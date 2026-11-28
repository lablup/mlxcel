# 기술 보고서: PR #1741 - feat(models): add the Kimi K3 text backbone

**날짜**: 2026-09-10
**작성**: mlxcel 메인테이너
**검토**: 구현 및 보안 리뷰 사이클
**상태**: 머지 전(계열 테스트 25개와 공용 모듈 스위트, clippy, fmt 모두 통과. 레이어를 잘라 낸 로컬 사본 두 개가 M5 Max에서 로드·프리필·디코드하고 유한 로짓을 남긴다. 93레이어 전체 모델은 이 프로젝트가 가진 어떤 하드웨어로도 돌릴 수 없고 토크나이저는 #1338이다)
**언어**: Rust, Markdown
**위험도**: 중간(새 계열에 더해, 다른 계열이 함께 쓰는 코드 세 곳을 건드렸다. gated delta의 게이트, `SwitchGLU`의 활성화 함수, 그리고 이 PR이 `kimi_linear`에서 끄집어내 가드를 붙인 `kv_b_proj` 분해다)

---

## 요약

Moonshot의 Kimi K3(`model_type: "kimi_k3"`, `text_config.model_type: "kimi_linear"`)는 2.8T 파라미터 93레이어 디코더다. Kimi Delta Attention 레이어 69개, 게이트가 붙은 NoPE MLA 레이어 24개, 7168폭 히든 상태를 3584폭으로 내린 잠재 공간에서 도는 896전문가 MoE, 그리고 12레이어 블록 단위의 Attention Residuals로 이루어진다. `src/models/kimi_linear.rs`에 KDA와 흡수형 MLA 하이브리드, 시그모이드 라우터가 이미 있으므로, 이 PR은 그 모듈에 없는 다섯 가지만 `src/models/kimi_k3.rs`로 새로 쓰고 있는 것 셋(`ShortConv1d`, `MultiLinear`, `kv_b_proj` 분해)은 빌려 쓴다. 공개 체크포인트의 compressed-tensors `mxfp4-pack-quantized` 전문가는 변환하지 않고 새니타이즈 시점에 MLX의 mxfp4 레이아웃으로 재해석해 읽는다.

이 PR에서 포팅 자체보다 값이 큰 것이 셋 있다. 보안 패스는 설정과 체크포인트가 어긋나는 23가지를 forward 도중의 MLX abort에서 텐서와 필드 이름을 부르는 로드 에러로 바꾸었고 4096토큰 잔차 창이 float32로 상주하지 않도록 Attention Residual 믹스를 두 번 다시 썼다. 두 계열이 한 벌만 쓰도록 여기서 끄집어낸 `decompose_kv_b_proj`는 Kimi Linear가 작성 이래 한 번도 갖지 못했던 reshape 대조 검사를 얻었다. 그리고 상태 리셋 한 줄을 그대로 옮겨 오는 과정에서 그 줄이 청크 프리필을 조용히 잘라 먹는다는 사실이 드러났다. 첫 청크 이후의 모든 청크가 앞선 청크가 쌓은 순환 상태를 버리는데 에러도 shape 불일치도 없다. Kimi K3는 청크 프리필을 끄고 나갔고 `kimi_linear`와 다른 다섯 계열의 같은 결함은 #1749가 맡는다.

---

## 1. `kimi_linear.rs`에 없는 다섯 가지

### 1.1 셋씩이던 프로젝션과 컨볼루션이 하나씩으로

Kimi Linear의 KDA 레이어는 `q_proj`, `k_proj`, `v_proj`를 따로 돌리고 각각 자기 `ShortConv1d`를 붙인다. Kimi K3는 `[3P, hidden]` 모양의 `qkv_proj` 하나와 `3P = 3 * 96 * 128 = 36864` 채널을 한꺼번에 지나가는 depthwise `qkv_conv` 하나를 싣는다. 헤드별 q / k / v로 쪼개는 것은 컨볼루션이 끝난 뒤다. 체크포인트는 여섯 텐서를 따로 저장하므로 합치는 일은 `sanitize_kda_layer`가 한다. `[C, 1, K]` 파이토치 컨볼루션 가중치 셋을 MLX의 `[C, K, 1]`로 옮겨 축 0으로 이어 붙이고 프로젝션 셋도 `.weight`, `.scales`, `.biases` 모두 같은 방식으로 이어 붙인다. 그래서 forward는 KDA 레이어마다 행렬곱 한 번과 컨볼루션 한 번만 돈다.

게이트가 나머지 절반이다. Kimi Linear는 `g = exp(-exp(A_log) * softplus(a + dt_bias))`를 쓰는데 `a`가 커지면 0에 닿고 순환 상태를 0까지 감쇠시킬 수 있다. Kimi K3는 `gate_lower_bound = -5.0`을 선언하고

```text
g = exp(gate_lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))
```

를 쓴다. FLA KDA 커널의 `safe_gate` 형태이고 모든 감쇠를 `(e^-5, 1)` 안에 가둔다. `A_log`는 공개 체크포인트에서 헤드 96개에 대해 128개 항목으로 실려 오므로 새니타이즈가 잘라 낸다. 출력 게이트는 저랭크 `g_b_proj(g_a_proj(x))` 쌍이 아니라 `[P, hidden]`짜리 전랭크 `g_proj`이고 `linear_attn_config.use_full_rank_gate`가 고른다. 설정 필드가 있으므로 두 형태 모두 구현했다.

미묘하게 틀리기 쉬운 대목 하나는 `KimiK3TextConfig::qk_norm_eps`에 적어 두었다. 레퍼런스는 FLA의 in-kernel l2norm `x / sqrt(sum(x^2) + 1e-6)`을 쓰고 이 트리에는 `rms_norm`이 있다. `D = head_dim`이라 하면 `sum(x^2) = D * mean(x^2)`이므로 l2norm은 `D^-0.5 * rms_norm(x, eps = 1e-6 / D)`와 같다. 엡실론을 헤드 차원으로 나누는 것은 장식이 아니다. `D = 128`에서는 제곱근 안의 값이 `1e-6`이냐 `7.8e-9`이냐가 갈린다. Kimi Linear는 mlx-lm의 평범한 `1e-6`을 쓰고 그대로 두었으므로 두 계열은 여기서 의도적으로 다르다.

### 1.2 아무것도 회전시키지 않는 MLA에 q-LoRA와 출력 게이트를 얹는다

MLA 레이어는 Kimi Linear의 흡수형을 그대로 유지한다. `embed_q`와 `unembed_out`은 새니타이즈 때 `kv_b_proj`에서 유도하고 `(kv_latent, k_pe)` 쌍은 `KVCache` 하나에 들어가며 디코드는 잠재 공간에서 점수를 내고 프리필은 잠재를 헤드별 k, v로 편다. K3가 그 둘레에 더한 것은 랭크 1536의 `q_a_proj`, RMSNorm, `q_b_proj`로 압축한 쿼리, `q_pe` / `k_pe` 절반에 회전을 전혀 걸지 않는 것, 그리고 `o_proj` 앞에서 어텐션 출력에 곱하는 `sigmoid(g_proj(x))`다.

RMSNorm 엡실론은 설정의 `rms_norm_eps`인 `1e-5`가 아니라 `1e-6`이다. 레퍼런스가 `q_a_layernorm`과 `kv_a_layernorm`을 둘 다 `KimiRMSNorm(rank)`으로 만들면서 설정 필드 대신 클래스 기본값을 쓰기 때문이다. 여기서 설정을 따르면 하필 쿼리와 키-밸류 압축 안쪽의 두 노름만 틀린다.

`mla_use_nope`는 구현하지 않고 false일 때 거부한다. 회전이 붙은 변종은 다른 어텐션이고 그런 설정이 들어오면 로드는 되면서 엉뚱한 위치에서 유창한 문장을 낸다.

### 1.3 SiTU, 그리고 그 대가 하나

이 계열의 게이트 MLP는 전부 SwiGLU가 아니라 SiTU를 쓴다.

```text
situ(up, gate) = (beta * tanh(gate / beta) * sigmoid(gate)) * (linear_beta * tanh(up / linear_beta))
```

`beta = 4.0`, `linear_beta = 25.0`이고 float32로 계산해 되돌린다. `linear_beta`가 null이면 두 번째 인자는 `up` 자신이 된다. `hidden_act`가 `"situ"`가 아니면 로드에서 실패한다.

라우팅 전문가도 이것이 필요한데 그 대목이 공용 코드 수정이다. `SwitchGLU`에 `SwitchGluActivation` 필드가 생겼고 기본값은 `SwiGlu`이며 그 기본값을 벗어나는 길은 `SwitchGLU::with_activation` 하나뿐이다. 대가는 대가가 발생하는 자리에 적어 두었다. `forward_fused_kernel`은 SwiGLU가 아닌 활성화에 `None`을 돌려준다. 그 커널이 융합하는 것은 `silu(gate) * up` 하나뿐이기 때문이고 그래서 Kimi K3의 전문가는 언제나 `gather_qmm` 경로를 탄다. 이 체크포인트에서는 오늘 잃는 것이 없다. 융합 커널은 affine 전용이고 이 전문가들은 mxfp4다. 다만 가드를 양자화가 아니라 활성화에 걸어 두었으므로 나중에 같은 계열을 affine으로 변환하더라도 잘못된 비선형을 계산하는 커널로 흘러들어 가지 않는다.

### 1.4 전문가는 히든 상태가 아니라 히든 상태의 투영 위에서 돈다

레이어 1부터 92까지는 잠재를 거쳐 라우팅한다. `routed_expert_down_proj`가 7168폭 잔차를 3584로 내리고 896개 중 16개 전문가가 거기서 돌며 가중합이 `routed_expert_norm`과 `routed_expert_up_proj`를 지나 7168로 돌아온다. 항상 켜져 있는 공유 전문가 둘은 원래 폭에서 돌아 더해진다. 전문가를 잠재 폭으로 재는 것이 896개를 감당 가능하게 만든다. `gate_proj` 플레인 하나가 `[3072, 7168]`이 아니라 `[3072, 3584]`가 되기 때문이다.

라우터는 게이트 로짓에 float32로 시그모이드를 건다. 선택은 `scores + e_score_correction_bias`로 하고 가중치는 바이어스 없는 `scores`에서 `take_along_axis`로 가져온 다음 자기들 합에 `1e-20`을 더한 값으로 정규화하고 `routed_scaling_factor`를 곱한다. 그룹 라우팅은 구현하지 않고 거부한다. `num_expert_group`과 `topk_group`이 둘 다 1이어야 하고 공개 설정이 그 조건을 만족한다.

### 1.5 Attention Residuals에 캐시가 필요 없다는 것이 설계의 전부다

`attn_res_block_size` 레이어마다 진행 중인 잔차를 블록 목록에 얼려 넣고 이후 각 어텐션과 MLP 서브레이어는 잔차 자체가 아니라 얼린 블록들과 현재 부분합의 소프트맥스 혼합을 입력으로 읽는다.

```text
logit_k = (raw_k @ w_eff) * rsqrt(mean(raw_k^2) + eps)      저장된 블록마다
logit_p = (partial @ w_eff) * rsqrt(mean(partial^2) + eps)
out     = sum_k softmax(...)_k * raw_k + softmax(...)_p * partial
```

`w_eff = res_norm.weight * res_proj.weight`는 로드 때 `[D, 1]` float32 열 하나로 접어 두므로 믹스는 저장된 블록당 gemv 한 번이다. 블록이 시작되는 레이어에서는 잔차가 다시 시작한다(`x + y`가 아니라 `partial = y`). 바로 앞에서 `x`를 블록 목록에 얼려 넣었기 때문이다. 레이어 0은 임베딩을 저장한다.

블록은 `[B, T, D]` 모양의 토큰별 행이고 한 토큰의 혼합은 그 토큰 자신의 얼린 상태만 읽으며 다른 위치를 읽지 않는다. 그래서 디코드 스텝은 블록 경계 레이어에서 자기 임베딩으로부터 자기 블록을 다시 계산하고 레퍼런스 forward가 하는 것과 같으며 호출 사이를 넘어가는 것이 없다. `ResidualBlocks`가 `run_layers`의 지역 변수이고 이것을 위한 캐시 변종이 없는 이유가 이것이다. 같은 이유로 파이프라인 병렬에서는 공짜가 아니라 문제가 된다. 레이어 24에서 시작하는 스테이지는 레이어 0과 12가 앞 노드에서 저장한 블록 둘이 필요하다. 그것이 #1734의 4번 항목이다.

---

## 2. 복사하지 않고 빌려 온 것

세 조각은 복제하지 않고 옮겼다. `MultiLinear`와 `ShortConv1d`가 `pub(crate)`이 되었고 후자는 이미 새니타이즈된 `[channels, kernel, 1]` 가중치를 감싸는 `new`를 얻어서 이어 붙인 컨볼루션으로 융합 인스턴스를 만들 수 있게 됐다. `decompose_kv_b_proj`는 `KimiLinearModel::sanitize_weights` 안에서 자유 함수로 빠져나와 두 새니타이저가 함께 부르며 #1026의 biases 누락 에러와 #958의 양자화 파라미터 경계 검사를 그대로 가져갔다.

두 계열 바깥까지 미치는 수정은 둘이고 둘 다 눈으로 확인한 것이 아니라 구조상 기존 호출자에게 무동작이다.

| 수정 | 기존 호출자 | 어긋날 수 없는 이유 |
|---|---|---|
| `gated_delta::compute_g_lower_bounded`와 `gated_delta_update_with_lower_bound` | Qwen3Next, Qwen 3.5, Kimi Linear | `lower_bound = None`이면 `compute_g(...)` 자체를 돌려준다. 동등한 그래프가 아니라 같은 그래프다. `gated_delta_update`는 한 줄짜리 전달자가 되었다. |
| `SwitchGLU::with_activation` | 트리의 모든 MoE 계열 | 생성자 둘 다 `SwitchGluActivation::SwiGlu`를 명시적으로 넣고 융합 디코드 커널은 그 밖의 값을 거절한다. 세터를 부르지 않는 호출자에게는 아무것도 달라지지 않는다. |

하한은 `gated_delta_ops` 앞에서 float32 게이트를 계산하는 방식만 바꾼다. Metal gated delta 커널도 ops 폴백도 어느 쪽이든 평범한 float32 게이트를 받으므로 두 경로 모두 손대지 않았다.

---

## 3. mxfp4 전문가는 변환이 아니라 재해석이다

compressed-tensors `mxfp4-pack-quantized`는 양자화된 선형 하나를 `weight_packed`(E2M1 코드, 바이트당 둘, uint8 `[out, in / 2]`)와 `weight_scale`(E8M0 블록 스케일, uint8 `[out, in / 32]`)로 저장한다. 공개 샤드에서는 `w1` / `w3`가 `[3072, 1792]`와 `[3072, 112]`, `w2`가 `[3584, 1536]`과 `[3584, 96]`이다. MLX 네이티브 mxfp4가 원하는 것은 코드 여덟 개를 낮은 니블부터 담은 uint32 워드와, 같은 uint8 E8M0 스케일과, 바이어스 없음이다.

같은 바이트다. `stack_expert_plane`은 전문가별 플레인을 새 선두 축으로 쌓고 `view(UINT32)`를 부르는데 이 호출은 마지막 축을 4로 나누고 버퍼는 건드리지 않는다. `gate_proj`와 `up_proj`는 `[896, 3072, 448]`, `down_proj`는 `[896, 3584, 384]`가 된다. 리틀엔디언 바이트 순서가 MLX mxfp4 커널이 읽는 낮은 니블 우선 코드 순서와 정확히 같다. 이 사실은 가정이 아니라 `mxfp4_repack_matches_scalar_dequant`가 스칼라 E2M1 역양자화와 대조해 고정한다.

92개 레이어에 전문가 896개라는 규모에서 실무적으로 중요한 것이 둘 있다. 플레인은 한 번에 하나씩 쌓아 평가하고 그 전문가별 원본을 다음 플레인 전에 놓아주므로 최대 상주량이 레이어 두 벌이 아니라 쌓은 플레인 하나다. 그리고 플레인 벡터는 `Vec::with_capacity(num_experts)`로 예약하지 않고 늘려 간다. `num_experts`는 상한 없는 `config.json` 필드라서 예약해 두면 말도 안 되는 선언이 루프가 이미 내놓는 부족 개수 에러 대신 capacity-overflow 패닉으로 바뀌기 때문이다.

`expert_quantization`은 모드를 설정이 아니라 데이터에서 정한다. uint8 `.scales` 플레인은 mxfp4의 서명이라 MLX `quantization` 블록이 무엇을 적었든 `(group_size 32, bits 4)`로 고정한다. 부동소수 스케일 플레인은 biases 유무로 모드를 추론해 설정의 쌍을 따르고 스케일이 없으면 밀집이다.

마지막 조각은 이 계열 코드 안에 있지도 않다. `src/models/sanitize.rs`의 `config_has_quantization_metadata`가 Apple Silicon의 bf16 to f16 변환이 기준으로 삼는 술어인데 Kimi K3는 최상위에 아무것도 선언하지 않는다. 유일한 선언이 `text_config` 아래 중첩된 compressed-tensors `quantization_config`다. 이 술어가 두 층 어디에서든 HuggingFace 표기를 보게 만든 것이, 어텐션과 공유 전문가와 밀집 MLP와 라우터와 `lm_head`(공개 `ignore` 목록이 부르는 전부)를 uint8·uint32 전문가 플레인 둘레에서 균일한 bf16으로 유지한다.

---

## 4. `config.json`은 신뢰할 수 없는 입력이고 MLX는 반환 대신 abort한다

이 경로의 MLX 호출 몇 개는 인자가 나쁘면 에러를 돌려주는 대신 프로세스를 내린다. MLX C++ throw가 cxx 브리지를 건너면 잡을 수 없는 `std::terminate`이기 때문이다. 여기 있는 호출은 전부 체크포인트 데이터나 `config.json` 값을, 혹은 둘 다를 받는다. 리뷰 커밋들은 각각을 텐서와 필드를 부르는 로드 시점 거부로 바꾸었다.

| MLX 호출 | 인자의 출처 | 가드가 없을 때의 실패 | 가드 |
|---|---|---|---|
| `decompose_kv_b_proj`의 `reshape` | `kv_b_proj`와 설정 차원 넷 | 키 이름도 없이 새니타이즈 도중 abort. 또는 나누어떨어지지만 `kv_lora_rank`가 아닌 폭이 새니타이즈를 통과해 첫 forward의 흡수형 MLA 행렬곱에서 abort | `num_heads * (qk_nope + v_head) * kv_lora_rank`를 풀어 적는 원소 수 검사 |
| 전문가 플레인의 `stack` | 체크포인트 | 새니타이즈 도중 abort | 어긋난 전문가 인덱스를 부르는 `check_uniform_shapes` |
| `view(UINT32)` | 패킹된 마지막 축 | 축이 uint32 워드의 정수배가 아니면 abort | 4의 양의 배수인지 명시적으로 검사 |
| q/k/v 융합의 `concatenate` | 컨볼루션 또는 프로젝션 플레인 셋 | 일부만 변환된 플레인 집합에서 새니타이즈 도중 abort | `check_concat_compatible` |
| `argpartition(kth = k - 1)` | `num_experts_per_token` | 체크포인트를 이미 다 올린 상태에서 첫 라우팅 토큰이 점수 축 밖을 가리킨다 | 두 로드 경로 모두에서 도는 `validate()`가 `k`를 `[1, num_experts]`로 묶는다 |
| KDA 게이트를 `[B, T, H, D]`로 바꾸는 `reshape` | `g_proj` 폭 | 첫 forward에서 abort | 로드 시점 `check_axis` |
| `attn_res_mix`의 `matmul` | `res_proj`와 `res_norm` 폭 | 둘끼리는 맞지만 `hidden_size`와 어긋나는 쌍이 로드를 통과해 첫 forward에서 abort | `load_attn_res_weight`가 곱하기 전에 둘을 각각 `hidden_size`와 대조 |

`kimi_k3.rs`는 이런 대조 검사를 23개 들고 있고(`check_axis` 15개, `check_numel` 8개) `hidden_size`는 `model.norm.weight` 원소 수 검사를 통해 간접적으로 덮이므로 입력 축 불일치도 로드에서 실패한다. 두 헬퍼 모두 키가 없는 것은 일부러 에러로 보지 않는다. 그 키가 필요한 로더가 이름을 대며 보고한다.

Attention Residual 믹스를 두 번 다시 쓴 이유도 같고 두 번째 재작성 쪽이 더 배울 것이 많다.

- **첫 형태.** 값을 `[B, T, n + 1, D]`로 쌓아 행렬곱 한 번으로 접는다. 합산 순서를 빼면 같은 연산이지만 블록들 위에 저장된 블록 전부의 연속 float32 사본을 따로 잡는다. `D = 7168`에 블록 창이 8개면 4096토큰 프리필에서 약 0.9GB의 일회성 메모리다.
- **두 번째 형태.** 스택을 없애고 항별로 접되 블록마다 float32 승격을 한 번 만들어 그 배열을 로짓 행렬곱과 가중항 두 곳에서 읽는다. 소프트맥스의 모든 소비자가 모든 로짓 뒤에 돌기 때문에 그 승격들이 첫 루프부터 접기까지 전부 상주한다. 역시 0.9GB다.
- **최종 형태.** 블록을 저장된 dtype 그대로 `matmul`과 `multiply`에 넘기고 float32 `w_eff`와 float32 확률 열에 대한 승격을 MLX가 각 연산 안에서 하도록 둔다. 코드가 손으로 하던 것과 같은 `astype`이므로 연산은 비트 단위로 같고 승격은 이제 연산이 끝나면 MLX가 놓아주는 연산 지역 임시값이다.

여기서 남길 만한 것은 일반적인 모양이다. 직접 쓴 `astype`은 모든 소비자에 걸치는 수명을 가진 그래프 노드이고 같은 승격을 연산 안에서 하면 임시값이다. 손으로 쓰면 작업을 공유하는 것처럼 보이지만 실제로는 수명을 늘린다.

보고했으나 고치지 않고 남긴 항목은 8절에 있다.

---

## 5. 자기 상태를 조용히 버리는 프리필

`forward_for_sequence`는 토큰 수를 기준으로 삼는 리셋으로 시작한다.

```rust
if seq_id.is_none() && seq_len > 1 {
    self.sequence_state.replace_internal(self.make_layer_caches());
}
```

이 리셋에는 역할이 있다. 캐시 없는 `LanguageModel::forward` 진입점을 서로 무관한 두 프롬프트에 연달아 써도 안전하게 만드는 것이 이 줄이다. 시퀀스 id 없이 들어온 다중 토큰 forward를 새 프롬프트의 시작으로 읽는다.

`supports_chunked_prefill`의 기본값은 true이고 `chunked_prefill_last_logits`는 시퀀스 id 없이 청크마다 `forward_last_logits`를 한 번씩 부르는데 그 호출은 `forward`에, 따라서 위 리셋에 떨어진다. 첫 청크 이후의 모든 청크가 앞선 청크들이 쌓은 KDA 컨볼루션 상태와 SSM 상태, MLA 잠재 쌍을 버린다. 아무것도 신호를 내지 않는다. 에러도 경고도 shape 불일치도 없다. 로짓은 마지막 `MLXCEL_PREFILL_CHUNK` 토큰(기본 2048)만으로 계산되고 다른 어떤 로짓과도 구별되지 않는다. 이 계열의 `max_position_embeddings`가 1048576이므로 청크를 넘는 프롬프트는 예외가 아니라 주된 사용례다.

Kimi K3는 `supports_chunked_prefill`을 false로 재정의한다. 이것이 보수적인 절반이다. #672의 청크별 프리필 메모리 상한을 포기하고(일회성 메모리가 청크마다가 아니라 프롬프트 전체 그래프 하나가 된다) 답을 정확하게 유지한다. 온전한 해결은 `forward_for_sequence`가 새 프롬프트와 이어지는 청크를 토큰 수 아닌 다른 것으로 구별하게 하는 것인데 그것은 공용 `ModelOwnedSequenceState` 계약을 바꾸는 일이라 forward 안에서 상태를 리셋하는 모든 계열을 놓고 검토해야 한다.

#1749의 감사는 `src/` 아래 `replace_internal` 호출 지점 42곳을 전부 훑었다. 일곱이 forward 안이고 가드가 전부 같은 모양이다. `kimi_k3`(여기서 완화), `kimi_linear`(#1749 자신), 그리고 `qwen3_next`, `rwkv7`, `bailing_moe_linear`, `recurrent_gemma`, `afmoe`. 뒤의 다섯은 각자 검증 체크포인트가 필요하므로 별도 후속 이슈가 필요하다. 서버 경로는 일곱 모두 영향이 없다. 실제 `SequenceId`로 `forward_for_sequence`를 부르므로 그 분기에 들어가지 않는다. CLI와 벤치마크의 결함이고 #672와 #674로 청크 프리필이 들어온 이래 `kimi_linear.rs`에 있었다.

---

## 6. 분산 관련 주장 하나를 바로잡고 하나를 메웠다

**파이프라인 병렬은 이 계열을 돌리지 못하는데 문서가 반대로 적혀 있었다.** 이 PR이 더한 것은 파티션 프로파일의 `kimi_k3_per_layer_bytes`다. 밀집인 레이어 0, 네 번째마다 오는 더 싼 q-LoRA MLA 레이어, 나머지 자리의 15~16GB MoE 레이어를 각각 다르게 잰다. 라우팅 전문가는 밀집 플레인을 설명하는 `bits_per_weight`와 무관하게 잠재 폭 위에서 4비트에 32가중치당 E8M0 1바이트로 센다. 이것은 계획 입력이다. `StageFamily` 변종도 스테이지 실행기도 없으므로 `--pp-size`와 `--pp-layers`는 `kimi_k3` 모델을 거부하고 `docs/distributed.md`와 `docs/supported-models.md`는 PP가 동작한다고 적는 대신 그 사실을 적는다. 실행기와 스테이지 경계를 넘는 Attention Residual 전송, uint32 와이어 dtype, 3노드 실행은 #1734가 맡는다.

**텐서 병렬에는 그럴듯해 보이는 실패로 이어지는 누락이 있었다.** replicate 목록에 항목이 없으면 `generate_shard_plan`은 `kimi_k3` 모델에 일반 트랜스포머 플랜을 준다. 그 플랜은 `q_proj`, `k_proj`, `v_proj`, `gate_proj`를 샤딩하겠다고 말하는데 이 모델의 프로젝션은 `qkv_proj`, `kv_a_proj_with_mqa`, `embed_q`, `switch_mlp`다. 이제 이 계열은 `kimi_linear` 옆에서 복제되고 `kimi_k3_is_replicated`가 그것을 고정한다. 다음 사람이 다시 발견하도록 남겨 두지 않았다.

---

## 7. 검증

호스트는 M5 Max 128GB, Metal이고 동시 빌드 에이전트 일곱과 공유한다. 아래 항목은 전부 최종 커밋으로 빌드한 바이너리에서 다시 실행했다.

### 7.1 유닛 테스트와 린트

`cargo test --profile test-fast --features metal,accelerate --lib models::kimi_k3`가 25개를 통과한다. 스칼라 레퍼런스와 대조한 SiTU, 하한 게이트의 범위, 분리 프로젝션과 대조한 융합 QKV, 프리필과 대조한 디코드, 비흡수 레퍼런스와 대조한 흡수형 q-LoRA MLA, 전문가별 루프와 대조한 잠재 MoE, 스칼라 소프트맥스와 대조한 AttnRes 믹스, `attn_res_block_size = 2`에서의 블록 개수, 스칼라 E2M1 역양자화와 대조한 mxfp4 리팩, 인과성, 공개 `config.json` 파싱, EOS 해석, 시퀀스 상태 격리, 그리고 리뷰 커밋이 추가한 모든 로드 시점 거부에 대한 음성 테스트다.

공용 모듈은 범위를 좁혀 돌렸다. `models::gated_delta`(6), `models::switch_layers`(24), `models::detection`(61), `distributed::tensor_parallel::plan_generator`(28), `distributed::pipeline::partition_profile`(7), `cli::turbo_args`(19). `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`와 `cargo fmt --all -- --check`는 깨끗하다.

### 7.2 잘라 낸 체크포인트 둘, 그리고 둘인 이유

전체 체크포인트는 4비트로 약 1.4TB 상주라서 실모델 게이트는 `models/kimi-k3-8l-mxfp4`에서 돈다. `moonshotai/Kimi-K3`의 로컬 사본에서 `text_config.num_hidden_layers`를 93에서 4로 낮추고 `model.safetensors.index.json`을 레이어 0~3과 임베딩·최종 노름·`lm_head`를 담은 샤드 7개로 다시 쓰고 `metadata.total_size`를 그 샤드들이 실제로 담은 54.46GiB로 고쳤다. 원본 둘은 옆에 남겨 두었다.

네 레이어는 공개 스케줄이 이 계열의 모든 레이어 종류를 처음으로 한 번씩 담는 지점이다. 레이어 0의 밀집 MLP, 레이어 0~2의 KDA, 레이어 1~3의 MoE, 그리고 인덱스 3(1-based 4, `full_attn_layers`의 첫 항목)의 첫 MLA 레이어다. 한 레이어를 더할 때마다 MoE 전문가로 약 17GB가 더 드는 반복이라서 메모리는 깊이 대신 두 번째 변종에 썼다.

| 게이트 | 명령 | 결과 |
|---|---|---|
| 로드 전 예산 | `mlxcel inspect -m models/kimi-k3-8l-mxfp4` | 가중치 54.46GiB, 8192토큰 KV 1.88GiB, 합계 67.69GiB, 여유 53.91GiB로 FITS |
| 로드·프리필·디코드 | `mlxcel generate -m models/kimi-k3-8l-mxfp4 -p "def fib(n):" -n 32` | 5.3초에 로드, 상주 45.41GB, 최대 49.71GB, 32토큰 0.60초(53.66 tok/s), exit 0, NaN 없음, abort 없음 |
| 유한 로짓 | `examples/logit_trace models/kimi-k3-8l-mxfp4 <corpus> 32 2 8 0` | teacher-forced 위치 64개, top-k 로짓 512개, 비유한값 0개. NLL 10.35~17.23(평균 13.65), top-k 로짓 6.66~10.38 |
| 다중 블록 Attention Residuals | `models/kimi-k3-4l-attnres2`에서 같은 명령 둘 | 54.61 tok/s로 32토큰, 위치 64개 추적, 비유한값 0개. NLL 10.56~18.19(평균 13.85) |

`models/kimi-k3-4l-attnres2`는 같은 샤드 7개를 심볼릭 링크로 두고 `attn_res_block_size`만 12에서 2로 낮춘 사본이다. 실제 가중치로 다중 항목 소프트맥스에 도달하는 방법은 이것뿐이다. 공개 블록 크기 12에서는 로드 가능한 크기의 사본이 블록을 정확히 하나만 저장하므로 `n = 1`이 되고 믹스의 루프가 아예 돌지 않는다. 블록 크기 2에서는 레이어 0과 2가 각각 블록을 얼리고 레이어 3의 믹스가 실제 `res_proj`, `res_norm` 가중치를 상대로 세 항목을 채점한다. 자연스럽게 그 경로를 태우는 13레이어 사본은 약 220GB다.

두 추적과 두 생성 모두 두 번째 실행에서 자릿수까지 재현되고 그것이 결정성 확인이기도 하다.

생성된 텍스트는 두 실행 모두 여러 언어가 섞인 잡음이다. 93레이어 중 4레이어가 내놓는 결과가 그것이고 #1338이 들어오기 전까지 범용 tiktoken 폴백이 `tiktoken.model`을 잘못된 사전 토큰화 정규식으로 읽는 것이 겹친다. 게이트는 로드와 shape과 NaN 부재와 로짓의 유한성이다. 텍스트가 아니며 디코드 속도를 품질 결과인 양 제시하는 대신 그렇다고 적어 둔다.

---

## 8. 확인하지 못한 것

**전체 모델.** 돌리지 않았고 돌릴 수도 없다. 4비트로 약 1.4TB인데 쓸 수 있는 가장 큰 호스트가 128GB이고 512GB 노드 셋이 생긴다 해도 트리에 파이프라인 스테이지 실행기가 없다. #1734가 둘 다 맡는다.

**유창한 출력.** 이 PR은 그것을 보이지 않는다. #1334의 승인 기준 두 개가 이 이유로 명시적으로 미체크다.

**어떤 레퍼런스와의 토큰 일치도.** mlx-lm에 `kimi_k3`가 없으므로 그리디 id를 대조할 외부 오라클이 없고 잘라 낸 사본의 실행은 자기 자신과 유닛 테스트의 스칼라 레퍼런스하고만 대조한다. 이 항목을 닫는 것은 #1734의 두 번째 승인 기준인 레이어별 float32 레퍼런스 추적이다.

**워크스페이스 게이트.** `cargo test --workspace`와 `cargo clippy --workspace --all-targets`는 이 호스트에서 돌리지 않았다. `mlxcel-core` 테스트 바이너리가 동시에 돌면 서로를 abort시키기 때문이다(#1008). 위의 범위 한정 스위트와 CI가 덮는다.

**보고했으나 남긴 대조 검사 공백(MEDIUM).** `lm_head.weight` 행 수와 `vocab_size`, `routed_expert_up_proj`와 잠재 폭, `moe_intermediate_size`와 쌓인 전문가 플레인, `intermediate_size`와 밀집 MLP. `num_hidden_layers`에 상한이 없으므로 악의적인 `config.json`은 `sanitize_weights`의 `0..num_hidden_layers` 루프를 멈추지 않는 루프로, `from_weights`의 `Vec::with_capacity`를 capacity-overflow 패닉으로 바꾼다.

**계산으로만 잰 비용 둘.** 잔차 창은 forward 내내 `ceil(num_hidden_layers / attn_res_block_size)`개의 전폭 잔차 사본을 들고 있고 공개 설정에서 4096토큰 프리필이면 블록 8개에 bf16으로 약 470MB이며 이 계열에는 청킹이 없다. 디코드는 레이어당 믹스를 두 번 치르고 각 믹스는 저장된 블록당 gemv 하나와 브로드캐스트 곱 하나라서 93레이어에 블록 8개면 토큰당 대략 1.5k개의 작은 연산이 더 붙는다. 둘 다 이 비용이 드러날 만큼 깊은 체크포인트에서 프로파일하지 않았다.

**남은 전문가 텐서를 버리는 경로.** `sanitize_moe_layer`는 쌓기가 끝난 뒤에도 `experts.` 아래 남은 것을 버린다. compressed-tensors export가 `weight_shape` 같은 사이드카를 실을 수 있다는 전제다. 공개 샤드는 전문가마다 `weight_packed`와 `weight_scale`만 싣고 있으므로 그 경로는 공개 샤드에서 한 번도 발화하지 않으며 유닛 테스트로만 검증된다.

**배칭, 패딩 프리필, 비전 타워.** `supports_batching`과 `supports_padded_prefill`은 둘 다 false이고 이유는 Kimi Linear와 같다. 비전 타워는 #1342다.

---

## 9. 변경 요약

### 통계

| 지표 | 값 |
|---|---|
| 변경 파일 | 29 |
| 추가 줄 | 5087 |
| 삭제 줄 | 117 |
| 새 모듈 | 3개(`kimi_k3.rs` 2086, `kimi_k3_sanitize.rs` 488, `kimi_k3_tests.rs` 1796) |
| 추가 테스트 | 계열 25개, 파티션 프로파일 1개, TP 플랜 1개, 탐지 1개, 새니타이즈 정책 1개 |

### 영역별 변경

- `src/models/kimi_k3.rs`: 설정과 `validate`, KDA와 MLA 블록, 밀집 SiTU MLP, 잠재 MoE, Attention Residual 믹스와 레이어별·모델 수준 가중치, 디코더 레이어, 모델 셸과 `LanguageModel` 구현, 그리고 설정 대 체크포인트 대조 검사 23개.
- `src/models/kimi_k3_sanitize.rs`: `language_model.` 범위 정리, MTP와 범위 밖 레이어 제거, 옛 잔차 키 표기, KDA 컨볼루션·프로젝션 융합, MLA 분해 호출, MoE 이름 변경과 mxfp4 전문가 쌓기. 설계상 두 번 돌려도 결과가 같다.
- `src/models/kimi_linear.rs`: `MultiLinear`와 `ShortConv1d`를 크레이트에 열고 `ShortConv1d::new` 추가, `decompose_kv_b_proj` 추출과 reshape 대조 검사 부여.
- `src/models/gated_delta.rs`: `compute_g_lower_bounded`와 `gated_delta_update_with_lower_bound`.
- `src/models/switch_layers.rs`: `SwitchGluActivation`, `situ_activation`, `SwitchGLU::with_activation`, 융합 커널 제외.
- `src/models/sanitize.rs`: `config_has_quantization_metadata`를 문서화하고 중첩 compressed-tensors 표기에 대해 테스트.
- `src/distributed/`: Kimi K3 파티션 프로파일, TP replicate 항목과 폴백 아키텍처 라벨, 그리고 양쪽 테스트.
- 등록: 탐지, 레지스트리, 메타데이터, `LoadedModel`, 특수 가중치 로더, 메모리 추정, 하이브리드 SSM 프롬프트 캐시 명단, MLA 잠재 캐시 계열 목록.
- 문서: `docs/supported-models.md`(계열 항목), `docs/distributed.md`(PP는 계획 입력일 뿐), `docs/turbo-kv-cache.md`와 `src/cli/turbo_args.rs`(MLA 잠재 양자화 제외).

### 커밋

| 해시 | 유형 | 제목 |
|---|---|---|
| `d7ae018c` | feat | add the Kimi K3 text backbone |
| `4fcddfd6` | fix | bound the router, cross-check config against the checkpoint |
| `9828b5ca` | fix | name the two remaining checkpoint-driven MLX aborts |
| `9c8ad3cd` | fix | bound the residual mix and the kv_b_proj decompose |
| `a6993da1` | fix | opt out of chunked prefill, correct the distributed claims |
| `b1e826e5` | docs | add the family to the shared-function and KV-cache rosters |
| `a424acbf` | test | pin the tensor-parallel plan as replicated |

### 관련 이슈

이 PR은 에픽 #1331의 하위 이슈 #1334를 닫는다. 같은 에픽의 형제는 #1338(토크나이저와 XTML 채팅 렌더링)과 #1342(MoonViT3D 비전 타워)다. 후속으로 #1734(파이프라인 스테이지 실행기와 전체 모델 검증)와 #1749(`kimi_linear`와 다섯 계열의 청크 프리필 상태 폐기)를 제출했다. #1026의 biases 누락 에러와 #958의 양자화 파라미터 경계 검사를 공용 `decompose_kv_b_proj`로 옮겨 왔다. 이 계열에 한해 #672의 청크별 프리필 메모리 상한을 포기했다.

---

## 10. 후속 작업

**이 PR이 확인하지 못한 모든 것의 병목은 #1734다.** 전체 모델 실행, 유창한 출력, 레이어별 float32 레퍼런스 추적이 전부 거기에 있고 Attention Residual 전송 문제도 마찬가지다. 레이어 `s`에서 시작하는 스테이지는 앞 노드에서 계산된 얼린 블록 `ceil(s / 12)`개가 필요하다.

**#1749를 `kimi_linear`에서 멈추면 안 된다.** 다섯 계열이 기본값 `supports_chunked_prefill`과 함께 동일한 forward 내 리셋을 들고 있다. 공용 해결을 맡는 쪽은 같은 변경에서 Kimi K3의 재정의도 걷어 내야 두 Kimi 계열이 갈라지지 않는다.

**MEDIUM 대조 검사 공백을 닫는다.** 8절에 남은 설정 대 체크포인트 폭 넷은 이미 닫은 것들과 실패 모양이 같고 상한 없는 `num_hidden_layers`는 악의적 설정을 에러가 아니라 정지 없는 루프로 바꾸는 항목이다.

**문서의 숫자 하나가 틀렸다.** `docs/supported-models.md` 항목은 쌓인 `down_proj` 플레인을 `[896, 3584, 192]`로 적는다. 공개 샤드는 `w2.weight_packed`를 uint8 `[3584, 1536]`으로 저장하고 uint32 view가 마지막 축을 4로 나누므로 `[896, 3584, 384]`다. `gate_proj`와 `up_proj`의 `[896, 3072, 448]`은 맞다.

### 옮겨 갈 만한 교훈

이 PR에서 가장 값이 큰 결함은 새로 더한 코드에 있지 않다. `forward_for_sequence`의 리셋은 `kimi_linear.rs`에서 그대로 옮겨 온 줄이고 #672 이래 그 모듈에서 청크 프리필을 조용히 잘라 먹고 있었다. 이것을 드러낸 것은 새 호출자를 위해 `seq_id.is_none() && seq_len > 1`이라는 조건이 실제로 무슨 뜻인지 적어야 했던 일이다. 그 뜻은 '호출자의 의도를 인자의 shape에서 추론한다'이고 shape에는 의도가 실리지 않는다. 무관한 두 호출자가 정반대 이유로 같은 shape을 보낼 수 있으며 그중 하나가 청크마다 그것을 보내는 `chunked_prefill_last_logits`다. 추론이 타입 관점에서 옳기 때문에 실패가 조용하다.

여기서 따라 나오는 규칙은 값이 싸다. 포팅이 목적을 그 자리에서 읽을 수 없는 줄을 복사할 때는 옮기지 말고 새 호출자에게 산문으로 설명한 다음, 그 설명이 전제하는 상태를 만들어 낼 수 있는 진입점을 전부 확인한다. 여기서는 `supports_chunked_prefill` 한 번과 `replace_internal` 한 번의 grep이었고 그것이 한 계열의 포팅을 일곱 계열의 감사로 바꾸었다.
