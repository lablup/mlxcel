# PR #3441: Raon-1 mini (raon1_mini) hybrid 아키텍처 (이슈 #3225)

**날짜**: 2026-11-26

**상태**: PR open(#3225를 닫는다). M3 Ultra에서 검증했다. 33개 파일, +4324 / -42.

## 요약

이 PR은 KRAFTON Raon-1 mini 아키텍처(`raon1_mini`, `KRAFTON/Raon-1-mini` revision `6691a076`)를 추가한다. 원본 bf16 + MXFP8 체크포인트로 `mlxcel generate`, `mlxcel-server`, `mlxcel arch`에서 동작한다. 이 모델은 mlxcel에 한꺼번에 존재하지 않던 네 요소를 결합한다: 4-stream Raon_mHC residual, RDA gated-delta linear attention 레이어, Kimi K3와 공유하는 gated NoPE MLA 레이어, SiTU activation과 MXFP8 expert를 쓰는 sigmoid 라우팅 top-6 MoE.

Greedy 출력은 짧은 프롬프트와 871-token 프롬프트(43 token) 모두에서 Transformers reference와 token 단위로 일치한다. Teacher-forced trace에서 4 x 256 corpus는 decided position 578개 중 0개, 긴 프롬프트는 549개 중 1개가 불일치한다. 그 1개는 reference 자신이 bf16과 f32 사이에서 다른 답을 내는 row다. 서버 batched decode는 native로 동작하며, 동시 요청 1 / 2 / 4개에서 aggregate decode가 29.9 / 75.0 / 127.0 tok/s다.

## 1. 문제 정의

### 1.1 배경

이슈 #3225는 steering alpha 0(reference의 `set_reasoning_effort("max")`)에서 `modeling_raon1_mini.py`와 일치하는 Raon-1 mini 지원을 요청했다. 이슈 본문은 그 파일을 읽고 forward pass를 rounding 위치까지 적어 두었다.

### 1.2 포팅 범위

- **Raon_mHC**: residual을 4개 stream `[B, L, 4, D]`로 운반한다. 각 sublayer site는 token마다 pre, post, residual-mix 계수를 계산하며, mix 행렬은 Helmert chart 위에서 만들어 doubly stochastic이 된다. DeepSeek-V4의 Sinkhorn mHC와는 다르다.
- **RDA**: 18개 gated-delta 레이어. f32 short conv, L2 정규화 q, k(eps 1e-6), write ratio, 하한이 있는 차원별 decay, `[B, 40, 128, 128]` f32 recurrent state.
- **Gated NoPE MLA**: 7개 레이어이며 KV를 갖는 유일한 레이어다.
- **MoE**: MXFP8 expert 128개, correction bias가 있는 top-6 sigmoid 라우팅, shared expert, expert와 dense MLP 양쪽의 SiTU(`4 tanh(g/4) sigmoid(g) * 25 tanh(v/25)`).
- 로드는 되어야 하지만(f32) 아직 적용하지 않는 reasoning-effort steering tensor 5개(#3435).

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| reference와 다른 위치의 rounding이 logit을 흘려 긴 프롬프트에서 greedy token이 뒤집힘 | High | High(실제로 발생, 3.2 참조) |
| hybrid MLA 계열의 KV 추정이 틀려 서버 context sizing과 prompt-cache 예산이 어긋남 | High | High(수정 전 80배 과대 추정) |
| 잘못된 config나 체크포인트가 이름 있는 오류 대신 panic하거나 cxx bridge 너머에서 abort | High | Medium |
| 서버 decode가 동시성에 따라 늘지 않음 | Medium | High(수정 전 per-row fallback) |

## 2. 기술적 검토 사항

### 2.1 보안 관점

config와 체크포인트는 신뢰할 수 없는 입력이다. 보안 검토(999aa995e)에서 다음 HIGH 항목을 수정했다.

| 문제 | 심각도 | 상태 |
|------|--------|------|
| `validate_layer_partition`이 `num_hidden_layers`로 bitmap 크기를 정해 `usize::MAX`에서 overflow, 큰 값이면 config만으로 메모리 할당 | High | 수정 |
| 0 width 허용(RDA `head_dim` 0이면 query scale이 `inf`), `raon_mhc_streams` 상한 없음, i32를 넘는 차원이 `as i32`에서 잘림 | High | 수정 |
| `stack_mxfp8_experts`가 shape이 다른 expert를 stack해 MLX 안에서 throw하고 프로세스가 abort | High | 수정 |
| mHC stream 2개 미만(Helmert basis가 비어 있음) | Medium | 수정 |
| projection 입력 width(RDA, MLA linear, expert `down_proj`)를 출력 row로만 검사해 잘못된 체크포인트가 첫 matmul에서 abort할 수 있음 | Medium | 미해결(Kimi K3와 동일) |
| oracle 스크립트가 체크포인트의 Python 파일을 import하며 digest는 기록하지만 고정하지 않음 | Low | 미해결 |

vendor MLX export와 NVFP4 export는 이름으로 거부한다(#3437). 검사는 deserialization 전 raw JSON에서 한 번, `validate()`에서 flatten된 extras로 다시 한 번 실행되므로 config 기반 weight 경로도 포함된다.

### 2.2 성능 관점

| 항목 | 이전 | 이후 |
|------|------|------|
| aggregate decode, 동시 요청 1 / 2 / 4 | 29.8 / 30.7 / 29.8 tok/s | 29.9 / 75.0 / 127.0 tok/s |
| token당 KV 추정 | 25 레이어 x 40 head x 320 element | 7 레이어 x 576 latent element(약 80배 작음) |
| single-stream decode | 해당 없음 | 32.9 tok/s |
| 로드 peak 메모리 | 해당 없음 | active 메모리와 동일, 32.8 GB |

f32 RDA state는 bf16 대비 token당 약 47 MB의 state traffic을 더한다(18 레이어 x 40 x 128 x 128 x 4 byte, 읽기와 쓰기). M3 Ultra 대역폭에서 1 ms에 한참 못 미친다. single-stream decode는 token당 약 30 ms이고 weight 읽기 하한은 약 8 ms(token당 약 6.5 GB)이다. 즉 step이 dispatch-bound다(8절 참조).

### 2.3 호환성/의존성 관점

- **Breaking change**: 없음. Kimi K3 MLA는 `NopeMlaSpec` / `KimiK3MLAAttention::from_spec` 뒤로 리팩터링했고 동작은 그대로다. 다른 gated-delta 계열은 표준 kernel을 유지한다.
- **새 의존성**: 없음. oracle 스크립트는 transformers가 필요하다(5.16.1로 검증).
- tensor parallel, DFlash2(#3436), NVFP4(#3437), Raon-1 nano(#3438)는 범위 밖이다. `raon_mhc.rs`와 `raon_rda.rs`는 Raon-1-mini config 타입을 받지 않으므로 nano가 그대로 재사용할 수 있다.

### 2.4 코드 품질 관점

새 모듈은 관심사별로 나뉜다: `raon_mhc.rs`, `raon_rda.rs`, `raon1_mini_moe.rs`, `raon1_mini_config.rs`, `raon1_mini_sanitize.rs`, `raon1_mini_lm.rs`, `raon1_mini.rs`, 각각 테스트 파일이 있다. 테스트는 Helmert basis, doubly stochastic residual, aggregate와 mix, layer partition, SiTU, RDA ratio와 decay(두 bound 형태), MoE router, NoPE MLA 순서 불변성, sanitize, MXFP8 native matmul 대 dequantize-then-matmul, chunked prefill 및 left-padded row decode 대비 prefill, per-sequence decode 대비 batched decode, config 검증, detection, export 거부, f32에서 0.998 decay가 64 step 동안 유지되는지를 다룬다.

## 3. 기술적 선택과 그 이유

### 3.1 rounding이 다를 때는 이슈 본문이 아니라 reference를 따른다

**컨텍스트:** 이슈 본문은 forward pass를 기술했지만 몇 군데에서 reference의 rounding 위치가 달랐다. 예를 들어 이슈는 MoE 출력을 `sum_k(experts.f32 * w_k) + shared.f32`를 한 번 rounding한다고 적었으나, `Raon1MiniMoE.forward`는 routed 합을 model dtype으로 rounding한 뒤 그 dtype에서 shared expert를 더한다. reference는 또한 L2 정규화한 q를 `head_dim ** -0.5` scale 전에 bf16으로 rounding하고 `beta`를 bf16으로 유지하며, 체크포인트에는 joint `qkv_conv1d` 하나가 아니라 conv 세 개(`q_conv1d`, `k_conv1d`, `v_conv1d`)가 들어 있다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 이슈 본문을 따름 | 문서화된 사양과 일치 | 잘못 놓인 rounding마다 bf16 noise가 생기고 trace 비교에서 드러남 |
| **선택: `6691a076`의 `modeling_raon1_mini.py`를 따름** | 중간값을 bit 단위로 비교 가능, trace를 불일치 0으로 맞출 수 있음 | 이슈 본문 일부가 낡은 내용이 됨 |

**선택 이유:** 수용 기준이 reference와의 일치이므로 reference가 기준이다. conv 세 개는 sanitize에서 q, k, v 순서로 depthwise conv 하나로 합친다. depthwise conv에서는 정확히 같은 결과이고 레이어당 dispatch 두 개를 줄인다.

### 3.2 RDA decay와 recurrent state를 f32로 유지한다

**컨텍스트:** 첫 커밋 이후에도 긴 프롬프트 trace의 p99 / max logit delta가 1.125 / 10.25였다. 원인은 f32 reference forward와 레이어 단위로 비교해서 찾았다. layer 0에 bit 단위로 같은 bf16 q, k, v, beta를 넣었을 때 mlxcel RDA core 출력은 f32 reference 대비 상대 오차 0.045였고, reference 자신의 bf16 forward는 0.0022였다. 오차는 입력이 아니라 recurrence에 있었다.

표준 Metal gated-delta kernel은 f32 decay `g`와 들어오는 state를 activation dtype으로 cast하고, 나가는 state도 그 dtype으로 저장한다. bf16에서 1 바로 아래 값의 간격은 2^-8이라 decay 0.998이 0.996이 된다. long-memory head는 token마다 그 decay를 곱하므로 오차가 프롬프트 길이에 따라 누적된다. reference는 log decay와 state를 fp32로 유지한다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 표준 kernel(bf16 decay와 state) | Qwen3Next, Kimi 계열과 공유, state traffic 적음 | 긴 프롬프트에서 token마다 decay가 흐름 |
| 모든 gated-delta 계열을 f32로 | 경로 하나 | bf16 reference로 검증된 계열의 출력이 바뀜 |
| **선택: RDA만 `gated_delta_ops_precise` 사용** | reference와 일치, 기존 precise kernel 경로 재사용(cast 없음, f32 state) | token당 state traffic 약 47 MB 증가 |

**결과:** p99 / max logit delta가 1.125 / 10.25에서 0.75 / 2.0으로, perplexity 차이가 +0.475%에서 +0.337%로 줄었다. 4 x 256 corpus는 578개 중 0개 그대로다. 0.998 decay가 f32 state로 64 step 유지되는지 확인하는 unit test를 추가했으며 표준 kernel에서는 실패한다.

### 3.3 모델 버그를 쫓기 전에 oracle의 presence penalty를 고친다

**컨텍스트:** 871-token 프롬프트 greedy decoding에서 token 하나가 뒤집혔다: reference는 "Linux and Windows", mlxcel은 "Linux/Windows". oracle은 `Raon1MiniForCausalLM.generate`에 `presence_penalty=None`을 넘겼는데, 이 함수는 인자가 None이 아닐 때만 generation config를 덮어쓴다. 그래서 `generation_config.json`의 1.5가 남았고 `_get_logits_processor`가 매 step에 적용했다. `/`는 "Linux/CUDA"에서 이미 나왔으므로 penalty가 1.5를 깎았고, " and"에 대한 1.125 우위가 사라졌다.

**결정:** 0.0을 넘겨 processor를 건너뛴다(f7c94abcc). 그러면 reference가 mlxcel과 같은 43 token을 낸다. 버그는 reference 쪽이었고, reference를 reference와 비교해서 드러났다.

### 3.4 row 866은 reference 쪽 bf16 noise로 본다

**컨텍스트:** 긴 프롬프트와 continuation에서 decided position 549개 중 1개가 여전히 불일치한다: row 866, `<|im_start|>assistant` 다음 token이다. 어느 쪽이 틀렸는지 가리려고 그 지점에서 reference를 f32로 실행했다. f32는 mlxcel과 같은 `\n`을 gap 0.48로 고른다. reference의 bf16 변형(프롬프트만, cached prefill, 더 긴 context)은 서로 일치하므로, bf16 reference가 그 row에서 무작위가 아니라 일관되게 어긋난다.

**결정:** row 866을 불일치 0 기준에서 예외로 두고 근거를 기록한다. 더 쫓으면 모델이 아니라 reference의 bf16 오차를 맞추는 일이 된다.

### 3.5 KV cache를 MLA 레이어의 latent pair로만 계산한다

**컨텍스트:** config에 `kv_lora_rank`가 있어 `kv_arch`가 DeepSeek V2 decompressed 분기로 빠졌고, 25개 레이어 전부를 token당 40 head x 320 element로 계산했다. 실제 cache의 약 80배다. 서버 context sizing과 prompt-cache snapshot 예산이 이 추정을 읽는다.

**결정(45dee535d):** `raon1_mini`를 `MLA_LATENT_CACHE_FAMILIES`에 넣고, `kv_arch`가 Kimi의 `kda_layers`처럼 `rda_config.rda_layers`에서 MLA 레이어를 센다. `QUANTIZED_LATENT_FAMILIES`에는 넣지 않으므로 int8 latent 요청은 int8 trace를 돌리기 전까지 fp16으로 해석된다. 고정 크기 RDA state(sequence당 약 47 MB)는 세지 않는데, 이는 recurrent 레이어에 대한 estimator의 관례다.

같은 커밋에서 `gate_lower_bound`가 null일 때의 decay도 고쳤다. 기존에는 `compute_g`를 거쳐 `exp(A_log)`가 softplus 밖에 있었으나 reference는 `-softplus(exp(A_log) * (a + dt_bias))`를 계산한다. 공개된 두 config 모두 -5.0을 쓰므로 출력은 바뀌지 않는다.

### 3.6 native batched decode

**컨텍스트:** 첫 커밋은 다른 recurrent hybrid처럼 `BatchedDecodeKind::PerRowFallback`을 선언했다. 그러면 서버가 활성 sequence마다 full forward를 한 번씩 돌려 expert와 260K-row head를 포함한 모든 weight를 row마다 읽는다. aggregate 처리량은 동시성과 무관하게 약 30 tok/s에 머물렀다.

**결정(48a466487):** Kimi Linear와 Kimi K3를 따른다. row들의 RDA conv state와 recurrent state를 step마다 `[B, ...]` state 하나로 stack하고 다시 나눈다. MLA 레이어는 `KimiK3MLAAttention::forward_batched_decode`로 각 row 자신의 latent cache를 읽으므로 길이가 다른 row도 한 step을 공유한다. multi-token step, mask, fresh row와 started row가 섞인 경우는 여전히 row 단위로 실행한다. 두 mHC site와 FFN은 `forward_with_mixer`를 통해 두 경로가 공유한다.

**트레이드오프:** 동시 greedy 프롬프트 4개 중 2개는 단독 실행과 정확히 일치했고, 나머지 2개는 약 280자 이후 갈라졌다. batched matmul shape에 따른 일반적인 bf16 효과다.

### 3.7 MXFP8 expert를 저장된 그대로 쓴다

체크포인트의 expert weight는 uint8 E4M3이고 scale은 입력 32개당 uint8 E8M0로, MLX의 native `mxfp8` 레이아웃과 같다. sanitize는 weight를 uint32로 view하고 `weight_scale_inv`를 `scales`로 이름을 바꾼 뒤 expert를 `switch_mlp`로 stack한다. byte는 하나도 바꾸지 않는다. uint8이 아닌 weight나 scale, shape이 다른 expert는 로드 오류다.

## 4. 구현 상세

```
embed -> 4개 stream으로 broadcast
  레이어마다:
    mHC(attn site): pre, post, R  -> x = norm(sum_s pre[s]*streams[s])
    mixer: RDA(18 레이어, f32 state) | gated NoPE MLA(7 레이어, latent KV)
    streams = R @ streams + post * branch      (f32, 한 번 rounding)
    mHC(mlp site) -> SiTU MLP 또는 MoE(routed rounding 후 + shared)
최종: norm(stream 평균) -> lm_head
```

주요 파일: `src/models/raon_mhc.rs`, `src/models/raon_rda.rs`, `src/models/raon1_mini_moe.rs`, `src/models/raon1_mini.rs`, `src/models/raon1_mini_lm.rs`, `src/models/raon1_mini_config.rs`, `src/models/raon1_mini_sanitize.rs`; `src/models/gated_delta.rs`(새 `gated_delta_ops_precise`); `src/models/kimi_k3.rs`(`NopeMlaSpec`); `src/execution/kv_arch.rs`; `scripts/raon1_mini_oracle_trace.py`(`generate`, `trace` 모드, presence penalty를 0으로 고정).

```rust
pub fn gated_delta_ops_precise(q, k, v, g, beta, state, mask)
    -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    gated_delta_ops_with_parity(q, k, v, g, beta, state, mask, false, true)
}
```

## 5. 학습 포인트

- **입력을 고정해 precision 버그를 찾는다.** 한 레이어에 bit 단위로 같은 bf16 입력을 넣고 출력을 f32 reference와 bf16 reference 양쪽에 비교하면 "입력이 다르다"와 "연산자가 다르게 rounding한다"를 분리할 수 있다. 20배 차이(0.045 대 0.0022)가 곧바로 recurrence를 가리켰다.
- **bf16은 1에 가까운 decay를 담지 못한다.** mantissa 8 bit에서 1 바로 아래 간격은 2^-8(약 0.0039)이다. token마다 그런 decay를 곱하면 1 ulp 오차가 다른 memory horizon이 된다.
- **모델보다 oracle을 먼저 확인한다.** config 기본값을 조용히 유지하는 `None` 인자(여기서는 presence penalty 1.5) 때문에 reference가 greedy가 아니었다. flip을 포팅 탓으로 돌리기 전에 reference를 reference와 비교한다(f32 대 bf16, cached 대 uncached).
- **reference noise도 유효한 결론이다.** reference의 f32 실행이 포팅과 일치하고 bf16 변형끼리 일치하면, 그 불일치는 reference의 precision 문제이며 근거와 함께 예외로 둘 수 있다.

## 6. 관련 이슈

- #3225: Raon-1 mini 지원(이 PR로 닫힘)
- #3435: reasoning-effort steering vector 적용
- #3436: Raon용 DFlash2
- #3437: NVFP4 export
- #3438: Raon-1 nano

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 33 |
| 추가 라인 | +4324 |
| 삭제 라인 | -42 |

| Hash | 유형 | 메시지 |
|------|------|--------|
| `028e5edea` | feat | add Raon-1 mini (raon1_mini) hybrid architecture |
| `4ed234d71` | fix | keep the Raon RDA decay and state in float32 |
| `45dee535d` | fix | size Raon-1 mini KV as a latent pair in its MLA layers |
| `f7c94abcc` | fix | run the Raon-1 mini greedy oracle without presence penalty |
| `999aa995e` | fix | refuse oversized Raon-1 mini geometry before it aborts |
| `48a466487` | fix | batch Raon-1 mini decode natively instead of per row |
| `a9e60c6fa` | test | cover the Raon-1 mini router, detection and export refusals |
| `8fb7c34f7` | docs | describe Raon-1 mini thinking control and final behavior |

## 8. 후속 조치

### 알려진 공백

- `generation_config.json`의 `presence_penalty`를 읽지 않는다. 기본값은 모든 계열에서 0.0이다. 읽도록 바꾸면 이 값을 설정한 모든 계열이 바뀐다.
- single-stream decode가 dispatch-bound다: 50개 mHC site가 각각 token당 약 30개의 작은 op을 내고, RDA 레이어마다 projection 9개를 따로 실행한다. 후보: mHC `route` / `aggregate` / `mix`의 compile fusion, 로드 시 q/k/v(및 `b_proj` / `rda_w_proj`)를 matmul 하나로 합치기.
- projection 입력 width를 로드 시 검사하지 않는다(Kimi K3와 같은 공백).
- fast RMSNorm은 gamma 곱 전에 `x * inv`를 bf16으로 rounding하지만 reference는 한 번만 rounding한다(모든 계열).
- `compute_g_lower_bounded`는 cast 전에 bf16 `a`를 f32 `dt_bias`에 더하며, CUDA overlay는 그 합을 bf16으로 rounding한다(Kimi K3와 공유, Metal은 정확).
- `raon1_mini`의 int8 latent KV는 int8 trace를 기다린다.

## 부록: 검증(M3 Ultra)

- `mlxcel arch`가 Raon-1 mini를 나열한다. 이슈의 `generate` 명령은 `42`로 답하며, template의 `enable_thinking=false` 렌더링을 `--no-chat-template`로 넣으면 reference와 같은 id인 `42<|im_end|>`를 낸다.
- Transformers 대비 greedy(bf16, CPU, transformers 5.16.1, `set_reasoning_effort("max")`): 짧은 프롬프트와 871-token 프롬프트(43 token)에서 동일.
- Teacher-forced trace(`compare_logit_traces.py --decided 1`): 4 x 256 token, decided position 578개 중 0개 불일치, perplexity +0.19%. 긴 프롬프트와 continuation: p99 / max logit delta 0.75 / 2.0, perplexity +0.34%, 549개 중 1개 불일치(row 866, 3.4 참조).
- `mlxcel-server` streaming: thinking을 켜면 reasoning은 `reasoning_content`, `42`는 `content`에 들어간다. `chat_template_kwargs.enable_thinking=false`면 reasoning 없이 `42`를 낸다.
- Raon unit test, clippy, fmt, `verify-*` contract target, 로컬 전체 `make verify` gate 통과.
