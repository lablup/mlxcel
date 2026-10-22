# PR #2552: 명시적 GELU 선택자 추가와 체크포인트가 tanh를 선언한 곳의 tanh 적용

**날짜**: 2026-10-23
**상태**: PR 열림; Refs #2423 (이슈는 열린 상태로 유지, 남은 패밀리는 #2548, mlx-vlm 프롬프트 길이 불일치는 #2547)

## 요약

이슈 #2423은 `mlxcel_core::gelu_approx`가 정확한 erf GELU를 계산하는 반면 MLX의 `nn.gelu_approx`와 `nn.GELU(approx="precise")`는 tanh 근사라는 점을 발견했다. 그래서 `gelu_approx`를 호출하는 여러 비전 포트가 mlx-vlm 레퍼런스와 다른 공식을 계산하고 있었다. PR #2552는 `from_hf_name`을 갖춘 `GeluActivation { Erf, Tanh, TanhF32, TanhF32Compiled, Fast }`를 추가하고, 개별 tanh 복사본 여섯 개를 각자의 정밀도를 유지한 채 여기에 통합하고(통합 테스트에서 비트 동일, BERT는 f32 입력 기준), erf가 `approx="precise"`와 같다고 주장하던 주석을 바로잡고, 체크포인트와 mlx-vlm 포트가 모두 tanh를 쓰는 비전 패밀리를 전환한다. 대상은 MiniCPM-V 4.6 인코더 MLP와 window merger, Gemma 3 VLM과 PaliGemma의 SigLIP 타워, Gemma 4 비전 GeGLU(DiffusionGemma와 EmbeddingGemma2도 이 경로를 쓴다)이다. 핵심 발견은 부정적 결과다. teacher-forced logit-trace 게이트는 측정한 어떤 패밀리에서도 erf와 tanh를 구별하지 못했다(결정된 위치의 불일치 0건, 로짓 이동은 최대 bf16 양자 크기인 0.125~0.5, Gemma 3에서 한 번 1.75). 그래서 각 패밀리는 체크포인트가 선언한 활성화 함수로 결정했고 mlx-vlm으로 교차 확인했다. bf16 안정성 재검증에서는 어느 arm에서도 NaN이나 Inf가 나오지 않았다. 예전 주석이 말한 NaN은 tanh가 아니라 `power(x, 3)`에서 나왔다. f32 tanh를 eager로 평가하면 Gemma 3 이미지 prefill이 +17.0% 늘었고, 컴파일된 커널로 바꾸자 pooled 기준 +2.8%가 되었다. 부호가 일관된 작은 실제 비용이다. 이 PR은 #2538 PTX 캐시 오버레이의 NAME_MAX 버그도 고친다. 이 버그 때문에 이름이 긴 커널은 캐시에 들어가지 못했다. 29개 파일, +1353/-178.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/lib/mlxcel-core/src/utils.rs`, `cpp/mlx_cxx_bridge.{cpp,h}`, `src/lib.rs` | `from_hf_name`, `in_f32`, `in_f32_compiled`, `apply`를 가진 `GeluActivation` enum; C++ `gelu_tanh_approx` 헬퍼(세제곱은 `x*x*x`) 위에 `gelu_tanh`, `gelu_tanh_f32`, `compiled_gelu_tanh_f32` 추가; `gelu_approx`는 erf 유지, 문서 수정 |
| `src/lib/mlxcel-core/src/utils_gelu_tests.rs` (신규) | 통합 테스트(f32/bf16/f16 비트 동일), 큰 \|x\|에서의 포화, 컴파일 대 eager 오차 한계 |
| `src/models/bert.rs`, `src/models/kokoro/ops.rs`, `src/vision/encoders/{molmo2,siglip,kimi_vl,qwen3_vl}.rs` | 개별 tanh 복사본을 각자의 정밀도 그대로 `TanhF32` 또는 `Tanh`로 통합 |
| `src/vision/config.rs`, `src/loading/vlm_gemma.rs`, `src/loading/vlm_siglip.rs` | `VisionHiddenActivation::SiglipDefaultTanh`와 `default_missing_hidden_act_to_siglip`, Gemma 3 VLM과 PaliGemma 로더만 호출 |
| `src/vision/encoders/minicpmv4_6.rs`, `minicpmo.rs` | 4.6 인코더 MLP는 `hidden_act`에서 결정, window merger는 tanh, downsample merger는 erf; MiniCPM-o는 erf 유지 |
| `src/vision/encoders/gemma4.rs` | 비전 GeGLU를 `hidden_activation`(기본값 `gelu_pytorch_tanh`)에서 결정 |
| `src/lib/mlxcel-xla/src/emitter/vision_config.rs` | XLA Gemma 3 비전 emitter도 누락된 키를 같은 방식으로 tanh로 해석 |
| `src/vision/encoders/internvl.rs`, `src/models/gemma3n.rs`, `deepseek_vl2.rs` | erf 유지, 명시화 또는 주석 |
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`, `tests/cuda_custom_kernel_disk_cache.rs` | PTX 캐시 임시 파일 이름을 suffix와 확장자로만 구성; 컴파일된 tanh 커널로 두 번째 프로세스 테스트 |
| `docs/gelu-call-sites.md`, `docs/benchmark_results/gelu-erf-vs-tanh-2026-10-23.md` (신규) | 약 110개 호출 지점과 고정된 업스트림 링크 목록; 측정 기록 |

## 2. 기술적 선택과 그 이유

### 2.1 근사가 아니라 공식에 이름을 붙인다

함정은 이름이다. 이 크레이트의 `gelu_approx`는 erf이고, MLX의 `nn.gelu_approx`는 tanh이며, MLX는 `"precise"`와 `"tanh"`를 모두 tanh로 매핑한다. 호출 이름을 그대로 옮긴 포트는 컴파일 시점에 아무 신호 없이 잘못된 공식을 얻는다. `GeluActivation`은 각 호출 지점이 erf, tanh, fast 중 무엇을 계산하는지 밝히게 하고, `from_hf_name`은 Hugging Face 이름을 `ACT2FN` 의미로 매핑한다(`gelu`는 erf, `gelu_pytorch_tanh`, `gelu_new`, `gelu_fast`는 tanh, `quick_gelu`는 fast). 그 밖의 이름에는 `None`을 돌려준다. `gelu_approx`는 호출 지점 대부분이 실제로 erf GELU를 포팅하므로 이름을 유지했고, 문서 주석이 이를 밝힌다.

### 2.2 게이트가 구별하지 못했으므로 체크포인트가 결정했다

mlxcel 안에서 같은 빌드와 같은 토큰 스트림으로 erf와 tanh를 바꿔도 어느 패밀리에서도 결정된 토큰은 바뀌지 않았다. 레퍼런스 선택에 대한 로짓 변화는 p50 0.000~0.250, 최대 0.031~1.750이었다.

| 체크포인트 | 결정된 위치 | 결정된 위치 불일치 | 로짓 변화 p50 / 최대 |
|---|---|---|---|
| internvl3-1b | 38 | 0 | 0.000 / 0.031 |
| minicpm-v-4.6-bf16 | 47 | 0 | 0.125 / 0.500 |
| gemma-3-4b-it-4bit | 50 | 0 | 0.250 / 1.750 |
| paligemma2-3b-6bit | 25 | 0 | 0.008 / 0.188 |
| gemma-4-e2b-it-4bit | 45 | 0 | 0.125 / 0.500 |
| gemma3n-e2b-4bit (텍스트, sparse 레이어) | 244 | 0 | 0.125 / 1.000 |

float32 mlx-vlm 레퍼런스도 같다. 레퍼런스 자체의 비전 GELU를 erf로 바꿔도 결정된 토큰은 바뀌지 않았다. 이 레퍼런스와 비교하면 MiniCPM-V 4.6(39개 중 2개)과 Gemma 4(46개 중 1개)에서 결정된 위치 불일치가 나타나지만, erf, tanh-f32, tanh-입력 dtype arm에서 모두 같다. 따라서 이 불일치는 GELU가 아니라 4-bit 가중치, bf16 활성값, 프롬프트에서 온다. mlxcel 대 레퍼런스 차이는 1~7 로짓이고, GELU 교체는 bf16 한 단계 수준이다. perplexity는 양쪽 방향으로 움직였고 두 tanh 정밀도 사이의 차이가 tanh와 erf의 차이만큼 컸다. 측정 기록은 이를 bf16 노이즈로 보며, 어느 쪽의 근거도 아니다.

어떤 측정도 두 공식 중 하나를 인증할 수 없었으므로, PR은 체크포인트의 `config.json` 또는 Hugging Face 코드와 mlx-vlm 포트가 모두 tanh라고 말하는 곳만 전환했다. mlx-vlm이 Hugging Face와 다르면 체크포인트를 따른다. InternViT는 `hidden_act = gelu`(erf)를 선언하고 MiniCPM-V 4.6 downsample merger는 `nn.GELU()`(erf)이므로, mlx-vlm이 tanh를 계산하더라도 둘 다 erf를 유지한다. Gemma3n sparse 레이어는 업스트림에서 tanh지만 여기서는 erf로 남겼다. 컴파일된 `gelu_topk` 커널, fused `gemma3n_mlp_forward` sparse 분기, XLA Gemma3n host oracle을 함께 바꿔야 하기 때문이다.

### 2.3 누락된 키는 로더별로 해석한다

Gemma 3와 PaliGemma 설정에는 `vision_config.hidden_act`가 없다. Hugging Face는 클래스 기본값과 같은 키를 저장하지 않고, `SiglipVisionConfig`의 기본값은 `gelu_pytorch_tanh`이므로 이 타워들은 tanh로 학습되었다. CLIP 클래스 기본값은 tanh가 아니라 `quick_gelu`이므로 다른 로더의 serde 기본값은 erf로 둔다. `default_missing_hidden_act_to_siglip`는 정확히 Gemma 3 VLM과 PaliGemma 로더만 호출한다. Aya Vision과 Granite Vision 3.x도 키를 생략하지만 #2548로 남겼다. XLA emitter는 `model_type == "gemma3"`인 경우만 바꾼다. 해석 결과는 `GeluPytorchTanh`가 아니라 별도의 `SiglipDefaultTanh` variant다. 그래서 `gelu_pytorch_tanh`를 명시한 경우는 eager 비트 동일 경로를 유지하고, 해석된 경우는 컴파일된 커널을 쓴다(2.5).

이제 MiniCPM-V 4.6 인코더와 Gemma 4 비전 타워에서 GELU 이름이 아닌 `hidden_act` 또는 `hidden_activation`은 조용히 erf를 쓰지 않고 로드를 실패시킨다. 이 패밀리의 로컬 설정은 모두 `gelu_pytorch_tanh`를 선언하거나 키를 생략한다.

### 2.4 bf16 안정성: NaN의 원인은 tanh가 아니라 `power`였다

예전 `gelu_approx` 주석은 4096 패치의 bf16 SigLIP MLP에서 tanh가 NaN을 냈다고 적었다. 모든 비전 MLP 활성화에 probe를 걸어 다시 측정한 결과 일곱 개 arm 모두 NaN 0, Inf 0이었다. 여기에는 gemma-3-4b-it-4bit의 bf16 타워 27 x [1, 4096, 4304](최대 |입력| 161, 페이지 이미지에서 177), minicpm-v-4.6-bf16(최대 214), f16인 paligemma2-3b-6bit(최대 70.4)이 포함되고, tanh는 입력 dtype과 f32 양쪽으로 평가했다. NaN은 `power(x, 3.0f)`에서 나왔다. MLX는 이를 `exp(3*log(x))`로 평가하므로 음수 x에서 정의되지 않는다. `gelu_tanh_approx`는 `x*x*x`로 세제곱한다. bf16의 지수 범위는 f32와 같으므로 |x| = 214에서의 세제곱(약 1e7)은 오버플로와 거리가 멀다. f16에서는 |x|가 약 40을 넘으면 세제곱이 오버플로하지만 tanh가 +-1로 포화하므로 출력은 NaN이 아니라 x 또는 0이 된다.

### 2.5 f32 tanh, 그리고 컴파일

두 정밀도 모두 유한했다. `TanhF32`를 고른 이유는 SigLIP `gelu_pytorch_tanh` 경로와 그 StableHLO 쌍이 이 형태로 검증되었고, PyTorch가 half 입력의 `gelu(approximate="tanh")`를 f32로 평가하기 때문이다(torch 2.14.1 CPU에서 12만 개 입력에 대해 f32 계산 후 캐스트한 결과와 비트 동일, 연산별 half 계산과는 약 15퍼센트에서 다름). mlx-vlm은 입력 dtype으로 평가한다.

eager로 평가하면 `TanhF32`는 원소별 연산이 약 11회로, f32 erf의 약 6회보다 많고, Gemma 3의 SigLIP은 이를 27 x [1, 4096, 4304]에 실행한다. 리뷰 단계 측정에서 erf 872 ms 대 `TanhF32` 1025 ms(중앙값, paired +151~+172 ms), 입력 dtype tanh 916 ms(+42 ms)였다. PR은 입력 dtype tanh로 내려가는 대신 같은 헬퍼를 upcast와 cast back까지 하나의 shapeless `mlx::core::compile` 영역으로 감싼 `compiled_gelu_tanh_f32`를 `TanhF32Compiled`로 노출하고, 전환된 패밀리만 이를 쓴다. 통합된 호출 지점은 대체한 복사본과 비트 단위로 고정된 eager `TanhF32`에 남는다. 컴파일된 커널은 곱셈-덧셈을 FMA로 합칠 수 있으므로, f32 출력은 4 f32 ulp, bf16/f16 출력은 해당 dtype의 1 ulp 한계로 검증한다. +-170까지의 이상값을 포함한 타워 형태의 bf16 [64, 4304] 입력에서는 eager 헬퍼와 비트 동일했고(275456개 lane 중 0개 차이), 다섯 개 smoke 체크포인트 모두에서 eager 빌드와 컴파일 빌드의 greedy 32 토큰 응답이 같았다.

이미지 prefill, 단일 바이너리, erf arm과 null arm은 `gelu`를 선언한 심볼릭 링크 체크포인트 복사본, arm 순서 회전, `MLX_ENABLE_TF32=0`, GB10:

| 패밀리 | n | erf 중앙값 | 컴파일 tanh 중앙값 | paired tanh - erf | null - erf | 결과 |
|---|---|---|---|---|---|---|
| Gemma 3 SigLIP | 16 | 859.5 ms | 878.7 ms | +23.8 ms, +2.8% (-11.8..+53.8); 블록별 +2.4%, +3.7% | -0.5 ms (-33.8..+34.8) | tanh 유지, 잔여 비용 미해명 |
| Gemma 3, eager `TanhF32` (이전 상태) | 8 | 869.3 ms | 1013.4 ms | +148.0 ms, +17.0% (+116.5..+174.7) | | 기준 |
| MiniCPM-V 4.6 | 6 | 910.8 ms | 897.1 ms | -14.0 ms, -1.5% (-36.1..+42.4) | +2.4 ms (-34.5..+31.6) | null 범위 안 |
| Gemma 4 vision | 12 | 661.7 ms | 677.9 ms | +13.2 ms, +2.0% (-6.0..+36.3) | +1.0 ms (-27.9..+24.8) | tanh 유지, 잔여 비용 미해명 |
| PaliGemma SigLIP | 6 | 995.5 ms | 987.6 ms | -15.1 ms, -1.5% (-37.5..+12.8) | +8.1 ms (-41.6..+20.3) | null 범위 안 |

Gemma 3와 Gemma 4의 잔여 비용은 모든 블록에서 부호가 같고 null arm은 0을 중심으로 분포하므로, 노이즈가 아니라 작은 실제 비용이다. pooled 기준으로는 3% 수용선 아래지만, Gemma 3 블록 하나(+3.7%)는 단독으로 그 선을 넘었다. 캐시된 모듈을 보면 세제곱과 tanh 꼬리가 별도 커널이므로, CUDA에서 컴파일 영역은 아직 하나 이상의 fused 커널로 실행된다. 완전히 하나로 합친 커널은 #2548의 항목이다.

### 2.6 PTX 캐시 임시 파일을 NAME_MAX 아래로

첫 컴파일 측정에서 커널과 무관한 고정 비용이 보였다. 모든 프로세스가 fused 커널을 NVRTC로 다시 컴파일했기 때문에 Gemma 3 prefill이 캐시 없이 2505 ms, 캐시 후 869 ms였다. #2538 이후 `jit_module.cpp` 오버레이는 각 캐시 파일을 `<module name>.txt.tmp<pid>_<tick>`을 거쳐 썼다. 컴파일된 tanh 커널의 모듈 이름은 234자이고 `get_ptx_path`는 이를 하나의 경로 요소로 유지하므로, 임시 파일 이름이 255바이트 NAME_MAX를 넘어 open이 ENAMETOOLONG으로 실패했고 캐시 항목은 기록되지 않았다. 모듈 이름이 대략 229~245자인 모든 커널이 영향을 받았다. 이제 임시 파일 이름은 고유 suffix와 확장자로만 만들고 최종 파일과 같은 디렉터리에 두므로 `rename(2)`는 원자적으로 유지된다. GELU에 한정되지 않고 이 길이 범위의 모든 커널에 해당하는 수정이다. `tests/cuda_custom_kernel_disk_cache.rs`는 이 커널을 두 번째 프로세스에서 실행해 아무것도 컴파일하지 않을 것을 요구하며, 이전 오버레이에서는 실패한다.

### 2.7 리뷰 중 변경

- `715ae4a8`: 구현 리뷰는 PR이 eager `TanhF32`의 prefill 비용(HIGH)과 GELU 이름이 아닌 활성화가 이제 로드를 실패시킨다는 점(MEDIUM)을 밝히지 않았고, `gemma4_activation_tests.rs`가 encoder가 없는 gemma-4-12b-it-4bit을 언급한다는 점(LOW)을 지적했다. 비용 절, 본문, 테스트 주석을 고쳤다.
- `ecabb220`: 컴파일된 f32 tanh 커널과 `TanhF32Compiled`, 전환된 패밀리의 eager 경로를 대체.
- `a78ffcf0`: `ecabb220`을 측정하다 발견한 NAME_MAX 수정.
- `f2617730`, `3734c1b4`: null arm을 포함한 패밀리별 컴파일 A/B, Gemma 3 두 블록 모두 기록. 리뷰의 미결 항목이던 PaliGemma, MiniCPM-V 4.6, Gemma 4 비용 미측정을 해소했다.
- `316fea12`에서의 보안 및 성능 리뷰는 CRITICAL, HIGH, MEDIUM 항목이 없었고 수정 커밋도 없었다.

## 3. 검증

PR 테스트 계획과 보안 리뷰 기준, GB10(kernel 7.0.0-1019-nvidia, driver 580.178.04, `MLX_CUDA_ARCHITECTURES=121`, `test-fast` + `cuda`, `MLX_ENABLE_TF32=0`, `--test-threads=1`, `gpu-lock` 하에서), `316fea12`:

| 검사 | 결과 |
|---|---|
| `-p mlxcel-core --lib gelu` | 24개 통과, 컴파일 대 eager 오차 한계 포함 |
| `ffi_tests::shapeless_compile_audit_harness` (`MLXCEL_SHAPELESS_COMPILE_AUDIT=1`) | `compiled_gelu_tanh_f32` PASS, 시그니처 3개 warm |
| `--test cuda_custom_kernel_disk_cache` | 이전 오버레이에서 실패, 수정 후 1개 통과(자식 프로세스는 아무것도 컴파일하지 않음, `.tmp` 잔여 파일 없음) |
| `-p mlxcel --lib` 필터 | siglip 39, minicpmv4_6 14, gemma4::activation 2, vision::config 2 통과; 브랜치 초기에는 internvl, molmo2, qwen3_vl, bert, kimi, kokoro, moonvit, `-p mlxcel-xla --lib vision_config emitter::vision`도 실행 |
| clippy `-D warnings`(mlxcel-core, mlxcel lib+tests), `cargo fmt --all --check` | 깨끗함 |
| CLI `--image` smoke | minicpm-v-4.6-bf16, gemma-3-4b-it-4bit, paligemma2-3b-6bit, gemma-4-e2b-it-4bit, internvl3-1b가 세 도형을 설명; eager 빌드와 컴파일 빌드의 텍스트 동일 |
| 서버 smoke (`/v1/chat/completions`) | 같은 체크포인트로 이전 커밋에서 실행; DiffusionGemma는 CLI만 |

통합 테스트는 SigLIP, Molmo2, Qwen3-VL, Kimi 복사본을 연산 단위로 재현하고 f32, bf16, f16에서 비트 동일을 확인한다. BERT 복사본은 f32 입력에서 f32 oracle과 같다. half 입력에서는 예전 복사본이 결과를 f32로 남겼고 선택자는 이제 원래 dtype으로 캐스트한다. 로컬 BERT 설정 중 tanh 분기를 타는 것은 없다.

## 4. 미검증 및 미해결 항목

- **이슈는 열린 상태:** #2423의 기준 3, 4, 6은 측정한 패밀리에서만 충족된다. 나머지 약 20개 불일치 패밀리(Gemma3n sparse, Gemma 4 bf16 dense MLP와 drafter, Gemma 4 TP gate, RecurrentGemma, MiniCPM-o, Aya/Granite SigLIP, CLIP `quick_gelu` 기본값, Phi, GPT-2, Moondream, LFM2-VL, DeepSeek-VL2 등)는 각각 하나의 수용 항목으로 #2548에 있다.
- **런타임 간 비교 무효:** InternVL(262 대 268), MiniCPM-V 4.6(78 대 76), Gemma 3(273 대 274)에서 `# prompt_tokens`가 mlx-vlm과 달라 해당 mlxcel 대 mlx-vlm 결과는 읽을 수 없다. #2547에서 추적한다. Gemma 4는 일치한다(275).
- **잔여 prefill 비용:** Gemma 3 pooled +2.8%(블록 하나 +3.7%)와 Gemma 4 +2.0%는 더 조사하지 않았다. MiniCPM-V 4.6 erf arm은 하드코딩된 tanh window merger를 그대로 실행했으므로 그 행은 차이를 약간 과소평가한다.
- **실행하지 않음:** Metal(컴파일 경로와 오버레이는 Metal에서 변경 없음), EmbeddingGemma2(로컬 체크포인트 없음), 서버를 통한 DiffusionGemma(기존부터 이미지 입력 거부), 워크스페이스 전체 test와 clippy.
- **보안 리뷰 LOW 항목, 변경하지 않음:** 크래시로 남은 고아 파일은 이제 모듈 이름 없는 dotfile이고 캐시를 정리하는 주체가 없다. `std::ofstream`은 임시 파일 이름의 심볼릭 링크를 따라간다(업스트림도 동일, `O_CREAT | O_EXCL | O_NOFOLLOW`로 막을 수 있음). JSON `null`인 `hidden_act`(MiniCPM-V 4.6)나 `hidden_activation`(Gemma 4 비전)은 `VisionConfig`와 달리 기본값으로 돌아가지 않고 로드를 실패시킨다.
- **리뷰 LOW 항목, 유지:** BERT의 `Activation`은 `relu`도 매핑하므로 `GeluActivation::from_hf_name` 대신 자체적으로 GELU 이름을 해석한다.
- **통합하지 않음:** `gemma3_backbone.rs`(Nemotron VoiceChat TTS)는 여전히 `power(x, 3)`로 tanh를 계산하며 프로덕션에서 쓰이고, `nemotron_codec`은 자체 erf를 유지한다. 통합하면 수치가 바뀐다.

## 5. 학습 포인트

- 결정된 위치에 대한 teacher-forced 게이트는 한 arm이 얼마나 벗어날 수 있는지 한계를 줄 뿐, 두 arm이 모두 bf16 한 단계 안에 있으면 어느 쪽이 옳은지 인증하지 못한다. 게이트가 구별하지 못하면 그 사실을 발견으로 기록하고 학습 설정으로 결정해야 하며, perplexity 노이즈를 신호로 읽어서는 안 된다.
- 레퍼런스 구현은 활성화 함수에 대해 정답이 아니다. mlx-vlm은 InternViT와 MiniCPM-V 4.6 downsample merger에서 체크포인트의 Hugging Face 활성화와 다르다. 불일치를 버그로 보기 전에 체크포인트의 `config.json`과 HF 클래스 기본값을 확인해야 한다.
- Hugging Face는 클래스 기본값과 같은 설정 키를 생략하므로, `hidden_act`가 없다는 것은 "클래스 기본값"을 뜻하고 그 기본값은 클래스마다 다르다(SigLIP은 tanh, CLIP은 `quick_gelu`). 공유 serde 기본값이 아니라 로더별로 해석해야 한다.
- 수치가 고정된 헬퍼에 더 빠른 경로가 필요하면, 새 사용처를 위한 별도 variant를 추가하고 고정된 쪽은 건드리지 않는다. 통합 테스트는 비트 동일을 유지하고 새 경로는 명시된 ulp 한계로 검증한다.
- 캐시 버그는 그 조건에 맞는 길이의 커널이 등장할 때까지 드러나지 않을 수 있다. 임시 파일 이름은 최종 파일 이름과 독립적으로 정하고, 캐시 적중은 두 번째 프로세스에서 테스트해야 한다.
