# PR #2744: GB10의 CUDA lib 테스트 실패 11건 해결

**날짜**: 2026-10-30 (머지 전에 작성한 보고서이며, 작성 시점에 PR은 열린 초안 상태)
**상태**: 열린 PR. CUDA(GB10, sm_121)에서만 검증했고, Metal 호스트가 없어 항목 1, 2, 4에 요청된 Metal 실행은 하지 못함
**언어**: Rust, C++ (CUDA 오버레이), Python, JSON, Markdown
**위험도**: 중간 (공유 경로의 프로덕션 변경 두 건: Qwen4-Exp gated-delta 정밀 정책과 SAM 3D Body 캐스트. 모든 CUDA 빌드의 conv와 컴파일 커널 동작을 바꾸는 CUDA 전용 MLX 오버레이 두 건)

## 요약

PR #2744(#2739 해결, 에픽 #2348)는 에픽의 머지 후 게이트가 GB10(aarch64, sm_121, `--profile test-fast --features cuda`, `--test-threads=1`)에서 CUDA lib 스위트를 처음 실행하며 발견한 lib 테스트 실패 11건을 해결한다. `#[ignore]`를 추가하지 않았고, 측정한 CUDA 값을 기록하지 않은 허용 오차 완화도 없다.

원인은 네 가지다. 첫째, mlxcel의 CUDA 전용 bf16+f32 승격 오버레이가 bf16을 돌려주면서 프로덕션 코드가 upstream MLX와 달라진 경우(Qwen4-Exp gated-delta 상태, SAM 3D Body의 linear, norm, layer scale, RoPE, 백본 stem, SAM 3D Objects 헤드 norm). 둘째, ROCm이나 x86_64에서 캡처한 백엔드 의존 오라클을 upstream `mlx` 0.32.4 CUDA에서 측정한 값으로 CUDA 키를 분리한 경우(SAM 3D Objects의 modulate, tanh GELU, bicubic resize, Sapiens2 RoPE) 또는 호스트 아키텍처로 분리한 경우(MHR ULP 한계). 셋째와 넷째는 새 MLX 소스 오버레이로 고친 실제 CUDA 백엔드 버그 두 건이다. 실행 계획이 없는 cuDNN conv는 이제 gemm conv로 대체되고, 융합된 `AsType`은 eager 캐스트를 사용해 complex64에서 실수로의 변환이 컴파일된다.

Kokoro는 테스트만의 문제가 아니라 프로덕션 버그다. `src/lib.rs`가 `MLX_ENABLE_TF32=0`을 기본값으로 두기 때문에 GB10에서 Kokoro 합성이 일반 사용 중에도 예외를 던졌다.

## 1. 문제 정의

이슈 #2739는 항목 8개, 테스트 11개를 나열한다. 이미지 생성 코드나 #2348이 바꾼 코드와는 무관하며, CUDA lib 스위트를 이 테스트들에 대해 실행한 적이 없었을 뿐이다(#2613의 "CUDA 데이터 포인트 없음", #2631의 Metal 전용 검증, Sapiens2 테스트의 "CUDA는 캡처가 없고 키가 없음"). 모두 재실행과 `MLX_ENABLE_TF32=0`에서도 재현되었다.

dtype 실패는 `overlays/cuda/mlx/dtype.cpp`에서 온다. 이 오버레이는 `copy_v` 커널 오버헤드를 피하려고 bf16과 f32 연산을 bf16으로 승격한다. #2087과 #2093의 규약이 적용된다: 오버레이는 유지하고, f32가 필요한 프로덕션 코드에서 명시적으로 캐스트하며, 승격만 고정하는 테스트에서만 `cfg!(feature = "cuda")`로 분기한다.

## 2. 변경 요약

파일 19개, +1278 / -36, 커밋 3개(`06d145af` 주 수정, `d9493e97` origin/main `060b5051` 병합, `2862e24d` conv 대체 범위 축소).

| 항목 | 영역 | 파일 | 변경 |
|---|---|---|---|
| 1 | qwen4_exp 상태 | `src/models/gated_delta.rs` | 비 Metal의 T=1 및 순차 경로에서 정밀 정책이 q, k, v, g, beta와 상태를 f32로 캐스트하고 출력을 쿼리 dtype으로 되돌림. `gated_delta_update_precise`만 쓰는 `compute_g_float32` 추가. |
| 2 | sam3d_body | `sam3d_body/layers.rs`, `backbone.rs`, `rope.rs`, `dense/mod.rs` | `Linear`, `Norm`, `LayerScale`, `apply_rope`, 백본 stem이 upstream이 승격하는 dtype으로 명시적 캐스트. 가중치는 bf16 유지. |
| 3 | MHR 기하 | `sam3d_body/mhr_tests.rs` | 비트 단위 일치는 x86_64 Linux에서만 요구하고 나머지 호스트는 2 ULP 한계(GB10 측정값 문서화). |
| 4 | Objects 레이어 | `mlx_cxx_objects.cpp`, `sam3d_objects/layers_tests.rs`, `cuda_oracle_tests.rs` | 헤드 norm f32 캐스트(오버레이 버그). modulate, residual, tanh GELU bf16 격자는 CUDA 키로 분리. |
| 5 | Sapiens2 RoPE | `sapiens2/backbone_tests.rs` | CUDA를 ROCm 값 `-4.559755325317383e-6`과 같은 키로 분리. |
| 6 | bicubic resize | `interpolate_objects_tests.rs` | CUDA 키 분리. f32 값 48개 중 12개가 1~3 ULP 차이. |
| 4/6 데이터 | 픽스처 | `tests/fixtures/sam3d_objects/cuda-oracle.json`, `generate_cuda_oracle.py`, `README.md` | upstream mlx 0.32.4 CUDA에서 캡처한 희소 오버라이드(키별 인덱스와 값). |
| 7 | Kokoro | `overlays/cuda/mlx/backend/cuda/conv.cpp`, `conv_no_plan_tests.rs` | cuDNN 실행 계획 없음 시 gemm conv로 대체. |
| 8 | complex64 캐스트 | `overlays/cuda/mlx/backend/cuda/compiled.cpp`, README | 융합 `AsType`이 eager `cast_to<T>()`를 생성. README 파일 수 27에서 29로. |

## 3. 기술적 선택과 그 이유

### 3.1 항목 1: 상태만이 아니라 피연산자도 캐스트

**배경:** 비 Metal에서 T=1은 `preserve_precision`을 무시하는 융합 디코드 스텝으로 가고, 오버레이 아래에서 `add(state_f32, k * delta_bf16)`는 bf16이다. 상태 dtype만 문제라는 이슈의 가설은 불완전했다. `exp(A_log)`(f32)와 bf16 softplus의 곱이 bf16이므로 `compute_g`도 CUDA에서 bf16 게이트를 돌려주었다.

**결정:** `PreciseOperands` 헬퍼가 스텝 전에 q, k, v, g, beta를 f32로 캐스트하고, 상태는 f32이며, `cast_output`이 출력을 쿼리 dtype으로 돌려준다. `compute_g_float32`는 softplus 항을 먼저 승격한다. 각 캐스트는 정확하며 Metal에서 MLX가 스스로 삽입하는 캐스트와 같으므로 Metal은 구조상 비트 단위로 동일하다. Qwen3Next, Qwen3.5, KimiLinear는 `preserve_precision = false`로 `compute_g`를 호출하므로 변하지 않는다.

**배제한 대안:** 모두를 위해 `compute_g`를 바꾸는 것. 실패하는 테스트가 없는 다른 모델 계열 세 곳의 CUDA 동작을 바꾸게 된다.

### 3.2 항목 4~6: 프로덕션을 먼저 고치고, 강제될 때만 오라클을 키로 분리

인수 기준에 따라 f32 중간값을 먼저 시도했다. ROCm 오라클을 재현하지 못했고(modulate 32개 중 13개 차이, 컴파일 그래프는 11개), 그래서 modulate, residual, tanh GELU bf16 격자를 CUDA 키로 분리했다. Objects 헤드 norm은 달랐다. f32 활성값에 bf16 감마를 곱하던 오버레이 유발 버그였고 `mlx_cxx_objects.cpp`의 f32 캐스트로 고쳤으며, 그 뒤 픽스처는 변경 없이 일치한다.

CUDA 값은 고정 오라클 위의 희소 오버라이드로 저장된다(`cuda_oracle_tests.rs::keyed`). 원소 수와 인덱스/값 길이를 단언하므로 낡은 픽스처는 즉시 실패한다. 생성기는 출처(mlx 0.32.4 CUDA, mlx-vlm 0.7.8, GB10 12.1, 드라이버 580.178.04, `MLX_ENABLE_TF32=0`)를 기록한다. 테스트 규약은 CUDA에서는 upstream CUDA와 비트 단위로 같고, 그 외에는 고정 오라클과 같아야 한다는 것이다.

### 3.3 항목 3: 아키텍처로 분리하고 한계는 유지

GB10 측정: `body_params` 266개 중 41개, `hand_params` 54개 중 6개가 정확히 1 ULP 다르고 나머지 필드는 비트 단위로 같다. 원인은 CUDA가 아니라 aarch64 glibc의 `atan2`/`sin`/`cos` 반올림이다. 조건을 `target_os = "linux"`에서 `all(linux, x86_64)`로 바꿨고 2 ULP 한계는 유지한다(M1 Ultra 2, GB10 1).

### 3.4 항목 7: 좁은 cuDNN 대체

**근본 원인:** TF32를 끄면 MLX가 cuDNN의 텐서 코어 계획을 제외한다. sm_121에서는 f32 dilated conv1d(Kokoro 생성기 형상 `x=(1,1200,256) w=(256,3,256) d=3`, 256과 128 채널의 `k=11 d=5`)에 계획이 남지 않고, upstream `graph.build()`가 "No execution plans support the graph"를 던진다. upstream mlx CUDA로 재현했고 TF32를 켜면 통과한다.

**결정:** 오버레이는 `build()`가 `GRAPH_EXECUTION_PLAN_CREATION_FAILED`로 실패할 때만 `nullopt`를 반환해 upstream의 기존 `gemm_conv` 경로를 타며, 형상별로 `CONV_FALLBACK`으로 캐시된다. 첫 커밋은 더 넓게 잡았고 `2862e24d`가 범위를 좁혀, 다른 cuDNN 빌드 오류는 upstream처럼 그대로 던지고 지원되는 conv는 기존 cuDNN 경로를 탄다.

**배제한 대안:** `MLX_ENABLE_TF32=0` 기본값 변경(모든 모델의 수치가 바뀜), Kokoro conv 레이아웃 재작성(한 계열만 고치고 다른 f32 dilated conv는 그대로 노출).

### 3.5 항목 8: ROCm 수정을 그대로 반영

upstream CUDA 컴파일 코드 생성이 `Abs` 뒤 `AsType`에 `static_cast<float>(complex64_t)`를 만들었다. 오버레이는 `cast_to<T>()`(실수부를 유지하고 bool은 두 부분을 모두 검사하는 eager `CastOp`)를 생성하고 JIT include에 `cast_op.cuh`를 추가한다. #2613의 ROCm `compiled.cpp`에 대응하는 CUDA 버전이다.

## 4. 검증

모두 GB10에서 `gpu-lock` 아래 `--test-threads=1`로 실행했다.

- 대상 테스트 11개 모두 통과. Kokoro 테스트 2개는 `models/kokoro-82m`이 있는 상태에서 실행.
- mlxcel lib 필터 모두 통과: `vision::dense` 193, `models::qwen4_exp` 70, `models::gated_delta` 10, `models::qwen3_next` 14, `models::qwen3_5` 32, `kimi` 173, `models::kokoro` 10. mlxcel_core: `compiled_dtype_cast_tests` 10, `compiled` 46, `conv` 16, `conv_no_plan_tests` 1(Kokoro 형상 세 개를 CPU 스트림과 비교).
- 항목 1 회귀 방지: `qwen3.5-0.8b-4bit`(`MLX_ENABLE_TF32=0`)의 로짓 추적에서 새 오버레이 없는 main의 `gated_delta.rs`와 이 브랜치가 바이트 단위로 같다(256 위치, 두 쪽 모두 ppl 28.6184).
- `mlxcel`과 `mlxcel-core`의 clippy `-D warnings`, `make verify-*`(versions, kernel-dtype-keys, kernel-port-dispatch, llama-compat, rocm-overlay, python-tooling, env-var-docs, fmt), `check_binary_assets.py` 통과. origin/main `060b5051` 병합과 conv 대체 범위 축소 뒤 최종 헤드에서 전체를 다시 실행했다.
- 새 오버레이 두 개가 빌드 시 `mlx-src`로 복사됨을 grep으로 확인했다.

## 5. 학습 포인트

- CUDA 승격 오버레이는 upstream이 f32인 곳에서 bf16을 만드는 상시적 원인이다. CUDA dtype 단언이 실패하면 커널을 의심하기 전에 bf16과 만나는 f32 피연산자를 찾고, 상태만이 아니라 모든 피연산자(여기서는 게이트)를 확인한다.
- TF32 끄기는 cuDNN에서 중립적인 스위치가 아니다. 계획 후보를 제거하며, 형상에 따라 후보가 하나도 남지 않을 수 있다. 오류 문구만으로는 TF32 기본값이 아니라 cuDNN을 가리킨다.
- 대체 조건은 실패에 맞춘다. 계획 생성 오류 코드만 잡으면 다른 cuDNN 진단은 그대로 유지된다.
- 다른 백엔드에서 캡처한 고정 오라클은 호스트 의존 값이다. 희소 오버라이드와 기록된 생성기는 텐서 전체를 복제하지 않고도 CUDA 키를 감사 가능하게 유지한다.
- 값싼 설명(f32 중간값)을 먼저 배제하고 실패했음을 기록해 다음 독자가 재시도하지 않게 한다.

## 6. 한계와 미검증 사항

- Metal 실행 없음. 이슈는 항목 1, 2, 4에 대해 Metal에서 `vision::dense`와 `models::qwen4_exp`를 요구했다. 안전성 근거는 구조적이다. 각 변경은 Metal의 비패치 승격 표가 이미 만드는 dtype으로의 명시적 `astype`이며, Metal 정밀 gated-delta 커널 경로는 변경된 대체 코드에 도달하지 않는다.
- 오버레이는 `MLX_BUILD_CUDA`에서만 컴파일되므로 다른 백엔드에서는 실행되지 않았다.
- 이 호스트에는 qwen4_exp 체크포인트가 없다. 항목 1의 증거는 패리티 및 합성 테스트와 Qwen3.5 로짓 추적이며, 다른 계열이 변하지 않았음을 보이지만 Qwen4-Exp 종단 실행은 아니다.
- CUDA 오라클 값은 GPU 한 대(GB10, 드라이버 하나)에서 나왔다. 다른 CUDA 아키텍처는 반올림이 다를 수 있고 별도 키가 필요하다.
- aarch64 MHR 한계는 측정한 호스트 두 곳에 근거한다.
- 선택 항목인 step3p7 TF32 메모(`tests/step3p7_parity.rs:566`)는 다루지 않았다.

## 참고

- 이슈 #2739(해결), 에픽 #2348, PR #2613(ROCm 컴파일 캐스트), PR #2631, #2580, #2578, #2550, #2475, #374(원인이 된 변경), #2087과 #2093(승격 오버레이 규약).
- 관련: PR #2740, PR #2730 보고서.
- Upstream: `mlx` 0.32.4 CUDA, mlx-vlm 0.7.8.
- 파일: `src/lib/mlxcelverse/overlays/cuda/README.md`, `tests/fixtures/sam3d_objects/README.md`.
