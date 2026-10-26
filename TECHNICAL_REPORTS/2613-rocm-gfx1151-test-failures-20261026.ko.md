# PR #2613: verify-rocm의 gfx1151 테스트 실패 9건 해소

**날짜**: 2026-10-26
**상태**: 구현 및 gfx1151 로컬 검증 완료(draft PR, 아직 머지 전)
**언어**: C++(HIP JIT 프리앰블), Rust, Markdown
**위험도**: 낮음~중간(프리앰블 변경은 모든 ROCm fused 커널의 소스 텍스트에 들어간다. 실수 타입 경로는 바뀌지 않았다)

## 요약

PR #2613(이슈 #2608)은 gfx1151에서 `make verify-rocm`이 실패하던 테스트 9건을 해소한다. 9건 모두 macOS에서는 통과하므로 전부 ROCm 전용 실패다. 원인은 서로 무관한 세 가지였고, 각각 다른 방식으로 처리했다.

- **실제 백엔드 버그, 수정.** ROCm의 fused(compiled) 커널은 complex64를 전혀 다루지 못했다. 공용 codegen이 내보내는 `complex64_t` 타입을 hiprtc 프리앰블이 정의하지 않았기 때문이다. 이제 프리앰블이 이 타입과 산술 및 성분 추출 연산의 complex64 오버로드를 정의하고, float 전용 연산은 실수부로 조용히 계산하는 대신 빌드 단계에서 complex64를 거부한다. `mlxcel-core` 테스트 2건.
- **알려진 백엔드 공백, 게이트 처리.** ROCm에는 NVFP4 커널이 없고, MLX 네이티브 NVFP4 plane은 설계상 로드 시점에 거부한다(#1806). `mlxcel` 테스트 6건이 그런 plane을 무조건 로드했다. 새 테스트 헬퍼 `expect_native_nvfp4_load`는 ROCm에서 거부를 단언하고, 공백을 명시한 skip 메시지를 출력하며, 각 테스트에서 백엔드와 무관한 부분은 계속 실행되게 한다.
- **기준값이 허용 오차 경계에 걸린 테스트, 수정.** PhiMoE fused expert 테스트는 fused 커널을 bf16 그래프 경로와 2% 한도로 비교했다. 이 fixture는 bf16 반올림 오차를 약 2.2%로 증폭하므로, 부정확한 쪽은 bf16 기준값 자체였다. 이제 기준값은 f32로 계산한다.

## 1. 문제 정의

이슈 #2608은 `make verify-rocm-held` 실행에서 나온 실패 9건을 나열했고, 수정하지 않은 main에서도 재현되었다. 원인은 진단되지 않은 상태였다. 이슈는 각 테스트가 ROCm에서 통과하거나, 확인된 백엔드 공백이면 그 공백을 명시하는 skip으로 게이트되어야 하며, 각 원인을 PR에 기록하라고 요구했다.

이슈가 요청한 첫 진단 단계는 ROCm 전용 실패와 일반 회귀를 구분하는 것이었다. 현재 main의 hosted CI는 테스트 job을 돌리지 않으므로, PR은 대신 macOS Nightly verify 실행(414ae122의 39703647781)을 사용했다. 거기서 9건 모두 통과한다. CUDA 데이터는 없다.

### 1.1 그룹 B: complex64의 compiled `real` / `imag`

`compiled_dtype_cast_tests::compiled_{real,imag}_of_complex64_matches_eager`는 fused 커널이 아예 컴파일되지 않아 실패했다. `overlays/rocm/mlx/backend/rocm/compiled.cpp`의 공용 codegen은 complex64 값을 `complex64_t`로 부르고(`dtype_to_hip_type`), 상수를 `static_cast<complex64_t>(...)`로 쓰며, `AsType`에는 양방향 `static_cast`를 내보낸다. 그런데 hiprtc 프리앰블(`g_jit_includes`)에는 그런 타입이 없었고, hiprtc는 `<hip/hip_complex.h>`를 include할 수 없다. 그래서 complex64가 들어간 모든 fused 그래프가 "unknown type name 'complex64_t'"로 실패했다. 프리앰블의 `Real`, `Imag`, `Conjugate` functor도 실수 입력용 스텁(항등, 0, 항등)이었으므로, 타입만 정의했다면 복소 입력에서 틀린 값을 냈을 것이다.

### 1.2 그룹 A: NVFP4 로드

테스트 6건(`switch_layers_global_scale_tests` 3건, gemma-unified 로딩, Mistral 4, PhiMoE sanitize 각 1건)은 MLX 네이티브 NVFP4 `SwitchLinear` / `SwitchGLU` plane을 만들어 로드한다. ROCm에서 `GpuBackendKind::quant_mode_support(Nvfp4)`는 `ConvertTo(Affine)`이고, 변환되는 것은 ModelOpt 체크포인트뿐이며, `validate_quantization_mode_runnable`은 네이티브 plane을 로드 시점에 거부한다. ROCm의 `GatherQMM`에는 expert별 global scale을 받을 입력도 없다. 이 실패들은 설계된 거부이지 회귀가 아니다.

### 1.3 그룹 A: PhiMoE fused expert 커널

`phimoe_tests::fused_expert_kernel_keeps_sparsemixer_scores`는 fused 단일 토큰 expert 커널이 PhiMoE의 sparsemixer 점수를 재정규화하지 않는지 지킨다. gfx1151에서 `MLXCEL_FUSED_MOE_PARITY_CHECK=1`로 측정한 결과는 다음과 같다. fused와 전부 f32로 계산한 참값의 차이 0.17%, `gather_qmm` 그래프 체인과 참값 2.2%, fused와 그래프 2.2%. 이 fixture는 down projection에서 상쇄가 심하다(|activation| 약 176, |output| 약 12). activation만 bf16으로 반올림해도 출력이 0.7% 움직이고, bf16 gate/up/activation 체인 전체로는 몇 퍼센트 움직인다. `quantized_matmul` 하나는 f32와 bf16 반올림 수준(0.16%)으로 일치한다. 어느 커널도 틀리지 않았다. bf16 기준값이 2% 경계에 있었고, ROCm의 반올림이 이를 넘겼을 뿐이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| hiprtc 프리앰블(`overlays/rocm/mlx/backend/rocm/compiled.cpp`) | `hipFloatComplex`와 같은 레이아웃의 `struct alignas(8) complex64_t { float x, y; }`를 새로 정의했다. 실수 타입에서 오는 explicit 템플릿 생성자(실수부, 허수부 0), 실수 타입으로 가는 explicit 템플릿 변환(실수부), 두 성분 중 하나라도 0이 아니면 참인 비템플릿 explicit `operator bool`을 둔다. 자유 함수 `+ - * /`, 단항 `-`, `==`, `!=`도 있다. |
| 연산 functor | `Add`, `Subtract`, `Multiply`, `Divide`, `Square`, `Real`(float32), `Imag`(float32), `Conjugate`, `Abs`(`hypotf`, float 반환), `NaNEqual`(MLX CUDA / Metal 규칙)에 complex64 오버로드를 추가했다. 오버로드를 넣기 위해 `Abs`를 `UNARY_FLOAT_OP`에서 독립 struct로 옮겼다. |
| 실수 전용 가드 | 새 `MLX_ROCM_REAL_ONLY(T)` 매크로(`static_assert(!__is_same(T, complex64_t), ...)`). `UNARY_FLOAT_OP`(Exp, Log, Sqrt, Rsqrt 등), `Power`, `ArcTan2`, `Remainder`, `FloorDivide`의 float 분기, `LogAddExp`, `Sigmoid`, `Sign`, `Reciprocal`에 넣었다. |
| `LOCAL_FIXES.md` | 항목 59로 fork 측 수정임을 기록했다. 백엔드 파일이므로 `CORE_RESIDUAL.diff`는 그대로다. |
| `switch_layers.rs` | 새 `#[cfg(test)] pub(crate) fn expect_native_nvfp4_load<T>(loaded, prefix, what) -> Option<T>`. NVFP4가 네이티브면 unwrap하고 실패 시 `what`으로 panic한다. 아니면 로드가 성공했을 때 panic하고, 오류에 plane prefix와 `nvfp4`가 들어 있는지 단언한 뒤 #1806을 명시한 skip을 출력하고 `None`을 반환한다. |
| NVFP4 테스트 6건 | 각 로드가 헬퍼를 거친다. 이후 레이어를 쓰는 테스트는 `None`이면 `return`(gemma-unified 루프에서는 `continue`)하고, 로드가 마지막 단계인 테스트는 헬퍼만 호출한다. `inv_global_scale`로의 사이드카 스택 확인과 "left unstacked" 거부 확인은 모든 백엔드에서 계속 실행된다. |
| `vlm_gemma_unified_tests.rs` | 게이트된 로드 전에 weight map에서 `<prefix>.<NATIVE_GLOBAL_SCALE_LEAF>`를 직접 읽어 expert별 scale 값을 확인한다. 그래서 split의 출력은 ROCm에서도 여전히 검증된다. |
| `phimoe_tests.rs` | 그래프 기준값은 `x`를 f32로 바꾸고 `moe_weighted_sum`을 f32로 계산한다. 2% 한도는 그대로다. |

## 3. 기술적 선택과 그 이유

### 3.1 complex64 fusion을 끄지 않고 프리앰블을 고친다

대안은 ROCm에서 complex64 그래프를 fusion에서 빼고 eager로 돌리는 것이었다. 그러면 타입 누락이 성능 차이 뒤에 숨고, `Real` / `Imag` / `Conjugate` 스텁이 남아 나중에 문제를 일으킨다. 프리앰블에 타입을 정의하는 것은 codegen이 이미 가정하는 바와 맞고, struct가 `hipFloatComplex` 및 `mlx::core::complex64_t`와 같은 레이아웃(float 두 개, 8바이트 정렬)이므로 디바이스 버퍼를 변환 없이 읽고 쓴다.

### 3.2 모든 변환을 explicit으로 둔다

실수 타입에서 오는 생성자와 실수 타입으로 가는 변환이 모두 `explicit`이다. 그래서 `complex64_t`는 실수 피연산자의 오버로드 해석에 끼어들지 않고, 실수 타입 `Add{}(a, b)`가 새 `Add{}(complex64_t, complex64_t)` 오버로드와 모호해지지 않으며, 실수 타입 커널은 이전과 똑같이 컴파일된다. codegen의 `AsType`은 explicit 변환도 보는 `static_cast`를 쓰므로 캐스트는 계속 동작한다.

### 3.3 `bool`과 `NaNEqual`은 일반 경로가 아니라 MLX CPU 및 CUDA를 따른다

템플릿 `operator T()`는 실수부를 반환하며, 이는 complex에서 실수로 가는 캐스트에 대한 MLX 규칙이다. 이를 `bool`에 그대로 적용하면 순허수 값의 fused `astype(z, bool)`이 거짓이 된다. 비템플릿 `operator bool`이 템플릿보다 우선하며 두 성분을 모두 본다. 마찬가지로 float `NaNEqual` 템플릿은 `static_cast<float>`를 거쳐 실수부만 비교하므로, 복소 입력의 fused `array_equal(..., equal_nan=True)`가 허수부를 무시했을 것이다. 둘 다 리뷰 중에 발견되었다. 타입이 없을 때는 도달할 수 없었지만, 타입이 생긴 뒤에는 빌드 실패가 아니라 조용히 틀린 결과를 냈을 것이다.

### 3.4 지원하지 않는 복소 연산은 빌드 단계에서 실패시킨다

`complex64_t`가 `float`로 변환되면 `static_cast<float>`로 계산하는 모든 연산 템플릿이 complex64로도 컴파일되어 조용히 실수부만 계산한다. `MLX_ROCM_REAL_ONLY`는 이를 읽기 쉬운 메시지의 hiprtc 컴파일 오류로 바꾼다. 다루지 않는 연산의 큰 실패는 이 PR 전에 사용자가 겪던 동작(그래프 빌드 실패)과 같으므로, 이 가드는 변경 범위를 실제로 구현한 연산으로 좁힌다.

### 3.5 NVFP4 테스트는 `cfg(rocm)`이 아니라 capability로 게이트한다

헬퍼는 ROCm 빌드인지가 아니라 `quant_mode_support(QuantMode::Nvfp4) == Native`인지 묻는다. 다른 테스트도 이미 같은 방식으로 분기하며, 어떤 백엔드가 네이티브 커널을 얻거나 잃어도 조건은 맞게 유지된다. 비네이티브 백엔드에서 단순히 건너뛰지도 않는다. 로드가 실패해야 하고, 그 오류가 plane과 모드를 모두 명시해야 한다. 나중에 ROCm이 NVFP4 plane을 조용히 로드하기 시작하면(예를 들어 global scale을 버리는 방식으로 변환하면) 테스트가 실패한다.

### 3.6 백엔드와 무관한 확인은 계속 실행한다

ROCm에서 일괄 `#[ignore]`를 걸면 사이드카 스택 확인까지 빠진다. 이 확인은 순수한 weight map 변환이라 어디서나 돈다. `Option` 반환 덕분에 각 테스트는 로드된 레이어가 처음 필요한 지점에서 정확히 멈춘다. gemma-unified 테스트는 한 걸음 더 나가 로드 전에 weight map에 저장된 expert별 native scale을 확인하므로, 이 테스트가 지키려는 split 로직은 ROCm에서도 검증된다.

### 3.7 허용 오차가 아니라 PhiMoE 기준값을 바꾼다

한도를 2% 넘게 넓히면 모든 백엔드에서 테스트가 약해진다. 검증하려는 성질은 fused 커널이 점수를 재정규화하지 않는다는 것이고, 재정규화하는 커널(점수 합 1.5025)은 어떤 기준값과도 약 1.5배 차이가 난다. 기준값을 f32로 옮기면 기준값이 검증 대상인 fused 커널보다 참값에 가까워진다. 기준값으로서 맞는 방향이며, 2% 한도의 의미도 유지된다.

## 4. 검증

PR 본문 기준, gfx1151에서 모든 GPU 실행을 `scripts/rocm_gpu_guard.sh --hold` 아래에서 수행했다.

- `mlxcel-core` `compiled_dtype_cast_tests`: 7/7(이전에는 2건 실패). 더 넓은 `compiled` 필터: `mlxcel-core` 42/42, `mlxcel` 8건 통과 및 1건 ignore. 리뷰 커밋 후 다시 실행했다.
- `mlxcel` `models::switch_layers` 57건 통과(1건 ignore, 체크포인트 필요), `models::phimoe` 14/14, `loading::vlm::gemma_unified` 9/9, `models::mistral4` 2/2.
- `cargo clippy --profile test-fast --features rocm -p mlxcel -p mlxcel-core --lib --tests -- -D warnings`와 `cargo fmt --all -- --check`: 깨끗함.
- `verify-rocm-overlay` 단계: `rocm_overlay.py records`(59개 항목), `rocm_overlay_test.sh`, `allocator_ownership_test.sh`, `rocm_runtime_dir_test.sh`, `rocm_overlay.py drift`: 통과.
- 전체 `make verify-rocm-held` 게이트는 머지 전에 중앙에서 실행한다. 그 결과는 이 보고서에 포함되지 않는다.

## 5. 알려진 공백

- **complex64의 fused `abs`는 여전히 빌드에 실패할 가능성이 높다.** MLX의 `abs`는 출력 dtype이 complex64인 `Abs` 노드를 만든 뒤 float32로 캐스트한다(`ops.cpp`). codegen은 각 tape 값을 노드의 dtype으로 선언하므로 `complex64_t tmp = Abs{}(tmp_z);`를 내보낸다. 새 `Abs` 오버로드는 `float`를 반환하고, 복사 초기화는 explicit `complex64_t(float)` 생성자를 쓸 수 없으므로 이 줄은 컴파일되지 않을 것이다. MLX CUDA는 `Abs`에서 `T`(허수부 0인 complex64)를 반환한다. 이전에도 빌드에 실패했으므로 회귀는 아니고 크게 실패하지만, 지금의 `Abs` 오버로드는 `LOCAL_FIXES.md` 항목 59가 설명하는 동작을 내지 못한다. `complex64_t(hypotf(z.x, z.y), 0.0f)`를 반환하면 CUDA와 맞는다. 실행으로 확인하지는 않았다.
- **테스트는 `real`과 `imag`만 있다.** 산술, `Square`, `Conjugate`, `bool` 캐스트, `NaNEqual` 오버로드에는 compiled 대 eager 테스트가 없다. 특히 `bool`과 `NaNEqual` 동작은 MLX CPU와 CUDA 소스에서 추론했을 뿐 실행해 보지 않았다.
- **복소 비교와 min/max는 정의되지 않았다.** complex64에 대한 `<`, `>`, `Minimum`, `Maximum`은 `MLX_ROCM_REAL_ONLY` 메시지가 아닌 일반 오류로 컴파일에 실패한다. MLX의 복소 순서(사전식)는 구현되지 않았다.
- **`Divide`는 교과서 공식을 쓴다.** `Abs`의 `hypotf`와 달리 overflow나 underflow에 대비한 스케일링이 없다. 크기가 큰 분모에서는 `b.x * b.x + b.y * b.y`가 overflow할 수 있다. eager ROCm 경로가 나눗셈을 스케일링하는지는 확인하지 않았으므로, 극단값에서 fused와 eager 결과가 다를 수 있다.
- **ROCm에서 NVFP4 수치는 검증되지 않는다.** ROCm에서 게이트된 테스트는 거부 지점에서 멈춘다. global scale 산술은 Metal과 CUDA에서만 확인되는데, PR은 변경 후 그쪽을 다시 돌리지 않았다. 근본 공백(ROCm에 expert별 global scale을 받는 NVFP4 `GatherQMM`이 없음)은 남아 있으며 #1806에서 추적한다.
- **PhiMoE 테스트는 ROCm에서만 다시 실행했다.** f32 기준값은 Metal과 CUDA에서 테스트를 통과하기 쉽게 만들 뿐이어야 하지만, 확인하지는 않았다.
- **JIT 캐시 갱신.** 프리앰블 변경으로 JIT 캐시 키가 바뀌므로, 업그레이드 후 캐시된 ROCm 커널 모듈이 한 번 다시 빌드된다. 예상된 무해한 동작이지만 첫 실행이 느려 보일 수 있다.

## 6. 학습 포인트

- 공용 codegen과 백엔드 프리앰블은 빌드 단계에서 아무도 확인하지 않는 계약이다. codegen은 프리앰블에 없는 타입을 이름으로 쓸 수 있고, 그 공백은 해당 dtype의 그래프가 런타임에 처음 컴파일될 때에야 드러난다.
- JIT 프리앰블에 변환을 추가하면 컴파일되는 템플릿의 범위가 넓어진다. 새 타입을 조용히 받아들일 템플릿에 그 타입에 대한 `static_assert`를 두면 "컴파일된다"가 "틀린 값을 계산한다"로 바뀌는 것을 막는다.
- C++에서 같은 대상 타입이면 비템플릿 변환 연산자가 템플릿보다 우선한다. 일반 규칙을 건드리지 않고 한 대상 타입(`bool`)만 특별 처리하는 깔끔한 방법이다.
- 백엔드 공백에 대한 테스트 skip은 그냥 일찍 반환하는 것보다 예상된 거부를 단언할 때 더 강하다. 백엔드 동작이 바뀌면 여전히 실패한다.
- parity 테스트가 경계에 걸려 있으면, 무엇이든 넓히기 전에 양쪽을 더 높은 정밀도의 참값과 비교해 본다. 이번에는 검증 대상 커널이 아니라 기준값이 덜 정확한 쪽이었다.
