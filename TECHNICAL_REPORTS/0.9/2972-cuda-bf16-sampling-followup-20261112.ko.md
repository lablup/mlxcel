# PR #2972: CUDA multivariate_normal 및 Rust 레퍼런스 top_p_filter의 float32 정밀도 보존

- **작성일**: 2026-11-12
- **상태**: 완료 (#2971 해결)

## 요약

PR #2963(#2958 해결) 머지 이후, CUDA 빌드의 `promote_types(bfloat16, float32) -> bfloat16` 타입 승격 오버레이(`src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`) 아래에서 나머지 난수 분포 및 샘플링 헬퍼 전체를 전수 점검하여 후속 정밀도/타입 결함 2건을 수정하고 의도적 유지 대상 1건을 검증했습니다.

1. **`mlx::core::random::multivariate_normal` (`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp`)**: 함수 진입부에서 `dtype == float32`를 명시적으로 강제(`if (dtype != float32) throw std::invalid_argument(...)`)하고 공분산 SVD 인자와 `scaled_out`을 모두 `float32`로 계산함에도, 마지막 `add(mean, scaled_out, stream)`에서 `mean.dtype() == bfloat16`이면 프로세스 전역에 적용되는 CUDA 승격 규칙(CPU 스트림 포함) 때문에 `scaled_out`(`float32`)이 `bfloat16`으로 강등되어 반환되었습니다.
2. **Rust 레퍼런스 `sampling::top_p_filter` (`src/lib/mlxcel-core/src/sampling.rs`)**: 바로 아래의 `top_p_filter_min_keep` 및 PR #2963에서 수정된 C++ `top_p_filter`와 달리, 입력 `logits`의 타입 그대로 `softmax`와 포함형 `cumsum`을 수행한 뒤 `where_cond(&mask, &sorted_logits, &neg_inf_f32)`로 마무리하고 있었습니다. 대규모 어휘(`[1, 262144]`)에서 `bfloat16`(또는 `float16`) 입력이 들어오면 누적 확률이 `0.5`(`bfloat16` 1 ULP = `2^-8 = 3.91e-3`)를 넘은 뒤 작은 꼬리 확률이 `cumsum`에서 증발해 마스킹이 풀리고, CUDA 빌드에서는 반환 타입까지 `FLOAT32` 대신 `BFLOAT16`으로 달라졌습니다.
3. **체인 내부 `min_p` 및 `XTC` `softmax(x, -1)` (`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`, 현행 유지)**: 두 필터 모두 `cumsum`을 사용하지 않고, MLX CUDA `softmax.cu` 커널이 행별 `max`와 `sum(exp)`를 이미 `float32`(`AccT = float`)로 누적하며, 마스킹된 `-inf` 항목은 `0.0_bf16 < threshold`로 누수가 불가능합니다. 또한 입출력을 `bfloat16`으로 유지해야 #636의 단일 타입 그래프와 `Softmax::eval_gpu`의 `copy_shared_buffer` 제자리 버퍼 재사용이 보존되므로 변경하지 않았습니다.

PR #2972는 `multivariate_normal`에서 `bfloat16` `mean`을 `add` 직전에 `float32`로 승격하고, `sampling::top_p_filter` 진입부에서 비-`FLOAT32` 입력만 `FLOAT32`로 변환(`FLOAT32` 입력 시 추가 FFI 호출 없음)하며, 오버레이 문서 갱신과 함께 수정 전 실패 / 수정 후 통과가 검증된 회귀 테스트 2종을 추가합니다.

## 1. 근본 원인 분석

### 1.1 `multivariate_normal`의 `bfloat16` 반환 강등 (`random.cpp:311`)

업스트림 `mlx/random.cpp`의 `multivariate_normal(mean, cov, shape, dtype, key, s)`은 `dtype == float32`를 검사한 뒤 `cov`를 `float32`로 변환해 `linalg::svd(covariance, true, stream)`를 수행하고, `float32` 표준정규 샘플로부터 `scaled_out`(`output_shape`, `float32`)을 만든 뒤 `add(mean, scaled_out, stream)`을 반환합니다.

표준 MLX 승격 규칙(Metal, ROCm, CPU 전용 빌드)에서는 `promote_types(bfloat16, float32)`가 `float32`이므로 `mean`이 `float32`로 올라가며 24비트 가수부 정밀도와 `float32` 반환 타입이 유지됩니다. 반면 `--features cuda` 빌드에서는 `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`가 공유 MLX 타입 테이블 전체(GPU 및 `linalg::svd`가 실행되는 CPU 스트림 모두)에서 `promote_types(bfloat16, float32) -> bfloat16`을 적용합니다. 그 결과 `mean`이 `bfloat16`일 때 `add(mean, scaled_out, stream)`이 전체 샘플 배치 텐서 `scaled_out`을 `bfloat16`으로 내리면서 하위 16비트 가수부를 잘라내고 `dtype == float32` 계약을 위반했습니다.

성능 측면에서도 수정 전에는 브로드캐스트된 전체 샘플 텐서 `scaled_out`(`[S, N]`)에 `float32 -> bfloat16` `AsType`이 걸리고 입출력 원소 크기 불일치(4바이트 대 2바이트)로 `scaled_out` 버퍼 재사용이 막혔던 반면, `mean`(`[N]`)만 `float32`로 올리면 변환 원소 수가 `1/S`로 줄고 `scaled_out`의 제자리 버퍼 재사용이 가능해집니다.

### 1.2 Rust 레퍼런스 `sampling::top_p_filter`의 꼬리 확률 증발 및 반환 타입 불일치 (`sampling.rs:3050`)

`src/lib/mlxcel-core/src/sampling.rs`의 `pub(crate) fn top_p_filter`는 단위 테스트(`top_p_filter_*`, `fused_sample_probs_equals_softmax_of_filtered_over_t`)에서 기준 구현으로 사용됩니다. 기존 코드는 입력 타입 정규화 없이 `ffi::softmax(logits, -1)`와 `ffi::cumsum(&sorted_probs, -1, false, true)`를 실행했습니다.

- 토큰 `0`의 확률이 `~0.601`, 토큰 `1..=10000`의 확률이 각각 `~2.73e-5`(실제 누적합 `~0.874 > top_p = 0.80`)인 `[1, 262144]` `bfloat16` 로짓 행에서 순차 `bfloat16` 덧셈은 `0.5` 구간의 `1 ULP`(`2^-8 = 3.91e-3`) 때문에 `0.601 + 2.73e-5 == 0.601`에 머물러 인덱스 `10,001..262,144`의 꼬리 토큰 `252,143`개가 마스킹되지 않았습니다.
- 6단계 `ffi::where_cond(&mask, &sorted_logits, &neg_inf)`(`neg_inf`는 `FLOAT32`)에서 Metal/CPU 빌드는 `FLOAT32`를 반환한 반면 CUDA 빌드는 `BFLOAT16`을 반환해 백엔드 간 반환 타입 일관성과 `to_vec_f32` 테스트 헬퍼 호환성이 깨졌습니다.

### 1.3 체인 내부 `min_p` 및 `XTC` `softmax`를 현행 유지한 근거

`mlx/backend/cuda/softmax.cu`와 `mlx/backend/cuda/arg_reduce.cu` 커널 구현을 대조한 결과, 체인 내부 `min_p`(`mlx_cxx_bridge.cpp:6289`)와 `XTC`(`mlx_cxx_bridge.cpp:6103`)는 `float32` 텐서 구체화가 불필요함을 확인했습니다.

- `softmax.cu`는 `max`와 `sum(exp)` 리덕션을 `float32`(`AccT = float`)로 계산하고 최종 확률 스칼라만 8비트 지수부를 가진 `bfloat16`(최소 정규수 `~1.17e-38`)으로 기록합니다.
- 두 필터 모두 누적합(`cumsum`) 없이 원소별 임계값 비교만 수행하며, `XTC`의 비후보 마스크(`+inf_bf16`)도 `ArgMin` 초기값(`Limits<__nv_bfloat16>::max()`)과 정확히 일치합니다.
- 입출력을 `bfloat16`으로 유지하면 #636의 단일 타입 체인과 `Softmax::eval_gpu`의 `copy_shared_buffer` 버퍼 재사용이 유지되어 매 디코드 스텝 행당 `1 MB`(`V = 262,144`)의 추가 `float32` 할당과 `AsType` 커널 호출을 피할 수 있습니다.

## 2. 변경 사항

1. **`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp` 및 `src/lib/mlxcelverse/overlays/cuda/README.md`**:
   - `multivariate_normal`의 마지막 덧셈 직전에 `bfloat16` `mean`을 `float32`로 승격(`auto mean_up = mean.dtype() == bfloat16 ? astype(mean, float32, stream) : mean; return add(mean_up, scaled_out, stream);`)하도록 수정했습니다.
   - 오버레이 상단 주석과 `overlays/cuda/README.md`에 `bernoulli`, `categorical_inverse_cdf`, 배치 `categorical_impl`, `laplace`에 이은 다섯 번째 보정 지점으로 `multivariate_normal`을 문서화했습니다(이슈 #2958, #2971).
2. **`src/lib/mlxcel-core/src/sampling.rs` (`top_p_filter`)**:
   - 입력이 이미 `dtype::FLOAT32`이면 추가 FFI 호출 없이 참조를 그대로 재사용하고, 비-`FLOAT32` 입력일 때만 진입부에서 `dtype::FLOAT32`로 승격하도록 수정했습니다.
   - 기본 디바이스 스트림과 CPU 스트림 양쪽에서 `[1, 262144]` `BFLOAT16` 입력의 꼬리 마스킹 및 `FLOAT32` 반환 타입을 검증하는 단위 테스트 `top_p_filter_bf16_large_vocab_masks_tail_and_returns_f32`를 추가했습니다.
3. **`tests/cuda_bf16_categorical_last_vocab.rs`**:
   - `BFLOAT16` `mean`과 `BFLOAT16`/`FLOAT32` 공분산 행렬에 대해 `mlxcel_core::random_multivariate_normal`의 `FLOAT32` 반환 타입, `[512, 2]` 형상, 표본 평균(`[3.5, -2.25]` 근방), 하위 16비트 `float32` 가수부 보존 여부를 검증하는 통합 테스트 `random_multivariate_normal_bf16_mean_returns_f32_samples`를 추가했습니다.

## 3. 검증 결과

모든 검증은 NVIDIA GB10(`sm_121`, `--features cuda`, `/usr/local/bin/gpu-lock` 직렬화)에서 수행했습니다.

- **수정 전 실패 확인**:
  - `sampling::tests::top_p_filter_bf16_large_vocab_masks_tail_and_returns_f32`: 수정 전 실패 확인(CUDA 기본 스트림에서 `BFLOAT16` 반환 `12 != 10`, 순차 CPU `cumsum`에서 꼬리 토큰 `252,143`개 미마스킹).
  - `random_multivariate_normal_bf16_mean_returns_f32_samples`: 수정 전 실패 확인(`BFLOAT16` 반환 `12 != 10`).
- **수정 후 통과 확인**:
  - `cargo test --profile test-fast --features cuda -p mlxcel-core --lib sampling::tests::top_p_filter -- --test-threads=1`: `4 passed; 0 failed` (`0.39s`).
  - `cargo test --profile test-fast --features cuda --test cuda_bf16_categorical_last_vocab -- --test-threads=1`: `5 passed; 0 failed` (`3.86s`).
- **정적 분석 및 오버레이 정합성 검증**:
  - `CONTRIBUTING.md` MLX 오버레이 베이스 커밋(`9f806b3606eaa9f7c807238f6bd465185cd94d6d`) 드리프트 검사: 드리프트 `0`건 / 누락 헤더 `0`건.
  - `make verify-fmt verify-versions verify-kernel-dtype-keys verify-env-var-docs verify-kernel-port-dispatch verify-rocm-overlay`: 전체 통과.
  - `cargo clippy --workspace --all-targets --features cuda -- -D warnings`: 경고 `0`건 통과.
