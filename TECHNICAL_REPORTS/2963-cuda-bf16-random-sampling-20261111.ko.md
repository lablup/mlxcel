# PR #2963: CUDA bfloat16 난수 샘플링의 float32 정밀도 보존

- **작성일**: 2026-11-11
- **상태**: 완료 (#2958 close)

## 요약

NVIDIA GB10(`sm_121`, CUDA 13)에서 `models/gemma-4-e2b-it-4bit`를 샘플링 생성하면 주변 문맥은 자연스러운 채로 약 수백 토큰마다 `<unused6226>`(토큰 id `262143`, `262144` 어휘의 마지막 인덱스 `vocab_size - 1`)이 간헐적으로 출력되었다. 초기 소표본 A/B에서는 양자화 행렬-벡터 곱(`qmv`)의 row-blocking 결함처럼 보였으나, 이는 Volta(`sm_70`) 이외의 아키텍처에서 `default_rows_per_warp = 1`을 선택하는 `qmv.cu` 소스 코드와 모순되었다.

런타임 GPU 커널 계측, 73번째 디코드 스텝의 로짓 캡처 및 CPU `f32` 기준값 대조, 그리고 조건당 `5,540` 토큰 규모의 A/B 실험으로 두 질문을 모두 해결했다.

1. **양자화 행렬곱(`qmv`)은 결함과 무관함**: `sm_121`에서 환경변수 미설정과 `MLXCEL_QMV_ROWS_PER_WARP=1`은 모두 동일한 기본 `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>`를 호출하며, 다섯 조건(`unset`, `1`, `2`, `3`, `4`) 모두 `65`개 시드 전체에서 `65/65` 바이트 단위 동일 출력과 동일한 `12`회 `<unused6226>` 발생(`0.217%`)을 기록했다. `<unused6226>`이 샘플링된 스텝(`--seed 13`, step `73`)에서 LM-head `quantized_matmul` 출력은 CPU `f32` `dequantize` + `matmul` 기준값과 `rel_err = 0.004860` 이내로 일치했고(`raw_gpu[262143] = -21.875000` 대 `raw_ref[262143] = -21.859125`), `top_k = 64, top_p = 0.95` 필터링 이후 `filtered[262143]`은 정상적으로 `-inf`로 마스킹되어 있었다.
2. **근본 원인은 CUDA `bfloat16` 타입 승격 규칙 아래의 `mlx/random.cpp`**: CUDA 빌드에서 `overlays/cuda/mlx/dtype.cpp`는 트랜스포머 그래프의 `copy_v<bfloat16, float>` 변환 오버헤드를 피하기 위해 `promote_types(bfloat16, float32) -> bfloat16` 및 `promote_types(uint32, bfloat16) -> bfloat16`으로 재정의한다. 어휘 크기 `vocab = 262144`가 `REJECTION_JOINT_VOCAB_MAX = 32768`을 초과하면 `fused_sample`은 Gemma 4(`top_k = 64, top_p = 0.95`)를 `fused_sample_argpartition_chain` -> `mlx::core::random::categorical(filtered, -1)` -> `categorical_inverse_cdf`로 라우팅하며, 이 때문에 `MLXCEL_SAMPLING_REJECTION=0`에서도 동일하게 증상이 재현되었다. `categorical_inverse_cdf` 내부의 `promote_types(logits.dtype(), float32)`가 `bfloat16`으로 평가되면서 `exp`, 배타적 `cumsum`, `sum`, `u = uniform_f32 * sum(w)`가 모두 `bfloat16`으로 실행되었다. 그 결과 `uniform_f32 >= 1 - 2^-8`(`1/256 = 0.391%`, 스캔 대 리덕션 1-ULP 반올림 차이로 실측 `~0.21%`)인 표본에서 `u`가 `sum(w) == cdf[262143]`으로 반올림되었고, `searchsorted(cdf, u, "right")`가 `262144`를 반환해 `subtract(..., 1u)`가 `w[262143] == 0`임에도 `262143`을 출력했다.

CUDA 전용 `overlays/cuda/mlx/random.cpp` 오버레이를 추가해 `categorical_inverse_cdf`, 배치 `categorical_impl` Gumbel-max, `bernoulli`, `laplace`의 내부 정밀도를 `float32`로 유지하고, `top_p_filter`의 `bfloat16` `softmax` + `cumsum`을 `float32`로 승격함으로써 `mlxcel generate`, `mlxcel run`, `mlxcel-server` 공통 경로에서 결함을 제거했다(시드 고정 E2B `6,750` 토큰 `0`건, 무작위 E2B `6,000` 토큰 `0`건, 시드 고정 E4B `2,898` 토큰 `0`건, 디코드 처리량 중앙값 `80.91` 대 `80.67 tok/s`, `-0.29%`).

## 1. 문제 정의

이슈 #2958은 `sm_121` GB10 환경에서 `models/gemma-4-e2b-it-4bit` 샘플링 생성 중 `<unused6226>`(`id = 262143 = vocab_size - 1`)이 간헐적으로 출력되는 현상을 보고했다.

- `mlxcel generate`와 대화형 `mlxcel run` 모두에서 발생하며 앞뒤 문맥은 정상적으로 이어졌다.
- `MLXCEL_SAMPLING_REJECTION=0` 및 새 `MLX_PTX_CACHE_DIR`에서도 재현되었고, `--temp 0`이나 Apple Silicon Metal에서는 재현되지 않았다.
- 초기 5회 실행(`~410` 토큰)에서 `MLXCEL_QMV_ROWS_PER_WARP=1`이 `0`건을 기록해 기본 설정(`~510` 토큰 중 `4`건)과 대비되었으나, 이는 `qmv.cu:1223`(`default_rows_per_warp = volta ? 2 : 1`)과 모순되었다.

이에 따라 런타임 계측과 수천 토큰 규모의 A/B 실험으로 커널 모순을 해소하고, 실제 발생 스텝의 원시 로짓을 캡처해 CPU `f32` `dequantize` + `matmul` 기준값과 비교하며, 타 모델 및 dtype 영향 범위를 확인하고, 새 옵션 스위치 없이 공통 코드 경로를 수정한 뒤 `tests/cuda_qmv_reference_parity.rs`와 `main`에서 실패하고 수정 후 통과하는 회귀 테스트를 추가해야 했다.

## 2. 원인 분석

### 2.1 `MLXCEL_SAMPLING_REJECTION=0`이 샘플러를 배제하지 못한 이유

`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`에서 비-ROCm 빌드의 `REJECTION_JOINT_VOCAB_MAX`는 `32768`이다. `top_k > 0`과 `top_p < 1.0`이 동시에 켜져 있고 `vocab > 32768`이면 `sampling_rejection_routes`는 `false`를 반환한다. Gemma 4는 `vocab_size = 262144`, `top_k = 64`, `top_p = 0.95`, `bfloat16` 활성화를 사용하므로 기본 실행과 `MLXCEL_SAMPLING_REJECTION=0` 모두 동일하게 `fused_sample_argpartition_chain` -> `fused_sample_filter_logits` -> `mlx::core::random::categorical(filtered, -1)` 경로를 탔다.

Metal에서는 `single_dtype_scalars`가 `false`여서 스칼라 `-inf` 마스킹 과정에서 `filtered`가 `float32`로 승격된 뒤 `categorical`로 전달된다. 반면 CUDA에서는 이슈 #636에 의해 `single_dtype_scalars`가 `true`이므로 `filtered`가 `bfloat16` 상태로 유지된다.

### 2.2 `mlx/random.cpp` 및 `top_p_filter`의 `bfloat16` 정밀도 강등

`src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`는 MLX 타입 승격 테이블을 패치해 `promote_types(bfloat16, float32) -> bfloat16`, `promote_types(uint32, bfloat16) -> bfloat16`을 반환한다. 업스트림 `mlx/random.cpp`의 네 분포 함수는 `float32`와의 승격 결과가 최소 `float32`라고 가정하고 작성되어 있었다.

1. **`categorical_inverse_cdf` (단일 행 디코드, `logits.size() == n`)**:
   ```cpp
   auto dtype = promote_types(logits.dtype(), float32); // CUDA에서 bfloat16으로 평가됨!
   auto x = astype(logits, dtype, s);
   ...
   auto cdf = cumsum(w, 0, /* reverse = */ false, /* inclusive = */ false, s);
   auto u = multiply(uniform(shape, float32, key, s), sum(w, s), s);
   return subtract(searchsorted(cdf, u, "right", s), array(1u, uint32), s);
   ```
   `dtype == bfloat16`이 되면 `w`, 배타적 `cdf`, `sum(w)`, `u`가 모두 `bfloat16`으로 계산된다. 마스킹된 꼬리 토큰들은 `w[i] = 0`이므로 `cdf`는 `cdf[vocab - 1]`에서 평탄해진다. `uniform_f32`가 `[1 - 2^-8, 1)` 구간에 들어와 `u`가 `sum(w)`로 반올림되거나 `bfloat16` 누적 스캔 반올림으로 `cdf[vocab - 1]`이 `sum(w)`보다 1 ULP 작아지면 `u >= cdf[vocab - 1]`이 성립하고, `searchsorted(..., "right")`(`upper_bound`)가 `u`보다 큰 원소를 찾지 못해 `vocab`을 반환하면서 `subtract(..., 1u)`가 `vocab - 1`(`262143`)을 반환한다.
2. **`categorical_impl` (배치 Gumbel-max 디코드, `B >= 2`)**: `add(g /*float32*/, logits /*bfloat16*/)`가 `float32` Gumbel 노이즈를 `bfloat16`으로 강등시켜 미세한 로짓 차이와 큰 로짓 크기에서의 이동 불변성을 무너뜨리고 인덱스 0 `argmax` 동률을 유발한다.
3. **`bernoulli`**: `multiply(p_bf16, upper_f32)`와 `less(bits_u32, bf16)`가 `upper`(`2^32`) 및 32비트 난수 비트를 `bfloat16`으로 강등시켜 `p == 1.0`에서도 약 `1/256` 확률로 `false`를 반환한다.
4. **`laplace`**: `multiply(array(-1.0f, bfloat16), abs(samples_f32))`가 `1.0` 근처의 균등 표본을 `-1.0_bf16`으로 반올림해 `log1p(-1.0_bf16) = -inf`를 발생시킨다.

또한 `top_p_filter`(`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`) 역시 `single_dtype_scalars`가 참일 때 `softmax`와 포함적 `cumsum`을 `bfloat16`으로 수행하고 있었다. 누적 확률이 `0.5`를 넘으면(`bfloat16` ULP = `2^-8 = 3.91e-3`) `2^-9`(`1.95e-3`) 미만의 꼬리 확률들이 `cumsum`에서 0으로 삼켜져 대형 어휘의 `top_p` 단독(`top_k = 0`) 필터링에서 절단 임계값에 도달하지 못하는 현상이 발생한다.

## 3. 변경 사항

1. **`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp` (신규 CUDA 전용 코어 오버레이)**:
   - 고정된 업스트림 MLX 커밋 `9f806b3606eaa9f7c807238f6bd465185cd94d6d`(`mlx/random.cpp`)에서 분기하고 출처 헤더 규칙을 준수했다.
   - `categorical_inverse_cdf`, `categorical_impl`, `bernoulli`, `laplace`에서 입력 또는 목표 dtype이 `bfloat16`일 때 내부 연산 정밀도를 `float32`로 보존하며, `float16`, `float32`, 정수 경로는 변경하지 않는다.
2. **`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` (`top_p_filter`)**:
   - `single_dtype_scalars`가 참이고 입력이 `bfloat16`일 때 `softmax` + `cumsum` 입력만 `float32`로 승격하고, `sorted_logits`와 `fill_scalar`는 `x.dtype()`으로 유지하며 Metal 경로는 그대로 둔다.
3. **`src/lib/mlxcelverse/CMakeLists.txt` 및 `src/lib/mlxcelverse/overlays/cuda/README.md`**:
   - `overlays/cuda/mlx/*.cpp` 글롭에 `CONFIGURE_DEPENDS`를 추가해 코어 오버레이 파일 추가·삭제 시 CMake가 자동 재구성되도록 했다.
   - 비-ROCm 빌드에서 재사용되는 `_deps/mlx-src` 체크아웃에 은퇴했거나 비활성화된 최상위 `mlx/*.cpp` 오버레이가 남아 있으면 `git checkout --`으로 자동 복원하도록 보강했다.
   - `overlays/cuda/README.md`의 코어 오버레이 파일 수(`3` -> `4`)를 갱신하고 백엔드 오버레이 파일 수 표기(`30` -> `33`)를 실제 목록과 일치시켰다.
4. **신규 CUDA 통합 테스트**:
   - `tests/cuda_qmv_reference_parity.rs`: `[1, K]` 및 `[1, 1, K]` 레이아웃, `N in {262144, 262147}`, `K in {512, 576}`, `float16` 및 `bfloat16`에 대해 4비트 `M = 1` `quantized_matmul`을 CPU `f32` `dequantize` + `matmul` 기준값과 비교하며(`rel < 2e-2`), 마지막 64개 행의 정확도와 유한성을 명시적으로 검증하고 별도 자식 프로세스에서 `MLXCEL_QMV_ROWS_PER_WARP` 미설정, `2`, `4`를 각각 검증한다.
   - `tests/cuda_bf16_categorical_last_vocab.rs`: `vocab = 262144` `bfloat16` 입력에서 `fused_sample`, `fused_sample_categorical`, 단일 행 `random_categorical`, 배치 `random_categorical` Gumbel-max, `top_p` 단독 `cumsum` 소실 문제를 결정론적으로 검증한다(수정 전 `main`에서 `0/3` 통과, 수정 후 `4/4` 통과).

## 4. 실측 결과 (GB10 `sm_121`)

### 4.1 AC 1: 런타임 LM-head 커널 계측

`mlxcel generate -m models/gemma-4-e2b-it-4bit -p "하이" -n 200 --show-reasoning --seed 13` 실행 중 `QuantizedMatmul::eval_gpu` 및 `cu::qmv` 계측 결과(`M = 1, N = 262144, K = 1536, B = 1`, LM-head 호출 `201`회):

| 조건 | 디스패치 경로 | 실행된 커널 | `rows_per_warp` | Grid / Block | `stock_qmv` / `rowblock` / `multirow` |
|------|---------------|-------------|-----------------|--------------|---------------------------------------|
| `unset` | `qmv` (`M*B=1 < 8`) | `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>` | `1` | `(32768,1,1)` / `(32,8,1)` | `201` / `0` / `0` |
| `MLXCEL_QMV_ROWS_PER_WARP=1` | `qmv` | `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>` | `1` | `(32768,1,1)` / `(32,8,1)` | `201` / `0` / `0` |
| `MLXCEL_QMV_ROWS_PER_WARP=2` | `qmv` | `qmv_rowblock_kernel` | `2` | `(16384,1,1)` / `(32,8,1)` | `0` / `201` / `0` |

### 4.2 AC 2: 발생 스텝 로짓 캡처 및 CPU `f32` 기준값 비교

수정 전 `models/gemma-4-e2b-it-4bit` `--seed 13`의 디코드 step `73`(`262143` `<unused6226>` 샘플링 지점) 캡처 결과:

- **Softcap 이전 LM-head `quantized_matmul` 대 CPU `f32` `dequantize` + `matmul`**: `max_abs_diff = 0.160517`(인덱스 `172211`), `max_abs_ref = 33.026920`, `rel_err = 0.004860`, top-1 토큰 일치(`236764`, GPU `33.000000` 대 CPU `33.026920`), 마지막 행 `262143`은 양쪽 모두 강한 음수(`raw_gpu[262143] = -21.875000`, `raw_ref[262143] = -21.859125`, `abs_diff = 0.015875`).
- **Softcap 및 필터 이후**: `max_abs_diff = 0.195107`, `rel_err = 0.008121`, `gpu[262143] = -18.750000` 대 `ref[262143] = -18.666931`, nucleus 토큰 `2`개 유지 및 `filtered[262143] = -inf`.
- **`categorical_inverse_cdf` 내부 상태**: `w[262143] = 0.0`, `cdf_bf16[262143] = 1.078125 (0x3f8a)`, `sum_bf16 = 1.078125 (0x3f8a)`, `uniform_f32 = 0.997557521 (0x3f7f5fee)`, `u_bf16 = 1.078125 >= cdf_bf16[262143] -> sampled 262143`. 반면 `float32`에서는 `cdf_f32[262143] = 1.082085013`, `u_f32 = 1.079442024 < cdf_f32[262143] -> sampled 236764`.

### 4.3 AC 3: 수정 전 QMV 커널 A/B (조건당 `5,540` 토큰)

| 수정 전 조건 | 시드 / 토큰 수 | `<unused6226>` 발생 (비율) | 발생 시드 | `unset` 대비 바이트 일치 | 디코드 중앙값 |
|--------------|----------------|----------------------------|-----------|--------------------------|---------------|
| `unset` | `65` / `5,540` | `12` (`0.217%`) | `2(x2),7,13,25,27,29,35,37,51,62,65` | 기준 | `81.09 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=1` | `65` / `5,540` | `12` (`0.217%`) | 동일 | `65 / 65` | `81.28 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=2` | `65` / `5,540` | `12` (`0.217%`) | 동일 | `65 / 65` | `80.87 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=3` | `65` / `5,540` | `12` (`0.217%`) | 동일 | `65 / 65` | `80.65 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=4` | `65` / `5,540` | `12` (`0.217%`) | 동일 | `65 / 65` | `80.77 tok/s` |

### 4.4 AC 4 및 AC 8: 타 모델 영향 범위 및 수정 후 검증

- **다른 크기의 Gemma 4 (`models/gemma-4-e4b-it-4bit`, `BF16`, `vocab = 262144`)**: 수정 전 `2,847` 토큰(`30` 시드)에서 `6`건(`0.211%`) 발생 -> 수정 후 **`2,898` 토큰(`30` 시드)에서 `0`건**.
- **비-Gemma `BF16` 대 `F16` 4비트 모델 비교 (`vocab = 151936`, `--top-k 64 --top-p 0.95 --temp 0.8`, `8` 시드 `1,608` 토큰)**: `models/qwen3-4b-4bit`(`BF16`)는 `vocab - 1 = 151935`를 `3`회(`0.187%`) 샘플링했으나(Qwen 3 토크나이저가 `0..=151668`만 정의하고 `151669..=151935`를 미매핑 패딩 슬롯으로 남겨 콘솔 문자열에는 보이지 않음), `models/qwen2.5-1.5b-instruct-4bit`(`F16`)는 `promote_types(float16, float32) == float32` 덕분에 `0`회 발생했다.
- **수정 후 `models/gemma-4-e2b-it-4bit`**: 시드 고정 **`6,750` 토큰(`75` 시드 `1..=75`)에서 `<unused6226>` `0`건** 및 시드 미지정 **`30 x 200` 재현 루프(`~6,000` 토큰)에서 `0`건**.
- **유휴 호스트 교차 디코드 처리량 (`gpu-lock` 아래 시드 `1, 3, 6, 8, 9`)**: 수정 전 중앙값 `80.91 tok/s`(평균 `80.85 tok/s`) 대 수정 후 중앙값 `80.67 tok/s`(평균 `80.66 tok/s`, `-0.29%`).

## 5. 검증 요약

- `make verify-fmt`, `make verify-versions`, `make verify-kernel-dtype-keys`, `make verify-env-var-docs`, `make verify-kernel-port-dispatch`, `make verify-rocm-overlay`, `make verify-python-tooling`: 모두 통과.
- `CONTRIBUTING.md` MLX 오버레이 출처 및 드리프트 검사(`PIN = 9f806b3606eaa9f7c807238f6bd465185cd94d6d`): CPU/Metal/CUDA 전체 오버레이 드리프트 `0`건.
- `cargo clippy --workspace --all-targets --features cuda -- -D warnings`: 경고 `0`건 통과.
- `cargo test --profile test-fast --features cuda --test cuda_bf16_categorical_last_vocab -- --test-threads=1`: `4`개 통과(수정 전 `main`에서 `0/3`).
- `cargo test --profile test-fast --features cuda --test cuda_qmv_reference_parity -- --test-threads=1`(`unset`, `MLXCEL_QMV_ROWS_PER_WARP=2`, `MLXCEL_QMV_ROWS_PER_WARP=4`): 각 조건 `2`개 통과.
- `make verify-test-cuda-qmm-determinism`: `5`개 통과.

## 6. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 6 |
| 추가 줄 | +1190 |
| 삭제 줄 | -10 |
| 커밋 | 2 (보고서 커밋 제외) |

관련: #2958(이슈), #2963(PR), #636(`single_dtype_scalars` CUDA 스칼라 타이핑).
