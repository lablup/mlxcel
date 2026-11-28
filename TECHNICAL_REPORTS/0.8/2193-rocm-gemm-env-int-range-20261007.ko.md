# 기술 보고서: PR #2193 - ROCm GEMM 환경 변수 정수 범위 검사와 오래된 그래프 주석 수정

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 origin/main `bbd05099` 위의 코드 커밋 `92de1604`, `e54ffc1e`와 이 보고서로 구현. 전체 ROCm 게이트(`make verify-rocm`) 통과 후 머지됨. #2152를 닫음.

**언어**: C++/HIP(`src/lib/mlx-cpp/patches-rocm/` 아래 ROCm 오버레이: 신규 `env_int.h`, `gemms/rocblas_gemm.{cpp,h}`, `matmul.cpp`, `quantized/qmm.hip`, `device.cpp`), Rust(신규 `tests/rocm_qmm_env.rs`), Markdown(`LOCAL_FIXES.md`, 오버레이 README, `docs/environment-variables.md`)

**위험도**: 낮음. 잘못된 값만 동작이 달라집니다. 예전에는 조용히 wrap되거나 무시되었고, 이제는 경고를 출력하고 기본값을 씁니다. 유효한 값과 설정하지 않은 변수는 결과가 전과 같습니다. Metal과 CUDA 빌드는 ROCm 오버레이를 복사하지 않습니다.

## 요약

ROCm 오버레이는 정수 GEMM 설정값을 `strtol`에 검사 없는 `static_cast<int>`를 붙이거나 `atoi`로 읽었습니다. 그래서 범위를 벗어난 입력이 아무 메시지 없이 wrap되었습니다. `MLX_ROCM_WMMA_QMM_MAX_M=4294967297`은 상한 1이 되어, 두 행 이상인 bf16 GEMM을 모두 fused WMMA 커널에서 dequantize + hipBLASLt로 보냈습니다.

PR #2193은 새 헤더 `mlx/backend/rocm/env_int.h`에 헬퍼 `env_int_or_default` 하나를 추가합니다. 이 헬퍼는 #2073이 `MLX_ROCM_GPU_WATCHDOG_SECS`와 `MLX_ROCM_FFT_CACHE_SIZE`에 정한 규칙을 따릅니다. 이제 설정값 11개가 이 헬퍼를 거칩니다. 각 읽기 결과는 함수 지역 static에 캐시되므로 잘못된 값은 프로세스당 stderr 한 줄만 출력합니다. prefill이 WMMA GEMM을 쓴다고 적혀 있던 `device.cpp`의 `use_hip_graphs()` 주석은 더 이상 prefill GEMM을 지목하지 않습니다.

첫 커밋의 리뷰에서 두 가지를 발견했습니다. solution index 읽기 함수가 아직 세 파일에 중복되어 잘못된 index가 최대 세 번 경고했고, threshold의 새 캐싱을 확인하는 테스트가 없었습니다. 둘 다 `e54ffc1e`에서 고쳤습니다. 새 child-process 테스트는 gfx1151에서 15개 케이스를 모두 통과하고, 오버레이 소스를 main에서 복원하면 그중 9개가 실패합니다.

## 1. 문제 정의

### 1.1 배경

LOCAL_FIXES 항목 25(#2073)가 ROCm 환경 변수 파싱 규칙을 정했습니다. `errno`를 0으로 두고 `strtol`로 10진수를 파싱하며, 문자열 전체가 숫자여야 합니다. `ERANGE`, 숫자가 아닌 문자, 범위 밖의 값은 `[ROCm] ignoring invalid NAME="value" (expected ...); using ...`을 출력하고 기본값을 씁니다. 설정하지 않았거나 빈 변수는 조용히 기본값을 씁니다. GEMM 설정값은 이 규칙보다 먼저 만들어졌습니다.

### 1.2 조용한 wrap

`parse_{positive,non_negative}_int_env`(qmm.hip, 그리고 matmul.cpp와 gemms/rocblas_gemm.cpp에 있던 non-negative 쪽의 복사본)와 `dequant_cache_capacity()`, `moe_segment_min_avg()`의 인라인 파싱은 `long`을 범위 검사 없이 `int`로 캐스팅했고 아무것도 출력하지 않았습니다. `4294967297`은 1이 되었고, `2147483648`은 음수가 되어 무시되었으며, `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=4294967304`는 8이 되었습니다. 숫자가 아닌 문자, 0, 음수 값은 메시지 없이 무시되었습니다.

### 1.3 atoi 읽기와 호출마다 읽기

- `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B`, `MLX_GRIDX_MULT`는 `atoi`를 썼습니다.
- `MLX_ROCM_QMV_TILE_N`은 게다가 qmv 호출마다 다시 읽혔고, tiled 커널의 `__launch_bounds__(TILE_N_MAX * WARP_SIZE)`(`TILE_N_MAX = 32`)를 넘을 수 있었습니다.
- `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD`는 `should_use_dequant_gemm_path`를 호출할 때마다 읽혔습니다. 그래서 이 값에 대한 경고는 GEMM마다 한 번씩 출력되었을 것입니다.

### 1.4 오래된 주석

device.cpp의 `use_hip_graphs()` 주석은 prefill이 WMMA GEMM을 쓴다고 했습니다. 항목 29(`select_qmm_route`) 이후로는 사실이 아닙니다.

### 1.5 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 설정값 오타가 GEMM 경로와 prefill 속도를 조용히 바꿈 | 중간 | 낮음 |
| 범위를 벗어난 tile 폭이 qmv launch를 깨뜨림 | 중간 | 낮음 |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlx/backend/rocm/env_int.h`(신규) | `env_int_or_default(name, default, min, max, expected, default_description = nullptr)`, #2073 규칙을 inline 함수 하나로 구현 |
| `gemms/rocblas_gemm.h` | `env_int.h`를 include하고, 네 solution index 변수를 읽는 유일한 공유 함수인 inline `gemm_solution_index_{f32,bf16}` 추가 |
| `gemms/rocblas_gemm.cpp` | 자체 `parse_non_negative_int_env`와 solution index 읽기 함수 복사본 제거 |
| `matmul.cpp` | 같은 복사본을 제거하고 `rocm::gemm_solution_index_*` 호출. `moe_segment_min_avg()`(`MLX_ROCM_MOE_SEG_MIN`)가 헬퍼 사용 |
| `quantized/qmm.hip` | `parse_positive_int_env`, `parse_non_negative_int_env`, `qmm_gemm_solution_index_*` 제거. threshold, 상한, 캐시 크기, `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B`, `MLX_GRIDX_MULT`가 static에서 헬퍼 사용. `grid_x`를 64비트로 계산 |
| `device.cpp` | `use_hip_graphs()` 주석이 prefill도 이 경로를 거치지 않으며 그 GEMM 경로는 `quantized/qmm.hip`의 `select_qmm_route`가 고른다고 기술 |
| `tests/rocm_qmm_env.rs`(신규) | child-process 테스트, 케이스 15개 |
| `LOCAL_FIXES.md`, `README.md`, `docs/environment-variables.md` | 항목 34, backend 파일 수 109, 범위와 경고 |

코드 커밋은 두 개입니다. 수정(`92de1604`, "range-check GEMM env integers and fix a stale graph comment")과 리뷰 후속(`e54ffc1e`, "read each GEMM solution index once per process")입니다. 이 보고서를 제외하고 파일 10개, 702줄 추가, 173줄 삭제.

## 3. 설계

### 3.1 #2073 규칙을 따르는 헬퍼 하나

`env_int_or_default(name, default, min, max, expected, default_description = nullptr)`는 설정하지 않았거나 빈 변수에 `default`를 돌려줍니다. 그 밖의 경우 `strtol`로 10진수를 파싱하고, `end == raw`, 뒤에 남는 문자, `ERANGE`, `[min, max]` 밖의 값을 거부하며 한 줄을 출력합니다. `default_description`은 `-1`을 그대로 출력하는 대신 sentinel 기본값의 이름("the per-device default", "the built-in crossover")을 출력합니다. 기본값 자체는 범위 검사를 하지 않으며, `max`는 `INT_MAX`를 넘으면 안 되므로 마지막 `int` 캐스팅은 wrap될 수 없습니다.

### 3.2 범위와 sentinel 기본값

- threshold, 상한, `MLX_ROCM_MOE_SEG_MIN`, `MLX_ROCM_GROUPED_PREFILL_MIN_B`, `MLX_GRIDX_MULT`는 1~`INT_MAX`입니다.
- solution index와 캐시 크기는 0~`INT_MAX`이고, 0은 여전히 캐시를 끕니다.
- `MLX_ROCM_QMV_TILE_N`은 1~`TILE_N_MAX`이며, tiled 커널의 launch 상한과 일치합니다.
- sentinel 기본값은 batched solution index, threshold, 상한이 -1이고 tile 폭이 0입니다. 의도적으로 허용 범위 밖에 있으며, 브리프대로 범위 검사를 하지 않습니다.

### 3.3 프로세스당 한 번 읽기

헬퍼는 잘못된 값을 볼 때마다 경고하므로, 모든 호출 지점이 결과를 함수 지역 static에 보관합니다. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD`와 `MLX_ROCM_QMV_TILE_N`은 이제 호출마다가 아니라 한 번만 읽힙니다. 그래서 잘못된 값이 GEMM이나 qmv launch마다가 아니라 한 줄만 출력합니다.

### 3.4 프로세스당 solution index 읽기 함수 하나

세 파일(matmul.cpp, gemms/rocblas_gemm.cpp, quantized/qmm.hip)이 각자 anonymous namespace 안에 solution index 읽기 함수 복사본을 두었으므로, static도 파일마다 따로 있었고 잘못된 `MLX_ROCM_GEMM_*_SOLUTION_INDEX`는 프로세스당 최대 세 번 경고했습니다. `e54ffc1e`는 복사본을 `gemms/rocblas_gemm.h`의 inline 함수 한 쌍 `gemm_solution_index_{f32,bf16}`으로 바꿉니다. external linkage를 가진 inline 함수 안의 static은 프로그램 전체에 객체가 하나뿐이므로, 각 변수는 한 번만 읽히고 한 번만 경고합니다. batched fallback 메시지는 "using MLX_ROCM_GEMM_F32_SOLUTION_INDEX" 대신 "using the MLX_ROCM_GEMM_*_SOLUTION_INDEX value"로 표시됩니다.

### 3.5 64비트 grid 크기

`MLX_GRIDX_MULT`는 expert-batched GatherQMM launch의 `grid_x`를 배율로 키웁니다. 범위가 이제 `INT_MAX`까지이므로 `avg_tiles * gridx_mult`가 `int`를 넘칠 수 있습니다. `grid_x`는 이제 좁히기 전에 `int64_t`에서 최솟값을 구하므로 결과가 `(B + 63) / 64`를 넘지 않습니다.

### 3.6 child-process 테스트

설정값이 static이므로 `tests/rocm_qmm_env.rs`는 값마다 child 프로세스를 하나씩 차례로 띄우며, 각 케이스를 적용하기 전에 테스트하는 변수를 모두 지웁니다. child는 bf16 `[64, 4096]` x 4비트 `[4096, 4096]` g64 GEMM이 어느 경로를 타는지 `quantized_matmul_matches_dense_gemm`에 묻고(이 함수는 `QuantizedMatmul::eval_gpu`와 `select_qmm_route`를 공유함), 이어서 128행 GEMM 두 번(캐시 크기를 읽는 dequantize 경로)과 1행 GEMV(tile 폭을 읽는 qmv 경로)를 GPU에서 실행해 각각을 CPU stream의 f32 기준값과 비교합니다. parent는 종료 상태, 보고된 경로, 정확한 경고 줄을 확인합니다. 상한과 캐시 케이스에는 `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1`을 설정해 상한만이 경로를 결정하게 합니다.

15개 케이스는 다음과 같습니다. 상한 미설정, `128`, 빈 값, `32`, `4294967297`, `2147483648`, `-1`, `0`, `12abc`; threshold `0`; 캐시 `4294967304`, `0`; tile 폭 `16`, `abc`, `33`. `e54ffc1e`에서 추가한 threshold 케이스는 child의 모든 probe와 GEMM이 threshold를 읽으므로, 캐시된 threshold가 정확히 한 번 경고하는지도 확인합니다.

## 4. 프로덕션 영향

유효한 값과 설정하지 않은 변수는 결과가 전과 같습니다. 잘못된 값은 이제 wrap되거나 조용히 무시되는 대신 프로세스당 stderr 한 줄을 출력하고 기본값을 씁니다.

전에는 받아들이던 값 두 가지의 동작이 바뀝니다.

- `MLX_GRIDX_MULT=0`이나 음수는 예전에 1로 clamp되었고, 이제는 경고하고 4를 씁니다.
- `MLX_ROCM_GROUPED_PREFILL_MIN_B=0`은 예전에 모든 batch 크기를 허용했고, 이제는 경고하고 64를 씁니다.

호출마다 새로 생기는 비용은 없습니다. 모든 읽기가 캐시되며, threshold와 tile 폭은 전보다 덜 자주 읽힙니다. `device.cpp` 변경은 주석입니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않으며, 테스트는 `#![cfg(feature = "rocm")]`입니다.

## 5. 문서

- `docs/environment-variables.md`는 1~2147483647 범위의 정수가 아닌 `MLX_ROCM_WMMA_QMM_MAX_M` 값이 stderr 경고와 함께 무시된다고 기술하고, 다른 GEMM 정수 설정값에 대한 문단을 추가합니다. 경고 형식, 설정하지 않았거나 빈 경우의 조용한 기본값, 설정값별 범위와 기본값을 담습니다.
- LOCAL_FIXES 항목 34가 변경을 기록하며, solution index 읽기 함수의 통합과 `MLX_ROCM_GROUPED_PREFILL_MIN_B=0`이 이제 모든 batch 크기를 허용하는 대신 경고한다는 점도 포함합니다. upstreaming 후보는 아닙니다.
- 오버레이 README의 backend 파일 수가 새 헤더로 인해 108에서 109가 됩니다.

## 6. 검증

gfx1151(Radeon 8060S), `bbd05099` 위로 리베이스한 상태에서 실행했습니다.

- **새 테스트.** `e54ffc1e`에서 `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1`: 15개 케이스 모두 통과. `92de1604`에서는 테스트 케이스가 14개였고 모두 통과했습니다.
- **수정 없이 실패.** qmm.hip, matmul.cpp, gemms/rocblas_gemm.{cpp,h}, device.cpp를 main에서 복원하고 다시 빌드하면 15개 중 9개가 실패합니다. `4294967297`은 dense 경로를 보고하고, 잘못된 값 9개 모두 경고를 출력하지 않습니다. threshold의 `static`만 제거하면 threshold 케이스가 GEMM마다 경고 한 줄씩을 출력하며 실패합니다.
- **기타 게이트.** `make verify-rocm-overlay`(backend 파일 109개, 항목 34개), `verify-versions`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-llama-compat`, `verify-fmt`, `cargo test --test dead_doc_pointers`, `cargo clippy --release --features rocm --test rocm_qmm_env -- -D warnings` 통과.
- **리뷰 확인.** 호출 지점마다 sentinel 기본값을 포함해 범위, 기본값, static 캐싱 방식을 확인했습니다. 새 헤더는 `mlx_apply_source_overlays`(`src/lib/mlx-cpp/CMakeLists.txt`)의 `patches-rocm/mlx/*` `GLOB_RECURSE`로 복사되고, `build.rs`는 `rerun-if-changed=../mlx-cpp/patches-rocm`로 이를 감지합니다. tile 폭의 범위 1~32는 `TILE_N_MAX`와 일치합니다. 항목 번호 34는 비어 있는 다음 번호이고, README의 개수 109는 실제 트리와 같습니다. LOCAL_FIXES 항목 34와 문서를 코드와 대조했습니다.
- **전체 ROCm 게이트.** `bbd05099` 위의 head `01a28a10`에서 `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`로 실행한 `make verify-rocm`: `[verify-rocm] OK`, cargo 테스트 스위트 153개에서 11997개 통과, 0개 실패, 384개 무시(mlxcel-core lib 스위트에서 8853개 통과, 156개 무시), ROCm 스모크는 GPU에서 32개 토큰을 생성했습니다.

검증하지 않은 것: 이 호스트에 없는 Metal과 CUDA. 변경은 해당 빌드가 복사하지 않는 `patches-rocm/`과 ROCm 전용 테스트만 건드립니다.

## 7. 기술적 선택과 그 이유

주된 선택은 solution index 읽기 함수를 어디에 둘지였습니다. 세 파일이 각자 anonymous namespace 안에 복사본을 두었으므로, static도 파일마다 따로 있었습니다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 복사본을 두고 "파일당 한 번"이라고 문서화 | 코드 이동 없음 | 경고가 여전히 반복됨. 이슈는 한 줄을 요구함 |
| **선택: `gemms/rocblas_gemm.h`의 inline 함수** | 프로세스당 static 하나(inline 함수의 static은 translation unit 간에 공유됨), 코드 44줄 감소 | `matmul.cpp`가 `rocblas_gemm.h`를 include하고 `rocblas_gemm.h`가 수정된 오버레이 파일이 됨 |

`rocblas_gemm.h`는 이미 qmm.hip과 rocblas_gemm.cpp가 include하고 있었고, matmul.cpp에는 이 헤더와 충돌하는 이름이 없습니다.

- **새 규칙이 아니라 #2073 규칙을 재사용.** GEMM 설정값은 이제 `MLX_ROCM_GPU_WATCHDOG_SECS`, `MLX_ROCM_FFT_CACHE_SIZE`와 같은 형식으로 경고합니다.
- **메시지에서 sentinel 기본값의 이름을 출력.** `default_description`은 `-1` 대신 "the per-device default"나 "the built-in crossover"를 출력합니다.
- **tile 폭을 `TILE_N_MAX`로 제한.** 더 큰 값은 tiled 커널의 launch 상한을 넘습니다.
- **0을 다른 범위 밖 값과 똑같이 처리.** `MLX_GRIDX_MULT=0`과 `MLX_ROCM_GROUPED_PREFILL_MIN_B=0`은 별도의 clamp나 전체 허용 값을 유지하는 대신, 다른 모든 설정값과 같은 규칙으로 경고하고 기본값을 씁니다.
- **child 프로세스에서 테스트.** static은 프로세스당 한 번 읽히므로, 값마다 새 프로세스를 띄워야만 파싱과 프로세스당 한 번 경고를 함께 테스트할 수 있습니다.

## 8. 남은 위험과 후속 작업

- **wave64 디바이스에서 `MLX_ROCM_QMV_TILE_N=32`**는 블록당 2048 스레드를 요청합니다. PR 이전부터 있던 문제이고 gfx1151은 wave32입니다. 나중에 wave64 디바이스에서 tile 폭의 상한을 런타임 wavefront 크기 기준(1024 / warp)으로 제한할 수 있습니다.
- **캐시 크기 0은 관찰하지 않음.** 테스트는 `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0`이 경고하지 않음을 확인하지만, 실제로 캐시를 끄는지는 관찰하지 않습니다.
- **#2180 문서.** #2180의 ROCm 표에 `MLX_ROCM_QMV_TILE_N`과 `MLX_ROCM_MOE_SEG_MIN`의 새 범위를 기술해야 합니다. boolean과 presence 설정값은 그곳에서도 범위 밖이며, 이 PR은 그 동작을 바꾸지 않습니다.
- **`MLX_GRAPH_REPLAY_SLOTS`**: device.cpp의 이 변수는 `std::max<size_t>(2, atoi(e))` 때문에 `-1`을 `SIZE_MAX`로 바꿉니다. #2180에 따라 범위 밖이며 별도 이슈가 필요합니다.

## 9. 학습 포인트

- **inline 함수 안의 static.** anonymous namespace 안에 있는 non-inline 함수의 지역 static은 그 함수를 정의한 translation unit마다 별도의 객체입니다. external linkage를 가진 `inline` 함수 안의 static은 프로그램 전체에 객체가 하나뿐입니다. 여러 파일이 같은 변수를 읽을 때 "프로세스당 한 번 경고"가 성립하는 이유가 이것입니다.
- **실패할 수 있는 테스트.** child-process 테스트의 가치는 수정이 없을 때 실패한다는 데 있습니다. 이를 두 번 측정했습니다. 한 번은 오버레이 변경 전체를 되돌렸고, 한 번은 threshold의 `static`만 제거했습니다.
