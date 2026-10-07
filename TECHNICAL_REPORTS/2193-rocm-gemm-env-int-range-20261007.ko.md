# 기술 리포트: PR #2193 - ROCm GEMM 환경 변수 정수 범위 검사와 오래된 그래프 주석 수정

**작성일**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. head `e54ffc1e`(`bbd05099` 기준), PR 열림, 머지 대기. Closes #2152.

**언어**: C++/HIP (`src/lib/mlx-cpp/patches-rocm/` 아래 ROCm 오버레이), Rust (`tests/rocm_qmm_env.rs`), Markdown (`LOCAL_FIXES.md`, 오버레이 README, `docs/environment-variables.md`)

**위험도**: 낮음. 잘못된 값만 동작이 달라진다. 예전에는 조용히 wrap되거나 무시되었고, 이제는 경고를 출력하고 기본값을 쓴다. 유효한 값과 설정하지 않은 변수는 결과가 전과 같다. Metal과 CUDA 빌드는 ROCm 오버레이를 복사하지 않는다.

## 요약

ROCm 오버레이는 정수 GEMM 설정값을 `strtol`에 검사 없는 `static_cast<int>`를 붙이거나 `atoi`로 읽었다. 그래서 범위를 벗어난 입력이 아무 메시지 없이 wrap되었다. `MLX_ROCM_WMMA_QMM_MAX_M=4294967297`은 상한 1이 되어, 두 행 이상인 bf16 GEMM을 모두 fused WMMA 커널 밖으로 보냈다. 이 PR은 새 헤더 `mlx/backend/rocm/env_int.h`에 헬퍼 `env_int_or_default` 하나를 추가한다. 이 헬퍼는 #2073이 `MLX_ROCM_GPU_WATCHDOG_SECS`와 `MLX_ROCM_FFT_CACHE_SIZE`에 정한 규칙을 그대로 따른다. 이제 설정값 11개가 이 헬퍼를 거친다. 각 읽기 결과는 static에 캐시되므로 잘못된 값은 프로세스당 stderr 한 줄만 출력한다. 리뷰에서 두 가지를 발견했다. solution index 읽기 함수가 아직 세 파일에 중복되어 잘못된 index가 최대 세 번 경고했고, threshold의 새 캐싱을 확인하는 테스트가 없었다. 둘 다 `e54ffc1e`에서 고쳤다.

## 1. 문제 정의

### 1.1 배경

LOCAL_FIXES 항목 25(#2073)가 ROCm 환경 변수 파싱 규칙을 정했다. `errno`를 0으로 두고 `strtol`로 10진수를 파싱하며, 문자열 전체가 숫자여야 한다. `ERANGE`, 숫자가 아닌 문자, 범위 밖의 값은 `[ROCm] ignoring invalid NAME="value" (expected ...); using ...`을 출력하고 기본값을 쓴다. 설정하지 않았거나 빈 변수는 조용히 기본값을 쓴다. GEMM 설정값은 이 규칙보다 먼저 만들어졌다.

### 1.2 기존 문제점

- **조용한 wrap**: `parse_{positive,non_negative}_int_env`(qmm.hip, 그리고 matmul.cpp와 gemms/rocblas_gemm.cpp의 복사본)와 `dequant_cache_capacity()`, `moe_segment_min_avg()`의 인라인 파싱은 `long`을 범위 검사 없이 `int`로 캐스팅했다. `4294967297`은 1이 되었고, `2147483648`은 음수가 되어 무시되었으며, `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=4294967304`는 8이 되었다.
- **atoi 읽기**: `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B`, `MLX_GRIDX_MULT`는 `atoi`를 썼다. 게다가 `MLX_ROCM_QMV_TILE_N`은 qmv 호출마다 다시 읽혔고, tiled 커널의 `__launch_bounds__(TILE_N_MAX * WARP_SIZE)`(`TILE_N_MAX = 32`)를 넘을 수 있었다.
- **호출마다 읽기**: `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD`는 `should_use_dequant_gemm_path`를 호출할 때마다 읽혔다. 그래서 이 값에 대한 경고는 GEMM마다 한 번씩 출력되었을 것이다.
- **오래된 주석**: device.cpp의 `use_hip_graphs()` 주석은 prefill이 WMMA GEMM을 쓴다고 했다. 항목 29(`select_qmm_route`) 이후로는 사실이 아니다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 설정값 오타가 GEMM 경로와 prefill 속도를 조용히 바꿈 | 중간 | 낮음 |
| 범위를 벗어난 tile 폭이 qmv launch를 깨뜨림 | 중간 | 낮음 |

## 2. 기술적 검토 사항

### 2.1 적용한 검토 기준

호출 지점마다 범위, 기본값, static 캐싱 방식을 확인했다. sentinel 기본값도 함께 보았다. batched solution index, threshold, 상한은 -1이고 tile 폭은 0이며, 브리프대로 이 기본값 자체는 범위 검사를 하지 않는다. `MLX_ROCM_QMV_TILE_N`의 상한을 tiled 커널 launch와 대조했고, `MLX_GRIDX_MULT`의 동작 변경과 64비트 `grid_x` 계산을 검토했다. 오버레이 통합 여부는 `src/lib/mlx-cpp/CMakeLists.txt`와 `build.rs`에서 확인했다. LOCAL_FIXES 항목 34와 문서를 코드와 대조했고, 테스트가 실패할 수 있음을 보이려고 수정을 되돌린 상태로도 실행했다.

### 2.2 발견 사항

| 발견 사항 | 심각도 | 상태 |
|-----------|--------|------|
| `gemm_solution_index_{f32,bf16}`이 세 translation unit에 있고 각자 static을 가져, 잘못된 `MLX_ROCM_GEMM_*_SOLUTION_INDEX`가 프로세스당 최대 세 번 경고함 | 중간 | 수정: `gemms/rocblas_gemm.h`의 inline 함수 한 쌍으로 통합 |
| threshold를 이제 한 번만 읽는지 확인하는 테스트가 없음 | 중간 | 수정: 잘못된 threshold 케이스 추가 |
| batched fallback 메시지가 "using MLX_ROCM_GEMM_F32_SOLUTION_INDEX"로 표시됨 | 낮음 | 수정: "using the ... value" |
| LOCAL_FIXES 항목 34가 `MLX_ROCM_GROUPED_PREFILL_MIN_B=0`의 동작 변경을 언급하지 않음 | 낮음 | 수정 |
| wave64 디바이스에서 `MLX_ROCM_QMV_TILE_N=32`는 블록당 2048 스레드를 요청함 | 낮음 | 남음: PR 이전부터 있던 문제이고 gfx1151은 wave32 |
| 캐시 크기 0이 실제로 캐시를 끄는지 테스트가 관찰하지 않음 | 낮음 | 남음 |

정확하다고 확인한 사항: 새 헤더는 `mlx_apply_source_overlays`의 `patches-rocm/mlx/*` `GLOB_RECURSE`로 복사되고, 빌드는 `rerun-if-changed=../mlx-cpp/patches-rocm`로 이를 감지한다. `grid_x`는 좁히기 전에 `int64_t`에서 최솟값을 구하므로 결과가 `(B + 63) / 64`를 넘지 않는다. tile 폭의 범위 1~32는 `TILE_N_MAX`와 일치한다. 항목 번호 34는 비어 있는 다음 번호이고, README의 개수 109는 실제 트리와 같다.

## 3. 기술적 선택과 그 이유

### 3.1 solution index 읽기 함수 통합

**컨텍스트:** 세 파일이 각자 anonymous namespace 안에 읽기 함수 복사본을 두었으므로, static도 파일마다 따로 있었다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 복사본을 두고 "파일당 한 번"이라고 문서화 | 코드 이동 없음 | 경고가 여전히 반복됨. 이슈는 한 줄을 요구함 |
| **선택: `gemms/rocblas_gemm.h`의 inline 함수** | 프로세스당 static 하나(inline 함수의 static은 translation unit 간에 공유됨), 코드 44줄 감소 | `matmul.cpp`가 `rocblas_gemm.h`를 include하고 `rocblas_gemm.h`가 수정된 오버레이 파일이 됨 |

`rocblas_gemm.h`는 이미 qmm.hip과 rocblas_gemm.cpp가 include하고 있었고, matmul.cpp에는 이 헤더와 충돌하는 이름이 없다.

## 4. 구현 상세

`env_int_or_default(name, default, min, max, expected, default_description = nullptr)`는 설정하지 않았거나 빈 변수에 `default`를 돌려준다. 그 밖의 경우 `strtol`로 10진수를 파싱하고, `end == raw`, 뒤에 남는 문자, `ERANGE`, `[min, max]` 밖의 값을 거부하며 한 줄을 출력한다. `default_description`은 `-1`을 그대로 출력하는 대신 sentinel 기본값의 이름("the per-device default", "the built-in crossover")을 출력한다.

범위: threshold, 상한, `MLX_ROCM_MOE_SEG_MIN`, `MLX_ROCM_GROUPED_PREFILL_MIN_B`, `MLX_GRIDX_MULT`는 1~`INT_MAX`이다. solution index와 캐시 크기는 0~`INT_MAX`이고, 0은 여전히 캐시를 끈다. `MLX_ROCM_QMV_TILE_N`은 1~`TILE_N_MAX`이다.

전에는 받아들이던 값 중 동작이 바뀐 것: `MLX_GRIDX_MULT=0`이나 음수는 예전에 1로 clamp되었고, 이제는 경고하고 4를 쓴다. `MLX_ROCM_GROUPED_PREFILL_MIN_B=0`은 예전에 모든 batch 크기를 허용했고, 이제는 경고하고 64를 쓴다.

## 5. 학습 포인트

### 5.1 inline 함수 안의 static

anonymous namespace 안에 있는 non-inline 함수의 지역 static은 그 함수를 정의한 translation unit마다 별도의 객체다. external linkage를 가진 `inline` 함수 안의 static은 프로그램 전체에 객체가 하나뿐이다. 여러 파일이 같은 변수를 읽을 때 "프로세스당 한 번 경고"가 성립하는 이유가 이것이다.

### 5.2 실패할 수 있는 테스트

설정값이 static이므로 child-process 테스트는 값마다 프로세스를 하나씩 띄운다. 이 테스트의 가치는 수정이 없을 때 실패한다는 데 있다. 이를 두 번 측정했다. 한 번은 오버레이 변경 전체를 되돌렸고, 한 번은 threshold의 `static`만 제거했다.

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|----|
| 변경된 파일 | 10 (이 리포트 제외) |
| 추가된 테스트 | 테스트 파일 1개 (child 케이스 15개) |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `92de1604` | fix | range-check GEMM env integers and fix a stale graph comment |
| `e54ffc1e` | fix | read each GEMM solution index once per process |

## 8. 후속 조치

### 완료 필요
- [ ] #2180: ROCm 표에 `MLX_ROCM_QMV_TILE_N`과 `MLX_ROCM_MOE_SEG_MIN`의 새 범위를 기술한다. boolean과 presence 설정값은 그곳에서도 범위 밖이다.

### 향후 개선 사항
- wave64 디바이스에서는 `MLX_ROCM_QMV_TILE_N`의 상한을 런타임 wavefront 크기 기준(1024 / warp)으로 제한한다.
- device.cpp의 `MLX_GRAPH_REPLAY_SLOTS`는 `std::max<size_t>(2, atoi(e))` 때문에 `-1`을 `SIZE_MAX`로 바꾼다. 별도 이슈가 필요하다.

## 부록

### A. 테스트 결과

gfx1151, `e54ffc1e` 기준: `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1`은 15개 케이스를 모두 통과했다. qmm.hip, matmul.cpp, gemms/rocblas_gemm.{cpp,h}, device.cpp를 main에서 복원하자 15개 중 9개가 실패했다. `4294967297`은 dense 경로를 보고했고, 잘못된 값 9개 모두 경고를 출력하지 않았다. threshold의 `static`만 제거하면 threshold 케이스가 GEMM마다 경고 한 줄씩을 출력하며 실패했다. `make verify-rocm-overlay`(backend 파일 109개, 항목 34개), `verify-versions`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-llama-compat`, `verify-fmt`, 그리고 테스트에 대한 `-D warnings` clippy가 모두 통과했다. Metal과 CUDA는 이 호스트에서 쓸 수 없어 검증하지 않았다.
