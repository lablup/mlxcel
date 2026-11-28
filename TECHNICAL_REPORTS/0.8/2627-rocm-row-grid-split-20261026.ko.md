# PR #2627: HIP 그리드 한계를 넘는 행 단위 softmax와 norm 그리드 분할

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 낮음

## 요약

ROCm의 softmax, logsumexp, rms_norm, layer_norm 런처는 행마다 블록 하나를 grid x에 배치했습니다. HIP는 `gridDim.x * blockDim.x >= 2^32`인 런치를 거부하므로, gfx1151에서 256 스레드로 2^24 + 1행을 처리하면 "invalid configuration argument"로 실패했습니다. 이제 각 런처가 grid x에 상한을 두고 남는 행을 grid y로 넘깁니다. 업스트림 MLX가 Metal에서 ml-explore/mlx#4717로 한 수정과 같은 방식입니다(이슈 #2619, LOCAL_FIXES 62번).

## 1. 문제 정의

수정 전 16,777,217 x 4 f32 입력에서 일곱 가지 런치가 실패했습니다. softmax(256 스레드 경로), logsumexp, rms_norm 일반과 strided, layer_norm, 그리고 rms_norm과 layer_norm의 VJP입니다. 이슈는 LOCAL_FIXES 11, 42, 46번에 기록된 한계에서 이를 추론했고, 새 테스트가 하드웨어에서 확인했습니다. 호스트의 `n_rows`도 `int`/`int32_t`였고, strided rms_norm 커널은 행 번호를 32비트로 분해했습니다.

## 3. 기술적 선택과 그 이유

- **11번 항목과 같은 그리드 분할.** `rocm::row_grid(n_rows, block_dim)`는 grid x를 `UINT32_MAX / block_dim`으로 제한하고 y에 `ceil(n_rows / grid_x)`를 둡니다. 상한 아래에서는 grid y가 1이고 런치 형태가 기존 `dim3(n_rows)`와 같으므로 기존 워크로드는 바뀌지 않습니다.
- **두 그리드 축에서 행 계산.** 커널은 `row = blockIdx.x + blockIdx.y * gridDim.x`를 `int64_t`로 계산하고, 새 인자 `n_rows`를 받아 `row >= n_rows`이면 반환합니다. 반환은 배리어 이전에 블록 단위로 일어나므로 일부 스레드만 `__syncthreads`를 건너뛰는 일이 없습니다.
- **파일 로컬 헬퍼.** `row_grid`와 `row_grid_index`는 모든 HIP 오브젝트가 포함하는 `kernel_utils.hpp` 대신 네 `.hip` 파일에 각각 둡니다. 공용 헤더를 고치면 백엔드 전체가 다시 빌드되기 때문입니다.
- **f64 호스트 기준값.** 처음에는 MLX CPU 스트림과 비교했습니다. 호스트가 디바이스 메모리를 읽어 테스트당 약 5분이 걸렸고, layer_norm 가중치 기울기에 대한 CPU 스트림의 f32 합계(2^24행)는 GPU 값과 2.7e5 차이가 났습니다. GPU 값은 f64 합계와 1.7e3 이내였습니다. 그래서 테스트는 호스트에서 계산한 f64 기준값과 비교합니다. 가중치 기울기는 rtol 1e-3을 쓰며, 모든 행이 실행되었다는 증거는 행별 `dx` 검사입니다.
- **테스트 전용 VJP 진입점.** mlxcel-core에는 VJP 바인딩이 없어서 `fast_rms_norm_vjp_for_test`와 `fast_layer_norm_vjp_for_test`가 `mlx::core::vjp`로 프리미티브를 실행합니다.

## 7. 변경 요약

- `overlays/rocm/mlx/backend/rocm/{softmax,logsumexp,rms_norm,layer_norm}.hip`: `int64_t n_rows`, 18개 런치 지점(forward, strided, VJP) 모두에 `row_grid` 적용, 경계 검사가 있는 64비트 행 인덱스.
- `overlays/rocm/LOCAL_FIXES.md`: 62번 항목.
- `mlxcel-core/src/lib.rs`, `cpp/mlx_cxx_bridge.{h,cpp}`: 테스트 전용 VJP 바인딩 두 개.
- `tests/rocm_hip_grid_clamp/row_grid.rs`(신규, `tests/rocm_hip_grid_clamp.rs`에서 포함): 2^24 + 1 x 4 f32 테스트 일곱 개와, 512/1024 스레드 softmax 경로용 `#[ignore]` 테스트 두 개(각 약 17 GiB).

## 검증

gfx1151에서 오버레이를 되돌린 상태로 일곱 테스트를 돌리면 `0 passed; 7 failed`였고, 모두 `hipLaunchKernel ... invalid configuration argument`로 실패했습니다. 수정 후에는 7.12초에 통과하며, `rocm_hip_grid_clamp` 전체는 20개 통과, 7개 무시입니다. 기존 mlxcel-core의 softmax(10), rms_norm(15), layer_norm(2), fused_norm_parity(10) 테스트도 통과합니다. 계약 검사와 clippy도 깨끗합니다.

실행하지 않은 것: 무시된 softmax 경로 테스트 두 개(31 GiB 공유 호스트에서 약 17 GiB)와, `mlxcel` lib 테스트 빌드가 필요한 `layer_norm_multiwave_mean_snapshot_is_stable`.

## 후속 과제

- `LayerNormVJP::eval_gpu`는 bias 기울기를 할당만 하고 계산하지 않습니다(`layer_norm.hip`의 TODO). 업스트림은 `g`의 행 합으로 계산합니다. 코드를 읽고 발견했으며 실행해 보지는 않았습니다.
- VJP의 `gw_temp` 형태가 `{int(n_rows), axis}`라서 2^31행을 넘으면 int32 shape가 넘칩니다. 실제로는 도달하지 않는 크기입니다.
