# PR #2624: ROCm gather QMV 출력 오프셋을 64비트로 확장

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 낮음

## 요약

업스트림 ml-explore/mlx#4738(CUDA gather-QMM int64 stride)을 따라, ROCm 오버레이의 gather QMV 경로에 남아 있던 32비트 오프셋 계산을 `int64_t`로 바꿨습니다. 이슈는 배치 `B = tokens * top_k`를 grid z에 두는 gather 커널이 65535 블록을 넘으면 실행되지 않을 수 있다고 의심했지만, gfx1151에서 측정한 결과 그런 제한은 없었으므로 launch 형상은 그대로 둡니다(이슈 #2620, LOCAL_FIXES 61번).

## 1. 문제 정의

`quantized/qmm.hip`의 `gather_qmv_kernel`은 `out[batch * M * N + row * N + col]`을 `int`로 계산했기 때문에 `[B, M, N]` 출력이 2^31 원소를 넘으면 인덱스가 넘쳤습니다. `quantized/qmv_tiled_kernel.hip`의 `gather_qmv_tiled_kernel`과 `gather_qmv_tiled_6bit_kernel`도 같았고, opt-in tiled 커널은 가중치와 활성값 오프셋도 `uint32_t` 인덱스로 만들었습니다. `gemms/gemv.hip`의 `gemv_gather_impl`은 출력 오프셋 `indices_idx * rows`를 `int`로 계산했습니다. 큰 MoE prefill(2048 토큰, top_k 32, 넓은 expert 출력)이면 이 크기에 도달합니다.

gfx1151에서 `hipDeviceProp_t::maxGridSize`가 y와 z에 대해 65535를 보고하므로, grid z에 65,537행 배치를 두면 실행이 거부되는지가 문제였습니다.

## 3. 기술적 선택과 그 이유

grid z 문제는 커널을 고치기 전에 먼저 확인했습니다. 테스트는 `B = 65537`(expert 128개, `N = K = 64`)인 unsorted `gather_qmm`을 dispatch가 고르는 각 커널(bf16 affine 4비트: wide 커널, bf16 affine 8비트와 bf16 mxfp4: warp-shared 커널, f32 affine 4비트: 행 단위 커널)로 실행했고, 별도의 `hipLaunchKernel` 프로브(ROCm 7.15)는 z 방향 16,777,216 블록, y 방향 262,144 블록까지 모든 블록이 실행됨을 확인했습니다. 보고된 65535는 강제되지 않으며, 실제 제한은 차원당 2^32 - 1 스레드(11번 항목)입니다. 배치를 상한이 있는 grid에 접고 stride 루프를 도는 방식은 모든 gather 커널에 루프와 barrier 위험을 더하면서 얻는 것이 없어 채택하지 않았습니다.

확장 방식은 `gather_qmv_warp_shared_kernel`이 이미 쓰는 패턴을 따릅니다. 64비트 기준 오프셋은 K 루프 전에 한 번 계산하고, 내부 루프 오프셋은 32비트로 두어 64비트 기준 포인터에 더하므로 hot loop에 64비트 연산이 늘지 않습니다.

리뷰에서 첫 bf16 테스트가 가중치를 f32로 양자화해 `gather_qmm`이 활성값을 승격하고 f32 행 단위 커널을 실행했다는 점이 드러났습니다. 이제 가중치를 활성값 dtype으로 양자화하고, 각 케이스가 출력 dtype이 유지되는지 확인하므로 f32 커널로 조용히 빠지면 테스트가 실패합니다.

어느 `CMakeLists.txt`에도 없고 어디에서도 include하지 않으며 같은 32비트 오프셋을 가진 `quantized/qmv_kernel.hip`은 삭제했습니다.

## 7. 변경 요약

- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: `gather_qmv_kernel` 출력 인덱스를 `int64_t`로.
- `overlays/rocm/mlx/backend/rocm/quantized/qmv_tiled_kernel.hip`: 두 tiled gather 커널의 출력 및 expert 오프셋을 `int64_t`로.
- `overlays/rocm/mlx/backend/rocm/gemms/gemv.hip`: `gemv_gather_impl` 출력 오프셋을 `int64_t`로.
- `overlays/rocm/mlx/backend/rocm/quantized/qmv_kernel.hip`: 삭제. `qdequant.hpp` 주석과 README 파일 수 갱신.
- `overlays/rocm/LOCAL_FIXES.md`: 61번 항목. 이 런타임에서 46번 항목의 y grid 거부가 재현되지 않았다는 점도 기록.
- `tests/rocm_hip_grid_clamp/gather_qmv.rs`(신규): `B = 65537` 케이스 4개와, 출력 원소 2^31 + 2^16개에서 2^31 경계 양쪽 행을 확인하는 `#[ignore]` f32 케이스.

## 검증

gfx1151에서 `rocm_gpu_guard.sh --hold`로 실행: `rocm_hip_grid_clamp` 빠른 테스트 13개 통과, bf16 테스트 수정과 rebase 후 `gather` 필터 재통과, ignore된 2^31 초과 테스트 5.8초에 통과. 음성 대조로 `gather_qmv_kernel`의 32비트 인덱스를 되돌리면 ignore 테스트가 grid `[256, 256, 32769]`에서 `hipErrorIllegalAddress`로 실패했습니다. mlxcel-core의 `gather`, `quantized` lib 필터, `rocm_gather_qmm_expert_batched`, 다섯 가지 계약 검사와 clippy 모두 깨끗합니다.

미검증: `MLX_ROCM_GATHER_QMV_USE_TILED`가 프로세스 전역 static으로 한 번만 읽혀 tiled 커널 확장에는 2^31 초과 런타임 테스트가 없습니다. bf16 6비트 tiled gather 커널을 다루는 `B = 65537` 케이스가 없습니다. `gemv_batched`와 `gemv_batched_inline`은 여전히 `int` 배치 오프셋을 씁니다(gather 경로 아님). 46번 항목의 y grid 기록은 프로브 결과와 충돌하므로 다시 확인해야 합니다.
