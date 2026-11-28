# PR #2636: ROCm norm VJP, gemv, GatherQMM의 빈 크기와 64비트 크기 보호

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 낮음

## 요약

ROCm overlay의 세 launch 경로가 크기 범위의 경계에서 실패했습니다. 행이 0개인 norm은 잘못된 grid를 launch했고, gemv는 행과 batch 오프셋을 `int`로 계산했으며, GatherQMM은 빈 출력에서 0으로 나눴습니다. 이제 각각 보호 조건이나 64비트 오프셋을 갖추었고, 각 경우마다 main에서 실패하거나 크래시하는 테스트가 있습니다 (이슈 #2633, LOCAL_FIXES 항목 65).

## 1. 문제

- `rms_norm.hip`과 `layer_norm.hip`의 `row_grid(0, ...)`는 `dim3(1, 0, 1)`을 반환하고, forward와 VJP 어디에서도 행 0개를 확인하지 않아 빈 입력의 `rms_norm`, `layer_norm`이 "invalid configuration argument"로 실패했습니다. VJP는 또한 `gw_temp`와 reduction plan을 `static_cast<int>(n_rows)`로 만들었는데, 이 값은 `INT_MAX`를 넘으면 wrap되어 2^31 + 2행에서 column reduce가 fault를 냈습니다.
- `gemv_impl`은 `mat + row * cols`를, batched 커널은 `out + batch_idx * rows`를 `int`로 계산해, 2^31 원소를 넘는 행렬이나 출력에서 fault가 났습니다 (gfx1151의 `gemv_single`, `gemv_batched_inline`).
- `GatherQMM::eval_gpu`는 nvfp4 분기 안에만 있던 빈 출력 검사보다 먼저 `B = out.size() / M / N`을 계산해, 모든 mode에서 `M == 0` 또는 `N == 0`이면 SIGFPE가 발생했습니다.

## 2. 기술적 결정

빈 norm은 출력이 준비된 뒤 반환합니다. forward는 `out` 할당 후, VJP는 `gx`를 할당하고 `gw`, `gb`를 `fill_gpu`로 0으로 채운 뒤 반환합니다 (weight 자체가 비어 있으면 0바이트 버퍼). 검사는 행 수가 아니라 원소 수로 하므로, 마지막 축이 비어 있어도 `data_size / axis_size` 나눗셈에 도달하지 않습니다.

이슈는 타입이 허용하면 `n_rows`를 cast 없이 넘기라고 했지만, `Shape`와 `ReductionPlan::shape`가 int32라서 허용되지 않습니다. 그래서 VJP는 예외를 던집니다. 이 검사는 `eval_gpu`의 맨 앞, `check_input`보다 먼저 실행되므로, 너무 큰 broadcast 입력은 수 GiB의 contiguous 복사 없이 실패합니다. weight나 bias 기울기를 reduce할 때만 적용되며, 둘 다 없는 VJP는 64비트 row grid를 그대로 씁니다.

GatherQMM 검사는 이슈가 지정한 대로 입력과 출력 등록 뒤, 나눗셈 앞으로 옮겼습니다. 빈 nvfp4 출력은 이제 nvfp4 layout 검사와 global scale 등록 전에 반환하는데, 그 검사들이 보호할 작업이 없기 때문입니다. 눈에 보이는 결과가 하나 있습니다. 출력이 비어 있는 잘못된 nvfp4 호출은 이제 예외 대신 빈 배열을 반환합니다.

gemv 테스트 두 개와 행 수 테스트는 `#[ignore]`입니다. gemv 테스트는 4~4.3 GB가 필요합니다. 행 수 테스트는 보호 조건이 있으면 아무것도 할당하지 않지만, 보호 조건이 없는 빌드에서는 VJP가 fault 전에 4 GiB 입력 두 개를 복사합니다. 큰 입력은 작은 `[rows, 1] + [1, cols]` 주기로 GPU에서 만들어 호스트가 보유하지 않으며, 기대값은 닫힌 식으로 2^31 경계 양쪽의 행에서 확인합니다.

## 3. 변경 요약

- `overlays/rocm/mlx/backend/rocm/layer_norm.hip`, `rms_norm.hip`: 빈 입력 조기 반환, 0으로 채운 기울기, `check_vjp_rows`.
- `overlays/rocm/mlx/backend/rocm/gemms/gemv.hip`: 64비트 행 및 batch 오프셋.
- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: 나눗셈 앞의 빈 출력 반환.
- `tests/rocm_hip_grid_clamp/size_guards.rs` (신규): 테스트 여섯 개, 그중 세 개는 ignore.
- `overlays/rocm/LOCAL_FIXES.md`: 항목 65.

## 검증

gfx1151에서 `rocm_gpu_guard.sh --hold`로 실행했을 때 새 테스트는 모두 main에서 실패하고 (invalid configuration, memory fault 세 건, SIGFPE 두 건) 수정 후 통과합니다. `rocm_hip_grid_clamp` 바이너리 전체 (25 통과, 10 무시), `rocm_nvfp4_quant`, `rocm_gather_qmm_expert_batched`, `mlxcel`과 `mlxcel-core`의 clippy, 다섯 가지 contract 검사가 통과합니다.

미검증: `gemv_batched`(inline batch 차원 한도를 넘을 때만 쓰는 non-inline 변형)는 같은 확장을 받았지만 런타임 테스트가 없고, strided `rms_norm` forward는 같은 검사로 조기 반환에 도달하지만 빈 strided view를 넣는 테스트는 없습니다.
