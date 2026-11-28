# PR #2635: ROCm에서 layer_norm VJP의 bias 기울기 기록

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 낮음

## 요약

ROCm overlay의 `LayerNormVJP::eval_gpu`는 bias 기울기 `gb`를 할당만 하고 값을 쓰지 않았기 때문에, `fast::layer_norm`의 `d/db`는 초기화되지 않은 디바이스 메모리였습니다. 이제 CUDA 백엔드와 같이 VJP 커널 실행 전에 기존 `col_reduce`로 cotangent의 열 합을 계산합니다 (이슈 #2632, LOCAL_FIXES 항목 64).

## 1. 문제

`layer_norm.hip`은 `gb`에 새 `malloc_async` 버퍼를 붙이고 `// TODO: Implement proper column reduction for gb`만 남겨 두었습니다. 반환값은 할당자가 준 메모리 그대로였습니다. 새 할당이면 0, 재사용된 버퍼면 이전 값이었습니다 (gfx1151 측정: 기대값 `-2.375`에 대해 `0.0`과 `-2.6875`). bias가 있는 ROCm LayerNorm을 거치는 학습이나 기울기 계산은 오류 없이 잘못된 bias 기울기를 받았습니다. #2627의 기존 cap 초과 VJP 테스트는 `wrt` 0과 1만 확인했습니다.

## 2. 기술적 결정

수정은 `mlx/backend/cuda/layer_norm.cu`를 따릅니다. `n_rows` x `axis_size`에 대한 `ContiguousStridedReduce` plan과 `col_reduce(..., Sum, {0}, plan)`으로 `gb`를 채우며, `gb`는 MLX가 정한 bias dtype을 유지합니다. VJP 커널은 `gx`나 `gw_temp`로 donate된 cotangent를 덮어쓰므로 reduction을 커널보다 먼저 enqueue하고, donate가 일어난 경우 CUDA처럼 `gb`를 커널 입력으로 등록합니다.

CUDA와 의도적으로 다른 점이 하나 있습니다. cotangent를 직접 넘기지 않고 버퍼의 `[n_rows, axis_size]` view(`copy_shared_buffer`, strides `{axis_size, 1}`)를 reduce합니다. `ColReduceArgs`는 입력 shape에서 axis 0을 뺀 나머지로 출력 축을 구성하므로, N차원 cotangent를 그대로 넘기면 그 축들이 어떻게 합쳐지는지에 결과가 달려 있습니다. 2차원 view를 쓰면 "앞쪽 모든 축에 대한 합"이 구조적으로 보장됩니다. 3차원 테스트 케이스가 이를 확인합니다.

테스트 브리지 `fast_layer_norm_vjp_for_test`는 이제 nullable weight(`*const MlxArray`, `fast_scaled_dot_product_attention`의 mask와 같은 방식)를 받아, MLX가 primitive에 스칼라 weight를 넘기는 weight 없는 경우도 테스트할 수 있습니다. weight가 없으면 `wrt` 1은 오류입니다.

2^24 + 1행 테스트는 cotangent를 바꿔야 했습니다. 기존 97값 주기는 주기마다 합이 거의 0이어서, 0으로 채워진 `gb`도 테스트 허용 오차 안에서 통과했습니다. +0.5 오프셋을 더해 각 열 합이 약 840만이 되게 했습니다.

## 3. 변경 요약

- `overlays/rocm/mlx/backend/rocm/layer_norm.hip`: `LayerNormVJP::eval_gpu`의 bias 기울기 reduction.
- `mlxcel-core` 브리지 (`mlx_cxx_bridge.{h,cpp}`, `src/lib.rs`): `fast_layer_norm_vjp_for_test`의 nullable weight.
- `tests/rocm_hip_grid_clamp/layer_norm_bias_grad.rs` (신규): f32, f16, bf16, weight 유무, `[24, 8]` (`col_reduce_small`)와 `[4, 37, 33]` (`col_reduce_looped`).
- `tests/rocm_hip_grid_clamp/row_grid.rs`: 오프셋 cotangent로 2^24 + 1행에서 `wrt = 2` 확인.
- `overlays/rocm/LOCAL_FIXES.md`: 항목 64. core 파일 변경이 없으므로 `CORE_RESIDUAL.diff`는 그대로입니다.

## 검증

gfx1151에서 `rocm_gpu_guard.sh --hold`로 실행했을 때, 새 검사 세 개 모두 main에서 실패하고 (`got 0.0, expected -2.375`; `got -2.6875, expected -2.375`; `got [0, 0, 0, 0], expected [8388608, ...]`) 수정 후 통과합니다. `rocm_hip_grid_clamp` 바이너리 전체가 통과하고 (22 통과, 7 무시), `mlxcel`과 `mlxcel-core`의 lib 및 test에 대한 clippy와 다섯 가지 contract 검사도 통과합니다.

미검증: donate된 cotangent의 순서 보장은 직접 테스트가 없습니다. 브리지가 cotangent를 보유하고 있어 MLX가 donate하지 않기 때문이며, 단일 stream에서 커널이 순서대로 enqueue되는 것에 의존합니다. 행이 0개인 입력은 #2633에서 다룹니다.
