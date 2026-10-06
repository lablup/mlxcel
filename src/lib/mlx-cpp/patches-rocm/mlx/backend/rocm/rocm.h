// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace mlx::core::rocm {

/* Check if the ROCm backend is available. */
MLX_API bool is_available();

// The wavefront width of the current HIP device as the hardware reports it
// (`hipDeviceAttributeWarpSize`): 32 on RDNA, 64 on CDNA. Unlike
// `Device::warp_size()`, it never follows `MLX_ROCM_FORCE_WARP_SIZE`, so a
// caller deciding whether a kernel written for one width may run reads the
// device, not a launch-width experiment. Returns 0 when the query fails
// (lablup/mlxcel#2147).
MLX_API int device_warp_size();

// Deterministic bump arena (shared decode/train region). Opt-in for future
// HIP-graph train capture — does NOT enable graphs by itself.
// capacity_bytes: backing HBM region; returns false on alloc failure.
MLX_API bool train_arena_begin(size_t capacity_bytes);
MLX_API void train_arena_reset();
MLX_API void train_arena_end();
MLX_API bool train_arena_active();
MLX_API size_t train_arena_high_water();
MLX_API bool train_arena_overflowed();

// Fused sorted-MoE SwiGLU (one D2H sync for the whole gate/up/silu/down).
// x: [T,D] bf16
// w_gate, w_up: [E,D,I] bf16  (lemonseed gather_mm layout after swapaxes)
// w_down: [E,I,D] bf16
// expert_ids: [T] uint32 sorted by expert id
// returns y: [T,D] bf16
MLX_API array moe_swiglu_sorted(
    const array& x,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids,
    StreamOrDevice s = {});

// Fused sorted-MoE SwiGLU VJP (one D2H sync for recompute + all grads).
// x: [T,D]  w_gate/up: [E,D,I]  w_down: [E,I,D]  ids: [T]  dy: [T,D]  (bf16)
// returns {dx[T,D], dw_gate[E,D,I], dw_up[E,D,I], dw_down[E,I,D]}
MLX_API std::vector<array> moe_swiglu_sorted_vjp(
    const array& x,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids,
    const array& dy,
    StreamOrDevice s = {});

// True when QuantizedMatmul on GPU `device_index`, for a transposed affine
// weight and one [M, K] input with no batch dimensions (M >= 2), dequantizes
// the weight in the input dtype and runs the GEMM through hipBLASLt. That is
// the path `matmul` takes for a dense f16 or bf16 weight, so on this route
// `dequantize` + `matmul` returns the same bytes as `quantized_matmul`; on the
// others (the fused WMMA kernel, the fp8 path, qmv) it does not. Reads the same
// route selection QuantizedMatmul uses, including its environment overrides
// (lablup/mlxcel#2081).
MLX_API bool quantized_matmul_runs_dequant_gemm(
    int device_index,
    int M,
    int N,
    int K,
    Dtype x_dtype,
    Dtype scales_dtype,
    std::optional<Dtype> biases_dtype,
    int group_size,
    int bits);

} // namespace mlx::core::rocm
