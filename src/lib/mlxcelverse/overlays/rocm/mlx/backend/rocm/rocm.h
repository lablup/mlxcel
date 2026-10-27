// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <cstddef>
#include <cstdint>
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

// Counters of QuantizedMatmul's dequantized-weight cache, process-wide since
// start (lablup/mlxcel#2151). `hits` and `misses` count lookups on the cached
// dequantize + GEMM route, `inserts` the entries stored (a refresh of an entry
// under the same key counts), `evictions` the entries dropped for the
// MLX_ROCM_QMM_DEQUANT_CACHE_SIZE or MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES
// limit, and `bypasses` the bf16 GEMMs above the fused kernel's row ceiling,
// which dequantize into a temporary without touching the cache. `entries` and
// `bytes` are what the cache holds now. All zero without ROCm.
// MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 prints the same numbers on stderr at
// process exit.
struct DequantCacheStats {
  uint64_t hits;
  uint64_t misses;
  uint64_t inserts;
  uint64_t evictions;
  uint64_t bypasses;
  size_t entries;
  size_t bytes;
};

MLX_API DequantCacheStats dequant_cache_stats();

// Host-shadow accounting (lablup/mlxcel#2643). On Windows every host access
// to a GPU buffer goes through a pinned host copy of the buffer, its shadow
// (`hipHostMalloc`, bounded by the RAM Windows sees, not by the device
// limit). `bytes` is what the live shadows hold now, summed over the shadowed
// buffers' allocator sizes; `downloads` counts device-to-host copies into a
// shadow (a host access to a buffer whose shadow is stale) and `uploads` the
// host-to-device flushes of a shadow the CPU wrote, both since process start.
// Near zero on Linux, where the CPU dereferences most device memory directly
// (only stream-ordered pool buffers, with the async pool enabled, get a
// shadow), and zero without ROCm. MLX_ROCM_HOST_SHADOW_STATS=1 prints the same numbers on
// stderr at process exit.
struct HostShadowStats {
  size_t bytes;
  uint64_t downloads;
  uint64_t uploads;
};

MLX_API HostShadowStats host_shadow_stats();

// Read-only host pointer to an evaluated array's bytes, the counterpart of
// `array::data<T>()` for a caller that will not write. `data<T>()` cannot tell
// a reader from a writer (its const overload casts to the non-const one), so
// on Windows it marks the shadow dirty and the next kernel reading the buffer
// pays a full host-to-device copy. This accessor refreshes the shadow when it
// is stale and leaves the buffer clean, so nothing is flushed. Same wait as
// `data<T>()`: the device is synchronized before the pointer is returned. For
// host-visible device memory (most buffers on Linux) it is the device pointer
// `data<T>()` returns. The caller evaluates the
// array first and must not write through the pointer (lablup/mlxcel#2643).
MLX_API const void* host_read_ptr(const array& a);

} // namespace mlx::core::rocm
