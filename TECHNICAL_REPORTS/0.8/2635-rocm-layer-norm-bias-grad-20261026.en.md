# PR #2635: Write the layer_norm VJP bias gradient on ROCm

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Low

## Executive Summary

`LayerNormVJP::eval_gpu` in the ROCm overlay allocated the bias gradient `gb` and never wrote it, so `d/db` of `fast::layer_norm` was uninitialized device memory. It is now the column sum of the cotangent, computed with the existing `col_reduce` before the VJP kernel runs, as in the CUDA backend (issue #2632, LOCAL_FIXES item 64).

## 1. Problem Statement

`layer_norm.hip` set `gb` to a fresh `malloc_async` buffer and left `// TODO: Implement proper column reduction for gb`. The value returned was whatever the allocator handed back: zeros for a fresh allocation, a stale buffer after reuse (measured on gfx1151: `0.0` and `-2.6875` where `-2.375` was expected). Any training or gradient path through a ROCm LayerNorm with a bias received a wrong bias gradient without an error. The existing past-the-cap VJP test from #2627 checked only `wrt` 0 and 1.

## 2. Technical Decisions

The fix follows `mlx/backend/cuda/layer_norm.cu`: a `ContiguousStridedReduce` plan over `n_rows` x `axis_size` and `col_reduce(..., Sum, {0}, plan)` into `gb`, which keeps the bias dtype MLX assigns. The reduction is enqueued before the VJP kernel because that kernel overwrites a cotangent donated into `gx` or `gw_temp`; when that donation happened, `gb` is registered as a kernel input as CUDA does.

One deliberate difference from CUDA: the cotangent is reduced through a `[n_rows, axis_size]` view of its buffer (`copy_shared_buffer` with strides `{axis_size, 1}`), not the cotangent itself. `ColReduceArgs` derives the output axes from the input's own shape minus axis 0, so passing an N-D cotangent depends on how those axes collapse; the 2-D view makes "sum over every leading axis" hold by construction. A 3-D test case covers it.

The test bridge `fast_layer_norm_vjp_for_test` now takes a nullable weight (`*const MlxArray`, the pattern of `fast_scaled_dot_product_attention`'s mask) so the no-weight case, where MLX passes a scalar weight to the primitive, is testable. Without a weight `wrt` 1 is an error.

The 2^24 + 1 row test needed a different cotangent: its 97-value cycle sums to about zero per cycle, and a zeroed `gb` passed the column-sum check at the test's tolerance. A +0.5 offset makes each sum about 8.4M.

## 3. Change Summary

- `overlays/rocm/mlx/backend/rocm/layer_norm.hip`: bias gradient reduction in `LayerNormVJP::eval_gpu`.
- `mlxcel-core` bridge (`mlx_cxx_bridge.{h,cpp}`, `src/lib.rs`): nullable weight on `fast_layer_norm_vjp_for_test`.
- `tests/rocm_hip_grid_clamp/layer_norm_bias_grad.rs` (new): f32, f16 and bf16, with and without a weight, on `[24, 8]` (`col_reduce_small`) and `[4, 37, 33]` (`col_reduce_looped`).
- `tests/rocm_hip_grid_clamp/row_grid.rs`: `wrt = 2` at 2^24 + 1 rows with the offset cotangent.
- `overlays/rocm/LOCAL_FIXES.md`: item 64. No core file changed, so `CORE_RESIDUAL.diff` is unchanged.

## Validation

On gfx1151 under `rocm_gpu_guard.sh --hold`, all three new checks fail on main (`got 0.0, expected -2.375`; `got -2.6875, expected -2.375`; `got [0, 0, 0, 0], expected [8388608, ...]`) and pass with the fix. The full `rocm_hip_grid_clamp` binary passes (22 passed, 7 ignored), clippy on `mlxcel` and `mlxcel-core` lib and tests is clean, and the five contract checks pass.

Unverified: the donated-cotangent ordering has no direct test, because the bridge holds the cotangent and MLX never donates it; it relies on in-order kernel enqueue on one stream. Zero-row input is handled by #2633.
