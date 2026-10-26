# PR #2627: Split per-row softmax and norm grids past the HIP limit

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Low

## Executive Summary

The ROCm softmax, logsumexp, rms_norm and layer_norm launchers placed one block per row in grid x. HIP rejects a launch when `gridDim.x * blockDim.x >= 2^32`, so 2^24 + 1 rows at 256 threads failed with "invalid configuration argument" on gfx1151. Each launcher now caps grid x and spills the remaining rows into grid y, the same fix upstream MLX made for Metal in ml-explore/mlx#4717 (issue #2619, LOCAL_FIXES item 62).

## 1. Problem Statement

Seven launches failed on a 16,777,217 x 4 f32 input before the fix: softmax (256-thread arm), logsumexp, rms_norm plain and strided, layer_norm, and the rms_norm and layer_norm VJPs. The issue had inferred this from the limit recorded in LOCAL_FIXES items 11, 42 and 46; the new tests confirmed it on hardware. `n_rows` was also held as `int`/`int32_t` on the host, and the strided rms_norm kernel decomposed its row in 32 bits.

## 3. Technical Decisions

- **Grid split as in item 11.** `rocm::row_grid(n_rows, block_dim)` caps grid x at `UINT32_MAX / block_dim` and puts `ceil(n_rows / grid_x)` in y. Below the cap grid y is 1 and the launch geometry equals the old `dim3(n_rows)`, so existing workloads see no change.
- **Row from both grid axes.** Kernels compute `row = blockIdx.x + blockIdx.y * gridDim.x` as `int64_t`, take `n_rows` as a new argument, and return when `row >= n_rows`. The return happens before any barrier and applies to whole blocks, so no thread waits on a `__syncthreads` that others skip.
- **File-local helpers.** `row_grid` and `row_grid_index` are repeated in each of the four `.hip` files instead of living in `kernel_utils.hpp`, which every HIP object includes; editing it would rebuild the whole backend.
- **f64 host reference.** The first version compared against the MLX CPU stream. That took about five minutes per test, because the host reads device memory, and the CPU stream's f32 sum over 2^24 rows for the layer_norm weight gradient was 2.7e5 from the GPU value, which is within 1.7e3 of an f64 sum. The tests compare against an f64 reference computed on the host. Weight gradients use rtol 1e-3; the per-row `dx` check is what proves every row ran.
- **Test-only VJP entry points.** mlxcel-core had no VJP binding, so `fast_rms_norm_vjp_for_test` and `fast_layer_norm_vjp_for_test` run the primitives through `mlx::core::vjp`.

## 7. Change Summary

- `overlays/rocm/mlx/backend/rocm/{softmax,logsumexp,rms_norm,layer_norm}.hip`: `int64_t n_rows`, `row_grid` at all 18 launch sites (forward, strided and VJP), 64-bit row index with a bounds return.
- `overlays/rocm/LOCAL_FIXES.md`: item 62.
- `mlxcel-core/src/lib.rs`, `cpp/mlx_cxx_bridge.{h,cpp}`: the two test-only VJP bindings.
- `tests/rocm_hip_grid_clamp/row_grid.rs` (new, included from `tests/rocm_hip_grid_clamp.rs`): seven tests on 2^24 + 1 x 4 f32 and two `#[ignore]`d tests for the 512 and 1024-thread softmax arms (about 17 GiB each).

## Validation

On gfx1151, with the overlay reverted, the seven tests gave `0 passed; 7 failed`, each with `hipLaunchKernel ... invalid configuration argument`. After the fix they pass in 7.12 s, and the whole `rocm_hip_grid_clamp` file gives 20 passed, 7 ignored. Existing mlxcel-core tests for softmax (10), rms_norm (15), layer_norm (2) and fused_norm_parity (10) pass. Contract checks and clippy are clean.

Not run: the two ignored softmax arm tests (about 17 GiB on a 31 GiB shared host) and `layer_norm_multiwave_mean_snapshot_is_stable`, which needs the `mlxcel` lib test build.

## Follow-ups

- `LayerNormVJP::eval_gpu` allocates the bias gradient but never computes it (a TODO in `layer_norm.hip`); upstream computes it as the row sum of `g`. Found by reading the code, not run.
- The VJP `gw_temp` is shaped `{int(n_rows), axis}`, so more than 2^31 rows would overflow the int32 shape. Not reachable in practice.
