# PR #2636: Guard empty and 64-bit sizes in ROCm norm VJP, gemv and GatherQMM

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Low

## Executive Summary

Three ROCm overlay launch paths failed at the edges of their size range: norms of zero rows launched an invalid grid, gemv computed row and batch offsets in `int`, and GatherQMM divided by zero for an empty output. Each now has a guard or a 64-bit offset, and each case has a test that fails or crashes on main (issue #2633, LOCAL_FIXES item 65).

## 1. Problem Statement

- `row_grid(0, ...)` in `rms_norm.hip` and `layer_norm.hip` returns `dim3(1, 0, 1)`, and no forward or VJP path checked for zero rows, so `rms_norm` or `layer_norm` of an empty input failed with "invalid configuration argument". The VJPs also built `gw_temp` and the reduction plans from `static_cast<int>(n_rows)`, which wraps past `INT_MAX`; at 2^31 + 2 rows the column reduce faulted.
- `gemv_impl` read `mat + row * cols` and the batched kernels wrote `out + batch_idx * rows` in `int`, so a matrix or an output past 2^31 elements faulted (`gemv_single` and `gemv_batched_inline` on gfx1151).
- `GatherQMM::eval_gpu` computed `B = out.size() / M / N` before its only empty check, which sat inside the nvfp4 branch, so `M == 0` or `N == 0` raised SIGFPE in every mode.

## 2. Technical Decisions

Empty norms return after their outputs exist: forward after allocating `out`, VJPs after allocating `gx` and zero-filling `gw` and `gb` with `fill_gpu` (a zero-byte buffer when the weight itself is empty). The checks use the element count rather than the row count, so an empty last axis does not reach the `data_size / axis_size` division either.

The issue asked to pass `n_rows` without the cast where the types allow. They do not: `Shape` and `ReductionPlan::shape` hold int32, so the VJPs throw instead. The check runs first in `eval_gpu`, before `check_input`, so an oversized broadcast input fails without a multi-GiB contiguous copy, and it applies only when a weight or bias gradient is reduced. A VJP without either keeps its 64-bit row grid.

The GatherQMM check moved above the division, after the input and output registration, as the issue specified. For an empty nvfp4 output it now returns before the nvfp4 layout checks and the global-scale registration; there is no work for those to guard. One visible consequence: a malformed nvfp4 call whose output is empty now returns an empty array instead of throwing.

The two gemv tests and the row-count test are `#[ignore]`d. The gemv tests need 4 to 4.3 GB; the row-count test allocates nothing with the guard, but on a build without it the VJP copies two 4 GiB inputs before faulting. The large inputs are built on the GPU from small `[rows, 1] + [1, cols]` cycles, so the host never holds them, and the expected values are closed forms checked on rows both sides of the 2^31 boundary.

## 3. Change Summary

- `overlays/rocm/mlx/backend/rocm/layer_norm.hip`, `rms_norm.hip`: empty early returns, zero-filled gradients, `check_vjp_rows`.
- `overlays/rocm/mlx/backend/rocm/gemms/gemv.hip`: 64-bit row and batch offsets.
- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: empty-output return above the division.
- `tests/rocm_hip_grid_clamp/size_guards.rs` (new): six tests, three ignored.
- `overlays/rocm/LOCAL_FIXES.md`: item 65.

## Validation

On gfx1151 under `rocm_gpu_guard.sh --hold`, every new test fails on main (invalid configuration, three memory faults, two SIGFPE) and passes with the fix. The full `rocm_hip_grid_clamp` binary (25 passed, 10 ignored), `rocm_nvfp4_quant`, `rocm_gather_qmm_expert_batched`, clippy on `mlxcel` and `mlxcel-core`, and the five contract checks pass.

Unverified: `gemv_batched` (the non-inline variant, used only above the inline batch-dimension limit) got the same widening without a runtime test, and the strided `rms_norm` forward reaches the empty return through the same check but no test feeds it an empty strided view.
