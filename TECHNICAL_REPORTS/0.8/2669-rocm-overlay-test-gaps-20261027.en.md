# PR #2669: Cover the donated LayerNorm cotangent, the gemv fallback and empty norms on ROCm

**Date**: 2026-10-27
**Status**: Complete on Linux gfx1151 (ROCm 7.15)
**Languages**: Rust, Markdown
**Risk Level**: Low (tests and records only)

## Executive Summary

Issue #2663 listed four places in the ROCm overlay whose tests passed only because no test reached the code. PR #2669 adds tests for three of them and corrects a stale record for the fourth. No kernel code changes. Each new test was shown to fail with its path deliberately broken. LOCAL_FIXES item 68 records the work.

## 1. Problem Statement

- `LayerNormVJP::eval_gpu` reduces the bias gradient before the VJP kernel, because the kernel overwrites a cotangent donated into `gx` or the weight-gradient temporary (item 64). The test bridge passes a caller-held cotangent, which is never donatable, so that ordering was untested.
- `gemv` launches `gemv_batched` with its shape and strides in device memory only past eight batch dimensions. No test had nine.
- The zero-row norm test did not cover `rms_norm` without a weight, or an empty input on the `strided` branch of `RMSNorm::eval_gpu`.
- LOCAL_FIXES item 46 said y grids above 65535 blocks are refused and that its row and tile tests failed without the fix. Item 61 had already measured that ROCm 7.15 does not enforce that limit.

## 2. Change Summary

| Area | Change |
|---|---|
| LayerNorm VJP | A transposed cotangent is copied by `check_input`, and the copy is donatable. With a held contiguous `x` it is donated into `gx`. With a transposed `x` and a weight it is donated into the weight-gradient temporary. Both cases, with and without a weight, run in f32, f16 and bf16 at 24 x 8 and 148 x 33. `gx`, `gw` and `gb` are compared with a host f64 reference. |
| gemv | `A @ v` over nine size-2 batch axes. The broadcast strides alternate between the matrix and the vector, so `collapse_contiguous_dims` cannot merge them. All 512 batches are checked on the host. |
| Norms | Adds zero-row `rms_norm` without a weight. Adds `rms_norm` with and without a weight on an empty `as_strided` view `[0, 8]` with strides `[16, 1]`. |
| Records | Item 46 and the comments in `tests/rocm_hip_grid_clamp.rs` and `gather_qmv.rs` now give the measured limit. The caps stay. |

## 3. Technical Decisions

### 3.1 No bridge change for donation

The issue offered a route that needs no new bridge function: `check_input` sets `donate_g |= g_copied`. A transposed cotangent is therefore enough. A transposed `x` adds the second case, because the copy of `x` takes `gx` and leaves the weight-gradient temporary to the cotangent.

### 3.2 The strided branch is reachable

MLX gives an empty slice fresh row-major strides, so it would be packed. `AsStrided::eval`, however, keeps the given strides and flags any empty view contiguous. A row stride of 16 against an axis of 8 then makes `packed` false and `strided` true. A temporary throw in that branch showed the test reaches it.

## 4. Validation

- **Red, gx donation**: `gb` reduced after the kernel gives `gb element 0 is -0.23338366, expected 1.125`.
- **Red, gw donation**: the same sabotage gives `-16.554958`. The existing item 64 tests still passed under this sabotage.
- **Red, gemv**: zeroed `d_vec_strides` gives `batch 1 (matrix 0, vector 1) row 0: got -23, expected -5`.
- **Red, strided norm**: the test fails with the thrown message.
- **Probe**: `hipLaunchKernel` on HIP 71526333 (`maxGridSize` reports 65535 for y and z) ran every block of y and z grids from 65535 to 16777216 blocks. It refused only 2^32 threads in one dimension.
- **Green**: `rocm_hip_grid_clamp` has 29 passed and 10 ignored. Overlay records, fmt and clippy on the touched test targets pass.

## 5. Known Gaps

- The fast grid tests can no longer be described as a red/green test of the y cap, because an uncapped y grid of that size runs on this runtime. They remain checks of the capped, grid-stride launch.
- The probe ran on ROCm 7.15 only. Another runtime might enforce `maxGridSize`, which is why the caps stay.
