# PR #2624: Widen ROCm gather QMV output offsets to 64 bits

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Low

## Executive Summary

Following upstream ml-explore/mlx#4738 (CUDA gather-QMM int64 strides), the ROCm overlay's remaining 32-bit offset sites in the gather QMV paths now compute in `int64_t`. The issue also suspected that gather kernels, which put the batch `B = tokens * top_k` in grid z, could not launch past 65535 blocks. Measurement on gfx1151 ruled that out, so the launch geometry is unchanged (issue #2620, LOCAL_FIXES item 61).

## 1. Problem Statement

`gather_qmv_kernel` in `quantized/qmm.hip` wrote `out[batch * M * N + row * N + col]` in `int`, so a `[B, M, N]` output wrapped once it passed 2^31 elements. `gather_qmv_tiled_kernel` and `gather_qmv_tiled_6bit_kernel` in `quantized/qmv_tiled_kernel.hip` did the same, and the opt-in tiled kernel also formed its weight and activation offsets from `uint32_t` indices. `gemv_gather_impl` in `gemms/gemv.hip` offset its output by `indices_idx * rows` in `int`. A large MoE prefill (2048 tokens at top_k 32 with a wide expert output) reaches these sizes.

`hipDeviceProp_t::maxGridSize` reports 65535 for y and z on gfx1151, which raised the question of whether a batch of 65,537 rows in grid z fails to launch.

## 3. Technical Decisions

The grid-z question was settled before any kernel change. A test ran an unsorted `gather_qmm` at `B = 65537` (128 experts, `N = K = 64`) through each kernel the dispatch picks (bf16 affine 4-bit: wide kernel; bf16 affine 8-bit and bf16 mxfp4: warp-shared kernel; f32 affine 4-bit: per-row kernel), and a standalone `hipLaunchKernel` probe (ROCm 7.15) launched grids of up to 16,777,216 blocks in z and 262,144 in y with every block executing. The advertised 65535 is not enforced; the limit that applies is 2^32 - 1 threads per dimension (item 11). Folding the batch into a capped grid with a stride loop would have added a loop and a barrier hazard to every gather kernel for no gain, so it was not done.

The widening follows the pattern already used by `gather_qmv_warp_shared_kernel`: the 64-bit base offset is computed once before the K loop, and inner-loop offsets stay 32-bit and are added to 64-bit base pointers, so the hot loops gain no 64-bit arithmetic.

Review found that the first version of the bf16 tests quantized weights in f32, which made `gather_qmm` promote the activations and run the f32 per-row kernel. The weights are now quantized in the activation dtype, and each case asserts that the output keeps that dtype, so a silent fallback to the f32 kernel fails the test.

The dead `quantized/qmv_kernel.hip` (in no `CMakeLists.txt`, included nowhere, and carrying the same 32-bit offsets) was removed.

## 7. Change Summary

- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: `gather_qmv_kernel` output index in `int64_t`.
- `overlays/rocm/mlx/backend/rocm/quantized/qmv_tiled_kernel.hip`: `int64_t` output and expert offsets in both tiled gather kernels.
- `overlays/rocm/mlx/backend/rocm/gemms/gemv.hip`: `int64_t` output offset in `gemv_gather_impl`.
- `overlays/rocm/mlx/backend/rocm/quantized/qmv_kernel.hip`: deleted; `qdequant.hpp` comment and README file count updated.
- `overlays/rocm/LOCAL_FIXES.md`: item 61, which also notes that this runtime did not reproduce item 46's y-grid refusal.
- `tests/rocm_hip_grid_clamp/gather_qmv.rs` (new): four `B = 65537` cases and an `#[ignore]` f32 case with 2^31 + 2^16 output elements, checking rows on both sides of element 2^31.

## Validation

On gfx1151 under `rocm_gpu_guard.sh --hold`: the 13 fast `rocm_hip_grid_clamp` tests pass, the `gather` filter passes again after the bf16 test fix and the rebase, and the ignored >2^31 test passes in 5.8 s. As a negative control, restoring the 32-bit index in `gather_qmv_kernel` made the ignored test fail with `hipErrorIllegalAddress` at grid `[256, 256, 32769]`. The mlxcel-core `gather` and `quantized` lib filters, `rocm_gather_qmm_expert_batched`, the five contract checks and clippy are clean.

Unverified: the tiled kernels' widening has no runtime >2^31 test, because `MLX_ROCM_GATHER_QMV_USE_TILED` is read once into a process-wide static; no `B = 65537` case covers the bf16 6-bit tiled gather kernel; `gemv_batched` and `gemv_batched_inline` still use `int` batch offsets (not gather paths); and the item 46 y-grid record conflicts with the probe and should be rechecked.
