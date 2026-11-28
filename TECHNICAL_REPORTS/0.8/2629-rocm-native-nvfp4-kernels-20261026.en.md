# PR #2629: Native NVFP4 qmm and gather_qmm kernels on ROCm

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Medium

## Executive Summary

ROCm had no NVFP4 kernel, so ModelOpt NVFP4 checkpoints were repacked to affine 4-bit at load and MLX-native NVFP4 planes were refused (#1806 policy). The ROCm overlay now runs NVFP4 (E2M1 values, E4M3 block scales over groups of 16, optional float32 global scale per expert) natively in `QuantizedMatmul` and `GatherQMM`, and `quant_mode_support(Rocm, Nvfp4)` is `Native` (issue #2617, LOCAL_FIXES item 63).

## 1. Problem Statement

`QuantizedMatmul::eval_gpu` dispatched only group sizes 32, 64 and 128, `qdequant.hpp` had no E4M3 decode, `GatherQMM` read no global-scale input, and the non-affine gather path required group 32. Six tests that load NVFP4 planes were gated to a skip branch on ROCm by `expect_native_nvfp4_load`, and `validate_native_global_scale` in `switch_layers.rs` admitted only Metal and CUDA.

## 3. Technical Decisions

A dedicated kernel file, `quantized/nvfp4_qmv.hip`, holds `nvfp4_qmv_kernel` and `nvfp4_dequantize_kernel` instead of threading a fifth mode through the tiled and warp-shared templates. `nvfp4_qmv_kernel` computes up to eight rows per block and reloads a weight group only when the expert changes, which serves M = 1 decode, the 6 to 16 row speculative-decode verify widths that upstream Metal #4712 targets, and sorted gather prefill with one kernel. Above a row crossover, `select_qmm_route` sends NVFP4 to the existing dequantize + hipBLASLt/rocBLAS route, which also covers non-transposed weights.

The E2M1 decode reuses the mxfp4 scaled-half-bits approach (upstream Metal #4709); the new E4M3 decode is branch-free and exact including subnormals. The global scale follows MLX's `fp_quantize`/`fp_dequantize` in `mlx/ops.cpp` and CUDA `fp_gather_qmv`: the block scale is multiplied by `global_scale[expert] / (448 * 6)`, applied once per output. `QuantizedMatmul` receives no global-scale input from MLX (dense global scales are applied at graph level), so a fourth input in a non-affine mode is now refused instead of ignored.

The kernels use 16-lane column groups, width-16 shuffles and no barriers, and never read `WARP_SIZE`, so they behave the same at wave32 and wave64; no wavefront gate was added. This was reasoned, not measured.

Review added host-side shape checks. `check_nvfp4_planes` requires the scale plane to be `w.shape[:-1] + [w.shape[-1] / 2]` with one-byte entries, because MLX's `validate_quantized_input` compares only the batch prefix and last axis, and an oversized plane would have made the dequantize kernel (which sized its grid from the scales) write past its output. The multiple-of-16 check applies to the quantized axis (`transpose ? K : N`), and grid x is capped at UINT32_MAX / 16.

## 7. Change Summary

- `overlays/rocm/mlx/backend/rocm/quantized/nvfp4_qmv.hip` (new): qmv and dequantize kernels, layout and plane checks.
- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: NVFP4 arms in `QuantizedMatmul` and `GatherQMM`, route selection, gather check ordering.
- `overlays/rocm/mlx/backend/rocm/quantized/qdequant.hpp`: E4M3 decode; fp4 decode moved here.
- `mlxcel-core/src/hardware.rs`: `(Rocm, Nvfp4)` is `Native`; table test updated. `layers.rs`, `ffi_tests.rs`, `src/models/sanitize.rs`, `sanitize_tests.rs`: tests that assumed conversion on ROCm now build that route explicitly.
- `src/models/switch_layers.rs`: `validate_native_global_scale` admits ROCm.
- `tests/rocm_nvfp4_quant.rs`, `tests/rocm_nvfp4_quant/gather.rs` (new): numerical tests against a CPU dense reference.
- Docs: installation ROCm table, environment variables, supported models.

## Validation

On gfx1151 after the post-rebase overlay rebuild: `rocm_nvfp4_quant` 7 of 7 (worst relative error vs `x @ dequantize(w).T` on CPU: f32 2.1e-6, f16 4.9e-4, bf16 5.4e-3, across qmv M = 1, multi-row, dequantize + GEMM, batched, non-transposed, gather sorted and unsorted with and without per-expert global scale, partial column tiles, out-of-range experts returning zeros, and malformed scale planes refused). The six formerly gated tests take the native branch and pass. `rocm_mxfp4_quant`, `rocm_hip_grid_clamp gather_qmv`, the five contract checks and clippy are clean.

Unverified: no real NVFP4 checkpoint was run (none cached, downloads out of scope); wave64 was not run; sorted NVFP4 gather prefill reads each expert's weights once per eight rows and may be slower than the former affine conversion with its expert-batched routes; `nvfp4_qmv_kernel` does not bound `lhs_indices` against the activation batch, like most existing gather kernels.
