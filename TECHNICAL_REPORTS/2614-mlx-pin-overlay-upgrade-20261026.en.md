# PR #2614: Advance MLX while preserving backend overlays

**Date**: 2026-10-26
**Status**: Implemented; CPU and GB10 validated, other hardware pending
**Languages**: C++, CUDA, CMake
**Risk Level**: Medium

## Executive Summary

Advance the MLX pin from `bff10220` to `7ca40033`, retaining local backend behavior. The 22 upstream commits include Metal small-batch kernels, CUDA large-expert offset fixes and single-root CUDA graphs for GB10. No throughput improvement has been measured in mlxcel.

## 1. Problem Statement

The existing pin already follows stable v0.32.4, so switching to that release would downgrade it. Merely updating the SHA would lose upstream changes behind whole-file overlays and break C++ SDPA callers because upstream inserted a window-size argument.

## 3. Technical Decisions

Keep a fixed main commit and rebase overlays from their declared bases. The CPU/Metal/CUDA audit covered 33 files: changed local deltas remain identical except for the Windows loader integration, which retains restricted dependency loading while adopting upstream versioned paths and exception handling. Metal QMV controls and gathered-row corrections, CUDA capture barriers, normalization precision, JIT paths and SDPA options remain intact.

### ROCm overlay

Keep fork `75915908` and retarget the core files. Preserve the fork's four-dimensional repeated-head GQA fallback while combining window/causal and explicit masks. Unsupported fused window attention falls back, or throws when forced. Regenerate the reviewed residual record and document the resolution in LOCAL_FIXES item 5.

## 7. Change Summary

- Update the single CMake pin and affected overlays; preserve unchanged local helpers.
- Pass `window_size=-1` at the five existing masked C++ SDPA call sites.
- Record the comparison and validation in [the pin review](../docs/mlxcelverse/mlx-pin-2026-10-26.md).

## Validation and Follow-up

CPU and CUDA sm_121 builds pass. Under gpu-lock on GB10, layer tests pass 104/104, SDPA 22/22 and grouped GEMM 3/3; filters overlap. Pin/parser tests, CPU SDPA and QMM tests, overlay records/drift and ROCm fast.cpp host syntax checks pass.

The CPU layer suite passes 103/104. The f16 dense-GEMM/QMM byte-equality failure reproduces with both cached old-pin and new-pin CPU libraries: 598,053 differing elements out of 1,048,576. The corresponding CUDA test passes. This pre-existing CPU issue is not fixed here.

Apple Silicon runtime/JIT parity, native Windows DLL loading and AMD HIP build/runtime checks remain pending. Hosted CI status is tracked on the PR separately from local evidence.
