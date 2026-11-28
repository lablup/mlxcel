# Technical Report: PR #2298 - Bump MLX Pin and Declare Overlay Bases

- **Date**: 2026-10-14
- **Status**: Completed; implementation reviewed and all contract gates passed
- **Risk Level**: Low (all upstream changes 3-way merged, all 32 overlays verified at 0 drift, local ROCm overlay and unit test suites green)

## Executive Summary

PR #2298 updates the pinned upstream `ml-explore/mlx` commit from `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a` to `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` (14 upstream commits) in `src/lib/mlx-cpp/CMakeLists.txt`. It establishes a machine-readable `// Modified from upstream MLX <sha> <path>` base-declaration header across all 32 overlay files in `src/lib/mlx-cpp/patches/` and `src/lib/mlx-cpp/patches-cuda/`, resolving each base commit by diff against upstream history. All 9 conflicting or drifted overlays across that range were 3-way merged onto the new pin while preserving all mlxcel custom kernel and runtime deltas. Upstream fix `dfe17bafb` (#3912) was confirmed intact. The ROCm overlay was retargeted to the new pin with fallback stubs for the new upstream `fast::GatedDeltaUpdate` primitive, and `CONTRIBUTING.md` was extended with documentation for the CUDA/Metal/CPU overlay lifecycle and 3-way merge workflow.

## 1. Problem Statement

### 1.1 Background
The `mlxcel` repository applies whole-file source overlays onto upstream `ml-explore/mlx` via `configure_file(... COPYONLY)`. When moving the upstream `GIT_TAG` forward, any upstream modifications to an overlaid file are silently overwritten unless the overlay is explicitly rebased against upstream. Previously, only the ROCm overlay tracked its base commits and drift systematically, while CUDA, Metal, and CPU overlays lacked a machine-readable declaration of their copy or rebase commits.

### 1.2 Identified Deficiencies
- **Missing Base Declarations**: 29 of 32 overlay files lacked a machine-readable `Modified from upstream MLX` header, preventing automated drift checking.
- **Outdated Reducers**: Three CUDA reduction overlays (`all_reduce.cu`, `col_reduce.cu`, `row_reduce.cu`) were pinned to `v0.31.1` (`ce45c525`), missing upstream const-correctness and 64-bit planning improvements.
- **Upstream Pin Advancement**: Upstream MLX advanced with new features, including global-scale support for quantized operations (#4507), Gated Delta Net kernels (#4020), and non-multiple-of-32 quantized dimension fixes (#3912).
- **ROCm Divergence**: Upstream addition of `fast::GatedDeltaUpdate` required ROCm fallback stubs to prevent undefined symbol errors during static linking.

## 2. Implementation Details

### 2.1 Phase 1: Machine-Readable Base Declarations
All 32 overlay files under `src/lib/mlx-cpp/patches/` and `src/lib/mlx-cpp/patches-cuda/` now carry a header comment of the form:
```cpp
// Modified from upstream MLX <sha> <upstream/relative/path>
```
Each SHA was established by diffing against upstream `ml-explore/mlx` history. All 32 SHAs resolve in the upstream git repository.

### 2.2 Phase 2: 3-Way Rebase of Drifted Overlays
Nine overlays touched between their base and `59d600b5e` were 3-way merged onto `59d600b5e`:
1. `patches/mlx/backend/metal/quantized.cpp`: Integrated upstream global-scale loading while preserving the three `MLXCEL_QMV_WIDE` runtime off-switch hunks (#1186, #1187).
2. `patches/mlx/backend/cuda/device/qmm_naive.cuh`: Preserved Volta sm_70 `MmaPath` parameterization (#1543).
3. `patches/mlx/backend/cuda/quantized/qmm/qmv.cu`: Added `global_scale` parameter threading while preserving multirow qmv (#725), row-window caps (#906), and pre-Ampere accumulator optimizations (#1539).
4. `patches/mlx/backend/cuda/quantized/quantized.cpp`: Preserved 65535 grid chunking (#648), input contiguity checks, and sorted MoE prefill grouped-GEMM routes (#629).
5. `patches-cuda/fast.cpp`: Added `gated_delta_update` fallback implementation.
6. `patches-cuda/ops.cpp`: Updated `gather_qmm` GPU device validation and `GatherQQMM` argument list.
7. `patches/mlx/backend/cuda/reduce/{all_reduce,col_reduce,row_reduce}.cu`: Incorporated upstream `const_cast` removal and large-input 64-bit all-reduce planning.

The drift check confirms 0 upstream commits between declared bases and `59d600b5e` across all 32 overlays. Upstream `dfe17bafb` (#3912) is intact: neither `fp_quantized.h` nor `quantized.h` is overlaid, so both are consumed unmodified from the new pin.

### 2.3 Phase 3: Pin Bump and ROCm Overlay Retargeting
- `GIT_TAG` in `src/lib/mlx-cpp/CMakeLists.txt` was updated to `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`.
- `patches-rocm/UPSTREAM` `retargeted_to_mlx_pin` was updated to `59d600b5e`.
- `patches-rocm/mlx/backend/rocm/primitives.cpp` added `fast::GatedDeltaUpdate::use_fallback` returning `true` and `NO_GPU_MULTI(GatedDeltaUpdate)`, documented as item 6 in `LOCAL_FIXES.md`.
- `patches-rocm/CORE_RESIDUAL.diff` was refreshed against `59d600b5e`.

### 2.4 Phase 4: Documentation
`CONTRIBUTING.md` was extended with the `Reconciling CUDA, Metal, and CPU source overlays` subsection, detailing:
- The `// Modified from upstream MLX <sha> <path>` header requirement.
- The automated shell loop for detecting overlay drift.
- The 3-way merge command sequence using `git merge-file`.
- Guidelines for retiring overlays absorbed upstream.

## 3. Verification

The following verification steps were executed and confirmed passing:
1. `bash scripts/ci/mlx_pinned_commit.sh` prints `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`.
2. `cargo test -p mlxcel-mlx-pin` passes all 27 tests (26 unit tests + 1 cross-parser integration test).
3. Overlay drift check across all 32 files reports 0 `NO BASE` lines and 0 `DRIFT` lines.
4. `make verify-rocm-overlay` passes with 29 checks OK, validating 109 backend + 15 core files, 45 `LOCAL_FIXES` entries, and 4 noted residuals against pin `59d600b5`.
5. Contract gates passed: `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-python-tooling verify-fmt`.
6. `git diff --check origin/main..HEAD` confirms no whitespace errors or leftover conflict markers.
7. Greedy generation validation on AMD `gfx1151` produces byte-identical tokens before and after the bump.
