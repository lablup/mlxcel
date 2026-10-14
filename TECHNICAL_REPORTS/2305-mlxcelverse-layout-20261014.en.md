# Technical Report: Organize MLX-Side Layer under `src/lib/mlxcelverse`

- **PR**: #2305
- **Issue**: #1816
- **Date**: 2026-10-14
- **Category**: Refactoring / Build System & Architecture

---

## 1. Executive Summary

Prior to this change, `src/lib/mlx-cpp/` held an ad-hoc collection of four subdirectories (`patches/`, `patches-cuda/`, `patches-rocm/`, and `turbo/`) that had accrued incrementally as mlxcel added backend-specific upstream MLX overrides and custom fused GPU kernels. From the directory tree alone, contributors could not readily distinguish files that served as whole-file overlays copied over upstream MLX sources during CMake configuration from files compiled directly into mlxcel's own C++ bridge library. Furthermore, the CUDA backend overlay was split across two separate roots (`patches/mlx/backend/cuda/` and `patches-cuda/`).

PR #2305 reorganizes `src/lib/mlx-cpp/` into `src/lib/mlxcelverse/` with an explicit two-tier layout:
- `src/lib/mlxcelverse/overlays/{cpu,metal,cuda,rocm}/mlx/...` for upstream MLX whole-file overlays copied into `${CMAKE_BINARY_DIR}/_deps/mlx-src` during CMake configuration via `mlx_apply_source_overlays`.
- `src/lib/mlxcelverse/kernels/` for mlxcel's own fused C++/Metal/CUDA/HIP kernels, launchers, and port dispatch compiled into `mlx_cxx_bridge` by `src/lib/mlxcel-core/build.rs`.

All files under `src/lib/mlx-cpp/` were moved with `git mv` to preserve commit history, four new `README.md` documents were added (`src/lib/mlxcelverse/README.md` and `src/lib/mlxcelverse/overlays/{cpu,metal,cuda}/README.md`), and all build scripts, CI workflows, static analysis gates, ROCm overlay sync tooling, documentation, and test fixtures were updated consistently.

---

## 2. Structural Transformation

| Previous Path | New Path | Role |
|---|---|---|
| `src/lib/mlx-cpp/CMakeLists.txt` | `src/lib/mlxcelverse/CMakeLists.txt` | Top-level CMake project that fetches pinned MLX, applies source overlays, and builds `libmlx` |
| `src/lib/mlx-cpp/patches/mlx/backend/cpu/` | `src/lib/mlxcelverse/overlays/cpu/mlx/backend/cpu/` | CPU backend overlay (`quantized.cpp`, float accumulation for scalar quantized matmul, #2248) |
| `src/lib/mlx-cpp/patches/mlx/backend/metal/` | `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/` | Metal backend overlay (4 files: `compiled.cpp`, `device.cpp`, `kernels/utils.h`, `quantized.cpp`) |
| `src/lib/mlx-cpp/patches/mlx/backend/cuda/` + `src/lib/mlx-cpp/patches-cuda/*.cpp` | `src/lib/mlxcelverse/overlays/cuda/mlx/` | Unified CUDA backend overlay (`backend/cuda/**` 24 files, and MLX core files `mlx/{dtype,fast,ops}.cpp`) |
| `src/lib/mlx-cpp/patches-rocm/` | `src/lib/mlxcelverse/overlays/rocm/` | ROCm backend overlay vendored from `NripeshN/mlx@rocm-support` (`mlx/backend/rocm/**` 109 files, 15 core files, records `UPSTREAM`, `LOCAL_FIXES.md`, `CORE_RESIDUAL.diff`) |
| `src/lib/mlx-cpp/turbo/` | `src/lib/mlxcelverse/kernels/` | mlxcel-owned fused kernels, launchers, and backend port dispatch (31 files) |

### Overlay Dispatch in `CMakeLists.txt`

In `src/lib/mlxcelverse/CMakeLists.txt`, `mlx_apply_source_overlays(mlx_source_dir)` now sources each backend overlay from `overlays/<backend>/`:
- `overlays/metal/`: Metal JIT type promotion fix, command buffer budget controls, SIMD group matrix helpers, and `MLXCEL_QMV_WIDE` runtime off-switch.
- `overlays/cpu/`: CPU quantized matmul float accumulation, applied across all builds (Metal, CUDA, ROCm, CPU-only).
- `overlays/cuda/mlx/backend/cuda/`: Custom grouped GEMM, quantized kernels, reduce ops, and binary ops (unconditionally copied; harmless on non-CUDA builds).
- `overlays/cuda/mlx/*.cpp`: `dtype.cpp`, `fast.cpp`, and `ops.cpp` copied only when `MLX_BUILD_CUDA` is `ON` to preserve Metal precision for bf16 vision weights.
- `overlays/rocm/`: 109 ROCm backend files and 15 MLX core files copied only when `MLX_BUILD_ROCM` is `ON`, with stale overlay cleanup for cached source directories.

---

## 3. Build, Tooling, and Verification Updates

1. **Rust Build & FFI Bridge (`src/lib/mlxcel-core/`)**:
   - `build.rs`: CMake project invocation updated to `Config::new("../mlxcelverse")`, C++ kernel launcher sources updated to `../mlxcelverse/kernels/*.cpp`, include directories updated to `../mlxcelverse/kernels`, and `cargo:rerun-if-changed` watches updated to the new paths.
   - `build_support/mlx_pin.rs`: CMakeLists paths updated to `../mlxcelverse/CMakeLists.txt` and `src/lib/mlxcelverse/CMakeLists.txt`.
   - `Cargo.toml`: ROCm comment updated to point to `../mlxcelverse/overlays/rocm`.
   - `src/lib/mlxcel-core/cpp/*.{cpp,h}`: `#include` directives updated from `"../../mlx-cpp/turbo/..."` to `"../../mlxcelverse/kernels/..."`.
2. **CI Workflows & Static Gates**:
   - `.github/workflows/ci.yml` and `.github/workflows/release.yml`: updated path filters (`mlxcelverse/**`, `overlays/metal/**`, `overlays/cuda/**`, `overlays/cpu/**`, `kernels/**`), MLX pin verification steps, and hardware converter checks.
   - `Makefile`: updated `verify-rocm-overlay` documentation and prerequisites.
   - `scripts/ci/mlx_pinned_commit.sh`: default CMake path updated to `src/lib/mlxcelverse/CMakeLists.txt`.
   - `scripts/ci/{check_cuda_arch_lists.py,check_kernel_dtype_keys.py,check_kernel_port_dispatch.py}` and test scripts: updated launcher target paths and probe locations to `src/lib/mlxcelverse/kernels/`.
   - `scripts/mlxcelverse/{rocm_overlay.py,rocm_overlay_test.sh,sync_from_fork.sh}`: default overlay path updated to `src/lib/mlxcelverse/overlays/rocm`.
3. **Tests & Documentation**:
   - Updated path citations in `src/lib/mlxcel-mlx-pin/tests/cross_parser.rs`, `src/lib/mlxcel-core/src/cache/paged_pool_offset_tests.rs`, and `src/execution/memory_estimate_inflight_tests.rs`.
   - Added `src/lib/mlxcelverse/README.md` and `src/lib/mlxcelverse/overlays/{cpu,metal,cuda}/README.md`, and updated `CONTRIBUTING.md`, `NOTICE`, and `docs/*` while preserving historical benchmark logs, ADRs, and prior technical reports as written.

---

## 4. Verification Results

- **Overlay Byte Equivalence**: Verified that files copied into `_deps/mlx-src` by `mlx_apply_source_overlays` are identical in code before and after reorganization across Metal (4 files), CPU (1 file), CUDA (27 files), and ROCm (125 files).
- **Unit and Contract Gates**:
  - `bash scripts/ci/mlx_pinned_commit_test.sh` and `cargo test -p mlxcel-mlx-pin` (27/27 passed).
  - `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-python-tooling verify-fmt` (all checks passed, including 123 Python tooling tests and cargo fmt).
  - `bash scripts/mlxcelverse/rocm_overlay_test.sh` (29/29 passed) and `python3 scripts/mlxcelverse/rocm_overlay.py drift --no-fetch` (passed, 0 drift).
  - `python3 scripts/insert_apache_header.py --check` (passed, license headers intact).
  - `python3 scripts/ci/check_cross_repo_refs.py` (passed).
  - `python3 scripts/ci/check_cuda_arch_lists.py` (passed, 7 architecture lists verified).
- **ROCm Hardware Smoke Verification**:
  - Guarded `scripts/ci/rocm_smoke.sh` on `Qwen3-0.6B-4bit` (`gfx1151`, AMD Radeon 8060S): generated 32 tokens in 0.13s (`255.73 tok/s`).
