# Technical Report: Issue #2108 - CPU-only Linux build fails to link kv_inplace_write

**Date**: 2026-10-05

**Status**: Implemented and validated on GB10 (CPU-only and CUDA builds); based on origin/main `33c45053`, pending merge.

**Languages**: C++ (bridge preprocessor guard), Rust (`build.rs`, build-policy predicate and unit tests), Markdown (installation guide)

**Risk Level**: Low (the GPU translation unit is unchanged apart from one predefined macro; the CPU-only build gains a throwing `eval_gpu` that no code path can reach)

## Executive Summary

`mlxcel-core/build.rs` compiles `src/lib/mlx-cpp/turbo/kv_inplace_write.cpp` (#1959) for every backend. Its `InplaceSliceWrite::eval_gpu` calls `mlx::core::copy_gpu_inplace`, which MLX defines only in `mlx/backend/gpu/copy.cpp`, a file built only with a GPU backend. A Linux build with no `cuda`, `metal` or `rocm` feature therefore failed at link time. The fix adds `MLXCEL_BRIDGE_GPU_BACKEND`, defined by `build.rs` exactly when MLX builds Metal, CUDA or ROCm, and puts the `copy.h` include and the `eval_gpu` body behind it. Without it `eval_gpu` throws, as `eval_cpu` already does.

## 1. Problem Statement

The bug was found while validating #2093, which needed `--unresolved-symbols=ignore-in-object-files` to link a CPU-only test binary. No CI job could catch it: the `ubuntu-latest` jobs only lint, every GB10 job passes `--features cuda`, and the remaining `cargo check` step does not link.

Reproduced on this branch by restoring main's `kv_inplace_write.cpp` and running `CARGO_TARGET_DIR=target/cpu cargo test -p mlxcel-core --profile test-fast --lib --no-run`: the link fails with two `undefined reference to mlx::core::copy_gpu_inplace(...)` errors from `InplaceSliceWrite::eval_gpu`, and no other unresolved symbol. `copy_gpu_inplace` is the only GPU-only MLX symbol the bridge references.

## 2. Change Summary

| Area | Change |
|---|---|
| `build_support/metal_backend.rs` | `gpu_backend_enabled(metal_backend, cuda, rocm)`, a pure predicate next to `resolve_metal_backend` |
| `build.rs` | Defines `MLXCEL_BRIDGE_GPU_BACKEND` from that predicate, using the resolved `metal_backend` value (not the `metal` Cargo feature) and the `cuda` and `rocm` features, which `build_mlx` maps one-to-one onto `MLX_BUILD_CUDA` and `MLX_BUILD_ROCM` |
| `kv_inplace_write.cpp` | `#include <mlx/backend/gpu/copy.h>` and the `eval_gpu` body under `#ifdef MLXCEL_BRIDGE_GPU_BACKEND`; otherwise `eval_gpu` throws `std::runtime_error` |
| `build_support/metal_backend_tests.rs` | Three tests: CPU-only is false, each backend alone is true, macOS with `MLXCEL_BUILD_METAL=OFF` and no other backend is false |
| `docs/installation.md` | "CPU-only link check" section with the local command and its measured cost |

## 3. Design Decisions

### 3.1 Why a new define instead of reusing the Metal and ROCm ones

`MLXCEL_BRIDGE_METAL_BACKEND` and `MLXCEL_BRIDGE_ROCM_BACKEND` gate backend-specific headers. `copy_gpu_inplace` is a backend-neutral GPU helper present in all three GPU builds, so its condition is the union, and CUDA had no bridge define at all. The predicate keys Metal on the resolved value because macOS enables Metal without the Cargo feature and `MLXCEL_BUILD_METAL=OFF` disables it with the feature (#1988).

### 3.2 Why keep compiling the file

The issue rejected dropping the `.file()` line: `cpp/mlx_cxx_ext.cpp` includes the header and the cxx FFI symbol `inplace_slice_write` must still link. At runtime every Rust caller (`cache.rs` decode write, rotating warmup write, and `cache/paged.rs` slab write) already requires `ffi::default_device_is_gpu()`, and `inplace_slice_write` itself refuses a non-GPU default device, so the throwing `eval_gpu` is unreachable in a CPU-only build.

### 3.3 GPU builds are byte-for-byte the same source

The include stays at its original position, wrapped in the guard, and the original body is inside the `#else` branch. With the define set, the preprocessed translation unit equals main's, which is the argument for Metal and ROCm, neither of which can be built on the development host.

## 4. Validation

| Check | Result |
|---|---|
| CPU-only `cargo test -p mlxcel-core --profile test-fast --lib --no-run` (separate `target/cpu`) | Links. Cold 2m24s including the CPU-only MLX tree; relink after touching the bridge file 6.0s |
| Same command with main's `kv_inplace_write.cpp` | Fails: `undefined reference to mlx::core::copy_gpu_inplace` (2 sites) |
| CPU-only `cargo build --release` | Links, 9m48s cold |
| CPU-only `metal_backend_build_policy` tests | 7 passed |
| CPU-only `inplace` tests | 4 passed (the CPU path keeps `slice_update`) |
| CPU-only `cargo clippy -p mlxcel-core --release --lib --tests -- -D warnings` | Clean |
| CUDA validation | See section 5 |

## 5. CUDA Regression

| Check | Result |
|---|---|
| `cargo build --release --features cuda --bin mlxcel-server` | Builds and links (17m41s cold) |
| CUDA `kv_inplace_write.o` | Still references `mlx::core::copy_gpu_inplace` and lacks the CPU-only error string, so the GPU body is compiled; the CPU-only object is the reverse |
| `cargo test --release -p mlxcel-core --features cuda --lib`, filter `inplace`, under `gpu-lock`, `--test-threads=1` | 4 passed |
| Same binary, filter `cache::` | 541 passed |
| Same binary, filter `metal_backend_build_policy` | 7 passed |
| `cargo clippy --release --features cuda --lib --tests -- -D warnings` | Clean |

## 6. Not Validated

Metal and ROCm could not be built on the GB10 host. Metal: `metal_backend` is true on a default macOS build, so the define is set and the source is main's. ROCm: `CARGO_FEATURE_ROCM` sets the define and `build_mlx` builds MLX with `MLX_BUILD_ROCM=ON`, whose GPU backend provides `copy_gpu_inplace`, so the source is again main's.

## 7. CI

No CI step was added here; `.github/workflows/ci.yml` is edited by #2111 in the same run. The proposed step is a GB10 job with its own persistent target dir (the pattern at `ci.yml:657-661`, for example `$HOME/.cargo-target/mlxcel-cpu-link-ci`) running `cargo test -p mlxcel-core --profile test-fast --lib --no-run`. Measured locally, a warm run costs 6.0s after a bridge C++ change and 6.6s after touching `mlxcel-core/src/lib.rs`, plus checkout; the first (cold) run costs about 2.5 minutes. That is cheap enough that the step is recommended for #2111. Until it lands, the command is documented in `docs/installation.md`.
