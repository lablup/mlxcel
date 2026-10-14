# PR #2307: Qualify OpenXLA/IREE Engine on ROCm/HIP

- **Date**: 2026-10-14
- **Status**: Implemented, reviewed, formatted, and verified
- **Risk**: Medium (build configuration, HIP HAL runtime linkage, and target validation)

## Summary

PR #2307 qualifies the OpenXLA/IREE execution backend (`--features xla-backend,xla-iree`) on AMD ROCm/HIP (`MLXCEL_XLA_DEVICE=hip`, `gfx1151` / AMD Radeon 8060S Graphics, ROCm 7.2.2) across all phases defined in issue #851. It delivers automated IREE toolchain setup (`scripts/iree/setup-rocm.sh`, `make iree-rocm`), runtime RPATH and HIP HAL driver linking, multi-line diagnostic reporting via `iree_status_fprint`, architecture/precision validation, fusion characterization under `#574`, and an end-to-end qualification report.

## Problem

Prior to this PR, mlxcel supported OpenXLA/IREE execution only on CPU (`local-task`, `local-sync`), Apple Silicon (`metal`), and NVIDIA CUDA (`cuda`). On AMD ROCm hosts, running `MLXCEL_BACKEND=xla` failed because:
- Prebuilt IREE binary distributions do not bundle the HIP HAL driver, requiring a pinned source build of `libiree_runtime_unified.a` with `IREE_HAL_DRIVER_HIP=ON`.
- IREE's HIP driver dynamically resolves `libamdhip64.so` at runtime via `dlopen`, which required either manual `LD_LIBRARY_PATH` exports or explicit ELF `DT_RPATH` tags.
- Diagnostics on bad target SKUs or missing device permissions previously printed truncated or opaque error strings rather than actionable remediation.
- No end-to-end qualification existed verifying numerical parity, continuous batching, HTTP serving, or memory stability on AMD hardware.

## Design and Implementation

- **Phase 0 & 1: Toolchain Provisioning (`scripts/iree/setup-rocm.sh`, `Makefile`)**: Clones and builds IREE `3.12.0rc20260721` (`dc9601f88654749456c7cee4ae87e13de2654e1e`) with `IREE_HAL_DRIVER_HIP=ON` and `IREE_BUILD_COMPILER=OFF` into `~/.cache/mlxcel/iree-rocm-3.12.0rc20260721`. Downloads and checksum-verifies the matching `iree-base-compiler` wheel for `iree-compile`. Automatically detects unambiguous `gfx*` architectures using `rocminfo` and `amdgpu-arch`.
- **Linking & RPATH (`build.rs`, `src/lib/mlxcel-xla/build.rs`, `src/lib/mlxcel-core/build_support/rocm_rpath.rs`)**: Consumes `IREE_ROCM_HOME`, links `libiree_hal_drivers_hip_registration_registration.a`, and passes `-Wl,--disable-new-dtags,-rpath,/opt/rocm/lib` to embed `DT_RPATH` so `dlopen("libamdhip64.so", ...)` resolves without `LD_LIBRARY_PATH`.
- **Diagnostics (`src/lib/mlxcel-xla/csrc/xla_iree.c`)**: Upgrades error handling from fixed 512-byte buffers to `iree_status_fprint(stderr, s)` and adds phase annotations (`[runtime-init]`, `[module-load]`, `[weight-upload]`) with concrete environment remediation.
- **Validation & Precision (`src/lib/mlxcel-xla/src/emitter/builder.rs`, `src/lib/mlxcel-xla/src/iree.rs`)**: Enforces single-SKU `gfx*` identifiers in `MLXCEL_XLA_ROCM_TARGET`, defaults `hip` precision to `f16`, trims whitespace on `MLXCEL_XLA_PRECISION`, and rejects invalid values.
- **Characterization & Qualification (`scripts/xla/fusion_spike.sh`, `docs/benchmark_results/xla_iree_rocm_qualification.md`)**: Analyzed packed int8 dequantization on `hip` (1,209 flow dispatches, lack of upstream fusion), confirmed data tiling failure matches CUDA `#574`, and documented complete benchmark and parity metrics.

## Review Findings and Fixes

1. **Formatting check (`cargo fmt --check`)**: Fixed line wrapping in `builder.rs`, `iree.rs`, and `phi4_audio.rs` so `cargo fmt --check` passes cleanly.
2. **Whitespace handling in precision override (`builder.rs`)**: Updated `precision_env_override` to trim whitespace so `MLXCEL_XLA_PRECISION=" f16 "` correctly matches `Precision::F16`, extracted `resolve_precision_checked_with_override` for pure testing without process environment mutation, and added unit test coverage.
3. **Qualification support verdict (`xla_iree_rocm_qualification.md`)**: Appended Section 6 ("Support Decision & Support Boundaries") explicitly stating the **Conditionally Supported** verdict and defining production-ready vs experimental boundaries.
4. **Target OS gate in ROCm rpath (`rocm_rpath.rs`)**: Added `if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") { return; }` to prevent macOS builds from emitting unsupported `-Wl,--disable-new-dtags` when `IREE_ROCM_HOME` is set.
5. **Non-empty validation for `MLXCEL_XLA_IREE_COMPILE` (`iree.rs`)**: Checked that `MLXCEL_XLA_IREE_COMPILE` is non-empty after trimming before returning `PathBuf`, avoiding opaque errors on empty environment overrides.
6. **Architecture normalization (`builder.rs`)**: Added `.to_ascii_lowercase()` in `validate_rocm_target` to accept uppercase target strings (e.g., `GFX1151`) and verified with unit test.
7. **Fast fail on download and temporary file cleanup (`setup-rocm.sh`, `validate_arch.sh`)**: Added `-f` flag to curl in `setup-rocm.sh` and registered an exit trap in `validate_arch.sh` to remove temporary oracle files.

## Verification

- `cargo fmt --check`: Passed with 0 differences.
- `cargo test -p mlxcel-xla --lib`: 282 passed, 0 failed, 11 ignored.
- `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings`: Passed with 0 warnings.
- `bash scripts/xla/validate_arch.sh --structural-only`: Passed.
- Hardware qualification recorded on `gfx1151` in `docs/benchmark_results/xla_iree_rocm_qualification.md` (64/64 token-exact parity, continuous batching 82.06 tok/s at B8, HTTP SSE streaming verified, 20-run soak with 0 leaked VRAM).
