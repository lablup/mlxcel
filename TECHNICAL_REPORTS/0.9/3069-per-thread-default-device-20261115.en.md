# Technical Report: PR #3069 - Make MLX's default device per thread

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Medium

## Executive Summary

MLX keeps one process-global default device, but its default streams are per thread. A test that pinned the CPU with `DefaultDeviceGuard::cpu()` therefore moved every concurrent test's ops onto the CPU. In a parallel `cargo test -p mlxcel-core --lib`, 5 of 6 runs failed (#3066). A core overlay, `mlx/device.cpp`, applied on every backend and mirrored in the ROCm copy, now makes the default device per thread. `set_process_default_device` sets the default for threads that never set their own, and the runtime uses it for `--device cpu`.

## 3. Technical Decisions

- **Per-thread default, plus a process default.** A thread-local default alone would leave `--device cpu` worker threads on the GPU, so a separate process default covers threads spawned after startup.
- **CMake keeps the core overlay.** On non-CUDA builds, the CMake step that restores modified top-level `mlx/*.cpp` files had been silently reverting the overlay.
- **ROCm records.** `CORE_RESIDUAL.diff` was regenerated, and `LOCAL_FIXES.md` item 75 explains the change.
- **Memory-reading tests moved.** The #3063 state-tail retention guards read process-wide memory, so they moved to their own test binary.

## 7. Change Summary

| Item | Value |
|------|-------|
| Parallel mlxcel-core lib runs | 5 of 5 pass (was 1 of 6) |
| New tests | a CPU guard confined to its thread; a process default reaching later threads |

## 8. Follow-up Actions

The new thread timing exposed a pre-existing race in 22 lazy kernel holders, fixed in #3089.
