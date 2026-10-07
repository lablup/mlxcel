# Technical Report: PR #2193 - Range-Check ROCm GEMM Env Integers and Fix a Stale Graph Comment

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); head `e54ffc1e` on `bbd05099`, PR open, pending merge. Closes #2152.

**Languages**: C++/HIP (ROCm overlay under `src/lib/mlx-cpp/patches-rocm/`), Rust (`tests/rocm_qmm_env.rs`), Markdown (`LOCAL_FIXES.md`, overlay README, `docs/environment-variables.md`)

**Risk Level**: Low. Only invalid values behave differently: they now warn and take the default instead of wrapping or being dropped silently. Valid values and unset variables give the same result as before. Metal and CUDA builds never copy the ROCm overlay.

## Executive Summary

The ROCm overlay read its integer GEMM knobs with `strtol` and a bare `static_cast<int>`, or with `atoi`, so out-of-range input wrapped without a message. `MLX_ROCM_WMMA_QMM_MAX_M=4294967297` became a ceiling of 1 and moved every bf16 GEMM of two or more rows off the fused WMMA kernel. The PR adds one helper, `env_int_or_default` in the new header `mlx/backend/rocm/env_int.h`, which follows the rule #2073 set for `MLX_ROCM_GPU_WATCHDOG_SECS` and `MLX_ROCM_FFT_CACHE_SIZE`. Eleven knobs now go through it. Each read is cached in a static, so a bad value prints one stderr line per process. The review found that the solution-index readers were still duplicated across three files, so a bad index warned up to three times, and that no test checked the threshold's new caching. Both are fixed in `e54ffc1e`.

## 1. Problem Statement

### 1.1 Background

LOCAL_FIXES item 25 (#2073) set the parsing rule for ROCm env knobs. Set `errno` to 0, parse base 10 with `strtol`, and require the whole string to be the number. `ERANGE`, junk and out-of-range values print `[ROCm] ignoring invalid NAME="value" (expected ...); using ...` and take the default. An unset or empty variable takes the default silently. The GEMM knobs predated the rule.

### 1.2 Existing Issues

- **Silent wrap**: `parse_{positive,non_negative}_int_env` (qmm.hip, plus copies in matmul.cpp and gemms/rocblas_gemm.cpp) and the inline parses in `dequant_cache_capacity()` and `moe_segment_min_avg()` cast a `long` to `int` without a bound. `4294967297` became 1, `2147483648` became negative and was dropped, and `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=4294967304` became 8.
- **atoi reads**: `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B` and `MLX_GRIDX_MULT` used `atoi`. `MLX_ROCM_QMV_TILE_N` was also read on every qmv call and could exceed the tiled kernels' `__launch_bounds__(TILE_N_MAX * WARP_SIZE)` with `TILE_N_MAX = 32`.
- **Per-call read**: `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD` was read on every call to `should_use_dequant_gemm_path`, so a warning there would print once per GEMM.
- **Stale comment**: `use_hip_graphs()` in device.cpp said prefill uses the WMMA GEMM. That has been untrue since item 29 (`select_qmm_route`).

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A mistyped knob silently changes the GEMM route and prefill speed | Medium | Low |
| An out-of-bound tile width breaks the qmv launch | Medium | Low |

## 2. Technical Review

### 2.1 Review Criteria Applied

Each call site was checked for its range, its default and how its static is cached. The checks also covered the sentinel defaults: -1 for the batched solution indexes, the threshold and the ceiling, and 0 for the tile width (not range-checked, as the brief specifies). The `MLX_ROCM_QMV_TILE_N` bound was compared with the tiled kernel launch, the `MLX_GRIDX_MULT` behavior change and the 64-bit `grid_x` math were reviewed, and overlay integration was confirmed in `src/lib/mlx-cpp/CMakeLists.txt` and `build.rs`. LOCAL_FIXES item 34 and the docs were read against the code, and the test was run with the fix reverted to show it can fail.

### 2.2 Findings

| Finding | Severity | Status |
|---------|----------|--------|
| `gemm_solution_index_{f32,bf16}` existed in three translation units, each with its own statics, so a bad `MLX_ROCM_GEMM_*_SOLUTION_INDEX` warned up to three times per process | Medium | Fixed: one inline pair in `gemms/rocblas_gemm.h` |
| No test checked that the threshold is now read once | Medium | Fixed: invalid-threshold case added |
| The batched fallback message read "using MLX_ROCM_GEMM_F32_SOLUTION_INDEX" | Low | Fixed: "using the ... value" |
| LOCAL_FIXES item 34 did not mention that `MLX_ROCM_GROUPED_PREFILL_MIN_B=0` changes behavior | Low | Fixed |
| `MLX_ROCM_QMV_TILE_N=32` on a wave64 device asks for 2048 threads per block | Low | Open: predates the PR; gfx1151 is wave32 |
| The test does not observe that a cache size of 0 turns the cache off | Low | Open |

Confirmed correct: the new header is copied by the `GLOB_RECURSE` over `patches-rocm/mlx/*` in `mlx_apply_source_overlays`, and the build picks it up through `rerun-if-changed=../mlx-cpp/patches-rocm`. `grid_x` takes the minimum in `int64_t` before narrowing, so the result never exceeds `(B + 63) / 64`. The tile width's 1-to-32 range matches `TILE_N_MAX`. Item 34 is the next free number, and the README count of 109 matches the tree.

## 3. Technical Decisions

### 3.1 Consolidating the Solution-Index Readers

**Context:** Each of the three files kept its own copy of the reader in an anonymous namespace, so each held a separate static.

| Option | Pros | Cons |
|--------|------|------|
| Leave the copies and document "once per file" | No code movement | The warning still repeats; the issue asks for one line |
| **Chosen: inline functions in `gemms/rocblas_gemm.h`** | One static per process (inline function statics are shared across translation units); 44 lines fewer | `matmul.cpp` now includes `rocblas_gemm.h` and `rocblas_gemm.h` becomes a modified overlay file |

`rocblas_gemm.h` was already included by qmm.hip and rocblas_gemm.cpp, and matmul.cpp declares no names that clash with it.

## 4. Implementation Details

`env_int_or_default(name, default, min, max, expected, default_description = nullptr)` returns `default` for an unset or empty variable. Otherwise it parses with `strtol` base 10 and rejects `end == raw`, trailing characters, `ERANGE` and values outside `[min, max]`, printing one line. `default_description` names a sentinel default ("the per-device default", "the built-in crossover") instead of printing `-1`.

Ranges: the threshold, the ceiling, `MLX_ROCM_MOE_SEG_MIN`, `MLX_ROCM_GROUPED_PREFILL_MIN_B` and `MLX_GRIDX_MULT` take 1 to `INT_MAX`. The solution indexes and the cache size take 0 to `INT_MAX`, and 0 still turns the cache off. `MLX_ROCM_QMV_TILE_N` takes 1 to `TILE_N_MAX`.

Behavior changes for previously accepted values: `MLX_GRIDX_MULT=0` or a negative value used to clamp to 1 and now warns and uses 4. `MLX_ROCM_GROUPED_PREFILL_MIN_B=0` used to admit every batch size and now warns and uses 64.

## 5. Learning Points

### 5.1 Statics in Inline Functions

A function-local static in a non-inline function inside an anonymous namespace is a separate object in every translation unit that defines it. A static inside an `inline` function with external linkage is one object for the whole program, which is what makes "warn once per process" hold when several files read the same variable.

### 5.2 A Test That Can Fail

The child-process test spawns one process per value because the knobs are statics. Its value depends on failing without the fix. That was measured twice, once by reverting the whole overlay change and once by removing only the threshold's `static`.

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 10 (excluding these reports) |
| Tests added | 1 test file (15 child cases) |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `92de1604` | fix | range-check GEMM env integers and fix a stale graph comment |
| `e54ffc1e` | fix | read each GEMM solution index once per process |

## 8. Follow-up Actions

### Required
- [ ] #2180: document the new ranges for `MLX_ROCM_QMV_TILE_N` and `MLX_ROCM_MOE_SEG_MIN` in its ROCm table; the boolean and presence knobs stay out of scope there.

### Future Improvements
- Bound `MLX_ROCM_QMV_TILE_N` by the runtime wavefront size (1024 / warp) on wave64 devices.
- `MLX_GRAPH_REPLAY_SLOTS` in device.cpp turns `-1` into `SIZE_MAX` through `std::max<size_t>(2, atoi(e))`; it needs its own issue.

## Appendix

### A. Test Results

On gfx1151 at `e54ffc1e`: `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1` passed 15 of 15 cases. With qmm.hip, matmul.cpp, gemms/rocblas_gemm.{cpp,h} and device.cpp restored from main, 9 of 15 failed: `4294967297` reported the dense route, and none of the nine invalid values warned. With only the threshold's `static` removed, the threshold case failed with one warning per GEMM. `make verify-rocm-overlay` (109 backend files, 34 entries), `verify-versions`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-llama-compat`, `verify-fmt` and clippy on the test with `-D warnings` all passed. Metal and CUDA were not verified (not available on this host).
