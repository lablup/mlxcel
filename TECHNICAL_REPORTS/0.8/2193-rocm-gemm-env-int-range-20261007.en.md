# Technical Report: PR #2193 - Range-Check ROCm GEMM Env Integers and Fix a Stale Graph Comment

**Date**: 2026-10-07

**Status**: Implemented on the gfx1151 host (ROCm/HIP) in code commits `92de1604` and `e54ffc1e` on origin/main `bbd05099`, plus this report. Merged after the full ROCm gate (`make verify-rocm`) passed. Closes #2152.

**Languages**: C++/HIP (ROCm overlay under `src/lib/mlx-cpp/patches-rocm/`: new `env_int.h`, `gemms/rocblas_gemm.{cpp,h}`, `matmul.cpp`, `quantized/qmm.hip`, `device.cpp`), Rust (new `tests/rocm_qmm_env.rs`), Markdown (`LOCAL_FIXES.md`, overlay README, `docs/environment-variables.md`)

**Risk Level**: Low. Only invalid values behave differently: they now warn and take the default instead of wrapping or being dropped silently. Valid values and unset variables give the same result as before. Metal and CUDA builds never copy the ROCm overlay.

## Executive Summary

The ROCm overlay read its integer GEMM knobs with `strtol` and a bare `static_cast<int>`, or with `atoi`, so out-of-range input wrapped without a message. `MLX_ROCM_WMMA_QMM_MAX_M=4294967297` became a ceiling of 1 and moved every bf16 GEMM of two or more rows off the fused WMMA kernel to dequantize + hipBLASLt.

PR #2193 adds one helper, `env_int_or_default` in the new header `mlx/backend/rocm/env_int.h`, which follows the rule #2073 set for `MLX_ROCM_GPU_WATCHDOG_SECS` and `MLX_ROCM_FFT_CACHE_SIZE`. Eleven knobs now go through it. Each read is cached in a function-local static, so a bad value prints one stderr line per process. The `use_hip_graphs()` comment in `device.cpp`, which said prefill uses the WMMA GEMM, no longer names a prefill GEMM.

The review of the first commit found that the solution-index readers were still duplicated across three files, so a bad index warned up to three times, and that no test checked the threshold's new caching. Both are fixed in `e54ffc1e`. The new child-process test passes all 15 cases on gfx1151, and 9 of them fail with the overlay sources restored from main.

## 1. Problem Statement

### 1.1 Background

LOCAL_FIXES item 25 (#2073) set the parsing rule for ROCm env knobs. Set `errno` to 0, parse base 10 with `strtol`, and require the whole string to be the number. `ERANGE`, junk and out-of-range values print `[ROCm] ignoring invalid NAME="value" (expected ...); using ...` and take the default. An unset or empty variable takes the default silently. The GEMM knobs predated the rule.

### 1.2 Silent wraps

`parse_{positive,non_negative}_int_env` (qmm.hip, plus copies of the non-negative one in matmul.cpp and gemms/rocblas_gemm.cpp) and the inline parses in `dequant_cache_capacity()` and `moe_segment_min_avg()` cast a `long` to `int` without a bound and printed nothing. `4294967297` became 1, `2147483648` became negative and was dropped, and `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=4294967304` became 8. Junk, zero and negative values were dropped with no message.

### 1.3 atoi reads and per-call reads

- `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B` and `MLX_GRIDX_MULT` used `atoi`.
- `MLX_ROCM_QMV_TILE_N` was also read on every qmv call and could exceed the tiled kernels' `__launch_bounds__(TILE_N_MAX * WARP_SIZE)` with `TILE_N_MAX = 32`.
- `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD` was read on every call to `should_use_dequant_gemm_path`, so a warning there would print once per GEMM.

### 1.4 The stale comment

`use_hip_graphs()` in device.cpp said prefill uses the WMMA GEMM. That has been untrue since item 29 (`select_qmm_route`).

### 1.5 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A mistyped knob silently changes the GEMM route and prefill speed | Medium | Low |
| An out-of-bound tile width breaks the qmv launch | Medium | Low |

## 2. Change Summary

| Area | Change |
|---|---|
| `mlx/backend/rocm/env_int.h` (new) | `env_int_or_default(name, default, min, max, expected, default_description = nullptr)`, the #2073 rule in one inline function |
| `gemms/rocblas_gemm.h` | Includes `env_int.h`; gains inline `gemm_solution_index_{f32,bf16}`, the one shared reader of the four solution-index variables |
| `gemms/rocblas_gemm.cpp` | Removes its `parse_non_negative_int_env` and its copies of the solution-index readers |
| `matmul.cpp` | Removes the same copies and calls `rocm::gemm_solution_index_*`; `moe_segment_min_avg()` (`MLX_ROCM_MOE_SEG_MIN`) uses the helper |
| `quantized/qmm.hip` | Removes `parse_positive_int_env`, `parse_non_negative_int_env` and `qmm_gemm_solution_index_*`. The threshold, the ceiling, the cache size, `MLX_ROCM_QMV_TILE_N`, `MLX_ROCM_GROUPED_PREFILL_MIN_B` and `MLX_GRIDX_MULT` use the helper from statics; `grid_x` is computed in 64 bits |
| `device.cpp` | The `use_hip_graphs()` comment says prefill does not go through this path and that `select_qmm_route` in `quantized/qmm.hip` chooses its GEMM route |
| `tests/rocm_qmm_env.rs` (new) | Child-process test, 15 cases |
| `LOCAL_FIXES.md`, `README.md`, `docs/environment-variables.md` | Item 34, backend file count 109, ranges and the warning |

Two code commits: the fix (`92de1604`, "range-check GEMM env integers and fix a stale graph comment") and the review follow-ups (`e54ffc1e`, "read each GEMM solution index once per process"). 10 files, 702 insertions and 173 deletions, excluding these reports.

## 3. Design

### 3.1 One helper with the #2073 rule

`env_int_or_default(name, default, min, max, expected, default_description = nullptr)` returns `default` for an unset or empty variable. Otherwise it parses with `strtol` base 10 and rejects `end == raw`, trailing characters, `ERANGE` and values outside `[min, max]`, printing one line. `default_description` names a sentinel default ("the per-device default", "the built-in crossover") instead of printing `-1`. The default itself is not range-checked, and `max` must not exceed `INT_MAX`, so the final cast to `int` cannot wrap.

### 3.2 Ranges and sentinel defaults

- The threshold, the ceiling, `MLX_ROCM_MOE_SEG_MIN`, `MLX_ROCM_GROUPED_PREFILL_MIN_B` and `MLX_GRIDX_MULT` take 1 to `INT_MAX`.
- The solution indexes and the cache size take 0 to `INT_MAX`, and 0 still turns the cache off.
- `MLX_ROCM_QMV_TILE_N` takes 1 to `TILE_N_MAX`, which matches the tiled kernel launch bound.
- The sentinel defaults are -1 for the batched solution indexes, the threshold and the ceiling, and 0 for the tile width. They lie outside the accepted ranges on purpose and are not range-checked, as the brief specifies.

### 3.3 One read per process

The helper warns on every call that sees a bad value, so every caller keeps the result in a function-local static. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD` and `MLX_ROCM_QMV_TILE_N` are now read once instead of per call, which is what makes a bad value print one line and not one per GEMM or qmv launch.

### 3.4 One solution-index reader per process

Each of the three files (matmul.cpp, gemms/rocblas_gemm.cpp, quantized/qmm.hip) kept its own copy of the solution-index reader in an anonymous namespace, so each held a separate static and a bad `MLX_ROCM_GEMM_*_SOLUTION_INDEX` warned up to three times per process. `e54ffc1e` replaces the copies with one pair of inline functions, `gemm_solution_index_{f32,bf16}` in `gemms/rocblas_gemm.h`. A static inside an inline function with external linkage is one object for the whole program, so each variable is read and warned about once. The batched fallback message now reads "using the MLX_ROCM_GEMM_*_SOLUTION_INDEX value" instead of "using MLX_ROCM_GEMM_F32_SOLUTION_INDEX".

### 3.5 64-bit grid size

`MLX_GRIDX_MULT` scales `grid_x` in the expert-batched GatherQMM launch. With the range now up to `INT_MAX`, `avg_tiles * gridx_mult` could overflow `int`. `grid_x` now takes the minimum in `int64_t` before narrowing, so the result never exceeds `(B + 63) / 64`.

### 3.6 The child-process test

The knobs are statics, so `tests/rocm_qmm_env.rs` spawns one child process per value, one at a time, with every tested variable cleared before each case is applied. The child asks `quantized_matmul_matches_dense_gemm` which route a bf16 `[64, 4096]` x 4-bit `[4096, 4096]` g64 GEMM takes (it shares `select_qmm_route` with `QuantizedMatmul::eval_gpu`), then runs a 128-row GEMM twice (dequantize path, which reads the cache size) and a one-row GEMV (qmv path, which reads the tile width) on the GPU and compares each with an f32 reference on the CPU stream. The parent checks the exit status, the reported route and the exact warning lines. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1` is set for the ceiling and cache cases so only the ceiling decides the route.

The 15 cases are: ceiling unset, `128`, empty, `32`, `4294967297`, `2147483648`, `-1`, `0`, `12abc`; threshold `0`; cache `4294967304`, `0`; tile width `16`, `abc`, `33`. The threshold case, added in `e54ffc1e`, also checks that the cached threshold warns exactly once, since every probe and GEMM in the child reads it.

## 4. Production Impact

Valid values and unset variables give the same result as before. An invalid value now prints one stderr line per process and takes the default, instead of wrapping or being dropped silently.

Two previously accepted values change behavior:

- `MLX_GRIDX_MULT=0` or a negative value used to clamp to 1 and now warns and uses 4.
- `MLX_ROCM_GROUPED_PREFILL_MIN_B=0` used to admit every batch size and now warns and uses 64.

There is no new per-call cost: every read is cached, and the threshold and the tile width are read less often than before. The `device.cpp` change is a comment. Metal and CUDA builds never copy `patches-rocm/`, and the test is `#![cfg(feature = "rocm")]`.

## 5. Documentation

- `docs/environment-variables.md` states that a `MLX_ROCM_WMMA_QMM_MAX_M` value that is not an integer from 1 to 2147483647 is ignored with a stderr warning, and adds a paragraph on the other GEMM integer knobs: the warning format, the silent default for unset or empty, and each knob's range and default.
- LOCAL_FIXES item 34 records the change, including the consolidation of the solution-index readers and that `MLX_ROCM_GROUPED_PREFILL_MIN_B=0` now warns instead of admitting every batch size. It is not an upstreaming candidate.
- The overlay README's backend file count goes from 108 to 109 for the new header.

## 6. Verification

On gfx1151 (Radeon 8060S), rebased on `bbd05099`:

- **New test.** `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1` at `e54ffc1e`: 15 of 15 cases pass. At `92de1604` the test had 14 cases, all passing.
- **Fails without the fix.** With qmm.hip, matmul.cpp, gemms/rocblas_gemm.{cpp,h} and device.cpp restored from main and rebuilt, 9 of 15 cases fail: `4294967297` reports the dense route, and none of the nine invalid values warns. With only the threshold's `static` removed, the threshold case fails with one warning per GEMM.
- **Other gates.** `make verify-rocm-overlay` (109 backend files, 34 entries), `verify-versions`, `verify-kernel-dtype-keys`, `verify-kernel-port-dispatch`, `verify-llama-compat`, `verify-fmt`, `cargo test --test dead_doc_pointers`, and `cargo clippy --release --features rocm --test rocm_qmm_env -- -D warnings` pass.
- **Review checks.** Each call site was checked for its range, its default and how its static is cached, including the sentinel defaults. The new header is copied by the `GLOB_RECURSE` over `patches-rocm/mlx/*` in `mlx_apply_source_overlays` (`src/lib/mlx-cpp/CMakeLists.txt`), and `build.rs` picks it up through `rerun-if-changed=../mlx-cpp/patches-rocm`. The tile width's 1-to-32 range matches `TILE_N_MAX`. Item 34 is the next free number, and the README count of 109 matches the tree. LOCAL_FIXES item 34 and the docs were read against the code.
- **Full ROCm gate.** `make verify-rocm` with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit` at head `01a28a10` on `bbd05099`: `[verify-rocm] OK`, 153 cargo test suites with 11997 passed, 0 failed and 384 ignored (8853 passed and 156 ignored in the mlxcel-core lib suite), and the ROCm smoke generated 32 tokens on the GPU.

Not verified: Metal and CUDA, which are not available on this host. The change touches only `patches-rocm/`, which those builds never copy, plus a ROCm-only test.

## 7. Technical Decisions

The main choice was where the solution-index readers live. Each of the three files kept its own copy in an anonymous namespace, so each held a separate static.

| Option | Pros | Cons |
|--------|------|------|
| Leave the copies and document "once per file" | No code movement | The warning still repeats; the issue asks for one line |
| **Chosen: inline functions in `gemms/rocblas_gemm.h`** | One static per process (inline function statics are shared across translation units); 44 lines fewer | `matmul.cpp` now includes `rocblas_gemm.h` and `rocblas_gemm.h` becomes a modified overlay file |

`rocblas_gemm.h` was already included by qmm.hip and rocblas_gemm.cpp, and matmul.cpp declares no names that clash with it.

- **Reuse the #2073 rule, not a new one.** The GEMM knobs now warn in the same format as `MLX_ROCM_GPU_WATCHDOG_SECS` and `MLX_ROCM_FFT_CACHE_SIZE`.
- **Name sentinel defaults in the message.** `default_description` prints "the per-device default" or "the built-in crossover" instead of `-1`.
- **Bound the tile width by `TILE_N_MAX`.** A larger value would exceed the tiled kernels' launch bounds.
- **Treat 0 like any other out-of-range value.** `MLX_GRIDX_MULT=0` and `MLX_ROCM_GROUPED_PREFILL_MIN_B=0` now warn and take the default, under the same rule as every other knob, instead of keeping a special clamp or an admit-all value.
- **Test in child processes.** The statics are read once per process, so only a fresh process per value can test both the parse and the once-per-process warning.

## 8. Residual Risks and Follow-ups

- **`MLX_ROCM_QMV_TILE_N=32` on a wave64 device** asks for 2048 threads per block. This predates the PR, and gfx1151 is wave32. A future change could bound the tile width by the runtime wavefront size (1024 / warp) on wave64 devices.
- **Cache size 0 is not observed.** The test checks that `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0` does not warn, but does not observe that it turns the cache off.
- **#2180 documentation.** #2180's ROCm table should describe the new ranges for `MLX_ROCM_QMV_TILE_N` and `MLX_ROCM_MOE_SEG_MIN`. The boolean and presence knobs stay out of scope there, and this PR leaves their behavior unchanged.
- **`MLX_GRAPH_REPLAY_SLOTS`** in device.cpp turns `-1` into `SIZE_MAX` through `std::max<size_t>(2, atoi(e))`. It is out of scope per #2180 and needs its own issue.

## 9. Learning Points

- **Statics in inline functions.** A function-local static in a non-inline function inside an anonymous namespace is a separate object in every translation unit that defines it. A static inside an `inline` function with external linkage is one object for the whole program, which is what makes "warn once per process" hold when several files read the same variable.
- **A test that can fail.** The child-process test's value depends on failing without the fix. That was measured twice, once by reverting the whole overlay change and once by removing only the threshold's `static`.
