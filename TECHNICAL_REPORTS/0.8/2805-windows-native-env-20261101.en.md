# PR #2805: Make environment set from Rust reach MLX's getenv on Windows

**Date**: 2026-11-01 (generated before merge)
**Status**: In review
**Languages**: Rust, C++, Python
**Risk Level**: Medium (settings that were inert on Windows start to apply, which changes serving behavior there; no change elsewhere)

## Executive Summary

On Windows every default mlxcel writes for MLX was inert, because `std::env::set_var` does not update the C runtime's `getenv`. A new `mlxcel_core::native_env` helper also calls `_wputenv_s`, all production writes go through it, and a CI guard refuses a direct one. The settings now apply, so Gemma 4, Laguna and long-lived servers behave differently on an RTX Spark N1X, and the owner measures that separately.

## 1. Problem Statement

`std::env::set_var` calls `SetEnvironmentVariableW`. The C runtime keeps its own narrow and wide tables, and `getenv` reads the narrow one. MLX's `env::get_var` is `atoi` over `std::getenv`, so a value written from Rust was seen by Rust and by nothing native. #2799 found this for `MLX_PTX_CACHE_DIR` and worked around that one variable in the overlay.

Measured with the fixed helper's tests and the `_wputenv_s` call removed: 5 of 8 tests fail, among them `the_cuda_cache_defaults_reach_mlx`, where `mlx::core::env::get_var("MLX_CUDA_GRAPH_CACHE_SIZE")` returned the default instead of 2000.

### Audit

| Variable | Writer | Native reader | Effective on Windows before |
|---|---|---|---|
| `MLX_USE_CUDA_GRAPHS` | `src/loading/mod.rs`, hoisted by `start_server` | `device.cpp` `use_cuda_graphs` | No: capture stayed on for Gemma 4 |
| `MLX_CUDA_GRAPH_CACHE_SIZE` | `hardware.rs` | `lru_cache.h` from `device.cpp` | No: 400, not 2000 |
| `MLX_CUDA_SDPA_CACHE_SIZE` | `hardware.rs` | `lru_cache.h` from `scaled_dot_product_attention.cpp` | No: 256, not 2000 |
| `MLX_MAX_OPS_PER_BUFFER` | `cuda_graph_budget.rs`; `hardware.rs` (Metal only) | `env::max_ops_per_buffer` from `get_graph_limits` | No: 20, not 100, for Laguna on 12.1 |
| `MLX_MAX_MB_PER_BUFFER` | `cuda_graph_budget.rs` | `env::max_mb_per_buffer` from `get_graph_limits` | No: 25, not 1000, for Laguna on 12.1 |
| `MLX_PTX_CACHE_DIR` | `ensure_persistent_ptx_cache` | `ptx_cache_dir_win.h`, `GetEnvironmentVariableW` | Yes, since #2799 |
| `MLXCEL_QMM_TILE_M` | `cuda_kernel_knobs.rs` (`mlxcel tune`, startup) | `qmm_sm80.cu`, per call | No: the tuner compared identical runs |
| `MLXCEL_QMV_MULTIROW_MAX_ROWS` | `cuda_kernel_knobs.rs` (startup) | `qmv.cu`, latched | No |

`MLXCEL_KV_BOUNDARY_V_LAYERS` and `MLXCEL_PREFILL_CHUNK` have Rust readers only, so they always worked. They go through the helper too, so that the guard has one rule.

`MLX_ENABLE_TF32` has test-only writers, the `#[cfg(test)]` ctors. `src/bin/mlx_server.rs:3019` and `:3043` are `ScopedEnv` inside `#[cfg(test)] mod tests`. Neither is production.

## 2. Technical Decisions

### 2.1 One C runtime, measured

The helper only works if Rust's `_wputenv_s` lands in the runtime MLX reads. `dumpbin /directives` shows `RuntimeLibrary=MD_DynamicRelease` and `/DEFAULTLIB:msvcrt` on `utils.cpp.obj`, the nvcc `sort.cu.obj`, `device.cpp.obj` and every member of `mlx_cxx_bridge.lib`. The Rust target has no `crt-static`, and the server imports `getenv` from `api-ms-win-crt-environment-l1-1-0.dll`. The tests read through `std::getenv` in the bridge and `env::get_var` inside `mlx.lib`, so a build that splits the runtimes fails them.

### 2.2 `_wputenv_s`, then `std::env`

`_wputenv_s` updates both tables and the OS block. An empty value removes the variable from all three, so `std::env` runs last to keep its contract, where an empty value stays set and empty. The C runtime has no empty value, so `getenv` returns NULL for it; the documented overlay rule already treats empty as unset.

### 2.3 Non-ASCII values

A scratch probe with ACP 949: Korean round-trips through `getenv`, and `é😀` comes back as `e??` with `_wputenv_s` returning 0. `std::env` and `GetEnvironmentVariableW` stay exact. The only audited value that can hold such characters is `MLX_PTX_CACHE_DIR`, and it keeps #2799's wide read for this reason. The other values are integers.

### 2.4 `_wputenv_s` ends the process on a long value

The same probe: a 32766-character value is accepted and a 32767-character one exits the process instead of returning an error. The helper checks the length first, prints a line, and leaves the value to `std::env`. No production value is near this, but a helper that can end the process on input should not.

### 2.5 The guard bans every production write

Whether native code reads a name is not visible at the write, and a reader added later should not need the writer to change. `check_native_env_writes.py` therefore allows no `std::env` write outside `native_env.rs`, and exempts `#[cfg(test)]` items and files declared by `#[cfg(test)] mod`. It goes by the declaration, never the file name. It runs in the existing `env var docs` job, which has no path filter.

### 2.6 Threading

The helper keeps the `set_var` contract, and every writer keeps its SAFETY comment. Checked per writer:

- `hardware.rs`, `cuda_graph_budget.rs`, `apply_env_once`, `ensure_persistent_ptx_cache`, `TurboKvCacheArgs`, `bench_engine`: on the main thread, before the runtime and any MLX op.
- `cuda_kernel_knobs.rs` `QmmTileOp::run`: not a startup write. It overwrites between candidates in `mlxcel tune`, and on Windows `_wputenv_s` frees the old string. `synchronize_default` drains the stream first and the tuner is offline, so no native reader holds that string. No race.
- `loading/mod.rs`: the hoist writes on the main thread, and the later load-site calls return at the `var_os` check. A load on a worker reaches the write only if the hoist saw nothing, for example an unreadable `config.json` at startup. That is a first write with other threads alive, and native `getenv` runs per call in `qmm_sm80.cu`. It is the same hazard as on Linux, it predates this PR, and MLX may have latched the switch by then. A first write frees no string.

## 3. Change Summary

- `native_env.rs`: `set_native_var`, `remove_native_var`, `native_getenv`, `mlx_env_int`, `log_effective_native_env`. Bridge gains `native_env_read` and `mlx_env_get_var_int`.
- Writers: `hardware.rs`, `cuda_graph_budget.rs`, `cuda_kernel_knobs.rs`, `ensure_persistent_ptx_cache`, `loading/mod.rs`, `turbo_args.rs`, `bench_engine.rs`.
- `RUST_LOG=mlxcel_core::native_env=debug` prints each audited variable as `getenv` returns it, from `main`, `start_server` and the Gemma 4 write.
- CI: guard and its tests in `env var docs`; `cpu-link` runs `native_env` on Linux.
- Docs: `docs/environment-variables.md`; the comment on `windows_env_value` in `ptx_cache_dir_win.h`.

## 4. Results

| Check | Where | Result |
|---|---|---|
| `native_env` tests, `_wputenv_s` call removed | N1X | 5 of 8 fail |
| `native_env` tests, this PR | N1X | 8 of 8 pass |
| `cuda_ptx_cache_trust_windows` (#2799) | N1X | 22 of 22 pass |
| `cuda_ptx_cache_win_test.sh` | N1X | pass; the symlink case skips, as in #2799 |
| Guard on this tree, and its unit tests | N1X | pass, 8 tests |
| `cargo clippy -p mlxcel-core --features cuda --lib --tests` | N1X | clean past two lints in `weights.rs`, below |
| `mlxcel generate` and `mlxcel-server`, config-only directories | N1X | table below |
| `mlxcel tune --op qmm-tile --op qmv-multirow`, temp cache | N1X | writes 2 tactics |
| `MLXCEL_AUTOTUNE=cache` with those tactics, both binaries | N1X | `MLXCEL_QMM_TILE_M=128`, `MLXCEL_QMV_MULTIROW_MAX_ROWS=8` through `getenv` |
| `MLX_USE_CUDA_GRAPHS=1`, `MLX_CUDA_GRAPH_CACHE_SIZE=400` set by the operator | N1X | kept |

Values `getenv` returns, in both binaries:

| Directory | `MLX_USE_CUDA_GRAPHS` | Graph cache | SDPA cache | Ops | MB | PTX dir |
|---|---|---|---|---|---|---|
| `llama` | unset | 2000 | 2000 | unset | unset | `~\.cache\mlxcel\cuda-ptx\7ca40033132d` |
| `gemma4_text` | 0 | 2000 | 2000 | unset | unset | same |
| `laguna`, 256 experts | unset | 2000 | 2000 | 100 | 1000 | same |

Laguna's values also confirm that this device reports compute capability 12.1, the only gate besides the family.

### What changes on this host

| Setting | Models | Before, after | GB10 datapoint (`cuda-graph-budget-gb10-2026-09-12.md`) |
|---|---|---|---|
| Graph capture off | Gemma 4 family | on, off | `gemma-4-26b-a4b-it-4bit`: decode -10.8%, prefill 164 to 473 tok/s |
| Graph budget | Laguna with experts | 20 ops and 25 "MB", 100 and 1000 | `laguna-xs-2.1-nvfp4`: decode +17.4%, peak +0.37 GB |
| Graph cache capacity | all | 400, 2000 | abort threshold 800 to 4000 lifetime misses; costs memory only as shapes accumulate |
| SDPA cache capacity | all | 256, 2000 | abort threshold 512 to 4000; memory of a full cache never measured |

Autotune stays inert without `MLXCEL_AUTOTUNE`, and this host had no tactic cache. The pre-fix settings can be reproduced on the new binary, because an operator value already reached MLX: `MLX_USE_CUDA_GRAPHS=1 MLX_CUDA_GRAPH_CACHE_SIZE=400 MLX_CUDA_SDPA_CACHE_SIZE=256 MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=25`.

## 5. Trade-offs and Residuals

- **Serving A/B not run here:** the owner runs it on the N1X, against the pre-fix server. The table above gives GB10 numbers, not N1X ones.
- **Reads are at `env::get_var`, not at the LRU:** no run grew a cache past 400 entries, and nothing here loaded weights. The reader in `lru_cache.h` is the same function.
- **`_wputenv_s` first-write safety under concurrent `getenv`** rests on the UCRT taking its environment lock, not on a stress test.
- **Windows CI does not run the `native_env` tests:** hosted Windows jobs are compile-only for these crates. The guard runs everywhere, and Linux runs the tests through `cpu-link`.
- **`MLX_ENABLE_TF32` test pin is inert on Windows:** the ctors use plain `set_var`, so MLX's TF32 default stays on in Windows test runs. Out of scope here; `tf32_pin_tests` was not run.
- **`src/lib/mlxcel-core/src/weights.rs:1051` and `:1061`:** an unused variable and unreachable code in a test, from its `#[cfg(not(unix))]` early return, fail `-D warnings` on the test target on Windows. They predate this PR and are untouched.

## 6. Learning Points

- The issue names `mlx_server.rs` writes that look like production and sit in a test module; checking the enclosing `#[cfg(test)]` settled it in one read.
- A startup log that reports a raised budget is not evidence that the reader saw it. On Windows that line was true of Rust's view only, which is why the new debug line reads through `getenv`.
- `_wputenv_s` turns a long value into process termination through the invalid parameter path, so its documented errno return does not cover that input.
