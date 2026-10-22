# PR #2538: Cache custom CUDA kernels on disk to drop a per-process compile

**Date**: 2026-10-22
**Status**: Completed (PR open; closes #2492)

## Executive Summary

Issue #2492 reported granite-4.0-h-tiny-4bit decode on GB10 (sm_121) at about 57 tok/s against about 77 on v0.7.0, bisected to #2283, which routed granite's single-token MoE through the fused decode kernel. The issue asked for a per-device routing default. PR #2538 shows the kernel is not slower per token on GB10. MLX builds every `mx.fast.cuda_kernel` with `use_disk_cache = false`, so each process NVRTC-compiled the fused gate-up and down pair again inside its first decode step, about 1.2 s that `decode_tok_per_sec` counts. The fix is in the `jit_module.cpp` overlay: custom kernels are written to `MLX_PTX_CACHE_DIR` like built-in ones, cache files are published by rename, and a damaged entry is treated as a miss and rebuilt. `MLXCEL_FUSED_MOE` keeps its default on every device and no per-device policy is added. With the cache, granite decode at `-n 256` is 89.20 tok/s against 81.63 with `MLXCEL_FUSED_MOE=0` and 53.63 on main. 9 files, +629/-78.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp` | Custom kernels written to the disk cache; publish by temporary file and rename, `.txt` map before `.ptx`; read rejects empty, short or malformed entries; a cached entry that fails to load is removed, recompiled and republished once; an uncreatable cache dir leaves the cache cold instead of throwing |
| `src/models/granitemoehybrid.rs` | `forward` delegates to new `forward_routed(x, try_fused)`; the gate itself is unchanged |
| `src/models/granitemoehybrid_tests.rs` | The #2283/#2245 routing test split into a forced fused arm, a forced gather arm, and a test that `forward` follows `MLXCEL_FUSED_MOE`; none skips under `=0` |
| `tests/cuda_custom_kernel_disk_cache.rs` (new) | Three-process cache test (populate, warm hit, damaged cache) |
| `src/models/switch_layers.rs`, `src/lib/mlxcel-core/src/lib.rs` | Doc comments on the gate and on the cache key |
| `docs/environment-variables.md`, `docs/installation.md` | `granitemoehybrid` added to the `MLXCEL_FUSED_MOE` family list; cache default and custom-kernel coverage documented |
| `docs/benchmark_results/fused-moe-jit-cache-gb10-2026-10-22.md` (new) | Measurement record |

## 2. Technical Decisions

### 2.1 The regression was a fixed cost, not a per-token one

`decode_tok_per_sec` counts from the first sampled token to the last, so it excludes prefill but includes the first single-token forward, which is where a custom kernel's first launch and its NVRTC compile land. A slower kernel would widen the gap as `-n` grows. The gap shrank instead:

| model | `-n` | default | `MLXCEL_FUSED_MOE=0` |
|---|---|---|---|
| granite | 2 | 0.82 tok/s (2.44 s) | 1.66 tok/s (1.20 s) |
| granite | 32 | 11.72 (2.73 s) | 21.16 (1.51 s) |
| granite | 256 | 53.03 (4.83 s) | 65.84 (3.89 s) |
| lfm2 | 1024 (970 generated) | 117.28 | 128.87 |

A fit of decode time = fixed + tokens x per-token gives granite fused 9.4 ms/token with 2.4 s fixed against gather 10.6 ms/token with 1.2 s fixed, and lfm2 fused 6.5 ms/token against gather 7.0 ms/token. The fused kernel is faster per token and carried about 1.2 to 1.3 s of extra fixed cost. On the fixed binary, granite `-n 2` is 3.78 tok/s with the default and 3.84 with `=0`, so the extra intercept is gone.

### 2.2 No per-device routing policy

The issue's second acceptance criterion (express the decision per device) is marked not applicable. On main the loss was not granite- or sm_121-specific: lfm2 lost more (`=0` +45.5% against granite's +24.1%), qwen3-30b-a3b showed `=0` +2.3%, and `MLXCEL_FUSED_MOE_SGY` at 2, 4 and 16 stayed within -1.3% to +0.5% of the default while `=0` was +25.6% ahead. After the fix the fused default wins on all three families (paired, `=0` is 8.1% slower on granite, 6.2% on lfm2 and 17.2% on qwen3-30b), so turning the kernel off on sm_121 would have given that up. The gather arm also gains (granite 65.84 to 81.63) because granite's SSM step is a custom kernel and is now cached too.

### 2.3 Caching custom kernels is as safe as caching built-ins

Upstream `custom_kernel.cpp` passes `use_disk_cache = false` to `get_jit_module`. `JitModule` attempted the cache read either way and only skipped the write, so custom kernels could already load a planted cache file before this PR. The module name is `custom_kernel_<name><template-args hash>_<hash of the full source>`, which keys a custom kernel tighter than a built-in (keyed by name alone, and hence the pin-scoped directory from `ensure_persistent_ptx_cache`). The overlay ignores the parameter (`(void)use_disk_cache`) and keeps it for signature parity with upstream.

### 2.4 Publish protocol

Once fused decode kernels sit in the cache, a bad entry would break decode on every later run, so the write path changed as well as the gate:

- Each file is written to `<path>.tmp<pid>_<tick>`, closed, checked, then renamed. A concurrent reader (parallel test binary, server next to a CLI run) never sees a partial file. Within one process `get_jit_module` serializes writers, so pid plus a steady-clock tick is unique.
- The `.txt` kernel map is published before the `.ptx`, and the `.ptx` only if the map succeeded. `read_cached_ptx` keys the hit on the `.ptx`, so an existing `.ptx` implies a complete map. Two processes racing on the same entry publish byte-identical content, so mixing one's map with the other's code is harmless.
- `read_cached_ptx` treats an empty `.ptx`, a short read, a map line without a tab or with an empty field, and an empty map as a miss. Upstream accepted all of these and failed later in `cuModuleLoadDataEx`.
- On a hit, a `load_module` failure clears the kernels, unloads the module handle if `cuModuleGetFunction` had already loaded one, removes the `.ptx`, recompiles, republishes and loads once more. A second failure takes the original error path.
- `create_directories` takes an `error_code`; a cache that cannot be written leaves the process on the cold path.

### 2.5 Granite routing made testable on any device

`forward_routed(x, try_fused)` takes the routing decision as a parameter, so the #2245 test no longer skips when `MLXCEL_FUSED_MOE=0` is set in the environment. Each arm is pinned bit-exact against its candidate, the shared helper asserts the two candidates differ (so a bit-exact match is a routing check), and a third test checks that the production `forward` follows the gate under either setting.

### 2.6 Changes during review

- `83df4487`: the publish step checked each temporary before closing it (a failed final flush passed) and published the `.ptx` even when its map failed. It now closes before checking, gates the `.ptx` on the map, and puts the process id in the temporary name.
- `5b8c4d2c`: the read rejects empty or malformed entries, and a failed load after a hit recompiles once.
- `af9d047c`: the retry unloads the first module handle before recompiling.

## 3. Verification

From the PR's test plan and measurement record, on GB10 (sm_121, kernel 7.0.0-1019-nvidia, driver 580.178.04, `MLX_CUDA_ARCHITECTURES=121`, test-fast profile, interleaved arms with rotating order, a null arm per block):

| Check | Result |
|---|---|
| granite-4.0-h-tiny, `-n 256`, n=7 | fixed default 89.20, fixed `=0` 81.63, main 53.63 tok/s; null floor -1.2 to +0.7% |
| lfm2-8b-a1b, n=5 | 109.65 / 102.25 / 70.74; null floor -0.3 to +2.3% |
| qwen3-30b-a3b, n=5 | 56.20 / 46.56 / 45.05; null arm not run |
| `cuda_custom_kernel_disk_cache` | Red with the overlay reverted, green with it |
| `granitemoehybrid` lib tests | 17 passed, also with `MLXCEL_FUSED_MOE=0` |
| `switch_layers::`, `fused_moe` | 57 and 4 passed, CUDA, `--test-threads=1` |
| clippy `-D warnings`, `cargo fmt --check`, `check_env_var_docs.py` | Clean |

The cache test sets `MLX_PTX_CACHE_DIR` to a temporary directory, launches the fused MoE pair, and checks that both `moe_gateup_kernel_cu` and `moe_down_kernel_cu` left a non-empty `.ptx` with a tab-separated map and no `.tmp` leftover. It re-runs its own binary as a child on the same directory and requires no JIT notice on stderr and output bit-identical to the parent's. It then empties the gate-up `.ptx` and replaces the down kernel's map with a line that has no tab. A third process must recompile (notice printed), match the parent bit for bit, and leave both entries complete. The test is the only one in its binary because MLX reads `MLX_PTX_CACHE_DIR` once per process, at the first JIT compile.

## 4. Unverified and Open Items

- **Untested branch:** a non-empty, well-formed entry that `cuModuleLoadDataEx` rejects. No test plants one; the retry reuses the cold path's compile and publish.
- **Pre-existing, follow-up:** the cache key carries no compute capability or NVRTC version, so a cache directory shared across GPU generations on one MLX pin loads a foreign cubin (built-in and custom kernels alike).
- **Pre-existing, follow-up:** only `mlxcel` and `mlxcel-server` call `ensure_persistent_ptx_cache`. Bench binaries, test binaries and embedders use MLX's `$TMPDIR/mlx/<version>/ptx`, which is trusted without an ownership check; on a multi-user host whoever creates it first can plant modules under derivable names. `MLXCEL_CACHE_DIR` on a shared mount has the same exposure.
- **Minor:** a process killed between writing and renaming a temporary leaves a `.tmp<pid>_<tick>` file that nothing removes. `get_ptx_path` nests directories for names over 245 characters; none of mlxcel's custom kernels reaches that length.
- **Not measured:** ROCm and Metal (the overlay is CUDA-only and does not reach them), `mlxcel-server` steady state (a server pays the compile once per lifetime either way; only its first request after a restart changes), release-profile absolute numbers, a Windows CUDA build.
- **Older fused-MoE numbers:** the #626 and #712 sweeps that set `FUSED_MOE_MAX_DFF_CUDA = 8192` used `mlxcel-bench-decode`, whose in-process warmup most likely absorbed this compile, so the bound is probably unaffected; GB10 fused-MoE numbers taken with `mlxcel generate` before this PR include the fixed cost. Neither was re-measured; #2540 tracks a re-measurement on the current MLX pin.

## 5. Learning Points

- A one-time cost that lands inside the first decode step reads as a per-token rate in `decode_tok_per_sec`. Before attributing a decode A/B to a kernel, run at least two `-n` values and fit intercept and slope; a gap that shrinks with `-n` is a fixed cost.
- A bisect finds the commit that exposed a cost, not necessarily the code that owns it. #2283 only routed granite through a custom kernel; the cost was MLX's cache policy for every custom kernel.
- A disk cache read by later runs needs a publish protocol (temporary file, close, check, rename, data before the file that keys the hit) and a read that treats anything doubtful as a miss. Otherwise one bad write fails every later run.
- When a process-global setting is read once (here `MLX_PTX_CACHE_DIR`), test cross-process behavior by re-executing the test binary as a child with `--exact` and an environment marker, and keep the test alone in its binary.
