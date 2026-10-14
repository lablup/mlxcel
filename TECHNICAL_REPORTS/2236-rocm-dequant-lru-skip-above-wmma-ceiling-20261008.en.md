# Technical Report: PR #2236 - Skip the Dequant LRU for GEMMs Routed Above the WMMA Ceiling

**Date**: 2026-10-08

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); branch commits `8964f610` (code) and `79a8c9bb` (docs), rebased onto origin/main `6dbfe7d7`, merged after the full `make verify-rocm` gate. Closes #2151. Follow-up: #2232.

**Languages**: HIP/C++ (ROCm overlay `qmm.hip`, `rocm.h`, `no_rocm.cpp`; mlxcel-core bridge `mlx_cxx_bridge.cpp`, `mlx_cxx_bridge.h`), Rust (mlxcel-core `lib.rs`, new `rocm_qmm_cache.rs`; new `tests/rocm_qmm_dequant_cache.rs`), Markdown (`LOCAL_FIXES.md`, `docs/environment-variables.md`, `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`)

**Risk Level**: Low. The ceiling-routed GEMM runs the same dequantize kernel and the same hipBLASLt GEMM as before; only the cache lookup and insert are skipped, and the logit traces are byte-identical. The f16 and non-WMMA callers of the cached route keep the cache unchanged. The new counters are relaxed atomics bumped once per GEMM on the dequantize route, and the cache refactor moves the existing statics into one struct without changing the eviction rule.

## Executive Summary

Since #2085, a bf16 affine GEMM of 128 rows or more on RDNA 3.5 leaves the fused WMMA kernel for dequantize plus hipBLASLt. That route stored every dequantized weight in an LRU of 8 matrices or 256 MB. A forward pass runs far more distinct projections than 8, so prefill cycled the cache without a single hit, and the last entries (up to 256 MB of bf16 weights, plus references to the quantized sources) stayed alive through decode.

PR #2236 adds `QmmRoute::DequantGemmAboveWmmaCeiling`, returned only from the ceiling branch of `select_qmm_route`. `QuantizedMatmul::eval_gpu` runs it like `DequantGemm` but dequantizes into a temporary that the command encoder frees after the GEMM. The cache now lives in one struct with hit, miss, insert, eviction and bypass counters, readable through `rocm::dequant_cache_stats()`, the bridge, and `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`.

On gfx1151 with Gemma 3 4B 4-bit (bf16) at pp2048/tg128, the cache made 0 hits in 340 lookups and held 6 entries (230 MiB) at exit before the change. After it, the GEMMs bypass the cache 340 times, the cache stays empty, and active memory after prefill drops from 2.80 GB to 2.56 GB. MLX peak memory stays at 4.52 GB and prefill and decode stay within run-to-run noise. Llama 3.1 8B 4-bit (f16 scales) is unchanged and also gets 0 hits in 320 lookups while holding 224 MiB; turning that default off is #2232.

## 1. Problem Statement

### 1.1 The ceiling route and its cache

`select_qmm_route` in `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/quantized/qmm.hip` decides how a quantized matmul runs. For a WMMA-eligible bf16 shape, it computes `hipblaslt_is_faster = env != 1 && dequant && M >= wmma_qmm_max_m(d) && !fp8()`, where `wmma_qmm_max_m` is 128 on RDNA 3.5. Before this PR that branch returned `QmmRoute::DequantGemm`, the same enumerator that f16 checkpoints and non-WMMA shapes reach through `if (dequant)`.

In `QuantizedMatmul::eval_gpu`, `DequantGemm` looked up a process-wide LRU keyed by the weight, scales and bias pointers (`DequantCacheKey`). On a miss it dequantized the weight and inserted it whenever `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` (default 8) and `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES` (default 256 MB) were both non-zero. Each entry also held references to the quantized source arrays.

### 1.2 Why the cache never hits

A prefill touches each projection once per forward pass, and a model has many more distinct projections than the cache holds (a Llama-class 8B has 32 layers of 7 projections). Every lookup is a miss, every insert evicts an older entry, and nothing is looked up again before it is evicted. The miss is structural: no key ordering or hashing change fixes it. Decode at batch 1 runs one row, takes qmv, and never reaches the cache, so the entries left at the end of prefill are never used.

`LOCAL_FIXES.md` item 29 and the #2085 technical report already recorded that the cache gets no hits in prefill and keeps up to 256 MB alive afterwards. The memory cost had not been measured.

### 1.3 The measured cost

With counters added (a counters-only build that routes and caches as `87538835` does), Gemma 3 4B 4-bit at pp2048/tg128 made 340 lookups: 0 hits, 340 misses, 340 inserts, 334 evictions, and 6 entries (241,172,480 bytes, 230 MiB) still cached at exit. Those 230 MiB stayed allocated from the end of prefill through decode.

## 2. Change Summary

| Area | Change |
|---|---|
| `patches-rocm/.../quantized/qmm.hip` | New `QmmRoute::DequantGemmAboveWmmaCeiling`, returned from the ceiling branch only. `eval_gpu` accepts it in the dequantize + GEMM arm; `use_cache` is true only for `DequantGemm`, so the new route dequantizes into a temporary and bumps `bypasses`. `quantized_matmul_runs_dequant_gemm` accepts both enumerators. The cache statics move into `struct DequantCache` with `evict_to()`; `DequantCacheCounters` holds relaxed atomic hit, miss, insert, eviction and bypass counters; `print_dequant_cache_stats()` is registered with `std::atexit` when `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1` |
| `patches-rocm/.../rocm.h`, `no_rocm.cpp` | `struct DequantCacheStats` and `MLX_API dequant_cache_stats()`; the stub returns zeros |
| `mlxcel-core/cpp/mlx_cxx_bridge.{cpp,h}` | `rocm_dequant_cache_stats()` returns the seven values as a `rust::Vec<uint64_t>`, seven zeros without `MLXCEL_BRIDGE_ROCM_BACKEND` |
| `mlxcel-core/src/lib.rs`, `rocm_qmm_cache.rs` (new) | FFI declaration and `pub mod rocm_qmm_cache` with `DequantCacheStats` and `dequant_cache_stats()`; unit test `stats_are_zero_off_rocm` |
| `tests/rocm_qmm_dequant_cache.rs` (new) | Four cases, one child process each, on ROCm |
| `LOCAL_FIXES.md` item 29, `docs/environment-variables.md`, `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md` | New behavior, the stats variable, and the dated measurement section |

Two commits: the code change (`8964f610`) and the measurements (`79a8c9bb`). 11 files, 812 insertions and 53 deletions.

## 3. Design

### 3.1 A separate enumerator, not a flag on the call

The ceiling branch is the only place that knows the GEMM came from a WMMA shape sent away by row count. A new enumerator carries that fact to `eval_gpu` without a second set of conditions there. Everything the two routes share stays shared: the same arm handles batch-shape checks, the dequantize kernel, the hipBLASLt GEMM and its rocBLAS fallback. Only one condition differs:

```cpp
const bool use_cache = route == QmmRoute::DequantGemm && cache_cap > 0 &&
    cache_max_bytes > 0;
```

With `use_cache` false, the weight takes the existing cache-off path: `w_dequant` gets a fresh allocation, and `enc.add_temporary(w_dequant)` keeps it alive until the GEMM completes. This is the path `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0` already used.

`quantized_matmul_runs_dequant_gemm`, which mlxcel's dense prefill path uses to know whether `quantized_matmul` and `dequantize` + `matmul` give the same bytes, is the other reader of the route. It now accepts both enumerators, so its answer for ceiling-routed shapes is unchanged.

### 3.2 Scope: only the ceiling route

The issue asked to leave the other `DequantGemm` callers alone and measure them first. f16 checkpoints and non-WMMA shapes still use the cache in this PR. The measurement below shows the f16 cache also never hits, but changing its default is a separate decision with its own case to check (#2156, f16 batch-4 decode on `DequantGemm`, where a decode loop reusing the same projections could hit), so it went to #2232.

Edge cases follow from the route logic: `MLX_ROCM_WMMA_QMM=1` sets `env == 1`, which never takes the ceiling branch, so the fused kernel runs and neither the cache nor the bypass counter moves. An `MLX_ROCM_WMMA_QMM_MAX_M` override changes the ceiling and still produces the new enumerator. The fp8 route is unaffected.

### 3.3 Counters and the stats hook

The issue's acceptance criteria needed a way to assert the cache is empty after a bf16 prefill. The cache's `static` locals inside `eval_gpu` could not be read from outside, so they moved into `struct DequantCache` (mutex, LRU list, byte count, entry map) behind `dequant_cache()`, and the two copies of the eviction loop became one `evict_to()` method. Eviction behavior is the same; `evict_to()` also bumps the eviction counter.

The counters are `std::atomic<uint64_t>` with relaxed ordering. They are bumped once per GEMM on the dequantize route and never read on the hot path. `read_dequant_cache_stats()` loads them and reads the entry count and bytes under the cache mutex. Three readers sit on top:

- `rocm::dequant_cache_stats()` in `rocm.h`, stubbed in `no_rocm.cpp`.
- `mlxcel_core::rocm_qmm_cache::dequant_cache_stats()` through the bridge, which returns zeros off ROCm.
- `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`, which registers an `std::atexit` handler that prints one `[ROCm] qmm dequant cache: ...` line on stderr. The handler is registered inside `dequant_cache()` after the static cache is constructed, so it runs before the cache is destroyed. The ceiling route calls `dequant_cache()` before counting the bypass, so the report is registered even when the cache itself is never used.

### 3.4 The test

`tests/rocm_qmm_dequant_cache.rs` runs each case in a fresh child process, because the cache, its counters and the environment knobs are process-wide and read once. The parent test runs the children one after another so they never share the GPU. Each child sets `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1` so the dequantize route is eligible at any row count, runs one quantized GEMM twice, compares each result with an f32 `dequantize` + `matmul` on the CPU stream (relative tolerance 2e-2), and checks the counters after each pass:

- `bf16_above_default_ceiling` (256 rows, 4096x4096): 1 then 2 bypasses, an empty cache, and output bytes equal to `dequantize` + `matmul` on the GPU.
- `bf16_above_env_ceiling` (64 rows, `MLX_ROCM_WMMA_QMM_MAX_M=32`): the same with an environment ceiling.
- `bf16_forced_wmma` (`MLX_ROCM_WMMA_QMM=1`): no counter moves.
- `f16_cached` (128 rows): a miss and an insert on the first pass, a hit on the second, one entry of `n * k * 2` bytes.

With the ceiling branch returning `DequantGemm` again, the two bf16 ceiling cases fail on their first pass (`misses=1 inserts=1 bypasses=0 entries=1`), and the forced-WMMA and f16 cases still pass: 2 of 4 cases fail.

## 4. Production Impact

Measured on gfx1151 (Radeon 8060S), `scripts/bench_decode.sh <model> --prompt-tokens 2048 --max-tokens 128`, before on main `87538835` and after on that commit plus the change, medians of 3 (range), each run under `scripts/rocm_gpu_guard.sh --idle-secs 60` with arms alternated:

| Model | Scales | Arm | Prefill tok/s | Decode tok/s | MLX peak | Active after prefill | Active at end |
|---|---|---|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | before | 2872 (2819-2910) | 62.75 (62.68-62.76) | 4.52 GB | 2.80 GB | 2.97 GB |
| gemma-3-4b-it-4bit | bf16 | after | 2868 (2839-2945) | 62.71 (62.55-62.73) | 4.52 GB | 2.56 GB | 2.73 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | before | 1136 (1106-1136) | 36.63 (32.36-36.84) | 6.92 GB | 4.75 GB | 4.75 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | after | 1132 (1048-1143) | 36.67 (36.58-36.92) | 6.92 GB | 4.75 GB | 4.75 GB |

Cache counters (warmup and measured prefill together; before from the counters-only build):

| Model | Arm | Hits | Misses | Inserts | Evictions | Bypassed | Entries at exit | Bytes at exit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | before | 0 | 340 | 340 | 334 | 0 | 6 | 241,172,480 |
| gemma-3-4b-it-4bit | after | 0 | 0 | 0 | 0 | 340 | 0 | 0 |
| Meta-Llama-3.1-8B-Instruct-4bit | before | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |
| Meta-Llama-3.1-8B-Instruct-4bit | after | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |

For bf16 checkpoints on RDNA 3.5, active memory after prefill and at the end of the run drops by 0.24 GB, the size of the cache's leftover entries. MLX peak memory does not move, because the peak comes from the per-call dequantized copies inside prefill, which the cache never removed (item 28 bounds those). Prefill and decode are within the run-to-run range; no speedup is claimed. Llama 3.1 8B is the f16 control: its GEMMs never take the ceiling route, so it reads the same in both arms.

One confirmation run per model on the change rebased onto main `ad844354` matched the after rows: Gemma 3 4B 2885 tok/s prefill, 62.14 tok/s decode, 4.52 GB peak, 2.56 GB active after prefill, 340 bypasses and an empty cache; Llama 3.1 8B 1119 tok/s prefill, 36.66 tok/s decode, 6.92 GB peak, 4.75 GB active, 320 misses, 0 hits and 2 entries.

Metal and CUDA are unaffected: the change is in the ROCm overlay, and the bridge function returns zeros off ROCm.

## 5. Documentation

- **`LOCAL_FIXES.md` item 29.** The sentence that said the LRU gets no hits in prefill but keeps its last entries alive now describes `QmmRoute::DequantGemmAboveWmmaCeiling`, the Gemma 3 4B numbers (0 hits in 340 lookups and 230 MiB cached before; 340 bypasses, an empty cache, 2.80 to 2.56 GB active after prefill, peak unchanged after), the counters, `dequant_cache_stats()`, `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`, and the new test.
- **`docs/environment-variables.md`.** The `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` text now says bf16 GEMMs sent above `MLX_ROCM_WMMA_QMM_MAX_M` never use the cache, and `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1` is documented.
- **`docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`.** A new section, "2026-10-07: the dequantized-weight cache above the WMMA ceiling (issue #2151)", with the method, both tables, the logit check and the confirmation run. It notes that the third Gemma before run was taken on 2026-10-08 after the first failed in the harness, and that Gemma's end-of-run active memory on the rebased head (2.56 GB against 2.73 GB on `87538835`) moved with main and was not isolated.

## 6. Verification

On gfx1151 (Radeon 8060S), ROCm 10.0.0 / HIP 7.15:

- **New test.** `cargo test --release --features rocm --test rocm_qmm_dequant_cache -- --test-threads=1`: pass. With the ceiling branch returning `DequantGemm` again, 2 of 4 cases fail (the two bf16 ceiling cases, on their first pass); the forced-WMMA and f16 cases still pass.
- **Route env test.** `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1`: pass.
- **Logits.** `examples/logit_trace` on gemma-3-4b-it-4bit (README.md as the corpus, 4 chunks of 256 tokens, so every projection runs at 256 rows and takes the ceiling route), before and after: byte-identical traces; `scripts/compare_logit_traces.py --decided 2.0` reports 0 of 1024 disagreements and an identical perplexity.
- **Targeted gates.** `make verify-rocm-overlay verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`, `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings` and `cargo clippy -p mlxcel --features rocm --test rocm_qmm_dequant_cache -- -D warnings`: pass.
- **Full gate.** `make verify-rocm` with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit` at head `79a8c9bb` on `6dbfe7d7`: `[verify-rocm] OK`, 162 cargo test suites with 12191 passed, 0 failed and 399 ignored (8906 passed and 165 ignored in the mlxcel-core lib suite), and the ROCm smoke generated 32 tokens on the GPU. An earlier run on the `ae343d84` base was also green (12190 passed, 0 failed); main then gained #2218, which touches the bridge files, so the branch was rebased and the gate re-run.

Not verified: Metal and CUDA, which are not available on this host. The bridge stub is covered by the unit test `stats_are_zero_off_rocm`, which has not been built or run on those backends.

## 7. Technical Decisions

- **Skip the cache only for the ceiling route.** That is the route the issue measured and the one where the cache's miss pattern was known. The f16 route also showed 0 hits, but #2156's decode-like case could hit, so its default change is tracked in #2232 rather than folded in here.
- **New enumerator over a global cache-size change.** Shrinking `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES` globally would keep the useless inserts and still pin memory; the issue rejected it. Changing the key was rejected too, since the miss is structural.
- **Reuse the cache-off path.** The temporary allocation and `add_temporary` already existed for `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0`, so the new route adds no new memory-management code.
- **Counters always on.** One relaxed atomic increment per GEMM is negligible next to a dequantize and a GEMM, and having them always present means a benchmark can read them with one environment variable and no rebuild.
- **One child process per test case.** The cache, the counters and the knobs are process-wide and read once, so each case needs a fresh process to start from zero.

## 8. Residual Risks and Follow-ups

- **#2232: the f16 cache.** Llama 3.1 8B (f16 scales) made 320 lookups with 0 hits, 320 inserts and 318 evictions, and kept 2 entries (224 MiB) alive after the run, identical in all three runs. #2232 proposes turning `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` off by default on ROCm, after checking #2156's decode-like case with the counters.
- **Ceiling measured on gfx1151 only.** The change follows wherever the ceiling route is taken, but the route and these numbers were measured on gfx1151.
- **End-of-run memory difference between bases.** Gemma's active memory at the end of the run was 2.56 GB on the rebased head against 2.73 GB on `87538835` plus the change. Main moved between the two bases and the difference was not isolated.
- **Off-ROCm stub not exercised.** `stats_are_zero_off_rocm` covers the stub but has not run on Metal or CUDA.

## 9. Learning Points

- **A cache needs a hit rate before it earns its memory.** The LRU was added for reuse that a forward pass at these row counts never gives. Counters made that visible in one run: 0 hits in 340 lookups.
- **Peak and resident memory are different numbers.** Skipping the cache lowered active memory after prefill by 0.24 GB and left the MLX peak unchanged, because the peak comes from per-call transients the cache never removed. Measuring only peak would have shown no effect.
- **Carry routing facts in the route.** Adding an enumerator at the point that knows why the GEMM was rerouted kept `eval_gpu` to a one-line difference and kept the other reader of the route, `quantized_matmul_runs_dequant_gemm`, correct with a one-line change.
- **Make the before state measurable on the old routing.** The before counts came from a counters-only build that routes like `87538835`, which separated the effect of the counters from the effect of the route change.
