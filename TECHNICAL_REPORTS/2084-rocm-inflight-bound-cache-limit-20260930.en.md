# Technical Report: PR #2084 - Bound in-flight batch memory and enforce the cache limit on ROCm

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host, rebased onto `3c9edea0`; pending merge.

**Languages**: C++ (ROCm overlay: allocator, command encoder, eval), Rust (runtime default, bench binary, tests), Bash (bench harness), Markdown

**Risk Level**: Medium (the eager ROCm path now blocks the host on GPU events during encoding, which touches every ROCm evaluation; Metal and CUDA builds never copy `patches-rocm/`, and the only shared Rust change keeps its old default off `rocm`)

## Executive Summary

Issue #2062 (split out of #1814, epic #1801) asked where the ROCm backend's memory goes on a UMA host and for a bounded default. At pp512/tg128 on gfx1151, `Meta-Llama-3.1-8B-Instruct-4bit` peaked at 20.60 GB of live MLX buffers for 4.75 GB of weights, and `Qwen3-30B-A3B-4bit` at 23.56 GB for 17.17 GB. The issue assumed buffer cache. The measurement showed otherwise: the cache was 0.4 to 1.7 GB at every phase boundary, and the peak was live transients inside the measured prefill. The ROCm eager path committed a command batch only every 2000 operations and never registered those commits with MLX's scheduler, so the host encoded far ahead of the GPU and every batch's temporaries were alive at once. Separately, the allocator's `set_cache_limit` stored its argument and nothing read it.

The PR adds two bounds, one per mechanism:

- **In-flight bound** (overlay, `LOCAL_FIXES.md` item 28): `gpu::eval` counts what each primitive allocates, the eager path commits a batch at a quarter of `MLX_ROCM_MAX_INFLIGHT_MB` (default 1024 MiB), and the host waits on the oldest committed batch while more than the budget is in flight. `0` restores the old behavior.
- **Cache bound**: `malloc_async` trims the cache on a miss once it is over the limit, down to three quarters of it, and mlxcel defaults `MLXCEL_CACHE_LIMIT` to 2 GiB on `rocm` builds. `memory_limit()`, which the pre-load estimate reads, is unchanged.

Through `scripts/bench_decode.sh` (three runs each, medians, GPU idle), the peaks fall from 20.60 to 6.14 GB and from 23.56 to 18.58 GB. Decode moves +1.5% and -0.2%, the 8B's prefill -1.8%. The investigation also found four allocator features that hold nothing in practice (decode arena, graph deferral, the async pool, the GTT/managed fallbacks), and a heap corruption at process exit in the first version of the bound, which the shipped version avoids by using raw HIP events.

## 1. Problem Statement

The ROCm spike under #1814 measured a peak of about 20 GB for an 8B model with about 4.5 GB of weights. On a UMA host (Radeon 8060S, 96 GiB VRAM carve-out, 31 GiB visible to the host) that memory is taken from the same pool the OS and every other GPU tenant use. #1814's split (2026-09-30) made this item #2062, with no dependency on the other eight sub-issues.

The issue's plan was to explain the peak first, compare every allocator knob, and then ship a default cache limit through the existing `resolve_cache_limit()` path. It also said what to do if the first step disproved the premise: "If step 1 shows the excess is not buffer cache (for example the arena or pool slack), fix it in the overlay instead and add a `LOCAL_FIXES.md` item, rather than capping a counter that does not hold the memory." Step 1 did disprove it, and the PR follows that branch while still shipping the cache default the acceptance criteria ask for.

## 2. The Diagnosis

### 2.1 The peak was live buffers, not cache

`mlxcel-bench-decode` now prints `active`, `cache` and a per-phase peak after load, after the warmup pass and after the measured pass. On origin/main:

| Model | Weights (`active` after warmup) | Cache after warmup | Warmup-pass peak | Measured-pass peak | Cache after measured pass | VRAM delta |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 4.75 GB | 1.64 GB | 6.19 GB | 20.60 GB | 0.42 GB | 21.19 GB |
| Qwen3-30B-A3B-4bit | 17.17 GB | 0.62 GB | 17.56 GB | 23.56 GB | 0.50 GB | 24.09 GB |

MLX's peak counts `active` only, never the cache, and the device-wide `mem_info_vram_used` delta sits within 0.6 GB of it, so the memory was held by live arrays. A 20-token and a 128-token measured pass both reach 20.60 GB, so it is set in the measured pass's prefill, not in decode. The warmup pass peaks low because its prefill overlaps the lazy weight load, and cross-stream waits split it into many small batches.

### 2.2 Why the transients were all alive at once

`CommandEncoder::add_temporary` keeps every input and scratch buffer of an operation alive until the completion handler of its batch runs. The eager path (`use_hip_graphs()` returns `false`) committed a batch every `MLX_MAX_OPS_PER_BUFFER` operations (2000), and it did not register those commits with MLX's scheduler. Two consequences follow:

- The scheduler's cap of 10 outstanding tasks never applied, so the host never waited for the GPU.
- `set_memory_limit` acts through that same task accounting, so `MLXCEL_MEMORY_LIMIT` never made the host wait either.

The host encodes a prefill much faster than the GPU runs it, so the temporaries of many batches were live together. `MLX_MAX_OPS_PER_BUFFER=50` did not help: more commits without a wait leave the host just as far ahead.

### 2.3 What the transients were

For the 8B, an f16 checkpoint, most of the excess (15.85 GB over the weights) is the f16 copy of each weight matrix that `QuantizedMatmul`'s dequantize-and-GEMM path allocates for a 512-row activation, 117 MB per MLP matrix. With that path off (`MLX_ROCM_QMM_DEQUANT_GEMM=0`) the peak was 6.78 GB, at a twentieth of the prefill speed (51.75 against 1069.34 tok/s). The MoE model is bf16, and bf16 affine 4-bit matmuls take the fork's fused WMMA kernel, which allocates no weight copy, so the same variable left its peak at 23.56 GB. Its excess is other operations' outputs held the same way, and the in-flight bound removes most of it too.

### 2.4 No existing knob fixed it

Every knob named in the issue was measured before the change (one run each, full table in `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`). `MLXCEL_CACHE_LIMIT=1GB`, `MLXCEL_MEMORY_LIMIT`, `MLX_ROCM_NO_ASYNC_POOL`, both managed-fallback switches, `MLX_ROCM_FINEGRAINED=0`, `MLX_GRAPH_NODEFER=1`, `MLX_MAX_OPS_PER_BUFFER=50` and `MLXCEL_CACHE_CLEAR_INTERVAL=0` (8B only) all left the peaks at 20.60 GB and 23.56 GB. Only `MLX_ROCM_QMM_DEQUANT_GEMM=0` moved the 8B, at the cost above, and `MLX_ROCM_USE_ASYNC_POOL=1` crashed both models. The periodic cache clear cannot matter in this run: its cadence is 256 tokens and the run decodes 128.

## 3. Dead Features Found

The issue listed several allocator mechanisms as candidates for the excess. None of them holds anything in these runs:

- **Decode arena** (`decode_arena_begin`): `decode_arena_begin` and `decode_capture_begin` have no caller in mlxcel or in the rest of the overlay, and `use_hip_graphs()` returns `false`. The arena is never allocated (capacity and high-water mark 0).
- **Graph deferral** (`MLX_GRAPH_NODEFER`): graph deferral never runs for the same reason, so the switch changes nothing.
- **Async pool** (`MLX_ROCM_USE_ASYNC_POOL`, `MLX_ROCM_FORCE_ASYNC_POOL`; `MLX_ROCM_NO_ASYNC_POOL` forces it off): off by default. Turned on, both models hit a GPU memory fault in `gather_rows_kernel` during the warmup pass and aborted, the failure the fork's own comment warns about.
- **Managed/GTT fallback** (`MLX_ROCM_ALLOW_MANAGED_FALLBACK`, `MLX_ROCM_NO_MANAGED_FALLBACK`): used only when a device allocation fails, which never happened. `mem_info_gtt_used` never rose by more than 0.01 GB in any run.
- **The cache limit itself**: `RocmAllocator::set_cache_limit` stored `max_pool_size_` and nothing read it, while the exact-size cache (`min_utilization` 1.0) keeps every buffer size it has seen until `clear_cache()`. The limit defaulted to the allocator's memory limit, 76.8 GiB on this host. `MLXCEL_CACHE_LIMIT` had never had any effect on ROCm.

These are recorded rather than removed. They are the fork's code, and the arena and deferral become live again if HIP graphs are turned back on.

## 4. Change Summary

Two commits on `update/issue-2062-rocm-cache-limit`, rebased onto `3c9edea0`:

- **`b72b03a6`** `fix(rocm): bound in-flight batch memory and enforce the cache limit`: the overlay changes, the ROCm cache default, the bench counters, the harness log option, the results page and data, the tests and the docs.
- **`5f665765`** `fix(rocm): count each eval's last batch and trim the cache with slack`: the two MEDIUM review findings. `gpu::finalize` now goes through `commit_and_throttle`, so the last batch of every eval and `async_eval` counts toward the budget, the miss-path trim goes to three quarters of the limit instead of to the limit, and a failed synchronize drops its in-flight tracking. Everything was re-measured on this commit.

Files by area:

- Overlay: `mlx/backend/rocm/device.cpp` and `device.h` (budget parsing, `commit_and_throttle`, `throttle_inflight`, `release_inflight`), `eval.cpp` (per-primitive byte count, finalize), `allocator.cpp` and `allocator.h` (`thread_allocated_bytes()`, miss-path trim), `LOCAL_FIXES.md` item 28.
- mlxcel: `src/execution/runtime.rs` (`DEFAULT_CACHE_LIMIT_BYTES`, pure `cache_limit_bytes`), `src/execution/runtime_tests.rs`, `src/lib/mlxcel-core/src/memory.rs` (cache-limit test), `tests/rocm_inflight_bound.rs`.
- Measurement: `src/bin/bench_decode.rs` (phase counters, applied cache limit), `scripts/bench_decode.sh` (`BENCH_RAW_DIR` keeps runner logs).
- Docs: `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md` with raw rows under `data/rocm-memory-gfx1151-2026-09-30/`, `docs/installation.md` (new "Memory footprint" subsection), `docs/environment-variables.md`.

## 5. The Fix

### 5.1 In-flight bound

The allocator keeps a thread-local, monotonic count of bytes handed out by `malloc` and `malloc_async`, cache hits included (`rocm::thread_allocated_bytes()`). `gpu::eval` reads it around each primitive's `eval_gpu` and adds the difference to the encoder's open batch (`add_batch_bytes`). Because the counter is thread-local and only increases, the worker thread freeing completed batches in the meantime does not disturb it.

`needs_commit()` on the eager path now fires at 2000 ops or once the open batch has allocated `inflight_budget_ / 4`. `commit_and_throttle()` commits, then `throttle_inflight()` records a raw HIP event after the batch, pops batches whose events have completed, and blocks on the oldest one while the committed total still exceeds the budget. The wait is `hipEventSynchronize` (the device runs in blocking-sync mode, so the thread sleeps), or a 50 microsecond polled wait that gives up at `MLX_ROCM_GPU_WATCHDOG_SECS` when that is set, matching the other host waits. A failed record or wait is stored as the stream's error, which the next synchronize throws, and ends the tracking. Nothing is tracked during a decode-step capture or any stream capture.

`MLX_ROCM_MAX_INFLIGHT_MB` must be a whole non-negative decimal integer that fits in `size_t` after scaling to bytes; anything else keeps 1024 with one stderr warning.

The bound is approximate by construction. It counts allocations, not only transients (a batch that grows the KV cache counts too), and a batch's buffers are dropped by the worker thread just after its event completes.

### 5.2 Cache-limit enforcement with trim hysteresis

On a cache miss in `malloc_async`, if the cache is over `max_pool_size_`, the allocator releases cached buffers down to three quarters of the limit (`get_cache_memory() - max_pool_size_ / 4 * 3`). Only a miss trims: it is the only place the footprint grows, and it already pays for a HIP allocation. `free()` and cache hits still never call `hipFree`, which the fork avoids because freeing tens of GB of same-sized activations between training steps turns into a blocking drain. With the trim, active plus cache stays within the live set plus the limit plus the one request. The quarter of slack means one round of blocking `hipFree` calls covers several misses of a workload whose shapes keep changing, instead of one `hipFree` per miss once the cache sits at the limit.

### 5.3 The ROCm default of `MLXCEL_CACHE_LIMIT`

`DEFAULT_CACHE_LIMIT_BYTES` is `Some(2 GiB)` under `cfg(feature = "rocm")` and `None` otherwise. `cache_limit_bytes(raw, default)` is a pure function with these rules: unset or empty gives the default; `0` or `none` in any case, or any size that parses to zero, gives no bound; a valid size gives that size; an unparseable value gives the default, so a typo cannot silently remove the ROCm bound. `resolve_cache_limit()` applies the result through `mlxcel_core::memory::set_cache_limit`. The harness logs show the applied value as `[Memory] cache limit: 2.15 GB`, which is 2 GiB in decimal GB.

### 5.4 `memory_limit()` left alone

On ROCm the estimate behind `mlxcel inspect` and `--estimate-memory` reads the allocator's `memory_limit()` as available memory (#1805). The cache default goes through `set_cache_limit` and the in-flight budget is internal to the backend, so `memory_limit()` stays 76.80 GiB and every estimate is what it was. `runtime_tests::the_cache_default_leaves_the_memory_limit_the_estimator_reads` checks that runtime bring-up leaves it unchanged. Lowering the memory limit instead would have bounded nothing (section 2.2) and would have changed every estimate.

## 6. Results

### 6.1 Before and after, through the harness

`scripts/bench_decode.sh` at pp512/tg128, three runs each, alternating, medians. "Before" is the same binary with both bounds off (`MLXCEL_CACHE_LIMIT=none MLX_ROCM_MAX_INFLIGHT_MB=0`), which reproduced the origin/main peaks exactly. Every run went through a guard that sampled `/sys/class/kfd/kfd/proc` and the process list every 0.25 s and rejected and reran any run that overlapped another GPU process or a compiler, because another development unit was profiling on the same GPU.

| Model | Setting | MLX peak | VRAM delta | Steady state (active + cache after the measured pass) | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | before | 20.60 GB | 21.19 GB | 5.16 GB | 1064.83 | 37.02 |
| Llama-3.1-8B-Instruct-4bit | defaults | 6.14 GB | 6.73 GB | 5.18 GB | 1045.87 | 37.57 |
| Qwen3-30B-A3B-4bit | before | 23.56 GB | 24.11 GB | 17.67 GB | 300.07 | 61.79 |
| Qwen3-30B-A3B-4bit | defaults | 18.58 GB | 19.16 GB | 17.56 GB | 303.75 | 61.69 |

The peak over the weights goes from 15.85 GB to 1.39 GB for the 8B and from 6.39 GB to 1.41 GB for the MoE model (computed from the table and the weights in 2.1): about 1.4 GB on both models, the same order as the 1 GiB budget. Peak and VRAM delta varied by at most 0.14 GB across the three runs of each row. Decode changes by +1.5% and -0.2%, inside the issue's 2% bound. The 8B's prefill loses 1.8%, the host now waiting on the GPU instead of queueing far ahead. The MoE model's +1.2% is noise: its prefill ranged from 264.33 to 315.02 tok/s over the three default runs and from 291.52 to 306.43 without the bounds. GTT rose by at most 0.01 GB and host `VmHWM` stayed between 0.75 and 0.87 GB. The steady state barely moves because `generate_with_stats` already clears the cache after prefill; what moved is the peak.

### 6.2 Choosing the numbers

Both sweeps ran on the first version of the bound (whose host wait spun on `hipEventQuery`, did not count each eval's last batch, and trimmed the cache to the limit), one run each; the shipped version is what 6.1 measured.

- **In-flight budget**: from 256 to 4096 MiB the 8B's peak went 5.28, 5.75, 6.36, 7.34, 9.36 GB and the MoE model's 17.82 to 21.66 GB, with decode flat. The peak tracks the budget as designed. 1024 MiB is the default because a transient larger than a quarter of the budget commits its own batch, and a model with larger weight matrices (a 70B MLP matrix is about 470 MB in f16, computed, not measured) keeps the GPU fed only if the budget holds a few of them. Smaller budgets did not buy measurably lower cost.
- **Cache limit**: decode stayed within 2% of no bound at every value down to 128 MiB. Prefill did not: at 128 MiB the 8B lost 16% (892.73 tok/s), most likely because each 117 MB f16 weight copy no longer survives in the cache and is reallocated for every matrix. From 512 MiB up nothing was measurable. 2 GiB is four times that smallest free value, for the larger weight copies of bigger models. In these runs the cache never reached 2 GiB, so the default did not bind; it bounds what a long-running process can accumulate between periodic clears, which the exact-size cache otherwise let grow toward 76.8 GiB.

## 7. The Pooled-HipEvent Heap Corruption

The first version of the in-flight bound tracked batches with the fork's `HipEvent` wrapper. `HipEvent` returns its handle to a function-local static pool when it is destroyed. A `CommandEncoder` can be destroyed after that pool during static destruction at process exit, so its `HipEvent` members returned handles into a pool that no longer existed. `tests/rocm_mxfp4_quant.rs` aborted at exit with `malloc_consolidate(): unaligned fastbin chunk detected`.

The shipped version stores raw `hipEvent_t` handles owned by the encoder: a deque of `InflightBatch { done, bytes }` and a vector of spare events for reuse. `release_inflight()` moves in-flight events back to the spare list, and the encoder's destructor calls it and then `hipEventDestroy`s every spare, ignoring errors because a faulted device returns the fault from every destroy. The comment on `InflightBatch` in `device.h` and `LOCAL_FIXES.md` item 28 both record why `HipEvent` is not used, so the next change to this code does not reintroduce it. `rocm_mxfp4_quant` then passed three runs.

## 8. Technical Decisions

- **Fix the mechanism that held the memory, not the counter the issue named.** The issue rejected changing `max_pool_size_` only inside the fork and preferred an operator-visible cache default. The measurement moved the problem to command-batch lifetime, so the main fix is in the overlay, as the issue's own fallback clause directs, and the cache default still ships through `MLXCEL_CACHE_LIMIT` because the cache limit was unenforced and unbounded.
- **Bound by bytes, not by op count or scheduler tasks.** More frequent commits without a host wait did nothing, and MLX's scheduler cap counts tasks, not bytes. Counting bytes per primitive bounds the quantity that actually grew, whatever the op mix.
- **Wait on the oldest batch, not on the whole stream.** Blocking only when the in-flight total exceeds the budget, and only on the oldest batch, keeps up to a budget of work queued so the GPU does not starve; synchronizing the stream at each commit would drain the queue every time.
- **Trim only on a miss, to three quarters.** It keeps `free()` and cache hits free of blocking `hipFree` (the fork's training-path concern) and amortizes trims when shapes keep changing.
- **Unparseable `MLXCEL_CACHE_LIMIT` keeps the default.** An operator disabling the bound has to say `0` or `none`. Before this change, garbage meant no bound; on ROCm that would now silently remove a default the docs promise.
- **Leave `memory_limit()` unchanged.** It is the pre-load estimate's input on ROCm, and neither bound needs it.
- **Keep `0` as an escape hatch for the in-flight bound.** It restores the exact old behavior, which also made the "before" rows reproducible from one binary.

## 9. Validation

From the PR body, on gfx1151:

- The before/after sweep (6.1) and the knob, in-flight-budget and cache-limit sweeps in the results page.
- `cargo test -p mlxcel-core --lib memory::tests::`, including the new `cache_limit_bounds_the_free_buffer_cache` (64 buffers of distinct sizes freed under a 1 MiB limit must leave under 8 MiB cached; with the trim removed it held 49.8 MB and failed).
- `cargo test -p mlxcel --lib execution::` (151 pass, including the new cache-default tests), and `--test rocm_inflight_bound` (40 f16 4-bit 8192x8192 matmuls over 512 rows in one evaluation, 5 GiB of dequantized copies: the peak rose by 1.75 GiB with the default, and by 6.25 GiB with `MLX_ROCM_MAX_INFLIGHT_MB=0`, which fails its 3 GiB bound).
- `rocm_gpu_faults`, `rocm_mxfp4_quant` (3 runs), `rocm_cpu_blas_finegrained`, `rocm_slice_update_source`, `rocm_strided_scan`, `rocm_fft_plan_cache`, `dead_doc_pointers`: pass. `scripts/ci/rocm_smoke.sh` OK. Clippy `-D warnings` on the touched crates and targets.
- Review (pr-reviewer): no CRITICAL or HIGH; both MEDIUM findings fixed in `5f665765`, and everything above re-run on it.

Orchestrator verification on gfx1151 with the branch rebased onto `3c9edea0`:

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, `verify-rocm-overlay` (108 backend and 15 core files, 28 `LOCAL_FIXES` entries), fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in exactly one target, `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible` (bf16), which is tracked in #2081 and is the last baseline failure, not introduced here. mlxcel-core lib passed 1838 tests.
- `tests/rocm_inflight_bound.rs` passed inside the full suite.

Rechecked from the committed data while writing this report: the medians in 6.1 match `harness-bounds-off.csv`, `harness-defaults.csv` and `memory-lines.txt` (peaks 6.14, 6.14, 6.14 and 18.57, 18.58, 18.58 GB with the defaults; 20.60 and 23.56 GB in all three runs without).

## 10. Learning Points

- **Measure which counter holds the memory before bounding one.** The issue's plan was a cache limit. The allocator's peak excludes the cache by definition, so a peak far above the weights with a small cache pointed at live buffers from the first measurement.
- **A limit that is stored is not a limit that is enforced.** `set_cache_limit` and `set_memory_limit` both "worked" on ROCm in the sense of returning, and neither changed anything: one was never read, the other depended on scheduler accounting the eager path bypassed. A test that asserts the effect (the cache test, the in-flight test) is what distinguishes them.
- **Batching without back-pressure only changes the batch size.** Committing more often moved nothing until the host was made to wait.
- **Static destruction order applies to GPU handle pools.** A pooled RAII wrapper with a function-local static pool is unsafe in any object that can outlive it at exit. The failure surfaced as heap corruption in an unrelated test's teardown.
- **Keep a switch that reproduces the old behavior exactly.** `MLX_ROCM_MAX_INFLIGHT_MB=0` plus `MLXCEL_CACHE_LIMIT=none` let one binary produce both halves of the before/after table, which removed build differences from the comparison.

## 11. Caveats, Not Verified, and Remaining Work

- **The pre-load estimate still sits under the 8B's peak.** `mlxcel inspect --max-tokens 640` estimates 5.58 GB for the 8B (weights and KV cache times 1.20 plus an activation term) and 20.72 GB for the MoE model. The 8B now peaks at 6.14 GB, 0.56 GB above its estimate (it was 15 GB above); the MoE model now peaks below its estimate. Recalibrating the 1.20 headroom factor for ROCm is left out of this PR.
- **The budget and cache sweeps ran on the first version of the bound.** The shipped version differs in counting each eval's last batch, blocking instead of spinning, and trimming with slack; only the harness table measured it.
- **The bound is approximate.** It counts allocations, including persistent ones such as KV cache growth, and buffers are released just after their event completes.
- **No automated test of `MLX_ROCM_MAX_INFLIGHT_MB` parsing** (the reviewer's LOW finding, not fixed).
- **Metal and CUDA were not run.** The shared code touched is `runtime.rs` (default stays `None` off `rocm`, pinned by `the_default_cache_limit_is_rocm_only`), `memory.rs` (a new test that MLX's own allocators satisfy there) and `bench_decode.rs` (print-only).
- **Two models, one host.** The 1024 MiB and 2 GiB defaults were chosen on an 8B and a 30B MoE at 512 prompt tokens; larger models and longer prompts are reasoned about, not measured.
- **Upstreaming.** Item 28 is an upstreaming candidate for the fork (#1813).

Refs: #2062, #1814, #1801, #1805, #1813, #2081, #627.
