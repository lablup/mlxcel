# Technical Report: PR #2049 - Set ROCm Blocking-Sync Before the First HIP Queue, Restore the FFT Plan Cache to 128

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (HIP runtime calls in the ROCm overlay), Rust (integration test), Markdown

**Risk Level**: Medium (changes when every ROCm process sets its device wait mode, on the allocator's hot path; ROCm-only overlay plus a `#![cfg(feature = "rocm")]` test, Metal and CUDA never copy `patches-rocm/`)

## Executive Summary

Issue #1876 (part of epic #1801) asked why the ROCm hipFFT plan cache had to be capped at 8 entries when the CUDA file uses 128. Above 8, an FFT-heavy workload hung forever with the GPU idle, and the cause had been written down as "somewhere inside rocFFT". The investigation in this PR shows the hang was not in rocFFT and did not depend on the number of live plans. It was an ordering bug in the ROCm backend: `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` ran in `rocm::device()`, after the allocator had already created the null stream's HIP queue through `hipStreamQuery(nullptr)`. CLR had given that queue non-interrupt completion signals, which cannot carry the async handler that a later `hipFree` relies on, so the first `hipFree` that had to synchronize the null stream never returned. rocFFT's `hipfftDestroy` at the first plan eviction was simply the first code in mlxcel that did null-stream work followed by `hipFree`.

A 30-line standalone HIP program with no MLX and no hipFFT reproduces the hang: touch the null stream, set the flag, then loop over `hipMalloc`, `hipMemcpy`, a null-stream kernel and `hipFree`. It hangs at the first `hipFree`, and completes when the early touch is removed.

The fix adds `rocm::ensure_device_flags(int)` and `ensure_current_device_flags()`, which set the flag once per device before any queue exists on it, and calls them from every path that can reach HIP before `rocm::device()`. The plan cache goes back to 128, and a new integration test, `tests/rocm_fft_plan_cache.rs`, drives 48 evicting round trips at cache size 16 in a child process and fails if the hang returns. The defect was never specific to FFT: any device-wide sync with pending null-stream work was exposed, so the fix is wider than the issue that found it.

## 1. Problem Statement

### What the issue recorded

The FFT port from #1825 (PRs #1856 and #1861) set `default_capacity` to 8 on the hipFFT `LRUBytesKeyCache` in `fft.hip`, with a comment saying the root cause inside rocFFT was unknown. The measurements were consistent and misleading: a probe cycling 26 transform shapes completed at capacity 8 and hung at 16, 24, 32 and 128, always at the same position in the sequence. Reordering the cases moved the hang to whatever ran at that position, and 60 repeats of a single shape never hung. That pattern pointed at "too many live plans", and the issue's proposed hypotheses followed from it: rocFFT runtime compilation, the manual work area set by `hipfftSetAutoAllocation(handle, 0)`, and plans outliving their cache entries through `shared_ptr` captures. A standalone program had also created 64 hipFFT plans without trouble, so the limit was known to depend on how MLX used the library, but not how.

The cost of the cap was real. Kokoro text to speech and the Phi-4-multimodal audio front end, the two mlxcel paths that reach a device FFT, rebuilt plans far more often than on CUDA.

### What the investigation found

This is the core of the PR, and the reasoning is worth keeping because the symptom pointed away from the cause at every step.

**Step 1: locate the hang in time.** The probe was rebuilt as `fft_numeric_probe --plans N` (originally from PR #1878). At cap 16 it hung after `plan 7`; at cap 32, after `plan 15`. Each probe iteration creates two plans (R2C and C2R), so the iteration after plan 7 creates the 17th plan and the iteration after plan 15 creates the 33rd: in both cases the first plan that forces an eviction. The hang was not at a live-plan threshold but at the first `hipfftDestroy`. Because the LRU cache evicts while inserting the new entry, the stall looked like "creating the next plan blocks", which is how the issue described it.

**Step 2: locate the hang in the stack.** rocgdb on the blocked thread gave `hipfftDestroy -> rocfft_plan_destroy -> ~ExecPlan -> ~TreeNode -> gpubuf_t::free -> hipFree -> ... -> std::condition_variable::wait`. The GPU was idle and no rocFFT RTC helper process was running, which ruled out the runtime-compilation hypothesis. rocFFT was only freeing a device buffer; the wait was inside HIP's `hipFree`, which synchronizes all streams on the device (`Device::SyncAllStreams`) before freeing.

**Step 3: find why the wait never wakes.** With `AMD_LOG_LEVEL=4`, the last line before the hang was `rocvirtual.cpp:844 hsa_amd_signal_async_handler() failed to set the handler!`. ROCr refuses async handlers on signals that have no event mailbox, that is, non-interrupt signals. `SyncAllStreams` enqueues a marker on the null stream and waits for the marker's completion handler to fire. The handler was never installed, so the condition variable was never notified.

**Step 4: find where the non-interrupt signal came from.** CLR types a queue's ring of completion signals according to the device wait mode in force when the queue is created. In the default active-wait mode, the host spins on signals, so CLR allocates cheap `HSA_AMD_SIGNAL_AMD_GPU_ONLY` (non-interrupt) signals. While active wait is on, `HwQueueTracker::ActiveSignal` swaps in an interrupt signal whenever a command needs a handler. Once `hipDeviceScheduleBlockingSync` turns active wait off, that swap stops, on the assumption that the queue was created with interrupt signals in the first place. The HIP log showed which queue violated that assumption: the null stream's queue was created by `hipStreamQuery(nullptr)` in `allocator::Buffer::raw_ptr()`, on the first host read of a unified buffer, and only afterwards did `rocm::device()` call `hipSetDeviceFlags`. The null stream's queue therefore kept non-interrupt signals for the life of the process, and the first marker that landed on one of them and needed a handler hung its waiter.

**Step 5: prove it outside MLX.** The standalone reproducer in the PR body strips everything else away (built with `hipcc --offload-arch=gfx1151 -O2`):

```cpp
if (early_touch) (void)hipStreamQuery(nullptr);        // creates the null-stream queue now
(void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync); // flips active wait off
// loop: hipMalloc, hipMemcpy H2D, kernel on the null stream,
//       keep 8 buffers live, hipFree the oldest
```

`timeout 30 ./repro 1 200` prints `iter 7` and exits 124, hanging at its first `hipFree`. `./repro 0 200`, which only drops the early touch, prints `done`. No hipFFT, no MLX, no plan cache: the whole defect is the order of two HIP calls.

### Why the issue's hypotheses were wrong, and what that rules out

- **Live plan count.** The hang depends on the first eviction, not on how many plans are alive. The standalone reproducer has no plans at all.
- **rocFFT runtime compilation.** No RTC helper was running, and the stack was in `hipFree`, not in a compile or cache lock.
- **Plans outliving their cache entries.** The `execute_fft` comment claimed the captured `shared_ptr` kept the plan alive for the asynchronous execution. It does not: `CommandEncoder::launch_kernel` calls the lambda synchronously and drops it on return. That premise was the reason the cap "had to leave headroom", and it was false.
- **rocFFT itself.** rocFFT's only role was to be the first caller of `hipFree` after null-stream work. Nothing needs to be reported to rocFFT. The reproducer is a candidate for a CLR report instead (not filed yet).

One part is not explained: why capacities of 12 or below completed while 16 and above hung at the first eviction, given that evictions also happen at capacity 8 with 26 shapes. The PR's working guess is the position in the null stream's signal ring at the time of the first `hipFree` (the reproducer also first hangs after eight kernels), but this was not characterized.

## 2. Change Summary

- **`device.cpp` / `device.h`: `ensure_device_flags(int)` and `ensure_current_device_flags()`.** Sets `hipDeviceScheduleBlockingSync` once per device index. A lock-free atomic bitmask covers indices below 64, because this runs on every unified allocation and every host read of a unified buffer; a mutex and a vector cover anything above. The helper switches to the target device only when the caller is on another one and restores the caller's device afterwards. If `hipSetDevice` fails, the device is not recorded, so a later call retries. If `hipSetDeviceFlags` fails, the error is cleared, one stderr line names the device and the HIP error, and the device is recorded so the failure is not retried on every allocation. `rocm::device()` now calls the helper instead of setting the flag inline. The flag value is unchanged.
- **Call sites that can create a queue before `rocm::device()`.** `unified_malloc` in `allocator.cpp` (often the first HIP work in the process); both branches of `Buffer::raw_ptr()` (the null-stream query on unified memory and the `hipDeviceSynchronize` on discrete memory); the opt-in async-pool free streams in the `RocmAllocator` constructor, per pooled device; `gpu::init()` in `eval.cpp` before its `hipFree(nullptr)`; and `staged_write` in `host_stage.cpp` before its device sync, because a checkpoint save can be the first sync on a thread's device.
- **`fft.hip`.** `default_capacity` returns to 128, matching `mlx/backend/cuda/fft.cu`. The cache comment now gives the real cause and points to `LOCAL_FIXES.md` entries 18 and 20. The `execute_fft` comment is corrected: the captured plan lives only as long as the enqueue, eviction is still safe for the eager launch because `hipFree` synchronizes the device first, and the opt-in `MLX_GRAPH_PREFILL_REPLAY=1` graph path was not examined.
- **`tests/rocm_fft_plan_cache.rs` (new, 285 lines).** The parent test re-invokes this binary on an ignored child test with `MLX_ROCM_FFT_CACHE_SIZE=16` and a 120 s budget, and kills the child if it does not finish. The child runs 48 rfft/irfft round trips at distinct lengths (96 plans through a 16-entry cache, so evictions start after eight round trips and repeat), then compares GPU against the CPU stream within 1e-5 for rfft, irfft, round trip and complex fft at lengths 2, 8, 400, 512, 1024, 1200 and 2048, plus a 512-batch rfft. The child body runs only when `MLXCEL_ROCM_FFT_CHILD` is set, so `--include-ignored` sweeps do not run it in a shared process.
- **`LOCAL_FIXES.md`.** Entry 18 (the FFT port) now describes the 128 capacity and the history. New entry 20 records the ordering defect, the CLR mechanism, every call site, the failure handling, the evidence and the reproducer, as a fork-side upstreaming candidate.
- **`docs/environment-variables.md`, `docs/installation.md`.** `MLX_ROCM_FFT_CACHE_SIZE` is documented with default 128 and the hang is described as fixed (#1876).

Commit history: `84e8ad19` is the fix; `6e4ac7d6` names the `LOCAL_FIXES.md` entries in the `fft.hip` comment; `9a9c1b06` makes the helper's failure paths visible and makes the test kill its child if polling it fails; `78a53b15` and `3ccab4e5` merge origin/main (the first to pick up `LOCAL_FIXES.md` entry 19 from PR #2046).

## 3. Technical Decisions

### Fix the order, not the cap

Keeping the cap at 8 would have left the defect in place for every other `hipFree` or device-wide sync that follows null-stream work. FFT was only the first caller to trip it; a future op that frees device memory at runtime could have hung the same way with no FFT involved. Fixing the order removes the class of hangs, and the cache capacity goes back to a memory-versus-rebuild tradeoff rather than a safety limit.

### Flag early at every entry point instead of once at process start

HIP has no hook that runs before the first HIP call made by any code, and the device index is not known until MLX picks a device. The flag has to be set on the right device before the first queue on it, so the helper runs at each place that can create a queue ahead of `rocm::device()`. The per-call cost after the first time is one atomic load and a bit test, which is why the fast path is lock-free: it sits on the allocation path.

### Keep blocking-sync rather than dropping it

Removing `hipSetDeviceFlags` altogether would also make the mode consistent (active wait everywhere), but it would change host CPU usage during every GPU wait, which is the reason the fork sets blocking-sync in the first place. The PR keeps the flag value and changes only when it is applied.

### Per-device, current-device-only

The helper tracks each index separately, so flagging device 0 first cannot leave device 1 unflagged, and it never iterates every device. The existing `rocm::device()` comment records why: on a multi-GPU host, creating a context or queue on the other GPU is what wedged the discrete GPU's queue over a TB5 link. The helper preserves that constraint and restores the caller's current device so it can be called from the allocator without side effects.

### Make failure visible, retry only what can succeed

A `hipSetDevice` failure means the flag was never attempted, so the device stays unrecorded and the next call retries. A `hipSetDeviceFlags` failure is recorded and reported once on stderr. Without that line, a device left in active-wait mode would reproduce #1876 with no trace, which is the silent hard hang this PR exists to remove.

### Test in a child process with a timeout

The hang has no timeout of its own and the plan cache capacity is read from the environment on the first FFT, so the test cannot run in-process. The child-process pattern gives both: a fresh environment and an external kill. The test was confirmed to fail (`the child did not finish within 120s`, after 8 of 48 round trips) with the early calls commented out, so it guards the ordering and not just FFT correctness.

## 4. Validation

Author's runs (gfx1151, Radeon 8060S, HIP 7.15.26333, ROCm 10.0 core, rocFFT 1.0.39):

- Probe at cap 128 (`--plans 40`), three consecutive runs: all exit 0, 40/40 round trips ok. This run evicts nothing, so the eviction runs below are the meaningful ones.
- Probe at caps 16 and 32, which hung before the fix after `plan 7` and `plan 15`: both exit 0, 40/40 ok. Cap 16 with `--plans 100` also completes.
- Probe correctness mode: all 25 comparisons within 1e-5 of the CPU stream; worst 4.2e-7 (round trip at n=2048 and n=1200).
- Ordering proof under `AMD_LOG_LEVEL=3`: `hipSetDeviceFlags ( 4 )` on line 26, before `hipExtMallocWithFlags` (line 32), `hipStreamQuery ( <null> )` (line 50) and the first `Created SWq` (line 52). Before the fix the flag came after the null-stream `Created SWq`.
- `cargo test --features rocm --test rocm_fft_plan_cache`: passes (9.97 s and 6.91 s); fails with the early calls commented out.
- Standalone reproducer: exit 124 at `iter 7` with the early touch, `done` without it.
- Clippy on the new test with `-D warnings`, `dead_doc_pointers`, and the fast `make verify-*` gates: pass.

Orchestrator verification (gfx1151, branch head `3ccab4e5`, which merges origin/main `ac02b2dc`):

- The regression test and `dead_doc_pointers` pass (child finishes in about 7 s); the regression test fails with the early calls commented out; the probe at cap 128 three times, at caps 16 and 32, and in correctness mode all complete; `scripts/ci/rocm_smoke.sh` with Qwen3-0.6B-4bit generates 32 tokens.
- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in three targets with exactly the 37 known baseline failures and nothing else:
  - `-p mlxcel-core --lib`: 35 failures. 34 are fused paged-attention tests without a ROCm port (#1814); the other is the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`.
  - `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`, both from #2037.
- The new `tests/rocm_fft_plan_cache.rs` passed inside the full suite.

## 5. Learning Points

- **A threshold that moves with the configuration is often an event, not a limit.** "Hangs above N live plans" became "hangs at the first eviction" as soon as the probe printed plan indices at two different caps. Measuring the position of the hang at more than one setting was what broke the live-count theory.
- **Get the stack before forming hypotheses.** The issue's four hypotheses were all reasonable and all wrong. One rocgdb backtrace moved the problem from rocFFT to `hipFree`, and one `AMD_LOG_LEVEL=4` line moved it from `hipFree` to signal typing.
- **Some HIP device flags must be set before the first queue, not just before the first stream.** CLR types a queue's completion signals by the wait mode at creation and does not re-type them. Any HIP call that touches the null stream (a query, a sync, a copy, a launch) can create a queue. Code that sets device flags "when the device is first used" has to define first use as the first HIP call of any kind, including those made by the allocator.
- **Reduce to the runtime before blaming a library.** The standalone reproducer shows the defect with two HIP calls in the wrong order. Without it, the natural next step would have been a rocFFT bug report about something rocFFT does not do.
- **Re-check comments that justify a limit.** The "captured plan outlives the call" comment was the stated reason for cache headroom. It described behavior `launch_kernel` does not have.

## 6. What Is Not Verified

- **Other GPUs and ROCm versions.** Only gfx1151 with HIP 7.15.26333 and rocFFT 1.0.39 was measured. The CLR behavior is generic, but a different HIP could type signals differently.
- **The threshold at small capacities.** Why capacities of 12 or below completed at all is not characterized; the signal-ring position is a guess.
- **The `MLX_GRAPH_PREFILL_REPLAY=1` path for FFT.** It records the hipFFT call into a graph that launches later, and the eviction-safety argument in `execute_fft` covers only the eager launch.
- **The `hipSetDeviceFlags` failure path.** The stderr message is reached only if the call fails, and no test forces that.
- **Metal and CUDA.** Not available on this host. The change touches only `patches-rocm/` and a `#![cfg(feature = "rocm")]` test, so neither build is affected by construction.
- **Multi-GPU hosts.** The helper touches only the requested device by design, but no multi-GPU ROCm host was tested.

## 7. Remaining Work

- File the standalone reproducer as a CLR report. None has been filed.
- Propose `LOCAL_FIXES.md` entry 20 (and the updated entry 18) to the ROCm fork as part of #1813.
- Pre-existing issues outside this diff: `lru_cache.h` evicts from an empty list when `MLX_ROCM_FFT_CACHE_SIZE=0`, and `std::stoul` throws on a non-numeric value.
- Examine FFT under `MLX_GRAPH_PREFILL_REPLAY=1`.
- Baseline test failures: #1814 (fused paged-attention ports, 34 failures), the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` failure, and the two #2037 failures.

Refs: #1876 (closed by this PR), #1801, #1825, #1878, #2033, #2046.
