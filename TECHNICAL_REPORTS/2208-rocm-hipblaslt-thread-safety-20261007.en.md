# Technical Report: PR #2208 - Make hipBLASLt GEMM State Safe Across Threads and Streams

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `df8bc628` on origin/main `5976af8d`, PR open, pending merge. Closes #2200 (part of #1801).

**Languages**: C++ and HIP (ROCm overlay `gemms/hipblaslt_gemm.cpp`, `gemms/hipblaslt_gemm.h`, `device.cpp`), Rust (new `tests/rocm_hipblaslt_concurrency.rs`, edited `tests/rocm_rocblas_handle_concurrency.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Medium. No GEMM argument or result changes, but the pipe cache moves from one process-wide map to one map per thread, and workspace buffers move from one per device to one per (device, stream) and are now freed by `~CommandEncoder`. The per-stream workspace path never ran on the test host (the heuristics asked for no workspace), so it is covered by construction only. Metal and CUDA code paths are untouched.

## Executive Summary

Issue #2200 found the same class of race as #2198 in the hipBLASLt GEMM layer, which is the default route for every bf16 and fp16 GEMM. `gemms/hipblaslt_gemm.cpp` published its per-device handle through a plain `bool` set before `hipblasLtCreate` ran, gave every stream on a device one shared 32 MB workspace that `ensure_workspace` freed and reallocated with no lock, and kept GEMM pipes (layouts, matmul descriptor, algorithm) in one process-wide map that threads used after unlocking and erased under each other. The server enqueues hipBLASLt GEMMs from several threads sharing one device, each on its own stream.

The PR does three things. The handle is created under the state mutex with the target device current and published through an atomic init state with release after success. The workspace is one buffer per (device, stream), allocated at the 32 MB cap on first use and released when the stream's `CommandEncoder` is destroyed. The pipe cache is thread-local, inserted only after a successful matmul. This last point deviates from the issue, which proposed `std::shared_ptr<GemmPipe>`: five experiments on gfx1151 showed that `hipblasLtMatmul` writes into the descriptor objects it is given, so a pipe cannot serve two threads at once however it is owned.

The new test reproduces the failure. With `hipblaslt_gemm.cpp` restored to main, 20 of 20 runs failed (50 of 200 child processes died, and 12 runs had a hung child). With the fix, 0 of 20 runs and 0 of 200 child processes failed. The heuristics requested workspace 0 for the tested shapes, so the per-stream workspace path is untested at runtime on this host. Decode throughput on Llama-3.1-8B-4bit is unchanged within noise. The orchestrator's `make verify-rocm` on `df8bc628` passed all but one test, a failure on main unrelated to this PR (section 4.3).

## 1. Problem Statement

### 1.1 Shared-state inventory

Read on main at `84d3a7bc`, all in `gemms/hipblaslt_gemm.cpp`:

| State | Guard before | Problem |
|---|---|---|
| `g_state[d].initialized`, `.available`, `.handle` | written under `state.mutex`, read unlocked | `initialized` set before `hipblasLtCreate` ran; plain `bool`, a data race. A reader that saw it could see `available == false` or a null handle. |
| `g_state[d].workspace`, `.workspace_size` | init under the mutex, `ensure_workspace` unlocked | One buffer for every stream on the device; unlocked `hipFree` and `hipMalloc`. |
| `g_caps[d]` | `g_caps_mutex` for the whole probe | `probed` set before `get_handle`; if that threw in the init window, every capability stayed `false` for the process. |
| `g_pipe_cache` and the descriptors each `GemmPipe` owns | map lookups under `g_pipe_mutex`, pipe used after unlock | A failed cached matmul destroyed and erased an entry another thread could be passing to `hipblasLtMatmul`. |
| `algo_cache` (and the fp8 one) | own mutex, entries copied out by value | Sound. |
| the `hipblasLtHandle_t` itself | none | Sound: `hipblaslt.h` asks callers to synchronize only handle-mutating helpers, and this file never mutates the handle after create. |

### 1.2 Failure modes

- **Shared workspace (corrupts output).** `hipblasLtMatmul` only enqueues, so two GEMMs on two streams whose algorithms request scratch memory run on the GPU at the same time over the same buffer. Stream order protects reuse only within one stream.
- **Workspace growth (GPU use-after-free).** `ensure_workspace` could free the buffer while another stream's queued GEMM still used it.
- **Pipe erase (host use-after-free).** On a failed cached matmul (a stale bucketed algorithm) the thread destroyed the entry's layouts and descriptor and erased it while another thread held them. The retry path fixed `algo_cache` but left the bad algorithm in the pipe, so the next call on that key failed and erased again.
- **Init window (wrong route).** A thread arriving between the flag and the create saw `available == false` and fell back to rocBLAS for that GEMM. Inside `gemm_caps`, the throw left the capability table all `false` for the process, which turns off the fp8 prefill route.
- **Device not current.** `hipblasLtCreate` ran without making `device_id` current.

On gfx1151 the workspace races do not run, because the heuristics request no workspace (section 4.3). The reachable ones there are the pipe descriptors and the init window.

## 2. Change Summary

| Area | Change |
|---|---|
| `gemms/hipblaslt_gemm.cpp` | `HipblasltState` holds the handle and `std::atomic<int> init_state`; `ensure_handle`. `stream_workspace` replaces `ensure_workspace`. Thread-local `GemmPipeCache` of `std::unique_ptr<GemmPipe>`; `GemmPipe` destroys its descriptors and cannot be copied. `gemm_caps` marks probed after probing. Heuristic-miss logging. Thread-safety comment at the top of the file. |
| `gemms/hipblaslt_gemm.h` | New `hipblaslt_release_stream_workspace(hipStream_t)`. |
| `device.cpp` | `~CommandEncoder` calls it before the stream is destroyed. |
| `tests/rocm_hipblaslt_concurrency.rs` (new) | Fresh-process concurrency test (section 4.1). |
| `tests/rocm_rocblas_handle_concurrency.rs` | `MLX_NO_HIPBLASLT=1` removed. |
| `patches-rocm/LOCAL_FIXES.md` | Item 38. |

One commit, 6 files, 665 insertions, 147 deletions.

## 3. Design

### 3.1 Handle init

`HipblasltState` keeps the handle, a mutex and `init_state` (untried, ready, unavailable). `ensure_handle` returns on a nonzero acquire load. Otherwise it locks `state.mutex`, rechecks, saves the current device, makes `device_id` current, calls `hipblasLtCreate` into a local, restores the device, and only on success stores `state.handle` and then `init_state` with release. `get_handle()` and `is_hipblaslt_available()` read the handle only after an acquire load says ready. A failed create warns once and stays unavailable for the process, as before, because `is_hipblaslt_available()` runs on every GEMM route decision and must not retry a create each time. The workspace allocation that init used to do (32 MB per device) is gone.

`gemm_caps` now marks `probed` only after the probes ran. The one early `probed = true` is on the path where `get_handle` threw, which happens only once the create has failed, which is permanent. Lock order stays `g_caps_mutex`, then `state.mutex`.

### 3.2 Per-stream workspace

`stream_workspace(device_id, stream, required)` returns the stream's buffer and its size under `g_ws_mutex`:

- `required == 0` returns `{nullptr, 0}` without locking; a request above the 32 MB cap throws.
- A stream with no buffer gets exactly `kMaxWorkspaceBytes`, allocated with `device_id` current. Every heuristic preference in the file caps at that size, so a buffer never grows or moves while its stream lives, which removes the growth race.
- A stream under capture (`hipStreamGetCaptureInfo` not `None`) with no buffer throws instead of calling `hipMalloc` into the capture.
- On `hipMalloc` failure it returns `{nullptr, 0}`: the tune loops skip that algorithm, and every other site throws `hipBLASLt: failed to allocate workspace of N bytes` instead of passing a null buffer to `hipblasLtMatmul`.

`hipblaslt_release_stream_workspace(stream)` finds the stream's buffer, erases the entry, synchronizes the stream and frees the buffer. `~CommandEncoder` calls it before its stream is destroyed, so a later stream that reuses the handle value cannot inherit a buffer a queued GEMM still reads. The table is leaked on purpose, like the JIT module cache (item 35), so static teardown never races an encoder destructor.

Cost: 32 MB per stream that runs a workspace-needing GEMM. A device whose heuristics request none allocates nothing, and the 32 MB per device that init allocated is gone.

### 3.3 The pipe cache

`thread_pipe_cache()` gives each OS thread its own `unordered_map<GemmPipeKey, unique_ptr<GemmPipe>>` (the HIP host-callback thread of the async MoE path gets one too), so no lock sits on the fast path. Each thread's map is leaked so a thread exiting during teardown never calls into hipBLASLt after it unloaded.

- **Insert after success.** A miss builds the pipe locally, runs the matmul and the retry, and inserts the pipe only after a matmul succeeded, with the algorithm that succeeded. A retry's fresh algorithm therefore replaces the stale one; before, it stayed in the cache.
- **Erase on failure.** A failed cached matmul erases the entry from the thread's own map, which destroys its descriptors; no other thread holds them.
- **Heuristics stay shared.** `algo_cache` keeps heuristic results process-wide by value, so a thread's first use of a geometry costs descriptor creation only, not `AlgoGetHeuristic`.

### 3.4 Debug logging

`MLX_ROCM_GEMM_DEBUG=1` now prints each heuristic miss as `[hipBLASLt algo] MNK=... batch=... algos=... workspace=...`, so a host can tell whether the per-stream workspace path runs for its shapes. This is how the workspace-0 result in section 4.3 was read.

### 3.5 Rejected

- **Per-thread handles** (upstream CUDA at the pin, `mlx/backend/cuda/cublas_utils.cpp`): `algo_cache` and the pipe cache hold heuristic results computed on one handle and shared across threads, and the async MoE path would create handles inside a HIP host callback.
- **A lease across `hipblasLtMatmul`**: the call returns before the kernel runs, so a lease cannot keep two streams' kernels off one buffer. Making it safe needs a stream synchronize (illegal under capture) or event handoff between unrelated streams, either of which serializes GEMMs.
- **A thread-local workspace**: the HIP callback thread runs async MoE GEMMs for every stream.
- **Per-call allocator temporaries** as upstream does: `hipblaslt_gemm_raw`, `hipblaslt_gemm_fp8_raw` and `hipblaslt_gemm_rowmajor_on_stream` receive a bare `hipStream_t`, not a `CommandEncoder`, and a per-call `hipMalloc` is not capturable.

## 4. Verification

### 4.1 The test

`tests/rocm_hipblaslt_concurrency.rs` (`#![cfg(feature = "rocm")]`) runs `hipblaslt_gemms_from_many_threads_in_fresh_processes`, which spawns the `#[ignore]` child 10 times in fresh processes with `MLX_NO_HIPBLASLT` and `MLX_HIPBLASLT_NO_PIPE_CACHE` removed from the environment. Each child starts 8 threads on their own streams behind a barrier, before any GPU matmul has run in the process. Each thread runs 33 rounds of a bf16 TN `[64, 64] x transpose([64, 64])`, a bf16 batched `[4, 64, 64] x [4, 64, 64]` and an f32 rocBLAS `[64, 128] x [128, 64]`, all threads on the same shapes so they contend on the same pipe keys. Every output is checked exactly against a host `i32` reference, and the parent requires the `[hipBLASLt caps] device` line exactly once per child, which proves the bf16 GEMMs reached hipBLASLt.

### 4.2 Deviations from the issue

**Deviation 1: thread-local pipes, not `shared_ptr`.** The issue proposed `std::shared_ptr<GemmPipe>` in `g_pipe_cache`. It was implemented first and failed. On gfx1151 (hipBLASLt 1.4), 8 threads enqueuing the same geometry, five child processes per variant:

| Variant | Children passed |
|---|---:|
| Issue's design: pipe shared through `shared_ptr` | 0 of 5 |
| Descriptors shared, algorithm copied out | 0 of 5 |
| Pipe cache bypassed (`MLX_HIPBLASLT_NO_PIPE_CACHE=1`, per-call descriptors on the same handle) | 5 of 5 |
| rocBLAS only (`MLX_NO_HIPBLASLT=1`) | 5 of 5 |
| Thread-local pipes | 5 of 5, then 200 of 200 |

The failing variants showed heap corruption (`malloc(): unaligned tcache chunk detected`), double frees, `SIGSEGV` and `HSA_STATUS_ERROR_MEMORY_FAULT`. The passing ones share the handle and `algo_cache` but never one descriptor object. The conclusion is that `hipblasLtMatmul` mutates the descriptor objects it is given, so a pipe cannot serve two threads at once, with or without shared ownership. The hipBLASLt source was not inspected, so the PR does not name the field it writes. The issue's acceptance item for `shared_ptr` is marked superseded by this measurement.

**Deviation 2: bf16 test inputs in `-1..=1`.** The issue and the first test draft used `-2..=2`. Building the exact check exposed #2206: on gfx1151, a bf16 GEMM returns residues of about 2^-22 at exact-zero outputs when the bf16 inputs mix zeros with magnitude 2. It is single-threaded and appears in both hipBLASLt and rocBLAS; `{-1, 0, 1}` and `{-2, -1, 1, 2}` are exact. bf16 output rounding hides it at non-zero outputs, so only an exact check sees it. The test uses `-1..=1` to stay exact until #2206 is resolved.

### 4.3 Results

On gfx1151 (ROCm 7.15), 20 runs of the test per build, 10 child processes each:

| Build | Runs failed | Child processes failed |
|---|---:|---:|
| `hipblaslt_gemm.cpp` restored to main (fix removed) | 20 of 20 | 50 of 200 |
| This PR | 0 of 20 | 0 of 200 |

Without the fix, the 50 dead children were 28 `SIGABRT` (tcache and double-free aborts), 19 `SIGSEGV`, and 3 with an unwritten output or `HSA_STATUS_ERROR_MEMORY_FAULT`. Twelve runs also had a child hang past its 120 s budget, one thread spinning on a GPU completion that never arrives. `tests/rocm_rocblas_handle_concurrency.rs` no longer sets `MLX_NO_HIPBLASLT` and passes.

**Workspace path not exercised.** The heuristics returned `workspace=0` for both bf16 test shapes (`MNK=64,64,64`, TN batch 1 and NN batch 4, 8 algorithms each), and the issue's own probe found 0 for all candidates of 15 shapes. So on gfx1151 the per-stream workspace path does not run, and the test guards it only by construction. What the test reaches there is the shared pipe descriptors and the init window. Architectures whose libraries pick split-K algorithms with global scratch (CDNA, gfx1201) were not measured.

**Decode, no change claimed.** Meta-Llama-3.1-8B-Instruct-4bit, 512-token prompt, 128 generated, both release binaries run back to back in one `scripts/rocm_gpu_guard.sh` session:

| Build | tok/s | Median |
|---|---|---:|
| Before (main's `hipblaslt_gemm.cpp`) | 37.96, 37.97, 38.02 | 37.97 |
| After | 37.89, 37.91, 37.92 | 37.91 |

The medians are 0.2% apart, and a separate guarded session read 35.63 tok/s for the same after binary, so the session-to-session spread is about 6%. Decode of a 4-bit model does not reach hipBLASLt at M = 1, so no change was expected.

### 4.4 Gates

- `cargo clippy --features rocm --test rocm_hipblaslt_concurrency --test rocm_rocblas_handle_concurrency -- -D warnings`: clean. `grep -n "ensure_workspace\|GemmPipe\*\|initialized" hipblaslt_gemm.cpp`: empty.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: pass.
- Orchestrator's `make verify-rocm` on `df8bc628` (rebased on main `5976af8d`): 158 suites, 12,104 passed, 1 failed, 386 ignored. The failure is `server::routes::props::props_tests::geometry_block_reports_the_resolved_batch_size_alias`. It fails deterministically on main since #2205 (default prefill chunk changed to 2048), fails identically when run alone, is unrelated to this PR and is being fixed separately.
- The unit's own gate on `ec067c6e` (before the rebase): 12,051 passed, 0 failed.

Not verified: Metal and CUDA (not available on this host; the change touches only files under `patches-rocm/` and `rocm`-gated tests).

## 5. LOCAL_FIXES Item 38

The entry goes under "Fixes to the fork's kernels". It records the shared state and how each piece is now owned, the three measured pipe variants, the rejected alternatives, the debug logging, and the test with its 20-of-20 count, and ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2200)." Nothing is added under `docs/mlxcelverse/upstream/`.

## 6. Technical Decisions

- **Measure before sharing a pipe.** The issue's design was reasonable on paper and failed 0 of 5. The fix follows the measurement.
- **Thread-local over locked.** Holding `g_pipe_mutex` across `hipblasLtMatmul` would serialize host enqueue of every hipBLASLt GEMM in the process, including the MoE segment loop. A per-thread map has no lock on the fast path, and the shared `algo_cache` keeps the first-use cost to descriptor creation.
- **Insert a pipe only after success.** It fixes the stale-algorithm retry as a side effect and makes a half-built pipe unobservable.
- **One buffer per stream at the cap.** It never grows, so there is no reallocation to race, and it is reused in stream order, which is the only ordering scratch memory needs.
- **Throw on a capturing stream without a buffer.** `hipMalloc` into a capture invalidates it. The only capture-capable hipBLASLt call is the fp8 path, which requires M >= 64 and is not reached from decode capture.
- **Failed create stays permanent.** Route decisions call it on every GEMM.

## 7. Follow-ups

- **#2206, bf16 residue at exact zeros.** Name the Tensile kernel, reproduce outside MLX, then fix or document a bound. Until then exact bf16 tests keep inputs in `-1..=1`.
- **#2197, the `rocm::device()` map.** Next in the series: an unlocked `find`/`try_emplace` on a process-global map, latent and multi-GPU only.
- **Leaked alpha/beta pairs.** The `new float[2]` pairs in the GEMM paths are never freed. Out of scope here.
- **GPU guard queueing.** The guarded decode session can queue for hours behind the host-wide lock when other units are compiling.

## 8. Residual Risks

- **Per-stream workspace untested at runtime.** No algorithm requested workspace on gfx1151. A bug in that path would show first on CDNA or gfx1201.
- **Cause of the descriptor mutation unnamed.** The thread-local design rests on measurement, not on the library source. A hipBLASLt version that changes this would not break it, but nothing here confirms which field is written.
- **Memory.** Each thread keeps its own pipe map and each workspace-needing stream holds 32 MB. Both are bounded by the number of worker threads and streams.
- **Exact-bf16 coverage is narrower than intended** until #2206 lands.
- **One host.** All runs are on one gfx1151.

## 9. Learning Points

- **Test the fix's premise.** The issue's shared ownership model looked right and was wrong, and only a five-way experiment showed why. Bypass, rocBLAS-only and per-thread variants separate "shared state" from "shared descriptor objects".
- **Ownership is not the same as safety.** `shared_ptr` kept the descriptors alive and they still corrupted the heap, because the callee mutates them.
- **Read the heuristics before reasoning about workspace.** Workspace 0 on this host turned a scary race into one the test cannot reach, and the report has to say so.
- **A fresh-process test turns a once-per-process window into a count.** Ten children per run, 20 runs.
- **An exact check finds what tolerance hides.** It is how #2206 was found.
