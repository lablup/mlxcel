# Technical Report: PR #2196 - Lock the ROCm JIT Module Cache and the HIP Event Pool

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `c8bf6cf3` on origin/main `84d3a7bc`, PR open, pending merge. Closes #2183 (part of #1801).

**Languages**: C++ and HIP (ROCm overlay `jit_module.cpp`, `jit_module.h`, `event.hip`; bridge `mlx_cxx_bridge.cpp`/`.h`), Rust (`src/lib/mlxcel-core/src/lib.rs`, `rocm_faults.rs`, new `tests/rocm_jit_module_concurrency.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Low to medium. The change adds locks to two process-globals on the ROCm launch path and does not change any kernel, key or result. The cost is serialization: hiprtc compiles now run one at a time and block other threads' cache lookups while they run, and every `HipEvent` construction and destruction takes a mutex. That cost was not measured. Metal and CUDA code paths are untouched; the bridge fixture compiles to a throwing stub off ROCm.

## Executive Summary

Issue #2183 reported that `rocm::get_jit_module` memoised compiled modules in an unlocked process-global `std::unordered_map`: an unlocked `find` followed by a `try_emplace` whose `JitModule` constructor compiles through hiprtc for tens to hundreds of milliseconds. The PR copies upstream CUDA's locking at the pin: a leaked map and a leaked `std::shared_mutex`, lookup under a shared lock, and on a miss a unique lock, a second `find`, and `try_emplace` only if the key is still missing. The unlocked `get_jit_module_cache()` accessor is removed, and `JitModule::get_kernel` sets the per-kernel configured flag under a new `kernels_mtx_`.

The new concurrency tests then exposed a second unlocked process-global on the same launch path: `HipEventPool` in `event.hip`. With the JIT cache locked and the event pool as on main, the test file failed in 20 of 20 runs (10 `invalid resource handle`, 8 SIGABRT, 2 SIGSEGV). A throwaway probe isolated it: 8 threads launching an already-compiled kernel on their own streams failed, and 1 thread doing it 40 times passed. Since every `HipEvent` takes a handle from the pool, this is a crash reachable in production for multi-threaded ROCm serving, more exposed than the JIT race the issue was filed for. The PR locks the pool with a `std::mutex`.

The tests are honest about what they prove. With the JIT lock removed (event pool locked), 0 of 20 runs failed, so for the JIT cache the tests are a crash guard, not a deterministic reproduction. For the event pool they reproduce the failure every run. The orchestrator's `make verify-rocm` on `c8bf6cf3` passed every step: 12,006 passed, 0 failed, 384 ignored, smoke OK.

## 1. Problem Statement

### 1.1 The JIT cache race (#2183)

On main at the time, `get_jit_module_cache()` returned a function-static `std::unordered_map<std::string, JitModule>`, and `get_jit_module` did `map.find(key)` and then `map.try_emplace(key, device(mlx_device), name, builder, cache)` with no lock. Custom kernels pass `use_disk_cache = false`, so every `fast::hip_kernel` compiles on its first launch in each process. Two failure shapes follow:

- Two threads miss the same key, both compile, both try to insert.
- An insert that grows the table rehashes it while another thread is walking buckets inside `find`, which is undefined behavior and can crash or return a dangling reference.

Every launch of a `Compiled` primitive, a `fast::hip_kernel` port and `compute_dynamic_offset` goes through `get_jit_module`, including cache hits. The server evaluates on several OS threads, each with its own thread-local stream and no global eval lock: the `BatchScheduler`, `EmbeddingWorker`, `RerankWorker` and `AudioWorker`. `-m <chat> --embedding-model <emb>` serves two models at once, so an embedding request can insert a module while the chat scheduler is inside `find`. New keys also appear late in a process's life whenever a new template-argument combination is first used.

`JitModule::get_kernel` also wrote the configured flag in `kernels_` unlocked. No ROCm caller passes `configure_kernel`, so that race was benign, but the CUDA overlay had already crashed on the equivalent at libtest's default thread count (#1566).

### 1.2 The event pool race, found by the new test

`HipEventPool` recycles `hipEvent_t` handles through a function-static `std::map<std::pair<int,int>, std::vector<HipEventHandle>>`. Every `HipEvent` takes a handle from it on construction and returns one on destruction, on whichever thread evaluates, waits or runs a completion handler. There was no lock. Concurrent `push_back` and `pop_back` on the same vector can hand one handle to two events or hand a destroyed handle to a record, which surfaces as `hipEventRecord(event_, stream) failed: invalid resource handle`, or as a heap corruption abort or segfault.

This was not in the issue. It appeared when the first version of the PR, with only the JIT lock, ran its own tests: they failed within a few rounds. The PR then spent a throwaway probe to separate the two causes before widening scope (section 4).

## 2. Change Summary

| Area | Change |
|---|---|
| `patches-rocm/mlx/backend/rocm/jit_module.cpp` | `get_jit_module` holds a leaked `static auto* modules` map and `static auto* mtx = new std::shared_mutex`; lookup under `std::shared_lock`, miss path under `std::unique_lock` with a second `find` before `try_emplace`. `get_jit_module_cache()` removed. `get_kernel` takes `kernels_mtx_` around the configured-flag check and set. `#include <shared_mutex>`. |
| `patches-rocm/mlx/backend/rocm/jit_module.h` | `get_jit_module_cache()` declaration removed; `std::mutex kernels_mtx_` added to `JitModule`. |
| `patches-rocm/mlx/backend/rocm/event.hip` | `HipEventPool::create` and `release` hold a `std::mutex` around the cache; `hipEventCreateWithFlags` runs outside the lock on a cache miss. `cache_for` made private with a "callers hold mutex()" contract. |
| Bridge (`mlx_cxx_bridge.cpp`/`.h`, `lib.rs`, `rocm_faults.rs`) | Test-only fixture `rocm_jit_race_probe_array(input, variant)` and wrapper `mlxcel_core::rocm_faults::jit_race_probe_array`: kernel `mlxcel_jit_race_probe_v<variant>`, `out[i] = inp[i] * 2 + 1`, one block of `n <= 1024` threads, throws off ROCm. |
| `tests/rocm_jit_module_concurrency.rs` (new) | `same_kernel_first_launch_from_many_threads` and `distinct_kernels_first_launch_while_others_hit_cache`, 8 threads on their own thread-local streams behind a barrier, 16 rounds, exact outputs. |
| `patches-rocm/LOCAL_FIXES.md` | Items 35 (JIT cache) and 36 (event pool). |

Two commits: the fix with tests and LOCAL_FIXES (`6b5a4f7c`), and a text correction of item 35 found in review (`c8bf6cf3`): the map is named `modules`, not `cache`, and the entry now states that a compile under the unique lock also holds up other threads' cache hits. 9 files, 378 insertions, 18 deletions.

## 3. Design

### 3.1 The JIT cache: upstream's pattern, not a new one

The issue asked for upstream's locking and nothing new, and the PR follows it line for line (`mlx/backend/cuda/jit_module.cpp` at pin 81ba1c6a, and the CUDA overlay):

- **Double-checked lookup.** The common case, a hit, takes only a shared lock, so concurrent launches of compiled kernels do not serialize against each other. A miss takes the unique lock and repeats the `find`, because another thread may have inserted the key between the two locks. Only one `JitModule` is constructed per key, and every caller gets a reference to the same one.
- **Leaked storage.** `static auto* modules = new std::unordered_map<...>` is never destroyed. Upstream's reason is that user code may still run JIT code after the main thread tears down statics. On this fork it has a second benefit: no `hipModuleUnload` runs at static teardown, where on a faulted device every HIP call fails (LOCAL_FIXES item 7). References returned from the map stay valid for the process, which matters because callers hold `JitModule&` after the lock is released; `unordered_map` rehash does not move nodes, so that reference stability holds under concurrent inserts.
- **Removing `get_jit_module_cache()`.** It was exported in the header and called by nothing. Leaving it would leave a way to reach the map without the lock, so it is gone rather than wrapped.
- **Error handling unchanged.** A compile or load failure throws out of the constructor before anything is inserted; unwinding releases the unique lock, and the next caller retries.
- **The configured flag.** `kernels_` is filled once in the constructor (under the write lock) and never restructured, so `kernels_.find` stays lock-free. Only the check-and-set of the configured bit takes `kernels_mtx_`, as the CUDA overlay does since #1566. The lock is held across `configure_kernel`, so a future caller's callback must not re-enter `get_kernel`; the code comment says so.

A side effect: serialized compiles stop two `StderrSuppressor` instances from swapping fd 2 under each other.

### 3.2 Rejected for the JIT cache: per-key locking

A per-key `std::once_flag` or future would let compiles of different keys run in parallel, and would let cache hits proceed while an unrelated key compiles. The issue rejected it because it diverges from upstream and a hiprtc compile happens once per key per process. The cost of that choice is now written into item 35: while one thread compiles under the unique lock, every other thread's `get_jit_module` call, hits included, waits. In a server, the first `fast::hip_kernel` launch of a new template combination can stall the other workers' launches for one compile. That stall was not measured.

### 3.3 The event pool: a mutex, not upstream's first-thread rule

Upstream CUDA's `CudaEventPool` (`mlx/backend/cuda/event.cu` at the pin) avoids the same race differently: it caches events only on the thread that first used the pool, and every other thread creates and destroys a fresh event each time. The PR did not copy that, and this is the one place it departs from upstream:

- In mlxcel's server, decode runs on worker threads (the batch scheduler and the per-model workers), not on the thread that first touched the pool. Under the first-thread rule, the hot path would create and destroy a `hipEvent_t` for every event, so pooling would effectively be off exactly where it matters.
- A `std::mutex` around `cache_for(...)` push and pop keeps pooling on every thread. The critical section is a map lookup and a vector push or pop. `hipEventCreateWithFlags` on a miss runs after the lock is released, so a slow driver call never holds other threads up.
- With one evaluating thread, the lock is uncontended.

What was not done: no benchmark compares the mutex against the first-thread rule, or either against main. The argument for the mutex is structural (keep the cache where decode runs), not measured. If a profile ever shows contention on the pool mutex, a per-thread cache with a shared fallback is the next step.

## 4. Verification

### 4.1 The 20-run table

On gfx1151, `tests/rocm_jit_module_concurrency.rs` was run 20 times per build:

| Build | Runs failed | Failure kinds |
|---|---:|---|
| Both locks (this PR) | 0 of 20 | none |
| JIT cache lock removed, event pool locked | 0 of 20 | none |
| JIT cache locked, `event.hip` from main | 20 of 20 | 10 test failures with `invalid resource handle`, 8 SIGABRT, 2 SIGSEGV |

How to read it:

- **For the event pool, the test is a reproduction.** Every run on the unlocked pool failed, within a few rounds.
- **For the JIT cache, the test is a crash guard, not a reproduction.** Removing the JIT lock did not cause a single failure on this host. The race is real by inspection (an unlocked `find` against a rehashing `try_emplace` is undefined behavior), and the tests exercise exactly the interleavings the issue described, but they did not catch it. A regression in the JIT lock could pass these tests. The issue's acceptance criteria required this to be stated if it happened, and the PR states it.

### 4.2 Isolating the event pool

Before widening scope, a throwaway probe that only launched an already-compiled kernel (no JIT miss, no insert) was run two ways:

- 8 threads, each on its own stream: failed with the same `invalid resource handle`.
- 1 thread, 40 launches: passed.

That removes the JIT cache from the picture: the failure needs concurrency but not compilation, and the only shared mutable state on that path is the event pool. With the pool locked, the 8-thread cache-hit probe passed 32 rounds.

### 4.3 Gates

- `cargo test --profile test-fast --features rocm --test rocm_jit_module_concurrency --test rocm_custom_kernel_jit_key --test rocm_gpu_faults`: 7 passed.
- `cargo clippy` with `-D warnings` on the new test and the mlxcel-core lib: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` and `cargo test --test dead_doc_pointers`: pass.
- Unit's `make verify-rocm` on `6b5a4f7c` (rebased on `84d3a7bc`): 154 test suites, 12,006 passed, 0 failed, 384 ignored. `c8bf6cf3` changes only LOCAL_FIXES text; `verify-rocm-overlay` passes on it.
- Orchestrator's `make verify-rocm` on `c8bf6cf3` (up to date with origin/main `84d3a7bc`): every step passed, 12,006 passed, 0 failed, 384 ignored, smoke OK.

Not verified: Metal and CUDA (not available on this host; no code on those paths changed), and the performance cost of either lock.

## 5. LOCAL_FIXES Items 35 and 36

Both entries go under "Fixes to the fork's kernels" in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`.

- **35, JIT module cache locked.** Records the unlocked `find`/`try_emplace`, the launch sites and server threads that reach it, the upstream locking now used, the unchanged `"<device index>:<name>"` key (item 33 supplies the name), the retry-on-throw behavior, the serialization cost including stalled cache hits during a compile, the teardown benefit relative to item 7, the removed accessor, the `kernels_mtx_` fill, and the test.
- **36, HIP event pool locked.** Records the unlocked pool, the 20-of-20 failure with the JIT cache locked, the 8-thread versus 1-thread isolation, the mutex with creation outside it, why upstream's first-thread rule was not used, and the 32-round pass.

Both end with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there." Nothing is added under `docs/mlxcelverse/upstream/`. Item 35 restores parity with upstream, so there is nothing to propose; item 36 is a fork-specific choice over upstream's design.

## 6. Technical Decisions

- **Copy upstream's JIT locking exactly.** Upstream already solved this on CUDA; matching it keeps the overlay diffable against the pin and avoids designing new concurrency in a fork.
- **Remove the unlocked accessor, do not wrap it.** An exported reference to a locked map is a lock bypass waiting for a caller.
- **Leak the cache.** Keeps references valid after teardown starts and avoids HIP calls on a faulted device at exit.
- **Widen scope to the event pool.** The new tests could not pass without it, and it is the more exposed crash: it needs no JIT miss, only two threads evaluating at once. Splitting it into a separate PR would have left a test file that fails every run.
- **Mutex over first-thread caching for the pool.** Keeps pooling on the server's worker threads where decode runs. Unmeasured; justified by where the hot path runs.
- **Report the crash-guard limitation instead of tuning the test until it fails.** A test that reproduces the JIT race on this host would need timing hooks inside the overlay; the issue asked for an honest count instead.

## 7. Follow-ups Filed From This Work

Each was found while working on or reviewing this PR and is pre-existing on main:

- **#2197, the `rocm::device()` map.** `device()` does an unlocked `find`/`try_emplace` on a process-global `std::unordered_map<int, Device>`, read on every launch. Latent and multi-GPU only: every insert comes from stream creation under the `all_streams()` lock, and on one GPU the only key is inserted before any thread holds a stream. It becomes reachable once a stream is placed on GPU 1 (#486, #488). The issue also covers the event pool's map being a destructible static, so a `HipEvent` released during static teardown pushes into a destroyed map; leaking it as the JIT cache is leaked would fix that.
- **#2198, rocBLAS handle init and stream rebind.** `Device::get_rocblas_handle()` sets its initialized flag before creating the handle, so a second thread can get a null handle; and every rocBLAS GEMM calls `set_rocblas_stream` on one shared handle unlocked, so thread B's rebind between thread A's rebind and A's GEMM puts A's GEMM on B's stream. Reachable on one GPU with two models served at once; the rebind race is open on every rocBLAS GEMM, not just the first.
- **#2200, hipBLASLt shared state.** One 32 MB workspace per device is shared by every stream (two concurrent GEMMs needing workspace corrupt each other's output), `ensure_workspace` frees and reallocates it unlocked, the init path publishes `initialized` before the handle and workspace exist, and a failed cached-pipe matmul destroys and erases a `GemmPipe` another thread may still be passing to `hipblasLtMatmul`. hipBLASLt is the default route for bf16 and fp16 GEMMs, so this is reachable on one GPU.

A smaller note from review: a moved-from `HipEvent` would return a null handle to the pool. Nothing moves one today.

## 8. Residual Risks

- **The JIT lock has no failing test.** If someone weakens it, these tests will probably still pass on gfx1151.
- **Lock cost unmeasured.** The shared lock on every `get_jit_module`, the stall of hits during a compile, and the pool mutex on every event are all unbenchmarked. They are expected to be small next to a GEMM, but that is an expectation.
- **More unlocked globals remain.** #2197, #2198 and #2200 are open. Until they land, multi-model ROCm serving can still race on rocBLAS and hipBLASLt state.
- **One host.** All runs are on one gfx1151; timing-dependent races may behave differently on other ROCm targets.

## 9. Learning Points

- **The multi-model server shares one `Device` across worker threads on separate streams, and this had never been exercised concurrently on ROCm.** Every process-global and every per-`Device` lazy field on the launch path was written assuming one submitting thread. The first test that actually ran 8 threads at once found a second race within a few rounds.
- **Each lock fix exposes the next unlocked global.** Locking the JIT cache moved the failure to the event pool; the review of this PR found the device map (#2197); the review of #2197 found rocBLAS (#2198); the review of #2198 found hipBLASLt (#2200). When one global on a hot path is unlocked, assume its neighbors are too and audit the whole path, rather than fixing the reported one.
- **Isolate before widening scope.** The 8-thread versus 1-thread probe on an already-compiled kernel took the JIT cache out of the picture with two runs, which made the scope change defensible in review.
- **Count failures with and without each fix separately.** The 20-run table shows which lock the test actually guards. Without the "JIT lock removed" row, the passing test would have looked like proof of the JIT fix.
- **Departing from upstream needs a reason tied to this codebase.** Upstream's first-thread event caching fits a single-submitter model; mlxcel's worker-thread decode does not, which is the whole argument for the mutex.
