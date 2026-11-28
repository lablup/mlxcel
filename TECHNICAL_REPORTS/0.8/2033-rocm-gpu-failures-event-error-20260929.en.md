# Technical Report: PR #2033 - Surface ROCm GPU Failures Through Event::error

**Date**: 2026-09-29

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (HIP), Rust

**Risk Level**: Medium (touches every ROCm wait and adds a check after every primitive launch; overlay-only, so Metal and CUDA are unchanged by construction but were not run)

## Executive Summary

Issue #1804 (phase 1 of epic #1801) asked for ROCm GPU failures to reach MLX's `Event::error` (ml-explore/mlx#3742) so that mlxcel reports them as errors. Before this PR the ROCm overlay had storage for the error and nothing ever wrote it. A rejected kernel launch left its output unwritten and the eval returned `Ok`; an asynchronous device fault made the waiting eval spin forever. The bridge's `stashed_launch_state` read both as `Landed`.

The PR ports the Metal model to the ROCm overlay. Each `CommandEncoder` owns an `Error`. Every host wait and query now returns a HIP status instead of assuming success, and a wait or query that finds its stream failed records the status on the encoder and attaches that `Error` to the event. `Event::wait()` throws it through `Event::check_error()`, and `Event::is_signaled()` reports a failed event as signaled with its error, so `array::is_available()` throws instead of leaving the array pending. Synchronous launch failures are caught by a per-primitive read of the thread's pending HIP error in `gpu::eval`. On gfx1151 an out-of-bounds write now fails the waiting eval in 195 ms, and later evals fail in 0 ms instead of hanging.

## 1. Problem Statement

The contract from ml-explore/mlx#3742 is that a failed GPU step surfaces as an exception carried by the event. mlxcel's bridge relies on it: `drain_pending_verification()` reads `stashed_launch_state()`, which reports `Failed` only when the signaled event carries an error pointer. On ROCm nothing set that pointer, so a failed launch read as `Landed` and its outputs were consumed as if written.

Measurement on gfx1151 with ROCm 7.15 (a standalone probe kernel, recorded in `patches-rocm/LOCAL_FIXES.md` item 7) showed that the backend's three waits were built on assumptions that do not hold after a queue fault:

- **The process is not aborted.** With or without `HIP_SKIP_ABORT_ON_GPU_ERROR`, the runtime keeps the process alive, so no crash stops the spin.
- **Every HIP call fails afterwards.** Every call on every thread, `hipDeviceReset` included, returns `hipErrorIllegalAddress`. `HipEvent::wait` looped "until `hipEventQuery` returns `hipSuccess`", which it never does again.
- **No queued host callback runs.** The backend signals `AtomicEvent` and runs completion handlers through `hipLaunchHostFunc`. A waiter on the counter, or `CommandEncoder::synchronize()` waiting on its promise, blocks forever.

Synchronous launch failures had a separate gap. About 60 launch sites use `hipLaunchKernelGGL` or `<<<>>>`, which return nothing; a rejected launch (oversized block, no code object for the gfx) only sets the thread's pending HIP error. `launch_module_kernel` discarded the `hipModuleLaunchKernel` status with `(void)`. The result was an eval that succeeded on an unwritten buffer, which is where the NaN outputs in the issue came from.

The issue's natural reproducers (the mxfp4 hang, the mxfp8 NaN, the `arg_reduce` grid overflow) had all been fixed at their source by PR #1818, so the fix needed a deliberately faulting test.

## 2. Change Summary

- `patches-rocm/mlx/backend/rocm/device.{h,cpp}`: `CommandEncoder` gains an `Error error_`, `set_device_error` (keeps the earliest error) and `check_launch(primitive)`. Free functions `describe_device_error` (names the HIP status and, if `hipGetDevice` also fails, says the context is gone and the process must restart), `record_stream_error` (finds the stream's encoder through a new non-binding `Device::find_encoder`, or a leaked process-wide fallback `Error` for CPU streams) and `gpu_watchdog_seconds`. `synchronize()` checks every HIP status, polls `hipStreamQuery` about once a millisecond while waiting for the handler promise, and throws the stream's error. `commit()` and the decode capture paths record a failed `Worker::commit`. Destructor-path graph destroys are explicitly `(void)`. A launch that throws inside `MLX_GRAPH_PREFILL_REPLAY=1` stream capture now ends the capture and falls back to the eager launch, which rethrows outside capture.
- `patches-rocm/mlx/backend/rocm/eval.cpp`: after each `eval_gpu`, `encoder.check_launch(name)` reads the thread's pending HIP error and throws it as that primitive's failure; if `eval_gpu` itself throws, the pending error is cleared first so the next primitive is not blamed.
- `patches-rocm/mlx/backend/rocm/event.{h,hip}`: `HipEvent::wait` and `AtomicEvent::wait` return `[[nodiscard]] hipError_t`; `HipEvent::completed()` became `query()`. The `HipEvent` loop stops on anything other than `hipErrorNotReady`. `AtomicEvent` moves its counter into a shared `State` that also records the GPU stream it was last signaled from and any status that stopped the callback being queued; `status()` reports that status or polls the stream. `Event::wait`, `Event::wait(Stream)`, `Event::signal(Stream)` and `Event::is_signaled` attach the stream's `Error` through `poison_event`/`fail_event`. CPU-stream waits and signals go through `scheduler::wait_event`/`scheduler::signal_event`, as on Metal and CUDA.
- `patches-rocm/mlx/backend/rocm/worker.{h,cpp}`: `Worker::commit` returns the `hipLaunchHostFunc` status.
- `patches-rocm/mlx/backend/rocm/fence.cpp`: `Fence::wait` throws when the event wait reports a failure.
- `patches-rocm/mlx/backend/rocm/jit_module.h`: `hipModuleLaunchKernel` is wrapped in `CHECK_HIP_ERROR`.
- `patches-rocm/mlx/backend/rocm/utils.h`: `HipHandle::reset` reports a failed destroy once per handle type on stderr instead of throwing from a destructor.
- `mlxcel-core`: a test-only bridge fixture `rocm_fault_probe_array(kind)` (a throwing stub off ROCm) and its typed wrapper `mlxcel_core::rocm_faults`, marked `#[doc(hidden)]`. The comment above `stashed_launch_state` now says ROCm attaches the error.
- `tests/rocm_gpu_faults.rs`: two tests, one for each failure class; the queue-fault test runs in a child process.
- Docs: `LOCAL_FIXES.md` item 7, the "GPU faults" row in `docs/installation.md`, and `MLX_ROCM_GPU_WATCHDOG_SECS` in `docs/environment-variables.md`.

## 3. Technical Decisions

### Follow the Metal model: one Error per stream, not per event

Metal's command buffer completion handler stores the encoder's error on every event it signals. CUDA at the pinned upstream attaches none, so it offers no model. The ROCm port copies Metal: the `Error` lives on the `CommandEncoder`, and any event signaled on a failed stream points at it. `Event::set_error` stores a raw pointer, so the pointee must outlive every event. Encoders live for the process (their `Device` keeps them), and the CPU-stream fallback is a deliberately leaked static. A per-event error object would have needed its own lifetime management for no gain, since after a queue fault every event on the device fails for the same reason.

### Waits return a status; the Event layer throws

The low-level waits (`HipEvent::wait`, `AtomicEvent::wait`) return `hipError_t` marked `[[nodiscard]]`, and only the `Event` layer converts a failure into an exception, through `poison_event` then `Event::check_error()`. Throwing through `check_error` is what the other backends do, and it consumes the message the same way, so a later wait on the same event does not report it twice. `[[nodiscard]]` makes the compiler flag any caller that ignores a status, which is how the old code lost them. `record_stream_error` and `poison_event` make no HIP call that can fail and do not bind the device, so they are safe on a dead device and from `const` queries.

### A failed event reports as signaled

`is_signaled()` returns `true` with the error attached when the signaling stream has failed. This is the only way to reach the two readers that never wait: `array::is_available()`, which detaches and throws an event's error once it is signaled, and the bridge's `stashed_launch_state()`, which checks the error pointer of a signaled event. Returning `false` would leave both waiting on a signal that cannot arrive. `AtomicEvent` rechecks the counter after reading a failure status, because the callback can land the value between the first check and the status query; a landed value counts as a completed wait regardless of what the stream reports afterwards.

### Catch synchronous launch failures once per primitive

Converting 60 unchecked `hipLaunchKernelGGL`/`<<<>>>` launch sites would have been a large, easily regressed diff. HIP already records a rejected launch as the thread's pending error, so reading it once after each `eval_gpu` in `gpu::eval`, on the launching thread, covers every site, including future ones. The measured cost is one thread-local `hipGetLastError`, about 17.5 ns per primitive. A probe confirmed that `hipErrorNotReady` from `hipEventQuery`/`hipStreamQuery` does not become the pending error, so pipelined decode does not trip the check.

Two refinements came from review. `hipErrorOutOfMemory` is skipped, because the allocator's cache-release retry, `unified_malloc`'s managed-memory fallback and hipBLASLt's no-workspace path all recover from a failed `hipMalloc` without clearing the pending error, and an unrecovered allocation already throws. Without the skip, evals that succeeded under memory pressure would have failed. And `check_launch` distinguishes a launch rejection from a dead device with a `hipGetDevice` call: a rejection throws a plain error and leaves the stream usable, while a dead context poisons the stream's `Error` so later waits fail fast.

### Poll the stream, but rarely

A faulted stream never runs the callback that lands `AtomicEvent`'s counter, so a waiter must ask the stream whether the signal can still arrive. The poll is rate-limited: the counter is read every spin, the clock every 64 spins, and `hipStreamQuery` (about 54 ns) at most once per millisecond. A wait that completes in under a millisecond, which is the normal decode case, makes no extra HIP call. `synchronize()` uses the same pattern, polling the handler future at 100 us and the stream every tenth poll. The design gives up instant detection for zero success-path cost; a fault is reported within about a second in practice (195 ms measured).

### Event state is owned by the callback payload

Previously the `AtomicEvent` counter was kept alive by `add_completed_handler([buf = mem_]{})`, a handler that a failed stream never runs. The counter, the signaling stream and the queue-failure status now live in a shared `State`, and the `hipLaunchHostFunc` payload holds a `shared_ptr` to it. Lifetime no longer depends on completion handlers, and `State` gives waiters on other threads a place to read the stream to poll and a failure that happened while queuing the callback.

### CPU streams go through the scheduler

`Event::wait(Stream)` and `Event::signal(Stream)` on a CPU stream now use `scheduler::wait_event` and `scheduler::signal_event`, so a CPU stream inherits a GPU event's error and a CPU stream that has already failed poisons the event it signals, as on Metal and CUDA. The first draft routed `Fence::update` on a CPU stream through a path that asserted a GPU encoder; `rocm_smoke.sh` caught it, and the second commit keeps CPU-stream fence signals on the scheduler path.

### The device is not recovered, and the watchdog is opt-in

After a queue fault HIP refuses every call, including `hipDeviceReset`, so in-process recovery is not possible. The PR reports the fault and says so in the error text instead of attempting a reset. `mlxcel-server`'s batch scheduler already maps a `try_eval` error to a per-request failure (`abort_sequence_with_error` in `decode_tick.rs` and `prefill.rs`), so no server change was needed: the faulting request fails, and later GPU requests fail with the same error until the process restarts. Shutdown may need a SIGKILL because HIP's teardown waits on the callbacks that never run.

A kernel that spins forever is indistinguishable from a slow one, so the issue's optional watchdog is off by default. It sits in the backend waits rather than the bridge drain the issue sketched, because that is where every wait path passes, and it is named `MLX_ROCM_GPU_WATCHDOG_SECS` like the overlay's other variables. Parsing uses `strtol` with full validation; `10s`, negative or overflowing values are ignored with one stderr warning rather than guessed at. It does not apply under `MLX_EVENT_BLOCKING`, whose blocking waits have no poll loop.

### The fault test runs in a child process

The probe fixture uses `fast::hip_kernel`, so the launch, wait and error attach are the production path; only the kernel bodies are contrived (a 2048-thread block, which HIP rejects, and a write 2^40 bytes past the buffer). Because a queue fault kills the device for the whole process, the out-of-bounds test re-executes the test binary as a child and parses its timing report. The child exits through `_exit` to skip HIP's teardown, which would otherwise wait for callbacks that never run. The fixture lives in the bridge behind `#[doc(hidden)]` and no production path calls it.

## 4. Validation

Author's runs, from the PR body (gfx1151, ROCm 7.15, `--features rocm`):

- `cargo test --features rocm --test rocm_gpu_faults`: 2 passed. The child reported the fault in 195 ms and the follow-up eval failed in 0 ms.
- Negative control: with the ten overlay files restored from `origin/main` and the fixture kept, both tests fail. The child hung for the full 120 s budget and was killed, and the oversized launch returned `Ok` on an unwritten buffer.
- After the review commits: `rocm_gpu_faults` 2 passed (also with `MLX_ROCM_GPU_WATCHDOG_SECS=10s`, which printed the invalid-value warning); `rocm_smoke.sh` OK (32 tokens, 258.8 tok/s); `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings` clean; `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` and `cargo test --features rocm --test dead_doc_pointers` pass.

Orchestrator verification (gfx1151, branch rebased onto origin/main `cdd531d5`, which includes PR #2030 for #1806):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in exactly three targets, none attributable to this PR:
  - `-p mlxcel-core --lib`: 1796 passed, 35 failed, 1 ignored. The 35 are the same as in the #2030 run: 34 fused paged-attention tests with no ROCm port (tracked by #1814), plus the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`.
  - `-p mlxcel --lib`: 8711 passed, 1 failed: `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`, off by one ULP on ROCm, a test added by #2037.
  - `-p mlxcel --bin mlxcel`: 248 passed, 1 failed: `family_order_is_exhaustive`, missing the `Speech` family added by #2037.
- Every other test binary passed, including `tests/rocm_gpu_faults.rs`.

With #1806 merged, the `mlxcel-core` lib tests run to completion on ROCm, so this is the first full-suite run of the `Event::error` change. That matters here because `check_launch` runs after every primitive and every wait now inspects a status: a false positive anywhere in the success path would have shown up as new failures across the suite, and the failure set is unchanged from #2030.

## 5. Learning Points

- **Measure the failure mode before designing the recovery.** The plan in the issue assumed a faulted queue could be marked failed and recovered from. The probe showed the device context is gone for the process, every HIP call fails, and no callback runs. That turned the goal from recovery into prompt, accurate reporting, and it explained why all three waits hung.
- **A loop that waits for success spins forever on a sticky error.** `while (query() != hipSuccess)` treats "not ready" and "failed" the same. Loop on the specific "not ready" status and return anything else.
- **Lifetime tied to completion handlers fails exactly when handlers do not run.** Keep-alive handlers are convenient on the success path and a leak or use-after-free risk on the failure path. Let the callback payload own what it touches.
- **One check at a choke point beats sixty at the call sites.** HIP's per-thread pending error is an existing channel for launch failures; reading it once per primitive in `gpu::eval` covers every present and future `hipLaunchKernelGGL` site at negligible cost, provided the statuses that recovery paths leave behind (`hipErrorOutOfMemory`) are excluded.
- **Test process-killing faults in a child.** A fault that poisons the device for the whole process cannot share a test binary with other GPU tests; re-executing the binary with a filter and `_exit` keeps the parent's device usable and the timing measurable.

## 6. What Is Not Verified

- **Metal and CUDA were not run.** The only file on their paths is `mlx_cxx_bridge.cpp`, where the fixture compiles to a throwing stub without `MLXCEL_BRIDGE_ROCM_BACKEND` and a comment changed. No behavior change is expected, but none was measured.
- **The out-of-memory skip was not exercised at runtime.** There is no way to force an out-of-memory-then-recover event through the bridge. The skip rests on a probe showing the pending error and on reading the allocator and hipBLASLt code.
- **Watchdog expiry was not exercised.** The tests cover parsing of an invalid value; no test drives a kernel that never finishes past a configured limit, so the `kWatchdogExpired` message and the stuck-stream behavior after it are untested.
- **The `MLX_EVENT_BLOCKING` path is not covered by the tests.** It relies on `hipEventSynchronize` returning the fault promptly.
- **The server path was checked by reading, not by a run.** No end-to-end request was sent to `mlxcel-server` while a fault was injected; the per-request failure mapping comes from the existing `abort_sequence_with_error` code.
- **No throughput baseline.** No pre-change binary existed in the worktree. The added success-path cost is estimated from micro-measurements (17.5 ns per primitive, 54 ns per unlanded `is_signaled` query on a GPU-signaled `AtomicEvent`), and the smoke run's 243 to 259 tok/s is not a controlled comparison.

## 7. Remaining Work

- Upstream the overlay change to the ROCm fork as part of #1813; `LOCAL_FIXES.md` item 7 records it.
- #1814: ROCm ports of the fused paged-attention kernels, which account for 34 of the 35 `mlxcel-core` failures.
- Triage the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` failure on ROCm.
- Fix the two failures introduced by #2037 on ROCm: the one-ULP `gelu_approx_matches_mlx_nn_bit_for_bit` mismatch and `family_order_is_exhaustive`, which does not list the `Speech` family.
- Optional: a test that exercises watchdog expiry with a kernel that spins, and an `mlxcel-server` run with an injected fault.
