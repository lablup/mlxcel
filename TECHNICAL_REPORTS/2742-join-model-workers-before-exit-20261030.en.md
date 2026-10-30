# PR #2742: Join model workers before the process exits

**Date**: 2026-10-30 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust
**Risk Level**: Low (shutdown paths only; error exits of `mlxcel-server` can now wait up to 120 s, or 150 s in router mode, for a model worker before the process ends)

## Executive Summary

The hosted `Pipeline Parallel CI / 2-host logical` job passed its test and then aborted with `pure virtual method called` while the test binary exited (issue #2738). The cause is a `ModelProvider` worker thread that nobody joined: it was still freeing its model's MLX arrays when `main` returned, and on the CPU backend the MLX allocator is a function-local C++ static whose destructor had already run. This PR adds one join primitive, `ModelProvider::shutdown_and_wait`, uses it in both real-model pipeline tests and in `InProcessServer::shutdown`, and routes every post-load exit of `mlxcel-server` (single-model and router mode) through a worker join. Local reproduction went from 44 aborts in 50 runs to 0 in 50, and the CI workflow passed ten consecutive dispatched runs on the runner image that produced every original abort.

## 1. Problem Statement

### The CI symptom

Since 2026-10-28 the job `Pipeline Parallel CI / 2-host logical (ubuntu-latest)` failed intermittently. The test `pipeline_multi_stage_two_host_logical_smoke` reported `ok`, libtest printed its summary, and about 40 ms later the process died with SIGABRT after `pure virtual method called` / `terminate called without an active exception`. The abort happened during process exit, not inside the test. It blocked every PR touching pipeline-parallel code.

The failures correlated with the GitHub runner image, not with a commit: all 10 aborts ran on `ubuntu24/20261025.353`, all 19 green runs on `ubuntu24/20261012.334`, and green and aborting runs shared the same base commit. The relevant image delta was the kernel (6.17 to 7.0); rustc, OpenBLAS and LAPACK were identical. The issue therefore treated it as a pre-existing exit-time race that new kernel scheduling exposed, and required the mechanism to be confirmed before any fix.

### Confirmed mechanism

Reproduced on a box with the same kernel (7.0.0) and the CI configuration (debug profile, no `cuda` feature, `MLXCEL_CI_PP_MODEL=qwen3-0.6b-4bit`). gdb with a breakpoint on `__cxa_pure_virtual` caught the abort on 3 of 3 attempts. The aborting thread was a model worker (`spawn_model_worker_with_batch_config` in `src/server/model_worker.rs`) dropping the scheduler's `PagedBlockPool`; the drop reached `MlxArray::~MlxArray`, then `mlx::core::allocator::Data::~Data`, which calls `allocator().free(buffer)`. At the same instant the main thread was in `exit` → `__run_exit_handlers` → `_dl_fini` (OpenBLAS's finalizer was on the stack).

Three facts make that an abort:

1. `impl Drop for ModelProvider` only sends `ModelRequest::Shutdown` and returns. The worker's `JoinHandle` is owned by the watcher thread created by `observe_worker_exit`, which gdb showed blocked in `pthread_join` on the worker. Nothing on the main thread waited for either.
2. Rust does not join spawned threads when `main` returns. The C runtime runs `exit()`, which runs static destructors while other threads keep executing.
3. On the CPU backend, `allocator()` returns `static CommonAllocator allocator_` in `mlx/backend/no_gpu/allocator.cpp`, a function-local static. Once its destructor has run, the object's vtable is the abstract `Allocator`'s and `free` is pure virtual. The Metal, CUDA and ROCm allocators are deliberately heap-leaked and never destroyed, which is why only the CPU CI build aborted.

The test helper `assert_multi_stage_coordinator_matches_dense_baseline` in `tests/pipeline_ci_multi_stage_real_models.rs` ended with `drop(remote_provider); drop(dense_provider);` and then joined only the stage service threads. The two provider workers were detached.

### The same pattern in production

`InProcessServer::shutdown` (used by `mlxcel run` and `chat` since #2224) and the router's model unload already waited on the worker's exit observer. Other process-owning paths did not, so the issue required a per-path verdict:

| Path | Verdict before this PR | After |
|---|---|---|
| `mlxcel-server` single-model (`start_server`) | Affected. A `serve_http` error (port taken, TLS failure) or any of the five side-model resolution errors after the chat worker started loading returned from `main` with the worker alive. | Fixed: every return after the provider exists goes through `join_model_worker_before_exit`. |
| Router mode (`run_router_server`) | Partly affected. Ctrl-C drained through `shutdown_all`, but the `serve_http` error arm returned without draining, so a bind failure after `load-on-startup` exited under loading workers. | Fixed: both arms drain; a Ctrl-C handler that fails to install also drains. |
| `--pp-peer` stage (`serve_remote_pipeline_stage`) | Not affected. `RemoteStageServiceHandle::shutdown` joins the stage thread, which owns the stage model. | Unchanged. |
| `mlxcel run` / `chat` | Not affected (#2224). | Unchanged; `InProcessServer::shutdown` now delegates to the shared primitive. |

SIGINT and SIGTERM in single-model mode still terminate by default signal disposition. No exit handlers run in that case, so the race does not occur there, and the PR leaves it unchanged.

## 2. Technical Decisions

### One join primitive on `ModelProvider`

`ModelProvider::shutdown_and_wait(timeout)` sends the shutdown signal, waits on the existing `WorkerExitObserver`, and returns:

- `Ok(())` when the worker exited, including when it had already exited before the call and when the request channel is closed (the worker is past its request loop, so a failed send is not an error);
- `Err("server model worker did not exit within {timeout:?}")` on expiry;
- `Err("server model worker panicked: ...")` when the watcher recorded a panic.

It is idempotent: a second call returns `Ok` at once because the observer stays observed. The body is the loop `InProcessServer::shutdown` already had, moved so that every owner uses one implementation instead of a second wait loop, which the issue explicitly ruled out. `WORKER_EXIT_TIMEOUT` (120 s) moves from `src/server/in_process/mod.rs` to `model_provider.rs` as `pub` and is re-exported from `mlxcel::server`, because the integration tests live outside the crate and need the same bound.

The timeout is formatted with `Debug` rather than `as_secs()`. For 120 s both print `120s`, so the production message is unchanged, but a sub-second test timeout prints `500ms` instead of the misleading `0s`.

### `Drop` stays non-blocking

Making `Drop for ModelProvider` join the worker would have fixed the test without touching callers, and the issue placed it out of scope for a reason: `ModelProvider` is dropped on tokio runtime threads during router entry teardown and `AppState` teardown. A blocking join there could stall a runtime worker thread for up to the timeout. The PR instead makes the wait explicit at the few places that own process exit.

### No exit shortcuts

`std::process::exit`, `libc::_exit`, `quick_exit`, `mem::forget` of the provider, and a CI retry would each have made the abort disappear. The issue excluded all of them unless the backtrace proved the exit handlers themselves were defective. It did not: the handlers ran correctly, and a thread nobody joined touched a destroyed static. Skipping exit handlers would also skip legitimate flushing in every other library, and a retry would hide the same race in production binaries.

### Make every `start_server` return reach the join structurally

The single-model path has many `?` returns between creating the provider and `serve_http` (warmup, side-model resolution, `AppState` construction). Adding a join before each would be easy to miss for the next `?` someone writes. The PR splits everything after `ModelProvider` creation into `serve_with_model_provider(startup, ChatServeInputs { .. })`. `start_server` keeps an `Arc` clone of the provider, awaits the inner function, and passes its `Result` to `join_model_worker_before_exit`. Any return from the inner function, `?` included, now passes through the join, and the compiler enforces that.

A scope guard with a blocking `Drop` was not an option for the same reason as above: it would block inside an async context. `ChatServeInputs` exists only to keep the split function's signature readable.

`join_model_worker_before_exit`:

- logs a serve error at error level first (`server stopped: ...; waiting for the model worker to exit`), so a bind or TLS failure during a long model load reads as a failure in the log rather than a silent wait;
- runs `shutdown_and_wait(WORKER_EXIT_TIMEOUT)` on `spawn_blocking`, so the wait does not occupy an HTTP runtime worker;
- logs `model worker exited before process exit` on success, or a warning on failure;
- returns the serve error unchanged when there is one. A shutdown failure is returned only when serving itself succeeded, so it never replaces the reason the server stopped.

### Router: drain on both `select!` arms

`run_router_server` previously drained the pool only inside the Ctrl-C arm. The `select!` now produces a `Result` from either arm, and `shutdown_pool.shutdown_all(ROUTER_SHUTDOWN_TIMEOUT)` runs after it unconditionally, with the existing completion and pending-resource logs. The listener error is logged before the drain starts. If `tokio::signal::ctrl_c()` fails to install its handler, that error is logged, the pool still drains, and the error is returned afterwards; before, the `signal?` returned before `shutdown_all`. `shutdown_all` already waits on each worker's exit observer and, through the entry operation guard, on an in-flight `build_model_app`, so no new wait logic was needed.

### Test ordering

Both real-model pipeline tests (`pipeline_ci_multi_stage_real_models.rs` and `pipeline_server_remote_real_models.rs`) now call `shutdown_and_wait` on the remote coordinator provider first, then the dense provider, and `expect` both before shutting the stage services down. The coordinator goes first so its last stage round trips complete while the stages are still serving. The second file only builds with CUDA or Metal today, where the allocators are leaked, but it had the same detached-worker pattern and would abort on a CPU runner.

The smoke test's doc comment also stops describing the setup as "two processes": the stages, coordinator and dense baseline are threads of one test process talking over loopback TCP.

## 3. Change Summary

9 files, +545/-48: two commits (e4312b89 for the fix, 03c1e365 for logging the exit reason before the join) plus a merge of origin/main (881231bd).

- `src/server/model_provider.rs`: `ModelProvider::shutdown_and_wait`; `pub const WORKER_EXIT_TIMEOUT`.
- `src/server/mod.rs`: re-exports `WORKER_EXIT_TIMEOUT`.
- `src/server/in_process/mod.rs`: `InProcessServer::shutdown` delegates to `shutdown_and_wait`; the local constant and wait loop are removed.
- `src/server/startup.rs`: `start_server` split into `serve_with_model_provider` plus `ChatServeInputs`; new `join_model_worker_before_exit`.
- `src/server/router_server.rs`: `run_router_server` drains on the listener-error and signal-error arms as well as Ctrl-C, logging the exit reason first.
- Tests:
  - `src/server/model_provider_tests.rs`: a `provider_with_worker` helper over a hand-rolled worker thread, and four tests: cooperative worker (`Ok`, observed, second call `Ok` without waiting), already-exited worker (`Ok` with `Duration::ZERO`), worker panicking during teardown (`Err` carrying `teardown boom`), worker ignoring shutdown (`Err` with the exact message `did not exit within 1s` after at least 1 s, then `Ok` once released).
  - `tests/pipeline_ci_multi_stage_real_models.rs`, `tests/pipeline_server_remote_real_models.rs`: explicit joins as described above.
  - `tests/server_exit_joins_worker.rs` (new): spawns a real `mlxcel-server` while the test holds the port, so the bind fails after the model started loading. The single-model test asserts exit code 1, no signal, no `pure virtual method called` / `terminate called` text, and the line `model worker exited before process exit`. The router test uses a preset with `load-on-startup = 1` and asserts `router: lifecycle shutdown completed attempted=1 completed=1`; it matches the counts because a failed `begin_load` logs the same message with `attempted=0`. Both skip when `qwen3-0.6b-4bit` is absent.

## 4. Trade-offs and Residuals

- **Error exits can now wait.** A bind or TLS failure while a large model is still loading used to exit at once (and could abort on CPU). It now waits up to 120 s (single-model) or 150 s (router) for the worker. The error is logged before the wait, so the operator sees the cause immediately.
- **A worker that outlives the timeout still exits under the handlers.** If the worker does not exit within `WORKER_EXIT_TIMEOUT` or `ROUTER_SHUTDOWN_TIMEOUT`, the process returns from `main` with it alive, and the original hazard applies on the CPU backend. The alternative is hanging indefinitely. A model load cannot be cancelled today, so a load longer than the timeout reaches this case; #2745 tracks it.
- **Side-model workers are not joined.** The audio, embedding, image and rerank workers are trait objects without an exit observer. The same hazard class applies to them on the CPU backend. #2745 tracks them, together with the sleeping-worker reload on shutdown and the `--pp-peer` signal path.
- **The fix applies to all backends but only CPU was exposed.** On Metal, CUDA and ROCm the leaked allocators mean the abort could not occur; the join still adds the bounded wait there.

## 5. Validation

**Mechanism.** gdb with `break __cxa_pure_virtual` on the CI-configured debug CPU binary: 3 of 3 attempts stopped in the worker's `PagedBlockPool` drop while the main thread was in `__run_exit_handlers`, as described in section 1.

**Before and after, local.** The smoke binary looped 50 times exactly as CI runs it (`MLXCEL_CI_PP_MODEL=qwen3-0.6b-4bit "$BIN" pipeline_multi_stage_two_host_logical_smoke --ignored --nocapture --test-threads=1`):

| Build | Aborts |
|---|---|
| origin/main 8385f40b | 44/50 (every abort log contains `pure virtual method called`) |
| this branch | 0/50 (same box, loops overlapped in time) |

**Suites.** `server::model_provider::` and `server::in_process::` lib tests (`--profile test-fast --features cuda`, `--test-threads=1`); both `server_exit_joins_worker` tests on the real checkpoint with `MLXCEL_REQUIRE_MODELS=1`; `cargo check` of `pipeline_server_remote_real_models` with `cuda,test-utils`; clippy `-D warnings` and `cargo fmt --check`.

**CI stability.** Ten consecutive `workflow_dispatch` runs of `pipeline-parallel-ci.yml` at head `03c1e365`, dispatched one at a time because the workflow's concurrency group cancels overlapping runs on the same ref. All ten passed, all on `ubuntu24/20261025.353`, and no job log contains `pure virtual method called`.

| # | Run | Result | Runner image |
|---|---|---|---|
| 1 | 41045436759 | success | `ubuntu24/20261025.353` |
| 2 | 41048002710 | success | `ubuntu24/20261025.353` |
| 3 | 41049748107 | success | `ubuntu24/20261025.353` |
| 4 | 41051499873 | success | `ubuntu24/20261025.353` |
| 5 | 41053271228 | success | `ubuntu24/20261025.353` |
| 6 | 41055089949 | success | `ubuntu24/20261025.353` |
| 7 | 41056817489 | success | `ubuntu24/20261025.353` |
| 8 | 41058570988 | success | `ubuntu24/20261025.353` |
| 9 | 41060493226 | success | `ubuntu24/20261025.353` |
| 10 | 41062232317 | success | `ubuntu24/20261025.353` |

The two `pull_request` runs of the same workflow on this PR (41024288311 at `e4312b89`, 41045135070 at `03c1e365`) also passed on that image.

**Not verified.**

- `pipeline_server_remote_real_models` was compiled, not run; it targets CUDA and Metal builds.
- On a GPU-backend build the `server_exit_joins_worker` assertion against abort text cannot fail, because those allocators are leaked. What pins the fix there is the join log line, which did not exist before this PR. The abort-text assertion only has teeth on a CPU-backend build.
- The router's `ROUTER_SHUTDOWN_TIMEOUT` expiry followed by process exit, the `ctrl_c` installation failure arm, and a worker exceeding `WORKER_EXIT_TIMEOUT` in `start_server` are covered by code review and the unit-level timeout test, not by a process-level test.
- No run on Metal or ROCm.

## 6. Learning Points

- **Returning from `main` does not wait for threads.** In Rust, a spawned thread that nobody joins keeps running while `exit()` runs C++ static destructors and `atexit` handlers. Any thread that may still touch native library state at that moment is a crash waiting for the right scheduling. Code that owns process exit must join such threads explicitly.
- **`pure virtual method called` after `main` returns usually means "use after static destruction".** During and after a C++ destructor, the object's vtable points at the base class, so a virtual call on a destroyed function-local static lands on `__cxa_pure_virtual`. A breakpoint there in gdb, plus `thread apply all bt`, shows both the late thread and the exit path in one stop.
- **Backend-specific singletons explain backend-specific crashes.** MLX leaks its Metal, CUDA and ROCm allocators on purpose and keeps a static for the CPU fallback. The same Rust code was safe on GPU builds and racy on the CPU build CI uses.
- **Correlate intermittent CI failures with the environment, not only with commits.** Splitting runs by runner image separated a 10-for-10 correlation from a coincidental commit range and pointed at timing rather than at #2687/#2688.
- **Enforce cleanup structurally.** Splitting the post-load half of `start_server` into its own function made the join reachable from every `?` without auditing each one, in a context where a blocking `Drop` guard was not acceptable.
- **When cleanup can fail too, decide which error wins.** `join_model_worker_before_exit` keeps the error that ended serving and returns a shutdown failure only when serving succeeded, so the operator always sees the root cause.
