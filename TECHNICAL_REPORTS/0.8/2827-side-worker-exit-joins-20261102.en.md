# PR #2827: Join side-model workers and close the remaining exit-path gaps

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2745)

## Executive Summary

PR #2827 closes the four teardown gaps that #2742 left after #2738. Every side worker (audio, realtime, embedding, image, rerank) now has the same `shutdown_and_wait` contract as the chat worker, and `serve_with_model_provider` joins the side workers first and then the chat worker under one shared deadline on the blocking pool. A sleeping worker now exits on `Shutdown` instead of reloading the model, and the `--pp-peer` stage path always drains its thread even when the signal handler cannot be installed. 23 files, +836/-75.

## 1. Problem Statement

On CPU builds the MLX `CommonAllocator` is a function-local static, so any thread that frees MLX arrays after `main` returns aborts with `pure virtual method called`. #2742 joined the chat worker, but four paths could still return from `main` with a worker alive:

- Side workers joined only in `Drop`, on whichever async thread dropped the last `Arc`, with no ordering before `main` returned.
- A sleeping worker treated `ModelRequest::Shutdown` as a wake request, reloaded the full model, and only then saw the shutdown.
- Model load has no cancellation point, so a load longer than `WORKER_EXIT_TIMEOUT` escaped the join silently.
- The `--pp-peer` stage returned a signal-handler error before calling `shutdown`, and ran the blocking join directly on a tokio worker thread.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/model_provider.rs` | Typed `WorkerShutdownError` (`Timeout` or `Panicked`), shared `observe_worker_exit`, `enqueue_shutdown_command` with a bounded backoff, `wait_for_worker_exit` |
| `src/server/startup.rs` | `SideWorkerExit` collector and `join_model_workers_before_exit`; side workers register as they load; stage teardown extracted to `wait_for_stage_shutdown` with a `StageServiceHandle` trait |
| `src/server/model_worker.rs` | Sleeping arm matches `Shutdown` and breaks without clearing `sleeping` or reloading |
| `audio_worker`, `embedding_worker`, `image_worker`, `rerank_worker`, `realtime_engine` | `shutdown_and_wait` implemented over `WorkerExitObserver`; `Drop` keeps its join |
| `startup_tests.rs`, `tests/server_exit_joins_worker.rs` | New unit and integration tests; `ModelProvider::slow_to_start_worker_for_tests` |
| `docs/server-features.md` | Documents the no-cancellation limitation next to `--sleep-idle-seconds` |

## 3. Technical Decisions

### 3.1 Side workers first, one deadline

`join_model_workers_before_exit` runs the side-worker shutdowns in one `spawn_blocking` closure, each with whatever remains of `WORKER_EXIT_TIMEOUT`, then passes the remaining budget to `join_model_worker_before_exit`. One deadline covers the whole teardown, and every blocking join stays off the tokio worker threads.

### 3.2 A side-worker failure never masks the serving error

A side model is optional and its shutdown failure is logged with `warn!`, then discarded. Reporting it instead of the real reason the server stopped (a bind error, an unloadable `--image-model`) would hide the failure that matters. Error precedence stays with the chat-worker join, which returns the original `served` error unchanged.

### 3.3 A typed shutdown error

`shutdown_and_wait` previously returned an untyped `anyhow` error. The exit path needs to distinguish a worker that is still running from one that panicked, because only the timeout case carries the CPU-build abort warning. `WorkerShutdownError::{Timeout, Panicked}` makes that distinction explicit, and the existing message text is preserved for callers that only print it.

### 3.4 Sleeping is a state, not a wake request

The sleeping arm now matches `Shutdown` before the generic request arm, logs `Shutdown requested while sleeping, worker exiting`, and breaks. It deliberately does not clear `sleeping`: the process is exiting and nothing reloads, so the observable state stays truthful until exit. A reload slower than the deadline would reopen the #2738 race.

### 3.5 Load has no cancellation point

The issue chose documentation over a cancellation point, and this PR follows that. Load runs through per-family `backend.load_model*` implementations with no shared loop to check a flag, and MLX evaluates lazily, so a flag between shards would not bound the time either. The timeout now says the worker is still running and may abort a CPU build at exit.

### 3.6 The stage helper takes the signal future

`wait_for_stage_shutdown` owns the select and the shutdown in that order, so a signal-handler failure still drains the stage. The handle is behind a small `StageServiceHandle` trait so the failure arm is unit-testable with a fake that records the shutdown call. The router half of item 4 was already fixed in #2742 and needed no code change here.

## 4. Verification

| Check | Result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo check -p mlxcel --lib --tests` | pass |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | pass |
| `cargo test -p mlxcel --lib server::startup::` | 99 passed |
| `cargo test -p mlxcel --lib server::model_provider::tests::shutdown_and_wait` | 4 passed |
| `cargo test -p mlxcel --test server_exit_joins_worker` | 5 passed, 0 failed |

The two non-model tests cover the timeout message and the stage-drain arm. The real-model cases (Whisper, embeddinggemma, qwen3-0.6b sleeping) report as passing skips because those checkpoints are absent on this host and `MLXCEL_REQUIRE_MODELS=1` would panic.

## 5. Unverified

The `--features cuda` run and the 2-host logical PP CI job from #2738 were not run; no CUDA host is available in this environment. Those remain the GPU and distributed gates for the PR. The four real-model exit tests have also not executed against their checkpoints here.

## 6. Learning Points

The exit race is a shape, not a single bug. #2742 fixed one instance, and the same unjoined-teardown shape remained in five side workers, the sleeping path, the bounded-load path, and the stage signal path. Fixing the shape required a shared observer and one collector, not five local patches.

## 7. Risks

Medium. Teardown ordering changed for every side model, and the shared deadline can now truncate a side worker's budget if an earlier one is slow. A side-worker truncation is logged and does not replace the serving error, but a future reader should know the deadline is shared rather than per worker.
