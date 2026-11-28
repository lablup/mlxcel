# Technical Report: PR #3015 - fix(python): reap the managed server off the asyncio event loop

**Date**: 2026-11-14
**Status**: Completed
**Languages**: Python
**Risk Level**: Low

---

## Executive Summary

`AsyncLLM.close()` stopped the managed `mlxcel` server by calling the synchronous `ManagedServer.close()` on the event loop thread. That call blocks in `proc.wait()` for up to the shutdown grace period (twice when the child has to be killed) and in a log-thread join, so every other task on the loop froze until the child exited. The shutdown now runs on the loop's default executor and is awaited. Cancellation is held until the child is reaped and then re-raised. With a child that takes 350 ms to exit, a 10 ms heartbeat task went from 0 beats and a 375 ms gap during `close()` to 36 beats and a gap of about 11 ms; `close()` itself takes as long as before.

---

## 1. Problem Statement

### 1.1 Background

The Python client can start and supervise a local `mlxcel` server (`ManagedServer` in `python/src/mlxcel/_server.py`). The sync `LLM` and the async `AsyncLLM` share it. `ManagedServer.close()` sends SIGTERM, waits, escalates to SIGKILL, joins the log-forwarding thread, and unlinks the socket. All of these are blocking calls.

### 1.2 Existing Issues

- `AsyncLLM.close()` called `self._server.close()` directly, so the waits ran on the event loop thread.
- A server that honors SIGTERM slowly, or ignores it, stalled the whole application for up to twice the grace period.
- #3011, merged just before this PR, made the server stop reachable from a `finally` after an HTTP close error or cancellation. This PR changes how that `finally` waits.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Cancellation abandons the worker and leaves a child behind a client already marked closed | Leaked process | Ruled out by design and tested |
| No asyncio loop (Trio through httpx) or an executor that refuses work | Close raises, child leaks | Handled by a synchronous fallback |
| Delayed cancellation surprises callers | `cancel()` takes up to the grace period | Low, documented |

---

## 2. Technical Review

### 2.1 Security

No new input surface. The change keeps the existing signal sequence and only moves where the waits run.

### 2.2 Performance

| | before | after |
|---|---|---|
| heartbeats during `close()` (Python 3.12) | 0, largest gap 375 ms | 36, largest gap 10.6 ms |
| duration of `close()` | 365 ms | 365 ms |

Figures from the contributor's runs on Linux aarch64. Shutdown uses one default-executor thread for its duration.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none in the API. A task cancelled during shutdown now sees its `CancelledError` after the child exits instead of immediately.
- **New Dependencies**: none.
- **Compatibility**: Python 3.9 to 3.13. Without a running asyncio loop, or when `run_in_executor` raises `RuntimeError` (executor shut down, interpreter exiting), shutdown runs synchronously as before.

### 2.4 Code Quality

`python/tests/test_async_shutdown.py` adds 16 cases with real child processes. The key one holds the child's SIGTERM handler until a file appears that only the event loop creates, so an implementation that blocks the loop cannot finish gracefully. Modes cover direct and context-manager close, HTTP-close error and cancellation, one and two cancellations during shutdown, a killed child, a worker error, and a worker error during cancellation. Further cases cover connect mode (no worker), a shut-down or refusing executor, and a coroutine driven without any asyncio loop.

---

## 3. Technical Decisions

### 3.1 Finish reaping before propagating cancellation

**Context:** once `_closed` is set, a second `close()` is a no-op, so anything left unfinished is never retried.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Propagate cancellation at once, worker continues | Immediate `cancel()` | Worker thread and possibly a live child outlive a client that reports itself closed |
| Cancel the worker | Simple | Thread work cannot be interrupted; the child may survive |
| Wait for the worker, then re-raise | Child always reaped before return | Cancellation delayed by up to the grace period |

**Decision:** wait, then re-raise. The delay is bounded by the existing grace period.

### 3.2 `asyncio.wait` instead of `asyncio.shield`

**Context:** the first revision awaited `asyncio.shield(cleanup)` in a loop. If the task had been cancelled and the worker then raised, the last shield await surfaced the worker's error and the recorded `CancelledError` was dropped, so a cancelled `close()` ended with an ordinary exception.

**Decision:** the review follow-up awaits `asyncio.wait((cleanup,))`, which neither cancels the worker nor raises its error, then reads the outcome once and raises a pending cancellation ahead of a worker error. The `cancel-worker-error` mode fails on the shield version.

### 3.3 Default executor, not a private one

A private thread pool could outlive the loop. The default executor is joined by `asyncio.run()` through `shutdown_default_executor()`, so no cleanup thread survives the loop.

---

## 4. Change Summary

| File | Change |
|------|--------|
| `python/src/mlxcel/_async_client.py` | `close()` runs `ManagedServer.close` on the default executor, holds cancellation until it finishes, falls back to a synchronous call when no loop or executor is available |
| `python/tests/test_async_shutdown.py` | New: 16 process-backed cases for loop responsiveness, cancellation, errors, and fallbacks |

---

## 5. Learning Points

- Blocking cleanup inside `async def` is easy to miss because it still "works"; the loop only shows it as latency elsewhere. A test that makes graceful exit depend on the loop running turns that latency into a failure.
- `asyncio.shield` protects the inner future from cancellation but still raises the inner future's result into the awaiting task. When the goal is "wait without being affected", `asyncio.wait` is the better primitive.
- Unverified: a real Trio run on the final revision (the contributor probed an earlier one) and macOS or Windows by the contributor. The maintainer reran the suite on macOS with Python 3.11 and 3.9.
