# PR #2284: Propagate MLX array readback failures without aborting

**Date**: 2026-10-13
**Status**: Implemented, independently reviewed, and focused tests passed
**Risk**: Medium (FFI error propagation and speculative scheduler cleanup)

## Summary

PR #2284 makes `array_evaluated_bytes` fallible across the C++/Rust cxx boundary. A failed MLX event now reaches engine and scheduler callers as a normal error instead of allowing a C++ exception to abort the process. Lookahead decoding drains an already-submitted next evaluation, restores speculative row positions, and aborts only unfinished rows in the affected cohort. Unrelated active requests continue and failed rows are not replayed.

## Problem

The previous bridge returned `Vec<u8>` directly. If synchronizing an MLX array surfaced an event error, the C++ exception crossed an infallible cxx boundary and terminated the process. The scheduler also needed a single cleanup path for failures discovered while collecting lookahead tokens: speculative state may include one append for the current lookahead and a second append for a submitted next prime.

## Design

- The cxx declaration now returns `Result<Vec<u8>>`; the C++ implementation keeps its native return type so the generated wrapper translates C++ exceptions into Rust errors.
- Engine token readback returns `Result<Vec<i32>, String>`, including fused-row and prompt-lookup callers.
- Lookahead failure cleanup evaluates any submitted `next.tokens` through the raw fallible path before trimming state, records the primary failure once, restores the correct one- or two-step speculative positions, removes lookaheads, and aborts only unfinished affected requests.
- A C++ fixture attaches a real signalled MLX `Event` error to an evaluated array. Its error object intentionally remains alive for the process because MLX stores a raw pointer. Direct FFI and engine tests consume this event exactly once.
- Scheduler coverage uses a thread-local one-shot readback fault at the collection boundary. Reusing the failed-event fixture as lookahead input would consume its one-shot event error earlier, while the next decoding step is built, and would not exercise scheduler readback cleanup. The seam preserves a valid submitted next step and injects the error at the boundary under test.

## Verification

The source commit is `9622b8c6`. The publication candidate merges base `a1c58265` at local head `d9a8c4ea`; its tree is `67d54aee7acc63c90a5395fda7673c6bc72a680f`. The reviewed 16-path binary diff against that base retains SHA-256 `60e33dd052fdb4405f281377c5875b80700573fdf0146c8bec95d9edd38489e7`.

- `cargo test --locked -p mlxcel-core -p mlxcel --lib --profile test-fast --features cuda -j4 --no-run`: passed. Log: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/build-no-run-restored-final.log`.
- Real failed-event FFI test: 1 passed. Log: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/ffi-failed-event-test-restored-final.log`.
- Engine failed-event propagation test: 1 passed. Log: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/engine-failed-event-test-restored-final.log`.
- Scheduler cleanup with and without a submitted next prime: 1 passed. It also verifies one health failure, restored model-owned offsets, no pending-token emission, no shutdown, and an unrelated survivor completing. Log: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/scheduler-read-failure-test-restored-final.log`.
- `cargo fmt --all -- --check`: passed. Log: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/fmt-check.log`.
- Refreshed repository contract gate: passed after merging current `main`, including version, kernel dtype-key, kernel port-dispatch, llama compatibility, ROCm overlay, 123 Python tooling tests, and 49 WebUI fixtures. Log: `/tmp/mlxcel-auto-20261012T1250/logs/contracts-1791-a1c-refreshed.log`.
- Negative control: the old infallible bridge was compiled in isolation and the direct FFI test ran in a child process. The child exited with `-6`, exactly `-SIGABRT`, proving the regression would abort without this change. Logs: `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-status.log`, `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-child.log`, and `/tmp/mlxcel-auto-20261012T1250/logs/issue-1791/negative-old-signature-build.log`.

The focused test binaries built from source commit `9622b8c6`, before the subsequent `main` refreshes that did not touch the 16 owned paths, were `mlxcel_core-be0ee8003b018a01` with SHA-256 `79ae35f3b958baab057da86d3abf2cc85058706a30840b2647cdb440e5dc1413` and `mlxcel-d5e5498c5ad868df` with SHA-256 `e9f19800ae9efc50ffaf2e2f96986a4123b8f601809ae621a76903c19288690a`.

## Review and limits

Independent implementation, security/performance, and static finalization reviews found no remaining issue in scope. The first issue contract gate passed before the final scheduler test-only fault seam was added; the refreshed gate above passed on the final test seam after merging current `main`. CI and the full integration checks remain with the orchestration gate before central merge. Broad workspace tests and Clippy were intentionally left to that gate.
