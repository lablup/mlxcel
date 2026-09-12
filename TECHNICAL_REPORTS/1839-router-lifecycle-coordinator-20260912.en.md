# Issue #1839 — Router lifecycle coordinator and worker-exit gating

## Summary

Issue #1839 adds the backend lifecycle coordination layer needed by the WebUI epic without adding a second model registry. Router entries now carry stable WebUI model identities, revisioned lifecycle snapshots, and a shared operation/event coordinator that future `/ui-api/v1` routes can reuse while legacy b10621 router routes continue to drive the same state transitions.

## Implementation

The router pool now owns a `LifecycleCoordinator` and each `RouterModelEntry` owns a `ModelLifecycle`. The lifecycle tracks inference state, download state, active response-body leases, revision/generation, admission stop, and worker-exit observation. Load, unload, autoload dispatch, cache download completion, removal, and the UI action adapter all synchronize through the same per-entry operation guard instead of creating a parallel WebUI path.

Model IDs are generated from the WebUI contract's canonical identity vectors: the raw source namespace stays server-side, clients only receive the SHA-256 source-key hash, and the opaque `mdl_...` ID is stable across rescans for the same configured source and entry. The coordinator keeps bounded operation and event history: active operations are capped, terminal operations are retained at 200/1h, SSE events at 1024/10m, and idempotency keys are pruned with their retained operation.

Unload now stops new admission, waits for active request leases to drain, sends the provider shutdown command, and only frees router capacity after the provider worker's actual thread exit is observed. The `ModelProvider` wraps every worker constructor with a `WorkerExitObserver`, including fake/test constructors, so the router observes model destruction rather than relying on dropping a registry `Arc`. A stream response now holds a lifecycle lease until the HTTP body finishes or is dropped, so unload cannot race a still-streaming response.

## Compatibility and validation

Legacy `/models`, `/models/load`, `/models/unload`, `/models/sse`, and router dispatch behavior remain b10621-compatible; transient lifecycle transitions are published through the new coordinator and do not inject misleading legacy status events. Tests cover contract identity vectors, busy-vs-capacity axes, request-lease drain behavior, operation idempotency, bounded idempotency pruning, SSE replay gap/restart classification, router pool regressions, router server regressions, and model provider worker wrapping.

## Real checkpoint acceptance

A GPU checkpoint harness was prepared at `/tmp/epic-1834-run-5tpfu3lc/issue-1839-real-lifecycle-harness.sh`. It loads model A, starts a long streaming request, unloads A while the response body is live, verifies new admission is refused while draining, drops the stream, checks the worker-exit observation log, then loads model B under `--models-max 1`. The harness also records scoped process RSS snapshots before/after load/unload/load; RSS is informational and not treated as a zero-memory promise.
