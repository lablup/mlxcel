# PR #1860 / Issue #1839 — Router lifecycle coordinator and worker-exit gating

## Summary

Issue #1839 adds the backend lifecycle coordination layer needed by the WebUI epic without adding a second model registry. Router entries now carry stable WebUI model identities, revisioned lifecycle snapshots, active request leases, and a shared operation/event coordinator used by both legacy router transitions and the new WebUI model-action/operation/event adapters.

## Implementation

The router pool now owns a `LifecycleCoordinator` and each `RouterModelEntry` owns a `ModelLifecycle`. The lifecycle tracks inference readiness, download state, busy versus capacity reservation, revision/generation, admission stop, active response-body leases, and observed worker exit. UI model actions submit idempotent operations against this coordinator, replay duplicate keys before revision checks, reject stale revisions with typed field errors, and make `202 Accepted` mean queued rather than ready.

Model IDs are generated from the WebUI contract's canonical identity vectors: the raw source namespace stays server-side, clients receive only the SHA-256 source-key hash, and the opaque `mdl_...` ID is stable for the same configured source and entry. The coordinator keeps bounded operation and event history: active operations are capped, terminal operations are retained at 200/1h, SSE events at 1024/10m, and idempotency keys are pruned with their retained operation. Event sequence assignment, ring insertion, and broadcast now share one mutex-ordered authority, and `/ui-api/v1/events` provides snapshot-fenced subscription, reconnect replay, and typed reset events for gaps or server restart.

Unload now stops new admission, waits for active request leases to drain, sends the provider shutdown command, and only frees router capacity after the provider worker's actual thread exit is observed. The `ModelProvider` wraps every worker constructor with a `WorkerExitObserver`, including fake/test constructors, so the router observes model destruction rather than relying on dropping a registry `Arc`. A streaming response holds a lifecycle lease until the HTTP body finishes or is dropped, so unload cannot race a still-streaming response. Router shutdown now drains all reserved/downloading entries within a bounded report window and logs any remaining resources instead of force-killing an in-process GPU worker.

## Compatibility and validation

Legacy `/models`, `/models/load`, `/models/unload`, `/models/sse`, and router dispatch behavior remain b10621-compatible; legacy cache removal remains the existing router deletion path, with lifecycle protection when a reserved entry must be stopped before deletion. The WebUI-specific removal operation is not claimed here and remains separate from this issue's model-action/operation/event adapters.

Validation run in the issue worktree: `cargo check --profile test-fast --features metal,accelerate --bin mlxcel-server`; `cargo test --profile test-fast --features metal,accelerate router_ -- --nocapture`; `python scripts/ci/check_webui_contract.py` (using the local validator environment). The focused tests cover contract identity vectors, DTO fixture round-trips, busy-vs-capacity axes, ordered event broadcast, ring gap replay, operations list/get/cancel, idempotency replay after revision change, UI load terminal failure, explicit eviction refusal, rescan/delete reservation barriers, response-body lease drop, bounded shutdown reporting, and WebUI HTTP action/operation/SSE adapters.

## Real checkpoint acceptance

A root-coordinated GPU checkpoint harness was prepared for maintainer execution. It loads model A, starts a long streaming request, unloads A while the response body is live, verifies new admission is refused while draining, drops the stream, checks the worker-exit observation log, then loads model B under `--models-max 1`. The harness also records scoped process RSS snapshots before/after load/unload/load; RSS is informational and not treated as a zero-memory promise.
