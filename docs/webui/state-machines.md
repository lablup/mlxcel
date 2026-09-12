# WebUI lifecycle and event state machines

## Per-model lifecycle

Inference lifecycle is independent from download state. A catalog entry always reports both `lifecycle.state` and `lifecycle.download`.

| From | Action | Preconditions | To | Invalid edge |
|---|---|---|---|---|
| `unloaded` | load | `expected_revision` matches, model complete/supported, capacity available or explicit eviction target accepted | `loading` | duplicate load for same revision returns 409 conflict or the original idempotent operation, never a second worker |
| `loading` | provider ready | tokenizer/provider ready and admission open | `ready` | unload may request drain/cancel but cannot pretend weights are already free |
| `loading` | load failure | loader reports error | `failed` | failed replacement does not promise rollback to a released previous model |
| `ready` | unload | new admission stopped | `draining` | unload with stale revision returns 409 |
| `ready` | load | same model already ready | `ready` | duplicate load returns 409 conflict unless idempotency key replays the original operation |
| `ready` | explicit eviction | caller names `eviction_target_id` and target is not busy beyond policy | `draining` | surprise eviction is forbidden |
| `draining` | requests complete | active request count reaches zero | `unloading` | load/unload races serialize through the coordinator |
| `unloading` | worker exit observed | provider sender dropped and worker exit/memory-release observation completes | `unloaded` | capacity is not freed before worker exit observation |
| `draining` | drain timeout | active requests remain | `draining` with model error | report recoverable blocked/failure on the operation, preserve resource ownership, and keep rejecting new loads until worker exit or operator-forced recovery is implemented |
| `failed` | load | expected revision matches current failed entry and no worker/resource owner remains | `loading` | retry with old revision returns 409 stale_revision; a failed load that still owns resources is not eligible |

Busy means loading, ready with active requests, draining, unloading, or downloading. Busy entries count against capacity until worker exit is observed. Existing `/models/load`, `/models/unload`, UI model actions, autoload dispatch and future page buttons must call the same coordinator.

## Download and removal lifecycle

| State | Action | Result |
|---|---|---|
| `absent` | `POST /downloads` | `downloading` operation after repo ID/revision validation and cache containment checks |
| `downloading` | progress with total | bytes update with determinate percentage derived by the client |
| `downloading` | progress without total | `total_bytes: null`, `indeterminate: true` |
| `downloading` | cancel | `cancelling` then `cancelled` only after the downloader stops and partial files are marked incomplete or removed |
| `downloading` | failure | `failed` with retryable reason when applicable |
| `complete` | removal | accepted only for managed cache entries that are not busy/loading/downloading/draining |
| `incomplete` | removal | accepted for managed cache entries when no worker owns them |

Deletion refusal cases are 409 for transient busy states and 422 for unsupported/non-cache sources. UI confirmation copy names the checkpoint and source but never exposes a raw filesystem path.

## Idempotency and stale revisions

`idempotency_key` scope is one `server_instance_id` and expires with operation history retention unless the operation is still active. Reusing the same key with the same request returns the original operation. Reusing the same key with a different normalized request returns 409 conflict. A changed `server_instance_id` means the client must not replay POSTs automatically; it must resnapshot and ask the user when necessary.

Every mutating model request carries `expected_revision`. If the catalog entry revision changed since the UI snapshot, return 409 `stale_revision` with the current operation/model pointer when possible.

## Snapshot to SSE fence

The client obtains full snapshots and records `server_instance_id` plus each resource `snapshot_sequence` from catalog, operations and selected runtime responses. `revision` is per-model concurrency control; `sequence` is global ordering across catalog, operations, settings and runtime events. Since separate snapshots can land at different global sequences, the replay cursor is the minimum held resource sequence, not the maximum. The client subscribes to `/events` with `Last-Event-ID` set to that minimum cursor; it applies an event only when the event sequence is newer than the corresponding resource snapshot sequence, so duplicate operation/runtime events are harmless and a catalog event between the catalog and operations snapshots is not lost. The server either delivers every later transition in order or emits `gap`/`server_restart`, after which the client discards incremental state and refetches. No transition may be lost between snapshot and subscription.

SSE events are monotonic per server instance. Each JSON payload carries `event_id` and `emitted_at`; the SSE `id:` field must equal `event_id`, and that ID is an opaque string containing sequence and instance information for replay only. Clients compare resource freshness with numeric `sequence`, not lexicographic event IDs. Unknown IDs, a ring miss, a changed instance, or a reset envelope force `/bootstrap`, `/catalog`, `/operations`, and selected `/runtime` refetch. Reconnect never re-POSTs actions.

## Required table fixtures

The table fixtures in `tests/fixtures/webui/scenarios/` are shared negative/edge contracts and must be consumed by backend route tests and frontend state tests in later issues. They cover duplicate load, load/unload race, busy eviction, stale revision, failed load, download cancel, deletion refusal, SSE gap, server restart, and unknown/null/partial-error handling. Invalid edges must be rejected by producers; consumers must render their typed error states instead of inventing recovery behavior.
