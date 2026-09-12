# PR #1860 / Issue #1839 — Router lifecycle coordinator and worker-exit gating

## Summary

Issue #1839 adds the backend lifecycle coordination layer needed by the WebUI epic without adding a second model registry. Router entries now carry stable WebUI model identities, monotonic lifecycle revisions, active request leases, and a shared operation/event coordinator used by both legacy router transitions and the new WebUI model-action/operation/event adapters.

## Implementation

The router pool now owns a `LifecycleCoordinator` and each `RouterModelEntry` owns a `ModelLifecycle`. The lifecycle tracks inference readiness, download state, busy versus capacity reservation, revision/generation, admission stop, active response-body leases, and observed worker exit. A pool-wide revision authority stamps recreated entries and all lifecycle transitions, so a remove/rescan/re-add ABA cannot reuse revision 1 for the same canonical model ID.

Model IDs are generated from the WebUI contract's canonical identity vectors: the raw source namespace stays server-side, clients receive only the SHA-256 source-key hash, and the opaque `mdl_...` ID is stable for the same configured source and entry. The coordinator keeps bounded operation and event history: active operations are capped, terminal operations are retained at 200/1h, SSE events at 1024/10m, and idempotency keys are pruned with their retained operation. Event sequence assignment, ring insertion, and broadcast share one mutex-ordered authority.

UI model actions submit idempotent operations against this coordinator, replay duplicate keys before revision checks, reject stale revisions with typed field errors, and make `202 Accepted` mean queued rather than ready. Queued actions now carry the requested primary entry `Arc`, stable model ID, and revision, and explicit eviction targets carry their own entry `Arc`, stable ID, and revision. The background executor revalidates those identities under the load/unload transition locks before mutating state, so replacement by rescan/remove/re-add is rejected even when the display name or canonical model ID is reused.

Unload now stops new admission, waits for active request leases to drain, sends the provider shutdown command, and only frees router capacity after the provider worker's actual thread exit is observed. The `ModelProvider` wraps every worker constructor with a `WorkerExitObserver`, including fake/test constructors, so the router observes model destruction rather than relying on dropping a registry `Arc`. A streaming response holds a lifecycle lease until the HTTP body finishes or is dropped, so unload cannot race a still-streaming response. Router shutdown drains all reserved/downloading entries within a bounded report window and logs any remaining resources instead of force-killing an in-process GPU worker.

## Review fixes in this cycle

The post-review fixes close the high-risk race windows around `begin_load`, rescan, remove, and queued UI actions. `begin_load` now takes the load lock before reading the registry, revalidates the captured entry under the entry transition guard, and performs the final current-entry check and `loading` reservation while retaining the registry read lock. A concurrent rescan cannot publish between those two steps; a deterministic barrier test exercises that exact boundary. Rescan preserves any current entry that became reserved after the stale snapshot clone. Remove revalidates the current `Arc` before cancel, after download stop, before unload, and while deleting from the registry; if a cancelled download's own terminal rescan has already dropped the entry, removal proceeds only when no replacement entry appeared. LRU eviction refuses serving/draining entries, and explicit eviction has typed `not_needed`, `displaced`, and `failed_after_displacement` outcomes.

The WebUI administrative adapters are no longer mounted by the production/base router. `create_router_app` keeps the legacy router only, while tests and future startup integration use `create_router_app_with_authenticated_ui`; that accessor rejects every `/ui-api/*` request unless an API key is configured and presented. Slow SSE clients now get client-local reset events instead of publishing a reset into the global ring for every lagging subscriber. Route and operation history errors are redacted to bounded typed messages, while raw build/load details are kept in server logs.

The WebUI contract was extended with `target.eviction_target_id` and typed `ModelEvictionReport`, regenerated TypeScript DTOs, and updated fixtures. Route-level producer tests compare the complete JSON returned by operation retrieval, operation listing, and a gap SSE envelope with canonical fixtures validated by the pinned JSON Schema gate. Only validated dynamic IDs, RFC3339 timestamps, revisions and sequences are normalized; required nullable fields, extra keys, lifecycle fields and discriminators must match exactly. A coordinator-seeded ready load result uses a distinct eviction victim; this tests serialization, not real worker execution. Negative controls reject missing nullable fields, unknown nested fields, invalid dynamic values, altered lifecycle/discriminator values, and envelope/list omissions.

## Compatibility and validation

Legacy `/models`, `/models/load`, `/models/unload`, and `/models/sse` response shapes are preserved. The deliberate safety difference is that unload waits for active responses and observed worker release; busy eviction and stale replacement are refused rather than freeing an in-use slot. Drain or worker-exit timeouts retain the reservation instead of reporting successful release; legacy cache removal remains the existing router deletion path, with lifecycle protection when a reserved entry must be stopped before deletion. The production router still does not expose WebUI administrative routes until the secure startup/auth issues mount them deliberately.

Validation run in the issue worktree:

- `cargo check --profile test-fast --features metal,accelerate --bin mlxcel-server`
- `cargo fmt --check`
- `git diff --check`
- `python3 scripts/insert_apache_header.py --check`
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` (32 fixtures, including validator negative tests)
- `cargo test --profile test-fast --features metal,accelerate router_server_tests:: -- --nocapture` (26 passed)
- `cargo test --profile test-fast --features metal,accelerate router_lifecycle_tests:: -- --nocapture` (9 passed)
- `cargo test --profile test-fast --features metal,accelerate router_models_tests:: -- --nocapture` (33 passed)

- `cargo test --workspace --profile test-fast --features metal,accelerate` (11,151 passed, 0 failed, 361 ignored; ignored tests are not counted as passes)

## Real checkpoint acceptance

The root-coordinated real checkpoint gate passed on runtime commit `bd85ff07` on macOS 27 / Apple Silicon. Under `--models-max 1`, model A `meta-llama-3.1-8b-instruct-4bit` streamed 963 response characters. While unload waited for the live response body, a new inference request received the expected drain refusal HTTP 400; after the stream was dropped, the server logged observed worker exit for A. Model B `granite-4.0-h-tiny-4bit` then produced `Affirmative.`. SIGINT shutdown reported one attempted and one completed lifecycle shutdown.

| Process RSS scope | KiB |
|---|---:|
| Server start | 33,184 |
| A loaded, before first request | 353,280 |
| Streaming A | 4,364,832 |
| After unloading A | 4,166,832 |
| After loading B | 4,254,192 |

These are process RSS snapshots, not allocator measurements or proof of zero retained memory. A negative control against an older binary (SHA-256 `ef3146d4a2722cce81b683bd48e67996fc9b9c4512c66ae4d51549e0844c6a78`, source commit unknown) failed the same harness with exit 4 at unload-before-stream-drop. This distinguishes the new observed-worker-exit behavior from the old registry-drop behavior without assigning an unverified source revision to that binary.

These measurements include the final atomic-reservation fix and producer-contract finalization. Temporary local harness files are not a published evidence archive; the measured outcomes are recorded here.

## User-approved GB10 CI outage waiver

On 2026-09-12, the user confirmed that the GB10 runner was down and explicitly authorized proceeding without its required CI jobs, using passing local CI. This waiver covers the unavailable GB10 cargo-clippy and OpenXLA feature compile jobs for this delivery; they are not reported as passed. CI configuration and branch protection are unchanged.

The local Metal/Accelerate workspace suite passed with 11,151 tests, zero failures and 361 ignored cases. Final workspace/all-target clippy, 27 CLI tests plus four compatibility tests, the 32-fixture contract gate and formatting checks passed. The real-checkpoint results above remain Apple Silicon evidence only. No CUDA or unavailable OpenXLA validation is claimed.
