# PR #1860 / Issue #1839 — Router lifecycle coordinator and worker-exit gating

## Summary

Issue #1839 adds the backend lifecycle coordination layer needed by the WebUI epic without adding a second model registry. Router entries now carry stable WebUI model identities, monotonic lifecycle revisions, active request leases, and a shared operation/event coordinator used by both legacy router transitions and the new WebUI model-action/operation/event adapters.

## Implementation

The router pool now owns a `LifecycleCoordinator` and each `RouterModelEntry` owns a `ModelLifecycle`. The lifecycle tracks inference readiness, download state, busy versus capacity reservation, revision/generation, admission stop, active response-body leases, and observed worker exit. A pool-wide revision authority stamps recreated entries and all lifecycle transitions, so a remove/rescan/re-add ABA cannot reuse revision 1 for the same canonical model ID.

Model IDs are generated from the WebUI contract's canonical identity vectors: the raw source namespace stays server-side, clients receive only the SHA-256 source-key hash, and the opaque `mdl_...` ID is stable for the same configured source and entry. The coordinator keeps bounded operation and event history: active operations are capped, terminal operations are retained at 200/1h, SSE events at 1024/10m, and idempotency keys are pruned with their retained operation. Event sequence assignment, ring insertion, and broadcast share one mutex-ordered authority.

UI model actions submit idempotent operations against this coordinator, replay duplicate keys before revision checks, reject stale revisions with typed field errors, and make `202 Accepted` mean queued rather than ready. Queued actions now carry the requested primary entry `Arc`, stable model ID, and revision, and explicit eviction targets carry their own entry `Arc`, stable ID, and revision. The background executor revalidates those identities under the load/unload transition locks before mutating state, so replacement by rescan/remove/re-add is rejected even when the display name or canonical model ID is reused.

Unload now stops new admission, waits for active request leases to drain, sends the provider shutdown command, and only frees router capacity after the provider worker's actual thread exit is observed. The `ModelProvider` wraps every worker constructor with a `WorkerExitObserver`, including fake/test constructors, so the router observes model destruction rather than relying on dropping a registry `Arc`. A streaming response holds a lifecycle lease until the HTTP body finishes or is dropped, so unload cannot race a still-streaming response. Router shutdown drains all reserved/downloading entries within a bounded report window and logs any remaining resources instead of force-killing an in-process GPU worker.

## Review fixes in this cycle

The post-review fixes close the high-risk race windows around `begin_load`, rescan, remove, and queued UI actions. `begin_load` now takes the load lock before reading the registry, revalidates the captured entry under the entry transition guard, and rechecks again immediately before marking `loading`. Rescan preserves any current entry that became reserved after the stale snapshot clone. Remove revalidates the current `Arc` before cancel, after download stop, before unload, and while deleting from the registry; if a cancelled download's own terminal rescan has already dropped the entry, removal proceeds only when no replacement entry appeared. LRU eviction refuses serving/draining entries, and explicit eviction has typed `not_needed`, `displaced`, and `failed_after_displacement` outcomes.

The WebUI administrative adapters are no longer mounted by the production/base router. `create_router_app` keeps the legacy router only, while tests and future startup integration use `create_router_app_with_authenticated_ui`; that accessor rejects every `/ui-api/*` request unless an API key is configured and presented. Slow SSE clients now get client-local reset events instead of publishing a reset into the global ring for every lagging subscriber. Route and operation history errors are redacted to bounded typed messages, while raw build/load details are kept in server logs.

The WebUI contract was extended with `target.eviction_target_id` and typed `ModelEvictionReport`, regenerated TypeScript DTOs, and updated fixtures. The route-level producer test now validates an actual serialized `/ui-api/v1/operations` response by round-tripping it through the Rust DTO, asserting schema-critical token/model-id/timestamp/nullability fields, and comparing operation/result discriminators plus eviction outcome against the schema-validated fixture.

## Compatibility and validation

Legacy `/models`, `/models/load`, `/models/unload`, `/models/sse`, and router dispatch behavior remain b10621-compatible; legacy cache removal remains the existing router deletion path, with lifecycle protection when a reserved entry must be stopped before deletion. The production router still does not expose WebUI administrative routes until the secure startup/auth issues mount them deliberately.

Validation run in the issue worktree:

- `cargo check --profile test-fast --features metal,accelerate --bin mlxcel-server`
- `cargo fmt --check`
- `git diff --check`
- `python3 scripts/insert_apache_header.py --check`
- `/tmp/mlxcel-webui-contract/bin/python scripts/ci/check_webui_contract.py`
- `cargo test --profile test-fast --features metal,accelerate router_server_tests:: -- --nocapture` (23 passed)
- `cargo test --profile test-fast --features metal,accelerate router_lifecycle_tests:: -- --nocapture` (9 passed)
- `cargo test --profile test-fast --features metal,accelerate router_models_tests:: -- --nocapture` (32 passed)

## Real checkpoint acceptance

Root-coordinated real checkpoint gate passed on the first post-implementation build (`584249a4`): model A `meta-llama-3.1-8b-instruct-4bit` streamed 572 response characters, unload while the response body was live produced the expected drain refusal HTTP 400, worker exit was observed for model A, and model B `granite-4.0-h-tiny-4bit` produced non-empty text (`Affirmative.`) under `--models-max 1`. Scoped RSS snapshots were: start 32656 KiB, after load A 351824 KiB, after unload A 4173872 KiB, after load B 4257760 KiB. RSS is informational and not a zero-memory promise.

Root also ran a negative control against an old binary (`SHA256 ef3146d4a2722cce81b683bd48e67996fc9b9c4512c66ae4d51549e0844c6a78`, source commit unknown), which failed the same real harness with exit 4 at unload-before-stream-drop. Evidence paths: `/tmp/epic-1834-run-5tpfu3lc/issue-1839-real-20260912-143914/summary.json` and `/tmp/epic-1834-run-5tpfu3lc/negative-control-ef3146d4.log`. A final real rerun after the review-fix commit is still pending root GPU coordination.
