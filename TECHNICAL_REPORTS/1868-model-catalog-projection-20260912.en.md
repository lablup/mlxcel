# Technical Report: PR #1868 — Model catalog projection

**Date**: 2026-09-12
**Status**: Open PR; focused validation plus root full-workspace and real-model gates complete; GB10 CI unavailable
**Languages**: Rust, JSON/OpenAPI, TypeScript, Markdown
**Risk Level**: Medium
**Implementation snapshot**: `6befcd26ea3d4624c76f64caf345908727922de6`

## Executive Summary

[PR #1868](https://github.com/lablup/mlxcel/pull/1868) implements the metadata-only catalog foundation for issue #1840 in epic #1834. It projects the existing router inventory and provider state into typed list/detail responses and coordinator-owned refresh operations, without creating a parallel model registry. Production startup mounting remains downstream in #1838; this report does not claim a completed end-user WebUI.

## 1. Problem Statement

A downloaded checkpoint, a recognized architecture, a supported backend, and a ready provider are different facts. A catalog built from directory names or vendor headings would conflate them and could offer chat or image actions that the current runtime cannot execute. Reusing unrestricted loader detection during library browsing would also risk weight/header reads and unnecessary work on the request executor.

The integration must preserve the existing cache/models-directory/preset authority, stable identities, and lifecycle coordinator while exposing incomplete or unsupported checkpoints honestly. Single-model mode must describe its existing provider instead of registering another one.

## 2. Technical Decisions

### One detection decision tree, different probes

`ModelDetectionProbes` separates the existing detection dispatch from filesystem evidence acquisition. Runtime loading retains its normal probe implementation. Catalog detection supplies bounded config/sidecar/index probes and declines classifications that require SafeTensors headers. Registry-derived tasks and backend support replace a new handwritten family table.

The trade-off is deliberate uncertainty: a checkpoint may be inspectable but lack a resolved variant until sufficient metadata or provider evidence exists. Guessing “text” when vision/MTP evidence is unavailable would give a more convenient but false answer.

### Cached metadata, current lifecycle

The catalog projects `RouterPool::catalog_snapshot()` on blocking workers. Cached metadata is keyed by stable identity plus router catalog epoch and model generation, while current revision, lifecycle, provider-confirmed capability, and removal status are reapplied. Ordinary GET cache hits no longer recompute content fingerprints or repeat heavy metadata filesystem probes; first cache fill and explicit refresh still perform bounded inspection. The fingerprint is not a checksum of weight contents.

Default pagination is 50, with a 200-item page limit and 1,000-entry inventory bound. Results sort by opaque ID. Stability is guaranteed for an unchanged inventory, not as a cross-request transaction during concurrent mutations.

### Execution ownership is stronger than operation reuse

Refresh requests use one server-instance-scoped execution owner while the coordinator operation is active. Returning an accepted/replayed operation alone would not prevent multiple background rescans. Only the owner dispatches the rescan; terminal cleanup permits a later refresh. Signature comparison counts detected changes even if before/after inventory lengths match.

### Keep API availability separate from integration readiness

The authenticated router constructor exposes catalog handlers for tests and future secure startup; the normal constructor keeps UI APIs unmounted. The single-model accessor reads the existing `AppState` provider and actual inference ID, reports read-only removal guidance, and constructs no second provider. Browser security and production CLI integration remain #1837/#1838 responsibilities, while deletion execution belongs to #1841.

## 3. Review Refinements

Independent correctness and security reviews cleared the final implementation snapshot after these refinements:

- Bounded metadata reads and shared detection probes prevent fallback to unrestricted weight-header inspection; unknown reasons replace unsupported guesses.
- Filesystem evidence rejects symlinked sidecars and nested pooling parents. The final fix specifically rejects `1_Pooling` symlink traversal rather than checking only its leaf file.
- Metadata caching preserves fresh pool lifecycle/provider facts instead of replaying stale readiness from a cached entry, and cache invalidation now follows router catalog epochs rather than per-GET fingerprint walks.
- Raw `model_type` is preserved exactly within the schema limit, `declared_architectures` is exposed as a bounded nullable raw array, and non-string/oversized values become null with reasons rather than truncated false exactness.
- Refresh singleflight separates operation acceptance from execution ownership and reports same-count changes through signatures; same-size config/index edits, legacy reloads, and cache downloads have route regressions.
- Producer tests compare the complete serialized catalog shape against pinned schema-validated fixtures, normalizing only temporary-path-derived IDs/fingerprint and disk bytes.

Config/sidecar reads are limited to 256 KiB and index JSON to 512 KiB. Disk traversal limits are 4,096 visited/pending entries and depth 8; unavailable measurements return null with a reason. No benchmarked latency or memory-reduction claim is made.

## 4. Validation Record

The following results were reported by the implementation and independent review stages for the final scoped change; the documentation stage did not start builds, tests, or model execution.

| Gate | Result at report preparation |
|---|---|
| Catalog-focused tests | 24 passed |
| HTTP catalog route tests | 4 passed |
| Existing catalog refresh singleflight route test | 1 passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Passed |
| `cargo fmt --check`, `git diff --check`, and `python3 scripts/insert_apache_header.py --check` | Passed |
| `make verify-webui-contract` with the isolated verifier Python | Passed, 40 fixtures |
| `make verify-llama-compat verify-versions verify-kernel-dtype-keys` with Python 3.14 on `PATH` | Passed |
| Root full workspace/local CI and actual Llama + Granite regression | Passed: 11187/0/361 plus clippy; Llama 572-character smoke; Granite affirmative smoke; SIGINT worker-exit 1/1 |
| GB10-required CI | Runner reported down; user explicitly authorized skipping unavailable runner gates |
| CUDA execution, production `--webui`, browser acceptance | Not established by this change's scoped evidence |

The runner exception applies only to unavailable GB10 CI. It does not waive local failures or imply CUDA compilation/inference success. The serialized root gate covered the shared detection risk with full workspace/local CI and real dense/hybrid smoke tests; production WebUI startup and browser acceptance remain downstream.

## 5. Change Summary

| Area | Change |
|---|---|
| Detection authority | Shared dispatch with injectable restricted catalog probes |
| Catalog projection | Typed identity, metadata, support reasons, capabilities, removal guidance, filtering and pagination |
| Router adapters | Authenticated list/detail/refresh accessors, singleflight execution, redacted refresh failures |
| Existing provider integration | Current pool capability snapshot and single-model `AppState` accessor |
| Contracts | OpenAPI, generated TypeScript, and catalog/identity fixtures updated together |
| Tests | Whole-producer contract checks, restricted filesystem evidence, cached lifecycle/provider projection, zero-heavy HTTP cache hits, same-size refresh edits, reload/download invalidation, 1,000-entry HTTP traversal |
| Documentation | English/Korean integration guide and this pre-merge report |

Implementation commits: `57c1f268` adds the projection and shared probes; `70809247` adds required test license headers; `fafc5cf5` hardens pooling-parent evidence and adds full HTTP refresh pagination coverage; `6befcd26` restores raw bounded metadata fields and makes catalog metadata caching epoch-based.

## 6. Learning Points and Follow-up

A read-only observer needs its own restricted evidence-acquisition boundary, not an independent model taxonomy. Separately, a cache hit may reuse expensive metadata while still requiring fresh lifecycle truth, and an idempotent response does not automatically mean single execution.

Downstream integration must preserve null/reason semantics, use catalog IDs for control and inference IDs for requests, recheck mutating operations at their authority boundary, and validate production startup/security rather than extrapolating from authenticated handler tests.

See [catalog integration](../docs/webui/catalog.md), [API contract](../docs/webui/api.yaml), and [architecture](../docs/webui/architecture.md).
