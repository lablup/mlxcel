# Technical Report: PR #1868 — Model catalog projection

**Date**: 2026-09-12
**Status**: Open PR; focused validation complete, full workspace and real-model gates pending
**Languages**: Rust, JSON/OpenAPI, TypeScript, Markdown
**Risk Level**: Medium
**Implementation snapshot**: `fafc5cf52835b4e7cc12a708c2f924b7cc88d7d1`

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

The catalog projects `RouterPool::catalog_snapshot()` on blocking workers. Cached metadata is keyed by identity plus source/path/fingerprint/provider-capability facts, while current revision, generation, lifecycle, and removal status are reapplied. Disk accounting is bounded and reused; fingerprint checks still perform filesystem metadata work. The fingerprint is not a checksum of weight contents.

Default pagination is 50, with a 200-item page limit and 1,000-entry inventory bound. Results sort by opaque ID. Stability is guaranteed for an unchanged inventory, not as a cross-request transaction during concurrent mutations.

### Execution ownership is stronger than operation reuse

Refresh requests use one server-instance-scoped execution owner while the coordinator operation is active. Returning an accepted/replayed operation alone would not prevent multiple background rescans. Only the owner dispatches the rescan; terminal cleanup permits a later refresh. Signature comparison counts detected changes even if before/after inventory lengths match.

### Keep API availability separate from integration readiness

The authenticated router constructor exposes catalog handlers for tests and future secure startup; the normal constructor keeps UI APIs unmounted. The single-model accessor reads the existing `AppState` provider and actual inference ID, reports read-only removal guidance, and constructs no second provider. Browser security and production CLI integration remain #1837/#1838 responsibilities, while deletion execution belongs to #1841.

## 3. Review Refinements

Independent correctness and security reviews cleared the final implementation snapshot after these refinements:

- Bounded metadata reads and shared detection probes prevent fallback to unrestricted weight-header inspection; unknown reasons replace unsupported guesses.
- Filesystem evidence rejects symlinked sidecars and nested pooling parents. The final fix specifically rejects `1_Pooling` symlink traversal rather than checking only its leaf file.
- Metadata caching preserves fresh pool lifecycle/provider facts instead of replaying stale readiness from a cached entry.
- Refresh singleflight separates operation acceptance from execution ownership and reports same-count changes through signatures.
- Producer tests compare the complete serialized catalog shape against pinned schema-validated fixtures, normalizing only temporary-path-derived IDs/fingerprint and disk bytes.

Config/sidecar reads are limited to 256 KiB and index JSON to 512 KiB. Disk traversal limits are 4,096 visited/pending entries and depth 8; unavailable measurements return null with a reason. No benchmarked latency or memory-reduction claim is made.

## 4. Validation Record

The following results were reported by the implementation and independent review stages for the final scoped change; the documentation stage did not start builds, tests, or model execution.

| Gate | Result at report preparation |
|---|---|
| Catalog-focused tests | 21 passed |
| Final pooling-parent symlink regression | 1 passed |
| Full 1,000-entry HTTP pagination across refresh | 1 passed |
| Router catalog/refresh tests | 2 passed |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Passed |
| `cargo fmt --check` and `git diff --check` | Passed in implementation validation |
| `make verify-webui-contract` with the isolated verifier Python | Passed, 32 fixtures |
| `make verify-llama-compat verify-versions verify-kernel-dtype-keys` | Passed |
| Root full workspace/local CI and actual Llama + Granite regression | Pending at report preparation; not counted as passed |
| GB10-required CI | Runner reported down; user explicitly authorized skipping unavailable runner gates |
| CUDA execution, production `--webui`, browser acceptance | Not established by this change's scoped evidence |

The runner exception applies only to unavailable GB10 CI. It does not waive local failures or imply CUDA compilation/inference success. Full workspace and real dense/hybrid checks are required because shared model detection changed; the orchestrator owns those serialized gates and the final PR validation update.

## 5. Change Summary

| Area | Change |
|---|---|
| Detection authority | Shared dispatch with injectable restricted catalog probes |
| Catalog projection | Typed identity, metadata, support reasons, capabilities, removal guidance, filtering and pagination |
| Router adapters | Authenticated list/detail/refresh accessors, singleflight execution, redacted refresh failures |
| Existing provider integration | Current pool capability snapshot and single-model `AppState` accessor |
| Contracts | OpenAPI, generated TypeScript, and catalog/identity fixtures updated together |
| Tests | Whole-producer contract checks, restricted filesystem evidence, cached lifecycle, refresh ownership, 1,000-entry HTTP traversal |
| Documentation | English/Korean integration guide and this pre-merge report |

Implementation commits: `57c1f268` adds the projection and shared probes; `70809247` adds required test license headers; `fafc5cf5` hardens pooling-parent evidence and adds full HTTP refresh pagination coverage.

## 6. Learning Points and Follow-up

A read-only observer needs its own restricted evidence-acquisition boundary, not an independent model taxonomy. Separately, a cache hit may reuse expensive metadata while still requiring fresh lifecycle truth, and an idempotent response does not automatically mean single execution.

Before finalizing the PR, append the actual full-workspace and Llama/Granite results. Downstream integration must preserve null/reason semantics, use catalog IDs for control and inference IDs for requests, recheck mutating operations at their authority boundary, and validate production startup/security rather than extrapolating from authenticated handler tests.

See [catalog integration](../docs/webui/catalog.md), [API contract](../docs/webui/api.yaml), and [architecture](../docs/webui/architecture.md).
