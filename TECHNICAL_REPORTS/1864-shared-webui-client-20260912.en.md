# Technical Report: PR #1864 — Shared WebUI client and state authority

**Date**: 2026-09-12
**Status**: Pre-merge implementation review complete; downstream application integration remains
**Languages**: TypeScript, TSX, Rust, Python, JSON-compatible YAML
**Risk Level**: Medium
**References**: [PR #1864](https://github.com/lablup/mlxcel/pull/1864), [issue #1842](https://github.com/lablup/mlxcel/issues/1842), [epic #1834](https://github.com/lablup/mlxcel/issues/1834)

## Executive Summary

This change establishes one browser transport layer and one headless React state authority for the bundled WebUI. It adds strict runtime validation against the canonical contract, authenticated fetch-based SSE, snapshot reconciliation and documented page hooks, while extending the existing Rust coordinator with paired numeric replay cursors. It does not mount production pages, introduce a second model registry, or claim completed browser/hardware acceptance.

## 1. Problem Statement

Models, Chat, Activity and Settings need the same answers about authentication, model identity, lifecycle state and pending operations. Independent page fetch wrappers or timers would duplicate credentials, trigger overlapping observations, and disagree after reconnects or server restarts. A transport failure after a load/download POST is particularly dangerous: retrying without knowing whether the server accepted it can execute a second user action.

Snapshots are not captured simultaneously. Replaying from the latest global event can skip an operation transition that happened after the operations snapshot but before a newer catalog snapshot. Opaque event IDs cannot safely be compared numerically, and a cursor must be interpreted together with its server instance.

## 2. Technical Decisions

### Canonical validation at the boundary

`webui/src/api/jsonSchema.ts` evaluates the supported canonical schema constructs; `validation.ts` exposes typed validators rather than maintaining a second handwritten DTO shape. The frontend test reads all 33 shared fixture files, including scenarios, identities and string catalogs, not merely selected happy-path response properties. Negative cases exercise unknown fields, missing required nullable fields, malformed timestamps and incorrect discriminators. Schema/generated declaration inputs now participate in the deterministic bundle source digest.

This approach avoids a new runtime validation dependency, but the evaluator is deliberately scoped to the canonical schema in this repository, not a general JSON Schema implementation. Future schema constructs need corresponding evaluator and independent negative-test coverage.

### One authenticated transport

`WebUiApiClient` accepts a validated same-origin path prefix, sends bearer authentication in headers, rejects redirects and external API bases, and keeps the credential only in memory. A 401 clears authentication and aborts outstanding work; a 403 remains a forbidden response. Bounded JSON reads and incremental SSE parsing constrain buffering. The fetch transport supports Authorization without putting credentials into an EventSource URL.

The same transport supports `/v1/chat/completions` and `/v1/responses`. Inference hooks resolve the opaque catalog ID to its request-facing `inference_id`, force streaming and `autoload=false`, and preserve raw content, reasoning, tool and usage frames for the later Chat presenter. Rendering, persistence and tool execution are not introduced here.

### Per-resource fences and atomic replay subscription

The reducer fences catalog snapshots, operation snapshots/records, model revisions and runtime snapshots independently. The synchronizer resumes from the minimum held sequence together with `server_instance_id`, not the maximum observed event. Paginated catalog/operation reads must agree on both server instance and snapshot sequence before being applied as a consistent collection.

Rust replay validation, coordinator subscription, schema, generated TypeScript and the request fixture change together. Paired query parameters are the primary cursor; an opaque `Last-Event-ID` remains available only as the legacy alternative. Missing, duplicate, conflicting, unsafe-integer and future cursors produce typed errors; a different instance or retained-ring gap produces a resnapshot signal. Subscription and replay are coordinated so events cannot be lost between reading history and subscribing to live updates.

### Shared headless hooks, not page-local stores

`WebUiProvider`, `useWebUi` and `useWebUiActions` own the authentication session, selected model, snapshots, operation reconciliation and stream teardown. Model actions resolve at acceptance, not readiness; pages render eventual completion from shared state. Unknown POST outcomes are reconciled against operations using the original idempotency key rather than resubmitted, with unresolved outcomes exposed after a 60-second reconciliation window.

One non-overlapping refresh loop uses 2 seconds while visible and 30 seconds while hidden. Reconnect scheduling is bounded and jittered; generation checks suppress obsolete work after teardown/restart, and model selection/logout abort outstanding transports. `lastUpdatedAt` records state transitions, while the separate nullable `lastSuccessfulAt` advances only when validated current-session data snapshots/events are accepted. Heartbeats, invalidation-only events and errors do not make stale data appear fresh; same-instance gaps/errors retain the previous timestamp, and logout/server replacement clears it. The architecture guide documents both timestamps and the complete public hook surface.

## 3. Review and Validation

Independent correctness and security reviews completed after three fix cycles without remaining findings, as recorded by the issue implementation workflow. The root froze Rust revision `c735bc99` for its full workspace gate. Documentation finalization exposed an acceptance gap in last-successful-data age; the follow-ups `cfabec5c` and `e728c9c9` add a separate timestamp, exclude unapplied notifications and add regression tests in TypeScript only, without changing Rust sources. The final freshness delta received independent approval. The finalizer itself changed documentation only.

| Check | Result and scope |
|---|---|
| `pnpm --dir webui run typecheck` | Passed strict TypeScript checking. |
| `pnpm --dir webui run lint` | Passed ESLint with zero warnings. |
| `pnpm --dir webui run unit` | 44 tests passed across 7 files, including full fixture validation, SSE byte splitting/CJK/CRLF, auth/abort, reducer, pagination, reconciliation, provider cleanup and successful-data freshness. |
| `pnpm --dir webui run verify-generated` | Two clean temporary builds matched each other and the committed bundle; no checked-in asset rewrite was needed during finalization. |
| `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | 33 fixtures, generated DTO drift and strictness/negative checks passed. |
| `make verify-llama-compat verify-versions verify-kernel-dtype-keys` | All passed with the isolated Python environment prepended to PATH. |
| `cargo fmt --check` | Passed. |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | Root executed the full workspace gate at frozen Rust revision `c735bc99`: 11,167 passed, 0 failed, 361 ignored across 123 summaries. Ignored tests are not passes. |
| `cargo test --profile test-fast --features webui ui_events` | Implementation validation: 3 selected Rust route tests passed. |
| `cargo test --profile test-fast --features webui lifecycle_coordinator_sequence_replay` | Implementation validation: 2 selected Rust coordinator tests passed. |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | Implementation workflow reported a passing scoped post-fix lint gate. |
| `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` | Root reported the full workspace/all-target gate passed at `c735bc99`; Rust sources are unchanged by the TypeScript-only freshness follow-up. |

The manifest reports 68,364 bytes of initial/total JavaScript gzip and 227,718 embedded asset bytes. These are the current scaffold's measurements: the new headless modules are not mounted into the scaffold and these numbers are not a prediction of the final Models/Chat application. No runtime performance improvement, GPU inference, actual Safari/VoiceOver, CUDA execution or production authentication end-to-end result is claimed by these checks.

The user confirmed the GB10 runner is down and explicitly authorized proceeding on passing local CI while unavailable GB10 jobs are skipped. This is a runner-availability waiver only: no CI workflow or branch protection changes were made, and unexecuted CUDA validation must not be reported as passed.

## 4. Change Summary

| Area | Durable change |
|---|---|
| `webui/src/api/` | Typed same-origin client, bounded streaming parser, canonical runtime validation and transport tests. |
| `webui/src/state/` | Shared provider/actions, resource-fenced reducer, synchronization/reconciliation loop and deterministic tests. |
| `src/server/router_lifecycle*.rs`, `router_server*.rs` | Paired sequence replay, atomic subscription and route/coordinator regressions. |
| `docs/webui/api.yaml`, generated declarations and replay fixture | Coordinated replay query contract and safe numeric cursor constraints. |
| Bundle script/config/manifest | Schema input loading and deterministic source-digest coverage without adding runtime dependencies. |
| `docs/webui/architecture.md` | Complete hook contract, inference exceptions and explicit headless integration/freshness boundaries. |

## 5. Learning Points and Follow-up

- A reconnect cursor is a lower bound across independently observed resources, not simply the newest event received. Test the transition that lies between two snapshots.
- Cancellation requires generation/session fencing as well as an AbortController; a late promise can otherwise restore state after logout or teardown.
- Generated types do not validate network input. Whole-fixture runtime checks and independent invalid-field mutations are complementary gates.
- Production startup/authentication, mounted login/schema-error views, page presenters, actual browser accessibility and hardware acceptance remain assigned to later epic units. Root README/CLI installation guidance is not expanded to advertise an unimplemented `--webui` startup; final user-facing documentation belongs to #1849.
