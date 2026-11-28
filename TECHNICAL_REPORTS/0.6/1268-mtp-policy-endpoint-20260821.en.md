# Technical Report: PR #1268 - A supported read interface for the settled MTP verdict

## Executive Summary

`MtpPolicy` (issue #333) settles a per-pairing MTP verdict on the actual machine and persists it as a `PolicyHint` under `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<hash>.json`. That verdict is the only machine-specific answer to "does MTP help for this pairing here", and a host application wants to show it. There was no API, so Backend.AI GO (lablup/backend.ai-go#4441) shipped a reader that parses mlxcel's private cache files from another process.

This PR adds `GET /v1/internal/mtp-policy`, the Option A the issue preferred. The cache files are untouched, because a consumer already depends on them.

## 1. Problem Statement

Coupling to the on-disk format is fragile in a way neither project detects. `HINT_VERSION` bumps whenever the schema or the verdict semantics change, `HINT_SUBDIR` is a private constant, and `PolicyHint` is `pub(crate)`. Any of those can move in a patch release. The consumer degrades to a blank surface rather than crashing, which is worse for diagnosis: the user sees nothing and no error is raised anywhere.

The issue also asked for a state the file format cannot express. A missing file means "still profiling", "no MTP pairing", and "the cache root resolved to the wrong place" all at once.

## 2. Technical Decisions

### 2.1 Publish into observability rather than lock the policy

`MtpPolicy` is documented as single-threaded: the scheduler owns it on the worker thread and it needs no locking. An axum handler runs elsewhere. Wrapping it in `Arc<Mutex<..>>` would put a lock on a decode-hot path to serve a diagnostic endpoint.

Instead the scheduler publishes a `MtpPolicySnapshot` into the already-shared `Arc<BatchObservability>`, and the handler reads it. Publishing happens at worker startup and then only while the state can still move: the burst path republishes under a `was_profiling` guard captured **before** `record_b1_sample`, so the profiling-to-settled transition is itself captured. A guard captured after the call would have missed exactly that transition. Once settled or forced, the steady state pays nothing.

### 2.2 Report `forced` as its own state

The issue named three states. The code has four. `PolicyState::Forced` means `MLXCEL_ENABLE_MTP_B1` pinned the decision and nothing was measured on this machine. Flattening it into `settled` would let a consumer render "this machine measured MTP as worth it" for a value that came from an environment variable. `forced` therefore reports `verdict: null`, no acceptance rate and no samples.

The related invariant holds across restarts: `from_parts` checks the pin before the stored hint, and `record_b1_sample` early-returns for any non-`Profiling` state, so a forced run never persists a hint that a later unpinned start could load back as a measured verdict.

### 2.3 Make `unavailable` carry a reason

`no_mtp_dispatch`, `adaptive_disabled` and `worker_not_ready` are distinct situations, and collapsing them reproduces the exact ambiguity the issue complained about. Under `adaptive_disabled` the response still reports what the static gate (or an operator pin) decided via `mtp_enabled`, so the answer is useful rather than merely honest.

### 2.4 Publish the rounded acceptance rate on settle

`PolicyHint::new` rounds to two decimals before persisting. The settled state now takes `hint.acceptance_rate` rather than the raw quotient, so a consumer migrating from the file to the endpoint cannot see the two sources disagree by a rounding step. The value is taken from the hint even when `store.save` fails.

## 3. Change Summary

| File | Change |
| --- | --- |
| `src/server/routes/mtp_policy.rs` | New handler, `MtpPolicyResponse`, `MTP_POLICY_SCHEMA_VERSION = 1`; mapping extracted as a pure helper, mirroring `cache::build_stats_response` |
| `src/server/batch/mtp_policy.rs` | `MtpPolicyStatus`, `MtpPolicyUnavailableReason`, `MtpPolicySnapshot`, `snapshot()`, `is_profiling()`; `Settled` became a struct variant carrying the rounded rate and sample count |
| `src/server/batch/observability.rs` | `mtp_policy: Mutex<Option<MtpPolicySnapshot>>` plus accessors, and the field on `ObservabilitySnapshot` |
| `src/server/batch/scheduler.rs` | Publishes at `with_mtp_policy`, republishes under the `was_profiling` guard |
| `src/server/model_worker.rs` | The legacy `--no-batch` and OpenXLA workers publish `no_mtp_dispatch` at startup |
| `src/server/app.rs`, `routes/mod.rs`, `batch/mod.rs` | Route registration and re-exports |
| `docs/mtp-policy-api.md` | New reference: body, state and reason tables, versioning and compatibility policy |

## 4. Review Findings

Two independent review passes ran. Neither found a CRITICAL or HIGH issue, and both confirmed the properties that matter: an operator pin can never surface as a measured verdict, `record_b1_sample` is the only mutator of `PolicyState` and has a single call site so the published snapshot cannot go stale, the `Mutex` fails closed (poisoning surfaces as `unavailable` rather than a stale verdict), and the `/health` addition leaves the Prometheus text payload byte-unchanged.

A second commit (`0f624651`) addressed what they did find, all of it accuracy rather than behavior:

- `#[non_exhaustive]` on the two newly public enums. `docs/mtp-policy-api.md` promises the label sets will grow, and the repository already uses the attribute in about 94 places.
- The legacy `--no-batch` and OpenXLA workers never called `with_mtp_policy`, so the endpoint answered `worker_not_ready` for their whole process lifetime when `no_mtp_dispatch` was the honest answer. Both now publish at startup.
- Four false statements in the new documentation: the endpoint is not mounted on the `--node-role router` front-end; `--num-draft-tokens` does not re-key the policy (`--draft-block-size` does, and the false claim was copied from a source comment that is now corrected too); `PolicyHint` already carries `hardware`, so describing it as an addition was wrong; and `adaptive_disabled` credited the static gate in a case where an operator pin actually decided.
- The `profiling` state's accuracy limit is now stated rather than left implicit. `with_mtp_policy` publishes `profiling` from dispatch being **configured**, but the burst has later runtime gates. If those reject every burst, the endpoint reports `profiling` with `samples: 0` for the process lifetime and `mtp_enabled: true` is not true of that configuration. Documented deliberately instead of patched, because a fix needs design thought.

## 5. Validation

Measured on GB10 (DGX Spark, CUDA sm_121, Linux aarch64), MLX pin `9a795735`.

- `make verify-test-cuda` at `9ff0a3ad`: **8188 passed, 0 failed, 310 ignored**, exit 0. That is +17 against `main` at `a940c737` (8171), and the diff adds exactly 17 `#[test]` functions and removes none, so the counts reconcile rather than merely look plausible.
- `make verify-test-cuda` re-run at `0f624651`: recorded in the PR thread.
- `cargo fmt --all -- --check`, `cargo clippy --lib --tests --features cuda -- -D warnings`: clean.

The endpoint has not been exercised against a live server with a loaded MTP pairing. Route wiring is compile-checked through `create_app`; the response mapping and the publish/read seam are unit-tested.

## 6. Related Work

- #1257: the issue this closes. lablup/backend.ai-go#4441 is the consumer whose interim cache read this replaces.
- #333: the adaptive policy whose verdict is now readable.
- The on-disk hint format is deliberately unchanged. `schema_version` for the body starts at 1 and is independent of `HINT_VERSION = 3` for the file; the documentation calls that out, since a consumer reading both will see two different version numbers.
