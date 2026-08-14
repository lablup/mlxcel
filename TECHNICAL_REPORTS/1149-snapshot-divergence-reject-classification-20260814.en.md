# Technical Report: PR #1149 - feat(server): classify snapshot-divergence rejects in prompt-cache stats

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (observability only; every existing lookup call site keeps its exact signature and semantics)

---

## Executive Summary

A snapshot candidate sitting in the request's own session bucket but diverging from the request before the entry's end used to come back from `lookup_snapshot_prefix` as a bare `None`. On `/v1/cache/stats` that made the structural multi-turn miss of epic #1148, present on every snapshot-only family probed, indistinguishable from an empty store. Diagnosing it required manual token-level detokenization to discover that a candidate existed, shared a long common prefix, and diverged on a template artifact.

This change makes the store report what it saw. A new `SnapshotLookupOutcome` distinguishes `Hit`, `Diverged` (carrying `common_prefix_len` and `stored_len`), and `NoCandidate`; the scheduler records a classified `snapshot_diverged` reject for the middle case only; and the geometry surfaces on `/v1/cache/stats` (`reject_snapshot_diverged`, `last_reject_context_len` / `last_reject_entry_len`) and as a `reason="snapshot_diverged"` series on `/metrics`. Verified live on two families on different hybrid paths, including a negative control where a genuine hit keeps the counter at zero.

---

## 1. Problem Statement

### 1.1 Background

Epic #1148's scope verification found the multi-turn prompt cache missing on every snapshot-only family probed (gemma-4-31b, qwen3.5-4b in both thinking modes, falcon-h1-tiny), each miss from a different divergence class: generation-prompt-only scaffold tokens, thinking blocks stripped from history, and retokenization drift. In every case `snapshot_lookups` advanced while `snapshot_hits` stayed at zero, which is also exactly what a cold store looks like. The store had the information (it walked the candidate and found the divergence point) and threw it away.

### 1.2 Existing Issues

- **The most common failure mode had no signal.** For thirteen families, "candidate exists but diverges" is the default outcome of turn 2, and it produced no counter, no reason, and no geometry anywhere on the stats surface.
- **Diagnosis cost was disproportionate.** Confirming the epic's misses meant dumping stored token vectors and detokenizing by hand to locate the divergence point. The 90/139 geometry on gemma-4-31b was found this way.
- **Downstream work needed the classification.** The partial-restore issue (#1145) needs to report "diverged but cannot truncate" as an outcome distinct from "no candidate at all"; without an outcome type there was no place to put it.
- **False classification was a live risk.** A naive implementation could count a cold store or a foreign session bucket as a divergence, poisoning the very signal being added.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Operators tune or file bugs against a cache they cannot distinguish from broken | High | High (before this change) |
| The new reject fires on empty stores or foreign sessions, making the counter unreadable | Medium | Low (pinned by tests) |
| Changing the lookup path alters hit behavior for existing callers | Medium | Low (old entry point kept as a wrapper) |

---

## 2. Technical Review

### 2.1 An Outcome Type, Not a Second Boolean

`lookup_snapshot_outcome` returns `SnapshotLookupOutcome::{Hit, Diverged(SnapshotDivergence), NoCandidate}` with `SnapshotDivergence { common_prefix_len, stored_len }`. The three-way split is the design: `NoCandidate` deliberately covers both the empty store and the foreign session bucket, so the caller cannot record a divergence for a request that never had a candidate. `lookup_snapshot_prefix` remains as a thin wrapper with its old signature, so every caller that only wants hits is untouched.

### 2.2 The Geometry Is Computed for Free

The common-prefix length falls out of the comparison the lookup already performs; no extra pass over the token vectors is added. When several candidates share the bucket, the best (longest common prefix) candidate's numbers are reported, matching what an operator would want to know: how close the nearest entry came.

### 2.3 The Recording Path Reuses the Reject Plumbing

`PromptCacheRejectReason::SnapshotDiverged` rides the existing `PromptCacheRejectCounters` mechanism. `PromptCacheLastReject` gains `entry_len` through a new `record_detailed`, and the existing `record` delegates with `None`, so no prior call site changes behavior. In the scheduler, `try_adopt_cached_prefix` consumes the outcome and records the classified reject only for `Diverged`.

### 2.4 The Negative Control Is the Load-Bearing Evidence

Class (c) retokenization drift is data-dependent: a reply may or may not re-tokenize identically. On falcon-h1-tiny the turn-2 reply did re-tokenize identically, the lookup correctly returned a hit, and `reject_snapshot_diverged` stayed at zero. That row proves the classifier moves only when no stored entry prefixes the request, not whenever the snapshot path runs. The cold-store row proves an empty store emits nothing.

---

## 3. Technical Decisions

### 3.1 Classify at the Store, Record at the Scheduler

The store computes the outcome because only it sees the candidates; the scheduler records it because only it knows a real request is being served. This split keeps test-driven and internal lookups from inflating operator-facing counters.

### 3.2 Geometry Over a Bare Counter

The issue required carrying `common_prefix_len` and `stored_len`, not just a count, and the live rows show why: 33/95 on qwen3.5 and 23/139 on falcon-h1 immediately localize the divergence to just past the history boundary, which is the diagnosis that previously needed detokenization. `last_reject_context_len` / `last_reject_entry_len` expose the most recent geometry without a log scrape.

### 3.3 The Epic's Verified Scenarios Became the Fixtures

`snapshot_divergence_tests.rs` pins the three divergence classes at the token-vector level the store compares, including the 90/139 gemma-4-31b geometry from the epic's live verification, plus the no-false-classification cases. The regression suite therefore encodes the real-world shapes, not synthetic ones.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 9 |
| Lines | +702 / -21 |
| New stats fields | 3 (`reject_snapshot_diverged`, `last_reject_entry_len`, `reason="snapshot_diverged"` on `/metrics`) |
| Behavior changes to existing lookups | 0 (`lookup_snapshot_prefix` wraps the new outcome) |
| New test file | 1 (`snapshot_divergence_tests.rs`, 343 lines) |

### Changes by Area

**`src/server/prompt_cache/store.rs`**
- `SnapshotDivergence`, `SnapshotLookupOutcome`, and `lookup_snapshot_outcome` reporting the best diverging same-session candidate; `lookup_snapshot_prefix` kept as a wrapper.

**`src/server/prompt_cache/metrics.rs`**
- `PromptCacheRejectReason::SnapshotDiverged` (label `snapshot_diverged`); `PromptCacheLastReject.entry_len` and `record_detailed`.

**`src/server/batch/scheduler.rs`, `src/server/batch/observability.rs`**
- `try_adopt_cached_prefix` consumes the outcome; `record_prompt_cache_reject_detailed`, the `prompt_cache_reject_snapshot_diverged` counter, and `entry_len` on the last-reject snapshot.

**`src/server/routes/cache.rs`, `src/server/routes/metrics.rs`**
- `reject_snapshot_diverged` and `last_reject_entry_len` on `/v1/cache/stats`; the `snapshot_diverged` series on `/metrics`.

**`src/server/prompt_cache/snapshot_divergence_tests.rs`** (new)
- The three verified divergence classes plus the no-false-classification cases.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --release --lib server::prompt_cache`: 153 passed.
- `cargo test --release --lib server::routes::cache`: 19 passed (route-level coverage of the new fields).
- `cargo test --release --lib server::batch::observability`: 15 passed.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`.
- Real checkpoints through the production `/v1/chat/completions` path: qwen3.5-0.8b-4bit (Attention + GatedDeltaNet) cold store emits nothing; turn 2 with `cached_tokens = 0` records reject 1 with geometry 33/95. falcon-h1-tiny-90m-instruct-4bit (Mamba hybrid): turn 2 is a genuine hit and the counter stays 0 (the negative control); an early-divergence turn records geometry 23/139. Two families on different hybrid paths both classify, so the reject is family-agnostic.

### Not Covered

- The gemma-4-31b-it-4bit scenario named in the issue was not run live on this branch; the orchestrator's large-model pass owns it. Its 90/139 geometry is pinned as a unit-test fixture instead.
- Wall-clock impact was not measured; the added work is a comparison already performed plus counter increments.

### Follow-up

- #1145's partial restore reports "diverged but wrapped, cannot truncate" through the outcome type this change introduced; that landed as PR #1152.
- The reject reason and its label are now a stable contract: the epic's verification and future regression checks key off `snapshot_diverged`, so renaming it is a breaking change to the diagnostic surface.
