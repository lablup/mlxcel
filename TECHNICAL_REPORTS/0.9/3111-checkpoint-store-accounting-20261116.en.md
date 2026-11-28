# Technical Report: PR #3111 - count state checkpoints and bound the orphan sweep

**Date**: 2026-11-16
**Status**: PR open (closes #3058)
**Reviewer**: independent review, no CRITICAL, HIGH or MEDIUM findings
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

PR #3055 added turn-boundary state checkpoints for hybrid models. That PR left four known gaps, filed together as #3058. Checkpoint bytes were invisible to the store's `len()`, `bytes()` and stats, and the #2883 snapshot growth guardrail did not cover them. The orphan sweep compared every checkpoint with every snapshot on each insert. Checkpoints stayed resident after their donor snapshot was evicted. A checkpoint whose rebuild failed sent the request to a cold prefill instead of a shorter whole-snapshot hit.

This PR fixes all four in one change. Checkpoints now count in the store footprint and pass through `snapshot_growth_allowed`. The orphan sweep is scoped to the conversation that changed. Orphans are dropped together with their donor. A failed rebuild discards the checkpoint and retries the lookup without checkpoints, and a checkpoint hit is counted only after the rebuild succeeds. The change is 6 files, +496 and -63 lines, with six new tests.

---

## 1. Problem Statement

### 1.1 Background

A checkpoint is the recurrent and convolution state at a history boundary. It is served only together with a donor, the conversation's newest whole snapshot, whose attention rows supply the prefix KV. A checkpoint without a qualifying donor is useless, so the store drops such orphans.

### 1.2 Existing Issues

- Checkpoint bytes sat outside `store.bytes()` and `store.len()`, so the prompt-cache gauges under-reported them, and `insert_state_checkpoint` skipped the growth guardrail.
- `drop_orphan_state_checkpoints` ran on checkpoint insert only, and compared every checkpoint against every snapshot under the write lock.
- LRU eviction, TTL expiry and cap enforcement of a donor left its checkpoints in place until their own TTL or LRU.
- When `adopt_state_checkpoint` failed, the request cold-prefilled even if a shorter whole snapshot existed. The lookup also counted the hit before the rebuild, so a failed rebuild was counted as a hit.

### 1.3 Risk Assessment

| Risk | Mitigation |
|------|------------|
| The supersede path could orphan checkpoints that the longer successor would donate to | `insert_snapshot` uses `remove_snapshot_unswept` and sweeps once after the new entry is stored |
| Scoping the sweep could miss an orphan in another conversation | Only a mutation in a conversation can change that conversation's donors, so other buckets and sessions are never affected |
| A retry loop on a persistently failing checkpoint | The failing checkpoint is discarded before the retry, and the retry lookup excludes checkpoints |
| The gauges now include checkpoints, which changes what operators read | Documented in `docs/CONTINUOUS_BATCHING.md` |

---

## 2. Technical Review

### 2.1 Security

No change to isolation. The scoped sweep takes the sessionless bucket and session of the mutation and passes only matching checkpoints to `donor_for`, which keeps the existing bucket and session rules. Checkpoints of other buckets and sessions are never read by the sweep, and a test pins this.

### 2.2 Performance

- The sweep cost drops from all checkpoints to the checkpoints of one conversation, still under the store write lock.
- The retry path runs one extra lookup only after a failed rebuild.
- No new allocation on the hit path.

### 2.3 Compatibility & Dependencies

No new dependencies or env vars. `stats().entries` and `bytes` now include checkpoints. New store methods: `state_checkpoint_growth_bytes`, `discard_state_checkpoint`, `record_state_checkpoint_hit`, `lookup_snapshot_outcome_without_checkpoints`, `remove_snapshot_unswept`.

### 2.4 Code Quality

The growth check reuses `snapshot_growth_allowed` rather than adding a second gate. The scheduler refreshes the gauges after the checkpoint insert instead of before it.

---

## 3. Technical Decisions

### 3.1 Count checkpoints in the existing totals

Checkpoints are added to `len()`, `bytes()` and the stats entries and bytes fields instead of exposing a parallel set of totals. Every consumer of the footprint, including the growth guardrail and the gauges, then sees one number. `state_checkpoint_growth_bytes` mirrors `snapshot_growth_bytes`, and a full checkpoint budget reports zero growth.

### 3.2 Scope the sweep to the changed conversation

A donor is found within a sessionless bucket and a session, so a mutation can only orphan checkpoints in that bucket and session. `drop_orphan_state_checkpoints` takes both and probes only those.

### 3.3 Sweep from `remove_snapshot`, but not mid-supersede

`remove_snapshot` runs the scoped sweep, which covers LRU, TTL and cap removal. In `insert_snapshot`, the supersede and same-key replacement removals use `remove_snapshot_unswept`, because sweeping there would drop checkpoints before the successor snapshot is stored. One sweep runs after the new entry is in place.

### 3.4 Retry without checkpoints after a failed rebuild

On a restore failure the scheduler calls `discard_state_checkpoint` and re-runs the lookup with `lookup_snapshot_outcome_without_checkpoints`, which adopts the best shorter whole snapshot. `CheckpointHit` is no longer counted by the lookup. The scheduler calls `record_state_checkpoint_hit` after a successful rebuild, so a failed rebuild plus the retry counts as one lookup.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
insert_snapshot:
  supersede / replace -> remove_snapshot_unswept
  store new entry
  one scoped sweep (bucket, session)

remove_snapshot (LRU, TTL, cap):
  remove entry -> scoped sweep (bucket, session)

scheduler, checkpoint hit:
  adopt_state_checkpoint ok   -> record_state_checkpoint_hit
  adopt_state_checkpoint fail -> discard_state_checkpoint
                              -> lookup_snapshot_outcome_without_checkpoints -> adopt whole snapshot
```

### 4.2 Key Code Changes

**Store** (`prompt_cache/store.rs`, `state_checkpoints.rs`): footprint accounting, `state_checkpoint_growth_bytes`, the scoped `drop_orphan_state_checkpoints`, `remove_snapshot` and `remove_snapshot_unswept`, `discard_state_checkpoint`, `record_state_checkpoint_hit`, `lookup_snapshot_outcome_without_checkpoints`.

**Scheduler** (`scheduler/prompt_cache.rs`): growth check before `insert_state_checkpoint`, gauge update after it, failed-rebuild retry.

**Docs** (`docs/CONTINUOUS_BATCHING.md`): the new sweep, retry and accounting behavior.

### 4.3 Data Model Changes

None. No persisted state, no new config.

---

## 5. Learning Points

### 5.1 Derived entries need the same accounting as their sources

Anything that holds memory must be in the totals that gate memory growth. A side map that skipped `bytes()` also skipped the guardrail.

### 5.2 Sweep at the point the dependency changes

An orphan appears when a donor leaves, so the sweep belongs in the donor removal path. A supersede is a removal followed by a successor, so it needs the sweep after the successor, not between.

### 5.3 Count an event when it succeeds

Counting the hit in the lookup made a failed rebuild look like a hit. Counting after the rebuild keeps the counters aligned with what served the request.

---

## 6. Further Learning

### Key Terms
| Keyword | Description | Relevance |
|---------|-------------|-----------|
| Checkpoint | Recurrent state at a history boundary | The entry this PR accounts for |
| Donor | Newest whole snapshot of the conversation | Its removal orphans checkpoints |
| Orphan | Checkpoint with no qualifying donor | Dropped by the scoped sweep |
| Supersede | A longer snapshot replaces the previous one in a session chain | Must not orphan checkpoints |

### Related PRs/Issues
- #3058: this PR's issue (closed by it)
- #3055, #3047: the checkpoint feature
- #2883: snapshot growth guardrail, now applied to checkpoints

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +496 |
| Lines deleted | -63 |
| New tests | 6 (5 in the prompt-cache and scheduler test files, 1 growth test) |

### Changes by Category

| Category | Summary |
|----------|---------|
| Bug fix | Footprint accounting, growth guardrail, scoped orphan sweep, orphan drop with donor, failed-rebuild retry |
| Tests | Accounting, sweep scope, donor eviction, hit counting and discard, failed rebuild, growth gate |
| Docs | `CONTINUOUS_BATCHING.md` |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `255ad02` | fix | count state checkpoints and bound the orphan sweep |

---

## 8. Follow-up Actions

### Required
None. The independent review found no CRITICAL, HIGH or MEDIUM issues.

### Monitoring Required
- `entries` and `bytes` in `/v1/cache/stats` now include checkpoints, so expect a step up in both after deploy.
- `state_checkpoint_hits` and `state_checkpoint_evictions` on long agent conversations.
- Any recurrence of the SIGSEGV described in Appendix A.

### Future Improvements
- (LOW) Observability counters see a failed rebuild plus its retry as two events.
- (LOW) `discard_state_checkpoint` does not refresh the gauges itself.
- (LOW) `state_checkpoint_growth_bytes` overestimates growth near the reserve, which errs on the safe side.
- (LOW) `donor_for` still scans all snapshots for each checkpoint.
- (LOW) The scoped-sweep test relies on a test-only recorder.

---

## Appendix

### A. Test Results

- `cargo test --lib server::prompt_cache::`: 215 passed.
- `cargo test --lib server::batch::`: 612 passed, 10 ignored.
- `state_checkpoint` tests: 21 passed.
- Mutation check: disabling the retry and the `remove_snapshot` sweep made 3 of the new tests fail.
- One `server::batch::` run crashed with SIGSEGV in unrelated model-owned cache tests. It did not reproduce in two later runs. The cause was not investigated, and it is listed under monitoring.
- No tests were run while writing this report.

### B. References
- Issue #3058
- `TECHNICAL_REPORTS/0.9/3055-turn-boundary-state-checkpoints-20261115.en.md`
- `docs/CONTINUOUS_BATCHING.md`
