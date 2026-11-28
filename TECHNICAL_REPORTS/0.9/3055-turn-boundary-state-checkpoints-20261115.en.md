# Technical Report: PR #3055 - checkpoint recurrent state at turn boundaries

**Date**: 2026-11-15
**Status**: PR open (draft, closes #3047)
**Reviewer**: -
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

Hybrid models (Qwen 3.5, Qwen3-Next, Qwen4-Exp) cannot rewind a GatedDeltaNet state, so a prompt-cache hit on an earlier turn needs a whole snapshot taken at exactly that turn. A whole snapshot at agent-chat length is 1 to 2 GB, almost all attention KV, and the session-chain supersede rule keeps only the newest history boundary per conversation. An edit one to three turns back therefore found nothing closer than the system-prompt root and re-prefilled about 23K tokens.

The PR stores only the recurrent and convolution state at every history boundary (a checkpoint, 151 MB to 449 MB in the replays) and rebuilds the full state on a hit from that checkpoint plus the conversation's newest whole snapshot (the donor). The donor is restored, its attention KV is cut back to the checkpoint length with the speculative-rollback trim, and the checkpoint's recurrent state is installed. The rebuild is exact only when the donor's leading rows were computed by the same sequences that produced the checkpoint, so every snapshot now carries a lineage and a checkpoint is served only over a donor whose lineage agrees. Checkpoints have their own byte budget, TTL, LRU and orphan removal.

On the M3 Ultra live-replies replay, an edit one turn back drops from 17.1 s to 1.7 s TTFT on `qwen3.5-4b-4bit` (16 GB limit) and from 87.0 s to 7.9 s on `qwen3.8-27b-4bit` (32 GB limit). Three turns back, 16.8 to 5.2 s and 85.6 to 25.0 s. Peak footprint also falls, because the off arm prefills about 23K tokens from the system root. The PR also fixes a Qwen4-Exp QSA detail that the checkpoint path exposed: block summaries that a direct prefill of the shorter prefix would not have pooled must be dropped after the cut.

---

## 1. Problem Statement

### 1.1 Background

Transformer attention state is positional: the KV rows for a prefix are the leading rows of the KV for any longer sequence sharing that prefix, so a cache can be trimmed to a shorter prefix. A GatedDeltaNet layer folds every consumed token into a fixed-size state, so it cannot be trimmed. For hybrid models the prompt cache therefore stores whole-state snapshots (KV plus recurrent state) at history boundaries, and a hit requires a snapshot of exactly the matched prefix.

The store's session-chain supersede rule keeps one history-boundary snapshot per conversation, the newest (#3035 and #3037 added `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` to keep one more). Snapshot sizes at 28K tokens are 1 to 1.8 GB, so keeping one per turn does not fit the default budget.

### 1.2 Existing Issues

- A request that edits turn N-1, N-2 or N-3 of a long conversation has no whole snapshot at the edited boundary and falls back to the system-prompt root (4,706 cached tokens on the 4b replay, 4,744 on 27b).
- The cost is large: A9 TTFT of 17 s on 4b and 87 s on 27b, and a peak footprint of 25.7 GB and 51.7 GB, because the edited turn prefills about 23K tokens.
- Almost all of the snapshot bytes are data that a longer snapshot of the same conversation already holds (the leading KV rows). Only the recurrent state is irreplaceable.

### 1.3 Risk Assessment

| Risk | Mitigation |
|------|------------|
| Mixing a checkpoint's recurrent state with KV rows computed by a different prefill partition gives a state no single run produces | Lineage check: the donor must share the checkpoint's producers over the checkpoint's length |
| Extra resident memory per conversation | Separate byte budget (default 12 checkpoints, capped at an eighth of free memory), `0` disables |
| Cross-session or cross-template leakage | Checkpoint and donor lookup require the same `SessionlessBucketKey` and the snapshot lookup's session rule |
| Qwen4-Exp QSA index differing from a direct prefill after the cut | `forget_unpooled_blocks`, plus bit-for-bit tests over four QSA regimes |

---

## 2. Technical Review

### 2.1 Security

Isolation holds. `best_state_checkpoint` and `donor_for` both require the request's `SessionlessBucketKey` (model, LoRA, template, multimodal digest, RoPE regime) and the same session rule as the snapshot lookup, check token-prefix containment, and require lineage agreement. Sequence ids come from a monotonic counter, so lineages never collide. Multimodal requests store no boundary snapshots and use a different digest bucket, so they never reach a checkpoint. The review notes that the checkpoint branch skips the `require_whole_entry` multimodal gate; no path reaches it today, and a guard would be defense in depth.

### 2.2 Performance

- Insert cost: a checkpoint is derived from the boundary snapshot with `ModelStateSnapshot::subset`, which shares the source tensors' buffers. No copy is made at insert.
- Lookup cost: one extra read-locked pass over the checkpoints, then a write lock to bump counters and touch entries.
- Memory: once the donor snapshot is superseded, the checkpoint's recurrent tensors stay resident on their own. Bounded by the checkpoint budget (151 MB at 6 entries on 4b, 449 MB on 27b at the end of the replay).
- Insert-time orphan drop is O(checkpoints x same-session snapshots x prompt length) under the store write lock (see section 8).

### 2.3 Compatibility & Dependencies

- No new dependencies. New env var `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES` (documented in `docs/environment-variables.md` and `docs/CONTINUOUS_BATCHING.md`); `/v1/cache/stats` gains `state_checkpoint_*` counters.
- `LanguageModel` gains two defaulted trait methods, `state_checkpoint_from_snapshot` (default `None`) and `restore_sequence_state_from_checkpoint` (default `Err`), so every other family stays on whole snapshots. The VL wrappers (`qwen3_5_vl`, `qwen4_exp_vl`, `minicpmv4_6_vl`) and `LanguageModelRef` forward them.
- `ModelSnapshotEntry` gains a `lineage` field, defaulted empty (matches nothing).

### 2.4 Code Quality

- The Qwen4-Exp linear-layer restore was extracted into `restore_linear_layer`, shared by the whole-snapshot and checkpoint paths.
- The cut reuses `KVCache::trim` and the Qwen4-Exp rollback trim instead of adding a new truncation path.
- `same_session` in `state_checkpoints.rs` duplicates the inline session match in `snapshot_lookup` (LOW, from review).

---

## 3. Technical Decisions

### 3.1 Checkpoint plus donor, not more whole snapshots

| Option | Outcome |
|--------|---------|
| Raise the snapshot budget or set `KEEP_PREVIOUS_TURN` | Costs a gigabyte per kept turn; 2,946 MB of whole snapshots on Flash-Next against 1,191 MB plus 662 MB of checkpoints |
| Store a snapshot per turn with copy-on-write KV | Would need paged or shared KV for snapshots, a much larger change |
| Store recurrent state only and borrow KV from the newest snapshot | Chosen: tens to hundreds of MB per turn, and reuses the trim and restore code |

### 3.2 Lineage instead of trusting token equality

Equal tokens do not imply equal rows. A chained hit forwards the history in different pieces than a cold run, which moves bf16 values by a few ULP (the existing partition effect). Cutting a donor whose leading rows came from a different partition and pairing them with the checkpoint's recurrent state would produce a state that no single run produces. `SnapshotLineage` records `(end, sequence id)` spans for each snapshot's rows. A restore copies rows verbatim, so an adopting sequence inherits the adopted snapshot's lineage clipped to the matched length and attributes every row it computes to its own id. `shares_prefix(other, len)` requires both lineages to cover `len` rows and be equal when clipped to `len`; an empty lineage never matches. Lineage follows a sequence through adoption, warm-up, queued re-prefill and handoff.

### 3.3 Separate budget, TTL, LRU and orphan removal

Checkpoints sit in their own map, so whole snapshots never push them out and they never push snapshots out. A checkpoint with no qualifying donor is useless, so each checkpoint insert drops checkpoints that no stored snapshot can rebuild (a snapshot of exactly the checkpoint's tokens counts, since its successor will donate). The default budget is 12 checkpoints of the model's recurrent state (float32 state, windows at the stream dtype, plus the PLE window for `qwen4_exp`), clamped to an eighth of the memory left after the model, which is half of the snapshot store's quarter because this budget is additional.

### 3.4 Lookup preference

A checkpoint is served only when it reaches further than any whole or partial snapshot, so a whole snapshot always wins. It must also stop short of the full request (`len < tokens.len()`), because prefill re-runs at least the last prompt token for its logits and a recurrent state cannot be rewound by one. A checkpoint restore failure falls back to a cold prefill and does not retry a shorter whole-snapshot hit (LOW, from review).

### 3.5 Qwen4-Exp: forget unpooled QSA blocks after the cut

The QSA indexer pools block summaries only once a query sees more complete blocks than `block_topk`. `index.trim` removes the summaries of blocks that are no longer complete, but a donor prefilled to a long length may hold pooled summaries that a direct prefill of the shorter prefix would never have computed. `QsaAttention::forget_unpooled_blocks` sets `block_keys = None` when `index.len() / compress_ratio <= block_topk`, so the cut cache equals what a direct prefill of the shorter prefix writes. `DecoderLayer::trim_cache_to_prefill_state` combines the rollback trim with this step. Without it, the four QSA regimes in the fixture tests do not match a direct prefill.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
Boundary snapshot insert (scheduler):
  snapshot --subset(recurrent tensors, shared buffers)--> checkpoint
  lineage = adopted_lineage.extended(len, this seq id)
  store.insert_snapshot(snapshot, lineage); store.insert_state_checkpoint(checkpoint, lineage)

Lookup (store.snapshot_lookup):
  best whole/partial snapshot -> reach
  best_state_checkpoint(prefix of request, len > reach, len < request) + donor_for(lineage shares prefix)
  -> CheckpointHit { checkpoint, donor, matched_len }

Adopt (scheduler.adopt_state_checkpoint):
  model.restore_sequence_state_from_checkpoint(seq, checkpoint, donor)
    attention layers: restore donor, trim to matched_len (+ QSA index, forget unpooled blocks)
    linear layers: install checkpoint state
  adopted_lineage[seq] = checkpoint.lineage
```

### 4.2 Key Code Changes

**Models** (`qwen3_5.rs`, `qwen3_next.rs`, `qwen4_exp.rs`, `qwen4_exp_snapshot.rs`, `qwen4_exp_qsa.rs`, VL wrappers): `Qwen3NextCache::is_checkpoint_tensor` keeps tensors whose name contains `.linear.`; Qwen4-Exp keeps everything except `.kv.` and `.qsa.` tensors (so the layer table, PLE window and n-gram contexts stay). `restore_from_checkpoint` and `checkpoint_caches` validate family, lengths and layer layout and return `Err` before installing anything.

**Core** (`generate.rs`, `language_model_ref.rs`): `ModelStateSnapshot::subset` and the two trait methods.

**Store** (`prompt_cache/store.rs`, `state_checkpoints.rs`, `lineage.rs`, `entry.rs`, `policy.rs`, `snapshot_sizing.rs`): the checkpoint map and counters inside `Inner`, `insert_state_checkpoint`, `best_state_checkpoint`, `donor_for`, `drop_orphan_state_checkpoints`, TTL sweep, LRU cap, `SnapshotLookupOutcome::CheckpointHit`, `clear()` coverage, the config field and the model-aware sizing (`recommend_state_checkpoint_capacity`).

**Scheduler** (`scheduler/prompt_cache.rs`, `queued_adoption.rs`, `handoff.rs`, `admission.rs`, `mod.rs`): `adopt_state_checkpoint`, checkpoint derivation in `insert_model_state_snapshot` for `Boundary` origin, lineage propagation, and startup wiring in `startup.rs` (operator override wins, else the model-aware default).

### 4.3 Data Model Changes

`ModelSnapshotEntry.lineage`, `PromptCacheConfig.state_checkpoint_capacity_bytes`, and five new `PromptCacheStats` fields (`state_checkpoint_entries`, `_bytes`, `_inserts`, `_hits`, `_evictions`). Nothing is persisted.

---

## 5. Learning Points

### 5.1 Separate what is rewindable from what is not

**Concept:** a hybrid model's state is two kinds of data. Positional rows (KV, QSA index) are prefixes of longer states and can be borrowed. Recurrent summaries cannot. Storing only the second kind per turn turns a gigabyte-per-turn problem into a hundred-megabyte one.

**Application in this PR:** the checkpoint is the non-rewindable subset, and the donor supplies the rest.

### 5.2 Equal tokens are not equal state

Under bf16, the way a history was chunked through prefill changes the rows by a few ULP. Any scheme that splices state from two runs has to track which computation produced each part. The lineage structure does that with a span list, which is cheap to clip and compare.

### 5.3 The partition effect explains the `--verify` misses

On `qwen3.5-4b` the off arm already loses A6 and A8 against cold runs, and the on arm additionally loses A9 (and A10 after it). A chained run that held the whole snapshot at 25,625 tokens (`MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN=1`, 128G) produces the same 16 completions byte for byte as the checkpoint run, A9 and A10 included. The checkpoint therefore reproduces the snapshot it replaces; the difference from cold is the documented chained-hit partition effect, which moves from one turn to another when the hit point moves. The tiny-fixture tests pin the same equality bit for bit.

---

## 6. Further Learning

### Key Terms
| Keyword | Description | Relevance |
|---------|-------------|-----------|
| GatedDeltaNet | Linear-attention layer with a fixed-size recurrent state | State that cannot be rewound |
| Donor | Newest whole snapshot of the conversation | Supplies attention rows for the checkpoint's prefix |
| Lineage | Per-span record of which sequence computed rows | Guards against mixing partitions |
| Partition effect | Different prefill chunking moves bf16 logits by a few ULP | Cause of `--verify` misses vs cold |
| QSA | Qwen4-Exp block-pooled sparse attention index | Needs `forget_unpooled_blocks` after a cut |

### Related PRs/Issues
- #3047: this feature (closed by this PR)
- #3035, #3037: keep-previous-turn restore point
- #3040: output-identity criterion for hybrid cache hits
- #2883: snapshot growth guardrail (not applied to checkpoints, see section 8)

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 86 (45 non-JSON, 41 replay result files) |
| Lines added | +9,176 (+2,434 outside the raw replay JSON) |
| Lines deleted | -70 |
| New test files | 4 (`qwen3_5_checkpoint_tests`, `qwen4_exp_checkpoint_tests`, `scheduler_state_checkpoint_tests`, `state_checkpoints_tests`), plus sizing, lineage and policy tests |

### Changes by Category

| Category | Summary |
|----------|---------|
| Feature | Checkpoint derivation, lineage, donor rebuild, own budget, lookup preference, scheduler adoption |
| Bug fix | Qwen4-Exp `forget_unpooled_blocks` for QSA cuts |
| Docs | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, replay results |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `5a22cf3` | feat | checkpoint recurrent state at turn boundaries |
| `6a21ac2` | docs | add the turn-boundary checkpoint replay results |

---

## 8. Follow-up Actions

### Required
- [ ] (MEDIUM) Checkpoint bytes are outside `store.bytes()` and `store.len()` (`store.rs`), so the prompt-cache gauges under-report them, and the #2883 growth guardrail covers only the snapshot insert, not `insert_state_checkpoint`. Memory is bounded by the checkpoint budget (an eighth of free memory by default) and released by `clear()`, but an operator reading the gauges will not see it.
- [ ] (MEDIUM) `drop_orphan_state_checkpoints` holds the store write lock while comparing every checkpoint with every snapshot (token `starts_with` plus two `clipped` allocations per pair) on each insert. Cheap at about 12 entries, but it grows as checkpoints x same-session snapshots x prompt length when the env override raises the budget.

### Monitoring Required
- `state_checkpoint_hits`, `_evictions` and `_bytes` in `/v1/cache/stats` on long agent conversations.
- Hit rate versus the default of 12 checkpoints when many conversations share a server.

### Future Improvements
- (LOW) Orphans are dropped only on checkpoint insert; if snapshot LRU or TTL removes a donor, dependent checkpoints wait for their own TTL or LRU.
- (LOW) Add a `require_whole_entry` guard to `adopt_state_checkpoint`.
- (LOW) Pre-existing: when a stored snapshot covers the whole request (regenerating an identical prompt), `reach` equals the prompt length and hybrids cold-prefill.
- (LOW) The env override is not clamped against free memory, matching documented operator semantics.
- (LOW) Share `same_session` with `snapshot_lookup`.

---

## Appendix

### A. Test Results

- Tiny-fixture tests: the rebuilt state equals a direct prefill of the shorter prefix bit for bit for qwen3_5 and qwen4_exp (four QSA regimes), and so do the decode logits after it and a suffix prefilled over it. Mismatched donors are refused.
- Scheduler test: an edit two turns back adopts the checkpoint and decodes the same as a reference that held the whole snapshot; with no budget it starts over.
- Targeted lib tests (prompt_cache, scheduler, qwen4_exp, qwen3_5, qwen3_next) and `cargo clippy --lib --tests -D warnings` pass, per the PR. No tests were run while writing this report.

### B. Replay Results (M3 Ultra)

`--turn-tokens 3072 --edit-turn 9 --live-replies`, off = checkpoint budget 0 (behavior of main), 2 to 3 reps per cell interleaved off/on in one GPU-lock hold. Build is main at `ab45ac59` plus #3047.

| Model, limit, edit | A9 cached off / on | A9 TTFT off / on | Checkpoint bytes | Peak footprint off / on |
|---|---|---|---|---|
| qwen3.5-4b, 16G, 1 turn back | 4,706 / 25,625 | 17.05-17.38 / 1.70-1.71 s | 151 MB | 25.7 / 13.5 GB |
| qwen3.5-4b, 16G, 3 turns back | 4,706 / 19,895 | 16.69-16.86 / 5.22-5.25 s | 151 MB | 27.2 / 17.3 GB |
| qwen3.8-27b, 32G, 1 turn back | 4,744 / 25,885 | 87.04-87.07 / 7.89-7.98 s | 449 MB | 51.7 / 33.5 GB |
| qwen3.8-27b, 32G, 3 turns back | 4,744 / 20,081 | 85.58-85.64 / 24.96-24.98 s | 449 MB | 52.7 / 36.4 GB |

Whole-snapshot bytes are unchanged (1,059 MB on 4b, 2,201 MB on 27b). Total replay TTFT: 4b 43.4 to 28.0 s (1 back) and 43.0 to 31.6 s (3 back); 27b 219.4 to 140.2 s and 218.0 to 157.3 s. Peak footprint comes from runs without `--verify` and includes weights and prefill activations. Load average at start was 1.2 to 7.7, with two 4b runs at 21.3 and 14.0; spread across reps stayed within 0.4 s of total TTFT.

Identity under `--verify` against cold runs:

| Model, edit | off match cold | on match cold |
|---|---|---|
| 4b, 1 turn back | 14 of 16 (A6, A8 differ) | 12 of 16 (A6, A8, A9, A10) |
| 4b, 3 turns back | 14 of 16 (A6, A8) | 13 of 16 (A6, A8, A9) |
| 27b, 1 and 3 turns back | 14 of 16 (B2, B3) | 14 of 16 (B2, B3) |
| Flash-Next, 1 turn back | n/a | 16 of 16 |

On Flash-Next both the checkpoint arm and the keep-previous-turn arm match all 16 cold runs; the checkpoint arm holds 1,191 MB of whole snapshots plus 662 MB of checkpoints, against 2,946 MB of whole snapshots in the keep-previous-turn arm. The 4b on-arm misses are the partition effect, shown byte-identical to a whole-snapshot run (section 5.3). Not measured: Flash-Next at three turns back, and any model other than these three.

### C. References
- Issue #3047
- `docs/benchmark_results/hybrid-turn-state-checkpoints-replay-m3ultra-2026-11-15.md`
- `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`
