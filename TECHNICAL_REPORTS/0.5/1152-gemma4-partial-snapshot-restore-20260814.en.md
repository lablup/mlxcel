# Technical Report: PR #1152 - feat(models): restore Gemma 4 snapshots at the longest common prefix

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (opt-in per family; the default capability answer is `false`, so recurrent and dense families keep exact-prefix semantics by construction)

---

## Executive Summary

The snapshot path required a stored entry to be an exact prefix of the incoming request, so an entry sharing a long common prefix but diverging before its own end was useless. For rotating-attention families that is stricter than the hardware demands: the "cannot truncate" constraint only binds after a sliding layer's ring buffer has wrapped. Before wrap the ring is linear and truncation is mechanically identical to the dense case, which is exactly the condition upstream mlx-lm's `can_trim_prompt_cache` checks.

This change converts exact-prefix into longest-prefix matching for Gemma 4 while leaving every recurrent family untouched. The capability is expressed as `LanguageModel::snapshot_truncatable_to`, defaulting to `false` so a family opts in explicitly; Gemma 4 answers per layer from the snapshot's own scalars. The store adopts the best diverging candidate at the longest common prefix once it clears `min_prefix_tokens`, and a refusal remains the classified `snapshot_diverged` reject from #1147. Real-checkpoint validation on gemma-4-e2b-it-4bit caught a genuine bug the synthetic fixture could not reach: KV-shared layers, which store nothing, failed the truncating restore against an offset of 0.

---

## 1. Problem Statement

### 1.1 Background

`PromptCacheStore::lookup_snapshot_prefix` demanded the stored token vector be an exact prefix of the request. Epic #1148 showed that on snapshot-only families the stored vector routinely shares a long prefix with the next turn and then diverges on a template artifact or retokenization drift, so the entry is discarded whole. Gemma 4 runs 60 layers in a 5:1 sliding:full pattern with `sliding_window = 1024` on the 31B checkpoint, so for short and medium conversations every layer is still linear and the divergence point is mechanically reachable by truncation. Upstream mlx-lm already encodes this: `can_trim_prompt_cache` permits trimming only while the rotating cache is unwrapped.

### 1.2 Existing Issues

- **Long common prefixes were thrown away.** A candidate matching 90 of its 139 stored tokens contributed nothing; the request re-prefilled from zero.
- **The constraint was applied at the wrong granularity.** "Rotating caches cannot truncate" is true only post-wrap and per layer; treating it as a family-wide absolute forfeited every short-conversation case.
- **Retokenization drift was unrescuable.** Small divergences near the end of an otherwise matching prefix (the falcon-h1 class: 120 sampled versus 118 re-tokenized reply tokens) lost the entire entry.
- **Recurrent families must not be dragged in.** GatedDeltaNet and SSM state cannot be truncated to an arbitrary boundary at all; any design that inferred capability from heuristics in the store risked corrupting them.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A wrapped or non-linear cache is truncated anyway, silently corrupting generation | High | Low (two independent failure modes each excluded and tested) |
| A recurrent family attempts a truncating restore | High | Low (default capability is `false`; opt-in per family) |
| The adopted-prefix state differs from a cold prefill of the same prefix | High | Low (output-parity test is the correctness bar) |
| Dense-KV families are perturbed | Low | Low (they never enter the snapshot path; verified live) |

---

## 2. Technical Review

### 2.1 Linearity Has Two Failure Modes, Both Excluded

`RotatingKVCacheSnapshotState::is_unwrapped` / `can_truncate_to` and `RotatingKVCache::is_trimmable` (mirroring upstream on the live cache) exclude two independent conditions: a wrapped ring, where `idx` no longer tracks `offset`, and the subtler trap where `update_concat` leaves an oversized temporary buffer for one step after an over-window prefill. The second keeps `idx == offset`, so a naive unwrapped check passes, but the next in-place update re-slices from the back and would misplace data after a truncation. The unit suite drives a live cache through its wrap and pins both refusals.

### 2.2 The Capability Is a Model Answer, Not a Store Heuristic

`LanguageModel::snapshot_truncatable_to` and `restore_sequence_state_truncated` default to `false` / `Err`, so a family that says nothing keeps exact-prefix semantics. Gemma 4 answers per layer from the snapshot's own scalars, and `restore_sequence_state_truncated` re-checks rather than trusting the caller. The store asks the capability through a closure and asks about the adopted length, not the stored length, which a dedicated test pins. This is the issue's requirement made structural: future hybrid families opt in explicitly or not at all.

### 2.3 The Store Change Is Small and Ordered

`lookup_snapshot_outcome` adopts the best diverging candidate at the longest common prefix once it clears `min_prefix_tokens`. An exact-prefix candidate always wins over a longer-common-prefix diverging one, and a capability refusal degrades to exactly the #1147 classified reject, so the diagnostic surface stays truthful: "diverged and unadoptable" is still visible, now with the reason being a wrapped cache rather than a missing mechanism.

### 2.4 The Bug Only a Real Checkpoint Could Find

The first gemma-4-e2b run failed with `layer15: Gemma 4 truncate: target 34 exceeds cached offset 0`. Gemma 4's KV-shared layers compute only Q and borrow K/V from an earlier layer, so `snapshot_into` skips them and `restore_from` leaves them fresh; the truncation guard `target_len <= offset` then failed against offset 0 on a restore that was actually sound. A single-layer synthetic fixture cannot represent a layer that stores nothing by design. The fix is a `Cache::is_populated` guard making the empty layer a no-op, and the regression test covers the negative half too, so the no-op cannot widen into a blanket bypass.

---

## 3. Technical Decisions

### 3.1 Follow the mlx-lm Precedent Rather Than Invent a Policy

Upstream's `can_trim_prompt_cache` embodies the same physical fact: unwrapped rings are linear. Mirroring it keeps the Rust implementation diffable against the reference and imports a condition that has already survived production use.

### 3.2 Per-Layer Check at the Target, Not a Conversation-Length Heuristic

A global "under 1024 tokens" rule would be both too strict (full-attention layers always truncate) and fragile (the window differs per checkpoint; e2b uses 512). The check reads each sliding layer's scalars at the truncation target, which is the only formulation that stays correct across the 5:1 layer pattern and across checkpoints.

### 3.3 Exact-Prefix Candidates Keep Priority

Partial adoption is a rescue, not the preferred path. When both an exact-prefix entry and a longer diverging one exist, the exact one wins, avoiding a truncating restore (and its extra checks) whenever a plain restore suffices.

### 3.4 The Correctness Bar Is Output Parity, Not Counter Movement

`gemma4_truncated_restore_matches_a_cold_prefill_of_the_same_prefix` requires that a restore truncated to N decodes identically to having only ever prefilled those N tokens. Counters prove the path ran; parity proves it ran correctly.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 10 |
| Lines | +927 / -33 |
| New `LanguageModel` methods | 2 (`snapshot_truncatable_to`, `restore_sequence_state_truncated`, both with safe defaults) |
| Families opted in | 1 (Gemma 4, including both VLM wrappers) |
| Recurrent-family behavior changes | 0 (verified by negative control) |

### Changes by Area

**`src/lib/mlxcel-core/src/cache.rs`**
- `RotatingKVCacheSnapshotState::{is_unwrapped, can_truncate_to}`; `RotatingKVCache::is_trimmable` mirroring upstream; the over-window `update_concat` trap excluded.

**`src/lib/mlxcel-core/src/generate.rs`**
- `LanguageModel::snapshot_truncatable_to` / `restore_sequence_state_truncated` defaulting to `false` / `Err`.

**`src/models/gemma4.rs`**
- Per-layer capability from snapshot scalars, `Cache::truncate_to`, `Cache::is_populated` for KV-shared layers, re-checking truncated restore.

**`src/server/prompt_cache/store.rs`, `src/server/batch/scheduler.rs`**
- `lookup_snapshot_outcome` takes the capability as a closure and adopts at the longest common prefix; the scheduler routes a `matched_len` shorter than the stored entry to the truncating restore.

**`src/loaded_model.rs`, `src/vision/gemma4_unified.rs`, `src/vision/gemma4_vl.rs`**
- Delegation so the capability survives every wrapper.

---

## 5. Validation and Follow-up

### Passed

- `mlxcel-core::cache::rotating_truncation_tests`: 9 passed, covering the inclusive boundary at `offset == max_size` (accepted) versus one token later (refused), a wrapped cache refused even for a target inside its live window, the over-window prefill trap, buffered speculative mode, non-FP16 storage, and a live cache driven through its wrap.
- `models::gemma4_tests::snapshot_prompt_cache`: 7 passed, including the cold-prefill parity test and the KV-shared empty-layer regression.
- `server::prompt_cache`: 165 passed, covering partial adoption, exact-prefix precedence, the `min_prefix_tokens` floor, a declining model keeping exact-prefix semantics, and capability queried at the adopted length.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`.
- Real checkpoints through `/v1/chat/completions`, counter-verified on `/v1/cache/stats`: gemma-4-e2b-it-4bit turn 2 adopts at 77 (`matched=77 stored=129 partial=true`) and a deliberately diverged turn at 66 (`matched=66 stored=168 partial=true`), both strictly shorter than the stored entry, which is impossible on the previous code. qwen3.5-0.8b-4bit is the load-bearing negative control: zero `partial=true` occurrences, `snapshot_hits` pinned at 0, the `snapshot_diverged` reject firing instead. llama-3.2-1b shows `snapshot_lookups = 0` throughout, so dense-KV families never touch this code.

### Not Covered

- The gemma-4-31b-it-4bit scenario named in the issue's acceptance criteria was exercised on gemma-4-e2b-it-4bit instead, the same rotating-attention family and identical code path under a 512-token window; the 31B run belongs to the orchestrator's large-model pass.
- Wall-clock benefit of partial adoption was not measured; the change is validated on correctness and counters.

### Follow-up

- The wrapped-decline reject currently reuses the `snapshot_diverged` classification; if operators need to distinguish "diverged, unadoptable because wrapped" from "diverged, no capability", a sub-reason can be added on the #1147 surface.
- Future rotating or hybrid families gain partial restore by implementing the two `LanguageModel` methods; the defaults guarantee nothing changes for a family that does not.
