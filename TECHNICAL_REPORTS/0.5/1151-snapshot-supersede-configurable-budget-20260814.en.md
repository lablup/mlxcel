# Technical Report: PR #1151 - feat(server): supersede session snapshot chains and expose the budget

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (additive store rule plus new knobs; defaults unchanged, dense-KV families untouched)

---

## Executive Summary

The exact-prefix snapshot store carried a fixed 512 MiB budget and no notion that a conversation's newer snapshot replaces its older one. Every turn of a multi-turn conversation donates a snapshot whose token vector extends the previous turn's, so the store accumulated the whole chain and left LRU to pick victims under byte pressure. With 31B-class snapshots measured at 300-370 MB, the second turn's insert evicted the first turn's entry, and two concurrent conversations evicted each other.

This change adds a session-chain supersede rule at insert time: a new snapshot removes every stored snapshot from the same session whose token vector is a strict prefix of the incoming one, before the new entry's bytes are accounted. A conversation's steady-state footprint becomes one entry regardless of turn count. The three existing `PromptCacheConfig` snapshot fields (capacity bytes, max entries, TTL) are wired through CLI flags and `MLXCEL_*` env vars on both server binaries, and a new `snapshot_supersedes` counter on `/v1/cache/stats` keeps deterministic replacement distinguishable from genuine budget pressure.

The issue's "model-aware default budget" sub-requirement was deliberately deferred with a documented hook point rather than half-built; see Technical Decisions.

---

## 1. Problem Statement

### 1.1 Background

Epic #1148 established that snapshot-only families (Gemma 4, Qwen 3.5, SSM hybrids; thirteen families reporting `supports_snapshot_reuse()`) never hit the multi-turn prompt cache. Issue #1146 covers the capacity half: even once matching is fixed by the sibling issues, the store's budget hygiene would thrash. `DEFAULT_SNAPSHOT_CAPACITY_BYTES` is `DEFAULT_CAPACITY_BYTES / 4` = 512 MiB, one 31B conversation snapshot is roughly 300-370 MB (observed 307,693,200 and 369,133,200 bytes), and there was no operator override on either binary.

### 1.2 Existing Issues

- **Chains accumulated.** Each turn's donated vector strictly extends the last, so a conversation held one entry per turn. All but the newest were dead weight: a lookup can only match the longest, yet each still counted against the byte budget.
- **LRU picked the wrong victim.** Under pressure the LRU rule could evict the entry being extended just as easily as an unrelated one. The observed `snapshot_inserts = 2, snapshot_entries = 1` on a 31B model was the second turn evicting the first.
- **No operator control.** Capacity, entry count, and TTL existed on `PromptCacheConfig` but were unreachable from the CLI or environment, unlike the sibling `prompt_cache_*` knobs.
- **Undiagnosable stats.** Every removal surfaced as `snapshot_evictions_lru`, so an operator could not tell healthy in-session replacement from real budget pressure.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Two concurrent 31B conversations thrash each other's snapshots | High | High (before this change) |
| Supersede removes an entry another request still matches (mid-conversation fork) | Low | Low |
| Anonymous-session sharing lets one caller supersede another's snapshot | Low | Low |

---

## 2. Technical Review

### 2.1 The Supersede Rule Is Deliberately Narrow

`insert_snapshot` removes a stored entry only when three conditions hold at once: same sessionless bucket (model, LoRA, template, multimodal identity), same non-`None` resolved `session_key`, and the stored token vector is a strict prefix of the incoming one. A `None` session carries no conversation identity to chain on and never triggers the rule. Two different sessions never touch each other, which the tests pin as cross-session isolation.

The anonymous-session case was examined rather than assumed away: requests that set neither `prompt_cache_key` nor `user` share `ANONYMOUS_SESSION_SENTINEL`, exactly as they already do in `lookup_longest_prefix`. For one such caller to supersede another's entry, its whole stored vector, prompt plus generated tail, must strictly prefix the other's, which in practice only holds for a genuine continuation of the same transcript.

### 2.2 Ordering Against the Capacity Check

The supersede runs before the new entry's bytes are accounted, so `enforce_snapshot_caps` sees the freed bytes and can admit an extension that would not have fit alongside its own ancestor. This ordering is what turns the 512 MiB budget from "fits one 31B snapshot, thrashes on the second turn" into "fits one 31B conversation, steady-state". A dedicated test proves the byte accounting by inserting an extension that only fits if the ancestor's bytes are freed first.

### 2.3 The Counter Split Is a Diagnostic Contract

`snapshot_supersedes` is held apart from `snapshot_evictions_lru` end to end (store internals, `PromptCacheStats`, `/v1/cache/stats`). A supersede is deterministic in-session replacement; an LRU eviction is capacity pressure. Folding them together would make the eviction counter unreadable for the exact tuning workflow the new flags enable. An attribution control was run live: an identical-request repeat advances `snapshot_inserts` and the LRU counter while `snapshot_supersedes` stays 0, so the new counter does not double-count idempotent replacement.

### 2.4 One Accepted Trade-off

A mid-conversation fork (an edited or regenerated earlier turn) loses its dropped ancestor and pays one re-prefill. This is the price of keeping a conversation's footprint at one entry, and it is recorded in the code comment at the rule rather than left for the next reader to rediscover.

---

## 3. Technical Decisions

### 3.1 Supersede at Insert, Not a Smarter Eviction Policy

The alternative was teaching LRU about chains at eviction time. Insert-time removal is strictly simpler: the information needed (which entries this one supersedes) is fully available at insert, the freed bytes benefit the very insert that triggered it, and eviction stays a pure capacity mechanism.

### 3.2 The Model-Aware Default Was Deferred, Honestly

Proposed Solution (b) asked to derive the default budget from the model's per-token state footprint. The PR documents a usable hook point (`start_server` constructs the store with `model_path` in hand, next to code that already reads `config.json` for the hybrid-SSM APC auto-disable) and names what is missing: a validated per-family state-size formula for thirteen families. Getting that formula wrong silently mis-sizes the store, so the default stays 512 MiB and the operator-facing need is covered by the knobs plus sizing guidance keyed to the measured, model-width-constant `snapshot_bytes`. An earlier draft justified the deferral with a config-timing constraint that review showed to be false; the PR body carries the correction.

### 3.3 Flags Mirror the Existing Pattern Exactly

The three flags and env vars follow the sibling `prompt_cache_*` precedent on both binaries (`src/main.rs`, `src/bin/mlx_server.rs`), including CLI-beats-env precedence and unparseable-value fallback, so the surface stays learnable and the tests could reuse the established fixtures.

### 3.4 Review Findings Were Folded In, Not Deferred

pr-reviewer and pr-security-checker ran independently; zero CRITICAL/HIGH. Both MEDIUMs were the same class of defect in the docs: sizing guidance phrased per conversation where the budget is actually consumed per entry, which would have caused the exact thrash this issue removes. Fixed in `ead5ac9f`, `8af676e4`, `3efebe30` along with a LOW pinning `snapshot_supersedes` on the wire format.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 14 |
| Lines | +609 / -4 |
| New CLI flags / env vars | 3 / 3 (both binaries) |
| New stats fields | 1 (`snapshot_supersedes`) |
| New store tests | 5 |
| New cli_input tests | 5 |
| Default behavior changes | 0 (supersede only fires with a session key and a strict-prefix chain) |

### Changes by Area

**`src/server/prompt_cache/store.rs`**
- Session-chain supersede in `insert_snapshot`, before byte accounting; `snapshot_supersedes` counter threaded through `Inner` and `stats()`.

**`src/server/prompt_cache/policy.rs`, `src/server/routes/cache.rs`**
- `snapshot_supersedes` on `PromptCacheStats` and `/v1/cache/stats`.

**`src/server/cli_input.rs`, `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`**
- `--prompt-cache-snapshot-capacity-bytes`, `--prompt-cache-snapshot-max-entries`, `--prompt-cache-snapshot-ttl` with `MLXCEL_*` env fallbacks, wired through `build_prompt_cache_config` via `with_snapshot_limits(...)`.

**`src/server/prompt_cache/entry.rs`**
- `ModelSnapshotEntry::new_for_test` (test-only), so budget tests control accounted size without allocating MLX tensors.

**`docs/environment-variables.md`**
- The three variables and sizing guidance: measure `snapshot_bytes`, multiply by expected resident entries.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --release --lib prompt_cache::store::tests`: 28 passed (5 new: chain collapse to one entry, cross-session isolation, no supersede without a session key, strict-extension requirement, byte accounting before the capacity check).
- `cargo test --release --lib server::cli_input`: 98 passed (5 new).
- `cargo test --release --lib server::routes::cache`: 17 passed.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt` clean.
- Real checkpoint (`qwen3.5-0.8b-4bit`, port 18146): flag path live end to end (startup log and `/v1/cache/stats` report the configured 40 MB against the compiled 512 MiB default); snapshot size constant at 13,246,464 bytes across turn vectors of 72, 98, 124, 151 tokens, confirming cost tracks model width, not prompt length; the reported thrash reproduced at HEAD under the small budget (`snapshot_evictions_lru` climbing while `snapshot_entries` stays pinned).

### Not Covered

- End-to-end supersede over HTTP. Measured, not assumed: donated vectors end with a generation prompt plus generated content that the next turn's re-rendered prompt does not reproduce, so no same-session strict extension reaches the store until the sibling boundary-snapshot work (#1143/#1144) lands. The rule itself is covered at the store layer.
- Two-conversation coexistence over HTTP, for the same reason.
- The model-aware default (deferred, see 3.2).

### Follow-up

- Once #1143/#1144 land, the steady-state claim (one resident entry per conversation, `snapshot_supersedes` advancing turn by turn) becomes observable over HTTP and belongs in the epic-level verification.
- A per-family state-size formula validated against measured `snapshot_bytes` would complete the model-aware default at the documented `start_server` hook.
