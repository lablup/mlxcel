# Technical Report: PR #1150 - feat(server): donate a history-boundary snapshot during prefill

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium (on-by-default extra forward and state copy on qualifying prefills; gated to snapshot-reuse families, kill switch provided, dense-KV families verified unchanged)

---

## Executive Summary

Snapshot-only families (every model reporting `supports_snapshot_reuse()`, thirteen families) reuse a prompt cache only through an exact-prefix match against a stored token vector. The vector donated at end of generation is `prompt + generated`, and epic #1148 measured that vector failing to prefix the next turn on every family tested, for three independent reasons that all live past the history boundary: templates append generation-prompt-only scaffolds, templates strip thinking blocks when re-rendering an assistant turn as history, and a sampled token sequence is not the canonical tokenization of its own text.

This change adds a second snapshot, captured during prefill at the history boundary and keyed by the tokenization of the `add_generation_prompt = false` render itself. Because the donated vector is produced by tokenizing the re-rendered history form, it is a prefix of every follow-up turn by construction, neutralizing all three divergence classes at once. Measured A/B against `main` at 9c154ff3 on qwen3.5-0.8b-4bit: turn 2 `cached_tokens` from 0/189 to 150/189, turn 3 from 0/214 to 184/214, `snapshot_hits` from 0 to 2; the llama-3.2-1b dense-KV control is identical in every cell across arms.

Rebasing onto the concurrently merged siblings surfaced a genuine cross-issue interaction: #1151's session-chain supersede deleted every boundary snapshot moments after creation. It was fixed here by tagging snapshots with a `SnapshotOrigin` and chaining per producer.

---

## 1. Problem Statement

### 1.1 Background

For the snapshot-only class, `PromptCacheStore::lookup_snapshot_prefix` is the only reuse path, and it demands the stored token vector be an exact prefix of the incoming request. Live verification on 2026-08-14 (main at 9c154ff3) showed turn 2 `cached_tokens = 0` on gemma-4-31b (169-token prompt), qwen3.5-4b in both thinking modes (123 and 177), and falcon-h1-tiny (250), while the llama-3.2-1b control hit 256 of 308 on the same binary. The store, key composition, and adopt paths all worked; the failure was specific to what the snapshot path stores.

### 1.2 Existing Issues

- **Class (a): generation-prompt-only tokens.** gemma-4-31b appends an empty thought scaffold only to the generation prompt, never to history; token-level proof showed the stored 139-token entry sharing exactly 90 with turn 2, the four divergence tokens detokenizing to the scaffold. qwen3.5 with `enable_thinking=false` has the same shape via its empty `<think>\n\n</think>\n\n` injection.
- **Class (b): thinking stripped from history.** qwen3.5's default mode primes `<think>\n` in the generation prompt but re-renders assistant history without the block, so the vectors diverge right after the assistant header.
- **Class (c): retokenization drift.** falcon-h1-tiny has a plain ChatML template and still missed: 120 sampled completion tokens versus 118 tokens when the same reply text is re-tokenized. Sampled sequences are not canonical tokenizations.
- **The shared property.** All three divergences happen after the history boundary. Everything rendered with `add_generation_prompt = false` is prefix-stable across turns regardless of template quirks, which is the invariant this change is built on.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Thirteen families never hit the multi-turn cache and pay full prefill every turn | High | Certain (before this change) |
| The extra forward or offset advance corrupts a prefill path (chunked, batched, speculative, VLM, distributed) | High | Low (each entry point traced; declines leave prefill unchanged) |
| Splitting one forward into two shifts greedy output on near-ties | Low | Medium (documented #203/#325/#326 class, already the status quo under chunked prefill and cache hits) |
| Dense-KV deployments pay a wasted second render and encode | Low | Certain (accepted, follow-up #1153) |

---

## 2. Technical Review

### 2.1 The Key Invariant: the Vector Comes from the Live Prompt

The snapshot's token vector is always `prompt_tokens[..boundary]`, taken from the live prompt rather than from the history render. `resolve_history_boundary` clips the history vector to the longest common prefix it shares with the live prompt tokens; the clip is what makes the vector a genuine prefix of the state the snapshot describes, and it also drops any token a BPE merge across the history/scaffold seam would have made unstable. No template output or message content can make the key describe state it does not match.

### 2.2 One Renderer, One Snapshot Producer Pair

`apply_history_with_kwargs` renders the same message list with `add_generation_prompt` forced off, and the existing entry points delegate to the same inner render, so there is one renderer and no behavior change for existing callers. On the store side, the capture-and-insert previously inlined in `donate_finished_sequence_cache` became `insert_model_state_snapshot`, shared by the boundary capture and the end-of-generation donate, so the existing donate keeps behaving exactly as before.

### 2.3 Cost Accounting

`capture_history_boundary_snapshot` runs one forward over `prompt_tokens[prefill_start_offset..boundary]`, snapshots there, inserts, and advances `prefill_start_offset`; both `execute_full_prefill` and `start_chunked_prefill` then process the remaining suffix unchanged. Total tokens forwarded are unchanged; the cost is one extra graph launch plus the state copy. A review finding removed a real memory hazard here: the segment forward originally evaluated its whole `[1, segment_len, vocab]` logits block, several GB of peak at 20k tokens and a 150k vocab, and now evaluates only the final position and clears the allocator cache. Re-validation produced byte-identical hit numbers, the proof the captured state did not change.

### 2.4 The Cross-Issue Break Found at Rebase

#1151's supersede rule removes stored snapshots whose token vector is a strict prefix of an incoming same-session one. A turn's history-boundary vector is always a strict prefix of that same turn's completion vector (`history + scaffold + sampled reply`), so every boundary snapshot was deleted by its own turn's completion donate. Measured on the rebased build: turn 2 back to 0 of 189 with `snapshot_inserts` still 6 and `snapshot_evictions_lru` 0, proving creation and non-capacity removal simultaneously. The supersede premise "longer same-session vector replaces shorter" holds within one producer but not across the two: the completion vector is longer and almost never usable, the boundary vector shorter and always usable. Snapshots now carry a `SnapshotOrigin` and the rule chains only within one origin; `ModelSnapshotEntry::new` defaults to `Completion` so every pre-existing caller keeps its behavior. Post-fix, the probe returned to 150/189 and 184/214 with four live entries (two chains of one).

### 2.5 Gating and Failure Behavior

The boundary path is reached only for `supports_snapshot_reuse()` models, on text-only requests, with the store enabled, and when the clipped boundary clears `min_prefix_tokens` and is strictly shorter than the prompt. The history render is dropped when the primary render fell back to `render_simple_fallback` (the fallback is not the template, so its history form describes a prompt the model never saw). An eval failure in the extra forward aborts the sequence exactly like existing prefill eval failures; every other decline leaves the request prefilling as before. `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` restores the pre-#1143 prefill without a rebuild, and after a review fix it also skips the second render and encode, reproducing the `main` baseline cell for cell.

---

## 3. Technical Decisions

### 3.1 Snapshot the Re-Rendered History, Not a Repaired Completion Vector

Trying to repair the end-of-generation vector (strip the scaffold, re-tokenize the reply) would chase each divergence class separately and remain hostage to template quirks. Keying the snapshot by the tokenization of the history render itself neutralizes all three classes by construction, because the next turn's prompt starts from exactly that render.

### 3.2 On by Default, with a Kill Switch

The feature is what makes the prompt cache hit at all for these families, so it defaults on. The cost, an extra graph launch and a full model-state copy on the foreground prefill of every qualifying request, is real for deployments serving only single-turn traffic; the kill switch is the interim opt-out and the automatic capability gate is filed as #1153 (the capability lives behind the worker thread and is not published to the HTTP layer).

### 3.3 The End-of-Generation Donate Stays

Suppressing the completion donate when a boundary snapshot exists was considered and rejected: that entry is not dead weight, it produced the longer hit on falcon-h1-tiny where the reply re-tokenized canonically. Both producers persist, each with its own supersede chain.

### 3.4 Batched-Cold Cohorts Are Left Out, Deliberately

A `BatchedCold` cohort of two or more rows forwards every row in one pass with no per-row split point. Marking those rows non-cold would route essentially every snapshot-family chat row to sequential prefill, giving up batched prefill for the whole family, which costs more than the missed reuse. Documented at the method.

### 3.5 Review Findings Were Folded In

Beyond the logits materialization and kill-switch scope fixes: a failed HTTP-side history tokenization no longer silently declines the request permanently (the scheduler's encode still gets its turn); the boundary decision was split into two pure functions with tests; a pre-existing preemption leak of `prompt_cache_seq_ctx` entries, made proportional to conversation length by this branch, is cleared at the eviction site; trace spans no longer report freshly forwarded boundary tokens as `cached`; and the capture takes the token vector out of the context map instead of cloning the whole context.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 18 |
| Lines | +1301 / -93 |
| New env vars | 1 (`MLXCEL_DISABLE_BOUNDARY_SNAPSHOT`) |
| New store concept | `SnapshotOrigin` (per-producer supersede chains) |
| Total tokens forwarded per prefill | Unchanged |
| Benchmark record | `docs/benchmark_results/history-boundary-snapshot-m1ultra-2026-08-14.md` |

### Changes by Area

**`src/server/chat_template.rs`, `src/server/chat_request.rs`, `src/server/config.rs`**
- History render (`apply_history_with_kwargs` and friends), `PreparedChatRequest::history_prompt` with its gating, and `PromptCacheRequestContext::{history_prompt, history_prefix_tokens}` mirroring the existing prompt/token split.

**`src/server/model_provider.rs`**
- The dispatch thread tokenizes the history render next to the prompt via the same `tokenize_prompt_for_generation` convention; the scheduler thread pays for neither encode.

**`src/server/batch/scheduler.rs`**
- `resolve_history_boundary`, `capture_history_boundary_snapshot`, `insert_model_state_snapshot` shared with the donate path, `prefill_start_offset` advance consumed by both full and chunked prefill.

**`src/server/prompt_cache/{entry,policy,store}.rs`**
- `SnapshotOrigin` with builder-set origin and origin-scoped supersede; the kill switch in policy.

**`src/server/routes/{chat,responses,anthropic}.rs`, `src/server/batch/observability.rs`**
- `build_prompt_cache_request_context` wired at all six call sites; `record_prefill_tokens` keeps `total_prefill_tokens` summing to the prompt across the split.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --release --lib server::prompt_cache` (142), `server::chat_request` (80), `server::chat_template` (100), `server::batch::` (353), `server::routes::` (116), `server::model_provider` (51); `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`; `cargo fmt --check`. New tests include `history_boundary_snapshot_hits_where_the_end_of_generation_one_cannot`, which encodes the epic's shape at the store level.
- Real-checkpoint A/B on Apple M1 Ultra, both arms through the production `/v1/chat/completions` path, baseline `main` at 9c154ff3. Attribution is self-checking: `snapshot_inserts` advances once per turn in the baseline arm and twice in the fix arm. qwen3.5-0.8b-4bit: turn 2 `cached_tokens` 0/189 to 150/189, turn 3 0/214 to 184/214, `snapshot_hits` 0 to 2; hit lengths equal the previous turn's history boundary exactly. llama-3.2-1b dense-KV control identical in every cell.
- Greedy determinism characterized rather than hidden: a kill-switch A/B on one greedy turn showed a single-word divergence after 168 identical characters, each arm byte-identical across three repeats. Two forwards do not reduce in the same order as one; this is the documented near-tie flip class and already the status quo under `--prefill-chunk-size` and cache hits.

### Not Covered

- Wall-clock latency and TTFT: the box carried load average 22 throughout, so no absolute timing claim would have survived; stated open rather than implied.
- Peak memory at 31B scale, concurrent-conversation budget pressure (#1146's domain), and gemma-4-31b-it-4bit, left to the epic-level verification since only small checkpoints were exercised.
- falcon-h1-tiny appears in the record but is not attribution evidence: its replies re-tokenized canonically, so the baseline arm hit too. It establishes only that adding the boundary snapshot does not break an already-hitting family.

### Follow-up

- #1153 (filed from this PR): dense-KV deployments pay a wasted second render and encode until `supports_snapshot_reuse()` is published across the thread boundary.
- The 512 MiB snapshot budget concern is still live for the gemma-4-31b verification: a conversation now legitimately holds two chains, so #1151 landing does not by itself resolve the capacity arithmetic (single 31B snapshots measured at 307-369 MB in the epic).
- The two new silent decline paths (history render not a prefix; boundary below `min_prefix_tokens`) belong in #1147's classified-reject surface; not added here to avoid conflicting with that issue's in-flight edits.
