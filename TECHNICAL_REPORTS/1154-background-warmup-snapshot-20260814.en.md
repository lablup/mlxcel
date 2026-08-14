# Technical Report: PR #1154 - feat(server): warm the next turn's history prefix in the background

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (background-only work dispatched exclusively from a fully idle scheduler; fire-and-forget failure semantics; kill switch provided)

---

## Executive Summary

The history-boundary snapshot from #1143 gets a conversation's next turn a hit covering everything up to the last user message, but the previous assistant reply is still prefilled on the foreground path of every turn. This change moves that work off the critical path: after a healthy completion, the server renders the next turn's expected history prefix and, when the scheduler has nothing else to do, restores the conversation's existing snapshot and prefills only the delta up to that prefix.

The load-bearing part of the design is the target vector. A warm-up stores a `Boundary` snapshot that supersedes the entry it chained from, so a target the next turn cannot match does not merely waste a background prefill: it destroys a working hit. Two intuitive constructions were built, measured, and rejected before the shipped one: the unclipped `render(messages + reply, add_generation_prompt = false)` target drove `cached_tokens` to 0 (worse than no warm-up), and a single-probe clip salvaged only 3 tokens. The shipped two-probe construction yields turn 2 cached tokens 150 to 194 of 227 and cuts turn 2 uncached prefill from 77 to 33 on qwen3.5-0.8b, with a counter-verified guarantee that warm-ups never run while foreground work exists.

---

## 1. Problem Statement

### 1.1 Background

For the snapshot-only class, #1143 made the multi-turn cache hit at the history boundary, and for the recurrent-state families (where partial restore is impossible, see #1145's scope) that boundary hit plus this warm-up is the complete reuse story. What remains on the foreground path is the previous reply: re-rendered as history, it must be prefilled at the start of every turn, tens of tokens on small models and avoidable 31B-scale work on large ones. Everything needed to compute that state (the boundary snapshot plus the reply text) exists the moment the completion finishes, and none of the computation has to happen while the user is waiting.

### 1.2 Existing Issues

- **Avoidable foreground work.** The delta between the boundary snapshot and the next turn's history prefix was paid on the critical path of every turn.
- **A naive target vector is actively harmful.** Because a warm-up supersedes the boundary entry it chains from, storing an unmatchable vector replaces a working hit with a useless one. This was measured, not hypothesized: the first construction regressed `cached_tokens` to 0 of 227 and 0 of 275.
- **Background work must not contend.** Model forwards are uninterruptible; a warm-up that starts while a request is arriving delays it. The scheduling condition has to be provably conservative.
- **No observability.** Without dedicated counters, a warm-up that never ran and one that ran and stored a useless entry would be indistinguishable.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Warm-up target does not match the next turn, superseding a working boundary hit | High | Low (two-probe clip; the failed constructions are pinned in tests) |
| Warm-up forward delays a foreground request | Medium | Low (idle-only dispatch requires empty batch, empty prefill queue, no parked chunked prefill; verified under load) |
| Warm-up failure breaks the completed request | Low | Very low (fire and forget: no response channel, no queue reservation, send failure is a debug line) |
| Streaming paths pay accumulation cost for replies that never warm | Low | Low (content accumulated only when a warm-up is possible) |

---

## 2. Technical Review

### 2.1 Three Target Constructions, Two Rejected by Measurement

1. **Unclipped `render(messages + reply, add_generation_prompt = false)`.** Intuitive and wrong: templates render the final assistant message differently from an earlier one, so the result is not a prefix of the next turn's prompt. Measured: `cached_tokens` fell to 0 of 227 and 0 of 275, strictly worse than no warm-up because the useless entry superseded the working boundary entry.
2. **The same, clipped against one probe render.** Safe but nearly useless: the two renders disagree immediately after the assistant header, so the clip discarded the reply. Measured: +3 cached tokens (153 versus 150).
3. **Two probes differing only in a trailing placeholder user turn.** Both place the reply where the next turn will place it, so their common prefix ends exactly where the next turn's own words begin, and the reply survives the clip. This shipped, as `render_next_turn_history` returning both probes and `clip_warmup_target` reducing them to the agreeing head.

The sequence is a worked example of the project's measurement discipline: the intuitive design was built, measured through the production path, shown harmful, and replaced, with the failed constructions recorded rather than erased.

### 2.2 Idle Means Idle

`run_next_prompt_cache_warmup` is dispatched only from the scheduler's `Idle` tick arm, and `can_run_prompt_cache_warmup` additionally requires an empty active batch, an empty prefill queue, and no parked chunked prefill. That conjunction is the only way to guarantee an uninterruptible forward never starts while foreground work exists. The queue is bounded and newest-wins, so a burst of completions cannot build a backlog of stale warm-ups.

### 2.3 The Job Itself Is Minimal

The warm-up restores the longest snapshot the conversation already has, prefills only the delta, evaluates only the last logits position (the same lesson #1150's review taught on the boundary forward), and stores the result as a `Boundary` snapshot so it supersedes the entry it chained from; the conversation keeps exactly one boundary snapshot. Every path returns silently: the client was answered before the job existed.

### 2.4 Attribution Is Built into the Counters

`snapshot_warmups_run` can only advance when a warm-up actually restored, prefilled, and stored, so it is the arm-attribution counter for the A/B and the yielding proof under load. `snapshot_warmups_skipped` accounts for queue discards. The concurrency probe leaned on the project's loaded-box methodology: a pinned-versus-advancing counter says the same thing at load average 20 as on a quiet machine, where a latency percentile would not.

---

## 3. Technical Decisions

### 3.1 Fire and Forget, by Design

`ModelRequest::PromptCacheWarmup` carries no response channel and takes no queue reservation; `submit_prompt_cache_warmup` treats a send failure as a debug line. The client has already been answered, so there is no caller to fail. This also caps the blast radius of every failure mode inside the job at "the conversation keeps the #1143 boundary snapshot".

### 3.2 Streaming Accumulates Conditionally

The streaming path accumulates filtered `delta.content` only when a warm-up is possible for the request, so an ordinary stream allocates nothing extra. Tool-calling turns are skipped entirely: that reply returns to the model as a tool result, not as assistant history content, so warming it would build the wrong prefix.

### 3.3 A Separate Kill Switch

`MLXCEL_DISABLE_CACHE_WARMUP` is independent of `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT`. The two features have different cost profiles (foreground copy versus idle background forward), and an operator may reasonably want either without the other. The switch also served as the A/B arm selector, keeping both arms on one binary.

### 3.4 The Fixture Choice Was Forced, and Documented

qwen3.5-0.8b with `enable_thinking = false` is divergence class (a) and therefore the only small checkpoint where the boundary snapshot is the sole thing a turn can hit, making warm-up gains cleanly attributable. The 4b default-thinking variant was unusable at this size: the 0.8b model never closes its `<think>` block within any tried token budget, so `content` comes back empty and there is no reply to warm. falcon-h1-tiny appears in the record but is not evidence: its replies re-tokenize canonically, leaving the warm-up a 2-token delta.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 17 |
| Lines | +907 / -8 |
| New env vars | 1 (`MLXCEL_DISABLE_CACHE_WARMUP`) |
| New stats fields | 2 (`snapshot_warmups_run`, `snapshot_warmups_skipped`) |
| Foreground behavior changes | 0 (dispatch requires a fully idle scheduler) |
| Benchmark record | `docs/benchmark_results/warmup-snapshot-m1ultra-2026-08-14.md` |

### Changes by Area

**`src/server/model_provider.rs`**
- `ModelRequest::PromptCacheWarmup { tokens, ctx }` and `submit_prompt_cache_warmup`, fire and forget.

**`src/server/batch/scheduler.rs`**
- Bounded newest-wins warm-up queue; `can_run_prompt_cache_warmup`; `run_next_prompt_cache_warmup` dispatched from the `Idle` arm only; restore, delta prefill, last-position eval, `Boundary`-origin store.

**`src/server/chat_request.rs`**
- `render_next_turn_history` (two probe renders) and `clip_warmup_target` (agreeing head).

**`src/server/routes/chat.rs`**
- `submit_next_turn_warmup` on both non-streaming and streaming paths; conditional content accumulation; tool-calling turns skipped.

**`src/server/prompt_cache/policy.rs`, `src/server/batch/observability.rs`, `src/server/routes/cache.rs`**
- The kill switch; the two counters through to `/v1/cache/stats`.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --release --lib server::chat_request` (84, including the two-probe render tests and `clip_warmup_target` over the ordinary case, mid-render divergence, and total divergence), `server::batch::` (357), `server::prompt_cache` (167), `server::routes::` (118), `server::model_provider` (51).
- `cargo clippy --release --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --check`.
- Real-checkpoint A/B on Apple M1 Ultra, one binary with the kill switch as the arm selector, three-turn conversation with 4s think-time through `/v1/chat/completions`: turn 2 cached 150 to 194 of 227 (identical prompt length in both arms), turn 2 uncached prefill 77 to 33 (a 57% cut), turn 3 uncached 59 to 24, `warmups_run` 0 versus 2.
- Concurrency probe on `--parallel 4` with two clients driving 12 requests in 2.6s: `warmups_run` pinned at 0 with 10 queue skips and zero foreground errors during load, reaching 2 only after the load stopped. Yielding is proven by counters, which survive a loaded box where latency numbers would not.

### Not Covered

- Wall-clock latency and TTFT (load average 4-20 throughout; stated open rather than implied), the idle-GPU energy cost of one background forward per turn, peak memory at 31B scale, and behavior under sustained partial load where idle windows are short but nonzero.
- gemma-4-31b-it-4bit, left to the epic-level verification; only small checkpoints were exercised on this branch.
- Turn 3 prompt lengths differ slightly across arms because replies diverge once the prefill shape changes, so uncached tokens (59 versus 24) are the comparable figure there, not cached tokens.

### Follow-up

- The epic-level verification owns the 31B-scale run, where the warm-up's absolute benefit is largest and the snapshot budget interaction (#1146's capacity arithmetic, still open per #1150's report) is most likely to bind.
- Behavior under sustained partial load is the one scheduling regime not probed; if short idle windows turn out to admit warm-ups that then collide with arrivals, the `can_run_prompt_cache_warmup` conjunction is the place to add hysteresis.
