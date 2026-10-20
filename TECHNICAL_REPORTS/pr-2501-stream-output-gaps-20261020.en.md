# PR #2501: remaining streamed vs non-streamed output gaps

**Date**: 2026-10-20
**Status**: Completed
**Risk Level**: Medium
**Languages**: Rust

---

## Executive Summary

Closes #2464 and #2470. Closes the six remaining streamed vs non-streamed output gaps left by #2458 plus two server output-shaping gaps found in the 0.8.0-beta pre-release smoke test: the Llama 3.x `<|python_tag|>` tool-call leak into streamed content, and `/v1/responses` dropping harmony reasoning and reporting `reasoning_tokens: 0`. Every fix is live on the chat and responses routes, not only inside the filter unit.

---

## 1. Problem Statement

### 1.1 Background

#2458 (#2394) aligned streamed tool-call and marker cleanup with the non-streamed output but listed six remaining gaps. #2470 added two more from the smoke test. Each one means a client sees different `content` or `reasoning` depending on whether it streams.

### 1.2 Existing Issues

- Harmony (gpt-oss) under `--reasoning-format none` sent analysis plus answer as content on the stream while the non-streamed reply returned answer only.
- `try_harmony` ignored a `to=` recipient placed in the role header; the stream suppressed it as a call, non-streamed treated it as commentary.
- Held whitespace after a closed call with no preceding text streamed as `\nB` while non-streamed trimmed to `B`.
- Harmony content split across messages joined with `\n` non-streamed but streamed back to back.
- MiniMax M3 left namespace residue on the no-tools path on both paths.
- Kimi-VL `◁think▷` / `◁/think▷` markers reached content on both paths.
- Llama 3.x `<|python_tag|>` tool call: non-streamed was correct (`finish_reason: tool_calls`, empty content); streamed, the raw call text arrived as `delta.content` and the same call again as `delta.tool_calls`.
- `/v1/responses` returned only the output_text item with `reasoning_tokens: 0` and no reasoning item, while the chat route exposed the analysis channel as `reasoning_content`.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Clients parse streamed tool-call text as content and double-execute calls | High | High (deterministic on Llama 3.x tool prompts) |
| Harmony analysis leaks into content under `none`/`deepseek-legacy` | Medium | High |
| Responses API clients cannot read reasoning or usage | Medium | High |

---

## 2. Technical Review

### 2.1 Security

Clean. The diff introduces no shared state (grep of all added lines for static/once_cell/OnceLock/Mutex/RwLock: nothing); the new StreamFilter flags are per-instance fields on a filter constructed per request and `FilterOutput.harmony_reasoning` is per-emit. The `split_reasoning` no-tools Harmony fallback fires at most once per request on the request's own raw output, bounded by max_tokens, and only `.reasoning_content` is taken from its result, so it cannot smuggle tool calls. No cross-request reasoning leak.

### 2.2 Performance

No per-token cost beyond O(1). `count_reasoning_tokens` is exactly one extra tokenizer encode per request (not per event), on the same order as the detokenization the request already performs. The six new delimiter-table rows are shorter than the existing max_delim_len (28 bytes), so the partial-match window and buffer bounds are unchanged; the per-feed table scan grows by a constant factor against ~41 rows.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Streamed output now matches non-streamed output; `reasoning_tokens` is non-zero on completed events and stays 0 on in-progress events (no counter exists until completion).
- **New Dependencies**: none.

### 2.4 Code Quality

- Test Coverage: 12 new parity tests + 4 helpers in `stream_parity_tests.rs`, 2 new unit tests in `routes/responses.rs`. The nine testable new parity tests were run against a pre-fix tree (test-only overlay on 6a115916): 0 passed / 9 failed, each panic showing its gap directly (python-tag raw text as content, payload not closed at `<|eom_id|>`, role-header mismatch, held-whitespace mismatch, both Harmony joins, analysis leaked as `delta.content`, M3 residue, Kimi-VL markers). The two responses tests are over helpers that do not exist on main.
- `clean_content_markers` now strips M3 tool blocks and python_tag payloads with the same spans the streaming suppression uses (open-through-close inclusive; unclosed runs to end of text on both paths).

---

## 3. Technical Decisions

### 3.1 Match the non-streamed reply for Harmony under `--reasoning-format none`

**Context:** Harmony has no canonical thinking-marker pair, so the rendering under `none` needed a decision.

**Chosen:** match the non-streamed reply (answer only). The analysis channel still reaches `reasoning_content` under deepseek formats via the unguarded `emits_reasoning_content` emissions. Recorded in the code comment and PR body.

### 3.2 Reuse the harmony path from chat.rs in /v1/responses

**Context:** `split_reasoning` / `extract_reasoning_from_raw` only knew think-marker forms, and `reasoning_tokens` was hard-coded to 0.

**Chosen:** `split_reasoning` takes `parsed.reasoning_content` first, then raw extraction, then a no-tools Harmony fallback gated on `parsed.is_none()` plus the raw channel markers, calling `tool_calls::parse_tool_calls(raw, None).reasoning_content` instead of a second harmony parser. `reasoning_tokens` counts by encoding the extracted reasoning text with the model tokenizer, in both paths.

### 3.3 The bare-JSON Llama 3.x form cannot be held back in the stream

Stated in the PR body as required: it is indistinguishable from ordinary JSON prose until fully parsed, and the stream cannot hold arbitrary text it would later emit as content.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/server/tool_calls/stream_filter.rs`**
- `FilterOutput`/`Drained` gained `pub harmony_reasoning: bool`; `StreamFilter` gained `has_emitted_content`, `harmony_content_open`, `harmony_join_pending`, `harmony_reasoning_open`, `harmony_reasoning_join_pending`.
- `CHAT_DELIMITERS` gained the Kimi-VL think pair, the bare `]<]minimax[>[` Strip row, `<|eom_id|>` ExitToolCall and `<|python_tag|>` EnterToolCall.
- `canonical_thinking_pair` gained the Kimi-VL pair; `emit_prefix`, `emit_content_prefix`, `after_transition` and `flush` carry the new flags, suppressions and owed join separators.

**File: `src/server/tool_calls/parser.rs` and `formats.rs`**
- `THINKING_MARKER_PAIRS` now 5 entries with Kimi-VL; `clean_content_markers` strips M3 and python_tag payloads and the stray marker bytes.
- `strip_minimax_m3_tools_block` and `strip_python_tag_payload` with spans matching the streaming suppression.

**File: `src/server/routes/chat.rs`, `router_front.rs`, `in_process/chat.rs`**
- The item-1 harmony guard `&& !emit.harmony_reasoning` on the content re-echo sites of the single-node route and the disaggregated router (the coordinator's fix), plus the construction sites for the new field.

**File: `src/server/routes/responses.rs`**
- `split_reasoning` priority chain; `count_reasoning_tokens` used at the non-streamed and streamed-final sites; encode failures logged at debug level.

---

## 5. Verification

| Check | Result |
|---|---|
| New parity tests on pre-fix tree | 0 passed / 9 failed (each gap pinned) |
| stream_parity | 31 passed / 0 failed |
| tool_calls | 474 passed / 0 failed |
| server:: | 3562 passed / 0 failed / 16 ignored (run twice on the final code) |
| cargo fmt / clippy | clean / clean |
| Full gate at fb7d7a1a | 161 bins build 0, clippy 0, doc 0, contract 0; 160/161 with the known #2467 mllama flake, rerun 8/8 |
| gpt-oss-20b-mxfp4 /v1/responses (temperature 0) | reasoning item with the analysis text, reasoning_tokens 44 streamed == non-streamed |
| meta-llama-3.1-8b-instruct-4bit get_weather (temperature 0) | streamed and non-streamed: finish_reason tool_calls, no content delta, exactly one call, no python-tag or raw-JSON leak |
| Security review | PASS; no shared state; fallback bounded once per request; no cross-request leak |

Checkpoint substitutions stated in the PR: models/mlx/gpt-oss-20b-mxfp4 (issue's gpt-oss-20b-mxfp4-q4 absent) and models/mlx/meta-llama-3.1-8b-instruct-4bit.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 8 (main commit), 2 (follow-ups) |
| Lines added | +587 |
| Lines deleted | -494 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `fb7d7a1a` | fix | fix(server): close remaining streamed vs non-streamed output gaps |
| `58e19969` | fix | fix(router): drop harmony analysis from content under none placement |
| `b46729b8` | refactor | refactor(server): gate split_reasoning fallback and log encode errors |

---

## 7. Follow-up Actions

### Monitoring Required

- Streamed tool calls on other python_tag-producing families in the first release after merge.
- `reasoning_tokens` on in-progress Responses events stays 0; clients must read it from response.completed.

### Future Improvements

- A streaming reasoning-token counter would need incremental encoding; deferred (no counter exists today).

---

## Appendix

### A. Review verdicts

pr-reviewer: PASS on all six focus areas; one HIGH (the item-1 harmony guard missing on the router_front re-echo sites) fixed by the coordinator in `58e19969`. pr-security-checker: PASS, no CRITICAL/HIGH/MEDIUM; two LOW adopted by the finalizer in `b46729b8` (redundant parse gated, encode errors logged).
