# PR #2522: Automatic Responses compaction (phase B)

**Date**: 2026-10-21
**Status**: Completed (PR open; closes #2397)

## Executive Summary

PR #2522 completes #2397. Phase A (PR #2520) added the explicit `POST /v1/responses/compact`. This PR lets `/v1/responses` compact on its own. A request that carries `context_management: [{"type":"compaction","compact_threshold":N}]` compacts its history before generation once the rendered history reaches N tokens. The response output then starts with the compaction item. Streaming sends `mlx.compaction.started` and `mlx.compaction.completed` before that item. Generation on these requests is capped at the remaining context, and a reply that fills the window ends `incomplete`. Requests without `context_management` are unchanged, and a test locks this for both the streamed and non-streamed paths. Review and security found no CRITICAL or HIGH issue. Four MEDIUM findings were fixed. The out-of-scope parts are filed as #2523. 15 files, +1356/-23, of which about 750 lines are tests.

## 1. Change Summary

| Area | Change |
|---|---|
| `routes/responses_auto_compact.rs` (new) | `context_management` parsing, threshold check, compaction call, output cap, `incomplete` marking, reasoning budget clamp |
| `routes/responses.rs` | Hook on both arms: count, compact, cap, then generate; stream events and the capsule at `output_index` 0 |
| `responses_compaction.rs` | `compact_history_counted` reuses the threshold count; `NoSplitPoint` error |
| `types/responses_stream.rs`, `types/responses_request.rs`, `responses_translator.rs` | Two event variants, the request field, expanded items kept only when the field is present |
| Tests | `responses_auto_compact_tests.rs`, `responses_auto_compact_chain_tests.rs`; the scripted provider can end on `length` |
| Docs | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

## 2. Technical Decisions

### 2.1 When compaction fires

The count covers the full history: chained `previous_response_id` or `conversation` items, plus `input`, with capsules expanded. It uses phase A's backend, which is the `/v1/responses/input_tokens` path. Compaction fires when the count is `>= compact_threshold`, which matches upstream and OpenAI's "reaches". The issue text said "exceeds". It also fires below the threshold when the history leaves no room for output. Phase A's selection, capsule and summary code run unchanged, and the threshold count is reused instead of being repeated. The automatic summary budget follows upstream: context/16, clamped to 128..1024.

### 2.2 The output cap applies only to `context_management` requests

Today every `length` stop reports `completed`, and with `--context-shift` generation can run past the window. A general cap would change that behavior, so the cap applies only to requests that carry the field, compacted or not. `max_tokens` becomes `min(max_tokens, context - prompt_tokens)`, computed with saturating arithmetic. When no room is left, the request gets `context_length_exceeded`. A reply that stops on `length` has these fields:
- `status: "incomplete"`;
- `incomplete_details.reason: "max_output_tokens"`, the OpenAI and upstream value;
- the cut-off message item has status `incomplete`;
- `completed_at` is null.

A stream ends with `response.incomplete`.

### 2.3 Failures happen before the stream opens

`ResponseStreamSender` sends with `blocking_send` from the generation task. Compaction therefore runs before the SSE response starts, so a failure is an HTTP error on both arms, as an overflow at admission already is. Malformed `context_management` is refused with 400 before any work: not an array, more than one entry, an unknown type, or a threshold outside 1..1e8. When nothing can be split, the request proceeds uncompacted if the history still fits, as upstream does. Otherwise it gets `context_length_exceeded`. The two events report the token counts before and after compaction, not live progress.

### 2.4 Stream order and stored shape

After `response.in_progress`, the stream sends `mlx.compaction.started`, then `mlx.compaction.completed`, then the capsule's `output_item.added` and `output_item.done` at index 0. Generated items start at index 1. `prepend_compaction_item` matches by the server-generated `cmp_` id. Both final-object paths therefore list the capsule exactly once at index 0, including the tool-call path from #2504 that rebuilds the object from recorded items. The stored response keeps only the instructions as input, with the capsule as `output[0]`. The next hop rebuilds instructions, capsule and tail through the existing output-to-input mapping, replays the summary, and does not compact again below the threshold.

### 2.5 Review findings

| Severity | Finding | Fix |
|---|---|---|
| MEDIUM (review) | No test of auto-compaction on the `conversation` path | `29dfec04` |
| MEDIUM (review) | No streamed test that a request without `context_management` is unchanged | `29dfec04`: exact event sequence and final object |
| MEDIUM (security) | The reasoning budget was resolved against the uncapped limit, so a capped reply could spend everything on reasoning | `b078c984`: budget clamped to one below the capped `max_tokens` |
| MEDIUM (security) | Token counts ran before slot admission, so a busy server still did the counting work | `b078c984`: `can_accept_request()` runs first and returns the existing 503 |
| LOW (review) | A capped reply kept item status `completed` and set `completed_at` | `288f9eec` |

## 3. Verification

| Check | Result |
|---|---|
| `responses_auto_compact` | 15 passed |
| `responses_compact` | 28 passed |
| `server::routes::` | 508 passed, 1 ignored |
| `server::responses`, `server::types`, `streaming_responses` | 74, 107, 4 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. The new busy-server test would also pass without the new gate in this harness, because the later stages already return 503 and text counts leave no visible trace. The gate was confirmed by reading the code.

## 4. Unverified and Open Items

- **Not run:** the integrated real-model check with `mlx-community/Qwen3-30B-A3B-4bit` and the OpenAI SDK. This host has no Metal or CUDA. Image and video prompts with `context_management` have no test.
- **Known limit (security MEDIUM, not fixed):** a streamed request that compacts runs the summary pass on the request task. A client disconnect does not cancel it. Non-streamed requests and `/compact` already behave this way, and slot admission bounds the cost.
- **LOW:**
  - `/v1/responses/input_tokens` ignores `context_management` without validating it.
  - When nothing can be split, no events are sent (upstream sends `started` alone).
  - A failed automatic summary fails the request instead of generating uncompacted.
  - The whole request is cloned once to build the count request.
- **Follow-up #2523:** Chat Completions and `/v1/messages` auto-compaction, Codex `compaction_trigger`, and multi-pass summarization.
- **Size:** +1356 lines, above the 1,000-line guideline; non-test code is about 600 lines.

## 5. Learning Points

- When a stream sender must not block the executor, any work that can fail has to finish before the stream opens. Then a failure keeps the plain HTTP error shape instead of becoming a half-open stream.
- A behavior change that applies only behind an opt-in field needs a test that the path without the field is byte-identical, on every transport. Otherwise the opt-in quietly becomes the default.
