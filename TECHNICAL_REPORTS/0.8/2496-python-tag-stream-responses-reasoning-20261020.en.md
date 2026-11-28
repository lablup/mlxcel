# PR #2496: Hide Llama 3.x python_tag calls from streams, add Responses reasoning

**Date**: 2026-10-20
**Status**: Completed (PR open; unverified items listed in section 5)

## Executive Summary

PR #2496 (closes #2470) fixes two server output gaps found in the 0.8.0-beta smoke test on GB10. Streamed Llama 3.x `<|python_tag|>` tool calls no longer leak as `delta.content` before arriving again as `delta.tool_calls`. `/v1/responses` now returns gpt-oss Harmony analysis as a `reasoning` output item and reports a real `reasoning_tokens` count instead of a hard-coded 0, equal between streaming and non-streaming. Review added three fixes: duplicated text with an unclosed primed `<think>`, a quadratic non-stream reasoning replay, and token counting on the async executor. 11 files, +439/-33.

## 1. Problem Statement

- **python_tag leak.** `try_llama3` (`formats.rs`) parsed `<|python_tag|>{...}` at end of generation, but `CHAT_DELIMITERS` in `stream_filter.rs` had no row for the tag. The stream filter passed the call text through as content, and the final parse then emitted the same call as `tool_calls`. Non-streamed output was correct, so the #2458 parity goal was unmet.
- **Responses reasoning.** `split_reasoning` and `extract_reasoning_from_raw` in `routes/responses.rs` knew only `<think>` forms, so Harmony analysis was dropped. `reasoning_tokens` was the literal `0` in both arms. The chat route already exposed the same analysis as `reasoning_content`.

## 2. Change Summary

| Area | Change |
|---|---|
| `tool_calls/stream_filter.rs` | `("<|python_tag|>", EnterToolCall)` and `("<|eom_id|>", ExitToolCall)` rows |
| `tool_calls/parser.rs` | `strip_python_tag_blocks`, called from `clean_content_markers` |
| `tool_calls/formats.rs` | `try_llama3` accepts the tag after visible text and stops at `<|eom_id|>` |
| `routes/responses.rs` | `split_reasoning` with three ordered sources; `count_reasoning_tokens` and `count_reasoning_tokens_off_executor` |
| `routes/chat.rs` | `strip_unclosed_primed_thinking` made `pub(crate)`; `collect_reasoning_chunked` with `REASONING_REPLAY_CHUNK_BYTES = 128` |
| Tests | 4 parity tests in `stream_parity_tests.rs`; 7 tests in `responses.rs`; chunked-replay test in `chat.rs` |
| Docs | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

Commits: `7026cfdb` (implementation), `27928305` (unclosed primed thinking), `82846066` (chunked replay), `9af71a31` (count off executor), `b7327309` (docs).

## 3. Technical Decisions

### 3.1 python_tag is tool markup in both paths

The filter enters tool-call state at `<|python_tag|>` and leaves it at `<|eom_id|>` or end of stream. `<|p` starts no other delimiter, and `<|eom_id|>` diverges from Muse's `<|eom|>` at byte 5, so partial-match hold-back resolves at every `chunkings()` split. `clean_content_markers` drops the same block in the non-streamed path, so a request without tools, or a call to an unknown tool, produces the same content both ways. This follows the #1743 rule that tool markup is never content.

Consequence: a non-JSON python_tag block (Llama's built-in code interpreter) is now dropped from content. At stream time the filter cannot know whether the block will be JSON, so suppressing it is the only behavior both paths can share. mlxcel does not run Llama's built-in tools.

**Bare JSON is not held back.** The Llama 3.x form without a tag has no marker. Classifying it in the stream would mean holding back every `{` until end of stream, which stalls every JSON-shaped answer. It still streams as content and is then parsed into `tool_calls`.

### 3.2 Reuse the chat route's reasoning sources instead of a second Harmony parser

`split_reasoning(raw, parsed, primed_open_thinking)` reads reasoning in this order:

1. `parsed.reasoning_content` from `parse_tool_calls` (set by Harmony and Muse).
2. The existing `<think>` extraction.
3. `chat::extract_reasoning_content`, the `StreamFilter` split that already routes Harmony channels.

Later sources run only when earlier ones return nothing, so reasoning is not extracted twice. The reported scenario had no tools, so source 3 is what fixes it. The issue explicitly ruled out a second Harmony parser.

**Review finding (HIGH, `27928305`).** With a primed `<think>` that never closes (`max_tokens` reached mid-thought), source 3 returned the whole output as reasoning while the visible text still held it. The non-streamed reply sent the text twice. `split_reasoning` now passes the visible text through `strip_unclosed_primed_thinking`, the chat route's existing rule.

### 3.3 Counting reasoning tokens by re-encoding

No reasoning-token counter existed. `count_reasoning_tokens` trims the extracted reasoning and calls `tokenizer.encode(text, false)`; an encode error logs a warning and counts 0. Both arms count the same text, so the counts are equal. A test feeds the text through `StreamFilter` one character at a time, as the streaming route does, and compares.

| Option | Pros | Cons |
|---|---|---|
| Count token positions in the filter while streaming | No re-encode | Needs per-token channel tracking in both arms; the non-stream arm has no token stream to observe |
| **Re-encode the extracted text** | One function for both arms; equal by construction | Re-encode cost; may differ by a token from the generated ids at segment joins |

**Security finding (MEDIUM, `9af71a31`).** Tiktoken BPE in this codebase is quadratic per piece, with about 26 s for a 25k-character punctuation run. The stream arm already counted inside `spawn_blocking`. The non-stream arm counted on the async executor, so `count_reasoning_tokens_off_executor` now moves that count to the blocking pool.

### 3.4 Bounded chunks for the non-stream reasoning replay

**Security finding (HIGH, `82846066`).** `extract_reasoning_content` fed the whole reply to one `StreamFilter::feed`. The filter rescans its buffer on every delimiter hit, so cost grew with hits times length: 32k `<tool_call>` repeats took 80 s and 32k `<|python_tag|>` 42 s. The slowness predates this PR on the chat route, but the PR added two triggering markers and sent non-stream `/v1/responses` through the same function. `collect_reasoning_chunked` now feeds 128-byte chunks cut on UTF-8 character boundaries, which bounds the buffer per feed. The same inputs take about 0.29 s. The filter internals are unchanged, and a test pins chunked output to per-character output for repeated markers, Hangul text and a Harmony reply.

## 4. Verification

| Check | Result |
|---|---|
| `server::tool_calls::` | 457 passed |
| `server::routes::responses` | 25 passed |
| `server::responses` | 56 passed |
| `server::muse_atem`, `server::reasoning_echo` | 11 and 30 passed |
| `server::routes::chat`, `server::chat_request` (security review) | 72 passed; 128 passed, 1 ignored |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. All four parity tests fail when the `stream_filter.rs`, `parser.rs` and `formats.rs` changes are stashed. The five original `responses.rs` tests call new code, so they were checked against main's logic restored in place, where they fail.

## 5. Unverified and Open Items

- **GB10 scenarios not run:** Llama 3.1 8B streamed tool call; gpt-oss-20b `/v1/responses` streamed and non-streamed.
- **Carried into #2464:** multi-segment Harmony analysis is joined with `\n` by the parser and with nothing by `StreamFilter`, so text, and possibly the count by one, can differ between arms. Whitespace just before a tool-call marker streams as content but is trimmed in the non-streamed reply.
- **Pre-existing:** `/v1/responses` ignores `--reasoning-format`. `extract_reasoning_from_raw` matches the first literal `</think>` anywhere, so a Harmony final channel that mentions `</think>` would lose text to reasoning in the non-streamed reply.
- **LOW:** dropped non-JSON python_tag blocks are not logged. `try_llama3` reads only the first python_tag call.

## 6. Learning Points

- A streaming filter and a whole-text parser must agree on what is markup. A delimiter missing from one of them shows up as a parity bug, not a crash, so parity tests over many chunkings are the main defense.
- Feeding a whole document to an incremental filter in one call can be much slower than feeding it in chunks, when the filter rescans its pending buffer per event. Bound the per-call input.
- Re-encoding text to count tokens is simple and keeps two code paths equal. Its cost belongs to the tokenizer, so run it where blocking is allowed.
