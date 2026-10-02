# Technical Report: PR #2094 - Forward echoed reasoning under `reasoning_content` too

**Date**: 2026-10-02

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (chat request rendering, tests), Markdown

**Risk**: Low (the raw-JSON message passed to chat templates gains a `reasoning_content` key carrying the same value as the existing `reasoning` key, only when a client echoed a trace)

## Summary

Issue #2089 reported that `AI21-Jamba-Reasoning-3B-4bit` (stored as `jamba-v0.1-4bit`) returned empty `content` and missed the prompt cache on every turn. The investigation split this into two findings. The empty `content` is `finish_reason=length`: the template always primes `<think>`, and a 64-token budget ends inside the reasoning block. With a 2048-token budget the model closes `</think>` and `content` is filled, so marker recognition is correct and no budget change was made. The cache miss had a server-side cause: an echoed trace reached chat templates only as `reasoning`, while this template reads `reasoning_content`.

## 1. Measurements (before the fix)

| max_tokens | turn 1 | turn 2 | turn 3 |
|---|---|---|---|
| 64 | length, content empty | length, content empty | length, content empty |
| 2048 | stop, `Paris.` | length (looped on a population question) | stop, `The Louvre.` |

Turn 2 and 3 reported `cached_tokens=0` even when turn 1 ended with non-empty content. Echoing `reasoning_content` did not change the prompt length (66 tokens in both cases), which showed the server dropped the echo before rendering.

## 2. Root cause

The Jamba template adds a thinking instruction to the last user turn. On an earlier user turn it keeps the instruction only when the next assistant message starts with `<think>` or defines `reasoning_content`. `build_raw_json_messages_with_thinking` folded both wire spellings into one field and emitted it as `reasoning` (the Gemma 4 spelling). The template therefore saw no reasoning, rendered turn 1's user message without the instruction on turn 2, and the turn-1 history-boundary snapshot (which contains the instruction) no longer prefixed turn 2's prompt.

## 3. Change

The forwarded trace is now written under both `reasoning` and `reasoning_content`, with the same gating as before (dropped for stripped turns and when the content already carries an inline `<think>` block). Templates on hand that accept both (Gemma 4, Laguna) read them as alternatives, so the trace is never rendered twice; a test pins this.

A client that echoes only `content` still gets the earlier user turn without the instruction. That is the template's own rule. It is pinned by a test, recorded in the prefix-stability notes of `chat_request.rs`, and documented in `docs/supported-models.md` together with the `max_tokens` guidance.

## 4. Validation

- `server::chat_request` tests: 126 passed, including 5 new tests in `chat_request_reasoning_content_tests.rs`.
- Clippy with `-D warnings` and rustfmt clean.
- Real server on GB10, fresh process, 3 turns at max_tokens 2048 echoing `reasoning_content`: finish stop/stop/stop, content `Paris.` / `4` / `Louvre`, prompt 46/93/135, cached_tokens 0/41/88.
- Same server, content-only echo: content non-empty, cached_tokens 0/0/0, as documented.

## 5. Not covered

Recovering cache reuse for content-only clients would need a snapshot at the point where the template stops rewriting history (before the last user turn) plus a warm-up that can start from it. That is a general change to the boundary snapshot design and is left for a separate issue.
