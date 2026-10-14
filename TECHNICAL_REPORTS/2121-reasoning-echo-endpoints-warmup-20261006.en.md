# Technical Report: PR #2121 - Extend reasoning re-injection to /v1/responses, /v1/messages and the warm-up

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (Responses and Anthropic routes, chat route warm-up, next-turn history render, tests), Markdown

**Risk**: Low to medium (content-only assistant turns on two more endpoints may now render with the reasoning the server returned for them, as on `/v1/chat/completions` since #2116; the chat warm-up now renders the reply with its reasoning and through the raw-JSON path when the history carries reasoning or tool fields)

## Summary

Issue #2118 followed up PR #2116, whose `ReasoningEchoStore` re-injected server-generated reasoning into content-only assistant turns on `/v1/chat/completions` only. `/v1/responses` and `/v1/messages` now fill and record the same way, and the #1144 next-turn warm-up now renders what a content-only follow-up renders after its own fill. On AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`) the two endpoints went from 0 cached tokens on every turn to hits on turns 2 and 3, and the chat warm-up, which never ran for this model, now supplies turn 2's hit.

## 1. /v1/responses and /v1/messages

Both handlers passed the translated request straight to `prepare_chat_request_with_cache`. They now mirror the chat route through two helpers in `reasoning_echo.rs`: `render_request` (scope plus `fill`, render only) and `record_reply`. The cache context, tool parsing and the record keep reading the translated request as the client sent it, so the key a later turn is looked up under never depends on an injected trace.

- Responses records the visible text and `reasoning` item from `split_reasoning` (non-streaming) and the message and reasoning accumulators of the stream emitter (streaming). A `previous_response_id` chain replays the stored message item but never its reasoning item, so chained turns are content-only history too; they are filled like an echoed turn.
- Anthropic records the text the client received (after stop-sequence handling) and the `thinking` block, non-streaming and streaming. It records only when the request enabled extended thinking, because otherwise the thinking block is never emitted and no client could have echoed it. Recording the hidden reasoning would change the store's "as if the client had echoed it" contract and is left to a separate decision.
- Precedence is unchanged: a Responses `reasoning` input item or an Anthropic `thinking` block maps to `Message.reasoning` and wins; tool-call turns are neither recorded nor filled.

## 2. Next-turn warm-up (#1144)

The warm-up rendered the unfilled request and appended the reply with `reasoning: None`, while the follow-up fills both. For Jamba the probe therefore rendered the earlier user turn without its instruction, failed the "extends this turn's history" check, and never ran. Three changes:

- The route records first, and `ReasoningEchoStore::record` now returns whether it stored the trace. The warm-up gets the reply reasoning exactly when it was stored, since only then will the follow-up's fill reproduce it.
- `submit_next_turn_warmup` receives the filled render request (non-streaming `render_request`, streaming a clone of it).
- `render_next_turn_history` takes `reply_reasoning` and picks the raw-JSON or typed render path with the predicate `prepare_chat_request_with_cache` uses. Only the raw path forwards `reasoning` / `reasoning_content`; this also aligns warm-ups whose history carries tool calls, which previously rendered on the typed path while the real request rendered raw.

The streaming path also appends the stream filter's flushed tail to the warm-up reply, which it already appended to the record.

## 3. Validation

New tests: `reasoning_echo_endpoint_tests.rs` drives the real `/v1/responses` (input array and `previous_response_id` chain) and `/v1/messages` handlers, streaming and non-streaming, with a scripted model and the reduced Jamba template; all three re-injection tests fail with `MLXCEL_REASONING_ECHO_MAX_BYTES=0`, and a fourth checks that `/v1/messages` without extended thinking records nothing. `reasoning_echo_warmup_tests.rs` checks that the warm-up target is a prefix of the follow-up's filled render on turn 1 to 2 and turn 2 to 3, that the reply without reasoning bails (the old behavior), that the unfilled request's target is not a prefix, and the `record` return value. The `reasoning_echo`, `server::routes`, `server::chat_request`, `server::batch::scheduler::prompt_cache`, `server::prompt_cache`, `server::responses`, `responses_translator`, `anthropic` and warm-up filters pass; clippy `--lib --tests -D warnings` and rustfmt are clean.

Real server on GB10, `jamba-v0.1-4bit`, 3 content-only turns, `max_tokens` 2048, temperature 0, prompt cache at defaults. "Before" is the same binary with `MLXCEL_REASONING_ECHO_MAX_BYTES=0`, which reproduces the old behavior on all three paths. `/v1/messages` reports no cached-token field, so its numbers come from the scheduler's `prompt-cache snapshot hit: restored N/M` debug line; Responses numbers are `usage.input_tokens_details.cached_tokens` and match the log.

| Endpoint | Mode | cached before | cached after |
|---|---|---|---|
| `/v1/responses` | non-streaming | 0 / 0 / 0 | 0 / 52 / 102 |
| `/v1/messages` (thinking enabled) | non-streaming | 0 / 0 / 0 | 0 / 51 / 101 |
| `/v1/chat/completions` | non-streaming | 0 / 0 / 0 | 0 / 91 / 101 |
| `/v1/chat/completions` | streaming | not run | 0 / 93 / 154 |

Warm-up attribution: before, `snapshot_warmups_run` stayed 0. After, non-streaming turn 2 logged `warm-up completed restored=51 delta=40 warmed=91` followed by `snapshot hit: restored 91/106 ... stored=91`, and `snapshot_warmups_run` went 0 to 1; the boundary snapshot alone would have given 51. Streaming turns 2 and 3 hit warmed entries of 93 and 154 tokens the same way. Non-streaming turn 3 hit the boundary (101) because turn 2 ended inside its thinking block with empty content, and the warm-up skips an empty reply.

## 4. Limits

- Streaming `/v1/responses` and `/v1/messages` build an unprimed `StreamFilter`, while the Jamba template primes `<think>`, so the reasoning streams as text and there is nothing to record (0 / 0 / 0 on both). This routing gap predates this change and is left as a follow-up; the record path itself is covered by the streaming route tests.
- `/v1/responses` rejects an `output_text` input part, so a client that echoes `response.output` items verbatim gets HTTP 400; the runs above echo the message text as a string.
- `/v1/messages` without extended thinking records nothing (see section 1).
- Not covered: the ASR compatibility stream, the prompt-inspection routes and the router front; a warm-up for `/v1/responses` or `/v1/messages` (they have none). The store remains per process.
