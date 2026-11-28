# Technical Report: Issue #2123 - Route primed-think reasoning on streaming /v1/responses and /v1/messages

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (Responses and Anthropic routes, route tests), Markdown

**Risk**: Low to medium (client-visible: on a template that primes an open thinking block, streamed reasoning moves out of the answer text into reasoning events or a thinking block; unprimed prompts are unchanged)

## Summary

AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`) renders every generation prompt ending in `<|im_start|>assistant\n<think>\n`, so the model writes its trace with no opener and only the `</think>` close. `/v1/chat/completions` already detected this with `is_prompt_primed_open_thinking`, started its stream filter in the thinking state, and set `thinking_enter_block_on_start`. The streaming `/v1/responses` and `/v1/messages` handlers built `StreamFilter::new()` unconditionally, and none of the four Responses and Anthropic arms set the budget flag. Streamed reasoning therefore reached clients as answer text, the #2121 reasoning record stored nothing for streaming turns, and a reasoning budget did not count primed tokens.

## 1. Change

- `stream_create_response` and `stream_messages` compute `primed_open_thinking` from `state.thinking_markers` and `prepared.prompt` before `prepared` moves into the blocking task, and choose `StreamFilter::new_primed_open_thinking()` when it holds.
- All four arms (both endpoints, streaming and non-streaming) set `options.thinking_enter_block_on_start = primed_open_thinking`. The non-streaming arms already split the trace through `split_reasoning` / `split_visible_reasoning`; only the budget flag was missing there.
- `ThinkingDelimiterEcho` (the chat route's #1470 marker re-emission) is not applied: both endpoints carry reasoning on a dedicated channel, so there is no delimiter to echo into the text.

## 2. Stream shapes

Before, streaming `/v1/responses`: `output_item.added` (message), `content_part.added`, then `response.output_text.delta` for every token of the trace and the answer (`</think>` itself was consumed by the filter), `output_text.done` and `completed` with a single `message` item whose text began "We need to answer: ...". No `reasoning` item.

After: `output_item.added` (reasoning), `response.reasoning_text.delta` for the trace, `reasoning_text.done`, `output_item.done` (reasoning), then the message item with `output_text.delta` holding only `\n\nParis.`; `completed` lists `reasoning` then `message`.

Before, streaming `/v1/messages` with extended thinking: one `content_block_start` of type `text` whose `text_delta` events carried the trace and the answer. After: a `thinking` block (`thinking_delta` events) followed by a `text` block holding only the answer. Without extended thinking the trace is dropped from the stream, as the non-streaming arm drops it.

## 3. Validation

New tests in `reasoning_echo_primed_stream_tests.rs` (mounted from `reasoning_echo_endpoint_tests.rs`) use the reduced Jamba template with a primed generation prompt and close-only scripted tokens. They assert the reasoning and text split of both streams, the `completed` output order, the budget flag on all four arms, the flag staying off for an unprimed prompt, and, for 3-turn content-only streaming chats on both endpoints, that turn 3 renders both earlier user turns with the thinking instruction (only possible when both streamed turns recorded non-empty reasoning). Run against the old route code, five of the six fail (the unprimed control passes). The `reasoning_echo`, `server::routes`, `responses_`, `anthropic_`, `reasoning_stream`, `stream_filter` and `stream` filters pass; clippy `--lib --tests -D warnings` and rustfmt are clean.

Real server on GB10, `jamba-v0.1-4bit`, `max_tokens` 2048, temperature 0, 3 content-only streaming turns with the client echoing the text it received. Cached tokens are the scheduler's `prompt-cache: request completed: cached=N/M` line (`/v1/responses` usage matches it).

| Endpoint | Before: reasoning surfaced | Before: cached | After: reasoning surfaced | After: cached |
|---|---|---|---|---|
| `/v1/responses` stream | none (all in `output_text`) | 0/46, 0/217, 0/358 | `reasoning` item, text `Paris.` | 0/46, 41/90, 85/142 |
| `/v1/messages` stream, thinking | none (one `text` block) | 41/46, 212/217, 353/358 | `thinking` + `text` blocks | 0/46, 41/90, 137/142 |

The old `/v1/messages` numbers are high only because the client echoed the leaked trace back as assistant content, which reproduces the generated text verbatim; that history is 2 to 3 times longer and carries the trace in the answer channel.

## 4. Limits

- The streamed text keeps the `\n\n` that follows `</think>`, as `/v1/chat/completions` streaming does; the non-streaming split trims it.
- A primed prompt whose generation never writes the close marker streams everything as reasoning and no text, which matches the chat stream and the non-streaming `max_tokens` behavior.
