# Technical Report: PR #2116 - Re-inject generated reasoning into content-only history

**Date**: 2026-10-05

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (chat route, new bounded store, tests), Markdown

**Risk**: Low to medium (a content-only assistant turn may now render with the reasoning the server generated for it, exactly as if the client had echoed `reasoning_content`; only while the prompt cache is on)

## Summary

Issue #2110 followed up PR #2094. After #2094, AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`) hit the prompt cache only when the client echoed `reasoning_content`. A client that sends back only `content`, which is the OpenAI SDK default, got `cached_tokens=0` on every turn. The chat route now remembers the reasoning each reply returned and fills it back into a later content-only assistant turn before rendering, under a key that only the conversation that produced the reply can match.

## 1. Root cause

The Jamba template keeps its thinking instruction on an earlier user turn only when the next assistant message has `reasoning_content` defined (or starts with `<think>`). Without an echoed trace, turn 2 re-rendered turn 1's user message without the instruction, so the turn-1 history-boundary snapshot (#1143), which contains it, no longer prefixed the turn-2 prompt. This is the template's own rule; the server cannot fix it by rendering differently without the trace.

## 2. Design

The issue's candidate fix keyed the store on `(template_sig, hash of assistant content)` narrowed by the session. That key alone is not sound for the anonymous session that most SDK requests share: two unrelated conversations whose replies were both "Yes." would cross-inject reasoning. The implemented key adds a BLAKE3 digest of every message that preceded the assistant turn, exactly as the client sent it. The full key is a domain separator plus model id, `template_sig`, resolved session key, that prefix digest, and the reply content with surrounding whitespace trimmed (a stream delivers the blank line after `</think>` as content while the non-streaming reply strips it). A trace is therefore found only by a request that could itself have produced it.

- Injection runs in the chat route before `prepare_chat_request_with_cache`, on a copy of the request. The render then takes the same raw-JSON path as an echoed trace, so the injected form renders byte-identically to the echoed form by construction. The cache context, tool parsing and the record step keep reading the request as received, so a re-injected turn does not change the key of a later turn.
- Echoed reasoning wins: a message with its own non-empty reasoning is never looked up. Tool-call turns and content with an inline `<think>` block are never filled.
- Edited content, an edited or reordered earlier message, changed kwargs or tools, another session or another model all miss, and a miss changes nothing.
- Replies cut off inside the thinking block have empty content and are recorded too. The prefix digest keeps such an entry tied to its own conversation; if that conversation is re-run, the newest trace replaces the older one, which that conversation could equally have produced. The Jamba streaming run below needed this (turn 2 hit `max_tokens`).
- Bounds: the store holds only the 32-byte key and the text, capped at 16 MiB and 4096 entries with least-recently-used eviction; a hit refreshes the entry, and one trace may use at most an eighth of the budget. `MLXCEL_REASONING_ECHO_MAX_BYTES` sets the budget, `0` disables.
- Off when the prompt cache is not installed, under `cache_prompt: false`, `--reasoning-format none` or `deepseek-legacy` (the trace already travels inline in content), and `--skip-chat-parsing`.

The fallback the issue named (snapshot before the last user turn) was not needed.

## 3. Validation

Unit tests in `reasoning_echo_tests.rs` cover identical rendering of the filled and echoed forms, unstored turns, echoed precedence, edited content and history, other session/template/model, tool-call and inline-think turns, prefill continuations, reasoning-only replies, turn-3 lookups, LRU and byte bounds. The route test in `reasoning_echo_route_tests.rs` drives the real handler (streaming and non-streaming) with a scripted model and fails when the store is disabled; it does not compile on main because it uses a new test-support constructor that captures prompts. The `server::chat_request`, `server::prompt_cache`, `server::routes` and `reasoning` filters pass, with clippy `-D warnings` and rustfmt clean.

Real server on GB10, content-only history, 3 turns, `max_tokens` 2048, temperature 0. "Before" is the same binary with `MLXCEL_REASONING_ECHO_MAX_BYTES=0`, which is the old behavior; the issue records 0 on 33c45053.

| Model | Mode | cached_tokens before | cached_tokens after |
|---|---|---|---|
| jamba-v0.1-4bit | non-streaming | 0 / 0 / 0 | 0 / 41 / 89 |
| jamba-v0.1-4bit | streaming | not run | 0 / 41 / 91 |
| qwen3-0.6b-4bit | non-streaming | 0 / 0 / 32 | 0 / 0 / 32 |
| qwen3.5-0.8b-4bit | non-streaming | 0 / 0 / 36 | 0 / 0 / 36 |

Qwen3 produced identical outputs and prompt lengths both ways: its template renders history assistant turns from content only, so the injected trace has no effect. Its turn-2 miss is pre-existing and out of scope. Qwen3.5-0.8b looped in its thinking block to `max_tokens` on every turn at temperature 0, so that row is weak evidence.

## 4. Limits

- Only `/v1/chat/completions` re-injects. `/v1/responses`, `/v1/messages`, the ASR compatibility stream, the prompt-inspection routes and the router front do not.
- The store is per process and empty after a restart.
- The next-turn warm-up (#1144) was not extended; for Jamba its probe check still bails, as before.
- All anonymous clients share one session bucket, so identical conversations from different anonymous clients can match each other's entry. Such a request could have produced the same reply itself, and the trace is only placed in the prompt, never returned.
