# PR #3330: Force the K2-Horizon budget close that matches the opened variant

**Date**: 2026-11-23
**Status**: PR open (closes #3321). Unit suites pass; verified on `models/k2-horizon-0.9b` through `mlxcel-server` on an M3 Ultra

## Executive Summary

IFM K2-Horizon's chat template opens the reasoning block with one of three tags chosen by `reasoning_effort`: `<ifm|think>` (high), `<ifm|think_fast>` (medium) or `<ifm|think_faster>` (low). Each has its own close. #3303 taught the thinking-budget tracker to recognize all three closes, but when a `thinking_budget_tokens` cap ran out, or a `reasoning_end` control request arrived, it always forced `</ifm|think>`. A `medium` or `low` request therefore saw a close the model was never trained to pair with its open. A variant open emitted by the model itself was also not recognized, so such a block was never budgeted.

This PR makes `ThinkingState` remember which close pairs with the open that started the block, from the primed prompt or from the model's own open, and makes both forcing paths emit that close. On the real checkpoint a `medium` request with a budget of 6 now ends its reasoning with `</ifm|think_fast>` after six tokens. The PR also replaces a stale ignored Qwen3.5 test that expected a panic #2442 had removed.

## 1. Problem Statement

`resolve_thinking_token_ids` returns `ThinkingTokenIds { open, close, alt_closes }` with `close = </ifm|think>`. `ThinkingState::decide_override` forced `ids.close` on both of its forcing paths: the primed Pending path (budget 0 or an armed force-end before the first token) and the InBlock path (cap reached or force-end). `observe` entered the block only on `ids.open`.

| Case | Before | After |
|---|---|---|
| `medium` primed, cap reached | `</ifm|think>` | `</ifm|think_fast>` |
| `low` primed, cap reached | `</ifm|think>` | `</ifm|think_faster>` |
| Model emits `<ifm|think_fast>` unprimed | stays Pending, never budgeted | InBlock, closed with `</ifm|think_fast>` |
| Qwen, Gemma 4, `<|content_thinking|>` | pair close | unchanged |

## 2. Change Summary

| Area | Change |
|---|---|
| `server/thinking_budget.rs` | `ThinkingTokenIds::alt_opens`, index-paired with `alt_closes` and filled from the same `THINK_EFFORT_VARIANTS` entry; `close_for_open`. `ThinkingState` gains a private `active_close`, set by `observe` on any open and by `with_primed_close` for a primed prompt (kept only when `ids.is_close(id)`). Both forcing paths call `forced_close()`, which is `active_close.unwrap_or(ids.close)`. New `resolve_primed_close_id` maps the marker string to a token id. |
| `server/config.rs` | `ServerGenerateOptions::thinking_primed_close: Option<String>`. |
| Routes | Every site that sets `thinking_enter_block_on_start` from the primed prompt also sets the new field from `primed_open_thinking_close_marker`: `prepare_chat_generation` (chat completions and the `mlxcel chat` / `mlxcel run` CLI through `InProcessServer::chat`), `/v1/responses` run and non-streaming paths, responses compaction, Anthropic `/v1/messages` (both paths), the ASR chat stream, and the router front. |
| `batch/scheduler` | `build_thinking_state` takes the marker, resolves it with `resolve_primed_close_id` and attaches it with `with_primed_close`. |
| `models/qwen3_5_tests.rs` | `rollback_speculative_cache_panics_on_ragged_batched_accepts` replaced by `rollback_speculative_cache_compacts_ragged_batched_accepts`. |
| Struct literals | `thinking_primed_close: None` in the warmup, engine probe, request-options builder and test literals. |
| Docs | `docs/supported-models.md` (K2-Horizon entry) and `CHANGELOG.md` (Fixed). |

## 3. Technical Decisions

### 3.1 Track the close on the state, not on the id table

The id table is per tokenizer and shared by all sequences; which variant a given block opened is per sequence. A single `Option<i32>` on `ThinkingState` costs nothing on the hot path: `forced_close` runs only when a force is due, and `close_for_open` runs only in unprimed Pending (one compare plus a two-slot scan). `ThinkingTokenIds` stays `Copy`.

### 3.2 Resolve the primed close at admission, from the string the routes already compute

The routes already call `primed_open_thinking_close_marker` for the stream filter and the #1470 delimiter echo. Carrying that string in `ServerGenerateOptions` and resolving it once at admission with the scheduler's own tokenizer keeps the routes free of token ids. The marker is resolved only when `enter_block_on_start` is true, and `with_primed_close` drops any id that is not one of the pair's closes, so a multi-token close (`<|close|>think<|sep|>`) or an unrelated family's marker falls back to the pair's close instead of forcing a stray token.

### 3.3 A variant open without a close opens nothing

`close_for_open` returns `None` for a variant whose close is missing from the vocabulary. Falling back to `ids.close` would reintroduce the mismatched pairing this PR removes. The real K2 vocabulary has all six tokens, so the case is theoretical; the doc comment records the reason.

### 3.4 The disaggregated handoff is out of scope

The router front sets the field, but the prefill frame's sampling payload (`kv_cache_serde`) does not carry it, so the decode node's `build_handoff_thinking_state` still forces `</ifm|think>` for a primed variant. Changing the wire format was explicitly excluded by #3321. A model-emitted variant open on the decode side is now tracked, because the handoff replays generated tokens through `observe`. A comment at the router-front assignment and the docs state the limit.

### 3.5 The Qwen3.5 test contract was stale, not the code

Since #2442, `rollback_speculative_cache_rows` compacts each row's accepted block and replays per-row gated-delta state instead of panicking on ragged accepts. The new test verifies rows `[1, 2, 3, 4]` and `[5, 6, 7, 0]` with accepts `[3, 1]` and checks each row's `conv_state` and `state_cache` against a B = 1 rollback of that row with its own accept, and the attention offset against `max(accepted) + 1` = 4. The shared helper now checks the returned count against `max(accepted)`, which is what the function documents for both uniform and ragged accepts.

## 4. Verification

| Selector | Result |
|---|---|
| `server::thinking_budget server::k2_horizon_reasoning_tests` | pass (thinking_budget suite unchanged; 12 K2 tests including 6 new) |
| `server::routes::chat` | pass |
| `models::qwen3_5_tests --ignored --test-threads=1` (GPU lock) | 7 of 7 pass |
| `coverage_tests` | 18 pass |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --check`, the seven CI contract scripts | clean |

Real checkpoint, `mlxcel-server --skip-chat-parsing` on `models/k2-horizon-0.9b`, greedy, `thinking_budget_tokens: 6`, "What is 17 times 23? Think step by step.":

| Effort | Raw output start |
|---|---|
| high | `The user asks: "What</ifm|think>\nWe need to answer: ...` |
| medium | `We need to compute 17</ifm|think_fast>\nWe need to answer: ...` |
| low | `</ifm|think_faster>\nLet's calculate ...` (the model closes on its own at the first token) |

With parsing on, the six budgeted reasoning tokens land in `reasoning_content` and the rest in `content` for all three efforts; with no budget, `medium` closes naturally with `</ifm|think_fast>` at token 39.

## 5. Known Limits

- The disaggregated prefill/decode handoff still forces `</ifm|think>` for a primed variant (section 3.4).
- Forced tokens do not appear in the chat `logprobs` array (observed during validation; pre-existing and unrelated to this change).
