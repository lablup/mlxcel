# PR #2415: Align count routes and message_start with usage

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with gemma-3-4b-it-4bit; `message_start` on the `xla-iree` serve loop and the disaggregated coordinator still carries the placeholder count for media bodies (see "Not verified and follow-up")
**Risk**: Medium (changes the Anthropic streaming event order for media bodies and the shared prompt render used by four routes)

## Summary

Issue #2337 follows #2321 (#1748) and #2335 (#1788), which made the count routes share `count_prompt_tokens`. Three older gaps remained where a token count disagreed with what `/v1/messages` or its siblings serve: streaming `message_start.usage.input_tokens` ignored media expansion, the count routes skipped the #2110 reasoning re-echo, and the count routes answered 200 when the server could not generate. This PR closes all three in one change.

## Problem

- `stream_messages` filled `message_start` from `tokenizer.encode(&prepared.prompt, true)`: the placeholder text, with `add_special` forced on and native `prompt_token_ids` ignored. On gemma-3-4b-it-4bit an image body reported 17 in `message_start` against 275 in `count_tokens` and non-streaming `usage`. A text-only body reported 16 against 15, because gemma-3's template already emits BOS.
- `render_admitted_chat_prompt`, shared by `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens` and `/apply-template`, rendered the request as sent. The generating routes also run `reasoning_echo::chat_scope` and `render_request`, so with the echo store on, a content-only assistant turn counted fewer tokens than the generation reported (52 against 78 in the new tests).
- `chat_input_tokens`, `responses_input_tokens` and `anthropic_count_tokens` never ran the chat-unavailable check their siblings run first. Under `--embeddings` or with no chat model, `/v1/chat/completions/input_tokens` answered 200 `{"input_tokens":0}`.

## Technical decisions

- The text-only rule from `count_prompt_tokens` is factored into `prompt_inspection::count_text_prompt_tokens`: native ids win, otherwise `add_special = !prompt_carries_bos`. Text-only streaming bodies use it and `message_start` still goes out immediately. This also removes the BOS over-count.
- A body with media (`PreparedChatRequest::carries_media()`) holds `message_start` until the worker's first `PrefillStats` and uses its `prompt_tokens`, the same post-expansion length that `usage` and the count-only request in `admission.rs` report. `streaming_anthropic::AnthropicMessageStart` holds the event and sends it exactly once. Every path that can emit another event calls `send` first with the best figure available: the prefill report, else the local count before a token or an error, else the result's own `prompt_tokens` on completion.
- Alternative rejected: call `count_prompt_tokens` before dispatch. It would take a second worker slot and repeat media preparation for every streamed media request. The prefill report costs nothing extra and no media is fetched twice.
- The ordering this relies on holds on the batch scheduler, which serves every regular model including the MLX vision families: the prefill completion step sends `GenerateEvent::Prefill` before the first token reaches `stream_decoded_text`. The speculative burst and chunked prefill paths also report before tokens. The Florence-2 and Nemotron-Parse image loops did not send `Prefill`; they now send a first-token snapshot right before their single answer token.
- Count routes only fill the reasoning echo store and never call `record_reply`. A hit refreshes the entry's LRU recency, as a generation would. `/apply-template` shares the render, so its prompt now shows the re-injected reasoning that the generating route would prefill.
- The availability check is the first statement of each count route and returns the sibling's 501 status and body. `anthropic::chat_unavailable_response` is new and shared with `anthropic_messages`.

Trade-off: during a long media queue wait and prefill the client receives only SSE keepalive comments before `message_start`, and with `--sse-ping-interval -1` nothing at all. No event precedes `message_start`, which is the ordering the Anthropic stream requires. `docs/llama-server-compat.md` states this.

## Changes

| Area | Files |
|------|-------|
| Shared text count, media detection | `src/server/routes/prompt_inspection.rs`, `src/server/chat_request.rs` |
| Deferred `message_start` | `src/server/streaming_anthropic.rs`, `src/server/routes/anthropic.rs` |
| Reasoning re-echo on count routes, availability checks | `src/server/routes/prompt_inspection.rs`, `src/server/routes/anthropic.rs` |
| Prefill report on image loops | `src/server/florence2_worker.rs`, `src/server/nemotron_parse_worker.rs` |
| Tests and test support | `src/server/routes/prompt_count_route_tests.rs`, `src/server/routes/availability_tests.rs`, `src/server/model_provider_test_support.rs` |
| Docs | `docs/llama-server-compat.md` |

10 files, +666 -64 against origin/main. Commits: `89f875e6` fix(server): align count routes and message_start with usage; `043063e5` fix(server): report prefill on Florence-2 and Nemotron-Parse loops.

## Verification

Each new test was confirmed to fail with its fix reverted:

- `anthropic_streaming_image_input_tokens_equal_count_and_usage`: `message_start` first, sent once, equal to `count_tokens` and `usage`. Reverted: left 33, right 1033.
- `{chat,responses,anthropic}_count_includes_the_reasoning_echo`: the count equals the length of the prompt the follow-up generation is handed. Reverted: left 52, right 78.
- `count_routes_refuse_like_their_generating_siblings`: all six count paths with no chat model and under `--embeddings`.
- `anthropic_streaming_media_without_a_prefill_report_still_opens_with_message_start`: a backend that streams a token with no prefill report still opens with one `message_start`.
- `message_start_goes_out_once_with_the_first_figure` and the text-only agreement test (the latter passes on the reverted code too, because the stub tokenizer adds no BOS).

Local gate (`--features cuda`, `--profile test-fast`, `--test-threads=1`): `prompt_inspection` 28 passed (1 ignored, needs ffmpeg), `availability_tests` 6, `routes::anthropic` 21, `streaming_anthropic` 4, `reasoning_echo` 30, `florence2_worker` 15, `nemotron_parse_worker` 7. Clippy with `-D warnings`, `cargo fmt --check` and the repository check scripts pass.

Real checkpoint `~/models/mlx/gemma-3-4b-it-4bit`, one body sent to `count_tokens`, non-streaming `/v1/messages` and streaming `/v1/messages`:

| Body | Build | count_tokens | usage.input_tokens | message_start.usage.input_tokens |
|---|---|---|---|---|
| image | main 93e5ef75 | 275 | 275 | 17 |
| image | this PR | 275 | 275 | 275 |
| text only | main 93e5ef75 | 15 | 15 | 16 |
| text only | this PR | 15 | 15 | 15 |

## Not verified and follow-up

- The `xla-iree` serve loop (`src/server/batch/xla_worker.rs`) and the disaggregated coordinator never send a prefill report. For a media body, `message_start` there carries the local placeholder count, as before. The XLA loop is not compiled by the local gate and the coordinator change belongs with that subsystem.
- The BOS over-count fix is shown on the real checkpoint only; the route-test stub tokenizer adds no BOS.
- The full `server::routes` test filter was not run locally (too large for the shared box); CI covers it.
- The gemma-3 checkpoint has no reasoning channel, so the re-echo is covered by route tests only.
