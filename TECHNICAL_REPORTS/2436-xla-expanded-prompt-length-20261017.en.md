# PR #2436: Report the expanded media prompt length on the xla-iree loop

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with gemma-3-4b-it-4bit through the XLA backend; Phi4MM audio was not run on a real checkpoint (see "Not verified and follow-up")
**Risk**: Medium (changes the prompt count that `usage`, the count routes and `message_start` report for media bodies on the `xla-iree` loop, and adds a new `Prefill` event to that loop)

## Summary

Issue #2420 follows #2415 (#2337), which made the Anthropic streaming `message_start` wait for the worker's first `GenerateEvent::Prefill`. The `xla-iree` serve loop never sent `Prefill` and reported the logical (placeholder) prompt length, so a media body's `usage`, `/v1/messages/count_tokens` and `message_start` all carried the placeholder count. This PR makes the loop report the expanded prepared length and send `Prefill`, matching the batch scheduler and `docs/llama-server-compat.md`.

## Problem

- `dispatch` in `src/server/batch/xla_worker.rs` sent only `Token`, `Done` and `Error`, so a streaming route waiting for the prefill report got none.
- `xla_worker_admission.rs` set `prompt_token_count` (the logical count) next to `effective_prefill_len` (`prepared.sequence_len()`), and the result carried the logical count. The #895 split kept the expanded length for internal throughput and KV accounting only.
- The XLA worker ignores `count_prompt_only` and answers the count route with a one-token generation, so `/v1/messages/count_tokens` reported the same placeholder figure.
- The existing image test expanded the request to 5 tokens and asserted `prompt_tokens == 3`, which pinned the old behavior.
- On gemma-3-4b-it-4bit through the XLA backend (main `cadc6366`), an image body reported 18 in `count_tokens`, `usage` and `message_start`, against 277 on the MLX path.

## Technical decisions

- `ServeState` keeps one count, `effective_prefill_len`: the expanded prepared length for media and the prompt length for text. The separate `prompt_token_count` field is removed, which reverses the #895 split. Text requests are unchanged because both lengths are equal.
- `ServeState::report_prefill` sends `PrefillStats { prompt_tokens, cached_tokens: 0, processed, first_token: true, prompt_ms }` once, guarded by a `prefill_reported` flag. It is called before the first `Token`, and before `Done` when the engine finishes with no token (a withheld EOS). It also stamps the decode state's first-token instant, as `sequence.rs` does on the batch scheduler.
- `Done`, the stop-string finish (`finalize_stop`) and the zero-budget media finish (`finish_zero_budget`) report the expanded length. `finish_zero_budget` now takes the length as a parameter, because `max_tokens = 0` returns before the old text-only count applied.
- Every XLA image family goes through `handle_preprocessed`, including the new Molmo (#918) and Youtu-VL (#919) producers, so they are covered. So are the DeepStack (Qwen3-VL) and Phi4MM audio and image paths.
- No doc change: `docs/llama-server-compat.md` has no XLA-specific statement, and its general promise now holds on this loop.
- The disaggregated coordinator stays out of scope, as the issue says. `coordinator.rs` drops `Prefill`, but the router front serves only `/v1/chat/completions` and `/v1/completions` and refuses media, so no media or Anthropic body reaches it.

## Changes

| Area | Files |
|------|-------|
| `ServeState` count, `report_prefill`, `Done` and stop-string finish | `src/server/batch/xla_worker.rs` |
| Admission paths, zero-budget finish | `src/server/batch/xla_worker_admission.rs` |
| Tests | `src/server/batch/xla_worker_tests.rs` |

3 files, +328 -40 against origin/main. Commit: `815a6da0` fix(server): report the expanded media prompt length on the xla-iree loop.

## Verification

The new and updated tests drive the real `admit`, preprocessing and `dispatch` path with a fake engine:

- Text gets `Prefill(2)`, the image gets `Prefill(5)`, and Phi4MM audio and image each get `Prefill(6)`. Every one is sent before its `Token`, and `Done` reports the same length.
- Zero-budget image, zero-budget audio and a bare EOS finish are covered.
- The fake Phi4MM producer now expands placeholders.
- Two existing assertions in this module were already failing on main because the error messages had changed (`structured` is now `constrained decoding`, and `refusing text fallback` is now `image resolution cardinality mismatch`). They were updated so the module runs green.

Local gate: `server::batch::xla_worker` 15 passed (`--release --features cuda,xla-iree`, `--test-threads=1`, under gpu-lock). With the production files restored to main, 9 tests failed: the 7 new or updated tests plus the 2 stale assertions. `cargo clippy --release --features cuda,xla-iree --lib --tests -- -D warnings`, `cargo check --release --features cuda,xla-iree --all-targets`, `cargo fmt --check` and the repository check scripts pass.

Real checkpoint `~/models/mlx/gemma-3-4b-it-4bit`, `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_CONTEXT_CAPACITY=512 mlxcel-server` on the GB10 (the default 256 capacity refuses a Gemma 3 image). One body (`tests/fixtures/test_image.png` plus a short prompt) was sent to `count_tokens`, non-streaming `/v1/messages` and streaming `/v1/messages`:

| Body | Build | count_tokens | usage.input_tokens | message_start.input_tokens |
|---|---|---|---|---|
| image | main cadc6366, XLA | 18 | 18 | 18 |
| image | this PR, XLA | 277 | 277 | 277 |
| image | this PR, MLX path | 277 | 277 | 277 |
| text | both builds, XLA | 17 | 17 | 17 |

In every stream `message_start` was the first event, and it was sent once.

## Not verified and follow-up

- Phi4MM audio was not run on a real checkpoint (none is present locally), so the fake-producer test covers it.
- The disaggregated coordinator is out of scope for this PR.
