# Technical Report: PR (issue #1979), Nemotron-H Nano Omni per-sequence state

**Date**: 2026-09-26

**Status**: Implemented and unit-tested; no Nano Omni checkpoint was available, so the wrapper has not run on a real server. Not yet merged.

**Languages**: Rust

**Risk level**: Low

## Executive summary

`NemotronHNanoOmniVlModel` implemented only the single-sequence subset of `LanguageModel`. The per-sequence entry points (`forward_with_sequence_id`, `prepare_sequence_state`, `release_sequence_state_by_id`, the snapshot methods) fell back to trait defaults, so on mlxcel-server every request ran on the Nemotron-H text model's one fallback slot: concurrent requests shared KV and Mamba state, and model-owned prompt reuse never stored a snapshot. The wrapper now forwards the whole per-sequence and snapshot surface to the inner `NemotronHModel`, and the embeddings prefill runs on the request's own slot.

## Problem statement

The scheduler allocates a model-owned slot per request and calls `prepare_sequence_state`, then prefills through `forward_last_logits_with_embeddings_and_sequence_id` (multimodal) or `forward_last_logits_with_sequence_id` (text or adopted-prefix suffix), and decodes through `forward_with_sequence_id`. With the defaults, all of these ignored the sequence id. The inner model's own `forward_last_logits_with_embeddings_and_sequence_id` also discards the embeddings and the id (it is a text model), so a plain delegation of that method would have dropped the image and audio rows.

## Change summary

- `NemotronHModel` gains `forward_with_inputs_embeds_and_sequence_id` and `last_logits_with_inputs_embeds_and_sequence_id`, which run the layer stack from precomputed embeddings on the slot of a given sequence id; `forward_with_inputs_embeds` (CLI) is now the `None`-slot case of the first.
- The wrapper implements both embeddings entry points itself (embeddings go to the new inner methods, the token path to the inner seq-id methods) and delegates `forward_with_sequence_id`, `forward_last_logits`, `forward_last_logits_with_sequence_id`, `sequence_state_layout`, `prepare_sequence_state`, `release_sequence_state_by_id`, `reset_runtime_state`, `supports_snapshot_reuse`, `snapshot_sequence_state`, `restore_sequence_state`, `supports_batching`, and `supports_padded_prefill` to the text model instead of hardcoding answers.
- No admission hook was added: Nemotron Omni image and audio preparation is stateless and returns `InputEmbeddings` carried on the sequence, so there is no fallback-slot side state to bind (unlike Qwen VL MRoPE or Gemma 4 per-layer inputs).
- The `EXEMPT` entry in `src/vision/snapshot_forwarding_tests.rs` is removed, so the source scan now covers the wrapper.

## Validation

- New test `sequence_ids_keep_isolated_state_on_token_and_embedding_paths` builds the wrapper around a one-layer attention Nemotron-H and the synthetic RADIO tower and checks that two sequence ids give identical prefill and decode logits for the same input, on both the token and the embeddings path, plus snapshot and release behavior. Against the old wrapper it fails (max logit diff 0.0596 on the second sequence's prefill), as does the source-scan test.
- `cargo test --release --features cuda --lib -- vision:: models::nemotron server::batch --test-threads=1`: 998 passed.
- `cargo clippy --release --features cuda --lib --tests -- -D warnings` and `cargo fmt --all --check`: clean.
- Text-only sanity on the inner model: `mlxcel-server -m nemotron-h-30b-4bit`, 3-turn chat, `cached_tokens` 0 / 3301 / 3320 of 3294 / 3313 / 3332 prompt tokens.

## Remaining work

The issue's real-server multi-turn check on the Nano Omni wrapper is not done: no `nemotron_h_nano_omni` checkpoint exists under `/home/inureyes/models/mlx`. It should be run once one is available.
