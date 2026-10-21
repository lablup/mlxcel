# PR #2326: Fail the LoRA load on unclaimed runtime terms and tied output heads

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints before merge; the tied-embedding follow-up is noted, not fixed (see "Not verified and follow-up")
**Risk**: Medium (adapters that used to load and silently serve base weights, or silently drop a delta, now fail the load)

## Summary

Issue #1577 closes three adapter-validation holes left after #1328 and PR #1576. In each, an adapter loaded successfully while part of it had no effect.

1. Runtime adapter terms that no model constructor claimed were reported as a `warn!` and dropped. `finish_runtime_staging()` now returns the sorted unclaimed prefixes and `load_runtime_adapted_model()` fails the load with one error listing them. The drain still runs on the construction-error path, so a failed load leaves nothing staged.
2. An `lm_head` adapter pair on a tied-embedding checkpoint is refused on every path (single-process fused, runtime, and the pipeline stage loader) by `lora::reject_tied_output_head_adapter`. The verdict comes from safetensors headers, before any weight is read.
3. 3-D MoE expert LoRA pairs (`LoRASwitchLinear`) are rejected once per adapter in `validate_adapter_tensors` with a mixture-of-experts message instead of "Expected 2D LoRA matrices".

## Problem

The unfused runtime path stages per-layer terms keyed by base weight prefix, and a term only takes effect if a layer constructor calls `runtime_lora::claim` with that prefix. Three constructors claim (`Linear::from_weights`, `UnifiedLinear::from_weights`, `FusedQKVLinear`). A projection built anywhere else, such as the DeepSeek MoE router read through `get_weight_copy`, leaves its term unclaimed. The old `finish_runtime_staging` logged one warning per prefix and returned `()`, so the server answered from the base model while the log said the adapter was staged. This is the #1328 failure mode through a second door, and it is the default channel for `--lora` and `--lora-scaled`.

The tied-head case was two verdicts for one adapter. Single-process refused an `lm_head` pair because the raw weight map has no `lm_head.weight`. The pipeline path loaded through `load_text_weights`, which synthesizes `lm_head.weight` from the embeddings, so the adapter fused into that copy. `LlamaStageExecutor` then built its head from `model.embed_tokens` when the config ties, so the delta was dropped without a message.

The MoE limit was already a hard error but surfaced as a tensor-rank complaint and was undocumented.

## Technical decisions

- Refuse the tied head rather than apply it. Applying it would need a delta written into a copy of the embedding matrix, which unties projections the checkpoint was trained to share. `mlx_lm.lora` only wraps modules inside `model.layers`, so no adapter it trains is affected.
- Decide from the checkpoint, not from the sanitized map. The rule is explicit `tie_word_embeddings: true`, or no stored `lm_head.weight`. A checkpoint with no flag that stores its own `lm_head.weight` counts as untied, because family defaults for the flag differ. Only safetensors headers are read, so the refusal is cheap and happens before weights load.
- Make the failure mode uniform. The helper is called by `load_model_with_adapter`, `load_model_with_adapter_specs` (fused and unfused) and the pipeline Llama stage loader, so the error text is identical on both paths and a test compares it word for word.
- Fix the constructor, not the check. A family that legitimately builds a targeted module outside the claiming constructors now refuses the adapter; the remedy is to make that constructor claim, not to relax the error.
- Sort the unclaimed list. `drain_unclaimed` drains a `HashMap`, so an unsorted message would be flaky across runs.
- Keep `drain_unclaimed` as the only source of unclaimed prefixes. No second reporting path.
- Report MoE once per adapter in the shared validator, so the fused, unfused and stage paths agree.

## Changes

- `src/lib/mlxcel-core/src/runtime_lora.rs`, `src/lora/runtime.rs`: `finish_runtime_staging` is `#[must_use]` and returns the sorted prefixes.
- `src/loading/mod.rs`: new `load_runtime_adapted_model` binds the drain result, keeps the construction `?`, then fails on a non-empty list; tied-head check wired into both single-process entries.
- `src/lora/tied_head.rs` (new): `reject_tied_output_head_adapter`. `src/lora/mod.rs` exports it.
- `src/lora/pair_rank.rs` (new), `src/lora/loader.rs`: 3-D pair screen in `validate_adapter_tensors`.
- `src/distributed/pipeline/stage_executor/llama.rs`: calls the shared tied-head check before composing the adapter.
- Tests: `src/loading/adapter_load_tests.rs`, `src/lora/tied_head_tests.rs`, `src/lora/pair_rank_tests.rs`, `src/lora/runtime_tests.rs`, `src/lora/test_support.rs`, `src/distributed/pipeline/stage_executor/llama_adapter_tests.rs`.
- `docs/server-features.md`, `docs/llama-server-compat.md`: the three rules.

## Behavior change

Adapters that previously loaded now fail: one with terms on modules built outside the claiming constructors, one with an `lm_head` pair on a tied checkpoint (pipeline path), and one with 3-D expert pairs now gets a clearer message. A failed runtime load reports every unclaimed layer at once.

## Verification

Real checkpoints, `mlxcel-server --lora <adapter>`, `/completion` at temperature 0, `MLXCEL_SDPA_DETERMINISTIC=1`. Adapters were synthetic mlx-lm-format with random non-zero `lora_a`/`lora_b`, rank 8, scale 20.

- `qwen3-0.6b-4bit`, q/k/v/o on all 28 layers (112 pairs, all claimed): loads and the output changes versus no adapter. At `--lora-scaled <adapter>:0.0` it matches the no-adapter output.
- `deepseek-v2-lite-chat-4bit-mlx`, pairs on `model.layers.{1,2}.mlp.gate` (the MoE router): the load fails with "Runtime LoRA adapter terms for 2 layers were not claimed by the model" naming both prefixes, and `/health` returns 503. The control with only `q_proj`/`o_proj` pairs loads and answers coherently.
- `qwen3-0.6b-4bit` with an `lm_head` pair: the load fails with the tied-head error.
- Revert check: with the unclaimed check, tied-head verdict and MoE screen all disabled, 11 new tests fail and the positive controls pass.
- `lora::` (49 pass), `loading::` (350 pass), `stage_executor::llama` (3 pass), `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings`, `cargo fmt --check` and the contract checks pass.

## Not verified and follow-up

- The sort of the unclaimed list is not pinned by a failing test: `finish_runtime_staging_returns_every_unclaimed_prefix_sorted...` still passes with the sort removed, because a three-key `HashMap` happened to drain in order. The sort is covered by inspection and the error-format tests.
- Known gap, not fixed: on a tied checkpoint an adapter pair on `model.embed_tokens` is claimed by the output head and applied as an output-head delta. It needs its own decision and issue.
- Only dense Llama-style and DeepSeek-V2 were run end to end. Other families that build targeted modules outside the three constructors will start refusing adapters; diff any new family the same way.
