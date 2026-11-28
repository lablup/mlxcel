# PR #2416: Refuse an embed_tokens LoRA pair on a tied checkpoint

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints before merge; mlx-lm's embedding-adapter semantics are not implemented (see "Not addressed and follow-up")
**Risk**: Low (a load-time refusal in one verdict function shared by three load paths; untied checkpoints and base runs are unchanged)

## Summary

Issue #2332 follows #2326. On a tied checkpoint the output head reads the embedding matrix, so a LoRA pair on `model.embed_tokens` addresses both ends of the tie. The fused path, the runtime path and the pipeline stage loader disagreed about such a pair, with each other and with mlx-lm. This PR extends `lora::reject_tied_output_head_adapter` so that an `embed_tokens` pair on a tied checkpoint is refused on every path with one error, before any weight is read.

## Problem

mlx-lm's `LoRAEmbedding` (`mlx_lm/tuner/lora.py`) stores the pair as `lora_a = [vocab, rank]` and `lora_b = [rank, hidden]`. `__call__` adds `scale * lora_a[x] @ lora_b` to the lookup, `as_linear` adds `scale * (x @ lora_b.T) @ lora_a.T` to the tied head, and `fuse` writes `W + scale * lora_a @ lora_b` into the one matrix both ends read. So mlx-lm adapts the lookup and the tied head alike. `mlx_lm.lora` creates the pair only when `lora_parameters.keys` names `model.embed_tokens`.

mlxcel reads every pair in the linear orientation, which produced three different outcomes:

- Fused path (`--adapter`, `--lora-fuse`) and pipeline stage: a linear-shaped pair (`[hidden] -> [vocab]`) was folded into the shared matrix, so both ends were adapted (dense base only). An mlx-lm-layout pair failed with a shape mismatch.
- Runtime path (`--lora`): the linear-shaped pair was claimed by `UnifiedLinear::from_weights(.., "model.embed_tokens", ..)`. On the Llama family it reached the logits only. On Qwen3, whose head goes through `as_linear`, it failed as unclaimed. An mlx-lm pair failed a different shape check.

On a real run before the fix, `mlxcel-server -m llama-3.2-1b-instruct-4bit --lora <linear pair>` loaded, and the first-token logprobs moved through the head alone (` Paris` -0.359 to -0.3125, `...` -3.477 to -3.852). The input embedding was untouched, which no other framework would produce for that adapter.

## Technical decisions

- Refuse instead of matching mlx-lm. Matching needs an embedding-orientation pair type, a runtime lookup term in `UnifiedEmbedding`, and a transposed runtime head term in every family that builds its head from the embedding. That is a separate feature. Refusing reuses the verdict and the call sites from #2326, which all three paths already invoke before reading weights.
- The verdict now models both ends with a private `TiedTarget` enum: `Head(stem)` and `Embedding(stem, head)`. The embedding variant carries the head whose stored weight decides the tie, so the tie test is made against the right head for each stem.
- Three embedding stems are matched, each paired with its head: `model.embed_tokens` with `lm_head`, `language_model.model.embed_tokens` with `language_model.lm_head`, and `model.language_model.embed_tokens` with `lm_head` (the transformers 4.52+ VLM layout). A checkpoint is tied for a pair when `config.json` sets `tie_word_embeddings` (top level or `text_config`) or when it stores no `<head>.weight` for that pair's head.
- A stem ending in `.weight` (`model.embed_tokens.weight.lora_a`, `lm_head.weight.lora_b`) is read the way the base-weight resolver reads it. This closes a bypass that existed for the head pair as well.
- The error text is built per kind. A head pair keeps the #2326 wording. An embedding pair gets an explanation of why it is refused and a pointer: if `mlx_lm.lora` trained the adapter, merge it with `mlx_lm.fuse` and load the merged model, otherwise remove the pair.

Trade-off: an adapter trained by `mlx_lm.lora` with `model.embed_tokens` in its keys cannot be loaded as an adapter on a tied checkpoint. The user has to fuse it first. This is a clear error in place of three different silent or confusing outcomes.

## Changes

- `src/lora/tied_head.rs`: `INPUT_EMBEDDINGS`, `TiedTarget`, `targeted_tied_ends`, per-kind error text, `.weight` stem handling.
- `src/lora/tied_head_tests.rs`: verdict unit tests for each stem, tied-by-config and tied-by-missing-head cases, the `.weight` stems, and untied pass-through.
- `src/loading/adapter_load_tests.rs`: a parity test asserting the fused, runtime and pipeline-stage errors are equal strings in both orientations, and a single-process test showing the runtime path no longer reaches the head.
- `src/distributed/pipeline/stage_executor/llama.rs`, `llama_adapter_tests.rs`: the stage loader call site and its test.
- `docs/llama-server-compat.md`, `docs/server-features.md`: describe the refusal.

## Behavior change

A LoRA pair on `model.embed_tokens`, `language_model.model.embed_tokens` or `model.language_model.embed_tokens` is refused at load on a tied checkpoint, on `mlxcel generate --adapter`, `--pp-size N --adapter`, `--lora-fuse` and `mlxcel-server --lora`. Untied checkpoints are unaffected. Head pairs behave as in #2326.

## Verification

- New tests fail with the fix functionally reverted (embedding match disabled). With the revert the runtime model loads with the term claimed and fails only later, at the missing tokenizer.
- `cargo clippy --profile test-fast --features cuda --lib --tests -D warnings`, `cargo fmt --check` and the five contract scripts pass.
- Real checkpoints with synthetic mlx-lm-format adapters (rank 2, `keys: ["model.embed_tokens"]`), `MLXCEL_SDPA_DETERMINISTIC=1`, prompt "The capital of France is". After the fix, `mlxcel generate --adapter`, `mlxcel generate --pp-size 2 --adapter` and `mlxcel-server --lora` on `qwen3-0.6b-4bit` and `llama-3.2-1b-instruct-4bit`, with both layouts, all print the same refusal. Base runs are unchanged.

## Not addressed and follow-up

- On an untied checkpoint, an `embed_tokens` pair still fuses on the fused path and is refused as unclaimed on the runtime path.
- Tied families with other embedding names (`transformer.wte`, `model.word_embeddings`, `backbone.embeddings`) are not on the list. Their heads use `as_linear`, so the runtime path refuses such a pair as unclaimed and never applies it to the head alone.
- Native support for mlx-lm embedding adapters (pair type, lookup term, transposed head term) would need its own issue.

## Statistics

7 files changed, 508 insertions, 55 deletions. Commits: `310e26b5` fix(lora): refuse an embed_tokens pair on a tied checkpoint, `56d0cd78` fix(lora): close the .weight stem gap in the tied-head verdict.
