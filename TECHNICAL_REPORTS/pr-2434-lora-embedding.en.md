# PR #2434: Support mlx-lm LoRAEmbedding adapters on embed_tokens

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with real checkpoints before merge; fusing into a quantized embedding table and other tied families are not covered (see "Not addressed")
**Risk**: Medium (adds a runtime LoRA term to the embedding lookup and to the tied head, and lifts the #2416 refusal for Llama, Qwen2 and Qwen3; other families keep the refusal)

## Summary

Closes #2421, which builds on #2416 (#2332). mlxcel now serves mlx-lm `LoRAEmbedding` adapters on `model.embed_tokens`, both fused (`--adapter`, `--lora-fuse`, pipeline stage loader) and unfused (`--lora`), with the semantics mlx-lm uses. On a tied checkpoint the pair adapts the lookup and the head together. On an untied checkpoint it adapts the lookup only.

## Problem

#2416 refused an `embed_tokens` pair on tied checkpoints on all three load paths. Two gaps remained:

- mlxcel could not serve these adapters unfused.
- On an untied checkpoint the fused path fused an `embed_tokens` pair while the runtime path refused it as unclaimed (`src/lora/runtime.rs:387`), because only linear constructors claimed terms (`src/lib/mlxcel-core/src/layers.rs:1319`).

The reference semantics are those of mlx-lm 0.32.2 with mlx 0.32.4. `LoRAEmbedding` stores `lora_a [vocab, r]` and `lora_b [r, hidden]`. `__call__` adds `scale * lora_a[x] @ lora_b`, `as_linear` adds `scale * (x @ lora_b.T) @ lora_a.T`, and `fuse` writes `W + (scale * lora_a @ lora_b).astype(W.dtype)`. `linear_to_lora_layers` wraps `model.embed_tokens` only when `lora_parameters.keys` names it.

## Technical decisions

- Pair layouts. A pair whose base weight stem ends in `embed_tokens` is read against the table's vocab rows in this order: mlx-lm `LoRAEmbedding` (`a [V, r]`, `b [r, H]`), the linear `[hidden] -> [vocab]` pair in mlx-lm orientation (`a [H, r]`, `b [r, V]`), and in PEFT orientation (`a [r, H]`, `b [V, r]`). All three canonicalize to `delta = A @ B`. When `vocab == hidden` the mlx-lm reading wins.
- Fusion dtype. The fused delta is computed in f32 and cast to the table dtype before the add, as `mlx_lm.fuse` does, so the table keeps its dtype. A fused linear still promotes to f32, unchanged.
- Dense base only for fusion. A quantized table (one with a `.scales` plane) is refused with an error pointing to `--lora`. The refusal keys on the `.scales` plane instead of a shape mismatch, which a delta as wide as the packed columns would have passed.
- Runtime staging. Embedding pairs go into their own map as `RuntimeLoraEmbeddingTerm` (f32 `a`, `b`, the adapter's `SharedLoraScale`), held in an `Arc`. `claim_embedding` hands the term to the lookup. `claim(prefix)` also attaches the tied-head form to any linear built from the embedding's prefix. `drain_unclaimed` reports an embedding term no lookup claimed, so a head-only attachment still fails the load.
- Width check at claim. `UnifiedEmbedding::from_weights` checks every term against the table's real `[rows, width]` (for a packed table, groups times the reconciled group size), so a pair of the wrong hidden width fails the load instead of aborting at the first forward.
- Refusal lift is narrow. The #2416 refusal is lifted for `model.embed_tokens` only, and only when the family's constructors read the head from the embedding (`applies_tied_embedding_pairs`): `ModelType::Llama` / `Qwen2` with `tie_word_embeddings: true`, and `Qwen3` unless the config sets it `false`. Everything else keeps the #2416 error with its wording unchanged.

Deviations from the issue design:

- The tied head on Llama gets a `RuntimeLoraTerm` built from the shared term's buffers (transposed views, same scale handle) instead of holding the `Arc` itself. The head is a `UnifiedLinear`, whose fused-kernel guards (`has_active_runtime_lora`, `as_quantized_weight`, `quantized_weight`) already key on its `runtime_loras`, so this keeps those guards correct without new branches in `UnifiedLinear`. No tensor data is copied.
- A Llama or Qwen2 checkpoint that is tied only because it stores no `lm_head.weight` (no `tie_word_embeddings: true`) keeps the refusal. Llama's `tie_word_embeddings` defaults to `false`, so it builds its head from the `lm_head` copy that `sanitize_tied_embeddings` makes. The fused path makes that copy after fusing and the runtime and stage paths before, so the paths would disagree. Qwen3 defaults to tied and reads `as_linear`, so it is lifted in that case.
- `mistral` (`ModelType::Llama`) is covered on the single-process paths because it loads through the same Llama constructors. Its pipeline stage executor refuses adapters entirely, as before. `qwen2` has no pipeline stage family, so the pipeline criterion is Llama only.

## Changes

- `src/lora/embedding.rs` (new): layout detection and canonicalization to `delta = A @ B`.
- `src/lora/loader.rs`: the fusion loop routes embedding targets to that delta. This covers `--adapter`, `--lora-fuse` and the pipeline stage loader, which share `fuse_lora_weights_into`.
- `src/lora/runtime.rs`: embedding pairs are canonicalized and staged as `PendingLoraEmbeddingTerm`s; the shape check reuses `validate_pair_shapes` with rows = vocab and columns = hidden.
- `src/lib/mlxcel-core/src/runtime_lora.rs`: `RuntimeLoraEmbeddingTerm`, `stage_with_embeddings`, `claim_embedding`, the head form via `head_term()`, and `drain_unclaimed`. `stage` clears the embedding map too.
- `src/lib/mlxcel-core/src/layers.rs` (embedding region only): `Embedding` and `QuantizedEmbedding` carry an `EmbeddingLoraTerms` built once at claim. `forward` adds the lookup term, cast to the output dtype before the add, as mlx-lm does. `as_linear` adds the head term through `apply_terms`. `clone_shared` does not carry terms, matching `Linear::clone_shared`.
- `src/lora/tied_head.rs`: the refusal lift described above.
- Tests: `src/loading/adapter_embedding_tests.rs`, `src/lora/embedding_tests.rs`, `src/lora/tied_head_tests.rs`, `src/lora/test_support.rs`, `src/loading/adapter_load_tests.rs`, `src/distributed/pipeline/stage_executor/llama_adapter_tests.rs`, and the `runtime_lora` tests.
- Docs: `docs/server-features.md`, `docs/llama-server-compat.md`.

Reachability by family: Llama and Qwen2 build the tied head as `UnifiedLinear::from_weights(weights, "model.embed_tokens", ..)` (`src/models/llama3.rs:1281`, and the stage loader), which now attaches the head term through `claim`. Qwen3 reads it through `embed_tokens.as_linear` (`src/models/qwen3.rs:720`).

## Behavior change

An `embed_tokens` LoRA pair is applied instead of refused on tied Llama, Qwen2 and Qwen3 checkpoints (conditions above), and on untied checkpoints the runtime path now claims it for the lookup. Other tied families, the VLM embedding stems, and an implicitly tied Llama or Qwen2 checkpoint keep the #2416 error. Base runs are unchanged.

## Verification

Commands (GB10, `--features cuda`, test-fast profile, every GPU run under `gpu-lock`):

- `cargo test ... --lib lora::`: 66 passed.
- `loading::adapter_`: 9 passed, 1 ignored (real-checkpoint gate). `llama::adapter_tests`: 4 passed. `partial_loading_adapter`: 16 passed. `-p mlxcel-core runtime_lora`: 8 passed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo fmt --all -- --check` and the contract scripts pass.

Revert check: with the fix functionally reverted (verdict lift off, embedding branch off in the fusion loop and in runtime staging), all three `adapter_embedding_tests`, the three new `tied_head` acceptance tests and `a_fully_claimed_runtime_adapter_still_loads` fail. With the fix restored they pass. The `embedding_tests` and core `runtime_lora` tests exercise the new functions directly and cannot be run against main.

Real checkpoints: adapters were trained with `mlx_lm.lora` (12 iters, rank 8, scale 20, keys `model.embed_tokens` and `self_attn.q_proj`, last 2 layers, seed 7). References came from `mlx_lm.fuse --dequantize`. Logits came from the ignored gate `real_checkpoint_embedding_adapter_parity` with `MLXCEL_SDPA_DETERMINISTIC=1`, last-position logits. The gate asserts equal argmax, top-5 equal up to ties within the tolerance, max |diff| <= 0.25, and runtime scale 0 byte-identical to the base model. Prompt "The capital of France is":

| Checkpoint | Paths | runtime vs fused | stage vs fused | scale 0 = base |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-bf16 (tied) | fused, runtime, stage | 0.125 | 0.0000 | exact |
| deepseek-coder-1.3b-instruct-mlx (untied) | fused, runtime, stage | 0.024 | 0.0000 | exact |
| qwen2.5-0.5b-instruct-bf16 (tied) | fused, runtime | 0.25 | n/a | exact |
| qwen3-0.6b-4bit (tied, quantized) | runtime | n/a | n/a | exact |

Against the `mlx_lm.fuse --dequantize` checkpoint loaded by mlxcel, runtime path: top-5 is equal on Llama-3.2-1B (max |diff| 0.125, 0.0625 on the reference top-5), deepseek-coder (0.047, 0.031) and qwen2.5-0.5b (0.50, 0.125). On qwen3-0.6b-4bit (0.75, 0.125) the first 4 are equal and ranks 4 to 6 of the reference are a three-way exact tie at 12.938. That run compares a quantized base against its dequantized fusion; with the second prompt its top-5 is equal in order. For scale, mlx-lm's own unfused LoRA against its own fused model differs by up to 0.56 (Llama) and 0.50 (Qwen).

Live server (`/v1/completions`, `logprobs: 5`, greedy) on Llama-3.2-1B-bf16 and qwen3-0.6b-4bit: the adapter changes the output at scale 1; after `POST /lora-adapters [{"id": 0, "scale": 0.0}]` text and every top-5 logprob equal a separate base server exactly; scaling back to 1.0 reproduces the first scale-1 response exactly. `mlxcel generate --adapter` and `--pp-size 2 --adapter` give the same text on the tied Llama and the untied deepseek-coder.

## Changes during review

- The security review found that the runtime loader's packed-table check (scales group count) admits up to four hidden widths, so a malformed adapter on a quantized table could load and then abort the process on the MLX shape mismatch. The width is now pinned at claim time. The same weak check on linear terms over quantized bases predates this PR and is unchanged.
- The fused path refuses a quantized table by its `.scales` plane.
- The Qwen3 `as_linear` head terms are built once at claim instead of per forward. After these changes the real-checkpoint runtime logits are byte-identical to the earlier run on all four checkpoints.
- The Pipeline Parallel CI smoke test (`pipeline_multi_stage_two_host_logical_smoke`, debug profile) overflowed the 2 MiB model-worker stack. In a debug build `try_load_config_backed_model_from_dir` gives every family arm its own stack slots (about 1.45 MB on main), and the 48 bytes the term set first added inline to `Embedding` and `QuantizedEmbedding` grew it by about 24 KB, past the limit. The term set is now `Option<Box<EmbeddingLoraTerms>>`, and the dispatcher loads each family through an out-of-line `load_dir_into`, so its frame is about 15 KB. The smallest passing `RUST_MIN_STACK` on GB10 went from about 2.31 MB to about 0.92 MB, and the real-checkpoint runtime logits are unchanged.

## Not addressed

- Fusing into a quantized embedding table (`--lora` serves it unfused).
- Tied families other than the Llama decoder and Qwen3, and the VLM embedding stems, keep the refusal.
- Embedding tables under other names (`transformer.wte`, `model.word_embeddings`) are still read as linear pairs.

## Statistics

17 files changed, 1733 insertions, 83 deletions. Commits: `63306d7d` feat(lora): support mlx-lm LoRAEmbedding adapters on embed_tokens, `bbeef290` fix(lora): justify the Arc around non-Send embedding terms, `848cd90b` fix(lora): pin an embedding term's width at claim and refuse packed fuse.
