# Technical Report: PR #1413 - EmbeddingGemma and Qwen3-Embedding forward passes

**Date**: 2026-08-26
**Author**: mlxcel contributors
**Status**: Completed; one layout covered by a synthetic checkpoint rather than a published one
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1413 implements the two decoder-backbone embedding families issue #1329 asks for, on top of the `/v1/embeddings` foundation that landed in #1408. EmbeddingGemma runs the existing Gemma 3 layers with bidirectional masks, mean pooling and two `Dense` projections; Qwen3-Embedding runs the existing Qwen3 layers causally and pools the last real token. Neither family adds a decoder: the whole change is masks, pooling, weight-key normalization, and one surgical split of `Qwen3Model::forward_impl`. Both load from their published checkpoints and reproduce their published reference numbers.

---

## 1. Problem Statement

### 1.1 Background

EmbeddingGemma (`model_type: gemma3_text`, `architectures: ["Gemma3TextModel"]`, `use_bidirectional_attention: true`) and Qwen3-Embedding (`model_type: qwen3`, `architectures: ["Qwen3ForCausalLM"]` plus a sentence-transformers `1_Pooling` module) are the two most downloaded small embedders, and both ship on decoder backbones mlxcel already implements. #1408 built the shared foundation: the `EmbeddingModel` trait, pooling, tokenization, micro-batching, the length limits, the mask builders in `mlxcel_core::utils`, detection into `ModelType::Gemma3Embedding` / `ModelType::Qwen3Embedding`, and a family dispatcher whose arms all returned "not yet supported". This PR fills in two of those arms.

### 1.2 Existing obstacles

- **Gemma3Model requires a head.** `Gemma3Model::from_weights` loads `lm_head` unconditionally and `forward_with_caches_and_embeddings` ends with `self.lm_head.forward(&h)`. An EmbeddingGemma checkpoint has no `lm_head` at all, so the generator constructor cannot be reused as is.
- **Gemma3Model always builds causal masks.** The generator's prefill path constructs `create_causal_mask` and `create_sliding_window_prefill_mask`. A bidirectional embedder needs the opposite of both, and each layer additionally carries `Attention.window_size`, which the Metal 4 attention path applies as a causal window even when an explicit mask is supplied.
- **Qwen3Model exposes only logits.** `forward_impl` ends with the head, so an embedder would have had to materialize a `[B, L, 151669]` tensor and discard it.
- **Neither checkpoint is keyed like its generator.** The sentence-transformers exports save the inner `...Model`, so `embed_tokens.weight`, `layers.{i}....` and `norm.weight` arrive without the `model.` prefix the constructors expect. EmbeddingGemma additionally stores its projections in two different places depending on the publisher.

### 1.3 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A causal mask reaches a bidirectional layer | High: embeddings are silently wrong, every gate except the numeric one still passes | Medium |
| The sliding period is off by one | High: the wrong layers get the wrong mask, output stays fluent-looking | Medium |
| `dense.0` and `dense.1` swapped | High: shapes still chain in one direction, output is wrong | Low |
| Tests interfere through concurrent MLX execution | Medium: gate numbers become unreliable evidence | High on this host |

---

## 2. Technical Decisions

### 2.1 Reuse the backbones; do not port two decoders

Both families are the existing decoders with different masks and no head. The implementation therefore builds `gemma3::TransformerBlock` and `Qwen3Model` directly and adds only what differs. The cost of this choice is that the two obstacles in 1.2 have to be handled at the seam rather than inside a fresh module; the benefit is that every future fix to Gemma 3 or Qwen3 attention, quantization, or RoPE reaches the embedders for free.

`src/models/gemma3.rs` needed no edit at all. `Attention.window_size` was already `pub`, so the embedding loader clears it after construction.

### 2.2 The embedding path owns every mask

`Gemma3EmbeddingModel::forward_hidden` builds both masks itself and passes `Some(mask)` to every layer:

- full-attention layers get `create_bidirectional_padding_mask(attention_mask)`, shape `[B, 1, 1, L]`, blocking only padding keys;
- sliding layers get `create_bidirectional_window_mask(attention_mask, sliding_window)`, shape `[B, 1, L, L]`, blocking a key when it is padding or when `|q - k| >= window`.

That second form is exactly the bidirectional sliding overlay `transformers` composes for Gemma 3 (`kv_idx > q_idx - w` and `kv_idx < q_idx + w`). For inputs up to `sliding_window` tokens the two masks are equivalent, so the window only starts to matter past 512 tokens, which is why the real-checkpoint gate deliberately embeds a 1749-token document.

Each layer's `self_attn.window_size` is set to `0` at load. This is not redundant with the mask: on the Metal 4 attention path `window_size` is applied by the kernel, so a layer given a bidirectional mask *and* a causal window would still be causal, and only on macOS. The rejected alternative, threading a per-call flag through `Attention::forward`, would have touched the generator's hot path for the benefit of a cold one.

Per-call caches are `gemma3::Cache::Standard(KVCache::new())` at offset 0, created and dropped inside the forward. Rotating caches are pointless here: nothing is reused between calls, and offset 0 keeps RoPE positions and the mask key axis aligned with the input length.

### 2.3 Derive the sliding period from `layer_types`, not from the scalar

`gemma3::ModelArgs` parses `sliding_window_pattern`. transformers 4.57 renamed that key to `_sliding_window_pattern` and made `layer_types` authoritative, so the parsed value on a modern EmbeddingGemma config is the family default of 6, not a value read from the checkpoint. On `mlx-community/embeddinggemma-300m-4bit` the default happens to be right, which is precisely the argument for pinning it: a period that is off by one still loads, still runs, and only shows up as a quietly wrong vector.

`resolve_sliding_window_pattern` reads `layer_types` when present, derives the period from the index of the first `full_attention` entry, then validates the entire list against that period rather than sampling it. An irregular list is a load error naming the offending layer. A config with no `full_attention` entry at all yields a period past the last layer, so no layer is treated as full attention. Falling back, it reads `sliding_window_pattern`, then `_sliding_window_pattern`, then the caller's default.

This resolution happens before the layers are constructed, because `gemma3::layer_rope_params` uses the same period to choose between `rope_theta` and `rope_local_base_freq`. Getting the period right after construction would have left the RoPE bases wrong.

### 2.4 Split `Qwen3Model::forward_impl` rather than exposing its internals

`forward_hidden` covers embedding lookup, the layer loop and the final norm; `forward_impl` calls it and then applies the head. The generator's output is unchanged by construction, and a test proves it by reassembling the tied head on top of `forward_hidden` and demanding a token-exact match on a synthetic two-layer model. The alternatives, making `layers` and `norm` public or duplicating the loop in the embedder, would both have let the two paths drift.

The same entry point is what a Qwen3 generative reranker (#1356) needs, so the doc comment names both consumers.

### 2.5 One shared weight-key sanitizer

A sentence-transformers export of a decoder backbone differs from the generator checkpoint in three mechanical ways, and every family that reuses a generator backbone has to undo all three. `src/models/embedding_sanitize.rs` does them in a fixed order: fold `{N}_Dense.linear.*` into `dense.{k}.*` by folder rank, drop `lm_head.*` and `head.*`, then prefix bare `embed_tokens.` / `layers.` / `norm.` with `model.`. The order matters: the `Dense` keys are not backbone roots and must be renamed before the prefix pass, or `2_Dense.linear.weight` would survive untouched while `layers.0...` moved.

The folder numbers are sentence-transformers module positions (`2_Dense` comes after `1_Pooling`), not projection indices, so they are ranked rather than used directly. That is what makes both published EmbeddingGemma layouts converge on the same `dense.0` / `dense.1` keys. The function is a no-op on an mlx conversion and idempotent, so a family may call it unconditionally.

### 2.6 Validate the `Dense` chain at load

`load_dense_stack` walks `dense.0`, `dense.1`, ... and requires each projection's input width to equal the previous stage's output, starting from the backbone hidden size. A swapped pair is a load error naming the projection. Without this, `768 -> 3072 -> 768` and its reverse both load, the forward pass runs, and only the numbers are wrong, which is the hardest class of failure to notice.

Reading the input width needs care on a quantized checkpoint: the weight is packed along the input axis (`dense.0.weight` is `[3072, 96]` at 4 bits), so `linear_features` reads the width from the `scales` grouping (`[3072, 12]` at group size 64, hence 768) and falls back to the weight's own second dimension when the tensor is dense.

The stack is a `Vec`, not a fixed pair, so a bidirectional Gemma 3 export with no `Dense` module at all loads and reports `embedding_dim == hidden_size`.

### 2.7 Force `tie_word_embeddings` in the Qwen3 embedding path

The embedder stops at the final norm, so no head is ever applied. Setting the flag before `Qwen3Model::from_weights` drops an untied `lm_head` from memory (on the 0.6B checkpoint that tensor is 151669 by 1024) and prevents the constructor from failing on a head this path would never read.

### 2.8 Cast back to the checkpoint dtype only when there is something to project

`pool` returns f32 by design, so the reductions run at full precision whatever the activation dtype. The `Dense` projections carry the checkpoint's dtype, so the pooled vector is cast back before those matmuls. With no projections the f32 vector goes straight to the engine instead of taking a needless round trip through f16.

### 2.9 Prompt prefixes stay caller-side, with one opt-in hook

Nothing is injected server-side; an input embeds exactly as it is sent. `Qwen3EmbeddingModel::format_text` wraps a query as `Instruct: {task}\nQuery: {query}` only when the request supplies an `instruction`, and is the identity otherwise, which matches what the `EmbeddingModel` trait's doc comment already promised for this family. `instruction` applies to every input of a request, so a mixed query-and-document batch has to split into two requests or format its queries in the text; `docs/embeddings.md` says so. EmbeddingGemma keeps the identity and documents its seven `config_sentence_transformers.json` prefixes in a table.

---

## 3. Implementation Details

### 3.1 EmbeddingGemma forward

```
h = embed_tokens[input_ids] * sqrt(hidden_size)
full    = create_bidirectional_padding_mask(attention_mask)      # [B, 1, 1, L]
sliding = create_bidirectional_window_mask(attention_mask, 512)  # [B, 1, L, L]
for i, layer in layers:                                          # window_size cleared
    h = layer.forward(h, fresh KVCache at offset 0,
                      if (i + 1) % pattern == 0 { full } else { sliding })
h      = norm(h)                                                 # GemmaRMSNorm
pooled = pool(h, attention_mask, Mean)                           # f32
pooled = dense.1(dense.0(astype(pooled, checkpoint dtype)))      # 768 -> 3072 -> 768
```

The engine L2-normalizes and applies `dimensions` truncation afterwards.

### 3.2 Qwen3-Embedding forward

```
mask   = create_causal_padding_mask(attention_mask, 0)           # [B, 1, L, L]
h      = model.forward_hidden(input_ids, None, fresh caches, Some(mask))
pooled = pool(h, attention_mask, LastToken)
```

Right padding is what makes this correct without a second code path: the pooled position is the last real token, padding sits after it, and padding keys are blocked for every real query, so the padded row reproduces the solo run exactly.

### 3.3 Weight-key mapping

| Published form | After sanitization |
|----------------|--------------------|
| `embed_tokens.weight` (sentence-transformers) | `model.embed_tokens.weight` |
| `model.embed_tokens.weight` (mlx) | unchanged |
| `2_Dense/model.safetensors` `linear.weight` | `dense.0.weight` |
| `3_Dense/model.safetensors` `linear.weight` | `dense.1.weight` |
| `dense.0.*`, `dense.1.*` (mlx) | unchanged |
| `lm_head.*`, `head.*` | dropped |

### 3.4 Registration

Only the `Gemma3Embedding` and `Qwen3Embedding` arms of `build_family_model` changed; every other family's "not yet supported" arm is untouched, which matters because sibling ports were landing in parallel. Detection, the `ModelType` variants and the `mlxcel arch` entries already existed from #1408, so `mlxcel arch` lists both under `Embedding` and `mlxcel list` shows both checkpoints with no further change.

---

## 4. Test Strategy

### 4.1 Two properties that no shape check can catch

The synthetic tests build a 16-wide Gemma 3 and a 16-wide Qwen3 from deterministic weights, so they need no checkpoint and no particular device.

- `bidirectional_prefill_is_not_causal`: over 96 tokens, flipping the last token must move the first token's hidden state. A causal mask makes that difference exactly zero.
- `causal_prefill_is_causal`: the mirror image for Qwen3. Flipping the last token must leave every earlier hidden state untouched within 1e-6, while the changed token itself moves.
- `global_layers_use_padding_mask_and_sliding_layers_use_window`: two one-layer models built from the *same* weights, one with the layer full-attention and one sliding with window 4. A token 6 positions away moves the first token's state in the first model and not in the second.
- `forward_hidden_then_head_matches_forward_impl`: the refactor guard, token-exact.
- `padding_invariance` and `last_token_pool_uses_appended_eos`: a right-padded two-row batch reproduces the unpadded single-row result.

### 4.2 Layout equivalence without the gated checkpoint

Only the mlx conversion of EmbeddingGemma is downloadable without accepting the gated `google/embeddinggemma-300m` terms, so `sentence_transformers_subfolder_layout_loads_from_disk_and_matches_the_mlx_layout` materializes a checkpoint instead: it writes the same synthetic weights in the sentence-transformers spelling (bare backbone roots, an unused `lm_head`, the projections in `2_Dense/` and `3_Dense/` module folders), loads it through the real `Gemma3EmbeddingModel::load(dir)`, and asserts the embeddings are bit-identical to the mlx-style in-memory build. This exercises `load_weights_from_dir_with_subfolders`, the sanitizer and the constructor together, which a key-mapping unit test alone does not.

### 4.3 Test-side MLX concurrency

`EmbeddingModel` is documented as single-thread and the product honors that through the embedding worker, so this hazard is test-side only. Two concurrent MLX forward passes in one process interfere in two observed ways on this tree: a CUDA graph capture aborts the process (`cudaStreamEndCapture ... operation failed due to a previous error during capture`, reproduced at roughly three runs in four on a wide filter), and results drift, with a sibling unit measuring two byte-identical rows of one batch at cosine 0.999912 instead of 1.0.

Every test here that builds a model or runs a forward pass takes a process-wide `mlx_test_guard()`, following the existing `llama4_helpers_tests::test_guard` pattern with poisoned-lock recovery so one panicking test fails alone. The guard cannot serialize against MLX work in *other* modules, which is why every gate this repository defines (`make verify-test`, `make verify-test-cuda`, `make test-fast`) already passes `--test-threads=1`; the Makefile documents that as load-bearing rather than tidiness. Every number below comes from that configuration, and the real-checkpoint gates print their values so a repeated run shows the spread instead of one pass/fail bit.

---

## 5. Real-Checkpoint Results

Linux, GB10, CUDA. Three consecutive single-threaded runs produced bit-identical numbers.

### `mlx-community/embeddinggemma-300m-4bit` (4-bit, 768 wide, `max_length` 2048)

| Gate | Observed | Requirement |
|------|----------|-------------|
| Vector width and norm | 768, norms 1.0 | 768-dim unit vectors |
| Query vs Mars | 0.642725 | highest by at least 0.1 |
| Query vs Venus | 0.323360 | below the match |
| Query vs Jupiter | 0.329467 | below the match |
| Match margin | 0.313 | at least 0.1 |
| `dimensions: 256` | width 256, norms 1.0, Mars 0.693574 > Venus 0.422158, Jupiter 0.412816 | unit vectors, ranking unchanged |
| Identical inputs in one batch | cosine 1.000000000 | 1.0 within 1e-6 |
| 1749-token document, solo vs padded batch | `max_abs_diff` 0.0 | within 1e-3 |
| Unrelated sentence | 0.0271 | below 0.5 |

### `Qwen/Qwen3-Embedding-0.6B` (bf16, 1024 wide, `max_length` 8192)

| Gate | Observed | Requirement |
|------|----------|-------------|
| Query-document matrix | `[[0.766633, 0.143439], [0.136450, 0.600714]]` | model card `[[0.7646, 0.1414], [0.1355, 0.6000]]` within 2e-2 |
| Largest deviation | 0.00204 | under 2e-2 |
| Identical inputs in one batch | cosine 1.000000119 | 1.0 within 1e-6 |
| Solo vs padded batch | `max_abs_diff` 0.0 | within 1e-3 |
| Non-finite components | none | none |
| Unrelated sentence | 0.1568 | below 0.5 |
| `dimensions: 256`, `encoding_format: base64` | decodes to a 256-wide unit vector | valid |

No PyTorch or `transformers` install exists on the validation host, so the Qwen3-Embedding reference is the checkpoint's published model card, as the issue prescribes. Nothing in this report is computed from a local reference implementation, and no number is estimated.

Both families return the same values through `mlxcel embed` and through `mlxcel-server` plus `POST /v1/embeddings`, and `/v1/models` lists the served embedding model in each case.

---

## 6. Validation Summary

| Command | Result |
|---------|--------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test ... --lib -- --test-threads=1 models::gemma3_embedding models::qwen3_embedding models::embedding_sanitize` | 22 passed, 0 failed, three times, identical gate numbers |
| `cargo test ... --lib embeddings:: -- --test-threads=1` | 62 passed, 0 failed |
| `cargo test ... --lib models::gemma3 -- --test-threads=1` | 23 passed, 5 ignored, 0 failed |
| `cargo test ... --lib models::qwen3 -- --test-threads=1` | 59 passed, 11 ignored, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |

The issue's acceptance criteria name the macOS `metal,accelerate` feature set. The CUDA gate is the equivalent on this host and is what ran; the macOS gate is CI's.

No performance numbers are reported. The epic runs its performance pass separately, on a quiet machine.

---

## 7. Change Summary

| File | Change |
|------|--------|
| `src/models/gemma3_embedding.rs` (new, 327) | `Gemma3EmbeddingModel`: bidirectional masks, mean pooling, `Dense` stack, period resolution |
| `src/models/gemma3_embedding_tests.rs` (new, 635) | Synthetic mask and pooling gates, the on-disk subfolder layout test, the real-checkpoint gates |
| `src/models/qwen3_embedding.rs` (new, 155) | `Qwen3EmbeddingModel`: causal-plus-padding mask, last-token pooling, instruction formatting |
| `src/models/qwen3_embedding_tests.rs` (new, 381) | Refactor guard, causality gate, padding gate, the real-checkpoint gate |
| `src/models/embedding_sanitize.rs` (new, 160) | Shared weight-key normalization plus `linear_features` |
| `src/models/embedding_sanitize_tests.rs` (new, 140) | Both layouts, idempotence, the unknown-submodule case |
| `src/models/embedding_test_support.rs` (new, 229) | Deterministic weights, shaped safetensors writer, checkpoint lookup, MLX guard |
| `src/models/qwen3.rs` (+24/-4) | `forward_impl` split into `forward_hidden` plus the head |
| `src/embeddings/loader.rs` (+6/-2) | Two family arms constructed |
| `src/embeddings/real_checkpoint_tests.rs` (+5/-3) | Qwen3-Embedding leaves the unported list |
| `src/models/mod.rs` (+5) | Three module declarations |
| `docs/embeddings.md` (+64/-21) | Family notes, prompt prefixes, source map; the deletions are the SigLIP and ModernBERT family sections folded into one `Family notes` heading during the rebase, not removed content |
| `docs/supported-models.md` (+2) | Two Embedding rows |

13 files, 2133 insertions, 30 deletions, on top of the SigLIP (#1410) and ModernBERT (#1412) ports that merged while this one was in flight.

---

## 8. What Remains Unverified

- **The published sentence-transformers EmbeddingGemma checkpoint.** `google/embeddinggemma-300m` is gated. The subfolder layout is proven against a synthetic checkpoint written in that spelling, not against the published artifact. A dense (non-quantized) EmbeddingGemma has therefore never been loaded end to end here either, though the same code path serves the dense Qwen3-Embedding checkpoint.
- **macOS and Metal.** Everything here ran on Linux with CUDA. The `window_size = 0` clearing exists specifically for the Metal 4 attention path and is exercised only indirectly on this host; the CI macOS gate is what will confirm it.
- **Larger Qwen3-Embedding variants.** 4B and 8B are the same code at a different size and are out of scope for the issue; neither was loaded.
- **Matryoshka widths other than 256.** 512 and 128 are trained widths and go through the same `truncate_dimensions` path, but only 256 was measured against the ranking requirement.

---

## 9. Follow-up Actions

- #1356 (Qwen3 generative reranker) can consume `Qwen3Model::forward_hidden` as is.
- If `google/embeddinggemma-300m` becomes available on a validation host, add it to `local_embedding_checkpoints_detect_to_their_families` and to the EmbeddingGemma gate so the published subfolder layout is covered directly.
- The epic's performance pass should include a long-input EmbeddingGemma case, since the sliding window only engages past 512 tokens and the two masks have different shapes (`[B, 1, 1, L]` against `[B, 1, L, L]`).

---

## References

- Issue #1329, epic #1348
- PR #1408 (embedding foundation), issue #1353
- `docs/embeddings.md`, `docs/supported-models.md`
- `mlx-community/embeddinggemma-300m-4bit`, `Qwen/Qwen3-Embedding-0.6B`
