# Technical Report: PR #1415 - Bidirectional Llama, Nemotron-3-Embed and LFM2.5-Embedding

**Date**: 2026-08-26
**Author**: mlxcel maintainers
**Status**: Completed. One correctness bug found and fixed during validation
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1415 implements issue #1325, three more family forward passes on the embedding foundation from #1353. All three are decoder backbones mlxcel already runs for generation, re-used bidirectionally with a padding-only mask and a pooling step: bidirectional Llama (the LLM2Vec recipe, `model_type: llama_bidirec`), Nemotron-3-Embed on the Ministral 3 backbone (`is_causal: false`), and LFM2.5-Embedding on the LFM2 short-conv plus attention hybrid.

Two of the three needed no backbone change at all. The third needed two, both confined to the short convolution: a `conv_causal` flag that splits the mixer's padding across both sides instead of prepending all of it, and a padding multiplier that zeroes the mixer's input at padding positions.

The second of those is the substantive finding. A convolution has no key axis for an attention mask to act on, so the attention mask that covers the rest of the model does not reach it. Without explicit zeroing, the pad-token embeddings mixed into the real positions next to the boundary and the six attention layers above spread that across the whole row. Measured on `LiquidAI/LFM2.5-Embedding-350M`, changing only the token ids underneath the mask moved the pooled vector by cosine 0.94. The fix mirrors the reference's `apply_mask_to_padding_states`, and every family in this PR now gates the property at zero tolerance.

Four real checkpoints load and serve through both `mlxcel embed` and `POST /v1/embeddings`, with retrieval margins of 0.42 to 0.50 against the issue's 0.15 bar and every unrelated pair below 0.06. The `mlx-community` 8-bit conversion of Nemotron-3-Embed agrees with the bf16 original to cosine 0.9998.

---

## 1. Problem Statement

### 1.1 Background

Epic #1348 ports embedding families one at a time onto the foundation in `src/embeddings/`. Before this PR, `ModelType::LlamaBidirec`, `ModelType::Ministral3Embedding` and `ModelType::Lfm2Embedding` were all detected correctly and all answered `<family> is detected as an embedding checkpoint, but this embedding family is not yet supported by /v1/embeddings`.

The three share a shape: a causal decoder whose only difference from the generator is the mask and the missing head. That makes them cheap in principle, which is why they were grouped. It also makes them easy to get quietly wrong, because a reused causal backbone loads, runs, produces finite unit vectors and only differs in quality.

### 1.2 Existing obstacles

| Obstacle | Where |
|---|---|
| `Llama3Model.lm_head` is not `Option`, so the constructor cannot build a headless model | `src/models/llama3.rs` |
| `get_llama4_attn_scale` and `Cache::as_interface` were private to the Ministral 3 module | `src/models/ministral3.rs` |
| `ShortConv::forward` always left-padded by `L_cache - 1` "so the depthwise conv stays causal" | `src/models/lfm2.rs` |
| `Lfm2Model` fields are private and it had no hidden-state output | `src/models/lfm2.rs` |
| All three checkpoints store backbone roots without the `model.` prefix every loader requires | the published exports |

### 1.3 Risk assessment

| Risk | Impact | Mitigation |
|---|---|---|
| A reused causal backbone stays causal | Silently degraded vectors, no error | Per-family gate: changing the last token must move position 0 |
| The Llama 4 attention scale is computed at the wrong offset | Scores drift from the generator's, growing with position | Gate against an independent restatement of the documented formula |
| The short conv reads padding | Contaminated embeddings for every batched request | Zero-tolerance gate holding batch shape fixed and changing only the masked content |
| A `1_Dense` late-interaction checkpoint loads as a single-vector embedder | A ColBERT model silently returns one pooled vector | Load-time rejection |

---

## 2. Technical Decisions

### 2.1 Build the Llama layers from the weight map instead of touching `llama3.rs`

The issue's plan proposed adding a `forward_hidden` split and a headless constructor to `src/models/llama3.rs`. The merged EmbeddingGemma family had already established the alternative: hold `UnifiedEmbedding`, `Vec<TransformerBlock>` and `RMSNorm` directly in the family struct and run the loop there.

That path was taken here because `llama3::TransformerBlock::from_weights_with_rope`, `Attention` and `MLP` are all already `pub`, so the family needs nothing new from the backbone. It also avoids materializing the tied head: `Llama3Model::from_weights` builds `lm_head` from `model.embed_tokens` when `tie_word_embeddings` is set, which on this checkpoint is a `128256 x 2048` `UnifiedLinear` view the embedder would never apply. The cost is one duplicated four-line layer loop; the benefit is a zero-line diff in a 1523-line file that three sibling units were editing concurrently.

### 2.2 Ministral 3 reuses its model type wholesale

`Ministral3Model` already has `lm_head: Option<UnifiedLinear>` and `pub` fields, so the embedder wraps the real model rather than re-assembling it. Only two visibility widenings were needed, both `pub(crate)` and neither a behaviour change: `get_llama4_attn_scale`, so the embedder computes the same per-position schedule the generator does, and `Cache::as_interface`, so the embedder can drive `TransformerBlock::forward` with the model's own mixed cache vector.

Driving `make_caches()` rather than a hand-rolled `Vec<KVCache>` means a future checkpoint that declares `sliding_attention` layers gets `RotatingKVCache` for those automatically, and the forward pass already picks `create_bidirectional_window_mask` for them. Neither published checkpoint exercises that path (`sliding_window` is `null` and there is no `layer_types`), so it is written but unverified.

### 2.3 The attention scale is computed at offset 0, and the reason is worth stating

`Ministral3Model::forward_with_caches` anchors the Llama 4 schedule on its full-attention cache offset. For a fresh prefill that offset is 0, which is exactly what the embedder passes, so the two paths index position `i` as token `i`. The schedule is `1 + beta * ln(1 + floor(pos / original_max_position_embeddings))` with `beta` 0.1 and a 16384 window, so on inputs capped at 8192 every scale is exactly 1.0 and the scaling is inert. It is still computed, because a checkpoint with a smaller window or a raised cap would need it, and because a silently-dropped multiplier is the kind of thing that only shows up as a quality regression.

### 2.4 `conv_causal` defaults to true, so generation is byte-identical

The one backbone behaviour change lives behind a serde-defaulted flag. No published `config.json` declares `conv_L_cache` alongside a directionality key, so LFM2 and LFM2-MoE generation keep the causal branch purely by the default, and that branch is the pre-change code path unchanged: all `L_cache - 1` zeros on the left, conv state written from the padded tail.

The non-causal branch splits the same total padding, `left = L_cache / 2` and `right = L_cache - 1 - left`, so the output length is `L` for any `L_cache`, and for the published odd `L_cache = 3` the window is exactly symmetric. It also stops writing a conv state, because a "state" holding the tail of a bidirectional window is not something a decode step could resume from.

### 2.5 The short conv needs its own padding mechanism

This is the finding of the PR. The embedding foundation gives families `create_bidirectional_padding_mask`, which blocks padding on the attention key axis. A convolution has no key axis. Zeroing the residual stream at padding positions before each layer would not work either, because the reference keeps the residual and re-masks at every mixer: what has to be zeroed is the conv input, at every conv layer, regardless of what the residual carries.

So `ShortConv::forward` takes an optional `[B, L, 1]` multiplier in the activation dtype and multiplies its input by it. Generation passes `None` and is unchanged. The embedder builds one multiplier per call and hands it to every layer; the attention layers ignore it because their mask already covers padding.

Before the fix, a `B = 1` run at fixed width and fixed mask that changed only the token ids underneath the mask moved the pooled vector by cosine 0.94, and the batched-versus-solo agreement sat at cosine 0.9968. After it, the first is exactly 0 and the second is 0.99992.

### 2.6 The LFM2 final-norm root is prefixed locally, not in the shared sanitizer

`sanitize_decoder_embedding_weights` prefixes `embed_tokens.`, `layers.` and `norm.` with `model.`. LFM2 spells its final norm `embedding_norm.weight`, which matches none of those. Adding a fourth root to the shared constant would have been semantically fine, but the shared module was being edited by concurrent sibling units, so the family carries a five-line `prefix_embedding_norm` of its own with the same idempotence contract. This is a deliberate trade of a small duplication for a smaller merge surface; folding it into the shared list is reasonable follow-up once the epic's waves have landed.

### 2.7 Unapplied `Dense` modules are a load error, not a silent drop

None of the three published checkpoints ships a post-pooling projection. All three families still refuse to load one: `sanitize_decoder_embedding_weights` returns the number of `Dense` folders it folded, and a non-zero count fails the load naming the count. A `Dense` module that loads and is never applied produces vectors that are wrong in a way nothing downstream can detect.

For LFM2 the same check does double duty, because a `1_Dense` folder is exactly the ColBERT late-interaction layout, and the rejection message says so. That layout is out of scope for this issue and is a candidate follow-up once #1337 lands.

### 2.8 Prompt prefixes stay caller-side

All three checkpoints declare their prefixes in `config_sentence_transformers.json` (`query: ` with `passage: ` for the two NVIDIA models, `query: ` with `document: ` for LFM2). None of the three overrides `format_text`, so an input embeds exactly as it is sent. This matches every merged family except Qwen3-Embedding, which has an explicit `instruction` hook because its format takes a caller-supplied task string.

---

## 3. Implementation Details

### 3.1 Bidirectional Llama forward

```
h    = embed_tokens[input_ids]
mask = create_bidirectional_padding_mask(attention_mask)      # [B, 1, 1, L]
for layer in layers:                                          # fresh KVCache::new() each
    h = layer.forward(h, cache, Some(mask))
h    = norm(h)
out  = mean_pool(h, attention_mask)                           # engine L2-normalizes
```

Weight sanitize, in order: drop keys ending in `rotary_emb.inv_freq` or `position_ids`; strip a `language_model.` wrapper prefix; then the shared pass folds `Dense` folders, drops `lm_head.*` / `head.*`, and prefixes `embed_tokens.` / `layers.` / `norm.` with `model.`. The order matters: stripping the wrapper has to precede the prefix decision or `language_model.layers.0.…` would never be recognized as a bare backbone root.

### 3.2 Nemotron-3-Embed forward

```
h          = model.embed_tokens[input_ids]
attn_scale = get_llama4_attn_scale(L, 0, beta, original_max_position_embeddings)
full_mask  = create_bidirectional_padding_mask(attention_mask)
window     = create_bidirectional_window_mask(attention_mask, sliding_window)  # only if any layer slides
for i, layer in model.layers:
    h = layer.forward(h, attn_scale, caches[i].as_interface(),
                      Some(window if layer.use_sliding else full_mask))
h          = model.norm(h)
out        = mean_pool(h, attention_mask)
```

`tie_word_embeddings` is forced true before `Ministral3Model::from_weights` runs, so it leaves `lm_head` as `None` rather than looking for a head the sanitize pass has just dropped.

### 3.3 LFM2.5-Embedding forward

```
h        = embed_tokens[input_ids]
mask     = create_bidirectional_padding_mask(attention_mask)
pad_mult = padding_multiplier(attention_mask, dtype_of(h))    # [B, L, 1], 1 real / 0 pad
for layer in layers:                                          # fresh caches, not sequence_state
    h = layer.forward(h, cache,
                      Some(mask) if layer.is_attention() else None,
                      Some(pad_mult))
h        = embedding_norm(h)
out      = cls_pool(h, attention_mask)
```

`Lfm2Model::from_weights` takes owned weights and runs its own sanitize (the `w1`/`w2`/`w3` feed-forward rename and the `[hidden, 1, L_cache]` to `[hidden, L_cache, 1]` conv transpose), so the family sanitizes for the `model.` prefix first and hands the map over.

CLS pooling reads the `<|startoftext|>` the tokenizer's post-processor prepends. With right padding that is index 0 in every row, though `pool` finds it by first-real-token argmax rather than assuming it, so a left-padded batch would pool correctly too.

### 3.4 Registration

`src/embeddings/loader.rs` gains three arms; every other family's `not yet supported` arm is untouched. Detection, `src/model_metadata.rs` and the `mlxcel arch` / `mlxcel list` output were already correct from the foundation, so this PR needed no change there. `src/embeddings/real_checkpoint_tests.rs` drops LFM2.5-Embedding from the unported-families list, the way EmbeddingGemma, Qwen3-Embedding and BERT did before it.

---

## 4. Test Strategy

### 4.1 Two properties no shape check can catch

**Bidirectionality.** A reused causal backbone has the right shapes, the right dtypes and finite unit vectors. The gate is that changing the last of 96 tokens must move position 0. For the two attention-only families the bar is a magnitude; for LFM2 the bar is "moved at all", because an all-conv control proves the discriminator: with the attention layer replaced by a conv, token 95 is far outside the stack's `L_cache / 2` per-layer reach and position 0 stays bit-identical, so anything above zero is the mask working.

**Padding invisibility.** The gate holds the batch shape, the slot and the mask all fixed and changes only the token ids underneath the mask, then demands bit-identical output. Nothing about the kernel geometry moves, so the tolerance is exactly 0 and any difference is a real leak. This is the test that caught the LFM2 short-conv bug.

### 4.2 The directional convolution is tested at the mixer, not through the model

`ShortConv` became `pub(crate)` so `lfm2_tests` can drive it with a one-hot impulse. The fixture is two channels: channel 0 carries the impulse, channel 1 is a constant 1, and the `in_proj` weight is chosen so the gate `C` and the value `x` are both 1 everywhere while `B` is the impulse. The output is then the raw convolution of the impulse, with three distinct taps so the landing position of each is unambiguous. Causal spreads an impulse at `t` to `t, t+1, t+2` and writes a conv state; bidirectional spreads it to `t-1, t, t+1` and writes none. Through sixteen layers of a whole model that one-index shift would be invisible.

### 4.3 The batch-geometry floor, and why the issue's numeric form of one gate was not used

The issue asks for the padded batch to match the single-input vectors "within 1e-3". The pooled vectors do agree to that in cosine, and the reports carry the figure. They do not agree to 1e-3 in the largest single component, and cannot on this hardware.

Embedding one text as 2, 3, 4, 5 and 8 **unpadded** copies of a single batch moves the largest component of `nvidia/Nemotron-3-Embed-1B-BF16` by up to 3.7e-3 while cosine stays above 0.99997. The already-merged `Qwen/Qwen3-Embedding-0.6B` behaves identically, and the four core primitives involved (`conv1d`, `matmul`, `attention_from_ptr` with a `[B,1,1,L]` mask, and GQA attention at the real head shapes) are each batch-slot deterministic in isolation. The effect is the MLX CUDA backend choosing its accumulation shape from the batch geometry in bf16, not anything the port controls, so a component-wise bound would gate the backend rather than the code.

The cosine form is gated at 1e-3, the component drift is printed for every run, and the property the port actually owes is gated at zero tolerance by the padding-content test in 4.1.

### 4.4 MLX test serialization

`EmbeddingModel` is documented as single-thread and the product honours that through the embedding worker, but `cargo test` runs one thread per logical CPU. Every MLX-evaluating test in the three new modules takes the shared `embedding_test_support::mlx_test_guard`, and the two pre-existing MLX-evaluating tests in `lfm2_tests.rs` were retrofitted with it. All gate numbers in this report were recorded under `--test-threads=1`.

---

## 5. Real-Checkpoint Results

Three consecutive `--test-threads=1` runs produced byte-identical output. The spread is zero, so single values are quoted.

| Checkpoint | dim | max_length | related | unrelated | margin | duplicate rows | pad-content leak | batch cosine |
|---|---|---|---|---|---|---|---|---|
| `nvidia/llama-nemotron-embed-1b-v2` | 2048 | 8192 | 0.432512 | 0.012978 | 0.419534 | 1.000000119 | 0 | 0.999944985 |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 | 8192 | 0.552243 | 0.057058 | 0.495185 | 1.000000119 | 0 | 0.999971032 |
| `mlx-community/Nemotron-3-Embed-1B-BF16-8bit` | 2048 | 8192 | 0.551046 | 0.056344 | 0.494702 | 1.000000119 | 0 | 1.000000000 |
| `LiquidAI/LFM2.5-Embedding-350M` | 1024 | 512 | 0.400611 | -0.026907 | 0.427518 | 1.000000119 | 0 | 0.999917984 |

"related" is the solar query against the photovoltaic passage, "unrelated" the same query against the recipe passage. The issue's bar is a 0.15 margin and every unrelated pair below 0.5.

**Quantization.** The 8-bit conversion agrees with the bf16 original per input at cosine 0.999839, 0.999839, 0.999857 and 0.999839, against the issue's 0.99 bar.

**Bidirectional prefill on the real weights.** A prompt of at least 64 tokens scores below 0.999 against its own one-third prefix: 0.977499 for bidirectional Llama, 0.961427 for LFM2. A causal prefill with the same pooling would leave the shared prefix positions untouched.

**Endpoint parity.** `mlxcel embed --json` and `mlxcel-server` `POST /v1/embeddings` were run on all four checkpoints and reproduce the table: unit-norm vectors of the right width, `model_type` reported as `LlamaBidirec`, `Ministral3Embedding` and `Lfm2Embedding`, and duplicate rows of one HTTP request at cosine 1.000000000.

---

## 6. Validation Summary

| Gate | Command | Result |
|---|---|---|
| Family unit and real-checkpoint tests | `cargo test --profile test-fast --features cuda --lib -- models::lfm2_embedding models::llama_bidirec models::ministral3_embedding models::lfm2_tests --test-threads=1` | 36 passed, 0 failed, three consecutive runs |
| Embedding subsystem | `cargo test --profile test-fast --features cuda --lib embeddings:: -- --test-threads=1` | 63 passed, 0 failed |
| Lint | `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | clean |
| Type check | `cargo check --profile test-fast --features cuda --all-targets` | clean |
| Format | `cargo fmt --all -- --check` | clean |
| CLI | `mlxcel embed --json` on four checkpoints | unit vectors, margins as tabled |
| HTTP | `mlxcel-server` plus `POST /v1/embeddings` on four checkpoints | unit vectors, margins as tabled |

The validation host is Linux with CUDA (GB10). The issue's `--features metal,accelerate` toolchain gates ran as their `--features cuda` equivalents; `metal` and `accelerate` are macOS-only and cannot be built there.

---

## 7. Change Summary

| File | Lines | Role |
|---|---|---|
| `src/models/llama_bidirec.rs` | +268 | `LlamaBidirecModel`, weight sanitize, adapter-only rejection |
| `src/models/llama_bidirec_tests.rs` | +522 | sanitize, adapter rejection, bidirectionality, padding, real checkpoint |
| `src/models/ministral3_embedding.rs` | +195 | `Ministral3EmbeddingModel`, attention scale at offset 0, sliding overlay |
| `src/models/ministral3_embedding_tests.rs` | +515 | attention-scale formula, bidirectionality, padding, both conversions |
| `src/models/lfm2_embedding.rs` | +222 | `Lfm2EmbeddingModel`, `embedding_norm` prefix, late-interaction rejection |
| `src/models/lfm2_embedding_tests.rs` | +589 | CLS pooling, all-conv control, padding, real checkpoint |
| `src/models/lfm2.rs` | +182 / -20 | `conv_causal`, directional padding, padding multiplier, `forward_hidden_bidirectional` |
| `src/models/lfm2_tests.rs` | +170 / -1 | impulse-response tests for both conv directions, MLX guard retrofit |
| `src/models/ministral3.rs` | +8 / -2 | two `pub(crate)` widenings, no behaviour change |
| `src/embeddings/loader.rs` | +12 / -3 | three family arms |
| `src/embeddings/real_checkpoint_tests.rs` | +12 / -5 | LFM2 leaves the unported list, which now holds only the two multimodal embedders |
| `src/models/mod.rs` | +3 | three module declarations |
| `docs/embeddings.md` | +58 | three family sections, source map, bf16 batch-geometry note |
| `docs/supported-models.md` | +3 | three Embedding rows |

---

## 8. What Remains Unverified

- **A sliding-window Ministral 3 embedder.** The `create_bidirectional_window_mask` branch and the `RotatingKVCache` selection are implemented and reachable, but neither published checkpoint declares `sliding_attention` layers, so the path has no real-checkpoint coverage.
- **A reference-implementation numeric diff.** No PyTorch or `transformers` install is available on the validation host, so no vector is compared against the reference framework. The gates are self-consistency, retrieval semantics, the published quantization tolerance and the model cards' declared prefixes.
- **Nemotron-3-Embed-8B.** Same code path, not downloaded, not run.
- **A merged LLM2Vec adapter.** The adapter-only rejection is gated on a synthetic directory; no real PEFT LLM2Vec checkpoint was exercised, merged or otherwise.
- **The non-causal path at even `L_cache`.** `conv_padding` handles it (the split is `L_cache / 2` and the remainder), but every published checkpoint uses `L_cache = 3`, so only the odd, exactly-symmetric case is covered by a test.

---

## 9. Follow-up Candidates

1. Fold `embedding_norm.` into the shared `BACKBONE_ROOTS` in `src/models/embedding_sanitize.rs` once the epic's parallel waves have merged, removing the local duplicate.
2. LFM2 / LFM2.5 ColBERT late-interaction support, which the load-time rejection currently reports as unsupported; a candidate once #1337 lands the multi-vector path.
3. Consider whether the generation path should also mask the short conv when a padded prefill becomes reachable there. It is `None` today because LFM2 generation runs one unpadded sequence, but that is an invariant worth asserting rather than assuming.

---

## References

- Issue #1325, epic #1348, foundation PR #1408 / issue #1353
- Merged sibling families: #1410 (SigLIP text), #1411 (BERT, XLM-RoBERTa), #1412 (ModernBERT), #1413 (EmbeddingGemma, Qwen3-Embedding), #1414 (ColIdefics3, ColQwen2.5)
- `docs/embeddings.md`, `docs/supported-models.md`
