# Technical Report: PR #1416 - Qwen3-VL-Embedding and Llama-Nemotron-VL-Embed

**Date**: 2026-08-26
**Author**: mlxcel maintainers
**Status**: Completed; no reference implementation available on the validation host, so no numeric parity is claimed
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1416 implements the two multimodal embedders issue #1345 asks for. They are the first families on `/v1/embeddings` that accept `image_url` items, so this is also the first end-to-end exercise of the image path the foundation (#1408 / #1353) built but never had a consumer for.

Neither family adds a decoder or a vision tower. Qwen3-VL-Embedding is the generative Qwen3-VL stack loaded by the generative loader, with the head replaced by last-token pooling. Llama-Nemotron-VL-Embed composes three parts mlxcel already runs: the SigLIP-400M tower, InternVL's pixel-shuffle `mlp1` connector, and a Llama 3.2 1B decoder driven bidirectionally and mean pooled. The only new algorithmic code is the checkpoint's own dynamic tiling, which differs from InternVL's in a way that made sharing the existing processor the wrong call.

The one structural change outside the two new families is a split of `Qwen3VLModel::forward_for_sequence` into a head-free `forward_hidden` plus `lm_head`, guarded by a token-exact test so generation is provably unchanged.

---

## 1. Problem Statement

### 1.1 Background

`Qwen/Qwen3-VL-Embedding-2B` ships as an ordinary `Qwen3VLForConditionalGeneration` export (`model_type: qwen3_vl`) with a sentence-transformers `1_Pooling` module declaring `pooling_mode: lasttoken`. Every weight is the generative one: a 24-layer ViT at `model.visual`, DeepStack mergers at indexes 5, 11 and 17, and a 28-layer Qwen3 text decoder with interleaved M-RoPE (`mrope_section: [24, 20, 20]`, `rope_theta: 5e6`, tied embeddings).

`nvidia/llama-nemotron-embed-vl-1b-v2` declares `model_type: llama_nemotron_vl` with `architectures: ["LlamaNemotronVLModel"]`, and its `llm_config` declares `LlamaBidirectionalModel` (`model_type: llama_bidirec`), which upstream defines as `LlamaModel` with `is_causal = False` set on every attention module. Its vision side is a 27-layer SigLIP tower at `vision_model.vision_model` (hidden 1152, `image_size` 512, `patch_size` 16), and the connector is InternVL's `mlp1`.

The foundation had already registered both variants for detection, listed them under `mlxcel arch`, and wired the request side: `image_url` items route to `EmbeddingEngine::embed_image` whenever `EmbeddingModel::supports_images()` is true, and `mlxcel embed --image` does the same offline. Both dispatcher arms returned "not yet supported". This PR fills them in.

### 1.2 Existing obstacles

- **`Qwen3VLModel` always applies the head.** `forward_for_sequence` ends with `self.lm_head.forward(&h)`, and so does its text-only fast path. An embedder would otherwise have to materialize a `[B, L, 151936]` tensor and discard it.
- **The engine tokenizes before the family sees the image.** `EmbeddingEngine::embed_image` calls `format_text` first, then `encode_row`, and only then hands the batch and the image to `EmbeddingModel::embed`. Neither family can emit the right number of visual placeholders at formatting time, because that count depends on the image's aspect ratio.
- **The `EmbeddingModel` trait carries no "this row has an image" flag.** `format_text(&self, text, instruction)` is all the family gets.
- **DeepStack injection is single-row.** `Qwen3VLModel::deepstack_process` handles `batch == 1` and copies `h` unchanged otherwise, and the M-RoPE delta is per sequence.
- **Nemotron's checkpoint is not keyed like any generator.** Its text weights live under `language_model.`, its SigLIP tower under `vision_model.vision_model.`, and the tower ships an attention-pooling `head.*` no embedder reads.
- **The existing InternVL processor implements a different tiling rule.** `InternVLProcessor::find_closest_aspect_ratio` minimizes the aspect difference and breaks ties on area. This checkpoint's `processing_llama_nemotron_vl.py` maximizes `min(area_ratio, 0.6) * min(target / actual, actual / target)`, a different objective, and normalizes with SigLIP constants rather than ImageNet ones.
- **Three sibling family ports were in flight in the same wave**, including one (#1325) that introduces a bidirectional Llama backbone this family could otherwise have depended on.

### 1.3 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| The `forward_hidden` split changes generation | High: every Qwen3-VL generative path regresses silently | Low, once the token-exact gate exists |
| Placeholder count mismatches the vision feature count | High: `merge_llava` scatters the wrong number of vectors and the embedding is quietly wrong | Medium |
| The tiling rule diverges from the reference | Medium: a different tile count means a different vector, correct-looking but not comparable to reference vectors | Medium |
| A causal mask reaches the Nemotron stack | High: retrieval quality drops with no visible symptom | Medium |
| Reusing `InternVLProcessor` changes InternVL VLM behaviour | High: an unrelated family regresses | Medium, if shared |
| Sibling ports conflict on shared files | Medium: rebase churn, not correctness | High |

---

## 2. Technical Decisions

### 2.1 Load Qwen3-VL-Embedding through the generative loader

`Qwen3VLEmbeddingModel::load` calls `crate::loading::load_qwen3_vl(model_dir)` and matches `LoadedModel::Qwen3VL`. This is deliberate rather than convenient. The alternative, reconstructing the stack in the embedding module, would have had to duplicate `read_sanitized_vlm_config`, `parse_required_vlm_subconfig`, the two quantization-inheritance helpers, `remap_qwen3_vl_weights` (which rewrites `model.language_model.` to `model.` and `model.visual.` to `vision_tower.`), `sanitize_tied_embeddings` and `qwen_vl_token_ids`. All of those are `pub(crate)` inside `loading::vlm` and none is family-specific, so duplicating them would have created a second copy of the Qwen3-VL layout contract that could drift from the first.

The cost is one line in `src/loading/mod.rs` re-exporting `load_qwen3_vl` crate-internally. The benefit is that any future fix to Qwen3-VL weight remapping, quantization inheritance or token-id resolution reaches the embedder for free, and that the embedder and the generator provably see the same tensors.

### 2.2 Split the head off `forward_for_sequence`, do not expose the internals

`forward_hidden_for_sequence` is the whole body up to and including the final norm, with the M-RoPE position resolution, the DeepStack injection and the state-clearing logic untouched. `forward_for_sequence` is now that plus `lm_head`. `forward_text_only` became `forward_text_only_hidden` for the same reason, and the caller applies the head once at the top level rather than in each branch.

`src/models/qwen3_vl_tests.rs` builds a synthetic two-layer Qwen3-VL from deterministic weights, runs `forward_impl`, then runs `forward_hidden` and applies `lm_head` by hand, and asserts `max_abs_diff == 0.0`. Not "close": exactly zero. That is what makes the split provably a refactor rather than an equivalence claim.

The rejected alternatives were making `layers` and `norm` public (which lets the two paths drift) and duplicating the loop in the embedder (same problem, plus the DeepStack and M-RoPE bookkeeping would have to be duplicated too). The doc comment names the third consumer, the Qwen3-VL reranker in #1356, so the next port does not re-derive this.

### 2.3 Expand the visual placeholder through `EmbeddingModel::expand_image_tokens`

Both families emit exactly one placeholder from `format_text` and grow it to the real count in `expand_image_tokens`, the hook the engine calls on the encoded row before padding. `expand_image_placeholders` walks the id row, replaces each placeholder with `counts[i]` copies, and expands the attention mask in lockstep so a padded row keeps its flags.

That hook landed in the foundation with the late-interaction port (#1414) while this one was in flight. An earlier revision of this port expanded inside `embed` instead, which worked but left `usage.prompt_tokens` counting the wrapper tokens rather than the visual block. Adopting the hook fixed that: an image row now reports 1332 tokens on Qwen3-VL and 3624 on Nemotron for the two-photograph CLI runs, against 134 and 55 before, and `embed` reads the batch as the engine built it instead of rewriting the ids.

The counts come from different places. For Qwen3-VL it is the Qwen2-VL processor's own grid: `t * (h / merge) * (w / merge)` per image, the same arithmetic the generative path uses to size `<|image_pad|>` runs. For Nemotron it is `num_image_token * tiles`, 256 per `512x512` tile.

Expanding at the id level is exactly equivalent to expanding the string before tokenization, because `<|image_pad|>` and `<IMG_CONTEXT>` are both added special tokens that always tokenize to one id. The function shares one implementation between the two families, and the shared error paths (placeholder count against image count, and a zero-token image) are unit-tested.

The counts are derived once per call from the same processor the forward pass uses, so the number the engine expands with and the number `embed` merges cannot disagree.

### 2.4 Use the empty string as the "this row carries an image" signal

`format_text` needs to know whether to emit the image block, and the trait gives it only the text. The engine's image path calls `format_text("", instruction)`; every text path rejects an empty string before it can reach the family, in two places: `EmbeddingEngine::embed_texts` returns `input[i] is an empty string`, and `validate_items` in the route returns the same as a `400`. So the empty string is unambiguous inside this codebase.

This is a real coupling and the doc comments on both families say so, naming `embed_image` and the two rejection sites. It is the least invasive option available to a wave-parallel family port; a follow-up that adds an explicit flag to the trait can delete both comments and the branch.

### 2.5 Do not reuse `InternVLProcessor` for Nemotron tiling

`src/models/llama_nemotron_vl_tiling.rs` is a separate 187-line module rather than a constructor variant on the shared InternVL processor. Two differences forced it:

1. `find_closest_aspect_ratio` is a different function. The InternVL port minimizes `|aspect - target|` and breaks ties on area proximity; this checkpoint maximizes `min(area_ratio, 0.6) * min(target / actual, actual / target)`. The two agree on most images and disagree on some, so the difference is a silent one, which is the reason to get it right rather than the reason to ignore it.
2. Normalization is SigLIP's (`mean = std = 0.5`, from `processor_config.json` `norm_type: "siglip"`), and the output must be channels-last because `SigLipVisionModel` convolves `[B, H, W, C]` while `InternVLProcessor` emits `[tiles, 3, H, W]`.

Making the shared processor configurable enough to cover both would have changed the InternVL VLM's tiling behaviour behind a flag, for the benefit of one caller. The new module keeps the shared one exactly as it is. What it does reuse is `InternVLConnector`, which is the actual model code: `pixel_shuffle(0.5)` with both `ps_version: v2` permutes, then `LayerNorm(4608) -> Linear(4608 -> 2048) -> GELU -> Linear(2048 -> 2048)`.

The candidate-grid enumeration sorts by `(cols * rows, cols, rows)` rather than by tile count alone. The reference sorts a Python set by `cols * rows`, leaving the intra-group order to set iteration; pinning it makes the strict-greater-than tie-break deterministic across runs and platforms.

### 2.6 Read the tile budget from `processor_config.json`, not `config.json`

The checkpoint declares `max_input_tiles` twice with different values: `2` at the top level of `config.json` and `6` in `processor_config.json`. The reference builds its processor from `processor_config.json` through `AutoProcessor`, so `6` is what actually runs, and that is what mlxcel reads. `image_size`, `use_thumbnail`, `num_image_token` and `passage_prefix` come from the same file, with the published values as fallbacks when a key or the whole file is missing.

### 2.7 Reuse `embedding_sanitize` by stripping the prefix first

Nemotron's text weights arrive as `language_model.embed_tokens.weight`, `language_model.layers.{i}....`, `language_model.norm.weight`, while `Llama3Model::from_weights` reads `model.`-prefixed keys. Rather than rename straight to `model.`, `sanitize_nemotron_vl_weights` strips `language_model.` and then calls the shared `sanitize_decoder_embedding_weights`, which re-adds `model.` to exactly the three backbone roots, drops any generation head, and folds any `{N}_Dense` module folder.

The detour through the shared helper is the point: this checkpoint has no `Dense` module and no `lm_head` today, but a re-export or a future revision might, and going through the shared path means it is handled without a second implementation. The two Nemotron-specific drops, the SigLIP attention-pooling `head.*` and the non-parameter buffers (`rotary_emb.inv_freq`, `*position_ids`), happen before the shared pass so they cannot be mistaken for backbone roots.

### 2.8 Drive the Llama layers locally instead of depending on the sibling port

Issue #1345 suggested `Llama3Backbone::from_weights_without_head` from #1325 "or a local equivalent". #1325 was running in the same wave against the same shared files, so `LlamaNemotronVLEmbeddingModel` constructs a plain `Llama3Model` with `tie_word_embeddings` forced true (which makes the constructor reuse `embed_tokens` for the head it never applies, instead of failing on an `lm_head` this checkpoint does not ship) and drives `embed_tokens`, `layers` and `norm` directly. All three are already `pub` fields.

The bidirectional behaviour comes entirely from the mask: `create_bidirectional_padding_mask` produces `[B, 1, 1, L]` blocking only padding keys, and `Attention::forward` uses the supplied mask instead of its causal fast path. Fresh `KVCache`s at offset 0 keep RoPE positions aligned with the input length.

This is a deliberate duplication of about fifteen lines against a merge conflict in wave 1. If #1325's backbone type is the better home once both have merged, collapsing the loop into it is a mechanical follow-up.

### 2.9 Take the pixel bounds from the checkpoint

`load_qwen3_vl` builds its processor with the Qwen2-VL defaults, whose ceiling is `16384 * 28 * 28` pixels: at `patch_size` 16 and `spatial_merge_size` 2 that is up to 12544 visual tokens for one image, well past the embedder's 8192-token budget. `Qwen/Qwen3-VL-Embedding-2B` declares `max_pixels: 1310720` in `preprocessor_config.json`, which caps one image at 1280 tokens. `apply_pixel_bounds` overwrites `min_pixels` and `max_pixels` after load from that file, so a re-export with different bounds stays correct instead of inheriting a hard-coded number.

### 2.10 Right padding, and why it matters more here than upstream

The reference Nemotron processor sets `tokenizer.padding_side = "left"` while computing `position_ids = arange(0, L)` regardless, so a padded row's real tokens sit at positions `pad_count..L-1` and a solo row's at `0..n-1`. The two therefore differ numerically upstream. mlxcel's engine right-pads, which puts real tokens at `0..n-1` in both cases, so the padded batch reproduces the solo result by construction rather than by tolerance. The same holds for Qwen3-VL: last-token pooling under a causal-plus-padding mask means padding sits after the pooled position and is blocked as a key for every real query.

### 2.11 Draw the cross-modal test images instead of committing fixtures

The repository's one image fixture is a flat orange square, which cannot separate two captions. `src/models/vl_embedding_test_images.rs` draws two scenes deterministically, a bar chart on white and a beach with sky, sun, sea and sand, at any requested aspect ratio. That keeps the gate reproducible byte for byte, adds no binary fixture, and lets the tiling gate ask for a 6:1 strip. The manual real-checkpoint runs used three downloaded photographs instead, which is the stronger evidence and is reported in section 5.

---

## 3. Implementation Details

### 3.1 Qwen3-VL-Embedding forward

Text rows:

```
mask   = create_causal_padding_mask(attention_mask, 0)          # [B, 1, L, L]
hidden = text_model.forward_hidden(input_ids, None, fresh caches, Some(mask))
pooled = pool(hidden, attention_mask, LastToken)
```

The M-RoPE and DeepStack slots are cleared first, so `forward_hidden` takes the text-only fast path. That path uses `fast_rope`, which is numerically the multimodal path for a sequence with no vision tokens: with `T == H == W` at every position, the three M-RoPE sections carry the same frequency and the interleaving is the identity.

Image rows (always `B == 1`; the engine has already expanded the placeholder):

```
pixels, grid = processor.preprocess_with_grid([image])
merged       = vlm.get_input_embeddings(input_ids, pixels, grid)  # sets M-RoPE + DeepStack state
hidden       = text_model.forward_hidden(input_ids, Some(merged), fresh caches,
                                         Some(create_causal_padding_mask(attention_mask, 0)))
pooled       = pool(hidden, attention_mask, LastToken)
```

The state is cleared again on the way out so a later text call cannot inherit it. Because `input_embeddings` is `Some`, `forward_hidden` skips both the state-clearing branch and the fast path and runs the general path, which is where DeepStack injection lives.

### 3.2 The rendered prompt

`format_text` runs the checkpoint's own `chat_template.jinja` through `ChatTemplateProcessor::apply_raw` on a two-message list, with `add_generation_prompt` at its default `true`. For a text row:

```
<|im_start|>system
Represent the user's input.<|im_end|>
<|im_start|>user
a photo of a dog<|im_end|>
<|im_start|>assistant
```

and for an image row the user turn is `<|vision_start|><|image_pad|><|vision_end|>` instead. Both strings are asserted exactly in the tests against the real template file, which needs the checkpoint directory but no weights.

The instruction defaults to `config_sentence_transformers.json` `prompts[default_prompt_name]`, which is `Represent the user's input.` on the published 2B. A caller-supplied instruction is trimmed and gets a trailing `.` when its last character is alphanumeric, so `Represent the user's input` becomes a sentence while `画像を表す。` and `Find the matching image?` are left alone.

The pooled position is the final newline of the assistant header, which is what `pooling_mode: lasttoken` selects on this prompt.

### 3.3 Llama-Nemotron-VL-Embed forward

```
h = embed_tokens[input_ids]                                       # [B, L, 2048]
if image:
    v = SigLipVisionModel.forward(pixels).hidden_states            # post_layernorm, [tiles, 1024, 1152]
    v = mlp1(pixel_shuffle(v, 0.5))                                # [tiles, 256, 2048]
    h = merge_llava(img_context_token_id, v, h, input_ids)
mask = create_bidirectional_padding_mask(attention_mask)           # [B, 1, 1, L]
for i, layer in layers: h = layer.forward(h, fresh KVCache, Some(mask))
h = norm(h); pooled = pool(h, attention_mask, Mean)
```

`select_layer: -1` means the reference reads the tower's `last_hidden_state`, which for a `SiglipVisionModel` is the `post_layernorm` output, which is what mlxcel's `SigLipVisionModel::forward` returns when no feature-layer selection is configured. The attention-pooling head is dropped at load precisely because this path never reaches it. The reference also skips its `vit_embeds[:, 1:, :]` CLS strip for SigLIP towers, and mlxcel's SigLIP embeddings carry no CLS token, so the two agree without a special case.

The `mlp1` LayerNorm uses eps `1e-5`, PyTorch's `nn.LayerNorm` default, because the checkpoint declares no eps for it.

### 3.4 The document prompt

`format_text` is the identity for text: the `query: ` and `passage: ` prefixes are caller-side, exactly as for the text-only sibling `nvidia/llama-nemotron-embed-1b-v2`. An image row carries no caller text, so the family emits the reference's document form itself:

```
passage: <img><IMG_CONTEXT></img><space>
```

The trailing space is not incidental. The reference builds `content = "<image>" + " " + text` and then `content = passage_prefix + " " + content`, so with an empty text the string ends in a space, and reproducing it keeps the token sequence identical to the reference's. The single `<IMG_CONTEXT>` is expanded to `256 * tiles` in `embed`; a real-checkpoint test tokenizes the emitted prompt with the checkpoint's own tokenizer, asserts exactly one placeholder and a leading `<|begin_of_text|>`, and checks the expanded width for both a 1-tile and a 7-tile image.

### 3.5 Weight-key mapping

| Published form | After sanitization |
|----------------|--------------------|
| `language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `language_model.layers.{i}.*` | `model.layers.{i}.*` |
| `language_model.norm.weight` | `model.norm.weight` |
| `vision_model.vision_model.*` | unchanged |
| `mlp1.0.*`, `mlp1.1.*`, `mlp1.3.*` | unchanged |
| `vision_model.vision_model.head.*` | dropped |
| `lm_head.*`, `*rotary_emb.inv_freq`, `*position_ids` | dropped |

`patch_embedding.weight` needs no handling here: `VisionEmbeddings::from_weights` already detects the PyTorch `[1152, 3, 16, 16]` layout and transposes it to channels-last.

### 3.6 Registration and the request side

Only the two arms of `build_family_model` changed; every other family's "not yet supported" arm is untouched, which mattered because three sibling ports were landing in parallel and one of them (#1411, BERT and XLM-RoBERTa) merged into this file mid-flight.

Detection, `src/model_metadata.rs` and the `mlxcel arch` entries already existed, so `mlxcel arch` lists both under `Embedding` and `mlxcel list` shows both checkpoints with no change. The route needed none either: `validate_items` already gates `image_url` on `supports_images()`, `fetch_images` already decodes under the shared image limits, `embed_items` already runs images one at a time and writes results back in request order, and `instruction` already reaches `format_text`. `mlxcel embed --image` already did the same offline.

---

## 4. Test Strategy

### 4.1 Without a checkpoint

- `forward_hidden_then_head_matches_forward_impl`: the refactor guard, token-exact on a synthetic two-layer Qwen3-VL.
- `expand_image_placeholders_*`: order across several placeholders, padding-flag preservation, and both error paths.
- `instruction_gets_trailing_period_only_when_it_ends_mid_sentence`: including a non-ASCII terminator.
- `sanitize_drops_the_vision_head_and_maps_the_language_model_prefix`: every key that must survive, and every class that must not.
- `a_square_image_uses_one_tile_and_a_wide_image_uses_the_budget` and `image_block_expands_to_num_image_token_per_tile`: the tiling budget for a square, a 6:1 strip and a 2:3 page.
- `preprocess_emits_channels_last_tiles_in_the_siglip_range`: shape `[tiles, H, W, 3]` and values inside `[-1, 1]`.
- `pixel_shuffle_then_mlp1_maps_one_tile_to_256_language_tokens`: `[1, 1024, 1152]` to `[1, 256, 2048]` on random weights.

### 4.2 With the checkpoint present, soft-skipping otherwise

- The two `format_text` gates render against the real `chat_template.jinja` and assert the exact prompt string.
- `bidirectional_prefill_lets_an_early_token_see_a_later_one`: over 96 tokens, changing only the last token must move the pooled vector. A causal stack cannot do that.
- `the_document_prompt_expands_to_256_tokens_per_tile`: the emitted prompt tokenized by the checkpoint's tokenizer, then expanded, for a 1-tile and a 7-tile image.
- Text gates for both families: identical rows, paraphrase against unrelated, the unrelated ceiling, and a padded batch against the single-input vector.
- Image gates for both families: unit vectors, finite components, the cross-modal margin in both directions, and re-embedding the same image twice.

Every test that builds a model or evaluates MLX ops takes the process-wide `mlx_test_guard()`, and all gate numbers were recorded under `--test-threads=1`.

### 4.3 Generative regression

The `forward_hidden` split is guarded by the token-exact unit test, and separately by running `mlxcel generate` on `Qwen/Qwen3-VL-Reranker-2B` (a generative Qwen3-VL checkpoint that happens to be downloaded here), once with text and once with an image. The text run answered the question coherently; the image run described a striped cat as a tiger, which is the checkpoint being a reranker rather than a captioner, and is far from the garbage a broken DeepStack or M-RoPE path would produce.

---

## 5. Real-Checkpoint Results

Linux, GB10, CUDA, bf16 (the Apple-Silicon bf16-to-f16 rule does not apply here). Three consecutive runs of every command produced bit-identical similarity matrices, on the CLI and through the server; the maximum absolute spread across runs was `0.000e+00` in all eight measurements.

### `Qwen/Qwen3-VL-Embedding-2B` (2048 wide, `max_length` 8192)

| Gate | Observed | Requirement |
|------|----------|-------------|
| Vector width and norm | 2048, norms 1.000000 | 2048-dim unit vectors |
| Identical inputs in one batch | cosine 1.000000000 | 1.0 within 1e-6 |
| Paraphrase vs unrelated | 0.7382 against 0.1245 | margin at least 0.15 |
| Unrelated ceiling | 0.1245 | below 0.5 |
| Padded batch vs single input | largest drift 9e-4 across shapes | within 1e-3 |
| Liberty caption, matching vs unrelated photo | 0.5575 against 0.1352 | margin at least 0.1 |
| Cat caption, matching vs unrelated photo | 0.5342 against 0.1676 | margin at least 0.1 |
| Non-finite components | none | none |

Measured through `mlxcel embed` and through `mlxcel-server` plus one `POST /v1/embeddings` request mixing text and `image_url` items; both surfaces returned the same values.

A third pairing was weaker and is reported because it is real: against a cluttered indoor photograph of powdered-sugar pastries, the caption preferred its own image by only 0.016 (0.1471 against 0.1291 and 0.1312 for two unrelated photographs). The direction is correct, so the epic's ordering gate holds, but the 0.1 margin the issue names does not hold for every pairing on this model. The two images in that set also score 0.6187 against each other, so this family's image-image similarity floor is high, the same anisotropy the SigLIP text tower shows.

### `nvidia/llama-nemotron-embed-vl-1b-v2` (2048 wide, `max_length` 8192)

| Gate | Observed | Requirement |
|------|----------|-------------|
| Vector width and norm | 2048, norms 1.000000 | 2048-dim unit vectors |
| Identical inputs in one batch | cosine 1.000000000 on the CLI batches | 1.0 within 1e-6 |
| Relevant vs irrelevant passage | 0.4284 against -0.0219 | margin at least 0.15 |
| Unrelated ceiling | -0.0219 | below 0.5 |
| Padded batch vs single input | within 1e-3 | within 1e-3 |
| Liberty query, matching vs unrelated photo | 0.4041 against 0.0963 | margin at least 0.1 |
| Cat query, matching vs unrelated photo | 0.4836 against 0.0440 | margin at least 0.1 |
| Pastry query, matching vs two unrelated photos | 0.4836 against 0.0792 and 0.0896 | margin at least 0.1 |
| 6-tile landscape plus thumbnail | `7 * 256` visual tokens, embeds without error | no truncation |

All three cross-modal pairings clear the margin on this family, including the pastry photograph that Qwen3-VL only ordered correctly.

### The identical-row observation

One server measurement did not reach cosine 1.0: a request whose text items were two identical Liberty captions plus one different caption returned 0.99995714 for the identical pair on Nemotron. A layer-by-layer probe located the divergence precisely. Rows 0 and 1 of three identical rows agree bit for bit through layer 4; at layer 5 row 2 differs by `9.766e-4`, exactly one bf16 ulp at that magnitude, and the difference compounds through the remaining eleven layers to `5.0e-1` in the pre-pooling hidden state, which is `4.3e-5` in the normalized vector.

Three controls place this outside the port:

- A sweep over batch sizes 2 to 8 and nine token lengths reproduces the same class of drift on the already-merged `Qwen3Embedding` family (6.0e-5 at one length), which shares none of this port's code.
- Isolated probes of `matmul` and `scaled_dot_product_attention` at the exact shapes involved, with tiled identical rows and both masked and unmasked, are exactly row-position invariant.
- The drift is fully deterministic: the same input produces the same value on every run, so it is not a race and not the MLX threading hazard.

Disabling `MLXCEL_FUSED_ADD_RMSNORM` and `MLXCEL_FUSED_ROPE_APPEND` changes nothing. The conclusion recorded in the test constants is that this is a property of the shared bf16 batched decode path at some shapes, not of either family's forward pass, and that a future failure of the 1e-6 assertion should be met with the sweep before the forward pass is suspected.

---

## 6. Validation Summary

| Command | Result |
|---------|--------|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo test ... --lib -- --test-threads=1 models::qwen3_vl models::llama_nemotron_vl` | 23 passed, 0 failed |
| `cargo test ... --lib -- --test-threads=1 embeddings::` | 69 passed, 0 failed |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |
| `mlxcel embed` on both checkpoints, text and image, 3 runs each | exit 0, bit-identical |
| `mlxcel-server` plus `POST /v1/embeddings` on both, mixed text and image items, 3 runs each | HTTP 200, bit-identical |
| `mlxcel generate` on `Qwen/Qwen3-VL-Reranker-2B`, text and image | exit 0, coherent output |

The issue's acceptance criteria name the macOS `metal,accelerate` feature set. The CUDA gate is the equivalent on this host and is what ran; the macOS gate is CI's.

An embedding-only server reports `503` on `/health` because no chat model loads, which is the foundation's existing behaviour and not a change here; readiness was probed on `/v1/embeddings` instead.

No performance numbers are reported. The epic runs its performance pass separately.

---

## 7. Change Summary

| File | Change |
|------|--------|
| `src/models/qwen3_vl_embedding.rs` (new, 402) | `Qwen3VLEmbeddingModel`: loader reuse, chat-template formatting, placeholder expansion, last-token pooling |
| `src/models/qwen3_vl_embedding_tests.rs` (new, 300) | Instruction rule, expansion, exact prompt rendering, the real-checkpoint text and image gates |
| `src/models/llama_nemotron_vl_embedding.rs` (new, 387) | `LlamaNemotronVLEmbeddingModel`: SigLIP plus `mlp1` plus bidirectional Llama, mean pooling, weight sanitization |
| `src/models/llama_nemotron_vl_embedding_tests.rs` (new, 495) | Sanitizer, tiling, connector shapes, bidirectionality, the real-checkpoint gates |
| `src/models/llama_nemotron_vl_tiling.rs` (new, 187) | The checkpoint's area-aware tiling and SigLIP normalization, channels-last |
| `src/models/qwen3_vl_tests.rs` (new, 173) | The token-exact `forward_hidden` refactor guard |
| `src/models/vl_embedding_test_images.rs` (new, 100, test-only) | Two deterministic synthetic scenes for the cross-modal gate |
| `src/models/qwen3_vl.rs` (+40/-6) | `forward_for_sequence` split into `forward_hidden_for_sequence` plus the head; the text-only path likewise |
| `src/embeddings/loader.rs` (+15/-7) | Two family arms constructed; the `not yet supported` arm removed, since every variant the epic enumerated now has one |
| `src/embeddings/real_checkpoint_tests.rs` (+27/-14) | The unported-family gate inverted: no embedding variant may report `not yet supported` |
| `src/loading/mod.rs` (+5) | `load_qwen3_vl` re-exported crate-internally |
| `src/models/mod.rs` (+5) | Four module declarations |
| `docs/embeddings.md` (+47) | Two family-notes subsections |
| `docs/supported-models.md` (+4) | Two Embedding rows plus the paragraph naming all four image-capable families |

14 files, 2202 insertions, 30 deletions. Rebased three times during the wave, onto the BERT and XLM-RoBERTa port (#1411), the late-interaction port (#1414) and the bidirectional-decoder port (#1415), all of which merged while this one was in flight. This is the last family of epic #1348, so `build_family_model` no longer has an unported arm.

---

## 8. What Remains Unverified

- **Numeric parity against the reference implementations.** No PyTorch or `transformers` install exists on the validation host, so nothing here is compared against `LlamaNemotronVLModel` or the Qwen3-VL-Embedding wrapper running upstream. Neither model card publishes a reference similarity matrix, so unlike the Qwen3-Embedding port there was no published number to check against either. Every figure in section 5 is a threshold from the issue, measured end to end, not a parity claim.
- **macOS and Metal.** Everything ran on Linux with CUDA. The bf16-to-f16 conversion rule, which only fires on Apple Silicon, is therefore untested for both families.
- **Several images in one request.** `expand_image_placeholders` consumes a list of counts and is unit-tested with two placeholders, but the engine sends one image per `embed` call, so the multi-placeholder path has no end-to-end coverage.
- **Video inputs.** Out of scope per the issue. `<|video_pad|>` is not expanded, and a video item would fail the placeholder-count check rather than silently produce a wrong vector.
- **`Qwen/Qwen3-VL-Embedding-8B`.** Same code at a different size, out of scope, not loaded.
- **Quantized exports.** Neither family has a published quantized checkpoint yet. The code threads `quantization_params` into the SigLIP tower and the connector, and `UnifiedLinear` falls back to a dense linear when `.scales` is absent, but no quantized artifact was loaded.
- **Nemotron's reference truncation.** `max_length` resolves to the shared cap of 8192 rather than the reference's `p_max_length` of 4096 and `q_max_length` of 512, because those keys live in `processor_config.json`, which the shared derivation does not read. `--embedding-max-length` reproduces it; the default does not.

---

## 9. Follow-up Actions

- Give `EmbeddingModel::format_text` an explicit "this row carries an image" input, or move formatting after preprocessing, so the empty-string convention in 2.4 can be deleted. This is foundation surface and should land after the epic's parallel wave.
- Once #1325 has merged, consider collapsing the Nemotron layer loop into its bidirectional Llama backbone.
- If a host with `transformers` becomes available, add a parity check against `LlamaNemotronVLModel.encode_queries` and `encode_documents`, which is the one class of evidence this report cannot offer.
- The epic's performance pass should include an image row for both families, since the vision tower and the tiling budget dominate that path and nothing here measured it.

---

## References

- Issue #1345, epic #1348
- PR #1408 (embedding foundation), issue #1353
- PR #1411 (BERT and XLM-RoBERTa), PR #1413 (EmbeddingGemma and Qwen3-Embedding)
- `docs/embeddings.md`, `docs/supported-models.md`
- `Qwen/Qwen3-VL-Embedding-2B`, `nvidia/llama-nemotron-embed-vl-1b-v2`
