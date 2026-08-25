# Technical Report: PR #1414 - ColIdefics3 and ColQwen2.5 late-interaction embedders

**Date**: 2026-08-26
**Author**: mlxcel maintainers
**Status**: Completed; two checkpoint layouts covered by unit tests rather than by a published artifact
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1414 implements the two late-interaction (ColBERT-style) visual document retrievers issue #1337 asks for, on top of the `/v1/embeddings` foundation from #1408. ColIdefics3 runs the existing SmolVLM / Idefics3 stack and ColQwen2.5 the existing Qwen2.5-VL stack; both stop at the decoder's final norm, project every token's hidden state to 128 dimensions with one `Linear`, L2-normalize each token vector, zero the padding rows, and are ranked with MaxSim instead of cosine. Neither family adds a decoder or a vision tower: the change is a head-free assembly, one projection, two prompt formats, three weight-key layouts, and one new trait hook so an expanded image prompt reports the token count it actually consumed.

Two things in the issue's specification did not survive contact with the checkpoints, and finding them is most of what this PR is about. The ColQwen2.5 document prompt closes with `<|endoftext|>` rather than an assistant header, and `vidore/colqwen2.5-base` stores its vision patch filter in `Conv3d`'s native PyTorch layout while the encoder in this tree only accepts the channels-last layout an mlx conversion produces. With both wrong the retriever ranked an unrelated page above the matching one; with both fixed it ranks correctly with a 50.7 percent margin.

---

## 1. Problem Statement

### 1.1 Background

ColIdefics3 and ColQwen2.5 are visual document retrievers: a page is rendered to an image, embedded as a set of token vectors, and scored against a query's token vectors with MaxSim, `sum_i max_j dot(q_i, d_j)`. mlxcel already ran both backbones for generation, so the missing pieces were narrow, but each one is load-bearing in a way that fails silently rather than loudly.

- Both loaders require an `lm_head`. `SmolVLMModel` builds a `Llama3Model`, whose constructor always loads a head, tied or untied. A ColIdefics3 checkpoint declares `tie_word_embeddings: false` and ships no head tensor at all, so that constructor cannot open it.
- `Qwen2VLModel::forward_for_sequence` ends in `lm_head`, so an embedder reaching the hidden states through it would materialize a `[B, L, 151936]` logit tensor per micro-batch and then discard it.
- The `/v1/embeddings` engine derives the number of returned rows and `usage.prompt_tokens` from the tokenized row, before padding. A vision-language prompt carries one placeholder that the processor expands into hundreds of image tokens, so without a hook the response would claim a dozen rows for a sequence the forward pass ran at eight hundred.
- `mlxcel embed` printed a cosine matrix, which is the wrong scoring rule for a multi-vector family and would silently rank badly rather than fail.

### 1.2 What the epic required

Real checkpoints, not synthetic ones. `vidore/colSmol-256M` and `vidore/colqwen2.5-v0.2` are the trained retrievers, and both are PEFT LoRA adapters on top of a `-base` repository, carrying the trained projection in a sentence-transformers `1_Dense/` folder and nothing else. Merging an adapter is explicitly out of scope for mlxcel, which leaves a validation problem: the base repositories load, but their projections are randomly initialized, so they prove the weight layout and nothing about retrieval.

---

## 2. Technical Decisions

### 2.1 Assemble a head-free Llama instead of adding one to `llama3.rs`

Issue #1337 proposed adding `forward_hidden` and `from_weights_without_head` to `src/models/llama3.rs`, noting that #1325 wants the same functions and that whichever lands first wins. Both ran in parallel here, and `llama3.rs` is the single most contended file in that pair.

The decision was to not touch it. `Llama3Model`'s fields are public and `TransformerBlock`, `UnifiedEmbedding` and `RMSNorm` are public types with public constructors, so a head-free backbone is an assembly difference rather than a second implementation of the architecture. `src/models/headless_llama.rs` is 98 lines that build the same blocks with the same resolved rope table and stop at the same final norm. The cost is one deliberate omission: `Llama3Model::forward` calls a private `pipeline_hint` between layers, which this path skips. That hint is a pipeline-parallel scheduling annotation with no numeric effect, and an embedding forward is a single pass on one device.

This is a deviation from the issue text, and it is worth stating why it is the better trade even ignoring the merge conflict. `from_weights_without_head` on `Llama3Model` would produce a `Llama3Model` whose `lm_head` field cannot be `None` without changing the type, so the alternative shapes were a dummy head, an `Option`, or a second struct. A second struct is what this is.

### 2.2 Split `forward_hidden` out of `Qwen2VLModel::forward_for_sequence`

`Qwen2VLModel`'s fields are private, so ColQwen2.5 genuinely needs the split. It is the mechanical kind: everything through `norm` moves into `forward_hidden`, and `forward_for_sequence` becomes that call followed by `self.lm_head.forward(&h)`. `forward_hidden_then_head_matches_forward` asserts the two produce a bit-identical logit tensor (`max_abs_diff == 0.0`) on a two-layer synthetic model, which is what makes "generation is unchanged" a checked statement rather than an assurance.

No sibling unit touches `qwen2_vl.rs`, so this split carries no coordination cost.

### 2.3 Expand the image prompt in the engine, not inside `embed`

The engine's `postprocess` slices `[B, L, D]` back into per-item matrices using `token_counts`, which come from `EncodedBatch`, which is built before the family sees anything. If a family expanded `<image>` into 832 tokens inside `embed`, the output would have 870 rows and the engine would return the first 12 of them while reporting 12 prompt tokens.

`EmbeddingModel::expand_image_tokens(ids, images) -> Result<Vec<u32>>` is therefore called by `EmbeddingEngine::embed_image` between `encode_row` and `EncodedBatch::from_rows`. It defaults to the identity, so no existing family changes and no other family is forced to think about it. Both new families implement it by computing only the cheap half of preprocessing (the tile layout for ColIdefics3, `compute_grid_thw` for ColQwen2.5) and delegating the actual token insertion to the generation path's `insert_smolvlm_image_tokens` and `insert_qwen_vl_image_tokens`. The expensive half, pixel extraction, still happens once, in `embed`.

The alternative considered was letting `EmbeddingOutput` carry its own row counts. That is a larger contract change for one family pair, and it would move the authority for `usage.prompt_tokens` out of the tokenizer, where every other family keeps it.

### 2.4 The `1_Dense` folder outranks the root projection

A merged export keeps two projections: the base repository's untrained `linear.*` or `custom_text_proj.*` in the main shard, and the trained one in `1_Dense/model.safetensors`, which `load_weights_from_dir_with_subfolders` surfaces as `1_Dense.linear.*`. sentence-transformers applies the module folder, so the folder has to win, and `apply_dense_projection_override` moves it over the root key.

One subtlety is handled explicitly: if the root projection were quantized and the folder dense, keeping the root's `scales` and `biases` would make `UnifiedLinear::from_weights` treat the folder's dense weight as a packed one. The override therefore drops any quantization tensor the folder did not itself supply.

`embedding_sanitize::fold_dense_modules` was deliberately not reused. It renames `{N}_Dense.linear.*` to `dense.{k}.*`, which is the EmbeddingGemma post-pooling chain; here the folder replaces a named projection rather than appending to a list.

### 2.5 Refuse LoRA-only repositories with the fix in the message

`reject_lora_only_checkpoint` fails a directory that has `adapter_model.safetensors` and no other top-level shard, naming the merge as the fix. The alternative, loading the base and ignoring the adapter, would serve an untrained retriever that returns correctly shaped, correctly normalized, entirely meaningless vectors. That is exactly the failure mode a gate does not catch unless it measures ranking, so it is refused at load.

### 2.6 MaxSim is reported raw, not averaged

`crate::embeddings::maxsim` returns the ColBERT sum, so a query against itself scores its own row count. `mlxcel embed` previously divided the multi-vector score by the query length, which reads like a cosine and hides the property that makes the number checkable: an exact `24.0000` on the diagonal is direct evidence that every query row is a unit vector and that the maximum over the document is attained at the identical row. Both real-checkpoint runs show that diagonal.

### 2.7 Clear the M-RoPE slot before a text batch

`Qwen2VLModel::forward_hidden` reuses a stored `[3, B, L]` position grid whenever its batch matches and it covers the requested range. An image request leaves exactly such a grid in the fallback slot. A text micro-batch of batch 1 and equal-or-shorter width would therefore be handed an image's spatial positions. `text_input_embeddings` calls `clear_mrope_state()` first, and `stale_image_positions_do_not_leak_into_a_text_batch` installs a fake repeating grid and asserts the text result is unchanged.

---

## 3. The two specification corrections

### 3.1 The ColQwen2.5 document prompt

The issue specifies `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|im_start|>assistant\n`. The reference `ColQwen2_5_Processor.visual_prompt_prefix` ends with `<|endoftext|>` instead, on both the current `colpali-engine` main branch and the 0.3.x line the `v0.2` checkpoint was trained under, and `vidore/colqwen2.5-v0.2` ships an `additional_chat_templates/sentence_transformers.jinja` that emits the same string. Nothing is generated after a retrieved page, so an assistant header is not merely different, it is a turn opening the model never saw in training.

### 3.2 The raw HuggingFace patch-embedding layout

`Qwen25VLVisionEncoder::PatchEmbed` reads a five-dimensional `visual.patch_embed.proj.weight` and permutes it `[0, 2, 3, 4, 1]` to reach `[out, T, C, H, W]`. That permutation is correct for the mlx-vlm conversion's `[out, kT, kH, kW, in]`, which is the only layout the generation loader ever sees, because `load_qwen2_5_vl` is used with `mlx-community` conversions. `vidore/colqwen2.5-base` is a raw HuggingFace export and stores `Conv3d`'s native `[out, in, kT, kH, kW]`, which is `[1280, 3, 2, 14, 14]`. Permuting that gives `[1280, 3, 14, 2, 14]`, a reshape of scrambled filters that still has the right element count and therefore fails no shape assertion.

`normalize_patch_embed_layout` detects the layout by the channel axis (`in_channels` is 3 and the patch size is 14, so only the converted layout has 3 on the trailing axis) and applies the mlx-vlm conversion step `[0, 2, 3, 4, 1]` before the encoder sees it.

### 3.3 How much the two mattered

Both were found by measurement, not by reading. The first ColQwen2.5 run on the merged checkpoint scored the unrelated page higher than the matching one, and the two pages scored 86 percent of their own self-similarity, which is the signature of a vision tower producing near-constant features rather than a prompt-format problem.

| Measurement | Before both corrections | After both corrections |
| --- | --- | --- |
| MaxSim, query against the revenue table | 7.8338 | 18.3363 |
| MaxSim, query against the unrelated page | 8.0446 | 9.0444 |
| Ranking | wrong | correct |
| MaxSim, table against unrelated page | 671.61 of 781 (86 percent) | 247.58 of 779 (32 percent) |

The ColIdefics3 prompt was kept exactly as the issue specifies. Its checkpoint also ships a second rendering of the same pieces (`<|im_start|>User: Describe the image.<image><end_of_utterance>`), and both were measured on the merged checkpoint: 53.64 percent relevance margin for the processor's form against 53.77 percent for the template's. They are equivalent in practice, so the form the issue and the reference processor agree on is the one kept, with the measurement recorded next to the constant.

---

## 4. Implementation Details

### 4.1 Module layout

| File | Role |
| --- | --- |
| `src/models/col_late_interaction.rs` | `embedding_dim`, `reject_lora_only_checkpoint`, `apply_dense_projection_override`, `project_and_normalize`, `format_query` |
| `src/models/headless_llama.rs` | `HeadlessLlama`: embedding table, blocks, final norm, `forward_hidden` |
| `src/models/colidefics3.rs` | `ColIdefics3Model`, the SmolVLM weight remap, the tile processor, marker encoding |
| `src/models/colqwen2_5.rs` | `ColQwen25Model`, `rewrite_colqwen25_key`, `normalize_patch_embed_layout`, `text_input_embeddings`, `token_vectors` |
| `src/embeddings/maxsim.rs` | `maxsim` over rows, `maxsim_mlx` over device arrays |

### 4.2 The shared forward tail

`project_and_normalize(hidden, projection, attention_mask)` casts the projection output to f32, divides each token row by `max(||row||, 1e-9)`, and multiplies by the mask broadcast to `[B, L, 1]`. Running the normalization in f32 rather than the activation dtype is what makes the measured unit-norm error 1e-7 instead of a bf16 ulp, and the epsilon is what keeps an all-zero row at exactly zero rather than NaN. The engine normalizes a second time, which is idempotent for unit rows and leaves zeros at zero, so `EmbeddingModel::normalize()` stays `true` and `dimensions` truncation still re-normalizes.

### 4.3 ColQwen2.5 key sanitization

Three layouts reach the loader. `rewrite_colqwen25_key` strips a leading `vlm.` first, so the remaining rules do not have to be written twice, then maps `embedding_proj_layer.` to `custom_text_proj.`, `model.language_model.` and a bare `language_model.` to `model.`, and `model.visual.` and `visual.` to `vision_tower.`. The tied `lm_head` is dropped and `tie_word_embeddings` is forced true at load, which keeps a 151936 by 2048 head out of memory and out of the constructor's way. `sanitize_tied_embeddings` is deliberately not called: it would copy the embedding table into `lm_head.*` for a head this path never reads.

### 4.4 An image batch is one row

`compute_rope_index` reads row 0 of `input_ids` and derives one grid from it. The engine embeds images one at a time, so that is an invariant rather than a limitation, and `ColQwen25Model::forward` now fails loudly when it is violated instead of giving rows 1 and up the first row's positions.

---

## 5. Test Strategy

### 5.1 Synthetic checkpoints run the real load path

`colidefics3_tests.rs` materializes a complete synthetic checkpoint on disk: `config.json`, `preprocessor_config.json`, a `WordLevel` `tokenizer.json` whose vocabulary holds the exact tile-marker strings, and one f32 safetensors shard covering the text backbone, the SigLIP tower, the connector and the projection, at an 8 pixel image with 4 pixel patches and a pixel-shuffle factor of 2, so one tile compresses to exactly one image token. Every test then goes through `ColIdefics3Model::load(dir, config)`, including the `1_Dense` override and the image-placeholder expansion. `sanitize_prefers_1_dense_projection` writes a folder whose weight is a constant, so a correct override makes every normalized row the same constant unit vector, which the random root projection could not produce.

`colqwen2_5_tests.rs` runs the text path on a bare `Qwen2VLModel` through the same `text_input_embeddings` and `token_vectors` the model uses, so the mask, the head-free forward and the normalization are exercised as product code rather than restated. Its vision half is covered by the real-checkpoint gates instead of by a hand-built 32-block windowed ViT, which would restate the encoder rather than test it.

### 5.2 The MLX test guard

Every test in these modules that builds a model or evaluates an MLX op takes `crate::models::embedding_test_support::mlx_test_guard`, the process-wide lock the merged embedding families already share, and gate numbers were recorded with `--test-threads=1`. `EmbeddingModel` is single-thread by contract and the product honors it; libtest does not, and concurrent MLX work on this tree has been measured both corrupting results silently and aborting inside `cudaStreamEndCapture`.

### 5.3 Real-checkpoint gates soft-skip

`real_colsmolvlm_base_loads_and_projects_to_128` and `real_colqwen25_base_loads_and_projects_to_128` load the published base repositories when present and assert the derived geometry (64 image feature rows per tile for ColIdefics3, a 768 by 28 by 28 pixel cap for ColQwen2.5), the token ids, and the prompt formats. Both return early with a message when the checkpoint is absent, the convention `src/embeddings/real_checkpoint_tests.rs` follows.

---

## 6. Real-Checkpoint Results

### 6.1 Producing a trained checkpoint

Both published retrievers are LoRA adapters. Each was merged into its base outside the repository with a standalone script (`W' = W + (B @ A) * alpha / r`, with `alpha / r = 32 / 32 = 1.0`), 210 merged tensors for ColIdefics3 and 253 for ColQwen2.5. The merge is not taken on faith: `vidore/colqwen2.5-v0.2` carries a LoRA on `custom_text_proj` and also ships the fully trained projection in `1_Dense/`, so the two must agree. They do, to 2.44e-4 on weights up to 0.104, which is bf16 rounding at that magnitude, with the bias exactly equal. That one comparison validates the scale, the `B @ A` orientation and the target-key mapping, and by extension the 252 backbone deltas that have no independent oracle.

### 6.2 Gates

Query `What was the total revenue in 2023?` against a rendered revenue table and an unrelated rendered page, three repeats each, on GB10 with CUDA.

| Gate | ColIdefics3 (colSmol-256M merged) | ColQwen2.5 (colqwen2.5-v0.2 merged) |
| --- | --- | --- |
| Shapes | `[24, 128]`, `[876, 128]`, `[876, 128]` | `[24, 128]`, `[779, 128]`, `[779, 128]` |
| Rows against `usage.prompt_tokens` | 1776 = 1776 | 1582 = 1582 |
| Worst unit-norm error (limit 1e-5) | 1.03e-7 | 0.0 |
| MaxSim, query against matching page | 18.7396, spread 0 over 3 runs | 18.3363, spread 0 over 3 runs |
| MaxSim, query against unrelated page | 8.6879, spread 0 over 3 runs | 9.0444, spread 0 over 3 runs |
| Relevance margin (limit 10 percent) | 53.6 percent | 50.7 percent |
| Same input, separate process | 0.0 | 0.0 |
| MaxSim of a query against itself | exactly 24.0000 | exactly 24.0000 |

`POST /v1/embeddings` reproduces every number, with the query as one text item and the two pages as `image_url` data URIs. In `base64` mode the payload decodes to the same floats bit for bit and carries the sibling `shape` field.

### 6.3 The one gate that is not met

The epic asks that a padded batch match the unpadded single-input result within 1e-3. That holds in f32, which is what the synthetic unit tests run and pass at that tolerance. It does not hold in bf16 on this CUDA box: the same row embedded at batch 1 and at batch 2 differs by up to 1.5e-2 for ColIdefics3 and 7.5e-3 for ColQwen2.5.

Four observations place this outside the change rather than inside it.

- Padding rows are exactly zero, so the mask and the zeroing are not implicated.
- Two identical, unpadded rows inside one batch already differ, by 8.6e-3 and 6.6e-3, so padding is not implicated either.
- Every measurement is deterministic: three repeats are bit-identical and two separate processes agree to 0.0, which rules out a race.
- `Qwen/Qwen3-Embedding-0.6B`, merged in #1413 and untouched here, drifts 3.8e-3 between batch 1 and batch 2 on the same box.

The conclusion recorded here is that batched bf16 prefill on this hardware is not batch-size invariant at 1e-3, and that a multi-vector family shows it plainly because it reports per-token vectors instead of a pooled average. It does not move either ranking: the margins above are 50 percent and the drift is under 2 percent of a component.

---

## 7. Validation Summary

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | pass |
| `cargo check --profile test-fast --features cuda --all-targets` | pass |
| `cargo test ... --lib models::col -- --test-threads=1` | 21 passed, 0 failed |
| `cargo test ... --lib embeddings:: -- --test-threads=1` | 69 passed, 0 failed |
| `mlxcel embed` on both merged checkpoints | pass, numbers in section 6 |
| `mlxcel-server` plus `POST /v1/embeddings`, float and base64 | pass, numbers in section 6 |

---

## 8. Change Summary

19 files, 2758 insertions, 35 deletions.

New: `src/models/col_late_interaction.rs` and its tests, `src/models/headless_llama.rs`, `src/models/colidefics3.rs` and its tests, `src/models/colqwen2_5.rs` and its tests, `src/embeddings/maxsim.rs` and its tests.

Modified: `src/models/qwen2_vl.rs` (the `forward_hidden` split), `src/embeddings/model.rs` (`expand_image_tokens`), `src/embeddings/engine.rs` (calling it), `src/embeddings/loader.rs` (two arms), `src/embeddings/mod.rs`, `src/models/mod.rs`, `src/embeddings/real_checkpoint_tests.rs`, `src/commands/embed.rs` (the MaxSim matrix), `docs/supported-models.md`, `docs/embeddings.md`.

---

## 9. What Remains Unverified

- **The native `colqwen2` / `ColQwen2ForRetrieval` layout.** No such checkpoint was available. Its five remap rules are covered by unit tests over key strings, not by a load.
- **A reference-implementation parity run.** `colpali-engine`, `transformers` and PyTorch are not installed on this validation host, so no side-by-side comparison against the reference forward pass was possible. What stands in for it is what each checkpoint itself ships: the trained `1_Dense/model.safetensors` (which the merge was checked against) and `additional_chat_templates/sentence_transformers.jinja` (which the prompts were checked against), plus the published processor source.
- **macOS and Metal.** Everything here ran on Linux with CUDA. The bf16 to f16 conversion rule that applies on Apple Silicon was never exercised, and neither was the Metal attention path.
- **Quantized conversions.** No quantized ColIdefics3 or ColQwen2.5 checkpoint is published. The quantization plumbing (`quantization_params`, the `1_Dense` override dropping a stale `scales` and `biases`) is written and unit tested but never loaded from a real quantized artifact.
- **More than one image per input.** The engine embeds images one at a time and ColQwen2.5 now refuses a multi-row image batch explicitly; multi-image inputs are neither implemented nor tested.
- **Documents longer than the derived `max_length`.** Image expansion happens after truncation, so a very large page could in principle exceed the cap. At the published geometries it cannot: ColIdefics3 tops out near 1100 image tokens and ColQwen2.5 at 768, against an 8192 cap.

---

## 10. Follow-up Actions

- The batch-size sensitivity of bf16 prefill is worth a focused issue against the epic's numeric-gate wording, since it applies to every embedding family on CUDA and not to this pair.
- `crate::embeddings::maxsim_mlx` currently has no production caller. A retrieval or scoring endpoint over stored multi-vector documents, which #1337 lists as out of scope, is where it belongs.
- If a native `ColQwen2ForRetrieval` export becomes available, add it to `local_embedding_checkpoints_detect_to_their_families` and to the ColQwen2.5 gate so the `vlm.`-wrapped layout is covered by a load rather than by string tests.
- `normalize_patch_embed_layout` fixes the raw HuggingFace Conv3d layout for this family only. The generation loader `load_qwen2_5_vl` has the same blind spot and would mis-load a raw Qwen2.5-VL export the same way; that is a separate, pre-existing issue worth filing.

---

## References

- Issue #1337, epic #1348
- PR #1408 (embedding foundation), issue #1353
- PR #1411 (BERT and XLM-RoBERTa), PR #1413 (EmbeddingGemma and Qwen3-Embedding)
- `docs/embeddings.md`, `docs/supported-models.md`
- `vidore/ColSmolVLM-Instruct-256M-base`, `vidore/colSmol-256M`, `vidore/colqwen2.5-base`, `vidore/colqwen2.5-v0.2`
- `colpali-engine` processors: `ColIdefics3Processor`, `ColQwen2_5_Processor`
