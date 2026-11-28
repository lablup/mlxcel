# PR #2773: Use the transformers mask for Gemma 3 image prefill

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust, Python
**Risk Level**: Medium (every Gemma 3 image request now prefills under a different attention mask, so Gemma 3 image outputs change; PaliGemma and the text-only Gemma 3 path are unchanged)

## Executive Summary

After #2750 made the Gemma 3 image prompt identical to transformers, the image logit trace still disagreed with the float32 reference on 5 of 43 decided positions. Issue #2760 traced the gap to the attention mask on both arms: mlxcel prefilled Gemma 3 image prompts with a fully bidirectional mask on every layer, and the reference ran causally, while transformers uses causal attention plus bidirectional attention inside each image block, intersected with the sliding window on the sliding layers. This PR builds the transformers mask in `vision::merge`, applies the window band to it on Gemma 3's sliding layers, gives PaliGemma its own merge strategy so it keeps the prefix-LM mask, and passes the same mask in the trace reference arm. With the transformers mask in both arms the decided disagreement drops from 7 of 48 to 0 of 48, PaliGemma's trace is byte-identical, and the GB10 CLI and server smoke now describe the test image correctly.

## 1. Problem Statement

### 1.1 What mlxcel did

`prepare_inputs_for_multimodal` (`src/vision/merge.rs`) built its 4D mask as `attention_mask[k] * attention_mask[q]`. With the all-ones padding mask every request carries, that is an all-zero additive mask: every prompt position attends to every other, so text tokens read later prompt tokens and image tokens read the question that follows them. `Gemma3Model::hidden_with_caches_and_embeddings` then handed that one mask to every layer, so the sliding-window layers (window 1024 on the 4B checkpoint) also skipped the band the text path builds with `text_prefill_sliding_mask`. Both users of `MergeStrategy::Gemma3` took this path: Gemma 3 (`load_gemma3_vlm`) and PaliGemma (`load_paligemma_vlm`). The mask reached the model on every surface that prefills images: the scheduler's embedding prefill, `mlxcel generate`, and the logit trace.

### 1.2 What the references do

| Runtime | Full-attention layers | Sliding layers |
|---|---|---|
| transformers (`create_masks_for_vision_model`) | `causal OR same_image_block` | `sliding_window AND (causal OR same_image_block)` |
| mlx-vlm generation | all ones (bidirectional) | same mask |
| trace reference arm before this PR (`mask=None`) | causal | causal |
| mlxcel before this PR | bidirectional | bidirectional, no band |

An image block is a maximal run of image soft tokens (`image_token_id`); `<start_of_image>` and `<end_of_image>` are text. PaliGemma is different by design: transformers runs it as a prefix LM, so the whole prompt is bidirectional at prefill, and mlxcel's old mask was already right for it.

### 1.3 Why it mattered

The mask is part of the model's training contract. A text token that can see the rest of the prompt computes different hidden states than the checkpoint was trained to produce, and the effect showed in generation: before the fix `mlxcel generate --image` described `test_image_shapes.png` (a red square, a blue circle and a green triangle) as "a white square-like shape-dividing it into four equal parts". The trace comparison could not catch this, because its reference arm was causal, which is wrong in a different direction; agreement between two wrong masks would not have meant correctness either.

## 2. Technical Decisions

### 2.1 The caller names the mask; PaliGemma gets its own strategy

`prepare_inputs_for_multimodal` now takes a `MultimodalMask` (`Gemma3ImageBlocks` or `PrefixLm`), and `MergeStrategy` gains a `PaliGemma` variant that `load_paligemma_vlm` uses. `VisionModule::prepare_inputs` maps each strategy to its mask in two plain match arms.

Inferring the mask from the token ids (for example, "PaliGemma prompts have no `<start_of_image>`") would tie a correctness property to a token-layout coincidence that a future checkpoint could break. The merge itself (scale the image features by `1/sqrt(hidden_size)`, masked-scatter them in) is identical for both families, so a separate merge function would have duplicated it for one differing step. The cost is that `prepare_inputs_for_multimodal` now has eight arguments and an `#[allow(clippy::too_many_arguments)]`.

### 2.2 Image blocks numbered in the graph, per batch row

`image_block_ids` marks a run start where a position is an image token and the previous one is not, takes the inclusive cumulative sum of run starts along the sequence, and multiplies by the image indicator, so each position gets `n >= 1` for the n-th run in its row or `0` for text. `same_block` is then `query_block > 0 AND query_block == key_block`, and the mask is `key_valid AND (causal OR same_block)`, encoded as additive f32 (`0.0` / `f32::MIN`) in the same `[B, 1, S, S]` shape as before.

Everything stays on the device with no host sync, which matters because the mask is built on the scheduler's prefill path. Numbering per row keeps two batch rows from sharing a block id even when their runs start at different positions. Two images in one prompt are two blocks, so the first image cannot read the second. Only padded keys are masked; a left-padded query row ends up with every key masked, which the additive `f32::MIN` encoding keeps finite where `-inf` would give NaN.

### 2.3 The sliding band is applied in the model, not in the merge

`vision::merge` still returns one mask. In `Gemma3Model::hidden_with_caches_and_embeddings`, global layers (`i % pattern == pattern - 1`, the same rule as the text path and `make_caches`) take the external mask as is, and sliding layers take `external_mask_with_window_band(mask, window)`: the external mask with every key `window` or more positions behind its query masked (`q - k < window`, the convention of `create_sliding_window_prefill_mask`). The banded mask is built once per forward, and only when the key axis is wider than the window; otherwise the function returns `None` and the sliding layers reuse the external mask. Queries are aligned to the end of the key axis, and a boolean input mask stays boolean.

Returning two masks from the merge would have changed `InputEmbeddings` and every caller that threads the mask (scheduler prefill, `generate`, the trace), and would have put the model's layer pattern and window into the vision module. Keeping the band in the model keeps the merge's output contract unchanged and puts the window where the window is configured. Every `Gemma3Wrapper` entry that carries an external mask funnels into `hidden_with_caches_and_embeddings`, so one branch covers all of them.

### 2.4 transformers over mlx-vlm

mlx-vlm's generation path builds the same all-ones mask mlxcel had, so matching mlx-vlm was the alternative. The issue rejected it: #2750 already chose transformers and the checkpoint over mlx-vlm where they disagree, and Gemma 3 was trained under the transformers mask. The smoke output (section 4) is the practical check that this was the right reference.

### 2.5 The reference arm refuses what it cannot represent

`scripts/mlx_vlm_image_logit_trace.py` now builds the Gemma 3 mask in numpy (`gemma3_prefill_mask`, the same run-numbering idea) and passes it to mlx-vlm's language model instead of `mask=None`. mlx-vlm applies one passed mask to every layer, so the script cannot express the sliding band; it exits with an error when the sequence is longer than the window instead of silently dropping the band. The traced sequence is 335 tokens against a 1024-token window, so the band is a no-op there and the reference is exact. PaliGemma's reference stays causal (its prefix-LM mask is not reproduced in the script).

## 3. Change Summary

9 files, +615/-42: three commits (e78337f2 fix, 4215da12 docs, 42c1acae refactor) plus a merge of origin/main (694f5227).

- `src/vision/merge.rs`: `MultimodalMask`; `prepare_inputs_for_multimodal` takes it and calls the public `multimodal_attention_mask`, which builds the mask from the ids and padding mask (`attention_mask_for_image_runs`, `image_block_ids`).
- `src/vision/mod.rs`: `MergeStrategy::PaliGemma`; one dispatch arm per strategy; strategy docs name their real loaders.
- `src/loading/vlm_siglip.rs`: `load_paligemma_vlm` uses `MergeStrategy::PaliGemma`.
- `src/models/gemma3.rs`: `external_mask_with_window_band`; the external-mask branch splits global and sliding layers.
- `scripts/mlx_vlm_image_logit_trace.py`: `gemma3_prefill_mask`, the window guard, and the mask passed for Gemma 3.
- `docs/benchmarks.md`: the Gemma 3 trace row and the paragraph that names the cause.
- Tests, all checkpoint-free:
  - `src/vision/merge_tests.rs`: `gemma3_mask_is_causal_with_bidirectional_image_blocks` (the 13-token matrix from the issue), `gemma3_mask_without_image_tokens_is_exactly_causal`, `gemma3_mask_keeps_padding_keys_masked` (right and left padding), `gemma3_mask_finds_image_blocks_per_batch_row`; the old all-ones test is renamed `paligemma_mask_stays_prefix_bidirectional` and uses `PrefixLm`.
  - `src/models/gemma3_vlm_mask_tests.rs` (new): `external_gemma3_mask_matches_text_prefill_past_the_window` (tiny f32 Gemma 3, window 4, pattern 2, 12 tokens: the external-mask branch must equal the text prefill within 1e-4), `sliding_layers_intersect_the_image_mask_with_the_window`, `sliding_band_is_skipped_when_the_keys_fit_the_window`.

## 4. Results

Trace pair on `gemma-3-4b-it-4bit`, `tests/fixtures/test_image_shapes.png`, `--mlxcel-prompt`, float32 CPU reference, decided gap 2:

| mlxcel build | Reference mask | Top-1 diff | Decided diff | Perplexity mlxcel vs reference |
|---|---|---|---|---|
| main | causal (old) | 20 / 64 | 5 / 43 | 4.8500 vs 5.1620 |
| main | transformers | 18 / 64 | 7 / 48 | 4.8500 vs 6.0156 |
| this PR | transformers | 4 / 64 | 0 / 48 | 6.2651 vs 6.0156 |

The first row is carried over from the #2750 run (same prompt ids and targets; the old-reference rerun timed out in this run, and neither the main build nor the old script changed). Correcting the reference alone made main look worse (7 of 48), which confirms that the old 5 of 43 was two wrong masks partly agreeing. The 4 remaining top-1 differences are all positions where the reference's own gap is under 1 logit, with mlxcel's choice at the reference's rank 2.

Both perplexities rose. They score a fixed reply stream recorded before the fix, so the absolute value is not a quality signal here; the agreement between the arms is.

- PaliGemma (`paligemma2-3b-6bit`): trace TSV byte-identical before and after (`cmp`), greedy `mlxcel generate --image` text identical.
- GB10 smoke, CLI and `mlxcel-server` (one image chat request), same answer: "three geometric shapes arranged vertically ... Top: A square, colored red. Middle: A circle, colored blue. Bottom: A triangle, colored green."
- Failing first: with the fix logic reverted, 6 of the new tests fail, including the tiny model (`row 0 logit 0: external mask 0.5711894, text prefill 0.6478121`). With only the `gemma3.rs` layer split reverted, the tiny model alone fails at the first row past the window (`row 4 logit 0: external mask 0.76741546, text prefill 0.72449356`), so the layer split has its own failing test independent of the merge tests.
- `vision::merge` 9 passed, `models::gemma3` 49 passed (11 ignored), `multimodal::` 318 passed (CUDA, `--test-threads=1`); fmt and clippy `-D warnings` clean.

## 5. Trade-offs and Residuals

- **Memory past the window.** For an image prompt longer than the sliding window, the sliding layers hold a second `[B, 1, S, S]` f32 mask, so persistent mask memory goes from 4·S² to 8·S² bytes per batch row, plus a transient int32 difference and bool band while it is built. It is built once per forward, not per layer, and stays below the per-layer `[B, H, S, S]` score buffer of masked SDPA. Prompts at or under the window allocate nothing extra.
- **The band is not verified on a real checkpoint against a reference.** The reference arm refuses sequences longer than the window, and the traced prompt is 335 tokens. The band is covered by the tiny-model test and the matrix test only.
- **OpenXLA Gemma 3 path.** `src/lib/mlxcel-xla/src/prepared_gemma3.rs` (`GEMMA3_VLM_MASK_MODE = "gemma3-vlm-bidirectional-padding-f32-min-v1"`) still mirrors the old bidirectional mask on every layer. The PR marks this for a follow-up issue; none was filed at the time of writing. Until then, the MLX and XLA backends give different Gemma 3 image outputs.
- **PaliGemma's reference arm** remains causal, while mlxcel runs PaliGemma as a prefix LM; PaliGemma trace numbers still compare different masks. Unchanged by this PR.
- **Not exercised:** a real-checkpoint request with two images (two blocks are unit-tested only), Gemma3n (its own path), and the non-decode `forward_batched_with_ids` path, which calls `forward_with_sequence_id` and takes no mask (no VLM caller reaches it with one).
- **Output change.** The same Gemma 3 image request produces different text than earlier releases.

## 6. Learning Points

- A parity harness checks agreement, not correctness. Here both arms were wrong in different directions (bidirectional vs causal), and fixing only the reference made the system under test look worse before it looked better. When a residual gap survives identical inputs, check what each arm is configured to compute before blaming the kernels.
- Name masking semantics explicitly at the call site. Two families sharing a merge routine with different attention contracts (prefix LM vs causal with image blocks) is a place where a shared default silently becomes wrong for one of them.
- Maximal runs can be numbered on the device with a shift, a run-start indicator and a cumulative sum; no host loop or sync is needed, and the numbering stays per row.
- Layer-dependent masks belong where the layer pattern lives. A merge that returns one mask plus a model that derives the sliding variant keeps the window configuration in one place and avoids widening every caller's interface.
- A reference script should refuse an input it cannot model faithfully rather than approximate it silently; the window guard makes the trace's exactness condition explicit.
