// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Token merging for vision-language models
//!
//! Merges vision encoder output features with text token embeddings
//! at image token positions.
//!
//! Strategies and current family users:
//! - `prepare_inputs_for_multimodal()`
//!   - Gemma3 and PaliGemma, which need additive 4D attention masks; the
//!     [`MultimodalMask`] argument picks Gemma3's causal image-block mask or
//!     PaliGemma's prefix-LM mask
//! - `merge_llava()` / `merge_llava_any()`
//!   - LLaVA/Bunny, Aya Vision, Pixtral/Mistral3, Gemma3n, Qwen-VL,
//!     Phi3V, Molmo2, and similar token-replacement paths
//!
//! Contract:
//! - merge helpers keep output embeddings in the text-model dtype
//! - Gemma-style merge returns an additive f32 4D mask
//!   (`0.0` at attended positions, `f32::MIN` at masked positions)
//! - LLaVA-style merge keeps standard causal masking and returns `None`
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/gemma3.py#L121-L168
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llava/llava.py#L85-L111

use mlxcel_core::{MlxArray, UniquePtr};

/// Which prefill attention mask [`prepare_inputs_for_multimodal`] builds.
///
/// The two families share the merge and differ only in the mask, so the
/// caller names it rather than having it inferred from the token ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultimodalMask {
    /// Gemma3: causal attention, plus bidirectional attention inside each
    /// image block (a maximal run of image soft tokens; `<start_of_image>`
    /// and `<end_of_image>` are text). A block never sees a later block.
    /// This is transformers' `create_masks_for_vision_model` for the
    /// full-attention layers; the sliding layers intersect it with their
    /// window in the model (`gemma3::external_mask_with_window_band`).
    Gemma3ImageBlocks,
    /// PaliGemma: prefix LM, so the whole prompt (image and text) attends
    /// bidirectionally at prefill; only padding is masked.
    PrefixLm,
}

/// Merged embeddings ready for text model
pub struct InputEmbeddings {
    pub inputs_embeds: UniquePtr<MlxArray>,
    pub attention_mask_4d: Option<UniquePtr<MlxArray>>,
}

/// Prepare multimodal inputs by merging text and vision embeddings
///
/// Port of Gemma3 Model.prepare_inputs_for_multimodal()
///
/// # Arguments
/// * `hidden_size` - Text model hidden size (for scaling vision features)
/// * `pad_token_id` - Padding token ID
/// * `image_token_index` - Token ID for image placeholder tokens
/// * `mask` - Which prefill attention mask to build ([`MultimodalMask`])
/// * `image_features` - Vision encoder output [batch, num_vision_tokens, embed_dim]
/// * `inputs_embeds` - Text token embeddings [batch, seq_len, embed_dim]
/// * `input_ids` - Token IDs [batch, seq_len]
/// * `attention_mask` - Attention mask [batch, seq_len]
///
/// # Returns
/// * `inputs_embeds` - Merged embeddings in the text-model dtype
/// * `attention_mask_4d` - Additive f32 4D mask of shape [batch, 1, seq_len, seq_len]
///   with `0.0` at attended positions and `f32::MIN` at masked positions. This is
///   consumable by `mx.fast.scaled_dot_product_attention`, which treats non-bool
///   masks as additive bias on pre-softmax scores.
///
/// Used by: Gemma3 VLM, PaliGemma
#[allow(clippy::too_many_arguments)]
pub fn prepare_inputs_for_multimodal(
    hidden_size: usize,
    pad_token_id: i32,
    image_token_index: i32,
    mask: MultimodalMask,
    image_features: &MlxArray,
    inputs_embeds: &MlxArray,
    input_ids: &MlxArray,
    attention_mask: &MlxArray,
) -> InputEmbeddings {
    let feat_shape = mlxcel_core::array_shape(image_features);
    let embed_dim = feat_shape[2];

    let ids_shape = mlxcel_core::array_shape(input_ids);
    let batch_size = ids_shape[0];
    let sequence_length = ids_shape[1];

    // Scale image features by 1/sqrt(hidden_size), matching Python mlx-vlm behavior.
    // Python: scaled_image_features = image_features / (config.hidden_size ** 0.5)
    // The language model's forward then multiplies ALL embeddings by sqrt(hidden_size),
    // so image features end up at approximately their original scale.
    // Used by: Gemma3 VLM, PaliGemma VLM
    let scale = 1.0 / (hidden_size as f32).sqrt();
    let scaled_image_features = mlxcel_core::multiply_scalar(image_features, scale);

    // Create zero embedding buffer
    let mut final_embedding = mlxcel_core::zeros(
        &[batch_size, sequence_length, embed_dim],
        mlxcel_core::dtype::FLOAT32,
    );

    // Create masks from input_ids
    let image_token_arr =
        mlxcel_core::full_f32(&[1], image_token_index as f32, mlxcel_core::dtype::INT32);
    let image_token_arr = mlxcel_core::astype(&image_token_arr, mlxcel_core::dtype::INT32);
    let pad_token_arr = mlxcel_core::full_f32(&[1], pad_token_id as f32, mlxcel_core::dtype::INT32);
    let pad_token_arr = mlxcel_core::astype(&pad_token_arr, mlxcel_core::dtype::INT32);

    let is_image = mlxcel_core::equal(input_ids, &image_token_arr);
    let is_pad = mlxcel_core::equal(input_ids, &pad_token_arr);
    let not_image = mlxcel_core::logical_not(&is_image);
    let not_pad = mlxcel_core::logical_not(&is_pad);
    let text_mask = mlxcel_core::logical_and(&not_image, &not_pad);

    // Expand masks to embedding dimension: [batch, seq_len] -> [batch, seq_len, embed_dim]
    let text_mask_expanded = mlxcel_core::expand_dims(&text_mask, -1);
    let text_mask_expanded = mlxcel_core::repeat(&text_mask_expanded, embed_dim, -1);
    let pad_mask_expanded = mlxcel_core::expand_dims(&is_pad, -1);
    let pad_mask_expanded = mlxcel_core::repeat(&pad_mask_expanded, embed_dim, -1);
    let image_mask_expanded = mlxcel_core::expand_dims(&is_image, -1);
    let image_mask_expanded = mlxcel_core::repeat(&image_mask_expanded, embed_dim, -1);

    // Place text embeddings at text positions
    final_embedding = mlxcel_core::where_cond(&text_mask_expanded, inputs_embeds, &final_embedding);

    // Zero out pad positions
    let zeros = mlxcel_core::zeros_like(&final_embedding);
    final_embedding = mlxcel_core::where_cond(&pad_mask_expanded, &zeros, &final_embedding);

    // Place scaled vision features at image positions via masked_scatter
    final_embedding = masked_scatter(
        &final_embedding,
        &image_mask_expanded,
        &scaled_image_features,
    );

    let attn_mask_4d =
        multimodal_attention_mask(mask, image_token_index, input_ids, attention_mask);

    // Cast final embedding to same dtype as inputs_embeds
    let dtype = mlxcel_core::array_dtype(inputs_embeds);
    let final_embedding = mlxcel_core::astype(&final_embedding, dtype);

    InputEmbeddings {
        inputs_embeds: final_embedding,
        attention_mask_4d: Some(attn_mask_4d),
    }
}

/// The additive `[B, 1, S, S]` prefill mask [`prepare_inputs_for_multimodal`]
/// returns, built from the token ids alone.
///
/// `attention_mask` is the `[B, S]` 0/1 padding mask. The result is f32 with
/// `0.0` at attended positions and `f32::MIN` at masked ones.
///
/// Used by: [`prepare_inputs_for_multimodal`], and tests that need the Gemma3
/// mask without vision features.
pub fn multimodal_attention_mask(
    mask: MultimodalMask,
    image_token_index: i32,
    input_ids: &MlxArray,
    attention_mask: &MlxArray,
) -> UniquePtr<MlxArray> {
    let image_token = mlxcel_core::from_slice_i32(&[image_token_index], &[1]);
    let is_image = mlxcel_core::equal(input_ids, &image_token);
    attention_mask_for_image_runs(mask, &is_image, attention_mask)
}

/// Build the additive mask from the `[B, S]` image-token positions and the
/// `[B, S]` padding mask.
///
/// Additive rather than multiplicative: `mx.fast.scaled_dot_product_attention`
/// adds a non-bool mask to the pre-softmax scores, so `f32::MIN` drops a key
/// while `0.0` leaves it alone. A 0/1 mask would leak padding into the
/// attention distribution.
fn attention_mask_for_image_runs(
    mask: MultimodalMask,
    is_image: &MlxArray,
    attention_mask: &MlxArray,
) -> UniquePtr<MlxArray> {
    let zero_i32 = mlxcel_core::from_slice_i32(&[0], &[1]);
    let key_valid = mlxcel_core::not_equal(attention_mask, &zero_i32); // [B, S] bool
    let allowed = match mask {
        // Prefix LM: every valid query sees every valid key.
        MultimodalMask::PrefixLm => mlxcel_core::logical_and(
            &mlxcel_core::expand_dims(&key_valid, 1), // [B, 1, S]
            &mlxcel_core::expand_dims(&key_valid, 2), // [B, S, 1]
        ),
        MultimodalMask::Gemma3ImageBlocks => {
            let seq_len = mlxcel_core::array_shape(is_image)[1];
            // k <= q, as a [1, S, S] bool (rows are queries, columns keys).
            let positions = mlxcel_core::arange_i32(0, seq_len, 1);
            let causal = mlxcel_core::less_equal(
                &mlxcel_core::reshape(&positions, &[1, 1, seq_len]),
                &mlxcel_core::reshape(&positions, &[1, seq_len, 1]),
            );
            // Same image block: both positions in the same run, a run id of
            // 0 meaning text.
            let blocks = image_block_ids(is_image);
            let query_block = mlxcel_core::expand_dims(&blocks, 2); // [B, S, 1]
            let key_block = mlxcel_core::expand_dims(&blocks, 1); // [B, 1, S]
            let same_block = mlxcel_core::logical_and(
                &mlxcel_core::greater(&query_block, &zero_i32),
                &mlxcel_core::equal(&query_block, &key_block),
            );
            mlxcel_core::logical_and(
                &mlxcel_core::expand_dims(&key_valid, 1),
                &mlxcel_core::logical_or(&causal, &same_block),
            )
        }
    };
    let zero_f32 = mlxcel_core::from_slice_f32(&[0.0], &[1]);
    let masked = mlxcel_core::from_slice_f32(&[f32::MIN], &[1]);
    let additive = mlxcel_core::where_cond(&allowed, &zero_f32, &masked); // [B, S, S]
    mlxcel_core::expand_dims(&additive, 1) // [B, 1, S, S]
}

/// Per-position image block id, `[B, S]` int32: `n >= 1` for a position in
/// the `n`-th maximal run of image tokens of its row, `0` for text.
///
/// A run starts where a position is an image token and the one before it is
/// not; the inclusive cumulative sum of run starts along the sequence numbers
/// the runs. Per row, so batch rows never share block ids.
fn image_block_ids(is_image: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(is_image);
    let (batch, seq_len) = (shape[0], shape[1]);
    let image = mlxcel_core::astype(is_image, mlxcel_core::dtype::INT32);
    let first = mlxcel_core::zeros(&[batch, 1], mlxcel_core::dtype::INT32);
    let previous = if seq_len > 1 {
        let shifted = mlxcel_core::slice(&image, &[0, 0], &[batch, seq_len - 1]);
        mlxcel_core::concatenate(&first, &shifted, 1)
    } else {
        first
    };
    let one = mlxcel_core::from_slice_i32(&[1], &[1]);
    let run_start = mlxcel_core::multiply(&image, &mlxcel_core::subtract(&one, &previous));
    let run_number = mlxcel_core::cumsum(&run_start, 1, false, true);
    mlxcel_core::multiply(&run_number, &image)
}

/// Merge image features with text embeddings for LLaVA
///
/// Port of LLaVA _merge_input_ids_with_image_features (llava.py:85-111)
///
/// Simpler than Gemma3: directly replaces image token positions with projected features.
/// Uses standard causal masking (no special 4D attention mask needed).
///
/// # Arguments
/// * `image_token_index` - Token ID for image placeholder tokens
/// * `image_features` - Projected vision features [num_images, num_patches, hidden]
/// * `inputs_embeds` - Text token embeddings [batch, seq_len, embed_dim]
/// * `input_ids` - Token IDs [batch, seq_len]
///
/// Used by: LLaVA/Bunny, Aya Vision, Pixtral, Mistral3, Gemma3n,
/// Qwen2/2.5/3/3.5-VL, Qwen3.8-Flash-Next, Phi3V, Molmo2, Llama4, Nemotron H Nano Omni
/// (vision and audio modalities), and Kimi K3, which wraps this call in
/// `vision::kimi_k3_vl::merge_media_features` so a placeholder count that
/// disagrees with the feature rows is an error rather than a silent
/// misalignment.
pub fn merge_llava(
    image_token_index: i32,
    image_features: &MlxArray,
    inputs_embeds: &MlxArray,
    input_ids: &MlxArray,
) -> InputEmbeddings {
    let feat_shape = mlxcel_core::array_shape(image_features);
    let embed_dim = feat_shape[feat_shape.len() - 1];

    // Flatten image features: [N_images, num_patches, hidden] -> [total_patches, hidden]
    let total_patches = feat_shape
        .iter()
        .take(feat_shape.len() - 1)
        .product::<i32>();
    let flat_features = mlxcel_core::reshape(image_features, &[total_patches, embed_dim]);

    // Cast image features to same dtype as text embeddings
    let embed_dtype = mlxcel_core::array_dtype(inputs_embeds);
    let flat_features = mlxcel_core::astype(&flat_features, embed_dtype);

    // Find image token positions and scatter features

    // Create image token comparison value
    let image_token_arr =
        mlxcel_core::full_f32(&[1], image_token_index as f32, mlxcel_core::dtype::INT32);
    let image_token_arr = mlxcel_core::astype(&image_token_arr, mlxcel_core::dtype::INT32);
    let is_image = mlxcel_core::equal(input_ids, &image_token_arr);

    // Expand mask to embedding dimension: [batch, seq_len] -> [batch, seq_len, embed_dim]
    let image_mask_expanded = mlxcel_core::expand_dims(&is_image, -1);
    let image_mask_expanded = mlxcel_core::repeat(&image_mask_expanded, embed_dim, -1);

    // Use masked_scatter to place features at image token positions
    let final_embedding = masked_scatter(inputs_embeds, &image_mask_expanded, &flat_features);

    InputEmbeddings {
        inputs_embeds: final_embedding,
        attention_mask_4d: None, // LLaVA uses standard causal masking
    }
}

/// Merge vision features into any of several visual placeholder token IDs.
///
/// Qwen-VL uses distinct `<|image_pad|>` and `<|video_pad|>` tokens, but the
/// projected feature tensor is one concatenated media stream in prompt order.
/// The scatter mask therefore has to be the union of both token IDs; using only
/// the image token silently leaves video features unmerged.
pub fn merge_llava_any(
    visual_token_ids: &[i32],
    image_features: &MlxArray,
    inputs_embeds: &MlxArray,
    input_ids: &MlxArray,
) -> InputEmbeddings {
    if visual_token_ids.is_empty() {
        return InputEmbeddings {
            inputs_embeds: mlxcel_core::copy(inputs_embeds),
            attention_mask_4d: None,
        };
    }

    let feat_shape = mlxcel_core::array_shape(image_features);
    let embed_dim = feat_shape[feat_shape.len() - 1];
    let total_patches = feat_shape
        .iter()
        .take(feat_shape.len() - 1)
        .product::<i32>();
    let flat_features = mlxcel_core::reshape(image_features, &[total_patches, embed_dim]);
    let embed_dtype = mlxcel_core::array_dtype(inputs_embeds);
    let flat_features = mlxcel_core::astype(&flat_features, embed_dtype);

    let mut is_visual = {
        let token_arr =
            mlxcel_core::full_f32(&[1], visual_token_ids[0] as f32, mlxcel_core::dtype::INT32);
        let token_arr = mlxcel_core::astype(&token_arr, mlxcel_core::dtype::INT32);
        mlxcel_core::equal(input_ids, &token_arr)
    };
    for &token_id in &visual_token_ids[1..] {
        let token_arr = mlxcel_core::full_f32(&[1], token_id as f32, mlxcel_core::dtype::INT32);
        let token_arr = mlxcel_core::astype(&token_arr, mlxcel_core::dtype::INT32);
        let mask = mlxcel_core::equal(input_ids, &token_arr);
        is_visual = mlxcel_core::logical_or(&is_visual, &mask);
    }

    let visual_mask_expanded = mlxcel_core::expand_dims(&is_visual, -1);
    let visual_mask_expanded = mlxcel_core::repeat(&visual_mask_expanded, embed_dim, -1);
    let final_embedding = masked_scatter(inputs_embeds, &visual_mask_expanded, &flat_features);

    InputEmbeddings {
        inputs_embeds: final_embedding,
        attention_mask_4d: None,
    }
}

/// Scatter scaled image features into image token positions
///
/// Port of masked_scatter from gemma3.py:54-72
///
/// Used by: all merge helpers in this module
fn masked_scatter(
    final_embedding: &MlxArray,
    image_mask_expanded: &MlxArray,
    scaled_image_features: &MlxArray,
) -> UniquePtr<MlxArray> {
    let final_shape = mlxcel_core::array_shape(final_embedding);

    // Flatten everything to 1D
    let features_flat = mlxcel_core::flatten(scaled_image_features);
    let embedding_flat = mlxcel_core::flatten(final_embedding);
    let mask_flat = mlxcel_core::flatten(image_mask_expanded);

    // Use where_cond to place features where mask is true
    // This works because the flattened image features align with the true positions
    // in the flattened mask, but only if num_true_positions == num_feature_elements.
    //
    // However, where_cond requires same shape. The image features may have fewer
    // elements than the full embedding. We need to expand features to full size
    // and use the mask to select.
    //
    // Alternative approach: build a full-size array where mask positions get feature values.
    // We use cumsum on the mask to create indices into the features array.
    let mask_i32 = mlxcel_core::astype(&mask_flat, mlxcel_core::dtype::INT32);
    let cumsum = mlxcel_core::cumsum(&mask_i32, 0, false, true); // inclusive cumsum
    // cumsum[i] = number of true values at or before position i
    // For mask positions: cumsum[i] - 1 = index into features_flat

    // Subtract 1 to get 0-based indices, clamp to valid range
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::INT32);
    let one = mlxcel_core::astype(&one, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::subtract(&cumsum, &one);
    // Clamp negative values to 0
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::INT32);
    let zero = mlxcel_core::astype(&zero, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::maximum(&indices, &zero);

    // Gather from features using indices
    let gathered = mlxcel_core::take(&features_flat, &indices, 0);

    // Use mask to select: where mask is true, use gathered features; else use original
    let result = mlxcel_core::where_cond(&mask_flat, &gathered, &embedding_flat);

    // Reshape back
    mlxcel_core::reshape(&result, &final_shape)
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod tests;
