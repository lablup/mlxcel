// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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
//!   - Gemma3 and PaliGemma-style paths that need additive 4D attention masks
//! - `merge_llava()`
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
pub fn prepare_inputs_for_multimodal(
    hidden_size: usize,
    pad_token_id: i32,
    image_token_index: i32,
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
    // Used by: PaliGemma VLM
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

    // Build additive 4D attention mask: f32 with 0.0 at attended positions
    // and f32::MIN at masked positions. mx.fast.scaled_dot_product_attention
    // treats non-bool masks as additive bias on pre-softmax scores, so
    // masked positions collapse to ~0 after softmax while attended positions
    // are unchanged. This is the correct semantics for padded sequences; a
    // multiplicative 0/1 mask would silently leak padding tokens into the
    // attention distribution.
    // Used by: PaliGemma (Gemma2 backbone), Gemma3 VLM
    let attn_mask_1 = mlxcel_core::expand_dims(attention_mask, 1); // [B, 1, S]
    let attn_mask_2 = mlxcel_core::expand_dims(attention_mask, 2); // [B, S, 1]
    let attn_product = mlxcel_core::multiply(&attn_mask_1, &attn_mask_2); // [B, S, S] int32 0/1
    let zero_i32 = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::INT32);
    let zero_i32 = mlxcel_core::astype(&zero_i32, mlxcel_core::dtype::INT32);
    let is_masked = mlxcel_core::equal(&attn_product, &zero_i32); // [B, S, S] bool
    let zero_f32 = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::MIN, mlxcel_core::dtype::FLOAT32);
    let additive = mlxcel_core::where_cond(&is_masked, &neg_inf, &zero_f32);
    let attn_mask_4d = mlxcel_core::expand_dims(&additive, 1); // [B, 1, S, S]

    // Cast final embedding to same dtype as inputs_embeds
    let dtype = mlxcel_core::array_dtype(inputs_embeds);
    let final_embedding = mlxcel_core::astype(&final_embedding, dtype);

    InputEmbeddings {
        inputs_embeds: final_embedding,
        attention_mask_4d: Some(attn_mask_4d),
    }
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
/// Qwen2/2.5/3/3.5-VL, Phi3V, Molmo2, Llama4, Nemotron H Nano Omni
/// (vision + audio modalities, and)
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

    // Cast to input dtype
    let final_embedding = mlxcel_core::astype(&final_embedding, embed_dtype);

    InputEmbeddings {
        inputs_embeds: final_embedding,
        attention_mask_4d: None, // LLaVA uses standard causal masking
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
