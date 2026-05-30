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

//! Gemma 4 multimodal embedder (vision and audio share the same shape).
//!
//! Pulled out of `gemma4_vl.rs` so that file stays under the project's
//! 500-line hard cap (added per-sequence `per_layer_inputs` plumbing alongside the existing VLM/audio code paths).
//!
//! Both `embed_vision` and `embed_audio` on
//! [`crate::vision::Gemma4VLModel`] use this module's
//! [`Gemma4MultimodalEmbedder`]: the encoder output is normalized with
//! the unscaled `RMSNormNoScale` and then projected into the language
//! model's hidden space via a `UnifiedLinear`. Upstream mlx-vlm reordered
//! the norm to run *before* the projection (renaming the field from
//! `embedding_post_projection_norm` to `embedding_pre_projection_norm`);
//! this implementation matches that ordering.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::{MlxArray, UniquePtr};

/// Vision/audio feature embedder used by Gemma 4 VLM.
///
/// Used by: [`crate::vision::Gemma4VLModel::embed_vision`] and
/// [`crate::vision::Gemma4VLModel::embed_audio`].
pub struct Gemma4MultimodalEmbedder {
    embedding_projection: UnifiedLinear,
    pre_projection_norm: crate::models::gemma4::RMSNormNoScale,
}

impl Gemma4MultimodalEmbedder {
    /// Construct the Gemma 4 multimodal embedder.
    ///
    /// `embedding_dim` is the input feature dimension (the vision or audio
    /// encoder output size) — i.e. the column count of the projection weight
    /// `embedding_projection.weight`. The RMS norm is applied on this dim
    /// BEFORE the projection, matching upstream mlx-vlm after the reordering
    /// that renamed `embedding_post_projection_norm` to
    /// `embedding_pre_projection_norm`.
    pub fn from_weights(
        weights: &mlxcel_core::weights::WeightMap,
        prefix: &str,
        embedding_dim: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            embedding_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.embedding_projection"),
                group_size,
                bits,
            )?,
            pre_projection_norm: crate::models::gemma4::RMSNormNoScale::new(
                embedding_dim as i32,
                eps,
            ),
        })
    }

    pub fn forward(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.pre_projection_norm.forward(inputs_embeds);
        self.embedding_projection.forward(&normed)
    }
}

/// Scatter audio encodings into the audio token positions of an
/// embedding tensor.
///
/// Equivalent to Python's `masked_scatter`: flattens the mask, computes
/// cumulative indices, aligns the source, and gates with `where_cond`.
/// Used by `Gemma4VLModel::get_input_embeddings_with_audio_and_cache`
/// during the audio-features merge step.
///
/// Shares the algorithm with [`crate::vision::merge`]'s `masked_scatter`
/// helper but operates with the rank/dtype assumptions specific to the
/// audio merge (which expects mask and source to already be aligned to
/// the embedding shape).
pub(crate) fn masked_scatter(
    input: &MlxArray,
    mask: &MlxArray,
    source: &MlxArray,
) -> UniquePtr<MlxArray> {
    let input_shape = mlxcel_core::array_shape(input);
    let total_size: i32 = input_shape.iter().product();

    let mask_flat = mlxcel_core::reshape(mask, &[total_size]);
    let mask_i32 = mlxcel_core::astype(&mask_flat, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::cumsum(&mask_i32, 0, false, true);
    let ones = mlxcel_core::ones(&[1], mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::subtract(&indices, &ones);

    let source_flat_size: i32 = mlxcel_core::array_shape(source).iter().product();
    let source_size_arr = mlxcel_core::from_slice_i32(&[source_flat_size], &[1]);
    let safe_indices = mlxcel_core::remainder(&indices, &source_size_arr);

    let source_flat = mlxcel_core::reshape(source, &[source_flat_size]);
    let aligned = mlxcel_core::take(&source_flat, &safe_indices, 0);

    let input_flat = mlxcel_core::reshape(input, &[total_size]);
    let result = mlxcel_core::where_cond(&mask_flat, &aligned, &input_flat);
    mlxcel_core::reshape(&result, &input_shape)
}
