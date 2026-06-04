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

//! Gemma 4 Unified encoder-free vision patch embedder.
//!
//! Unlike the ViT-backed `gemma4` VLM (see
//! [`crate::vision::encoders::gemma4::Gemma4VisionModel`]), the
//! `gemma4_unified` architecture has **no** vision transformer. Images are
//! turned into soft tokens by a small patch projector:
//!
//! 1. `patch_ln1` — LayerNorm over the flat patch dim (`model_patch_size² · 3`,
//!    e.g. `48·48·3 = 6912`).
//! 2. `patch_dense` — Linear (`patch_dim → mm_embed_dim`).
//! 3. `patch_ln2` — LayerNorm over `mm_embed_dim`.
//! 4. learned 2-D positional embedding `pos_embedding`
//!    (`[mm_posemb_size, 2, mm_embed_dim]`) added per patch, indexed by the
//!    patch's `(x, y)` grid position (x → slot 0, y → slot 1). Padding patches
//!    carry position `-1` and contribute zero.
//! 5. `pos_norm` — LayerNorm over `mm_embed_dim`.
//!
//! The resulting per-patch features (at `output_proj_dims == mm_embed_dim`) are
//! then projected into the language-model hidden space by the shared
//! [`crate::vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder`]
//! (`embed_vision`), exactly as the audio feature path uses `embed_audio`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::gemma4_unified_config::Gemma4UnifiedVisionConfig;

/// Encoder-free Gemma 4 Unified vision patch embedder (`vision_embedder.*`).
pub struct Gemma4UnifiedVisionEmbedder {
    patch_ln1: LayerNorm,
    patch_dense: UnifiedLinear,
    patch_ln2: LayerNorm,
    /// Learned 2-D positional embedding of shape
    /// `[mm_posemb_size, 2, mm_embed_dim]`.
    pos_embedding: UniquePtr<MlxArray>,
    pos_norm: LayerNorm,
    mm_embed_dim: i32,
    mm_posemb_size: i32,
}

/// Build a [`LayerNorm`] (weight + bias) from a weight-map prefix.
fn layer_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

impl Gemma4UnifiedVisionEmbedder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma4UnifiedVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let eps = config.rms_norm_eps;
        let patch_ln1 = layer_norm_from_weights(weights, &format!("{prefix}.patch_ln1"), eps)?;
        let patch_dense = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.patch_dense"),
            group_size,
            bits,
        )?;
        let patch_ln2 = layer_norm_from_weights(weights, &format!("{prefix}.patch_ln2"), eps)?;
        let pos_embedding = weights
            .get(&format!("{prefix}.pos_embedding"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.pos_embedding"))?;
        let pos_norm = layer_norm_from_weights(weights, &format!("{prefix}.pos_norm"), eps)?;

        Ok(Self {
            patch_ln1,
            patch_dense,
            patch_ln2,
            pos_embedding,
            pos_norm,
            mm_embed_dim: config.mm_embed_dim as i32,
            mm_posemb_size: config.mm_posemb_size as i32,
        })
    }

    /// Project a single image's patches into per-patch vision features.
    ///
    /// * `patches` — `[num_soft_tokens, patch_dim]` float32, where
    ///   `patch_dim == model_patch_size² · 3`. Padding rows may be zeros.
    /// * `positions` — `[num_soft_tokens, 2]` int32 `(x, y)` grid coordinates.
    ///   Padding patches use `-1` on both axes and contribute zero positional
    ///   embedding.
    ///
    /// Returns `[num_soft_tokens, mm_embed_dim]` (== `output_proj_dims`).
    pub fn forward(&self, patches: &MlxArray, positions: &MlxArray) -> UniquePtr<MlxArray> {
        // Run the patch path in the model's native dtype (the dtype of the
        // learned weights, e.g. bf16/f16) so the positional-embedding `add`
        // below does not mix f32 patches with bf16 tables.
        let native = mlxcel_core::array_dtype(&self.pos_embedding);
        let patches = mlxcel_core::astype(patches, native);

        // 1. patch_ln1 over patch_dim.
        let h = self.patch_ln1.forward(&patches);
        // 2. patch_dense: patch_dim -> mm_embed_dim.
        let h = self.patch_dense.forward(&h);
        let h = mlxcel_core::astype(&h, native);
        // 3. patch_ln2 over mm_embed_dim.
        let h = self.patch_ln2.forward(&h);

        // 4. Add learned 2-D positional embeddings indexed by (x, y).
        let pos = self.gather_position_embeddings(positions);
        let h = mlxcel_core::add(&h, &pos);

        // 5. pos_norm over mm_embed_dim.
        self.pos_norm.forward(&h)
    }

    /// Gather and sum the per-patch positional embedding.
    ///
    /// `pos_embedding[x, 0, :]` is the x-axis embedding and
    /// `pos_embedding[y, 1, :]` is the y-axis embedding. `-1` positions
    /// contribute zero. Returns `[num_patches, mm_embed_dim]`.
    fn gather_position_embeddings(&self, positions: &MlxArray) -> UniquePtr<MlxArray> {
        let pos_i32 = mlxcel_core::astype(positions, mlxcel_core::dtype::INT32);
        let num_patches = mlxcel_core::array_shape(&pos_i32)[0];

        // Split into x and y index vectors of shape [num_patches].
        let x_ids = mlxcel_core::reshape(
            &mlxcel_core::slice(&pos_i32, &[0, 0], &[num_patches, 1]),
            &[num_patches],
        );
        let y_ids = mlxcel_core::reshape(
            &mlxcel_core::slice(&pos_i32, &[0, 1], &[num_patches, 2]),
            &[num_patches],
        );

        // Validity masks: positions >= 0 contribute, -1 contributes zero.
        let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
        let x_valid = mlxcel_core::greater_equal(&x_ids, &zero);
        let y_valid = mlxcel_core::greater_equal(&y_ids, &zero);

        // Clamp -1 to 0 before gathering so the take stays in bounds; the
        // validity mask zeroes those rows afterwards.
        let x_safe = mlxcel_core::maximum(&x_ids, &zero);
        let y_safe = mlxcel_core::maximum(&y_ids, &zero);

        // pos_embedding: [mm_posemb_size, 2, mm_embed_dim]. Slice axis-1
        // slots 0 (x) and 1 (y) to [mm_posemb_size, mm_embed_dim].
        let x_table = mlxcel_core::reshape(
            &mlxcel_core::slice(
                &self.pos_embedding,
                &[0, 0, 0],
                &[self.mm_posemb_size, 1, self.mm_embed_dim],
            ),
            &[self.mm_posemb_size, self.mm_embed_dim],
        );
        let y_table = mlxcel_core::reshape(
            &mlxcel_core::slice(
                &self.pos_embedding,
                &[0, 1, 0],
                &[self.mm_posemb_size, 2, self.mm_embed_dim],
            ),
            &[self.mm_posemb_size, self.mm_embed_dim],
        );

        let x_emb = mlxcel_core::take(&x_table, &x_safe, 0); // [num_patches, mm_embed_dim]
        let y_emb = mlxcel_core::take(&y_table, &y_safe, 0);

        // Zero out invalid (padding) rows.
        let emb_dtype = mlxcel_core::array_dtype(&self.pos_embedding);
        let x_mask = mlxcel_core::expand_dims(&x_valid, -1);
        let x_mask = mlxcel_core::broadcast_to(&x_mask, &[num_patches, self.mm_embed_dim]);
        let y_mask = mlxcel_core::expand_dims(&y_valid, -1);
        let y_mask = mlxcel_core::broadcast_to(&y_mask, &[num_patches, self.mm_embed_dim]);
        let zeros = mlxcel_core::zeros(&[num_patches, self.mm_embed_dim], emb_dtype);

        let x_emb = mlxcel_core::where_cond(&x_mask, &x_emb, &zeros);
        let y_emb = mlxcel_core::where_cond(&y_mask, &y_emb, &zeros);

        mlxcel_core::add(&x_emb, &y_emb)
    }
}
