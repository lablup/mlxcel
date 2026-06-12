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

//! Gemma3 AvgPool Multi-Modal Projector
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/gemma3.py#L13-L51
//!
//! Architecture:
//! 1. Reshape + transpose vision features to spatial grid
//! 2. Average pool to reduce spatial resolution
//! 3. Flatten back to sequence
//! 4. GemmaRMSNorm
//! 5. Matrix multiply with projection weight (einsum "btm,md->btd" == matmul)

use super::MultiModalConnector;
use mlxcel_core::layers::GemmaRMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct AvgPoolProjector {
    mm_input_projection_weight: UniquePtr<MlxArray>,
    mm_soft_emb_norm: GemmaRMSNorm,
    patches_per_image: usize,
    tokens_per_side: usize,
    kernel_size: i32,
}

impl AvgPoolProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vision_hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        mm_tokens_per_image: usize,
        layer_norm_eps: f32,
    ) -> Result<Self, String> {
        let proj_key = format!("{}.mm_input_projection_weight", prefix);
        let norm_key = format!("{}.mm_soft_emb_norm.weight", prefix);

        let mm_input_projection_weight = weights
            .get(&proj_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", proj_key))?;

        let norm_weight = weights
            .get(&norm_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", norm_key))?;

        let mm_soft_emb_norm = GemmaRMSNorm::new(norm_weight, layer_norm_eps);

        let patches_per_image = image_size / patch_size;
        let tokens_per_side = (mm_tokens_per_image as f64).sqrt() as usize;
        let kernel_size = (patches_per_image / tokens_per_side) as i32;

        // Verify projection weight shape
        let proj_shape = mlxcel_core::array_shape(&mm_input_projection_weight);
        if proj_shape[0] != vision_hidden_size as i32 {
            return Err(format!(
                "Projection weight shape mismatch: expected first dim {}, got {}",
                vision_hidden_size, proj_shape[0]
            ));
        }

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            tokens_per_side,
            kernel_size,
        })
    }
}

impl MultiModalConnector for AvgPoolProjector {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let _num_patches = shape[1]; // num_patches
        let l = shape[2]; // hidden_size

        // Transpose: [B, num_patches, hidden] -> [B, hidden, num_patches]
        let reshaped = mlxcel_core::transpose_axes(x, &[0, 2, 1]);

        // Reshape to spatial grid: [B, hidden, patches_per_image, patches_per_image]
        let reshaped = mlxcel_core::reshape(
            &reshaped,
            &[
                b,
                l,
                self.patches_per_image as i32,
                self.patches_per_image as i32,
            ],
        );

        // Transpose to [B, patches_per_image, patches_per_image, hidden] for avg_pool2d
        let reshaped = mlxcel_core::transpose_axes(&reshaped, &[0, 2, 3, 1]);

        // Average pool with kernel_size stride
        let pooled = mlxcel_core::avg_pool2d(
            &reshaped,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            0,
            0,
        );

        // Transpose back: [B, tokens_per_side, tokens_per_side, hidden]
        // -> [B, hidden, tokens_per_side, tokens_per_side]
        let pooled = mlxcel_core::transpose_axes(&pooled, &[0, 3, 1, 2]);

        // Flatten spatial dims: [B, hidden, tokens_per_side, tokens_per_side]
        // -> [B, hidden, num_tokens]
        let num_tokens = (self.tokens_per_side * self.tokens_per_side) as i32;
        let pooled = mlxcel_core::reshape(&pooled, &[b, l, num_tokens]);

        // Transpose to [B, num_tokens, hidden]
        let pooled = mlxcel_core::transpose_axes(&pooled, &[0, 2, 1]);

        // Apply RMS norm
        let normed = self.mm_soft_emb_norm.forward(&pooled);

        // Project: einsum("btm,md->btd") == matmul(normed, weight)
        let projected = mlxcel_core::matmul(&normed, &self.mm_input_projection_weight);

        // Cast to input dtype
        let dtype = mlxcel_core::array_dtype(x);
        mlxcel_core::astype(&projected, dtype)
    }
}
