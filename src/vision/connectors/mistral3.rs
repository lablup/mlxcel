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

//! Mistral 3 Multi-Modal Projector
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/mistral3/mistral3.py
//!
//! Architecture: RMSNorm → PatchMerger(unfold 2×2, Linear(4*D→D)) → Linear(D→H) → GELU → Linear(H→H)
//!
//! The PatchMerger spatially merges 2×2 patches before MLP projection, reducing
//! the sequence length by 4× while preserving spatial information.
//!
//! Used by: Mistral 3 VLM (Mistral Small 3.1)

use super::MultiModalConnector;
use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Mistral3Projector {
    norm: RMSNorm,
    merging_layer: UnifiedLinear,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    patch_h: i32,
    spatial_merge_size: i32,
}

impl Mistral3Projector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        patch_h: i32,
        spatial_merge_size: i32,
        rms_norm_eps: f32,
    ) -> Result<Self, String> {
        // RMSNorm
        let norm_key = format!("{}.norm.weight", prefix);
        let norm_weight = weights
            .get(&norm_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", norm_key))?;
        let norm = RMSNorm::new(norm_weight, rms_norm_eps);

        // PatchMerger merging_layer: Linear(hidden*4, hidden, no bias)
        let merging_layer = UnifiedLinear::from_weights(
            weights,
            &format!("{}.patch_merger.merging_layer", prefix),
            group_size,
            bits,
        )?;

        // MLP: Linear(vision_hidden → text_hidden) → GELU → Linear(text_hidden → text_hidden)
        let linear_1 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_1", prefix),
            group_size,
            bits,
        )?;
        let linear_2 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_2", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            norm,
            merging_layer,
            linear_1,
            linear_2,
            patch_h,
            spatial_merge_size,
        })
    }

    /// PatchMerger: unfold 2×2 patches with stride 2, then project
    ///
    /// Matches Python's unfold (im2col) channels-first ordering:
    /// [c0_s0, c0_s1, c0_s2, c0_s3, c1_s0, c1_s1, ...]
    ///
    /// `grid_h` / `grid_w` are the pre-merge patch rows/cols for this image
    /// (`target_h / patch_size`, `target_w / patch_size`); for a fixed-square
    /// input they both equal `patch_h`, for a dynamic aspect-ratio input they
    /// differ. Mirrors
    /// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/mistral3/mistral3.py
    /// (`Mistral3PatchMerger.__call__`), which reshapes each image's tokens to
    /// its own `(h, w)` grid before the unfold.
    ///
    /// Input: [N, D] (squeezed from [1, N, D]), N == grid_h * grid_w
    /// 1. Reshape: [H*W, D] → [H, W, D]
    /// 2. Reshape: [H/2, 2, W/2, 2, D]
    /// 3. Transpose to channels-first: [H/2, W/2, D, 2, 2]
    /// 4. Reshape: [H/2 * W/2, D*4]
    /// 5. merging_layer: [H/2 * W/2, D]
    fn patch_merge(&self, x: &MlxArray, grid_h: i32, grid_w: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let d = shape[shape.len() - 1];
        let h = grid_h;
        let w = grid_w;
        let s = self.spatial_merge_size;
        let h2 = h / s;
        let w2 = w / s;

        // [H*W, D] → [H, W, D]
        let x = mlxcel_core::reshape(x, &[h, w, d]);
        // [H, W, D] → [H/2, 2, W/2, 2, D]
        let x = mlxcel_core::reshape(&x, &[h2, s, w2, s, d]);
        // [H/2, 2, W/2, 2, D] → [H/2, W/2, D, 2, 2] (channels-first ordering)
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 1, 3]);
        // [H/2, W/2, D, 2, 2] → [H/2 * W/2, D*4]
        let x = mlxcel_core::reshape(&x, &[h2 * w2, d * s * s]);
        // merging_layer: [H/2 * W/2, D*4] → [H/2 * W/2, D]
        self.merging_layer.forward(&x)
    }
}

impl Mistral3Projector {
    /// RMSNorm → PatchMerger(grid_h, grid_w) → Linear → GELU → Linear.
    fn project(&self, vision_features: &MlxArray, grid_h: i32, grid_w: i32) -> UniquePtr<MlxArray> {
        // Squeeze batch dim: [1, N, D] → [N, D]
        let shape = mlxcel_core::array_shape(vision_features);
        let x = if shape.len() == 3 && shape[0] == 1 {
            mlxcel_core::squeeze_axis(vision_features, 0)
        } else {
            mlxcel_core::copy(vision_features)
        };

        // RMSNorm
        let x = self.norm.forward(&x);
        // PatchMerger over this image's actual patch grid
        let x = self.patch_merge(&x, grid_h, grid_w);
        // MLP: Linear → GELU → Linear
        let x = self.linear_1.forward(&x);
        let x = mlxcel_core::gelu(&x);
        let x = self.linear_2.forward(&x);

        // Add batch dim back: [N', D'] → [1, N', D']
        mlxcel_core::expand_dims(&x, 0)
    }
}

impl MultiModalConnector for Mistral3Projector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        // Fixed-square fallback: assume the configured square patch grid.
        self.project(vision_features, self.patch_h, self.patch_h)
    }

    fn forward_with_grid(
        &self,
        vision_features: &MlxArray,
        grid_h: i32,
        grid_w: i32,
    ) -> UniquePtr<MlxArray> {
        self.project(vision_features, grid_h, grid_w)
    }
}
