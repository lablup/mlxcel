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

//! PaddleOCR-VL vision-to-text projector (spatial-merge connector).
//!
//! Groups each `spatial_merge_size x spatial_merge_size` block of vision tokens
//! into one token, then projects `merge^2 * embed` down to the text hidden size
//! with a two-layer MLP (`linear_1 -> GELU -> linear_2`). A `pre_norm`
//! LayerNorm is applied to the per-token features before merging.
//!
//! This needs the per-image `grid_thw`, so it exposes `forward_with_grid`
//! rather than the grid-agnostic `MultiModalConnector` trait.
//!
//! Used by: PaddleOCR-VL
//! Reference: mlx-vlm `paddleocr_vl/vision.py` (`PaddleOCRProjector`).

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct PaddleOcrProjector {
    pre_norm: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    spatial_merge_size: i32,
}

impl PaddleOcrProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spatial_merge_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let pre_weight = weights
            .get(&format!("{prefix}.pre_norm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.pre_norm.weight"))?;
        let pre_bias = weights
            .get(&format!("{prefix}.pre_norm.bias"))
            .map(|b| mlxcel_core::copy(b));
        let pre_norm = LayerNorm::new(pre_weight, pre_bias, 1e-6);

        let linear_1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_1"), gs, bits)?;
        let linear_2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_2"), gs, bits)?;

        Ok(Self {
            pre_norm,
            linear_1,
            linear_2,
            spatial_merge_size: spatial_merge_size as i32,
        })
    }

    /// Project one image's tokens. `x`: `[t*h*w, embed]` in row-major order.
    fn project_image(&self, x: &MlxArray, t: i32, h: i32, w: i32) -> UniquePtr<MlxArray> {
        let normed = self.pre_norm.forward(x);
        let d = mlxcel_core::array_shape(&normed);
        let d = d[d.len() - 1];
        let m = self.spatial_merge_size;
        let h_block = h / m;
        let w_block = w / m;

        // [t*h*w, d] -> [t, h_block, m, w_block, m, d]
        let x = mlxcel_core::reshape(&normed, &[t, h_block, m, w_block, m, d]);
        // -> [t, h_block, w_block, m, m, d]
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
        // -> [t*h_block*w_block, m*m*d]
        let x = mlxcel_core::reshape(&x, &[t * h_block * w_block, m * m * d]);

        let x = self.linear_1.forward(&x);
        let x = mlxcel_core::gelu(&x);
        self.linear_2.forward(&x)
    }

    /// Project all images. `hidden_states`: `[total_tokens, embed]`.
    /// Returns `[merged_tokens, text_hidden]`.
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        if grid_thw.len() == 1 {
            let (t, h, w) = grid_thw[0];
            return self.project_image(hidden_states, t, h, w);
        }

        let mut parts: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(grid_thw.len());
        let mut offset = 0i32;
        for &(t, h, w) in grid_thw {
            let len = t * h * w;
            let shape = mlxcel_core::array_shape(hidden_states);
            let chunk = mlxcel_core::slice(hidden_states, &[offset, 0], &[offset + len, shape[1]]);
            parts.push(self.project_image(&chunk, t, h, w));
            offset += len;
        }

        let mut result = mlxcel_core::copy(parts[0].as_ref().unwrap());
        for part in &parts[1..] {
            result = mlxcel_core::concatenate(result.as_ref().unwrap(), part.as_ref().unwrap(), 0);
        }
        result
    }
}
