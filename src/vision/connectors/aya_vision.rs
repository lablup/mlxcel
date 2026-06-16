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

//! Aya Vision Multi-Modal Projector (SwiGLU + PixelShuffle + LayerNorm)
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/aya_vision/aya_vision.py (AyaVisionMultiModalProjector)
//!
//! Architecture:
//!   pixel_shuffle(x) → LayerNorm → Linear(in, mid) → split → SiLU(gate) * x → Linear(mid/2, out)
//!
//! Used by: Aya Vision

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct AyaVisionProjector {
    layernorm_weight: UniquePtr<MlxArray>,
    layernorm_bias: UniquePtr<MlxArray>,
    layernorm_eps: f32,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    downsample_factor: usize,
}

impl AyaVisionProjector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        downsample_factor: usize,
        layernorm_eps: f32,
    ) -> Result<Self, String> {
        let ln_w = weights
            .get(&format!("{}.layernorm.weight", prefix))
            .ok_or_else(|| format!("Missing {}.layernorm.weight", prefix))?;
        let ln_b = weights
            .get(&format!("{}.layernorm.bias", prefix))
            .ok_or_else(|| format!("Missing {}.layernorm.bias", prefix))?;

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
            layernorm_weight: mlxcel_core::copy(ln_w),
            layernorm_bias: mlxcel_core::copy(ln_b),
            layernorm_eps,
            linear_1,
            linear_2,
            downsample_factor,
        })
    }

    /// Pixel shuffle: reshape spatial patches to reduce spatial resolution
    /// while increasing channel dimension.
    /// Input: [B, S, D] where S = H*W (spatial patches)
    /// Output: [B, S/(factor²), D*factor²]
    fn pixel_shuffle(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch_size = shape[0];
        let seq_length = shape[1];
        let feature_dim = shape[2];
        let factor = self.downsample_factor as i32;

        let height = (seq_length as f64).sqrt() as i32;
        let width = height;

        // [B, S, D] → [B, W, H, D]
        let x = mlxcel_core::reshape(x, &[batch_size, width, height, feature_dim]);

        // [B, W, H, D] → [B, W, H/factor, D*factor]
        let channels = feature_dim;
        let x = mlxcel_core::reshape(&x, &[batch_size, width, height / factor, channels * factor]);

        // Transpose: [B, W, H/factor, D*factor] → [B, H/factor, W, D*factor]
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);

        // [B, H/factor, W, D*factor] → [B, H/factor, W/factor, D*factor²]
        let new_channels = channels * factor;
        let x = mlxcel_core::reshape(
            &x,
            &[
                batch_size,
                height / factor,
                width / factor,
                new_channels * factor,
            ],
        );

        // Transpose back: [B, H/factor, W/factor, D*factor²] → [B, W/factor, H/factor, D*factor²]
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);

        // Flatten spatial: [B, W/factor * H/factor, D*factor²]
        let new_seq = (width / factor) * (height / factor);
        let final_dim = channels * factor * factor;
        mlxcel_core::reshape(&x, &[batch_size, new_seq, final_dim])
    }
}

impl MultiModalConnector for AyaVisionProjector {
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        // 1. Pixel shuffle
        let x = self.pixel_shuffle(vision_features);

        // 2. LayerNorm
        let x = mlxcel_core::layer_norm(
            &x,
            &self.layernorm_weight,
            &self.layernorm_bias,
            self.layernorm_eps,
        );

        // 3. Linear 1 (projects to alignment_intermediate_size)
        let x = self.linear_1.forward(&x);

        // 4. SwiGLU: split in half, apply SiLU to gate, multiply
        let shape = mlxcel_core::array_shape(&x);
        let mid = shape[2] / 2;
        let batch = shape[0];
        let seq = shape[1];

        // Split: x_part = x[..., :mid], gate_part = x[..., mid:]
        let x_part = mlxcel_core::slice(&x, &[0, 0, 0], &[batch, seq, mid]);
        let gate_part = mlxcel_core::slice(&x, &[0, 0, mid], &[batch, seq, shape[2]]);
        let gate_activated = mlxcel_core::silu(&gate_part);
        let x = mlxcel_core::multiply(&gate_activated, &x_part);

        // 5. Linear 2 (projects to text hidden size)
        self.linear_2.forward(&x)
    }
}
