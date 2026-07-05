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

//! DeepSeek-VL2 `downsample_mlp_gelu` projector.
//!
//! Folds each non-overlapping `ds x ds` block of vision patches into one feature
//! vector (space-to-depth), then a two-layer GELU MLP maps `input_dim * ds * ds`
//! to the decoder width. The block feature order is channel-outermost, index
//! `c * ds * ds + dy * ds + dx`; the first Linear was trained against exactly
//! that layout.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseek_vl_v2/` (`MlpProjector`,
//! `downsample_mlp_gelu`).
//!
//! Used by: DeepSeek-VL2

use super::MultiModalConnector;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DownsampleMlpGelu {
    layers_0: UnifiedLinear,
    layers_2: UnifiedLinear,
    downsample_ratio: i32,
}

impl DownsampleMlpGelu {
    /// `prefix` is the projector namespace (`projector`), holding
    /// `layers.0` (`input_dim * ds^2 -> hidden`) and `layers.2`
    /// (`hidden -> n_embed`) with a GELU between them.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        downsample_ratio: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let layers_0 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.0"), group_size, bits)?;
        let layers_2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.2"), group_size, bits)?;
        Ok(Self {
            layers_0,
            layers_2,
            downsample_ratio,
        })
    }
}

impl MultiModalConnector for DownsampleMlpGelu {
    /// `vision_features`: `(tiles, N, input_dim)` with `N = g * g` a perfect
    /// square. Returns `(tiles, N', n_embed)` with `N' = (ceil(g / ds))^2`.
    fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(vision_features);
        let (tiles, n, c) = (s[0], s[1], s[2]);
        let ds = self.downsample_ratio;
        let g = (n as f64).sqrt().round() as i32;

        // (tiles, N, C) -> (tiles, g, g, C), token axis row-major over (h, w).
        let x = mlxcel_core::reshape(vision_features, &[tiles, g, g, c]);

        // Zero-pad the two spatial axes at the high end up to a multiple of ds.
        let g_pad = ((g + ds - 1) / ds) * ds;
        let pad_after = g_pad - g;
        let x = if pad_after > 0 {
            mlxcel_core::pad(&x, &[0, 0, 0, pad_after, 0, pad_after, 0, 0], 0.0)
        } else {
            x
        };
        let hp = g_pad / ds;

        // Space-to-depth: (tiles, hp, ds, wp, ds, C) with (dy, dx) the intra-block
        // offsets, then gather each block channel-outermost into one vector.
        // transpose [0,1,3,5,2,4] -> (tiles, hp, wp, C, dy, dx); flat feature
        // index is c*ds*ds + dy*ds + dx.
        let x = mlxcel_core::reshape(&x, &[tiles, hp, ds, hp, ds, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 5, 2, 4]);
        let x = mlxcel_core::reshape(&x, &[tiles, hp * hp, c * ds * ds]);

        let x = self.layers_0.forward(&x);
        let x = mlxcel_core::gelu(&x); // exact erf GELU
        self.layers_2.forward(&x)
    }
}
