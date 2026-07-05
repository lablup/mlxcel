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

//! ERNIE-4.5-VL variable-resolution resampler (`resampler_model.*`).
//!
//! Compresses the DFNRope tower output into text-embedding rows:
//!
//! 1. **Spatial fold**: `(N, in_dim)` rows arrive in merge-window order, so
//!    each 2x2 merge window is 4 consecutive rows; reshape to
//!    `(N/4, in_dim * 4)` and run `spatial_linear` (Linear, exact GELU, Linear,
//!    LayerNorm at sequential indices 0, 2, 3).
//! 2. **Temporal fold**: per grid entry, gather even-frame rows as A and
//!    odd-frame rows as B (a single image duplicates its one frame), concat
//!    along the feature axis to `(rows, in_dim * 8)` and run `temporal_linear`
//!    (Linear in_dim*8 -> in_dim*4, GELU, Linear, LayerNorm).
//! 3. `mlp` Linear to the text hidden size, then `after_norm` RMSNorm (eps 1e-5).
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/ernie4_5_moe_vl.py>
//! (`VariableResolutionResamplerModel`).

use crate::vision::encoders::qwen2_vl::concat_many;
use mlxcel_core::layers::{LayerNorm, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

struct LinearGeluLinearNorm {
    lin0: UnifiedLinear,
    lin2: UnifiedLinear,
    norm3: LayerNorm,
}

impl LinearGeluLinearNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        let ln_w_key = format!("{prefix}.layers.3.weight");
        let ln_w = weights
            .get(&ln_w_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {ln_w_key}"))?;
        let ln_b = weights
            .get(&format!("{prefix}.layers.3.bias"))
            .map(|b| mlxcel_core::copy(b));
        Ok(Self {
            lin0: UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.0"), gs, bits)?,
            lin2: UnifiedLinear::from_weights(weights, &format!("{prefix}.layers.2"), gs, bits)?,
            norm3: LayerNorm::new(ln_w, ln_b, 1e-6),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.lin0.forward(x);
        let h = mlxcel_core::gelu(&h); // exact erf GELU
        let h = self.lin2.forward(&h);
        self.norm3.forward(&h)
    }
}

pub struct Ernie45VlResampler {
    spatial_linear: LinearGeluLinearNorm,
    temporal_linear: Option<LinearGeluLinearNorm>,
    mlp: UnifiedLinear,
    after_norm: RMSNorm,
    spatial_conv_size: i32,
}

impl Ernie45VlResampler {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spatial_conv_size: i32,
        use_temporal_conv: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let after_norm_key = format!("{prefix}.after_norm.weight");
        let after_norm_w = weights
            .get(&after_norm_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {after_norm_key}"))?;
        Ok(Self {
            spatial_linear: LinearGeluLinearNorm::from_weights(
                weights,
                &format!("{prefix}.spatial_linear"),
                gs,
                bits,
            )?,
            temporal_linear: if use_temporal_conv {
                Some(LinearGeluLinearNorm::from_weights(
                    weights,
                    &format!("{prefix}.temporal_linear"),
                    gs,
                    bits,
                )?)
            } else {
                None
            },
            mlp: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp"), gs, bits)?,
            after_norm: RMSNorm::new(after_norm_w, 1e-5),
            spatial_conv_size,
        })
    }

    /// `x`: `(N, in_dim)` tower output in merge-window row order; `grid_thw`
    /// per image `(t, h, w)` in patch units. Returns `(rows, hidden_size)`.
    pub fn forward_with_grid(
        &self,
        x: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        let sq = self.spatial_conv_size * self.spatial_conv_size;
        let in_shape = mlxcel_core::array_shape(x);
        let (n, c) = (in_shape[0], in_shape[1]);

        // Spatial fold: 4 consecutive merge-window rows -> one feature row.
        let folded = mlxcel_core::reshape(x, &[n / sq, c * sq]);
        let mut h = self.spatial_linear.forward(&folded);

        if let Some(ref temporal) = self.temporal_linear {
            // Temporal pair fold: per grid entry with S = h*w/4 rows per frame,
            // gather even frames as A and odd frames as B (t == 1 duplicates the
            // single frame), then concat along the feature axis.
            let feat = mlxcel_core::array_shape(&h)[1];
            let mut even_slices: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut odd_slices: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut offset = 0i32;
            for &(t, gh, gw) in grid_thw {
                let s = gh * gw / sq;
                let mut f = 0;
                while f < t {
                    even_slices.push(mlxcel_core::slice(
                        &h,
                        &[offset + f * s, 0],
                        &[offset + (f + 1) * s, feat],
                    ));
                    f += 2;
                }
                let mut f = if t > 1 { 1 } else { 0 };
                while f < t {
                    odd_slices.push(mlxcel_core::slice(
                        &h,
                        &[offset + f * s, 0],
                        &[offset + (f + 1) * s, feat],
                    ));
                    f += 2;
                }
                offset += t * s;
            }
            let even = if even_slices.len() == 1 {
                even_slices.into_iter().next().unwrap()
            } else {
                concat_many(&even_slices, 0)
            };
            let odd = if odd_slices.len() == 1 {
                odd_slices.into_iter().next().unwrap()
            } else {
                concat_many(&odd_slices, 0)
            };
            let paired = mlxcel_core::concatenate(&even, &odd, -1);
            h = temporal.forward(&paired);
        }

        let h = self.mlp.forward(&h);
        self.after_norm.forward(&h)
    }
}
