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

//! Step-3.7 vision-to-text connector (downsamplers + projector).
//!
//! Reshapes the per-token encoder output back to the spatial grid, applies two
//! stride-2 pad-1 3x3 convs (`vit_downsampler1`: `width -> 2*width`,
//! `vit_downsampler2`: `2*width -> 4*width`) that each halve the grid per axis,
//! flattens the resulting `grid/4 x grid/4` tokens, and projects `4*width` down
//! to the text hidden size with `vit_large_projector` (bias iff
//! `projector_bias`). A `52x52` base grid collapses to `13x13 = 169` tokens; a
//! `36x36` patch grid to `9x9 = 81` tokens.
//!
//! The downsampler conv weights live under `vision_model.*` (owned by the
//! tower) but are applied here after the encoder output is reshaped, matching
//! the reference module split.
//!
//! Used by: Step-3.7 (step3p7).

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Step3p7Connector {
    downsampler1_weight: UniquePtr<MlxArray>,
    downsampler1_bias: Option<UniquePtr<MlxArray>>,
    downsampler2_weight: UniquePtr<MlxArray>,
    downsampler2_bias: Option<UniquePtr<MlxArray>>,
    projector: UnifiedLinear,
    stride: i32,
    width: i32,
}

fn load_conv(
    weights: &WeightMap,
    prefix: &str,
    in_ch: i32,
) -> Result<(UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>), String> {
    let raw = weights
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    // The loader permutes to channels-last `(out, kH, kW, in)`; the guard here
    // keeps the connector robust to already-converted weights.
    let weight = crate::vision::encoders::step3p7::permute_conv_weight_to_channels_last(raw, in_ch);
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok((weight, bias))
}

impl Step3p7Connector {
    pub fn from_weights(
        weights: &WeightMap,
        vision_prefix: &str,
        projector_prefix: &str,
        width: usize,
        stride: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let width = width as i32;
        let (downsampler1_weight, downsampler1_bias) =
            load_conv(weights, &format!("{vision_prefix}.vit_downsampler1"), width)?;
        let (downsampler2_weight, downsampler2_bias) = load_conv(
            weights,
            &format!("{vision_prefix}.vit_downsampler2"),
            2 * width,
        )?;
        let projector = UnifiedLinear::from_weights(weights, projector_prefix, gs, bits)?;

        Ok(Self {
            downsampler1_weight,
            downsampler1_bias,
            downsampler2_weight,
            downsampler2_bias,
            projector,
            stride: stride as i32,
            width,
        })
    }

    fn downsample(
        &self,
        x: &MlxArray,
        weight: &MlxArray,
        bias: Option<&UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        // stride-2 pad-1 3x3 conv, channels-last.
        let out = mlxcel_core::conv2d(x, weight, self.stride, self.stride, 1, 1, 1, 1, 1);
        match bias {
            Some(b) => mlxcel_core::add(&out, b),
            None => out,
        }
    }

    /// Project one homogeneous batch of grid tokens.
    ///
    /// `hidden_states`: `(batch, grid_h*grid_w, width)` in row-major order.
    /// Returns `(batch, (grid_h/4)*(grid_w/4), text_hidden)`.
    pub fn forward(
        &self,
        hidden_states: &MlxArray,
        grid_h: i32,
        grid_w: i32,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];

        // Back to the spatial grid: (B, grid_h, grid_w, width).
        let grid = mlxcel_core::reshape(hidden_states, &[batch, grid_h, grid_w, self.width]);

        let d1 = self.downsample(
            &grid,
            &self.downsampler1_weight,
            self.downsampler1_bias.as_ref(),
        );
        let d2 = self.downsample(
            &d1,
            &self.downsampler2_weight,
            self.downsampler2_bias.as_ref(),
        );

        let dshape = mlxcel_core::array_shape(&d2);
        let (out_h, out_w, out_ch) = (dshape[1], dshape[2], dshape[3]);
        let flat = mlxcel_core::reshape(&d2, &[batch, out_h * out_w, out_ch]);

        self.projector.forward(&flat)
    }
}
