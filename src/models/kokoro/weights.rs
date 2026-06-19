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

//! Weight-map accessors for the Kokoro checkpoint.
//!
//! Wraps a [`WeightMap`] with helpers that fetch tensors by name, reconstruct
//! weight-normalized convolution kernels, and apply convolutions. Conv weights
//! in the Kokoro checkpoint are stored as `weight_g` / `weight_v` pairs (PyTorch
//! `torch.nn.utils.weight_norm`); plain layers store `weight` / `bias`.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ops;

/// Borrowed view over a Kokoro [`WeightMap`] with named-tensor accessors.
pub(crate) struct Weights<'a> {
    map: &'a WeightMap,
}

impl<'a> Weights<'a> {
    pub(crate) fn new(map: &'a WeightMap) -> Self {
        Self { map }
    }

    /// Fetch a tensor by exact name, cloning the handle. Errors if absent.
    pub(crate) fn get(&self, name: &str) -> Result<UniquePtr<MlxArray>, String> {
        self.map
            .get(name)
            .map(|w| mlxcel_core::copy(ops::r(w)))
            .ok_or_else(|| format!("kokoro weight not found: {name}"))
    }

    /// Plain `weight` / `bias` linear weights for `prefix` (`prefix.weight`,
    /// `prefix.bias`). Bias is optional.
    pub(crate) fn linear(
        &self,
        prefix: &str,
    ) -> Result<(UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>), String> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let b = self.get(&format!("{prefix}.bias")).ok();
        Ok((w, b))
    }

    /// Reconstruct a weight-norm conv kernel `w = g * v / ||v||` for `prefix`
    /// (`prefix.weight_g`, `prefix.weight_v`), with optional `prefix.bias`.
    pub(crate) fn conv_wn(
        &self,
        prefix: &str,
    ) -> Result<(UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>), String> {
        let g = self.get(&format!("{prefix}.weight_g"))?;
        let v = self.get(&format!("{prefix}.weight_v"))?;
        let w = ops::weight_norm(&g, &v);
        let b = self.get(&format!("{prefix}.bias")).ok();
        Ok((w, b))
    }
}

/// A weight-normalized convolution layer with cached reconstructed weights.
pub(crate) struct ConvWn {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    transposed: bool,
    output_padding: i32,
}

impl ConvWn {
    /// Load a weight-norm conv from `prefix`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self, String> {
        let (weight, b) = w.conv_wn(prefix)?;
        Ok(Self {
            weight,
            bias: if bias { b } else { None },
            stride,
            padding,
            dilation,
            groups,
            transposed: false,
            output_padding: 0,
        })
    }

    /// Load a weight-norm transposed conv from `prefix`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_transposed(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        output_padding: i32,
        groups: i32,
    ) -> Result<Self, String> {
        let (weight, b) = w.conv_wn(prefix)?;
        Ok(Self {
            weight,
            bias: b,
            stride,
            padding,
            dilation: 1,
            groups,
            transposed: true,
            output_padding,
        })
    }

    /// A plain (non-weight-norm) conv from `prefix.weight` / `prefix.bias`.
    pub(crate) fn load_plain(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
    ) -> Result<Self, String> {
        let (weight, bias) = w.linear(prefix)?;
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation: 1,
            groups: 1,
            transposed: false,
            output_padding: 0,
        })
    }

    /// Apply the convolution to a `(C, L)` activation, returning `(C_out, L_out)`.
    pub(crate) fn forward(&self, x: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        if self.transposed {
            ops::conv_transpose1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                self.dilation,
                self.output_padding,
                self.groups,
            )
        } else {
            ops::conv1d(
                x,
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        }
    }
}
