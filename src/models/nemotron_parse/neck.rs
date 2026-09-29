// Copyright 2025-2026 Lablup Inc.
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

//! Nemotron-Parse compression neck (`RadioWithNeck` in the reference).
//!
//! `conv1` is a 1x1 Conv1d (applied here as a Linear), `conv2` a
//! `(1, 4)`-kernel, `(1, 4)`-stride Conv2d that shrinks the patch grid 4x
//! horizontally. Because the kernel tiles each row without overlap, the conv
//! is exactly a reshape to `[B, hp, wp/4, 4 * C]` followed by one matmul
//! against the kernel flattened in `(kw, c_in)` order, which is what
//! [`NemotronParseNeck::forward`] does. The summary vector is projected into
//! the same width and appended as one extra encoder row.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::NemotronParseVisionConfig;
use super::encoder::{VISION_LN_EPS, dense_or_quantized, layer_norm_eps};

pub(crate) struct NemotronParseNeck {
    conv1: UnifiedLinear,
    layer_norm1: LayerNorm,
    /// `[out, kw * in]`, the `conv2` kernel flattened in `(kw, c_in)` order.
    conv2: UniquePtr<MlxArray>,
    layer_norm2: LayerNorm,
    sum_proj: UnifiedLinear,
    layer_norm3: LayerNorm,
    dim: i32,
    kernel_w: i32,
}

impl NemotronParseNeck {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronParseVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let dim = config.neck_dim;
        let kw = config.neck_kernel_w;
        let conv2_key = format!("{prefix}.conv2.weight");
        let conv2 = weights
            .get(&conv2_key)
            .ok_or_else(|| format!("Nemotron-Parse weight not found: {conv2_key}"))?;
        let shape = mlxcel_core::array_shape(conv2);
        if shape != [dim, 1, kw, dim] {
            return Err(format!(
                "Nemotron-Parse {conv2_key} has shape {shape:?}, expected [{dim}, 1, {kw}, {dim}]"
            ));
        }
        let conv2 = mlxcel_core::reshape(conv2, &[dim, kw * dim]);
        // Transposed once at load so the forward is a plain matmul.
        let conv2 = mlxcel_core::transpose_axes(&conv2, &[1, 0]);
        let conv2 = mlxcel_core::copy(&conv2);
        Ok(Self {
            conv1: dense_or_quantized(weights, &format!("{prefix}.conv1"), group_size, bits)?,
            layer_norm1: layer_norm_eps(weights, &format!("{prefix}.layer_norm1"), VISION_LN_EPS)?,
            conv2,
            layer_norm2: layer_norm_eps(weights, &format!("{prefix}.layer_norm2"), VISION_LN_EPS)?,
            sum_proj: dense_or_quantized(weights, &format!("{prefix}.sum_proj"), group_size, bits)?,
            layer_norm3: layer_norm_eps(weights, &format!("{prefix}.layer_norm3"), VISION_LN_EPS)?,
            dim,
            kernel_w: kw,
        })
    }

    /// `features: [B, hp * wp, C_vit]`, `summary: [B, n_summary * C_vit]`
    /// -> `[B, hp * (wp / kw) + 1, dim]`.
    pub(crate) fn forward(
        &self,
        features: &MlxArray,
        summary: &MlxArray,
        grid: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        let (hp, wp) = grid;
        let b = mlxcel_core::array_shape(features)[0];
        let (d, kw) = (self.dim, self.kernel_w);

        let f = self.layer_norm1.forward(&self.conv1.forward(features));
        let f = mlxcel_core::reshape(&f, &[b, hp, wp, d]);
        let f = row_window_conv(&f, &self.conv2, kw);
        let f = self.layer_norm2.forward(&f);

        let s = self.layer_norm3.forward(&self.sum_proj.forward(summary));
        let s = mlxcel_core::expand_dims(&s, 1);
        mlxcel_core::concatenate(&f, &s, 1)
    }
}

/// A `(1, kw)`-kernel, `(1, kw)`-stride conv over channels-last
/// `[B, hp, wp, C]`, as one matmul: each output column reads a disjoint run
/// of `kw` input columns, so the windows are a plain reshape. `w_t` is the
/// MLX `[out, 1, kw, in]` kernel reshaped to `[out, kw * in]` and transposed.
/// A trailing partial window is dropped, like torch does. Returns
/// `[B, hp * (wp / kw), out]`.
pub(crate) fn row_window_conv(x: &MlxArray, w_t: &MlxArray, kw: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let (b, hp, wp, c) = (s[0], s[1], s[2], s[3]);
    let wo = wp / kw;
    let x = if wo * kw != wp {
        mlxcel_core::slice(x, &[0, 0, 0, 0], &[b, hp, wo * kw, c])
    } else {
        mlxcel_core::copy(x)
    };
    let x = mlxcel_core::reshape(&x, &[b, hp * wo, kw * c]);
    mlxcel_core::matmul(&x, w_t)
}
