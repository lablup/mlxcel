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

//! Acoustic text encoder: phoneme ids -> aligned acoustic features.
//!
//! Embedding (`178 -> 512`), three `[weight-norm Conv1d(k=5) + LayerNorm +
//! LeakyReLU(0.2)]` blocks over the channel axis, then a bi-LSTM (`512 -> 512`).
//! The block LayerNorm normalizes the channel axis (its affine params are stored
//! as `gamma` / `beta`). Output is `(512, L)`, expanded to per-frame features by
//! the alignment matrix in the top-level forward.

use mlxcel_core::{MlxArray, UniquePtr};

use super::lstm::BiLstm;
use super::ops;
use super::weights::{ConvWn, Weights};

const LN_EPS: f32 = 1e-5;
const LEAKY: f32 = 0.2;

/// Channel-axis LayerNorm with `gamma` / `beta` affine.
struct ChannelLayerNorm {
    gamma: UniquePtr<MlxArray>,
    beta: UniquePtr<MlxArray>,
}

impl ChannelLayerNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            gamma: w.get(&format!("{prefix}.gamma"))?,
            beta: w.get(&format!("{prefix}.beta"))?,
        })
    }

    /// Apply to `(C, L)`: normalize each time step over the channel axis.
    fn forward(&self, x: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        // Transpose to (L, C), normalize over C, scale/shift, transpose back.
        let xt = ops::swap_axes(x, 0, 1); // (L, C)
        let mu = ops::mean_axis(&xt, -1);
        let xc = ops::sub(&xt, &mu);
        let var = ops::var_axis(&xt, -1);
        let denom = mlxcel_core::rsqrt(ops::r(&ops::add_scalar(&var, LN_EPS)));
        let normed = ops::mul(&xc, &denom);
        let scaled = ops::add(&ops::mul(&normed, &self.gamma), &self.beta);
        ops::swap_axes(&scaled, 0, 1) // (C, L)
    }
}

/// The acoustic text encoder.
pub(crate) struct TextEncoder {
    embedding: UniquePtr<MlxArray>,
    convs: Vec<ConvWn>,
    norms: Vec<ChannelLayerNorm>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub(crate) fn load(w: &Weights) -> Result<Self, String> {
        let embedding = w.get("text_encoder.embedding.weight")?;
        let mut convs = Vec::new();
        let mut norms = Vec::new();
        for i in 0..3 {
            // Conv1d(512, 512, k=5, pad=2), weight-norm.
            convs.push(ConvWn::load(
                w,
                &format!("text_encoder.cnn.{i}.0"),
                1,
                2,
                1,
                1,
                true,
            )?);
            norms.push(ChannelLayerNorm::load(
                w,
                &format!("text_encoder.cnn.{i}.1"),
            )?);
        }
        let lstm = BiLstm::load(w, "text_encoder.lstm", 256)?;
        Ok(Self {
            embedding,
            convs,
            norms,
            lstm,
        })
    }

    /// Encode a phoneme id sequence (length `l`), returning `(512, L)`.
    pub(crate) fn forward(&self, ids: &[i32]) -> UniquePtr<MlxArray> {
        let l = ids.len();
        let emb = ops::embed(&self.embedding, ids); // (L, 512)
        let mut x = ops::swap_axes(&emb, 0, 1); // (512, L)
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            x = conv.forward(&x);
            x = norm.forward(&x);
            x = ops::leaky_relu(&x, LEAKY);
        }
        let seq = ops::swap_axes(&x, 0, 1); // (L, 512)
        let out = self.lstm.forward(&seq, l); // (L, 512)
        ops::swap_axes(&out, 0, 1) // (512, L)
    }
}
