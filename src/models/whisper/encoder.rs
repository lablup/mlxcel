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

//! Whisper audio encoder: two strided Conv1d stems, additive sinusoidal
//! position embeddings, N self-attention transformer blocks, and a final
//! LayerNorm.

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::WhisperDims;
use super::layers::ResidualAttentionBlock;

fn conv_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Whisper weight not found: {key}"))
}

/// Sinusoidal position table `[length, channels]`, matching the reference
/// `sinusoids()`: `[sin(t * inv_ts) || cos(t * inv_ts)]`.
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let max_timescale = 10_000.0f64;
    let log_increment = max_timescale.ln() / (half as f64 - 1.0).max(1.0);
    let inv_timescales: Vec<f64> = (0..half)
        .map(|i| (-log_increment * i as f64).exp())
        .collect();
    let mut out = vec![0.0f32; length * channels];
    for t in 0..length {
        for (i, &inv) in inv_timescales.iter().enumerate() {
            let scaled = t as f64 * inv;
            out[t * channels + i] = scaled.sin() as f32;
            out[t * channels + half + i] = scaled.cos() as f32;
        }
    }
    out
}

pub(crate) struct AudioEncoder {
    conv1_weight: UniquePtr<MlxArray>,
    conv1_bias: UniquePtr<MlxArray>,
    conv2_weight: UniquePtr<MlxArray>,
    conv2_bias: UniquePtr<MlxArray>,
    positional: UniquePtr<MlxArray>,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        dims: &WhisperDims,
        dtype: i32,
    ) -> Result<Self, String> {
        let positional_data = sinusoids(dims.n_audio_ctx as usize, dims.n_audio_state as usize);
        let positional =
            mlxcel_core::from_slice_f32(&positional_data, &[dims.n_audio_ctx, dims.n_audio_state]);
        let positional = mlxcel_core::astype(&positional, dtype);

        let mut blocks = Vec::with_capacity(dims.n_audio_layer as usize);
        for i in 0..dims.n_audio_layer {
            blocks.push(ResidualAttentionBlock::from_weights(
                weights,
                &format!("encoder.blocks.{i}"),
                dims.n_audio_head,
                false,
            )?);
        }

        let ln_weight = conv_weight(weights, "encoder.ln_post.weight")?;
        let ln_bias = weights
            .get("encoder.ln_post.bias")
            .map(|w| mlxcel_core::copy(w));

        Ok(Self {
            conv1_weight: conv_weight(weights, "encoder.conv1.weight")?,
            conv1_bias: conv_weight(weights, "encoder.conv1.bias")?,
            conv2_weight: conv_weight(weights, "encoder.conv2.weight")?,
            conv2_bias: conv_weight(weights, "encoder.conv2.bias")?,
            positional,
            blocks,
            ln_post: LayerNorm::new(ln_weight, ln_bias, 1e-5),
        })
    }

    /// Encode a mel spectrogram `[batch, n_frames, n_mels]` into audio features
    /// `[batch, n_audio_ctx, n_audio_state]`.
    pub(crate) fn forward(&self, mel: &MlxArray) -> UniquePtr<MlxArray> {
        // conv1: stride 1, pad 1. conv2: stride 2, pad 1. Both kernel 3.
        let x = mlxcel_core::conv1d(mel, &self.conv1_weight, 1, 1, 1, 1);
        let x = mlxcel_core::add(&x, &self.conv1_bias);
        let x = mlxcel_core::gelu(&x);

        let x = mlxcel_core::conv1d(&x, &self.conv2_weight, 2, 1, 1, 1);
        let x = mlxcel_core::add(&x, &self.conv2_bias);
        let x = mlxcel_core::gelu(&x);

        let mut x = mlxcel_core::add(&x, &self.positional);
        for block in &self.blocks {
            x = block.forward(&x, None, None, &mut None, &mut None);
        }
        self.ln_post.forward(&x)
    }
}
