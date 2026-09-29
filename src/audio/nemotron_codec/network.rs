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

//! Codec encoder and decoder stacks.
//!
//! Ports `AudioEncoder` and `AudioDecoder` from
//! `mlx_audio/codec/models/nemotron_voicechat/codec.py`. Layer indices match
//! the checkpoint's `encoder.layers.N` / `decoder.layers.N` keys:
//!
//! - encoder: `[Conv1d(stft_channels -> c0, 1)]`, then per stage
//!   `blocks_per_stage` ConvNeXt blocks and a strided `Conv1d(K = stride = rate)`
//!   to the next width (`latent_dim` after the last stage);
//! - decoder: per reversed stage a `ConvTranspose1d(K = stride = rate)` and
//!   `blocks_per_stage` ConvNeXt blocks, then `Conv1d(c0 -> stft_channels, 1)`.
//!
//! ConvNeXt block ids restart at 0 in each stack; only the decoder uses them
//! as streaming cache keys.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::cache::CausalConv1dCache;
use super::config::CodecConfig;
use super::layers::{Conv1d, ConvNeXtBlock1d, ConvTranspose1d};

enum EncoderLayer {
    Conv(Conv1d),
    Block(ConvNeXtBlock1d),
}

/// Spectrogram features `[B, F, stft_channels]` -> latents `[B, T, latent_dim]`.
pub(crate) struct AudioEncoder {
    layers: Vec<EncoderLayer>,
}

impl AudioEncoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &CodecConfig,
    ) -> Result<Self, String> {
        let channels = config.stage_channels();
        let key = |index: usize| format!("{prefix}.layers.{index}");
        let mut layers = vec![EncoderLayer::Conv(Conv1d::from_weights(
            weights,
            &key(0),
            config.stft_channels(),
            channels[0],
            1,
            1,
            1,
            false,
        )?)];
        let mut block_index = 0;
        for (stage, (&width, &rate)) in channels.iter().zip(&config.downsample_rates).enumerate() {
            for _ in 0..config.blocks_per_stage {
                layers.push(EncoderLayer::Block(ConvNeXtBlock1d::from_weights(
                    weights,
                    &key(layers.len()),
                    width,
                    config.block_kernel_size,
                    block_index,
                )?));
                block_index += 1;
            }
            let next = channels
                .get(stage + 1)
                .copied()
                .unwrap_or(config.latent_dim);
            layers.push(EncoderLayer::Conv(Conv1d::from_weights(
                weights,
                &key(layers.len()),
                width,
                next,
                rate,
                rate,
                1,
                false,
            )?));
        }
        Ok(Self { layers })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let mut hidden = mlxcel_core::copy(x);
        for layer in &self.layers {
            hidden = match layer {
                EncoderLayer::Conv(conv) => conv.forward(&hidden)?,
                EncoderLayer::Block(block) => block.forward(&hidden, None, false)?,
            };
        }
        Ok(hidden)
    }
}

enum DecoderLayer {
    Upsample(ConvTranspose1d),
    Block(ConvNeXtBlock1d),
    Conv(Conv1d),
}

/// Latents `[B, T, latent_dim]` -> spectrogram features
/// `[B, T * prod(rates), stft_channels]`.
pub(crate) struct AudioDecoder {
    layers: Vec<DecoderLayer>,
}

impl AudioDecoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &CodecConfig,
    ) -> Result<Self, String> {
        let channels = config.stage_channels();
        let key = |index: usize| format!("{prefix}.layers.{index}");
        let mut layers = Vec::new();
        let mut source = config.latent_dim;
        let mut block_index = 0;
        for (&width, &rate) in channels
            .iter()
            .rev()
            .zip(config.downsample_rates.iter().rev())
        {
            layers.push(DecoderLayer::Upsample(ConvTranspose1d::from_weights(
                weights,
                &key(layers.len()),
                source,
                width,
                rate,
                rate,
            )?));
            for _ in 0..config.blocks_per_stage {
                layers.push(DecoderLayer::Block(ConvNeXtBlock1d::from_weights(
                    weights,
                    &key(layers.len()),
                    width,
                    config.block_kernel_size,
                    block_index,
                )?));
                block_index += 1;
            }
            source = width;
        }
        layers.push(DecoderLayer::Conv(Conv1d::from_weights(
            weights,
            &key(layers.len()),
            channels[0],
            config.stft_channels(),
            1,
            1,
            1,
            false,
        )?));
        Ok(Self { layers })
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mut cache: Option<&mut CausalConv1dCache>,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let mut hidden = mlxcel_core::copy(x);
        for layer in &self.layers {
            hidden = match layer {
                DecoderLayer::Upsample(conv) => conv.forward(&hidden),
                DecoderLayer::Block(block) => {
                    block.forward(&hidden, cache.as_deref_mut(), flush)?
                }
                DecoderLayer::Conv(conv) => conv.forward(&hidden)?,
            };
        }
        Ok(hidden)
    }
}
