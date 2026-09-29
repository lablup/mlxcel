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

//! NemotronLabs VoiceChat codec configuration.
//!
//! Mirrors `NemotronVoiceChatCodecConfig` from
//! `mlx_audio/codec/models/nemotron_voicechat/config.py`. Defaults equal the
//! released checkpoint's `codec_config`, so a missing or partial JSON object
//! deserializes to the shipped codec.
//!
//! Used by: [`super::NemotronCodec`].

use serde::Deserialize;

/// Codec hyper-parameters (the checkpoint's `codec_config` object).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct CodecConfig {
    pub sample_rate: usize,
    pub base_channels: usize,
    pub channel_multipliers: Vec<usize>,
    pub downsample_rates: Vec<usize>,
    pub blocks_per_stage: usize,
    pub block_kernel_size: usize,
    pub latent_dim: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            base_channels: 384,
            channel_multipliers: vec![1, 2, 4],
            downsample_rates: vec![7, 7, 9],
            blocks_per_stage: 3,
            block_kernel_size: 7,
            latent_dim: 512,
            n_fft: 16,
            hop_length: 4,
            num_quantizers: 31,
            codebook_size: 1024,
        }
    }
}

impl CodecConfig {
    /// Waveform samples per codec frame: `hop_length * prod(downsample_rates)`.
    pub fn waveform_to_token_ratio(&self) -> usize {
        self.downsample_rates
            .iter()
            .fold(self.hop_length, |ratio, rate| ratio * rate)
    }

    /// Codec frames per second of audio.
    pub fn frame_rate(&self) -> f64 {
        self.sample_rate as f64 / self.waveform_to_token_ratio() as f64
    }

    /// Number of rFFT bins (`n_fft / 2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Spectrogram feature channels (real and imaginary halves).
    pub fn stft_channels(&self) -> usize {
        2 * self.num_bins()
    }

    /// Per-stage channel widths (`base_channels * multiplier`).
    pub fn stage_channels(&self) -> Vec<usize> {
        self.channel_multipliers
            .iter()
            .map(|m| self.base_channels * m)
            .collect()
    }

    /// Reject configurations the port cannot run.
    ///
    /// The overlap-add in [`super::stft`] relies on `n_fft % hop_length == 0`,
    /// which holds for the released codec (16 / 4).
    pub fn validate(&self) -> Result<(), String> {
        if self.channel_multipliers.is_empty()
            || self.channel_multipliers.len() != self.downsample_rates.len()
        {
            return Err(format!(
                "codec config: channel_multipliers ({}) and downsample_rates ({}) must be \
                 non-empty and of equal length",
                self.channel_multipliers.len(),
                self.downsample_rates.len()
            ));
        }
        let positive = [
            ("sample_rate", self.sample_rate),
            ("base_channels", self.base_channels),
            ("block_kernel_size", self.block_kernel_size),
            ("latent_dim", self.latent_dim),
            ("n_fft", self.n_fft),
            ("hop_length", self.hop_length),
            ("num_quantizers", self.num_quantizers),
            ("codebook_size", self.codebook_size),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(format!("codec config: {name} must be positive"));
            }
        }
        if self.downsample_rates.contains(&0) || self.channel_multipliers.contains(&0) {
            return Err("codec config: rates and multipliers must be positive".to_string());
        }
        if self.hop_length > self.n_fft || !self.n_fft.is_multiple_of(self.hop_length) {
            return Err(format!(
                "codec config: n_fft ({}) must be a multiple of hop_length ({})",
                self.n_fft, self.hop_length
            ));
        }
        if self.codebook_size > i32::MAX as usize {
            return Err("codec config: codebook_size exceeds i32 range".to_string());
        }
        Ok(())
    }
}
