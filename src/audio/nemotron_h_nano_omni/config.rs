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

//! Nemotron H Nano Omni audio config — Parakeet flavour.
//!
//! Mirrors upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/config.py (AudioConfig).
//!
//! Used by: Nemotron H Nano Omni audio modality.

use serde::Deserialize;

fn default_hidden_size() -> usize {
    1024
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_attention_bias() -> bool {
    false
}
fn default_convolution_bias() -> bool {
    false
}
fn default_conv_kernel_size() -> usize {
    9
}
fn default_subsampling_factor() -> usize {
    8
}
fn default_subsampling_conv_channels() -> usize {
    256
}
fn default_num_mel_bins() -> usize {
    128
}
fn default_subsampling_conv_kernel_size() -> usize {
    3
}
fn default_subsampling_conv_stride() -> usize {
    2
}
fn default_max_position_embeddings() -> usize {
    5_000
}
fn default_scale_input() -> bool {
    false
}
fn default_projection_hidden_size() -> usize {
    4_096
}
fn default_projection_bias() -> bool {
    false
}
fn default_sampling_rate() -> u32 {
    16_000
}
fn default_hop_length() -> usize {
    160
}
fn default_n_fft() -> usize {
    512
}
fn default_win_length() -> usize {
    400
}
fn default_preemphasis() -> f32 {
    0.97
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// Parakeet-flavour audio configuration.
///
/// Maps 1:1 to the upstream `AudioConfig` dataclass. Only fields that
/// influence inference-time math are surfaced; training-only knobs
/// (dropout, layerdrop, initializer_range) are intentionally omitted.
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronOmniAudioConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_convolution_bias")]
    pub convolution_bias: bool,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,
    #[serde(default = "default_subsampling_conv_channels")]
    pub subsampling_conv_channels: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_subsampling_conv_kernel_size")]
    pub subsampling_conv_kernel_size: usize,
    #[serde(default = "default_subsampling_conv_stride")]
    pub subsampling_conv_stride: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_scale_input")]
    pub scale_input: bool,
    #[serde(default = "default_projection_hidden_size")]
    pub projection_hidden_size: usize,
    #[serde(default = "default_projection_bias")]
    pub projection_bias: bool,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_win_length")]
    pub win_length: usize,
    #[serde(default = "default_preemphasis")]
    pub preemphasis: f32,
    /// Eps for the projector's pre-norm RMSNorm. Defaults to 1e-5 to
    /// match the upstream `nn.RMSNorm(config.hidden_size, eps=1e-5)`
    /// and is not part of the upstream config dataclass.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

impl Default for NemotronOmniAudioConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_attention_heads: default_num_attention_heads(),
            num_hidden_layers: default_num_hidden_layers(),
            intermediate_size: default_intermediate_size(),
            attention_bias: default_attention_bias(),
            convolution_bias: default_convolution_bias(),
            conv_kernel_size: default_conv_kernel_size(),
            subsampling_factor: default_subsampling_factor(),
            subsampling_conv_channels: default_subsampling_conv_channels(),
            num_mel_bins: default_num_mel_bins(),
            subsampling_conv_kernel_size: default_subsampling_conv_kernel_size(),
            subsampling_conv_stride: default_subsampling_conv_stride(),
            max_position_embeddings: default_max_position_embeddings(),
            scale_input: default_scale_input(),
            projection_hidden_size: default_projection_hidden_size(),
            projection_bias: default_projection_bias(),
            sampling_rate: default_sampling_rate(),
            hop_length: default_hop_length(),
            n_fft: default_n_fft(),
            win_length: default_win_length(),
            preemphasis: default_preemphasis(),
            rms_norm_eps: default_rms_norm_eps(),
        }
    }
}

impl NemotronOmniAudioConfig {
    /// `head_dim` derived from `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of subsampling Conv2D layers — `log2(subsampling_factor)`.
    /// The upstream config default of `subsampling_factor=8` produces 3
    /// stride-2 stages.
    pub fn num_subsampling_layers(&self) -> usize {
        // log2 implemented without `f32::log2` so it stays exact for
        // power-of-two factors (the only ones the upstream model uses).
        // Non-power-of-two factors silently truncate log2; guard in debug
        // builds so a bad config.json value is caught early.
        debug_assert!(
            self.subsampling_factor.is_power_of_two(),
            "subsampling_factor must be a power of two (got {})",
            self.subsampling_factor
        );
        let mut n = self.subsampling_factor;
        let mut count = 0;
        while n > 1 {
            n /= 2;
            count += 1;
        }
        count
    }

    /// Output length of subsampling for an input length, mirroring upstream
    /// `ParakeetEncoder._get_subsampling_output_length`.
    ///
    /// The upstream implementation uses
    ///   `add_pad = (((kernel_size - 1) // 2) * 2) - kernel_size`
    /// per stage, then `lengths = floor(lengths + add_pad) / stride + 1`.
    pub fn subsampling_output_length(&self, mut length: usize) -> usize {
        let kernel = self.subsampling_conv_kernel_size as i64;
        let stride = self.subsampling_conv_stride as i64;
        let add_pad = ((kernel - 1) / 2) * 2 - kernel;
        for _ in 0..self.num_subsampling_layers() {
            let signed = length as i64 + add_pad;
            let next = if signed < 0 { 0 } else { signed / stride + 1 };
            length = next.max(0) as usize;
        }
        length
    }

    /// Subsampling output length per layer, used for masking inside
    /// the subsampling Conv2D pipeline. Mirrors upstream
    /// `_get_output_length` per layer (stride==1 stages return input).
    pub fn subsampling_output_length_after_layer(
        &self,
        length: usize,
        layer_stride: usize,
    ) -> usize {
        if layer_stride == 1 {
            return length;
        }
        let kernel = self.subsampling_conv_kernel_size as i64;
        let stride = layer_stride as i64;
        let padding = (kernel - 1) / 2;
        let signed = length as i64 + 2 * padding - kernel;
        if signed < 0 {
            0
        } else {
            (signed / stride + 1) as usize
        }
    }
}
