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

//! Cache-aware FastConformer encoder settings (`audio_config.encoder`).
//!
//! Mirrors `ConformerArgs` in `mlx_audio/stt/models/nemotron_asr/config.py`,
//! with defaults equal to the NemotronLabs VoiceChat checkpoint.

use serde::{Deserialize, Deserializer};

/// `conv_context_size`: `"causal"` or an explicit `[left, right]` padding.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum ConvContextSize {
    Named(String),
    Explicit([usize; 2]),
}

impl Default for ConvContextSize {
    fn default() -> Self {
        Self::Named("causal".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ConformerArgs {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub conv_kernel_size: usize,
    pub causal_downsampling: bool,
    pub conv_context_size: ConvContextSize,
    pub conv_norm_type: String,
    pub self_attention_model: String,
    pub att_context_style: String,
    /// Every `[left, right]` look-ahead the model was trained with; the first
    /// entry is used for offline inference. Accepts `[70, 0]` or `[[70, 0]]`.
    #[serde(deserialize_with = "deserialize_att_context_size")]
    pub att_context_size: Vec<[i64; 2]>,
    pub pos_emb_max_len: usize,
    pub use_bias: bool,
    pub xscaling: bool,
}

impl Default for ConformerArgs {
    fn default() -> Self {
        Self {
            feat_in: 128,
            n_layers: 24,
            d_model: 1024,
            n_heads: 8,
            ff_expansion_factor: 4,
            subsampling: "dw_striding".to_string(),
            subsampling_factor: 8,
            subsampling_conv_channels: 256,
            conv_kernel_size: 9,
            causal_downsampling: true,
            conv_context_size: ConvContextSize::default(),
            conv_norm_type: "layer_norm".to_string(),
            self_attention_model: "rel_pos".to_string(),
            att_context_style: "chunked_limited".to_string(),
            att_context_size: vec![[70, 0]],
            pos_emb_max_len: 5000,
            use_bias: false,
            xscaling: false,
        }
    }
}

impl ConformerArgs {
    /// Reject configurations this port does not implement.
    pub fn validate(&self) -> Result<(), String> {
        if self.subsampling != "dw_striding" {
            return Err(format!("unsupported subsampling {:?}", self.subsampling));
        }
        if !self.subsampling_factor.is_power_of_two() || self.subsampling_factor < 2 {
            return Err(format!(
                "subsampling_factor must be a power of two >= 2, got {}",
                self.subsampling_factor
            ));
        }
        if self.conv_norm_type != "layer_norm" {
            return Err(format!(
                "conv_norm_type {:?} not supported (expected layer_norm)",
                self.conv_norm_type
            ));
        }
        if self.self_attention_model != "rel_pos" {
            return Err(format!(
                "self_attention_model {:?} not supported (expected rel_pos)",
                self.self_attention_model
            ));
        }
        if self.n_heads == 0
            || !self.d_model.is_multiple_of(self.n_heads)
            || !self.d_model.is_multiple_of(2)
        {
            return Err(format!(
                "d_model {} must be even and divisible by n_heads {}",
                self.d_model, self.n_heads
            ));
        }
        if self.conv_kernel_size == 0 {
            return Err("conv_kernel_size must be positive".to_string());
        }
        self.conv_padding()?;
        Ok(())
    }

    /// Number of stride-2 stages in the subsampling stack.
    pub fn sampling_num(&self) -> usize {
        self.subsampling_factor.trailing_zeros() as usize
    }

    /// `(left, right)` padding of every 3x3 stride-2 subsampling conv.
    pub fn subsampling_padding(&self) -> (usize, usize) {
        if self.causal_downsampling {
            (2, 1) // kernel - 1, stride - 1
        } else {
            (1, 1) // (kernel - 1) / 2
        }
    }

    /// Output length of the subsampling stack for `length` input frames (the
    /// same formula applies to the frequency axis).
    pub fn subsampled_length(&self, length: usize) -> usize {
        let (left, right) = self.subsampling_padding();
        (0..self.sampling_num()).fold(length, |n, _| (n + left + right).saturating_sub(3) / 2 + 1)
    }

    /// `(left, right)` time padding of the depthwise convolution.
    pub fn conv_padding(&self) -> Result<(usize, usize), String> {
        match &self.conv_context_size {
            ConvContextSize::Named(name) if name == "causal" => Ok((self.conv_kernel_size - 1, 0)),
            ConvContextSize::Named(other) => {
                Err(format!("unsupported conv_context_size {other:?}"))
            }
            ConvContextSize::Explicit([left, right]) => {
                if left + right + 1 != self.conv_kernel_size {
                    return Err(format!(
                        "conv_context_size [{left}, {right}] does not match kernel {}",
                        self.conv_kernel_size
                    ));
                }
                Ok((*left, *right))
            }
        }
    }

    /// The `[left, right]` attention context used for offline inference.
    pub fn default_att_context(&self) -> [i64; 2] {
        self.att_context_size.first().copied().unwrap_or([-1, -1])
    }
}

fn deserialize_att_context_size<'de, D>(deserializer: D) -> Result<Vec<[i64; 2]>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape {
        Flat([i64; 2]),
        Nested(Vec<[i64; 2]>),
    }
    Ok(match Shape::deserialize(deserializer)? {
        Shape::Flat(pair) => vec![pair],
        Shape::Nested(list) => list,
    })
}
