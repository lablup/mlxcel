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

use serde::Deserialize;

fn hidden_size() -> usize {
    1536
}
fn input_feat_size() -> usize {
    128
}
fn vocab_size() -> usize {
    128
}
fn vocab_offset() -> i32 {
    262_272
}
fn rms_norm_eps() -> f32 {
    1e-6
}
fn gradient_clipping() -> f32 {
    1e10
}
fn chunk_size() -> usize {
    12
}
fn context_left() -> usize {
    13
}
fn attention_logit_cap() -> f32 {
    50.0
}
fn attention_heads() -> usize {
    8
}
fn hidden_layers() -> usize {
    12
}
fn conv_kernel_size() -> usize {
    5
}
fn reduction_factor() -> usize {
    4
}
fn residual_weight() -> f32 {
    0.5
}
fn conv_channels() -> Vec<usize> {
    vec![128, 32]
}
fn group_norm_eps() -> f32 {
    1e-3
}
fn conv_kernel_sizes() -> Vec<[usize; 2]> {
    vec![[3, 3], [3, 3]]
}
fn conv_stride_sizes() -> Vec<[usize; 2]> {
    vec![[2, 2], [2, 2]]
}

/// Official `Gemma3nAudioConfig` fields from Transformers commit
/// `181beb3ba4c47098ed8cbc97ee250d1d45ae0107`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Gemma3nAudioConfig {
    pub vocab_size: usize,
    pub vocab_offset: i32,
    pub input_feat_size: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
    pub gradient_clipping: f32,
    pub conf_attention_chunk_size: usize,
    pub conf_attention_context_left: usize,
    pub conf_attention_context_right: usize,
    pub conf_attention_logit_cap: f32,
    pub conf_num_attention_heads: usize,
    pub conf_num_hidden_layers: usize,
    pub conf_conv_kernel_size: usize,
    pub conf_reduction_factor: usize,
    pub conf_residual_weight: f32,
    pub sscp_conv_channel_size: Vec<usize>,
    pub sscp_conv_group_norm_eps: f32,
    pub sscp_conv_kernel_size: Vec<[usize; 2]>,
    pub sscp_conv_stride_size: Vec<[usize; 2]>,
}

impl Default for Gemma3nAudioConfig {
    fn default() -> Self {
        Self {
            vocab_size: vocab_size(),
            vocab_offset: vocab_offset(),
            input_feat_size: input_feat_size(),
            hidden_size: hidden_size(),
            rms_norm_eps: rms_norm_eps(),
            gradient_clipping: gradient_clipping(),
            conf_attention_chunk_size: chunk_size(),
            conf_attention_context_left: context_left(),
            conf_attention_context_right: 0,
            conf_attention_logit_cap: attention_logit_cap(),
            conf_num_attention_heads: attention_heads(),
            conf_num_hidden_layers: hidden_layers(),
            conf_conv_kernel_size: conv_kernel_size(),
            conf_reduction_factor: reduction_factor(),
            conf_residual_weight: residual_weight(),
            sscp_conv_channel_size: conv_channels(),
            sscp_conv_group_norm_eps: group_norm_eps(),
            sscp_conv_kernel_size: conv_kernel_sizes(),
            sscp_conv_stride_size: conv_stride_sizes(),
        }
    }
}

impl Gemma3nAudioConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.conf_num_attention_heads == 0
            || !self
                .hidden_size
                .is_multiple_of(self.conf_num_attention_heads)
        {
            return Err("Gemma3n audio hidden size must be divisible by the head count".into());
        }
        if self.conf_attention_chunk_size == 0
            || self.conf_reduction_factor == 0
            || self.conf_conv_kernel_size == 0
        {
            return Err(
                "Gemma3n audio chunk, reduction, and convolution sizes must be positive".into(),
            );
        }
        if self.sscp_conv_channel_size.len() != 2
            || self.sscp_conv_kernel_size.len() != 2
            || self.sscp_conv_stride_size.len() != 2
        {
            return Err("Gemma3n audio SSCP requires exactly two convolution stages".into());
        }
        if self.input_feat_size == 0 || self.vocab_size == 0 {
            return Err("Gemma3n audio feature and vocabulary sizes must be positive".into());
        }
        if self.sscp_conv_channel_size.contains(&0)
            || self
                .sscp_conv_kernel_size
                .iter()
                .flatten()
                .any(|size| *size == 0)
            || self
                .sscp_conv_stride_size
                .iter()
                .flatten()
                .any(|size| *size == 0)
        {
            return Err("Gemma3n audio SSCP dimensions and strides must be positive".into());
        }
        let mut frequency = self.input_feat_size;
        for (kernel, stride) in self
            .sscp_conv_kernel_size
            .iter()
            .zip(&self.sscp_conv_stride_size)
        {
            if frequency + 2 < kernel[1] {
                return Err("Gemma3n audio SSCP frequency kernel exceeds padded input".into());
            }
            frequency = (frequency + 2 - kernel[1]) / stride[1] + 1;
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.sscp_conv_group_norm_eps.is_finite()
            || self.sscp_conv_group_norm_eps <= 0.0
            || !self.gradient_clipping.is_finite()
            || self.gradient_clipping <= 0.0
            || !self.conf_attention_logit_cap.is_finite()
            || self.conf_attention_logit_cap <= 0.0
            || !self.conf_residual_weight.is_finite()
        {
            return Err("Gemma3n audio normalization and clipping values must be finite".into());
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.conf_num_attention_heads
    }

    pub fn max_past_horizon(&self) -> usize {
        self.conf_attention_context_left.saturating_sub(1)
    }

    pub fn context_size(&self) -> usize {
        self.conf_attention_chunk_size + self.max_past_horizon() + self.conf_attention_context_right
    }

    pub fn time_stride_product(&self) -> usize {
        self.sscp_conv_stride_size
            .iter()
            .map(|pair| pair[0])
            .product()
    }
}
