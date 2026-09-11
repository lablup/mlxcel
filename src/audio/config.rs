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

//! Gemma4 audio encoder configuration.
//!
//! Used by: Gemma4 audio encoder

use serde::Deserialize;

fn default_hidden_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    12
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_subsampling_conv_channels() -> Vec<usize> {
    vec![128, 32]
}
fn default_conv_kernel_size() -> usize {
    5
}
fn default_residual_weight() -> f32 {
    0.5
}
fn default_attention_chunk_size() -> usize {
    12
}
fn default_attention_context_left() -> usize {
    13
}
fn default_attention_context_right() -> usize {
    0
}
fn default_attention_logit_cap() -> f32 {
    50.0
}
fn default_attention_invalid_logits_value() -> f32 {
    -1e9
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_gradient_clipping() -> f32 {
    1e10
}
fn default_output_proj_dims() -> Option<usize> {
    Some(1536)
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_subsampling_conv_channels")]
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f32,
    #[serde(default = "default_attention_chunk_size")]
    pub attention_chunk_size: usize,
    #[serde(default = "default_attention_context_left")]
    pub attention_context_left: usize,
    #[serde(default = "default_attention_context_right")]
    pub attention_context_right: usize,
    #[serde(default = "default_attention_logit_cap")]
    pub attention_logit_cap: f32,
    #[serde(default = "default_attention_invalid_logits_value")]
    pub attention_invalid_logits_value: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f32,
    #[serde(default = "default_output_proj_dims")]
    pub output_proj_dims: Option<usize>,
}

impl AudioConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn max_past_horizon(&self) -> usize {
        self.attention_context_left.saturating_sub(1)
    }

    pub fn context_size(&self) -> usize {
        self.attention_chunk_size + self.max_past_horizon() + self.attention_context_right
    }
}
