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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Deserialize an `AudioConfig` from `overrides`, leaving every field that
    /// is absent at its serde default.
    fn config(overrides: Value) -> AudioConfig {
        serde_json::from_value(overrides).expect("audio config should deserialize")
    }

    #[test]
    fn empty_config_yields_the_documented_default_geometry() {
        let cfg = config(json!({}));

        // The raw defaults the derived values are computed from.
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.attention_chunk_size, 12);
        assert_eq!(cfg.attention_context_left, 13);
        assert_eq!(cfg.attention_context_right, 0);

        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.max_past_horizon(), 12);
        // chunk (12) + past horizon (13 - 1) + right context (0).
        assert_eq!(cfg.context_size(), 24);
    }

    #[test]
    fn head_dim_is_hidden_size_over_attention_heads() {
        for (hidden_size, heads, expected) in [
            (1024, 8, 128),
            (512, 4, 128),
            (768, 12, 64),
            (256, 1, 256),
            (64, 64, 1),
        ] {
            let cfg = config(json!({
                "hidden_size": hidden_size,
                "num_attention_heads": heads,
            }));
            assert_eq!(
                cfg.head_dim(),
                expected,
                "hidden_size={hidden_size} heads={heads}"
            );
        }
    }

    #[test]
    fn max_past_horizon_is_context_left_minus_one_saturating_at_zero() {
        for (left, expected) in [(0, 0), (1, 0), (2, 1), (13, 12), (128, 127)] {
            let cfg = config(json!({ "attention_context_left": left }));
            assert_eq!(cfg.max_past_horizon(), expected, "left={left}");
        }
    }

    #[test]
    fn context_size_sums_chunk_past_horizon_and_right_context() {
        // (chunk, left, right, expected context_size)
        for (chunk, left, right, expected) in [
            (12, 13, 0, 24),
            // `attention_context_left == 0` saturates the past horizon to 0
            // instead of underflowing, leaving only chunk + right context.
            (12, 0, 0, 12),
            // One left position is the query itself, so it adds no past horizon.
            (12, 1, 0, 12),
            (12, 2, 0, 13),
            (12, 13, 5, 29),
            (12, 0, 5, 17),
            (1, 1, 0, 1),
        ] {
            let cfg = config(json!({
                "attention_chunk_size": chunk,
                "attention_context_left": left,
                "attention_context_right": right,
            }));
            assert_eq!(
                cfg.context_size(),
                expected,
                "chunk={chunk} left={left} right={right}"
            );
        }
    }
}
