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

//! Z1T `config.json` schema and its load-time validation.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Z1tQuantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Z1tConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_aft_kind")]
    pub aft_kind: String,
    #[serde(default = "default_aft_heads")]
    pub aft_heads: usize,
    #[serde(default = "default_aft_ksize")]
    pub aft_ksize: usize,
    #[serde(default = "default_dyt_alpha")]
    pub dyt_alpha: f32,
    /// Shared fixed sparse fan-in; `None` means dense.
    #[serde(default)]
    pub linear_fan_in: Option<usize>,
    #[serde(default)]
    pub attn_fan_in: Option<usize>,
    #[serde(default)]
    pub mlp_fan_in: Option<usize>,
    /// Wrap every internal linear as `tanh(Wx + b)`; the LM head is exempt.
    #[serde(default)]
    pub tanh_linear: bool,
    /// MLP activation is `tanh` instead of SiLU.
    #[serde(default)]
    pub tanh_mlp: bool,
    #[serde(default)]
    pub quantization: Option<Z1tQuantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_max_positions() -> usize {
    256
}
fn default_aft_kind() -> String {
    "conv".to_string()
}
fn default_aft_heads() -> usize {
    4
}
fn default_aft_ksize() -> usize {
    4
}
fn default_dyt_alpha() -> f32 {
    0.5
}

impl Z1tConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.aft_kind != "conv" {
            return Err(format!(
                "Z1T aft_kind {:?} is not supported; only \"conv\" is implemented",
                self.aft_kind
            ));
        }
        if self.vocab_size == 0 || self.hidden_size == 0 || self.num_hidden_layers == 0 {
            return Err(
                "Z1T vocab_size, hidden_size and num_hidden_layers must be positive".into(),
            );
        }
        if self.aft_heads == 0 || !self.hidden_size.is_multiple_of(self.aft_heads) {
            return Err(format!(
                "Z1T hidden_size {} must be a positive multiple of aft_heads {}",
                self.hidden_size, self.aft_heads
            ));
        }
        if self.aft_ksize == 0 || self.max_position_embeddings == 0 {
            return Err("Z1T aft_ksize and max_position_embeddings must be positive".into());
        }
        // Every derived width (qkv out 2D + heads, MLP hidden 4D) is an i32 shape.
        if self.hidden_size > (i32::MAX as usize) / 8 || self.vocab_size > i32::MAX as usize {
            return Err("Z1T hidden_size or vocab_size exceeds the supported range".into());
        }
        Ok(())
    }

    pub(crate) fn fan_in_attn(&self) -> Option<usize> {
        self.attn_fan_in.or(self.linear_fan_in)
    }

    pub(crate) fn fan_in_mlp(&self) -> Option<usize> {
        self.mlp_fan_in.or(self.linear_fan_in)
    }

    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }
}
