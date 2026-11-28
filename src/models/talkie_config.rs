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

//! Talkie `config.json` parsing and validation.

use serde::Deserialize;

/// Talkie `config.json`. Every optional key defaults to the 13B value used by
/// `mlx_lm/models/talkie.py`.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Talkie calls `F.rms_norm` without an eps, so the reduction uses the
    /// fp32 machine epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);

/// `eos_token_id` may be a single int or a list of ints.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "talkie".to_string()
}
fn default_hidden_size() -> usize {
    5120
}
fn default_num_hidden_layers() -> usize {
    40
}
fn default_num_attention_heads() -> usize {
    40
}
fn default_head_dim() -> usize {
    128
}
fn default_intermediate_size() -> usize {
    13696
}
fn default_vocab_size() -> usize {
    65540
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rms_norm_eps() -> f32 {
    f32::EPSILON
}

/// `<|endoftext|>` and `<|end|>`: the stop ids the published packs declare,
/// used when the config omits `eos_token_id`.
pub const TALKIE_EOS_TOKEN_IDS: [i32; 2] = [65535, 65536];

/// Upper bounds on architecture scalars read from an untrusted `config.json`.
/// Orders of magnitude above the 13B model.
const MAX_HIDDEN_SIZE: usize = 1 << 16;
const MAX_LAYERS: usize = 1024;
const MAX_HEADS: usize = 1024;
const MAX_HEAD_DIM: usize = 1024;
const MAX_VOCAB: usize = 1 << 22;
const MAX_INTERMEDIATE: usize = 1 << 20;

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![*id],
            Some(EosTokenId::Multiple(ids)) if !ids.is_empty() => ids.clone(),
            _ => TALKIE_EOS_TOKEN_IDS.to_vec(),
        }
    }

    /// Reject impossible scalars before any of them sizes an allocation,
    /// indexes a table, or divides.
    pub fn validate(&self) -> Result<(), String> {
        let bounded = |name: &str, v: usize, max: usize| {
            if v == 0 || v > max {
                Err(format!("talkie config: {name} = {v} is outside 1..={max}"))
            } else {
                Ok(())
            }
        };
        bounded("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        bounded("num_hidden_layers", self.num_hidden_layers, MAX_LAYERS)?;
        bounded("num_attention_heads", self.num_attention_heads, MAX_HEADS)?;
        bounded("head_dim", self.head_dim, MAX_HEAD_DIM)?;
        bounded("vocab_size", self.vocab_size, MAX_VOCAB)?;
        bounded(
            "intermediate_size",
            self.intermediate_size,
            MAX_INTERMEDIATE,
        )?;
        if self.head_dim < 2 || !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "talkie config: head_dim {} must be even and at least 2 for RoPE",
                self.head_dim
            ));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(format!(
                "talkie config: rope_theta {} must be finite and positive",
                self.rope_theta
            ));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return Err(format!(
                "talkie config: rms_norm_eps {} must be finite and positive",
                self.rms_norm_eps
            ));
        }
        Ok(())
    }
}
