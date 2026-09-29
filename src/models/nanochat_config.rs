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

//! nanochat `config.json` parsing and validation.

use serde::Deserialize;

/// nanochat `config.json`. Every optional key defaults to the d20 value.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Defaults to `4 * hidden_size` when absent.
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Output logit softcap. Outer `None` = key absent (default 15.0),
    /// `Some(None)` = explicit `null` (softcap disabled). Read through
    /// [`ModelArgs::soft_cap`].
    #[serde(default, deserialize_with = "present_or_null")]
    pub logits_soft_cap: Option<Option<f32>>,
    /// Alias spelling. Kept as a separate field because serde rejects a
    /// config that carries both an alias and its primary key, and the MLX
    /// conversions write both.
    #[serde(default, deserialize_with = "present_or_null")]
    pub logits_softcap: Option<Option<f32>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

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
    "nanochat".to_string()
}
fn default_hidden_size() -> usize {
    1280
}
fn default_num_hidden_layers() -> usize {
    20
}
fn default_num_attention_heads() -> usize {
    10
}
fn default_vocab_size() -> usize {
    65536
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
/// Distinguishes an absent key (`None`) from an explicit `null` (`Some(None)`).
fn present_or_null<'de, D>(d: D) -> Result<Option<Option<f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<f32>::deserialize(d).map(Some)
}

/// `<|assistant_end|>`: the end-of-turn token used when the config omits one.
pub const NANOCHAT_EOS_TOKEN_ID: i32 = 65531;

/// Upper bounds on architecture scalars read from an untrusted `config.json`.
/// Orders of magnitude above the d32 model (2048 wide, 32 layers).
const MAX_HIDDEN_SIZE: usize = 1 << 16;
const MAX_LAYERS: usize = 1024;
const MAX_HEADS: usize = 1024;
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

    /// Effective softcap: an explicit value or `null` wins over the 15.0
    /// default; `logits_soft_cap` wins over its alias.
    pub fn soft_cap(&self) -> Option<f32> {
        self.logits_soft_cap
            .or(self.logits_softcap)
            .unwrap_or(Some(15.0))
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![*id],
            Some(EosTokenId::Multiple(ids)) if !ids.is_empty() => ids.clone(),
            _ => vec![NANOCHAT_EOS_TOKEN_ID],
        }
    }

    /// Reject impossible scalars before any of them sizes an allocation,
    /// indexes a table, or divides.
    pub fn validate(&self) -> Result<(), String> {
        let bounded = |name: &str, v: usize, max: usize| {
            if v == 0 || v > max {
                Err(format!(
                    "nanochat config: {name} = {v} is outside 1..={max}"
                ))
            } else {
                Ok(())
            }
        };
        bounded("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        bounded("num_hidden_layers", self.num_hidden_layers, MAX_LAYERS)?;
        bounded("num_attention_heads", self.num_attention_heads, MAX_HEADS)?;
        bounded("num_key_value_heads", self.num_key_value_heads, MAX_HEADS)?;
        bounded("vocab_size", self.vocab_size, MAX_VOCAB)?;
        bounded(
            "intermediate_size",
            self.intermediate_size(),
            MAX_INTERMEDIATE,
        )?;
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "nanochat config: hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.num_key_value_heads != self.num_attention_heads {
            return Err(format!(
                "nanochat config: num_key_value_heads {} must equal num_attention_heads {} \
                 (multi-head attention only)",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        let head_dim = self.head_dim();
        if head_dim < 2 || !head_dim.is_multiple_of(2) {
            return Err(format!(
                "nanochat config: head_dim {head_dim} must be even and at least 2 for RoPE"
            ));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(format!(
                "nanochat config: rope_theta {} must be finite and positive",
                self.rope_theta
            ));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return Err(format!(
                "nanochat config: rms_norm_eps {} must be finite and positive",
                self.rms_norm_eps
            ));
        }
        if let Some(cap) = self.soft_cap()
            && !(cap.is_finite() && cap > 0.0)
        {
            return Err(format!(
                "nanochat config: logits_soft_cap {cap} must be finite and positive or null"
            ));
        }
        Ok(())
    }
}
