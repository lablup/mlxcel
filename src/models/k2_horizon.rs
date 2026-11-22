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

//! IFM K2-Horizon dense text models (`k2_horizon`, `K2HorizonForCausalLM`).
//!
//! Reference: mlx-vlm's `mlx_vlm/models/k2_horizon` (Blaizzy/mlx-vlm#2153) and
//! the HF `modeling_k2_horizon.py` shipped with each checkpoint.
//!
//! The 0.9B checkpoint is a plain Llama-shaped decoder: GQA, SwiGLU, RMSNorm
//! before attention and before the MLP, an untied head, Llama's weight key
//! names, and YaRN RoPE. It therefore runs on [`Llama3Model`] unchanged; this
//! module only translates and screens the config.
//!
//! # RoPE
//!
//! K2-Horizon writes RoPE under the transformers 5 `rope_parameters` block
//! (`{"rope_type": "yarn", "factor": 16, ...}` on 0.9B, `{"rope_theta": 1e7,
//! "rope_type": "default"}` with a null top-level `rope_theta` on 7B). The shared
//! lift ([`mlxcel_core::impl_rope_lifting_deserialize!`]) copies that block into
//! the flat `rope_theta` / `rope_scaling` keys this struct reads. The block's
//! `attention_factor` (`0.1 * ln(factor) + 1`) and `truncate: true` are not read:
//! the shared YaRN path computes the same magnitude from `mscale 1` /
//! `mscale_all_dim 0` and the same floor/ceil correction range.
//!
//! # What is refused
//!
//! The HF modeling supports several features the shared decoder does not.
//! [`K2HorizonConfig::validate`] refuses each by name rather than loading a
//! checkpoint that would decode wrongly without any error:
//!
//! - `layernorm_num_groups != 1`: `K2HorizonRMSNorm` normalizes each of
//!   `n_groups` contiguous slices separately (3.7B / 7B / 32B use 2 / 4 / 4).
//! - `query_key_norm: true`: per-head Q/K RMSNorm.
//! - `num_experts > 0` or `mova_num_experts > 0`: the MoE and MoVA variants.
//! - a non-null `attention_gate_func`: gated attention output.
//! - `use_sliding_window: true`.
//! - `rope_head_dim` different from `head_dim`: partial rotary, where HF splits
//!   Q/K at `rope_head_dim` and rotates only the first slice.

use mlxcel_core::weights::WeightMap;
use serde::Deserialize;
use std::path::Path;

use super::llama3::{self, EosTokenId, Llama3Model, RopeScaling};

/// K2-Horizon `config.json`, restricted to the fields the dense path reads or
/// must screen.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct K2HorizonConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,

    /// Filled from `rope_parameters.rope_theta` by the lift when the flat key is
    /// absent or null. `None` only when neither spells a base.
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Filled from a non-default `rope_parameters` block by the lift.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub quantization: Option<llama3::Quantization>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,

    // Fields screened by `validate`; none of them reaches the decoder.
    #[serde(default = "default_layernorm_num_groups")]
    pub layernorm_num_groups: usize,
    #[serde(default)]
    pub query_key_norm: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub mova_num_experts: usize,
    #[serde(default)]
    pub attention_gate_func: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "null_as_false")]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_head_dim: Option<usize>,
}

mlxcel_core::impl_rope_lifting_deserialize!(K2HorizonConfig);

fn default_model_type() -> String {
    "k2_horizon".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_layernorm_num_groups() -> usize {
    1
}
/// The base mlx-vlm's `ModelConfig` defaults to, and the one the larger
/// K2-Horizon variants declare under `rope_parameters`.
const DEFAULT_ROPE_THETA: f32 = 1e7;

fn null_as_false<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(false))
}

impl K2HorizonConfig {
    /// The RoPE base after the `rope_parameters` lift.
    pub fn rope_base(&self) -> f32 {
        self.rope_theta.unwrap_or(DEFAULT_ROPE_THETA)
    }

    /// Head width: the declared `head_dim`, else `hidden_size / heads`.
    ///
    /// Only valid after [`Self::validate`], which rejects zero heads.
    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Refuse a config that names a K2-Horizon feature the shared Llama
    /// decoder does not implement, or a geometry it cannot run.
    pub fn validate(&self) -> Result<(), String> {
        if self.layernorm_num_groups != 1 {
            return Err(format!(
                "K2-Horizon layernorm_num_groups ({}) is not supported: grouped RMSNorm \
                 (K2HorizonRMSNorm with n_groups > 1, used by the 3.7B / 7B / 32B variants) \
                 normalizes each slice of the hidden vector separately, and the shared decoder \
                 only implements layernorm_num_groups = 1",
                self.layernorm_num_groups
            ));
        }
        if self.query_key_norm {
            return Err(
                "K2-Horizon query_key_norm = true is not supported: the shared decoder has no \
                 per-head Q/K RMSNorm"
                    .to_string(),
            );
        }
        if self.num_experts > 0 {
            return Err(format!(
                "K2-Horizon num_experts ({}) is not supported: the MoE variants are not \
                 implemented, only the dense checkpoints",
                self.num_experts
            ));
        }
        if self.mova_num_experts > 0 {
            return Err(format!(
                "K2-Horizon mova_num_experts ({}) is not supported: the MoVA (mixture of value \
                 attention) variants are not implemented",
                self.mova_num_experts
            ));
        }
        if let Some(gate) = self.attention_gate_func.as_ref().filter(|v| !v.is_null()) {
            return Err(format!(
                "K2-Horizon attention_gate_func ({gate}) is not supported: the shared decoder \
                 has no gated attention output"
            ));
        }
        if self.use_sliding_window {
            return Err(
                "K2-Horizon use_sliding_window = true is not supported: the shared decoder \
                 attends over the full context"
                    .to_string(),
            );
        }
        self.validate_geometry()?;
        if let Some(rope_head_dim) = self.rope_head_dim {
            let head_dim = self.resolved_head_dim();
            if rope_head_dim != head_dim {
                return Err(format!(
                    "K2-Horizon rope_head_dim ({rope_head_dim}) differs from head_dim \
                     ({head_dim}): partial rotary is not supported, the shared decoder rotates \
                     the full head"
                ));
            }
        }
        let base = self.rope_base();
        if !base.is_finite() || base <= 0.0 {
            return Err(format!(
                "K2-Horizon rope_theta ({base}) must be a finite positive number"
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "K2-Horizon rms_norm_eps ({}) must be a finite positive number",
                self.rms_norm_eps
            ));
        }
        if let Some(q) = self.quantization.as_ref() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("K2-Horizon config.json: {e}"))?;
        }
        Ok(())
    }

    /// Reject head counts and widths that would divide by zero or reach an MLX
    /// kernel with an odd rotary dimension.
    fn validate_geometry(&self) -> Result<(), String> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("intermediate_size", self.intermediate_size),
            ("num_attention_heads", self.num_attention_heads),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return Err(format!("K2-Horizon {name} must be at least 1"));
            }
        }
        if self.num_hidden_layers > MAX_NUM_HIDDEN_LAYERS {
            return Err(format!(
                "K2-Horizon num_hidden_layers ({}) exceeds {MAX_NUM_HIDDEN_LAYERS}",
                self.num_hidden_layers
            ));
        }
        let kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        if kv_heads == 0 || !self.num_attention_heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "K2-Horizon num_attention_heads ({}) must be a positive multiple of \
                 num_key_value_heads ({kv_heads})",
                self.num_attention_heads
            ));
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "K2-Horizon hidden_size ({}) must be divisible by num_attention_heads ({}) when \
                 head_dim is not declared",
                self.hidden_size, self.num_attention_heads
            ));
        }
        let head_dim = self.resolved_head_dim();
        if head_dim < 2 || !head_dim.is_multiple_of(2) || head_dim > MAX_HEAD_DIM {
            return Err(format!(
                "K2-Horizon head_dim ({head_dim}) must be even and between 2 and {MAX_HEAD_DIM}"
            ));
        }
        Ok(())
    }

    /// Build the `llama3::ModelArgs` that drives the shared dense decoder.
    ///
    /// Written as a full struct literal on purpose (the pattern of
    /// `helium::ModelArgs::to_llama3_args`): a field added to
    /// [`llama3::ModelArgs`] fails this file to compile instead of silently
    /// defaulting for K2-Horizon.
    pub fn to_llama3_args(&self) -> llama3::ModelArgs {
        llama3::ModelArgs {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            head_dim: Some(self.resolved_head_dim()),
            num_key_value_heads: self.num_key_value_heads,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            rope_theta: self.rope_base(),
            rope_scaling: self.rope_scaling.clone(),
            max_position_embeddings: self.max_position_embeddings,
            quantization: self.quantization.clone(),
            quantization_config: self.quantization_config.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id.clone(),
            // HF builds the standard split-half rotation (`rotate_half`).
            rope_traditional: false,
            checkpoint_label: None,
        }
    }
}

/// Ceiling on `num_hidden_layers`, so an untrusted config cannot size the
/// layer `Vec` in [`Llama3Model::from_weights`] arbitrarily. The 32B uses 64.
const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
/// Ceiling on the head width, so `heads * head_dim` stays inside `i32`.
const MAX_HEAD_DIM: usize = 1024;

/// Parse and screen a K2-Horizon `config.json`.
pub fn parse_config(config_str: &str) -> Result<K2HorizonConfig, String> {
    let config: K2HorizonConfig = serde_json::from_str(config_str)
        .map_err(|e| format!("Failed to parse K2-Horizon config.json: {e}"))?;
    config.validate()?;
    Ok(config)
}

/// Load a K2-Horizon checkpoint directory onto the shared Llama decoder.
pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Llama3Model, K2HorizonConfig), String> {
    let model_dir = model_dir.as_ref();
    let config_str = std::fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| format!("Failed to read config.json: {e}"))?;
    let config = parse_config(&config_str)?;
    let weights = crate::models::load_text_weights(model_dir, None)?;
    let mut args = config.to_llama3_args();
    args.set_checkpoint_label(model_dir);
    let model = Llama3Model::from_weights(&weights, &args)?;
    Ok((model, config))
}

/// Build the shared Llama decoder from already-loaded weights (adapter path).
pub fn from_weights(weights: &WeightMap, config: &K2HorizonConfig) -> Result<Llama3Model, String> {
    config.validate()?;
    Llama3Model::from_weights(weights, &config.to_llama3_args())
}

#[cfg(test)]
#[path = "k2_horizon_tests.rs"]
mod tests;
