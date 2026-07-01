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

//! DeepSeek-V4 backbone.
//!
//! DeepSeek-V4 is the next-generation DeepSeek MoE. The full architecture adds
//! two features on top of the DeepSeek-V3 backbone:
//! - HiSA (hierarchical sparse attention), and
//! - a shared-expert SwiGLU clamped to a `swiglu_limit`.
//!
//! This module ports the **backbone only**: MLA (Multi-head Latent Attention),
//! group-limited MoE routing, the unclamped shared-expert `DenseMLP`, and the
//! multi-token-prediction (MTP) layer handling, all reused verbatim from
//! [`crate::models::deepseek_v3`]. The two V4-specific features above are tracked
//! as separate follow-up issues and are intentionally NOT implemented here.
//!
//! Concretely, [`DeepSeekV4Model`] is a thin wrapper around
//! [`DeepSeekV3Model`]: the V4 config is parsed independently (so the follow-up
//! issues can extend it without touching V3), then mapped to a
//! [`DeepSeekV3Config`] via [`DeepSeekV4Config::to_dsv3_config`]. This mirrors the
//! [`crate::models::glm_moe_dsa`] wrapper, which maps GLM-MoE-DSA onto DeepSeek
//! V3.2 the same way.
//!
//! Reference: `references/mlx-vlm/mlx_vlm/models/deepseek_v4/` (Python). The
//! reference's novel attention/HyperConnection/HiSA stack is out of scope for
//! this backbone port per the issue directive; only the config surface and the
//! V3-equivalent MLA + MoE + shared-expert semantics are reproduced here.

#[cfg(test)]
#[path = "deepseek_v4_tests.rs"]
mod deepseek_v4_tests;

use crate::models::deepseek_v3::{DeepSeekV3Config, DeepSeekV3Model, QuantizationConfig};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
//
// Field defaults track `references/mlx-vlm/mlx_vlm/models/deepseek_v4/config.py`.
// The MLA-latent dims (`kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`) are not
// part of the V4 reference config, which uses a different attention parameterization;
// the backbone reuses V3 MLA, so those fields fall back to the V3 defaults and are
// honored when a checkpoint supplies them.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV4Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: usize,
    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: usize,
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: usize,
    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    // V4-specific fields, parsed for forward-compatibility with the deferred
    // follow-up features. They are NOT consumed by the backbone (see the module
    // docs): `head_dim`/`num_nextn_predict_layers` describe the HiSA attention
    // and MTP drafters, and `swiglu_limit` is the shared-expert clamp. Keeping
    // them here means a genuine `deepseek_v4` config parses cleanly and the
    // follow-up issues can act on already-parsed values.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default = "default_num_nextn_predict_layers")]
    pub num_nextn_predict_layers: usize,
}

fn default_moe_intermediate_size() -> usize {
    2048
}
fn default_routed_scaling_factor() -> f32 {
    1.5
}
fn default_kv_lora_rank() -> usize {
    512
}
fn default_q_lora_rank() -> usize {
    1024
}
fn default_qk_rope_head_dim() -> usize {
    64
}
fn default_v_head_dim() -> usize {
    128
}
fn default_qk_nope_head_dim() -> usize {
    128
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_scoring_func() -> String {
    "sqrtsoftplus".to_string()
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    6
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    1048576
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_head_dim() -> usize {
    512
}
fn default_swiglu_limit() -> f32 {
    10.0
}
fn default_num_nextn_predict_layers() -> usize {
    1
}

impl DeepSeekV4Config {
    /// Map the V4 config onto the DeepSeek-V3 backbone config.
    ///
    /// Every field the V3 MLA + group-limited MoE + shared-expert `DenseMLP`
    /// needs is threaded through directly. The V4-only fields (`head_dim`,
    /// `swiglu_limit`, `num_nextn_predict_layers`) have no V3 counterpart and are
    /// dropped here: HiSA and the shared-expert clamp are deferred, and V3's
    /// weight sanitizer already drops the trailing MTP layer.
    pub fn to_dsv3_config(&self) -> DeepSeekV3Config {
        DeepSeekV3Config {
            model_type: self.model_type.clone(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            n_shared_experts: self.n_shared_experts,
            n_routed_experts: self.n_routed_experts,
            routed_scaling_factor: self.routed_scaling_factor,
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            topk_method: self.topk_method.clone(),
            scoring_func: self.scoring_func.clone(),
            norm_topk_prob: self.norm_topk_prob,
            n_group: self.n_group,
            topk_group: self.topk_group,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_layer_freq: self.moe_layer_freq,
            first_k_dense_replace: self.first_k_dense_replace,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling.clone(),
            attention_bias: self.attention_bias,
            quantization: self.quantization.clone(),
        }
    }
}

/// DeepSeek-V4 backbone. Thin wrapper over [`DeepSeekV3Model`]; see module docs.
pub struct DeepSeekV4Model {
    inner: DeepSeekV3Model,
}

impl DeepSeekV4Model {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, DeepSeekV4Config), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: DeepSeekV4Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let v3_args = args.to_dsv3_config();
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = DeepSeekV3Model::sanitize_weights_with_args(weights, &v3_args);
        let inner = DeepSeekV3Model::from_weights(&weights, &v3_args)?;

        Ok((Self { inner }, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &DeepSeekV4Config) -> Result<Self, String> {
        let v3_args = args.to_dsv3_config();
        let inner = DeepSeekV3Model::from_weights(weights, &v3_args)?;
        Ok(Self { inner })
    }
}

impl LanguageModel for DeepSeekV4Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.inner.forward(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.inner.make_caches()
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.inner)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }
}
