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

//! GLM MoE DSA (GLM5) model implementation
//!
//! A thin wrapper around DeepSeek V3.2 with GLM-specific config mapping.
//! The key difference is that GLM MoE DSA uses `rope_parameters` dict
//! instead of separate `rope_scaling` and `rope_theta` fields.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm_moe_dsa.py

use crate::models::deepseek_v32::{self, DeepSeekV32Model};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default)]
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
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<deepseek_v32::Quantization>,

    // GLM-specific: rope_parameters dict that contains rope_theta and scaling info
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, serde_json::Value>>,

    // These may be provided directly or derived from rope_parameters
    #[serde(default)]
    pub rope_scaling: Option<deepseek_v32::RopeScaling>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // GLM-specific fields (unused by DSV32 but present in config)
    #[serde(default)]
    pub index_head_dim: Option<usize>,
    #[serde(default)]
    pub index_n_heads: Option<usize>,
    #[serde(default)]
    pub index_topk: Option<usize>,
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_kv_lora_rank() -> usize {
    512
}
fn default_q_lora_rank() -> usize {
    1536
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
    "sigmoid".to_string()
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
    8
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    163840
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    /// Convert GLM MoE DSA config to DeepSeek V3.2 config.
    /// Extracts rope_theta from rope_parameters and maps rope_parameters → rope_scaling.
    pub fn to_dsv32_args(&self) -> deepseek_v32::ModelArgs {
        // Extract rope_theta from rope_parameters if available
        let rope_theta = self
            .rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(self.rope_theta);

        // Convert rope_parameters to RopeScaling
        let rope_scaling = if let Some(ref rp) = self.rope_parameters {
            Some(deepseek_v32::RopeScaling {
                scaling_type: rp
                    .get("type")
                    .or_else(|| rp.get("rope_type"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                factor: rp.get("factor").and_then(|v| v.as_f64()).map(|v| v as f32),
                mscale_all_dim: rp
                    .get("mscale_all_dim")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
            })
        } else {
            self.rope_scaling.clone()
        };

        deepseek_v32::ModelArgs {
            model_type: self.model_type.clone(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            moe_intermediate_size: self.moe_intermediate_size,
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
            rope_theta,
            rope_scaling,
            attention_bias: self.attention_bias,
            tie_word_embeddings: self.tie_word_embeddings,
            quantization: self.quantization.clone(),
        }
    }
}

// GLM MoE DSA Model (wraps DeepSeekV32Model).
pub struct GlmMoeDsaModel {
    inner: DeepSeekV32Model,
}

impl GlmMoeDsaModel {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let dsv32_args = args.to_dsv32_args();
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = DeepSeekV32Model::sanitize_weights_with_args(weights, &dsv32_args);
        let inner = DeepSeekV32Model::from_weights(&weights, &dsv32_args)?;

        Ok((Self { inner }, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let dsv32_args = args.to_dsv32_args();
        let inner = DeepSeekV32Model::from_weights(weights, &dsv32_args)?;
        Ok(Self { inner })
    }
}

impl LanguageModel for GlmMoeDsaModel {
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
        vec![151329, 151336, 151338] // GLM EOS tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_args() -> ModelArgs {
        ModelArgs {
            model_type: "glm_moe_dsa".to_string(),
            vocab_size: 10,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            moe_intermediate_size: 64,
            n_shared_experts: Some(1),
            n_routed_experts: Some(8),
            routed_scaling_factor: 1.25,
            kv_lora_rank: 128,
            q_lora_rank: 256,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            qk_nope_head_dim: 128,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            n_group: 2,
            topk_group: 1,
            num_experts_per_tok: 4,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            tie_word_embeddings: true,
            quantization: None,
            rope_parameters: None,
            rope_scaling: None,
            rope_theta: 10000.0,
            index_head_dim: None,
            index_n_heads: None,
            index_topk: None,
        }
    }

    #[test]
    fn glm_moe_dsa_prefers_rope_parameters_over_direct_fields() {
        let mut args = sample_args();
        let mut rope_parameters = HashMap::new();
        rope_parameters.insert("rope_theta".to_string(), serde_json::json!(500000.0));
        rope_parameters.insert("type".to_string(), serde_json::json!("yarn"));
        rope_parameters.insert("factor".to_string(), serde_json::json!(32.0));
        rope_parameters.insert("mscale_all_dim".to_string(), serde_json::json!(1.5));
        args.rope_parameters = Some(rope_parameters);
        args.rope_theta = 1234.0;
        args.rope_scaling = Some(deepseek_v32::RopeScaling {
            scaling_type: Some("ignored".to_string()),
            factor: Some(2.0),
            mscale_all_dim: Some(3.0),
        });

        let mapped = args.to_dsv32_args();

        assert_eq!(mapped.rope_theta, 500000.0);
        let rope_scaling = mapped.rope_scaling.expect("rope scaling");
        assert_eq!(rope_scaling.scaling_type.as_deref(), Some("yarn"));
        assert_eq!(rope_scaling.factor, Some(32.0));
        assert_eq!(rope_scaling.mscale_all_dim, Some(1.5));
    }

    #[test]
    fn glm_moe_dsa_falls_back_to_direct_rope_fields() {
        let mut args = sample_args();
        args.rope_theta = 320000.0;
        args.rope_scaling = Some(deepseek_v32::RopeScaling {
            scaling_type: Some("linear".to_string()),
            factor: Some(8.0),
            mscale_all_dim: None,
        });

        let mapped = args.to_dsv32_args();

        assert_eq!(mapped.rope_theta, 320000.0);
        let rope_scaling = mapped.rope_scaling.expect("rope scaling");
        assert_eq!(rope_scaling.scaling_type.as_deref(), Some("linear"));
        assert_eq!(rope_scaling.factor, Some(8.0));
        assert_eq!(rope_scaling.mscale_all_dim, None);
    }
}
