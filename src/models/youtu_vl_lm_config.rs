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

//! Youtu-VL language-model configuration types.
//!
//! Lifted out of `youtu_vl_lm.rs` so the runtime module stays under the
//! 500-line soft target and so the loader can depend on the pure data types
//! without pulling in the full language-model implementation.

use serde::Deserialize;
use std::collections::HashMap;

use crate::models::deepseek_v3;

/// Youtu-VL text-side configuration.
///
/// Mirrors https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/config.py (TextConfig).
/// MoE-related fields are kept for forward compatibility with possible larger
/// variants but are not exercised by the standard `youtu_vl` checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct YoutuTextConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default = "default_true")]
    pub rope_traditional: bool,

    #[serde(default = "default_true")]
    pub rope_interleave: bool,

    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    // MoE fields (kept for future variants; standard Youtu-VL leaves these unset).
    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,

    #[serde(default = "default_one")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_one")]
    pub n_group: usize,

    #[serde(default = "default_one")]
    pub topk_group: usize,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_one")]
    pub moe_layer_freq: usize,

    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "youtu_vl".to_string()
}
fn default_max_position_embeddings() -> usize {
    32_768
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    500_000.0
}
fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}

impl YoutuTextConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Build the carrier `DeepSeekV3Config` used to construct an
    /// MLA `DeepSeekV3Attention` from shared weights. Only the fields read by
    /// `DeepSeekV3Attention::from_weights` and `get_attention_scale` are
    /// populated meaningfully; MoE-related defaults are kept neutral.
    pub(super) fn to_deepseek_v3_config(&self) -> deepseek_v3::DeepSeekV3Config {
        deepseek_v3::DeepSeekV3Config {
            model_type: "deepseek_v3".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size.unwrap_or(self.intermediate_size),
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            n_shared_experts: self.n_shared_experts,
            n_routed_experts: self.n_routed_experts,
            routed_scaling_factor: self.routed_scaling_factor,
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
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
            quantization: self
                .quantization
                .as_ref()
                .map(|q| deepseek_v3::QuantizationConfig {
                    group_size: q.group_size,
                    bits: q.bits,
                }),
        }
    }
}
