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

//! Mistral4 model implementation using mlxcel-core
//!
//! Key features:
//! - Multi-Latent Attention (MLA) with compressed KV projections (similar to DeepSeek-V3)
//! - Optional query LoRA compression (q_lora_rank)
//! - KV LoRA compression (kv_lora_rank) with single shared latent per layer
//! - Split rope vs non-rope head dimensions (qk_rope_head_dim / qk_nope_head_dim)
//! - Separate value head dimension (v_head_dim)
//! - MoE layers with SwitchGLU routing and shared experts
//! - Llama-4 position-dependent attention scaling

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use super::switch_layers::SwitchGLU;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Mistral4Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f32,

    // MLA parameters
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,

    // MoE parameters
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub first_k_dense_replace: usize,

    // Rope parameters
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "mistral4".to_string()
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_tie_word_embeddings() -> bool {
    false
}

impl Mistral4Config {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn rope_base(&self) -> f32 {
        // First try rope_parameters.rope_theta, then fall back to rope_theta field
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .or(self.rope_theta)
            .unwrap_or(10000.0)
    }

    pub fn llama4_scaling_beta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.get("llama_4_scaling_beta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0)
    }

    pub fn original_max_position_embeddings(&self) -> usize {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.max_position_embeddings)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace && self.n_routed_experts.unwrap_or(0) > 0
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

// Llama 4 Attention Scaling.
// Used by: Mistral4, Ministral3
fn get_llama4_attn_scale(
    size: i32,
    offset: i32,
    beta: f32,
    max_position_embeddings: usize,
) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let pos = (i + offset) as f32;
            let max_pos = max_position_embeddings as f32;
            1.0 + beta * (1.0 + (pos / max_pos).floor()).ln()
        })
        .collect()
}

// MLA Attention.
// Used by: Mistral4
pub struct Mistral4Attention {
    // Q projection: either direct or through LoRA
    pub q_proj: Option<UnifiedLinear>,
    pub q_a_proj: Option<UnifiedLinear>,
    pub q_a_layernorm: Option<RMSNorm>,
    pub q_b_proj: Option<UnifiedLinear>,

    // KV projection with compression
    pub kv_a_proj_with_mqa: UnifiedLinear,
    pub kv_a_layernorm: RMSNorm,
    pub kv_b_proj: UnifiedLinear,

    pub o_proj: UnifiedLinear,

    pub num_heads: i32,
    pub qk_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub qk_nope_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl Mistral4Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        attn_scale: &[f32],
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Query projection (optionally through LoRA)
        let q = if let Some(ref q_proj) = self.q_proj {
            q_proj.forward(x)
        } else {
            let q_a = self.q_a_proj.as_ref().unwrap().forward(x);
            let q_a_norm = self.q_a_layernorm.as_ref().unwrap().forward(&q_a);
            self.q_b_proj.as_ref().unwrap().forward(&q_a_norm)
        };

        // Reshape Q to [batch, seq, heads, qk_head_dim] then transpose
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.qk_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // KV projection: compressed representation + rope component
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);

        // Split into compressed latent and k_pe
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // k_pe is single-head (MQA for rope): [batch, seq, 1, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // Decompress KV through second projection
        let kv_latent = self.kv_a_layernorm.forward(&compressed);
        let kv = self.kv_b_proj.forward(&kv_latent);
        // Reshape: [batch, seq, num_heads * (qk_nope_head_dim + v_head_dim)]
        //       -> [batch, seq, num_heads, qk_nope_head_dim + v_head_dim]
        let kv = mlxcel_core::reshape(
            &kv,
            &[
                b,
                l,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ],
        );
        let kv = mlxcel_core::transpose_axes(&kv, &[0, 2, 1, 3]);
        let k_nope = slice_axis(&kv, -1, 0, self.qk_nope_head_dim);
        let values = slice_axis(&kv, -1, self.qk_nope_head_dim, -1);

        // Apply RoPE to positional components only
        let offset = cache.offset;
        let q_pe = mlxcel_core::fast_rope(
            &q_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );
        let k_pe = mlxcel_core::fast_rope(
            &k_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );

        // Broadcast k_pe from [B, 1, L, rope_dim] to [B, num_heads, L, rope_dim]
        let k_pe = mlxcel_core::broadcast_to(&k_pe, &[b, self.num_heads, l, self.qk_rope_head_dim]);

        // Concatenate nope and pe parts to form full keys
        let keys = mlxcel_core::concatenate(&k_nope, &k_pe, -1);

        // Update KV cache
        let (keys, values) = cache.update_and_fetch(keys, values);

        // Concatenate query nope and pe parts
        let queries = mlxcel_core::concatenate(&q_nope, &q_pe, -1);

        // Apply Llama-4 position-dependent attention scaling
        let scale_shape = vec![1, 1, l, 1];
        let scale_arr = mlxcel_core::from_slice_f32(attn_scale, &[l]);
        let scale_arr = mlxcel_core::reshape(&scale_arr, &scale_shape);
        let queries = mlxcel_core::multiply(&queries, &scale_arr);

        // Scaled dot-product attention
        let attn_out = if l > 1 {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &queries, &keys, &values, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&queries, &keys, &values, self.scale, 0.0, 0)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.v_head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Mistral4Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let qk_head_dim = args.qk_head_dim() as i32;

        // Query projection: direct or LoRA
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = if args.q_lora_rank.is_none() {
            let q = UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?;
            (Some(q), None, None, None)
        } else {
            let q_a = UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_a_proj"),
                group_size,
                bits,
            )?;
            let q_a_norm_w = get_weight_copy(weights, &format!("{prefix}.q_a_layernorm.weight"))?;
            let q_a_norm = RMSNorm::new(q_a_norm_w, 1e-6);
            let q_b = UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_b_proj"),
                group_size,
                bits,
            )?;
            (None, Some(q_a), Some(q_a_norm), Some(q_b))
        };

        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa"),
            group_size,
            bits,
        )?;
        let kv_a_norm_w = get_weight_copy(weights, &format!("{prefix}.kv_a_layernorm.weight"))?;
        let kv_a_layernorm = RMSNorm::new(kv_a_norm_w, args.rms_norm_eps);

        let kv_b_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.kv_b_proj"), group_size, bits)?;

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), group_size, bits)?;

        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            qk_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            scale: (qk_head_dim as f32).powf(-0.5),
            rope_base: args.rope_base(),
        })
    }
}

// Dense MLP.
// Used by: Mistral4 (dense layers and shared experts)
pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let gate_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.gate_proj"), group_size, bits)?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), group_size, bits)?;
        let down_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.down_proj"), group_size, bits)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// MoE Block.
// Used by: Mistral4
pub struct Mistral4MoE {
    pub gate: UnifiedLinear,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Option<DenseMLP>,
    pub top_k: i32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
}

impl Mistral4MoE {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Route tokens to experts
        let gates = self.gate.forward(x);
        let gates = mlxcel_core::softmax(&gates, -1);

        // Top-k expert selection via argpartition
        let neg_gates = mlxcel_core::negative(&gates);
        let part = mlxcel_core::argpartition(&neg_gates, self.top_k - 1, -1);
        let inds = slice_axis(&part, -1, 0, self.top_k);
        let scores = mlxcel_core::take_along_axis(&gates, &inds, -1);

        // Normalize scores
        let scores = if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&sum));
            let denom = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&scores, &denom)
        } else {
            scores
        };

        // Apply routed scaling factor
        let scale_val = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&scores),
        );
        let scores = mlxcel_core::multiply(&scores, &scale_val);

        // Dispatch to selected experts
        let y = self.switch_mlp.forward(x, &inds);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &y,
            &scores,
            mlxcel_core::array_dtype(x),
        );

        // Add shared expert output
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Mistral4Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.gate"), group_size, bits)?;

        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{prefix}.switch_mlp"), group_size, bits)?;

        let shared_experts = match args.n_shared_experts {
            Some(n) if n > 0 => {
                // Shared experts use a single DenseMLP with scaled intermediate size
                // (intermediate_size = moe_intermediate_size * n_shared_experts, set in config)
                Some(DenseMLP::from_weights(
                    weights,
                    &format!("{prefix}.shared_experts"),
                    group_size,
                    bits,
                )?)
            }
            _ => None,
        };

        Ok(Self {
            gate,
            switch_mlp,
            shared_experts,
            top_k: args.num_experts_per_tok as i32,
            norm_topk_prob: args.norm_topk_prob,
            routed_scaling_factor: args.routed_scaling_factor,
        })
    }
}

// MLP variant: Dense or MoE.
pub enum MLPType {
    Dense(DenseMLP),
    MoE(Mistral4MoE),
}

// Transformer Block.
pub struct Mistral4TransformerBlock {
    pub self_attn: Mistral4Attention,
    pub mlp: MLPType,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl Mistral4TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        attn_scale: &[f32],
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, attn_scale, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed),
            MLPType::MoE(moe) => moe.forward(&normed),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Mistral4Config,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let layer_prefix = format!("{prefix}model.layers.{layer_idx}");
        let group_size = args.group_size();
        let bits = args.bits();

        let self_attn =
            Mistral4Attention::from_weights(weights, args, &format!("{layer_prefix}.self_attn"))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            MLPType::MoE(Mistral4MoE::from_weights(
                weights,
                args,
                &format!("{layer_prefix}.mlp"),
            )?)
        } else {
            MLPType::Dense(DenseMLP::from_weights(
                weights,
                &format!("{layer_prefix}.mlp"),
                group_size,
                bits,
            )?)
        };

        let input_norm_w =
            get_weight_copy(weights, &format!("{layer_prefix}.input_layernorm.weight"))?;
        let post_attn_norm_w = get_weight_copy(
            weights,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_w, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_w, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Mistral4 Model.
pub struct Mistral4Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<Mistral4TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub config: Mistral4Config,
}

impl Mistral4Model {
    /// Shared transformer body: mask + scale + layer loop + norm + lm_head.
    fn transformer_body(
        &self,
        mut h: UniquePtr<MlxArray>,
        caches: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1];

        // Create causal mask for prefill
        let mask = if l > 1 {
            let offset = caches[0].offset;
            Some(create_causal_mask(l, offset))
        } else {
            None
        };

        // Compute Llama 4 attention scale
        let offset = caches[0].offset;
        let beta = self.config.llama4_scaling_beta();
        let attn_scale = if beta > 0.0 {
            get_llama4_attn_scale(
                l,
                offset,
                beta,
                self.config.original_max_position_embeddings(),
            )
        } else {
            vec![1.0; l as usize]
        };

        // Pass through layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(
                &h,
                &attn_scale,
                &mut caches[i],
                mask.as_ref().map(|m| m.as_ref().unwrap() as &MlxArray),
            );
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        self.transformer_body(h, caches)
    }

    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from a standalone directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Mistral4Config), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config: Mistral4Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;

        // Run weight sanitization for MoE expert weight splitting
        sanitize_weights(&mut weights, &config, "");

        let model = Self::from_weights(&weights, &config)?;
        Ok((model, config))
    }

    /// Load model from VLM wrapper directory (extracts text_config)
    pub fn load_from_text_config<P: AsRef<Path>>(
        model_dir: P,
    ) -> Result<(Self, Mistral4Config), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        let text_config = config
            .get("text_config")
            .ok_or("Missing text_config in VLM wrapper config")?;
        let args: Mistral4Config = serde_json::from_value(text_config.clone())
            .map_err(|e| format!("Failed to parse text_config: {e}"))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;

        // Sanitize with language_model. prefix for VLM weights
        sanitize_weights(&mut weights, &args, "language_model.");

        let model = Self::from_weights_with_prefix(&weights, &args, "language_model.")?;
        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &Mistral4Config) -> Result<Self, String> {
        Self::from_weights_with_prefix(weights, args, "")
    }

    /// Create model from loaded weights with optional prefix (for VLM wrappers)
    pub fn from_weights_with_prefix(
        weights: &WeightMap,
        args: &Mistral4Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{prefix}model.embed_tokens"),
            group_size,
            bits,
        )?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = Mistral4TransformerBlock::from_weights(weights, args, i, prefix)?;
            layers.push(layer);
        }

        let norm_w = get_weight_copy(weights, &format!("{prefix}model.norm.weight"))?;
        let norm = RMSNorm::new(norm_w, args.rms_norm_eps);

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}lm_head"),
                group_size,
                bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: args.clone(),
        })
    }
}

/// Sanitize weights for Mistral4 models.
/// Handles fused gate_up_proj splitting for MoE expert layers.
///
/// Used by: Mistral4
pub fn sanitize_weights(weights: &mut WeightMap, config: &Mistral4Config, prefix: &str) {
    let n_routed = match config.n_routed_experts {
        Some(n) if n > 0 => n,
        _ => return,
    };

    for l in 0..config.num_hidden_layers {
        if !config.is_moe_layer(l) {
            continue;
        }

        let layer_prefix = format!("{prefix}model.layers.{l}.mlp");

        // Split fused gate_up_proj: (n_experts, 2*intermediate, hidden)
        // -> gate_proj: (n_experts, intermediate, hidden)
        // -> up_proj: (n_experts, intermediate, hidden)
        let fused_key = format!("{layer_prefix}.experts.gate_up_proj");
        if let Some(gate_up) = weights.remove(&fused_key) {
            let shape = mlxcel_core::array_shape(&gate_up);
            let half = shape[1] / 2;
            let gate_proj = slice_axis(&gate_up, 1, 0, half);
            let up_proj = slice_axis(&gate_up, 1, half, -1);
            weights.insert(
                format!("{layer_prefix}.switch_mlp.gate_proj.weight"),
                gate_proj,
            );
            weights.insert(format!("{layer_prefix}.switch_mlp.up_proj.weight"), up_proj);
        }

        // Rename down_proj if needed
        let down_key = format!("{layer_prefix}.experts.down_proj");
        if let Some(down) = weights.remove(&down_key) {
            weights.insert(format!("{layer_prefix}.switch_mlp.down_proj.weight"), down);
        }

        // Stack individual expert weights if they exist (experts.N format)
        let first_key = format!("{layer_prefix}.experts.0.gate_proj.weight");
        if weights.contains_key(&first_key) {
            for m in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let check_key = format!("{layer_prefix}.experts.0.{m}.{k}");
                    if weights.contains_key(&check_key) {
                        let mut expert_arrays = Vec::new();
                        for e in 0..n_routed {
                            let key = format!("{layer_prefix}.experts.{e}.{m}.{k}");
                            if let Some(w) = weights.get(&key) {
                                expert_arrays.push(mlxcel_core::copy(w));
                            }
                        }
                        if !expert_arrays.is_empty() {
                            let stacked = mlxcel_core::utils::stack_arrays(&expert_arrays, 0);
                            let new_key = format!("{layer_prefix}.switch_mlp.{m}.{k}");
                            weights.insert(new_key, stacked);
                            for e in 0..n_routed {
                                let key = format!("{layer_prefix}.experts.{e}.{m}.{k}");
                                weights.remove(&key);
                            }
                        }
                    }
                }
            }
        }
    }

    // Remove unused precomputed rotary freqs
    let rotary_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("rotary_emb.inv_freq"))
        .cloned()
        .collect();
    for key in rotary_keys {
        weights.remove(&key);
    }

    // Remove weight_scale_inv / activation_scale keys (FP8 quantization artifacts)
    let scale_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("weight_scale_inv") || k.contains("activation_scale"))
        .cloned()
        .collect();
    for key in scale_keys {
        weights.remove(&key);
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

// LanguageModel trait implementation.
impl LanguageModel for Mistral4Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_caches_impl()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // Mistral EOS token
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // For VLM: use input_embeddings if provided, otherwise embed input_ids
        let h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };
        self.transformer_body(h, caches)
    }
}
