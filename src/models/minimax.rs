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

//! MiniMax-M2 MoE model implementation
//!
//! Implements the MiniMax architecture with sparse mixture of experts (MoE).
//! Key features:
//! - Sparse MoE with top-k expert selection (256 experts, top-8)
//! - Sigmoid scoring with e_score_correction_bias (not softmax)
//! - Q/K RMSNorm normalization on full projections
//! - Partial RoPE (rotary_dim < head_dim)
//! - Pre-stacked expert weights (switch_mlp format)

use crate::models::switch_layers::SwitchGLU;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_local_experts: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub head_dim: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_true")]
    pub use_qk_norm: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

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

    /// MiniMax quantized models use 8-bit quantization for MoE gates
    pub fn gate_bits(&self) -> i32 {
        if self.quantization.is_some() {
            8
        } else {
            self.bits()
        }
    }
}

// ============================================================================
// Attention with Q/K normalization
// ============================================================================

pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let mut q = self.q_proj.forward(x);
        let mut k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Apply Q/K normalization BEFORE reshape (on full flattened projections)
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q);
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k);
        }

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply partial RoPE (rotary_dim may be < head_dim)
        let q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            false, // traditional
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load Q/K normalization weights (optional)
        let (q_norm, k_norm) = if args.use_qk_norm {
            let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
            let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
            (
                Some(RMSNorm::new(q_norm_weight, args.rms_norm_eps)),
                Some(RMSNorm::new(k_norm_weight, args.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        let head_dim = args.head_dim as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rotary_dim as i32,
            rope_base: args.rope_theta,
        })
    }
}

// ============================================================================
// Sparse MoE Block with sigmoid scoring
// ============================================================================

pub struct SparseMoeBlock {
    pub router: UnifiedLinear,
    pub experts: SwitchGLU,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Compute gate logits in float32
        let x_f32 = mlxcel_core::astype(&x_flat, mlxcel_core::dtype::FLOAT32);
        let logits = self.router.forward(&x_f32);

        // Sigmoid scoring (not softmax)
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add correction bias for expert selection
        let biased_scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Top-k selection using argpartition on negated biased scores
        let k = self.num_experts_per_tok as i32;
        let neg_biased = mlxcel_core::negative(&biased_scores);
        let part = mlxcel_core::argpartition(&neg_biased, k - 1, -1);
        let part_shape = mlxcel_core::array_shape(&part);
        let topk_indices = mlxcel_core::slice(&part, &[0, 0], &[part_shape[0], k]);

        // Get original (unbiased) scores for the selected experts
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize scores to sum to 1
        let score_sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
        let eps = mlxcel_core::full_f32(&[1], 1e-20, mlxcel_core::array_dtype(&topk_scores));
        let score_sum = mlxcel_core::add(&score_sum, &eps);
        let norm_scores = mlxcel_core::divide(&topk_scores, &score_sum);
        let norm_scores = mlxcel_core::astype(&norm_scores, mlxcel_core::array_dtype(&x_flat));

        // Apply experts - returns [n_tokens, k, hidden]
        let expert_out = self.experts.forward(&x_flat, &topk_indices);

        let result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &norm_scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        // Gate uses 8-bit quantization for MiniMax
        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            args.group_size(),
            args.gate_bits(),
        )?;

        // Expert weights are pre-stacked as switch_mlp.{gate_proj,down_proj,up_proj}
        let experts = SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            args.group_size(),
            args.bits(),
        )?;

        // Load e_score_correction_bias
        let bias_key = format!("{}.e_score_correction_bias", prefix);
        let e_score_correction_bias = weights
            .get(&bias_key)
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| {
                mlxcel_core::full_f32(
                    &[args.num_local_experts as i32],
                    0.0,
                    mlxcel_core::dtype::FLOAT32,
                )
            });

        Ok(Self {
            router,
            experts,
            e_score_correction_bias,
            num_experts_per_tok: args.num_experts_per_tok,
        })
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

pub struct DecoderLayer {
    pub self_attn: Attention,
    pub block_sparse_moe: SparseMoeBlock,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm MoE
        let normed = self.post_attention_layernorm.forward(&h);
        let moe_out = self.block_sparse_moe.forward(&normed);
        mlxcel_core::add(&h, &moe_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let block_sparse_moe =
            SparseMoeBlock::from_weights(weights, args, &format!("{}.block_sparse_moe", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// ============================================================================
// MiniMax Model
// ============================================================================

pub struct MiniMaxModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
}

impl MiniMaxModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = DecoderLayer::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = if !args.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// ============================================================================
// LanguageModel trait implementation
// ============================================================================

impl LanguageModel for MiniMaxModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MiniMaxModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        MiniMaxModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // MiniMax-M2 EOS token: [e~[ → token ID 200020
        vec![200020]
    }
}
