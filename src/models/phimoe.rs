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

//! PhiMoE model implementation using mlxcel-core
//!
//! Key features:
//! - Mixture of Experts (MoE) architecture
//! - LayerNorm (not RMSNorm) with bias
//! - SiLU activation (SwiGLU) — hidden_act=silu per model config
//! - Softmax scoring for experts
//! - Standard RoPE (simplified from SuScaledRoPE)
//! - bias=True for attention projections

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "phimoe".to_string()
}
fn default_vocab_size() -> usize {
    32064
}
fn default_hidden_size() -> usize {
    4096
}
fn default_intermediate_size() -> usize {
    6400
}
fn default_num_hidden_layers() -> usize {
    32
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_num_local_experts() -> usize {
    16
}
fn default_num_experts_per_tok() -> usize {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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
}

// Sparse MoE Block.
/// PhiMoE sparse mixture of experts layer
pub struct SparseMoeBlock {
    pub router: UnifiedLinear, // Router/gate network
    pub experts: crate::models::switch_layers::SwitchGLU,
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

        // Get router logits
        let logits = self.router.forward(&x_flat);

        // Top-k selection using argpartition
        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);

        // Slice to get top-k: indices[..., kth:]
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Get scores for top-k experts and apply softmax (PhiMoE uses softmax)
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk_indices, -1);
        let scores = mlxcel_core::softmax(&topk_logits, -1);

        // Apply experts with optional fused single-token decode dispatch.
        let result = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.experts
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &topk_indices);
                    crate::models::switch_layers::moe_weighted_sum(
                        &expert_out,
                        &scores,
                        mlxcel_core::array_dtype(&x_flat),
                    )
                }
            }
        };

        // Reshape back to original shape
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }
}

// Attention.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear, // PhiMoE uses o_proj (not dense)
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

        // Project Q, K, V (with bias)
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE (standard RoPE, not traditional)
        q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token or explicit mask
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

        // Output projection (with bias)
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

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim, // Full RoPE
            rope_base: args.rope_theta,
        })
    }
}

// Transformer Block.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub block_sparse_moe: SparseMoeBlock,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
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

        // LayerNorm with bias
        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let input_norm_bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let input_layernorm =
            LayerNorm::new(input_norm_weight, input_norm_bias, args.layer_norm_eps);

        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_norm_bias = weights
            .get(&format!("{}.post_attention_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let post_attention_layernorm =
            LayerNorm::new(post_norm_weight, post_norm_bias, args.layer_norm_eps);

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// PhiMoE Model.
pub struct PhiMoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: LayerNorm,
    pub lm_head: UnifiedLinear,
}

impl PhiMoeModel {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head (with bias)
        self.lm_head.forward(&h)
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Sanitize weights (stack expert weights if needed)
        let weights = Self::sanitize_weights(&weights, &args)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = DecoderLayer::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm with bias
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm_bias = weights.get("model.norm.bias").map(|w| mlxcel_core::copy(w));
        let norm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_eps);

        // Load LM head (with bias)
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Sanitize weights: stack expert weights if they're provided separately
    fn sanitize_weights(weights: &WeightMap, args: &ModelArgs) -> Result<WeightMap, String> {
        use std::collections::HashMap;

        let mut new_weights = HashMap::new();

        // Check if experts are already stacked
        let check_key = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight".to_string();
        if weights.contains_key(&check_key) {
            // Already stacked, just clone
            for (k, v) in weights.iter() {
                new_weights.insert(k.clone(), mlxcel_core::copy(v));
            }
            return Ok(new_weights);
        }

        // Check if experts are separate
        let check_sep = "model.layers.0.block_sparse_moe.experts.0.w1.weight".to_string();
        let needs_stacking = weights.contains_key(&check_sep);

        if !needs_stacking {
            // Neither format, just copy
            for (k, v) in weights.iter() {
                new_weights.insert(k.clone(), mlxcel_core::copy(v));
            }
            return Ok(new_weights);
        }

        // Stack expert weights
        for l in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{}", l);

            // Map: (original_name, target_name)
            let proj_mapping = [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")];

            for (orig_name, target_name) in &proj_mapping {
                for weight_type in &["weight", "scales", "biases"] {
                    let first_key = format!(
                        "{}.block_sparse_moe.experts.0.{}.{}",
                        prefix, orig_name, weight_type
                    );

                    if weights.contains_key(&first_key) {
                        let mut expert_arrays: Vec<UniquePtr<MlxArray>> = Vec::new();
                        for e in 0..args.num_local_experts {
                            let key = format!(
                                "{}.block_sparse_moe.experts.{}.{}.{}",
                                prefix, e, orig_name, weight_type
                            );
                            if let Some(w) = weights.get(&key) {
                                expert_arrays.push(mlxcel_core::copy(w));
                            }
                        }

                        if !expert_arrays.is_empty() {
                            let expert_ptrs: Vec<*const MlxArray> = expert_arrays
                                .iter()
                                .map(|a| a.as_ref().unwrap() as *const _)
                                .collect();
                            let stacked = mlxcel_core::stack(&expert_ptrs, 0);
                            let new_key = format!(
                                "{}.block_sparse_moe.switch_mlp.{}.{}",
                                prefix, target_name, weight_type
                            );
                            new_weights.insert(new_key, stacked);
                        }
                    }
                }
            }
        }

        // Copy all other weights
        for (k, v) in weights.iter() {
            if !k.contains(".experts.") {
                new_weights.insert(k.clone(), mlxcel_core::copy(v));
            }
        }

        Ok(new_weights)
    }
}

// MoE Implementation Details.
impl SparseMoeBlock {
    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let router = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            args.group_size(),
            args.bits(),
        )?;

        let experts = crate::models::switch_layers::SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            args.group_size(),
            args.bits(),
        )?;

        Ok(Self {
            router,
            experts,
            num_experts_per_tok: args.num_experts_per_tok,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for PhiMoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        PhiMoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        PhiMoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // PhiMoE EOS token (commonly <|endoftext|> = 32000)
        vec![32000]
    }
}
