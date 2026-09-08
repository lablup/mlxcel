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

//! Phi v1/v2 model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Partial RoPE (partial_rotary_factor - only part of head_dim uses RoPE)
//! - GELU approximate activation (not SiLU)
//! - LayerNorm (not RMSNorm)
//! - bias=True for all projections
//! - Parallel attention + MLP (attn_h + ff_h + x)
//! - Simple MLP: fc2(gelu(fc1(x))) - no gate/up pattern

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

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

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
    "phi".to_string()
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_vocab_size() -> usize {
    51200
}
fn default_hidden_size() -> usize {
    2560
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_hidden_layers() -> usize {
    32
}
fn default_partial_rotary_factor() -> f32 {
    0.4
}
fn default_intermediate_size() -> usize {
    10240
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of dimensions to apply RoPE to
    pub fn rope_dims(&self) -> i32 {
        (self.partial_rotary_factor * self.head_dim() as f32) as i32
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

// Attention with Partial RoPE.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub dense: UnifiedLinear, // Note: dense, not o_proj (Phi naming)
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32, // Partial RoPE dimensions
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

        // Apply partial RoPE (only to first rope_dims dimensions)
        q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // **The attention runs in float32, and that is load-bearing**, for the
        // reason [`crate::models::phixtral`] already documents: upstream writes
        // `queries.astype(mx.float32)` around the SDPA call
        // (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phi.py),
        // Phi-2 carries large outlier activations, and this checkpoint ships
        // float16, whose 65504 ceiling the `q @ k^T` products reach in the deep
        // layers. `phi-2-4bit` goes NaN at layer 29 of 32 in f16, after which
        // every sampled token is the argmax of a NaN row and decodes as `!`.
        // The cache stays f16; only the arithmetic widens.
        // Upstream widens the queries alone, and MLX promotes the score matmul to
        // match its wider operand, so that is enough to keep `q @ k^T` out of f16.
        // The values are only ever multiplied by probabilities in [0, 1] and cannot
        // overflow, so widening them would double the V read on every decode step
        // and buy nothing.
        let dtype = mlxcel_core::array_dtype(&cache_v);
        let q = mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32);

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
        let attn_out = mlxcel_core::astype(&attn_out, dtype);

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection (dense with bias)
        self.dense.forward(&attn_out)
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
        let dense =
            UnifiedLinear::from_weights(weights, &format!("{}.dense", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rope_dims(),
            rope_base: args.rope_theta,
        })
    }
}

// MLP (Simple GELU - no gate/up pattern).
pub struct MLP {
    pub fc1: UnifiedLinear,
    pub fc2: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Simple MLP: fc2(gelu(fc1(x)))
        let h = self.fc1.forward(x);
        let h = mlxcel_core::utils::gelu_approx(&h);
        self.fc2.forward(&h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), group_size, bits)?;

        Ok(Self { fc1, fc2 })
    }
}

// Transformer Block (Parallel Attention + MLP).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: LayerNorm,
}

impl TransformerBlock {
    /// Parallel computation: attn(norm(x)) + mlp(norm(x)) + x
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Normalize input once
        let normed = self.input_layernorm.forward(x);

        // Compute attention and MLP in parallel (on the same normed input)
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let mlp_out = self.mlp.forward(&normed);

        // Sum all three: x + attn + mlp
        let h = mlxcel_core::add(x, &attn_out);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        // LayerNorm with bias
        let weight = get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let input_layernorm = LayerNorm::new(weight, bias, args.layer_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }
}

// Phi Model.
pub struct PhiModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub final_layernorm: LayerNorm,
    pub lm_head: UnifiedLinear, // Phi has separate lm_head with bias
}

impl PhiModel {
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
        let h = self.final_layernorm.forward(&h);

        // LM head (with bias)
        self.lm_head.forward(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

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

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load embedding (auto-detects quantization)
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final layernorm with bias
        let norm_weight = get_weight_copy(weights, "model.final_layernorm.weight")?;
        let norm_bias = weights
            .get("model.final_layernorm.bias")
            .map(|w| mlxcel_core::copy(w));
        let final_layernorm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_eps);

        // Load lm_head (with bias)
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
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
impl LanguageModel for PhiModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        PhiModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        PhiModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![50256] // GPT-2 style EOS token
    }
}
