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

//! Nemotron model implementation using mlxcel-core
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron.py
//!
//! Key features:
//! - NemotronLayerNorm1P: uses (weight + 1) instead of just weight
//! - relu_squared activation: relu(x)^2
//! - Partial RoPE: apply rotary only to partial_rotary_factor * head_dim dimensions
//! - Simple MLP without gating (just up_proj, relu_squared, down_proj)
//! - Standard GQA attention with partial RoPE

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::relu_squared;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,

    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type", alias = "rope_type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_partial_rotary_factor() -> f32 {
    0.5
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_scale(&self) -> f32 {
        if let Some(ref scaling) = self.rope_scaling
            && let Some(ref stype) = scaling.scaling_type
            && stype == "linear"
            && let Some(factor) = scaling.factor
        {
            return 1.0 / factor;
        }
        1.0
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

// NemotronLayerNorm1P: Uses (weight + 1) instead of just weight.
pub struct NemotronLayerNorm1P {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl NemotronLayerNorm1P {
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // NemotronLayerNorm1P: uses (weight + 1) instead of just weight
        let ones = mlxcel_core::ones(
            &[mlxcel_core::array_shape(&self.weight)[0]],
            mlxcel_core::array_dtype(&self.weight),
        );
        let weight_plus_one = mlxcel_core::add(&self.weight, &ones);

        // Use fast_layer_norm with (weight + 1) and no bias
        let weight_ptr = weight_plus_one.as_ref().unwrap() as *const MlxArray;
        unsafe { mlxcel_core::fast_layer_norm(x, weight_ptr, std::ptr::null(), self.eps) }
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        Ok(Self::new(weight, eps))
    }
}

// Partial RoPE Attention.
pub struct NemotronAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32, // Only apply RoPE to this many dimensions
    pub rope_base: f32,
    pub rope_scale: f32,
}

impl NemotronAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply partial RoPE - only to first rope_dims dimensions
        let q = if self.rope_dims < self.head_dim {
            self.apply_partial_rope(&q, offset)
        } else {
            mlxcel_core::fast_rope(
                &q,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            )
        };

        let k = if self.rope_dims < self.head_dim {
            self.apply_partial_rope(&k, offset)
        } else {
            mlxcel_core::fast_rope(
                &k,
                self.rope_dims,
                false,
                self.rope_base,
                self.rope_scale,
                offset,
            )
        };

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

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    /// Apply partial RoPE: only to the first rope_dims dimensions
    fn apply_partial_rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        // x shape: [batch, n_heads, seq_len, head_dim]
        // Split into rotary part and passthrough part
        let x_rot = mlxcel_core::slice_last_dim(x, 0, self.rope_dims);
        let x_pass = mlxcel_core::slice_last_dim(x, self.rope_dims, self.head_dim);

        // Apply RoPE to rotary part
        let x_rot_roped = mlxcel_core::fast_rope(
            &x_rot,
            self.rope_dims,
            false,
            self.rope_base,
            self.rope_scale,
            offset,
        );

        // Concatenate rotary and passthrough parts
        mlxcel_core::concatenate(&x_rot_roped, &x_pass, -1)
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
        let rope_dims = (args.partial_rotary_factor * head_dim as f32) as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims,
            rope_base: args.rope_theta,
            rope_scale: args.rope_scale(),
        })
    }
}

// Simple MLP with relu_squared (no gating).
pub struct NemotronMLP {
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl NemotronMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Simple MLP: down_proj(relu_squared(up_proj(x)))
        let up = self.up_proj.forward(x);
        let activated = relu_squared(&up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self { up_proj, down_proj })
    }
}

// Transformer Block.
pub struct NemotronBlock {
    pub self_attn: NemotronAttention,
    pub mlp: NemotronMLP,
    pub input_layernorm: NemotronLayerNorm1P,
    pub post_attention_layernorm: NemotronLayerNorm1P,
}

impl NemotronBlock {
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

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            NemotronAttention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = NemotronMLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_layernorm = NemotronLayerNorm1P::from_weights(
            weights,
            &format!("{}.input_layernorm", prefix),
            args.norm_eps,
        )?;
        let post_attention_layernorm = NemotronLayerNorm1P::from_weights(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            args.norm_eps,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Nemotron Model.
pub struct NemotronModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<NemotronBlock>,
    pub norm: NemotronLayerNorm1P,
    pub lm_head: UnifiedLinear,
}

impl NemotronModel {
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

        // LM head
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
            let layer = NemotronBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm = NemotronLayerNorm1P::from_weights(weights, "model.norm", args.norm_eps)?;

        // Load LM head
        let lm_head = if args.tie_word_embeddings {
            // Use embedding weights for lm_head
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for NemotronModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        NemotronModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        NemotronModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Common EOS token ID for Nemotron models
        vec![0]
    }
}

// Type alias for compatibility
pub type Model = NemotronModel;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_args_defaults() {
        let json = r#"{
            "model_type": "nemotron",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.head_dim(), 128);
        assert_eq!(args.partial_rotary_factor, 0.5);
        assert_eq!(args.rope_theta, 10000.0);
        assert_eq!(args.norm_eps, 1e-5);
    }

    #[test]
    fn test_rope_scale() {
        let json = r#"{
            "model_type": "nemotron",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.rope_scale(), 0.5);
    }
}
