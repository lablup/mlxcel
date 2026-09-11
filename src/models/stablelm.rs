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

//! StableLM model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Partial RoPE (only applied to first portion of head dimensions)
//! - Q/K LayerNorm per head
//! - Optional parallel residual connections

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub use_qkv_bias: bool,
    pub partial_rotary_factor: f32, // Fraction of head_dim to apply RoPE to
    pub layer_norm_eps: f32,

    #[serde(default)]
    pub use_parallel_residual: bool,

    #[serde(default)]
    pub qk_layernorm: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn rope_dims(&self) -> usize {
        (self.head_dim() as f32 * self.partial_rotary_factor) as usize
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

// Q/K Normalization (LayerNorm per head).
/// LayerNorm applied per head: [n_heads, head_dim]
pub struct LayerNormPerHead {
    pub weight: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl LayerNormPerHead {
    /// Forward pass: x is [B, L, n_heads, head_dim]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Fast layer norm without bias
        unsafe {
            mlxcel_core::fast_layer_norm(
                x,
                self.weight.as_ref().unwrap() as *const _,
                std::ptr::null(),
                self.eps,
            )
        }
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_name))?;

        Ok(Self { weight, eps })
    }
}

// Attention with Partial RoPE and Q/K Normalization.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_layernorm: Option<LayerNormPerHead>, // Q normalization
    pub k_layernorm: Option<LayerNormPerHead>, // K normalization
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

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply Q/K normalization if enabled
        if let Some(ref q_norm) = self.q_layernorm {
            q = q_norm.forward(&q);
        }
        if let Some(ref k_norm) = self.k_layernorm {
            k = k_norm.forward(&k);
        }

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply partial RoPE: only to first rope_dims dimensions
        // Split each tensor into RoPE part and pass-through part
        let q_rope = mlxcel_core::slice_last_dim(&q, 0, self.rope_dims);
        let q_pass = mlxcel_core::slice_last_dim(&q, self.rope_dims, self.head_dim);
        let q_rope =
            mlxcel_core::fast_rope(&q_rope, self.rope_dims, false, self.rope_base, 1.0, offset);
        q = mlxcel_core::concatenate(&q_rope, &q_pass, -1);

        let k_rope = mlxcel_core::slice_last_dim(&k, 0, self.rope_dims);
        let k_pass = mlxcel_core::slice_last_dim(&k, self.rope_dims, self.head_dim);
        let k_rope =
            mlxcel_core::fast_rope(&k_rope, self.rope_dims, false, self.rope_base, 1.0, offset);
        k = mlxcel_core::concatenate(&k_rope, &k_pass, -1);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention in float32. Upstream widens both the
        // queries and the keys before the call and narrows the result after
        // (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/stablelm.py),
        // the same guard [`crate::models::phixtral`] and [`crate::models::phi`]
        // carry: the `q @ k^T` products can pass f16's 65504 ceiling in the deep
        // layers, and the softmax turns the resulting inf into NaN. The cache
        // stays in its own dtype; only the arithmetic widens.
        // Upstream widens the queries and the keys, and leaves the values alone:
        // values are multiplied by probabilities in [0, 1] and cannot overflow, so
        // widening them would double the V read on every decode step for nothing.
        let dtype = mlxcel_core::array_dtype(&cache_v);
        let f32_dtype = mlxcel_core::dtype::FLOAT32;
        let q = mlxcel_core::astype(&q, f32_dtype);
        let cache_k = mlxcel_core::astype(&cache_k, f32_dtype);

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

        // Output projection
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

        // Load Q/K normalization if enabled
        let q_layernorm = if args.qk_layernorm {
            Some(LayerNormPerHead::from_weights(
                weights,
                &format!("{}.q_layernorm", prefix),
                args.layer_norm_eps,
            )?)
        } else {
            None
        };

        let k_layernorm = if args.qk_layernorm {
            Some(LayerNormPerHead::from_weights(
                weights,
                &format!("{}.k_layernorm", prefix),
                args.layer_norm_eps,
            )?)
        } else {
            None
        };

        let head_dim = args.head_dim() as i32;
        let rope_dims = args.rope_dims() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_layernorm,
            k_layernorm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims,
            rope_base: args.rope_theta,
        })
    }
}

// MLP (SwiGLU).
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(result) = mlxcel_core::layers::compiled_swiglu_mlp(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// Transformer Block (with optional parallel residual).
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: Option<LayerNorm>,
    pub use_parallel_residual: bool,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if self.use_parallel_residual {
            // Parallel: x + attn(norm(x)) + mlp(norm(x))
            let normed = self.input_layernorm.forward(x);
            let attn_out = self.self_attn.forward(&normed, cache, mask);
            let mlp_out = self.mlp.forward(&normed);
            let h = mlxcel_core::add(x, &attn_out);
            mlxcel_core::add(&h, &mlp_out)
        } else {
            // Sequential: x + mlp(norm(x + attn(norm(x))))
            let normed = self.input_layernorm.forward(x);
            let attn_out = self.self_attn.forward(&normed, cache, mask);
            let h = mlxcel_core::add(x, &attn_out);

            let normed = self
                .post_attention_layernorm
                .as_ref()
                .expect("post_attention_layernorm required for sequential")
                .forward(&h);
            let mlp_out = self.mlp.forward(&normed);
            mlxcel_core::add(&h, &mlp_out)
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        // Load input LayerNorm
        let ln_weight_name = format!("{}.input_layernorm.weight", prefix);
        let ln_bias_name = format!("{}.input_layernorm.bias", prefix);

        let weight = weights
            .get(&ln_weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", ln_weight_name))?;

        let bias = weights.get(&ln_bias_name).map(|b| mlxcel_core::copy(b));

        let input_layernorm = LayerNorm::new(weight, bias, args.layer_norm_eps);

        // Load post-attention LayerNorm (only if not using parallel residual)
        let post_attention_layernorm = if !args.use_parallel_residual {
            let ln_weight_name = format!("{}.post_attention_layernorm.weight", prefix);
            let ln_bias_name = format!("{}.post_attention_layernorm.bias", prefix);

            let weight = weights
                .get(&ln_weight_name)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {}", ln_weight_name))?;

            let bias = weights.get(&ln_bias_name).map(|b| mlxcel_core::copy(b));

            Some(LayerNorm::new(weight, bias, args.layer_norm_eps))
        } else {
            None
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            use_parallel_residual: args.use_parallel_residual,
        })
    }
}

// StableLM Model.
pub struct StableLMModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: UnifiedLinear,
    eos_token_ids: Vec<i32>,
}

impl StableLMModel {
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

        // Load final norm
        let norm_weight_name = "model.norm.weight";
        let norm_bias_name = "model.norm.bias";

        let weight = weights
            .get(norm_weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", norm_weight_name))?;

        let bias = weights.get(norm_bias_name).map(|b| mlxcel_core::copy(b));

        let norm = LayerNorm::new(weight, bias, args.layer_norm_eps);

        // Load LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        // Parse EOS token IDs from config
        let eos_token_ids = match &args.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_i64().unwrap_or(0) as i32]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
            _ => vec![0],
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for StableLMModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        StableLMModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        StableLMModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
