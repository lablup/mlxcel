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

//! StarCoder2 model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - LayerNorm instead of RMSNorm (with affine parameters)
//! - GELU activation instead of SiLU
//! - Attention/MLP bias enabled

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct StarCoder2Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_epsilon: f32,
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
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

fn default_rope_theta() -> f32 {
    100000.0
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl StarCoder2Config {
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

// Attention.
pub struct StarCoder2Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl StarCoder2Attention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &StarCoder2Config,
    ) -> Result<Self, String> {
        let n_heads = cfg.num_attention_heads as i32;
        let n_kv_heads = cfg.num_key_value_heads as i32;
        let head_dim = cfg.head_dim() as i32;
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        // Fused QKV: concatenate q/k/v weights into one projection at load time
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights, prefix, group_size, bits, n_heads, n_kv_heads, head_dim,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                group_size,
                bits,
            )?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: cfg.rope_theta,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection: single matmul → split into Q, K, V
        let (q, k, v) = self.qkv_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE using fast_rope directly
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

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
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// MLP.
pub struct StarCoder2MLP {
    pub c_fc: UnifiedLinear,
    pub c_proj: UnifiedLinear,
}

impl StarCoder2MLP {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &StarCoder2Config,
    ) -> Result<Self, String> {
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        Ok(Self {
            c_fc: UnifiedLinear::from_weights(
                weights,
                &format!("{}.c_fc", prefix),
                group_size,
                bits,
            )?,
            c_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.c_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.c_fc.forward(x);
        // Use compiled GELU for kernel fusion (matches Python's gelu activation)
        let x = mlxcel_core::compiled_gelu(&x);
        self.c_proj.forward(&x)
    }
}

// Transformer Block.
pub struct StarCoder2TransformerBlock {
    pub self_attn: StarCoder2Attention,
    pub mlp: StarCoder2MLP,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
}

impl StarCoder2TransformerBlock {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &StarCoder2Config,
    ) -> Result<Self, String> {
        let self_attn =
            StarCoder2Attention::from_weights(weights, &format!("{}.self_attn", prefix), cfg)?;
        let mlp = StarCoder2MLP::from_weights(weights, &format!("{}.mlp", prefix), cfg)?;

        // Load LayerNorm weights manually
        let input_weight = get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let input_bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let input_layernorm = LayerNorm::new(input_weight, input_bias, cfg.norm_epsilon);

        let post_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_bias = weights
            .get(&format!("{}.post_attention_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let post_attention_layernorm = LayerNorm::new(post_weight, post_bias, cfg.norm_epsilon);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention block with residual
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP block with residual
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Model.
pub struct StarCoder2Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<StarCoder2TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Option<UnifiedLinear>,
}

impl StarCoder2Model {
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
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, StarCoder2Config), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: StarCoder2Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &StarCoder2Config) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Load embedding (auto-detects quantization)
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            layers.push(StarCoder2TransformerBlock::from_weights(
                weights, &prefix, config,
            )?);
        }

        // Load final norm (with weight and bias)
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm_bias = weights.get("model.norm.bias").map(|w| mlxcel_core::copy(w));
        let norm = LayerNorm::new(norm_weight, norm_bias, config.norm_epsilon);

        // Load lm_head (if not tied)
        let lm_head = if !config.tie_word_embeddings {
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

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for StarCoder2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        StarCoder2Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        StarCoder2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // StarCoder2 EOS token
        vec![0] // <|endoftext|>
    }
}
