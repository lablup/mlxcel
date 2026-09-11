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

//! ExaOne model implementation using mlxcel-core (LG AI)
//!
//! Key differences from Llama:
//! - Different weight naming: wte, h, ln_f, ln_1/ln_2, out_proj, c_fc_0/c_fc_1/c_proj
//! - Model structure name: transformer (not model)
//! - Attention wrapper: attn.attention (nested)
//! - Standard Llama-style architecture otherwise (RMSNorm + SiLU)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct ExaOneConfig {
    pub model_type: String,
    pub hidden_size: usize,
    #[serde(default = "default_num_layers", rename = "num_layers")]
    pub num_hidden_layers: usize, // Note: can be either "num_layers" or "num_hidden_layers"
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_num_layers() -> usize {
    32
}

fn default_layer_norm_epsilon() -> f32 {
    1e-6
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl ExaOneConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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
pub struct ExaOneAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub out_proj: UnifiedLinear, // Note: out_proj, not o_proj
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub rope_traditional: bool,
}

impl ExaOneAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &ExaOneConfig,
    ) -> Result<Self, String> {
        let n_heads = cfg.num_attention_heads as i32;
        let n_kv_heads = cfg.num_key_value_heads as i32;
        let head_dim = cfg.head_dim() as i32;
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_proj", prefix),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.k_proj", prefix),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.v_proj", prefix),
                group_size,
                bits,
            )?,
            out_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.out_proj", prefix),
                group_size,
                bits,
            )?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_base: cfg.rope_theta,
            rope_traditional: cfg.rope_traditional,
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

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

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
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

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
        self.out_proj.forward(&attn_out)
    }
}

// MLP.
pub struct ExaOneMLP {
    pub c_fc_0: UnifiedLinear, // gate_proj equivalent
    pub c_fc_1: UnifiedLinear, // up_proj equivalent
    pub c_proj: UnifiedLinear, // down_proj equivalent
}

impl ExaOneMLP {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &ExaOneConfig,
    ) -> Result<Self, String> {
        let group_size = cfg.group_size();
        let bits = cfg.bits();

        Ok(Self {
            c_fc_0: UnifiedLinear::from_weights(
                weights,
                &format!("{}.c_fc_0", prefix),
                group_size,
                bits,
            )?,
            c_fc_1: UnifiedLinear::from_weights(
                weights,
                &format!("{}.c_fc_1", prefix),
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
        let gate = self.c_fc_0.forward(x);
        let up = self.c_fc_1.forward(x);

        // Use compiled SwiGLU activation for kernel fusion
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.c_proj.forward(&activated)
    }
}

// Decoder Layer.
pub struct ExaOneDecoderLayer {
    pub attn: ExaOneAttention,
    pub mlp: ExaOneMLP,
    pub ln_1: RMSNorm, // input_layernorm
    pub ln_2: RMSNorm, // post_attention_layernorm
}

impl ExaOneDecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &ExaOneConfig,
    ) -> Result<Self, String> {
        // Note: ExaOne uses nested attention wrapper: attn.attention
        let attn_prefix = format!("{}.attn.attention", prefix);
        let attn = ExaOneAttention::from_weights(weights, &attn_prefix, cfg)?;
        let mlp = ExaOneMLP::from_weights(weights, &format!("{}.mlp", prefix), cfg)?;

        // Load RMSNorm weights manually
        let ln_1_weight = get_weight_copy(weights, &format!("{}.ln_1.weight", prefix))?;
        let ln_2_weight = get_weight_copy(weights, &format!("{}.ln_2.weight", prefix))?;

        let ln_1 = RMSNorm::new(ln_1_weight, cfg.layer_norm_epsilon);
        let ln_2 = RMSNorm::new(ln_2_weight, cfg.layer_norm_epsilon);

        Ok(Self {
            attn,
            mlp,
            ln_1,
            ln_2,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention with residual
        let normed = self.ln_1.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP with residual
        let normed = self.ln_2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// Model.
pub struct ExaOneModel {
    pub wte: UnifiedEmbedding,      // embed_tokens equivalent
    pub h: Vec<ExaOneDecoderLayer>, // layers equivalent
    pub ln_f: RMSNorm,              // norm equivalent
    pub lm_head: Option<UnifiedLinear>,
}

impl ExaOneModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.wte.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.h.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.ln_f.forward(&h);

        // LM head
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h)
        } else {
            self.wte.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.h.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ExaOneConfig), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: ExaOneConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    pub fn from_weights(weights: &WeightMap, config: &ExaOneConfig) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Load embedding (wte instead of embed_tokens)
        let wte = UnifiedEmbedding::from_weights(weights, "transformer.wte", group_size, bits)?;

        // Load transformer layers (h instead of layers)
        let mut h = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("transformer.h.{}", i);
            h.push(ExaOneDecoderLayer::from_weights(weights, &prefix, config)?);
        }

        // Load final norm (ln_f instead of norm)
        let ln_f_weight = get_weight_copy(weights, "transformer.ln_f.weight")?;
        let ln_f = RMSNorm::new(ln_f_weight, config.layer_norm_epsilon);

        // Load lm_head (if not tied)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            wte,
            h,
            ln_f,
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
impl LanguageModel for ExaOneModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ExaOneModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        ExaOneModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.h.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // ExaOne EOS token
        vec![0] // <|endoftext|>
    }
}
