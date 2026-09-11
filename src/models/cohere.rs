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

//! Cohere model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - LayerNorm instead of RMSNorm
//! - Parallel attention+MLP (both computed on normed input, then summed)
//! - Q/K normalization
//! - Logit scaling

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
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    pub logit_scale: f32,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub layer_norm_bias: bool,

    #[serde(default)]
    pub use_qk_norm: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
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

// Q/K Normalization (2D LayerNorm per head).
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

// Attention with Q/K Normalization.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: Option<LayerNormPerHead>, // Q normalization
    pub k_norm: Option<LayerNormPerHead>, // K normalization
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

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply Q/K normalization if enabled
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q);
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k);
        }

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let mut k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            true, // traditional=true for Cohere
            self.rope_base,
            1.0,
            offset,
        );
        k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            true, // traditional=true for Cohere
            self.rope_base,
            1.0,
            offset,
        );

        // Update KV cache and get sliced views
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
        let q_norm = if args.use_qk_norm {
            Some(LayerNormPerHead::from_weights(
                weights,
                &format!("{}.q_norm", prefix),
                args.layer_norm_eps,
            )?)
        } else {
            None
        };

        let k_norm = if args.use_qk_norm {
            Some(LayerNormPerHead::from_weights(
                weights,
                &format!("{}.k_norm", prefix),
                args.layer_norm_eps,
            )?)
        } else {
            None
        };

        let head_dim = args.head_dim() as i32;

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
            rope_dims: head_dim,
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

// Transformer Block (Parallel Attention+MLP).
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

        // Load LayerNorm weight and bias
        let ln_weight_name = format!("{}.input_layernorm.weight", prefix);
        let ln_bias_name = format!("{}.input_layernorm.bias", prefix);

        let weight = weights
            .get(&ln_weight_name)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", ln_weight_name))?;

        let bias = if args.layer_norm_bias {
            weights.get(&ln_bias_name).map(|b| mlxcel_core::copy(b))
        } else {
            None
        };

        let input_layernorm = LayerNorm::new(weight, bias, args.layer_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }
}

// Cohere Model.
pub struct CohereModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub logit_scale: f32,
}

impl CohereModel {
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

        // LM head (tied embeddings) with logit scaling
        let logits = self.embed_tokens.as_linear(&h);

        // Apply logit scaling
        if self.logit_scale != 1.0 {
            let scale =
                mlxcel_core::full_f32(&[1], self.logit_scale, mlxcel_core::array_dtype(&logits));
            mlxcel_core::multiply(&logits, &scale)
        } else {
            logits
        }
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

        // Load quantized embedding
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

        let bias = if args.layer_norm_bias {
            weights.get(norm_bias_name).map(|b| mlxcel_core::copy(b))
        } else {
            None
        };

        let norm = LayerNorm::new(weight, bias, args.layer_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            logit_scale: args.logit_scale,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for CohereModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        CohereModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        CohereModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![255001] // Cohere EOS token
    }
}
