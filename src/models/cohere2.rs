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

//! Cohere2 (Command R+) model implementation using mlxcel-core
//!
//! Key features:
//! - Sliding window attention pattern (every Nth layer is global, others use sliding window)
//! - RoPE only applied to sliding window layers (not global attention layers)
//! - Standard LayerNorm (not RMSNorm)
//! - Parallel attention + MLP (both on same normalized input)
//! - Logit scaling

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_causal_mask_with_window};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
/// Cohere2 model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Cohere2Config {
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
    pub head_dim: usize,
    pub sliding_window: usize,
    // sliding_window_pattern is optional, we'll fall back to layer_switch
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    // layer_switch is an alternative name
    #[serde(default)]
    pub layer_switch: Option<usize>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub layer_norm_bias: bool,

    // Both tie_word_embeddings and use_embedding_sharing can be used
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_embedding_sharing: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    // BOS/EOS token IDs
    #[serde(default)]
    pub bos_token_id: Option<i32>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl Cohere2Config {
    /// Get the sliding window pattern value (from either field)
    fn get_sliding_window_pattern(&self) -> usize {
        self.sliding_window_pattern
            .or(self.layer_switch)
            .unwrap_or(4) // Default to 4
    }

    /// Get whether to use tied embeddings (from either field)
    fn should_tie_embeddings(&self) -> bool {
        self.tie_word_embeddings || self.use_embedding_sharing
    }

    /// Returns true if the layer uses sliding window attention
    pub fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        // Every Nth layer is global (not sliding), others are sliding
        // Pattern: layer 0,1,2 = sliding, layer 3 = global, layer 4,5,6 = sliding, layer 7 = global, ...
        let pattern = self.get_sliding_window_pattern();
        !(layer_idx + 1).is_multiple_of(pattern)
    }

    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// Cohere2 Attention (with conditional RoPE).
pub struct Cohere2Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub use_sliding_window: bool,
    pub window_size: i32,
}

impl Cohere2Attention {
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
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE only if this is a sliding window layer
        let (q, k) = if self.use_sliding_window {
            // Use traditional RoPE for Cohere
            let q_rope =
                mlxcel_core::fast_rope(&q, self.rope_dims, true, self.rope_base, 1.0, offset);
            let k_rope =
                mlxcel_core::fast_rope(&k, self.rope_dims, true, self.rope_base, 1.0, offset);
            (q_rope, k_rope)
        } else {
            (q, k)
        };

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 {
            // Prefill: use mask. If a windowed mask was supplied for a
            // sliding-window layer, post-earlier it is shaped
            // `(l, window_size)` (capped). Plain `KVCache` returns
            // full-length K/V, so slice K/V to the last `window_size` slots
            // here, mirroring `causal_attention`'s internal slicing.
            let k_shape = mlxcel_core::array_shape(&cache_k);
            let k_len = k_shape[2];
            let (k_used, v_used) = if self.window_size > 0 && k_len > self.window_size {
                let v_shape = mlxcel_core::array_shape(&cache_v);
                let start = k_len - self.window_size;
                (
                    Some(mlxcel_core::slice(
                        &cache_k,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], k_len, k_shape[3]],
                    )),
                    Some(mlxcel_core::slice(
                        &cache_v,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], k_len, v_shape[3]],
                    )),
                )
            } else {
                (None, None)
            };
            let k_ref: &MlxArray = k_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_k.as_ref().unwrap());
            let v_ref: &MlxArray = v_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_v.as_ref().unwrap());

            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    k_ref,
                    v_ref,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        } else {
            // Single token: use causal SDPA (slices internally when needed)
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2Config,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let use_sliding_window = args.is_sliding_window_layer(layer_idx);

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;
        let head_dim = args.head_dim as i32;

        // Fused QKV: concatenate q/k/v weights into one projection at load time
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            prefix,
            group_size,
            bits,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            use_sliding_window,
            window_size: if use_sliding_window {
                args.sliding_window as i32
            } else {
                0
            },
        })
    }
}

// Cohere2 MLP.
pub struct Cohere2MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl Cohere2MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2Config,
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

// Cohere2 Transformer Block (parallel attention + MLP).
pub struct Cohere2TransformerBlock {
    pub self_attn: Cohere2Attention,
    pub mlp: Cohere2MLP,
    pub input_layernorm: LayerNorm,
}

impl Cohere2TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Cohere2 uses parallel attention + MLP
        // h = norm(x)
        // out = attn(h) + mlp(h) + x
        let h = self.input_layernorm.forward(x);
        let attn_h = self.self_attn.forward(&h, cache, mask);
        let ff_h = self.mlp.forward(&h);

        // attn_h + ff_h + x
        let sum = mlxcel_core::add(&attn_h, &ff_h);
        mlxcel_core::add(&sum, x)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Cohere2Attention::from_weights(
            weights,
            args,
            layer_idx,
            &format!("{}.self_attn", prefix),
        )?;
        let mlp = Cohere2MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        // Load layer norm weights
        let norm_weight = get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let norm_bias = weights
            .get(&format!("{}.input_layernorm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let input_layernorm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }
}

// Cohere2 Model.
pub struct Cohere2Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<Cohere2TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: UnifiedLinear,
    pub logit_scale: f32,
    pub config: Cohere2Config,
    // Indices of first sliding window and first global attention layers
    swa_idx: usize,
    ga_idx: usize,
}

impl Cohere2Model {
    /// Forward pass through the entire model
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1] as usize;

        // Create masks for full and sliding attention
        let (full_mask, sliding_mask) = if l > 1 {
            let ga_offset = caches[self.ga_idx].offset;
            let swa_offset = caches[self.swa_idx].offset;

            let full = Some(create_causal_mask(l as i32, ga_offset));
            let sliding = Some(create_causal_mask_with_window(
                l as i32,
                swa_offset,
                Some(self.config.sliding_window as i32),
            ));
            (full, sliding)
        } else {
            (None, None)
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.config.is_sliding_window_layer(i) {
                sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                full_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // Output projection
        let logits = self.lm_head.forward(&h);

        // Apply logit scaling
        let scale_arr =
            mlxcel_core::full_f32(&[1], self.logit_scale, mlxcel_core::array_dtype(&logits));
        mlxcel_core::multiply(&logits, &scale_arr)
    }

    /// Get token embeddings (for VLM merge)
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward with pre-computed embeddings (for VLM prefill)
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1] as usize;

        let (full_mask, sliding_mask) = if l > 1 {
            let ga_offset = caches[self.ga_idx].offset;
            let swa_offset = caches[self.swa_idx].offset;

            let full = Some(create_causal_mask(l as i32, ga_offset));
            let sliding = Some(create_causal_mask_with_window(
                l as i32,
                swa_offset,
                Some(self.config.sliding_window as i32),
            ));
            (full, sliding)
        } else {
            (None, None)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.config.is_sliding_window_layer(i) {
                sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                full_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);
        let logits = self.lm_head.forward(&h);
        let scale_arr =
            mlxcel_core::full_f32(&[1], self.logit_scale, mlxcel_core::array_dtype(&logits));
        mlxcel_core::multiply(&logits, &scale_arr)
    }

    /// Load model from directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Cohere2Config), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: Cohere2Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &Cohere2Config) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Find first indices for each attention type
        let sliding_window_pattern = args.get_sliding_window_pattern();
        let swa_idx = (0..args.num_hidden_layers)
            .find(|&i| args.is_sliding_window_layer(i))
            .unwrap_or(0);
        let ga_idx = (0..args.num_hidden_layers)
            .find(|&i| !args.is_sliding_window_layer(i))
            .unwrap_or(sliding_window_pattern - 1);

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = Cohere2TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm_bias = weights.get("model.norm.bias").map(|w| mlxcel_core::copy(w));
        let norm = LayerNorm::new(norm_weight, norm_bias, args.layer_norm_eps);

        // Load LM head (tied to embeddings in Cohere2)
        let lm_head = if args.should_tie_embeddings() {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            logit_scale: args.logit_scale,
            config: args.clone(),
            swa_idx,
            ga_idx,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for Cohere2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        match &self.config.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![*id],
            Some(EosTokenId::Multiple(ids)) => ids.clone(),
            None => vec![255001], // Default Cohere2 EOS token
        }
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}
