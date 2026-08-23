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

//! InternLM2 model implementation using mlxcel-core
//!
//! Key difference from Llama:
//! - Fused wqkv projection (single linear layer for Q, K, V)
//! - Dynamic NTK RoPE scaling through the shared
//!   [`crate::models::dynamic_ntk_rope`] schedule
//!
//! # The `rope_scaling` block (issue #1324)
//!
//! `ModelArgs` did not declare `rope_scaling` at all, so the
//! `{"type": "dynamic", "factor": 2.0}` that InternLM2 checkpoints ship
//! (`models/internlm2-7b-4bit`, and the long-context 1M variants) was dropped
//! at deserialization and the rotary base stayed at `rope_theta` for every
//! sequence length. Inside `max_position_embeddings` that is the correct
//! schedule, which is why nothing shorter than a 32768-token prompt could
//! observe it; past that boundary the base has to grow and did not.
//!
//! The position scale was already right here (a hardcoded `1.0`), which is
//! where this family differs from `internlm3` and from
//! [`mlx_lm/models/internlm2.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm2.py):
//! upstream computes `rope_scale = ... else 2.0` and so doubles every position
//! on exactly these checkpoints.

use crate::models::dynamic_ntk_rope::DynamicNtkRope;
use crate::models::rope_utils::RopeScalingSpec;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
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
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default)]
    pub bias: bool,

    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_traditional: bool,

    /// Read through the shared [`RopeScalingSpec`], which looks up `type` then
    /// `rope_type`. InternLM2 configs spell it `type`; reading both means one
    /// helper serves this family and `internlm3`, whose configs spell it
    /// `rope_type`.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingSpec>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_max_pos() -> usize {
    32768
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

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// The rotary schedule this config selects.
    ///
    /// Returns `Err` for a scheme the family does not implement, or for a
    /// `"linear"` / `"dynamic"` block without a usable `factor`. That mirrors
    /// upstream's `__post_init__`, which raises on anything outside
    /// `{"linear", "dynamic"}`.
    pub fn rope(&self) -> Result<DynamicNtkRope, String> {
        DynamicNtkRope::from_scaling(
            self.head_dim() as i32,
            self.rope_theta,
            self.rope_traditional,
            self.max_position_embeddings,
            self.rope_scaling.as_ref(),
            &self.model_type,
        )
    }
}

// Attention with Fused QKV.
pub struct Attention {
    pub wqkv: UnifiedLinear, // Fused Q, K, V projection
    pub wo: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// The dynamic NTK schedule, resolved once at load.
    pub rope: DynamicNtkRope,
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

        // Fused QKV projection
        let qkv = self.wqkv.forward(x);

        // InternLM2 QKV layout: interleaved per KV head
        // Reshape to [B, L, n_kv_heads, 2 + n_kv_groups, head_dim]
        // where n_kv_groups = n_heads / n_kv_heads
        let n_kv_groups = self.num_heads / self.num_kv_heads;
        let qkv = mlxcel_core::reshape(
            &qkv,
            &[b, l, self.num_kv_heads, 2 + n_kv_groups, self.head_dim],
        );

        // Extract Q: first n_kv_groups elements per kv_head -> [B, L, n_kv_heads, n_kv_groups, head_dim]
        let q = mlxcel_core::utils::slice_axis(&qkv, 3, 0, n_kv_groups);
        // Reshape to [B, L, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);

        // Extract K: second-to-last element per kv_head -> [B, L, n_kv_heads, 1, head_dim]
        let k = mlxcel_core::utils::slice_axis(&qkv, 3, n_kv_groups, n_kv_groups + 1);
        // Reshape to [B, L, n_kv_heads, head_dim] (equivalent to squeeze axis 3)
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);

        // Extract V: last element per kv_head -> [B, L, n_kv_heads, 1, head_dim]
        let v = mlxcel_core::utils::slice_axis(&qkv, 3, n_kv_groups + 1, n_kv_groups + 2);
        // Reshape to [B, L, n_kv_heads, head_dim] (equivalent to squeeze axis 3)
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Now Q is [B, L, n_heads, head_dim], K and V are [B, L, n_kv_heads, head_dim]
        let mut q = q;
        let mut k = k;

        // Transpose to [batch, n_heads, seq_len, head_dim]
        q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let seq_len = l + offset;

        // Apply RoPE. Positions are handed over unscaled; only the base moves,
        // and only once the sequence passes `max_position_embeddings`.
        q = self.rope.apply(&q, offset, seq_len);
        k = self.rope.apply(&k, offset, seq_len);

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
        self.wo.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let wqkv =
            UnifiedLinear::from_weights(weights, &format!("{}.wqkv", prefix), group_size, bits)?;
        let wo = UnifiedLinear::from_weights(weights, &format!("{}.wo", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            wqkv,
            wo,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope: args.rope()?,
        })
    }
}

// MLP (SwiGLU).
pub struct MLP {
    pub w1: UnifiedLinear, // gate_proj
    pub w2: UnifiedLinear, // down_proj
    pub w3: UnifiedLinear, // up_proj
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.w1.forward(x);
        let up = self.w3.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.w2.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let w1 = UnifiedLinear::from_weights(weights, &format!("{}.w1", prefix), group_size, bits)?;
        let w2 = UnifiedLinear::from_weights(weights, &format!("{}.w2", prefix), group_size, bits)?;
        let w3 = UnifiedLinear::from_weights(weights, &format!("{}.w3", prefix), group_size, bits)?;

        Ok(Self { w1, w2, w3 })
    }
}

// Transformer Block.
pub struct TransformerBlock {
    pub attention: Attention,
    pub feed_forward: MLP,
    pub attention_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.ffn_norm.forward(&h);
        let ff_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let attention = Attention::from_weights(weights, args, &format!("{}.attention", prefix))?;
        let feed_forward = MLP::from_weights(weights, args, &format!("{}.feed_forward", prefix))?;

        let attention_norm_weight =
            get_weight_copy(weights, &format!("{}.attention_norm.weight", prefix))?;
        let ffn_norm_weight = get_weight_copy(weights, &format!("{}.ffn_norm.weight", prefix))?;

        let attention_norm = RMSNorm::new(attention_norm_weight, args.rms_norm_eps);
        let ffn_norm = RMSNorm::new(ffn_norm_weight, args.rms_norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

// InternLM2 Model.
pub struct InternLM2Model {
    pub tok_embeddings: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub output: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
}

impl InternLM2Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.tok_embeddings.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(ref output) = self.output {
            output.forward(&h)
        } else {
            self.tok_embeddings.as_linear(&h)
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
        let tok_embeddings =
            UnifiedEmbedding::from_weights(weights, "model.tok_embeddings", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load output projection (or use tied embeddings)
        let output = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "output", group_size, bits,
            )?)
        };

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            tie_word_embeddings: args.tie_word_embeddings,
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
impl LanguageModel for InternLM2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        InternLM2Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        InternLM2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // Default EOS token
    }
}

#[cfg(test)]
#[path = "internlm2_tests.rs"]
mod tests;
