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

//! InternLM 3 model implementation using mlxcel-core
//!
//! Key features:
//! - Dynamic NTK RoPE scaling through the shared
//!   [`crate::models::dynamic_ntk_rope`] schedule: positions are never scaled,
//!   and the rotary base is recomputed per forward from the live sequence
//!   length, departing from `rope_theta` only once the sequence passes
//!   `max_position_embeddings`.
//! - Optional qkv_bias
//! - Standard Llama-style architecture
//!
//! # The position scale (issue #1324)
//!
//! `ModelArgs::rope_scale` used to end in `.unwrap_or(2.0)` and matched only
//! `"linear"`, so an absent block and a `"dynamic"` block both produced a
//! position scale of `2.0`: every query and key was rotated at twice its true
//! position, at every context length, on the very checkpoint this family is
//! validated against
//! (`mlx-community/internlm3-8b-instruct-4bit`, which ships
//! `{"factor": 6.0, "rope_type": "dynamic"}`). That is not a long-context
//! defect: doubling the position doubles the relative angle between any two
//! tokens, so attention differed from training everywhere. It was inherited
//! from
//! [`mlx_lm/models/internlm3.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm3.py),
//! which still computes `rope_scale = ... else 2.0` and then reuses that same
//! `2.0` as the NTK factor; the checkpoint's own remote code
//! (`modeling_internlm3.py`, via transformers'
//! `_compute_dynamic_ntk_parameters`) is the reference this file now follows.

use crate::models::dynamic_ntk_rope::DynamicNtkRope;
use crate::models::rope_utils::RopeScalingSpec;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
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
    pub bias: bool,

    #[serde(default)]
    pub qkv_bias: bool,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_traditional: bool,

    /// Read through the shared [`RopeScalingSpec`] rather than a derived
    /// struct, so both the `rope_type` spelling this family's configs use and
    /// the older `type` spelling resolve, and a config carrying both parses
    /// instead of hitting serde's `duplicate field`.
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

fn default_max_position_embeddings() -> usize {
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
    /// `"linear"` / `"dynamic"` block without a usable `factor`; see
    /// [`DynamicNtkRope::from_scaling`] for why that is a load error here.
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

// Attention.
pub struct Attention {
    /// Fused QKV projection: Q, K, V weights concatenated along output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
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
        let seq_len = l + offset;

        // Apply RoPE. The base is recomputed from the live sequence length on
        // every forward, so it tracks the cache as it grows; the position is
        // handed over unscaled, which is what the dynamic schedule requires.
        let q = self.rope.apply(&q, offset, seq_len);
        let k = self.rope.apply(&k, offset, seq_len);

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention (handles GQA expansion internally).
        //
        // A multi-token prefill with no caller-supplied mask MUST take the
        // causal path. Every generation path (CLI text prefill, the VLM
        // embeddings prefill, and the chunked prefill) calls the model with
        // `mask == None` and expects the model to apply its own causal mask,
        // exactly like `create_attention_mask(h, cache[0])` in the reference
        // https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm3.py.
        // Handing that `None` straight to the unmasked SDPA made every prefill
        // fully bidirectional, so each layer wrote future-contaminated K/V into
        // the cache and the first sampled token was already wrong (issue #999).
        // An explicit mask (the padded-prefill case) still goes to the masked
        // call, and decode is unchanged: `causal_attention` takes its
        // single-query maskless fast path at `l == 1`.
        let attn_out = if l > 1 && mask.is_some() {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
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

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_kv_heads() as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

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
            scale,
            rope: args.rope()?,
        })
    }
}

// MLP.
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SiLU(gate_proj(x)) * up_proj(x)
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SiLU activation
        let gate_silu = mlxcel_core::silu(&gate);

        // Element-wise product
        let activated = mlxcel_core::multiply(&gate_silu, &up);

        // Down projection
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

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
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

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// InternLM3 Model.
pub struct InternLM3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
}

impl InternLM3Model {
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
        if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
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
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head (or use tied embeddings)
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
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
impl LanguageModel for InternLM3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        InternLM3Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        InternLM3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // InternLM EOS token
    }
}

#[cfg(test)]
#[path = "internlm3_tests.rs"]
mod tests;
