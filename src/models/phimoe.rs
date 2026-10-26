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

//! PhiMoE model implementation using mlxcel-core
//!
//! Key features:
//! - Mixture of Experts (MoE) architecture
//! - LayerNorm (not RMSNorm) with bias
//! - SiLU activation (SwiGLU) — hidden_act=silu per model config
//! - sparsemixer expert routing (HF `sparsemixer`, inference path)
//! - LongRoPE through [`phi3::SuRope`]: the short table while the sequence
//!   fits in `original_max_position_embeddings`, the long one beyond it, with
//!   the configured mscale; plain RoPE when `rope_scaling` is null or absent
//! - bias=True for attention projections

use mlxcel_core::batched_decode::{SplitAttention, SplitAttentionShape, forward_split_attention};
use mlxcel_core::cache::BatchedAttentionMetadata;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode};
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

use crate::models::phi3;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
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

    /// Norm epsilon for the LayerNorm blocks. Phi-3.5-MoE checkpoints spell it
    /// `rms_norm_eps` (the key mlx-lm's `phimoe.py` reads, default 1e-6);
    /// configs using the `layer_norm_eps` spelling keep parsing. 1e-6 when
    /// neither key exists (issue #2465).
    #[serde(default = "default_layer_norm_eps", alias = "rms_norm_eps")]
    pub layer_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// `longrope` block (Phi-3.5-MoE). Null or absent keeps plain RoPE
    /// (Phi-mini-MoE).
    #[serde(default)]
    pub rope_scaling: Option<phi3::RopeScaling>,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Trained context length when the checkpoint spells it at the top level
    /// rather than inside `rope_scaling`. Resolved through
    /// [`ModelArgs::original_max_position_embeddings`].
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,

    /// sparsemixer's `jitter_eps`: how far below the current top gate an
    /// expert may sit and still share the softmax that scores it.
    #[serde(default = "default_router_jitter_noise")]
    pub router_jitter_noise: f32,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    /// A ModelOpt NVFP4 export declares its quantization here instead of in
    /// MLX's `quantization` block; [`Self::group_size`] reads its 16-element
    /// blocks from it (issue #2351).
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);

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
fn default_max_position_embeddings() -> usize {
    131072
}
fn default_router_jitter_noise() -> f32 {
    0.01
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
            .or_else(|| {
                crate::models::sanitize::modelopt_nvfp4_group_size(
                    self.quantization_config.as_ref(),
                )
            })
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Trained context `L` that selects the LongRoPE table: the
    /// `rope_scaling` block's value, then the top-level key, then 4096.
    pub fn original_max_position_embeddings(&self) -> usize {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.original_max_position_embeddings)
            .or(self.original_max_position_embeddings)
            .unwrap_or_else(crate::models::config::default_original_max_position_embeddings)
    }

    /// Both LongRoPE tables, or `None` when `rope_scaling` is null, absent, or
    /// names another scheme.
    pub(crate) fn su_rope(&self) -> Option<phi3::SuRope> {
        phi3::SuRope::from_scaling(
            self.rope_scaling.as_ref()?,
            self.head_dim(),
            self.rope_theta,
            self.max_position_embeddings,
            self.original_max_position_embeddings(),
        )
    }

    /// Reject a routing config sparsemixer cannot run: it picks
    /// `num_experts_per_tok` distinct experts, one per round.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_experts_per_tok == 0 {
            return Err("num_experts_per_tok must be at least 1".to_string());
        }
        if self.num_experts_per_tok > self.num_local_experts {
            return Err(format!(
                "num_experts_per_tok ({}) is larger than num_local_experts ({})",
                self.num_experts_per_tok, self.num_local_experts
            ));
        }
        Ok(())
    }
}

// sparsemixer routing.

/// Inference path of HF PhiMoE's `sparsemixer`, as mlx-lm ports it
/// (ml-explore/mlx-lm#1939).
///
/// Runs `top_k` rounds over `gates` (`[..., n_experts]`). Each round takes the
/// best remaining expert, then scores it with a softmax over only the experts
/// whose gate is within `2 * jitter_eps` of that best one, relative to
/// `max(|gate|, top)`; experts already chosen stay at `-inf`. The scores are
/// not renormalized across rounds, so a confident first expert and a
/// confident second expert can both score 1.0.
///
/// Returns `(indices, scores)`, both `[..., top_k]`.
pub(crate) fn sparsemixer(
    gates: &MlxArray,
    top_k: usize,
    jitter_eps: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let dtype = mlxcel_core::array_dtype(gates);
    let shape = mlxcel_core::array_shape(gates);
    let n_experts = *shape.last().expect("gates have an expert axis");
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, dtype);
    let threshold = mlxcel_core::full_f32(&[1], 2.0 * jitter_eps, dtype);
    let abs_gates = mlxcel_core::abs(gates);
    let expert_ids = mlxcel_core::arange_i32(0, n_experts, 1);

    let mut masked = mlxcel_core::copy(gates);
    let mut indices = Vec::with_capacity(top_k);
    let mut scores = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        let top = mlxcel_core::max_axis(&masked, -1, true);
        let ind = mlxcel_core::argmax(&masked, -1, true);
        let factor = mlxcel_core::maximum(&abs_gates, &top);
        let relative = mlxcel_core::divide(&mlxcel_core::subtract(&top, gates), &factor);
        let band = mlxcel_core::where_cond(
            &mlxcel_core::greater(&relative, &threshold),
            &neg_inf,
            &masked,
        );
        let band = mlxcel_core::softmax_precise(&band, -1);
        scores.push(mlxcel_core::take_along_axis(&band, &ind, -1));
        // One-hot `-inf` at the chosen expert so the next round skips it.
        let chosen = mlxcel_core::equal(
            &expert_ids,
            &mlxcel_core::astype(&ind, mlxcel_core::dtype::INT32),
        );
        masked = mlxcel_core::where_cond(&chosen, &neg_inf, &masked);
        indices.push(ind);
    }
    let concat = |parts: &[UniquePtr<MlxArray>]| {
        let refs: Vec<&MlxArray> = parts.iter().map(|p| p.as_ref().unwrap()).collect();
        mlxcel_core::concatenate_many(&refs, -1)
    };
    (concat(&indices), concat(&scores))
}

// Sparse MoE Block.
/// PhiMoE sparse mixture of experts layer
pub struct SparseMoeBlock {
    pub router: UnifiedLinear, // Router/gate network
    pub experts: crate::models::switch_layers::SwitchGLU,
    pub num_experts_per_tok: usize,
    /// sparsemixer `jitter_eps`, from `router_jitter_noise`.
    pub jitter_eps: f32,
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

        // Route with sparsemixer. Both expert paths below take these scores
        // as they are: the fused kernel multiplies each expert's down
        // projection by its score in f32 and sums over k without
        // renormalizing, the same weighted sum `moe_weighted_sum` computes.
        let gates = self.router.forward(&x_flat);
        let (topk_indices, scores) = sparsemixer(&gates, self.num_experts_per_tok, self.jitter_eps);

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
    /// LongRoPE tables, `None` for a null or absent `rope_scaling`.
    pub su_rope: Option<phi3::SuRope>,
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

        // Apply RoPE (non-traditional). LongRoPE picks its table for the whole
        // sequence and pre-scales Q and K by its mscale, as phi3 does.
        if let Some(table) = self.su_rope.as_ref().map(|su| su.table_for(offset, l)) {
            if let Some(scaled) = table.scale_rotary_prefix(&q, self.rope_dims) {
                q = scaled;
            }
            if let Some(scaled) = table.scale_rotary_prefix(&k, self.rope_dims) {
                k = scaled;
            }
            q = mlxcel_core::fast_rope_with_freqs(
                &q,
                self.rope_dims,
                false,
                1.0,
                offset,
                &table.freqs,
            );
            k = mlxcel_core::fast_rope_with_freqs(
                &k,
                self.rope_dims,
                false,
                1.0,
                offset,
                &table.freqs,
            );
        } else {
            q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
            k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);
        }

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
            su_rope: args.su_rope(),
        })
    }
}

impl SplitAttention for Attention {
    fn split_shape(&self) -> SplitAttentionShape {
        SplitAttentionShape {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }

    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let seq_len = mlxcel_core::array_shape(q)[2];
        let mut q_rows = Vec::with_capacity(offsets.len());
        let mut k_rows = Vec::with_capacity(offsets.len());
        for (row, &offset) in offsets.iter().enumerate() {
            let q_row = mlxcel_core::slice(
                q,
                &[row as i32, 0, 0, 0],
                &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            let k_row = mlxcel_core::slice(
                k,
                &[row as i32, 0, 0, 0],
                &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
            );
            if let Some(table) = self
                .su_rope
                .as_ref()
                .map(|su| su.table_for(offset, seq_len))
            {
                let q_row = table
                    .scale_rotary_prefix(&q_row, self.rope_dims)
                    .unwrap_or(q_row);
                let k_row = table
                    .scale_rotary_prefix(&k_row, self.rope_dims)
                    .unwrap_or(k_row);
                q_rows.push(mlxcel_core::fast_rope_with_freqs(
                    &q_row,
                    self.rope_dims,
                    false,
                    1.0,
                    offset,
                    &table.freqs,
                ));
                k_rows.push(mlxcel_core::fast_rope_with_freqs(
                    &k_row,
                    self.rope_dims,
                    false,
                    1.0,
                    offset,
                    &table.freqs,
                ));
            } else {
                q_rows.push(mlxcel_core::fast_rope(
                    &q_row,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    1.0,
                    offset,
                ));
                k_rows.push(mlxcel_core::fast_rope(
                    &k_row,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    1.0,
                    offset,
                ));
            }
        }
        let q_refs: Vec<&MlxArray> = q_rows.iter().map(|row| row.as_ref().unwrap()).collect();
        let k_refs: Vec<&MlxArray> = k_rows.iter().map(|row| row.as_ref().unwrap()).collect();
        (
            mlxcel_core::concatenate_many(&q_refs, 0),
            mlxcel_core::concatenate_many(&k_refs, 0),
        )
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

    pub fn forward_batched(
        &self,
        x: &MlxArray,
        caches: &mut [&mut KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let q = self.self_attn.q_proj.forward(&normed);
        let k = self.self_attn.k_proj.forward(&normed);
        let v = self.self_attn.v_proj.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(caches, seq_len, 0)
            .expect("valid PhiMoE batched attention metadata");
        let attn = forward_split_attention(&self.self_attn, &q, &k, &v, caches, &metadata, mask);
        let attn_out = self.self_attn.o_proj.forward(&attn);
        let h = mlxcel_core::add(x, &attn_out);
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

    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for layer_idx in 0..self.layers.len() {
            let mut layer_caches: Vec<&mut KVCache> = batch_caches
                .iter_mut()
                .map(|caches| &mut caches[layer_idx])
                .collect();
            h = self.layers[layer_idx].forward_batched(&h, &mut layer_caches, mask);
        }
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
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
        args.validate()?;
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
                // The shared leaf list includes the ModelOpt NVFP4 scale
                // sidecar, which lands as `inv_global_scale` (issue #2351).
                for weight_type in &crate::models::switch_layers::EXPERT_PLANE_LEAVES {
                    let first_key = format!(
                        "{}.block_sparse_moe.experts.0.{}.{}",
                        prefix, orig_name, weight_type
                    );

                    if weights.contains_key(&first_key) {
                        // Borrowed pointers, not copies (issue #948). The
                        // intermediate `mlxcel_core::copy` produced a `Copy`
                        // graph node per expert tensor that the following
                        // `mx::stack` then consumed, and nothing else used, so
                        // it was pure indirection. It was NOT redundant memory
                        // traffic: `Copy::eval` is
                        // `out.copy_shared_buffer(inputs[0])`, which aliases the
                        // input buffer rather than materializing it. See the
                        // longer note in `switch_layers.rs`. `weights` is only
                        // read and is not mutated while the pointers are live;
                        // the stacked result goes into the separate
                        // `new_weights` map and keeps its inputs alive on its
                        // own, because the C++ `stack` shim copies each
                        // refcounted `mx::array` handle into its own vector.
                        let mut expert_ptrs: Vec<*const MlxArray> = Vec::new();
                        for e in 0..args.num_local_experts {
                            let key = format!(
                                "{}.block_sparse_moe.experts.{}.{}.{}",
                                prefix, e, orig_name, weight_type
                            );
                            if let Some(w) = weights.get(&key) {
                                expert_ptrs
                                    .push(w.as_ref().expect("weight map holds no null handles")
                                        as *const MlxArray);
                            }
                        }

                        if !expert_ptrs.is_empty() {
                            let (leaf, stacked) = crate::models::switch_layers::stack_expert_leaf(
                                weight_type,
                                &expert_ptrs,
                            );
                            let new_key = format!(
                                "{}.block_sparse_moe.switch_mlp.{}.{}",
                                prefix, target_name, leaf
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
            jitter_eps: args.router_jitter_noise,
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
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

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

    /// LongRoPE picks its table from the whole prompt length, so a cached
    /// prefix is only valid for a request on the same side of
    /// `original_max_position_embeddings` (as `phi3`, #1358). A checkpoint
    /// with no `longrope` block answers `None`.
    fn rope_table_regime(&self, total_prompt_len: usize) -> Option<u8> {
        let su = self.layers.first()?.self_attn.su_rope.as_ref()?;
        Some(su.regime(total_prompt_len))
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode(self, input_ids, batch_caches, |batch_caches| {
            self.forward_batched_impl(input_ids, batch_caches, mask)
        })
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::*;

    /// ModelOpt NVFP4 experts keep a `weight_scale_2` per expert plane, which
    /// the repack leaves as `experts.{e}.{w}.global_scale`. Stacking must turn
    /// it into the `[num_experts]` output multiplier rather than drop it, and
    /// the stacked plane must load through the shared `SwitchGLU` at the
    /// group size a ModelOpt config declares (issue #2351).
    #[test]
    fn modelopt_expert_sidecars_stack_into_the_output_multiplier() {
        use crate::models::switch_layers::{
            SwitchGLU, expect_native_nvfp4_load, insert_modelopt_expert_plane,
            stacked_inv_global_scale,
        };
        let args: ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "phimoe",
            "num_hidden_layers": 1,
            "num_local_experts": 3,
            "num_experts_per_tok": 2,
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}
            }
        }))
        .expect("tiny config parses");
        assert_eq!(args.group_size(), 16, "ModelOpt NVFP4 blocks are 16 wide");

        let mut weights = WeightMap::new();
        let scale2 = [[0.5f32, 1.5, 2.5], [3.0, 0.25, 1.0], [2.0, 4.0, 0.125]];
        for (j, (leaf, out, input)) in [("w1", 32, 64), ("w2", 64, 32), ("w3", 32, 64)]
            .into_iter()
            .enumerate()
        {
            for (e, scale) in scale2[j].iter().enumerate() {
                insert_modelopt_expert_plane(
                    &mut weights,
                    &format!("model.layers.0.block_sparse_moe.experts.{e}.{leaf}"),
                    out,
                    input,
                    *scale,
                    (e * 3 + j) as u32,
                );
            }
        }
        let stacked = PhiMoeModel::sanitize_weights(&weights, &args).expect("stacking succeeds");
        let prefix = "model.layers.0.block_sparse_moe.switch_mlp";
        for (j, proj) in ["gate_proj", "down_proj", "up_proj"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                stacked_inv_global_scale(&stacked, &format!("{prefix}.{proj}")),
                scale2[j].to_vec(),
                "{proj}"
            );
        }
        expect_native_nvfp4_load(
            SwitchGLU::from_weights(&stacked, prefix, args.group_size(), args.bits()),
            prefix,
            "the stacked ModelOpt plane loads through the shared SwitchGLU",
        );
    }

    /// Per-expert stacking builds its `stack` inputs from pointers borrowed out
    /// of the source `WeightMap` rather than from `mlxcel_core::copy` results
    /// (issue #948). Drop the source map before evaluating anything, so a stack
    /// that held a borrow instead of a refcounted handle reads freed memory
    /// rather than failing an assertion. Inline rather than in
    /// `phimoe_tests.rs`, which is a sibling module and cannot reach the private
    /// `sanitize_weights`; this mirrors how `deepseek_v3.rs` tests its own.
    #[test]
    fn stacked_experts_outlive_the_weight_map_they_were_borrowed_from() {
        let args: ModelArgs = serde_json::from_str(
            r#"{
                "model_type": "phimoe",
                "num_hidden_layers": 1,
                "num_local_experts": 2,
                "num_experts_per_tok": 2
            }"#,
        )
        .expect("tiny config parses");

        // w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj.
        let mut weights = WeightMap::new();
        for e in 0..2 {
            let base = (e + 1) as f32;
            for (leaf, bump) in [("w1", 0.0), ("w2", 10.0), ("w3", 20.0)] {
                weights.insert(
                    format!("model.layers.0.block_sparse_moe.experts.{e}.{leaf}.weight"),
                    mlxcel_core::from_slice_f32(
                        &[
                            base + bump,
                            base + bump + 0.5,
                            base + bump,
                            base + bump + 0.5,
                        ],
                        &[2, 2],
                    ),
                );
            }
        }
        // A non-expert weight, to pin that the passthrough branch still runs.
        weights.insert(
            "model.norm.weight".into(),
            mlxcel_core::from_slice_f32(&[1.0, 1.0], &[2]),
        );

        let stacked = PhiMoeModel::sanitize_weights(&weights, &args).expect("stacking succeeds");
        drop(weights);

        for (proj, bump) in [("gate_proj", 0.0), ("down_proj", 10.0), ("up_proj", 20.0)] {
            let key = format!("model.layers.0.block_sparse_moe.switch_mlp.{proj}.weight");
            let tensor = stacked.get(&key).unwrap_or_else(|| panic!("missing {key}"));
            mlxcel_core::eval(tensor);
            assert_eq!(
                mlxcel_core::array_shape(tensor),
                vec![2, 2, 2],
                "{key} must stack both experts"
            );
            for e in 0..2i32 {
                let first = mlxcel_core::slice(tensor, &[e, 0, 0], &[e + 1, 1, 1]);
                mlxcel_core::eval(&first);
                assert_eq!(
                    mlxcel_core::item_f32(&first),
                    (e + 1) as f32 + bump,
                    "{key} expert {e} landed at the wrong plane or read freed memory"
                );
            }
        }

        assert!(
            stacked.contains_key("model.norm.weight"),
            "non-expert weights must still be carried over"
        );
        assert!(
            !stacked
                .keys()
                .any(|k| k.contains("block_sparse_moe.experts.")),
            "the per-expert keys must not survive stacking"
        );
    }
}
