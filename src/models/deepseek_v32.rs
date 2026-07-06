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

//! DeepSeek V3.2 model implementation using mlxcel-core
//!
//! Key features:
//! - MLA (Multi-head Latent Attention) with LoRA-style projections
//! - Yarn RoPE for extended context (reuses from V2)
//! - MoE with group expert selection
//! - DeepSeek Sparse Attention (DSA) "lightning indexer": per layer, scores
//!   every cached key against the query and attends only to the top-`index_topk`
//!   positions (see [`indexer`]).
//!
//! Short-context parity: when the running `kv_len <= index_topk` the indexer
//! returns no selection and attention reduces to the dense fallback, which is
//! numerically identical to the pre-#509 behavior. The dense fallback can also
//! be forced for A/B via the `MLXCEL_DSA_DENSE` environment variable.
//!
//! The single-token decode path (`L == 1`) gathers the selected KV entries
//! directly with `take_along_axis` instead of materializing a per-step sparse
//! mask; longer prefills build a sparse additive mask (mirrors upstream).
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py

#[path = "deepseek_v32_indexer.rs"]
mod indexer;

#[cfg(test)]
#[path = "deepseek_v32_tests.rs"]
mod deepseek_v32_tests;

use crate::distributed::pipeline::{LayerFilter, StageExecutionOutput};
use indexer::Indexer;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{silu, slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(alias = "type", alias = "rope_type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f32>,
    pub mscale_all_dim: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: usize,

    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: usize,

    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,

    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: usize,

    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,

    // DeepSeek Sparse Attention (DSA) lightning indexer.
    #[serde(default = "default_index_topk")]
    pub index_topk: usize,

    #[serde(default = "default_index_n_heads")]
    pub index_n_heads: usize,

    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: usize,

    /// Indexer RoPE `traditional` flag. Per-model default: `false`
    /// (non-interleaved) for deepseek_v32, `true` (interleaved) for
    /// glm_moe_dsa. `#[serde(default)]` yields `false` for a raw
    /// deepseek_v32 config; glm_moe_dsa supplies `true` explicitly.
    #[serde(default)]
    pub indexer_rope_interleave: bool,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,

    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub attention_bias: bool,

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

fn default_moe_intermediate_size() -> usize {
    1407
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_kv_lora_rank() -> usize {
    512
}
fn default_q_lora_rank() -> usize {
    1536
}
fn default_qk_rope_head_dim() -> usize {
    64
}
fn default_v_head_dim() -> usize {
    128
}
fn default_qk_nope_head_dim() -> usize {
    128
}
pub(crate) fn default_index_topk() -> usize {
    2048
}
pub(crate) fn default_index_n_heads() -> usize {
    64
}
pub(crate) fn default_index_head_dim() -> usize {
    128
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    163840
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
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

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && (layer_idx - self.first_k_dense_replace).is_multiple_of(self.moe_layer_freq)
    }

    pub fn mscale_all_dim(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.mscale_all_dim)
            .unwrap_or(0.0)
    }

    pub fn scaling_factor(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|s| s.factor)
            .unwrap_or(1.0)
    }
}

// MLA Attention (LoRA-style).
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

struct MLAAttention {
    q_a_proj: UnifiedLinear,
    q_a_layernorm: RMSNorm,
    q_b_proj: UnifiedLinear,

    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    embed_q: MultiLinear,
    unembed_out: MultiLinear,

    o_proj: UnifiedLinear,

    // DSA lightning indexer. `None` when the checkpoint carries no indexer
    // weights or when forced dense via `MLXCEL_DSA_DENSE`; in that case the
    // forward pass is byte-identical to the pre-#509 full-attention path.
    indexer: Option<Indexer>,

    num_heads: i32,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    q_head_dim: i32,
    scale: f32,
    rope_base: f32,
}

impl MLAAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // LoRA-style Q projection. `qr` (post-layernorm, pre-q_b_proj) is the
        // LoRA-reduced query hidden the indexer also consumes.
        let q_lora = self.q_a_proj.forward(x);
        let qr = self.q_a_layernorm.forward(&q_lora);
        let q = self.q_b_proj.forward(&qr);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // Compressed KV with MQA
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // kv_latent = layernorm(compressed)
        let kv_latent = self.kv_a_layernorm.forward(&compressed);

        // Apply RoPE. `offset` is the monotonic rope position; `live_before`
        // is the number of K rows already in the cache BEFORE this call's
        // update_and_fetch, which the causal fallback below needs (mirrors
        // deepseek_v3's `live_before`, issue #619).
        let offset = cache.offset;
        let live_before = cache.live_len();
        let q_pe = mlxcel_core::fast_rope(
            &q_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );
        let k_pe = mlxcel_core::fast_rope(
            &k_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );

        // Indexer q/k/weights for the new tokens. The roped indexer key rides
        // alongside `kv_latent` in the KV cache (concatenated on the feature
        // axis) so a decode step can score against every cached position.
        let indexer_new = self.indexer.as_ref().map(|idx| {
            (
                idx.keys(x, offset),
                idx.queries(&qr, offset),
                idx.weights(x),
            )
        });

        // Expand kv_latent for caching: [B, L, kv_lora_rank] → [B, 1, L, kv_lora_rank]
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Cache stores (kv_latent [+ indexer key], k_pe) for memory efficiency.
        let cache_keys = match &indexer_new {
            Some((idx_key, _, _)) => mlxcel_core::concatenate(&kv_latent, idx_key, -1),
            None => kv_latent,
        };
        let (cache_keys, k_pe) = cache.update_and_fetch(cache_keys, k_pe);

        // Slice the cached indexer key back off the shared buffer.
        let (kv_latent, indexer_keys) = match &self.indexer {
            Some(_) => {
                let kvl = slice_axis(&cache_keys, -1, 0, self.kv_lora_rank);
                let ik = slice_axis(&cache_keys, -1, self.kv_lora_rank, -1);
                (kvl, Some(ik))
            }
            None => (cache_keys, None),
        };

        // Causal fallback (issue #619, same class as #618): both standard
        // generation paths pass `mask == None` for prefill, and neither the
        // SDPA wrapper nor the indexer applies implicit causality, so bake a
        // causal mask in here for any maskless multi-token forward. It must
        // reach BOTH consumers: the indexer top-k selection (or future keys
        // get selected) and the additive `pe_scores` mask (which
        // `apply_sparse_prefill_mask` documents as already-causal). Decode
        // (`l == 1`) stays maskless: every cached position is causally valid.
        let causal_storage;
        let effective_mask: Option<&MlxArray> = if mask.is_some() {
            mask
        } else if l > 1 {
            causal_storage = mlxcel_core::utils::create_causal_mask(l, live_before);
            Some(causal_storage.as_ref().unwrap())
        } else {
            None
        };

        // Top-k key selection. `None` at short context (kv_len <= index_topk),
        // in which case attention falls through to the dense path unchanged.
        let topk_indices = match (&self.indexer, &indexer_new, &indexer_keys) {
            (Some(idx), Some((_, idx_q, idx_w)), Some(idx_k)) => {
                idx.top_indices(idx_q, idx_k, idx_w, effective_mask)
            }
            _ => None,
        };

        // Compute positional encoding scores: pe_scores = (q_pe * scale) @ k_pe.T
        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_scalar);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);

        // Apply the (caller or fallback) causal mask to pe_scores.
        let pe_scores = if let Some(m) = effective_mask {
            mlxcel_core::add(&pe_scores, m)
        } else {
            pe_scores
        };

        // MLA attention: different paths for decode vs prefill
        let output = if l == 1 {
            // Decode: project Q into latent space, use kv_latent as K=V
            let q_projected = self.embed_q.forward(&q_nope);

            if let Some(indices) = &topk_indices {
                // Sparse decode fast path: gather the selected KV subset with
                // take_along_axis (no per-step sparse mask). Decode is always
                // called with `mask == None` here, and every gathered position
                // is causally valid, so pe_scores is recomputed maskless over
                // the gathered k_pe.
                let (kv_latent_g, k_pe_g) = gather_topk_kv(
                    &kv_latent,
                    &k_pe,
                    indices,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                );
                let k_pe_g_t = mlxcel_core::transpose_axes(&k_pe_g, &[0, 1, 3, 2]);
                let pe_scores_g = mlxcel_core::matmul(&q_pe_scaled, &k_pe_g_t);
                let pe_mask_ptr = &*pe_scores_g as *const MlxArray;
                let output = unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q_projected,
                        &kv_latent_g,
                        &kv_latent_g,
                        self.scale,
                        pe_mask_ptr,
                        0.0,
                        0,
                    )
                };
                self.unembed_out.forward(&output)
            } else {
                let pe_mask_ptr = &*pe_scores as *const MlxArray;
                let output = unsafe {
                    mlxcel_core::layers::attention_from_ptr(
                        &q_projected,
                        &kv_latent,
                        &kv_latent,
                        self.scale,
                        pe_mask_ptr,
                        0.0,
                        0,
                    )
                };
                // Project output from latent space to v_head_dim
                self.unembed_out.forward(&output)
            }
        } else {
            // Prefill: restrict pe_scores to the selected keys via a sparse
            // additive mask (mirrors upstream's boolean sparse_mask & causal),
            // then project kv_latent to K and V.
            let pe_scores = match &topk_indices {
                Some(indices) => {
                    let kv_len = mlxcel_core::array_shape(&kv_latent)[2];
                    apply_sparse_prefill_mask(&pe_scores, indices, kv_len)
                }
                None => pe_scores,
            };
            let k = self.embed_q.forward_no_transpose(&kv_latent);
            let v = self.unembed_out.forward(&kv_latent);
            let pe_mask_ptr = &*pe_scores as *const MlxArray;
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_nope,
                    &k,
                    &v,
                    self.scale,
                    pe_mask_ptr,
                    0.0,
                    0,
                )
            }
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.v_head_dim]);

        self.o_proj.forward(&output)
    }
}

/// Gather the top-`index_topk` cached `kv_latent`/`k_pe` entries for the single
/// decode query. `indices` is `[b, 1, 1, index_topk]`; the gathered tensors are
/// `[b, 1, index_topk, feat]`. Used by the `L == 1` sparse-decode fast path.
fn gather_topk_kv(
    kv_latent: &MlxArray,
    k_pe: &MlxArray,
    indices: &MlxArray,
    kv_lora_rank: i32,
    qk_rope_head_dim: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let ishape = mlxcel_core::array_shape(indices);
    let b = ishape[0];
    let topk = ishape[3];
    // [b, 1, 1, topk] -> [b, 1, topk, 1] so the gather indexes the key axis (2).
    let idx = slice_axis(indices, 2, 0, 1);
    let idx = mlxcel_core::reshape(&idx, &[b, 1, topk, 1]);

    let kv_idx = mlxcel_core::broadcast_to(&idx, &[b, 1, topk, kv_lora_rank]);
    let kv_latent_g = mlxcel_core::take_along_axis(kv_latent, &kv_idx, 2);

    let pe_idx = mlxcel_core::broadcast_to(&idx, &[b, 1, topk, qk_rope_head_dim]);
    let k_pe_g = mlxcel_core::take_along_axis(k_pe, &pe_idx, 2);

    (kv_latent_g, k_pe_g)
}

/// Build a sparse additive mask for the multi-token prefill path: positions not
/// in the per-query top-k become `-inf`. `pe_scores` already carries the causal
/// mask, so keeping it where selected reproduces upstream's `sparse_mask & mask`.
///
/// `indices` is `[b, 1, s, index_topk]`; `pe_scores` is `[b, n_heads, s, kv_len]`.
fn apply_sparse_prefill_mask(
    pe_scores: &MlxArray,
    indices: &MlxArray,
    kv_len: i32,
) -> UniquePtr<MlxArray> {
    let ishape = mlxcel_core::array_shape(indices);
    let b = ishape[0];
    let s = ishape[2];
    let topk = ishape[3];

    // Scatter 1.0 into the selected columns, then threshold to a bool keep-mask.
    // FP32 scatter avoids relying on boolean put_along_axis semantics; the
    // scatter values match the index shape so the binding needs no broadcast.
    let base = mlxcel_core::zeros(&[b, 1, s, kv_len], mlxcel_core::dtype::FLOAT32);
    let ones = mlxcel_core::ones(&[b, 1, s, topk], mlxcel_core::dtype::FLOAT32);
    let filled = mlxcel_core::put_along_axis(&base, indices, &ones, -1);
    let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
    let keep = mlxcel_core::greater(&filled, &half);

    let neg_inf =
        mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::array_dtype(pe_scores));
    mlxcel_core::where_cond(&keep, pe_scores, &neg_inf)
}

// MLP (Dense and MoE - reusing from V2).
struct DenseMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DenseMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h)
    }
}

struct MoEGate {
    weight: UniquePtr<MlxArray>,
    e_score_correction_bias: UniquePtr<MlxArray>,
    top_k: usize,
    routed_scaling_factor: f32,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
}

impl MoEGate {
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // gates = x @ weight.T
        let weight_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &weight_t);

        // Sigmoid scoring in native dtype (f16 sigmoid saturation at |x|>5.5
        // preserves relative ordering for top-k selection, safe for MoE gating)
        let scores = mlxcel_core::sigmoid(&gates);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add correction bias
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups)
        let scores = if self.n_group > 1 {
            super::switch_layers::group_mask_scores(
                &scores,
                self.n_group as i32,
                self.topk_group as i32,
            )
        } else {
            scores
        };

        // Get top-k expert indices using argpartition
        let k = self.top_k as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, k);

        // Get scores from orig_scores (before bias addition)
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize if needed
        let topk_scores = if self.top_k > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            mlxcel_core::divide(&topk_scores, &sum)
        } else {
            topk_scores
        };

        // Scale scores
        let scale = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_scores),
        );
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        (topk_indices, topk_scores)
    }
}

// Kept local: forward() hardcodes sorted_indices=false, per-expert stacking in load
enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
    },
}

impl SwitchLinear {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => unsafe {
                mlxcel_core::gather_qmm(
                    x,
                    weight,
                    scales,
                    biases
                        .as_ref()
                        .map(|b| b as *const _)
                        .unwrap_or(std::ptr::null()),
                    std::ptr::null(),
                    indices as *const _,
                    true,
                    *group_size,
                    *bits,
                    false,
                    "affine",
                )
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, false)
                }
            }
        }
    }
}

struct SwitchGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        let gate = self.gate_proj.forward(&x_expanded, indices);
        let gate = silu(&gate);
        let up = self.up_proj.forward(&x_expanded, indices);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h, indices)
    }
}

struct MoEBlock {
    gate: MoEGate,
    experts: SwitchGLU,
    shared_experts: Option<DenseMLP>,
}

impl MoEBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.gate.forward(x);
        let y = self.experts.forward(x, &indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &y,
            &scores,
            mlxcel_core::array_dtype(x),
        );

        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }
}

enum MLPType {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

impl MLPType {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPType::Dense(mlp) => mlp.forward(x),
            MLPType::MoE(moe) => moe.forward(x),
        }
    }
}

// Decoder Layer.
struct DecoderLayer {
    self_attn: MLAAttention,
    mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let r = self.self_attn.forward(&normed, mask, cache);
        let h = mlxcel_core::add(x, &r);

        let normed = self.post_attention_layernorm.forward(&h);
        let r = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &r)
    }
}

// DeepSeek V3.2 Model.
pub struct DeepSeekV32Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
}

impl DeepSeekV32Model {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        let mask = mask.map(mlxcel_core::copy);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_deref(), cache);
        }

        let h = self.norm.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, &args);
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Public entry point for sanitizing weights with external args (used by GLM MoE DSA)
    pub fn sanitize_weights_with_args(weights: WeightMap, args: &ModelArgs) -> WeightMap {
        Self::sanitize_weights(weights, args)
    }

    /// Decompose kv_b_proj into embed_q and unembed_out for MLA, and stack expert weights
    fn sanitize_weights(mut weights: WeightMap, args: &ModelArgs) -> WeightMap {
        // Remove multi-token prediction (MTP) layers beyond num_hidden_layers
        let mtp_layer = args.num_hidden_layers;
        weights.retain(|k, _| {
            let parts: Vec<&str> = k.split('.').collect();
            if parts.len() >= 3
                && parts[1] == "layers"
                && let Ok(layer_idx) = parts[2].parse::<usize>()
            {
                return layer_idx < mtp_layer;
            }
            true
        });

        // Stack expert weights if needed (individual experts.N format → switch format)
        if let Some(n_routed) = args.n_routed_experts {
            for l in 0..args.num_hidden_layers {
                let prefix = format!("model.layers.{}", l);
                let first_key = format!("{}.mlp.experts.0.gate_proj.weight", prefix);
                if weights.contains_key(&first_key) {
                    for m in ["gate_proj", "down_proj", "up_proj"] {
                        for k in ["weight", "scales", "biases"] {
                            let check_key = format!("{}.mlp.experts.0.{}.{}", prefix, m, k);
                            if weights.contains_key(&check_key) {
                                let mut expert_arrays = Vec::new();
                                for e in 0..n_routed {
                                    let key = format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                    if let Some(w) = weights.get(&key) {
                                        expert_arrays.push(mlxcel_core::copy(w));
                                    }
                                }
                                if !expert_arrays.is_empty() {
                                    let stacked = stack_arrays(&expert_arrays, 0);
                                    let new_key = format!("{}.mlp.experts.{}.{}", prefix, m, k);
                                    weights.insert(new_key, stacked);
                                    for e in 0..n_routed {
                                        let key =
                                            format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                        weights.remove(&key);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Decompose kv_b_proj into embed_q and unembed_out
        let num_heads = args.num_attention_heads as i32;
        let head_dim = (args.qk_nope_head_dim + args.v_head_dim) as i32;
        let qk_nope_head_dim = args.qk_nope_head_dim as i32;

        for l in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{}.self_attn", l);
            let kv_b_key = format!("{}.kv_b_proj.weight", prefix);
            let embed_q_key = format!("{}.embed_q.weight", prefix);

            // Skip if already decomposed
            if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
                continue;
            }

            // Check if quantized
            let scales_key = format!("{}.kv_b_proj.scales", prefix);
            let is_quantized = weights.contains_key(&scales_key);

            let w = weights.remove(&kv_b_key).unwrap();

            let w_full = if is_quantized {
                let s = weights
                    .remove(&format!("{}.kv_b_proj.scales", prefix))
                    .unwrap();
                let b = weights
                    .remove(&format!("{}.kv_b_proj.biases", prefix))
                    .unwrap();
                let w_shape = mlxcel_core::array_shape(&w);
                let s_shape = mlxcel_core::array_shape(&s);
                let kv_lora_rank = args.kv_lora_rank as i32;
                let inferred_bits = (w_shape[w_shape.len() - 1] * 32) / kv_lora_rank;
                let inferred_gs = kv_lora_rank / s_shape[s_shape.len() - 1];
                unsafe {
                    mlxcel_core::dequantize(
                        &w,
                        &s,
                        &*b as *const _,
                        inferred_gs,
                        inferred_bits,
                        "affine",
                    )
                }
            } else {
                mlxcel_core::copy(&w)
            };

            // Reshape: [num_heads * head_dim, kv_lora_rank] → [num_heads, head_dim, kv_lora_rank]
            let w_3d = mlxcel_core::reshape(&w_full, &[num_heads, head_dim, -1]);

            // Split: wk = [:, :qk_nope_head_dim, :], wv = [:, qk_nope_head_dim:, :]
            let wk = slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
            let wv = slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

            // embed_q: wk.swapaxes(-1, -2) = [num_heads, kv_lora_rank, qk_nope_head_dim]
            let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
            let wk = mlxcel_core::copy(&wk);
            let wv = mlxcel_core::copy(&wv);

            weights.insert(format!("{}.embed_q.weight", prefix), wk);
            weights.insert(format!("{}.unembed_out.weight", prefix), wv);
        }

        // Remove rotary embedding frequencies
        let keys_to_remove: Vec<String> = weights
            .keys()
            .filter(|k| k.contains("rotary_emb.inv_freq"))
            .cloned()
            .collect();
        for key in keys_to_remove {
            weights.remove(&key);
        }

        weights
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = load_decoder_layer(weights, args, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = if !args.tie_word_embeddings {
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

// Weight Loading Helpers.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

fn load_decoder_layer(
    weights: &WeightMap,
    args: &ModelArgs,
    layer_idx: usize,
) -> Result<DecoderLayer, String> {
    let prefix = format!("model.layers.{}", layer_idx);
    let group_size = args.group_size();
    let bits = args.bits();

    let self_attn = load_mla_attention(weights, args, &format!("{}.self_attn", prefix))?;

    let mlp = if args.is_moe_layer(layer_idx) {
        MLPType::MoE(load_moe_block(weights, args, &format!("{}.mlp", prefix))?)
    } else {
        MLPType::Dense(load_dense_mlp(
            weights,
            &format!("{}.mlp", prefix),
            group_size,
            bits,
        )?)
    };

    let input_layernorm_weight =
        get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
    let post_attention_layernorm_weight = get_weight_copy(
        weights,
        &format!("{}.post_attention_layernorm.weight", prefix),
    )?;

    Ok(DecoderLayer {
        self_attn,
        mlp,
        input_layernorm: RMSNorm::new(input_layernorm_weight, args.rms_norm_eps),
        post_attention_layernorm: RMSNorm::new(post_attention_layernorm_weight, args.rms_norm_eps),
    })
}

fn load_mla_attention(
    weights: &WeightMap,
    args: &ModelArgs,
    prefix: &str,
) -> Result<MLAAttention, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    let q_head_dim = args.q_head_dim() as i32;

    // Compute scale with mscale adjustment
    let mut scale = (q_head_dim as f32).powf(-0.5);
    if args.mscale_all_dim() > 0.0 && args.scaling_factor() > 1.0 {
        let ms = yarn_get_mscale(args.scaling_factor(), args.mscale_all_dim());
        scale *= ms * ms;
    }

    // Load embed_q and unembed_out (decomposed from kv_b_proj by sanitize_weights)
    let embed_q =
        MultiLinear::from_weights(weights, &format!("{}.embed_q", prefix), group_size, bits)?;
    let unembed_out = MultiLinear::from_weights(
        weights,
        &format!("{}.unembed_out", prefix),
        group_size,
        bits,
    )?;

    Ok(MLAAttention {
        q_a_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_a_proj", prefix),
            group_size,
            bits,
        )?,
        q_a_layernorm: RMSNorm::new(
            get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?,
            1e-6,
        ),
        q_b_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_b_proj", prefix),
            group_size,
            bits,
        )?,
        kv_a_proj_with_mqa: UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            group_size,
            bits,
        )?,
        kv_a_layernorm: RMSNorm::new(
            get_weight_copy(weights, &format!("{}.kv_a_layernorm.weight", prefix))?,
            1e-6,
        ),
        embed_q,
        unembed_out,
        o_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.o_proj", prefix),
            group_size,
            bits,
        )?,
        indexer: Indexer::load(weights, args, prefix)?,
        num_heads: args.num_attention_heads as i32,
        kv_lora_rank: args.kv_lora_rank as i32,
        qk_nope_head_dim: args.qk_nope_head_dim as i32,
        qk_rope_head_dim: args.qk_rope_head_dim as i32,
        v_head_dim: args.v_head_dim as i32,
        q_head_dim,
        scale,
        rope_base: args.rope_theta,
    })
}

fn load_dense_mlp(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<DenseMLP, String> {
    Ok(DenseMLP {
        gate_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?,
        up_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.up_proj", prefix),
            group_size,
            bits,
        )?,
        down_proj: UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?,
    })
}

fn load_moe_block(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<MoEBlock, String> {
    let group_size = args.group_size();
    let bits = args.bits();
    let n_routed = args.n_routed_experts.unwrap_or(1);

    let gate_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
    let e_score_correction_bias =
        get_weight_copy(weights, &format!("{}.gate.e_score_correction_bias", prefix))?;
    let gate = MoEGate {
        weight: gate_weight,
        e_score_correction_bias,
        top_k: args.num_experts_per_tok,
        routed_scaling_factor: args.routed_scaling_factor,
        n_group: args.n_group,
        topk_group: args.topk_group,
        norm_topk_prob: args.norm_topk_prob,
    };

    let experts = load_switch_glu(
        weights,
        &format!("{}.experts", prefix),
        n_routed,
        group_size,
        bits,
    )?;

    let shared_experts = if let Some(_n_shared) = args.n_shared_experts {
        Some(DenseMLP {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.shared_experts.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    } else {
        None
    };

    Ok(MoEBlock {
        gate,
        experts,
        shared_experts,
    })
}

fn load_switch_glu(
    weights: &WeightMap,
    prefix: &str,
    num_experts: usize,
    group_size: i32,
    bits: i32,
) -> Result<SwitchGLU, String> {
    Ok(SwitchGLU {
        gate_proj: load_switch_linear(weights, num_experts, prefix, "gate_proj", group_size, bits)?,
        up_proj: load_switch_linear(weights, num_experts, prefix, "up_proj", group_size, bits)?,
        down_proj: load_switch_linear(weights, num_experts, prefix, "down_proj", group_size, bits)?,
    })
}

fn load_switch_linear(
    weights: &WeightMap,
    num_experts: usize,
    prefix: &str,
    weight_name: &str,
    group_size: i32,
    bits: i32,
) -> Result<SwitchLinear, String> {
    let mut expert_weights = Vec::with_capacity(num_experts);

    // Check if first expert has scales (quantized vs non-quantized)
    let is_quantized = weights.contains_key(&format!("{}.0.{}.scales", prefix, weight_name));

    if is_quantized {
        let mut expert_scales = Vec::with_capacity(num_experts);
        let mut expert_biases = Vec::with_capacity(num_experts);

        for expert_idx in 0..num_experts {
            expert_weights.push(get_weight_copy(
                weights,
                &format!("{}.{}.{}.weight", prefix, expert_idx, weight_name),
            )?);
            expert_scales.push(get_weight_copy(
                weights,
                &format!("{}.{}.{}.scales", prefix, expert_idx, weight_name),
            )?);
            expert_biases.push(get_weight_copy(
                weights,
                &format!("{}.{}.{}.biases", prefix, expert_idx, weight_name),
            )?);
        }

        let weight = mlxcel_core::utils::stack_arrays(&expert_weights, 0);
        let scales = mlxcel_core::utils::stack_arrays(&expert_scales, 0);
        let biases = mlxcel_core::utils::stack_arrays(&expert_biases, 0);
        Ok(SwitchLinear::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    } else {
        for expert_idx in 0..num_experts {
            expert_weights.push(get_weight_copy(
                weights,
                &format!("{}.{}.{}.weight", prefix, expert_idx, weight_name),
            )?);
        }
        let weight = mlxcel_core::utils::stack_arrays(&expert_weights, 0);
        Ok(SwitchLinear::Regular { weight })
    }
}

pub(crate) struct DeepSeekV32StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<DecoderLayer>,
    norm: Option<RMSNorm>,
    lm_head: Option<UnifiedLinear>,
}

impl DeepSeekV32StageModel {
    pub(crate) fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let load_embeddings =
            filter.has_embedding || (args.tie_word_embeddings && filter.has_lm_head);

        let embed_tokens = if load_embeddings {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "model.embed_tokens",
                group_size,
                bits,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            layers.push(load_decoder_layer(weights, args, layer_idx)?);
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let norm = if filter.has_lm_head {
            Some(RMSNorm::new(
                get_weight_copy(weights, "model.norm.weight")?,
                args.rms_norm_eps,
            ))
        } else {
            None
        };

        let lm_head = if filter.has_lm_head && !args.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub(crate) fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
    ) -> Result<StageExecutionOutput, String> {
        let hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        self.execute_hidden(hidden, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        hidden_states: UniquePtr<MlxArray>,
        caches: &mut [KVCache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }
        self.execute_hidden(hidden_states, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [KVCache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(&hidden, None, cache);
        }

        let hidden = if let Some(norm) = &self.norm {
            norm.forward(&hidden)
        } else {
            hidden
        };

        if self.filter.has_lm_head {
            let logits = if let Some(lm_head) = &self.lm_head {
                lm_head.forward(&hidden)
            } else {
                self.embed_tokens
                    .as_ref()
                    .ok_or_else(|| {
                        "final tied-word-embedding stage missing embeddings".to_string()
                    })?
                    .as_linear(&hidden)
            };
            Ok(StageExecutionOutput::Logits(logits))
        } else {
            Ok(StageExecutionOutput::HiddenStates(hidden))
        }
    }
}

// LanguageModel trait implementation.
impl LanguageModel for DeepSeekV32Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        DeepSeekV32Model::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        DeepSeekV32Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![100001] // <|end▁of▁sentence|>
    }
}
