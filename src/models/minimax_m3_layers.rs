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

//! MiniMax-M3 attention block and the clamp-SwiGLU ("swigluoai") activation.
//!
//! Attention is GQA (64 query heads, 4 KV heads, head_dim 128) with per-head
//! Gemma-style Q/K RMSNorm (`qk_norm_type: "per_head"`) and partial RoPE
//! (`rotary_dim` of `head_dim`). On the sparse layers a block-sparse indexer is
//! attached: its index key is cached on the regular K buffer (feature-axis
//! concat) and, once the live cache outgrows the selected window, it contributes
//! an additive block mask on top of the causal mask.

use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::indexer::{BlockSparseIndexer, check_out_dim};
use super::{ModelArgs, SparseAttentionConfig, get_weight_copy};
use crate::models::gemma::GemmaRMSNorm;

/// Clamp-SwiGLU ("swigluoai") activation shared with gpt-oss.
///
/// `out = clamp(x_glu, max=limit) * sigmoid(alpha * clamp(x_glu, max=limit)) *
/// (clamp(x_linear, [-limit, limit]) + 1)`. For the MiniMax-M3 defaults
/// (`swiglu_alpha = 1.702`, `swiglu_limit = 7.0`) this reuses the compiled
/// gpt-oss kernel (`mlxcel_core::compiled_gpt_oss_swiglu_activation`), which
/// hardcodes alpha 1.702 / beta 1.0; other values fall back to the explicit
/// clamp path.
pub(super) fn swigluoai(
    x_linear: &MlxArray,
    x_glu: &MlxArray,
    alpha: f32,
    limit: f32,
) -> UniquePtr<MlxArray> {
    if (alpha - 1.702).abs() <= f32::EPSILON && (limit - 7.0).abs() <= f32::EPSILON {
        return mlxcel_core::compiled_gpt_oss_swiglu_activation(x_linear, x_glu);
    }

    let dtype = mlxcel_core::array_dtype(x_linear);
    let neg_limit = mlxcel_core::full_f32(&[1], -limit, dtype);
    let pos_limit = mlxcel_core::full_f32(&[1], limit, dtype);
    let x_glu = mlxcel_core::minimum(x_glu, &pos_limit);
    let x_linear = mlxcel_core::maximum(x_linear, &neg_limit);
    let x_linear = mlxcel_core::minimum(&x_linear, &pos_limit);

    let alpha_arr = mlxcel_core::full_f32(&[1], alpha, dtype);
    let glu_scaled = mlxcel_core::multiply(&alpha_arr, &x_glu);
    let sig = mlxcel_core::sigmoid(&glu_scaled);
    let out_glu = mlxcel_core::multiply(&x_glu, &sig);

    let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
    let x_linear_plus_1 = mlxcel_core::add(&x_linear, &one);
    let result = mlxcel_core::multiply(&out_glu, &x_linear_plus_1);
    mlxcel_core::astype(&result, dtype)
}

/// GQA attention with per-head Q/K norm, partial RoPE, and an optional
/// block-sparse indexer.
pub(super) struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
    indexer: Option<BlockSparseIndexer>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_dims: i32,
    rope_base: f32,
}

impl Attention {
    pub(super) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // [b, l, heads, head_dim] -> per-head RMSNorm on the last axis.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // [b, heads, l, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let live_before = cache.live_len();

        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        // The single-head index key for the new tokens rides on the regular K
        // buffer via a head-axis concat (index_dim == head_dim, guaranteed at
        // load), so a later decode step can score against every cached position.
        // Combined K is [b, num_kv_heads + 1, s, head_dim]; the extra head is the
        // index key, sliced back off after the cache fetch.
        let index_new = self
            .indexer
            .as_ref()
            .map(|idx| (idx.keys(x, offset), idx.queries(x, offset)));

        let cache_keys = match &index_new {
            Some((idx_key, _)) => mlxcel_core::concatenate(&k, idx_key, 1),
            None => k,
        };
        let (cache_keys, cache_v) = cache.update_and_fetch(cache_keys, v);

        let (cache_k, index_keys) = match &self.indexer {
            Some(_) => {
                let kk = slice_axis(&cache_keys, 1, 0, self.num_kv_heads);
                let ik = slice_axis(&cache_keys, 1, self.num_kv_heads, self.num_kv_heads + 1);
                (kk, Some(ik))
            }
            None => (cache_keys, None),
        };

        let kv_len = mlxcel_core::array_shape(&cache_k)[2];

        // Build the effective mask. Non-indexer layers keep the dense fast path
        // (causal_attention) when no mask is supplied; indexer layers always
        // build at least a causal mask so the block mask can fold into it.
        let causal_storage;
        let causal_mask: Option<&MlxArray> = if mask.is_some() {
            mask
        } else if l > 1 {
            causal_storage = mlxcel_core::utils::create_causal_mask(l, live_before);
            Some(causal_storage.as_ref().unwrap())
        } else {
            None
        };

        // Index scores for this step, computed once and shared by the fused
        // sparse decode below and the mask path that backs it up.
        let sparse_ctx = match (&self.indexer, &index_new, &index_keys) {
            (Some(idx), Some((_, index_q)), Some(index_k)) if idx.should_apply_sparse(kv_len) => {
                Some((idx, idx.token_scores(index_q, index_k, causal_mask)))
            }
            _ => None,
        };

        // Fused sparse decode (issue #904): attend to the selected blocks
        // through a page table instead of computing every dropped block and
        // then adding `-inf` to it. Declines fall through to the mask path
        // below with the reason announced once per kind.
        if l == 1
            && let Some((idx, token_scores)) = &sparse_ctx
            && let Some(attn_out) = self.try_fused_sparse_decode(&q, cache, idx, token_scores, kv_len)
        {
            let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
            let attn_out =
                mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
            return self.o_proj.forward(&attn_out);
        }

        let sparse_storage;
        let effective_mask: Option<&MlxArray> = match &sparse_ctx {
            Some((idx, token_scores)) => {
                let block_drop = idx.block_drop_mask(token_scores, offset);
                sparse_storage = match causal_mask {
                    Some(m) => mlxcel_core::add(m, &block_drop),
                    None => block_drop,
                };
                Some(sparse_storage.as_ref().unwrap())
            }
            None => causal_mask,
        };

        let attn_out = if l > 1 && effective_mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = effective_mask
                .map(|m| m as *const _)
                .unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    /// Attend to the indexer's selected blocks through a `page_size = 1` page
    /// table, or `None` to fall back to the additive-mask path (issue #904).
    ///
    /// The win is structural rather than incremental. The mask path computes
    /// every score over the full live window and then adds `-inf` to the
    /// dropped ones, so the dropped blocks cost exactly as much as the kept
    /// ones; here the kernel visits the selected rows and nothing else. The
    /// selection is a device value throughout, so no synchronization happens.
    ///
    /// Every early return announces itself through
    /// [`mlxcel_core::paged_v2::report_sparse_outcome_once`], once per kind. A
    /// sparse path that silently falls back to dense looks exactly like
    /// "sparsity does not help", and this decode path runs under the `mlxcel`
    /// CLI, which installs no tracing subscriber.
    fn try_fused_sparse_decode(
        &self,
        q: &MlxArray,
        cache: &KVCache,
        idx: &BlockSparseIndexer,
        token_scores: &MlxArray,
        kv_len: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        use mlxcel_core::cache::sparse_csr::{ContiguousCacheLayout, selection_from_blocks};
        use mlxcel_core::paged_v2::{
            SparseDecodeInputs, SparseDecodeOutcome, report_sparse_outcome_once, run_sparse_decode,
            sparse_paged_enabled,
        };

        if !sparse_paged_enabled() {
            report_sparse_outcome_once(&SparseDecodeOutcome::KillSwitch);
            return None;
        }
        let Some(shape) = idx.decode_shape(kv_len) else {
            report_sparse_outcome_once(&SparseDecodeOutcome::NotServable(
                "the block budget covers the whole context",
            ));
            return None;
        };
        // The raw allocations, not the fetched window: the window is a
        // token-axis slice of a step-padded buffer and reshaping it into a pool
        // view would copy the whole cache.
        let Some((k_alloc, v_alloc, live_len)) = cache.raw_kv_allocations() else {
            report_sparse_outcome_once(&SparseDecodeOutcome::NotServable(
                "the cache rows are not raw fp16 K/V",
            ));
            return None;
        };
        if live_len != kv_len {
            report_sparse_outcome_once(&SparseDecodeOutcome::NotServable(
                "the cache window disagrees with the attended length",
            ));
            return None;
        }
        // The K allocation carries the index key as one extra head, so it is
        // the wider of the two; the row mapping is checked against both inside
        // `run_sparse_decode`.
        let layout = match ContiguousCacheLayout::from_shape(&mlxcel_core::array_shape(k_alloc)) {
            Ok(layout) => layout,
            Err(reason) => {
                report_sparse_outcome_once(&SparseDecodeOutcome::UnservableLayout(reason));
                return None;
            }
        };

        let blocks = idx.decode_blocks(token_scores, &shape);
        let selection = match selection_from_blocks(
            &layout,
            self.num_kv_heads,
            idx.block_size(),
            &blocks,
            shape.tail_start,
            live_len,
        ) {
            Ok(selection) => selection,
            Err(reason) => {
                report_sparse_outcome_once(&SparseDecodeOutcome::SelectionRejected(reason));
                return None;
            }
        };

        let inputs = SparseDecodeInputs {
            q,
            k_alloc,
            v_alloc,
            kv_heads: self.num_kv_heads,
            scale: self.scale,
        };
        let (out, outcome) = run_sparse_decode(&inputs, &selection);
        report_sparse_outcome_once(&outcome);
        out.map(|o| mlxcel_core::astype(&o, mlxcel_core::array_dtype(q)))
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        sparse: Option<&SparseAttentionConfig>,
        prefix: &str,
    ) -> Result<Self, String> {
        if args.attention_output_gate {
            return Err(
                "minimax_m3: attention_output_gate is not supported by this text decoder"
                    .to_string(),
            );
        }
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;

        // Defensive shape checks: fail with a clear error on a GQA layout
        // mismatch instead of corrupting the forward pass. q_proj is
        // `[num_heads * head_dim, hidden]`, k/v_proj are
        // `[num_kv_heads * head_dim, hidden]`.
        check_out_dim(
            weights,
            &format!("{}.q_proj.weight", prefix),
            num_heads * head_dim,
        )?;
        check_out_dim(
            weights,
            &format!("{}.k_proj.weight", prefix),
            num_kv_heads * head_dim,
        )?;
        check_out_dim(
            weights,
            &format!("{}.v_proj.weight", prefix),
            num_kv_heads * head_dim,
        )?;

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Per-head Q/K norm (Gemma-style weight+1), weight width == head_dim.
        let q_norm = GemmaRMSNorm::new(
            get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?,
            args.rms_norm_eps,
        );
        let k_norm = GemmaRMSNorm::new(
            get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?,
            args.rms_norm_eps,
        );

        let indexer = match sparse {
            Some(cfg) => BlockSparseIndexer::load(weights, args, cfg, prefix)?,
            None => None,
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            indexer,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dims: args.rotary_dim as i32,
            rope_base: args.rope_theta,
        })
    }
}

/// Dense SwiGLU MLP for the leading dense layers (`dense_intermediate_size`
/// wide) and for the separate MoE shared expert. Uses the same clamp-SwiGLU
/// activation as the routed experts.
pub(super) struct DenseMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    alpha: f32,
    limit: f32,
}

impl DenseMlp {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        // gate is the gated ("glu") branch, up is the linear branch.
        let activated = swigluoai(&up, &gate, self.alpha, self.limit);
        self.down_proj.forward(&activated)
    }

    pub(super) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
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
            alpha: args.swiglu_alpha,
            limit: args.swiglu_limit,
        })
    }
}
