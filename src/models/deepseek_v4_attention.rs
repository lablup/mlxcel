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

//! DeepSeek-V4 attention (`LocalAttention` / `CompressedAttention` /
//! `SparseCompressedAttention` and `v4_attention_factory` in the reference).
//!
//! The three reference classes share the whole projection pipeline and
//! differ only in what they concatenate onto the local sliding-window KV, so
//! this port models them as one struct with an [`AttnKind`], selected per
//! layer from `compress_ratios[layer_idx]` exactly as `v4_attention_factory`
//! does: `0` local, `128` compressed, `4` sparse-compressed.
//!
//! Shared pipeline facts that are easy to get silently wrong:
//!
//! * `num_key_value_heads == 1`: KV is a SINGLE shared head of width
//!   `head_dim` (512), broadcast across the 64 query heads by SDPA.
//! * Q is RMS-normed WITHOUT a weight after the head reshape
//!   (`mx.fast.rms_norm(q, None, eps)`).
//! * Per-head learned sinks (`attn_sink`, float32) join every softmax.
//! * The attention OUTPUT is un-rotated (`rope(out, offset, inverse=True)`)
//!   before the grouped `wo_a` (`MultiLinear`) / `wo_b` output projection.
//! * The local KV lives in a `RotatingKVCache` of `sliding_window` (128).
//! * Local layers rope at `rope_theta` with NO Yarn scaling; compressed and
//!   sparse layers rope at `compress_rope_theta` WITH the Yarn config.
//!
//! The sparse path's `_sparse_pooled_attention` computes a split softmax over
//! the local KV and the gathered top-k pooled KV sharing one log-normalizer
//! (plus the per-head sink), in float32.

use mlxcel_core::layers::{MultiLinear, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::compress::{Compressor, align_local_mask, extend_mask, pool_mask_additive};
use super::indexer::Indexer;
use super::rope::V4Rope;
use super::{ModelArgs, OVERLAP_COMPRESS_RATIO, V4LayerCache, get_weight_copy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttnKind {
    Local,
    Compressed,
    Sparse,
}

impl AttnKind {
    pub(crate) fn for_ratio(ratio: i32) -> Result<Self, String> {
        match ratio {
            0 => Ok(Self::Local),
            128 => Ok(Self::Compressed),
            OVERLAP_COMPRESS_RATIO => Ok(Self::Sparse),
            other => Err(format!(
                "Unsupported DeepSeek-V4 compress ratio {other} (supported: 0, 4, 128)"
            )),
        }
    }
}

pub(crate) struct V4Attention {
    pub(crate) kind: AttnKind,
    wq_a: UnifiedLinear,
    q_norm: RMSNorm,
    wq_b: UnifiedLinear,
    wkv: UnifiedLinear,
    kv_norm: RMSNorm,
    wo_a: MultiLinear,
    wo_b: UnifiedLinear,
    /// `[n_heads]` float32 learned sinks.
    attn_sink: UniquePtr<MlxArray>,
    rope: V4Rope,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    n_heads: i32,
    head_dim: i32,
    o_groups: i32,
    o_lora_rank: i32,
    rms_norm_eps: f32,
    scale: f32,
}

impl V4Attention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let ratio = args.compress_ratios[layer_idx] as i32;
        let kind = AttnKind::for_ratio(ratio)?;
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.head_dim as i32;
        let n_heads = args.num_attention_heads as i32;
        let index_head_dim = args.index_head_dim as i32;

        let rope = match kind {
            AttnKind::Local => V4Rope::new(
                args.qk_rope_head_dim as i32,
                args.rope_theta,
                None,
                1,
                &[(head_dim, false), (head_dim, true)],
            )?,
            AttnKind::Compressed => V4Rope::new(
                args.qk_rope_head_dim as i32,
                args.compress_rope_theta,
                args.rope_scaling.as_ref(),
                1,
                &[(head_dim, false), (head_dim, true)],
            )?,
            AttnKind::Sparse => V4Rope::new(
                args.qk_rope_head_dim as i32,
                args.compress_rope_theta,
                args.rope_scaling.as_ref(),
                1,
                &[(head_dim, false), (head_dim, true), (index_head_dim, false)],
            )?,
        };

        let compressor = match kind {
            AttnKind::Local => None,
            AttnKind::Compressed | AttnKind::Sparse => Some(Compressor::from_weights(
                weights,
                args,
                &format!("{prefix}.compressor"),
                ratio,
                head_dim,
            )?),
        };
        let indexer = match kind {
            AttnKind::Sparse => Some(Indexer::from_weights(
                weights,
                args,
                &format!("{prefix}.indexer"),
                ratio,
            )?),
            _ => None,
        };

        let attn_sink = get_weight_copy(weights, &format!("{prefix}.attn_sink"))?;
        let sink_shape = mlxcel_core::array_shape(&attn_sink);
        if sink_shape != [n_heads] {
            return Err(format!(
                "{prefix}.attn_sink: expected shape [{n_heads}], checkpoint ships {sink_shape:?}"
            ));
        }

        Ok(Self {
            kind,
            wq_a: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wq_a"),
                group_size,
                bits,
            )?,
            q_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{prefix}.q_norm.weight"))?,
                args.rms_norm_eps,
            ),
            wq_b: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wq_b"),
                group_size,
                bits,
            )?,
            wkv: UnifiedLinear::from_weights(weights, &format!("{prefix}.wkv"), group_size, bits)?,
            kv_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{prefix}.kv_norm.weight"))?,
                args.rms_norm_eps,
            ),
            wo_a: MultiLinear::from_weights(weights, &format!("{prefix}.wo_a"), group_size, bits)?,
            wo_b: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wo_b"),
                group_size,
                bits,
            )?,
            attn_sink,
            rope,
            compressor,
            indexer,
            n_heads,
            head_dim,
            o_groups: args.o_groups as i32,
            o_lora_rank: args.o_lora_rank as i32,
            rms_norm_eps: args.rms_norm_eps,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut V4LayerCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        // Monotonic token offset BEFORE this call's update, used for RoPE,
        // pooling window bases, and the pooled-visibility mask.
        let offset = cache.local.offset;

        let q_residual = self.q_norm.forward(&self.wq_a.forward(x));
        let q = self.wq_b.forward(&q_residual);
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let q = mlxcel_core::fast_rms_norm_no_weight(&q, self.rms_norm_eps);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q = self.rope.apply(&q, offset, false);

        let kv = self.kv_norm.forward(&self.wkv.forward(x));
        let kv = mlxcel_core::reshape(&kv, &[b, 1, l, self.head_dim]);
        let kv = self.rope.apply(&kv, offset, false);
        // Single shared KV head: K and V are the same tensor. Store it as
        // both so the rotating window trims them in lockstep.
        let kv_values = mlxcel_core::copy(&kv);
        let (kv, _values) = cache.local.update_and_fetch(kv, kv_values);
        let local_len = mlxcel_core::array_shape(&kv)[2];

        let aligned_mask = mask.map(|m| align_local_mask(m, local_len));
        let sinks = mlxcel_core::astype(&self.attn_sink, mlxcel_core::array_dtype(&q));

        let out = match self.kind {
            AttnKind::Local => self.dense_sdpa(&q, &kv, aligned_mask.as_deref(), &sinks),
            AttnKind::Compressed => {
                let compressor = self.compressor.as_ref().expect("compressed layer");
                let pool = cache.pool.as_mut().expect("compressed layer pool cache");
                let ratio = compressor.ratio;
                let pooled = compressor.forward(x, pool, offset);
                let np = mlxcel_core::array_shape(&pooled)[1];
                if np > 0 {
                    let pmask = pool_mask_additive(l, offset, ratio, np);
                    let kv_full =
                        mlxcel_core::concatenate(&kv, &mlxcel_core::expand_dims(&pooled, 1), 2);
                    let full_mask = aligned_mask
                        .as_ref()
                        .map(|m| extend_mask(m, pmask.as_deref(), np));
                    self.dense_sdpa(&q, &kv_full, full_mask.as_deref(), &sinks)
                } else {
                    self.dense_sdpa(&q, &kv, aligned_mask.as_deref(), &sinks)
                }
            }
            AttnKind::Sparse => {
                let compressor = self.compressor.as_ref().expect("sparse layer");
                let indexer = self.indexer.as_ref().expect("sparse layer");
                let ratio = compressor.ratio;
                let pooled = {
                    let pool = cache.pool.as_mut().expect("sparse layer pool cache");
                    compressor.forward(x, pool, offset)
                };
                let np = mlxcel_core::array_shape(&pooled)[1];
                let pmask = pool_mask_additive(l, offset, ratio, np);
                let topk = {
                    let idx_pool = cache.idx_pool.as_mut().expect("sparse layer indexer cache");
                    indexer.forward(x, &q_residual, &self.rope, idx_pool, offset)
                };

                if np == 0 {
                    self.dense_sdpa(&q, &kv, aligned_mask.as_deref(), &sinks)
                } else if np <= indexer.index_topk {
                    // Short context: dense concat of local + pooled, exactly
                    // the compressed path.
                    let kv_full =
                        mlxcel_core::concatenate(&kv, &mlxcel_core::expand_dims(&pooled, 1), 2);
                    let full_mask = aligned_mask
                        .as_ref()
                        .map(|m| extend_mask(m, pmask.as_deref(), np));
                    self.dense_sdpa(&q, &kv_full, full_mask.as_deref(), &sinks)
                } else {
                    let topk = topk.expect(
                        "indexer returns selections whenever pooled rows exist \
                         (same ratio over the same token stream)",
                    );
                    // Gather each query's pooled-visibility values at its
                    // selected indices: `[L, Np]` additive -> `[B, 1, L, k]`.
                    let sparse_mask = pmask.as_ref().map(|pm| {
                        let pm3 = mlxcel_core::expand_dims(pm, 0);
                        let gathered = mlxcel_core::take_along_axis(&pm3, &topk, 2);
                        mlxcel_core::expand_dims(&gathered, 1)
                    });
                    sparse_pooled_attention(
                        &q,
                        &kv,
                        &pooled,
                        &topk,
                        aligned_mask.as_deref(),
                        sparse_mask.as_deref(),
                        self.scale,
                        &sinks,
                    )
                }
            }
        };

        // Un-rotate the attention output, then the grouped output projection.
        let out = self.rope.apply(&out, offset, true);
        let heads_per_group = self.n_heads / self.o_groups;
        let out =
            mlxcel_core::reshape(&out, &[b, self.o_groups, heads_per_group, l, self.head_dim]);
        let out = mlxcel_core::transpose_axes(&out, &[0, 1, 3, 2, 4]);
        let out = mlxcel_core::reshape(
            &out,
            &[b, self.o_groups, l, heads_per_group * self.head_dim],
        );
        let out = self.wo_a.forward(&out);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.o_groups * self.o_lora_rank]);
        self.wo_b.forward(&out)
    }

    fn dense_sdpa(
        &self,
        q: &MlxArray,
        kv: &MlxArray,
        mask: Option<&MlxArray>,
        sinks: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mask_ptr = mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let sinks_ptr = sinks as *const MlxArray;
        unsafe {
            mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                q, kv, kv, self.scale, mask_ptr, sinks_ptr,
            )
        }
    }
}

/// `_sparse_pooled_attention`: split softmax over the local KV and the
/// gathered top-k pooled KV sharing one log-normalizer plus the per-head
/// sink. Scores are computed in float32; `sinks` arrives in the activation
/// dtype exactly as the reference passes it (the `logaddexp` against the f32
/// normalizer promotes it), and the output is cast back to `q`'s dtype.
#[allow(clippy::too_many_arguments)]
fn sparse_pooled_attention(
    q: &MlxArray,
    local_kv: &MlxArray,
    pooled: &MlxArray,
    topk: &MlxArray,
    local_mask: Option<&MlxArray>,
    pooled_mask: Option<&MlxArray>,
    scale: f32,
    sinks: &MlxArray,
) -> UniquePtr<MlxArray> {
    let q_shape = mlxcel_core::array_shape(q);
    let (b, h, l, d) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let k = *mlxcel_core::array_shape(topk).last().expect("topk rank");
    let q_dtype = mlxcel_core::array_dtype(q);

    // Gather each query's selected pooled rows: [B, Np, D] -> [B, L, k, D].
    let idx = mlxcel_core::reshape(topk, &[b, l * k]);
    let idx = mlxcel_core::expand_dims(&idx, -1);
    let idx = mlxcel_core::broadcast_to(&idx, &[b, l * k, d]);
    let gathered = mlxcel_core::take_along_axis(pooled, &idx, 1);
    let gathered = mlxcel_core::reshape(&gathered, &[b, l, k, d]);
    let gathered = mlxcel_core::astype(&gathered, mlxcel_core::dtype::FLOAT32);

    let q_f = mlxcel_core::astype(q, mlxcel_core::dtype::FLOAT32);
    let q_f = mlxcel_core::multiply_scalar(&q_f, scale);
    let local_f = mlxcel_core::astype(local_kv, mlxcel_core::dtype::FLOAT32);

    // Local scores and normalizer.
    let local_t = mlxcel_core::transpose_axes(&local_f, &[0, 1, 3, 2]);
    let mut local_scores = mlxcel_core::matmul(&q_f, &local_t); // [B, H, L, S]
    if let Some(m) = local_mask {
        local_scores = mlxcel_core::add(&local_scores, m);
    }
    let mut normalizer = mlxcel_core::logsumexp_axis(&local_scores, -1, true);

    // Pooled scores share the normalizer.
    let q_bl = mlxcel_core::transpose_axes(&q_f, &[0, 2, 1, 3]); // [B, L, H, D]
    let gathered_t = mlxcel_core::transpose_axes(&gathered, &[0, 1, 3, 2]);
    let pooled_scores = mlxcel_core::matmul(&q_bl, &gathered_t); // [B, L, H, k]
    let mut pooled_scores = mlxcel_core::transpose_axes(&pooled_scores, &[0, 2, 1, 3]);
    if let Some(m) = pooled_mask {
        pooled_scores = mlxcel_core::add(&pooled_scores, m);
    }
    let pooled_norm = mlxcel_core::logsumexp_axis(&pooled_scores, -1, true);
    normalizer = mlxcel_core::logaddexp(&normalizer, &pooled_norm);

    // Per-head sink joins the shared normalizer.
    let sinks = mlxcel_core::reshape(sinks, &[1, h, 1, 1]);
    normalizer = mlxcel_core::logaddexp(&normalizer, &sinks);

    let local_weights = mlxcel_core::exp(&mlxcel_core::subtract(&local_scores, &normalizer));
    let pooled_weights = mlxcel_core::exp(&mlxcel_core::subtract(&pooled_scores, &normalizer));

    let out = mlxcel_core::matmul(&local_weights, &local_f); // [B, H, L, D]
    let pw_bl = mlxcel_core::transpose_axes(&pooled_weights, &[0, 2, 1, 3]); // [B, L, H, k]
    let pooled_out = mlxcel_core::matmul(&pw_bl, &gathered); // [B, L, H, D]
    let pooled_out = mlxcel_core::transpose_axes(&pooled_out, &[0, 2, 1, 3]);
    let out = mlxcel_core::add(&out, &pooled_out);
    mlxcel_core::astype(&out, q_dtype)
}
