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

//! Hy4 gated MLA attention with learnable sinks and cross-layer top-k
//! sharing (`HyV4Attention` in `hy_v4/language.py` upstream).
//!
//! The projections, the latent cache layout and the DSA indexer are the
//! DeepSeek-V3.2 ones ([`MLAAttention`], loaded by `load_mla_attention`).
//! Three things differ, which is why the forward lives here instead of
//! growing parameters on the V3.2 forward:
//!
//! * **Sinks.** `learnable_sink_param [heads]` is one extra logit per head in
//!   the softmax denominator with no value row. None of the V3.2 kernels
//!   (`rope_folded_attention`, the fused absorbed decode, the packed
//!   quantized decode) takes a sink, so every step here runs MLX's SDPA with
//!   sinks: a single query attends absorbed against the latent with the rope
//!   scores as its additive mask, and a multi-token step materializes
//!   per-head K/V and folds the rope half into the head dimension (#3071).
//! * **Gate.** `sigmoid(linear_gate(x))`, `[B, heads, L, v_head_dim]`, is
//!   multiplied into the per-head output before `o_proj`.
//! * **Shared top-k.** A `full` layer owns the indexer and caches the indexer
//!   key beside the latent, `[kv_latent | indexer key]`, as V3.2 does. A
//!   `shared` layer owns no indexer, caches the latent alone, and attends to
//!   the positions the nearest earlier `full` layer selected in this forward
//!   (`None` there means dense).
//!
//! A quantized latent window (`--mla-latent-quant int8` / `int4`) is read the
//! way the V3.2 multi-token step reads it: the indexer scores the indexer-key
//! columns, the selected rows are gathered from the packed triple, and only
//! those rows are dequantized before the sink SDPA. The packed decode kernels
//! are not used because they cannot carry a sink.

use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::mla::{FoldedMask, LatentWindow, MlaDecodePath};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::{MLAAttention, apply_sparse_prefill_mask, latent};
use crate::models::mla_batched::{dense_latent_in, store_latent_step};

/// One Hy4 attention block.
pub(super) struct HyV4Attention {
    /// The DeepSeek-V3.2 MLA pieces. `mla.indexer` is `Some` on a `full`
    /// layer (unless `MLXCEL_DSA_DENSE` drops it) and `None` on a `shared`
    /// one.
    pub(super) mla: MLAAttention,
    /// `linear_gate`; `None` only under the `no_gate` parity arm.
    pub(super) linear_gate: Option<UnifiedLinear>,
    /// `learnable_sink_param`, `[heads]` float32; `None` only under the
    /// `no_sinks` parity arm.
    pub(super) sinks: Option<UniquePtr<MlxArray>>,
    /// `indexer_types[layer] == "shared"`.
    pub(super) shares_indexer: bool,
    /// RoPE pairing of the rope half: interleaved (`true`, upstream mlx-vlm)
    /// or half-split (`false`, the `rope_half_split` arm).
    pub(super) rope_traditional: bool,
}

fn cast_to(array: UniquePtr<MlxArray>, dtype: i32) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(&array) == dtype {
        array
    } else {
        mlxcel_core::astype(&array, dtype)
    }
}

/// `softmax([scale * q . k + mask, sinks])` with the sink column dropped,
/// times `v`: MLX's SDPA with one extra logit per head, or plain SDPA when
/// `sinks` is `None`. `k` and `v` may carry one head for `q`'s `H` (the
/// absorbed decode, where the latent is both).
fn sink_sdpa(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    sinks: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let Some(sinks) = sinks else {
        return mlxcel_core::layers::attention(q, k, v, scale, mask, 0.0, 0);
    };
    let sinks = mlxcel_core::astype(sinks, mlxcel_core::array_dtype(q));
    let mask_ptr = mask.map_or(std::ptr::null(), |mask| mask as *const MlxArray);
    // SAFETY: `mask_ptr` is null or borrows `mask` for the call, and `sinks`
    // is a live local array.
    unsafe {
        mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
            q,
            k,
            v,
            scale,
            mask_ptr,
            &*sinks as *const MlxArray,
        )
    }
}

impl HyV4Attention {
    /// Attend over `x` (`[B, L, hidden]`, already normed).
    ///
    /// `topk` carries the DSA selection between layers of one forward: a
    /// `full` layer overwrites it with its own selection (`None` when
    /// `kv_len <= index_topk`), a `shared` layer reads it.
    pub(super) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        topk: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let m = &self.mla;
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let gate = self.linear_gate.as_ref().map(|linear| {
            let gate = linear.forward(x);
            let gate = mlxcel_core::reshape(&gate, &[b, l, m.num_heads, m.v_head_dim]);
            let gate = mlxcel_core::sigmoid(&gate);
            mlxcel_core::transpose_axes(&gate, &[0, 2, 1, 3])
        });

        // `qr` (post-layernorm, pre-q_b_proj) is what the indexer also reads.
        let qr = m.q_a_layernorm.forward(&m.q_a_proj.forward(x));
        let q = m.q_b_proj.forward(&qr);
        let q = mlxcel_core::reshape(&q, &[b, l, m.num_heads, m.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q_nope = slice_axis(&q, -1, 0, m.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, m.qk_nope_head_dim, -1);

        let compressed_kv = m.kv_a_proj_with_mqa.forward(x);
        let compressed = slice_axis(&compressed_kv, -1, 0, m.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, m.kv_lora_rank, -1);
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, m.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);
        let kv_latent = m.kv_a_layernorm.forward(&compressed);

        // `offset` is the monotonic rope position; `live_before` the rows
        // already cached, which the causal fallback below needs (#619).
        let offset = cache.offset;
        let live_before = cache.live_len();
        // The shared DeepSeek-V3.2 RoPE (`DsaRope`), resolved from this
        // model's own `rope_parameters`: plain RoPE for the shipped
        // `rope_type: default`, never V3.2's YaRN table (#3279).
        let rope = |t: &MlxArray| m.rope.apply(t, self.rope_traditional, offset);
        let q_pe = rope(&q_pe);
        let k_pe = rope(&k_pe);

        let indexer_new = m.indexer.as_ref().map(|indexer| {
            (
                indexer.keys(x, offset),
                indexer.queries(&qr, offset),
                indexer.weights(x),
            )
        });

        // A full layer caches `[kv_latent | indexer key]`, a shared layer the
        // latent alone.
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);
        let cache_keys = match &indexer_new {
            Some((index_key, _, _)) => mlxcel_core::concatenate(&kv_latent, index_key, -1),
            None => kv_latent,
        };
        let window = store_latent_step(cache, m.geometry(), cache_keys, k_pe, "hy_v4", true);

        // Both generation paths pass no mask for a prefill, and neither the
        // indexer nor a sparse mask is causal by itself (#619).
        let causal_storage;
        let effective_mask: Option<&MlxArray> = if mask.is_some() {
            mask
        } else if l > 1 {
            causal_storage = mlxcel_core::utils::create_causal_mask(l, live_before);
            Some(&*causal_storage)
        } else {
            None
        };

        let dtype = mlxcel_core::array_dtype(&q_nope);
        let index_query = indexer_new.as_ref().map(|(_, q, w)| (&**q, &**w));
        let (kv_latent, k_pe, gathered) =
            self.select(window, index_query, effective_mask, l, dtype, topk);

        mlxcel_core::mla::stats::record(if l == 1 {
            MlaDecodePath::AbsorbedComposed
        } else {
            MlaDecodePath::Decompressed
        });
        let output = if l == 1 {
            // Gathered rows are all causally valid and the caller's mask is
            // indexed by ungathered positions, so a sparse step is maskless.
            let step_mask = if gathered { None } else { mask };
            self.decode_step(&q_nope, &q_pe, &kv_latent, &k_pe, step_mask)
        } else {
            self.prefill_step(
                &q_nope,
                &q_pe,
                &kv_latent,
                &k_pe,
                mask,
                effective_mask,
                topk.as_deref(),
            )
        };

        let output = match &gate {
            Some(gate) => mlxcel_core::multiply(&output, gate),
            None => output,
        };
        m.project_attention_output(&output, b, l)
    }

    /// Run this layer's key selection and return the dense `(kv_latent,
    /// k_pe)` the attention reads, plus whether the rows were gathered.
    ///
    /// A `full` layer scores the indexer-key columns of `window` and stores
    /// its selection in `topk`; a `shared` layer leaves `topk` as the last
    /// `full` layer set it. A single query gathers the selected rows (from
    /// the packed triple when the window is quantized); a multi-token step
    /// keeps every row and masks instead.
    fn select(
        &self,
        window: LatentWindow,
        index_query: Option<(&MlxArray, &MlxArray)>,
        mask: Option<&MlxArray>,
        l: i32,
        dtype: i32,
        topk: &mut Option<UniquePtr<MlxArray>>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>, bool) {
        let m = &self.mla;
        let rank = m.kv_lora_rank;
        if !self.shares_indexer {
            *topk = match (&m.indexer, index_query) {
                (Some(indexer), Some((query, weights))) => {
                    let keys = window.dense_key_columns(rank, window.key_width());
                    indexer.top_indices(query, &keys, weights, mask)
                }
                _ => None,
            };
        }
        let latent_window = window.narrow_keys(0, rank);
        match topk.as_deref() {
            Some(indices) if l == 1 => {
                if !self.shares_indexer {
                    latent::trace_topk(m.layer, || latent_window.seq_len(), indices);
                }
                let (ckv, kpe) = dense_latent_in(latent_window.gather_rows(indices), dtype);
                (ckv, kpe, true)
            }
            _ => {
                let (ckv, kpe) = dense_latent_in(latent_window, dtype);
                (ckv, kpe, false)
            }
        }
    }

    /// SDPA with this layer's sinks, or plain SDPA under the `no_sinks` arm.
    fn sdpa(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        sink_sdpa(q, k, v, self.mla.scale, mask, self.sinks.as_deref())
    }

    /// One absorbed query: `q_nope` moves into latent space, the latent is K
    /// and V, and the rope scores are the additive mask. Returns
    /// `[B, H, 1, v_head_dim]`.
    fn decode_step(
        &self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        kv_latent: &MlxArray,
        k_pe: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let m = &self.mla;
        let dtype = mlxcel_core::array_dtype(q_pe);
        let scale = mlxcel_core::full_f32(&[1], m.scale, dtype);
        let q_pe_scaled = mlxcel_core::multiply(q_pe, &scale);
        let k_pe_t = mlxcel_core::transpose_axes(k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);
        let pe_scores = match mask {
            Some(mask) => cast_to(mlxcel_core::add(&pe_scores, mask), dtype),
            None => pe_scores,
        };
        let q_latent = m.embed_q.forward(q_nope);
        let output = self.sdpa(&q_latent, kv_latent, kv_latent, Some(&pe_scores));
        m.unembed_out.forward(&output)
    }

    /// A multi-token step: per-head K and V are materialized from the latent
    /// and the rope half is folded into the head dimension, so the mask that
    /// remains is the `[.., L, S]` causal (or caller) mask with the top-k
    /// selection folded in. Returns `[B, H, L, v_head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_step(
        &self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        kv_latent: &MlxArray,
        k_pe: &MlxArray,
        caller_mask: Option<&MlxArray>,
        effective_mask: Option<&MlxArray>,
        indices: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let m = &self.mla;
        let k = m.embed_q.forward_no_transpose(kv_latent);
        let v = m.unembed_out.forward(kv_latent);
        let kv_len = mlxcel_core::array_shape(kv_latent)[2];

        let sparse_storage;
        let folded = match (indices, caller_mask) {
            (Some(indices), _) => {
                let causal = effective_mask.expect("a multi-token step always has a mask");
                sparse_storage = apply_sparse_prefill_mask(causal, indices, kv_len);
                FoldedMask::Array(&sparse_storage)
            }
            (None, Some(mask)) => FoldedMask::Array(mask),
            (None, None) => FoldedMask::Causal,
        };

        let Some(sinks) = &self.sinks else {
            return mlxcel_core::mla::rope_folded_attention(
                q_nope, q_pe, &k, k_pe, &v, m.scale, folded,
            );
        };
        let dtype = mlxcel_core::array_dtype(q_nope);
        let queries = mlxcel_core::concatenate(q_nope, q_pe, -1);
        let k_pe = mlxcel_core::utils::repeat_kv(k_pe, m.num_heads);
        let keys = mlxcel_core::concatenate(&k, &k_pe, -1);
        match folded {
            FoldedMask::Causal => {
                let sinks = mlxcel_core::astype(sinks, dtype);
                mlxcel_core::causal_prefill_attention(&queries, &keys, &v, m.scale, 0, Some(&sinks))
            }
            FoldedMask::Array(mask) => {
                let mask = mlxcel_core::astype(mask, dtype);
                self.sdpa(&queries, &keys, &v, Some(&mask))
            }
            FoldedMask::None => self.sdpa(&queries, &keys, &v, None),
        }
    }
}

#[cfg(test)]
#[path = "hy_v4_attention_tests.rs"]
mod tests;
