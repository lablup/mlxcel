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

//! The DSA attention of one query, and a verify block run as its rows'
//! single queries (issue #3421).
//!
//! A decode step attends absorbed over the stored latent, after the
//! indexer picks its top-k rows from a one-row score. A block forms the
//! scores of all its rows in one product and attends through rope-folded
//! SDPA, so a token-only verify round (prompt lookup, a full-model drafter)
//! read different logits than plain decode. Inside the row-invariant scope
//! (`MLXCEL_MLA_ROW_INVARIANT=1`) a verify block runs every row through
//! [`MLAAttention::single_query_attention`] over the prefix of the stored
//! window that row's decode step would see.

use mlxcel_core::mla::LatentWindow;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{MLAAttention, gather_topk_kv, latent};

impl MLAAttention {
    /// One query `[B, H, 1, ..]` against a `[kv_latent | indexer key]`
    /// window, absorbed: the output `[B, H, 1, v_head_dim]` after
    /// `unembed_out`. `index_query` is the step's `(indexer query, indexer
    /// weights)`, absent when the layer has no indexer.
    ///
    /// A quantized window stays packed (`latent.rs`) and its attention
    /// records its own path. A dense window takes the indexer's top-k rows
    /// when it selects any and the composed absorbed decode.
    ///
    /// Used by: `MLAAttention::forward`, for a decode step and for every row
    /// of a verify block.
    pub(super) fn single_query_attention(
        &self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        window: LatentWindow,
        index_query: Option<(&MlxArray, &MlxArray)>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if window.is_quantized() {
            let q_projected = self.embed_q.forward(q_nope);
            let output = self.quantized_decode(window, &q_projected, q_pe, index_query, mask);
            return self.unembed_out.forward(&output);
        }
        let (cache_keys, k_pe) = window.into_dense();

        // Slice the cached indexer key back off the shared buffer.
        let (kv_latent, indexer_keys) = match &self.indexer {
            Some(_) => {
                let kvl = slice_axis(&cache_keys, -1, 0, self.kv_lora_rank);
                let ik = slice_axis(&cache_keys, -1, self.kv_lora_rank, -1);
                (kvl, Some(ik))
            }
            None => (cache_keys, None),
        };

        // Top-k key selection. `None` at short context (kv_len <= index_topk),
        // in which case the step attends over the whole window.
        let topk_indices = match (&self.indexer, index_query, &indexer_keys) {
            (Some(idx), Some((idx_q, idx_w)), Some(idx_k)) => {
                idx.top_indices(idx_q, idx_k, idx_w, mask)
            }
            _ => None,
        };
        if let Some(indices) = &topk_indices {
            latent::trace_topk(
                self.layer,
                || mlxcel_core::array_shape(&kv_latent)[2],
                indices,
            );
        }
        mlxcel_core::mla::stats::record(mlxcel_core::mla::MlaDecodePath::AbsorbedComposed);

        // Single-token decode keeps the rope term as an additive mask over
        // one `[B, H, 1, S]` score row.
        // pe_scores = (q_pe * scale) @ k_pe.T
        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(q_pe));
        let q_pe_scaled = mlxcel_core::multiply(q_pe, &scale_scalar);
        // Absorbed: project Q into latent space, use kv_latent as K=V
        let q_projected = self.embed_q.forward(q_nope);
        if let Some(indices) = &topk_indices {
            // Sparse decode fast path: gather the selected KV subset with
            // take_along_axis (no per-step sparse mask). Decode is always
            // called with `mask == None` here, and every gathered position is
            // causally valid, so pe_scores is computed maskless over the
            // gathered k_pe.
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
            // SAFETY: `pe_mask_ptr` points into `pe_scores_g`, which outlives
            // the call; the other operands are live borrows.
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
            return self.unembed_out.forward(&output);
        }
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);
        let pe_scores = match mask {
            Some(m) => mlxcel_core::add(&pe_scores, m),
            None => pe_scores,
        };
        let pe_mask_ptr = &*pe_scores as *const MlxArray;
        // SAFETY: `pe_mask_ptr` points into `pe_scores`, which outlives the
        // call; the other operands are live borrows.
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

    /// A verify block's attention `[1, H, L, v_head_dim]`: row `r` is the
    /// single query of the decode step that appended position
    /// `total - L + r`, over the first `total - L + r + 1` stored positions
    /// and with its own indexer selection. The block's window already holds
    /// every row's key, so each row reads a prefix of it. The rows count as
    /// one MLA call in the path stats.
    ///
    /// Used by: `MLAAttention::forward` inside the row-invariant scope.
    pub(super) fn verify_block_attention(
        &self,
        q_nope: &MlxArray,
        q_pe: &MlxArray,
        window: &LatentWindow,
        index_query: Option<(&MlxArray, &MlxArray)>,
    ) -> UniquePtr<MlxArray> {
        let l = mlxcel_core::array_shape(q_nope)[2];
        let total = window.seq_len();
        let rows: Vec<UniquePtr<MlxArray>> = mlxcel_core::mla::stats::as_one_call(|| {
            (0..l)
                .map(|r| {
                    // Indexer query `[B, heads, L, D]`, weights `[B, L, heads]`.
                    let row_query = index_query
                        .map(|(q, w)| (slice_axis(q, 2, r, r + 1), slice_axis(w, 1, r, r + 1)));
                    self.single_query_attention(
                        &slice_axis(q_nope, 2, r, r + 1),
                        &slice_axis(q_pe, 2, r, r + 1),
                        crate::models::mla_batched::latent_window_prefix(window, total - l + r + 1),
                        row_query.as_ref().map(|(q, w)| (&**q, &**w)),
                        None,
                    )
                })
                .collect()
        });
        let refs: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
        mlxcel_core::concatenate_many(&refs, 2)
    }
}
