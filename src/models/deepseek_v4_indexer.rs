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

//! DeepSeek-V4 HiSA indexer (`Indexer` in `language.py`, batched selection in
//! `hisa_kernel.py`; paper <https://arxiv.org/abs/2603.28458>).
//!
//! The indexer owns its OWN `Compressor` over the same token stream (same
//! ratio, `index_head_dim`-wide pooled rows), scores pooled positions with a
//! ReLU'd, per-head-weighted dot product against `wq_b(q_residual)`, and
//! returns the top-`index_topk` pooled indices for the sparse attention path.
//!
//! Three selection paths, all required, mirroring the reference exactly:
//!
//! * **Decode fast path** (`L == 1`, no pool mask to honour, and
//!   `Np >= index_block * index_keep`): coarse-score `index_block`-sized
//!   block means, keep the best `index_keep` blocks, fine-score only their
//!   members, then top-k.
//! * **Batched path** (`L > 1`, same size gate): the same two-stage
//!   hierarchy, honouring causality through a per-query `valid_len` count
//!   (the pool mask is contiguous-causal, so a count IS the mask), tiled over
//!   `L` with `fine_chunk` so the candidate tensor stays bounded.
//! * **Flat fallback**: score every pooled position, mask, `argpartition`.
//!   The hierarchical paths must agree with this one on identical inputs;
//!   the unit tests assert it.
//!
//! `index_block` (64) and `index_keep` (16) are NOT in the real checkpoint's
//! config.json; they come from the reference dataclass defaults and the
//! `ModelArgs` defaults preserve them.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::compress::{Compressor, PoolingCache, pool_visible_counts};
use super::rope::V4Rope;
use super::{ModelArgs, masked_fill_min};

/// Tile size over `L` for the batched fine stage (reference default).
const FINE_CHUNK: i32 = 512;

pub(crate) struct Indexer {
    wq_b: UnifiedLinear,
    weights_proj: UnifiedLinear,
    pub(crate) compressor: Compressor,
    n_heads: i32,
    head_dim: i32,
    pub(crate) index_topk: i32,
    index_block: i32,
    index_keep: i32,
    scale: f32,
}

impl Indexer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        compress_ratio: i32,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let head_dim = args.index_head_dim as i32;
        Ok(Self {
            wq_b: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wq_b"),
                group_size,
                bits,
            )?,
            weights_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.weights_proj"),
                group_size,
                bits,
            )?,
            compressor: Compressor::from_weights(
                weights,
                args,
                &format!("{prefix}.compressor"),
                compress_ratio,
                head_dim,
            )?,
            n_heads: args.index_n_heads as i32,
            head_dim,
            index_topk: args.index_topk as i32,
            index_block: args.index_block as i32,
            index_keep: args.index_keep as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `weights_proj(x)` in f32, scaled by `n_heads^-0.5`: `[B, L, H]`.
    fn head_weights(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let w = self.weights_proj.forward(x);
        let w = mlxcel_core::astype(&w, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::multiply_scalar(&w, (self.n_heads as f32).powf(-0.5))
    }

    /// Select top-`min(index_topk, Np)` pooled indices for every query.
    ///
    /// Returns `None` when no pooled rows exist yet. `rope` is the OWNING
    /// attention's positional rope (`position_rope` in the reference), used
    /// on the indexer queries at `index_head_dim`.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        q_residual: &MlxArray,
        rope: &V4Rope,
        pool: &mut PoolingCache,
        offset: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        // Borrowed straight out of the cache: `update_and_fetch` hands back
        // the live pooled buffer rather than a copy of it, and the borrow
        // ends at the `astype` below, before any selection path runs.
        let pooled = self.compressor.forward(x, pool, offset);
        let np = mlxcel_core::array_shape(pooled)[1];
        if np == 0 {
            return None;
        }

        let q = self.wq_b.forward(q_residual);
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q = rope.apply(&q, offset, false);
        let q = mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32);
        let pooled = mlxcel_core::astype(pooled, mlxcel_core::dtype::FLOAT32);

        let k = self.index_topk.min(np);
        let ratio = self.compressor.ratio;
        let w = self.head_weights(x);
        // `index_block * index_keep` in i64. Both sides of this gate use the
        // product, and an i32 multiply that wraps in release can let the gate
        // pass while `nb = np / index_block` is 0, which drives `kb` to 0 and
        // reaches `topk_indices(.., 0)` -> `argpartition(kth = -1)`, an MLX
        // throw and therefore an uncatchable abort. `ModelArgs::validate` now
        // caps both scalars at `i32::MAX`, which makes the wrap unreachable
        // from config, but this gate is the load-bearing arithmetic and is
        // kept safe on its own.
        let block_span = i64::from(self.index_block) * i64::from(self.index_keep);
        let hierarchical =
            self.index_block > 0 && i64::from(np) >= block_span && block_span >= i64::from(k);

        // Decode fast path: no pool mask exists for L == 1.
        if l == 1 && hierarchical {
            return Some(self.hisa_select_decode(&q, &pooled, &w, k));
        }

        if l > 1 && hierarchical {
            let counts = pool_visible_counts(l, offset, ratio, np);
            return Some(self.hisa_select_batched(&q, &pooled, &w, k, &counts));
        }

        let counts = (l > 1).then(|| pool_visible_counts(l, offset, ratio, np));
        Some(self.flat_select(&q, &pooled, &w, k, counts.as_deref()))
    }

    /// Flat fallback: score all `Np` pooled positions, mask, `argpartition`.
    /// `q` is `[B, H, L, D]` f32, `pooled` `[B, Np, D]` f32, `w` the f32
    /// per-head weights `[B, L, H]` (already scaled by `n_heads^-0.5`).
    pub(crate) fn flat_select(
        &self,
        q: &MlxArray,
        pooled: &MlxArray,
        w: &MlxArray,
        k: i32,
        valid_counts: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        let l = mlxcel_core::array_shape(q)[2];
        let np = mlxcel_core::array_shape(pooled)[1];
        let pooled_b = mlxcel_core::expand_dims(pooled, 1);
        let pooled_t = mlxcel_core::transpose_axes(&pooled_b, &[0, 1, 3, 2]);
        let scores = mlxcel_core::matmul(q, &pooled_t);
        let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let scores = mlxcel_core::maximum(&scores, &zero);
        let scores = mlxcel_core::multiply_scalar(&scores, self.scale);
        let w_t = mlxcel_core::transpose_axes(w, &[0, 2, 1]);
        let w_t = mlxcel_core::expand_dims(&w_t, -1);
        let scores = mlxcel_core::multiply(&scores, &w_t);
        let scores = mlxcel_core::sum_axis(&scores, 1, false);
        let scores = if let Some(counts) = valid_counts {
            let counts = mlxcel_core::from_slice_i32(counts, &[1, l, 1]);
            let idx = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, np, 1), &[1, 1, np]);
            let visible = mlxcel_core::less(&idx, &counts);
            masked_fill_min(&scores, &visible)
        } else {
            scores
        };
        topk_indices(&scores, k)
    }

    /// Decode-time hierarchical selection (`Indexer._hisa_select`).
    /// `q` is `[B, H, 1, D]` f32, `pooled` `[B, Np, D]` f32, `w` the f32
    /// per-head weights `[B, 1, H]`; returns `[B, 1, k]` int32 indices into
    /// the pooled prefix.
    pub(crate) fn hisa_select_decode(
        &self,
        q: &MlxArray,
        pooled: &MlxArray,
        w: &MlxArray,
        k: i32,
    ) -> UniquePtr<MlxArray> {
        let pshape = mlxcel_core::array_shape(pooled);
        let (b, np, hd) = (pshape[0], pshape[1], pshape[2]);
        let blk = self.index_block;
        let nb = np / blk;
        let usable = nb * blk;

        // (B, 1, H) -> (B, H, 1, 1)
        let wq = mlxcel_core::transpose_axes(w, &[0, 2, 1]);
        let wq = mlxcel_core::expand_dims(&wq, -1);

        // Coarse: block-mean representatives.
        let rep = mlxcel_core::utils::slice_axis(pooled, 1, 0, usable);
        let rep = mlxcel_core::reshape(&rep, &[b, nb, blk, hd]);
        let rep = mlxcel_core::mean_axis(&rep, 2, false);
        let rep_b = mlxcel_core::expand_dims(&rep, 1);
        let rep_t = mlxcel_core::transpose_axes(&rep_b, &[0, 1, 3, 2]);
        let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let cs = mlxcel_core::maximum(&mlxcel_core::matmul(q, &rep_t), &zero);
        let cs = mlxcel_core::multiply_scalar(&cs, self.scale);
        let cs = mlxcel_core::multiply(&cs, &wq);
        let cscore = mlxcel_core::sum_axis(&cs, 1, false); // (B, 1, nb)

        let kb = self.index_keep.min(nb);
        let top_blk = topk_indices(&cscore, kb); // (B, 1, Kb) int32

        // Fine: score only positions inside the retained blocks.
        let pos = block_member_positions(&top_blk, blk); // (B, 1, Kb*blk)
        let c = kb * blk;
        let idx = mlxcel_core::reshape(&pos, &[b, c]);
        let idx = mlxcel_core::expand_dims(&idx, -1);
        let idx = mlxcel_core::broadcast_to(&idx, &[b, c, hd]);
        let cand = mlxcel_core::take_along_axis(pooled, &idx, 1); // (B, C, hd)
        let cand_b = mlxcel_core::expand_dims(&cand, 1);
        let cand_t = mlxcel_core::transpose_axes(&cand_b, &[0, 1, 3, 2]);
        let fs = mlxcel_core::maximum(&mlxcel_core::matmul(q, &cand_t), &zero);
        let fs = mlxcel_core::multiply_scalar(&fs, self.scale);
        let fs = mlxcel_core::multiply(&fs, &wq);
        let fscore = mlxcel_core::sum_axis(&fs, 1, false); // (B, 1, C)

        let sel = topk_indices(&fscore, k);
        mlxcel_core::take_along_axis(&pos, &sel, -1)
    }

    /// Batched hierarchical selection (`hisa_kernel.hisa_select`): honours
    /// causality through `valid_len` and tiles the fine stage over `L`.
    /// `q` is `[B, H, L, D]` f32, `w` the f32 per-head weights `[B, L, H]`;
    /// returns `[B, L, k]` int32.
    pub(crate) fn hisa_select_batched(
        &self,
        q: &MlxArray,
        pooled: &MlxArray,
        w: &MlxArray,
        k: i32,
        valid_counts: &[i32],
    ) -> UniquePtr<MlxArray> {
        let pshape = mlxcel_core::array_shape(pooled);
        let (b, np, hd) = (pshape[0], pshape[1], pshape[2]);
        let l = mlxcel_core::array_shape(q)[2];
        let blk = self.index_block;
        let nb = np / blk;
        let usable = nb * blk;
        let neg_big = mlxcel_core::full_f32(&[1], -1e30, mlxcel_core::dtype::FLOAT32);

        // wk = weights * scale (per-head multiplier, scale folded in).
        let wk = mlxcel_core::multiply_scalar(w, self.scale); // (B, L, H)
        let wk_h = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
        let wk_h = mlxcel_core::expand_dims(&wk_h, -1); // (B, H, L, 1)

        let valid = mlxcel_core::from_slice_i32(valid_counts, &[1, l, 1]); // broadcast over B

        // Coarse stage.
        let rep = mlxcel_core::utils::slice_axis(pooled, 1, 0, usable);
        let rep = mlxcel_core::reshape(&rep, &[b, nb, blk, hd]);
        let rep = mlxcel_core::mean_axis(&rep, 2, false);
        let rep_b = mlxcel_core::expand_dims(&rep, 1);
        let rep_t = mlxcel_core::transpose_axes(&rep_b, &[0, 1, 3, 2]);
        let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
        let cs = mlxcel_core::maximum(&mlxcel_core::matmul(q, &rep_t), &zero); // (B,H,L,nb)
        let cs = mlxcel_core::multiply(&cs, &wk_h);
        let cscore = mlxcel_core::sum_axis(&cs, 1, false); // (B, L, nb)
        // Mask blocks whose START is not yet visible.
        let block_start = mlxcel_core::multiply(
            &mlxcel_core::arange_i32(0, nb, 1),
            &mlxcel_core::from_slice_i32(&[blk], &[1]),
        );
        let block_start = mlxcel_core::reshape(&block_start, &[1, 1, nb]);
        let block_visible = mlxcel_core::less(&block_start, &valid);
        let cscore = mlxcel_core::where_cond(&block_visible, &cscore, &neg_big);

        let kb = self.index_keep.min(nb);
        let top_blk = topk_indices(&cscore, kb); // (B, L, Kb) int32

        let c = kb * blk;
        let chunk = FINE_CHUNK.min(l);
        let mut parts: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut s = 0;
        while s < l {
            let e = (s + chunk).min(l);
            let rows = e - s;
            let blk_c = mlxcel_core::utils::slice_axis(&top_blk, 1, s, e); // (B, rows, Kb)
            let pos_c = block_member_positions(&blk_c, blk); // (B, rows, C)
            let idx = mlxcel_core::reshape(&pos_c, &[b, rows * c]);
            let idx = mlxcel_core::expand_dims(&idx, -1);
            let idx = mlxcel_core::broadcast_to(&idx, &[b, rows * c, hd]);
            let cand = mlxcel_core::take_along_axis(pooled, &idx, 1);
            let cand = mlxcel_core::reshape(&cand, &[b, rows, c, hd]);
            let qbl = mlxcel_core::utils::slice_axis(q, 2, s, e); // (B, H, rows, D)
            let qbl = mlxcel_core::transpose_axes(&qbl, &[0, 2, 1, 3]); // (B, rows, H, D)
            let cand_t = mlxcel_core::transpose_axes(&cand, &[0, 1, 3, 2]);
            let fs = mlxcel_core::maximum(&mlxcel_core::matmul(&qbl, &cand_t), &zero); // (B,rows,H,C)
            let wk_c = mlxcel_core::utils::slice_axis(&wk, 1, s, e); // (B, rows, H)
            let wk_c = mlxcel_core::expand_dims(&wk_c, -1);
            let fs = mlxcel_core::multiply(&fs, &wk_c);
            let oc = mlxcel_core::sum_axis(&fs, 2, false); // (B, rows, C)
            let valid_c = mlxcel_core::utils::slice_axis(&valid, 1, s, e); // (1, rows, 1)
            let vis = mlxcel_core::less(&pos_c, &valid_c);
            let oc = mlxcel_core::where_cond(&vis, &oc, &neg_big);
            if chunk < l {
                mlxcel_core::eval(&oc);
            }
            parts.push(oc);
            s = e;
        }
        let mut fscore = parts.remove(0);
        for part in &parts {
            fscore = mlxcel_core::concatenate(&fscore, part, 1);
        }

        let sel = topk_indices(&fscore, k); // (B, L, k)
        let pos = block_member_positions(&top_blk, blk); // (B, L, C)
        mlxcel_core::take_along_axis(&pos, &sel, -1)
    }
}

/// Unordered top-`k` indices along the last axis via `argpartition`, cast to
/// int32 so callers can do index arithmetic on them.
pub(crate) fn topk_indices(scores: &MlxArray, k: i32) -> UniquePtr<MlxArray> {
    let neg = mlxcel_core::negative(scores);
    let idx = mlxcel_core::argpartition(&neg, k - 1, -1);
    let idx = mlxcel_core::utils::slice_axis(&idx, -1, 0, k);
    mlxcel_core::astype(&idx, mlxcel_core::dtype::INT32)
}

/// Expand block indices `[..., Kb]` (int32) into the flat positions of every
/// block member: `[..., Kb * blk]` (`top_blk[..., None] * blk + arange(blk)`).
fn block_member_positions(top_blk: &MlxArray, blk: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(top_blk);
    let kb = *shape.last().expect("block indices need a last axis");
    let base = mlxcel_core::multiply(top_blk, &mlxcel_core::from_slice_i32(&[blk], &[1]));
    let base = mlxcel_core::expand_dims(&base, -1); // [..., Kb, 1]
    let member = mlxcel_core::arange_i32(0, blk, 1); // [blk]
    let pos = mlxcel_core::add(&base, &member); // [..., Kb, blk]
    let mut out_shape = shape;
    *out_shape.last_mut().expect("checked above") = kb * blk;
    mlxcel_core::reshape(&pos, &out_shape)
}
