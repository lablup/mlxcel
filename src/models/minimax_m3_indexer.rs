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

//! MiniMax-M3 block-sparse attention ("MSA") lightning indexer.
//!
//! Structurally this mirrors the DeepSeek DSA indexer
//! (`deepseek_v32_indexer.rs`): separate index Q/K projections, per-head index
//! RMSNorm, RoPE on the index projections, and a running side cache for the
//! index keys that rides alongside the regular KV cache. The delta is the
//! selection granularity: DSA selects the top-`index_topk` individual key
//! *positions*, MSA scores fixed-size key *blocks* (`sparse_block_size`), then
//! selects the top-`sparse_topk_blocks` blocks and expands the block choice to a
//! per-token additive mask that is added to the dense causal mask.
//!
//! Degeneration invariant: when `sparse_topk_blocks` covers every block of the
//! current cache, every block is kept, the additive block mask is all zeros, and
//! block-sparse attention reduces exactly to dense causal attention. This is the
//! property the `minimax_m3` unit tests pin.
//!
//! The real 427B checkpoint cannot be loaded on the development machine, so the
//! index projection/norm weight *shapes* here follow the config
//! (`sparse_num_index_heads`, `sparse_index_dim`) but are exercised only through
//! the synthetic reduced-config unit tests. `load` returns `Ok(None)` whenever
//! the checkpoint carries no index weights, preserving a dense fallback rather
//! than failing to load.

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{ModelArgs, SparseAttentionConfig, get_weight_copy};
use crate::models::gemma::GemmaRMSNorm;
use mlxcel_core::layers::UnifiedLinear;

/// Environment variable that forces the dense full-attention fallback,
/// disabling the block-sparse indexer entirely for A/B comparison. When set
/// (to any value) the indexer is not loaded.
pub(super) const DENSE_FALLBACK_ENV: &str = "MLXCEL_MINIMAX_M3_DENSE";

/// The block-sparse "MSA" indexer for one attention block.
///
/// MQA-style, matching the real checkpoint: `index_q_proj` produces
/// `sparse_num_index_heads` query heads of `sparse_index_dim`, while
/// `index_k_proj` produces a SINGLE shared index-key stream of `sparse_index_dim`
/// (checkpoint shapes: `index_q_proj [512, 6144]`, `index_k_proj [128, 6144]`
/// with 4 heads x 128). The query heads all score against the one key stream.
pub(super) struct BlockSparseIndexer {
    index_q_proj: UnifiedLinear,
    index_k_proj: UnifiedLinear,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
    /// Number of index query heads (`sparse_num_index_heads`). The key is a
    /// single shared head (MQA).
    n_query_heads: i32,
    /// Index vector dimension (`sparse_index_dim`).
    index_dim: i32,
    /// Partial-RoPE dimension applied to the index projections.
    rope_dims: i32,
    rope_base: f32,
    softmax_scale: f32,
    block_size: i32,
    topk_blocks: i32,
    init_blocks: i32,
    local_blocks: i32,
}

impl BlockSparseIndexer {
    /// Load the indexer for `{attn_prefix}.{index_q_proj,index_k_proj,...}`.
    /// Returns `Ok(None)` when the checkpoint carries no index weights (dense
    /// fallback preserved), when the sparse config is absent, or when
    /// [`DENSE_FALLBACK_ENV`] is set. Returns `Err` when `sparse` is
    /// degenerate (see [`SparseAttentionConfig::validate`]).
    ///
    /// The single-head index key rides on the regular K buffer (head-axis
    /// concat), which requires `sparse_index_dim == head_dim`; when they differ
    /// the indexer is disabled (dense fallback) rather than silently mis-caching.
    pub(super) fn load(
        weights: &WeightMap,
        args: &ModelArgs,
        sparse: &SparseAttentionConfig,
        attn_prefix: &str,
    ) -> Result<Option<Self>, String> {
        if std::env::var_os(DENSE_FALLBACK_ENV).is_some() {
            return Ok(None);
        }
        let q_key = format!("{}.index_q_proj.weight", attn_prefix);
        if !weights.contains_key(&q_key) {
            return Ok(None);
        }
        // Reject a degenerate `sparse_block_size`/`sparse_topk_blocks` here,
        // once, instead of guarding the masking hot path (`should_apply_sparse`,
        // `build_block_drop_mask`) on every call.
        sparse.validate()?;

        let n_query_heads = sparse.sparse_num_index_heads as i32;
        let index_dim = sparse.sparse_index_dim as i32;
        let head_dim = args.head_dim as i32;
        if index_dim != head_dim {
            // The single-head index key is cached on the regular K buffer via a
            // head-axis concat, which needs a matching last-axis width. Fall back
            // to dense otherwise instead of mis-caching.
            return Ok(None);
        }

        // Defensive shape checks: catch an MHA-vs-MQA layout mismatch at load
        // time instead of corrupting the forward pass. index_q_proj is
        // `[n_query_heads * index_dim, hidden]`; index_k_proj is a single shared
        // head `[index_dim, hidden]`.
        check_out_dim(
            weights,
            &format!("{}.index_q_proj.weight", attn_prefix),
            n_query_heads * index_dim,
        )?;
        check_out_dim(
            weights,
            &format!("{}.index_k_proj.weight", attn_prefix),
            index_dim,
        )?;

        let group_size = args.group_size();
        let bits = args.bits();

        let index_q_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.index_q_proj", attn_prefix),
            group_size,
            bits,
        )?;
        let index_k_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.index_k_proj", attn_prefix),
            group_size,
            bits,
        )?;
        let q_norm = GemmaRMSNorm::new(
            get_weight_copy(weights, &format!("{}.index_q_norm.weight", attn_prefix))?,
            args.rms_norm_eps,
        );
        let k_norm = GemmaRMSNorm::new(
            get_weight_copy(weights, &format!("{}.index_k_norm.weight", attn_prefix))?,
            args.rms_norm_eps,
        );

        Ok(Some(Self {
            index_q_proj,
            index_k_proj,
            q_norm,
            k_norm,
            n_query_heads,
            index_dim,
            rope_dims: args.rotary_dim.min(sparse.sparse_index_dim) as i32,
            rope_base: args.rope_theta,
            softmax_scale: (index_dim as f32).powf(-0.5),
            block_size: sparse.sparse_block_size as i32,
            topk_blocks: sparse.sparse_topk_blocks as i32,
            init_blocks: sparse.sparse_init_block as i32,
            local_blocks: sparse.sparse_local_block as i32,
        }))
    }

    /// Single shared index key for the new tokens: `rope(norm(index_k_proj(x)))`.
    /// Shape `[b, 1, s, index_dim]`. This is what gets cached on the K side
    /// buffer (concatenated onto the regular K along the head axis).
    pub(super) fn keys(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, s) = (shape[0], shape[1]);
        let k = self.index_k_proj.forward(x);
        // Single head: [b, s, index_dim] -> norm over index_dim -> [b, 1, s, dim].
        let k = mlxcel_core::reshape(&k, &[b, s, 1, self.index_dim]);
        let k = self.k_norm.forward(&k);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset)
    }

    /// Per-head index query: `rope(norm(index_q_proj(x)))`.
    /// Shape `[b, n_query_heads, s, index_dim]`.
    pub(super) fn queries(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, s) = (shape[0], shape[1]);
        let q = self.index_q_proj.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, s, self.n_query_heads, self.index_dim]);
        let q = self.q_norm.forward(&q);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset)
    }

    /// Whether the block-sparse mask should actually be applied at the current
    /// cache length, or attention can stay dense. Decode/prefill stay dense
    /// while the selected window (`topk_blocks * block_size`) still covers at
    /// least half the live cache; the sparse mask only kicks in beyond that.
    pub(super) fn should_apply_sparse(&self, kv_len: i32) -> bool {
        kv_len > 2 * self.topk_blocks * self.block_size
    }

    /// Token-level index scores `[b, 1, s, kv_len]`, mean-combined over the
    /// index heads, with the additive causal `mask` folded in so future
    /// positions score `-inf` and never enter a selected block.
    pub(super) fn token_scores(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // q: [b, n_query_heads, s, index_dim]; k: [b, 1, kv_len, index_dim].
        // scores = (q @ k^T) * scale -> [b, n_query_heads, s, kv_len], the single
        // key head broadcasting across the query heads.
        let k_t = mlxcel_core::transpose_axes(k, &[0, 1, 3, 2]);
        let scores = mlxcel_core::matmul(q, &k_t);
        let scale =
            mlxcel_core::full_f32(&[1], self.softmax_scale, mlxcel_core::array_dtype(&scores));
        let scores = mlxcel_core::multiply(&scores, &scale);
        // Mean over query heads -> [b, 1, s, kv_len].
        let scores = mlxcel_core::mean_axis(&scores, 1, true);
        match mask {
            Some(m) => mlxcel_core::add(&scores, m),
            None => scores,
        }
    }

    /// Build the additive block-sparse mask from causally-masked token scores.
    /// See [`build_block_drop_mask`].
    pub(super) fn block_drop_mask(
        &self,
        token_scores: &MlxArray,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        build_block_drop_mask(
            token_scores,
            offset,
            self.block_size,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
        )
    }
}

/// Assert that a weight's output dimension (first axis) matches `expected`.
/// Returns a clear error on mismatch so a wrong tensor layout fails at load
/// instead of corrupting the forward pass. A quantized weight still carries its
/// true output dim on axis 0 (packing is on the input axis), so this holds for
/// both quantized and unquantized tensors.
pub(super) fn check_out_dim(weights: &WeightMap, name: &str, expected: i32) -> Result<(), String> {
    if let Some(w) = weights.get(name) {
        let shape = mlxcel_core::array_shape(w);
        if let Some(&out) = shape.first()
            && out != expected
        {
            return Err(format!(
                "minimax_m3: {} has output dim {}, expected {} (tensor layout mismatch)",
                name, out, expected
            ));
        }
    }
    Ok(())
}

/// Build the additive per-token block-sparse mask.
///
/// `token_scores` is `[b, 1, s, kv_len]`, already causally masked (future
/// positions `-inf`). Keys are grouped into `[num_blocks]` contiguous blocks of
/// `block_size` (the tail block is `-inf`-padded); each block is scored by the
/// max token score inside it (`sparse_score_type: "max"`). The first
/// `init_blocks` blocks and, per query row, the trailing `local_blocks` blocks
/// up to and including the query's own block are force-kept; the remaining
/// budget selects the top `topk_blocks` blocks by score. The result is
/// `[b, 1, s, kv_len]` with `0.0` for kept positions and `-inf` for dropped
/// ones, ready to be added to the dense causal mask.
///
/// When `topk_blocks` covers every block the function short-circuits to an
/// all-zero mask, so the caller's masked attention is bit-identical to dense.
pub(super) fn build_block_drop_mask(
    token_scores: &MlxArray,
    offset: i32,
    block_size: i32,
    topk_blocks: i32,
    init_blocks: i32,
    local_blocks: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(token_scores);
    let (b, s, kv_len) = (shape[0], shape[2], shape[3]);
    // Ceiling division; `kv_len` and `block_size` are positive. (`i32::div_ceil`
    // is still unstable, so compute it directly.)
    let num_blocks = (kv_len + block_size - 1) / block_size;
    let padded = num_blocks * block_size;

    // Every block fits in the budget: nothing is dropped, so the additive mask
    // is exactly zero and attention degenerates to dense.
    if topk_blocks >= num_blocks {
        return mlxcel_core::zeros(&[b, 1, s, kv_len], mlxcel_core::dtype::FLOAT32);
    }

    // Pad the key axis up to a whole number of blocks with -inf, then reduce
    // each block to its max token score: [b, 1, s, num_blocks].
    let scores = mlxcel_core::astype(token_scores, mlxcel_core::dtype::FLOAT32);
    let scores = if padded > kv_len {
        mlxcel_core::pad(
            &scores,
            &[0, 0, 0, 0, 0, 0, 0, padded - kv_len],
            f32::NEG_INFINITY,
        )
    } else {
        scores
    };
    let blocked = mlxcel_core::reshape(&scores, &[b, 1, s, num_blocks, block_size]);
    let block_scores = mlxcel_core::max_axis(&blocked, -1, false);

    // Force-keep the initial and per-query local blocks by lifting their score
    // to +inf before the top-k. block_idx: [1, 1, 1, num_blocks].
    let block_idx = mlxcel_core::reshape(
        &mlxcel_core::astype(
            &mlxcel_core::arange_i32(0, num_blocks, 1),
            mlxcel_core::dtype::FLOAT32,
        ),
        &[1, 1, 1, num_blocks],
    );
    // query_block: [1, 1, s, 1] = floor((row + offset) / block_size).
    let rows = mlxcel_core::astype(
        &mlxcel_core::arange_i32(offset, offset + s, 1),
        mlxcel_core::dtype::FLOAT32,
    );
    let bs_scalar = mlxcel_core::full_f32(&[1], block_size as f32, mlxcel_core::dtype::FLOAT32);
    let query_block = mlxcel_core::astype(
        &mlxcel_core::reshape(&mlxcel_core::divide(&rows, &bs_scalar), &[1, 1, s, 1]),
        mlxcel_core::dtype::INT32,
    );
    let query_block = mlxcel_core::astype(&query_block, mlxcel_core::dtype::FLOAT32);

    // init_keep: block_idx < init_blocks.
    let init_scalar = mlxcel_core::full_f32(&[1], init_blocks as f32, mlxcel_core::dtype::FLOAT32);
    let init_keep = mlxcel_core::less(&block_idx, &init_scalar);
    // local_keep: (query_block - local_blocks) < block_idx <= query_block.
    let local_scalar =
        mlxcel_core::full_f32(&[1], local_blocks as f32, mlxcel_core::dtype::FLOAT32);
    let lower = mlxcel_core::subtract(&query_block, &local_scalar);
    let above_lower = mlxcel_core::greater(&block_idx, &lower);
    let below_upper = mlxcel_core::less_equal(&block_idx, &query_block);
    let local_keep = mlxcel_core::logical_and(&above_lower, &below_upper);
    let forced = mlxcel_core::logical_or(&init_keep, &local_keep);

    let pos_inf = mlxcel_core::full_f32(&[1], f32::INFINITY, mlxcel_core::dtype::FLOAT32);
    let block_scores = mlxcel_core::where_cond(&forced, &pos_inf, &block_scores);

    // Top-`topk_blocks` blocks by score: negate, argpartition, take the front.
    let neg = mlxcel_core::negative(&block_scores);
    let part = mlxcel_core::argpartition(&neg, topk_blocks - 1, -1);
    let topk_idx = slice_axis(&part, -1, 0, topk_blocks);

    // Scatter 1.0 into the kept block columns, threshold to a bool keep-mask.
    let base = mlxcel_core::zeros(&[b, 1, s, num_blocks], mlxcel_core::dtype::FLOAT32);
    let ones = mlxcel_core::ones(&[b, 1, s, topk_blocks], mlxcel_core::dtype::FLOAT32);
    let filled = mlxcel_core::put_along_axis(&base, &topk_idx, &ones, -1);
    let half = mlxcel_core::full_f32(&[1], 0.5, mlxcel_core::dtype::FLOAT32);
    let keep = mlxcel_core::greater(&filled, &half);

    // Additive block mask (0 keep / -inf drop), then expand each block to
    // `block_size` token columns and slice back to `kv_len`.
    let zero = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::dtype::FLOAT32);
    let block_mask = mlxcel_core::where_cond(&keep, &zero, &neg_inf);
    let block_mask = mlxcel_core::expand_dims(&block_mask, -1); // [b,1,s,num_blocks,1]
    let token_mask = mlxcel_core::broadcast_to(&block_mask, &[b, 1, s, num_blocks, block_size]);
    let token_mask = mlxcel_core::reshape(&token_mask, &[b, 1, s, padded]);
    slice_axis(&token_mask, -1, 0, kv_len)
}
