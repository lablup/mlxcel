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

//! Common utility functions for mlxcel-core
//!
//! This module provides shared utility functions used across multiple models,
//! reducing code duplication and ensuring consistency.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::sync::OnceLock;

// Array Slicing Utilities.
/// Slice an array along a specified axis.
///
/// # Arguments
/// * `x` - Input array
/// * `axis` - Axis to slice along (supports negative indexing)
/// * `start` - Start index (supports negative indexing)
/// * `end` - End index. Use -1 to mean "to the end of axis" (Python slice semantics)
///
/// # Example
/// ```ignore
/// // Slice x[:, 0:10, :] along axis 1
/// let sliced = slice_axis(&x, 1, 0, 10);
///
/// // Slice x[:, 5:, :] along axis 1 (5 to end)
/// let sliced = slice_axis(&x, 1, 5, -1);
/// ```
pub fn slice_axis(x: &MlxArray, axis: i32, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let ndim = shape.len();

    // Handle negative axis
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;

    let dim_size = shape[axis];

    // Handle end index:
    // - end = -1 means "to the end of axis" (Python slice semantics)
    // - other negative values are relative to end
    let end = if end == -1 {
        dim_size
    } else if end < 0 {
        dim_size + end
    } else {
        end.min(dim_size)
    };

    // Handle negative start
    let start = if start < 0 {
        (dim_size + start).max(0)
    } else {
        start.min(dim_size)
    };

    // Build starts and stops vectors
    let mut starts = vec![0i32; ndim];
    let mut stops: Vec<i32> = shape.clone();
    starts[axis] = start;
    stops[axis] = end;

    ffi::slice(x, &starts, &stops)
}

// Attention Mask Utilities.
/// Create a causal attention mask.
/// Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4,
/// MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal attention callers
///
/// Creates a lower triangular mask of shape [size, size + offset] where:
/// - 1.0 indicates positions that can be attended to
/// - -inf indicates positions that should be masked
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (number of previously cached tokens)
///
/// # Returns
/// Mask of shape [size, size + offset] with -inf in upper triangular region
pub fn create_causal_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
    additive_causal_window_mask(size, offset, None)
}

/// Shared builder for full-width additive causal / windowed-causal masks.
///
/// Query row `q` (logical position `q + offset`) attends key column `k` iff
/// `k <= q + offset` and, when `window` is set, `k >= q + offset - window + 1`.
/// Output is `[size, size + offset]` f32 with `0.0` = attend, `-inf` = block,
/// identical to the previous `ones -> tril [-> triu -> multiply] -> greater ->
/// where` chain.
///
/// Built from broadcast index comparisons so the only full-size intermediates
/// are one or two bool `[size, total]` arrays plus the f32 result. The old
/// chain materialized up to six f32 `[size, total]` buffers, which at a 32k
/// single-pass prefill is ~4 GiB each and dominated the allocator high-water
/// mark of the very models the dense masks serve (issue #672).
///
/// Intentional FP32 output: additive attention masks carry 0/-inf sentinels
/// and are added to attention scores, not propagated as model activations.
fn additive_causal_window_mask(size: i32, offset: i32, window: Option<i32>) -> UniquePtr<MlxArray> {
    let total_len = size + offset;

    // Row coordinates as logical key positions [size, 1]; columns [1, total].
    let q_idx = ffi::reshape(&ffi::arange_i32(offset, offset + size, 1), &[size, 1]);
    let k_idx = ffi::reshape(&ffi::arange_i32(0, total_len, 1), &[1, total_len]);

    // Causal upper bound: k <= q + offset.
    let mut allowed = ffi::less_equal(&k_idx, &q_idx);

    // Sliding-window lower bound: k >= q + offset - window + 1.
    if let Some(w) = window {
        let q_low = ffi::reshape(
            &ffi::arange_i32(offset - w + 1, offset - w + 1 + size, 1),
            &[size, 1],
        );
        allowed = ffi::logical_and(&allowed, &ffi::greater_equal(&k_idx, &q_low));
    }

    let zero = crate::from_slice_f32(&[0.0], &[1, 1]);
    let neg_inf = crate::from_slice_f32(&[f32::NEG_INFINITY], &[1, 1]);
    ffi::where_cond(&allowed, &zero, &neg_inf)
}

/// Create a causal attention mask with per-sequence left-padding support.
///
/// Mirrors the Python `mlx_lm.models.base.create_causal_mask` with the
/// `left_padding` argument.  Used by [`crate::cache::batch_quant::BatchQuantizedKVCache::make_mask`]
/// and [`crate::cache::batch_quant::BatchTurboQuantKVCache::make_mask`].
///
/// # Arguments
/// * `n` — Number of query tokens in the current step (usually 1 for decode).
/// * `offset` — Actual number of tokens already in the KV buffer (`_idx` in Python
///   terminology, **not** the logical `offset` that starts negative for padded
///   sequences).  The total key length returned is `n + offset`.
/// * `left_padding` — Per-sequence number of leading padding tokens.  The mask
///   zeroes out (sets to −∞) key positions that are padding for each sequence.
///   When empty or all-zero the result is identical to [`create_causal_mask`].
///
/// # Returns
/// Additive mask with 0 for attended positions and −∞ for masked positions.
///
/// # Shape note
/// * **No padding** (`left_padding` empty or all-zero): `[n, n+offset]`, same
///   as [`create_causal_mask`].
/// * **With padding** (`left_padding` has at least one non-zero element):
///   `[B, 1, n, n+offset]` where `B = left_padding.len()`.  The `[B, 1]`
///   leading dims allow broadcasting against a `[B, H, n, n+offset]` score
///   tensor in batched SDPA.
///
/// # NaN-safe invariant for fully-masked padding query rows
/// A query row `q` whose absolute position `q + offset < left_padding[r]` is a
/// leading-padding query: its causal-AND-padding key set is empty, so without
/// special handling its mask row would be all −∞ and `softmax` over it yields
/// NaN. This builder rescues exactly the self/diagonal column (`k == q + offset`)
/// for those rows, so **every** query row attends at least one key and softmax
/// over any row is finite regardless of the fused-SDPA masked-column
/// semantics. The rescued column lies on the causal diagonal (distance 0), so
/// the padding-row output is garbage-but-finite; it is never consumed because
/// those padding key positions stay masked for every real query. Real query
/// rows (`q + offset >= left_padding[r]`) are byte-identical to the pre-rescue
/// construction. Removing the NaN at the source keeps every downstream layer's
/// K/V at padding positions finite, instead of relying on the kernel's hard
/// skip of additive −∞ columns to confine NaN to never-consumed slots.
///
/// Used by: BatchQuantizedKVCache, BatchTurboQuantKVCache
pub fn create_causal_mask_with_left_padding(
    n: i32,
    offset: i32,
    left_padding: &[i32],
) -> UniquePtr<MlxArray> {
    let total_len = n + offset;

    // ── Base causal (lower-triangular) mask ─────────────────────────────────
    // Shape: [n, total_len]  (0 = attend, -inf = mask after conversion)
    let ones = ffi::ones(&[n, total_len], dtype::FLOAT32);
    // `tril(ones, offset)` keeps the lower triangle starting `offset` columns
    // to the right of the main diagonal — i.e. the first `offset + q` entries
    // of query row `q`.  That matches the causal condition
    // `q_pos (= q + offset) >= k_pos`.
    let causal_tril = ffi::tril(&ones, offset);

    if left_padding.is_empty() || left_padding.iter().all(|&p| p == 0) {
        // Fast path: no per-sequence padding — identical to create_causal_mask.
        let zeros = ffi::zeros(&[n, total_len], dtype::FLOAT32);
        let neg_inf = ffi::full_f32(&[n, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
        let bool_mask = ffi::greater(&causal_tril, &zeros);
        return ffi::where_cond(&bool_mask, &zeros, &neg_inf);
    }

    // ── Left-padding filter ─────────────────────────────────────────────────
    // For each sequence `b`, key positions `k < left_padding[b]` are padding
    // and must be masked.  We build:
    //
    //   lp_tensor : [B, 1, 1, 1]  — per-sequence padding count
    //   rinds     : [1, 1, 1, total_len]  — key position indices
    //   lp_mask   : [B, 1, 1, total_len]  — True where key pos >= lp[b]
    //
    // Then broadcast-multiply with the causal mask.

    let b = left_padding.len() as i32;

    // Key position indices: 0, 1, …, total_len-1  (shape [1, 1, 1, total_len])
    let rinds_1d = ffi::arange_i32(0, total_len, 1);
    let rinds = ffi::reshape(&rinds_1d, &[1, 1, 1, total_len]);

    // Per-sequence left-padding: shape [B, 1, 1, 1]
    let lp_tensor = ffi::from_slice_i32(left_padding, &[b, 1, 1, 1]);

    // lp_mask[b,0,0,k] = (k >= left_padding[b])  — True = attend, False = mask
    // Using `greater_equal(rinds, lp_tensor)` broadcasts [B,1,1,total_len].
    let lp_mask = ffi::greater_equal(&rinds, &lp_tensor);

    // Causal mask broadcast: [1, 1, n, total_len]
    let causal_4d = ffi::reshape(&causal_tril, &[1, 1, n, total_len]);

    // Cast lp_mask to float for multiply (it is currently bool/int8 from the
    // greater_equal; we need float 0/1 to combine with the causal float mask).
    // Trick: use where_cond with ones/zeros to convert.
    let ones_lp = ffi::ones(&[b, 1, 1, total_len], dtype::FLOAT32);
    let zeros_lp = ffi::zeros(&[b, 1, 1, total_len], dtype::FLOAT32);
    let lp_mask_f32 = ffi::where_cond(&lp_mask, &ones_lp, &zeros_lp);

    // Combined: shape [B, 1, n, total_len]  (causal broadcasts over B)
    let combined = ffi::multiply(&causal_4d, &lp_mask_f32);

    // ── NaN-safe diagonal rescue for fully-masked padding query rows ─────────
    // A leading-padding query row `q` (absolute position `q + offset <
    // left_padding[r]`) has an empty causal-AND-padding key set, so its
    // `combined` row is all-zero and would convert to an all-−∞ mask row →
    // NaN softmax. Re-enable exactly the self/diagonal column (`k == q +
    // offset`) for those rows so every query row attends at least one key.
    // Zero extra cost on real query rows: their `padding_query` factor is 0,
    // so `rescue` is 0 and `combined` is unchanged (byte-identical).
    let qinds_1d = ffi::arange_i32(offset, offset + n, 1);
    let qinds = ffi::reshape(&qinds_1d, &[1, 1, n, 1]);
    let one_f = ffi::ones(&[1, 1, 1, 1], dtype::FLOAT32);
    let zero_f = ffi::zeros(&[1, 1, 1, 1], dtype::FLOAT32);
    // self_cond[_,_,q,k] = 1.0 where k == q + offset  (shape [1,1,n,total_len]).
    let self_cond = ffi::where_cond(&ffi::equal(&rinds, &qinds), &one_f, &zero_f);
    // padding_query[r,_,q,_] = 1.0 where q + offset < left_padding[r] (shape [B,1,n,1]).
    let padding_query = ffi::where_cond(&ffi::less(&qinds, &lp_tensor), &one_f, &zero_f);
    // rescue[r,_,q,k] = 1.0 only at the self column of a padding query row.
    let rescue = ffi::multiply(&self_cond, &padding_query);
    let combined = ffi::add(&combined, &rescue);

    // Convert 0/1 float mask to additive 0 / -inf mask.
    let zeros_out = ffi::zeros(&[b, 1, n, total_len], dtype::FLOAT32);
    let neg_inf_out = ffi::full_f32(&[b, 1, n, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_out = ffi::greater(&combined, &zeros_out);
    ffi::where_cond(&bool_out, &zeros_out, &neg_inf_out)
}

/// Create a sliding-window causal attention mask with per-sequence
/// left-padding support.
///
/// This is the windowed counterpart of [`create_causal_mask_with_left_padding`]
/// and the left-padding-aware counterpart of [`create_causal_mask_with_window`].
/// It is used by the **ragged batched MTP prefill** path
/// ([`crate::speculative::mtp`] via the Gemma 4 batched target adapter): when a
/// B > 1 burst window mixes prompts of different lengths, every row is
/// left-padded to `max_prompt_len` and the sliding-attention layers need a mask
/// that (a) enforces the sliding-window causal band AND (b) prevents real query
/// positions from attending to a row's leading padding keys.
///
/// # Arguments
/// * `size` — Number of query tokens in the current step (the padded prompt
///   width `max_prompt_len` for prefill).
/// * `offset` — Tokens already in the KV buffer before this call. For a fresh
///   prefill this is `0`. The total (uncapped) key length is `size + offset`.
/// * `window` — Sliding window size. `None` collapses to
///   [`create_causal_mask_with_left_padding`] (no windowing).
/// * `left_padding` — Per-sequence number of leading padding tokens. Key
///   positions `k < left_padding[b]` are masked for sequence `b`. When empty or
///   all-zero the result is byte-identical to [`create_causal_mask_with_window`].
///
/// # Returns
/// Additive mask (0 for attended positions, −∞ for masked positions).
///
/// # Shape note
/// * **No padding** (`left_padding` empty or all-zero): same shape as
///   [`create_causal_mask_with_window`] — `[size, T_k]` where
///   `T_k = min(size + offset, window)`.
/// * **With padding**: `[B, 1, size, T_k]` where `B = left_padding.len()`. The
///   `[B, 1]` leading dims broadcast against a `[B, H, size, T_k]` score tensor.
///
/// # NaN-safe invariant for fully-masked padding query rows
/// Like [`create_causal_mask_with_left_padding`], leading-padding query rows
/// (`q + offset < left_padding[r]`) keep their self/diagonal column attended,
/// so every query row has at least one attended key and softmax over any row is
/// finite regardless of the fused-SDPA masked-column semantics. The self column
/// is always inside the sliding-window band (distance 0 from the diagonal), so
/// the rescue never re-admits an out-of-window key; padding-row outputs are
/// garbage-but-finite and never consumed.
///
/// # Full-key-axis precondition (not "size + offset <= window")
/// The left-padding column filter assumes the K axis is the **full**
/// `size + offset` (column `k` maps 1:1 to logical key position `k`). The
/// sliding-window upper bound is enforced by an explicit `triu` band term, so
/// `size + offset > window` is fully supported as long as the backing cache has
/// not evicted/compacted any front keys. This is exactly the **MTP-buffered**
/// sliding-cache regime: the `RotatingKVCache` rollback buffer (`buffer_size`)
/// keeps the cache uncompacted up to a logical capacity of
/// `window + buffer_size`, so the resident prompt padding at `[0, lp)` stays in
/// the returned K and must keep being masked every verify step even once
/// `size + offset > window`. (An earlier version asserted `size + offset <=
/// window` and the Gemma 4 caller fell back to a padding-UNAWARE plain windowed
/// mask above the window, which leaked the resident padding into the
/// most-left-padded row and broke greedy parity.) The only unsupported case is a
/// genuinely *compacted* axis (`actual_kv_len < size + offset`); the ragged
/// caller never reaches buffer compaction in the eligible regime, and a
/// compacted axis has already evicted the (oldest) padding so a plain windowed
/// mask is padding-free there.
///
/// Used by: ragged batched MTP prefill + verify (Gemma 4 batched target adapter).
#[must_use]
pub fn create_causal_mask_with_window_and_left_padding(
    size: i32,
    offset: i32,
    window: Option<i32>,
    left_padding: &[i32],
) -> UniquePtr<MlxArray> {
    let no_padding = left_padding.is_empty() || left_padding.iter().all(|&p| p == 0);

    // Fast path: no per-sequence padding -> identical to the windowed mask.
    if no_padding {
        return create_causal_mask_with_window(size, offset, window);
    }

    let uncapped_len = size + offset;

    // Precondition: the **key axis is the full `size + offset`** (column k maps
    // 1:1 to logical key position k), i.e. the backing cache has NOT evicted /
    // compacted any front keys. The sliding-window upper bound is then enforced
    // by the `triu` band term below, so `size + offset > window` is fully
    // supported — this is the MTP-buffered sliding-cache regime, where the
    // rollback buffer (`buffer_size`) keeps the cache uncompacted (logical
    // capacity `window + buffer_size`) and the resident prompt padding at
    // `[0, lp)` must keep being masked even though `size + offset > window`. The
    // only unsupported case is a *compacted* axis (`actual_kv_len < size +
    // offset`), which the ragged caller avoids: the eligible regime never
    // reaches buffer compaction, and a compacted axis has already evicted the
    // (oldest) padding so a plain windowed mask would be padding-free anyway.
    let total_len = uncapped_len;

    // ── Windowed causal (band) base mask, [size, total_len] ─────────────────
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut band = ffi::tril(&ones, offset);
    if let Some(w) = window {
        // Enforce the sliding-window upper bound q <= k + window - 1, identical
        // to the non-capped branch of `create_causal_mask_with_window`.
        let upper_mask = ffi::triu(&ones, offset - w + 1);
        band = ffi::multiply(&band, &upper_mask);
    }

    // ── Left-padding column filter ──────────────────────────────────────────
    // For sequence `b`, key positions `k < left_padding[b]` are padding and must
    // be masked. Build a [B, 1, 1, total_len] boolean (k >= lp[b]) and multiply
    // with the band mask broadcast to [1, 1, size, total_len].
    let b = left_padding.len() as i32;

    let rinds_1d = ffi::arange_i32(0, total_len, 1);
    let rinds = ffi::reshape(&rinds_1d, &[1, 1, 1, total_len]);
    let lp_tensor = ffi::from_slice_i32(left_padding, &[b, 1, 1, 1]);
    let lp_mask = ffi::greater_equal(&rinds, &lp_tensor);

    let band_4d = ffi::reshape(&band, &[1, 1, size, total_len]);

    let ones_lp = ffi::ones(&[b, 1, 1, total_len], dtype::FLOAT32);
    let zeros_lp = ffi::zeros(&[b, 1, 1, total_len], dtype::FLOAT32);
    let lp_mask_f32 = ffi::where_cond(&lp_mask, &ones_lp, &zeros_lp);

    let combined = ffi::multiply(&band_4d, &lp_mask_f32);

    // ── NaN-safe diagonal rescue for fully-masked padding query rows ─────────
    // Identical to the non-windowed builder: re-enable the self/diagonal column
    // (`k == q + offset`) for leading-padding query rows (`q + offset <
    // left_padding[r]`). The self column lies on the causal diagonal (distance
    // 0), so it is always inside the sliding-window band; the rescue can never
    // re-admit an out-of-window key. Real query rows are byte-identical.
    let qinds_1d = ffi::arange_i32(offset, offset + size, 1);
    let qinds = ffi::reshape(&qinds_1d, &[1, 1, size, 1]);
    let one_f = ffi::ones(&[1, 1, 1, 1], dtype::FLOAT32);
    let zero_f = ffi::zeros(&[1, 1, 1, 1], dtype::FLOAT32);
    let self_cond = ffi::where_cond(&ffi::equal(&rinds, &qinds), &one_f, &zero_f);
    let padding_query = ffi::where_cond(&ffi::less(&qinds, &lp_tensor), &one_f, &zero_f);
    let rescue = ffi::multiply(&self_cond, &padding_query);
    let combined = ffi::add(&combined, &rescue);

    // Convert the 0/1 float mask to an additive 0 / -inf mask.
    let zeros_out = ffi::zeros(&[b, 1, size, total_len], dtype::FLOAT32);
    let neg_inf_out = ffi::full_f32(&[b, 1, size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_out = ffi::greater(&combined, &zeros_out);
    ffi::where_cond(&bool_out, &zeros_out, &neg_inf_out)
}

/// Exclude each row's stale key-tail gap from a batched additive attention mask.
///
/// After divergent (mixed) accepts in a B > 1 batched speculative verify round,
/// row `r`'s logical valid key end `per_row_valid_end[r]` lags the physical
/// cache offset (`gap_end`, the global max across rows). The keys in
/// `[per_row_valid_end[r], gap_end)` are that row's stale rejected-draft K/V:
/// resident in the unbounded full-attention `Cache::Standard` (whose
/// `zero_partial_accept_tail` is a no-op) and present as zeroed phantom columns
/// in the sliding `Cache::Rotating` (zeroed K still carries softmax weight).
/// The `mask == None` verify forward derives its mask from the global offset, so
/// row `r` would attend that gap; the standalone B = 1 reference trims its cache
/// exactly and has no such gap. Adding `-inf` to exactly those columns moves the
/// batched logits onto the B = 1 semantics, so it can only improve parity.
///
/// # Arguments
/// * `base`: additive attention mask carrying 0 (attend) / −∞ (mask)
///   sentinels, either 2-D `[n, K]` (broadcasts over the batch) or 4-D
///   `[B | 1, 1, n, K]`.
/// * `per_row_valid_end`: per-row logical valid key end (length `B`). Column
///   `k` is penalised for row `r` when `per_row_valid_end[r] <= k < gap_end`.
/// * `gap_end`: exclusive upper bound of the stale gap (the physical / global
///   cache offset). Rows with `per_row_valid_end[r] >= gap_end` are unchanged.
///
/// # Returns
/// `[B, 1, n, K]` = `base + penalty`. Penalty cells are 0 or −∞; adding 0/−∞ to
/// the 0/−∞ `base` cannot produce a NaN (there are no `+∞` operands), so the
/// result stays a clean additive mask.
#[must_use]
pub fn mask_stale_key_gap(
    base: &MlxArray,
    per_row_valid_end: &[i32],
    gap_end: i32,
) -> UniquePtr<MlxArray> {
    let b = per_row_valid_end.len() as i32;
    let shape = ffi::array_shape(base);
    // The key axis is always the last dim (2-D `[n, K]` or 4-D `[B,1,n,K]`).
    let k_len = *shape.last().expect("attention mask has at least one dim");

    // Column indices [0, K): shape [1, 1, 1, K].
    let kinds_1d = ffi::arange_i32(0, k_len, 1);
    let kinds = ffi::reshape(&kinds_1d, &[1, 1, 1, k_len]);

    // Per-row valid end and the (shared) gap end: shape [B,1,1,1] / [1,1,1,1].
    let ve_tensor = ffi::from_slice_i32(per_row_valid_end, &[b, 1, 1, 1]);
    let gap_end_tensor = ffi::from_slice_i32(&[gap_end], &[1, 1, 1, 1]);

    let one_f = ffi::ones(&[1, 1, 1, 1], dtype::FLOAT32);
    let zero_f = ffi::zeros(&[1, 1, 1, 1], dtype::FLOAT32);
    // in_gap[r,_,_,k] = (k >= ve[r]) AND (k < gap_end)  -> 0/1 f32, shape [B,1,1,K].
    let ge_ve = ffi::where_cond(&ffi::greater_equal(&kinds, &ve_tensor), &one_f, &zero_f);
    let lt_end = ffi::where_cond(&ffi::less(&kinds, &gap_end_tensor), &one_f, &zero_f);
    let in_gap = ffi::multiply(&ge_ve, &lt_end);

    // penalty = in_gap ? -inf : 0   (shape [B,1,1,K]).
    let neg_inf = ffi::full_f32(&[1, 1, 1, 1], f32::NEG_INFINITY, dtype::FLOAT32);
    let gap_bool = ffi::greater(&in_gap, &zero_f);
    let penalty = ffi::where_cond(&gap_bool, &neg_inf, &zero_f);

    // base + penalty broadcasts base over B (when base is 2-D / B==1) and
    // penalty over the query axis `n`. Result is [B, 1, n, K].
    match shape.len() {
        2 => {
            let base_4d = ffi::reshape(base, &[1, 1, shape[0], shape[1]]);
            ffi::add(&base_4d, &penalty)
        }
        4 => ffi::add(base, &penalty),
        other => panic!("mask_stale_key_gap expects a 2-D or 4-D base mask, got {other}-D"),
    }
}

/// Create a boolean causal attention mask.
/// Used by: same as `create_causal_mask` (experimental path)
///
/// Returns a bool mask where `true` means "allowed attention".
pub fn create_causal_bool_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
    let total_len = size + offset;
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mask = ffi::tril(&ones, offset);
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    ffi::greater(&mask, &zeros)
}

/// Create a causal attention mask with sliding window.
/// Used by: Gemma2, Gemma3, Gemma3n, Gemma4, Qwen3, Ministral and other windowed-attention callers
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (tokens already in the cache before this call)
/// * `window` - Sliding window size (None for full attention)
///
/// # Returns
/// Mask with sliding window constraint applied, shaped `(size, T_k)` where
/// `T_k = min(size + offset, window)`.
///
/// ## Shape semantics when `size + offset > window`
///
/// A `RotatingKVCache` with `max_size = window` returns at most `window` K tokens
/// (the most recent ones).  The mask must match this T_k dimension so that
/// `mx::fast::scaled_dot_product_attention` can broadcast it against the score
/// tensor `(B, H, T_q, T_k)`.
///
/// When `total_len (= size + offset) > window`, the mask is produced as if we
/// took the last `window` columns of the full `(size, total_len)` causal mask.
/// Cache slot `k_cache` corresponds to logical key position
/// `k_cache + (total_len - window)`, and query row `q` corresponds to logical
/// query position `q + offset`. The causal condition is
/// `q_logical >= k_logical`:
///
/// ```text
/// q + offset >= k_cache + (total_len - window)
/// q + offset >= k_cache + (size + offset - window)
/// q         >= k_cache + (size - window)
/// ```
///
/// Hence the `tril` diagonal offset is `-(size - window) = window - size`,
/// independent of `offset`. The resulting mask shape is `(size, window)`,
/// matching the RotatingKVCache output and allowing broadcast to
/// `(B, H, size, window)`.
///
/// ## Why the window upper-bound term is elided in the capped path
///
/// In the full-length path the `triu` enforces `q <= k + window - 1`.  In the
/// capped path the column range is already restricted to the window; the upper
/// bound is always satisfied, so `triu` is omitted.
pub fn create_causal_mask_with_window(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    let uncapped_len = size + offset;

    // When a window is specified and the K sequence would exceed the window
    // (i.e. RotatingKVCache returns fewer than `uncapped_len` tokens), cap the
    // mask width so it matches the actual K dimension returned by the cache.
    //
    // Example: size=4096, offset=0, window=1024
    //   uncapped_len = 4096, which is > 1024.
    //   The cache returns K of shape (B, H, 1024, D).
    //   The score tensor is (B, H, 4096, 1024).
    //   A mask of (4096, 4096) cannot broadcast to (1, 8, 4096, 1024) — SIGABRT.
    //   Fix: produce mask of (4096, 1024) using adjusted tril offset.
    let (total_len, tril_offset) = if let Some(w) = window {
        if uncapped_len > w {
            // Cap: take the last `w` columns of the full (size, uncapped_len) mask.
            // Cache slot k_c holds logical key position k_c + (uncapped_len - w);
            // query row q holds logical query position q + offset. The causal
            // condition q + offset >= k_c + (uncapped_len - w) simplifies to
            // q >= k_c + (size - w), so the tril diagonal offset is
            // -(size - w) = w - size — independent of `offset`.
            (w, w - size)
        } else {
            (uncapped_len, offset)
        }
    } else {
        (uncapped_len, offset)
    };

    // Create lower triangular mask (1 = attend, 0 = mask)
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut mask = ffi::tril(&ones, tril_offset);

    // Apply sliding window upper-bound only when the mask is NOT capped.
    // In the capped path the column range is already the window; the upper
    // bound (q <= k + window - 1) is trivially satisfied.
    if let Some(w) = window
        && uncapped_len <= w
    {
        // Non-capped path: enforce window upper bound.
        let upper_mask = ffi::triu(&ones, offset - w + 1);
        mask = ffi::multiply(&mask, &upper_mask);
    }

    // Convert to attention mask format: where mask=1 -> 0, where mask=0 -> -inf
    // Use where_cond to avoid NaN from 0 * -inf
    // Intentional FP32: additive attention masks carry 0/-inf sentinels and are
    // added to attention scores, not propagated as model activations.
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&mask, &zeros); // mask > 0 gives bool mask

    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Create a sliding-window causal mask sized to the *full* key axis, without
/// the `min(size + offset, window)` cap applied by [`create_causal_mask_with_window`].
/// Used by: Gemma 3, Gemma 4 single-pass prefill longer than the sliding window, Cohere2/Gemma3n/Olmo3 dense prefill (#413)
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Tokens already resident in the KV cache before this call
/// * `window` - Sliding window size (`None` for full attention)
///
/// # Returns
/// An additive `[size, size + offset]` mask (`0.0` = attend, `-inf` = block)
/// where query row `q` (logical position `q + offset`) attends to key column
/// `k` iff `k <= q + offset` (causal) and `k >= q + offset - window + 1`
/// (sliding-window lower bound).
///
/// ## Why an uncapped variant is needed
///
/// [`create_causal_mask_with_window`] caps the key axis to `window` when
/// `size + offset > window`, on the assumption that a [`RotatingKVCache`] with
/// `max_size = window` only ever returns the most recent `window` keys. That
/// assumption holds for decode and for cache rollover, but **not** for a
/// single-pass prefill whose length exceeds the window: the rotating cache
/// keeps every prefill key (it only trims to `window` on the subsequent decode
/// step), so the cache returns all `size` keys. With the capped `[size, window]`
/// mask the attention layer slices K/V down to the trailing `window` keys, which
/// strands the earliest query rows (logical position `< size - window`) with no
/// visible key, producing an all-masked row that softmaxes to NaN and degenerates the
/// output. Mirroring mlx-lm's `RotatingKVCache`, the correctness of the window
/// comes from the *mask*, not from physically dropping keys, so prefill must use
/// this full-width mask. Consumers backed by a rotating cache pair this with a
/// `trim_mask_to_keys` step so that any later capped fetch still aligns. See
/// issue #401.
///
/// [`RotatingKVCache`]: crate::cache::RotatingKVCache
pub fn create_causal_mask_with_window_full(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    additive_causal_window_mask(size, offset, window)
}

/// Build the sliding-window attention mask for a multi-token prefill
/// (`size > 1`), sized to the keys the cache actually returns.
///
/// A fresh single-pass prefill that exceeds the window is the degenerate case
/// behind issue #401/#408: both [`RotatingKVCache`] (on its first prefill
/// append) and a dense `KVCache` keep *every* prefill key, so the window must
/// be enforced by the mask over the full key axis, not by capping the mask and
/// dropping keys. For that case (`size > window && sliding_offset == 0`) this
/// returns the uncapped [`create_causal_mask_with_window_full`] `[size, size]`
/// mask. In every other case (within-window prefill, or a rolled-over rotating
/// cache that already returns at most `window` keys) it returns the clamped
/// [`create_causal_mask_with_window`] mask with the same
/// `sliding_offset.min((window - size).max(0))` adjustment the per-model mask
/// builders used before, so greedy output stays byte-identical there.
///
/// The `sliding_offset == 0` gate mirrors the gemma3 prior art: only a fresh
/// prefill is guaranteed to hold all `size` keys; once a rotating cache has
/// rolled over (`sliding_offset > 0`) it returns at most `window` keys and the
/// clamped mask is the matching shape. Models whose attention layer slices K/V
/// to the trailing window must slice to the mask's key dimension (not blindly
/// to `window`) so they keep the full key set when this returns the full mask.
///
/// Used by: GptOss, Mellum, Exaone4, ExaoneMoE, Ministral3, Step3P5, Gemma3,
/// Gemma4 sliding-window prefill mask construction.
///
/// Gemma3 carries the documented `sliding_offset == 0` invariant (no trim step,
/// so it can legitimately see `sliding_offset > 0` with `size > window` under
/// chunked prefill or multi-turn reuse and needs the clamped path). Gemma4
/// routes through `trim_mask_to_keys`: when `size > window && sliding_offset > 0`,
/// RotatingKVCache trims to exactly `window` keys and `trim_mask_to_keys` crops
/// the full `[size, size+offset]` mask to its trailing `window` columns, which
/// is the same band as the clamped output of this helper (`q-size+1 <= k <=
/// q-size+window`, independent of offset). Old-trimmed equals new for every
/// input, so the migration is behaviour-preserving (#410).
///
/// [`RotatingKVCache`]: crate::cache::RotatingKVCache
pub fn create_sliding_window_prefill_mask(
    size: i32,
    sliding_offset: i32,
    window: i32,
) -> UniquePtr<MlxArray> {
    if size > window && sliding_offset == 0 {
        create_causal_mask_with_window_full(size, 0, Some(window))
    } else {
        let effective_offset = sliding_offset.min((window - size).max(0));
        create_causal_mask_with_window(size, effective_offset, Some(window))
    }
}

/// Dense-`KVCache` variant of [`create_sliding_window_prefill_mask`].
///
/// A dense `KVCache` retains EVERY key (it never trims to `window`, unlike a
/// `RotatingKVCache`), and the consumer slices K/V to the mask's key axis. So
/// the correct prefill mask is ALWAYS the full `[size, size + sliding_offset]`
/// windowed-causal mask over all retained keys, for every size/offset
/// combination. There is no clamped branch: the full mask keeps every key and
/// the window is enforced by the mask, not by physically dropping keys.
///
/// For the common within-window prefill (`size + sliding_offset <= window`)
/// this is byte-identical to [`create_sliding_window_prefill_mask`]: the
/// clamped builder takes its non-capped path and produces the same
/// `tril(offset)` intersect `triu(offset - window + 1)` band over `[size,
/// size + sliding_offset]`, so greedy output there is unchanged. The full mask
/// fixes the two dense-cache degeneration cases the clamped path caused once
/// the total exceeded the window:
///
/// 1. `size > window` (at any offset): the clamped `[size, window]` mask's
///    `tril` diagonal `window - size < 0` stranded the earliest query rows
///    (logical position `< size - window`) with an all-`-inf` row, which
///    softmaxed to NaN and decoded to `<pad>`.
/// 2. `size <= window && size + sliding_offset > window`: the clamped path's
///    trailing-`window` K/V slice dropped the OLDEST in-window keys for the
///    earliest query rows, silently narrowing their attention window
///    (coherent-but-wrong output rather than NaN/`<pad>`).
///
/// The rotating-cache helper [`create_sliding_window_prefill_mask`] keeps the
/// clamped path because a `RotatingKVCache` physically returns at most `window`
/// keys once it has rolled over, so the clamped mask is the matching shape.
/// See issue #413.
///
/// Used by: Cohere2, Gemma3n, Olmo3 sliding-window prefill mask construction.
pub fn create_sliding_window_prefill_mask_dense(
    size: i32,
    sliding_offset: i32,
    window: i32,
) -> UniquePtr<MlxArray> {
    create_causal_mask_with_window_full(size, sliding_offset, Some(window))
}

/// Create a boolean causal attention mask with optional sliding window.
/// Used by: same as `create_causal_mask_with_window` (experimental path)
///
/// Returns a bool mask where `true` means "allowed attention".
/// Shape is `(size, min(size + offset, window))` when window is specified.
/// See `create_causal_mask_with_window` for the shape-capping rationale.
pub fn create_causal_bool_mask_with_window(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    let uncapped_len = size + offset;

    let (total_len, tril_offset) = if let Some(w) = window {
        if uncapped_len > w {
            // See `create_causal_mask_with_window` for the derivation:
            // tril diagonal offset is `w - size`, independent of `offset`.
            (w, w - size)
        } else {
            (uncapped_len, offset)
        }
    } else {
        (uncapped_len, offset)
    };

    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut mask = ffi::tril(&ones, tril_offset);

    if let Some(w) = window
        && uncapped_len <= w
    {
        let upper_mask = ffi::triu(&ones, offset - w + 1);
        mask = ffi::multiply(&mask, &upper_mask);
    }

    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    ffi::greater(&mask, &zeros)
}

// KV Cache Utilities.
/// Repeat key/value tensors for grouped-query attention.
///
/// When n_kv_heads < n_heads, we need to repeat K and V to match Q dimensions.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, n_kv_heads, seq_len, head_dim]
/// * `n_rep` - Number of times to repeat (n_heads / n_kv_heads)
///
/// # Returns
/// Tensor of shape [batch, n_heads, seq_len, head_dim]
pub fn repeat_kv(x: &MlxArray, n_rep: i32) -> UniquePtr<MlxArray> {
    if n_rep == 1 {
        // No repetition needed — return a zero-copy view via reshape
        let shape = ffi::array_shape(x);
        return ffi::reshape(x, &shape);
    }

    let shape = ffi::array_shape(x);
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // Reshape to [batch, n_kv_heads, 1, seq_len, head_dim]
    let x_exp = ffi::reshape(x, &[batch, n_kv_heads, 1, seq_len, head_dim]);

    // Broadcast to [batch, n_kv_heads, n_rep, seq_len, head_dim]
    let x_broad = ffi::broadcast_to(&x_exp, &[batch, n_kv_heads, n_rep, seq_len, head_dim]);

    // Reshape to [batch, n_kv_heads * n_rep, seq_len, head_dim]
    ffi::reshape(&x_broad, &[batch, n_kv_heads * n_rep, seq_len, head_dim])
}

// Activation Functions.
/// SiLU (Swish) activation: x * sigmoid(x) — compiled kernel fusion
#[inline]
pub fn silu(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_silu(x)
}

/// GELU activation with sigmoid approximation: x * sigmoid(1.702 * x)
///
/// NOTE: This is NOT the same as exact GELU or tanh-approximate GELU.
/// For exact GELU, use `ffi::gelu()` (re-exported as `mlxcel_core::gelu`).
/// For tanh-approximate GELU, use `gelu_approx()`.
#[inline]
pub fn gelu_sigmoid(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x_dtype = ffi::array_dtype(x);
    let coef = ffi::full_f32(&[1], 1.702, x_dtype);
    let scaled = ffi::multiply(&coef, x);
    let sigmoid_x = ffi::sigmoid(&scaled);
    ffi::multiply(x, &sigmoid_x)
}

/// ReLU squared activation: max(0, x)^2 — compiled kernel fusion
#[inline]
pub fn relu_squared(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_relu_squared(x)
}

///// Numerically stable softplus activation: log(1 + exp(x)).
/// Uses logaddexp(x, 0) internally to match Python's mx.logaddexp(x, 0).
/// This avoids float16 overflow for values >= ~11.09 (exp(x) > float16 max).
/// Used by: Mamba, Mamba2, Jamba, GatedDelta, RecurrentGemma
#[inline]
pub fn softplus(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::softplus(x)
}

/// GELU approximate activation (erf-based for numerical stability with bf16)
/// Used by many models like Phi
#[inline]
pub fn gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::gelu_approx(x)
}

/// GeGELU activation for Phi3Small
/// Splits input into gelu and linear parts (interleaved), applies gelu to first half,
/// then computes: gelu(x[::2]) * (x[1::2] + 1.0)
///
/// # Arguments
/// * `x` - Input array where last dim will be split into interleaved gelu/linear parts
/// * `limit` - Clipping limit for numerical stability
pub fn gegelu(x: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    let x_dtype = ffi::array_dtype(x);
    let shape = ffi::array_shape(x);
    let ndim = shape.len();
    let last_dim = shape[ndim - 1];
    let half_dim = last_dim / 2;

    // Split into gelu part (even indices) and linear part (odd indices)
    // Reshape: [B, L, D] -> [B, L, D/2, 2]
    let mut new_shape = shape.clone();
    new_shape[ndim - 1] = half_dim;
    new_shape.push(2);

    let x_reshaped = ffi::reshape(x, &new_shape);

    // Select gelu_part (index 0) and linear_part (index 1) along last axis
    // Using slice: gelu_part = x_reshaped[..., :, 0], linear_part = x_reshaped[..., :, 1]
    let mut starts = vec![0i32; ndim + 1];
    let mut stops: Vec<i32> = new_shape.clone();

    // gelu_part: slice [..., :, 0:1] then squeeze
    starts[ndim] = 0;
    stops[ndim] = 1;
    let gelu_part = ffi::slice(&x_reshaped, &starts, &stops);
    let gelu_part = ffi::squeeze_axis(&gelu_part, ndim as i32);

    // linear_part: slice [..., :, 1:2] then squeeze
    starts[ndim] = 1;
    stops[ndim] = 2;
    let linear_part = ffi::slice(&x_reshaped, &starts, &stops);
    let linear_part = ffi::squeeze_axis(&linear_part, ndim as i32);

    // Clip both parts for numerical stability
    let neg_limit = ffi::full_f32(&[1], -limit, x_dtype);
    let pos_limit = ffi::full_f32(&[1], limit, x_dtype);

    let a_gelu = ffi::clip(&gelu_part, &neg_limit, &pos_limit);
    let a_linear = ffi::clip(&linear_part, &neg_limit, &pos_limit);

    // Apply GELU approximation: x * sigmoid(1.702 * x)
    let coef = ffi::full_f32(&[1], 1.702, x_dtype);
    let scaled = ffi::multiply(&coef, &a_gelu);
    let sigmoid_x = ffi::sigmoid(&scaled);
    let out_gelu = ffi::multiply(&a_gelu, &sigmoid_x);

    // Compute: out_gelu * (a_linear + 1.0)
    let ones = ffi::full_f32(&[1], 1.0, x_dtype);
    let linear_plus_one = ffi::add(&a_linear, &ones);
    let out = ffi::multiply(&out_gelu, &linear_plus_one);
    if ffi::array_dtype(&out) == x_dtype {
        out
    } else {
        ffi::astype(&out, x_dtype)
    }
}

// Gemma-specific Functions.
/// Softcap function for Gemma 2/3 attention and logits.
///
/// Applies tanh(x / cap) * cap to prevent extreme values.
///
/// # Arguments
/// * `x` - Input array
/// * `cap` - Softcapping value
///
/// # Returns
/// Softcapped array
pub fn softcap(x: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let scaled = crate::divide_scalar(x, cap);
    let tanhed = ffi::tanh(&scaled);
    crate::multiply_scalar(&tanhed, cap)
}

/// Clip residual addition for float16 overflow prevention (Gemma 3).
///
/// When using float16, casts to float32, adds, clips to float16 range,
/// and casts back to float16. For other dtypes, performs normal addition.
///
/// # Arguments
/// * `x` - First input array
/// * `y` - Second input array (to be added to x)
///
/// # Returns
/// Clipped residual sum
pub fn clip_residual_f16(x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype_code = ffi::array_dtype(x);

    // Check if dtype is float16 (dtype code 9)
    if dtype_code != dtype::FLOAT16 {
        // Not float16, just add normally
        return ffi::add(x, y);
    }

    // float16 max is approximately 65504
    let bound = 65504.0f32;

    // Intentional FP32: the residual is widened only for overflow-safe clipping
    // and is cast back to f16 before returning.
    let x_f32 = ffi::astype(x, dtype::FLOAT32);
    let y_f32 = ffi::astype(y, dtype::FLOAT32);

    // Add
    let sum = ffi::add(&x_f32, &y_f32);

    // Create bound arrays
    let min_bound = ffi::full_f32(&[1], -bound, dtype::FLOAT32);
    let max_bound = ffi::full_f32(&[1], bound, dtype::FLOAT32);

    // Clip
    let clipped = ffi::clip(&sum, &min_bound, &max_bound);

    // Cast back to f16
    ffi::astype(&clipped, dtype::FLOAT16)
}

// Neural Accelerator Tile Alignment Utilities.

/// Tile size for the M5 Neural Accelerator optimal matrix operation.
pub const NA_TILE_SIZE: usize = 32;

/// Align a sequence length up to the nearest multiple of `NA_TILE_SIZE`.
///
/// When the sequence is already aligned (i.e. `len % NA_TILE_SIZE == 0`),
/// the value is returned unchanged. Otherwise it is rounded up so that
/// the prefill input perfectly fills complete 32×32 tiles, enabling peak
/// Neural Accelerator throughput on M5+ hardware.
///
/// # Examples
/// ```ignore
/// assert_eq!(align_to_na_tile(10), 32);
/// assert_eq!(align_to_na_tile(32), 32);
/// assert_eq!(align_to_na_tile(33), 64);
/// assert_eq!(align_to_na_tile(0),   0);
/// ```
#[inline]
pub fn align_to_na_tile(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    len.div_ceil(NA_TILE_SIZE) * NA_TILE_SIZE
}

/// Create a causal attention mask for a tile-aligned padded prefill.
///
/// The input sequence has `actual_len` real tokens followed by `pad_len =
/// padded_len - actual_len` padding tokens. The returned mask has shape
/// `[padded_len, padded_len]` and encodes two constraints:
///
/// 1. **Causal**: query position `q` may only attend to key positions `k ≤ q`.
/// 2. **No padding leakage**: key positions `k ≥ actual_len` are always masked
///    with −∞, even for query positions that are themselves padding tokens.
///
/// This ensures that after the padded forward pass:
/// - The logits at position `actual_len - 1` correctly predict the next token.
/// - Padding tokens do not pollute the KV cache values that will be trimmed.
///
/// # Arguments
/// * `actual_len` - Number of real (non-padding) tokens in the sequence.
/// * `padded_len` - Total sequence length after alignment (≥ `actual_len`).
/// * `offset`     - Number of tokens already in the KV cache (typically 0 for
///   fresh prefill, non-zero for multi-turn continuation).
pub fn create_padded_prefill_mask(
    actual_len: i32,
    padded_len: i32,
    offset: i32,
) -> UniquePtr<MlxArray> {
    let total_kv = padded_len + offset;

    // Step 1: causal lower-triangular mask over the full padded shape.
    let ones = ffi::ones(&[padded_len, total_kv], dtype::FLOAT32);
    let causal = ffi::tril(&ones, offset);

    // Step 2: build a key-padding mask that zeros out positions ≥ actual_len.
    // Shape: [1, total_kv]  (broadcast along the query axis).
    // Value: 1 for valid key positions, 0 for padding key positions.
    let mut valid_mask_data = vec![0f32; total_kv as usize];
    for v in valid_mask_data
        .iter_mut()
        .take((actual_len + offset) as usize)
    {
        *v = 1.0;
    }
    let valid_mask = ffi::from_slice_f32(&valid_mask_data, &[1, total_kv]);

    // Combine: both constraints must hold (multiply, then convert to -inf mask).
    let combined = ffi::multiply(&causal, &valid_mask);

    // Convert to additive mask: 1 → 0.0,  0 → -inf
    // Intentional FP32: additive attention masks carry 0/-inf sentinels and are
    // added to attention scores, not propagated as model activations.
    let zeros = ffi::zeros(&[padded_len, total_kv], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[padded_len, total_kv], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&combined, &zeros);
    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

// Shape Utilities.
/// Concatenate two arrays along the specified axis.
#[inline]
pub fn concatenate(a: &MlxArray, b: &MlxArray, axis: i32) -> UniquePtr<MlxArray> {
    crate::concatenate(a, b, axis)
}

/// Stack arrays along a new axis.
pub fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let ptrs: Vec<*const MlxArray> = arrays
        .iter()
        .map(|a| a.as_ref().unwrap() as *const _)
        .collect();
    unsafe { ffi::stack(&ptrs, axis) }
}

// Pipeline Hint for Layer-Level async_eval

/// Granularity setting for layer-boundary pipeline hints.
///
/// Controlled via the `MLXCEL_PIPELINE_GRANULARITY` environment variable:
/// - `layer`   — call `async_eval` after every transformer layer
/// - `block:N` — call `async_eval` every N layers (e.g. `block:4`)
/// - `off`     — no intermediate eval (default; preserves MLX graph fusion)
#[derive(Debug, Clone, Copy)]
enum PipelineMode {
    /// No intermediate eval — current MLX default behavior.
    Off,
    /// Evaluate after every transformer layer.
    PerLayer,
    /// Evaluate every N layers.
    PerBlock(usize),
}

fn get_pipeline_mode() -> PipelineMode {
    match std::env::var("MLXCEL_PIPELINE_GRANULARITY")
        .as_deref()
        .unwrap_or("off")
    {
        "layer" => PipelineMode::PerLayer,
        s if s.starts_with("block:") => {
            let n = s[6..].parse::<usize>().unwrap_or(4);
            PipelineMode::PerBlock(n.max(1))
        }
        _ => PipelineMode::Off,
    }
}

/// Insert an `async_eval` pipeline hint at a transformer layer boundary.
///
/// Calling this after each layer's `forward()` allows MLX's lazy evaluation
/// engine to begin executing the current layer's compute graph while the next
/// layer's weights are prefetched into L2 cache, hiding memory latency.
///
/// On M5 (Neural Accelerator + GPU shader cores), this can improve throughput
/// by overlapping NA compute for layer N with weight loads for layer N+1.
///
/// Activation is controlled by `MLXCEL_PIPELINE_GRANULARITY`:
/// - `layer`   — hint after every layer
/// - `block:N` — hint every N layers
/// - `off`     — no hints (default; preserves MLX graph fusion)
///
/// # Arguments
/// * `hidden` - The hidden state tensor output from the current layer.
/// * `layer_idx` - Zero-based index of the layer that was just executed.
/// * `total_layers` - Total number of transformer layers in the model.
///
/// Used by: Llama3, Qwen3, Gemma, Gemma2, Gemma3
#[inline]
pub fn pipeline_hint(hidden: &MlxArray, layer_idx: usize, total_layers: usize) {
    static MODE: OnceLock<PipelineMode> = OnceLock::new();
    let mode = MODE.get_or_init(get_pipeline_mode);

    // Never emit a hint after the last layer — the caller will eval the output.
    if layer_idx + 1 >= total_layers {
        return;
    }

    match mode {
        PipelineMode::Off => {}
        PipelineMode::PerLayer => {
            ffi::async_eval(hidden);
        }
        PipelineMode::PerBlock(n) => {
            if (layer_idx + 1).is_multiple_of(*n) {
                ffi::async_eval(hidden);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_axis_basic() {
        // Create a simple test array
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice middle portion
        let sliced = slice_axis(&x, 1, 2, 5);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_slice_axis_end_minus_one() {
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice from index 5 to end using -1
        let sliced = slice_axis(&x, 1, 5, -1);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 5, 4]); // 10 - 5 = 5
    }

    #[test]
    fn test_repeat_kv() {
        let x = ffi::ones(&[1, 4, 10, 64], dtype::FLOAT32);

        // Repeat 2 times (4 heads -> 8 heads)
        let repeated = repeat_kv(&x, 2);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let x = ffi::ones(&[1, 8, 10, 64], dtype::FLOAT32);

        // No repeat needed
        let repeated = repeat_kv(&x, 1);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }

    #[test]
    fn test_align_to_na_tile_zero() {
        assert_eq!(align_to_na_tile(0), 0);
    }

    #[test]
    fn test_align_to_na_tile_exact() {
        // Already aligned
        assert_eq!(align_to_na_tile(32), 32);
        assert_eq!(align_to_na_tile(64), 64);
        assert_eq!(align_to_na_tile(128), 128);
    }

    #[test]
    fn test_align_to_na_tile_short() {
        // Prompts shorter than one tile
        assert_eq!(align_to_na_tile(1), 32);
        assert_eq!(align_to_na_tile(10), 32);
        assert_eq!(align_to_na_tile(31), 32);
    }

    #[test]
    fn test_align_to_na_tile_cross_boundary() {
        assert_eq!(align_to_na_tile(33), 64);
        assert_eq!(align_to_na_tile(63), 64);
        assert_eq!(align_to_na_tile(65), 96);
    }

    #[test]
    fn test_create_padded_prefill_mask_shape() {
        // actual_len=10, padded_len=32, offset=0
        let mask = create_padded_prefill_mask(10, 32, 0);
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![32, 32]);
    }

    #[test]
    fn test_create_padded_prefill_mask_no_padding() {
        // When actual_len == padded_len, result equals a standard causal mask
        let mask = create_padded_prefill_mask(8, 8, 0);
        let ref_mask = create_causal_mask(8, 0);
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![8, 8]);
        let ref_shape = ffi::array_shape(&ref_mask);
        assert_eq!(ref_shape, vec![8, 8]);
    }

    // --- Sliding window mask shape regression tests -------------

    /// Gemma3-4B trigger: seq_len=4096, window=1024, offset=0.
    ///
    /// Before the fix the mask had shape (4096, 4096).  MLX SDPA falls back to
    /// software when head_dim=256 (not in the Metal fast-kernel list) and its
    /// fallback tried to broadcast (4096, 4096) against score (B, H, 4096, 1024)
    /// → SIGABRT.  After the fix the mask must be (4096, 1024).
    #[test]
    fn test_sliding_window_mask_shape_capped_when_seq_exceeds_window() {
        let mask = create_causal_mask_with_window(4096, 0, Some(1024));
        let shape = ffi::array_shape(&mask);
        // Must be (T_q=4096, T_k=min(4096+0, 1024)=1024), NOT (4096, 4096).
        assert_eq!(
            shape,
            vec![4096, 1024],
            "mask shape must match RotatingKVCache output (4096, 1024); \
             got {shape:?} — broadcast mismatch against score (B,H,4096,1024) would SIGABRT"
        );
    }

    /// When seq_len < window the mask must retain its full (T_q, T_q+offset)
    /// shape — no spurious capping.
    #[test]
    fn test_sliding_window_mask_shape_uncapped_when_seq_within_window() {
        // seq=512, offset=0, window=1024: total=512 < 1024 → no cap
        let mask = create_causal_mask_with_window(512, 0, Some(1024));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![512, 512]);
    }

    /// When total_len exactly equals window the mask must NOT be capped.
    #[test]
    fn test_sliding_window_mask_shape_at_window_boundary() {
        // seq=512, offset=512, window=1024: total=1024 == window → no cap
        let mask = create_causal_mask_with_window(512, 512, Some(1024));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![512, 1024]);
    }

    /// In the capped path, queries below the cache start horizon must be fully
    /// masked (-inf).  For seq=4, window=2, offset=0:
    ///   cache holds last 2 of the 4 input tokens (positions 2..3).
    ///   q=0 and q=1 cannot attend to any cached key → all -inf.
    ///   q=2 attends to k=0 (input pos 2≥input pos 2). → row 2, col 0 = 0.
    ///   q=3 attends to k=0,1 (input 3≥2, 3≥3). → row 3, cols 0 and 1 = 0.
    #[test]
    fn test_sliding_window_mask_values_when_capped() {
        // Produce (4, 2) mask: rows=T_q, cols=T_k=window=2
        let mask = create_causal_mask_with_window(4, 0, Some(2));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![4, 2]);

        // Extract values (the mask is additive: 0.0 = attend, -inf = block)
        let row0_col0 = ffi::item_f32(&ffi::slice(&mask, &[0, 0], &[1, 1]));
        let row1_col0 = ffi::item_f32(&ffi::slice(&mask, &[1, 0], &[2, 1]));
        let row2_col0 = ffi::item_f32(&ffi::slice(&mask, &[2, 0], &[3, 1]));
        let row3_col0 = ffi::item_f32(&ffi::slice(&mask, &[3, 0], &[4, 1]));
        let row3_col1 = ffi::item_f32(&ffi::slice(&mask, &[3, 1], &[4, 2]));

        // q=0,1 cannot see any cache key (cache starts at input pos 2)
        assert!(
            row0_col0.is_infinite() && row0_col0 < 0.0,
            "row0_col0 should be -inf, got {row0_col0}"
        );
        assert!(
            row1_col0.is_infinite() && row1_col0 < 0.0,
            "row1_col0 should be -inf, got {row1_col0}"
        );
        // q=2 attends to cache-k=0 (input pos 2 ≥ input pos 2)
        assert_eq!(row2_col0, 0.0, "row2_col0 should be 0.0 (attend)");
        // q=3 attends to both cache keys
        assert_eq!(row3_col0, 0.0, "row3_col0 should be 0.0 (attend)");
        assert_eq!(row3_col1, 0.0, "row3_col1 should be 0.0 (attend)");
    }

    // --- Uncapped windowed mask (single-pass prefill > window, issue #401) --

    /// The uncapped windowed mask keeps the full `[size, size + offset]` key
    /// axis, so a single-pass prefill longer than the window leaves NO query
    /// row fully masked. Contrast `test_sliding_window_mask_values_when_capped`,
    /// where the capped `[4, 2]` mask strands rows 0 and 1 (all -inf) because it
    /// drops the keys those early rows still need.
    #[test]
    fn full_windowed_mask_over_window_has_no_all_masked_row() {
        // size=4 > window=2, offset=0. Capped variant would be [4, 2] with rows
        // 0 and 1 entirely -inf; the full variant is [4, 4] with every row valid.
        let mask = create_causal_mask_with_window_full(4, 0, Some(2));
        let shape = ffi::array_shape(&mask);
        assert_eq!(
            shape,
            vec![4, 4],
            "full windowed mask must NOT cap the key axis"
        );

        let at = |q: i32, k: i32| ffi::item_f32(&ffi::slice(&mask, &[q, k], &[q + 1, k + 1]));
        let blocked = |v: f32| v.is_infinite() && v < 0.0;

        // Diagonal always open: no query row is fully masked.
        for q in 0..4 {
            assert_eq!(at(q, q), 0.0, "row {q} must attend to itself");
        }
        // Window band (window = 2): query q attends to keys {q-1, q}.
        assert_eq!(at(1, 0), 0.0, "row1 attends to in-window key0");
        assert_eq!(at(2, 1), 0.0, "row2 attends to in-window key1");
        // Older-than-window keys are blocked.
        assert!(blocked(at(2, 0)), "key0 is older than row2's window");
        assert!(blocked(at(3, 1)), "key1 is older than row3's window");
        // Future keys stay causally blocked.
        assert!(blocked(at(0, 1)), "row0 future key blocked (causal)");
        assert!(blocked(at(1, 2)), "row1 future key blocked (causal)");
    }

    /// Within the window (`size + offset <= window`) the full and capped
    /// builders agree cell-for-cell, so the `<= window` prefill path is
    /// unchanged.
    #[test]
    fn full_windowed_mask_within_window_matches_capped() {
        // size=3, offset=2, window=8: total=5 <= 8 → capped builder does not cap.
        let full = create_causal_mask_with_window_full(3, 2, Some(8));
        let capped = create_causal_mask_with_window(3, 2, Some(8));
        assert_eq!(
            ffi::array_shape(&full),
            ffi::array_shape(&capped),
            "within-window shapes must match"
        );
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..3 {
            for k in 0..5 {
                let a = at(&full, q, k);
                let b = at(&capped, q, k);
                assert_eq!(
                    a.is_finite(),
                    b.is_finite(),
                    "cell ({q},{k}) finiteness must match: full={a}, capped={b}"
                );
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value mismatch: full={a}, capped={b}");
                }
            }
        }
    }

    // --- Sliding-window prefill mask selector (issue #408) ----------------

    /// A fresh single-pass prefill (`sliding_offset == 0`) longer than the
    /// window must select the uncapped full `[size, size]` mask, so no early
    /// query row is stranded with an all-`-inf` row.
    #[test]
    fn sliding_window_prefill_mask_selects_full_when_fresh_over_window() {
        let prefill = create_sliding_window_prefill_mask(4, 0, 2);
        assert_eq!(
            ffi::array_shape(&prefill),
            vec![4, 4],
            "fresh over-window prefill must use the full [size, size] mask"
        );
        let full = create_causal_mask_with_window_full(4, 0, Some(2));
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..4 {
            for k in 0..4 {
                let a = at(&prefill, q, k);
                let b = at(&full, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
            // No row is fully masked: each attends to at least its own key.
            assert_eq!(at(&prefill, q, q), 0.0, "row {q} must attend to itself");
        }
    }

    /// A rolled-over cache (`sliding_offset > 0`) keeps the clamped path,
    /// byte-identical to the per-model `create_causal_mask_with_window`
    /// construction it replaced (same `min((window - size).max(0))` clamp).
    #[test]
    fn sliding_window_prefill_mask_clamps_when_rolled_over() {
        let window = 4;
        let size = 2;
        let sliding_offset = 9; // well past the window → clamped path
        let prefill = create_sliding_window_prefill_mask(size, sliding_offset, window);
        let effective_offset = sliding_offset.min((window - size).max(0));
        let clamped = create_causal_mask_with_window(size, effective_offset, Some(window));
        assert_eq!(
            ffi::array_shape(&prefill),
            ffi::array_shape(&clamped),
            "rolled-over prefill must match the clamped mask shape"
        );
        let cols = ffi::array_shape(&prefill)[1];
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..size {
            for k in 0..cols {
                let a = at(&prefill, q, k);
                let b = at(&clamped, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
        }
    }

    /// A within-window fresh prefill (`size <= window`) selects the clamped
    /// (here non-capped) mask, identical to `create_causal_mask_with_window`.
    #[test]
    fn sliding_window_prefill_mask_within_window_matches_capped_builder() {
        let prefill = create_sliding_window_prefill_mask(3, 0, 8);
        let capped = create_causal_mask_with_window(3, 0, Some(8));
        assert_eq!(ffi::array_shape(&prefill), ffi::array_shape(&capped));
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..3 {
            for k in 0..3 {
                let a = at(&prefill, q, k);
                let b = at(&capped, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
        }
    }

    // --- Dense-cache sliding-window prefill mask selector (issue #413) -----

    /// The in-scope fix: a dense `KVCache` over-window prefill at a non-zero
    /// offset (`size > window && sliding_offset > 0`) must select the full
    /// `[size, size + offset]` mask so every retained key stays visible and no
    /// early query row is stranded all-`-inf` (the NaN/`<pad>` failure of the
    /// old clamped path). Equals `create_causal_mask_with_window_full` exactly.
    #[test]
    fn sliding_window_prefill_mask_dense_full_when_rolled_over_window() {
        let size = 4;
        let offset = 3;
        let window = 2;
        let prefill = create_sliding_window_prefill_mask_dense(size, offset, window);
        assert_eq!(
            ffi::array_shape(&prefill),
            vec![size, size + offset],
            "dense over-window prefill must use the full [size, size + offset] mask"
        );
        let full = create_causal_mask_with_window_full(size, offset, Some(window));
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..size {
            for k in 0..(size + offset) {
                let a = at(&prefill, q, k);
                let b = at(&full, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
            // No query row is fully masked: each attends to its own logical
            // key at column `q + offset`. This is the row the old clamped path
            // left all-`-inf` for logical position `< size - window`.
            assert_eq!(
                at(&prefill, q, q + offset),
                0.0,
                "row {q} must attend to its own key at column {}",
                q + offset
            );
        }
    }

    /// The adjacent latent fix: a within-window chunk (`size <= window`) whose
    /// offset pushes the total past the window (`size + sliding_offset >
    /// window`). The dense helper now returns the full `[size, size + offset]`
    /// mask instead of the clamped `[size, window]` mask, so the earliest query
    /// rows keep their OLDEST in-window keys instead of having them sliced away
    /// (coherent-but-wrong output under the old clamped path, not NaN/`<pad>`).
    #[test]
    fn sliding_window_prefill_mask_dense_full_when_within_window_over_total() {
        let size = 2;
        let offset = 9;
        let window = 4;
        let prefill = create_sliding_window_prefill_mask_dense(size, offset, window);
        assert_eq!(
            ffi::array_shape(&prefill),
            vec![size, size + offset],
            "within-window over-total prefill must use the full [size, size + offset] mask, not the clamped [size, window]"
        );
        let full = create_causal_mask_with_window_full(size, offset, Some(window));
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..size {
            for k in 0..(size + offset) {
                let a = at(&prefill, q, k);
                let b = at(&full, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
        }
        // Row 0 holds logical position offset=9; its full window is keys
        // [6..=9]. The old clamped path sliced K/V to the trailing `window`=4
        // keys and dropped key 6 (the oldest in-window key) for this row; the
        // full mask now attends to it.
        assert_eq!(
            at(&prefill, 0, 6),
            0.0,
            "row 0 must attend oldest in-window key 6"
        );
        assert_eq!(at(&prefill, 0, 9), 0.0, "row 0 causal boundary key 9");
        assert!(
            !at(&prefill, 0, 5).is_finite(),
            "row 0 key 5 is below the window"
        );
        assert!(
            !at(&prefill, 0, 10).is_finite(),
            "row 0 key 10 is in the future"
        );
        // Row 1 holds logical position 10; its window is keys [7..=10].
        assert_eq!(
            at(&prefill, 1, 7),
            0.0,
            "row 1 must attend oldest in-window key 7"
        );
        assert_eq!(at(&prefill, 1, 10), 0.0, "row 1 causal boundary key 10");
        assert!(
            !at(&prefill, 1, 6).is_finite(),
            "row 1 key 6 is below the window"
        );
    }

    /// A fresh dense over-window prefill (`sliding_offset == 0`) is identical
    /// to the rotating helper: both build the full `[size, size]` mask.
    #[test]
    fn sliding_window_prefill_mask_dense_matches_fresh_over_window() {
        let dense = create_sliding_window_prefill_mask_dense(4, 0, 2);
        let rotating = create_sliding_window_prefill_mask(4, 0, 2);
        assert_eq!(
            ffi::array_shape(&dense),
            ffi::array_shape(&rotating),
            "fresh over-window dense and rotating helpers must match shape"
        );
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for q in 0..4 {
            for k in 0..4 {
                let a = at(&dense, q, k);
                let b = at(&rotating, q, k);
                assert_eq!(a.is_finite(), b.is_finite(), "cell ({q},{k}) finiteness");
                if a.is_finite() {
                    assert_eq!(a, b, "cell ({q},{k}) value");
                }
            }
        }
    }

    /// For within-total prefills (`size + sliding_offset <= window`) the dense
    /// helper is byte-identical to the rotating helper: the rotating clamped
    /// builder takes its non-capped path there, so greedy output for the common
    /// within-window prefill is unchanged. (The `size + sliding_offset > window`
    /// cases now diverge on purpose and are covered by the dense-full tests.)
    #[test]
    fn sliding_window_prefill_mask_dense_within_window_byte_identical() {
        let at =
            |m: &MlxArray, q: i32, k: i32| ffi::item_f32(&ffi::slice(m, &[q, k], &[q + 1, k + 1]));
        for &(size, offset, window) in &[(3, 0, 8), (3, 2, 8)] {
            let dense = create_sliding_window_prefill_mask_dense(size, offset, window);
            let rotating = create_sliding_window_prefill_mask(size, offset, window);
            assert_eq!(
                ffi::array_shape(&dense),
                ffi::array_shape(&rotating),
                "({size},{offset},{window}) dense and rotating shapes must match"
            );
            let cols = ffi::array_shape(&dense)[1];
            for q in 0..size {
                for k in 0..cols {
                    let a = at(&dense, q, k);
                    let b = at(&rotating, q, k);
                    assert_eq!(
                        a.is_finite(),
                        b.is_finite(),
                        "({size},{offset},{window}) cell ({q},{k}) finiteness"
                    );
                    if a.is_finite() {
                        assert_eq!(a, b, "({size},{offset},{window}) cell ({q},{k}) value");
                    }
                }
            }
        }
    }

    // --- Windowed left-padding mask (ragged batched MTP prefill) ----------

    /// No-padding fast path must be byte-identical to the plain windowed mask.
    #[test]
    fn windowed_left_padding_mask_no_padding_matches_windowed() {
        // Non-capped regime: size=6, offset=0, window=8 (>= size). No padding.
        let ref_mask = create_causal_mask_with_window(6, 0, Some(8));
        let lp_mask = create_causal_mask_with_window_and_left_padding(6, 0, Some(8), &[0, 0]);
        let ref_shape = ffi::array_shape(&ref_mask);
        let lp_shape = ffi::array_shape(&lp_mask);
        assert_eq!(
            ref_shape,
            vec![6, 6],
            "non-capped windowed mask is [size, size]"
        );
        assert_eq!(
            ref_shape, lp_shape,
            "no-padding windowed left-padding mask must match plain windowed mask shape"
        );

        // Empty left_padding slice also collapses to the windowed mask.
        let lp_empty = create_causal_mask_with_window_and_left_padding(6, 0, Some(8), &[]);
        assert_eq!(ffi::array_shape(&lp_empty), ref_shape);

        // Spot-check a couple of cells are identical (additive 0 / -inf).
        for (q, k) in [(0_i32, 0_i32), (3, 0), (3, 3), (5, 1), (5, 5)] {
            let a = ffi::item_f32(&ffi::slice(&ref_mask, &[q, k], &[q + 1, k + 1]));
            let b = ffi::item_f32(&ffi::slice(&lp_mask, &[q, k], &[q + 1, k + 1]));
            assert_eq!(
                a.is_finite(),
                b.is_finite(),
                "cell ({q},{k}) finiteness must match: ref={a}, lp={b}"
            );
            if a.is_finite() {
                assert_eq!(a, b, "cell ({q},{k}) value mismatch: ref={a}, lp={b}");
            }
        }
    }

    /// With per-row left-padding the mask is `[B, 1, size, total_len]` and the
    /// padding columns are masked for the padded rows while remaining unmasked
    /// for the non-padded row.
    #[test]
    fn windowed_left_padding_mask_masks_padding_columns() {
        // B=2, size=4 (padded width), offset=0, window large enough not to
        // trigger sliding-window upper-bound masking (window >= size).
        // Row 0: left_padding=2 (real tokens at padded indices 2,3).
        // Row 1: left_padding=0 (all real).
        let mask = create_causal_mask_with_window_and_left_padding(4, 0, Some(4), &[2, 0]);
        let shape = ffi::array_shape(&mask);
        assert_eq!(
            shape,
            vec![2, 1, 4, 4],
            "padded mask must be [B,1,size,total]"
        );

        // Helper to read cell [b,0,q,k].
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };

        // Row 0 (left_padding=2): the first real query is at padded index q=2.
        // It must NOT attend to padding keys k=0,1, but MUST attend to k=2.
        assert!(
            cell(0, 2, 0).is_infinite() && cell(0, 2, 0) < 0.0,
            "row0 q=2 -> padding k=0 must be -inf"
        );
        assert!(
            cell(0, 2, 1).is_infinite() && cell(0, 2, 1) < 0.0,
            "row0 q=2 -> padding k=1 must be -inf"
        );
        assert_eq!(cell(0, 2, 2), 0.0, "row0 q=2 -> real k=2 must attend");
        // Causal upper bound: q=2 must NOT see future k=3.
        assert!(
            cell(0, 2, 3).is_infinite() && cell(0, 2, 3) < 0.0,
            "row0 q=2 -> future k=3 must be -inf (causal)"
        );

        // Row 1 (no padding): standard causal band, k=0 attended by q=0.
        assert_eq!(
            cell(1, 0, 0),
            0.0,
            "row1 q=0 -> k=0 must attend (no padding)"
        );
        assert!(
            cell(1, 0, 1).is_infinite() && cell(1, 0, 1) < 0.0,
            "row1 q=0 -> future k=1 must be -inf (causal)"
        );
    }

    /// In the non-capped prefill regime (`offset == 0`, `size <= window`) the
    /// sliding-window upper bound is inert, so for the real-token sub-block the
    /// windowed left-padding mask is byte-identical to the plain (non-windowed)
    /// left-padding mask. This is the exact invariant the ragged batched MTP
    /// prefill relies on: a short prefix prefill within the window sees no
    /// windowing effect, only causal + left-padding.
    #[test]
    fn windowed_left_padding_mask_matches_plain_left_padding_when_uncapped() {
        // size=5 <= window=8, offset=0 -> non-capped, upper bound inert.
        let windowed = create_causal_mask_with_window_and_left_padding(5, 0, Some(8), &[1, 0]);
        let plain = create_causal_mask_with_left_padding(5, 0, &[1, 0]);

        let wshape = ffi::array_shape(&windowed);
        let pshape = ffi::array_shape(&plain);
        assert_eq!(wshape, vec![2, 1, 5, 5]);
        assert_eq!(
            wshape, pshape,
            "non-capped windowed left-padding mask must share the plain shape"
        );

        let wcell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(
                &windowed,
                &[b, 0, q, k],
                &[b + 1, 1, q + 1, k + 1],
            ))
        };
        let pcell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(
                &plain,
                &[b, 0, q, k],
                &[b + 1, 1, q + 1, k + 1],
            ))
        };
        for b in 0..2 {
            for q in 0..5 {
                for k in 0..5 {
                    let w = wcell(b, q, k);
                    let p = pcell(b, q, k);
                    assert_eq!(
                        w.is_finite(),
                        p.is_finite(),
                        "cell ({b},{q},{k}) finiteness mismatch: windowed={w}, plain={p}"
                    );
                    if w.is_finite() {
                        assert_eq!(w, p, "cell ({b},{q},{k}) value mismatch");
                    }
                }
            }
        }
    }

    /// Ragged batched MTP **verify-round** left-padding regression (issue #161 /
    /// PR #162).
    ///
    /// Greedy parity for a variable-length B>1 burst requires that EVERY verify
    /// round mask each row's resident `[0, left_padding[r])` prompt-padding keys
    /// — not just the prefill. The padding K/V (token 0) is never evicted from
    /// the unbounded full-attention cache, so a verify query at a nonzero cache
    /// offset that attends those padding columns diverges from the row's
    /// standalone B=1 run, and the divergence scales with `left_padding[r]`
    /// (only the most-padded / shortest row breaks in the real-model gate).
    ///
    /// This pins the verify-frame mask the fixed `mask == None` forward path
    /// builds: `create_causal_mask_with_left_padding(width, offset, left_padding)`
    /// with `offset > 0` (cache already holds the padded prompt plus accepted
    /// tokens) and a LARGE per-row padding gap. For the most-padded row the
    /// leading `left_padding` key columns must be `-inf` and every real key
    /// (the columns the standalone B=1 run would expose) must be `0.0`.
    #[test]
    fn left_padding_mask_masks_padding_in_verify_round_with_large_gap() {
        // Verify round: width=2 query tokens, cache offset O=10 (e.g. padded
        // prompt max_len=8 + 2 accepted), so the key axis is O+width=12.
        // Row 0 (shortest / most padded): left_padding=6 — keys [0,6) are
        //   prompt padding, real keys live at [6, 12).
        // Row 1 (full length): left_padding=0 — every key is real.
        let width = 2_i32;
        let offset = 10_i32;
        let total = width + offset; // 12
        let mask = create_causal_mask_with_left_padding(width, offset, &[6, 0]);
        assert_eq!(
            ffi::array_shape(&mask),
            vec![2, 1, width, total],
            "verify left-padding mask must be [B, 1, width, offset+width]",
        );

        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };

        // Row 0: the leading 6 padding columns must be masked for every query.
        for q in 0..width {
            for k in 0..6 {
                let v = cell(0, q, k);
                assert!(
                    v.is_infinite() && v < 0.0,
                    "row0 (lp=6) padding key {k} for query {q} must be -inf, got {v}",
                );
            }
        }
        // Row 0: real keys [6, offset) are in the causal past of both queries
        // (query absolute positions are offset and offset+1) and must attend.
        for q in 0..width {
            for k in 6..offset {
                let v = cell(0, q, k);
                assert_eq!(
                    v, 0.0,
                    "row0 real key {k} for query {q} must attend (0.0), got {v}",
                );
            }
        }
        // Row 0: causal upper bound still holds for the appended query columns —
        // query q (absolute offset+q) must not see future key offset+q+1.
        assert!(
            cell(0, 0, offset + 1).is_infinite() && cell(0, 0, offset + 1) < 0.0,
            "row0 query 0 must not attend future key {}",
            offset + 1,
        );

        // Row 1 (no padding): the same real-key columns are attended and no
        // column is spuriously masked — the byte-identical baseline.
        for q in 0..width {
            for k in 0..=(offset + q) {
                let v = cell(1, q, k);
                assert_eq!(
                    v, 0.0,
                    "row1 (lp=0) key {k} for query {q} must attend (0.0), got {v}",
                );
            }
        }
    }

    /// Ragged batched MTP **buffered sliding-cache** verify regression (issue
    /// #161 / PR #162).
    ///
    /// The MTP rollback buffer keeps the sliding `RotatingKVCache` UNCOMPACTED
    /// far past the bare `sliding_window` (logical capacity `window +
    /// buffer_size`), so the resident prompt padding survives at columns
    /// `[0, lp)` even when `size + offset > window`. The verify forward must
    /// therefore use `create_causal_mask_with_window_and_left_padding` with the
    /// FULL key axis (`size + offset`) in this regime — enforcing BOTH the
    /// sliding-window band AND the `[0, lp)` padding filter. (The pre-fix gate
    /// fell back to a padding-UNAWARE plain windowed mask once `size + offset >
    /// window`, leaking the resident padding into the most-padded row.)
    ///
    /// This pins that, with `size + offset > window`, the most-padded row's
    /// leading padding is masked, the in-window real keys attend, and keys
    /// OLDER than the sliding window are excluded by the band.
    #[test]
    fn windowed_left_padding_mask_masks_padding_and_band_when_buffered_over_window() {
        // Single verify query (width=1) at cache offset 11 (buffered: 7-token
        // padded prompt + 4 accepted), window=8 -> size+offset=12 > window, but
        // the buffered cache returns the full 12-key axis (no compaction).
        // Row 0 (most padded): lp=5; row 1: lp=0.
        let size = 1_i32;
        let offset = 11_i32;
        let window = 8_i32;
        let total = size + offset; // 12
        let mask =
            create_causal_mask_with_window_and_left_padding(size, offset, Some(window), &[5, 0]);
        assert_eq!(
            ffi::array_shape(&mask),
            vec![2, 1, size, total],
            "buffered windowed left-padding mask must keep the FULL [B,1,size,size+offset] axis",
        );

        let cell = |b: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, 0, k], &[b + 1, 1, 1, k + 1]))
        };

        // The single query is at absolute position `offset` (= 11). Its
        // sliding-window admits keys [offset - window + 1, offset] = [4, 11].
        // Row 0 padding occupies [0, 5); the *intersection* of "real" and
        // "in-window" is [5, 11].
        for k in 0..5 {
            let v = cell(0, k);
            assert!(
                v.is_infinite() && v < 0.0,
                "row0 padding key {k} must be -inf, got {v}",
            );
        }
        // Key 4 is real but OUTSIDE the sliding window (4 < offset-window+1 = 4?
        // 11-8+1 = 4, so key 4 is the oldest in-window slot) -> attends. Keys
        // [4, 11] attend; here [5, 11] are real+in-window (key 4 is padding-free
        // only for row 1).
        for k in 5..=offset {
            let v = cell(0, k);
            assert_eq!(v, 0.0, "row0 real in-window key {k} must attend, got {v}");
        }

        // Row 1 (no padding): the sliding-window band excludes keys older than
        // `offset - window + 1` = 4, i.e. keys [0, 4) are -inf, [4, 11] attend.
        for k in 0..(offset - window + 1) {
            let v = cell(1, k);
            assert!(
                v.is_infinite() && v < 0.0,
                "row1 out-of-window key {k} must be -inf (sliding band), got {v}",
            );
        }
        for k in (offset - window + 1)..=offset {
            let v = cell(1, k);
            assert_eq!(v, 0.0, "row1 in-window key {k} must attend, got {v}");
        }
    }

    // --- NaN-safe diagonal rescue for fully-masked padding query rows (#163) --

    /// (a) With `lp = [2, 0]` at prefill offset 0, batch row 0's leading-padding
    /// query rows (`q < 2`) have an empty causal-AND-padding key set. The
    /// diagonal rescue keeps their self column attended, so EVERY query row of
    /// EVERY batch row has at least one attended (0.0) cell, so softmax is finite.
    #[test]
    fn left_padding_mask_every_query_row_has_an_attended_cell() {
        let n = 4_i32;
        let offset = 0_i32;
        let lp = [2_i32, 0];
        let mask = create_causal_mask_with_left_padding(n, offset, &lp);
        assert_eq!(ffi::array_shape(&mask), vec![2, 1, n, n]);
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        for b in 0..2 {
            for q in 0..n {
                let any_attend = (0..n).any(|k| cell(b, q, k) == 0.0);
                assert!(
                    any_attend,
                    "row b={b} q={q} must have at least one attended (0.0) cell"
                );
            }
        }
    }

    /// (b) Leading-padding query rows attend EXACTLY their self/diagonal column
    /// (`k == q + offset`) and nothing else. With `lp = [3, 0]` at offset 0,
    /// batch row 0's padding query rows are `q in {0, 1, 2}`.
    #[test]
    fn left_padding_mask_padding_query_rows_attend_exactly_self_column() {
        let n = 5_i32;
        let offset = 0_i32;
        let lp = [3_i32, 0];
        let mask = create_causal_mask_with_left_padding(n, offset, &lp);
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        for q in 0..3 {
            for k in 0..n {
                let v = cell(0, q, k);
                if k == q {
                    assert_eq!(v, 0.0, "padding query q={q} must attend self column k={k}");
                } else {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "padding query q={q} must mask non-self k={k}, got {v}"
                    );
                }
            }
        }
    }

    /// (c) Real query rows (`q + offset >= lp[r]`) are byte-identical to the
    /// pre-rescue causal-AND-padding construction: attend iff
    /// `lp[r] <= k <= q + offset`. The rescue touches only padding query rows.
    #[test]
    fn left_padding_mask_real_query_rows_match_causal_and_padding() {
        let n = 4_i32;
        let offset = 0_i32;
        let lp = [2_i32, 0];
        let mask = create_causal_mask_with_left_padding(n, offset, &lp);
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        for b in 0..2 {
            let lp_b = lp[b as usize];
            for q in 0..n {
                let q_abs = q + offset;
                if q_abs < lp_b {
                    continue; // padding query row, covered by (a)/(b)
                }
                for k in 0..n {
                    let expected_attend = k <= q_abs && k >= lp_b;
                    let v = cell(b, q, k);
                    if expected_attend {
                        assert_eq!(v, 0.0, "real row b={b} q={q} k={k} must attend");
                    } else {
                        assert!(
                            v.is_infinite() && v < 0.0,
                            "real row b={b} q={q} k={k} must be -inf, got {v}"
                        );
                    }
                }
            }
        }
    }

    /// (d) The windowed builder applies the identical rescue: with an ACTIVE
    /// window (`window = 3 < n = 5`) at offset 0 and `lp = [2, 0]`, (i) every
    /// query row has an attended cell, (ii) padding query rows attend exactly
    /// the (always in-window) self column, and (iii) real query rows match
    /// `lp[r] <= k <= q+offset` AND `k >= q+offset-window+1` (the band).
    #[test]
    fn windowed_left_padding_mask_nan_safe_rescue_trio() {
        let size = 5_i32;
        let offset = 0_i32;
        let window = 3_i32;
        let lp = [2_i32, 0];
        let mask = create_causal_mask_with_window_and_left_padding(size, offset, Some(window), &lp);
        assert_eq!(ffi::array_shape(&mask), vec![2, 1, size, size]);
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&mask, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        // (i) every query row has at least one attended cell.
        for b in 0..2 {
            for q in 0..size {
                assert!(
                    (0..size).any(|k| cell(b, q, k) == 0.0),
                    "windowed row b={b} q={q} must have an attended cell"
                );
            }
        }
        // (ii) padding query rows (batch row 0, q < lp[0] = 2) attend exactly k == q.
        for q in 0..2 {
            for k in 0..size {
                let v = cell(0, q, k);
                if k == q {
                    assert_eq!(
                        v, 0.0,
                        "windowed padding query q={q} must attend self k={k}"
                    );
                } else {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "windowed padding query q={q} must mask non-self k={k}, got {v}"
                    );
                }
            }
        }
        // (iii) real query rows match causal-AND-padding-AND-window.
        for b in 0..2 {
            let lp_b = lp[b as usize];
            for q in 0..size {
                let q_abs = q + offset;
                if q_abs < lp_b {
                    continue;
                }
                for k in 0..size {
                    let expected = k <= q_abs && k >= lp_b && k > q_abs - window;
                    let v = cell(b, q, k);
                    if expected {
                        assert_eq!(v, 0.0, "windowed real row b={b} q={q} k={k} must attend");
                    } else {
                        assert!(
                            v.is_infinite() && v < 0.0,
                            "windowed real row b={b} q={q} k={k} must be -inf, got {v}"
                        );
                    }
                }
            }
        }
    }

    // --- mask_stale_key_gap: per-row valid-length tail exclusion (#163) -------

    /// Gap columns `[ve[r], gap_end)` become −∞ for the right rows only; columns
    /// `>= gap_end` and `< ve[r]` are untouched; a 2-D base broadcasts to
    /// `[B, 1, n, K]`; and rows with `ve[r] == gap_end` are unchanged.
    #[test]
    fn mask_stale_key_gap_excludes_only_the_per_row_gap() {
        // Base is a 2-D all-attend mask [n=2, K=8]; the helper must broadcast it
        // over B and over the query axis. ve=[3, 6], gap_end=6.
        let n = 2_i32;
        let k_len = 8_i32;
        let base = ffi::zeros(&[n, k_len], dtype::FLOAT32);
        let ve = [3_i32, 6];
        let gap_end = 6_i32;
        let out = mask_stale_key_gap(&base, &ve, gap_end);
        assert_eq!(
            ffi::array_shape(&out),
            vec![2, 1, n, k_len],
            "2-D base must broadcast to [B,1,n,K]"
        );
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&out, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        for q in 0..n {
            // Row 0 (ve=3): only columns [3, 6) are -inf; [0,3) and [6,8) attend.
            for k in 0..k_len {
                let v = cell(0, q, k);
                if (3..6).contains(&k) {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "row0 gap col k={k} must be -inf, got {v}"
                    );
                } else {
                    assert_eq!(v, 0.0, "row0 non-gap col k={k} must stay attended");
                }
            }
            // Row 1 (ve=6 == gap_end): empty gap, every column unchanged.
            for k in 0..k_len {
                assert_eq!(
                    cell(1, q, k),
                    0.0,
                    "row1 (ve==gap_end) col k={k} must be unchanged"
                );
            }
        }
    }

    /// A 4-D base `[B,1,n,K]` is preserved cell-for-cell outside the gap, and the
    /// additive penalty composes with existing −∞ base cells without NaN.
    #[test]
    fn mask_stale_key_gap_preserves_4d_base_outside_gap() {
        // Base: a per-row left-padding causal mask at a nonzero offset (4-D
        // [B,1,n,K]) so some cells are already -inf. Then carve a stale gap.
        let n = 2_i32;
        let offset = 4_i32;
        let base = create_causal_mask_with_left_padding(n, offset, &[1, 0]);
        let k_len = n + offset; // 6
        assert_eq!(ffi::array_shape(&base), vec![2, 1, n, k_len]);
        let base_cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&base, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        // Snapshot the base before consuming it.
        let mut base_vals = [[[0.0_f32; 6]; 2]; 2];
        for b in 0..2 {
            for q in 0..n {
                for k in 0..k_len {
                    base_vals[b as usize][q as usize][k as usize] = base_cell(b, q, k);
                }
            }
        }
        let ve = [3_i32, 6]; // row 0 valid end 3 < gap_end 5; row 1 == gap_end.
        let gap_end = 5_i32;
        let out = mask_stale_key_gap(&base, &ve, gap_end);
        assert_eq!(ffi::array_shape(&out), vec![2, 1, n, k_len]);
        let cell = |b: i32, q: i32, k: i32| -> f32 {
            ffi::item_f32(&ffi::slice(&out, &[b, 0, q, k], &[b + 1, 1, q + 1, k + 1]))
        };
        for b in 0..2 {
            for q in 0..n {
                for k in 0..k_len {
                    let in_gap = b == 0 && (ve[0]..gap_end).contains(&k);
                    let v = cell(b, q, k);
                    if in_gap {
                        assert!(
                            v.is_infinite() && v < 0.0,
                            "row {b} gap col k={k} must be -inf, got {v}"
                        );
                    } else {
                        let base_v = base_vals[b as usize][q as usize][k as usize];
                        assert_eq!(
                            v.is_finite(),
                            base_v.is_finite(),
                            "row {b} q={q} k={k} finiteness must match base outside the gap"
                        );
                        if base_v.is_finite() {
                            assert_eq!(
                                v, base_v,
                                "row {b} q={q} k={k} must equal base outside gap"
                            );
                        }
                    }
                }
            }
        }
    }
}
