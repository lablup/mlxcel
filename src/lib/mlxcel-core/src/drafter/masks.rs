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

//! Bidirectional attention-mask helpers for the Gemma 4 MTP drafter.
//!
//! The drafter attends from `query_len` query positions (placed just past
//! the end of the target's KV cache) over the target's last-layer K/V —
//! bidirectionally for both full-attention and sliding-window layers. The
//! masks here mirror the upstream Python reference at
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/gemma4_assistant/masks.py.
//!
//! ## Scope
//!
//! - [`bidirectional_full_mask`] — full-attention bias (`None` in the
//!   unbatched no-padding case, additive bias when batched rows have
//!   different `kv_valid_len`).
//! - [`bidirectional_swa_mask`] — sliding-window bias (`None` when the
//!   window already covers every query, additive bias otherwise).
//! - [`make_drafter_masks`] — dispatcher keyed by [`LayerType`].
//! - [`normalize_batched_shared_kv_states`] — left-rolls per-row K/V so the
//!   drafter sees the prefix-valid layout the unbatched path already uses.
//!
//! ## Dtype invariant
//!
//! All produced masks carry the requested dtype (typically `bf16` or `f16`
//! — the drafter's native dtype). No silent f32 promotion: the `0` /
//! `-inf` entries are materialised via [`crate::ffi::full_f32`] with the
//! caller-provided dtype, which is the same path the existing
//! [`crate::utils::create_causal_mask`] family uses.
//!
//! ## Upstream parity
//!
//! Every public function in this module ports a single upstream Python
//! function with identical numerical semantics. Test fixtures pin the
//! exact upstream behaviour for the matrix in the issue acceptance
//! criteria.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Layer type
// ---------------------------------------------------------------------------

/// Attention-layer flavour the drafter's K/V comes from.
///
/// Mirrors upstream's string-keyed dictionary (`"full_attention"` /
/// `"sliding_attention"`), with a Rust enum for compile-time exhaustiveness.
/// The enum is `#[non_exhaustive]` so future Gemma variants that add a new
/// attention type do not break downstream `match` exhaustiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LayerType {
    /// Full bidirectional attention layer (the drafter's `mtp.layers.*.attn`
    /// last full-attention slot).
    FullAttention,
    /// Sliding-window attention layer (the drafter's last SWA slot).
    SlidingWindowAttention,
}

impl LayerType {
    /// Canonical string key used by upstream `make_drafter_masks`.
    pub const fn as_str(self) -> &'static str {
        match self {
            LayerType::FullAttention => "full_attention",
            LayerType::SlidingWindowAttention => "sliding_attention",
        }
    }
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Per-row scalar metadata
// ---------------------------------------------------------------------------

/// Per-row batch metadata: either a single scalar broadcast across all rows
/// (the B=1 fast path) or a B-length `int32` `MlxArray` carrying one value
/// per row.
///
/// Mirrors upstream's `Union[int, mx.array]` typing on `kv_valid_len`,
/// `query_offset`, and `left_padding`. Carrying both shapes through one
/// enum lets the call-sites stay branch-free at the public API.
pub enum BatchScalar<'a> {
    /// Single scalar value (B=1 path; the upstream `int` arm).
    Scalar(i32),
    /// Per-row vector of shape `[B]`, dtype `int32`. Borrowed because the
    /// caller (drafter forward) owns the tensor and we do not want to copy
    /// it across the FFI boundary just to read it.
    PerRow(&'a MlxArray),
}

// Manual Debug impl: `MlxArray` is an opaque cxx FFI type that does not
// derive `Debug`, so we render the per-row arm as `PerRow { len }` from
// `array_shape` rather than printing the (GPU-resident) buffer body.
impl<'a> std::fmt::Debug for BatchScalar<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchScalar::Scalar(v) => f.debug_tuple("Scalar").field(v).finish(),
            BatchScalar::PerRow(arr) => {
                let shape = ffi::array_shape(arr);
                f.debug_struct("PerRow").field("shape", &shape).finish()
            }
        }
    }
}

impl<'a> BatchScalar<'a> {
    /// Convenience for B=1 fast-path callers and tests.
    pub fn scalar(v: i32) -> Self {
        BatchScalar::Scalar(v)
    }

    /// Convenience for batched callers.
    pub fn per_row(v: &'a MlxArray) -> Self {
        BatchScalar::PerRow(v)
    }

    /// Whether this is the cheap scalar arm (used to short-circuit the
    /// "no mask needed" checks for the B=1 path).
    fn is_scalar(&self) -> bool {
        matches!(self, BatchScalar::Scalar(_))
    }
}

// ---------------------------------------------------------------------------
// Bidirectional full-attention mask
// ---------------------------------------------------------------------------

/// Build the bidirectional full-attention bias.
///
/// Ports upstream `bidirectional_full_mask`. The unbatched no-padding case
/// is a no-op (returns `None`) and the SDPA path handles it without any
/// allocated mask tensor.
///
/// When a batched call has rows with different `kv_valid_len`, each row
/// must mask keys beyond its own valid prefix. The returned bias has shape
/// `[B, 1, 1, kv_len]` (broadcasts over heads and queries) with `0` at
/// valid positions and `-inf` (in the requested dtype) at padded positions.
///
/// - `query_len`: unused for the full mask (kept for API symmetry with
///   [`bidirectional_swa_mask`] — upstream takes the same parameter for
///   the same reason).
/// - `kv_len`: length of the padded K/V axis.
/// - `kv_valid_len`: either `None` (no padding; mask is `None`), or per-row
///   valid prefix length.
/// - `dtype`: target dtype for the produced mask. Pass the drafter's
///   native dtype (`bf16` / `f16`) — no f32 promotion.
pub fn bidirectional_full_mask(
    _query_len: i32,
    kv_len: i32,
    kv_valid_len: Option<&BatchScalar<'_>>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    let key_offset = BatchScalar::Scalar(0);
    bidirectional_full_mask_with_key_offset(_query_len, kv_len, kv_valid_len, &key_offset, dtype)
}

/// Build the bidirectional full-attention bias with an absolute K/V
/// `key_offset`.
///
/// Latest upstream Gemma 4 MTP passes `key_offset = max(kv_valid_len -
/// kv_len, 0)` for full-attention caches so a sliced post-rollback K/V slab
/// whose local axis is `[0, kv_len)` still masks against the correct absolute
/// key positions. [`bidirectional_full_mask`] is kept as the legacy
/// `key_offset = 0` wrapper for existing tests and callers.
pub fn bidirectional_full_mask_with_key_offset(
    _query_len: i32,
    kv_len: i32,
    kv_valid_len: Option<&BatchScalar<'_>>,
    key_offset: &BatchScalar<'_>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    let valid = kv_valid_len?;

    match (valid, key_offset) {
        (BatchScalar::Scalar(v), BatchScalar::Scalar(ko)) => {
            // B=1 fast path. Skip when the scalar valid length already
            // covers the buffer.
            if *ko + kv_len <= *v {
                return None;
            }

            // Build bias of shape [kv_len], 0 at valid positions and -inf
            // elsewhere, then broadcast to [1, 1, 1, kv_len].
            let k_idx = ffi::arange_i32(*ko, *ko + kv_len, 1);
            let valid_arr = ffi::from_slice_i32(&[*v], &[1]);
            let inside = ffi::less(&k_idx, &valid_arr); // bool [kv_len]
            let bias = build_bias_from_bool(&inside, dtype);
            // Reshape from [kv_len] to [1, 1, 1, kv_len].
            Some(ffi::reshape(&bias, &[1, 1, 1, kv_len]))
        }
        (BatchScalar::Scalar(v), BatchScalar::PerRow(key_offset_per_row)) => {
            let batch = ffi::array_shape(key_offset_per_row)[0];
            let ko_2d = ffi::reshape(key_offset_per_row, &[batch, 1]);
            let k_range = ffi::reshape(&ffi::arange_i32(0, kv_len, 1), &[1, kv_len]);
            let k_idx = ffi::add(&ko_2d, &k_range);
            let valid_arr = ffi::from_slice_i32(&[*v], &[1]);
            let inside = ffi::less(&k_idx, &valid_arr); // bool [B, kv_len]
            let bias_2d = build_bias_from_bool(&inside, dtype);
            Some(ffi::reshape(&bias_2d, &[batch, 1, 1, kv_len]))
        }
        (BatchScalar::PerRow(valid_per_row), BatchScalar::Scalar(ko)) => {
            // Per-row valid lengths: build bias of shape [B, 1, 1, kv_len].
            let batch = ffi::array_shape(valid_per_row)[0];

            // k_idx: shape [1, kv_len] (broadcasts over batch).
            let k_idx_1d = ffi::arange_i32(*ko, *ko + kv_len, 1);
            let k_idx = ffi::reshape(&k_idx_1d, &[1, kv_len]);

            // valid_per_row reshaped to [B, 1] for broadcast.
            let valid_2d = ffi::reshape(valid_per_row, &[batch, 1]);

            let inside = ffi::less(&k_idx, &valid_2d); // bool [B, kv_len]
            let bias_2d = build_bias_from_bool(&inside, dtype);
            // Reshape from [B, kv_len] to [B, 1, 1, kv_len].
            Some(ffi::reshape(&bias_2d, &[batch, 1, 1, kv_len]))
        }
        (BatchScalar::PerRow(valid_per_row), BatchScalar::PerRow(key_offset_per_row)) => {
            let batch = ffi::array_shape(valid_per_row)[0];
            let ko_2d = ffi::reshape(key_offset_per_row, &[batch, 1]);
            let k_range = ffi::reshape(&ffi::arange_i32(0, kv_len, 1), &[1, kv_len]);
            let k_idx = ffi::add(&ko_2d, &k_range); // [B, kv_len]
            let valid_2d = ffi::reshape(valid_per_row, &[batch, 1]);
            let inside = ffi::less(&k_idx, &valid_2d);
            let bias_2d = build_bias_from_bool(&inside, dtype);
            Some(ffi::reshape(&bias_2d, &[batch, 1, 1, kv_len]))
        }
    }
}

// ---------------------------------------------------------------------------
// Bidirectional sliding-window mask
// ---------------------------------------------------------------------------

/// Build the bidirectional sliding-window bias.
///
/// Ports upstream `bidirectional_swa_mask`. Returns `None` when the SWA
/// window already covers every query position (the "no mask needed" fast
/// path — a real performance win because SDPA can then skip the mask
/// add entirely).
///
/// When the window does not cover everything, produces an additive bias
/// where `bias[..., q, k] = 0` iff `|q_logical - k_logical| < window`,
/// else `-inf` (in the requested dtype). When `kv_valid_len` is also
/// provided, positions with `k >= kv_valid_len` are additionally masked.
///
/// - `query_len`: number of query rows (`L` upstream).
/// - `query_offset`: absolute KV-space offset of the first query. Either
///   a scalar (B=1 / single offset shared across rows) or a `[B]` per-row
///   `int32` vector.
/// - `kv_len`: length of the padded K/V axis.
/// - `window`: sliding-window half-extent (exclusive on both sides, like
///   upstream).
/// - `kv_valid_len`: optional per-row valid prefix. When `None`, no extra
///   tail-masking is applied.
/// - `dtype`: target dtype for the produced mask.
pub fn bidirectional_swa_mask(
    query_len: i32,
    query_offset: &BatchScalar<'_>,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<&BatchScalar<'_>>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    let key_offset = BatchScalar::Scalar(0);
    bidirectional_swa_mask_with_key_offset(
        query_len,
        query_offset,
        kv_len,
        window,
        kv_valid_len,
        &key_offset,
        dtype,
    )
}

/// Build the bidirectional sliding-window bias with an absolute K/V
/// `key_offset`.
///
/// The latest upstream MTP mask logic maps rotating-cache query/valid
/// positions onto the local K/V window before calling this helper, but keeps
/// this explicit `key_offset` parameter for full absolute-distance parity.
/// [`bidirectional_swa_mask`] is the legacy `key_offset = 0` wrapper.
pub fn bidirectional_swa_mask_with_key_offset(
    query_len: i32,
    query_offset: &BatchScalar<'_>,
    kv_len: i32,
    window: i32,
    kv_valid_len: Option<&BatchScalar<'_>>,
    key_offset: &BatchScalar<'_>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    // ----- "no mask needed" fast path -------------------------------------
    //
    // The window already covers every query in both directions when:
    //   - query_offset is scalar,
    //   - kv_valid_len is None or scalar,
    //   - key_offset is scalar,
    //   - kv_len <= window,
    //   - the first key is less than `window` behind the first query,
    //   - the last key is less than `window` ahead of the last query.
    let kv_valid_is_scalar = kv_valid_len.map(BatchScalar::is_scalar).unwrap_or(true);
    if let (BatchScalar::Scalar(qo), true, BatchScalar::Scalar(ko)) =
        (query_offset, kv_valid_is_scalar, key_offset)
        && kv_len <= window
        && *qo - *ko < window
        && *ko + kv_len - (*qo + query_len) < window
    {
        return None;
    }

    // ----- materialisation -------------------------------------------------
    //
    // Build q_idx and k_idx so dist = q_idx - k_idx broadcasts to the
    // appropriate mask shape (either [query_len, kv_len] for scalar
    // offsets or [B, query_len, kv_len] for per-row offsets).
    match (query_offset, key_offset) {
        (BatchScalar::Scalar(qo), BatchScalar::Scalar(ko)) => {
            // q_idx: shape [query_len, 1].
            let q_idx_1d = ffi::arange_i32(*qo, *qo + query_len, 1);
            let q_idx = ffi::reshape(&q_idx_1d, &[query_len, 1]);

            // k_idx: shape [1, kv_len].
            let k_idx_1d = ffi::arange_i32(*ko, *ko + kv_len, 1);
            let k_idx = ffi::reshape(&k_idx_1d, &[1, kv_len]);

            // dist = q_idx - k_idx, broadcast to [query_len, kv_len].
            let dist = ffi::subtract(&q_idx, &k_idx);
            let inside = swa_inside_from_dist(&dist, window);

            // Optional kv_valid_len tail-mask.
            let inside = apply_kv_valid_tail(inside, &k_idx, kv_valid_len);
            let bias_2d = build_bias_from_bool(&inside, dtype);
            // Reshape from [query_len, kv_len] to [1, 1, query_len, kv_len].
            Some(ffi::reshape(&bias_2d, &[1, 1, query_len, kv_len]))
        }
        (BatchScalar::Scalar(qo), BatchScalar::PerRow(ko_vec)) => {
            let batch = ffi::array_shape(ko_vec)[0];
            let qo_vec = scalar_batch_vector(*qo, batch);
            bidirectional_swa_mask_with_key_offset(
                query_len,
                &BatchScalar::PerRow(&qo_vec),
                kv_len,
                window,
                kv_valid_len,
                key_offset,
                dtype,
            )
        }
        (BatchScalar::PerRow(qo_vec), key_offset) => {
            let batch = ffi::array_shape(qo_vec)[0];

            // q_idx[b, q] = qo_vec[b] + q. Shape [B, query_len].
            let qo_col = ffi::reshape(qo_vec, &[batch, 1]); // [B, 1]
            let q_range_1d = ffi::arange_i32(0, query_len, 1); // [query_len]
            let q_range = ffi::reshape(&q_range_1d, &[1, query_len]); // [1, query_len]
            let q_idx_2d = ffi::add(&qo_col, &q_range); // [B, query_len]
            // Reshape for the [B, query_len, kv_len] computation:
            //   q_idx -> [B, query_len, 1], k_idx -> [1, 1, kv_len].
            let q_idx = ffi::reshape(&q_idx_2d, &[batch, query_len, 1]);

            let k_idx = match key_offset {
                BatchScalar::Scalar(ko) => {
                    let k_idx_1d = ffi::arange_i32(*ko, *ko + kv_len, 1);
                    ffi::reshape(&k_idx_1d, &[1, 1, kv_len])
                }
                BatchScalar::PerRow(ko_vec) => {
                    let ko_3d = ffi::reshape(ko_vec, &[batch, 1, 1]);
                    let k_range = ffi::reshape(&ffi::arange_i32(0, kv_len, 1), &[1, 1, kv_len]);
                    ffi::add(&ko_3d, &k_range)
                }
            };

            let dist = ffi::subtract(&q_idx, &k_idx); // [B, query_len, kv_len]
            let inside = swa_inside_from_dist(&dist, window);

            // Per-row kv_valid_len tail-mask.
            let inside = apply_kv_valid_tail_batched(inside, &k_idx, kv_valid_len, batch);
            let bias_3d = build_bias_from_bool(&inside, dtype); // [B, query_len, kv_len]
            // Reshape to [B, 1, query_len, kv_len].
            Some(ffi::reshape(&bias_3d, &[batch, 1, query_len, kv_len]))
        }
    }
}

fn scalar_batch_vector(value: i32, batch: i32) -> UniquePtr<MlxArray> {
    let values = vec![value; batch as usize];
    ffi::from_slice_i32(&values, &[batch])
}

/// `inside = (dist > -window) & (dist < window)` — the bidirectional
/// window condition that upstream uses for the SWA mask.
fn swa_inside_from_dist(dist: &MlxArray, window: i32) -> UniquePtr<MlxArray> {
    let neg_window = ffi::from_slice_i32(&[-window], &[1]);
    let pos_window = ffi::from_slice_i32(&[window], &[1]);
    let lower = ffi::greater(dist, &neg_window);
    let upper = ffi::less(dist, &pos_window);
    ffi::logical_and(&lower, &upper)
}

/// AND the `inside` mask with `k_idx < kv_valid_len` when `kv_valid_len`
/// is supplied. Scalar arm.
fn apply_kv_valid_tail(
    inside: UniquePtr<MlxArray>,
    k_idx: &MlxArray,
    kv_valid_len: Option<&BatchScalar<'_>>,
) -> UniquePtr<MlxArray> {
    let Some(valid) = kv_valid_len else {
        return inside;
    };
    match valid {
        BatchScalar::Scalar(v) => {
            let v_arr = ffi::from_slice_i32(&[*v], &[1]);
            let in_valid = ffi::less(k_idx, &v_arr);
            ffi::logical_and(&inside, &in_valid)
        }
        BatchScalar::PerRow(_) => {
            // The caller paired a per-row kv_valid_len with a scalar
            // query_offset — upstream allows this but produces a [B, ...]
            // mask. Defer to the batched helper, which lifts the inside
            // bool to a [1, ...] tensor first.
            inside
        }
    }
}

/// Same as [`apply_kv_valid_tail`] but for the per-row `query_offset` arm
/// where `inside` already has shape `[B, query_len, kv_len]`.
fn apply_kv_valid_tail_batched(
    inside: UniquePtr<MlxArray>,
    k_idx: &MlxArray,
    kv_valid_len: Option<&BatchScalar<'_>>,
    batch: i32,
) -> UniquePtr<MlxArray> {
    let Some(valid) = kv_valid_len else {
        return inside;
    };
    match valid {
        BatchScalar::Scalar(v) => {
            // Broadcast the scalar against `k_idx` (shape `[1, 1, kv_len]`)
            // to produce a `[1, 1, kv_len]` bool; AND it with the batched
            // inside.
            let v_arr = ffi::from_slice_i32(&[*v], &[1]);
            let in_valid = ffi::less(k_idx, &v_arr);
            ffi::logical_and(&inside, &in_valid)
        }
        BatchScalar::PerRow(valid_per_row) => {
            // Shape `valid_per_row` to `[B, 1, 1]` so `k_idx < valid` has
            // shape `[B, 1, kv_len]`, broadcasting cleanly over the
            // `query_len` axis.
            let valid_3d = ffi::reshape(valid_per_row, &[batch, 1, 1]);
            let in_valid = ffi::less(k_idx, &valid_3d);
            ffi::logical_and(&inside, &in_valid)
        }
    }
}

/// `bias = where(bool_mask, 0, -inf)` materialised in the requested dtype.
///
/// This is the only allocation of mask-shaped tensors in this module. The
/// `dtype` passed here is the drafter's native dtype (bf16 / f16) and is
/// load-bearing for the Apple Silicon precision invariant: silently
/// promoting to f32 would cost a `astype` per layer per draft step.
fn build_bias_from_bool(bool_mask: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(bool_mask);
    let zero = ffi::full_f32(&shape, 0.0, dtype);
    let neg_inf = ffi::full_f32(&shape, f32::NEG_INFINITY, dtype);
    ffi::where_cond(bool_mask, &zero, &neg_inf)
}

// ---------------------------------------------------------------------------
// make_drafter_masks
// ---------------------------------------------------------------------------

/// Build the per-layer-type drafter mask map.
///
/// Ports upstream `make_drafter_masks`. The returned map mirrors the
/// `shared_kv_states` input keying so the drafter forward can look up
/// `masks.get(&layer_type)` without juggling string keys.
///
/// - `shared_kv_states`: per-layer-type tuple of `(keys, values)` — only
///   the keys' shape is consulted here (to read `kv_len = K.shape[-2]`),
///   the buffers themselves are not touched.
/// - `query_len`: forwarded to the per-mask helpers.
/// - `query_offset`: forwarded. The wrapper supplies `kv_valid_len =
///   query_offset + 1` to the `_with_valid_len` path so the documented
///   invariant `query_offset == kv_valid_len - 1` holds and the SWA query
///   position derives as `query_offset` (issue #163). Upstream's note
///   `kv_valid_len = query_offset` refers to the position anchor, not the
///   valid-length count, which is one greater.
/// - `sliding_window`: SWA window size.
/// - `dtype`: target mask dtype.
///
/// `None` entries in the returned map carry the upstream "no mask needed"
/// signal — callers should skip the mask-add entirely on those layers.
pub fn make_drafter_masks(
    shared_kv_states: &HashMap<LayerType, (&MlxArray, &MlxArray)>,
    query_len: i32,
    query_offset: &BatchScalar<'_>,
    sliding_window: i32,
    dtype: i32,
) -> HashMap<LayerType, Option<UniquePtr<MlxArray>>> {
    // Honour the documented invariant `query_offset == kv_valid_len - 1`
    // (issue #163). The `_with_valid_len` path falls back to `effective_valid =
    // query_offset` when `kv_valid_len` is `None`, which makes
    // `make_swa_mask_with_local_offsets` derive the SWA query position as
    // `effective_valid - 1 = query_offset - 1`, one short of the invariant.
    // Passing `kv_valid_len = query_offset + 1` recovers the SWA query position
    // `query_offset`. Declare the owned per-row tensor BEFORE the `BatchScalar`
    // that borrows it so it outlives the borrow.
    let per_row_plus_one;
    let kv_valid_len = match query_offset {
        BatchScalar::Scalar(v) => BatchScalar::Scalar(v + 1),
        BatchScalar::PerRow(arr) => {
            let one = ffi::from_slice_i32(&[1], &[1]);
            per_row_plus_one = ffi::add(arr, &one);
            BatchScalar::PerRow(&per_row_plus_one)
        }
    };
    make_drafter_masks_with_valid_len(
        shared_kv_states,
        query_len,
        query_offset,
        sliding_window,
        dtype,
        Some(&kv_valid_len),
    )
}

/// Build the per-layer-type drafter mask map with an explicit
/// `kv_valid_len`.
///
/// Upstream Gemma 4 MTP now distinguishes the drafter query/RoPE anchor
/// (`position = kv_valid_len - 1`) from the valid target-cache length used
/// to mask shared K/V (`kv_valid_len`). This helper ports that split while
/// keeping [`make_drafter_masks`] as the backwards-compatible
/// `kv_valid_len = query_offset` wrapper.
pub fn make_drafter_masks_with_valid_len(
    shared_kv_states: &HashMap<LayerType, (&MlxArray, &MlxArray)>,
    query_len: i32,
    query_offset: &BatchScalar<'_>,
    sliding_window: i32,
    dtype: i32,
    kv_valid_len: Option<&BatchScalar<'_>>,
) -> HashMap<LayerType, Option<UniquePtr<MlxArray>>> {
    let mut masks: HashMap<LayerType, Option<UniquePtr<MlxArray>>> =
        HashMap::with_capacity(shared_kv_states.len());

    for (&layer_type, (keys, _values)) in shared_kv_states {
        let kv_len = kv_len_of(keys);
        let effective_valid = match kv_valid_len {
            Some(v) => match v {
                BatchScalar::Scalar(s) => BatchScalar::Scalar(*s),
                BatchScalar::PerRow(arr) => BatchScalar::PerRow(arr),
            },
            None => match query_offset {
                BatchScalar::Scalar(v) => BatchScalar::Scalar(*v),
                BatchScalar::PerRow(arr) => BatchScalar::PerRow(arr),
            },
        };
        let mask = match layer_type {
            LayerType::SlidingWindowAttention => make_swa_mask_with_local_offsets(
                query_len,
                query_offset,
                kv_len,
                sliding_window,
                &effective_valid,
                dtype,
            ),
            LayerType::FullAttention => {
                make_full_mask_with_absolute_key_offset(query_len, kv_len, &effective_valid, dtype)
            }
        };
        masks.insert(layer_type, mask);
    }

    masks
}

fn make_full_mask_with_absolute_key_offset(
    query_len: i32,
    kv_len: i32,
    kv_valid_len: &BatchScalar<'_>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    match kv_valid_len {
        BatchScalar::Scalar(v) => {
            let key_offset = BatchScalar::Scalar((*v - kv_len).max(0));
            bidirectional_full_mask_with_key_offset(
                query_len,
                kv_len,
                Some(kv_valid_len),
                &key_offset,
                dtype,
            )
        }
        BatchScalar::PerRow(valid_per_row) => {
            let kv_len_arr = ffi::from_slice_i32(&[kv_len], &[1]);
            let diff = ffi::subtract(valid_per_row, &kv_len_arr);
            let zero = ffi::from_slice_i32(&[0], &[1]);
            let key_offset_arr = ffi::maximum(&diff, &zero);
            let key_offset = BatchScalar::PerRow(&key_offset_arr);
            bidirectional_full_mask_with_key_offset(
                query_len,
                kv_len,
                Some(kv_valid_len),
                &key_offset,
                dtype,
            )
        }
    }
}

fn make_swa_mask_with_local_offsets(
    query_len: i32,
    query_offset: &BatchScalar<'_>,
    kv_len: i32,
    sliding_window: i32,
    kv_valid_len: &BatchScalar<'_>,
    dtype: i32,
) -> Option<UniquePtr<MlxArray>> {
    // The shared K/V handed to the drafter is normalized so every row's valid
    // keys occupy `[0, kv_valid_len[r])` in the **prefix-valid** frame (see
    // `normalize_batched_shared_kv_states`). The SWA window distance must
    // therefore be measured in that same frame: the bonus query is logically
    // the last valid token, i.e. at position `kv_valid_len - 1`. The caller's
    // `query_offset` is the drafter's **RoPE anchor** in the **padded** frame
    // (`bonus_position = left_padding + kv_valid_len - 1`); using it directly
    // for the SWA distance would inject a spurious `+left_padding` shift that
    // over-masks the front keys of a left-padded (ragged batched MTP) row by
    // exactly `left_padding`, breaking greedy parity for the most-padded row
    // while leaving the others untouched.
    //
    // For every non-padded path the established invariant is `query_offset ==
    // kv_valid_len - 1` (B = 1 decode, equal-length batched, rotating-cache
    // slice), so deriving the SWA query position from `kv_valid_len - 1` is
    // byte-identical there and only diverges (correctly) when `left_padding >
    // 0`. `key_offset` stays 0 because the normalized keys start at logical 0.
    //
    // `query_offset` is now unused for the SWA distance; it is retained in the
    // signature for symmetry with the full-attention path and potential future
    // callers that pass already-valid-frame offsets.
    let _ = query_offset;
    let key_offset = BatchScalar::Scalar(0);
    match kv_valid_len {
        BatchScalar::Scalar(v) => {
            // Valid-frame query position: clamp `kv_valid_len - 1` into
            // `[0, kv_len]` (the local K/V axis). The clamp matches the prior
            // `min(.., kv_len)` behaviour and additionally floors at 0 so a
            // degenerate `kv_valid_len == 0` cannot produce a negative offset.
            let local_query = BatchScalar::Scalar((*v - 1).clamp(0, kv_len));
            let local_valid = BatchScalar::Scalar((*v).min(kv_len));
            bidirectional_swa_mask_with_key_offset(
                query_len,
                &local_query,
                kv_len,
                sliding_window,
                Some(&local_valid),
                &key_offset,
                dtype,
            )
        }
        BatchScalar::PerRow(v_arr) => {
            // local_query[r] = clamp(kv_valid_len[r] - 1, 0, kv_len).
            let local_query_arr = local_query_offset_array(v_arr, kv_len);
            let local_valid_arr = local_window_offset_array(v_arr, kv_len);
            let local_query = BatchScalar::PerRow(&local_query_arr);
            let local_valid = BatchScalar::PerRow(&local_valid_arr);
            bidirectional_swa_mask_with_key_offset(
                query_len,
                &local_query,
                kv_len,
                sliding_window,
                Some(&local_valid),
                &key_offset,
                dtype,
            )
        }
    }
}

/// Per-row valid-frame SWA query offset: `clamp(kv_valid_len[r] - 1, 0, kv_len)`.
///
/// The bonus query is the last valid token, so its position relative to the
/// normalized `[0, kv_valid_len)` key prefix is `kv_valid_len - 1`. The clamp
/// floors at 0 (degenerate empty prefix) and caps at the local K/V axis
/// length, matching [`local_window_offset_array`]'s upper bound.
fn local_query_offset_array(kv_valid_len: &MlxArray, kv_len: i32) -> UniquePtr<MlxArray> {
    let one = ffi::from_slice_i32(&[1], &[1]);
    let minus_one = ffi::subtract(kv_valid_len, &one);
    let zero = ffi::from_slice_i32(&[0], &[1]);
    let kv_len_arr = ffi::from_slice_i32(&[kv_len], &[1]);
    let floored = ffi::maximum(&minus_one, &zero);
    ffi::minimum(&floored, &kv_len_arr)
}

fn local_window_offset_array(value: &MlxArray, kv_len: i32) -> UniquePtr<MlxArray> {
    let kv_len_arr = ffi::from_slice_i32(&[kv_len], &[1]);
    ffi::minimum(value, &kv_len_arr)
}

/// Read `K.shape[-2]` — the K/V sequence-length axis. K is always 4-D in
/// the drafter cross-attention path (`[B, n_kv_heads, kv_len, head_dim]`).
fn kv_len_of(keys: &MlxArray) -> i32 {
    let shape = ffi::array_shape(keys);
    debug_assert!(
        shape.len() >= 2,
        "shared K/V key tensor must be at least 2-D, got shape {shape:?}",
    );
    shape[shape.len() - 2]
}

// ---------------------------------------------------------------------------
// normalize_batched_shared_kv_states
// ---------------------------------------------------------------------------

/// Normalize batched shared K/V into the drafter's prefix-valid layout.
///
/// Ports upstream `normalize_batched_shared_kv_states`. The target's
/// shared K/V may be left-padded (initial batched prefill) and may carry
/// per-row rollback slack in the tail after mixed speculative accepts.
/// The Gemma drafter expects the simpler invariant used by the unbatched
/// path: each row's real keys occupy `[0, kv_valid_len)` and any invalid
/// slots are zeroed in the tail.
///
/// For each `(K, V)` pair in `shared_kv_states`:
/// 1. Roll along axis 2 (seq_len) by `-left_padding[b]` per row so the
///    valid prefix moves to indices `[0, kv_valid_len[b])`.
/// 2. Zero out the tail `[kv_valid_len[b], kv_len)` — defensive, since
///    mid-pipeline ops may have dirty tails.
///
/// Returns a fresh map of owned tensors; the caller's input is not
/// mutated. The map preserves the input's key set.
///
/// ## When this is a no-op
///
/// - `left_padding` is `None` — there is nothing to roll, return inputs
///   wrapped as owned tensors.
/// - `B == 1`, `left_padding[0] == 0`, and `kv_valid_len[0] >= kv_len` —
///   the row is already in the canonical layout; we still return an
///   owned (copied) tensor so the caller has a uniform ownership story.
pub fn normalize_batched_shared_kv_states(
    shared_kv_states: &HashMap<LayerType, (&MlxArray, &MlxArray)>,
    kv_valid_len: &BatchScalar<'_>,
    left_padding: Option<&BatchScalar<'_>>,
) -> HashMap<LayerType, (UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let mut out = HashMap::with_capacity(shared_kv_states.len());

    let Some(left) = left_padding else {
        // No left-padding ⇒ no roll. Still hand back owned copies so the
        // caller has a uniform ownership story across both arms.
        for (&layer_type, (keys, values)) in shared_kv_states {
            out.insert(layer_type, (ffi::copy(keys), ffi::copy(values)));
        }
        return out;
    };

    for (&layer_type, (keys, values)) in shared_kv_states {
        let k_out = normalize_shared_kv_tensor(keys, kv_valid_len, left);
        let v_out = normalize_shared_kv_tensor(values, kv_valid_len, left);
        out.insert(layer_type, (k_out, v_out));
    }
    out
}

/// Per-tensor normalisation. Ports upstream `_normalize_shared_kv_tensor`.
fn normalize_shared_kv_tensor(
    tensor: &MlxArray,
    kv_valid_len: &BatchScalar<'_>,
    left_padding: &BatchScalar<'_>,
) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(tensor);
    if shape.len() != 4 {
        // Upstream returns the tensor unchanged if it is not 4-D. We
        // mirror that by handing back a copy (callers expect owned).
        return ffi::copy(tensor);
    }

    let batch = shape[0];
    let n_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // Broadcast/clip per-row vectors against the batch and seq_len axes.
    let valid = broadcast_batch_vector(kv_valid_len, batch, seq_len);
    let left = broadcast_batch_vector(left_padding, batch, seq_len);

    // B=1 short-circuit when no roll is needed AND the row's prefix already
    // covers the full buffer (mirrors upstream's `int(left[0].item()) == 0
    // and int(valid[0].item()) >= seq_len` check).
    //
    // We deliberately do NOT short-circuit when `B==1, left[0]==0, valid[0]
    // < seq_len` — the tail still needs zeroing. The upstream `keep`
    // multiplication path handles that uniformly with the rolled tensor.
    if batch == 1 && shape_eq_one_scalar(&left, 0) && shape_ge_one_scalar(&valid, seq_len) {
        return ffi::copy(tensor);
    }

    // Per-row roll along axis 2 by -left[b]. Implemented via
    // take_along_axis with a broadcasted index tensor of shape
    // [B, 1, seq_len, 1]: index[b, 0, t, 0] = (t + left[b]) % seq_len.
    let rolled = roll_left_per_row(tensor, &left, batch, n_heads, seq_len, head_dim);

    // keep[b, 0, t, 0] = 1.0 iff t < valid[b]. Shape [B, 1, seq_len, 1].
    let keep = build_keep_mask(&valid, batch, seq_len, ffi::array_dtype(tensor));

    ffi::multiply(&rolled, &keep)
}

/// Build a `[B, 1, seq_len, 1]` mask whose entries are `1.0` at positions
/// `t < valid[b]` and `0.0` otherwise, materialised in `dtype` so the
/// subsequent `multiply(rolled, keep)` preserves the K/V dtype.
fn build_keep_mask(valid: &MlxArray, batch: i32, seq_len: i32, dtype: i32) -> UniquePtr<MlxArray> {
    let valid_4d = ffi::reshape(valid, &[batch, 1, 1, 1]);
    let t_1d = ffi::arange_i32(0, seq_len, 1);
    let t_4d = ffi::reshape(&t_1d, &[1, 1, seq_len, 1]);
    let bool_keep = ffi::less(&t_4d, &valid_4d); // bool [B, 1, seq_len, 1]
    ffi::astype(&bool_keep, dtype)
}

/// Build a `[B, n_heads, seq_len, head_dim]` index tensor and apply
/// `take_along_axis(tensor, idx, axis=2)` so each row independently rolls
/// left by `left[b]`.
///
/// We materialise the indices to the full source shape because MLX's
/// `take_along_axis` requires the indices to broadcast to the source
/// shape along all axes other than the indexed one. The cost is one
/// `[B, n_heads, seq_len, head_dim]` `int32` allocation per call; in the
/// drafter-rebind hot path this is amortised against four K/V
/// normalisations per draft block.
fn roll_left_per_row(
    tensor: &MlxArray,
    left: &MlxArray,
    batch: i32,
    n_heads: i32,
    seq_len: i32,
    head_dim: i32,
) -> UniquePtr<MlxArray> {
    // base = arange(seq_len) reshaped to [1, 1, seq_len, 1].
    let base_1d = ffi::arange_i32(0, seq_len, 1);
    let base_4d = ffi::reshape(&base_1d, &[1, 1, seq_len, 1]);

    // shift = left reshaped to [B, 1, 1, 1].
    let shift_4d = ffi::reshape(left, &[batch, 1, 1, 1]);

    // raw_idx[b, t] = t + left[b]; modular reduction along seq_len keeps
    // the indices in-bounds even when left[b] could be >= seq_len. Upstream
    // clips left to [0, seq_len], so the modulus is mostly defensive.
    let raw_idx = ffi::add(&base_4d, &shift_4d);
    let seq_len_arr = ffi::from_slice_i32(&[seq_len], &[1]);
    let idx_mod = ffi::remainder(&raw_idx, &seq_len_arr);

    // Broadcast indices to the full source shape.
    let idx_full = ffi::broadcast_to(&idx_mod, &[batch, n_heads, seq_len, head_dim]);
    ffi::take_along_axis(tensor, &idx_full, 2)
}

/// Mirror upstream's `_broadcast_batch_vector`: lift `value` to an
/// `int32` 1-D tensor of length `batch`, repeating B=1 inputs across rows,
/// and clip into `[0, limit]`.
fn broadcast_batch_vector(value: &BatchScalar<'_>, batch: i32, limit: i32) -> UniquePtr<MlxArray> {
    let raw = match value {
        BatchScalar::Scalar(v) => ffi::from_slice_i32(&[*v], &[1]),
        BatchScalar::PerRow(arr) => {
            // Reduce to a flat int32 vector then assert length == batch.
            let flat = ffi::astype(arr, dtype::INT32);
            let flat_shape = ffi::array_shape(&flat);
            // If 0-D, lift to [1].
            if flat_shape.is_empty() {
                ffi::reshape(&flat, &[1])
            } else if flat_shape.len() == 1 {
                flat
            } else {
                ffi::flatten(&flat)
            }
        }
    };

    // Repeat single-element vectors across the batch axis.
    let raw_shape = ffi::array_shape(&raw);
    let raw_len = if raw_shape.is_empty() {
        1
    } else {
        raw_shape[0]
    };
    let repeated = if raw_len == 1 && batch != 1 {
        ffi::repeat(&raw, batch, 0)
    } else {
        debug_assert_eq!(
            raw_len, batch,
            "BatchScalar::PerRow length {raw_len} != batch {batch}"
        );
        raw
    };

    // Clip into [0, limit] so downstream gather indices stay in bounds.
    let lo = ffi::from_slice_i32(&[0], &[1]);
    let hi = ffi::from_slice_i32(&[limit], &[1]);
    let clamped_lo = ffi::maximum(&repeated, &lo);
    ffi::minimum(&clamped_lo, &hi)
}

/// `true` iff `vec` has shape `[1]` and `vec[0] == expected`. The lookup
/// is materialised on the GPU and forced by `eval` so the comparison is
/// against a real i32, not a symbolic placeholder.
fn shape_eq_one_scalar(vec: &MlxArray, expected: i32) -> bool {
    let shape = ffi::array_shape(vec);
    if shape != [1] {
        return false;
    }
    ffi::eval(vec);
    ffi::item_i32(vec) == expected
}

/// `true` iff `vec` has shape `[1]` and `vec[0] >= expected`.
fn shape_ge_one_scalar(vec: &MlxArray, expected: i32) -> bool {
    let shape = ffi::array_shape(vec);
    if shape != [1] {
        return false;
    }
    ffi::eval(vec);
    ffi::item_i32(vec) >= expected
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: extract one f32 from a mask tensor at index `[..., q, k]`
    /// (last two dims). Casts to f32 first so bf16/f16 masks can be
    /// inspected through the same `item_f32` API.
    fn mask_at_qk(mask: &MlxArray, indices: &[i32]) -> f32 {
        let f32_mask = ffi::astype(mask, dtype::FLOAT32);
        let mut starts = indices.to_vec();
        let mut stops: Vec<i32> = indices.iter().map(|v| v + 1).collect();
        // ffi::slice requires len(starts) == ndim, so pad with full slices
        // on any leading axes the caller did not specify.
        let shape = ffi::array_shape(&f32_mask);
        while starts.len() < shape.len() {
            starts.insert(0, 0);
            stops.insert(0, shape[stops.len()]);
        }
        // Recompute stops to be 1-past-start on the trailing dims.
        let scalar = ffi::slice(&f32_mask, &starts, &stops);
        ffi::item_f32(&scalar)
    }

    // ----- LayerType round-trip --------------------------------------------

    #[test]
    fn layer_type_canonical_strings_match_upstream() {
        assert_eq!(LayerType::FullAttention.as_str(), "full_attention");
        assert_eq!(
            LayerType::SlidingWindowAttention.as_str(),
            "sliding_attention"
        );
    }

    // ----- bidirectional_full_mask -----------------------------------------

    #[test]
    fn full_mask_returns_none_when_no_padding() {
        // kv_valid_len == None: degenerate case, mask is None.
        let mask = bidirectional_full_mask(
            /*query_len=*/ 1,
            /*kv_len=*/ 8,
            None,
            dtype::FLOAT32,
        );
        assert!(mask.is_none(), "full mask must be None without padding");
    }

    #[test]
    fn full_mask_returns_none_when_scalar_valid_covers_full_kv() {
        // kv_valid_len >= kv_len ⇒ no padding ⇒ mask is None.
        let valid = BatchScalar::Scalar(8);
        let mask = bidirectional_full_mask(1, 8, Some(&valid), dtype::FLOAT32);
        assert!(mask.is_none());

        let valid = BatchScalar::Scalar(100);
        let mask = bidirectional_full_mask(1, 8, Some(&valid), dtype::FLOAT32);
        assert!(mask.is_none());
    }

    #[test]
    fn full_mask_scalar_short_prefix_produces_expected_bias() {
        // kv_len=5, kv_valid_len=3: positions [0,1,2]=0, [3,4]=-inf.
        let valid = BatchScalar::Scalar(3);
        let mask = bidirectional_full_mask(
            /*query_len=*/ 1,
            /*kv_len=*/ 5,
            Some(&valid),
            dtype::FLOAT32,
        )
        .expect("mask must materialise when valid < kv_len");
        assert_eq!(ffi::array_shape(&mask), vec![1, 1, 1, 5]);

        // Inspect each column.
        for k in 0..3 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert_eq!(v, 0.0, "valid prefix col {k} must be 0, got {v}");
        }
        for k in 3..5 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert!(
                v.is_infinite() && v < 0.0,
                "padded col {k} must be -inf, got {v}",
            );
        }
    }

    #[test]
    fn full_mask_per_row_short_prefix_produces_expected_bias() {
        // Issue acceptance criterion verbatim: B=2, kv_valid_lens=[5, 3],
        // kv_len=5.
        //   row 0: all positions valid -> bias = 0 everywhere.
        //   row 1: positions [3, 4] masked -> bias = -inf there, 0 in [0..3].
        let kv_valid_lens = ffi::from_slice_i32(&[5, 3], &[2]);
        let valid = BatchScalar::PerRow(&kv_valid_lens);
        let mask =
            bidirectional_full_mask(1, 5, Some(&valid), dtype::FLOAT32).expect("must materialise");
        assert_eq!(ffi::array_shape(&mask), vec![2, 1, 1, 5]);

        // Row 0: every column must be 0.
        for k in 0..5 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert_eq!(v, 0.0, "row 0 col {k} must be 0, got {v}");
        }
        // Row 1: [0..3] = 0, [3..5] = -inf.
        for k in 0..3 {
            let v = mask_at_qk(&mask, &[1, 0, 0, k]);
            assert_eq!(v, 0.0, "row 1 col {k} must be 0, got {v}");
        }
        for k in 3..5 {
            let v = mask_at_qk(&mask, &[1, 0, 0, k]);
            assert!(
                v.is_infinite() && v < 0.0,
                "row 1 col {k} must be -inf, got {v}",
            );
        }
    }

    #[test]
    fn full_mask_key_offset_uses_absolute_key_positions() {
        // Latest upstream uses key_offset=max(kv_valid_len-kv_len, 0) for
        // sliced full-attention K/V. With kv_valid_len=10 and a local slab
        // carrying absolute keys [7, 8, 9, 10], only the last column is past
        // the valid prefix.
        let valid = BatchScalar::Scalar(10);
        let key_offset = BatchScalar::Scalar(7);
        let mask = bidirectional_full_mask_with_key_offset(
            1,
            4,
            Some(&valid),
            &key_offset,
            dtype::FLOAT32,
        )
        .expect("absolute key 10 must be masked");
        assert_eq!(ffi::array_shape(&mask), vec![1, 1, 1, 4]);

        for k in 0..3 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert_eq!(v, 0.0, "absolute key {} must be valid", 7 + k);
        }
        let v = mask_at_qk(&mask, &[0, 0, 0, 3]);
        assert!(
            v.is_infinite() && v < 0.0,
            "absolute key 10 must be masked, got {v}",
        );
    }

    // ----- bidirectional_swa_mask ------------------------------------------

    #[test]
    fn swa_mask_returns_none_when_window_covers_everything() {
        // kv_len=4, window=8, query_offset=0, query_len=1.
        // kv_len <= window AND query_offset + query_len = 1 <= 4 + 8 = 12.
        let qo = BatchScalar::Scalar(0);
        let mask = bidirectional_swa_mask(
            /*query_len=*/ 1,
            &qo,
            /*kv_len=*/ 4,
            /*window=*/ 8,
            None,
            dtype::FLOAT32,
        );
        assert!(
            mask.is_none(),
            "SWA mask should be None when window dominates"
        );
    }

    #[test]
    fn swa_mask_handcoded_fixture_query_len_1_kv_8_window_4() {
        // Issue acceptance criterion verbatim: query_len=1, kv_len=8,
        // sliding_window=4 must produce bias where bias[..., q, k] = 0
        // iff |q - k| < 4. With query_offset=0 the single query is q=0.
        let qo = BatchScalar::Scalar(0);
        let mask = bidirectional_swa_mask(
            /*query_len=*/ 1,
            &qo,
            /*kv_len=*/ 8,
            /*window=*/ 4,
            None,
            dtype::FLOAT32,
        )
        .expect("mask must materialise when kv_len > window");
        assert_eq!(ffi::array_shape(&mask), vec![1, 1, 1, 8]);

        // |0 - k| < 4 means k in {0, 1, 2, 3} -> 0; k in {4..7} -> -inf.
        for k in 0..4 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert_eq!(v, 0.0, "k={k} must be 0 (inside window), got {v}");
        }
        for k in 4..8 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert!(
                v.is_infinite() && v < 0.0,
                "k={k} must be -inf (outside window), got {v}",
            );
        }
    }

    #[test]
    fn swa_mask_query_offset_shifts_window() {
        // query_offset=3, query_len=1, kv_len=8, window=2.
        // q = 3, |3 - k| < 2 ⇒ k ∈ {2, 3, 4} -> 0; else -inf.
        let qo = BatchScalar::Scalar(3);
        let mask = bidirectional_swa_mask(1, &qo, 8, 2, None, dtype::FLOAT32).expect("materialise");
        for k in 0..2 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert!(v.is_infinite() && v < 0.0, "k={k} must be -inf");
        }
        for k in 2..5 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert_eq!(v, 0.0, "k={k} must be 0");
        }
        for k in 5..8 {
            let v = mask_at_qk(&mask, &[0, 0, 0, k]);
            assert!(v.is_infinite() && v < 0.0, "k={k} must be -inf");
        }
    }

    // ----- make_drafter_masks ----------------------------------------------

    #[test]
    fn make_drafter_masks_returns_none_for_both_layer_types_in_fast_path() {
        // Issue acceptance criterion verbatim: when B=1, kv_len <=
        // sliding_window, query_offset + query_len <= kv_len + sliding_window,
        // BOTH masks must be None. Note: kv_valid_len == query_offset
        // upstream, so we use query_offset = kv_len so the SWA "no mask"
        // criterion (kv_len <= window) drives the result.
        let kv_len = 4_i32;
        let sliding_window = 8_i32;
        let query_len = 1_i32;
        // query_offset = kv_len so that kv_valid_len == kv_len, hence the
        // full mask returns None as well.
        let qo = BatchScalar::Scalar(kv_len);

        // Stub K/V shaped [B=1, n_kv_heads=1, kv_len, head_dim=1].
        let k_full = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let v_full = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let k_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let v_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);

        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::FullAttention, (&k_full, &v_full));
        shared.insert(LayerType::SlidingWindowAttention, (&k_swa, &v_swa));

        let masks = make_drafter_masks(&shared, query_len, &qo, sliding_window, dtype::FLOAT32);

        assert_eq!(masks.len(), 2);
        assert!(
            masks.get(&LayerType::FullAttention).unwrap().is_none(),
            "full mask must be None in the fast path",
        );
        assert!(
            masks
                .get(&LayerType::SlidingWindowAttention)
                .unwrap()
                .is_none(),
            "SWA mask must be None in the fast path",
        );
    }

    #[test]
    fn make_drafter_masks_with_valid_len_maps_swa_to_local_window() {
        // Regression pin for the upstream #1166 mask change: after a
        // rotating-cache slice the absolute drafter position can be much
        // larger than the local K/V axis. SWA masks must compare against
        // min(position, kv_len), while still using kv_valid_len for tail
        // validity.
        let kv_len = 4_i32;
        let query_len = 1_i32;
        let sliding_window = 5_i32;
        let query_offset = BatchScalar::Scalar(9);
        let kv_valid_len = BatchScalar::Scalar(10);

        let k_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let v_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);

        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::SlidingWindowAttention, (&k_swa, &v_swa));

        let masks = make_drafter_masks_with_valid_len(
            &shared,
            query_len,
            &query_offset,
            sliding_window,
            dtype::FLOAT32,
            Some(&kv_valid_len),
        );
        assert!(
            masks
                .get(&LayerType::SlidingWindowAttention)
                .unwrap()
                .is_none(),
            "SWA should be mask-free after mapping absolute position 9 to local offset 4",
        );
    }

    /// The `make_drafter_masks` wrapper restores `query_offset == kv_valid_len -
    /// 1` by delegating with `kv_valid_len = query_offset + 1` (issue #163). Pin
    /// that on a NON-fast-path shape where the SWA mask materializes (`kv_len >
    /// sliding_window`), so a regression dropping the `+1` changes the emitted
    /// mask. The wrapper output must be byte-identical to the explicit
    /// `_with_valid_len(.., Some(query_offset + 1))` call.
    #[test]
    fn make_drafter_masks_wrapper_matches_valid_len_query_offset_plus_one() {
        let kv_len = 6_i32;
        let query_len = 1_i32;
        let sliding_window = 3_i32;
        let qo = 4_i32;

        let k_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let v_swa = ffi::zeros(&[1, 1, kv_len, 1], dtype::FLOAT32);
        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::SlidingWindowAttention, (&k_swa, &v_swa));

        let via_wrapper = make_drafter_masks(
            &shared,
            query_len,
            &BatchScalar::Scalar(qo),
            sliding_window,
            dtype::FLOAT32,
        );
        let via_valid_len = make_drafter_masks_with_valid_len(
            &shared,
            query_len,
            &BatchScalar::Scalar(qo),
            sliding_window,
            dtype::FLOAT32,
            Some(&BatchScalar::Scalar(qo + 1)),
        );

        let wrapper_swa = via_wrapper
            .get(&LayerType::SlidingWindowAttention)
            .unwrap()
            .as_ref()
            .expect("non-fast-path SWA must materialize a mask");
        let valid_swa = via_valid_len
            .get(&LayerType::SlidingWindowAttention)
            .unwrap()
            .as_ref()
            .expect("explicit valid-len SWA must materialize a mask");

        let wshape = ffi::array_shape(wrapper_swa);
        assert_eq!(
            wshape,
            ffi::array_shape(valid_swa),
            "wrapper and explicit valid-len masks must share a shape"
        );

        // Flatten and compare element-wise (finite and -inf both matched).
        let total: i32 = wshape.iter().product();
        let wflat = ffi::astype(&ffi::reshape(wrapper_swa, &[total]), dtype::FLOAT32);
        let vflat = ffi::astype(&ffi::reshape(valid_swa, &[total]), dtype::FLOAT32);
        for i in 0..total {
            let a = ffi::item_f32(&ffi::slice(&wflat, &[i], &[i + 1]));
            let b = ffi::item_f32(&ffi::slice(&vflat, &[i], &[i + 1]));
            assert_eq!(a.is_finite(), b.is_finite(), "elem {i} finiteness mismatch");
            if a.is_finite() {
                assert_eq!(a, b, "elem {i} value mismatch: wrapper={a} valid_len={b}");
            }
        }
    }

    /// Ragged batched MTP greedy-parity regression (issue #161 / PR #162).
    ///
    /// When a B > 1 burst mixes prompt lengths, each row's shared K/V is
    /// normalized (rolled left by `left_padding[r]`) so its valid keys occupy
    /// `[0, kv_valid_len[r])` in the **prefix-valid** frame, while the drafter
    /// query's RoPE anchor stays in the **padded** frame at `bonus_position[r]
    /// = left_padding[r] + kv_valid_len[r] - 1` (so the query's RoPE phase
    /// matches the baked keys' `+left_padding[r]` phase). The SWA mask distance,
    /// however, is computed against the normalized keys at `[0, kv_valid_len)`,
    /// so it MUST use the valid-frame query position `kv_valid_len[r] - 1`, NOT
    /// the padded `bonus_position[r]`. Conflating the two over-masks the
    /// most-left-padded (shortest) row's SWA layers by exactly `left_padding[r]`
    /// keys — the bonus query can attend to ZERO valid keys, which is why the
    /// shortest prompt emitted empty content and never hit EOS in the real-model
    /// gate while the other rows were byte-identical.
    ///
    /// Numeric reproduction (the unit proxy for the 31B parity gate):
    /// - kv_len (padded buffer width) = 8, sliding_window = 4.
    /// - Row 0 (shortest, most padded): kv_valid_len = 3, left_padding = 5, so
    ///   bonus_position = 5 + 3 - 1 = 7 = kv_len - 1. Its valid keys are [0, 3).
    ///   Correct SWA distance uses query position kv_valid_len-1 = 2, so dist[k]
    ///   = 2 - k and every valid key k in {0,1,2} is inside the window. With the
    ///   bug (query position = 7) dist[k] = 7 - k, so `dist < window` (= 4)
    ///   requires k > 3 while `k < kv_valid_len` requires k < 3 — an empty set,
    ///   i.e. the bonus query attends to nothing.
    /// - Row 1 (full length): kv_valid_len = 8, left_padding = 0, so
    ///   bonus_position = 7 = kv_valid_len - 1 (the unaffected baseline).
    ///
    /// The test asserts row 0's three valid SWA keys are attended (bias 0.0) and
    /// its padded tail is masked, which fails on the pre-fix code and passes
    /// after the SWA query offset is derived from kv_valid_len.
    #[test]
    fn swa_mask_ragged_left_padding_attends_short_rows_valid_keys() {
        let kv_len = 8_i32;
        let query_len = 1_i32;
        let sliding_window = 4_i32;

        // Padded-frame RoPE anchors (bonus_position) — uniform max_len - 1.
        let query_offsets = ffi::from_slice_i32(&[7, 7], &[2]);
        let query_offset = BatchScalar::PerRow(&query_offsets);
        // Per-row valid prefix lengths after K/V normalization.
        let kv_valid_lens = ffi::from_slice_i32(&[3, 8], &[2]);
        let kv_valid_len = BatchScalar::PerRow(&kv_valid_lens);

        let k_swa = ffi::zeros(&[2, 1, kv_len, 1], dtype::FLOAT32);
        let v_swa = ffi::zeros(&[2, 1, kv_len, 1], dtype::FLOAT32);

        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::SlidingWindowAttention, (&k_swa, &v_swa));

        let masks = make_drafter_masks_with_valid_len(
            &shared,
            query_len,
            &query_offset,
            sliding_window,
            dtype::FLOAT32,
            Some(&kv_valid_len),
        );

        let swa = masks
            .get(&LayerType::SlidingWindowAttention)
            .expect("SWA entry present")
            .as_ref()
            .expect("per-row SWA mask must materialise (kv_valid differs from kv_len)");
        assert_eq!(
            ffi::array_shape(swa),
            vec![2, 1, query_len, kv_len],
            "per-row SWA mask must be [B, 1, query_len, kv_len]",
        );

        // Row 0 (most padded): its three valid keys [0, 3) must be attended.
        // Pre-fix these are all -inf because the padded query offset 7 pushes
        // every valid key outside the window.
        for k in 0..3 {
            let v = mask_at_qk(swa, &[0, 0, 0, k]);
            assert_eq!(
                v, 0.0,
                "row 0 (lp=5) valid SWA key {k} must be attended (0.0), got {v}",
            );
        }
        // Row 0 padded tail [3, kv_len) must remain masked.
        for k in 3..kv_len {
            let v = mask_at_qk(swa, &[0, 0, 0, k]);
            assert!(
                v.is_infinite() && v < 0.0,
                "row 0 padded SWA key {k} must be masked (-inf), got {v}",
            );
        }

        // Row 1 (full length, lp=0) is the byte-identical baseline: with window
        // 4 and query position 7 it attends keys {4,5,6,7} and masks {0,1,2,3}.
        for k in 0..4 {
            let v = mask_at_qk(swa, &[1, 0, 0, k]);
            assert!(
                v.is_infinite() && v < 0.0,
                "row 1 SWA key {k} must be outside the window (-inf), got {v}",
            );
        }
        for k in 4..kv_len {
            let v = mask_at_qk(swa, &[1, 0, 0, k]);
            assert_eq!(
                v, 0.0,
                "row 1 SWA key {k} must be inside the window (0.0), got {v}",
            );
        }
    }

    // ----- dtype preservation ----------------------------------------------

    #[test]
    fn full_mask_dtype_matches_request_bfloat16() {
        let valid = BatchScalar::Scalar(3);
        let mask =
            bidirectional_full_mask(1, 5, Some(&valid), dtype::BFLOAT16).expect("materialise");
        assert_eq!(
            ffi::array_dtype(&mask),
            dtype::BFLOAT16,
            "mask dtype must be bf16, no silent f32 promotion",
        );
    }

    #[test]
    fn full_mask_dtype_matches_request_float16() {
        let valid = BatchScalar::Scalar(3);
        let mask =
            bidirectional_full_mask(1, 5, Some(&valid), dtype::FLOAT16).expect("materialise");
        assert_eq!(
            ffi::array_dtype(&mask),
            dtype::FLOAT16,
            "mask dtype must be f16, no silent f32 promotion",
        );
    }

    #[test]
    fn swa_mask_dtype_matches_request_bfloat16() {
        let qo = BatchScalar::Scalar(0);
        let mask =
            bidirectional_swa_mask(1, &qo, 8, 4, None, dtype::BFLOAT16).expect("materialise");
        assert_eq!(
            ffi::array_dtype(&mask),
            dtype::BFLOAT16,
            "SWA mask dtype must be bf16",
        );
    }

    // ----- normalize_batched_shared_kv_states ------------------------------

    #[test]
    fn normalize_no_op_when_left_padding_is_none() {
        // No left_padding ⇒ identity copy. Verify shapes propagate.
        let k = ffi::zeros(&[1, 1, 4, 2], dtype::FLOAT32);
        let v = ffi::zeros(&[1, 1, 4, 2], dtype::FLOAT32);
        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::FullAttention, (&k, &v));

        let kv_valid_len = BatchScalar::Scalar(4);
        let out = normalize_batched_shared_kv_states(&shared, &kv_valid_len, None);

        let (k_out, v_out) = out.get(&LayerType::FullAttention).unwrap();
        assert_eq!(ffi::array_shape(k_out), vec![1, 1, 4, 2]);
        assert_eq!(ffi::array_shape(v_out), vec![1, 1, 4, 2]);
    }

    #[test]
    fn normalize_per_row_roll_zeros_tail_beyond_kv_valid_len() {
        // Issue acceptance criterion verbatim: B=2, kv_valid_lens=[5, 3],
        // left_padding=[2, 4], kv_len=8. After normalization:
        //   - row 0's K is at indices [0..5] (rolled left by 2),
        //   - row 1's K is at indices [0..3] (rolled left by 4),
        //     with [3..8] zeroed.
        //
        // Use kv_len=8 (rather than the issue's kv_len=5 which is
        // inconsistent with left_padding=4) so the roll is exercisable.
        // We build a K tensor where K[b, 0, t, 0] = bi * 100 + t so the
        // post-roll values are easy to read.
        let kv_len = 8_i32;
        let batch = 2_i32;
        // Build K via from_slice_f32 then reshape.
        let mut k_data = Vec::with_capacity((batch * kv_len) as usize);
        for bi in 0..batch {
            for t in 0..kv_len {
                k_data.push((bi as f32) * 100.0 + (t as f32));
            }
        }
        let k_flat = ffi::from_slice_f32(&k_data, &[batch, 1, kv_len, 1]);
        let v_flat = ffi::copy(&k_flat);

        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::FullAttention, (&k_flat, &v_flat));

        let kv_valid_arr = ffi::from_slice_i32(&[5, 3], &[batch]);
        let left_arr = ffi::from_slice_i32(&[2, 4], &[batch]);
        let kv_valid = BatchScalar::PerRow(&kv_valid_arr);
        let left = BatchScalar::PerRow(&left_arr);

        let out = normalize_batched_shared_kv_states(&shared, &kv_valid, Some(&left));
        let (k_out, _v_out) = out.get(&LayerType::FullAttention).unwrap();
        assert_eq!(ffi::array_shape(k_out), vec![batch, 1, kv_len, 1]);

        // Row 0: rolled left by 2.
        //   raw t  : 0   1   2   3   4   5   6   7
        //   value  : 0   1   2   3   4   5   6   7
        //   after  : 2   3   4   5   6   7   0   1
        //   keep   : 1   1   1   1   1   0   0   0   (kv_valid_len=5)
        //   final  : 2   3   4   5   6   0   0   0
        let row0_expected: [f32; 8] = [2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0];
        for (t, expected) in row0_expected.iter().enumerate() {
            let v = mask_at_qk(k_out, &[0, 0, t as i32, 0]);
            assert!(
                (v - expected).abs() < 1e-5,
                "row 0 t={t} expected {expected} got {v}",
            );
        }

        // Row 1: rolled left by 4.
        //   raw t  : 0   1   2   3   4   5   6   7
        //   value  : 100 101 102 103 104 105 106 107
        //   after  : 104 105 106 107 100 101 102 103
        //   keep   : 1   1   1   0   0   0   0   0   (kv_valid_len=3)
        //   final  : 104 105 106 0   0   0   0   0
        let row1_expected: [f32; 8] = [104.0, 105.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (t, expected) in row1_expected.iter().enumerate() {
            let v = mask_at_qk(k_out, &[1, 0, t as i32, 0]);
            assert!(
                (v - expected).abs() < 1e-5,
                "row 1 t={t} expected {expected} got {v}",
            );
        }

        // Reference the kv_len variable so clippy does not complain about
        // unused bindings now that the index loop no longer uses it.
        let _ = kv_len;
    }

    #[test]
    fn normalize_preserves_dtype_bfloat16() {
        // K/V values do not matter for this check, only the dtype.
        let k = ffi::zeros(&[1, 1, 4, 2], dtype::BFLOAT16);
        let v = ffi::zeros(&[1, 1, 4, 2], dtype::BFLOAT16);
        let mut shared: HashMap<LayerType, (&MlxArray, &MlxArray)> = HashMap::new();
        shared.insert(LayerType::FullAttention, (&k, &v));

        let kv_valid_arr = ffi::from_slice_i32(&[2], &[1]);
        let left_arr = ffi::from_slice_i32(&[1], &[1]);
        let kv_valid = BatchScalar::PerRow(&kv_valid_arr);
        let left = BatchScalar::PerRow(&left_arr);

        let out = normalize_batched_shared_kv_states(&shared, &kv_valid, Some(&left));
        let (k_out, v_out) = out.get(&LayerType::FullAttention).unwrap();
        assert_eq!(
            ffi::array_dtype(k_out),
            dtype::BFLOAT16,
            "K dtype must be preserved",
        );
        assert_eq!(
            ffi::array_dtype(v_out),
            dtype::BFLOAT16,
            "V dtype must be preserved",
        );
    }
}
