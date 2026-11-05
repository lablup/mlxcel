// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Stacked KV buffers for batched decode over per-sequence dense caches.
//!
//! A model that owns one dense `KVCache` per sequence decodes a batch by
//! attending row by row: an update, a fetch and an SDPA launch per row and
//! layer, because the rows' live lengths differ. [`StackedKvReuse`] keeps a
//! `[B, n_kv_heads, capacity, head_dim]` buffer per attention layer instead:
//! one scatter writes every row's new K/V at its own position and one masked
//! SDPA attends the whole batch. Each row's cache adopts a view of the stacked
//! buffer, so it stays a valid dense cache for every other consumer, and the
//! next step reuses the buffer as long as every row still holds its view.
//!
//! A view keeps the whole stacked buffer alive, so a row that leaves the batch
//! or is about to be snapshotted is materialized first ([`StackedKvReuse::owns`]
//! and [`materialize_kv`]).
//!
//! The masked batch SDPA reduces over the padded length, so a row's attention
//! output is not bit-identical to the same row attended alone.
//!
//! Used by: Qwen 3.5 (`qwen3_5`) batched decode.

use std::collections::HashMap;

use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Capacity granularity of the stacked buffer, matching `KVCache`'s own step.
const STACKED_KV_STEP: i32 = 256;

struct StackedKvLayer {
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
    /// `array_id` of each row's (keys, values) view, in batch order.
    row_ids: Vec<(usize, usize)>,
    /// Handles to those views, held so their ids stay unique.
    _views: Vec<UniquePtr<MlxArray>>,
}

/// Per-layer stacked KV buffers of the current decode batch.
#[derive(Default)]
pub(crate) struct StackedKvReuse {
    layers: HashMap<usize, StackedKvLayer>,
}

fn row_ids(cache: &KVCache) -> Option<(usize, usize)> {
    Some((
        mlxcel_core::array_id(cache.keys.as_ref()?),
        mlxcel_core::array_id(cache.values.as_ref()?),
    ))
}

impl StackedKvReuse {
    /// Drop every stacked buffer (the batch changed).
    pub(crate) fn reset(&mut self) {
        self.layers.clear();
    }

    /// Whether `cache` still holds a view of a stacked buffer.
    pub(crate) fn owns(&self, cache: &KVCache) -> bool {
        let Some((k, v)) = row_ids(cache) else {
            return false;
        };
        self.layers
            .values()
            .any(|layer| layer.row_ids.iter().any(|&(lk, lv)| lk == k || lv == v))
    }

    /// One single-token decode step of attention layer `layer_idx` for the
    /// whole batch, against the stacked buffer.
    ///
    /// `q` is `[B, n_heads, 1, head_dim]` and `k` / `v` are
    /// `[B, n_kv_heads, 1, head_dim]`, already normed and rotated. Returns
    /// `[B, n_heads, 1, head_dim]`, or `None` (with no cache touched) when a
    /// row's cache is not a dense FP16 cache the buffer can stand in for.
    pub(crate) fn attend(
        &mut self,
        layer_idx: usize,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        caches: &mut [&mut KVCache],
        scale: f32,
    ) -> Option<UniquePtr<MlxArray>> {
        if caches.is_empty() || !caches.iter().all(|c| c.is_dense_fp16_from_zero()) {
            self.layers.remove(&layer_idx);
            return None;
        }
        let batch = caches.len() as i32;
        let offsets: Vec<i32> = caches.iter().map(|c| c.offset).collect();
        let live_max = offsets.iter().copied().max().unwrap_or(0) + 1;

        let reused = self.layers.remove(&layer_idx).filter(|layer| {
            layer.row_ids.len() == caches.len()
                && caches
                    .iter()
                    .zip(&layer.row_ids)
                    .all(|(c, ids)| row_ids(c) == Some(*ids))
        });
        let (mut keys, mut values) = match reused {
            Some(layer) => (layer.keys, layer.values),
            None => {
                let capacity = (live_max + STACKED_KV_STEP - 1) / STACKED_KV_STEP * STACKED_KV_STEP;
                (
                    stack_rows(caches, &offsets, capacity, |c| c.keys.as_deref()),
                    stack_rows(caches, &offsets, capacity, |c| c.values.as_deref()),
                )
            }
        };
        // The rows' views of the old buffer are replaced below; drop them now
        // so the scatter can update the buffer in place.
        for cache in caches.iter_mut() {
            cache.keys = None;
            cache.values = None;
        }

        let capacity = mlxcel_core::array_shape(&keys)[2];
        if live_max > capacity {
            keys = grow(&keys, STACKED_KV_STEP);
            values = grow(&values, STACKED_KV_STEP);
        }

        // Row b's new token goes to slot offsets[b].
        let positions = mlxcel_core::from_slice_i32(&offsets, &[batch, 1, 1, 1]);
        let k_idx = mlxcel_core::broadcast_to(&positions, &mlxcel_core::array_shape(k));
        let v_idx = mlxcel_core::broadcast_to(&positions, &mlxcel_core::array_shape(v));
        keys = mlxcel_core::put_along_axis(&keys, &k_idx, k, 2);
        values = mlxcel_core::put_along_axis(&values, &v_idx, v, 2);

        // Row b attends to its first offsets[b] + 1 slots.
        let window_k = slice_len(&keys, live_max);
        let window_v = slice_len(&values, live_max);
        let lengths = mlxcel_core::from_slice_i32(
            &offsets.iter().map(|o| o + 1).collect::<Vec<_>>(),
            &[batch, 1, 1, 1],
        );
        let slots = mlxcel_core::reshape(
            &mlxcel_core::arange_i32(0, live_max, 1),
            &[1, 1, 1, live_max],
        );
        let mask = mlxcel_core::less(&slots, &lengths);
        let output =
            mlxcel_core::layers::attention(q, &window_k, &window_v, scale, Some(&mask), 0.0, 0);

        let mut ids = Vec::with_capacity(caches.len());
        let mut views = Vec::with_capacity(caches.len() * 2);
        for (row, cache) in caches.iter_mut().enumerate() {
            let row = row as i32;
            let k_view = mlxcel_core::utils::slice_axis(&keys, 0, row, row + 1);
            let v_view = mlxcel_core::utils::slice_axis(&values, 0, row, row + 1);
            ids.push((
                mlxcel_core::array_id(&k_view),
                mlxcel_core::array_id(&v_view),
            ));
            views.push(mlxcel_core::array_share(&k_view));
            views.push(mlxcel_core::array_share(&v_view));
            cache.adopt_dense_fp16(k_view, v_view, offsets[row as usize] + 1);
        }
        self.layers.insert(
            layer_idx,
            StackedKvLayer {
                keys,
                values,
                row_ids: ids,
                _views: views,
            },
        );
        Some(output)
    }
}

/// `[B, H, capacity, D]` from each row's first `offsets[b]` slots, zero padded.
fn stack_rows(
    caches: &[&mut KVCache],
    offsets: &[i32],
    capacity: i32,
    buffer: impl Fn(&KVCache) -> Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let rows: Vec<UniquePtr<MlxArray>> = caches
        .iter()
        .zip(offsets)
        .map(|(cache, &offset)| {
            let live = slice_len(buffer(cache).expect("dense cache checked"), offset);
            mlxcel_core::pad(&live, &[0, 0, 0, 0, 0, capacity - offset, 0, 0], 0.0)
        })
        .collect();
    let refs: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
    mlxcel_core::concatenate_many(&refs, 0)
}

/// The first `len` slots of a `[B, H, L, D]` buffer.
fn slice_len(buffer: &MlxArray, len: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::utils::slice_axis(buffer, 2, 0, len)
}

/// `buffer` with `extra` zero slots appended on the length axis.
fn grow(buffer: &MlxArray, extra: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::pad(buffer, &[0, 0, 0, 0, 0, extra, 0, 0], 0.0)
}

/// Replace a row's K/V views with copies of its live window, scheduled for
/// evaluation, so the row stops holding the stacked buffer.
pub(crate) fn materialize_kv(cache: &mut KVCache) {
    let offset = cache.offset;
    let (Some(keys), Some(values)) = (cache.keys.take(), cache.values.take()) else {
        return;
    };
    let keys = mlxcel_core::contiguous(&slice_len(&keys, offset), false);
    let values = mlxcel_core::contiguous(&slice_len(&values, offset), false);
    let pending = [&*keys as *const MlxArray, &*values as *const MlxArray];
    // SAFETY: both pointers refer to arrays alive for the call.
    unsafe { mlxcel_core::async_eval_all(&pending) };
    cache.adopt_dense_fp16(keys, values, offset);
}

/// [`mlxcel_core::batched_decode::forward_split_attention`] with a
/// single-token unmasked step served from `reuse`'s stacked buffer of
/// `layer_idx`; any other step, or a batch the buffer cannot stand in for,
/// takes the usual per-row attention.
pub(crate) fn forward_split_attention_stacked<A: mlxcel_core::batched_decode::SplitAttention>(
    attn: &A,
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    caches: &mut [&mut KVCache],
    rope_offsets: &[i32],
    reuse: &mut StackedKvReuse,
    layer_idx: usize,
) -> UniquePtr<MlxArray> {
    let batch = caches.len() as i32;
    let shape = attn.split_shape();
    let seq_len = mlxcel_core::array_shape(q)[1];
    let q = mlxcel_core::reshape(q, &[batch, seq_len, shape.num_heads, shape.head_dim]);
    let k = mlxcel_core::reshape(k, &[batch, seq_len, shape.num_kv_heads, shape.head_dim]);
    let v = mlxcel_core::reshape(v, &[batch, seq_len, shape.num_kv_heads, shape.head_dim]);
    let (q, k) = attn.qk_norm_heads(q, k);
    let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
    let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
    let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
    let (q, k) = attn.rope_batched(&q, &k, rope_offsets);

    let stacked = if seq_len == 1 && !attn.custom_row_attend() {
        reuse.attend(layer_idx, &q, &k, &v, caches, shape.scale)
    } else {
        None
    };
    let output = stacked.unwrap_or_else(|| {
        reuse.layers.remove(&layer_idx);
        mlxcel_core::cache::attend_batched(&q, &k, &v, caches, shape.scale, None)
    });
    let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
    mlxcel_core::reshape(&output, &[batch, seq_len, shape.num_heads * shape.head_dim])
}
