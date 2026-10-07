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

//! Per-batch-row MTP verify for Gemma 4 on the row-wise geometries (issue
//! #2190).
//!
//! The B = 1 linear verify (#2185) is byte-identical to classic decode because
//! every query row takes the call decode makes. A `[B, K]` verify block cannot
//! keep that: each quantized matmul sees `B * K` rows, which leaves the
//! per-row-exact qmv window at `B * K >= 8` (and the compiled GeGLU gate counts
//! the same rows), and attention over the shared cache would reduce over each
//! row's padding and stale gap. So a B > 1 verify runs each batch row as its
//! own `[1, K]` call through the same helpers the B = 1 verify uses, and only
//! the cache write stays batched: the rows' new K/V go into the shared
//! `[B, ...]` cache in one update, which keeps the adapter's finalize, rollback
//! and drafter slab contracts unchanged.
//!
//! Each row's keys are physically cut to what that row's standalone run would
//! hold, history `[0, ve[r])` plus the verify block, because masked slots still
//! change a reduction's width. Sliding layers reduce in the ring order classic
//! decode would have at the row's own logical length.
//!
//! Ceiling: the buffered sliding cache compacts against the SHARED offset
//! (`RotatingKVCache::buffered_planned_drop`). Once it has compacted, a row
//! that lags the shared offset has lost the oldest keys of its own window, and
//! no reordering recovers them. That case is reported, never silently run.
use mlxcel_core::{MlxArray, UniquePtr, utils::slice_axis};

use super::{Attention, AttentionProjection, Cache, CacheInterface, DecoderLayer, SubopTimer};

/// Apply `f` to each `[1, ...]` batch row of `x` and concatenate the results
/// on axis 0.
pub(super) fn per_row(
    x: &MlxArray,
    mut f: impl FnMut(&MlxArray) -> UniquePtr<MlxArray>,
) -> UniquePtr<MlxArray> {
    let batch = mlxcel_core::array_shape(x)[0];
    let rows: Vec<UniquePtr<MlxArray>> = (0..batch)
        .map(|row| f(&slice_axis(x, 0, row, row + 1)))
        .collect();
    concat_rows(&rows)
}

fn concat_rows(rows: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let rows: Vec<&MlxArray> = rows
        .iter()
        .map(|row| row.as_ref().expect("row result is non-null"))
        .collect();
    mlxcel_core::concatenate_many(&rows, 0)
}

impl DecoderLayer {
    /// One decoder layer of a per-batch-row verify block.
    ///
    /// `offsets[r]` is row `r`'s logical valid end: its RoPE offset and the
    /// end of its history in the shared cache. Returns the `[B, K, D]` output,
    /// the stored K/V (as the ordinary forward does) and whether any row lost
    /// keys it needed (see the module docs).
    #[allow(clippy::type_complexity)]
    pub(super) fn forward_verify_rows(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
        offsets: &[i32],
        layer_idx: usize,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
        bool,
    ) {
        let batch = mlxcel_core::array_shape(x)[0];
        let rows: Vec<UniquePtr<MlxArray>> = (0..batch)
            .map(|row| slice_axis(x, 0, row, row + 1))
            .collect();
        let normed: Vec<UniquePtr<MlxArray>> = rows
            .iter()
            .map(|row| self.input_layernorm.forward(row))
            .collect();
        let (attn_rows, stored, inexact) = self
            .self_attn
            .attend_batch_rows(&normed, cache, shared_kv, offsets);
        let mut timer = SubopTimer::new(false, layer_idx);
        let out: Vec<UniquePtr<MlxArray>> = rows
            .iter()
            .zip(attn_rows.iter())
            .enumerate()
            .map(|(row, (x_row, attn))| {
                let input = per_layer_input.map(|p| slice_axis(p, 0, row as i32, row as i32 + 1));
                self.finish_layer(
                    x_row,
                    attn,
                    input.as_ref().map(|p| p.as_ref().unwrap()),
                    &mut timer,
                    false,
                    layer_idx,
                )
            })
            .collect();
        (concat_rows(&out), stored, inexact)
    }
}

impl Attention {
    /// Q, and K/V unless this is a KV-shared layer, for one `[1, K, D]` row,
    /// rotated at `offset` exactly as the B = 1 linear verify rotates them.
    fn project_verify_row(
        &self,
        x: &MlxArray,
        offset: i32,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let (raw_q, raw_kv) = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (q, k, v) = proj.forward(x);
                (q, Some((k, Some(v))))
            }
            AttentionProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let raw_values = if self.use_k_eq_v {
                    None
                } else {
                    Some(
                        v_proj
                            .as_ref()
                            .expect("Gemma4 attention expected v_proj for non-k_eq_v layer")
                            .forward(x),
                    )
                };
                (q_proj.forward(x), Some((k_proj.forward(x), raw_values)))
            }
            AttentionProjection::KvShared { q_proj } => (q_proj.forward(x), None),
        };
        let queries = self.rotate_verify_row(&raw_q, &self.q_norm, self.n_heads, offset);
        let kv = raw_kv.map(|(raw_keys, raw_values)| {
            let k_norm = self
                .k_norm
                .as_ref()
                .expect("k_norm must be Some for non-KV-shared layers");
            let keys = self.rotate_verify_row(&raw_keys, k_norm, self.n_kv_heads, offset);
            let raw_values = raw_values.as_ref().unwrap_or(&raw_keys);
            let len = mlxcel_core::array_shape(raw_values)[1];
            let values =
                mlxcel_core::reshape(raw_values, &[1, len, self.n_kv_heads, self.head_dim]);
            let values = self.v_norm.forward(&values);
            (keys, mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]))
        });
        (queries, kv)
    }

    /// The B = 1 linear verify's head chain: row-at-a-time RoPE where MLX's
    /// RoPE kernel choice differs (`mtp_row_rope`, CUDA), else the uniform
    /// chain.
    fn rotate_verify_row(
        &self,
        raw: &MlxArray,
        norm: &mlxcel_core::layers::RMSNorm,
        heads: i32,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        if self.mtp_row_rope {
            self.head_rows_like_decode(raw, norm, heads, offset)
        } else {
            self.head_branch(raw, norm, heads, offset)
        }
    }

    /// Attention for every batch row of a verify block, each as the B = 1
    /// linear verify would run it on that row's own cache. `rows` are the
    /// normed `[1, K, D]` inputs. Returns the per-row `[1, K, D]` outputs
    /// (after `o_proj`), the stored K/V and the inexact flag.
    #[allow(clippy::type_complexity)]
    fn attend_batch_rows(
        &self,
        rows: &[UniquePtr<MlxArray>],
        cache: &mut dyn CacheInterface,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
        offsets: &[i32],
    ) -> (
        Vec<UniquePtr<MlxArray>>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
        bool,
    ) {
        let len = mlxcel_core::array_shape(&rows[0])[1];
        let offset = cache.offset();
        let cursor = cache.speculative_ring_cursor();
        let use_shared = self.is_kv_shared_layer && shared_kv.is_some();

        let mut queries = Vec::with_capacity(rows.len());
        let mut new_keys = Vec::with_capacity(rows.len());
        let mut new_values = Vec::with_capacity(rows.len());
        for (row, &row_offset) in rows.iter().zip(offsets) {
            let (q, kv) = self.project_verify_row(row, row_offset);
            queries.push(q);
            if !use_shared {
                let (k, v) = kv.expect(
                    "a non-shared Gemma 4 layer must project K/V; the shared-KV \
                     layer path needs `shared_kv`",
                );
                new_keys.push(k);
                new_values.push(v);
            }
        }
        let (keys, values, stored) = match shared_kv {
            Some((keys, values)) if use_shared => {
                (mlxcel_core::copy(keys), mlxcel_core::copy(values), false)
            }
            _ => {
                let (keys, values) =
                    cache.update_and_fetch(concat_rows(&new_keys), concat_rows(&new_values));
                (keys, values, self.store_full_length_kv)
            }
        };

        let key_len = mlxcel_core::array_shape(&keys)[2];
        // Logical position of physical key index 0. Zero for the unbounded
        // full-attention cache; the buffered sliding cache's retained start.
        let base = offset + len - key_len;
        let block_start = key_len - len;
        let window = self.window_size;
        let mut inexact = false;
        let outputs = queries
            .iter()
            .zip(offsets)
            .enumerate()
            .map(|(row, (query, &row_end))| {
                // The earliest key this row's own run would still attend.
                let needed = if window > 0 {
                    (row_end + 1 - window).max(0)
                } else {
                    0
                };
                inexact |= base > needed || row_end > offset;
                let history_end = (row_end - base).clamp(0, block_start);
                let row_keys = visible_row(&keys, row as i32, history_end, block_start, key_len);
                let row_values =
                    visible_row(&values, row as i32, history_end, block_start, key_len);
                // The ring cursor is linear in the logical length, so the
                // cursor classic decode has at this row's length is the shared
                // one moved back by the row's lag.
                let row_cursor = cursor
                    .filter(|_| window > 0)
                    .map(|c| (c - (offset - row_end)).rem_euclid(window));
                let attn = self.attend_verify_row(query, &row_keys, &row_values, row_cursor);
                self.project_output(&attn, 1, len)
            })
            .collect();
        let stored = stored.then_some((keys, values));
        (outputs, stored, inexact)
    }

    /// One row's verify attention: the row-wise path the B = 1 verify takes,
    /// or its maskless fallback when that path declines (a single query on a
    /// full-attention layer, or an unbuffered sliding ring past the window).
    fn attend_verify_row(
        &self,
        query: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        ring_cursor: Option<i32>,
    ) -> UniquePtr<MlxArray> {
        if self.mtp_row_verify
            && let Some(rows) = super::verify_attention::attend_verify_rows(
                query,
                keys,
                values,
                self.scale,
                self.window_size,
                ring_cursor,
            )
        {
            return rows;
        }
        let query_len = mlxcel_core::array_shape(query)[2];
        let key_len = mlxcel_core::array_shape(keys)[2];
        let window = if query_len > 1 && self.window_size > 0 && key_len <= self.window_size {
            0
        } else {
            self.window_size
        };
        mlxcel_core::causal_attention(query, keys, values, self.scale, 0.0, window)
    }
}

/// Row `row` of a `[B, H, S, D]` key or value tensor, keeping history
/// `[0, history_end)` and the verify block `[block_start, key_len)` and
/// physically dropping the stale gap between them.
fn visible_row(
    array: &MlxArray,
    row: i32,
    history_end: i32,
    block_start: i32,
    key_len: i32,
) -> UniquePtr<MlxArray> {
    let row = slice_axis(array, 0, row, row + 1);
    if history_end == block_start {
        return row;
    }
    let history = slice_axis(&row, 2, 0, history_end);
    let block = slice_axis(&row, 2, block_start, key_len);
    mlxcel_core::concatenate(&history, &block, 2)
}

/// Stack per-row prefilled `[1, ...]` cache vectors into one `[B, ...]` cache
/// vector for the batched MTP adapter (issue #2190).
///
/// Each row is prefilled on its own, through the classic chunk partition, so
/// its K/V are exactly what a standalone run holds. Shorter rows are padded
/// with zeros AFTER their history (never before it), so every row keeps
/// physical slot == logical position and its RoPE frame: a short row is a row
/// whose valid end lags the shared offset, the same layout a divergent verify
/// round leaves behind. Equal-length rows share every scalar of cache state
/// and concatenate as they are; rows of different lengths are supported while
/// no sliding ring has wrapped (each row's slot `t` holds token `t`).
///
/// Only dense FP16 caches are stacked. Returns `Err` for anything else (a
/// quantized or paged KV mode, or ragged rows past the sliding window), which
/// the adapter turns into a declined burst.
pub(crate) fn stack_prefilled_rows(rows: Vec<Vec<Cache>>) -> Result<Vec<Cache>, String> {
    let Some(num_layers) = rows.first().map(Vec::len) else {
        return Err("no rows to stack".to_string());
    };
    if rows.iter().any(|row| row.len() != num_layers) {
        return Err("rows have different layer counts".to_string());
    }
    let mut per_layer: Vec<Vec<Cache>> = (0..num_layers).map(|_| Vec::new()).collect();
    for row in rows {
        for (layer, cache) in row.into_iter().enumerate() {
            per_layer[layer].push(cache);
        }
    }
    per_layer
        .into_iter()
        .enumerate()
        .map(|(layer, caches)| stack_layer(caches).map_err(|e| format!("layer {layer}: {e}")))
        .collect()
}

fn stack_layer(caches: Vec<Cache>) -> Result<Cache, String> {
    if caches.len() == 1 {
        return Ok(caches.into_iter().next().expect("one cache"));
    }
    let offsets: Vec<i32> = caches.iter().map(Cache::offset).collect();
    let longest = (0..caches.len())
        .max_by_key(|&row| (offsets[row], std::cmp::Reverse(row)))
        .expect("non-empty");
    let equal = offsets.iter().all(|&o| o == offsets[0]);
    let mut arrays: Vec<(&MlxArray, &MlxArray)> = Vec::with_capacity(caches.len());
    for cache in &caches {
        let (keys, values) = match cache {
            Cache::Standard(c) => {
                if c.mode != mlxcel_core::cache::KVCacheMode::Fp16
                    || c.is_paged_backed()
                    || c.live_len() != c.offset
                {
                    return Err("only dense FP16 full-attention caches can be stacked".into());
                }
                (c.keys.as_deref(), c.values.as_deref())
            }
            Cache::Rotating(c) => {
                if c.mode != mlxcel_core::cache::KVCacheMode::Fp16 || c.buffer_size != 0 {
                    return Err("only unbuffered FP16 sliding caches can be stacked".into());
                }
                if !equal && !c.snapshot_state().is_unwrapped() {
                    return Err("rows of different lengths past the sliding window".into());
                }
                (c.keys.as_deref(), c.values.as_deref())
            }
        };
        match (keys, values) {
            (Some(k), Some(v)) => arrays.push((k, v)),
            _ => return Err("an empty cache cannot be stacked".into()),
        }
    }
    let same_kind = caches
        .iter()
        .all(|c| matches!(c, Cache::Standard(_)) == matches!(caches[0], Cache::Standard(_)));
    if !same_kind {
        return Err("rows disagree on the layer's cache kind".into());
    }
    if equal
        && let Cache::Rotating(first) = &caches[0]
        && caches.iter().any(|c| match c {
            Cache::Rotating(c) => {
                c.buffer_write_idx() != first.buffer_write_idx()
                    || c.logical_start() != first.logical_start()
            }
            Cache::Standard(_) => true,
        })
    {
        return Err("equal-length sliding rows disagree on ring state".into());
    }

    let width = |array: &MlxArray| mlxcel_core::array_shape(array)[2];
    let target = width(arrays[longest].0);
    let pad = |array: &MlxArray, valid: i32| -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(array);
        if equal && shape[2] == target {
            return Ok(mlxcel_core::copy(array));
        }
        if valid > target || valid > shape[2] {
            return Err(format!(
                "row history {valid} does not fit the stacked width {target}"
            ));
        }
        let history = slice_axis(array, 2, 0, valid);
        if valid == target {
            return Ok(history);
        }
        let zeros = mlxcel_core::zeros(
            &[shape[0], shape[1], target - valid, shape[3]],
            mlxcel_core::array_dtype(array),
        );
        Ok(mlxcel_core::concatenate(&history, &zeros, 2))
    };
    let mut keys = Vec::with_capacity(arrays.len());
    let mut values = Vec::with_capacity(arrays.len());
    for (&(k, v), &valid) in arrays.iter().zip(&offsets) {
        keys.push(pad(k, valid)?);
        values.push(pad(v, valid)?);
    }
    let keys = concat_rows(&keys);
    let values = concat_rows(&values);

    let mut base = caches
        .into_iter()
        .nth(longest)
        .expect("longest row index is in range");
    match &mut base {
        Cache::Standard(c) => {
            c.keys = Some(keys);
            c.values = Some(values);
        }
        Cache::Rotating(c) => {
            c.keys = Some(keys);
            c.values = Some(values);
        }
    }
    Ok(base)
}

#[cfg(test)]
#[path = "gemma4_verify_rows_tests.rs"]
mod tests;
