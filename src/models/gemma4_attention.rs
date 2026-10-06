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

//! Query-row attention for Gemma 4 MTP verify on the row-wise geometries.
use mlxcel_core::{MlxArray, UniquePtr, layers::RMSNorm, utils::slice_axis};

/// Masked sibling/padding slots change the physical reduction width. This
/// correction establishes chain parity only for a single chronological row.
pub(super) fn linear_verify_layout(
    batch: i32,
    tree_positions: Option<&[i32]>,
    padded_or_divergent: bool,
) -> bool {
    batch == 1
        && !padded_or_divergent
        && tree_positions.is_none_or(|positions| {
            positions
                .iter()
                .enumerate()
                .all(|(index, position)| *position == index as i32)
        })
}

/// Row-wise verify attention for the geometries where a batched M=K block
/// and classic M=1 decode select different kernels or reductions (31B
/// everywhere, 12B on CUDA; see `TextConfig::mtp_requires_linear_singleton`).
///
/// `window == 0` is a full-attention layer. A sliding layer whose cache holds
/// the MTP rotating buffer reports a `ring_cursor` and gets the physical ring
/// order classic decode reduces over. Without a cursor (no buffer) the keys
/// match classic order only until the ring wraps, so that case is taken only
/// while every key fits in the window; serving and the startup probe always
/// enable the buffer for these geometries, so it is a fallback, not the
/// serving path. `None` means the caller keeps its ordinary dispatch (a
/// single full-attention query is already the decode shape).
pub(super) fn attend_verify_rows(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    window: i32,
    ring_cursor: Option<i32>,
) -> Option<UniquePtr<MlxArray>> {
    let query_len = mlxcel_core::array_shape(queries)[2];
    if window > 0
        && let Some(cursor) = ring_cursor
    {
        return Some(attend_ring_rows(
            queries, keys, values, scale, window, cursor,
        ));
    }
    let key_len = mlxcel_core::array_shape(keys)[2];
    (query_len > 1 && (window == 0 || key_len <= window))
        .then(|| attend_query_rows(queries, keys, values, None, scale, window))
}

/// Preserve every structural mask constraint while matching the decode
/// matmul geometry. Future draft keys are physically excluded: masked zeros
/// alone still change a reduction's width and can change its last bit.
pub(super) fn attend_query_rows(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    mask: Option<&MlxArray>,
    scale: f32,
    window: i32,
) -> UniquePtr<MlxArray> {
    let query_len = mlxcel_core::array_shape(queries)[2];
    let key_len = mlxcel_core::array_shape(keys)[2];
    let mut output = Vec::with_capacity(query_len as usize);
    for position in 0..query_len {
        let end = key_len - query_len + position + 1;
        let query = slice_axis(queries, 2, position, position + 1);
        let key = slice_axis(keys, 2, 0, end);
        let value = slice_axis(values, 2, 0, end);
        let row_mask = mask.map(|mask| {
            let shape = mlxcel_core::array_shape(mask);
            let row = if shape[shape.len() - 2] == 1 {
                0
            } else {
                position
            };
            let mask = slice_axis(mask, -2, row, row + 1);
            slice_axis(&mask, -1, 0, end)
        });
        let row = if let Some(mask) = row_mask.as_ref() {
            mlxcel_core::layers::attention(&query, &key, &value, scale, mask.as_ref(), 0.0, window)
        } else {
            mlxcel_core::causal_attention(&query, &key, &value, scale, 0.0, window)
        };
        output.push(row);
    }
    let rows: Vec<&MlxArray> = output
        .iter()
        .map(|row| row.as_ref().expect("attention result is non-null"))
        .collect();
    mlxcel_core::concatenate_many(&rows, 2)
}

/// Buffered keys are chronological; classic decode reduces over the physical
/// ring. Recreate that exact order separately for each query, after physically
/// dropping keys outside its window (including later draft positions).
pub(super) fn attend_ring_rows(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    window: i32,
    cursor: i32,
) -> UniquePtr<MlxArray> {
    let query_len = mlxcel_core::array_shape(queries)[2];
    let key_len = mlxcel_core::array_shape(keys)[2];
    let mut output = Vec::with_capacity(query_len as usize);
    for position in 0..query_len {
        let end = key_len - query_len + position + 1;
        let start = (end - window).max(0);
        let query = slice_axis(queries, 2, position, position + 1);
        let reorder = |array: &MlxArray| {
            let visible = slice_axis(array, 2, start, end);
            let next = (cursor + position + 1).rem_euclid(window);
            if end - start == window && next > 0 {
                let tail = slice_axis(&visible, 2, window - next, window);
                let head = slice_axis(&visible, 2, 0, window - next);
                mlxcel_core::concatenate(&tail, &head, 2)
            } else {
                visible
            }
        };
        let key = reorder(keys);
        let value = reorder(values);
        // Every physical slot is visible to this M=1 query; a causal/window
        // mask here would interpret ring order as chronology.
        output.push(mlxcel_core::layers::attention(
            &query, &key, &value, scale, None, 0.0, 0,
        ));
    }
    let rows: Vec<&MlxArray> = output.iter().map(|row| row.as_ref().unwrap()).collect();
    mlxcel_core::concatenate_many(&rows, 2)
}

impl super::Attention {
    /// Project one head branch (Q or K) of a verify block row by row, each
    /// row through exactly the call classic decode makes for that token.
    ///
    /// On CUDA, MLX rotates a `B = 1, L = 1` row-contiguous input with its
    /// `rope_single` kernel and anything wider with the general `rope` kernel,
    /// and the two disagree in the last float bit (measured on GB10, #2160).
    /// After bf16 rounding that flips a few query or key elements per layer,
    /// which is enough to break verify-vs-chain byte identity. Rotating each
    /// row at `offset + row` as its own `L = 1` call reproduces decode's
    /// kernel choice; norm and projection are row-wise already.
    pub(super) fn head_rows_like_decode(
        &self,
        raw: &MlxArray,
        norm: &RMSNorm,
        heads: i32,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let width = mlxcel_core::array_shape(raw)[1];
        let rows: Vec<UniquePtr<MlxArray>> = (0..width)
            .map(|row| {
                let raw_row = slice_axis(raw, 1, row, row + 1);
                self.head_branch(&raw_row, norm, heads, offset + row)
            })
            .collect();
        let rows: Vec<&MlxArray> = rows
            .iter()
            .map(|row| row.as_ref().expect("head branch result is non-null"))
            .collect();
        mlxcel_core::concatenate_many(&rows, 2)
    }

    /// The uniform `reshape -> norm -> transpose -> RoPE` chain, `[B, L, H*D]`
    /// in and `[B, H, L, D]` out: the compiled proportional chain on
    /// full-attention layers, the op-at-a-time chain otherwise.
    fn head_branch(
        &self,
        raw: &MlxArray,
        norm: &RMSNorm,
        heads: i32,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2
                * ((self.proportional_partial_rotary_factor as f64 * self.head_dim as f64 / 2.0)
                    .floor() as i32)
                    .max(0);
            return mlxcel_core::compiled_q_path_proportional(
                raw,
                &norm.weight,
                freqs,
                norm.eps,
                heads,
                self.head_dim,
                rotated_dims,
                offset,
            );
        }
        let shape = mlxcel_core::array_shape(raw);
        let heads_in = mlxcel_core::reshape(raw, &[shape[0], shape[1], heads, self.head_dim]);
        let normed = norm.forward(&heads_in);
        let transposed = mlxcel_core::transpose_axes(&normed, &[0, 2, 1, 3]);
        self.apply_rope(&transposed, offset)
    }
}

#[cfg(test)]
#[path = "gemma4_attention_tests.rs"]
mod tests;
