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

//! Query-row attention for Gemma 4 MTP's unfused full-attention heads.
use mlxcel_core::{MlxArray, UniquePtr, utils::slice_axis};

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

#[cfg(test)]
#[path = "gemma4_attention_tests.rs"]
mod tests;
