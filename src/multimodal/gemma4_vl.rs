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

//! Gemma 4 VLM batching helpers (mlx-vlm PR #1127).
//!
//! This module owns the **per-row prompt-kwarg alignment** helpers used
//! when a batched VLM call ships a `[B, T_max, ...]` tensor (sequence-
//! aligned along axis 1) and the prompt builder needs to slice it row-by-
//! row and pad each slice to the target sequence length before
//! recombining. Upstream mlx-vlm PR #1127
//! (`_split_prompt_kwargs_per_row` + `_pad_sequence_aligned_prompt_kwarg`)
//! addresses the same problem in Python; the helpers below are the Rust
//! analogue.
//!
//! Per-row batched dispatch on `LanguageModel::forward_batched_with_context_and_ids`
//! lives in [`super::batched_dispatch::forward_batched_with_seq_ids_dispatch`]
//! and is shared with the Qwen VL families. Both Qwen VL and
//! Gemma 4's vision wrappers route their override through that helper so
//! mixed-length batches reach `forward_with_sequence_id` row-by-row with
//! the correct `seq_id`. After's runtime fix landed in this
//! same PR, [`crate::vision::Gemma4VLModel::supports_batching`] returns
//! `true` and the scheduler actually drives this batched path on
//! Gemma 4.
//!
//! The kwarg-alignment helpers here are scoped narrow enough that issue
//! (Gemma 4 E4B/E2B `per_layer_inputs` alignment in batched prefill)
//! can extend them with the 4D `per_layer_inputs` shape without
//! duplicating the row-slice / pad-to-max-length math.

// Re-export the shared per-row dispatch helper so existing call sites
// (`crate::vision::gemma4_vl`) and tests can keep importing it from this
// module. The implementation lives in `super::batched_dispatch`.
pub use super::batched_dispatch::forward_batched_with_seq_ids_dispatch;

use mlxcel_core::{MlxArray, UniquePtr};

/// Direction to pad a per-row tensor when its `T_i < T_target`.
///
/// - `Right` mirrors upstream `_pad_sequence_aligned_prompt_kwarg(left=False)`
///   used in the cold batched-prefill path.
/// - `Left` mirrors upstream `_pad_sequence_aligned_prompt_kwarg(left=True)`
///   used in the warm/mixed APC path where the suffix anchors to the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    Left,
    Right,
}

/// Slice row `row_idx` out of a batched tensor `[B, T, ...]`.
///
/// Mirrors upstream `_prompt_kwarg_row`:
/// - When the tensor is full-batched (`shape[0] == batch_size`), takes
///   `v[row_idx : row_idx + 1]`.
/// - When it is already broadcast-shaped (`shape[0] == 1`), returns the
///   single existing row regardless of `row_idx`.
///
/// Returns `None` when the tensor has rank 0 or an empty leading axis.
///
/// Used by: Gemma 4 mixed-length batching prep and — when
/// extended for `per_layer_inputs`.
#[must_use]
pub fn prompt_kwarg_row(
    v: &MlxArray,
    row_idx: usize,
    batch_size: usize,
) -> Option<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(v);
    if shape.is_empty() || shape[0] <= 0 {
        return None;
    }
    let leading = shape[0] as usize;

    let (start_row, end_row) = if leading == batch_size && batch_size > 1 {
        (row_idx as i32, row_idx as i32 + 1)
    } else {
        // Already a single-row (broadcast) tensor — keep row 0.
        (0, 1)
    };

    // Slice [start_row:end_row, :, :, ...] — preserve every other axis.
    let mut starts: Vec<i32> = vec![0; shape.len()];
    let mut stops: Vec<i32> = shape.clone();
    starts[0] = start_row;
    stops[0] = end_row;
    Some(mlxcel_core::slice(v, &starts, &stops))
}

/// Pad a per-row tensor along axis 1 (the sequence axis) up to
/// `target_length`. Padding is filled with zeros in the same dtype as `v`.
///
/// Matches upstream `_pad_sequence_aligned_prompt_kwarg`. Returns the
/// original tensor unchanged when the current length already meets or
/// exceeds `target_length`.
///
/// `v` must have rank >= 2; the seq axis is axis 1 by convention. This
/// covers all sequence-aligned prompt kwargs the upstream fix touches:
/// 2D (`attention_mask`, `position_ids`), 3D (`inputs_embeds`,
/// `decoder_inputs_embeds`, `deepstack_visual_embeds`), and 4D
/// (`per_layer_inputs`, the shape that will exercise).
///
/// # Panics
/// Panics if `v` has rank < 2 — sequence-aligned kwargs always have at
/// least a `[B, T]` shape, so a smaller rank is a programmer error.
///
/// Used by: Gemma 4 mixed-length batching prep; designed
/// for re-use.
#[must_use]
pub fn pad_sequence_aligned_prompt_kwarg(
    v: &MlxArray,
    target_length: i32,
    side: PadSide,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(v);
    assert!(
        shape.len() >= 2,
        "pad_sequence_aligned_prompt_kwarg: rank must be >= 2; got {shape:?}"
    );
    let current = shape[1];
    if current >= target_length {
        return mlxcel_core::copy(v);
    }
    let pad = target_length - current;

    // Build the padding tensor with the same shape as `v` except axis 1
    // replaced with `pad`. Padding uses zeros in the same dtype.
    let mut pad_shape = shape.clone();
    pad_shape[1] = pad;
    let dtype = mlxcel_core::array_dtype(v);
    let pad_arr = mlxcel_core::zeros(&pad_shape, dtype);

    match side {
        PadSide::Right => mlxcel_core::concatenate(v, pad_arr.as_ref().unwrap(), 1),
        PadSide::Left => mlxcel_core::concatenate(pad_arr.as_ref().unwrap(), v, 1),
    }
}

/// Convenience: combine [`prompt_kwarg_row`] and
/// [`pad_sequence_aligned_prompt_kwarg`] for a single row.
///
/// Returns the per-row, padded tensor when slicing succeeds. Callers
/// that need different padding semantics for non-sequence-aligned
/// kwargs (e.g. `pixel_values` shaped `[B, ...]`) should call
/// [`prompt_kwarg_row`] directly and skip the padding step.
///
/// Used by: Gemma 4 mixed-length batching prep; re-used
/// for `per_layer_inputs`.
#[must_use]
pub fn align_per_row_prompt_kwarg(
    v: &MlxArray,
    row_idx: usize,
    batch_size: usize,
    target_length: i32,
    side: PadSide,
) -> Option<UniquePtr<MlxArray>> {
    let row = prompt_kwarg_row(v, row_idx, batch_size)?;
    let row_shape = mlxcel_core::array_shape(row.as_ref().unwrap());
    if row_shape.len() < 2 {
        // Rank-1 (or rank-0) leading-batch tensors do not have a seq axis
        // to pad. Return the row as-is.
        return Some(row);
    }
    Some(pad_sequence_aligned_prompt_kwarg(
        row.as_ref().unwrap(),
        target_length,
        side,
    ))
}

#[cfg(test)]
#[path = "gemma4_vl_tests.rs"]
mod tests;
