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

//! Unit tests for the Gemma 4 mixed-length batching helpers.
//!
//! These tests cover two distinct surfaces:
//!
//! 1. The per-row prompt-kwarg alignment helpers ([`prompt_kwarg_row`],
//!    [`pad_sequence_aligned_prompt_kwarg`], [`align_per_row_prompt_kwarg`]).
//!    A follow-up will extend the same helpers to handle Gemma 4 E4B/E2B
//!    `per_layer_inputs` (shape `[B, T, num_layers, hidden_per_layer]`),
//!    so the 4D-shape coverage here is also forward-looking guard rails
//!    for that follow-up.
//!
//! 2. The per-row batched dispatch helper
//!    ([`forward_batched_with_seq_ids_dispatch`]). Mirrors the integration
//!    tests added for Qwen VL.

use super::{
    PadSide, align_per_row_prompt_kwarg, forward_batched_with_seq_ids_dispatch,
    pad_sequence_aligned_prompt_kwarg, prompt_kwarg_row,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};
use std::cell::RefCell;

// -- prompt_kwarg_row --------------------------------------------------

/// Pull a `Vec<i32>` snapshot of an MLX tensor for assertions. Evaluates
/// the array first so the resulting bytes are materialized.
fn arr_to_vec_i32(arr: &MlxArray) -> Vec<i32> {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn arr_to_vec_f32(arr: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "requires serial MLX execution"]
fn prompt_kwarg_row_extracts_each_row_of_a_full_batch() {
    // 3D `inputs_embeds`-shaped tensor `[B=3, T=2, H=2]`.
    let data: Vec<f32> = (0..3 * 2 * 2).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[3, 2, 2]);

    for row in 0..3 {
        let sliced = prompt_kwarg_row(v.as_ref().unwrap(), row, 3).unwrap();
        let shape = mlxcel_core::array_shape(sliced.as_ref().unwrap());
        assert_eq!(
            shape,
            vec![1, 2, 2],
            "row {row} must produce shape [1, T, H]"
        );

        let values = arr_to_vec_f32(sliced.as_ref().unwrap());
        // Row `row` covers data[row * 4 .. row * 4 + 4].
        let expected: Vec<f32> = data[row * 4..(row + 1) * 4].to_vec();
        assert_eq!(values, expected, "row {row} contents wrong");
    }
}

#[test]
#[ignore = "requires serial MLX execution"]
fn prompt_kwarg_row_handles_4d_per_layer_inputs_shape() {
    // 4D `per_layer_inputs`-shaped tensor `[B=2, T=3, L=2, H=4]`. Issue
    // will exercise this exact rank for Gemma 4 E4B/E2B.
    let total = 2 * 3 * 2 * 4;
    let data: Vec<f32> = (0..total).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[2, 3, 2, 4]);

    let row0 = prompt_kwarg_row(v.as_ref().unwrap(), 0, 2).unwrap();
    let row1 = prompt_kwarg_row(v.as_ref().unwrap(), 1, 2).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(row0.as_ref().unwrap()),
        vec![1, 3, 2, 4]
    );
    assert_eq!(
        mlxcel_core::array_shape(row1.as_ref().unwrap()),
        vec![1, 3, 2, 4]
    );

    // Every element of row 1 should equal `original + (3 * 2 * 4)`.
    let row0_values = arr_to_vec_f32(row0.as_ref().unwrap());
    let row1_values = arr_to_vec_f32(row1.as_ref().unwrap());
    let stride = 3 * 2 * 4;
    for i in 0..stride {
        assert!((row1_values[i] - row0_values[i] - stride as f32).abs() < 1e-5);
    }
}

#[test]
#[ignore = "requires serial MLX execution"]
fn prompt_kwarg_row_returns_single_row_for_broadcast_shape() {
    // When `shape[0] == 1` the helper returns the single row regardless of
    // the requested `row_idx`. Mirrors upstream's `v[:1]` fallback.
    let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[1, 3, 2]);

    for row_idx in 0..3 {
        let sliced = prompt_kwarg_row(v.as_ref().unwrap(), row_idx, 3).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(sliced.as_ref().unwrap()),
            vec![1, 3, 2]
        );
        let values = arr_to_vec_f32(sliced.as_ref().unwrap());
        assert_eq!(values, data, "broadcast row content must be unchanged");
    }
}

#[test]
#[ignore = "requires serial MLX execution"]
fn prompt_kwarg_row_returns_none_for_empty_leading_axis() {
    // Defensive: the helper must not crash when handed a `[0, ...]` tensor.
    let v = mlxcel_core::zeros(&[0, 3, 2], mlxcel_core::dtype::FLOAT32);
    assert!(prompt_kwarg_row(v.as_ref().unwrap(), 0, 0).is_none());
}

// -- pad_sequence_aligned_prompt_kwarg ----------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn pad_sequence_aligned_right_pads_2d_attention_mask() {
    // 2D attention-mask shape: `[1, T]`. Pad from T=2 to T=4 with zeros
    // on the right (cold prefill path).
    let v = mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]);
    let padded = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 4, PadSide::Right);
    assert_eq!(
        mlxcel_core::array_shape(padded.as_ref().unwrap()),
        vec![1, 4]
    );
    let values = arr_to_vec_i32(padded.as_ref().unwrap());
    assert_eq!(values, vec![1, 1, 0, 0]);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn pad_sequence_aligned_left_pads_2d_attention_mask() {
    // Mixed APC path uses left-padding so the suffix anchors right.
    let v = mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]);
    let padded = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 4, PadSide::Left);
    assert_eq!(
        mlxcel_core::array_shape(padded.as_ref().unwrap()),
        vec![1, 4]
    );
    let values = arr_to_vec_i32(padded.as_ref().unwrap());
    assert_eq!(values, vec![0, 0, 1, 1]);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn pad_sequence_aligned_right_pads_3d_inputs_embeds() {
    // 3D `inputs_embeds`-shaped tensor `[1, T=2, H=3]`. Pad to T=4.
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[1, 2, 3]);

    let padded = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 4, PadSide::Right);
    assert_eq!(
        mlxcel_core::array_shape(padded.as_ref().unwrap()),
        vec![1, 4, 3]
    );

    let values = arr_to_vec_f32(padded.as_ref().unwrap());
    // First two seq positions = original [1..6], next two = zeros [0..6].
    assert_eq!(
        values,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn pad_sequence_aligned_left_pads_4d_per_layer_inputs() {
    // 4D `per_layer_inputs`-shaped tensor `[1, T=2, L=2, H=2]`. Pad to T=3.
    // This is the exact rank will exercise — the helper must
    // already produce correct output for this case so only has to
    // touch the call site (not re-derive the math).
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[1, 2, 2, 2]);

    let padded = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 3, PadSide::Left);
    assert_eq!(
        mlxcel_core::array_shape(padded.as_ref().unwrap()),
        vec![1, 3, 2, 2]
    );

    let values = arr_to_vec_f32(padded.as_ref().unwrap());
    // First seq position = zeros (4 elements), next two seq positions =
    // original data [1..8]. Total 12 elements.
    let expected: Vec<f32> = (0..4).map(|_| 0.0).chain(data.iter().copied()).collect();
    assert_eq!(values, expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn pad_sequence_aligned_is_noop_when_target_already_met() {
    // `target_length <= current_length` must return a tensor with the
    // same shape and contents as the input.
    let v = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);
    let padded = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 2, PadSide::Right);
    assert_eq!(
        mlxcel_core::array_shape(padded.as_ref().unwrap()),
        vec![1, 3]
    );
    assert_eq!(arr_to_vec_i32(padded.as_ref().unwrap()), vec![1, 1, 1]);

    let exact = pad_sequence_aligned_prompt_kwarg(v.as_ref().unwrap(), 3, PadSide::Right);
    assert_eq!(
        mlxcel_core::array_shape(exact.as_ref().unwrap()),
        vec![1, 3]
    );
}

// -- align_per_row_prompt_kwarg ----------------------------------------

#[test]
#[ignore = "requires serial MLX execution"]
fn align_per_row_combines_slice_and_pad_for_inputs_embeds() {
    // Build a `[B=2, T=3, H=2]` batched tensor. After per-row alignment
    // to target length 4 with right-padding, each row's slice is
    // `[1, 3, 2]` then padded out to `[1, 4, 2]`.
    let data: Vec<f32> = (0..2 * 3 * 2).map(|x| x as f32).collect();
    let v = mlxcel_core::from_slice_f32(&data, &[2, 3, 2]);

    let row0 = align_per_row_prompt_kwarg(v.as_ref().unwrap(), 0, 2, 4, PadSide::Right).unwrap();
    let row1 = align_per_row_prompt_kwarg(v.as_ref().unwrap(), 1, 2, 4, PadSide::Right).unwrap();

    assert_eq!(
        mlxcel_core::array_shape(row0.as_ref().unwrap()),
        vec![1, 4, 2]
    );
    assert_eq!(
        mlxcel_core::array_shape(row1.as_ref().unwrap()),
        vec![1, 4, 2]
    );

    let row0_values = arr_to_vec_f32(row0.as_ref().unwrap());
    let row1_values = arr_to_vec_f32(row1.as_ref().unwrap());
    // Each row's flat layout: 3 real seq positions of 2 hidden = 6
    // elements, then 1 padding seq position of 2 hidden = 2 zeros.
    // Total 8 elements per row matches the shape `[1, 4, 2]`.
    let mut expected_row0 = data[0..6].to_vec();
    expected_row0.extend([0.0; 2]);
    assert_eq!(row0_values, expected_row0);

    let mut expected_row1 = data[6..12].to_vec();
    expected_row1.extend([0.0; 2]);
    assert_eq!(row1_values, expected_row1);
}

// -- forward_batched_with_seq_ids_dispatch -----------------------------

/// Minimal `LanguageModel` stub that records every
/// `(seq_id_or_neg1, row_value)` pair it observes via
/// `forward_with_sequence_id`. Same pattern used in `qwen_vl_tests.rs`.
struct StubTextModel {
    calls: RefCell<Vec<(i64, i32)>>,
}

impl StubTextModel {
    fn new() -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(i64, i32)> {
        self.calls.borrow().clone()
    }
}

impl LanguageModel for StubTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let row_val = read_first_i32(input_ids);
        // No seq id — record with sentinel -1.
        self.calls.borrow_mut().push((-1, row_val));
        // Shape `[1, 1, 1]` so concatenate along axis 0 produces `[B, 1, 1]`.
        mlxcel_core::from_slice_f32(&[row_val as f32], &[1, 1, 1])
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let row_val = read_first_i32(input_ids);
        let id = seq_id.map(|s| s.as_u64() as i64).unwrap_or(-1);
        self.calls.borrow_mut().push((id, row_val));
        mlxcel_core::from_slice_f32(&[row_val as f32], &[1, 1, 1])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

fn read_first_i32(arr: &MlxArray) -> i32 {
    mlxcel_core::eval(arr);
    let head = mlxcel_core::slice(arr, &[0, 0], &[1, 1]);
    mlxcel_core::eval(&head);
    mlxcel_core::item_i32(&head)
}

/// Mixed-length batch dispatch: each row's `seq_id` must reach the
/// stub model's `forward_with_sequence_id` independently.
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_routes_each_row_to_its_seq_id() {
    let model = StubTextModel::new();

    // input_ids `[B=2, T=1]` with row 0 = 100, row 1 = 200.
    let input_ids = mlxcel_core::from_slice_i32(&[100, 200], &[2, 1]);

    let mut row0_caches: Vec<KVCache> = Vec::new();
    let mut row1_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> =
        vec![row0_caches.as_mut_slice(), row1_caches.as_mut_slice()];

    let seq_ids = [SequenceId::from_raw(7), SequenceId::from_raw(13)];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        Some(&seq_ids),
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    let calls = model.calls();
    assert_eq!(
        calls.len(),
        2,
        "must dispatch once per row, got {:?}",
        calls
    );
    assert_eq!(calls[0], (7, 100));
    assert_eq!(calls[1], (13, 200));

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(shape, vec![2, 1, 1]);
}

/// Single-row fast path with `seq_ids`: must still route through
/// `forward_with_sequence_id` so the model's per-sequence state resolves.
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_single_row_uses_forward_with_sequence_id() {
    let model = StubTextModel::new();
    let input_ids = mlxcel_core::from_slice_i32(&[42], &[1, 1]);
    let mut row_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> = vec![row_caches.as_mut_slice()];
    let seq_ids = [SequenceId::from_raw(99)];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        Some(&seq_ids),
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], (99, 42));
}

/// `seq_ids = None` must fall through to `forward_batched_with_context`.
/// CLI/single-process callers (e.g. the legacy `mlxcel generate` path)
/// rely on this fallback so they never have to plumb a fake seq id.
#[test]
#[ignore = "requires serial MLX execution"]
fn forward_batched_with_seq_ids_dispatch_no_seq_ids_falls_through_to_batched() {
    let model = StubTextModel::new();
    let input_ids = mlxcel_core::from_slice_i32(&[7, 8], &[2, 1]);
    let mut row0_caches: Vec<KVCache> = Vec::new();
    let mut row1_caches: Vec<KVCache> = Vec::new();
    let mut batch_caches: Vec<&mut [KVCache]> =
        vec![row0_caches.as_mut_slice(), row1_caches.as_mut_slice()];

    let logits = forward_batched_with_seq_ids_dispatch(
        &model,
        &input_ids,
        None,
        batch_caches.as_mut_slice(),
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    // Without seq_ids the trait default `forward_batched_with_context`
    // routes through `forward_batched`, which loops calling `forward()` —
    // so each call is logged under the sentinel id `-1`.
    let calls = model.calls();
    assert_eq!(
        calls,
        vec![(-1, 7), (-1, 8)],
        "no seq_ids must use forward (no id) per row; got {:?}",
        calls
    );
}
