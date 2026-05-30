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

//! Structural unit tests for the Gemma 4 [`MtpTarget`] adapter
//!
//! These tests pin the adapter's **trait surface** without loading real
//! Gemma 4 weights. The full on-hardware greedy-parity test for the
//! adapter ships in `tests/speculative_parity.rs` and is gated behind
//! `#[ignore]` so CI hosts without the model checkpoints don't red-flag
//! the build.
//!
//! ## What this file pins
//!
//! 1. `Gemma4MtpTargetAdapter::new` constructs cleanly with a `&'a
//!    Gemma4Wrapper` and `Option<SequenceId>`.
//! 2. The adapter is `Send`-able only in the same way `&Gemma4Wrapper`
//!    is — i.e. the lifetime bound is properly threaded.
//! 3. The shared-K/V slicing helper (`slice_shared_kv`) returns the
//!    input vector unchanged when `rejected == 0` (the common
//!    full-accept fast-path).
//!
//! Tests that need to actually call `forward_with_speculative_sinks`
//! require a loaded model and live in `tests/speculative_parity.rs`.

use super::*;

#[test]
fn mtp_rotating_buffer_size_matches_upstream_clamp() {
    assert_eq!(mtp_rotating_buffer_size(1), 32);
    assert_eq!(mtp_rotating_buffer_size(4), 32);
    assert_eq!(mtp_rotating_buffer_size(8), 64);
    assert_eq!(mtp_rotating_buffer_size(16), 128);
    assert_eq!(mtp_rotating_buffer_size(64), 128);
}

#[test]
fn slice_shared_kv_with_zero_rejected_is_identity() {
    // Build a synthetic 4-tensor shared K/V vector to verify the
    // fast-path. We use the FFI `from_slice_f32` to build small tensors;
    // since `rejected == 0` the slice helper must return them unchanged.
    let _runtime = crate::initialize_runtime();

    let make =
        || mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make(), make(), make()];
    let original_shapes: Vec<Vec<i32>> = tensors
        .iter()
        .map(|t| mlxcel_core::array_shape(t.as_ref().unwrap()))
        .collect();

    let sliced = Gemma4MtpTargetAdapter::slice_shared_kv(tensors, 0);
    assert_eq!(sliced.len(), 4);
    for (i, s) in sliced.iter().enumerate() {
        let shape = mlxcel_core::array_shape(s.as_ref().unwrap());
        assert_eq!(
            shape, original_shapes[i],
            "rejected=0 must return identical shapes; entry {i} drifted"
        );
    }
}

#[test]
fn slice_shared_kv_with_rejected_one_shrinks_kv_axis() {
    // Build a `[1, 2, 4, 2]` synthetic tensor (B=1, num_kv_heads=2,
    // kv_len=4, head_dim=2). `rejected = 1` should produce
    // `[1, 2, 3, 2]`.
    let _runtime = crate::initialize_runtime();

    let make = || {
        // Total cells = 1 (batch) * 2 (heads) * 4 (kv_len) * 2 (head_dim) = 16
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[1, 2, 4, 2])
    };
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make()];
    let sliced = Gemma4MtpTargetAdapter::slice_shared_kv(tensors, 1);
    assert_eq!(sliced.len(), 2);
    for s in &sliced {
        let shape = mlxcel_core::array_shape(s.as_ref().unwrap());
        assert_eq!(
            shape,
            vec![1, 2, 3, 2],
            "rejected=1 must shrink kv_len axis from 4 to 3"
        );
    }
}

#[test]
fn argmax_per_position_returns_one_id_per_position() {
    // `[1, 3, 4]` logits with deterministic per-row argmax:
    //   row 0: max at index 2 -> argmax = 2
    //   row 1: max at index 0 -> argmax = 0
    //   row 2: max at index 3 -> argmax = 3
    let _runtime = crate::initialize_runtime();

    let data: Vec<f32> = vec![
        // row 0
        0.1, 0.2, 0.9, 0.3, // row 1
        0.9, 0.1, 0.2, 0.3, // row 2
        0.1, 0.2, 0.3, 0.9,
    ];
    let logits = mlxcel_core::from_slice_f32(&data, &[1, 3, 4]);
    let argmax = Gemma4MtpTargetAdapter::argmax_per_position(logits.as_ref().unwrap());
    assert_eq!(argmax, vec![2, 0, 3]);
}

// ===========================================================================
// Batched MTP target adapter helper tests
//
// These pin the pure-tensor helper methods of `Gemma4MtpBatchedTargetAdapter`
// without loading real Gemma 4 weights. The full on-hardware B = 4
// byte-identical greedy-parity test ships in `tests/speculative_parity.rs`
// gated behind `#[ignore]`.
// ===========================================================================

#[test]
fn batched_rectangular_input_builds_b_by_width_tensor() {
    let _runtime = crate::initialize_runtime();
    // 3 rows, width 4 — a valid rectangular batch.
    let per_row = vec![
        vec![10, 11, 12, 13],
        vec![20, 21, 22, 23],
        vec![30, 31, 32, 33],
    ];
    let (arr, width) =
        Gemma4MtpBatchedTargetAdapter::rectangular_input(&per_row, 3).expect("rectangular");
    assert_eq!(width, 4);
    let shape = mlxcel_core::array_shape(arr.as_ref().unwrap());
    assert_eq!(shape, vec![3, 4], "must build a [B, width] tensor");
}

#[test]
fn batched_rectangular_input_rejects_variable_width_rows() {
    let _runtime = crate::initialize_runtime();
    // Row 1 is shorter — the batched verify forward requires a
    // rectangular input, so this must error.
    // `rectangular_input`'s `Ok` variant carries a `UniquePtr<MlxArray>`
    // (not `Debug`), so we `match` rather than `expect_err`.
    let per_row = vec![vec![10, 11, 12, 13], vec![20, 21], vec![30, 31, 32, 33]];
    let msg = match Gemma4MtpBatchedTargetAdapter::rectangular_input(&per_row, 3) {
        Ok(_) => panic!("variable-width rows must be rejected"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("rectangular") || msg.contains("width"),
        "error must explain the rectangular-input requirement, got: {msg}"
    );
}

#[test]
fn batched_rectangular_input_rejects_batch_size_mismatch() {
    let _runtime = crate::initialize_runtime();
    let per_row = vec![vec![10, 11], vec![20, 21]];
    // Adapter expects 3 rows but only 2 supplied.
    let msg = match Gemma4MtpBatchedTargetAdapter::rectangular_input(&per_row, 3) {
        Ok(_) => panic!("batch-size mismatch must be rejected"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("3"),
        "error must mention the expected batch size, got: {msg}"
    );
}

#[test]
fn batched_argmax_per_row_returns_b_by_width_ids() {
    let _runtime = crate::initialize_runtime();
    // `[2, 3, 4]` logits. Row 0: argmax per position = [2, 0, 3].
    //                     Row 1: argmax per position = [1, 3, 0].
    let data: Vec<f32> = vec![
        // row 0
        0.1, 0.2, 0.9, 0.3, // pos 0 -> 2
        0.9, 0.1, 0.2, 0.3, // pos 1 -> 0
        0.1, 0.2, 0.3, 0.9, // pos 2 -> 3
        // row 1
        0.1, 0.9, 0.2, 0.3, // pos 0 -> 1
        0.1, 0.2, 0.3, 0.9, // pos 1 -> 3
        0.9, 0.1, 0.2, 0.3, // pos 2 -> 0
    ];
    let logits = mlxcel_core::from_slice_f32(&data, &[2, 3, 4]);
    let argmax = Gemma4MtpBatchedTargetAdapter::argmax_per_row(logits.as_ref().unwrap(), 2, 3);
    assert_eq!(argmax, vec![vec![2, 0, 3], vec![1, 3, 0]]);
}

#[test]
fn batched_slice_shared_kv_zero_rejected_is_identity() {
    let _runtime = crate::initialize_runtime();
    // `[2, 2, 4, 2]` slabs (B=2). rejected = 0 must return unchanged.
    let make = || {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[2, 2, 4, 2])
    };
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make(), make(), make()];
    let original: Vec<Vec<i32>> = tensors
        .iter()
        .map(|t| mlxcel_core::array_shape(t.as_ref().unwrap()))
        .collect();
    let sliced = Gemma4MtpBatchedTargetAdapter::slice_shared_kv_batched(tensors, 0);
    assert_eq!(sliced.len(), 4);
    for (i, s) in sliced.iter().enumerate() {
        assert_eq!(
            mlxcel_core::array_shape(s.as_ref().unwrap()),
            original[i],
            "rejected=0 must return identical shapes; entry {i} drifted"
        );
    }
}

#[test]
fn batched_slice_shared_kv_rejected_shrinks_kv_axis_keeping_batch() {
    let _runtime = crate::initialize_runtime();
    // `[3, 2, 5, 2]` slabs (B=3, kv_len=5). rejected = 2 -> [3, 2, 3, 2].
    let make = || {
        let data: Vec<f32> = (0..(3 * 2 * 5 * 2)).map(|i| i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[3, 2, 5, 2])
    };
    let tensors: Vec<UniquePtr<MlxArray>> = vec![make(), make()];
    let sliced = Gemma4MtpBatchedTargetAdapter::slice_shared_kv_batched(tensors, 2);
    for s in &sliced {
        assert_eq!(
            mlxcel_core::array_shape(s.as_ref().unwrap()),
            vec![3, 2, 3, 2],
            "rejected=2 must shrink only the kv_len axis (5 -> 3), batch dim stays 3"
        );
    }
}

#[test]
fn batched_last_position_hidden_slices_to_b_by_one() {
    let _runtime = crate::initialize_runtime();
    // `[2, 4, 3]` hidden -> last-position slice `[2, 1, 3]`.
    let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
    let hidden = mlxcel_core::from_slice_f32(&data, &[2, 4, 3]);
    let last = Gemma4MtpBatchedTargetAdapter::last_position_hidden(hidden.as_ref().unwrap());
    let shape = mlxcel_core::array_shape(last.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![2, 1, 3],
        "must slice [B, T, H] down to [B, 1, H]"
    );
}

#[test]
fn mtp_hidden_at_position_slices_single_position() {
    let _runtime = crate::initialize_runtime();
    // `[1, 4, 3]` hidden -> position 2 slice `[1, 1, 3]`.
    let data: Vec<f32> = (0..(4 * 3)).map(|i| i as f32).collect();
    let hidden = mlxcel_core::from_slice_f32(&data, &[1, 4, 3]);
    let selected = Gemma4MtpTargetAdapter::hidden_at_position(hidden.as_ref().unwrap(), 2);
    let shape = mlxcel_core::array_shape(selected.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![1, 1, 3],
        "B=1 MTP target must pass a singleton-position hidden state to the drafter"
    );
}

#[test]
fn batched_hidden_at_positions_slices_per_row() {
    let _runtime = crate::initialize_runtime();
    // `[2, 4, 3]` hidden with different accepted positions per row
    // must still produce `[2, 1, 3]`.
    let data: Vec<f32> = (0..(2 * 4 * 3)).map(|i| i as f32).collect();
    let hidden = mlxcel_core::from_slice_f32(&data, &[2, 4, 3]);
    let selected = Gemma4MtpBatchedTargetAdapter::hidden_at_positions_batched(
        hidden.as_ref().unwrap(),
        &[1, 3],
    )
    .expect("per-row hidden slice");
    let shape = mlxcel_core::array_shape(selected.as_ref().unwrap());
    assert_eq!(
        shape,
        vec![2, 1, 3],
        "Batched MTP target must select each row's accepted hidden position"
    );
}

#[test]
fn batched_scalar_tokens_per_row_extracts_one_per_row() {
    let _runtime = crate::initialize_runtime();
    // A `[3, 1]` token tensor (the shape `sample_token_optimized`
    // returns for a 3-row batch).
    let token_arr = mlxcel_core::from_slice_i32(&[42, 7, 99], &[3, 1]);
    let tokens = scalar_tokens_per_row(token_arr.as_ref().unwrap(), 3);
    assert_eq!(tokens, vec![42, 7, 99]);
}

#[test]
fn batched_capture_layer_ids_is_last_layer_only() {
    // The batched adapter must capture the last decoder layer's pre-norm
    // hidden state only (`None`), matching the B = 1 adapter's
    // `forward_with_speculative_sinks` call shape. A regression here
    // (e.g. someone hard-coding the DFlash `[1, 8, 15, 22, 29]` list)
    // would change `MtpBatchedVerifyOutput::next_hidden`'s feature dim.
    assert!(
        BATCHED_CAPTURE_LAYER_IDS.is_none(),
        "batched MTP adapter must capture the last layer only (None)"
    );
}
