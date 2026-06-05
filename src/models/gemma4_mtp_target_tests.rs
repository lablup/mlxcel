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

// ===========================================================================
// Gemma 4 Unified MTP adapter — structural / trait-surface pins (issue #154)
//
// The Unified adapters are pure delegators to `Gemma4MtpTargetAdapter` /
// `Gemma4MtpBatchedTargetAdapter` over `unified.text_model`, mirroring the VL
// adapters. Constructing a real `Gemma4UnifiedModel` requires the encoder-free
// vision embedder + multimodal embedder + processor, which are out of scope
// for a weights-free unit test, so the full forward-driving greedy-parity
// round-trip ships in `tests/speculative_parity.rs`
// (`greedy_parity_mtp_gemma4_unified_12b`, `#[ignore]`). These pins guarantee
// the delegation types are wired correctly at compile time.
// ===========================================================================

/// The Unified B = 1 adapter must implement [`MtpTarget`] and be usable as a
/// trait object — the burst dispatch drives it via a generic `T: MtpTarget`,
/// and the type must satisfy that bound for any borrowed `&Gemma4UnifiedModel`.
#[test]
fn unified_b1_adapter_implements_mtp_target() {
    // Type-level assertion: a no-op function that only type-checks if the
    // adapter implements `MtpTarget` for an arbitrary borrow lifetime. Never
    // called — it exists purely to fail compilation on a broken impl.
    #[allow(dead_code)]
    fn assert_mtp_target<T: MtpTarget>() {}
    let _ = assert_mtp_target::<Gemma4UnifiedMtpTargetAdapter<'_>>;
}

/// The Unified batched adapter must implement [`MtpTarget`] (including the
/// `*_batched` surface forwarded to the inner batched adapter) and expose the
/// `batch_size` accessor used by diagnostics.
#[test]
fn unified_batched_adapter_implements_mtp_target() {
    #[allow(dead_code)]
    fn assert_mtp_target<T: MtpTarget>() {}
    let _ = assert_mtp_target::<Gemma4UnifiedMtpBatchedTargetAdapter<'_>>;
    // Pin the accessor name/signature the dispatch + tests rely on.
    #[allow(dead_code)]
    fn assert_batch_size_accessor(a: &Gemma4UnifiedMtpBatchedTargetAdapter<'_>) -> usize {
        a.batch_size()
    }
}

// ===========================================================================
// Ragged (variable-length-prompt) batched MTP prefill helpers (issue #161)
//
// These pin the pure left-padding builder and the shifted-frame seed-anchor
// formula without loading real Gemma 4 weights. The on-hardware greedy-parity
// validation of the ragged forward ships behind the orchestrator's real-model
// gate.
// ===========================================================================

/// `left_padded_input` right-aligns each row to `max_prompt_len`, reports the
/// per-row left-padding (`max_len - L_r`) and valid length (`L_r`), and builds
/// a `[B, max_len]` tensor.
#[test]
fn ragged_left_padded_input_right_aligns_rows() {
    let _runtime = crate::initialize_runtime();
    // Rows of length 2, 5, 3 -> max_len = 5.
    let per_row = vec![vec![10, 11], vec![20, 21, 22, 23, 24], vec![30, 31, 32]];
    let prefill =
        Gemma4MtpBatchedTargetAdapter::left_padded_input(&per_row, 3).expect("left-padded");
    assert_eq!(prefill.max_len, 5);
    assert_eq!(prefill.left_padding, vec![3, 0, 2], "lp = max_len - L_r");
    assert_eq!(prefill.valid_len, vec![2, 5, 3], "valid = L_r");
    let shape = mlxcel_core::array_shape(prefill.arr.as_ref().unwrap());
    assert_eq!(shape, vec![3, 5], "must build a [B, max_len] tensor");
}

/// The padded tensor places each row's real tokens at indices `[lp, max_len)`
/// with the leading `lp` columns zeroed (the padding token id).
#[test]
fn ragged_left_padded_input_places_tokens_at_suffix() {
    let _runtime = crate::initialize_runtime();
    let per_row = vec![vec![7, 8], vec![1, 2, 3, 4]];
    let prefill =
        Gemma4MtpBatchedTargetAdapter::left_padded_input(&per_row, 2).expect("left-padded");
    assert_eq!(prefill.max_len, 4);
    assert_eq!(prefill.left_padding, vec![2, 0]);

    // Read individual cells of the [2, 4] tensor.
    let cell = |r: i32, c: i32| -> i32 {
        let v = mlxcel_core::slice(prefill.arr.as_ref().unwrap(), &[r, c], &[r + 1, c + 1]);
        let scalar = mlxcel_core::reshape(&v, &[]);
        mlxcel_core::item_i32(&scalar)
    };
    // Row 0 (lp=2): [PAD, PAD, 7, 8].
    assert_eq!(cell(0, 0), 0, "row0 leading pad");
    assert_eq!(cell(0, 1), 0, "row0 leading pad");
    assert_eq!(cell(0, 2), 7, "row0 first real token at index lp=2");
    assert_eq!(cell(0, 3), 8);
    // Row 1 (lp=0): [1, 2, 3, 4].
    assert_eq!(cell(1, 0), 1);
    assert_eq!(cell(1, 3), 4);
}

/// Empty / zero-length rows are rejected.
#[test]
fn ragged_left_padded_input_rejects_empty_row() {
    let _runtime = crate::initialize_runtime();
    let per_row = vec![vec![1, 2], vec![], vec![3]];
    let msg = match Gemma4MtpBatchedTargetAdapter::left_padded_input(&per_row, 3) {
        Ok(_) => panic!("empty row must be rejected"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("non-empty"),
        "error must explain the non-empty requirement, got: {msg}"
    );
}

/// Shifted-frame seed-anchor formula: `kv_offset = left_padding + kv_valid_len`,
/// `bonus_position = kv_offset - 1`. This is the load-bearing per-row position
/// derivation that keeps each ragged row in its own RoPE frame.
#[test]
fn ragged_seed_anchors_use_shifted_frame() {
    // Ragged window: max_len = 5, rows of valid length 2, 5, 3 ->
    // left_padding = [3, 0, 2].
    let kv_valid_len = vec![2_usize, 5, 3];
    let left_padding = vec![3_usize, 0, 2];
    let (kv_offset, bonus_position) =
        super::seed_anchors_from_valid_len(&kv_valid_len, &left_padding);
    // Every row's physical/padded offset collapses to max_len = 5 at the seed.
    assert_eq!(
        kv_offset,
        vec![5, 5, 5],
        "kv_offset = lp + valid (== max_len)"
    );
    assert_eq!(bonus_position, vec![4, 4, 4], "bonus = kv_offset - 1");
}

/// Equal-length rows (`left_padding == 0`) reduce to the legacy metadata:
/// `kv_offset == kv_valid_len`, `bonus_position == kv_valid_len - 1`.
#[test]
fn ragged_seed_anchors_equal_length_reduces_to_legacy() {
    let kv_valid_len = vec![7_usize, 7, 7];
    let left_padding = vec![0_usize, 0, 0];
    let (kv_offset, bonus_position) =
        super::seed_anchors_from_valid_len(&kv_valid_len, &left_padding);
    assert_eq!(kv_offset, vec![7, 7, 7]);
    assert_eq!(bonus_position, vec![6, 6, 6]);
}

/// After a verify round, each row advances its logical valid length by its own
/// `accepted + 1` while `left_padding` stays constant — so per-row anchors
/// diverge correctly. Models the `verify_finalize_batched` bookkeeping.
#[test]
fn ragged_seed_anchors_track_per_row_round_advance() {
    // Seed valid lengths (= unpadded prompt lengths) and constant lp.
    let mut kv_valid_len = vec![2_usize, 5, 3];
    let left_padding = vec![3_usize, 0, 2];
    // Round 1 per-row accepts: row0 accepts 0, row1 accepts 3, row2 accepts 1.
    let accepted = [0_usize, 3, 1];
    for (v, a) in kv_valid_len.iter_mut().zip(accepted) {
        *v += a + 1;
    }
    // Logical valid lengths now 3, 9, 5.
    assert_eq!(kv_valid_len, vec![3, 9, 5]);
    let (kv_offset, bonus_position) =
        super::seed_anchors_from_valid_len(&kv_valid_len, &left_padding);
    // Physical/padded offsets: lp + valid.
    assert_eq!(kv_offset, vec![6, 9, 7]);
    assert_eq!(bonus_position, vec![5, 8, 6]);
}

// ===========================================================================
// END-TO-END ragged greedy parity (the real-generation proxy the parity gate
// needs). Drives the *real* `Gemma4MtpBatchedTargetAdapter` — ragged prefill,
// multi-round verify, per-row finalize/rollback, persisted per-row
// `left_padding`/position bookkeeping — against the synthetic 1-layer Gemma 4
// wrapper, and asserts the most-left-padded (shortest) row's GREEDILY GENERATED
// token sequence (including its EOS stop) is byte-identical to that same row run
// alone as B = 1.
//
// This is the unit proxy that the prior logit/mask-level tests missed: it
// actually *generates* multiple tokens through the same prefill->verify->
// finalize loop the round-loop driver uses (driven here target-only, i.e. a
// perfect/full-accept drafter, so no drafter checkpoint is required), so any
// per-row frame/position/EOS-retirement defect that flips the most-padded row's
// argmax or drops its EOS surfaces as a sequence mismatch.
// ===========================================================================

/// Greedily extend `prompt` token-by-token through the *batched* adapter for a
/// single row, stepping the real verify+finalize loop. Returns the emitted
/// sequence (seed bonus first), stopping at the first `eos` token or after
/// `max_new` tokens. The other rows are extended in lockstep but ignored.
///
/// `row` selects which batch row to read; the verify block width is 1 (pure
/// autoregressive greedy), which still drives the real per-row mask /
/// `left_padding` / position bookkeeping every round.
fn batched_greedy_extend_row(
    adapter: &Gemma4MtpBatchedTargetAdapter<'_>,
    seed_bonuses: &[i32],
    row: usize,
    eos: i32,
    max_new: usize,
    sampler: &SamplingConfig,
) -> Vec<i32> {
    let batch = seed_bonuses.len();
    let mut bonus_per_row = seed_bonuses.to_vec();
    let mut emitted = vec![seed_bonuses[row]];
    if emitted[0] == eos {
        return emitted;
    }
    while emitted.len() < max_new {
        // Width-1 verify block per row = [bonus].
        let verify_input: Vec<Vec<i32>> = bonus_per_row.iter().map(|&b| vec![b]).collect();
        let fwd = adapter
            .verify_forward_batched(&verify_input, sampler)
            .expect("batched verify forward");
        // Per-row next greedy token (argmax at the single verify position).
        let next_per_row: Vec<i32> = (0..batch)
            .map(|r| fwd.target_tokens_per_row[r][0])
            .collect();
        // accepted = 0 for every row (no draft proposals); finalize advances
        // each row by 1 and rolls back the (zero) speculative tail.
        let _seed = adapter
            .verify_finalize_batched(&vec![0usize; batch], 1, fwd.captured)
            .expect("batched verify finalize");
        let tok = next_per_row[row];
        emitted.push(tok);
        bonus_per_row = next_per_row;
        if tok == eos {
            break;
        }
    }
    emitted
}

/// Standalone B = 1 analogue of [`batched_greedy_extend_row`] driving the
/// single-row adapter's real verify+finalize loop.
fn b1_greedy_extend(
    adapter: &Gemma4MtpTargetAdapter<'_>,
    seed_bonus: i32,
    eos: i32,
    max_new: usize,
    sampler: &SamplingConfig,
    logprobs: &mlxcel_core::sampling::LogprobsConfig,
) -> Vec<i32> {
    let mut bonus = seed_bonus;
    let mut emitted = vec![seed_bonus];
    if emitted[0] == eos {
        return emitted;
    }
    while emitted.len() < max_new {
        let vout = adapter.verify_forward(&[bonus], sampler, logprobs);
        let next = vout.target_tokens[0];
        let _seed = adapter.verify_finalize(0, 1, vout.captured);
        emitted.push(next);
        bonus = next;
        if next == eos {
            break;
        }
    }
    emitted
}

/// Drive the full prefill + multi-round verify + finalize loop for BOTH rows of
/// a large-length-gap ragged batch and the two standalone B = 1 runs, returning
/// `(batched_short, std_short, batched_long, std_long)`. `eos` is the stop
/// token threaded into every walk.
fn run_ragged_vs_b1(
    short_row: &[i32],
    long_row: &[i32],
    eos: i32,
    max_new: usize,
    layer_type: &str,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let sampler = SamplingConfig::greedy();
    let logprobs = mlxcel_core::sampling::LogprobsConfig::default();
    let build = || crate::models::gemma4_tests::build_synthetic_wrapper_with_layer(layer_type);

    // ---- Standalone B = 1 ground truth ----
    let w_short = build();
    let a_short = Gemma4MtpTargetAdapter::new(&w_short, None);
    let (b_short, _, _) = a_short.prefill_and_seed(short_row, &sampler, &[], &logprobs);
    let std_short = b1_greedy_extend(&a_short, b_short, eos, max_new, &sampler, &logprobs);

    let w_long = build();
    let a_long = Gemma4MtpTargetAdapter::new(&w_long, None);
    let (b_long, _, _) = a_long.prefill_and_seed(long_row, &sampler, &[], &logprobs);
    let std_long = b1_greedy_extend(&a_long, b_long, eos, max_new, &sampler, &logprobs);

    // ---- Ragged B = 2: short row (cache consumed by its walk) ----
    let w_batch = build();
    let adapter = Gemma4MtpBatchedTargetAdapter::new(&w_batch, 2);
    let (bonuses, _seed) = adapter
        .prefill_and_seed_batched(&[short_row.to_vec(), long_row.to_vec()], &sampler)
        .expect("ragged prefill");
    let batched_short = batched_greedy_extend_row(&adapter, &bonuses, 0, eos, max_new, &sampler);

    // Fresh adapter to extend the long row in isolation.
    let w_batch2 = build();
    let adapter2 = Gemma4MtpBatchedTargetAdapter::new(&w_batch2, 2);
    let (bonuses2, _seed2) = adapter2
        .prefill_and_seed_batched(&[short_row.to_vec(), long_row.to_vec()], &sampler)
        .expect("ragged prefill");
    let batched_long = batched_greedy_extend_row(&adapter2, &bonuses2, 1, eos, max_new, &sampler);

    (batched_short, std_short, batched_long, std_long)
}

/// End-to-end MULTI-ROUND greedy parity for the most-left-padded row. A 2-row
/// ragged batch with a large length gap must produce, for the SHORT (most-
/// padded) row, a byte-identical multi-token greedy sequence to that short
/// prompt run alone as B = 1. EOS is a sentinel the synthetic model never emits,
/// so both paths run the full `max_new` rounds — this is the path where the
/// documented failure mode ("verify-round perturbation accumulates and flips the
/// most-padded row's greedy argmax") would surface as a sequence mismatch.
#[test]
fn ragged_end_to_end_greedy_parity_most_padded_row_multiround() {
    let _runtime = crate::initialize_runtime();
    for layer_type in ["sliding_attention", "full_attention"] {
        // Large length gap: short row lp = 5, long row lp = 0.
        let short_row: Vec<i32> = vec![2, 3];
        let long_row: Vec<i32> = vec![4, 5, 6, 7, 2, 3, 1];
        let max_new = 10usize;
        // Sentinel EOS the degenerate fixture never produces, forcing
        // full-length multi-round generation in every walk.
        let eos = -1i32;

        let (batched_short, std_short, batched_long, std_long) =
            run_ragged_vs_b1(&short_row, &long_row, eos, max_new, layer_type);

        eprintln!(
            "E2E multiround [{layer_type}] short: standalone={std_short:?} \
             batched={batched_short:?}; long: standalone={std_long:?} batched={batched_long:?}"
        );

        // Both paths must actually GENERATE multiple tokens (not stop at the
        // seed), otherwise the multi-round parity assertion would be vacuous.
        assert!(
            std_short.len() >= 3,
            "[{layer_type}] short-row multi-round walk must generate >= 3 tokens \
             (got {std_short:?})"
        );
        // The load-bearing assertion: the most-left-padded row's multi-round
        // greedy output is byte-identical batched-vs-standalone every step. The
        // `full_attention` iteration is the one that exercises the unbounded
        // KVCache verify-round left-padding mask (where the prompt padding is
        // resident forever).
        assert_eq!(
            batched_short, std_short,
            "[{layer_type}] SHORT (most-left-padded) row multi-round greedy \
             sequence must be byte-identical to its standalone B = 1 run"
        );
        assert_eq!(
            batched_long, std_long,
            "[{layer_type}] LONG (lp == 0) control row multi-round greedy sequence \
             must match standalone"
        );
    }
}

/// End-to-end EOS-STOP parity for the most-left-padded row: when the short row's
/// standalone greedy decode hits EOS early, the batched ragged most-padded row
/// must stop at the SAME position (not run to the cap emitting pad — the exact
/// "empty + full token budget + no EOS" symptom the real 31B gate caught).
///
/// EOS is set to the token the synthetic model greedily emits first, so the
/// short row's standalone decode is a known 1-token sequence ending in EOS; the
/// ragged most-padded row must reproduce that EOS stop rather than continuing.
#[test]
fn ragged_end_to_end_greedy_parity_most_padded_row_hits_eos() {
    let _runtime = crate::initialize_runtime();
    let sampler = SamplingConfig::greedy();
    let logprobs = mlxcel_core::sampling::LogprobsConfig::default();
    let short_row: Vec<i32> = vec![2, 3];
    let long_row: Vec<i32> = vec![4, 5, 6, 7, 2, 3, 1];
    let max_new = 12usize;

    // The token the model greedily emits first becomes EOS, so the short row's
    // standalone greedy decode terminates on EOS within budget.
    let w_probe = crate::models::gemma4_tests::build_synthetic_wrapper();
    let a_probe = Gemma4MtpTargetAdapter::new(&w_probe, None);
    let (probe_bonus, _, _) = a_probe.prefill_and_seed(&short_row, &sampler, &[], &logprobs);
    let eos = probe_bonus;

    let (batched_short, std_short, batched_long, std_long) =
        run_ragged_vs_b1(&short_row, &long_row, eos, max_new, "sliding_attention");

    eprintln!(
        "E2E eos-stop eos={eos} short: standalone={std_short:?} batched={batched_short:?}; \
         long: standalone={std_long:?} batched={batched_long:?}"
    );

    // The short row's standalone greedy decode must terminate on EOS within
    // budget (a "known sequence ending in EOS"), not run to the cap.
    assert!(
        std_short.last() == Some(&eos) && std_short.len() < max_new,
        "short row standalone greedy must end in EOS within budget (got {std_short:?})"
    );
    assert_eq!(
        batched_short, std_short,
        "SHORT (most-left-padded) row must reproduce the standalone EOS stop, not \
         run to the token cap emitting pad"
    );
    assert_eq!(
        batched_long, std_long,
        "LONG (lp == 0) control row must match standalone EOS behaviour"
    );
}

/// Hidden-state contamination probe — the most sensitive signal, immune to the
/// synthetic fixture's degenerate argmax. After the ragged prefill, run several
/// width-1 verify rounds so the shared cache offset grows past the prompt + its
/// padding, then compare the most-left-padded row's captured verify hidden state
/// against the standalone B = 1 run at the same logical step.
///
/// Note on the tolerance: exact bitwise equality is NOT achievable, because
/// left-padding shifts every real token's *absolute* RoPE position by `lp`. RoPE
/// is relative — `rotate(q, p_q) · rotate(k, p_k)` depends only on `p_q - p_k` —
/// so the attention scores are mathematically identical, but the two absolute
/// rotations round differently in fp, leaving a tiny (~1e-3) residual. A padding
/// *contamination* bug, by contrast, makes the most-padded row attend `lp`
/// spurious padding keys and perturbs the hidden by a LARGE amount (pre-fix this
/// probe saw first-byte divergences, i.e. O(1) relative error). The tolerance
/// below is loose enough to pass the unavoidable RoPE rounding yet far tighter
/// than any contamination, so it is a real proxy for the greedy-parity gate.
/// Runs for both attention families — `full_attention` exercises the unbounded
/// KVCache (padding resident forever) and `sliding_attention` the MTP-buffered
/// RotatingKVCache (padding resident until the buffer compacts).
#[test]
fn ragged_most_padded_row_verify_hidden_has_no_padding_contamination() {
    let _runtime = crate::initialize_runtime();
    let sampler = SamplingConfig::greedy();
    let logprobs = mlxcel_core::sampling::LogprobsConfig::default();

    // Read the most-padded row (row 0) hidden from a verify capture's
    // `VerifyCaptured.tensors[0]` (`[B, width, hidden]`) as f32 values.
    fn row0_hidden_f32(captured: &VerifyCaptured, width: i32, hidden: i32) -> Vec<f32> {
        let h = captured
            .tensors
            .first()
            .expect("verify capture carries hidden at index 0");
        let row0 = mlxcel_core::slice(h.as_ref().unwrap(), &[0, 0, 0], &[1, width, hidden]);
        let row0_f32 = mlxcel_core::astype(&row0, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&row0_f32);
        let bytes = mlxcel_core::array_to_raw_bytes(&row0_f32);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    for layer_type in ["sliding_attention", "full_attention"] {
        let build = || crate::models::gemma4_tests::build_synthetic_wrapper_with_layer(layer_type);
        let short_row: Vec<i32> = vec![2, 3];
        let long_row: Vec<i32> = vec![4, 5, 6, 7, 2, 3, 1];
        let hidden_dim = 4i32; // synthetic fixture hidden_size.
        // Walk a few rounds so the SWA cache offset grows past `sliding_window`
        // (= 8) while the MTP rollback buffer (`buffer_size = 32`) keeps the
        // prompt padding RESIDENT and uncompacted — the exact regime the
        // ragged path hits on the real 31B and the one the pre-fix mask logic
        // mis-handled. At this depth BOTH defects bite: the full-attention path
        // (unbounded cache, padding resident forever) and the sliding path
        // (buffered cache, `sliding_offset + l > window` yet padding resident).
        let warmup_rounds = 4usize;

        // ---- Standalone B = 1 short row ----
        let w_short = build();
        let a_short = Gemma4MtpTargetAdapter::new(&w_short, None);
        let (mut b_short, _, _) = a_short.prefill_and_seed(&short_row, &sampler, &[], &logprobs);
        for _ in 0..warmup_rounds {
            let vout = a_short.verify_forward(&[b_short], &sampler, &logprobs);
            b_short = vout.target_tokens[0];
            let _ = a_short.verify_finalize(0, 1, vout.captured);
        }
        let std_capture = a_short.verify_forward(&[b_short], &sampler, &logprobs);
        let std_h = row0_hidden_f32(&std_capture.captured, 1, hidden_dim);

        // ---- Ragged B = 2, most-padded row (row 0) ----
        let w_batch = build();
        let adapter = Gemma4MtpBatchedTargetAdapter::new(&w_batch, 2);
        let (bonuses, _seed) = adapter
            .prefill_and_seed_batched(&[short_row.clone(), long_row.clone()], &sampler)
            .expect("ragged prefill");
        let mut bonus_per_row = bonuses;
        for _ in 0..warmup_rounds {
            let verify_input: Vec<Vec<i32>> = bonus_per_row.iter().map(|&b| vec![b]).collect();
            let fwd = adapter
                .verify_forward_batched(&verify_input, &sampler)
                .expect("batched verify forward");
            bonus_per_row = (0..2).map(|r| fwd.target_tokens_per_row[r][0]).collect();
            let _ = adapter
                .verify_finalize_batched(&[0usize, 0usize], 1, fwd.captured)
                .expect("batched verify finalize");
        }
        let verify_input: Vec<Vec<i32>> = bonus_per_row.iter().map(|&b| vec![b]).collect();
        let batched_capture = adapter
            .verify_forward_batched(&verify_input, &sampler)
            .expect("batched verify forward");
        let batched_h = row0_hidden_f32(&batched_capture.captured, 1, hidden_dim);

        // Max relative deviation across the hidden vector.
        let max_rel = std_h
            .iter()
            .zip(&batched_h)
            .map(|(&s, &b)| (s - b).abs() / s.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        eprintln!(
            "[{layer_type}] most-padded-row hidden max_rel_dev={max_rel:.6} std={std_h:?} batched={batched_h:?}"
        );
        // Measured on this fixture at `warmup_rounds = 4`: the pure RoPE-rounding
        // residual (fix applied) is ~1.5e-5, while the pre-fix padding
        // contamination is ~1.6e-4 (sliding, buffered) / ~2.8e-4 (full,
        // unbounded) and grows with depth. An 8e-5 ceiling sits cleanly between
        // them, so this assertion FAILS without the verify-round padding mask and
        // PASSES with it — a real unit proxy for the 31B greedy-parity gate.
        assert!(
            max_rel < 8e-5,
            "[{layer_type}] most-left-padded row verify hidden deviates from the \
             standalone B = 1 run by max_rel={max_rel} — far above the ~1.5e-5 RoPE \
             fp-rounding floor, indicating the row attends resident prompt padding \
             (greedy-parity break)"
        );
    }
}
