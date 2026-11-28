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

//! Gemma 3 snapshot donation and restore under a quantized KV mode table
//! (issue #3250).
//!
//! The two-layer fixture has a sliding layer and a global one, which is Gemma
//! 4's situation in miniature. Under `Int8` or `Turbo4` the sliding ring holds
//! plain rows behind a label while the global layer holds real quantized
//! state; under `Turbo4Asym` on a window that is a multiple of the turbo block
//! both layers are really quantized. Before the change every one of these
//! tables made `snapshot_sequence_state` return `None`.
//!
//! The reference is always a cold run of the same mode: the logits a restored
//! sequence produces for the rest of a prompt must equal, bit for bit, the
//! logits of a sequence that prefilled the prefix itself.

use super::super::Gemma3Wrapper;
use super::snapshot_prompt_cache::{build_wrapper_with_window, ids};
use crate::models::embedding_test_support::mlx_test_guard;
use crate::models::pooled_model::Pooled;
use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::LanguageModel;

/// Sequence-id base for this module ("issue 3250, sequence N").
const SEQ_BASE: u64 = 3_250_000;

/// The mode tables under test with a window each: `(mode, window, what the
/// sliding layer stores)`. `Turbo4Asym` quantizes a ring only on a window
/// that is a multiple of the turbo block (32); on the 8-token window the
/// model builds that ring as plain FP16.
const TABLES: [(KVCacheMode, i32, KVCacheMode); 6] = [
    (KVCacheMode::Int8, 8, KVCacheMode::Fp16),
    (KVCacheMode::Int8, 64, KVCacheMode::Fp16),
    (KVCacheMode::Turbo4, 8, KVCacheMode::Fp16),
    (KVCacheMode::Turbo4Asym, 8, KVCacheMode::Fp16),
    (KVCacheMode::Turbo4Asym, 32, KVCacheMode::Turbo4Asym),
    (KVCacheMode::Turbo4Asym, 64, KVCacheMode::Turbo4Asym),
];

fn wrapper(window: i32, mode: KVCacheMode) -> Pooled<Gemma3Wrapper> {
    let wrapper = build_wrapper_with_window(window);
    wrapper.set_kv_cache_layer_modes(vec![mode; wrapper.num_layers()]);
    wrapper
}

/// Forward `tokens` on a sequence that may already hold state and return the
/// logits as raw bytes.
fn logit_bytes(wrapper: &Pooled<Gemma3Wrapper>, seq: SequenceId, tokens: &[i32]) -> Vec<u8> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = wrapper.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    mlxcel_core::eval(&logits);
    mlxcel_core::array_to_raw_bytes(&logits)
}

/// A fresh sequence that prefilled `prefix` itself.
fn cold(
    window: i32,
    mode: KVCacheMode,
    id: u64,
    prefix: &[i32],
) -> (Pooled<Gemma3Wrapper>, SequenceId) {
    let wrapper = wrapper(window, mode);
    let seq = SequenceId::from_raw(SEQ_BASE + id);
    wrapper.prepare_sequence_state(seq);
    if !prefix.is_empty() {
        let _ = logit_bytes(&wrapper, seq, prefix);
    }
    (wrapper, seq)
}

#[test]
fn the_tables_store_what_the_tests_assume() {
    let _guard = mlx_test_guard();
    for (mode, window, sliding) in TABLES {
        let wrapper = wrapper(window, mode);
        assert_eq!(
            wrapper.kv_cache_storage_modes(),
            Some(vec![Some(sliding), Some(mode)]),
            "{mode:?} at window {window}: a sliding layer, then a global one"
        );
    }
}

/// A snapshot taken at `boundary` tokens and restored into a fresh sequence
/// gives the rest of the prompt the logits a cold split prefill gives it,
/// bit for bit, and so does the decode step after it.
///
/// The boundaries are one token, a ring that has not wrapped, one exactly as
/// long as the window, a prefill longer than the window, and both sides of
/// the global layer's 256-token buffer step.
#[test]
fn a_restored_quantized_state_reproduces_a_cold_split_prefill_bit_for_bit() {
    let _guard = mlx_test_guard();
    for (mode, window, _) in TABLES {
        for boundary in [1, 6, window, window + 5, 256, 257] {
            let tokens = ids(0..boundary + 6);
            let (prefix, rest) = tokens.split_at(boundary as usize);
            let (suffix, next) = rest.split_at(5);
            let what = format!("{mode:?}, window {window}, boundary {boundary}");

            let (cold_model, cold_seq) = cold(window, mode, 1, prefix);
            let cold_suffix = logit_bytes(&cold_model, cold_seq, suffix);
            let cold_next = logit_bytes(&cold_model, cold_seq, next);

            let (donor, donor_seq) = cold(window, mode, 2, prefix);
            let snapshot = donor
                .snapshot_sequence_state(donor_seq, prefix.len())
                .unwrap_or_else(|| panic!("{what}: the state must be donated"));
            assert_eq!(snapshot.token_len(), prefix.len());
            assert!(!snapshot.is_empty(), "{what}");

            let (restored, restored_seq) = cold(window, mode, 3, &[]);
            restored
                .restore_sequence_state(restored_seq, &snapshot)
                .unwrap_or_else(|err| panic!("{what}: restore failed: {err}"));
            assert!(
                logit_bytes(&restored, restored_seq, suffix) == cold_suffix,
                "{what}: the suffix logits after a restore differ from the cold split prefill"
            );
            assert!(
                logit_bytes(&restored, restored_seq, next) == cold_next,
                "{what}: the next decode step after a restore differs from the cold run"
            );

            // The donor went on being used; the snapshot it shares buffers
            // with must not have moved.
            let _ = logit_bytes(&donor, donor_seq, suffix);
            let (again, again_seq) = cold(window, mode, 4, &[]);
            again
                .restore_sequence_state(again_seq, &snapshot)
                .unwrap_or_else(|err| panic!("{what}: second restore failed: {err}"));
            assert!(
                logit_bytes(&again, again_seq, suffix) == cold_suffix,
                "{what}: a second restore of the same snapshot differs"
            );
        }
    }
}

/// A snapshot cut back to a shorter prefix is the state a cold prefill of
/// that prefix builds, while the sliding ring is unwrapped. Under a label
/// that leaves the ring holding plain rows, that used to be refused because
/// of the label alone.
#[test]
fn a_truncated_quantized_restore_reproduces_a_cold_prefill_of_the_prefix() {
    let _guard = mlx_test_guard();
    for (mode, window, _) in TABLES {
        for (stored, target) in [(6, 4), (6, 1), (6, 6), (window, window - 1)] {
            let tokens = ids(0..stored + 4);
            let what = format!("{mode:?}, window {window}, {stored} tokens cut to {target}");
            let (donor, donor_seq) = cold(window, mode, 5, &tokens[..stored as usize]);
            let snapshot = donor
                .snapshot_sequence_state(donor_seq, stored as usize)
                .unwrap_or_else(|| panic!("{what}: the state must be donated"));
            assert!(
                donor.snapshot_truncatable_to(&snapshot, target as usize),
                "{what}: an unwrapped ring truncates"
            );
            assert!(
                !donor.snapshot_truncatable_to(&snapshot, stored as usize + 1),
                "{what}: cannot invent tokens the snapshot never held"
            );

            let (adopted, adopted_seq) = cold(window, mode, 6, &[]);
            adopted
                .restore_sequence_state_truncated(adopted_seq, &snapshot, target as usize)
                .unwrap_or_else(|err| panic!("{what}: truncated restore failed: {err}"));

            let (control, control_seq) = cold(window, mode, 7, &tokens[..target as usize]);
            let suffix = &tokens[target as usize..target as usize + 3];
            assert!(
                logit_bytes(&adopted, adopted_seq, suffix)
                    == logit_bytes(&control, control_seq, suffix),
                "{what}: the suffix logits differ from a cold prefill of the prefix"
            );
        }
    }
}

/// Once the ring has wrapped, the snapshot does not truncate, whatever the
/// label says, and a truncated restore of it is refused rather than run.
#[test]
fn a_wrapped_ring_does_not_truncate_under_any_table() {
    let _guard = mlx_test_guard();
    for (mode, window, _) in TABLES {
        let stored = window + 16;
        let tokens = ids(0..stored);
        let what = format!("{mode:?}, window {window}");
        let (donor, donor_seq) = cold(window, mode, 8, &tokens);
        let snapshot = donor
            .snapshot_sequence_state(donor_seq, stored as usize)
            .unwrap_or_else(|| panic!("{what}: the state must be donated"));
        assert!(
            !donor.snapshot_truncatable_to(&snapshot, stored as usize - 1),
            "{what}: a wrapped ring does not truncate"
        );
        let (adopted, adopted_seq) = cold(window, mode, 9, &[]);
        assert!(
            adopted
                .restore_sequence_state_truncated(adopted_seq, &snapshot, stored as usize - 1)
                .is_err(),
            "{what}: a truncated restore of a wrapped ring is refused"
        );
    }
}

/// `Turbo4Delegated` and `Turbo3Asym` on the global layer are still not
/// carried: the table reports the mode for that layer and nothing is donated.
///
/// Only the delegated table is run through a prefill. The 3-bit packing needs
/// a head dim that is a multiple of 8, so it cannot run on this fixture's
/// head dim of 2 at all; its refusal is pinned on 64-wide caches in
/// `kv_snapshot_quantized_tests.rs`.
#[test]
fn a_table_with_an_uncarried_mode_donates_nothing() {
    let _guard = mlx_test_guard();
    for mode in [KVCacheMode::Turbo4Delegated, KVCacheMode::Turbo3Asym] {
        assert_eq!(
            wrapper(8, mode).kv_cache_storage_modes(),
            Some(vec![Some(KVCacheMode::Fp16), Some(mode)]),
            "{mode:?}: the ring stores plain rows, the global layer the mode"
        );
    }
    let tokens = ids(0..6);
    let (donor, donor_seq) = cold(8, KVCacheMode::Turbo4Delegated, 10, &tokens);
    assert!(
        donor.snapshot_sequence_state(donor_seq, 6).is_none(),
        "a state with a Turbo4Delegated layer is not donated"
    );
}

/// A snapshot taken under one table does not restore under another.
#[test]
fn a_snapshot_restores_only_under_the_table_it_was_taken_with() {
    let _guard = mlx_test_guard();
    let tokens = ids(0..6);
    let (donor, donor_seq) = cold(8, KVCacheMode::Int8, 11, &tokens);
    let snapshot = donor
        .snapshot_sequence_state(donor_seq, 6)
        .expect("an Int8 state is donated");
    for other in [
        KVCacheMode::Fp16,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Asym,
    ] {
        let (target, target_seq) = cold(8, other, 12, &[]);
        assert!(
            target
                .restore_sequence_state(target_seq, &snapshot)
                .is_err(),
            "an Int8 snapshot must not restore under a {other:?} table"
        );
        // The refused sequence is still a fresh one and prefills cold.
        let (control, control_seq) = cold(8, other, 13, &[]);
        assert!(
            logit_bytes(&target, target_seq, &tokens)
                == logit_bytes(&control, control_seq, &tokens),
            "{other:?}: a refused restore leaves a fresh sequence"
        );
    }
}
