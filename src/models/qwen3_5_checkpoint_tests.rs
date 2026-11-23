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

//! Turn-boundary state checkpoints for Qwen 3.5 (issue #3047), on the
//! two-layer synthetic model: a GatedDeltaNet layer and a full-attention
//! layer.
//!
//! The rebuilt state, the checkpoint's GatedDeltaNet state plus a longer
//! snapshot's attention rows cut back, must decode exactly like a direct
//! prefill of the shorter prefix.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::{MlxArray, UniquePtr};

use super::pooled_model::Pooled;
use super::qwen3_5_tests::pooled_two_layer_model;
use crate::models::Qwen35Model;

/// `(checkpoint, donor)` lengths: a long cut, a one-token cut, and a cut that
/// crosses the KV buffer's 256-row allocation step.
const CASES: &[(usize, usize)] = &[(9, 23), (16, 17), (200, 300)];

fn prompt(len: usize) -> Vec<i32> {
    (0..len).map(|i| ((i * 5 + 3) % 8) as i32).collect()
}

fn forward(model: &Pooled<Qwen35Model>, seq: SequenceId, tokens: &[i32]) -> UniquePtr<MlxArray> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    mlxcel_core::eval(&logits);
    logits
}

fn bytes(array: &MlxArray) -> Vec<u8> {
    let evaluated = mlxcel_core::contiguous(array, false);
    mlxcel_core::eval(&evaluated);
    mlxcel_core::array_to_raw_bytes(&evaluated)
}

/// Logits of a prefill of `suffix` and eight decode steps after it.
fn continue_from(model: &Pooled<Qwen35Model>, seq: SequenceId, suffix: &[i32]) -> Vec<Vec<u8>> {
    let mut out = vec![bytes(&forward(model, seq, suffix))];
    for step in 0..8 {
        out.push(bytes(&forward(model, seq, &[(step * 3 + 1) % 8])));
    }
    out
}

/// The checkpoint at `keep` and the whole snapshot at `tokens.len()` from one
/// sequence, the way a conversation's turns produce them.
fn chain(
    model: &Pooled<Qwen35Model>,
    seq: SequenceId,
    tokens: &[i32],
    keep: usize,
) -> (ModelStateSnapshot, ModelStateSnapshot) {
    model.prepare_sequence_state(seq);
    let _ = forward(model, seq, &tokens[..keep]);
    let at_keep = model
        .snapshot_sequence_state(seq, keep)
        .expect("snapshot at the turn boundary");
    let checkpoint = model
        .state_checkpoint_from_snapshot(&at_keep)
        .expect("Qwen 3.5 splits its state");
    drop(at_keep);
    let _ = forward(model, seq, &tokens[keep..]);
    let donor = model
        .snapshot_sequence_state(seq, tokens.len())
        .expect("snapshot of the newest turn");
    model.release_sequence_state_by_id(seq);
    (checkpoint, donor)
}

/// The acceptance check of #3047 for Qwen 3.5: a checkpoint plus a longer
/// snapshot cut back decodes exactly like a direct prefill of the prefix.
#[test]
fn a_checkpoint_over_a_longer_snapshot_matches_a_direct_prefill_bit_for_bit() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = pooled_two_layer_model();
    let suffix = [6, 1, 4, 4, 2];
    for (case, &(keep, have)) in CASES.iter().enumerate() {
        let tokens = prompt(have);
        let base = 3_047_000 + case as u64 * 10;
        let (checkpoint, donor) = chain(&model, SequenceId::from_raw(base), &tokens, keep);
        assert_eq!(checkpoint.token_len(), keep);
        assert!(checkpoint.tensor("layer1.attention.keys").is_none());
        assert!(checkpoint.tensor("layer0.linear.state_cache").is_some());
        assert!(checkpoint.nbytes() < donor.nbytes());

        let direct = SequenceId::from_raw(base + 1);
        model.prepare_sequence_state(direct);
        let _ = forward(&model, direct, &tokens[..keep]);
        let want = continue_from(&model, direct, &suffix);

        let rebuilt = SequenceId::from_raw(base + 2);
        model.prepare_sequence_state(rebuilt);
        model
            .restore_sequence_state_from_checkpoint(rebuilt, &checkpoint, &donor)
            .expect("rebuild from the checkpoint");
        let got = continue_from(&model, rebuilt, &suffix);
        for (step, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                w == g,
                "({keep}, {have}): step {step} after the rebuilt state differs from a direct prefill"
            );
        }
        for seq in [direct, rebuilt] {
            model.release_sequence_state_by_id(seq);
        }
    }
}

/// A shorter donor, a foreign family, and a checkpoint without the
/// GatedDeltaNet state are refused, and the sequence is left untouched.
#[test]
fn mismatched_checkpoints_are_refused() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = pooled_two_layer_model();
    let tokens = prompt(23);
    let (checkpoint, donor) = chain(&model, SequenceId::from_raw(3_047_100), &tokens, 9);
    let (_, short_donor) = chain(&model, SequenceId::from_raw(3_047_101), &tokens[..7], 4);
    let target = SequenceId::from_raw(3_047_102);
    model.prepare_sequence_state(target);
    assert!(
        model
            .restore_sequence_state_from_checkpoint(target, &checkpoint, &short_donor)
            .is_err()
    );
    let foreign = ModelStateSnapshot::new("qwen3_next", 9);
    assert!(
        model
            .restore_sequence_state_from_checkpoint(target, &foreign, &donor)
            .is_err()
    );
    let gutted = checkpoint.subset(|name| !name.ends_with("state_cache"));
    let err = model
        .restore_sequence_state_from_checkpoint(target, &gutted, &donor)
        .expect_err("a checkpoint without GatedDeltaNet state must be refused");
    assert!(err.contains("state_cache"), "{err}");
    model.release_sequence_state_by_id(target);
}

/// Qwen 3.5 is scheduler-owned (issue #3254): its table, its pool state and
/// its snapshot vocabulary, and the bare model refuses a sequence id on the
/// `KVCache`-slice entry.
#[test]
fn qwen35_is_scheduler_owned() {
    use mlxcel_core::cache::{LayerState, LayerStateKind, SequenceStateBackend};

    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = pooled_two_layer_model();
    assert_eq!(
        model.sequence_state_layout().backend,
        SequenceStateBackend::SchedulerOwned
    );
    assert_eq!(
        model.layer_state_kinds(),
        Some(vec![LayerStateKind::Recurrent, LayerStateKind::Full])
    );
    let state = model.make_layer_state().expect("layer state");
    let [LayerState::Recurrent(recurrent), LayerState::Kv(_)] = state.as_slice() else {
        panic!("a GatedDeltaNet layer then a full-attention layer");
    };
    assert_eq!(
        recurrent.names().collect::<Vec<_>>(),
        vec!["conv_state", "state_cache"]
    );
    assert_eq!(
        model.layer_state_snapshot_spec().map(|spec| spec.family),
        Some("qwen3_5")
    );
    assert!(model.supports_snapshot_reuse());

    let bare: &Qwen35Model = &model;
    let input = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bare.forward_with_sequence_id(&input, Some(SequenceId::from_raw(3_254_900)), &mut [], None)
    }));
    assert!(refused.is_err(), "a sequence id must go through the pool");
}

/// The acceptance check of #3254 for Qwen 3.5: prefill in two pieces,
/// snapshot through the generic path, restore into a fresh pool sequence and
/// decode 16 tokens. Every step equals a cold run prefilled with the same
/// partition, bit for bit. A recurrent state cannot be cut back, so a whole
/// snapshot never truncates.
#[test]
fn a_whole_snapshot_round_trips_bit_for_bit() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = pooled_two_layer_model();
    for (case, partition) in [[5usize, 4], [200, 100]].into_iter().enumerate() {
        let tokens = prompt(partition[0] + partition[1]);
        let base = 3_254_000 + case as u64 * 10;
        let prefill = |seq: SequenceId| {
            model.prepare_sequence_state(seq);
            let _ = forward(&model, seq, &tokens[..partition[0]]);
            let _ = forward(&model, seq, &tokens[partition[0]..]);
        };
        let decode = |seq: SequenceId| -> Vec<Vec<u8>> {
            (0..16)
                .map(|step| bytes(&forward(&model, seq, &[(step * 3 + 1) % 8])))
                .collect()
        };
        let (source, cold, restored) = (
            SequenceId::from_raw(base),
            SequenceId::from_raw(base + 1),
            SequenceId::from_raw(base + 2),
        );
        prefill(source);
        let snapshot = model
            .snapshot_sequence_state(source, tokens.len())
            .expect("whole snapshot");
        assert_eq!(snapshot.family(), "qwen3_5");
        assert!(snapshot.tensor("layer0.linear.conv_state").is_some());
        assert!(snapshot.tensor("layer1.attention.keys").is_some());
        assert!(!model.snapshot_truncatable_to(&snapshot, partition[0]));

        prefill(cold);
        let want = decode(cold);
        model
            .restore_sequence_state(restored, &snapshot)
            .expect("restore");
        let got = decode(restored);
        for (step, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                w == g,
                "partition {partition:?}: step {step} after the restore differs from the cold run"
            );
        }
        for seq in [source, cold, restored] {
            model.release_sequence_state_by_id(seq);
        }
    }
}
