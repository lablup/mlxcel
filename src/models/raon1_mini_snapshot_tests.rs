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

//! Prompt-cache snapshots of a Raon-1 mini sequence (issue #3458) on the tiny
//! random model.

use super::super::tests::{VOCAB, ids, tiny_model};

/// A prompt-cache snapshot (issue #3458): the state after a prefill, restored
/// into a fresh sequence, decodes the next eight tokens exactly as the
/// original sequence does, and the snapshot is tied to this family.
#[test]
fn a_restored_snapshot_decodes_bit_for_bit_like_the_original_sequence() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let prompt: Vec<i32> = (0..11).map(|i| (i * 5 + 2) % VOCAB as i32).collect();
    let (original, restored) = (SequenceId::from_raw(34_580), SequenceId::from_raw(34_581));
    let forward = |id: SequenceId, tokens: &[i32]| {
        let out = model.forward_with_sequence_id(
            &ids(&[tokens]),
            Some(id),
            &mut model.make_caches(),
            None,
        );
        mlxcel_core::eval(&out);
        mlxcel_core::array_to_raw_bytes(&out)
    };

    assert!(model.supports_snapshot_reuse());
    model.prepare_sequence_state(original);
    let _ = forward(original, &prompt);
    let snapshot = model
        .snapshot_sequence_state(original, prompt.len())
        .expect("a prefilled sequence donates a snapshot");
    assert_eq!(snapshot.family(), "raon1_mini");
    assert_eq!(snapshot.token_len(), prompt.len());
    // The recurrent state cannot rewind, so only the whole prefix restores.
    assert!(!model.snapshot_truncatable_to(&snapshot, prompt.len() - 1));
    // A state that does not hold exactly the keyed tokens donates nothing:
    // a DFlash2 burst leaves the sequence's own state untouched.
    assert!(
        model
            .snapshot_sequence_state(original, prompt.len() + 3)
            .is_none()
    );

    model.prepare_sequence_state(restored);
    model
        .restore_sequence_state(restored, &snapshot)
        .expect("restore into a fresh sequence");
    let next: Vec<i32> = (0..8).map(|i| (i * 3 + 1) % VOCAB as i32).collect();
    for (step, token) in next.iter().enumerate() {
        assert_eq!(
            forward(restored, &[*token]),
            forward(original, &[*token]),
            "decode step {step} after the restore"
        );
    }
    // The snapshot is a copy: decoding the original did not change it.
    let again = SequenceId::from_raw(34_582);
    model.prepare_sequence_state(again);
    model.restore_sequence_state(again, &snapshot).unwrap();
    let fresh = SequenceId::from_raw(34_583);
    model.prepare_sequence_state(fresh);
    let _ = forward(fresh, &prompt);
    assert_eq!(forward(again, &next[..1]), forward(fresh, &next[..1]));

    let foreign = ModelStateSnapshot::new("falcon_h1", prompt.len());
    let err = model
        .restore_sequence_state(restored, &foreign)
        .unwrap_err();
    assert!(err.contains("falcon_h1"), "{err}");
    for id in [original, restored, again, fresh] {
        model.release_sequence_state_by_id(id);
    }
}
