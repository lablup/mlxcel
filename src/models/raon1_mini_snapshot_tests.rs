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

/// A snapshot that lacks a layer, or whose RDA state has the wrong shape or
/// offset, fails the restore (the scheduler then prefills cold) and leaves no
/// sequence state behind.
#[test]
fn a_snapshot_missing_a_layer_or_with_wrong_shapes_is_refused() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let prompt: Vec<i32> = (0..9).map(|i| (i * 7 + 1) % VOCAB as i32).collect();
    let source = SequenceId::from_raw(34_590);
    model.prepare_sequence_state(source);
    let out = model.forward_with_sequence_id(
        &ids(&[&prompt]),
        Some(source),
        &mut model.make_caches(),
        None,
    );
    mlxcel_core::eval(&out);
    let good = model
        .snapshot_sequence_state(source, prompt.len())
        .expect("a prefilled sequence donates a snapshot");
    let target = SequenceId::from_raw(34_591);
    model.prepare_sequence_state(target);
    model
        .restore_sequence_state(target, &good)
        .expect("the intact snapshot restores");

    // Rebuild the snapshot without the tensors whose name contains `drop`,
    // or with one tensor replaced by `swap`.
    let rebuild =
        |drop: Option<&str>,
         swap: Option<(&str, mlxcel_core::UniquePtr<mlxcel_core::MlxArray>)>| {
            let mut out = ModelStateSnapshot::new("raon1_mini", prompt.len());
            let mut swap = swap;
            for name in (0..64).flat_map(|i| {
                [
                    "rda.conv",
                    "rda.ssm",
                    "rda.offset",
                    "mla.keys",
                    "mla.values",
                    "mla.offset",
                    "mla.mode",
                ]
                .map(|kind| format!("layer{i}.{kind}"))
            }) {
                let Some(tensor) = good.tensor(&name) else {
                    continue;
                };
                if drop.is_some_and(|d| name.contains(d)) {
                    continue;
                }
                match swap.take_if(|(n, _)| *n == name) {
                    Some((_, replacement)) => out.push_tensor(name, &replacement),
                    None => out.push_tensor(name, tensor),
                }
            }
            out
        };
    let refused = |snapshot: &ModelStateSnapshot| {
        let id = SequenceId::from_raw(34_592);
        model.prepare_sequence_state(id);
        let err = model.restore_sequence_state(id, snapshot).unwrap_err();
        model.release_sequence_state_by_id(id);
        err
    };
    assert!(rebuild(None, None).token_len() == prompt.len());
    model
        .restore_sequence_state(target, &rebuild(None, None))
        .expect("the rebuilt copy is intact");

    // A missing RDA layer, a missing MLA layer.
    let err = refused(&rebuild(Some("layer0.rda"), None));
    assert!(err.contains("layer0"), "{err}");
    let err = refused(&rebuild(Some("layer2.mla"), None));
    assert!(err.contains("layer2"), "{err}");
    // An RDA state of the wrong rank, and one of the wrong offset.
    let flat = mlxcel_core::from_slice_f32(&[0.0; 4], &[2, 2]);
    let err = refused(&rebuild(None, Some(("layer0.rda.ssm", flat))));
    assert!(err.contains("layer0"), "{err}");
    let err = refused(&rebuild(
        None,
        Some(("layer0.rda.offset", mlxcel_core::from_slice_i32(&[2], &[]))),
    ));
    assert!(err.contains("offset"), "{err}");

    for id in [source, target] {
        model.release_sequence_state_by_id(id);
    }
}
