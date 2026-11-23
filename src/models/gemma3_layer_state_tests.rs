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

//! Gemma 3 on scheduler-owned layer state (issue #3254).
//!
//! The fixture is the two-layer model of `snapshot_prompt_cache`: layer 0
//! slides over an 8-token window and layer 1 is global. A snapshot taken
//! through the generic prompt-cache path and restored into a fresh pool
//! sequence must decode bit for bit like a cold run prefilled with the same
//! partition (the #2974 rule: a different chunking is a different graph, so
//! the cold run never uses one chunk where the source used two).

use super::super::{Cache, Gemma3Wrapper};
use super::snapshot_prompt_cache::{build_wrapper, ids};
use crate::models::kv_snapshot::KvSnapshotNames;
use crate::models::pooled_model::Pooled;
use mlxcel_core::cache::{LayerState, LayerStateKind, SequenceId, SequenceStateBackend};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::{MlxArray, UniquePtr};

const SEQ_BASE: u64 = 3_254_000;
const DECODE_STEPS: i32 = 16;

fn bytes(array: &MlxArray) -> Vec<u8> {
    let evaluated = mlxcel_core::contiguous(array, false);
    mlxcel_core::eval(&evaluated);
    mlxcel_core::array_to_raw_bytes(&evaluated)
}

fn forward(model: &Pooled<Gemma3Wrapper>, seq: SequenceId, tokens: &[i32]) -> UniquePtr<MlxArray> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let logits = model.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    mlxcel_core::eval(&logits);
    logits
}

/// Prefill `tokens` in pieces of the given lengths on a fresh sequence.
fn prefill_partition(
    model: &Pooled<Gemma3Wrapper>,
    seq: SequenceId,
    tokens: &[i32],
    partition: &[usize],
) {
    assert_eq!(partition.iter().sum::<usize>(), tokens.len());
    model.prepare_sequence_state(seq);
    let mut start = 0;
    for &piece in partition {
        let _ = forward(model, seq, &tokens[start..start + piece]);
        start += piece;
    }
}

/// The logits of [`DECODE_STEPS`] single-token steps after position `from`.
fn decode_logits(model: &Pooled<Gemma3Wrapper>, seq: SequenceId, from: i32) -> Vec<Vec<u8>> {
    ids(from..from + DECODE_STEPS)
        .into_iter()
        .map(|token| bytes(&forward(model, seq, &[token])))
        .collect()
}

fn assert_bit_exact(got: &[Vec<u8>], want: &[Vec<u8>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: step count");
    for (step, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g == w, "{what}: decode step {step} logits differ");
    }
}

#[test]
fn gemma3_is_scheduler_owned() {
    let model = build_wrapper();
    assert_eq!(
        model.sequence_state_layout().backend,
        SequenceStateBackend::SchedulerOwned
    );
    assert_eq!(
        model.layer_state_kinds(),
        Some(vec![LayerStateKind::sliding(8), LayerStateKind::Full])
    );
    let state = model.make_layer_state().expect("layer state");
    assert!(matches!(
        state.as_slice(),
        [LayerState::Rotating(_), LayerState::Kv(_)]
    ));
    // The lookahead rewind is on, so the sliding layer logs its decode writes.
    assert!(model.supports_decode_lookahead_rewind());
    assert!(model.supports_snapshot_reuse());
    assert_eq!(
        model.layer_state_snapshot_spec().map(|spec| spec.family),
        Some("gemma3")
    );
}

#[test]
#[should_panic(expected = "Gemma 3 state is scheduler-owned")]
fn a_sequence_id_on_the_kv_slice_entry_is_refused() {
    let model = build_wrapper();
    let input = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    // The bare model, not the pooled driver: this is a caller that bypassed
    // the pool.
    let bare: &Gemma3Wrapper = &model;
    let _ =
        bare.forward_with_sequence_id(&input, Some(SequenceId::from_raw(SEQ_BASE)), &mut [], None);
}

/// Prefill `prompt_len` tokens in `partition`, snapshot, restore into a fresh
/// sequence and compare 16 decode steps with a cold run of the same
/// partition.
fn round_trip(case: u64, prompt_len: i32, partition: &[usize]) -> ModelStateSnapshot {
    let model = build_wrapper();
    let prompt = ids(0..prompt_len);
    let (source, cold, restored) = (
        SequenceId::from_raw(SEQ_BASE + case * 10 + 1),
        SequenceId::from_raw(SEQ_BASE + case * 10 + 2),
        SequenceId::from_raw(SEQ_BASE + case * 10 + 3),
    );
    prefill_partition(&model, source, &prompt, partition);
    let snapshot = model
        .snapshot_sequence_state(source, prompt.len())
        .expect("Gemma 3 donates a snapshot");
    assert_eq!(snapshot.family(), "gemma3");
    assert_eq!(snapshot.token_len(), prompt.len());

    prefill_partition(&model, cold, &prompt, partition);
    let want = decode_logits(&model, cold, prompt_len);

    model
        .restore_sequence_state(restored, &snapshot)
        .expect("restore");
    let got = decode_logits(&model, restored, prompt_len);
    assert_bit_exact(
        &got,
        &want,
        &format!("prompt {prompt_len}, partition {partition:?}"),
    );
    snapshot
}

#[test]
#[ignore = "requires serial MLX execution"]
fn an_unwrapped_ring_round_trips_bit_exact() {
    // Six tokens under the 8-token window: the ring has not wrapped.
    let snapshot = round_trip(1, 6, &[4, 2]);
    let model = build_wrapper();
    assert!(model.snapshot_truncatable_to(&snapshot, 4));
    assert!(model.state_checkpoint_from_snapshot(&snapshot).is_none());
}

#[test]
#[ignore = "requires serial MLX execution"]
fn a_wrapped_ring_round_trips_bit_exact() {
    // Twenty-four tokens in two pieces: the ring has wrapped, so the restore
    // has to reproduce the ring geometry and not only the buffer contents.
    let snapshot = round_trip(2, 24, &[10, 14]);
    let model = build_wrapper();
    assert!(!model.snapshot_truncatable_to(&snapshot, 12));
    let checkpoint = model
        .state_checkpoint_from_snapshot(&snapshot)
        .expect("a wrapped ring has a checkpoint");
    assert!(
        checkpoint
            .tensor_names()
            .all(|name| name.starts_with("layer0.rotating."))
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn a_truncation_inside_the_window_matches_the_shorter_prefill_bit_exact() {
    // The source prefilled [4, 2]. Cut back to 4 tokens it holds exactly the
    // rows the first piece computed, which is what a cold prefill of those 4
    // tokens holds.
    let model = build_wrapper();
    let prompt = ids(0..6);
    let (source, cold, restored) = (
        SequenceId::from_raw(SEQ_BASE + 31),
        SequenceId::from_raw(SEQ_BASE + 32),
        SequenceId::from_raw(SEQ_BASE + 33),
    );
    prefill_partition(&model, source, &prompt, &[4, 2]);
    let snapshot = model.snapshot_sequence_state(source, 6).expect("snapshot");
    assert!(model.snapshot_truncatable_to(&snapshot, 4));

    prefill_partition(&model, cold, &prompt[..4], &[4]);
    let want = decode_logits(&model, cold, 4);

    model
        .restore_sequence_state_truncated(restored, &snapshot, 4)
        .expect("truncated restore");
    let got = decode_logits(&model, restored, 4);
    assert_bit_exact(&got, &want, "truncated to 4 of 6");
}

#[test]
#[ignore = "requires serial MLX execution"]
fn a_ring_checkpoint_over_a_longer_donor_matches_the_boundary_bit_exact() {
    // One sequence passes the 12-token boundary and goes on to 20 tokens.
    // The ring checkpoint taken at the boundary plus the 20-token snapshot
    // rebuild the boundary state.
    let model = build_wrapper();
    let prompt = ids(0..20);
    let (source, cold, rebuilt) = (
        SequenceId::from_raw(SEQ_BASE + 41),
        SequenceId::from_raw(SEQ_BASE + 42),
        SequenceId::from_raw(SEQ_BASE + 43),
    );
    prefill_partition(&model, source, &prompt[..12], &[12]);
    let boundary = model.snapshot_sequence_state(source, 12).expect("boundary");
    let checkpoint = model
        .state_checkpoint_from_snapshot(&boundary)
        .expect("ring checkpoint");
    let _ = forward(&model, source, &prompt[12..]);
    let donor = model.snapshot_sequence_state(source, 20).expect("donor");

    prefill_partition(&model, cold, &prompt[..12], &[12]);
    let want = decode_logits(&model, cold, 12);

    model
        .restore_sequence_state_from_checkpoint(rebuilt, &checkpoint, &donor)
        .expect("checkpoint restore");
    let got = decode_logits(&model, rebuilt, 12);
    assert_bit_exact(&got, &want, "checkpoint at 12 over a 20-token donor");
}

/// The tensors of `snapshot` as `(name, shape, dtype, bytes)`.
fn contents(snapshot: &ModelStateSnapshot) -> Vec<(String, Vec<i32>, i32, Vec<u8>)> {
    snapshot
        .tensor_names()
        .map(|name| {
            let tensor = snapshot.tensor(name).expect("named tensor");
            (
                name.to_string(),
                mlxcel_core::array_shape(tensor),
                mlxcel_core::array_dtype(tensor),
                bytes(tensor),
            )
        })
        .collect()
}

#[test]
#[ignore = "requires serial MLX execution"]
fn a_snapshot_has_the_tensors_the_model_owned_path_wrote() {
    // Byte compatibility with the snapshots stored before Gemma 3 became
    // scheduler-owned. The reference is that path itself: the same forward
    // over the model-owned `Cache` enum, serialized by `Cache::snapshot_into`
    // under the Gemma 3 names, which is what `snapshot_sequence_state` did.
    for (case, partition) in [(0u64, &[6usize][..]), (1, &[10, 14][..])] {
        let model = build_wrapper();
        let len: usize = partition.iter().sum();
        let prompt = ids(0..len as i32);
        let seq = SequenceId::from_raw(SEQ_BASE + 50 + case);
        prefill_partition(&model, seq, &prompt, partition);
        for token in ids(len as i32..len as i32 + 3) {
            let _ = forward(&model, seq, &[token]);
        }
        let got = model
            .snapshot_sequence_state(seq, len + 3)
            .expect("snapshot");

        // The pool sequence's sliding layer logs decode writes for the
        // lookahead rewind; the model-owned sequence did too.
        let mut caches = model.model.make_caches();
        for cache in &mut caches {
            if let Cache::Rotating(rotating) = cache {
                rotating.set_decode_undo_depth(
                    mlxcel_core::cache::DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS,
                );
            }
        }
        let mut run = |tokens: &[i32]| {
            let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
            let logits = model.model.forward_with_caches(&input, &mut caches);
            mlxcel_core::cache::flush_decode_undo_rows(caches.iter_mut().filter_map(|c| match c {
                Cache::Rotating(rotating) => Some(rotating),
                Cache::Standard(_) => None,
            }));
            mlxcel_core::eval(&logits);
        };
        let mut start = 0;
        for &piece in partition {
            run(&prompt[start..start + piece]);
            start += piece;
        }
        for token in ids(len as i32..len as i32 + 3) {
            run(&[token]);
        }
        let names = KvSnapshotNames::new("Gemma 3", "standard", "rotating");
        let mut want = ModelStateSnapshot::new("gemma3", len + 3);
        for (idx, cache) in caches.iter().enumerate() {
            cache
                .snapshot_into(&mut want, &format!("layer{idx}"), names)
                .expect("model-owned snapshot");
        }

        assert_eq!(got.family(), want.family());
        assert_eq!(got.token_len(), want.token_len());
        let (got, want) = (contents(&got), contents(&want));
        let name_list = |c: &[(String, Vec<i32>, i32, Vec<u8>)]| -> Vec<String> {
            c.iter().map(|t| t.0.clone()).collect()
        };
        assert_eq!(
            name_list(&got),
            name_list(&want),
            "partition {partition:?}: names"
        );
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.1, w.1, "partition {partition:?}: shape of {}", g.0);
            assert_eq!(g.2, w.2, "partition {partition:?}: dtype of {}", g.0);
            assert!(g.3 == w.3, "partition {partition:?}: contents of {}", g.0);
        }
    }
}
