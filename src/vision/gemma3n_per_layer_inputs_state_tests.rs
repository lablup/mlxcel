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

//! Unit tests for the Gemma 3n per-layer-inputs per-sequence state container
//! (issue #85).
//!
//! These cover the burst-enqueue race the container is designed to close:
//! when 2+ Gemma 3n VLM requests are enqueued within one
//! `drain_incoming_requests` tick, each `prepare_request_vlm_embeddings`
//! writes the fallback slot and the scheduler's
//! `bind_fallback_to_sequence(seq_id)` must isolate the freshly written
//! tensor under that sequence id before the next request overwrites the
//! fallback. The "preemption round trip" test covers the take/install
//! round trip mirroring [`super::Gemma3nPerLayerInputsState::bind_for_sequence`].

use super::Gemma3nPerLayerInputsState;
use mlxcel_core::cache::SequenceId;

/// Build a tiny `[1, 1, 1, 1]` per-layer-inputs tensor whose single value
/// identifies the producer. The real shape is `[1, T, L, h_per_layer]`
/// but the state container is shape-agnostic — the tests only need to
/// verify which writer's tensor surfaces at each lookup.
fn make_pli(value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_f32(&[value], &[1, 1, 1, 1])
}

fn read_pli_scalar(arr: &mlxcel_core::MlxArray) -> f32 {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
#[ignore = "requires serial MLX execution"]
fn fallback_round_trip_take_returns_value() {
    // Legacy CLI / bench path: `set_fallback` then `take_fallback`
    // returns the tensor and leaves the fallback empty for the next
    // caller.
    let state = Gemma3nPerLayerInputsState::new();
    state.set_fallback(Some(make_pli(7.0)));
    let taken = state.take_fallback().expect("fallback held a value");
    assert!((read_pli_scalar(taken.as_ref().unwrap()) - 7.0).abs() < 1e-5);
    assert!(
        state.take_fallback().is_none(),
        "second take must observe an empty fallback"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn set_fallback_with_none_clears_slot() {
    // Defensive: passing `None` must wipe a stale value left by an
    // earlier turn so the next prefill does not inherit it.
    let state = Gemma3nPerLayerInputsState::new();
    state.set_fallback(Some(make_pli(3.0)));
    state.set_fallback(None);
    assert!(
        state.take_fallback().is_none(),
        "set_fallback(None) must clear the slot"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn bind_fallback_drains_into_per_sequence_slot() {
    // Server flow: vision wrapper writes the freshly projected tensor
    // to the fallback before the scheduler knows the request's id; the
    // scheduler then binds it to the allocated id. After bind the
    // fallback must be empty so a subsequent request cannot inherit it.
    let state = Gemma3nPerLayerInputsState::new();
    let seq = SequenceId::from_raw(11);
    state.set_fallback(Some(make_pli(42.0)));
    state.bind_fallback_to_sequence(seq);

    let consumed = state
        .take_for_sequence(seq)
        .expect("seq must hold the bound tensor");
    assert!((read_pli_scalar(consumed.as_ref().unwrap()) - 42.0).abs() < 1e-5);
    assert!(
        state.take_fallback().is_none(),
        "bind_fallback_to_sequence must drain the fallback"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn bind_fallback_with_empty_slot_leaves_seq_absent() {
    // Text-only request after a Gemma 3n VLM model load: the wrapper
    // sets the fallback to `None` and the scheduler still calls
    // `bind_fallback_to_sequence`. The map must be left without an
    // entry so a `take_for_sequence` returns `None` and the prefill
    // path falls through to the (still empty) fallback slot — which
    // matches the no-per-layer-inputs semantics.
    let state = Gemma3nPerLayerInputsState::new();
    let seq = SequenceId::from_raw(12);
    state.bind_fallback_to_sequence(seq);
    assert!(
        state.take_for_sequence(seq).is_none(),
        "binding an empty fallback must leave seq absent so the prefill consumer surfaces None"
    );
}

/// Issue #85 root-cause regression: simulate the burst-enqueue race.
///
/// `prepare_request_vlm_embeddings` writes the fallback slot once per
/// request; the scheduler's bind step claims it before the next
/// request runs. Without the bind step, the second
/// `prepare_request_vlm_embeddings` would overwrite the cell and the
/// first prefill would consume the wrong tensor — or, when the timing
/// flips, the second prefill would observe `None` and the legacy
/// `take().unwrap()` would panic.
#[test]
#[ignore = "requires serial MLX execution"]
fn burst_enqueue_race_each_seq_consumes_its_own_tensor() {
    let state = Gemma3nPerLayerInputsState::new();
    let row_a = SequenceId::from_raw(101);
    let row_b = SequenceId::from_raw(102);

    // Enqueue tick: two Gemma 3n VLM requests arrive back-to-back.

    // Request A: vision wrapper writes its tensor to the fallback.
    state.set_fallback(Some(make_pli(1.0)));
    // Scheduler binds it under row A's seq id immediately after.
    state.bind_fallback_to_sequence(row_a);

    // Request B: same producer/consumer pair, different value.
    state.set_fallback(Some(make_pli(2.0)));
    state.bind_fallback_to_sequence(row_b);

    // Prefill tick: each row consumes its own tensor.
    let pli_a = state.take_for_sequence(row_a).expect("row_a tensor");
    let pli_b = state.take_for_sequence(row_b).expect("row_b tensor");

    assert!(
        (read_pli_scalar(pli_a.as_ref().unwrap()) - 1.0).abs() < 1e-5,
        "row_a must consume its own tensor (1.0), not row_b's (2.0)"
    );
    assert!(
        (read_pli_scalar(pli_b.as_ref().unwrap()) - 2.0).abs() < 1e-5,
        "row_b must consume its own tensor (2.0)"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn release_drops_only_target_sequence() {
    let state = Gemma3nPerLayerInputsState::new();
    let a = SequenceId::from_raw(201);
    let b = SequenceId::from_raw(202);
    state.set_fallback(Some(make_pli(10.0)));
    state.bind_fallback_to_sequence(a);
    state.set_fallback(Some(make_pli(20.0)));
    state.bind_fallback_to_sequence(b);

    state.release_sequence(a);

    assert!(
        state.take_for_sequence(a).is_none(),
        "released seq must no longer resolve"
    );
    assert!(
        state.take_for_sequence(b).is_some(),
        "untouched seq must still resolve"
    );
}

/// Preemption round trip — mirrors the Gemma 4 / Qwen MRoPE preemption
/// flows.
///
/// Eviction releases the old id; the saved tensor must survive the
/// release and reinstall under the freshly allocated id so the
/// re-prefill sees the same projection (`prepare_request_vlm_embeddings`
/// does not re-run on re-prefill).
#[test]
#[ignore = "requires serial MLX execution"]
fn preemption_round_trip_carries_tensor_to_new_id() {
    let state = Gemma3nPerLayerInputsState::new();
    let id_a = SequenceId::from_raw(301);
    let id_b = SequenceId::from_raw(302);

    // Original prefill: tensor stored under id_a.
    state.set_fallback(Some(make_pli(99.0)));
    state.bind_fallback_to_sequence(id_a);

    // Eviction simulation: take out the entry, release the id.
    let saved = state.take_for_sequence(id_a).expect("entry must exist");
    state.release_sequence(id_a);

    assert!(
        state.take_for_sequence(id_a).is_none(),
        "release must drain id_a"
    );

    // Re-queue under a fresh id; install the saved entry.
    state.bind_for_sequence(id_b, saved);

    let consumed = state
        .take_for_sequence(id_b)
        .expect("id_b must hold tensor");
    assert!((read_pli_scalar(consumed.as_ref().unwrap()) - 99.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn take_for_sequence_returns_none_for_unknown_id() {
    let state = Gemma3nPerLayerInputsState::new();
    let unknown = SequenceId::from_raw(401);
    assert!(state.take_for_sequence(unknown).is_none());
}

#[test]
#[ignore = "requires serial MLX execution"]
fn unknown_sequence_cannot_consume_another_requests_fallback() {
    let state = Gemma3nPerLayerInputsState::new();
    state.set_fallback(Some(make_pli(77.0)));
    assert!(state.take_for_sequence(SequenceId::from_raw(999)).is_none());
    let fallback = state
        .take_fallback()
        .expect("unknown sequence must not drain fallback");
    assert!((read_pli_scalar(fallback.as_ref().unwrap()) - 77.0).abs() < 1e-5);
}
