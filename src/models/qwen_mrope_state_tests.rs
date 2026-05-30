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

//! Per-sequence MRoPE state tests.

use super::{MRopeEntry, MRopeState};
use mlxcel_core::cache::SequenceId;

/// Build a tiny `[3, 1, 1]` position-id tensor so the entry has a non-trivial
/// `position_ids` slot that we can identify by its scalar payload.
fn make_pos_ids(value: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(&[value, value, value], &[3, 1, 1])
}

fn entry_delta(entry: &MRopeEntry) -> Option<i32> {
    entry.rope_deltas
}

#[test]
fn fallback_starts_empty() {
    let state = MRopeState::new();
    state.with_entry(None, |entry| {
        assert!(entry.position_ids.is_none());
        assert!(entry.rope_deltas.is_none());
    });
}

#[test]
fn fallback_set_then_read() {
    let state = MRopeState::new();
    state.set_fallback(make_pos_ids(42), 7);
    state.with_entry(None, |entry| {
        assert_eq!(entry_delta(entry), Some(7));
        assert!(entry.position_ids.is_some());
    });
}

#[test]
fn fallback_clear_resets_entry() {
    let state = MRopeState::new();
    state.set_fallback(make_pos_ids(11), 3);
    state.clear_fallback();
    state.with_entry(None, |entry| {
        assert!(entry.position_ids.is_none());
        assert_eq!(entry_delta(entry), None);
    });
}

/// invariant: per-sequence deltas must be isolated. Set the
/// fallback to a "VL-row" delta of 5, then register a text-only sequence
/// with delta 0 — looking up either by seq id should yield delta=0, while
/// a lookup with `None` still returns the fallback's delta=5.
#[test]
fn per_sequence_state_isolates_text_only_row_from_vl_row() {
    let state = MRopeState::new();
    let text_seq = SequenceId::from_raw(1);

    // Pretend a VL prefill ran first and stored its delta in the fallback
    // slot (mirrors a legacy CLI path or an early call site that did not
    // yet plumb a seq_id).
    state.set_fallback(make_pos_ids(99), 5);
    // A subsequent text-only request registers under its own sequence id.
    state.set_for_sequence(text_seq, make_pos_ids(0), 0);

    state.with_entry(Some(text_seq), |entry| {
        assert_eq!(entry_delta(entry), Some(0));
    });
    state.with_entry(None, |entry| {
        assert_eq!(entry_delta(entry), Some(5));
    });
}

#[test]
fn missing_sequence_falls_back_to_legacy_slot() {
    let state = MRopeState::new();
    state.set_fallback(make_pos_ids(0), 0);
    let unknown = SequenceId::from_raw(99);
    state.with_entry(Some(unknown), |entry| {
        // No per-sequence entry — fallback delta surfaces.
        assert_eq!(entry_delta(entry), Some(0));
    });
}

#[test]
fn release_drops_only_target_sequence() {
    let state = MRopeState::new();
    let a = SequenceId::from_raw(1);
    let b = SequenceId::from_raw(2);
    state.set_for_sequence(a, make_pos_ids(0), 5);
    state.set_for_sequence(b, make_pos_ids(0), 11);

    state.release_sequence(a);

    state.with_entry(Some(a), |entry| {
        // a was released — falls back to fallback (which is empty).
        assert!(entry.rope_deltas.is_none());
    });
    state.with_entry(Some(b), |entry| {
        // b is still present.
        assert_eq!(entry_delta(entry), Some(11));
    });
}

/// Acceptance criterion #3 (unit-test variant): for a
/// contrived mixed batch where row 0 has delta=5 and row 1 has delta=0,
/// the per-row delta tensor produced by stacking each sequence's entry
/// is `[5, 0]`. We construct this by reading back each row's stored
/// delta in turn.
#[test]
fn mixed_batch_per_row_delta_tensor_matches_expectation() {
    let state = MRopeState::new();
    let row0 = SequenceId::from_raw(101);
    let row1 = SequenceId::from_raw(102);

    state.set_for_sequence(row0, make_pos_ids(0), 5);
    state.set_for_sequence(row1, make_pos_ids(0), 0);

    let row_ids = [row0, row1];
    let per_row: Vec<i32> = row_ids
        .iter()
        .map(|sid| state.with_entry(Some(*sid), |entry| entry.rope_deltas.unwrap_or(0)))
        .collect();

    assert_eq!(
        per_row,
        vec![5, 0],
        "Per-row MRoPE delta tensor must match upstream PR #1095 invariant"
    );
}

/// Server flow: the vision wrapper writes the freshly computed MRoPE state
/// to the fallback slot before the scheduler knows the request's sequence
/// id. After the scheduler binds it to the seq id, looking up by id must
/// see those values *and* the fallback slot must be empty so the next
/// request cannot inherit them.
#[test]
fn bind_fallback_to_sequence_transfers_state_and_clears_fallback() {
    let state = MRopeState::new();
    state.set_fallback(make_pos_ids(0), 9);

    let seq = SequenceId::from_raw(7);
    state.bind_fallback_to_sequence(seq);

    state.with_entry(Some(seq), |entry| {
        assert_eq!(entry_delta(entry), Some(9));
        assert!(entry.position_ids.is_some());
    });
    state.with_entry(None, |entry| {
        // Fallback was drained by the bind operation.
        assert!(entry.position_ids.is_none());
        assert_eq!(entry_delta(entry), None);
    });
}

/// Even when the fallback slot is empty (text-only request, no VL prefill
/// ran), binding must still register an entry under the sequence id so
/// subsequent lookups see `rope_deltas == None` for that sequence — i.e.,
/// the text-only path with no leaked delta from a prior request.
#[test]
fn bind_fallback_to_sequence_registers_empty_entry_for_text_only() {
    let state = MRopeState::new();
    let seq = SequenceId::from_raw(8);
    state.bind_fallback_to_sequence(seq);

    state.with_entry(Some(seq), |entry| {
        assert!(entry.position_ids.is_none());
        assert_eq!(entry_delta(entry), None);
    });
}

/// Cross-request leak regression: register a VL request first, then bind
/// a second text-only request. The text-only request must not inherit the
/// VL delta even though the fallback slot may have been left set by some
/// earlier non-server caller.
#[test]
fn text_only_request_does_not_inherit_vl_delta_after_bind() {
    let state = MRopeState::new();

    // First sequence: VL prefill placed delta=5 into the fallback slot,
    // then the scheduler bound it to row0.
    state.set_fallback(make_pos_ids(0), 5);
    let row0 = SequenceId::from_raw(201);
    state.bind_fallback_to_sequence(row0);

    // Second sequence: text-only — no fallback writeback happens, but the
    // scheduler still calls bind. The new entry must show delta=None.
    let row1 = SequenceId::from_raw(202);
    state.bind_fallback_to_sequence(row1);

    state.with_entry(Some(row0), |entry| {
        assert_eq!(entry_delta(entry), Some(5));
    });
    state.with_entry(Some(row1), |entry| {
        assert_eq!(entry_delta(entry), None);
    });
}

/// follow-up: preemption regression coverage.
///
/// The server preemption path evicts a victim sequence (releasing its old
/// id), then re-allocates a fresh id and re-queues the same request. The
/// MRoPE entry must survive that round trip. Without `take_for_sequence` +
/// `bind_for_sequence`, the entry would be dropped along with the old id
/// and the re-prefill would treat a VL prompt as text-only.
#[test]
fn preemption_round_trip_carries_mrope_entry_to_new_id() {
    let state = MRopeState::new();
    let id_a = SequenceId::from_raw(301);
    let id_b = SequenceId::from_raw(302);

    // Original VL prefill: delta=5 stored under id_a.
    state.set_for_sequence(id_a, make_pos_ids(7), 5);

    // Eviction simulation: take the entry out, then drop the old id
    // (release_sequence is called by the scheduler's
    // release_sequence_caches path).
    let entry = state.take_for_sequence(id_a).expect("entry must exist");
    state.release_sequence(id_a);

    // After release the old id resolves only via the (still-empty)
    // fallback — confirms the take + release actually drained id_a.
    state.with_entry(Some(id_a), |e| {
        assert!(e.position_ids.is_none());
        assert_eq!(entry_delta(e), None);
    });

    // Re-queue under a fresh id; install the saved entry.
    state.bind_for_sequence(id_b, entry);

    state.with_entry(Some(id_b), |e| {
        assert_eq!(entry_delta(e), Some(5));
        assert!(e.position_ids.is_some());
    });
}

/// `take_for_sequence` returns `None` when no entry exists for the id —
/// callers (e.g. text-only preemption victims, non-Qwen-VL models) must
/// be able to ask without panicking.
#[test]
fn take_for_sequence_returns_none_for_unknown_id() {
    let state = MRopeState::new();
    let unknown = SequenceId::from_raw(401);
    assert!(state.take_for_sequence(unknown).is_none());
}
