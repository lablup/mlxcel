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

//! Per-sequence image state for Llama 3.2 Vision (`mllama`) (#2567).
//!
//! Mllama keeps image features out of the token stream: the vision tower and
//! projector produce `cross_attention_states` once per request, and every
//! cross-attention layer projects them into a key/value pair exactly once and
//! reuses it on every decode step. Both belong to one request. Before this
//! module the states sat in one `RefCell` on the wrapper and the projected
//! K/V in one `RefCell` per layer, so a text-only request that followed an
//! image request still attended to the image, and two concurrent image
//! requests overwrote each other mid-decode.
//!
//! The container mirrors `Gemma4PerLayerInputsState`: `set_fallback` is
//! written during request preparation (the sequence id is not known yet),
//! `bind_fallback_to_sequence` moves it under the scheduler's id at admission
//! and drains the fallback, `with_entry_mut` resolves a row, and
//! `take_for_sequence` / `bind_for_sequence` carry the entry across a
//! preempt-and-reallocate round trip. A sequence the scheduler never bound
//! reads the fallback, like `MRopeState` and `DeepStackState`: the CLI's
//! raw-completion client prepares the image into the fallback and then
//! forwards under its own `SequenceId` with no bind, and `seq_id == None`
//! reads it too. A sequence bound with an empty fallback is recorded as
//! text-only and never reads the fallback, so an image left there by a
//! preparation whose admission failed cannot reach it; its cross-attention
//! layers are the identity.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

/// One cross-attention layer's fill-once projected image key/value.
pub type CrossKvSlot = Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>;

/// One request's image state: the projected vision features and, per decoder
/// layer index, the key/value the cross-attention layer at that index derived
/// from them (self-attention layers keep an unused `None`).
pub struct MllamaImageState {
    pub(crate) cross_states: UniquePtr<MlxArray>,
    pub(crate) cross_kv: Vec<CrossKvSlot>,
}

impl MllamaImageState {
    /// Fresh state for `cross_states` with every layer's K/V still unfilled.
    pub(crate) fn new(cross_states: UniquePtr<MlxArray>, num_layers: usize) -> Self {
        Self {
            cross_states,
            cross_kv: (0..num_layers).map(|_| None).collect(),
        }
    }
}

/// Fallback slot plus per-`SequenceId` map for [`MllamaImageState`].
///
/// Used by: MllamaVLModel.
///
/// A map value of `None` marks a sequence bound as text-only.
#[derive(Default)]
pub(crate) struct MllamaImageStateStore {
    sequences: RefCell<HashMap<SequenceId, Option<MllamaImageState>>>,
    fallback: RefCell<Option<MllamaImageState>>,
}

impl MllamaImageStateStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Replace the fallback slot (request preparation, CLI, tests).
    pub(crate) fn set_fallback(&self, state: MllamaImageState) {
        *self.fallback.borrow_mut() = Some(state);
    }

    /// Drop the fallback slot; per-sequence entries are untouched.
    pub(crate) fn clear_fallback(&self) {
        *self.fallback.borrow_mut() = None;
    }

    pub(crate) fn has_fallback(&self) -> bool {
        self.fallback.borrow().is_some()
    }

    /// Whether a forward for `seq_id` attends to an image: the sequence's own
    /// entry when it was bound, the fallback slot when it was not.
    pub(crate) fn resolves_image(&self, seq_id: SequenceId) -> bool {
        match self.sequences.borrow().get(&seq_id) {
            Some(entry) => entry.is_some(),
            None => self.has_fallback(),
        }
    }

    /// Move the fallback slot under `seq_id`, draining the fallback. With an
    /// empty fallback (text-only request) the sequence is recorded as
    /// text-only, replacing any stale entry.
    pub(crate) fn bind_fallback_to_sequence(&self, seq_id: SequenceId) {
        let state = self.fallback.borrow_mut().take();
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    /// Remove and return the entry under `seq_id` (preemption). `None` for a
    /// text-only or unbound sequence.
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<MllamaImageState> {
        self.sequences.borrow_mut().remove(&seq_id).flatten()
    }

    /// Re-install a taken entry under a (possibly new) `seq_id`; `None`
    /// records the sequence as text-only.
    pub(crate) fn bind_for_sequence(&self, seq_id: SequenceId, state: Option<MllamaImageState>) {
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    /// Drop a sequence's entry (sequence release or abort).
    pub(crate) fn release_sequence(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    /// Run `f` with mutable access to the row's image state: a bound
    /// sequence's own entry (`None` when it was bound text-only), otherwise
    /// the fallback slot (an unbound `Some(seq_id)` and `seq_id == None`).
    ///
    /// The state is moved out for the duration of `f` and put back afterwards,
    /// so no `RefCell` borrow is held while the forward graph is built.
    pub(crate) fn with_entry_mut<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(Option<&mut MllamaImageState>) -> R,
    ) -> R {
        if let Some(id) = seq_id {
            let taken = self.sequences.borrow_mut().remove(&id);
            if let Some(mut entry) = taken {
                let result = f(entry.as_mut());
                self.sequences.borrow_mut().insert(id, entry);
                return result;
            }
        }
        let mut taken = self.fallback.borrow_mut().take();
        let result = f(taken.as_mut());
        *self.fallback.borrow_mut() = taken;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{MllamaImageState, MllamaImageStateStore};
    use mlxcel_core::cache::SequenceId;

    fn state(value: f32) -> MllamaImageState {
        MllamaImageState::new(
            mlxcel_core::full_f32(&[1, 2, 4], value, mlxcel_core::dtype::FLOAT32),
            3,
        )
    }

    #[test]
    fn bind_moves_fallback_into_the_map_and_drains_it() {
        let store = MllamaImageStateStore::new();
        let seq = SequenceId::from_raw(1);
        store.set_fallback(state(1.0));
        assert!(store.has_fallback());
        store.bind_fallback_to_sequence(seq);
        assert!(!store.has_fallback());
        assert!(store.resolves_image(seq));
    }

    #[test]
    fn bind_with_empty_fallback_records_text_only() {
        let store = MllamaImageStateStore::new();
        let seq = SequenceId::from_raw(2);
        store.set_fallback(state(1.0));
        store.bind_fallback_to_sequence(seq);
        store.bind_fallback_to_sequence(seq);
        assert!(!store.resolves_image(seq));
        // A later, never-bound fallback does not reach the text-only sequence.
        store.set_fallback(state(3.0));
        assert!(!store.resolves_image(seq));
        assert!(!store.with_entry_mut(Some(seq), |entry| entry.is_some()));
    }

    #[test]
    fn with_entry_mut_resolves_sequence_then_fallback_then_text_only() {
        let store = MllamaImageStateStore::new();
        let bound = SequenceId::from_raw(3);
        let unbound = SequenceId::from_raw(4);
        store.set_fallback(state(2.0));
        store.bind_fallback_to_sequence(bound);
        store.set_fallback(state(5.0));

        assert!(store.with_entry_mut(Some(bound), |entry| entry.is_some()));
        // An unbound sequence (the CLI's raw-completion client) reads the
        // fallback, like `seq_id == None`.
        assert!(store.with_entry_mut(Some(unbound), |entry| entry.is_some()));
        assert!(store.with_entry_mut(None, |entry| entry.is_some()));
        // Mutations through the closure persist on the owning slot.
        store.with_entry_mut(Some(bound), |entry| {
            entry.expect("bound").cross_kv[1] = Some((
                mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
                mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
            ));
        });
        assert!(store.with_entry_mut(Some(bound), |entry| {
            entry.expect("bound").cross_kv[1].is_some()
        }));
        assert!(store.with_entry_mut(None, |entry| entry.expect("fallback").cross_kv[1].is_none()));
    }

    #[test]
    fn take_install_release_round_trip() {
        let store = MllamaImageStateStore::new();
        let old = SequenceId::from_raw(5);
        let new = SequenceId::from_raw(6);
        store.set_fallback(state(1.0));
        store.bind_fallback_to_sequence(old);
        let taken = store.take_for_sequence(old).expect("entry was bound");
        assert!(!store.resolves_image(old));
        store.bind_for_sequence(new, Some(taken));
        assert!(store.resolves_image(new));
        store.release_sequence(new);
        assert!(!store.resolves_image(new));
    }
}
