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

//! Per-sequence DeepStack state for the Qwen3-VL family and Cohere Compass
//! (#2567).
//!
//! The vision wrapper computes a request's DeepStack features (one residual
//! injection per early decoder layer, plus the mask of image positions) while
//! preparing its embeddings at enqueue time. The text model consumes them at
//! the request's offset-0 prefill. Before this module the two tensors lived in
//! model-level `RefCell`s: two image requests admitted in one drain tick made
//! the first prefill inject the second request's features, and a text-only
//! prefill landing in between cleared them.
//!
//! The container follows [`crate::models::qwen_mrope_state::MRopeState`], which
//! the same text models already key by sequence, so the scheduler wires both
//! through one bind call:
//!
//! - `set_fallback` is written by the vision wrapper during preparation (it
//!   does not know the sequence id yet).
//! - `bind_fallback_to_sequence(seq_id)` runs at admission, right after
//!   preparation, moving the fallback into the per-`SequenceId` map and
//!   leaving the fallback empty so the next request cannot inherit it.
//! - `with_entry(seq_id, ..)` resolves a row's entry. A bound sequence reads
//!   its own entry; a sequence with no entry, and the CLI's `seq_id == None`,
//!   read the fallback, exactly like `MRopeState::with_entry`.
//! - `take_for_sequence` / `bind_for_sequence` carry an entry across the
//!   scheduler's preempt-and-reallocate round trip, since the re-prefill
//!   injects again without re-running the vision tower.
//! - `release_sequence` drops the entry with the sequence.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

/// One request's DeepStack injection inputs.
///
/// `visual_pos_masks` is the `[1, prompt_len]` boolean mask of image and video
/// token positions; `embeds[i]` is the `[num_image_tokens, hidden]` residual
/// the text model adds after layer `i`.
pub struct DeepStackEntry {
    pub(crate) visual_pos_masks: UniquePtr<MlxArray>,
    pub(crate) embeds: Vec<UniquePtr<MlxArray>>,
}

/// Fallback slot plus per-`SequenceId` map for [`DeepStackEntry`].
///
/// Used by: Qwen3VLModel, Qwen3VLMoeModel (also behind Qwen3-Omni) and
/// CohereCompassTextModel.
#[derive(Default)]
pub(crate) struct DeepStackState {
    sequences: RefCell<HashMap<SequenceId, DeepStackEntry>>,
    fallback: RefCell<Option<DeepStackEntry>>,
}

impl DeepStackState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Store a freshly computed entry in the fallback slot (vision wrapper
    /// during preparation, CLI, single-row tests).
    pub(crate) fn set_fallback(
        &self,
        visual_pos_masks: UniquePtr<MlxArray>,
        embeds: Vec<UniquePtr<MlxArray>>,
    ) {
        *self.fallback.borrow_mut() = Some(DeepStackEntry {
            visual_pos_masks,
            embeds,
        });
    }

    /// Drop the fallback slot. The per-sequence map is untouched. The server
    /// scheduler also calls this (through the text model) when admission
    /// aborts between media preparation and the bind (issue #2625).
    pub(crate) fn clear_fallback(&self) {
        *self.fallback.borrow_mut() = None;
    }

    /// Move the fallback slot's entry under `seq_id`, draining the fallback.
    /// With an empty fallback (text-only request) any stale entry under
    /// `seq_id` is removed instead, so the sequence starts without features.
    pub(crate) fn bind_fallback_to_sequence(&self, seq_id: SequenceId) {
        let entry = self.fallback.borrow_mut().take();
        let mut sequences = self.sequences.borrow_mut();
        match entry {
            Some(entry) => {
                sequences.insert(seq_id, entry);
            }
            None => {
                sequences.remove(&seq_id);
            }
        }
    }

    /// Drop a sequence's entry (sequence release or abort).
    pub(crate) fn release_sequence(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    /// Remove and return the entry under `seq_id` (preemption).
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<DeepStackEntry> {
        self.sequences.borrow_mut().remove(&seq_id)
    }

    /// Re-install a taken entry under a (possibly new) `seq_id`.
    pub(crate) fn bind_for_sequence(&self, seq_id: SequenceId, entry: DeepStackEntry) {
        self.sequences.borrow_mut().insert(seq_id, entry);
    }

    /// Whether the row resolved by `seq_id` has DeepStack features to inject.
    pub(crate) fn has_entry(&self, seq_id: Option<SequenceId>) -> bool {
        self.with_entry(seq_id, |entry| {
            entry.is_some_and(|entry| !entry.embeds.is_empty())
        })
    }

    /// Run `f` with the entry for `seq_id`: the sequence's own entry when it
    /// has one, otherwise the fallback slot (which the scheduler drains at
    /// admission, so a bound text-only row sees `None`).
    pub(crate) fn with_entry<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(Option<&DeepStackEntry>) -> R,
    ) -> R {
        if let Some(id) = seq_id {
            let map = self.sequences.borrow();
            if let Some(entry) = map.get(&id) {
                return f(Some(entry));
            }
        }
        let fallback = self.fallback.borrow();
        f(fallback.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn has_sequence(&self, seq_id: SequenceId) -> bool {
        self.sequences.borrow().contains_key(&seq_id)
    }

    #[cfg(test)]
    pub(crate) fn has_fallback(&self) -> bool {
        self.fallback.borrow().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::DeepStackState;
    use mlxcel_core::cache::SequenceId;

    fn entry(state: &DeepStackState, layers: usize) {
        state.set_fallback(
            mlxcel_core::zeros(&[1, 4], mlxcel_core::dtype::BOOL),
            (0..layers)
                .map(|_| mlxcel_core::zeros(&[2, 8], mlxcel_core::dtype::FLOAT32))
                .collect(),
        );
    }

    #[test]
    fn bind_moves_fallback_into_the_map_and_drains_it() {
        let state = DeepStackState::new();
        let seq = SequenceId::from_raw(1);
        entry(&state, 1);
        assert!(state.has_fallback());
        state.bind_fallback_to_sequence(seq);
        assert!(!state.has_fallback());
        assert!(state.has_sequence(seq));
        assert!(state.has_entry(Some(seq)));
        assert!(!state.has_entry(None));
    }

    #[test]
    fn bind_with_empty_fallback_removes_a_stale_entry() {
        let state = DeepStackState::new();
        let seq = SequenceId::from_raw(2);
        entry(&state, 1);
        state.bind_fallback_to_sequence(seq);
        state.bind_fallback_to_sequence(seq);
        assert!(!state.has_sequence(seq));
        assert!(!state.has_entry(Some(seq)));
    }

    #[test]
    fn unbound_sequence_reads_the_fallback_like_mrope() {
        let state = DeepStackState::new();
        entry(&state, 1);
        assert!(state.has_entry(Some(SequenceId::from_raw(3))));
        state.clear_fallback();
        assert!(!state.has_entry(Some(SequenceId::from_raw(3))));
    }

    #[test]
    fn take_install_release_round_trip() {
        let state = DeepStackState::new();
        let old = SequenceId::from_raw(4);
        let new = SequenceId::from_raw(5);
        entry(&state, 2);
        state.bind_fallback_to_sequence(old);
        let taken = state.take_for_sequence(old).expect("entry was bound");
        assert!(!state.has_sequence(old));
        state.bind_for_sequence(new, taken);
        assert!(state.has_sequence(new));
        state.release_sequence(new);
        assert!(!state.has_sequence(new));
        assert!(state.take_for_sequence(new).is_none());
    }
}
