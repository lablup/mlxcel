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

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::{KVCacheMode, SequenceId};

pub(crate) struct ModelOwnedSequenceState<T> {
    internal: RefCell<Vec<T>>,
    sequences: RefCell<HashMap<SequenceId, Vec<T>>>,
}

/// Shared per-layer KV cache mode table for families whose attention caches
/// live inside the model wrapper instead of the external generator cache slice.
///
/// The table is injected by the CLI generator or server scheduler after they
/// resolve requested/effective KV mode and Boundary-V policy. Constructors use
/// `mode_for_layer` so absent or short tables conservatively fall back to FP16.
pub(crate) struct KvCacheLayerModes {
    modes: RefCell<Option<Vec<KVCacheMode>>>,
}

impl KvCacheLayerModes {
    pub(crate) fn new() -> Self {
        Self {
            modes: RefCell::new(None),
        }
    }

    pub(crate) fn set(&self, modes: Vec<KVCacheMode>) {
        *self.modes.borrow_mut() = Some(modes);
    }

    pub(crate) fn clone_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.modes.borrow().clone()
    }

    pub(crate) fn mode_for_layer(&self, layer_idx: usize) -> KVCacheMode {
        self.modes
            .borrow()
            .as_ref()
            .and_then(|modes| modes.get(layer_idx).copied())
            .unwrap_or(KVCacheMode::Fp16)
    }
}

impl Default for KvCacheLayerModes {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ModelOwnedSequenceState<T> {
    pub(crate) fn new(internal: Vec<T>) -> Self {
        Self {
            internal: RefCell::new(internal),
            sequences: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn replace_internal(&self, internal: Vec<T>) {
        *self.internal.borrow_mut() = internal;
    }

    pub(crate) fn prepare_sequence_state(&self, seq_id: SequenceId, state: Vec<T>) {
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    pub(crate) fn replace_sequence_state(&self, seq_id: SequenceId, state: Vec<T>) {
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    pub(crate) fn with_sequence_state_ref<R>(
        &self,
        seq_id: SequenceId,
        f: impl FnOnce(&[T]) -> R,
    ) -> Option<R> {
        let sequences = self.sequences.borrow();
        sequences.get(&seq_id).map(|state| f(state.as_slice()))
    }

    pub(crate) fn with_sequence_state<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> R {
        let sequence_state = seq_id.and_then(|id| self.sequences.borrow_mut().remove(&id));
        if let Some(mut sequence_state) = sequence_state {
            let result = f(&mut sequence_state);
            self.sequences
                .borrow_mut()
                .insert(seq_id.expect("sequence id must exist"), sequence_state);
            return result;
        }

        let mut internal = self.internal.borrow_mut();
        f(&mut internal)
    }

    pub(crate) fn with_or_create_sequence_state<R>(
        &self,
        seq_id: Option<SequenceId>,
        init: impl FnOnce() -> Vec<T>,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> R {
        if let Some(seq_id) = seq_id {
            let mut sequence_state = self
                .sequences
                .borrow_mut()
                .remove(&seq_id)
                .unwrap_or_else(init);
            let result = f(&mut sequence_state);
            self.sequences.borrow_mut().insert(seq_id, sequence_state);
            return result;
        }

        let mut internal = self.internal.borrow_mut();
        f(&mut internal)
    }

    pub(crate) fn with_existing_sequence_state<R>(
        &self,
        seq_id: SequenceId,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R, String> {
        let mut sequence_state =
            self.sequences.borrow_mut().remove(&seq_id).ok_or_else(|| {
                format!("missing model-owned sequence state for sequence {seq_id}")
            })?;
        let result = f(&mut sequence_state);
        self.sequences.borrow_mut().insert(seq_id, sequence_state);
        Ok(result)
    }

    pub(crate) fn with_batched_sequence_states<R>(
        &self,
        seq_ids: &[SequenceId],
        f: impl FnOnce(&mut [Vec<T>]) -> R,
    ) -> Result<R, String> {
        let mut extracted = Vec::with_capacity(seq_ids.len());
        {
            let mut sequences = self.sequences.borrow_mut();
            for &seq_id in seq_ids {
                let state = sequences.remove(&seq_id).ok_or_else(|| {
                    format!("missing model-owned sequence state for sequence {seq_id}")
                })?;
                extracted.push(state);
            }
        }

        let result = f(&mut extracted);

        let mut sequences = self.sequences.borrow_mut();
        for (seq_id, state) in seq_ids.iter().copied().zip(extracted) {
            sequences.insert(seq_id, state);
        }
        Ok(result)
    }

    pub(crate) fn release_sequence_state(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    #[cfg(test)]
    pub(crate) fn has_sequence_state(&self, seq_id: SequenceId) -> bool {
        self.sequences.borrow().contains_key(&seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelOwnedSequenceState;
    use mlxcel_core::cache::SequenceId;

    #[test]
    fn model_owned_sequence_state_preserves_fallback_between_steps() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);

        state.with_sequence_state(None, |fallback| fallback[0] += 1);
        state.with_sequence_state(None, |fallback| fallback[0] += 1);

        let value = state.with_sequence_state(None, |fallback| fallback[0]);
        assert_eq!(value, 2);
    }

    #[test]
    fn model_owned_sequence_state_isolates_prepared_sequences() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let seq_a = SequenceId::from_raw(1220);
        let seq_b = SequenceId::from_raw(1221);

        state.prepare_sequence_state(seq_a, vec![10]);
        state.prepare_sequence_state(seq_b, vec![20]);

        state
            .with_existing_sequence_state(seq_a, |a| a[0] += 1)
            .expect("seq_a exists");
        state
            .with_existing_sequence_state(seq_b, |b| b[0] += 2)
            .expect("seq_b exists");

        assert_eq!(
            state
                .with_existing_sequence_state(seq_a, |a| a[0])
                .expect("seq_a still exists"),
            11
        );
        assert_eq!(
            state
                .with_existing_sequence_state(seq_b, |b| b[0])
                .expect("seq_b still exists"),
            22
        );
    }

    #[test]
    fn model_owned_sequence_state_refuses_missing_sequence_ids() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let seq = SequenceId::from_raw(1220);

        let err = state
            .with_existing_sequence_state(seq, |slot| slot[0])
            .expect_err("unprepared sequence must fail closed");

        assert!(
            err.contains("missing model-owned sequence state for sequence seq-1220"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn model_owned_sequence_state_release_drops_prepared_slot() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let seq = SequenceId::from_raw(1220);

        state.prepare_sequence_state(seq, vec![1]);
        assert!(state.has_sequence_state(seq));

        state.release_sequence_state(seq);

        assert!(!state.has_sequence_state(seq));
        assert!(
            state
                .with_existing_sequence_state(seq, |slot| slot[0])
                .is_err()
        );
    }
}
