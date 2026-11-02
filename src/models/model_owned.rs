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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use mlxcel_core::cache::{KVCacheMode, RotatingKVCache, SequenceId};

pub(crate) struct ModelOwnedSequenceState<T> {
    internal: RefCell<Vec<T>>,
    sequences: RefCell<HashMap<SequenceId, Vec<T>>>,
}

/// What a model-owned layer keeps for attention, as its cache builder
/// creates it. Families report storage through
/// [`KvCacheLayerModes::storage_modes`] (or [`table_less_storage_modes`])
/// with the same predicate their builder uses, so the two cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKv {
    /// A `KVCache` built with the injected mode.
    Full,
    /// A `RotatingKVCache` over this window, which stores what
    /// `RotatingKVCache::stored_mode` says.
    Sliding(i32),
    /// An attention cache that ignores the mode and stores FP16 (Llama 4's
    /// `ChunkedKVCache`, a family with no mode table).
    Fp16Only,
    /// No attention KV at all (a linear-attention, recurrent, SSM or
    /// convolution layer, or a skipped attention block).
    None,
}

impl LayerKv {
    /// The mode a layer of this kind stores when `mode` is injected, `None`
    /// for a layer that holds no KV cache.
    pub(crate) fn stored_mode(self, mode: KVCacheMode) -> Option<KVCacheMode> {
        match self {
            Self::Full => Some(mode),
            Self::Sliding(window) => Some(RotatingKVCache::stored_mode(window, mode)),
            Self::Fp16Only => Some(KVCacheMode::Fp16),
            Self::None => None,
        }
    }
}

/// Storage report of a model-owned family that has no mode table and never
/// reads `KVCacheMode`: every layer holding attention KV stores FP16, the
/// rest hold none. Returning it (rather than `None`) keeps the server from
/// falling back to the configured table and claiming every layer applied
/// (issue #2790).
pub(crate) fn table_less_storage_modes(
    n_layers: usize,
    layer_kind: impl Fn(usize) -> LayerKv,
) -> Option<Vec<Option<KVCacheMode>>> {
    Some(
        (0..n_layers)
            .map(|layer_idx| layer_kind(layer_idx).stored_mode(KVCacheMode::Fp16))
            .collect(),
    )
}

// The shared window helpers live in `mlxcel_core::cache` (issue #2798) so the
// drafters and `mlxcel-xla` can call them; the model layer re-exports the one
// its families use.
pub(crate) use mlxcel_core::cache::cache_window;

/// Shared per-layer KV cache mode table for families whose attention caches
/// live inside the model wrapper instead of the external generator cache slice.
///
/// The table is injected by the CLI generator or server scheduler after they
/// resolve requested/effective KV mode and Boundary-V policy. Constructors use
/// `mode_for_layer` so absent or short tables conservatively fall back to FP16.
/// Sliding-window layers use `rotating_mode_for_layer` instead, and report
/// what they really store through `storage_modes` (issue #2783).
pub(crate) struct KvCacheLayerModes {
    modes: RefCell<Option<Vec<KVCacheMode>>>,
    /// Set once the misaligned-window `Turbo4Asym` fallback has been logged,
    /// so rebuilding caches per sequence does not repeat it.
    misaligned_window_warned: Cell<bool>,
}

impl KvCacheLayerModes {
    pub(crate) fn new() -> Self {
        Self {
            modes: RefCell::new(None),
            misaligned_window_warned: Cell::new(false),
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

    /// Mode to build layer `layer_idx`'s `RotatingKVCache` of size `window`
    /// with.
    ///
    /// `RotatingKVCache::new_with_mode` panics for `Turbo4Asym` on a window
    /// that is not a positive multiple of `turbo::BLOCK_SIZE`, so that case
    /// falls back to FP16 with one warning per table. Every other mode passes
    /// through unchanged on purpose: the cache already stores FP16 for them,
    /// and rewriting the mode would change `can_truncate_to` and snapshot
    /// reuse for those caches.
    pub(crate) fn rotating_mode_for_layer(&self, layer_idx: usize, window: i32) -> KVCacheMode {
        let mode = self.mode_for_layer(layer_idx);
        if mode == KVCacheMode::Turbo4Asym
            && RotatingKVCache::stored_mode(window, mode) != KVCacheMode::Turbo4Asym
        {
            if !self.misaligned_window_warned.replace(true) {
                tracing::warn!(
                    window,
                    "fp16+turbo4 needs a sliding window that is a positive multiple of {}; \
                     sliding-window layers with window {window} use fp16 KV instead",
                    mlxcel_core::cache::turbo::BLOCK_SIZE
                );
            }
            return KVCacheMode::Fp16;
        }
        mode
    }

    /// Per-layer storage modes for the injected table.
    ///
    /// `layer_kind(layer_idx)` says what the family's cache builder creates
    /// for that layer, with the same predicate the builder uses; see
    /// [`LayerKv::stored_mode`]. `None` entries are layers with no KV cache.
    pub(crate) fn storage_modes(
        &self,
        layer_kind: impl Fn(usize) -> LayerKv,
    ) -> Option<Vec<Option<KVCacheMode>>> {
        let modes = self.clone_modes()?;
        Some(
            modes
                .into_iter()
                .enumerate()
                .map(|(layer_idx, mode)| layer_kind(layer_idx).stored_mode(mode))
                .collect(),
        )
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

    /// Validate a complete identified batch before any row can mutate state.
    pub(crate) fn validate_sequence_states(&self, seq_ids: &[SequenceId]) -> Result<(), String> {
        let sequences = self.sequences.borrow();
        let mut unique = HashSet::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            if !unique.insert(seq_id) {
                return Err(format!(
                    "duplicate model-owned sequence state for sequence {seq_id}"
                ));
            }
            if !sequences.contains_key(&seq_id) {
                return Err(format!(
                    "missing model-owned sequence state for sequence {seq_id}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn with_batched_sequence_states<R>(
        &self,
        seq_ids: &[SequenceId],
        f: impl FnOnce(&mut [Vec<T>]) -> R,
    ) -> Result<R, String> {
        // Validate the complete request before removing any entries. Without
        // this pass a duplicate or later missing id returned early after
        // earlier states had already been extracted, permanently dropping
        // those live sequences from the map.
        self.validate_sequence_states(seq_ids)?;

        let mut extracted = Vec::with_capacity(seq_ids.len());
        {
            let mut sequences = self.sequences.borrow_mut();
            for &seq_id in seq_ids {
                let state = sequences
                    .remove(&seq_id)
                    .expect("prevalidated model-owned sequence state must exist");
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

    #[test]
    fn model_owned_batched_state_updates_distinct_rows() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let seq_a = SequenceId::from_raw(1220);
        let seq_b = SequenceId::from_raw(1221);
        state.prepare_sequence_state(seq_a, vec![10]);
        state.prepare_sequence_state(seq_b, vec![20]);

        state
            .with_batched_sequence_states(&[seq_a, seq_b], |rows| {
                rows[0][0] += 1;
                rows[1][0] += 2;
            })
            .expect("distinct prepared rows must be borrowed together");

        assert_eq!(
            state
                .with_existing_sequence_state(seq_a, |row| row[0])
                .expect("seq_a remains available"),
            11
        );
        assert_eq!(
            state
                .with_existing_sequence_state(seq_b, |row| row[0])
                .expect("seq_b remains available"),
            22
        );
    }

    #[test]
    fn model_owned_batched_state_rejects_duplicates_without_dropping_state() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let seq = SequenceId::from_raw(1220);
        state.prepare_sequence_state(seq, vec![10]);

        let err = state
            .with_batched_sequence_states(&[seq, seq], |_| ())
            .expect_err("duplicate ids must be rejected");
        assert!(err.contains("duplicate model-owned sequence state"));
        assert_eq!(
            state
                .with_existing_sequence_state(seq, |row| row[0])
                .expect("rejected batch must preserve seq state"),
            10
        );
    }

    #[test]
    fn model_owned_batched_state_rejects_missing_id_without_dropping_state() {
        let state = ModelOwnedSequenceState::new(vec![0usize]);
        let present = SequenceId::from_raw(1220);
        let missing = SequenceId::from_raw(1221);
        state.prepare_sequence_state(present, vec![10]);

        let err = state
            .with_batched_sequence_states(&[present, missing], |_| ())
            .expect_err("missing ids must be rejected");
        assert!(err.contains("missing model-owned sequence state"));
        assert_eq!(
            state
                .with_existing_sequence_state(present, |row| row[0])
                .expect("rejected batch must preserve preceding state"),
            10
        );
    }
}
