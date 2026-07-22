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

//! Per-sequence `per_layer_inputs` state for Gemma 3n VLM (issue #85).
//!
//! `Gemma3nVLModel::get_input_embeddings` projects the Gemma 3n
//! `per_layer_inputs` tensor (shape `[1, T, L, h]`) while preparing a
//! request's VLM embeddings. That tensor is consumed later, at prefill
//! time, by `Gemma3nVLModel::forward_with_embeddings_and_sequence_id`
//! when it calls
//! `text_model.language_model.forward_with_inputs(embeds, &pli, caches)`.
//!
//! Before this module landed, the projected tensor lived on a single
//! `RefCell<Option<UniquePtr<MlxArray>>>` field (`cached_per_layer_inputs`)
//! on `Gemma3nVLModel`. The scheduler's `drain_incoming_requests` runs
//! the embedding-prep step once per request at enqueue time and the
//! prefill step asynchronously at scheduling time. When 2+ Gemma 3n VLM
//! requests are enqueued before the first one prefills (a typical burst
//! on an idle server), each `prepare_request_vlm_embeddings` overwrites
//! the same cell. The first prefill then `take()`s **the latest writer's**
//! `per_layer_inputs` instead of its own, or — if the prepares interleave
//! with the prefills such that the slot is already drained — panics on
//! `Option::unwrap()` at `gemma3n_vl.rs:145`.
//!
//! The container below mirrors
//! [`crate::vision::gemma4_per_layer_inputs_state::Gemma4PerLayerInputsState`]
//! so the scheduler wires both per-sequence states identically. The same
//! comment about lifecycle ownership applies:
//!
//! - **Per-`SequenceId` map** is the primary slot;
//!   `bind_fallback_to_sequence` transfers a freshly written fallback
//!   into this map under the scheduler-allocated id, draining the
//!   fallback in the same step so the next request cannot inherit the
//!   previous row's tensor.
//! - **Fallback slot** preserves only legacy single-instance callers (CLI
//!   `mlxcel generate`, `mlxcel-bench-decode`, single-row tests). A request
//!   carrying a `SequenceId` never reads this slot, because it could belong to
//!   a different concurrently-prepared request.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-sequence container for Gemma 3n VLM `per_layer_inputs` tensors.
///
/// Used by: [`crate::vision::Gemma3nVLModel`]. The value stored is the
/// projected `per_layer_inputs` tensor that
/// `Gemma3nLanguageModel::project_per_layer_inputs` produces; the prefill
/// path then passes it into `forward_with_inputs` so the language model
/// can blend the per-layer projection into its layer stack.
#[derive(Default)]
pub(crate) struct Gemma3nPerLayerInputsState {
    sequences: RefCell<HashMap<SequenceId, UniquePtr<MlxArray>>>,
    fallback: RefCell<Option<UniquePtr<MlxArray>>>,
}

impl Gemma3nPerLayerInputsState {
    /// Construct an empty state container.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Store `per_layer_inputs` in the fallback slot. Used by the legacy
    /// single-instance call path (`mlxcel-bench-decode`, CLI
    /// `mlxcel generate`, single-row tests) before any sequence id has
    /// been allocated.
    ///
    /// `per_layer_inputs == None` clears the fallback.
    pub(crate) fn set_fallback(&self, per_layer_inputs: Option<UniquePtr<MlxArray>>) {
        *self.fallback.borrow_mut() = per_layer_inputs;
    }

    /// Move the fallback slot's contents into the per-sequence map under
    /// `seq_id`. Drains the fallback so the next request cannot inherit
    /// the previous row's tensor — this is the burst-enqueue race fix
    /// from issue #85.
    ///
    /// If the fallback slot is empty (text-only request after a Gemma 3n
    /// VLM model load — `prepare_request_vlm_embeddings` returns `None`
    /// in that case and never writes the slot), `seq_id` is left absent
    /// from the map. A subsequent `take_for_sequence(seq_id)` returns
    /// `None`, and the prefill consumer can decide whether to fall back
    /// to the slot or treat that as a hard error.
    pub(crate) fn bind_fallback_to_sequence(&self, seq_id: SequenceId) {
        let mut fallback = self.fallback.borrow_mut();
        if let Some(value) = fallback.take() {
            self.sequences.borrow_mut().insert(seq_id, value);
        }
    }

    /// Take a row's per-layer-inputs out of the map for prefill
    /// consumption. Returns `None` when no entry exists; sequence-scoped
    /// consumers treat that as an invariant violation and never fall back.
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<UniquePtr<MlxArray>> {
        self.sequences.borrow_mut().remove(&seq_id)
    }

    /// Drain the fallback slot. Used by the legacy CLI / bench prefill
    /// path (`Gemma3nVLModel::forward_with_embeddings_and_sequence_id`
    /// with `seq_id == None`) to consume the tensor exactly once,
    /// mirroring the original `RefCell::take()` semantics.
    pub(crate) fn take_fallback(&self) -> Option<UniquePtr<MlxArray>> {
        self.fallback.borrow_mut().take()
    }

    /// Drop a single sequence's stored tensor. Called from
    /// `Gemma3nVLModel::release_sequence_state_by_id` so the map drains
    /// in step with the scheduler's per-sequence cache release. No-op
    /// when the map has no entry for the id (text-only request, request
    /// that finished before the binding step ran).
    pub(crate) fn release_sequence(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    /// Re-install a previously taken tensor under `seq_id`. Used by the
    /// preemption path so a VL row that was evicted before completing
    /// prefill can resume under its freshly allocated id without
    /// rerunning the vision encoder.
    pub(crate) fn bind_for_sequence(&self, seq_id: SequenceId, value: UniquePtr<MlxArray>) {
        self.sequences.borrow_mut().insert(seq_id, value);
    }
}

#[cfg(test)]
#[path = "gemma3n_per_layer_inputs_state_tests.rs"]
mod tests;
