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

//! Per-sequence `per_layer_inputs` state for Gemma 4 E2B/E4B VLM.
//!
//! `Gemma4VLModel::get_input_embeddings_with_audio_and_cache` projects
//! the Gemma 4 E2B/E4B `per_layer_inputs` tensor (shape `[1, T, L, h]`)
//! while preparing a request's VLM embeddings. That tensor is consumed
//! later, at prefill time, by
//! `Gemma4VLModel::forward_with_embeddings_and_sequence_id` when it calls
//! the text wrapper's
//! `forward_with_inputs_and_sequence_id(per_layer_inputs)`.
//!
//! Before this module landed, the projected tensor lived on a single
//! `RefCell<Option<UniquePtr<MlxArray>>>` field on `Gemma4VLModel`. The
//! scheduler's `drain_incoming_requests` runs the embedding-prep step
//! once per request *at enqueue time* and the prefill step asynchronously
//! at scheduling time. When 2+ Gemma 4 VLM requests are enqueued before
//! the first one prefills (a typical burst on an idle server), each
//! `prepare_request_vlm_embeddings` overwrites the same cell. The first
//! prefill then `take()`s **the latest writer's** `per_layer_inputs`
//! instead of its own — every subsequent request runs against the wrong
//! per-layer projection. That is the production root cause behind issue
//! the upstream mlx-vlm PR #1123 fix is for a related
//! batched-concat shape mismatch that does not apply to mlxcel's
//! sequential VLM prefill path.
//!
//! The container below mirrors [`crate::models::qwen_mrope_state::MRopeState`]
//! so the scheduler wires both per-sequence
//! states identically:
//!
//! - **Per-`SequenceId` map** is the primary slot; `bind_fallback_to_sequence`
//!   transfers a freshly written fallback into this map under the
//!   scheduler-allocated id, draining the fallback in the same step so
//!   the next request cannot inherit the previous row's tensor.
//! - **Fallback slot** preserves legacy single-instance callers (CLI
//!   `mlxcel generate`, single-row tests). It also acts as a last-resort
//!   when `with_entry(Some(id), ..)` is called with an id that was never
//!   bound (e.g., a request that started before per-sequence wiring
//!   landed).
//!
//! Lifecycle ownership matches Qwen MRoPE:
//! - `set_fallback` is called by `get_input_embeddings_with_audio_and_cache`
//!   right before the result is returned to the caller.
//! - `bind_fallback_to_sequence(seq_id)` is called by the scheduler
//!   immediately after `prepare_request_vlm_embeddings`, alongside
//!   `bind_qwen_vl_mrope_state_to_sequence`.
//! - `take_for_sequence` / `bind_for_sequence` carry the entry across a
//!   preemption-and-reallocate round trip so the re-prefill sees the
//!   same projection (per_layer_inputs is **not** recomputed during
//!   re-prefill — the request's vision encoder and per-layer projection
//!   ran exactly once at enqueue time).
//! - `release_sequence` is called from `Gemma4VLModel::release_sequence_state_by_id`
//!   so the map drains in lock step with the scheduler's per-sequence
//!   cache cleanup.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-sequence container for Gemma 4 VLM `per_layer_inputs` tensors.
///
/// Used by: [`crate::vision::Gemma4VLModel`] for E2B/E4B variants where
/// `hidden_size_per_layer_input > 0`. The value stored is the projected
/// `[1, T, num_layers, h_per_layer]` tensor that
/// `Gemma4TextModel::project_per_layer_inputs` produces; the prefill
/// path then adds it to its own freshly-projected output (see
/// `Gemma4TextModel::forward` line `let sum = mlxcel_core::add(...)`).
#[derive(Default)]
pub(crate) struct Gemma4PerLayerInputsState {
    sequences: RefCell<HashMap<SequenceId, UniquePtr<MlxArray>>>,
    fallback: RefCell<Option<UniquePtr<MlxArray>>>,
}

impl Gemma4PerLayerInputsState {
    /// Construct an empty state container.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Store `per_layer_inputs` in the fallback slot. Used by the legacy
    /// single-instance call path (CLI `mlxcel generate`, single-row
    /// tests) before any sequence id has been allocated.
    ///
    /// `per_layer_inputs == None` clears the fallback (the E1B variant
    /// has `hidden_size_per_layer_input == 0` and produces no per-layer
    /// projection — clearing keeps the slot consistent).
    pub(crate) fn set_fallback(&self, per_layer_inputs: Option<UniquePtr<MlxArray>>) {
        *self.fallback.borrow_mut() = per_layer_inputs;
    }

    /// Move the fallback slot's contents into the per-sequence map under
    /// `seq_id`. Drains the fallback so the next request cannot inherit
    /// the previous row's tensor — this is the burst-enqueue race fix
    ///
    /// If the fallback slot is empty (E1B variant or text-only request
    /// after a Gemma 4 VLM model load — `prepare_request_vlm_embeddings`
    /// returns `None` in that case and never writes the slot), `seq_id`
    /// is left absent from the map. A subsequent
    /// `with_entry(Some(seq_id), ..)` will fall back to the (still empty)
    /// fallback slot and the prefill consumer sees `per_layer_inputs ==
    /// None`, which matches the no-VLM-prefill semantics.
    pub(crate) fn bind_fallback_to_sequence(&self, seq_id: SequenceId) {
        let mut fallback = self.fallback.borrow_mut();
        if let Some(value) = fallback.take() {
            self.sequences.borrow_mut().insert(seq_id, value);
        }
    }

    /// Take a row's per-layer-inputs out of the map for prefill
    /// consumption. Returns `None` when no entry exists — the prefill
    /// path then falls back to the legacy single-row slot via
    /// [`Self::take_fallback`].
    ///
    /// The borrow is released before the value is returned so callers
    /// can hold the resulting `UniquePtr<MlxArray>` across other state
    /// accesses without re-entering `RefCell::borrow_mut`.
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<UniquePtr<MlxArray>> {
        self.sequences.borrow_mut().remove(&seq_id)
    }

    /// Drain the fallback slot. Used by the legacy CLI prefill path
    /// (`Gemma4VLModel::forward_with_embeddings_and_sequence_id` with
    /// `seq_id == None`) to consume the tensor exactly once, mirroring
    /// the original `RefCell::take()` semantics.
    pub(crate) fn take_fallback(&self) -> Option<UniquePtr<MlxArray>> {
        self.fallback.borrow_mut().take()
    }

    /// Drop a single sequence's stored tensor. Called from
    /// `Gemma4VLModel::release_sequence_state_by_id` so the map drains
    /// in step with the scheduler's per-sequence cache release. No-op
    /// when the map has no entry for the id (text-only request, E1B
    /// variant, request that finished before the binding step ran).
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
#[path = "gemma4_per_layer_inputs_state_tests.rs"]
mod tests;
