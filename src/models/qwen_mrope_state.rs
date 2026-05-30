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

//! Per-sequence MRoPE state for Qwen VL families.
//!
//! mlx-vlm PR #1095 (commit `e94b76`): when a single text-model
//! instance serves several sequences in a server (or a mixed batch with both
//! VL and text-only rows), the cached MRoPE `rope_deltas` and the spatially
//! computed `position_ids` must be tracked **per sequence**. A shared scalar
//! delta lets the most recent VL prefill leak into other sequences' decode
//! steps and produces wrong attention positions for text-only rows.
//!
//! This module provides a small container that the four in-scope Qwen VL
//! text models (`qwen2_vl`, `qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`) share.
//! It tracks state in two slots:
//!
//! - A per-`SequenceId` map keyed by the scheduler's identifier; written
//!   when the server-side `set_for_sequence` API is used (one row at a
//!   time during sequential prefill) and read back on decode steps that
//!   come through `forward_with_sequence_id`.
//! - A "current" fallback slot used by non-server callers (CLI generate,
//!   single-process VLM batches), and as a last-resort fallback when
//!   `forward_with_sequence_id` does not have an entry for the sequence
//!   yet — for example when `forward()` (no seq_id) is the dispatch path.
//!
//! The fallback preserves the original single-row behavior for every test
//! and CLI path. Only the server's per-request flow opts into the
//! per-sequence map by calling `set_for_sequence`.

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

/// MRoPE state for a single sequence.
///
/// `position_ids` is `[3, 1, prefill_len]` and is populated only by the
/// vision-language prefill path (when image/video tokens are present).
/// `rope_deltas` is the scalar that decode steps add to `cache_offset`
/// to recover the absolute MRoPE position; it is non-zero when the row
/// went through a multimodal prefill, and zero (or absent) for text-only
/// rows.
pub(crate) struct MRopeEntry {
    pub position_ids: Option<UniquePtr<MlxArray>>,
    pub rope_deltas: Option<i32>,
}

impl MRopeEntry {
    pub(crate) fn empty() -> Self {
        Self {
            position_ids: None,
            rope_deltas: None,
        }
    }
}

/// Container that resolves MRoPE state by `SequenceId` first and falls
/// back to a "current" slot for legacy/non-server call sites.
pub(crate) struct MRopeState {
    sequences: RefCell<HashMap<SequenceId, MRopeEntry>>,
    fallback: RefCell<MRopeEntry>,
}

impl MRopeState {
    // Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn new() -> Self {
        Self {
            sequences: RefCell::new(HashMap::new()),
            fallback: RefCell::new(MRopeEntry::empty()),
        }
    }

    /// Store MRoPE state under the given sequence identifier.
    ///
    /// This is the per-sequence path used by the server scheduler after
    /// it allocates a sequence id but before the prefill forward runs.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn set_for_sequence(
        &self,
        seq_id: SequenceId,
        position_ids: UniquePtr<MlxArray>,
        rope_deltas: i32,
    ) {
        self.sequences.borrow_mut().insert(
            seq_id,
            MRopeEntry {
                position_ids: Some(position_ids),
                rope_deltas: Some(rope_deltas),
            },
        );
    }

    /// Store MRoPE state in the fallback slot (legacy single-instance
    /// callers, CLI tools, single-row tests).
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn set_fallback(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = Some(position_ids);
        entry.rope_deltas = Some(rope_deltas);
    }

    /// Drop everything in the fallback slot — used when the language
    /// model is reset for a fresh image/video request without a seq id.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn clear_fallback(&self) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = None;
        entry.rope_deltas = None;
    }

    /// Drop a single sequence's per-row MRoPE entry. Called on
    /// sequence release / abort so the map does not grow without bound.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn release_sequence(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }

    /// Move whatever the fallback slot holds into the per-sequence map
    /// under `seq_id`, leaving the fallback empty afterward. This is how
    /// the server scheduler associates the MRoPE state computed by
    /// `set_fallback` (during the vision-encoder pass that did not yet
    /// know the sequence id) with the sequence that will eventually
    /// consume it.
    ///
    /// If the fallback slot is empty (no VL prefill ran for this row),
    /// an empty entry is registered so subsequent decode steps see
    /// `rope_deltas == None` for the sequence — i.e., the text-only
    /// path with no leaked delta.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn bind_fallback_to_sequence(&self, seq_id: SequenceId) {
        let mut fallback = self.fallback.borrow_mut();
        let entry = MRopeEntry {
            position_ids: fallback.position_ids.take(),
            rope_deltas: fallback.rope_deltas.take(),
        };
        self.sequences.borrow_mut().insert(seq_id, entry);
    }

    /// Remove the per-sequence entry under `seq_id` and return it without
    /// dropping the contained `UniquePtr<MlxArray>`. Used by the server
    /// preemption path so the entry can be carried across the eviction
    /// (which releases the old sequence id) and reinstalled under the
    /// freshly allocated id (follow-up).
    ///
    /// Returns `None` when the map has no entry for `seq_id`.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<MRopeEntry> {
        self.sequences.borrow_mut().remove(&seq_id)
    }

    /// Re-install a previously taken `MRopeEntry` under `seq_id`. Used to
    /// rebind state that survived a preemption-and-reallocate round trip.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn bind_for_sequence(&self, seq_id: SequenceId, entry: MRopeEntry) {
        self.sequences.borrow_mut().insert(seq_id, entry);
    }

    /// Run a closure with the resolved MRoPE entry for `seq_id`. If
    /// `seq_id` is `None` *or* the sequence has no entry yet, the closure
    /// runs against the fallback slot. The borrow is held for the
    /// duration of the closure, which is the only safe way to hand a
    /// reference to the contained `UniquePtr<MlxArray>` back to the
    /// caller without another allocation.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5
    pub(crate) fn with_entry<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(&MRopeEntry) -> R,
    ) -> R {
        if let Some(id) = seq_id {
            let map = self.sequences.borrow();
            if let Some(entry) = map.get(&id) {
                return f(entry);
            }
        }
        let fallback = self.fallback.borrow();
        f(&fallback)
    }
}

impl Default for MRopeState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "qwen_mrope_state_tests.rs"]
mod tests;
