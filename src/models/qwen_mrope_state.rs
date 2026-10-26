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
use mlxcel_core::layers::KVCache;
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
    // Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    // GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
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
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
    pub(crate) fn set_fallback(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = Some(position_ids);
        entry.rope_deltas = Some(rope_deltas);
    }

    /// Drop everything in the fallback slot. Used when the language model is
    /// reset for a fresh image/video request without a seq id, and by the
    /// server scheduler when a request's admission aborts after its media
    /// preparation wrote the slot but before it was bound (issue #2625).
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
    pub(crate) fn clear_fallback(&self) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = None;
        entry.rope_deltas = None;
    }

    /// Drop a single sequence's per-row MRoPE entry. Called on
    /// sequence release / abort so the map does not grow without bound.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
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
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
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
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
    pub(crate) fn take_for_sequence(&self, seq_id: SequenceId) -> Option<MRopeEntry> {
        self.sequences.borrow_mut().remove(&seq_id)
    }

    /// Re-install a previously taken `MRopeEntry` under `seq_id`. Used to
    /// rebind state that survived a preemption-and-reallocate round trip.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
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
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp, GLM-4V,
    /// GLM-4V-MoE, Cohere Compass, ERNIE-4.5-MoE-VL, Hunyuan-VL, PaddleOCR-VL
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

    /// Build the per-row MRoPE positions for a single-token decode batch.
    ///
    /// Each row advances from its own monotonic cache `offset` and its own
    /// stored `rope_delta`, yielding `[3, B, 1]`. Missing sequence ids resolve
    /// through the legacy fallback entry, exactly like [`Self::with_entry`].
    ///
    /// A stored `position_ids` tensor that still covers the requested decode
    /// position cannot be represented by the scalar-delta fast path. In that
    /// case this returns `None` so the caller preserves the exact per-row
    /// forward path and slices the stored tensor there.
    ///
    /// Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, GLM-4V, GLM-4V-MoE,
    /// Hunyuan-VL, ERNIE-4.5-MoE-VL, PaddleOCR-VL and Cohere Compass native
    /// decode.
    pub(crate) fn batched_decode_positions(
        &self,
        seq_ids: &[SequenceId],
        offsets: &[i32],
    ) -> Option<UniquePtr<MlxArray>> {
        self.batched_decode_positions_with_axes(seq_ids, offsets, 3)
    }

    /// Generalized decode positions for families such as Hunyuan-VL whose
    /// multimodal rotary layout has four axes instead of three.
    pub(crate) fn batched_decode_positions_with_axes(
        &self,
        seq_ids: &[SequenceId],
        offsets: &[i32],
        axes: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        assert_eq!(
            seq_ids.len(),
            offsets.len(),
            "MRoPE batched decode sequence/offset length mismatch"
        );

        let mut positions = Vec::with_capacity(seq_ids.len());
        for (&seq_id, &offset) in seq_ids.iter().zip(offsets) {
            let position = self.with_entry(Some(seq_id), |entry| {
                if let Some(stored) = entry.position_ids.as_ref() {
                    let shape = mlxcel_core::array_shape(stored);
                    if shape.len() == 3 && shape[1] == 1 && shape[2] > offset {
                        return None;
                    }
                }
                Some(offset + entry.rope_deltas.unwrap_or(0))
            })?;
            positions.push(position);
        }

        Some(decode_positions_from_offsets(&positions, axes))
    }
}

/// Build `[axes, B, 1]` decode positions from one scalar position per row.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths.
pub(crate) fn decode_positions_from_offsets(offsets: &[i32], axes: i32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity(offsets.len() * axes as usize);
    for _ in 0..axes {
        data.extend_from_slice(offsets);
    }
    mlxcel_core::from_slice_i32(&data, &[axes, offsets.len() as i32, 1])
}

/// Read each row's monotonic layer-zero cache offset with family/row/layer
/// diagnostics.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths.
pub(crate) fn decode_offsets(batch_caches: &[&mut [KVCache]], family: &str) -> Vec<i32> {
    batch_caches
        .iter()
        .enumerate()
        .map(|(row, caches)| {
            caches
                .first()
                .unwrap_or_else(|| panic!("{family} batched decode cache mismatch at row {row}, layer 0: row has 0 cache layers"))
                .offset
        })
        .collect()
}

/// Select rows from an array while preserving their ascending request order.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths, and the Mllama image/text row
/// split (#2567).
pub(crate) fn take_rows(array: &MlxArray, rows: &[usize], axis: i32) -> UniquePtr<MlxArray> {
    let indices: Vec<i32> = rows.iter().map(|&row| row as i32).collect();
    let indices = mlxcel_core::from_slice_i32(&indices, &[indices.len() as i32]);
    mlxcel_core::take(array, &indices, axis)
}

/// Borrow the cache rows for one ascending decode group.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths, and the Mllama image/text row
/// split (#2567).
pub(crate) fn select_cache_rows<'a>(
    batch_caches: &'a mut [&mut [KVCache]],
    rows: &[usize],
) -> Vec<&'a mut [KVCache]> {
    let mut selected = Vec::with_capacity(rows.len());
    let mut next = 0;
    for (row, caches) in batch_caches.iter_mut().enumerate() {
        if next < rows.len() && rows[next] == row {
            selected.push(&mut **caches);
            next += 1;
        }
    }
    assert_eq!(
        selected.len(),
        rows.len(),
        "MRoPE decode group rows must be sorted and in batch bounds"
    );
    selected
}

/// Scatter a group's logits back into their original request rows.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths, and the Mllama image/text row
/// split (#2567).
pub(crate) fn scatter_group_logits(
    outputs: &mut [Option<UniquePtr<MlxArray>>],
    rows: &[usize],
    logits: &MlxArray,
) {
    let shape = mlxcel_core::array_shape(logits);
    for (local_row, &original_row) in rows.iter().enumerate() {
        outputs[original_row] = Some(mlxcel_core::slice(
            logits,
            &[local_row as i32, 0, 0],
            &[local_row as i32 + 1, shape[1], shape[2]],
        ));
    }
}

/// Concatenate fully scattered row logits with a family-specific diagnostic.
///
/// Used by: the Qwen VL, GLM-4V, Hunyuan-VL, ERNIE-VL, PaddleOCR-VL and
/// Cohere Compass native grouped-decode paths, and the Mllama image/text row
/// split (#2567).
pub(crate) fn concatenate_scattered(
    outputs: Vec<Option<UniquePtr<MlxArray>>>,
    family: &str,
) -> UniquePtr<MlxArray> {
    let outputs: Vec<UniquePtr<MlxArray>> = outputs
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            value.unwrap_or_else(|| panic!("{family} batched decode row {row} was not produced"))
        })
        .collect();
    let refs: Vec<&MlxArray> = outputs.iter().map(|row| &**row).collect();
    mlxcel_core::concatenate_many(&refs, 0)
}

impl Default for MRopeState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "qwen_mrope_state_tests.rs"]
mod tests;
