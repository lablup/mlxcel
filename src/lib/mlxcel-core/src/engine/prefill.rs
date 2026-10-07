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

//! The prefill half of the engine: one [`crate::prefill_plan::PrefillPlan`]
//! piece per call ([`Engine::prefill`]), the padded batched cohort window
//! ([`Engine::prefill_cohort`]), the pad trim that follows a padded forward,
//! and [`piece_input`], which builds a piece's input and mask.

use super::{Engine, EngineError};
use crate::cache::{SequenceId, SequenceStateBackend};
use crate::generate::LanguageModel;
use crate::prefill_plan::{PrefillPiece, PrefillPlan};
use crate::utils::create_padded_prefill_mask;
use crate::{MlxArray, UniquePtr};

/// One prefill piece for one sequence.
#[derive(Clone, Copy)]
pub struct PrefillStep<'a> {
    pub seq_id: SequenceId,
    /// `[1, padded_len]` token ids (pad positions zero-filled).
    pub input: &'a MlxArray,
    /// Precomputed input embeddings (VLM prefill); the model then runs its
    /// embedding forward instead of embedding `input`.
    pub embeddings: Option<&'a MlxArray>,
    pub mask: Option<&'a MlxArray>,
    /// Position of the last real token; only its logits are returned.
    pub last_pos: usize,
    /// Pad positions to drop from the sequence's state after the forward
    /// (`PrefillPiece::trim_after`). Zero when the piece is unpadded.
    pub trim_excess: i32,
    /// Force-evaluate the logits before returning (a non-terminal piece,
    /// whose transients must be released before the next piece's graph).
    /// A terminal piece leaves the eval to the first sample.
    pub eval: bool,
}

/// The result of [`Engine::prefill`].
pub struct PrefillOutcome {
    /// `[1, 1, vocab]` logits at `PrefillStep::last_pos`.
    pub logits: UniquePtr<MlxArray>,
    /// Outcome of the forced eval when `PrefillStep::eval` was set (`Ok` when
    /// it was not): an MLX throw caught at the FFI boundary (#822).
    pub eval: Result<(), String>,
    /// Outcome of the pad trim: `Err` means the state may disagree with the
    /// token count (#1755). The caller records `eval` first (it feeds the
    /// backend health counter) and then acts on this.
    pub trim: Result<(), String>,
}

/// The `[1, padded_len]` input ids of `piece` (`tokens` are its real
/// tokens, zero-filled to the padded length) and, when the plan needs one
/// for a padded piece, the padded prefill mask anchored at `kv_offset`, the
/// positions already in the sequence's KV state.
///
/// Used by: `BatchScheduler::run_prefill_piece`, `DirectEngine` (the
/// parity probe), so the probe builds each piece exactly as the server does.
pub fn piece_input(
    plan: &PrefillPlan,
    piece: &PrefillPiece,
    tokens: &[i32],
    kv_offset: i32,
) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
    debug_assert_eq!(tokens.len(), piece.len(), "one token per real position");
    let mask = (piece.is_padded() && plan.pad_mask_required()).then(|| {
        create_padded_prefill_mask(piece.len() as i32, piece.padded_len as i32, kv_offset)
    });
    let input = if piece.is_padded() {
        let mut padded = tokens.to_vec();
        padded.resize(piece.padded_len, 0);
        crate::from_slice_i32(&padded, &[1, piece.padded_len as i32])
    } else {
        crate::from_slice_i32(tokens, &[1, piece.padded_len as i32])
    };
    (input, mask)
}

impl<M: LanguageModel> Engine<M> {
    /// Run one prefill piece for one sequence and return the logits at its
    /// last real position. The forward, the forced eval and the pad trim all
    /// run while the sequence's caches are borrowed, in that order; the
    /// outcomes are reported separately so the caller can record the eval
    /// against the backend health counter before acting on the trim.
    pub fn prefill(&mut self, step: &PrefillStep<'_>) -> Result<PrefillOutcome, EngineError> {
        let caches = self
            .pool
            .get_caches_mut(step.seq_id)
            .ok_or(EngineError::MissingSequence(step.seq_id))?;
        let logits = match step.embeddings {
            Some(embeddings) => self
                .model
                .forward_last_logits_with_embeddings_and_sequence_id(
                    step.input,
                    Some(embeddings),
                    Some(step.seq_id),
                    caches,
                    step.mask,
                    step.last_pos,
                ),
            None => self.model.forward_last_logits_with_sequence_id(
                step.input,
                Some(step.seq_id),
                caches,
                step.mask,
                step.last_pos,
            ),
        };
        let eval = if step.eval {
            crate::try_eval(&logits).map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        let trim = trim_padded_prefill(&self.model, step.seq_id, caches, step.trim_excess);
        if step.embeddings.is_some() {
            self.model.after_prefill();
        }
        Ok(PrefillOutcome { logits, eval, trim })
    }

    /// A prefill-only scoring pass: the forward over every position of
    /// `step.input` for `step.seq_id`, returning the `[1, T, vocab]` logits of
    /// the whole window rather than the last position's (`step.last_pos` is
    /// not read). The per-token log-likelihoods a perplexity gate needs are
    /// computed from these by the caller; the sequence is closed afterwards,
    /// so the state the pass wrote is never decoded from.
    ///
    /// Used by: `DirectEngine::loglikelihoods`.
    pub fn score(&mut self, step: &PrefillStep<'_>) -> Result<PrefillOutcome, EngineError> {
        let caches = self
            .pool
            .get_caches_mut(step.seq_id)
            .ok_or(EngineError::MissingSequence(step.seq_id))?;
        let logits = match step.embeddings {
            Some(embeddings) => self.model.forward_with_embeddings_and_sequence_id(
                step.input,
                Some(embeddings),
                Some(step.seq_id),
                caches,
                step.mask,
            ),
            None => self.model.forward_with_sequence_id(
                step.input,
                Some(step.seq_id),
                caches,
                step.mask,
            ),
        };
        let eval = if step.eval {
            crate::try_eval(&logits).map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        let trim = trim_padded_prefill(&self.model, step.seq_id, caches, step.trim_excess);
        Ok(PrefillOutcome { logits, eval, trim })
    }

    /// One padded batched prefill pass over a cohort: `input` is
    /// `[B, padded_len]`, `mask` the stacked per-row padded masks (or `None`
    /// for a maskless family). Returns `[B, padded_len, vocab]` logits; the
    /// caller slices each row's last real position and trims its padding
    /// with [`Engine::trim_padding`].
    pub fn prefill_cohort(
        &mut self,
        seq_ids: &[SequenceId],
        input: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, EngineError> {
        let mut batch_caches = self
            .pool
            .get_batch_caches_mut(seq_ids)
            .map_err(EngineError::Batch)?;
        if batch_caches.len() != seq_ids.len() {
            return Err(EngineError::Batch(format!(
                "cohort of {} rows resolved {} cache sets",
                seq_ids.len(),
                batch_caches.len()
            )));
        }
        Ok(self
            .model
            .forward_batched_with_ids(input, Some(seq_ids), &mut batch_caches, mask))
    }

    /// Drop `excess` pad positions a padded prefill wrote for `id`, from the
    /// pool caches and from a model-owned family's own state (#1755).
    pub fn trim_padding(&mut self, id: SequenceId, excess: i32) -> Result<(), EngineError> {
        let caches = self
            .pool
            .get_caches_mut(id)
            .ok_or(EngineError::MissingSequence(id))?;
        trim_padded_prefill(&self.model, id, caches, excess).map_err(EngineError::Trim)
    }
}

/// Drop the `excess` pad positions a padded prefill pass wrote for `seq_id`.
///
/// `caches` is the sequence's pool entry. It is trimmed for every layout, and
/// is empty (or shadow paged accounting) for a model-owned family, whose own
/// state is rewound through the model hook. The model's natural layout
/// decides, not the allocated backend: a model-owned family allocated on the
/// paged backend for block accounting still keeps its K/V to itself (#1346).
pub fn trim_padded_prefill<M: LanguageModel + ?Sized>(
    model: &M,
    seq_id: SequenceId,
    caches: &mut [crate::layers::KVCache],
    excess: i32,
) -> Result<(), String> {
    if excess <= 0 {
        return Ok(());
    }
    for cache in caches.iter_mut() {
        cache.trim(excess);
    }
    if model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
        model.trim_state(Some(seq_id), excess)?;
    }
    Ok(())
}
