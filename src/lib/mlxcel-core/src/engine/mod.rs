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

//! The batch-native decode engine (ADR 0007, epic #2166 Phase 4b, #2172).
//!
//! [`Engine`] owns a model and the [`CachePool`] that holds every open
//! sequence's KV state, and it is the only path through which a model forward
//! runs: the server's `BatchScheduler` keeps admission, tick policy,
//! preemption, prompt-cache lookup and the speculative burst generators, and
//! drives the model through [`Engine::open`], [`Engine::prefill`],
//! [`Engine::step`] and [`Engine::close`]. A single sequence is a
//! [`StepBatch`] of one; there is no separate single-sequence decode loop.
//!
//! The engine composes the contracts the earlier phases of the epic fixed:
//! attention dispatch is a property of the cache ([`crate::cache::KVCache::attend`],
//! ADR 0008), prefill geometry comes from a [`crate::prefill_plan::PrefillPlan`]
//! piece, the per-row sampling step is [`crate::sampling_row_step::RowSampler`]
//! and the post-sample finish step is [`crate::decode_finish::finish_step`].
//!
//! Errors are per sequence: an [`EngineError`] names the row it concerns so a
//! caller can finish that row with `FinishReason::Error` and keep serving the
//! rest of the batch (#822). The engine never aborts a batch for one row.

use std::fmt;

use crate::cache::{
    CachePool, DecodeLookaheadAppendScope, DetachedCacheSet, DetachedPagedCacheSet, SequenceId,
    SequenceStateBackend, SequenceStateLayout,
};
use crate::decode_finish::FinishCause;
use crate::generate::LanguageModel;
use crate::sampling::{FusedSampleParams, TokenBiasMap, batched_fused_sample_tokens};
use crate::{MlxArray, UniquePtr};

mod prefill;
pub mod rows;

pub use prefill::{PrefillOutcome, PrefillStep, piece_input, trim_padded_prefill};
pub use rows::{
    FusedRowGate, RowError, RowOutcome, StepRow, StepRowHooks, finish_row, fused_params,
    row_biases, row_logits, sample_and_finish, sample_and_finish_row, shared_fused_params,
    tokens_to_host,
};

/// A per-sequence engine failure.
///
/// Every variant names the sequence it concerns (or the whole batch for
/// [`EngineError::Batch`]); callers map it to that row's finish reason.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The sequence has no cache entry in the pool (it was closed, evicted or
    /// never opened).
    #[error("sequence {0} is not open")]
    MissingSequence(SequenceId),
    /// The pool could not assemble the batch view for these rows.
    #[error("batch caches: {0}")]
    Batch(String),
    /// Allocating a sequence's state failed (pool capacity, paged budget).
    #[error("open: {0}")]
    Open(String),
    /// Padding trim after a prefill piece failed; the sequence's state may
    /// disagree with its token count and must be finished with an error.
    #[error("trim: {0}")]
    Trim(String),
    /// A speculative-append rewind failed; the row is desynchronized (#2182).
    #[error("rewind: {0}")]
    Rewind(String),
    /// Scheduling a pipelined step threw at the MLX boundary (#822). The
    /// speculative appends of that forward are still in place; the caller
    /// unwinds them with [`Engine::unwind_appends`].
    #[error("eval: {0}")]
    Eval(String),
}

/// What [`Engine::open`] needs to allocate a sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SequenceSpec {
    /// Override the model's natural sequence-state layout (the server's
    /// `--decode-storage-backend` resolution, ADR 0008). `None` lets the model
    /// decide.
    pub layout_override: Option<SequenceStateLayout>,
}

/// The result of [`Engine::close`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosedSequence {
    pub id: SequenceId,
    /// Whether the pool held an entry for the sequence when it was closed.
    pub was_open: bool,
}

/// One decode step over `B` rows: `input` is `[B, 1]`, one new token per row
/// of `seq_ids`, in that order.
#[derive(Clone, Copy)]
pub struct StepBatch<'a> {
    pub seq_ids: &'a [SequenceId],
    pub input: &'a MlxArray,
}

/// The result of [`Engine::step`]: one [`RowOutcome`] per row of the
/// [`StepBatch`], in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutput {
    pub rows: Vec<RowOutcome>,
}

/// The batch-native decode engine. See the module docs.
pub struct Engine<M: LanguageModel> {
    model: M,
    pool: CachePool,
}

impl<M: LanguageModel> fmt::Debug for Engine<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("open_sequences", &self.pool.active_count())
            .finish_non_exhaustive()
    }
}

impl<M: LanguageModel> Engine<M> {
    /// Build an engine over `model` and the pool that will hold its
    /// sequences.
    pub fn new(model: M, pool: CachePool) -> Self {
        Self { model, pool }
    }

    /// Build an engine with a fresh pool of `max_sequences` dense entries.
    pub fn with_capacity(model: M, max_sequences: usize) -> Self {
        Self::new(model, CachePool::new(max_sequences))
    }

    /// The model, for capability queries, snapshot and restore hooks, and the
    /// family-specific bindings a front end needs. Forwards go through
    /// [`Engine::step`] and [`Engine::prefill`], not through this.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The pool, for admission accounting, detach and adopt, and handoff.
    pub fn pool(&self) -> &CachePool {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut CachePool {
        &mut self.pool
    }

    /// Model and pool together, for callers that hand a sequence's caches to
    /// a model hook (snapshot, restore, release) or restore a disaggregated
    /// handoff into the pool. The model is shared, as from [`Engine::model`]:
    /// forwards still belong to the engine entries, which the scheduler's
    /// source guard (`prefill_span_coverage_tests`) enforces.
    pub fn parts_mut(&mut self) -> (&M, &mut CachePool) {
        (&self.model, &mut self.pool)
    }

    pub fn into_parts(self) -> (M, CachePool) {
        (self.model, self.pool)
    }

    /// Allocate a sequence: its pool entry (dense caches from
    /// `make_caches`, a paged state, or a model-owned placeholder, per the
    /// layout) and the model's own per-sequence state.
    pub fn open(&mut self, spec: SequenceSpec) -> Result<SequenceId, EngineError> {
        let id = self
            .pool
            .allocate_with_layout(&self.model, spec.layout_override)
            .map_err(EngineError::Open)?;
        self.model.prepare_sequence_state(id);
        Ok(id)
    }

    /// Release a sequence's model state and pool entry. Closing a sequence
    /// that is not open is a no-op reported through
    /// [`ClosedSequence::was_open`].
    pub fn close(&mut self, id: SequenceId) -> ClosedSequence {
        self.model.release_sequence_state_by_id(id);
        let was_open = match self.pool.get_caches_mut(id) {
            Some(caches) => {
                self.model.release_sequence_state(caches);
                true
            }
            None => false,
        };
        self.pool.release(id);
        ClosedSequence { id, was_open }
    }

    /// Whether `id` has a pool entry.
    pub fn is_open(&self, id: SequenceId) -> bool {
        self.pool.get(id).is_some()
    }

    /// Open a sequence over a detached dense cache set (prompt-cache adopt):
    /// the pool re-attaches the set under a fresh id and the model is told
    /// about the adopted prefix.
    pub fn adopt(&mut self, detached: DetachedCacheSet) -> Result<SequenceId, EngineError> {
        self.pool
            .adopt(&self.model as &dyn LanguageModel, detached)
            .map_err(EngineError::Open)
    }

    /// [`Engine::adopt`] for a detached paged cache set.
    pub fn adopt_paged(
        &mut self,
        detached: DetachedPagedCacheSet,
    ) -> Result<SequenceId, EngineError> {
        self.pool
            .adopt_paged(&self.model as &dyn LanguageModel, detached)
            .map_err(EngineError::Open)
    }

    /// One decode step for every row of `batch`: the forward, then each
    /// row's sampling and finish step ([`sample_and_finish`]).
    ///
    /// A batch of one runs the model's single-row forward, which is what the
    /// provided batched entry does at `b == 1` and what the measured B=1
    /// throughput rests on, and samples through the per-row chain; a larger
    /// batch runs the batched entry and may take the fused draw. After the
    /// forward each row's model-owned storage is mirrored into the pool
    /// (`sync_sequence_storage`), and every row that continues advances its
    /// pool offset by one. `rows` must be the batch's rows in
    /// `batch.seq_ids` order.
    pub fn step<H: StepRowHooks>(
        &mut self,
        batch: &StepBatch<'_>,
        rows: &mut [StepRow<'_, H>],
    ) -> Result<StepOutput, EngineError> {
        debug_assert_eq!(rows.len(), batch.seq_ids.len(), "one StepRow per batch row");
        let logits = self.forward(batch)?;
        let outcomes = sample_and_finish(&logits, rows);
        self.advance_continuing(&outcomes);
        Ok(StepOutput { rows: outcomes })
    }

    /// The submit half of a pipelined step: a speculative forward for
    /// `batch`, the rows' token biases and the shared fused row filters
    /// folded in, one fused `[B, vocab] -> [B]` draw, scheduled with
    /// `async_eval` and returned as the lazy `[B]` device tokens so the GPU
    /// runs ahead while the caller finishes the previous step on the host.
    ///
    /// The forward runs inside a [`DecodeLookaheadAppendScope`], so a
    /// model-owned family that rewinds its own state (#2182) logs the rows
    /// these writes overwrite. The caller collects with
    /// [`Engine::finish_rows`] once the tokens are read back, or unwinds the
    /// appends with [`Engine::unwind_appends`] when it tears the step down,
    /// including after an `Err`: an [`EngineError::Eval`] leaves the
    /// speculative appends in place.
    pub fn submit(
        &mut self,
        batch: &StepBatch<'_>,
        params: &FusedSampleParams,
        biases: &[&TokenBiasMap],
    ) -> Result<UniquePtr<MlxArray>, EngineError> {
        let logits = {
            let _speculative = DecodeLookaheadAppendScope::enter();
            self.forward(batch)?
        };
        // The synchronous fused path's draw (token bias in the per-row
        // sampler's chain position, then the pre-fused row filters), so the
        // pipelined draw samples from the identical distribution.
        let tokens = batched_fused_sample_tokens(&logits, params, biases);
        crate::try_async_eval(&tokens).map_err(|e| EngineError::Eval(e.to_string()))?;
        Ok(tokens)
    }

    /// The collect half of a pipelined step: `tokens[i]` is the host token
    /// drawn for `rows[i]`; each row runs the finish step and every row that
    /// continues advances its pool offset.
    pub fn finish_rows<H: StepRowHooks>(
        &mut self,
        tokens: &[i32],
        rows: &mut [StepRow<'_, H>],
    ) -> Vec<RowOutcome> {
        debug_assert_eq!(rows.len(), tokens.len(), "one token per row");
        let outcomes: Vec<RowOutcome> = rows
            .iter_mut()
            .zip(tokens)
            .map(|(row, &token)| finish_row(row, token, token, None, false))
            .collect();
        self.advance_continuing(&outcomes);
        outcomes
    }

    /// Sample and finish a prefill's first token from its last-position
    /// `logits` (`[1, 1, vocab]`) through the per-row chain. The pool offset
    /// is the caller's: a prefill sets it from the prompt length.
    pub fn complete_prefill<H: StepRowHooks>(
        &mut self,
        logits: &MlxArray,
        row: &mut StepRow<'_, H>,
    ) -> RowOutcome {
        sample_and_finish_row(logits, row)
    }

    fn advance_continuing(&mut self, outcomes: &[RowOutcome]) {
        for outcome in outcomes {
            if outcome.error.is_some() || outcome.finish == Some(FinishCause::Eos) {
                continue;
            }
            if let Some(set) = self.pool.get_mut(outcome.seq_id) {
                set.current_offset += 1;
            }
        }
    }

    /// The forward for `batch`: `[B, 1, vocab]` lazy logits, with each row's
    /// model-owned storage mirrored into the pool afterwards.
    fn forward(&mut self, batch: &StepBatch<'_>) -> Result<UniquePtr<MlxArray>, EngineError> {
        let logits = self.forward_rows(batch)?;
        for &id in batch.seq_ids {
            self.sync_sequence_storage(id);
        }
        Ok(logits)
    }

    fn forward_rows(&mut self, batch: &StepBatch<'_>) -> Result<UniquePtr<MlxArray>, EngineError> {
        match batch.seq_ids {
            [] => Err(EngineError::Batch("empty step batch".to_string())),
            [id] => {
                let caches = self
                    .pool
                    .get_caches_mut(*id)
                    .ok_or(EngineError::MissingSequence(*id))?;
                Ok(self
                    .model
                    .forward_with_sequence_id(batch.input, Some(*id), caches, None))
            }
            ids => {
                let mut batch_caches = self
                    .pool
                    .get_batch_caches_mut(ids)
                    .map_err(EngineError::Batch)?;
                Ok(self.model.forward_batched_with_ids(
                    batch.input,
                    Some(ids),
                    &mut batch_caches,
                    None,
                ))
            }
        }
    }

    /// Undo `n` speculative KV appends on `id`: trim the pool caches and, for
    /// a model-owned family, rewind the model's own state. A trim that
    /// removes fewer positions than asked is logged (the KV may be out of
    /// sync); a failed model rewind is [`EngineError::Rewind`], and the caller
    /// must finish that row with an error rather than donate its state.
    pub fn unwind_appends(&mut self, id: SequenceId, n: i32) -> Result<(), EngineError> {
        if n <= 0 {
            return Ok(());
        }
        let Some(caches) = self.pool.get_caches_mut(id) else {
            return Ok(());
        };
        for (layer, cache) in caches.iter_mut().enumerate() {
            let trimmed = cache.trim(n);
            if trimmed != n {
                tracing::warn!(
                    seq_id = %id,
                    layer,
                    requested = n,
                    trimmed,
                    "lookahead teardown: trim removed fewer positions than requested, KV may be out of sync"
                );
            }
        }
        if self.model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
            self.model
                .rewind_decode_appends(id, n)
                .map_err(EngineError::Rewind)?;
        }
        self.sync_sequence_storage(id);
        Ok(())
    }

    /// Mirror a model-owned family's sequence storage into the pool's paged
    /// bookkeeping (a no-op for a pure dense pool and for pool-backed rows).
    pub fn sync_sequence_storage(&mut self, id: SequenceId) {
        if let Err(err) = self.model.sync_sequence_storage(id, &mut self.pool) {
            tracing::warn!("Failed to sync paged state for {id}: {err}");
        }
    }
}

#[cfg(test)]
mod tests;
