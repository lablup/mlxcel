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

//! The decode loop of the raw-completion client (#2176), after its first
//! token: the fused lookahead pipeline when the sequence is eligible, the
//! per-row lookahead pipeline (`direct_decode_rows`, #2229) for the samplers
//! the fused draw cannot run, the synchronous per-row chain otherwise.
//!
//! The pipeline overlaps step n+1's forward with step n's host read, as the
//! retired CLI generator did and the server scheduler does (#632): every
//! iteration first submits the next forward fed by the still-unread tokens
//! on the device ([`super::Engine::submit`]), then reads step n's token back and
//! runs the finish step on it ([`super::Engine::finish_rows`]). A token that
//! finishes the sequence (EOS or a stop id, the budget, a repetition loop)
//! or a callback that stops it leaves exactly one speculative append, the
//! forward fed by that token, and the teardown unwinds it
//! ([`super::Engine::unwind_appends`]), so the stream and the sequence's state are
//! what the synchronous loop leaves. The token that spends the budget never
//! has a next forward submitted after it.
//!
//! Eligibility mirrors the scheduler's gate (`BatchScheduler::lookahead_params`)
//! through the same two rules: [`shared_fused_params`] (the draw is one fused
//! sample, so no history penalty, mask, override, per-token logprobs or
//! sampler feedback state, and the host token is never needed before the
//! next forward) and [`super::Engine::can_unwind_lookahead`] (a trimmable state, or
//! a model-owned family that rewinds its own). [`super::lookahead::FORCE_SYNC_ENV`]
//! turns it off. A sequence the second rule admits but whose sampler fails
//! the first (history penalties, DRY, the feedback samplers) decodes on the
//! per-row pipeline, which keeps the forward ahead and builds each draw on
//! the host after the previous token is committed (mirostat and adaptive-p read
//! the draw on the host, so for them only the finish step overlaps the next
//! forward). Unlike the scheduler, the client keeps loop detection on the
//! pipeline: the finish step runs on every committed token here, so the
//! post-commit scan the scheduler's steady path skips is not skipped.
//!
//! The client also pipelines a model-owned family that cannot rewind its own
//! state ([`Teardown::Discard`]), as the retired CLI loop pipelined every
//! family: every caller of [`DirectEngine::decode`] closes the sequence right
//! after it, so the one append past the finishing token is released with the
//! sequence instead of unwound, and the stream is the synchronous loop's.
//! What such a family cannot do is fall back to the synchronous loop after a
//! failed submit that left an append in place, so that run ends with the
//! submit's error. The scheduler keeps these families synchronous: its
//! sequences outlive a tick (prompt-cache donation, preemption).

use super::direct::{BareHooks, DirectEngine, DirectEngineError, delivers_to_callback};
use super::lookahead::lookahead_feedback_input;
use super::rows::{FusedRowGate, shared_fused_params, tokens_to_host};
use super::{EngineError, RowOutcome, StepBatch, StepRow};
use crate::cache::{SequenceId, SequenceStateBackend};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::generation_policy::initial_token_history;
use crate::sampling::{FusedSampleParams, LogprobsConfig};
use crate::sampling_row_step::RowSampler;
use crate::{MlxArray, UniquePtr};

/// One raw completion's per-sequence decode state, which every step row of
/// that sequence borrows.
pub(super) struct DecodeState<'s> {
    pub(super) id: SequenceId,
    pub(super) sampling: &'s SamplingConfig,
    pub(super) sampler: RowSampler,
    pub(super) history: Vec<i32>,
    pub(super) generated: Vec<i32>,
    /// Merged EOS ids (model EOS plus the request's stop ids).
    pub(super) eos: Vec<i32>,
    pub(super) max_tokens: usize,
    pub(super) logprobs: LogprobsConfig,
}

impl<'s> DecodeState<'s> {
    pub(super) fn new(
        id: SequenceId,
        prompt_tokens: &[i32],
        sampling: &'s SamplingConfig,
        eos: Vec<i32>,
        max_tokens: usize,
    ) -> Self {
        Self {
            id,
            sampling,
            sampler: RowSampler::new(sampling),
            history: initial_token_history(prompt_tokens, sampling.needs_token_history()),
            generated: Vec::new(),
            eos,
            max_tokens,
            logprobs: LogprobsConfig::default(),
        }
    }

    /// The sequence as one step row.
    pub(super) fn row(&mut self) -> StepRow<'_, BareHooks> {
        StepRow {
            seq_id: self.id,
            sampler: &mut self.sampler,
            sampling: self.sampling,
            token_history: &mut self.history,
            generated: &mut self.generated,
            eos: &self.eos,
            max_tokens: self.max_tokens,
            logprobs: &self.logprobs,
            needs_mask: false,
            needs_override: false,
            hooks: BareHooks,
        }
    }

    pub(super) fn last_token(&self) -> Result<i32, DirectEngineError> {
        self.generated
            .last()
            .copied()
            .ok_or_else(|| DirectEngineError::Row("decode without a token".into()))
    }
}

/// The client's input to the fused-eligibility rule: it streams no
/// per-token logprobs and carries no mask or override.
fn client_fused_params(sampling: &SamplingConfig) -> Option<FusedSampleParams> {
    shared_fused_params([Some(FusedRowGate {
        sampling,
        needs_mask: false,
        needs_override: false,
        logprobs_enabled: false,
    })])
}

pub(super) fn single(outcomes: Vec<RowOutcome>) -> Result<RowOutcome, DirectEngineError> {
    outcomes
        .into_iter()
        .next()
        .ok_or_else(|| DirectEngineError::Row("step returned no row".into()))
}

/// How [`DirectEngine::decode`]'s pipeline tears down the speculative
/// appends left past a finishing token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Teardown {
    /// [`super::Engine::unwind_appends`] undoes them exactly
    /// ([`super::Engine::can_unwind_lookahead`]), so the closed sequence holds
    /// what the synchronous loop leaves.
    Unwind,
    /// A model-owned family that cannot rewind its own state: the appends
    /// stay and are released when the caller closes the sequence, which every
    /// caller of `decode` does right after it (see the module docs).
    Discard,
}

impl<M: LanguageModel> DirectEngine<M> {
    /// The fused sampling parameters of the lookahead pipeline when sequence
    /// `id` may decode on it under `sampling`, `None` for the synchronous
    /// chain (see the module docs for the rules). The client streams no
    /// per-token logprobs and carries no mask or override.
    pub(super) fn lookahead_params(
        &self,
        id: SequenceId,
        sampling: &SamplingConfig,
    ) -> Option<FusedSampleParams> {
        if self.force_sync() || !self.engine().can_unwind_lookahead(id) {
            return None;
        }
        client_fused_params(sampling)
    }

    /// How a pipeline on sequence `id` would tear down its speculative
    /// appends, whatever the sampler: [`Teardown::Unwind`] under the exact
    /// rule ([`super::Engine::can_unwind_lookahead`]), [`Teardown::Discard`]
    /// for a model-owned family that cannot rewind its own state and holds
    /// the sequence on its natural backend, `None` (the synchronous chain)
    /// otherwise or under [`super::lookahead::FORCE_SYNC_ENV`].
    pub(super) fn lookahead_teardown(&self, id: SequenceId) -> Option<Teardown> {
        if self.force_sync() {
            return None;
        }
        if self.engine().can_unwind_lookahead(id) {
            return Some(Teardown::Unwind);
        }
        let model_owned = |backend| backend == SequenceStateBackend::ModelOwned;
        let discards = model_owned(self.model().sequence_state_layout().backend)
            && self
                .engine()
                .pool()
                .get(id)
                .is_some_and(|set| model_owned(set.backend));
        discards.then_some(Teardown::Discard)
    }

    /// The fused pipeline of [`DirectEngine::decode`]: the fused sampling
    /// parameters and the teardown when the sampler is fused-eligible and
    /// [`DirectEngine::lookahead_teardown`] admits the sequence, `None`
    /// otherwise.
    pub(super) fn decode_lookahead(
        &self,
        id: SequenceId,
        sampling: &SamplingConfig,
    ) -> Option<(FusedSampleParams, Teardown)> {
        let teardown = self.lookahead_teardown(id)?;
        client_fused_params(sampling).map(|params| (params, teardown))
    }

    /// The per-row pipeline of [`DirectEngine::decode`] (#2229): its teardown
    /// when the sampler is not fused-eligible but the sequence may still
    /// pipeline, `None` for the synchronous chain. The B9 pre-bias counters
    /// ([`crate::lang_bias_counters`]) read the pre-bias argmax on the host
    /// at every draw, so they keep a request synchronous. The client's rows
    /// carry no mask, override or logprobs payload, so every such row meets
    /// [`super::Engine::draw_row`]'s contract.
    pub(super) fn decode_row_lookahead(
        &self,
        id: SequenceId,
        sampling: &SamplingConfig,
    ) -> Option<Teardown> {
        if client_fused_params(sampling).is_some() || crate::lang_bias_counters::enabled() {
            return None;
        }
        self.lookahead_teardown(id)
    }

    /// Decode `state` after its first token until the finish step ends it or
    /// `on_token` returns `false`.
    pub(super) fn decode<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        on_token: &mut F,
    ) -> Result<(), DirectEngineError> {
        if let Some((params, teardown)) = self.decode_lookahead(state.id, state.sampling) {
            // The raised command-buffer input budget pays off only where
            // step n+1 is encoded while the device still runs step n; a
            // synchronous step encodes and then waits, and loses the
            // encode / execute overlap inside the step under it (the
            // scheduler measured this on M1 Ultra, `run_decode_tick`).
            let _decode_budget = crate::DecodeCommandBufferBudget::enter();
            return self.decode_pipelined(state, &params, teardown, on_token);
        }
        if let Some(teardown) = self.decode_row_lookahead(state.id, state.sampling) {
            let _decode_budget = crate::DecodeCommandBufferBudget::enter();
            return self.decode_pipelined_rows(state, teardown, on_token);
        }
        self.decode_sync(state, on_token)
    }

    /// The synchronous loop: forward, per-row sample, host read, finish.
    pub(super) fn decode_sync<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        on_token: &mut F,
    ) -> Result<(), DirectEngineError> {
        loop {
            let id = state.id;
            let input = crate::from_slice_i32(&[state.last_token()?], &[1, 1]);
            let batch = StepBatch {
                seq_ids: std::slice::from_ref(&id),
                input: &input,
            };
            let before = state.generated.len();
            let out = self
                .engine_mut()
                .step(&batch, &mut [state.row()])
                .map_err(DirectEngineError::Step)?;
            let outcome = single(out.rows)?;
            if let Some(error) = outcome.error {
                return Err(DirectEngineError::Row(error.message().to_string()));
            }
            if delivers_to_callback(&state.generated, before, &outcome) && !on_token(outcome.token)
            {
                return Ok(());
            }
            if outcome.finish.is_some() {
                return Ok(());
            }
        }
    }

    /// The lookahead pipeline (see the module docs). Holds at most two
    /// speculative appends: the forward fed by the last committed token,
    /// whose draw is still unread, and the next one submitted before the read.
    fn decode_pipelined<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        params: &FusedSampleParams,
        teardown: Teardown,
        on_token: &mut F,
    ) -> Result<(), DirectEngineError> {
        let id = state.id;
        let input = crate::from_slice_i32(&[state.last_token()?], &[1, 1]);
        let mut pending = match self.submit_lookahead(id, state.sampling, &input, params) {
            Ok(tokens) => tokens,
            Err(err) => {
                self.unwind_failed_submit(id, &err, 0, teardown)?;
                return self.decode_sync(state, on_token);
            }
        };
        loop {
            // `pending` holds token number `generated.len()`; when that one
            // spends the budget no forward follows it.
            let spends_budget = state.generated.len() + 1 >= state.max_tokens;
            let next = if spends_budget {
                None
            } else {
                let input = lookahead_feedback_input(&pending);
                match self.submit_lookahead(id, state.sampling, &input, params) {
                    Ok(tokens) => Some(tokens),
                    Err(err) => {
                        // The pending forward's append is still uncommitted.
                        drop(pending);
                        self.unwind_failed_submit(id, &err, 1, teardown)?;
                        return self.decode_sync(state, on_token);
                    }
                }
            };
            // The sync point: waits for `pending` only, while `next` runs.
            let host = tokens_to_host(&pending);
            drop(pending);
            let before = state.generated.len();
            let outcome = single(self.engine_mut().finish_rows(&host, &mut [state.row()]))?;
            if let Some(error) = outcome.error {
                // Nothing was committed: undo the read step's append and the
                // next one, so the closed sequence holds no speculative state.
                let n = 1 + i32::from(next.is_some());
                self.retire(id, next.as_deref(), n, teardown)?;
                return Err(DirectEngineError::Row(error.message().to_string()));
            }
            let stopped = delivers_to_callback(&state.generated, before, &outcome)
                && !on_token(outcome.token);
            if stopped || outcome.finish.is_some() {
                if next.is_some() {
                    self.retire(id, next.as_deref(), 1, teardown)?;
                }
                return Ok(());
            }
            match next {
                Some(tokens) => pending = tokens,
                // The budget check above and the finish step's length rule
                // agree, so this is unreachable; the synchronous loop
                // continues from the committed state if it ever is not.
                None => return self.decode_sync(state, on_token),
            }
        }
    }

    /// Submit one pipelined step for `id` on `input` (`[1, 1]`).
    pub(super) fn submit_lookahead(
        &mut self,
        id: SequenceId,
        sampling: &SamplingConfig,
        input: &MlxArray,
        params: &FusedSampleParams,
    ) -> Result<UniquePtr<MlxArray>, EngineError> {
        let batch = StepBatch {
            seq_ids: std::slice::from_ref(&id),
            input,
        };
        self.engine_mut()
            .submit(&batch, params, &[&sampling.token_bias])
    }

    /// Unwind after a failed [`super::Engine::submit`]: the `uncommitted`
    /// appends already in place plus the failed forward's own when it threw
    /// at the schedule ([`EngineError::Eval`]; any other error appended
    /// nothing). The caller then decodes synchronously, which re-runs the
    /// step through the guarded per-row chain and reports a persistent
    /// failure there. Under [`Teardown::Discard`] a failure that left an
    /// append in place cannot be undone, so it ends the run with `err`.
    pub(super) fn unwind_failed_submit(
        &mut self,
        id: SequenceId,
        err: &EngineError,
        uncommitted: i32,
        teardown: Teardown,
    ) -> Result<(), DirectEngineError> {
        tracing::debug!(seq_id = %id, error = %err, "decode lookahead: submit failed, decoding synchronously");
        let n = uncommitted + i32::from(matches!(err, EngineError::Eval(_)));
        if teardown == Teardown::Discard && n > 0 {
            return Err(DirectEngineError::Step(err.clone()));
        }
        self.engine_mut()
            .unwind_appends(id, n)
            .map_err(DirectEngineError::Step)
    }

    /// Tear down `n` speculative appends, the last of which produced `next`.
    /// A model-owned family's rewind edits the model's own state, so its
    /// in-flight forward is synced first, as the scheduler's finishing
    /// teardown does; a pool trim only moves a host offset, and the lazy
    /// slices it implies are ordered after the append on the stream. Under
    /// [`Teardown::Discard`] the appends stay for the caller's close.
    pub(super) fn retire(
        &mut self,
        id: SequenceId,
        next: Option<&MlxArray>,
        n: i32,
        teardown: Teardown,
    ) -> Result<(), DirectEngineError> {
        if let Some(next) = next
            && self.model().sequence_state_layout().backend == SequenceStateBackend::ModelOwned
            && let Err(err) = crate::try_eval(next)
        {
            tracing::debug!(seq_id = %id, error = %err, "decode lookahead: discarded step failed to evaluate");
        }
        if teardown == Teardown::Discard {
            return Ok(());
        }
        self.engine_mut()
            .unwind_appends(id, n)
            .map_err(DirectEngineError::Step)
    }
}
