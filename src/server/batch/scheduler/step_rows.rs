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

//! The scheduler's side of an engine step row (#2172).
//!
//! [`step_row`] views one [`SequenceInfo`] as the [`StepRow`] the engine
//! samples and finishes: its sampler, config, history and stream, plus
//! [`SequenceStepHooks`], which supplies the structured-output mask and
//! matcher, the thinking-budget override, the first-token stamp and the
//! [`FinishHooks`] half (stop matcher, generation bounds, context bound,
//! streaming) that `finish.rs` already implements. [`apply_row_outcomes`]
//! turns each [`RowOutcome`] back into the sequence's state: a row's error is
//! that row's `FinishReason::Error` and nothing else's (#822).

use std::time::Instant;

use mlxcel_core::FinishHooks;
use mlxcel_core::engine::{RowError, RowOutcome, StepRow, StepRowHooks};
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::sampling_row_step::LogitMask;

use super::run_loop::{SharedStructuredConstraint, StructuredMask};
use super::*;
use crate::server::batch::finish::{ContextBound, SequenceFinishHooks};
use crate::server::batch::sequence::{FirstTokenOrigin, stamp_first_token};

/// [`StepRowHooks`] over the disjoint `SequenceInfo` fields a step touches
/// besides what the [`StepRow`] itself borrows.
pub(crate) struct SequenceStepHooks<'s> {
    finish: SequenceFinishHooks<'s>,
    thinking: &'s mut ThinkingState,
    constraint: Option<&'s SharedStructuredConstraint>,
    mask: Option<StructuredMask<'s>>,
    /// Set for a prefill completion: the emitted token is the request's
    /// first, so its time is stamped and the prefill stats event streamed.
    first_token: Option<(&'s mut Option<Instant>, FirstTokenOrigin)>,
}

impl FinishHooks for SequenceStepHooks<'_> {
    fn stop_text(&mut self, token: i32, generated_len: usize) -> bool {
        self.finish.stop_text(token, generated_len)
    }

    fn bound_stopped(&self) -> bool {
        self.finish.bound_stopped()
    }

    fn context_bound_due(&self, generated_len: usize) -> bool {
        self.finish.context_bound_due(generated_len)
    }
}

impl StepRowHooks for SequenceStepHooks<'_> {
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask> {
        self.mask.as_mut().map(|m| m as &mut dyn LogitMask)
    }

    fn override_token(&mut self, sampled: i32) -> i32 {
        BatchScheduler::apply_thinking_budget(self.thinking, sampled)
    }

    fn consume_sampled(&mut self, sampled: i32) -> Result<bool, String> {
        match self.constraint {
            Some(constraint) => BatchScheduler::consume_structured_token(constraint, sampled),
            None => Ok(false),
        }
    }

    fn resolved(&mut self, _sampled: i32, _token: i32) {
        if let Some((first_token_time, origin)) = self.first_token.take() {
            stamp_first_token(
                first_token_time,
                self.finish.decode_state,
                self.finish.response_tx,
                origin,
                Instant::now(),
            );
        }
    }

    fn set_logprobs(&mut self, logprobs: Option<TokenLogprobData>) {
        self.finish.logprobs = logprobs;
    }
}

/// `seq` as the engine's step row. `first_token` marks a prefill completion.
pub(crate) fn step_row<'s>(
    seq: &'s mut SequenceInfo,
    tokenizer: &'s MlxcelTokenizer,
    context: ContextBound,
    first_token: bool,
) -> StepRow<'s, SequenceStepHooks<'s>> {
    let SequenceInfo {
        seq_id,
        sampler,
        sampling,
        token_history,
        generated_tokens,
        merged_eos,
        max_tokens,
        logprobs_config,
        decode_state,
        stop_matcher,
        bounds,
        response_tx,
        prompt_tokens,
        vlm_embeddings,
        thinking,
        structured,
        first_token_time,
        already_cached_tokens,
        created_at,
        ..
    } = seq;
    let constraint = structured.as_ref();
    let hooks = SequenceStepHooks {
        finish: SequenceFinishHooks {
            decode_state,
            stop_matcher,
            bounds,
            response_tx,
            tokenizer,
            logprobs: None,
            prompt_len: prompt_tokens.len(),
            is_vlm: vlm_embeddings.is_some(),
            context,
        },
        thinking,
        constraint,
        mask: constraint.map(StructuredMask),
        first_token: first_token.then_some((
            first_token_time,
            FirstTokenOrigin {
                prompt_tokens: prompt_tokens.len(),
                cached_tokens: *already_cached_tokens,
                created_at: *created_at,
            },
        )),
    };
    StepRow {
        seq_id: *seq_id,
        needs_mask: constraint.is_some(),
        needs_override: !hooks.thinking.is_disabled(),
        sampler,
        sampling,
        token_history,
        generated: generated_tokens,
        eos: merged_eos,
        max_tokens: *max_tokens,
        logprobs: logprobs_config,
        hooks,
    }
}

/// The user-facing prefix of a row error's message.
pub(crate) fn row_error_prefix(error: &RowError) -> &'static str {
    match error {
        RowError::Structured(_) => "structured output",
        RowError::Eval(_) => "inference backend",
    }
}

impl BatchScheduler {
    /// Record each row's outcome: a successful eval clears the backend
    /// health counter, a failed one bumps it (#822), and a failed row is
    /// finished with its error. Returns `false` when the health counter is
    /// exhausted and the caller must stop.
    pub(super) fn apply_row_outcomes(&mut self, outcomes: &[RowOutcome]) -> bool {
        for outcome in outcomes {
            match &outcome.error {
                None => self.note_eval_success(),
                Some(error) => {
                    let eval = matches!(error, RowError::Eval(_));
                    if eval {
                        let _ = self.record_eval_outcome(Err(error.message().to_string()));
                    }
                    Self::abort_sequence_with_error(
                        self.active_batch.get_mut(outcome.seq_id),
                        row_error_prefix(error),
                        error.message(),
                    );
                    if eval && self.eval_failures_exhausted() {
                        return false;
                    }
                }
            }
        }
        true
    }
}
