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

//! One verify round of a token-only drafter, shared by every caller
//! (#2255): the raw-completion client's drafter loop
//! ([`super::DirectEngine::generate_with_drafter`], `mlxcel generate
//! --prompt-lookup`) and the server scheduler's prompt-lookup rows run the
//! same [`Engine::verify_round`], and decide whether a sequence may run one
//! through the same [`Engine::verify_rounds_unsupported`].
//!
//! A round forwards `[current, d_0, .., d_{k-1}]` as speculative appends
//! ([`Engine::verify`]), decides every position in order through the row's
//! finish step (one batched argmax and a single host read on the pure greedy
//! path, the per-row chain otherwise), keeps the longest agreeing prefix plus
//! the target's own token, and leaves the sequence's state exactly where the
//! equivalent run of one-token steps would: the accepted proposals are
//! committed ([`Engine::commit_appends`]) and the rejected tail unwound
//! ([`Engine::unwind_appends`]). A position that finishes the row (EOS, a
//! stop string, `max_tokens`, the context bound) ends the round there and the
//! positions after it are unwound. A position that fails unwinds every
//! append of the round and is reported on the last outcome, so a caller that
//! keeps serving other rows (the scheduler, #822) never sees a half-applied
//! round.

use super::direct::{delivers_to_callback, logits_vocab};
use super::rows::{RowError, RowOutcome, StepRow, StepRowHooks, finish_row, sample_and_finish_row};
use super::speculative::SpeculativeRunError;
use super::{Engine, EngineError};
use crate::cache::{SequenceId, can_trim_prompt_cache};
use crate::decode_finish::FinishCause;
use crate::drafter::Drafter;
use crate::ffi;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::speculative::prompt_lookup::prompt_lookup_unsupported_reason;
use crate::{MlxArray, UniquePtr};

/// What one [`Engine::verify_round`] decided.
pub struct VerifyRound {
    /// One outcome per decided position, in order: the accepted proposals,
    /// then the target's own token (the replacement at the first
    /// disagreement, or the bonus token after a fully accepted block). A
    /// position that finishes the row, or after which `on_token` stopped,
    /// is the last. A failed position is the last outcome and carries the
    /// error; every append of the round was then unwound.
    pub outcomes: Vec<RowOutcome>,
    /// Proposals the target confirmed.
    pub accepted: usize,
    /// Whether `on_token` stopped the round.
    pub stopped: bool,
    /// The verify forward's `[1, k + 1, vocab]` logits, for the hidden-state
    /// slot of [`Drafter::accept_verified_tokens`] (a token-only drafter
    /// ignores it).
    pub logits: UniquePtr<MlxArray>,
}

impl VerifyRound {
    /// The failure the round ended on, if any.
    pub fn error(&self) -> Option<&RowError> {
        self.outcomes.last().and_then(|o| o.error.as_ref())
    }

    /// The finish the round ended on, if any.
    pub fn finish(&self) -> Option<FinishCause> {
        self.outcomes.last().and_then(|o| o.finish)
    }

    /// Whether the sequence is done after this round: it finished, failed,
    /// or `on_token` stopped it.
    pub fn ended(&self) -> bool {
        self.stopped || self.finish().is_some() || self.error().is_some()
    }
}

/// The sequence-independent half of [`Engine::verify_rounds_unsupported`]:
/// the drafter must propose from the token context alone, and the sampler
/// must not carry feedback state across tokens (a verify block decides
/// several positions from one forward and cannot replay that state).
pub fn token_only_rounds_unsupported(
    sampling: &SamplingConfig,
    drafter: &dyn Drafter,
) -> Option<SpeculativeRunError> {
    if !drafter.drafts_from_tokens_only() {
        return Some(SpeculativeRunError::NeedsHiddenStates);
    }
    if sampling.needs_sampler_feedback_state() {
        return Some(SpeculativeRunError::SamplerFeedbackState);
    }
    None
}

impl<M: LanguageModel> Engine<M> {
    /// Why sequence `id` cannot run token-only verify rounds of `drafter`
    /// under `sampling`, or `None` when it can: the one eligibility rule of
    /// the drafter loop and the scheduler's prompt-lookup rows. On top of
    /// [`token_only_rounds_unsupported`], a rejected block is dropped through
    /// the pool caches, so they must be trimmable (a family with no external
    /// caches has nothing to trim), and the model must keep all of its
    /// sequence state in them ([`prompt_lookup_unsupported_reason`]).
    pub fn verify_rounds_unsupported(
        &mut self,
        id: SequenceId,
        sampling: &SamplingConfig,
        drafter: &dyn Drafter,
    ) -> Option<SpeculativeRunError> {
        if let Some(err) = token_only_rounds_unsupported(sampling, drafter) {
            return Some(err);
        }
        let trimmable = self
            .pool
            .get_caches_mut(id)
            .is_some_and(|caches| !caches.is_empty() && can_trim_prompt_cache(caches));
        if !trimmable {
            return Some(SpeculativeRunError::NotTrimmable);
        }
        prompt_lookup_unsupported_reason(&self.model).map(SpeculativeRunError::ModelUnsupported)
    }

    /// Run one verify forward at every block width a verify round can use
    /// (2 through `max_draft + 1`) on the open sequence `id`, feeding `token`
    /// at every position, and unwind each, so a later round does not pay a
    /// width's first-use kernel cost in the middle of a decode (on GB10 that
    /// cost more than the rest of a short reply). The sequence's state is
    /// left as it was, also when a width fails: its appends are unwound
    /// before the error is returned.
    pub fn warm_up_verify_widths(
        &mut self,
        id: SequenceId,
        token: i32,
        max_draft: usize,
    ) -> Result<(), EngineError> {
        for width in 2..=max_draft + 1 {
            let tokens = vec![token; width];
            let input = ffi::from_slice_i32(&tokens, &[1, width as i32]);
            let logits = self.verify(id, &input)?;
            let argmax = ffi::argmax_last_axis(&logits);
            let evaluated = crate::try_eval(&argmax);
            self.unwind_appends(id, width as i32)?;
            evaluated.map_err(|err| EngineError::Eval(err.to_string()))?;
        }
        Ok(())
    }

    /// One verify round for `row` (see the module docs): `current` is the
    /// row's last emitted token, which its state does not hold yet, and
    /// `draft` the proposed block (non-empty, already capped so the round
    /// cannot pass `max_tokens`). `on_token` sees every token the finish
    /// step streams, in order, and stops the round by returning `false`.
    ///
    /// `Err` is a failure before any position was decided (the sequence is
    /// not open) or a rewind that left the state desynchronized
    /// ([`EngineError::Rewind`]); the caller finishes the row with it.
    pub fn verify_round<H: StepRowHooks>(
        &mut self,
        row: &mut StepRow<'_, H>,
        current: i32,
        draft: &[i32],
        mut on_token: impl FnMut(i32) -> bool,
    ) -> Result<VerifyRound, EngineError> {
        let id = row.seq_id;
        let verify_len = draft.len() + 1;
        let mut verify_tokens = Vec::with_capacity(verify_len);
        verify_tokens.push(current);
        verify_tokens.extend_from_slice(draft);
        let input = ffi::from_slice_i32(&verify_tokens, &[1, verify_len as i32]);
        let logits = self.verify(id, &input)?;

        let vocab = match logits_vocab(&ffi::array_shape(&logits), verify_len) {
            Ok(vocab) => vocab,
            Err(msg) => {
                return Ok(self.failed_round(
                    id,
                    verify_len,
                    logits,
                    vec![row_failure(id, RowError::Eval(msg))],
                ));
            }
        };
        // Pure argmax with nothing that depends on the emitted prefix or on
        // the row's hooks: every position is decided from one batched argmax
        // and a single host read. Anything else goes position by position
        // through the per-row chain.
        let sampling = row.sampling;
        let batched_argmax = sampling.is_greedy_path()
            && !sampling.needs_token_history()
            && sampling.token_bias.is_empty()
            && !row.needs_mask
            && !row.needs_override
            && !row.logprobs.enabled;
        let greedy_targets = if batched_argmax {
            let argmax = ffi::argmax_last_axis(&logits);
            if let Err(err) = crate::try_eval(&argmax) {
                let outcome = row_failure(id, RowError::Eval(err.to_string()));
                return Ok(self.failed_round(id, verify_len, logits, vec![outcome]));
            }
            let targets = crate::drafter::dflash::materialize_argmax_i32_vec(&argmax, verify_len);
            if targets.len() != verify_len {
                let msg = format!(
                    "verify argmax returned {} tokens for {verify_len} positions",
                    targets.len()
                );
                return Ok(self.failed_round(
                    id,
                    verify_len,
                    logits,
                    vec![row_failure(id, RowError::Eval(msg))],
                ));
            }
            Some(targets)
        } else {
            None
        };

        let mut outcomes = Vec::with_capacity(verify_len);
        let mut accepted = 0usize;
        let mut stopped = false;
        for pos in 0..verify_len {
            let emitted_before = row.generated.len();
            let outcome = match &greedy_targets {
                Some(targets) => finish_row(row, targets[pos], targets[pos], None, false),
                None => {
                    let pos_logits =
                        ffi::slice(&logits, &[0, pos as i32, 0], &[1, pos as i32 + 1, vocab]);
                    sample_and_finish_row(&pos_logits, row)
                }
            };
            if outcome.error.is_some() {
                outcomes.push(outcome);
                return Ok(self.failed_round(id, verify_len, logits, outcomes));
            }
            let delivered = delivers_to_callback(row.generated, emitted_before, &outcome);
            let token = outcome.token;
            let finished = outcome.finish.is_some();
            outcomes.push(outcome);
            if delivered && !on_token(token) {
                stopped = true;
                break;
            }
            if finished {
                break;
            }
            // The target's token is emitted either way; it only extends the
            // round when it confirms the proposal in the same slot.
            if pos < draft.len() && token == draft[pos] {
                accepted += 1;
            } else {
                break;
            }
        }

        // The forward appended `current, d_0..d_{k-1}`. The state must end
        // holding every emitted token except the next round's current token,
        // i.e. through `d_{a-1}`: keep `a + 1`, drop `k - a`.
        let rejected = (draft.len() - accepted) as i32;
        self.unwind_appends(id, rejected)?;
        // A step does not advance the pool offset for the token that finishes
        // the sequence on an EOS, so the EOS slot is not committed here either.
        let ended_on_eos = outcomes.last().and_then(|o| o.finish) == Some(FinishCause::Eos);
        self.commit_appends(id, accepted as i32 + i32::from(!ended_on_eos));
        Ok(VerifyRound {
            outcomes,
            accepted,
            stopped,
            logits,
        })
    }

    /// A round that ended on a failed position: every append of the round
    /// is unwound, best effort, because the row finishes with the error
    /// either way and its state is never donated.
    fn failed_round(
        &mut self,
        id: SequenceId,
        verify_len: usize,
        logits: UniquePtr<MlxArray>,
        outcomes: Vec<RowOutcome>,
    ) -> VerifyRound {
        if let Err(err) = self.unwind_appends(id, verify_len as i32) {
            tracing::warn!(seq_id = %id, error = %err, "verify round: unwind after a failed position");
        }
        VerifyRound {
            outcomes,
            accepted: 0,
            stopped: false,
            logits,
        }
    }
}

fn row_failure(seq_id: SequenceId, error: RowError) -> RowOutcome {
    RowOutcome {
        seq_id,
        sampled: 0,
        token: 0,
        finish: None,
        error: Some(error),
    }
}
