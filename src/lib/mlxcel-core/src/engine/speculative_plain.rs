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

//! The pipelined plain rounds of the token-only drafter loop
//! ([`DirectEngine::generate_with_drafter`], #2176), restored from the
//! retired prompt-lookup loop: once the drafter reports a run of rounds
//! without a proposal ([`Drafter::pipelines_plain_rounds`]), each one-token
//! forward is submitted from the still-unread draw of the one before, on
//! the same split step and teardown as `generate`'s pipeline. A round that
//! finds a proposal while a step is in flight reads that step and verifies
//! from it synchronously on the next round; the switch costs no forward.

use super::direct::{DirectEngine, DirectEngineError, delivers_to_callback};
use super::direct_decode::{DecodeState, single};
use super::lookahead::lookahead_feedback_input;
use super::rows::tokens_to_host;
use super::speculative::{SpeculativeRounds, SpeculativeRunError};
use crate::drafter::Drafter;
use crate::generate::LanguageModel;
use crate::sampling::FusedSampleParams;
use crate::{MlxArray, UniquePtr};

impl<M: LanguageModel> DirectEngine<M> {
    /// One drafter-loop round with a plain step in flight (`in_flight`, the
    /// lazy draw of the forward fed by the last emitted token). The next
    /// one-token forward is submitted before the read unless the round's
    /// lookup found something to verify (`draft`) or the in-flight token is
    /// the last one allowed; then the in-flight token is read, committed and
    /// emitted. `asked` says whether `draft` came from
    /// [`Drafter::draft_block`]; a call no forward follows is retracted.
    /// Returns whether the run is done. A finish or a callback stop unwinds
    /// the forward submitted past it, so the state is the synchronous loop's.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pipelined_plain_round<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        in_flight: &mut Option<UniquePtr<MlxArray>>,
        draft: &[i32],
        asked: bool,
        params: Option<&FusedSampleParams>,
        drafter: &mut dyn Drafter,
        rounds: &mut SpeculativeRounds,
        on_token: &mut F,
    ) -> Result<bool, SpeculativeRunError> {
        let Some(pending) = in_flight.take() else {
            return Ok(false);
        };
        let id = state.id;
        let remaining = state.max_tokens.saturating_sub(state.generated.len());
        let mut next = None;
        if draft.is_empty()
            && remaining > 1
            && let Some(params) = params
        {
            let input = lookahead_feedback_input(&pending);
            match self.submit_lookahead(id, state.sampling, &input, params) {
                Ok(tokens) => {
                    rounds.rounds += 1;
                    next = Some(tokens);
                }
                Err(err) => {
                    // Undo the in-flight append too: the loop re-runs this
                    // round synchronously from the last emitted token.
                    drop(pending);
                    if asked {
                        drafter.retract_draft(draft);
                    }
                    self.unwind_failed_submit(id, &err, 1)?;
                    return Ok(false);
                }
            }
        }
        if next.is_none() && asked {
            drafter.retract_draft(draft);
        }
        let host = tokens_to_host(&pending);
        let before = state.generated.len();
        let outcome = single(self.engine_mut().finish_rows(&host, &mut [state.row()]))?;
        if let Some(error) = outcome.error {
            let n = 1 + i32::from(next.is_some());
            self.retire(id, next.as_deref(), n)?;
            return Err(DirectEngineError::Row(error.message().to_string()).into());
        }
        let emitted = &state.generated[before..];
        drafter.accept_verified_tokens(&pending, &[], 0, emitted, state.sampling)?;
        let stopped =
            delivers_to_callback(&state.generated, before, &outcome) && !on_token(outcome.token);
        if stopped || outcome.finish.is_some() {
            if next.is_some() {
                self.retire(id, next.as_deref(), 1)?;
            }
            return Ok(true);
        }
        *in_flight = next;
        Ok(false)
    }
}
