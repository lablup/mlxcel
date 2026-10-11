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

//! The raw-completion client's per-row lookahead pipeline (#2229), for the
//! requests the fused pipeline turns away: history penalties, DRY, mirostat,
//! adaptive-p and the extended chain.
//!
//! Their draw for token t+1 reads token t on the host (it must be in the
//! penalty history, or confirmed to the feedback state), so the draw cannot
//! ride the forward as the fused draw does. The forward can: it is fed by
//! the still-unread device token, exactly as the retired CLI generator's
//! `sample_next_step` did. Each iteration submits the forward fed by the
//! pending draw ([`super::Engine::submit_forward`]), reads the pending token
//! while that forward runs, commits it ([`super::Engine::finish_rows`], which
//! pushes it to the history), and only then builds the next draw from the
//! submitted logits ([`super::Engine::draw_row`]). The draw for t+1 is built
//! from the same logits, the same history and the same sampler state as the
//! synchronous step's, and forwards consume no randomness, so greedy and
//! seeded streams equal [`super::DirectEngine::decode_sync`]'s token for token.
//!
//! A failed draw restores sampler feedback before synchronous fallback. A
//! stochastic draw already consumed a thread-local RNG key while building its
//! graph. This per-row path does not retain an RNG snapshot, so a discarded
//! draw's key is not restored and the failure may shift a seeded stream by one
//! draw. A failed forward submit commits the pending draw first.
//!
//! The overlap differs by sampler. Penalties, DRY and the extended chain
//! without adaptive-p keep the draw lazy on the device, so the draw and the
//! finish step both overlap the next forward. Mirostat and adaptive-p read the
//! uniform draw on the host inside the sampler, so for them only the finish
//! step overlaps the next forward.
//!
//! The teardown rules are the fused pipeline's (`direct_decode`): one
//! uncommitted append past a finishing token or a callback stop, none after
//! the token that spends the budget, and a synchronous fallback from the last
//! committed token after a failed submit or draw. A draw that fails at the
//! host read ends the run with that error, as the synchronous step's does.

use super::direct::{DirectEngine, DirectEngineError, delivers_to_callback};
use super::direct_decode::{DecodeState, Teardown, single};
use super::lookahead::lookahead_feedback_input;
use super::{EngineError, StepBatch};
use crate::cache::SequenceId;
use crate::generate::LanguageModel;
use crate::{MlxArray, UniquePtr};

impl<M: LanguageModel> DirectEngine<M> {
    /// The per-row lookahead pipeline (see the module docs). Holds at most
    /// two speculative appends: the forward whose draw is still unread, and
    /// the next one, fed by that draw, submitted before the read.
    pub(super) fn decode_pipelined_rows<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        teardown: Teardown,
        on_token: &mut F,
    ) -> Result<(), DirectEngineError> {
        let id = state.id;
        let input = crate::from_slice_i32(&[state.last_token()?], &[1, 1]);
        let logits = match self.submit_row_forward(id, &input) {
            Ok(logits) => logits,
            Err(err) => {
                self.unwind_failed_submit(id, &err, 0, teardown)?;
                return self.decode_sync(state, on_token);
            }
        };
        let mut pending = match self.engine_mut().draw_row(&logits, &mut state.row()) {
            Ok(token) => token,
            Err(err) => {
                self.unwind_failed_draw(id, &err, &logits, teardown)?;
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
                match self.submit_row_forward(id, &input) {
                    Ok(logits) => Some(logits),
                    Err(err) => {
                        self.unwind_failed_submit(id, &err, 0, teardown)?;
                        if self.commit_row_pending(state, pending, None, teardown, on_token)? {
                            return Ok(());
                        }
                        return self.decode_sync(state, on_token);
                    }
                }
            };
            if self.commit_row_pending(state, pending, next.as_deref(), teardown, on_token)? {
                return Ok(());
            }
            // The budget check above and the finish step's length rule agree,
            // so `None` is unreachable; the synchronous loop continues from
            // the committed state if it ever is not.
            let Some(logits) = next else {
                return self.decode_sync(state, on_token);
            };
            // The token just committed is now in the history: draw the next.
            pending = match self.engine_mut().draw_row(&logits, &mut state.row()) {
                Ok(token) => token,
                Err(err) => {
                    self.unwind_failed_draw(id, &err, &logits, teardown)?;
                    return self.decode_sync(state, on_token);
                }
            };
        }
    }

    /// Read and commit the pending draw, shared by normal and failed submits.
    /// Returns whether the run finished; `next` owns only an uncommitted append.
    fn commit_row_pending<F: FnMut(i32) -> bool>(
        &mut self,
        state: &mut DecodeState<'_>,
        pending: UniquePtr<MlxArray>,
        next: Option<&MlxArray>,
        teardown: Teardown,
        on_token: &mut F,
    ) -> Result<bool, DirectEngineError> {
        let id = state.id;
        // Wait only for pending, preserving overlap with the next forward.
        let host = match super::rows::try_read_token(&pending) {
            Ok(host) => host,
            Err(msg) => {
                drop(pending);
                self.retire(id, next, 1 + i32::from(next.is_some()), teardown)?;
                return Err(DirectEngineError::Row(msg));
            }
        };
        drop(pending);
        let before = state.generated.len();
        let outcome = single(self.engine_mut().finish_rows(&[host], &mut [state.row()]))?;
        if let Some(error) = outcome.error {
            self.retire(id, next, 1 + i32::from(next.is_some()), teardown)?;
            return Err(DirectEngineError::Row(error.message().to_string()));
        }
        let stopped =
            delivers_to_callback(&state.generated, before, &outcome) && !on_token(outcome.token);
        if stopped || outcome.finish.is_some() {
            if next.is_some() {
                self.retire(id, next, 1, teardown)?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Submit the forward of one per-row pipelined step for `id` on `input`
    /// (`[1, 1]`).
    fn submit_row_forward(
        &mut self,
        id: SequenceId,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, EngineError> {
        let batch = StepBatch {
            seq_ids: std::slice::from_ref(&id),
            input,
        };
        self.engine_mut().submit_forward(&batch)
    }

    /// Unwind after a failed [`super::Engine::draw_row`]: the forward that
    /// produced `logits` appended one position nothing committed, so it is
    /// retired (synced first for a model-owned family) before the caller
    /// decodes synchronously from the last committed token. Under
    /// [`Teardown::Discard`] that append cannot be undone, so the run ends
    /// with `err`.
    fn unwind_failed_draw(
        &mut self,
        id: SequenceId,
        err: &EngineError,
        logits: &MlxArray,
        teardown: Teardown,
    ) -> Result<(), DirectEngineError> {
        tracing::debug!(seq_id = %id, error = %err, "decode row lookahead: draw failed, decoding synchronously");
        if teardown == Teardown::Discard {
            return Err(DirectEngineError::Step(err.clone()));
        }
        self.retire(id, Some(logits), 1, teardown)
    }
}
