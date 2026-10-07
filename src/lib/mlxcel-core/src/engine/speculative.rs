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

//! Speculative decoding on the engine for a token-only drafter (#2176).
//!
//! A [`Drafter`] whose proposals come from the token context alone
//! ([`Drafter::drafts_from_tokens_only`], prompt lookup today) needs no
//! hidden states from the target, so one round loop can drive it over any
//! model through the engine: [`Engine::verify`] forwards the current token
//! and the proposed block as speculative appends, the per-row chain samples
//! or argmaxes every position in order, and [`Engine::unwind_appends`] drops
//! the rejected tail so the sequence's state holds exactly the emitted
//! tokens. [`DirectEngine::generate_with_drafter`] is that loop for one
//! sequence; the server's scheduler can offer the same drafter through the
//! same two engine entries.

use std::time::Instant;

use super::direct::{
    BareHooks, DirectEngine, DirectEngineError, DirectRequest, DirectRun, delivers_to_callback,
};
use super::rows::{finish_row, sample_and_finish_row};
use super::{Engine, EngineError, StepBatch, StepRow};
use crate::cache::{DecodeLookaheadAppendScope, SequenceId, can_trim_prompt_cache};
use crate::drafter::{Drafter, DrafterError};
use crate::ffi;
use crate::generate::{GenerationStats, LanguageModel, SamplingConfig};
use crate::generation_policy::{initial_token_history, merged_eos_token_ids, seed_rng_if_needed};
use crate::sampling::LogprobsConfig;
use crate::sampling_row_step::RowSampler;
use crate::streams::install_thread_local_default_stream;
use crate::{MlxArray, UniquePtr};

impl<M: LanguageModel> Engine<M> {
    /// A verify forward for `id`: `input` is `[1, n]`, the current token
    /// followed by a proposed block, appended to the sequence's state as
    /// speculative positions (inside a [`DecodeLookaheadAppendScope`], so a
    /// model-owned family logs what the appends overwrote). Returns the
    /// `[1, n, vocab]` logits; the caller samples every position, keeps the
    /// accepted prefix with [`Engine::commit_appends`] and drops the rest
    /// with [`Engine::unwind_appends`].
    pub fn verify(
        &mut self,
        id: SequenceId,
        input: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, EngineError> {
        let caches = self
            .pool
            .get_caches_mut(id)
            .ok_or(EngineError::MissingSequence(id))?;
        let logits = {
            let _speculative = DecodeLookaheadAppendScope::enter();
            self.model
                .forward_with_sequence_id(input, Some(id), caches, None)
        };
        self.sync_sequence_storage(id);
        Ok(logits)
    }

    /// Keep `n` of the positions a verify forward appended for `id`: the
    /// pool offset advances by that many, as a step advances it by one.
    pub fn commit_appends(&mut self, id: SequenceId, n: i32) {
        if n <= 0 {
            return;
        }
        if let Some(set) = self.pool.get_mut(id) {
            set.current_offset += n;
        }
    }
}

/// Why a speculative run on the engine stopped early.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpeculativeRunError {
    #[error(transparent)]
    Engine(#[from] DirectEngineError),
    #[error("drafter: {0}")]
    Drafter(String),
    #[error(
        "the drafter needs the target's hidden states, which the token-only round loop does not produce"
    )]
    NeedsHiddenStates,
    #[error("the sequence's KV caches cannot drop a rejected block (not trimmable)")]
    NotTrimmable,
    #[error("the sampler carries feedback state across tokens, which a verify block cannot replay")]
    SamplerFeedbackState,
}

impl From<DrafterError> for SpeculativeRunError {
    fn from(err: DrafterError) -> Self {
        SpeculativeRunError::Drafter(err.to_string())
    }
}

/// Acceptance accounting for one [`DirectEngine::generate_with_drafter`] run,
/// as the round loop sees it: one round per forward.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeRounds {
    /// Forwards after the prefill: plain steps and verify blocks.
    pub rounds: usize,
    /// Rounds that verified a proposed block.
    pub drafted_rounds: usize,
    pub proposed_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
}

/// The result of a speculative run.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeRun {
    pub run: DirectRun,
    pub rounds: SpeculativeRounds,
}

impl<M: LanguageModel> DirectEngine<M> {
    /// One completion with a token-only `drafter` proposing up to
    /// `block_size` tokens per round. Every round forwards the current token
    /// and the proposal in one [`Engine::verify`], emits the longest agreeing
    /// prefix plus the target's own token at the first disagreement (or the
    /// bonus token after a full block), and unwinds the rest; a round with
    /// no proposal is a plain [`Engine::step`]. Greedy output is identical
    /// to [`DirectEngine::generate`]'s.
    ///
    /// `on_token` sees every emitted token in order, as in `generate`. The
    /// drafter is bound to the model, told the prompt and the first token,
    /// and told every round's outcome through
    /// [`Drafter::accept_verified_tokens`] (with the verify logits, or the
    /// step's `[1, 1, vocab]` logits, in the hidden-state slot a token-only
    /// drafter ignores).
    pub fn generate_with_drafter<F: FnMut(i32) -> bool>(
        &mut self,
        request: &DirectRequest<'_>,
        drafter: &mut dyn Drafter,
        block_size: usize,
        on_token: F,
    ) -> Result<SpeculativeRun, SpeculativeRunError> {
        if !drafter.drafts_from_tokens_only() {
            return Err(SpeculativeRunError::NeedsHiddenStates);
        }
        let sampling = self.compose_sampling(request.sampling);
        if sampling.needs_sampler_feedback_state() {
            return Err(SpeculativeRunError::SamplerFeedbackState);
        }
        if request.max_tokens == 0 {
            return Ok(SpeculativeRun {
                run: DirectRun::empty(request.prompt_tokens.len()),
                rounds: SpeculativeRounds::default(),
            });
        }
        install_thread_local_default_stream(self.generation_stream());
        drafter.bind(self.model())?;
        let id = self.open_sequence()?;
        let result = self.speculate(id, request, &sampling, drafter, block_size, on_token);
        self.close_sequence(id);
        result
    }

    /// Prefill `prompt_tokens` and run one verify forward at every block
    /// width `generate_with_drafter` can use (2 through `max_draft + 1`), so
    /// a later run does not pay first-use kernel costs in the middle of its
    /// decode. The first forward at a width this process has not run yet
    /// builds or traces kernels for it, which on GB10 cost more than the
    /// rest of a short reply.
    pub fn warm_up_verify_widths(
        &mut self,
        prompt_tokens: &[i32],
        max_draft: usize,
    ) -> Result<(), SpeculativeRunError> {
        let Some(&last) = prompt_tokens.last() else {
            return Err(DirectEngineError::EmptyPrompt.into());
        };
        install_thread_local_default_stream(self.generation_stream());
        let id = self.open_sequence()?;
        let result = (|| -> Result<(), SpeculativeRunError> {
            let logits = self.prefill_text(id, prompt_tokens)?;
            crate::try_eval(&logits).map_err(|e| DirectEngineError::PrefillEval(e.to_string()))?;
            for width in 2..=max_draft + 1 {
                let tokens = vec![last; width];
                let input = ffi::from_slice_i32(&tokens, &[1, width as i32]);
                let logits = self
                    .engine_mut()
                    .verify(id, &input)
                    .map_err(DirectEngineError::Step)?;
                let argmax = ffi::argmax_last_axis(&logits);
                crate::try_eval(&argmax).map_err(|e| DirectEngineError::Row(e.to_string()))?;
                self.engine_mut()
                    .unwind_appends(id, width as i32)
                    .map_err(DirectEngineError::Step)?;
            }
            Ok(())
        })();
        self.close_sequence(id);
        crate::clear_memory_cache();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn speculate<F: FnMut(i32) -> bool>(
        &mut self,
        id: SequenceId,
        request: &DirectRequest<'_>,
        sampling: &SamplingConfig,
        drafter: &mut dyn Drafter,
        block_size: usize,
        mut on_token: F,
    ) -> Result<SpeculativeRun, SpeculativeRunError> {
        let prompt_tokens = request.prompt_tokens;
        let max_tokens = request.max_tokens;
        // A rejected block is dropped through the pool caches, so they must
        // be trimmable; a family with no external caches has nothing to trim.
        let trimmable = self
            .engine_mut()
            .pool_mut()
            .get_caches_mut(id)
            .is_some_and(|caches| !caches.is_empty() && can_trim_prompt_cache(caches));
        if !trimmable {
            return Err(SpeculativeRunError::NotTrimmable);
        }
        let eos = merged_eos_token_ids(self.model().eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut history = initial_token_history(prompt_tokens, needs_history);
        let mut generated: Vec<i32> = Vec::new();
        let mut sampler = RowSampler::new(sampling);
        let logprobs = LogprobsConfig::default();
        let mut rounds = SpeculativeRounds::default();
        // Pure argmax with nothing that depends on the emitted prefix: every
        // verify position is decided from one batched argmax and a single
        // host read. Anything else goes position by position through the
        // per-row chain.
        let batched_argmax =
            sampling.is_greedy_path() && !needs_history && sampling.token_bias.is_empty();

        let prefill_start = Instant::now();
        let logits = self.prefill_text(id, prompt_tokens)?;
        seed_rng_if_needed(sampling);
        let first = {
            let mut row = StepRow {
                seq_id: id,
                sampler: &mut sampler,
                sampling,
                token_history: &mut history,
                generated: &mut generated,
                eos: &eos,
                max_tokens,
                logprobs: &logprobs,
                needs_mask: false,
                needs_override: false,
                hooks: BareHooks,
            };
            self.engine_mut().complete_prefill(&logits, &mut row)
        };
        if let Some(error) = first.error {
            return Err(DirectEngineError::FirstToken(error.message().to_string()).into());
        }
        drafter.prefill_from_target_hidden(prompt_tokens, &logits, first.token, sampling)?;
        self.prepare_decode(id, max_tokens);
        let _decode_budget = crate::DecodeCommandBufferBudget::enter();
        let prefill_time = prefill_start.elapsed();
        crate::clear_memory_cache();

        let decode_start = Instant::now();
        let mut done = first.finish.is_some();
        if delivers_to_callback(&generated, 0, &first) && !on_token(first.token) {
            done = true;
        }
        while !done {
            let current = *generated
                .last()
                .ok_or_else(|| DirectEngineError::Row("decode without a token".into()))?;
            // Never propose past `max_tokens`: every accepted proposal is
            // emitted, and the round also emits one target token.
            let remaining = max_tokens.saturating_sub(generated.len());
            let budget = block_size.min(remaining.saturating_sub(1));
            let draft = if budget == 0 {
                Vec::new()
            } else {
                let mut draft = drafter.draft_block(current, None, budget, sampling)?;
                draft.truncate(budget);
                draft
            };
            rounds.rounds += 1;

            if draft.is_empty() {
                // A plain step, as `generate` runs it.
                let input = ffi::from_slice_i32(&[current], &[1, 1]);
                let batch = StepBatch {
                    seq_ids: std::slice::from_ref(&id),
                    input: &input,
                };
                let before = generated.len();
                let out = {
                    let row = StepRow {
                        seq_id: id,
                        sampler: &mut sampler,
                        sampling,
                        token_history: &mut history,
                        generated: &mut generated,
                        eos: &eos,
                        max_tokens,
                        logprobs: &logprobs,
                        needs_mask: false,
                        needs_override: false,
                        hooks: BareHooks,
                    };
                    self.engine_mut()
                        .step(&batch, &mut [row])
                        .map_err(DirectEngineError::Step)?
                };
                let outcome = out
                    .rows
                    .into_iter()
                    .next()
                    .ok_or_else(|| DirectEngineError::Row("step returned no row".into()))?;
                if let Some(error) = outcome.error {
                    return Err(DirectEngineError::Row(error.message().to_string()).into());
                }
                let emitted = &generated[before..];
                drafter.accept_verified_tokens(&input, &[], 0, emitted, sampling)?;
                if delivers_to_callback(&generated, before, &outcome) && !on_token(outcome.token) {
                    break;
                }
                done = outcome.finish.is_some();
                continue;
            }

            // Verify `[current, d_0, .., d_{k-1}]` in one forward. Position
            // `i` is the target's choice for the slot `draft[i]` claims;
            // position `k` is the bonus token after a fully accepted block.
            rounds.drafted_rounds += 1;
            rounds.proposed_draft_tokens += draft.len();
            let mut verify_tokens = Vec::with_capacity(draft.len() + 1);
            verify_tokens.push(current);
            verify_tokens.extend_from_slice(&draft);
            let verify_len = verify_tokens.len();
            let input = ffi::from_slice_i32(&verify_tokens, &[1, verify_len as i32]);
            let logits = self
                .engine_mut()
                .verify(id, &input)
                .map_err(DirectEngineError::Step)?;
            let vocab = ffi::array_shape(&logits)[2];
            let greedy_targets = if batched_argmax {
                let argmax = ffi::argmax_last_axis(&logits);
                crate::try_eval(&argmax).map_err(|e| DirectEngineError::Row(e.to_string()))?;
                Some(crate::drafter::dflash::materialize_argmax_i32_vec(
                    &argmax, verify_len,
                ))
            } else {
                None
            };

            let before = generated.len();
            let mut accepted = 0usize;
            let mut stop = false;
            for pos in 0..verify_len {
                let emitted_before = generated.len();
                let outcome = {
                    let mut row = StepRow {
                        seq_id: id,
                        sampler: &mut sampler,
                        sampling,
                        token_history: &mut history,
                        generated: &mut generated,
                        eos: &eos,
                        max_tokens,
                        logprobs: &logprobs,
                        needs_mask: false,
                        needs_override: false,
                        hooks: BareHooks,
                    };
                    match &greedy_targets {
                        Some(targets) => {
                            finish_row(&mut row, targets[pos], targets[pos], None, false)
                        }
                        None => {
                            let pos_logits = ffi::slice(
                                &logits,
                                &[0, pos as i32, 0],
                                &[1, pos as i32 + 1, vocab],
                            );
                            sample_and_finish_row(&pos_logits, &mut row)
                        }
                    }
                };
                if let Some(error) = outcome.error {
                    return Err(DirectEngineError::Row(error.message().to_string()).into());
                }
                if delivers_to_callback(&generated, emitted_before, &outcome)
                    && !on_token(outcome.token)
                {
                    stop = true;
                    break;
                }
                if outcome.finish.is_some() {
                    stop = true;
                    break;
                }
                // The target's token is emitted either way; it only extends
                // the round when it confirms the proposal in the same slot.
                if pos < draft.len() && outcome.token == draft[pos] {
                    accepted += 1;
                } else {
                    break;
                }
            }
            rounds.accepted_draft_tokens += accepted;

            // The forward appended `current, d_0..d_{k-1}`. The state must
            // end holding every emitted token except the next round's
            // current token, i.e. through `d_{a-1}`: keep `a + 1`, drop
            // `k - a`.
            let rejected = (draft.len() - accepted) as i32;
            self.engine_mut()
                .unwind_appends(id, rejected)
                .map_err(DirectEngineError::Step)?;
            self.engine_mut().commit_appends(id, accepted as i32 + 1);
            let emitted = &generated[before..];
            drafter.accept_verified_tokens(&logits, &draft, accepted, emitted, sampling)?;
            if crate::memory::should_clear_cache_at(
                generated.len(),
                crate::memory::cache_clear_interval(),
            ) {
                crate::clear_memory_cache();
            }
            done = stop;
        }
        let decode_time = decode_start.elapsed();

        let prompt_count = prompt_tokens.len();
        let gen_count = generated.len();
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;
        // The first token comes out of the prefill forward; the decode rate
        // counts only what the rounds produced in `decode_time`.
        let decode_count = gen_count.saturating_sub(1);
        let stats = GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                decode_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        };
        Ok(SpeculativeRun {
            run: DirectRun {
                tokens: generated,
                stats,
            },
            rounds,
        })
    }
}
