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

//! Prompt-lookup speculative decoding inside the batch scheduler (#2255).
//!
//! Under [`crate::server::SpeculativeDispatch::PromptLookup`] each eligible
//! sequence carries a [`PromptLookupDrafter`] on its [`SequenceInfo`]
//! ([`PromptLookupRow`]), primed at prefill completion
//! ([`BatchScheduler::prime_prompt_lookup`]). A decode tick asks every
//! gated-in row for a proposal on its committed tokens; a row that proposes
//! runs one [`mlxcel_core::engine::Engine::verify_round`] on its own (the
//! round `mlxcel generate --prompt-lookup` runs), and every other row takes
//! the regular batched step. There is no run-to-completion burst: at most one
//! verify round per tick per row, so concurrent rows keep getting a token
//! every tick. An ineligible request (multimodal, structured output,
//! logprobs, a feedback or extended sampler chain, a model whose state a
//! trim cannot roll back) decodes plainly, with a `debug` log naming why.
//!
//! ## Lookahead
//!
//! Proposal-free ticks stay on the lookahead pipeline, mirroring the
//! drafter loop's `pipelined_plain_round` at batch level:
//!
//! 1. With a step n in flight ([`DecodeLookahead`]) and before the step n+1
//!    prime, every gated-in row is asked for a proposal on its committed
//!    tokens.
//! 2. If none proposes, the tick pipelines exactly as without the flag.
//! 3. If any proposes, no prime is submitted: step n is read and committed,
//!    every asked proposal is retracted ([`Drafter::retract_draft`], no
//!    forward followed it), and the next tick runs synchronously, where the
//!    proposing rows verify from their newly committed token and the rest
//!    take the synchronous batched step.
//! 4. A prime after a synchronous tick is submitted only when no row
//!    proposed on that tick and every gated-in row's drafter reports
//!    [`Drafter::pipelines_plain_rounds`].
//!
//! ## Invariants
//!
//! - **I1, synchronous state.** With no lookahead stored, every decoding
//!   row's state holds its prompt and every emitted token but the last. A
//!   verify round appends `k + 1` positions and, before it returns, commits
//!   the accepted `a` proposals plus the current token (`a` on an EOS) and
//!   unwinds the other `k - a`; a round that fails at a position unwinds all
//!   `k + 1` and the row finishes with `FinishReason::Error` (#822).
//! - **I2, one append in flight.** While `decode_lookahead` holds step n,
//!   each of its rows has exactly one uncommitted append (the forward fed by
//!   its last emitted token). Only the steady commit
//!   ([`BatchScheduler::pipelined_steady_decode`],
//!   [`BatchScheduler::commit_lookahead_without_prime`]) commits it, and only
//!   [`BatchScheduler::discard_lookahead`] /
//!   [`BatchScheduler::apply_lookahead_trim`] unwind it. A verify round never
//!   runs while a lookahead is stored: the prompt-lookup tick commits or
//!   unwinds it first.
//! - **I3, no prime past a proposal.** A step n+1 prime is submitted only
//!   after every gated-in row was asked this tick and none proposed.
//! - **I4, drafter context.** A drafter's context is the prompt plus
//!   `generated_tokens[..observed]`; plain-step tokens are reported lazily
//!   ([`PromptLookupDrafter::observe_emitted`]) before the next
//!   [`Drafter::draft_block`], verify-round tokens through
//!   [`Drafter::accept_verified_tokens`]. Every `draft_block` call is followed
//!   by the forward it was asked for (a verify round, or a plain step fed by
//!   the same token) or by a `retract_draft`.
//! - **I5, the verify gate.** A row is asked only when the tick's active
//!   decode batch is at most `--prompt-lookup-max-batch`, no unified KV
//!   budget is set (verify rounds commit several tokens per tick, as the
//!   speculative bursts that also decline it), the row has room for a
//!   proposal under `max_tokens` and under the context-bound stop (a
//!   proposal is capped so the verify forward never appends past the
//!   positions the plain step that ends the row would hold), and the paged
//!   context limit
//!   ([`PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT`]) does not engage. A row that is
//!   not asked calls no `draft_block`, so its governor records no miss, and
//!   it keeps observing committed tokens, so it resumes when the batch
//!   shrinks. With nobody gated in, the tick is the plain tick.
//! - **I6, membership changes.** Admission, preemption, block reclaim and
//!   completion tear a stored lookahead down through `discard_lookahead`
//!   (one position) exactly as without the flag, before anything else
//!   touches the caches. A request admitted while a lookahead is in flight is
//!   primed at its own prefill completion, after that teardown, and is first
//!   asked on a later tick. Preemption drops the row's drafter; the
//!   re-prefill primes a fresh one.
//! - **I7, errors.** A verify forward that cannot run, or a position that
//!   fails, finishes only that row with an error after unwinding its
//!   appends; a rewind failure finishes it through
//!   `fail_desynchronized_sequence`, as a failed lookahead teardown does. A
//!   lookahead read that throws (#822) unwinds the in-flight append and the
//!   tick decodes synchronously, which fails the affected rows cleanly.
//! - **I8, cancellation.** A cancelled row is not interrupted mid-round; the
//!   round leaves its state consistent (I1) and `finalize_completed` finishes
//!   it after the tick.
//!
//! ## Verify-width warmup
//!
//! The first eligible prefill on a scheduler (normally the server's startup
//! warmup request, before its finish at prefill returns) runs one verify
//! forward at every width a round can use and unwinds each
//! ([`BatchScheduler::warm_up_prompt_lookup_widths`]), as `mlxcel generate
//! --prompt-lookup` does, so first-use kernel costs do not land mid-decode.
//!
//! ## Paged storage
//!
//! A multi-token verify forward cannot take the paged single-token kernel
//! and goes through the pool intercept and gather, which ADR 0001 measured at
//! two to three times contiguous SDPA past 4096 tokens at batch 4. Whether
//! that erases the gain is measured once (#2255 measurement (d)); until then
//! [`PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT`] is `None` and no limit applies.

use mlxcel_core::drafter::Drafter;
use mlxcel_core::engine::EngineError;
use mlxcel_core::speculative::prompt_lookup::PromptLookupConfig;
use mlxcel_core::speculative::prompt_lookup_drafter::PromptLookupDrafter;

use super::*;
use crate::server::model_provider::SpeculativeStats;

/// Context length past which a sequence on paged storage stops running
/// verify rounds and decodes plainly.
///
/// PROVISIONAL (#2255): `None`, no limit, pending measurement (d), the verify
/// cost at 8192 tokens on `--decode-storage-backend paged`. The measurement
/// fills in either a length (with its numbers cited here) or a note that no
/// limit applies and why.
pub(crate) const PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT: Option<usize> = None;

/// A sequence's prompt-lookup state, carried on [`SequenceInfo`] across
/// ticks (see the module docs).
#[derive(Debug, Clone)]
pub(crate) struct PromptLookupRow {
    pub(crate) drafter: PromptLookupDrafter,
    /// Generated tokens the drafter has been told about (I4).
    pub(crate) observed: usize,
}

impl PromptLookupRow {
    /// The acceptance counters a finished request reports, `None` when no
    /// verify round ran.
    pub(crate) fn speculative_stats(&self) -> Option<SpeculativeStats> {
        let stats = self.drafter.stats();
        SpeculativeStats::from_counts(
            mlxcel_core::drafter::DrafterKind::PromptLookup,
            stats.drafted_rounds,
            stats.proposed_draft_tokens,
            stats.accepted_draft_tokens,
        )
    }
}

/// Proposals a row may verify this tick: never past `max_tokens`, since
/// every accepted proposal is emitted and the round emits one target token
/// too, and never past the context-bound stop
/// ([`ContextBound::verify_room`]), so a prompt that ends just under
/// `--max-kv-size` cannot make the verify forward append positions beyond
/// what the plain step that ends the row would hold.
pub(super) fn proposal_budget(
    seq: &SequenceInfo,
    config: &PromptLookupConfig,
    context: ContextBound,
) -> usize {
    let remaining = seq.max_tokens.saturating_sub(seq.generated_tokens.len());
    let budget = config.max_draft.min(remaining.saturating_sub(1));
    match context.verify_room(
        seq.prompt_tokens.len(),
        seq.generated_tokens.len(),
        seq.vlm_embeddings.is_some(),
    ) {
        Some(room) => budget.min(room),
        None => budget,
    }
}

impl BatchScheduler {
    /// The prompt-lookup drafter config when this tick's rows may be asked
    /// for proposals at all (I5): the dispatch is prompt lookup, the active
    /// decode batch is at most `--prompt-lookup-max-batch`, and no unified
    /// KV budget is set.
    fn prompt_lookup_gate(&self, seq_ids: &[SequenceId]) -> Option<PromptLookupConfig> {
        let (config, max_batch) = self.speculative_dispatch.prompt_lookup_config()?;
        (seq_ids.len() <= max_batch && self.shared_kv_budget().is_none()).then_some(*config)
    }

    /// Whether `seq` is gated in this tick (I5), given an open gate.
    fn prompt_lookup_askable(&self, seq: &SequenceInfo, config: &PromptLookupConfig) -> bool {
        if seq.prompt_lookup.is_none()
            || seq.state.is_finished()
            || proposal_budget(seq, config, self.context_bound()) == 0
        {
            return false;
        }
        let paged_limit_engaged = PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT.is_some_and(|limit| {
            seq.prompt_tokens.len() + seq.generated_tokens.len() > limit
                && self
                    .engine
                    .pool()
                    .get(seq.seq_id)
                    .is_some_and(|set| set.backend == SequenceStateBackend::PagedKvCache)
        });
        !paged_limit_engaged
    }

    /// Whether some row of this tick is gated in, which routes the tick
    /// through [`Self::run_prompt_lookup_tick`]; otherwise it is the plain
    /// tick.
    pub(super) fn prompt_lookup_tick_applies(&self, seq_ids: &[SequenceId]) -> bool {
        let Some(config) = self.prompt_lookup_gate(seq_ids) else {
            return false;
        };
        seq_ids.iter().any(|&id| {
            self.active_batch
                .get(id)
                .is_some_and(|seq| self.prompt_lookup_askable(seq, &config))
        })
    }

    /// Rule 4: a gated-in row whose drafter expects a proposal soon keeps the
    /// next tick synchronous, so the proposal is verified without waiting a
    /// step. Read by [`Self::maybe_prime_lookahead`].
    pub(super) fn prompt_lookup_blocks_prime(&self, seq_ids: &[SequenceId]) -> bool {
        let Some(config) = self.prompt_lookup_gate(seq_ids) else {
            return false;
        };
        seq_ids.iter().any(|&id| {
            self.active_batch.get(id).is_some_and(|seq| {
                self.prompt_lookup_askable(seq, &config)
                    && seq
                        .prompt_lookup
                        .as_ref()
                        .is_some_and(|row| !row.drafter.pipelines_plain_rounds())
            })
        })
    }

    /// Ask every gated-in row for a proposal on its committed tokens (rule
    /// 1), after telling its drafter about the tokens plain steps emitted
    /// since it last looked (I4). Returns `(row, proposal)` for every row
    /// asked, in `seq_ids` order; an empty proposal is a row that decodes
    /// plainly. A drafter that fails is dropped and its row decodes plainly
    /// from then on.
    fn ask_prompt_lookup(&mut self, seq_ids: &[SequenceId]) -> Vec<(SequenceId, Vec<i32>)> {
        let Some(config) = self.prompt_lookup_gate(seq_ids) else {
            return Vec::new();
        };
        let context = self.context_bound();
        let mut asked = Vec::new();
        for &id in seq_ids {
            let askable = self
                .active_batch
                .get(id)
                .is_some_and(|seq| self.prompt_lookup_askable(seq, &config));
            if !askable {
                continue;
            }
            let Some(seq) = self.active_batch.get_mut(id) else {
                continue;
            };
            let budget = proposal_budget(seq, &config, context);
            let SequenceInfo {
                generated_tokens,
                prompt_lookup,
                sampling,
                ..
            } = seq;
            let (Some(row), Some(&current)) = (prompt_lookup.as_mut(), generated_tokens.last())
            else {
                continue;
            };
            // I4 keeps `observed <= generated_tokens.len()`; a row that ever
            // breaks it decodes plainly instead of panicking the scheduler
            // thread on the slice.
            let Some(unseen) = generated_tokens.get(row.observed..) else {
                tracing::debug!(seq_id = %id, observed = row.observed, generated = generated_tokens.len(), "prompt lookup: drafter is ahead of the row, decoding plainly");
                *prompt_lookup = None;
                continue;
            };
            row.drafter.observe_emitted(unseen);
            row.observed = generated_tokens.len();
            match row.drafter.draft_block(current, None, budget, sampling) {
                Ok(mut draft) => {
                    draft.truncate(budget);
                    asked.push((id, draft));
                }
                Err(err) => {
                    tracing::debug!(seq_id = %id, error = %err, "prompt lookup: drafter failed, decoding plainly");
                    *prompt_lookup = None;
                }
            }
        }
        asked
    }

    /// Retract every asked proposal: no forward followed the ask (rule 3).
    fn retract_prompt_lookup(&mut self, asked: &[(SequenceId, Vec<i32>)]) {
        for (id, draft) in asked {
            if let Some(row) = self
                .active_batch
                .get_mut(*id)
                .and_then(|seq| seq.prompt_lookup.as_mut())
            {
                row.drafter.retract_draft(draft);
            }
        }
    }

    /// One decode tick with prompt-lookup rows gated in (see the module
    /// docs for the switch rule and the invariants it keeps).
    pub(super) fn run_prompt_lookup_tick(&mut self, seq_ids: &[SequenceId]) {
        let params = self.lookahead_params(seq_ids);
        match self.decode_lookahead.take() {
            Some(la) if la.ids == seq_ids && params.is_some() && self.lookahead_safe() => {
                let asked = self.ask_prompt_lookup(seq_ids);
                if asked.iter().all(|(_, draft)| draft.is_empty()) {
                    // Rule 2: nobody proposes, so the tick pipelines exactly
                    // as without the flag. The empty asks are consumed by the
                    // step their tokens fed.
                    let _decode_budget = mlxcel_core::DecodeCommandBufferBudget::enter();
                    if let Some(params) = params {
                        self.pipelined_steady_decode(la, seq_ids, &params);
                    }
                    return;
                }
                // Rule 3: no prime; step n is committed and the next tick is
                // synchronous.
                self.retract_prompt_lookup(&asked);
                self.commit_lookahead_without_prime(la, seq_ids);
            }
            Some(la) => {
                // Stale id set or no longer eligible / safe: unwind the one
                // in-flight append (I2) and decode synchronously.
                let failed =
                    self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
                drop(la);
                let live: Vec<SequenceId> = seq_ids
                    .iter()
                    .copied()
                    .filter(|id| !failed.contains(id))
                    .collect();
                if !live.is_empty() {
                    self.prompt_lookup_sync_tick(&live);
                }
            }
            None => self.prompt_lookup_sync_tick(seq_ids),
        }
    }

    /// A synchronous prompt-lookup tick: proposing rows verify on their own,
    /// every other row takes the batched step, and the next step is primed
    /// only when nobody proposed (rule 4, through
    /// [`Self::maybe_prime_lookahead`]).
    fn prompt_lookup_sync_tick(&mut self, seq_ids: &[SequenceId]) {
        let asked = self.ask_prompt_lookup(seq_ids);
        let proposing: Vec<(SequenceId, Vec<i32>)> = asked
            .into_iter()
            .filter(|(_, draft)| !draft.is_empty())
            .collect();
        let plain: Vec<SequenceId> = seq_ids
            .iter()
            .copied()
            .filter(|id| proposing.iter().all(|(p, _)| p != id))
            .collect();
        if !plain.is_empty() {
            self.dispatch_sync_decode(&plain);
        }
        let verified = !proposing.is_empty();
        for (id, draft) in proposing {
            self.run_prompt_lookup_round(id, &draft);
        }
        if !verified {
            let _decode_budget = mlxcel_core::DecodeCommandBufferBudget::enter();
            self.maybe_prime_lookahead(seq_ids);
        }
    }

    /// Rule 3's read: commit the in-flight step n without submitting step
    /// n+1. The tokens are evaluated through the fallible boundary first
    /// (#822); a throw, or a readback of the wrong length, unwinds the
    /// in-flight append and decodes the tick synchronously (I7).
    pub(super) fn commit_lookahead_without_prime(
        &mut self,
        la: DecodeLookahead,
        seq_ids: &[SequenceId],
    ) {
        let toks = match mlxcel_core::try_eval(&la.tokens) {
            Ok(()) => mlxcel_core::engine::tokens_to_host(&la.tokens),
            Err(err) => {
                let _ = self.record_eval_outcome(Err(err.to_string()));
                Vec::new()
            }
        };
        if toks.len() != seq_ids.len() {
            let failed = self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
            drop(la);
            let live: Vec<SequenceId> = seq_ids
                .iter()
                .copied()
                .filter(|id| !failed.contains(id))
                .collect();
            if !live.is_empty() {
                self.dispatch_sync_decode(&live);
            }
            return;
        }
        drop(la);
        // The collect half of the step, as a synchronous step finishes it: a
        // row whose token finishes it (EOS, a stop, `max_tokens`) finishes
        // here and leaves the state a synchronous step leaves.
        self.apply_fused_decode_tokens(seq_ids, &toks);
        self.batch_observability.record_lookahead_step();
    }

    /// One verify round for row `id` on `draft` (see the module docs).
    fn run_prompt_lookup_round(&mut self, id: SequenceId, draft: &[i32]) {
        // A --max-kv-size trim runs before the verify forward, as before a
        // step.
        let retention = self
            .active_batch
            .get(id)
            .map(|seq| seq.retention)
            .unwrap_or_default();
        self.enforce_max_kv_size_for(id, retention);
        // The decode reservation covers one position per row; a verify block
        // that would need a paged block the budget cannot give decodes the
        // row plainly instead of reclaiming for speculative work.
        let width = draft.len() + 1;
        let fits = self
            .available_paged_blocks()
            .is_none_or(|free| free >= self.engine.pool().paged_blocks_to_append(id, width));
        if !fits {
            self.retract_prompt_lookup(&[(id, draft.to_vec())]);
            self.decode_single_step(id);
            return;
        }
        let context = self.context_bound();
        let round = {
            let Some(seq) = self.active_batch.get_mut(id) else {
                return;
            };
            let Some(&current) = seq.generated_tokens.last() else {
                return;
            };
            let tokenizer = &self.tokenizer;
            let mut row = step_rows::step_row(seq, tokenizer, context, false);
            self.engine.verify_round(&mut row, current, draft, |_| true)
        };
        let round = match round {
            Ok(round) => round,
            Err(EngineError::Rewind(msg)) => {
                self.fail_desynchronized_sequence(id, &msg);
                return;
            }
            Err(err) => {
                Self::abort_sequence_with_error(
                    self.active_batch.get_mut(id),
                    "inference backend",
                    &err.to_string(),
                );
                return;
            }
        };
        self.finish_rows_state(&round.outcomes);
        if !self.apply_row_outcomes(&round.outcomes) || round.error().is_some() {
            return;
        }
        let Some(seq) = self.active_batch.get_mut(id) else {
            return;
        };
        let SequenceInfo {
            generated_tokens,
            prompt_lookup,
            sampling,
            ..
        } = seq;
        if let Some(row) = prompt_lookup.as_mut() {
            let emitted = &generated_tokens[row.observed.min(generated_tokens.len())..];
            if let Err(err) = row.drafter.accept_verified_tokens(
                &round.logits,
                draft,
                round.accepted,
                emitted,
                sampling,
            ) {
                tracing::debug!(seq_id = %id, error = %err, "prompt lookup: drafter failed, decoding plainly");
                *prompt_lookup = None;
                return;
            }
            row.observed = generated_tokens.len();
        }
    }

    /// Run one verify forward at every width a prompt-lookup round can use
    /// (2 through `max_draft + 1`) on `seq`'s freshly prefilled state and
    /// unwind each, once per scheduler, as `mlxcel generate --prompt-lookup`
    /// does before it decodes: the first forward at a width builds or traces
    /// its kernels, which on GB10 costs more than the rest of a short reply,
    /// and would otherwise land in the middle of the first requests that use
    /// each width. Runs on the first text-only prefill that could run verify
    /// rounds at all; the server's startup warmup request is one. The state
    /// is left as the prefill left it (the prompt, not the first token). A
    /// failure is logged and not retried: the request decodes as it would
    /// have, and its own rounds fail it cleanly if the backend is broken.
    pub(super) fn warm_up_prompt_lookup_widths(&mut self, seq: &SequenceInfo) {
        if self.prompt_lookup_widths_warmed {
            return;
        }
        let Some((config, _)) = self.speculative_dispatch.prompt_lookup_config() else {
            return;
        };
        let max_draft = config.max_draft;
        let drafter = PromptLookupDrafter::new(*config);
        if seq.vlm_embeddings.is_some()
            || !seq.images.is_empty()
            || !seq.audio.is_empty()
            || self.shared_kv_budget().is_some()
            || self
                .engine
                .verify_rounds_unsupported(seq.seq_id, &seq.sampling, &drafter)
                .is_some()
        {
            return;
        }
        let Some(&first) = seq.generated_tokens.last() else {
            return;
        };
        // A prompt that ends near the context-bound stop cannot take the
        // widest block (`proposal_budget` would never forward it); leave the
        // warmup to a later, shorter prefill.
        if self
            .context_bound()
            .verify_room(seq.prompt_tokens.len(), seq.generated_tokens.len(), false)
            .is_some_and(|room| room < max_draft)
        {
            return;
        }
        let widest = max_draft + 1;
        let fits = self.available_paged_blocks().is_none_or(|free| {
            free >= self
                .engine
                .pool()
                .paged_blocks_to_append(seq.seq_id, widest)
        });
        if !fits {
            return;
        }
        self.prompt_lookup_widths_warmed = true;
        match self
            .engine
            .warm_up_verify_widths(seq.seq_id, first, max_draft)
        {
            Ok(()) => tracing::debug!(max_draft, "prompt lookup: verify widths warmed up"),
            Err(err) => {
                tracing::warn!(seq_id = %seq.seq_id, error = %err, "prompt lookup: verify-width warmup failed")
            }
        }
    }

    /// Give `seq` its prompt-lookup drafter at prefill completion when the
    /// dispatch offers prompt lookup and the request is eligible; an
    /// ineligible request decodes plainly with a `debug` log naming why.
    /// `logits` are the prefill's last-position logits, in the hidden-state
    /// slot the token-only drafter ignores.
    pub(super) fn prime_prompt_lookup(
        &mut self,
        seq: &mut SequenceInfo,
        logits: &mlxcel_core::MlxArray,
    ) {
        seq.prompt_lookup = None;
        let Some((config, _)) = self.speculative_dispatch.prompt_lookup_config() else {
            return;
        };
        let mut drafter = PromptLookupDrafter::new(*config);
        let reason: Option<String> =
            if seq.vlm_embeddings.is_some() || !seq.images.is_empty() || !seq.audio.is_empty() {
                Some("multimodal payload attached".to_string())
            } else if seq.structured.is_some() {
                Some("structured-output constraint attached".to_string())
            } else if seq.logprobs_config.enabled {
                Some("per-token logprobs requested".to_string())
            } else if seq.sampling.needs_extended_chain() {
                Some("the sampler runs the extended chain".to_string())
            } else {
                self.engine
                    .verify_rounds_unsupported(seq.seq_id, &seq.sampling, &drafter)
                    .map(|err| err.to_string())
            };
        if let Some(reason) = reason {
            tracing::debug!(seq_id = %seq.seq_id, "prompt lookup declined: {reason}; decoding plainly");
            return;
        }
        let Some(&first) = seq.generated_tokens.last() else {
            return;
        };
        if let Err(err) =
            drafter.prefill_from_target_hidden(&seq.prompt_tokens, logits, first, &seq.sampling)
        {
            tracing::debug!(seq_id = %seq.seq_id, error = %err, "prompt lookup declined: drafter prefill failed");
            return;
        }
        seq.prompt_lookup = Some(Box::new(PromptLookupRow {
            drafter,
            observed: seq.generated_tokens.len(),
        }));
    }
}
