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

//! An ordinary full model as a token-only [`Drafter`] (#2757): the pairing
//! the classic draft-model loop served until #2754, moved onto the engine's
//! round loop so `mlxcel generate --draft-model`, `mlxcel run`, `mlxcel serve`
//! and `mlxcel-server --draft-model` all drive it the way they drive prompt
//! lookup. The drafter runs the small model autoregressively from its own KV
//! caches, proposes the argmax at every step, and reads nothing from the
//! target: the target verifies every proposal in one
//! [`crate::engine::Engine::verify_round`], so the drafter's numerics reach
//! acceptance only, never the output.
//!
//! ## Ownership
//!
//! The draft model is shared and the per-sequence state is not: on the server
//! one loaded checkpoint serves every row, and each row owns a
//! [`DraftModelState`] (its own caches plus the bookkeeping that rolls them
//! back). [`FullModelDrafter`] is the short-lived view that pairs the two for
//! one call; the CLI builds it around its own model and one state, the
//! scheduler around the worker's model and the row's state.
//!
//! ## Cache invariant
//!
//! The target's engine state holds the prompt and every emitted token except
//! the current one, and the drafter mirrors that: before every
//! [`Drafter::draft_block`] its caches hold a prefix of the context (prompt
//! plus emitted tokens) that stops short of the current token. The call
//! forwards the uncached tail of the context (the current token, plus the
//! last proposal of a fully accepted round, which was never fed) and then
//! each proposal but the last, so a round of `k` proposals costs `k` draft
//! forwards and leaves `d_0 .. d_{k-2}` in the caches. After the target
//! decides, [`Drafter::accept_verified_tokens`] keeps the proposals that are
//! now context and trims the rest ([`DraftModelState::realign`]), which is
//! the classic loop's `rejected - 1` trim written in terms of the context
//! so an early finish or a retracted ask ([`Drafter::retract_draft`]) takes
//! the same path. A trim is only sound when every piece of the draft model's
//! sequence state lives in its external caches, so a draft model that fails
//! [`prompt_lookup_unsupported_reason`] is refused at construction, the
//! same rule the target is held to by `Engine::verify_rounds_unsupported`
//! (#2495).

use super::prompt_lookup::prompt_lookup_unsupported_reason;
use crate::cache::can_trim_prompt_cache;
use crate::drafter::{Drafter, DrafterError, DrafterKind, TokenOnlyCounts, TokenOnlyDrafter};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, ffi};

/// Why a model cannot serve as a full-model drafter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FullModelDrafterError {
    /// The draft model keeps sequence state a cache trim cannot roll back
    /// (the same rule [`prompt_lookup_unsupported_reason`] applies to a
    /// target).
    #[error("the draft model cannot roll a rejected block back: {0}")]
    Unsupported(&'static str),
}

/// Acceptance accounting for one sequence of a full-model drafter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FullModelDraftStats {
    /// `draft_block` calls that stood (not retracted).
    pub rounds: usize,
    /// Rounds that proposed at least one token.
    pub drafted_rounds: usize,
    pub proposed_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    /// Draft-model forwards run, prefill chunks included.
    pub draft_forwards: usize,
    /// Proposals that followed a draft-model EOS inside a block (the chain
    /// cannot stop at one without a host read; see `draft_block`).
    pub eos_truncated_proposals: usize,
}

impl FullModelDraftStats {
    /// One-line summary for the CLI, after `generated` tokens.
    #[must_use]
    pub fn summary_line(&self, generated: usize) -> String {
        let rate = if self.proposed_draft_tokens > 0 {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64
        } else {
            0.0
        };
        format!(
            "[full-model drafter: {generated} tokens in {} rounds ({} drafted), {}/{} proposals accepted ({rate:.3}), {} draft forwards]",
            self.rounds,
            self.drafted_rounds,
            self.accepted_draft_tokens,
            self.proposed_draft_tokens,
            self.draft_forwards
        )
    }
}

/// One sequence's state of a full-model drafter: the draft model's own KV
/// caches and the bookkeeping that keeps them aligned with the context (see
/// the module docs).
pub struct DraftModelState {
    caches: Vec<KVCache>,
    /// Draft-model EOS ids: a proposal that is one ends the block.
    eos: Vec<i32>,
    /// Prompt followed by every emitted token.
    context: Vec<i32>,
    /// How many leading tokens of `context` the caches hold.
    held: usize,
    /// Proposals fed after `context[..held]`, still in the caches.
    held_drafts: Vec<i32>,
    stats: FullModelDraftStats,
    /// Whether the last `draft_block` proposed anything, for `retract_draft`.
    last_drafted: bool,
}

impl std::fmt::Debug for DraftModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftModelState")
            .field("layers", &self.caches.len())
            .field("context", &self.context.len())
            .field("held", &self.held)
            .field("held_drafts", &self.held_drafts)
            .field("stats", &self.stats)
            .finish()
    }
}

impl DraftModelState {
    /// Fresh state for `model`, or the reason it cannot draft (#2495).
    pub fn new<M: LanguageModel + ?Sized>(model: &M) -> Result<Self, FullModelDrafterError> {
        if let Some(reason) = prompt_lookup_unsupported_reason(model) {
            return Err(FullModelDrafterError::Unsupported(reason));
        }
        let caches = model.make_caches();
        if caches.is_empty() || !can_trim_prompt_cache(&caches) {
            return Err(FullModelDrafterError::Unsupported(
                "the draft model's KV caches are not trimmable",
            ));
        }
        Ok(Self {
            caches,
            eos: model.eos_token_ids(),
            context: Vec::new(),
            held: 0,
            held_drafts: Vec::new(),
            stats: FullModelDraftStats::default(),
            last_drafted: false,
        })
    }

    /// Acceptance accounting so far.
    #[must_use]
    pub fn stats(&self) -> FullModelDraftStats {
        self.stats
    }

    /// Positions the caches hold (every layer agrees, or the trim failed).
    #[must_use]
    pub fn cached_positions(&self) -> usize {
        self.held + self.held_drafts.len()
    }

    /// Extend the context with tokens plain steps emitted
    /// ([`TokenOnlyDrafter::observe_emitted`]).
    fn observe(&mut self, tokens: &[i32]) {
        self.context.extend_from_slice(tokens);
        self.realign();
    }

    /// Bring the caches back to a prefix of the context: keep the fed
    /// proposals the context now confirms, trim the rest.
    fn realign(&mut self) {
        let confirmed = self
            .held_drafts
            .iter()
            .zip(&self.context[self.held.min(self.context.len())..])
            .take_while(|(d, c)| d == c)
            .count();
        let drop = self.held_drafts.len() - confirmed;
        if drop > 0 {
            trim_caches(&mut self.caches, drop);
        }
        self.held += confirmed;
        self.held_drafts.clear();
    }

    fn clear(&mut self, caches: Vec<KVCache>) {
        self.caches = caches;
        self.context.clear();
        self.held = 0;
        self.held_drafts.clear();
        self.stats = FullModelDraftStats::default();
        self.last_drafted = false;
    }
}

/// Trim `n` positions from every cache; a layer that trims fewer leaves the
/// state inconsistent, which the next forward would surface as a shape
/// error, so it is logged here where the cause is visible.
fn trim_caches(caches: &mut [KVCache], n: usize) {
    for (layer, cache) in caches.iter_mut().enumerate() {
        let trimmed = cache.trim(n as i32);
        if trimmed != n as i32 {
            tracing::warn!(
                layer,
                requested = n,
                trimmed,
                "full-model drafter: short trim"
            );
        }
    }
}

/// A draft model paired with one sequence's [`DraftModelState`] for the
/// duration of a round-loop call (see the module docs).
pub struct FullModelDrafter<'a, M: LanguageModel + ?Sized> {
    model: &'a M,
    state: &'a mut DraftModelState,
}

impl<'a, M: LanguageModel + ?Sized> FullModelDrafter<'a, M> {
    /// Pair `model` with `state`, which [`DraftModelState::new`] built for it.
    pub fn new(model: &'a M, state: &'a mut DraftModelState) -> Self {
        Self { model, state }
    }

    /// Acceptance accounting so far.
    #[must_use]
    pub fn stats(&self) -> FullModelDraftStats {
        self.state.stats
    }

    /// One draft forward of `input` (`[1, n]`, possibly still unevaluated),
    /// continuing from the caches; returns the lazy `[1, 1]` argmax at the
    /// last position, which is the next forward's input as it stands.
    fn step(
        &mut self,
        input: &MlxArray,
        last_pos: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let logits = self
            .model
            .try_forward_last_logits_with_sequence_id(
                input,
                None,
                &mut self.state.caches,
                None,
                last_pos,
            )
            .map_err(|reason| DrafterError::DraftFailed { reason })?;
        self.state.stats.draft_forwards += 1;
        Ok(ffi::argmax_last_axis(&logits))
    }

    /// Forward the prompt in chunks, evaluating only the cache state between
    /// them so the LM head never runs over a chunk.
    fn prefill(&mut self, prompt_tokens: &[i32]) -> Result<(), DrafterError> {
        let chunk_len = crate::prefill_plan::prefill_chunk_len().max(1);
        for chunk in prompt_tokens.chunks(chunk_len) {
            let input = ffi::from_slice_i32(chunk, &[1, chunk.len() as i32]);
            let _logits = self
                .model
                .try_forward_with_sequence_id(&input, None, &mut self.state.caches, None)
                .map_err(|reason| DrafterError::DraftFailed { reason })?;
            self.state.stats.draft_forwards += 1;
            for cache in &self.state.caches {
                cache.eval_state();
            }
            self.state.held += chunk.len();
            crate::clear_memory_cache();
        }
        Ok(())
    }
}

impl<M: LanguageModel + ?Sized> Drafter for FullModelDrafter<'_, M> {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn drafts_from_tokens_only(&self) -> bool {
        true
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        _hidden: &MlxArray,
        first_bonus: i32,
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.state.clear(self.model.make_caches());
        self.state.context.extend_from_slice(prompt_tokens);
        self.prefill(prompt_tokens)?;
        self.state.context.push(first_bonus);
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        debug_assert_eq!(
            self.state.context.last().copied(),
            Some(last_bonus),
            "the loop reports every emitted token before it asks for a proposal"
        );
        debug_assert!(
            self.state.held_drafts.is_empty(),
            "a round's proposals are realigned before the next ask"
        );
        self.state.stats.rounds += 1;
        self.state.last_drafted = false;
        if block_size == 0 {
            return Ok(Vec::new());
        }
        // The uncached tail of the context: the current token, preceded by
        // the last proposal of a fully accepted round. Never empty while the
        // invariant holds; re-feed the current token if it somehow is.
        if self.state.held >= self.state.context.len() {
            let over = self.state.held - self.state.context.len() + 1;
            trim_caches(&mut self.state.caches, over);
            self.state.held -= over;
        }
        let tail: Vec<i32> = self.state.context[self.state.held..].to_vec();
        // The block is one lazy graph: each proposal is the previous
        // forward's argmax fed back as a `[1, 1]` device array, so the round
        // reads the host once, after the last forward, instead of once per
        // proposal. A proposal that is a draft-model EOS is therefore not
        // seen in time to end the block; the ones after it are wasted verify
        // positions, and the target decides either way.
        let input = ffi::from_slice_i32(&tail, &[1, tail.len() as i32]);
        let mut proposals: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(block_size);
        let mut next = self.step(&input, tail.len() - 1)?;
        self.state.held = self.state.context.len();
        while proposals.len() + 1 < block_size {
            let fed = self.step(&next, 0)?;
            proposals.push(next);
            next = fed;
        }
        proposals.push(next);
        let mut draft = Vec::with_capacity(block_size);
        for proposal in &proposals {
            crate::try_eval(proposal).map_err(|e| DrafterError::DraftFailed {
                reason: e.to_string(),
            })?;
            draft.push(ffi::item_i32(proposal));
        }
        // Every proposal but the last was fed, so it is in the caches.
        self.state
            .held_drafts
            .extend_from_slice(&draft[..draft.len() - 1]);
        if let Some(eos_at) = draft.iter().position(|t| self.state.eos.contains(t)) {
            self.state.stats.eos_truncated_proposals += draft.len() - eos_at - 1;
        }
        self.state.stats.drafted_rounds += 1;
        self.state.stats.proposed_draft_tokens += draft.len();
        self.state.last_drafted = true;
        Ok(draft)
    }

    fn pipelines_plain_rounds(&self) -> bool {
        false
    }

    fn retract_draft(&mut self, draft: &[i32]) {
        self.state.stats.rounds = self.state.stats.rounds.saturating_sub(1);
        if self.state.last_drafted {
            self.state.stats.drafted_rounds = self.state.stats.drafted_rounds.saturating_sub(1);
            self.state.stats.proposed_draft_tokens = self
                .state
                .stats
                .proposed_draft_tokens
                .saturating_sub(draft.len());
            self.state.last_drafted = false;
        }
        // The context did not move, so every fed proposal is dropped.
        self.state.realign();
    }

    fn accept_verified_tokens(
        &mut self,
        _verify_hidden: &MlxArray,
        draft_tokens: &[i32],
        accepted: usize,
        new_tokens: &[i32],
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        if !draft_tokens.is_empty() {
            self.state.stats.accepted_draft_tokens += accepted;
        }
        self.state.observe(new_tokens);
        Ok(())
    }

    fn reset(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.state.clear(self.model.make_caches());
        Ok(())
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::FullModel
    }
}

impl<M: LanguageModel + ?Sized> TokenOnlyDrafter for FullModelDrafter<'_, M> {
    fn observe_emitted(&mut self, tokens: &[i32]) {
        self.state.observe(tokens);
    }

    fn counts(&self) -> TokenOnlyCounts {
        let stats = self.state.stats;
        TokenOnlyCounts {
            rounds: stats.rounds,
            drafted_rounds: stats.drafted_rounds,
            proposed_draft_tokens: stats.proposed_draft_tokens,
            accepted_draft_tokens: stats.accepted_draft_tokens,
        }
    }
}

#[cfg(test)]
#[path = "full_model_drafter_tests.rs"]
mod tests;
