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
//! forwards the uncached tail of the context and then each proposal but the
//! last, so a round of `k` proposals costs `k` draft forwards and leaves
//! `d_0 .. d_{k-2}` in the caches. The tail is normally the current token,
//! plus the last proposal of a fully accepted round, which was never fed.
//! It is the whole context on the first ask, because the prefill only records
//! the prompt, and every token plain steps emitted on the ask after a row was
//! gated out; a tail longer than one prefill chunk is forwarded in chunks. A
//! row the scheduler never asks therefore runs no draft forward and holds no
//! draft KV, however large the batch it sits in. After the target
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
    /// Size of the draft model's embedding table when known
    /// ([`Self::with_input_vocab`]).
    input_vocab: Option<usize>,
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
            input_vocab: None,
        })
    }

    /// Declare the draft model's vocabulary size. The target's context can
    /// hold ids past a smaller draft table (a pairing that passes
    /// [`crate::drafter::full_model_vocab_problem`]). Such an id cannot be
    /// embedded correctly, so it is fed as id 0: the position stays aligned,
    /// and the drafter's numerics reach acceptance only.
    #[must_use]
    pub fn with_input_vocab(mut self, vocab: Option<usize>) -> Self {
        self.input_vocab = vocab;
        self
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

    /// Positions the first cache layer really holds, for checking
    /// [`Self::cached_positions`] against the caches themselves.
    #[must_use]
    pub fn cache_seq_len(&self) -> usize {
        self.caches
            .first()
            .map_or(0, |cache| cache.seq_len().max(0) as usize)
    }

    /// The draft-model input for `tokens`; see [`Self::with_input_vocab`].
    fn input_array(&self, tokens: &[i32]) -> UniquePtr<MlxArray> {
        let shape = [1, tokens.len() as i32];
        match self.input_vocab {
            Some(vocab) if tokens.iter().any(|&t| t < 0 || t as usize >= vocab) => {
                let clamped: Vec<i32> = tokens
                    .iter()
                    .map(|&t| if t < 0 || t as usize >= vocab { 0 } else { t })
                    .collect();
                ffi::from_slice_i32(&clamped, &shape)
            }
            _ => ffi::from_slice_i32(tokens, &shape),
        }
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

/// Why `target` and `draft` cannot run as a full-model pair, by the one rule
/// (#2495) applied to both, or `None` when they can. A server checks this once
/// at startup so a pair that could never verify a round is refused rather than
/// decoding plainly with a per-request debug line.
///
/// Used by: `server::model_worker` (startup).
pub fn full_model_pair_problem<T, D>(target: &T, draft: &D) -> Option<String>
where
    T: LanguageModel + ?Sized,
    D: LanguageModel + ?Sized,
{
    if let Some(reason) = prompt_lookup_unsupported_reason(target) {
        return Some(format!("the target cannot verify a draft: {reason}"));
    }
    DraftModelState::new(draft).err().map(|err| err.to_string())
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

    /// One draft forward of `tokens`, continuing from the caches; returns the
    /// argmax at the last position. One host read per proposal: chaining the
    /// block as a single lazy graph was measured on M1 Ultra at no gain on a
    /// 4-bit pair and 15 percent slower on a bf16 one (#2757).
    fn step(&mut self, tokens: &[i32]) -> Result<i32, DrafterError> {
        let input = self.state.input_array(tokens);
        let logits = self
            .model
            .try_forward_last_logits_with_sequence_id(
                &input,
                None,
                &mut self.state.caches,
                None,
                tokens.len() - 1,
            )
            .map_err(|reason| DrafterError::DraftFailed { reason })?;
        self.state.stats.draft_forwards += 1;
        let argmax = ffi::argmax_last_axis(&logits);
        crate::try_eval(&argmax).map_err(|e| DrafterError::DraftFailed {
            reason: e.to_string(),
        })?;
        Ok(ffi::item_i32(&argmax))
    }

    /// Forward context tokens in chunks, evaluating only the cache state
    /// between them so the LM head never runs over a chunk.
    fn forward_chunks(&mut self, tokens: &[i32], chunk_len: usize) -> Result<(), DrafterError> {
        for chunk in tokens.chunks(chunk_len) {
            let input = self.state.input_array(chunk);
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
        // No forward here: the first ask feeds the prompt as the uncached
        // tail, so a row that is never asked costs nothing.
        self.state.clear(self.model.make_caches());
        self.state.context.extend_from_slice(prompt_tokens);
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
        // The uncached tail of the context (see the module docs). Never empty
        // while the invariant holds; re-feed the current token if it somehow
        // is.
        if self.state.held >= self.state.context.len() {
            let over = self.state.held - self.state.context.len() + 1;
            trim_caches(&mut self.state.caches, over);
            self.state.held -= over;
        }
        let tail: Vec<i32> = self.state.context[self.state.held..].to_vec();
        // `MLXCEL_PREFILL_CHUNK=0` disables chunking.
        let chunk_len = match crate::prefill_plan::prefill_chunk_len() {
            0 => tail.len(),
            len => len,
        };
        let (leading, last) = tail.split_at(tail.len().saturating_sub(chunk_len));
        self.forward_chunks(leading, chunk_len)?;
        let mut draft = Vec::with_capacity(block_size);
        let mut next = self.step(last)?;
        self.state.held = self.state.context.len();
        draft.push(next);
        while draft.len() < block_size && !self.state.eos.contains(&next) {
            // `next` is now a proposal in the caches; the last one is not fed.
            self.state.held_drafts.push(next);
            next = self.step(&[next])?;
            draft.push(next);
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
