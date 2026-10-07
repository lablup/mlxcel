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

//! The one per-row sampling step (issue #2169, ADR 0007 "Sampling step").
//!
//! Every decode path that samples one row at a time goes through
//! [`RowSampler`]: the four `CxxGenerator` loops behind `mlxcel generate`,
//! `mlxcel run` and the chat REPL, and the server, where the engine's per-row
//! chain (`mlxcel_core::engine::sample_and_finish_row`, #2172) draws every
//! decode row and the first token of a prefill. Before this module each side
//! owned a copy of the step and the copies drifted: #2090 (the penalty history
//! one token stale) hit only the CLI, and the CLI created `SamplerState` only
//! for history-reading samplers, so mirostat and adaptive-p would have lost
//! their feedback state between tokens.
//!
//! ## What the step owns, per sequence
//!
//! - **State lifecycle.** [`RowSampler::new`] creates the [`SamplerState`]
//!   when [`RowSampler::needs_state`] says so (`needs_token_history() ||
//!   needs_sampler_feedback_state()`), once, at construction, and the state
//!   lives as long as the sampler. The chain's own lazy creator stays as a
//!   safety net for a sampler built before its config was known
//!   ([`RowSampler::default`]).
//! - **History ordering (#2090).** Before step N is drawn, the history the
//!   caller passes ends with token N-1, and the prompt is included. The step
//!   does not own the history, because the finish step pushes to it, but it
//!   says when the caller must read the previous token on the host first:
//!   [`RowSampler::needs_host_token_before_sample`]. That is true only for a
//!   history-reading sampler (penalties, DRY). A feedback-only sampler
//!   (mirostat, adaptive-p) carries state but reads no history, so a
//!   pipelined caller keeps building the next sample straight from the lazy
//!   graph.
//! - **The chain.** [`RowSampler::draw`] calls the existing sampler chain
//!   (`sample_token_optimized_with_state`, or
//!   `sample_token_with_state_and_distribution` when the caller wants the
//!   post-chain distribution); it copies no chain stage.
//! - **Structured-output mask hook.** A [`LogitMask`] runs on the row's logits
//!   before the chain. A mask that rejects the row returns its message, which
//!   the caller turns into the same structured-output error as before.
//! - **Post-draw override hook.** [`RowSampler::resolve`] applies an override
//!   such as the thinking budget's forced end-of-think to the sampled token and
//!   returns `(sampled, final)`: the grammar advances on `sampled` (its mask
//!   described the unaltered logits) and everything downstream uses `final`.
//!   It then confirms `final` to the feedback state (`accept_token`), so an
//!   override leaves the adaptive-p EMA untouched.
//! - **Fusion.** [`RowSampler::fused_eligible`] decides whether a row may join
//!   the batched fused dispatch, derived from [`SamplerStage::ALL`], the list
//!   of stages the step runs. Adding a stage means adding a variant, and the
//!   exhaustive matches in [`SamplerStage::is_active`] and
//!   [`SamplerStage::fusion`] force a decision about fusing it.
//!
//! Token-bias composition, the other half of #2169, is
//! [`crate::sampling_token_bias::compose_token_bias`].

use crate::ffi;
use crate::ffi::MlxArray;
use crate::generate::SamplingConfig;
use crate::sampling::{
    SamplerState, sample_token_optimized_with_state, sample_token_with_state_and_distribution,
};
use cxx::UniquePtr;

/// A per-row logit mask applied before the sampler chain, such as a
/// structured-output grammar mask.
///
/// The server passes its grammar constraint; the CLI passes none.
pub trait LogitMask {
    /// Mask `logits` (one row, `[1, 1, vocab]` or `[1, vocab]`) for this step.
    ///
    /// `vocab_size` is the logits' last dimension. `Err` carries the reason
    /// the mask could not be applied (for a grammar mask, a matcher in an
    /// error state or one that allows no token); the caller aborts the row
    /// with it.
    fn apply(
        &mut self,
        logits: UniquePtr<MlxArray>,
        vocab_size: usize,
    ) -> Result<UniquePtr<MlxArray>, String>;
}

/// The result of one [`RowSampler::draw`], left unevaluated so a pipelined
/// caller keeps its lookahead.
pub struct TokenDraw {
    /// The sampled token id, a lazy `[1]` array.
    pub token: UniquePtr<MlxArray>,
    /// The logits after token bias and the history penalties, the view the
    /// chain's logprobs are computed from.
    pub adjusted_logits: UniquePtr<MlxArray>,
    /// The float32 `[1, vocab]` post-chain distribution the draw came from,
    /// present only when the caller asked for it.
    pub distribution: Option<UniquePtr<MlxArray>>,
}

impl TokenDraw {
    /// The `(token, adjusted_logits)` pair the pipelined CLI loops carry.
    pub fn into_token_and_logits(self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (self.token, self.adjusted_logits)
    }
}

/// One sequence's sampling step and its [`SamplerState`].
///
/// Used by: `CxxGenerator` decode loops, `engine::sample_and_finish_row`
/// (the scheduler's decode steps and prefill completions)
#[derive(Debug, Clone, Default)]
pub struct RowSampler {
    state: Option<SamplerState>,
    needs_history: bool,
}

impl RowSampler {
    /// The sampler for one sequence sampled under `config`, with its state
    /// created now when the config needs one.
    pub fn new(config: &SamplingConfig) -> Self {
        Self {
            state: Self::needs_state(config).then(|| SamplerState::for_config(config)),
            needs_history: config.needs_token_history(),
        }
    }

    /// The state-creation rule: a sampler that reads the token history
    /// (penalties, DRY) or carries feedback state across steps (mirostat,
    /// adaptive-p) gets a [`SamplerState`].
    pub fn needs_state(config: &SamplingConfig) -> bool {
        config.needs_token_history() || config.needs_sampler_feedback_state()
    }

    /// Whether the caller must append the previous token to the history,
    /// which means reading it on the host, before drawing the next one.
    ///
    /// `false` keeps a pipelined loop free of a per-step host read: the greedy
    /// and plain stochastic paths, and the feedback-only samplers.
    pub fn needs_host_token_before_sample(&self) -> bool {
        self.needs_history
    }

    /// The sequence's sampler state, if it has one.
    pub fn state(&self) -> Option<&SamplerState> {
        self.state.as_ref()
    }

    /// Draw one token from `logits` (one row) under `config`.
    ///
    /// `history` must already end with the previous token (see
    /// [`Self::needs_host_token_before_sample`]). `mask` runs before the
    /// chain; its error is returned unchanged and nothing is sampled.
    /// `want_distribution` attaches the post-chain distribution, drawn in the
    /// same chain pass so both share one XTC gate.
    ///
    /// The draw is not confirmed to the feedback state yet: call
    /// [`Self::resolve`] once the token is on the host, or
    /// [`Self::draw_and_accept`] when there is no override.
    pub fn draw(
        &mut self,
        logits: &MlxArray,
        config: &SamplingConfig,
        history: &[i32],
        mask: Option<&mut dyn LogitMask>,
        want_distribution: bool,
    ) -> Result<TokenDraw, String> {
        let Some(mask) = mask else {
            return Ok(self.sample(logits, config, history, want_distribution));
        };
        let vocab_size = ffi::array_shape(logits).last().copied().unwrap_or(0) as usize;
        let masked = mask.apply(ffi::copy(logits), vocab_size)?;
        Ok(self.sample(&masked, config, history, want_distribution))
    }

    /// [`Self::draw`] without a mask or a distribution, confirmed at once as
    /// the emitted token.
    ///
    /// For callers that never override a draw (the CLI): confirming at draw
    /// time needs no host read, and the next draw sees the updated feedback
    /// state exactly as it would after [`Self::resolve`].
    pub fn draw_and_accept(
        &mut self,
        logits: &MlxArray,
        config: &SamplingConfig,
        history: &[i32],
    ) -> TokenDraw {
        let draw = self.sample(logits, config, history, false);
        self.accept_drawn();
        draw
    }

    /// Apply `token_override` to the host value of the last draw and confirm
    /// the result to the feedback state. Returns `(sampled, final)`.
    ///
    /// Pass `|t| t` when nothing overrides the draw.
    pub fn resolve(&mut self, sampled: i32, token_override: impl FnOnce(i32) -> i32) -> (i32, i32) {
        let final_token = token_override(sampled);
        if let Some(state) = self.state.as_mut() {
            state.accept_token(final_token);
        }
        (sampled, final_token)
    }

    /// Confirm the last draw unchanged, without reading it on the host.
    ///
    /// Adaptive-p parks the token it drew; accepting that same id is what
    /// `resolve(sampled, |t| t)` does. Every other sampler has nothing to
    /// confirm.
    fn accept_drawn(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.accept_pending_token();
        }
    }

    fn sample(
        &mut self,
        logits: &MlxArray,
        config: &SamplingConfig,
        history: &[i32],
        want_distribution: bool,
    ) -> TokenDraw {
        if want_distribution {
            let (token, adjusted_logits, distribution) =
                sample_token_with_state_and_distribution(logits, config, history, &mut self.state);
            TokenDraw {
                token,
                adjusted_logits,
                distribution: Some(distribution),
            }
        } else {
            // With no state this is `sample_token_optimized` exactly: the
            // chain's lazy creator allocates nothing for a config that needs
            // no state.
            let (token, adjusted_logits) =
                sample_token_optimized_with_state(logits, config, history, &mut self.state);
            TokenDraw {
                token,
                adjusted_logits,
                distribution: None,
            }
        }
    }

    /// Whether a row sampled under `config` may join the batched fused
    /// dispatch, which samples `[B, vocab] -> [B]` from shared scalar
    /// parameters.
    ///
    /// Every active stage of [`SamplerStage::ALL`] must be one the dispatch
    /// runs ([`StageFusion::InDispatch`]) or a static row edit the scheduler
    /// folds into the logits first ([`StageFusion::FoldedRowEdit`], the token
    /// bias). The token bias stops folding while the B9 pre-bias suppression
    /// counters are on ([`crate::lang_bias_counters`]): they read the pre-bias
    /// argmax on the host every step, which only the per-row sampler does.
    ///
    /// Per-row obligations that are not sampler stages (a grammar mask, a
    /// thinking-budget override, a logprobs payload) are the caller's to add;
    /// see [`crate::sampling::row_supports_fused_batch_except_bias`].
    pub fn fused_eligible(config: &SamplingConfig) -> bool {
        SamplerStage::ALL
            .iter()
            .filter(|stage| stage.is_active(config))
            .all(|stage| match stage.fusion() {
                StageFusion::InDispatch => true,
                StageFusion::FoldedRowEdit => !crate::lang_bias_counters::enabled(),
                StageFusion::PerRowOnly => false,
            })
    }
}

/// The stages of the per-row sampler chain, in chain order.
///
/// Used by: [`RowSampler::fused_eligible`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SamplerStage {
    /// Additive per-token bias (`logit_bias`, language bias, suppression).
    TokenBias,
    /// Repetition penalty over the history window.
    RepetitionPenalty,
    /// DRY penalty over the history.
    Dry,
    /// Frequency and presence penalties over the history window.
    FrequencyPresencePenalty,
    /// Row-wise, history-free filters: top-n-sigma, p-less, typical-p.
    RowFilters,
    /// Temperature, top-k, top-p, min-p and the categorical draw (or argmax).
    FusedChain,
    /// XTC, gated by one random draw per step.
    Xtc,
    /// Mirostat, which replaces the chain and carries `mu`.
    Mirostat,
    /// Dynamic temperature (extended chain).
    DynamicTemperature,
    /// The `min_keep` floor on the truncation filters (extended chain).
    MinKeep,
    /// Adaptive-p, which replaces the draw and carries an EMA.
    AdaptiveP,
}

/// Whether the batched fused dispatch can run a [`SamplerStage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageFusion {
    /// The dispatch runs it from its shared scalar parameters.
    InDispatch,
    /// A static per-row logit edit the scheduler folds into the batch logits
    /// before the dispatch (`apply_token_bias_rows`).
    FoldedRowEdit,
    /// Needs per-row history, per-row state, per-row randomness or Rust filter
    /// arithmetic the dispatch has no parameters for.
    PerRowOnly,
}

impl SamplerStage {
    /// Every stage, in chain order.
    pub const ALL: [SamplerStage; 11] = [
        SamplerStage::TokenBias,
        SamplerStage::RepetitionPenalty,
        SamplerStage::Dry,
        SamplerStage::FrequencyPresencePenalty,
        SamplerStage::RowFilters,
        SamplerStage::FusedChain,
        SamplerStage::Xtc,
        SamplerStage::Mirostat,
        SamplerStage::DynamicTemperature,
        SamplerStage::MinKeep,
        SamplerStage::AdaptiveP,
    ];

    /// Whether the step runs this stage for `config`. Mirrors the conditions
    /// the chain itself checks (mirostat replaces the whole chain, so it turns
    /// the history stages off).
    pub fn is_active(self, config: &SamplingConfig) -> bool {
        let chain = config.effective_mirostat() == 0;
        match self {
            SamplerStage::TokenBias => !config.token_bias.is_empty(),
            SamplerStage::RepetitionPenalty => {
                chain && config.penalty_last_n != 0 && config.repetition_penalty != 1.0
            }
            SamplerStage::Dry => {
                chain && config.dry_multiplier > 0.0 && config.dry_penalty_last_n != 0
            }
            SamplerStage::FrequencyPresencePenalty => {
                chain
                    && config.penalty_last_n != 0
                    && (config.frequency_penalty != 0.0 || config.presence_penalty != 0.0)
            }
            SamplerStage::RowFilters => {
                config.effective_top_n_sigma() > 0.0
                    || config.effective_p_less()
                    || config.effective_typical_p() < 1.0
            }
            SamplerStage::FusedChain => chain,
            SamplerStage::Xtc => config.xtc_probability > 0.0,
            SamplerStage::Mirostat => !chain,
            SamplerStage::DynamicTemperature => config.effective_dynatemp_range() > 0.0,
            SamplerStage::MinKeep => config.effective_min_keep() >= 2,
            SamplerStage::AdaptiveP => config.effective_adaptive_target() >= 0.0,
        }
    }

    /// Whether the batched fused dispatch can run this stage.
    pub fn fusion(self) -> StageFusion {
        match self {
            SamplerStage::TokenBias => StageFusion::FoldedRowEdit,
            SamplerStage::RowFilters | SamplerStage::FusedChain => StageFusion::InDispatch,
            SamplerStage::RepetitionPenalty
            | SamplerStage::Dry
            | SamplerStage::FrequencyPresencePenalty
            | SamplerStage::Xtc
            | SamplerStage::Mirostat
            | SamplerStage::DynamicTemperature
            | SamplerStage::MinKeep
            | SamplerStage::AdaptiveP => StageFusion::PerRowOnly,
        }
    }
}

// CLI-loop vs server-step token-stream identity, the feedback-state carry, and
// the hooks.
#[cfg(test)]
#[path = "sampling_row_step_tests.rs"]
mod tests;

// Fused eligibility derived from the stage list.
#[cfg(test)]
#[path = "sampling_row_step_fused_tests.rs"]
mod fused_tests;
