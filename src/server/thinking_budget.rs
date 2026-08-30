// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Thinking-token budget enforcement for Qwen3-family reasoning models.
//!
//! Qwen3/Qwen3.5/Qwen3.6 generate a `<think>...</think>` reasoning block before
//! producing the final answer. This module adds a per-request/per-server cap on
//! the number of tokens spent inside that block, matching the semantics of
//! llama.cpp's `--reasoning-budget` flag and vLLM's `thinking_token_budget`
//! sampling parameter.
//!
//! Resolution precedence (highest wins):
//! 1. Per-request body field (`thinking_budget_tokens` primary,
//!    `thinking_token_budget` vLLM alias, `thinking_budget` Qwen alias).
//! 2. `mlxcel-server` CLI flag `--reasoning-budget`.
//! 3. `LLAMA_ARG_THINK_BUDGET` env var.
//! 4. Legacy `LLAMA_ARG_REASONING_BUDGET` env alias.
//! 5. Default: unbounded (`-1`).
//!
//! Value semantics (match llama.cpp):
//! - `-1` = unrestricted (unbounded reasoning).
//! - `0`  = immediate end of thinking; the very first reasoning token is
//!   replaced by `</think>`.
//! - `N > 0` = cap reasoning at `N` tokens counted **inside** the block; the
//!   opening `<think>` token is not counted. When the limit is hit, the
//!   next sampled token is overridden with `</think>`.
//! - Other negative values (request-side) are rejected with a 400 error.
//!
//! The forced `</think>` is emitted as a normal delta chunk in SSE streams and
//! participates in `finish_reason` accounting. Forced tokens bypass logits-
//! based penalties for the override step only; subsequent sampling resumes
//! normally (no lingering history effect because `</think>` is treated as a
//! regular generated token).
//!
//! Used by: `server::batch::scheduler` (enforcement), `server::routes::chat`
//! and `server::routes::native_completion` (per-request parsing).

use std::num::NonZeroU32;

/// A validated thinking-token budget value.
///
/// Construct via [`Self::from_raw_i32`]; the `Option<Self>` wrapper distinguishes
/// "no budget plumbed" (`None` → unbounded) from "budget == 0" (immediate close).
///
/// A `-1` raw value is normalized to `None` at the boundary rather than stored
/// as a sentinel inside the enum so that all later code sees a clean "bounded
/// or not?" binary decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingBudget {
    /// Budget is `0`: the very first reasoning token is replaced by `</think>`.
    ImmediateClose,
    /// Budget is `N > 0`: allow up to `N` tokens inside the `<think>` block
    /// before forcing `</think>`.
    Limited(NonZeroU32),
}

impl ThinkingBudget {
    /// Effective cap as a `u32`.
    ///
    /// Returns `0` for [`Self::ImmediateClose`] and `N` for
    /// [`Self::Limited(N)`]. Useful for the "generated-inside-block count >=
    /// cap" comparison that drives forced `</think>` emission.
    #[inline]
    pub fn cap(self) -> u32 {
        match self {
            Self::ImmediateClose => 0,
            Self::Limited(n) => n.get(),
        }
    }

    /// Parse a raw `i32` into a validated budget.
    ///
    /// - `-1` → `Ok(None)` (unbounded).
    /// - `0`  → `Ok(Some(ImmediateClose))`.
    /// - `N > 0` → `Ok(Some(Limited(N)))`.
    /// - Other negatives → `Err(ThinkingBudgetError::InvalidNegative)`.
    pub fn from_raw_i32(raw: i32) -> Result<Option<Self>, ThinkingBudgetError> {
        match raw {
            -1 => Ok(None),
            0 => Ok(Some(Self::ImmediateClose)),
            n if n > 0 => {
                // SAFETY: n > 0 → non-zero; cast is safe because n <= i32::MAX < u32::MAX.
                let nz = NonZeroU32::new(n as u32).expect("n > 0 yields NonZeroU32");
                Ok(Some(Self::Limited(nz)))
            }
            other => Err(ThinkingBudgetError::InvalidNegative(other)),
        }
    }
}

/// Validation error for a user-supplied budget value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingBudgetError {
    /// A negative value other than `-1` was supplied.
    InvalidNegative(i32),
    /// The budget exceeds the request-level `max_tokens` / `n_predict` cap.
    ///
    /// `budget` is the requested reasoning cap; `max_tokens` is the generation cap.
    ExceedsMaxTokens { budget: u32, max_tokens: usize },
}

impl std::fmt::Display for ThinkingBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNegative(v) => write!(
                f,
                "thinking_budget_tokens must be >= -1 (got {v}); use -1 for unlimited, 0 for immediate end, or a positive cap"
            ),
            Self::ExceedsMaxTokens { budget, max_tokens } => write!(
                f,
                "thinking_budget_tokens ({budget}) must be <= max_tokens ({max_tokens})"
            ),
        }
    }
}

impl std::error::Error for ThinkingBudgetError {}

/// Apply the canonical `LLAMA_ARG_THINK_BUDGET` env fallback and the legacy
/// `LLAMA_ARG_REASONING_BUDGET` alias on top of a CLI value.
///
/// CLI wins when both are set. Unparseable env values trigger a warning and
/// the CLI value (or default) is kept.
///
/// Returns the raw `i32` ready for [`ThinkingBudget::from_raw_i32`].
pub fn resolve_server_default_reasoning_budget(cli_value: i32, cli_was_set: bool) -> i32 {
    const DEFAULT: i32 = -1;
    const CANONICAL_KEY: &str = "LLAMA_ARG_THINK_BUDGET";
    const LEGACY_KEY: &str = "LLAMA_ARG_REASONING_BUDGET";

    if cli_was_set || cli_value != DEFAULT {
        for key in [CANONICAL_KEY, LEGACY_KEY] {
            if std::env::var_os(key).is_none() {
                continue;
            }
            tracing::info!(
                "{key} env var is set but --reasoning-budget CLI flag takes precedence; ignoring {key}"
            );
        }
        return cli_value;
    }

    for key in [CANONICAL_KEY, LEGACY_KEY] {
        let Ok(raw) = std::env::var(key) else {
            continue;
        };
        return match raw.trim().parse::<i32>() {
            Ok(value) => value,
            Err(_) => {
                tracing::warn!(
                    "{key} env var has unparseable value {:?}; ignoring (expected integer >= -1)",
                    raw
                );
                DEFAULT
            }
        };
    }

    DEFAULT
}

/// Resolve the effective request-level budget given per-request override and
/// server default.
///
/// The per-request value, when present, fully overrides the server default
/// (including reverting to unbounded via `-1`).
pub fn resolve_request_budget(
    per_request: Option<i32>,
    server_default: Option<ThinkingBudget>,
    max_tokens: usize,
) -> Result<Option<ThinkingBudget>, ThinkingBudgetError> {
    if let Some(raw) = per_request {
        let parsed = ThinkingBudget::from_raw_i32(raw)?;
        if let Some(ThinkingBudget::Limited(n)) = parsed {
            let cap = n.get();
            if (cap as usize) > max_tokens {
                return Err(ThinkingBudgetError::ExceedsMaxTokens {
                    budget: cap,
                    max_tokens,
                });
            }
        }
        return Ok(parsed);
    }
    // No per-request override: inherit the server default.
    Ok(server_default)
}

/// Pick the per-request value from the three accepted aliases, in priority
/// order (primary wins).
///
/// The aliases are:
/// - `thinking_budget_tokens` (primary, llama.cpp)
/// - `thinking_token_budget` (vLLM alias)
/// - `thinking_budget` (Qwen alias)
pub fn pick_budget_alias(
    thinking_budget_tokens: Option<i32>,
    thinking_token_budget: Option<i32>,
    thinking_budget: Option<i32>,
) -> Option<i32> {
    thinking_budget_tokens
        .or(thinking_token_budget)
        .or(thinking_budget)
}

/// Token-id pair resolved once per tokenizer load.
///
/// Populated by [`resolve_thinking_token_ids`]. `None` for non-thinking models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingTokenIds {
    /// The `<think>` opening token id.
    pub open: i32,
    /// The `</think>` closing token id.
    pub close: i32,
}

/// Resolve the reasoning-block open/close token IDs from the tokenizer.
///
/// Prefers the Qwen3-family `<think>` / `</think>` pair when present; falls
/// back to the Gemma 4 `<|channel>` / `<channel|>` pair when the Qwen tokens
/// are absent. Both pairs are non-special added tokens that the tokenizer
/// exposes through `token_to_id` (`<think>` / `</think>` for Qwen3/Qwen3.5;
/// IDs 100 / 101 for `<|channel>` / `<channel|>` in Gemma 4's
/// `tokenizer.json`). The returned `ThinkingTokenIds` is opaque to the rest
/// of the scheduler — callers never need to know which family produced it.
///
/// Non-HF tokenizers are not supported for thinking-budget enforcement;
/// models without either pair return `None` and the budget parameter is
/// silently ignored.
///
/// Used by: `server::model_worker` at startup to cache per-model IDs.
pub fn resolve_thinking_token_ids(
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
) -> Option<ThinkingTokenIds> {
    let hf = tokenizer.hf_tokenizer()?;
    if let (Some(open), Some(close)) = (hf.token_to_id("<think>"), hf.token_to_id("</think>")) {
        return Some(ThinkingTokenIds {
            open: open as i32,
            close: close as i32,
        });
    }
    // Gemma 4 fallback: the reasoning block is wrapped in
    // `<|channel>thought\n…<channel|>`. We track the outer delimiter pair;
    // the `thought\n` stream between them is ordinary content that
    // `ThinkingState` counts via `in_block_count`.
    if let (Some(open), Some(close)) = (hf.token_to_id("<|channel>"), hf.token_to_id("<channel|>"))
    {
        return Some(ThinkingTokenIds {
            open: open as i32,
            close: close as i32,
        });
    }
    None
}

/// Runtime state tracking thinking-block position for one sequence.
///
/// Transitions:
/// - `Pending` → `InBlock` on first detected `<think>` (by token-id match or
///   sequence-start sentinel on `enter_block_on_start`).
/// - `InBlock` → `Closed` when `</think>` is emitted (either naturally or
///   forced by the budget).
/// - `Closed` is terminal for budget-tracking purposes; further tokens are
///   normal answer content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingPhase {
    /// Not inside a `<think>` block (either before it starts or not applicable
    /// because the model never emitted one).
    Pending,
    /// Currently inside the block. `in_block_count` tracks generated tokens
    /// that belong to the block body (i.e., excluding the opening `<think>`).
    InBlock,
    /// Block has closed; budget logic is inactive for the remainder of the
    /// request.
    Closed,
}

/// Per-sequence thinking-state tracker.
///
/// Stored on `SequenceInfo`. The scheduler drives transitions through the
/// [`Self::observe`] method on each generated token.
#[derive(Debug, Clone)]
pub struct ThinkingState {
    /// Current phase in the `<think>...</think>` lifecycle.
    pub phase: ThinkingPhase,
    /// Tokens generated **inside** the block so far. Does not count the
    /// opening `<think>` itself (vLLM behavior).
    pub in_block_count: u32,
    /// Resolved token ids; `None` → this sequence's model is not a thinking
    /// model (budget field is silently ignored).
    pub token_ids: Option<ThinkingTokenIds>,
    /// Effective cap for this sequence; `None` → unbounded.
    pub budget: Option<ThinkingBudget>,
    /// When `true`, the very first generated token is treated as
    /// "already-inside-block" (no opening `<think>` token expected). This
    /// matches the Qwen3 default where the chat template primes
    /// `<think>\n` so the model's first emitted token is reasoning content.
    pub enter_block_on_start: bool,
    /// Runtime "end reasoning now" flag, armed by the b10621
    /// `reasoning_control` request field and set by
    /// `POST /v1/chat/completions/control` with `action: "reasoning_end"`
    /// (#1444). When the flag is set, the next sampled token inside the
    /// thinking block is replaced by `</think>`, exactly as an exhausted
    /// budget would; committed tokens are unaffected. `Some` keeps the state
    /// active even when no budget is configured, which is upstream's
    /// "create the budget sampler on demand" semantics.
    pub force_end: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// b10621 `--reasoning-budget-message` (#1470), tokenized against this
    /// model's vocabulary. Upstream composes the forced run as
    /// `message_tokens + end_tag_tokens` and its budget sampler forces that
    /// whole sequence token by token, so the message is emitted on the
    /// exhausted-budget path AND on the runtime `reasoning_end` path, which
    /// enter the same forcing state. Empty (or `None`) forces the close token
    /// alone, which is the pre-#1470 behavior.
    forced_message: Option<std::sync::Arc<Vec<i32>>>,
    /// How many of [`forced_message`](Self::forced_message) have been emitted.
    /// Advanced by [`observe`](Self::observe) when the token it sees is the one
    /// [`decide_override`](Self::decide_override) forced, so the cursor cannot
    /// run ahead of what the client received.
    forced_message_cursor: usize,
}

impl ThinkingState {
    /// Construct a state that tracks the block for the given sequence.
    ///
    /// Returns a state that will short-circuit every observation (always
    /// returning [`ThinkingDecision::NoOverride`]) when either `budget` or
    /// `token_ids` is `None`.
    pub fn new(
        token_ids: Option<ThinkingTokenIds>,
        budget: Option<ThinkingBudget>,
        enter_block_on_start: bool,
    ) -> Self {
        Self {
            phase: ThinkingPhase::Pending,
            in_block_count: 0,
            token_ids,
            budget,
            enter_block_on_start,
            force_end: None,
            forced_message: None,
            forced_message_cursor: 0,
        }
    }

    /// Attach the tokenized `--reasoning-budget-message` (#1470).
    ///
    /// An empty run is stored as `None`, so the forcing path stays exactly the
    /// pre-#1470 single close token when the flag was not passed or tokenized
    /// to nothing.
    #[must_use]
    pub fn with_forced_message(mut self, tokens: Option<std::sync::Arc<Vec<i32>>>) -> Self {
        self.forced_message = tokens.filter(|run| !run.is_empty());
        self
    }

    /// Attach the runtime force-end flag armed by the request's
    /// `reasoning_control` field (#1444). A `Some` flag keeps the state
    /// active even without a configured budget.
    #[must_use]
    pub fn with_force_end(
        mut self,
        force_end: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Self {
        self.force_end = force_end;
        self
    }

    /// A no-op state used when thinking-budget enforcement is disabled for
    /// this sequence (either the server did not resolve a budget, or the
    /// model lacks `<think>`/`</think>`).
    pub const fn disabled() -> Self {
        Self {
            phase: ThinkingPhase::Closed,
            in_block_count: 0,
            token_ids: None,
            budget: None,
            enter_block_on_start: false,
            force_end: None,
            forced_message: None,
            forced_message_cursor: 0,
        }
    }

    /// `true` when this state will not affect sampling (zero overhead on the
    /// hot path beyond a single branch). An armed `force_end` flag keeps the
    /// state active even without a budget (#1444).
    #[inline]
    pub fn is_disabled(&self) -> bool {
        self.token_ids.is_none() || (self.budget.is_none() && self.force_end.is_none())
    }

    /// `true` when `reasoning_control` armed this sequence and a control
    /// request has since asked for the reasoning phase to end.
    #[inline]
    fn force_end_requested(&self) -> bool {
        self.force_end
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Inspect the upcoming sampled token and decide whether to override it
    /// with `</think>`.
    ///
    /// Call order: sampler yields `sampled_token`; caller invokes
    /// `state.decide_override(sampled_token)`. If the decision is
    /// `ForceClose`, the caller substitutes the close token id for
    /// `sampled_token` before streaming / committing it.
    ///
    /// After the final token value is known (whether overridden or not),
    /// the caller must invoke [`Self::observe`] to advance state.
    pub fn decide_override(&self, sampled: i32) -> ThinkingDecision {
        let Some(ids) = self.token_ids else {
            return ThinkingDecision::NoOverride;
        };
        if self.budget.is_none() && self.force_end.is_none() {
            return ThinkingDecision::NoOverride;
        }
        // A live `reasoning_end` control request acts exactly like an
        // exhausted budget from this step onward (#1444).
        let forced = self.force_end_requested();

        match self.phase {
            ThinkingPhase::Closed => ThinkingDecision::NoOverride,
            ThinkingPhase::Pending => {
                // If the chat template primes `<think>\n`, budget=0 (or a
                // pending force-end) takes effect at the very first
                // generated token.
                if self.enter_block_on_start
                    && (forced || self.budget.is_some_and(|b| b.cap() == 0))
                {
                    return self.forcing_decision(ids.close);
                }
                // Otherwise wait until we detect `<think>` open; no override
                // yet. (Budget=0 without a primed template still waits for
                // the open token so we don't mistakenly close before the
                // model enters the block.)
                ThinkingDecision::NoOverride
            }
            ThinkingPhase::InBlock => {
                // Natural close — let it through.
                if sampled == ids.close {
                    return ThinkingDecision::NoOverride;
                }
                if forced || self.budget.is_some_and(|b| self.in_block_count >= b.cap()) {
                    self.forcing_decision(ids.close)
                } else {
                    ThinkingDecision::NoOverride
                }
            }
        }
    }

    /// The token to force at this step: the next unemitted
    /// `--reasoning-budget-message` token, or the close tag once the message
    /// has been emitted in full (#1470).
    ///
    /// Upstream forces `message_tokens + end_tag_tokens` as one run, so the
    /// message lands immediately before the tag and nowhere else.
    fn forcing_decision(&self, close: i32) -> ThinkingDecision {
        match self.next_forced_message_token() {
            Some(token) => ThinkingDecision::ForceMessage(token),
            None => ThinkingDecision::ForceClose(close),
        }
    }

    /// The next message token to emit, or `None` when the run is finished, no
    /// message was configured, or no force is due at this step.
    ///
    /// The force-due half matters: without it a model that happens to sample
    /// the message's first token while still inside its budget would advance
    /// the cursor and lose a token of its own reasoning.
    fn next_forced_message_token(&self) -> Option<i32> {
        if !self.forcing_due() {
            return None;
        }
        self.forced_message
            .as_ref()
            .and_then(|run| run.get(self.forced_message_cursor))
            .copied()
    }

    /// Whether the close tag is being forced at this step: the budget is
    /// exhausted, or a runtime `reasoning_end` is armed. Mirrors the conditions
    /// [`decide_override`](Self::decide_override) forces on, so the cursor
    /// advances exactly when a message token was emitted.
    fn forcing_due(&self) -> bool {
        if self.token_ids.is_none() || (self.budget.is_none() && self.force_end.is_none()) {
            return false;
        }
        let forced = self.force_end_requested();
        match self.phase {
            ThinkingPhase::Closed => false,
            ThinkingPhase::Pending => {
                self.enter_block_on_start && (forced || self.budget.is_some_and(|b| b.cap() == 0))
            }
            ThinkingPhase::InBlock => {
                forced || self.budget.is_some_and(|b| self.in_block_count >= b.cap())
            }
        }
    }

    /// Record that `final_token` was emitted; advance phase and counters.
    ///
    /// `final_token` is the token value as it will be streamed to the client
    /// (i.e., after any override applied by [`Self::decide_override`]).
    pub fn observe(&mut self, final_token: i32) {
        let Some(ids) = self.token_ids else { return };
        if self.budget.is_none() && self.force_end.is_none() {
            return;
        }
        // A forced message token is bookkeeping for the forcing run, not
        // reasoning content: it must advance the cursor and nothing else, or
        // the run would repeat its first token forever (#1470). Matching by
        // value is exact, because the caller emits precisely the token
        // `decide_override` returned.
        if self.next_forced_message_token() == Some(final_token) {
            self.forced_message_cursor += 1;
            // The block is still open; the close tag follows the run.
            if self.phase == ThinkingPhase::Pending {
                self.phase = ThinkingPhase::InBlock;
            }
            return;
        }
        match self.phase {
            ThinkingPhase::Closed => {}
            ThinkingPhase::Pending => {
                if self.enter_block_on_start {
                    // Treat every emitted token as in-block from the start.
                    if final_token == ids.close {
                        self.phase = ThinkingPhase::Closed;
                    } else {
                        self.phase = ThinkingPhase::InBlock;
                        self.in_block_count = 1;
                    }
                } else if final_token == ids.open {
                    self.phase = ThinkingPhase::InBlock;
                    self.in_block_count = 0;
                }
                // Otherwise remain Pending — the model chose not to reason.
            }
            ThinkingPhase::InBlock => {
                if final_token == ids.close {
                    self.phase = ThinkingPhase::Closed;
                } else {
                    self.in_block_count = self.in_block_count.saturating_add(1);
                }
            }
        }
    }
}

/// Outcome of [`ThinkingState::decide_override`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingDecision {
    /// Keep the sampled token.
    NoOverride,
    /// Replace the sampled token with the given `</think>` token id.
    ForceClose(i32),
    /// Replace the sampled token with the next token of the configured
    /// `--reasoning-budget-message` (#1470). The run is emitted immediately
    /// before [`ForceClose`](Self::ForceClose), which follows it, so the
    /// caller substitutes this id exactly as it substitutes the close tag.
    ForceMessage(i32),
}

#[cfg(test)]
mod tests {
    use super::*;
    // Env-var fallback tests must serialize through the crate-wide
    // `ENV_LOCK`; per-module locks race with env mutations
    // in unrelated modules of the same test binary.
    use crate::test_support::env_lock::env_lock;

    // -- ThinkingBudget::from_raw_i32 --

    #[test]
    fn from_raw_minus_one_is_unbounded() {
        assert_eq!(ThinkingBudget::from_raw_i32(-1).unwrap(), None);
    }

    #[test]
    fn from_raw_zero_is_immediate_close() {
        assert_eq!(
            ThinkingBudget::from_raw_i32(0).unwrap(),
            Some(ThinkingBudget::ImmediateClose)
        );
    }

    #[test]
    fn from_raw_positive_is_limited() {
        let b = ThinkingBudget::from_raw_i32(64).unwrap().unwrap();
        assert_eq!(b.cap(), 64);
        assert!(matches!(b, ThinkingBudget::Limited(_)));
    }

    #[test]
    fn from_raw_other_negatives_rejected() {
        let err = ThinkingBudget::from_raw_i32(-2).unwrap_err();
        assert_eq!(err, ThinkingBudgetError::InvalidNegative(-2));
    }

    // -- pick_budget_alias precedence --

    #[test]
    fn alias_primary_wins() {
        assert_eq!(pick_budget_alias(Some(8), Some(16), Some(32)), Some(8));
    }

    #[test]
    fn alias_vllm_wins_over_qwen() {
        assert_eq!(pick_budget_alias(None, Some(16), Some(32)), Some(16));
    }

    #[test]
    fn alias_qwen_picked_last() {
        assert_eq!(pick_budget_alias(None, None, Some(32)), Some(32));
    }

    #[test]
    fn alias_all_none_is_none() {
        assert_eq!(pick_budget_alias(None, None, None), None);
    }

    // -- resolve_request_budget --

    #[test]
    fn request_none_inherits_server_default() {
        let server = ThinkingBudget::from_raw_i32(32).unwrap();
        let out = resolve_request_budget(None, server, 1024).unwrap();
        assert_eq!(out, server);
    }

    #[test]
    fn request_minus_one_reverts_even_when_server_bounded() {
        let server = ThinkingBudget::from_raw_i32(32).unwrap();
        let out = resolve_request_budget(Some(-1), server, 1024).unwrap();
        assert_eq!(out, None, "per-request -1 must revert to unbounded");
    }

    #[test]
    fn request_overrides_server_default() {
        let server = ThinkingBudget::from_raw_i32(32).unwrap();
        let out = resolve_request_budget(Some(8), server, 1024).unwrap();
        assert_eq!(out.unwrap().cap(), 8);
    }

    #[test]
    fn request_zero_means_immediate_close() {
        let out = resolve_request_budget(Some(0), None, 1024).unwrap();
        assert_eq!(out, Some(ThinkingBudget::ImmediateClose));
    }

    #[test]
    fn request_exceeding_max_tokens_rejected() {
        let err = resolve_request_budget(Some(2048), None, 1024).unwrap_err();
        assert!(matches!(
            err,
            ThinkingBudgetError::ExceedsMaxTokens {
                budget: 2048,
                max_tokens: 1024
            }
        ));
    }

    #[test]
    fn request_equal_to_max_tokens_accepted() {
        let out = resolve_request_budget(Some(1024), None, 1024).unwrap();
        assert_eq!(out.unwrap().cap(), 1024);
    }

    #[test]
    fn request_invalid_negative_rejected() {
        let err = resolve_request_budget(Some(-7), None, 1024).unwrap_err();
        assert_eq!(err, ThinkingBudgetError::InvalidNegative(-7));
    }

    // -- ThinkingState transitions --

    fn ids() -> ThinkingTokenIds {
        ThinkingTokenIds {
            open: 100,
            close: 200,
        }
    }

    #[test]
    fn disabled_state_never_overrides() {
        let s = ThinkingState::disabled();
        assert_eq!(s.decide_override(100), ThinkingDecision::NoOverride);
        assert_eq!(s.decide_override(200), ThinkingDecision::NoOverride);
    }

    #[test]
    fn unbounded_state_never_overrides() {
        let s = ThinkingState::new(Some(ids()), None, true);
        assert_eq!(s.decide_override(100), ThinkingDecision::NoOverride);
        assert_eq!(s.decide_override(999), ThinkingDecision::NoOverride);
    }

    #[test]
    fn budget_zero_with_template_primed_forces_immediate() {
        let s = ThinkingState::new(Some(ids()), Some(ThinkingBudget::ImmediateClose), true);
        assert_eq!(s.decide_override(42), ThinkingDecision::ForceClose(200));
    }

    // -- reasoning_control force-end (#1444) --

    fn force_flag() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))
    }

    #[test]
    fn armed_without_budget_is_active_and_tracks_the_block() {
        let flag = force_flag();
        let mut s = ThinkingState::new(Some(ids()), None, true).with_force_end(Some(flag.clone()));
        assert!(!s.is_disabled());

        // Flag not yet set: behaves like an unbounded budget.
        assert_eq!(s.decide_override(50), ThinkingDecision::NoOverride);
        s.observe(50);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));

        // Control request lands mid-block: next sampled reasoning token is
        // replaced by `</think>`; already-observed tokens are untouched.
        flag.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(s.decide_override(51), ThinkingDecision::ForceClose(200));
        s.observe(200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));

        // After the close the request continues as ordinary content.
        assert_eq!(s.decide_override(52), ThinkingDecision::NoOverride);
    }

    // -- #1470 `--reasoning-budget-message` injection --

    /// The tokenized message, standing in for `--reasoning-budget-message`.
    fn message() -> Option<std::sync::Arc<Vec<i32>>> {
        Some(std::sync::Arc::new(vec![301, 302]))
    }

    /// Drive the applier the decode loop uses: force what `decide_override`
    /// says, then `observe` exactly that token.
    fn step(state: &mut ThinkingState, sampled: i32) -> i32 {
        let final_token = match state.decide_override(sampled) {
            ThinkingDecision::NoOverride => sampled,
            ThinkingDecision::ForceClose(id) | ThinkingDecision::ForceMessage(id) => id,
        };
        state.observe(final_token);
        final_token
    }

    #[test]
    fn an_exhausted_budget_emits_the_message_then_the_close_tag() {
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(2).unwrap().unwrap()),
            false,
        )
        .with_forced_message(message());
        assert_eq!(step(&mut s, 100), 100, "the model opens the block");
        assert_eq!(step(&mut s, 50), 50, "in-block token 1 of 2");
        assert_eq!(step(&mut s, 51), 51, "in-block token 2 of 2");
        // Budget exhausted: the message run comes first, the close tag last,
        // which is upstream's `message_tokens + end_tag_tokens` order.
        assert_eq!(step(&mut s, 52), 301);
        assert_eq!(step(&mut s, 53), 302);
        assert_eq!(step(&mut s, 54), 200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));
        // Past the close the request is ordinary content again.
        assert_eq!(step(&mut s, 55), 55);
    }

    #[test]
    fn a_runtime_force_end_emits_the_message_too() {
        // Upstream enters the same FORCING state for a runtime `reasoning_end`
        // as for an exhausted budget, so the message is emitted on both paths.
        let flag = force_flag();
        let mut s = ThinkingState::new(Some(ids()), None, false)
            .with_force_end(Some(flag.clone()))
            .with_forced_message(message());
        assert_eq!(step(&mut s, 100), 100);
        assert_eq!(step(&mut s, 50), 50);
        flag.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(step(&mut s, 51), 301);
        assert_eq!(step(&mut s, 52), 302);
        assert_eq!(step(&mut s, 53), 200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));
    }

    #[test]
    fn the_message_is_emitted_on_the_primed_template_path() {
        let mut s = ThinkingState::new(Some(ids()), Some(ThinkingBudget::ImmediateClose), true)
            .with_forced_message(message());
        assert_eq!(step(&mut s, 42), 301);
        assert_eq!(step(&mut s, 43), 302);
        assert_eq!(step(&mut s, 44), 200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));
    }

    #[test]
    fn no_message_keeps_the_pre_1470_single_close_token() {
        let mut s = ThinkingState::new(Some(ids()), Some(ThinkingBudget::ImmediateClose), true);
        assert_eq!(step(&mut s, 42), 200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));
    }

    #[test]
    fn an_empty_message_is_stored_as_none() {
        let s = ThinkingState::new(Some(ids()), Some(ThinkingBudget::ImmediateClose), true)
            .with_forced_message(Some(std::sync::Arc::new(Vec::new())));
        assert_eq!(s.decide_override(42), ThinkingDecision::ForceClose(200));
    }

    #[test]
    fn a_sampled_token_equal_to_the_message_head_does_not_advance_the_cursor() {
        // The cursor advances on the token `decide_override` forced, and a
        // force is only due once the budget is exhausted. Without that gate a
        // model that happened to sample 301 inside its budget would lose a
        // token of its own reasoning to the injection run.
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(3).unwrap().unwrap()),
            false,
        )
        .with_forced_message(message());
        assert_eq!(step(&mut s, 100), 100, "open");
        assert_eq!(step(&mut s, 301), 301, "the model's own token, kept");
        assert_eq!(step(&mut s, 50), 50);
        assert_eq!(step(&mut s, 51), 51);
        // Budget exhausted now: the whole message still runs from its head.
        assert_eq!(step(&mut s, 52), 301);
        assert_eq!(step(&mut s, 53), 302);
        assert_eq!(step(&mut s, 54), 200);
    }

    #[test]
    fn a_natural_close_never_injects_the_message() {
        // The message belongs to the forced end only: a model that closes its
        // own block within budget must not have text inserted into it.
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(8).unwrap().unwrap()),
            false,
        )
        .with_forced_message(message());
        assert_eq!(step(&mut s, 100), 100);
        assert_eq!(step(&mut s, 50), 50);
        assert_eq!(step(&mut s, 200), 200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));
    }

    #[test]
    fn force_end_before_first_primed_token_closes_immediately() {
        let flag = force_flag();
        flag.store(true, std::sync::atomic::Ordering::Release);
        let s = ThinkingState::new(Some(ids()), None, true).with_force_end(Some(flag));
        assert_eq!(s.decide_override(42), ThinkingDecision::ForceClose(200));
    }

    #[test]
    fn force_end_outside_a_block_waits_for_the_open_token() {
        let flag = force_flag();
        flag.store(true, std::sync::atomic::Ordering::Release);
        let mut s = ThinkingState::new(Some(ids()), None, false).with_force_end(Some(flag));
        // No block open yet: ordinary content flows untouched.
        assert_eq!(s.decide_override(55), ThinkingDecision::NoOverride);
        s.observe(55);
        assert!(matches!(s.phase, ThinkingPhase::Pending));
        // The model opens a block anyway: the sticky flag closes it on the
        // next step, like a zero budget would.
        s.observe(100);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));
        assert_eq!(s.decide_override(56), ThinkingDecision::ForceClose(200));
    }

    #[test]
    fn force_end_lets_a_natural_close_through() {
        let flag = force_flag();
        flag.store(true, std::sync::atomic::Ordering::Release);
        let mut s = ThinkingState::new(Some(ids()), None, true).with_force_end(Some(flag));
        s.observe(50);
        // The sampled token IS the close: no override needed.
        assert_eq!(s.decide_override(200), ThinkingDecision::NoOverride);
    }

    #[test]
    fn unarmed_state_without_budget_stays_disabled() {
        let s = ThinkingState::new(Some(ids()), None, true).with_force_end(None);
        assert!(s.is_disabled());
    }

    #[test]
    fn budget_limited_enters_block_and_counts() {
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(3).unwrap().unwrap()),
            true,
        );
        // Qwen3 default: first token is treated as in-block.
        // 1st reasoning token.
        assert_eq!(s.decide_override(50), ThinkingDecision::NoOverride);
        s.observe(50);
        assert_eq!(s.in_block_count, 1);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));

        // 2nd reasoning token.
        assert_eq!(s.decide_override(51), ThinkingDecision::NoOverride);
        s.observe(51);
        assert_eq!(s.in_block_count, 2);

        // 3rd reasoning token — still within cap.
        assert_eq!(s.decide_override(52), ThinkingDecision::NoOverride);
        s.observe(52);
        assert_eq!(s.in_block_count, 3);

        // 4th reasoning token — would exceed cap (3) → force close.
        let dec = s.decide_override(53);
        assert_eq!(dec, ThinkingDecision::ForceClose(200));
        // Caller substitutes the close token and calls observe(200).
        s.observe(200);
        assert!(matches!(s.phase, ThinkingPhase::Closed));

        // Subsequent tokens are post-thinking and never overridden.
        assert_eq!(s.decide_override(54), ThinkingDecision::NoOverride);
    }

    #[test]
    fn observe_natural_close_marks_closed() {
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(10).unwrap().unwrap()),
            true,
        );
        s.observe(50); // in-block
        s.observe(200); // close
        assert!(matches!(s.phase, ThinkingPhase::Closed));
    }

    #[test]
    fn non_primed_waits_for_open_token() {
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(1).unwrap().unwrap()),
            false,
        );
        // Before the open token, state is Pending; budget doesn't fire.
        assert_eq!(s.decide_override(999), ThinkingDecision::NoOverride);
        s.observe(999);
        assert!(matches!(s.phase, ThinkingPhase::Pending));
        assert_eq!(s.in_block_count, 0);

        // Open token arrives.
        s.observe(100);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));
        assert_eq!(s.in_block_count, 0);

        // First reasoning token allowed (cap=1).
        assert_eq!(s.decide_override(50), ThinkingDecision::NoOverride);
        s.observe(50);
        assert_eq!(s.in_block_count, 1);

        // Second reasoning token would exceed cap → force close.
        assert_eq!(s.decide_override(51), ThinkingDecision::ForceClose(200));
    }

    #[test]
    fn non_primed_budget_zero_waits_for_open_token_first() {
        // Raw-text endpoints (e.g. `/completion`) don't prime `<think>` in
        // the prompt, so the scheduler constructs ThinkingState with
        // enter_block_on_start=false. Under that setting, budget=0 must NOT
        // inject `</think>` before the model has emitted `<think>` itself.
        // Without this behavior, a request like `{prompt: "Hello",
        // thinking_budget_tokens: 0}` would force `</think>` on the first
        // ordinary answer token — corrupting the output.
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::ImmediateClose),
            /*enter_block_on_start=*/ false,
        );
        // Regular answer tokens must pass through unchanged.
        for tok in [10, 20, 30, 40] {
            assert_eq!(
                s.decide_override(tok),
                ThinkingDecision::NoOverride,
                "budget=0 with enter_block_on_start=false must not override non-open tokens"
            );
            s.observe(tok);
        }
        assert!(matches!(s.phase, ThinkingPhase::Pending));
        assert_eq!(s.in_block_count, 0);

        // Once the model actually emits `<think>`, the in-block state begins.
        s.observe(ids().open);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));
        assert_eq!(s.in_block_count, 0);

        // Now budget=0 fires on the very first in-block token.
        assert_eq!(
            s.decide_override(42),
            ThinkingDecision::ForceClose(ids().close)
        );
    }

    #[test]
    fn natural_close_within_block_is_not_overridden_even_at_cap() {
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(2).unwrap().unwrap()),
            true,
        );
        s.observe(50);
        s.observe(51);
        // in_block_count == 2, cap == 2: if model chose to emit </think>
        // itself, we MUST let it through unchanged.
        assert_eq!(s.decide_override(200), ThinkingDecision::NoOverride);
    }

    #[test]
    fn budget_one_emits_exactly_one_reasoning_token() {
        // Regression guard: budget=1 means "exactly one token inside
        // <think>, then force </think>." Covers the AC "budget=N emits
        // exactly N reasoning tokens."
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(1).unwrap().unwrap()),
            true,
        );

        // 1st reasoning token allowed.
        assert_eq!(s.decide_override(50), ThinkingDecision::NoOverride);
        s.observe(50);

        // 2nd reasoning token would exceed cap -> force close.
        assert_eq!(s.decide_override(51), ThinkingDecision::ForceClose(200));
    }

    #[test]
    fn forced_close_does_not_advance_phase_past_closed() {
        // After a forced close, further tokens must NOT trigger another
        // override.
        let mut s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::from_raw_i32(1).unwrap().unwrap()),
            true,
        );
        s.observe(50); // in-block
        let d = s.decide_override(60);
        assert_eq!(d, ThinkingDecision::ForceClose(200));
        s.observe(200); // forced close committed
        assert!(matches!(s.phase, ThinkingPhase::Closed));

        for tok in [61, 62, 63, 200, 100] {
            assert_eq!(
                s.decide_override(tok),
                ThinkingDecision::NoOverride,
                "tok {tok} must not be overridden after Closed"
            );
        }
    }

    #[test]
    fn non_thinking_model_state_is_noop() {
        // Model without <think>/</think> — state always disabled.
        let mut s = ThinkingState::new(None, Some(ThinkingBudget::ImmediateClose), true);
        assert!(s.is_disabled());
        assert_eq!(s.decide_override(100), ThinkingDecision::NoOverride);
        s.observe(100);
        // Phase still starts as Pending; observe is a no-op when
        // token_ids is None.
        assert!(matches!(s.phase, ThinkingPhase::Pending));
        assert_eq!(s.in_block_count, 0);
    }

    // -- resolve_server_default_reasoning_budget env fallback --
    //
    // These three cases must not race on the shared process-wide env var, so
    // they're serialized through a single test using a mutex. Keeping them in
    // one test function also means `set_var` / `remove_var` pairs always run
    // in the same thread (required by the unsafe contract on edition 2024).

    #[test]
    fn server_default_env_fallback_all_cases() {
        let _guard = env_lock();

        // Case 1: CLI wins over env.
        // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
        unsafe {
            std::env::set_var("LLAMA_ARG_REASONING_BUDGET", "64");
        }
        assert_eq!(
            resolve_server_default_reasoning_budget(32, true),
            32,
            "CLI set to 32 must win over env=64"
        );

        // Case 2: canonical env is used when CLI is at default (-1), and wins
        // over the retained legacy spelling.
        unsafe {
            std::env::set_var("LLAMA_ARG_REASONING_BUDGET", "128");
            std::env::set_var("LLAMA_ARG_THINK_BUDGET", "96");
        }
        assert_eq!(resolve_server_default_reasoning_budget(-1, false), 96);

        // Case 3: unparseable canonical env falls back to default (-1).
        unsafe {
            std::env::set_var("LLAMA_ARG_THINK_BUDGET", "not-a-number");
        }
        assert_eq!(resolve_server_default_reasoning_budget(-1, false), -1);

        // Case 4: no env var and CLI at default (-1) → unbounded (-1).
        // Regression guard: `mlxcel-server` with no `--reasoning-budget` flag
        // must leave reasoning unbounded (AC#7 unit test requirement).
        unsafe {
            std::env::remove_var("LLAMA_ARG_THINK_BUDGET");
            std::env::remove_var("LLAMA_ARG_REASONING_BUDGET");
        }
        assert_eq!(
            resolve_server_default_reasoning_budget(-1, false),
            -1,
            "no CLI flag and no env var must yield unbounded (-1)"
        );
    }

    // -- Gemma 4 token-id fallback in resolve_thinking_token_ids --
    //
    // We can't easily construct a full `MlxcelTokenizer` without loading a
    // real model, so these tests exercise the decision directly against a
    // minimal in-memory HF tokenizer built from an added-vocab set. The
    // intent is to guard the "prefer Qwen, fall back to Gemma 4" ordering
    // and the "neither → None" non-thinking case so a future refactor
    // doesn't silently drop a family.

    use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};

    fn mlxcel_from(tokens: &[&str]) -> crate::tokenizer::MlxcelTokenizer {
        // Minimal BPE base (same shape `MlxcelTokenizer::stub` uses). We
        // only care about the added-vocab mappings, so the base model's
        // details are irrelevant here.
        let mut hf = Tokenizer::new(BPE::default());
        let added: Vec<AddedToken> = tokens
            .iter()
            .map(|s| AddedToken::from(*s, /*special=*/ true))
            .collect();
        hf.add_tokens(&added);
        crate::tokenizer::MlxcelTokenizer::HuggingFace(hf)
    }

    #[test]
    fn resolve_prefers_qwen_think_pair_when_present() {
        // Simulate a tokenizer that has BOTH pairs (hypothetical, but
        // exercises the ordering contract). Qwen pair wins.
        let tok = mlxcel_from(&["<think>", "</think>", "<|channel>", "<channel|>"]);
        let ids = resolve_thinking_token_ids(&tok).expect("qwen pair should resolve");
        // `<think>`/`</think>` are added first, so they get the lowest IDs
        // among the added set. Re-look them up to be robust against the
        // underlying tokenizer's id-assignment scheme.
        let hf = tok.hf_tokenizer().unwrap();
        assert_eq!(ids.open as u32, hf.token_to_id("<think>").unwrap());
        assert_eq!(ids.close as u32, hf.token_to_id("</think>").unwrap());
    }

    #[test]
    fn resolve_falls_back_to_gemma4_channel_pair() {
        // Qwen pair absent, Gemma 4 pair present. Budget should be usable.
        let tok = mlxcel_from(&["<|channel>", "<channel|>"]);
        let ids = resolve_thinking_token_ids(&tok).expect("gemma4 pair should resolve");
        let hf = tok.hf_tokenizer().unwrap();
        assert_eq!(ids.open as u32, hf.token_to_id("<|channel>").unwrap());
        assert_eq!(ids.close as u32, hf.token_to_id("<channel|>").unwrap());
    }

    #[test]
    fn resolve_returns_none_for_non_thinking_tokenizer() {
        let tok = mlxcel_from(&["<|user|>", "<|assistant|>"]);
        assert!(resolve_thinking_token_ids(&tok).is_none());
    }

    #[test]
    fn resolve_partial_gemma4_pair_does_not_resolve() {
        // Guard: only the opening marker is present (malformed config).
        // Must not silently pretend the pair exists.
        let tok = mlxcel_from(&["<|channel>"]);
        assert!(resolve_thinking_token_ids(&tok).is_none());

        let tok2 = mlxcel_from(&["<channel|>"]);
        assert!(resolve_thinking_token_ids(&tok2).is_none());
    }

    // -- ThinkingState behavior with the prompt-primed open channel path --

    #[test]
    fn primed_open_thinking_with_bounded_budget_force_closes_at_cap() {
        // Reproduces the Gemma 4 `enable_thinking=true` wiring the scheduler
        // applies: token_ids match `<|channel>`/`<channel|>`, budget=3, and
        // `enter_block_on_start=true` because the prompt left the model
        // inside the open thinking channel. After three in-block tokens the
        // next decide_override must force the close marker so generation
        // exits the channel and the model can emit a tool_call afterwards.
        let mut s = ThinkingState::new(
            Some(ids()), // reuses the module-local (100, 200) pair
            Some(ThinkingBudget::from_raw_i32(3).unwrap().unwrap()),
            /*enter_block_on_start=*/ true,
        );
        // Three reasoning tokens allowed.
        for token in [10, 20, 30] {
            assert_eq!(s.decide_override(token), ThinkingDecision::NoOverride);
            s.observe(token);
        }
        assert_eq!(s.in_block_count, 3);
        assert!(matches!(s.phase, ThinkingPhase::InBlock));

        // Fourth would exceed cap → force-emit close.
        let decision = s.decide_override(40);
        assert_eq!(decision, ThinkingDecision::ForceClose(ids().close));
        s.observe(ids().close);
        assert!(matches!(s.phase, ThinkingPhase::Closed));

        // Post-close tokens pass through unchanged — this is where the model
        // gets to emit its `<|tool_call>…<tool_call|>` for the tool-requiring
        // prompt after budget intervened.
        for token in [50, 60] {
            assert_eq!(s.decide_override(token), ThinkingDecision::NoOverride);
        }
    }

    #[test]
    fn primed_open_thinking_with_budget_zero_forces_close_immediately() {
        // Budget 0 + primed prompt = "no reasoning allowed, skip to action"
        // — the first decode step emits the close marker so token #1 is
        // already outside the thinking channel.
        let s = ThinkingState::new(
            Some(ids()),
            Some(ThinkingBudget::ImmediateClose),
            /*enter_block_on_start=*/ true,
        );
        assert_eq!(
            s.decide_override(10),
            ThinkingDecision::ForceClose(ids().close)
        );
    }
}
