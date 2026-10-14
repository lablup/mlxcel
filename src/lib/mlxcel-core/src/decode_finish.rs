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

//! The one post-sample finish step shared by every decode site (#2168).
//!
//! After a token is sampled, a decode site decides whether the sequence is
//! finished. The engine's per-row chain, which every `BatchScheduler` site
//! (per-row batched decode, fused batched decode, single-step decode, prefill
//! completion and the speculative burst stream) call [`finish_step`], so the
//! order of the checks and the set of checks cannot drift between them again.
//!
//! The parts that depend on the serving path (the text stop matcher, the
//! generation bounds and the context bound) are reached through
//! [`FinishHooks`]. The CLI passes [`NoStopHooks`], whose hooks never fire.

use crate::loop_detection::{LoopDetectionConfig, detect_repetition_loop};
use crate::memory::{cache_clear_interval, should_clear_cache_at};

/// Why [`finish_step`] ended a sequence.
///
/// The server maps each cause onto its `FinishReason`: `Eos` and
/// `StructuredStop` report `stop`, `ContextExhausted` reports `length` and
/// marks the response truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishCause {
    /// The sampled token is an end-of-sequence id. It was not pushed.
    Eos,
    /// A request stop string completed on this token.
    StopSequence,
    /// The structured-output grammar reached an accepting stop state.
    StructuredStop,
    /// The token budget is spent, or a generation bound (b10621 `n_indent`,
    /// `t_max_predict_ms`, #1477) fired.
    Length,
    /// The next token would not fit the KV window with context shifting
    /// disabled (#1472).
    ContextExhausted,
    /// The generated stream collapsed into a short repeated pattern (#432).
    RepetitionLoop,
}

/// The serving-path specific checks of [`finish_step`].
///
/// Every hook runs after the token was pushed, so `generated_len` counts it.
pub trait FinishHooks {
    /// Feed the token through the text stop matcher, emitting whatever text it
    /// releases. Returns `true` when a stop string completed.
    fn stop_text(&mut self, token: i32, generated_len: usize) -> bool;
    /// Whether a generation bound (b10621 #1477) has fired.
    fn bound_stopped(&self) -> bool;
    /// Whether the next token would overflow the KV window with context shift
    /// disabled, given `generated_len` generated tokens.
    fn context_bound_due(&self, generated_len: usize) -> bool;
}

/// Hooks for a decode loop with no stop strings, generation bounds or context
/// bound: every hook returns `false`. Used by the engine's raw-completion
/// client (`engine::BareHooks` builds on the same rule) and its tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoStopHooks;

impl FinishHooks for NoStopHooks {
    #[inline]
    fn stop_text(&mut self, _token: i32, _generated_len: usize) -> bool {
        false
    }

    #[inline]
    fn bound_stopped(&self) -> bool {
        false
    }

    #[inline]
    fn context_bound_due(&self, _generated_len: usize) -> bool {
        false
    }
}

/// The per-token state [`finish_step`] reads and updates.
pub struct FinishInput<'a> {
    /// The token id the decode loop already read back from the device.
    pub token: i32,
    /// Merged EOS ids (model EOS plus per-request stop tokens).
    pub eos: &'a [i32],
    /// The generated stream; the token is pushed here unless it is EOS.
    pub generated: &'a mut Vec<i32>,
    /// Penalty history; the token is pushed here too when `Some`.
    pub history: Option<&'a mut Vec<i32>>,
    /// `generated` length at which the token budget is spent.
    pub max_tokens: usize,
    /// Whether the structured-output matcher stopped on this token.
    pub structured_stopped: bool,
    /// Repetition-loop detection settings; disabled is the default.
    pub loop_detection: &'a LoopDetectionConfig,
}

/// Decide whether the sequence that just sampled `input.token` is finished.
///
/// The checks run in this order, and the first one that fires wins:
///
/// 1. EOS: [`FinishCause::Eos`]; the token is not pushed.
/// 2. Push the token to `generated`, and to `history` when present.
/// 3. Stop string ([`FinishHooks::stop_text`]): [`FinishCause::StopSequence`].
/// 4. Generation bound ([`FinishHooks::bound_stopped`]): [`FinishCause::Length`].
/// 5. Structured-output stop: [`FinishCause::StructuredStop`].
/// 6. Token budget (`generated.len() >= max_tokens`): [`FinishCause::Length`].
/// 7. Context bound ([`FinishHooks::context_bound_due`]):
///    [`FinishCause::ContextExhausted`].
/// 8. Repetition loop ([`detect_repetition_loop`]):
///    [`FinishCause::RepetitionLoop`].
/// 9. Not finished: the periodic cache clear ([`should_clear_cache_at`]).
///
/// It reads only `input.token`, which the caller has already read back, so it
/// never forces a host sync the decode loop would not otherwise do. With loop
/// detection disabled (the default), step 8 costs one branch.
#[inline]
pub fn finish_step(input: FinishInput<'_>, hooks: &mut dyn FinishHooks) -> Option<FinishCause> {
    let FinishInput {
        token,
        eos,
        generated,
        history,
        max_tokens,
        structured_stopped,
        loop_detection,
    } = input;

    if eos.contains(&token) {
        return Some(FinishCause::Eos);
    }

    generated.push(token);
    if let Some(history) = history {
        history.push(token);
    }
    let generated_len = generated.len();

    let cause = if hooks.stop_text(token, generated_len) {
        Some(FinishCause::StopSequence)
    } else if hooks.bound_stopped() {
        Some(FinishCause::Length)
    } else if structured_stopped {
        Some(FinishCause::StructuredStop)
    } else if generated_len >= max_tokens {
        Some(FinishCause::Length)
    } else if hooks.context_bound_due(generated_len) {
        Some(FinishCause::ContextExhausted)
    } else if loop_detection.is_enabled() && detect_repetition_loop(generated, loop_detection) {
        Some(FinishCause::RepetitionLoop)
    } else {
        None
    };

    // Periodic cache clearing, backend-aware cadence (#627): disabled by
    // default on CUDA (the clear churns the pool and defeats CUDA-graph
    // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
    if cause.is_none() && should_clear_cache_at(generated_len, cache_clear_interval()) {
        crate::ffi::clear_memory_cache();
    }

    cause
}

#[cfg(test)]
#[path = "decode_finish_tests.rs"]
mod tests;
