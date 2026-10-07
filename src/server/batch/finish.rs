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

//! The server side of the shared post-sample finish step (#2168).
//!
//! Every `BatchScheduler` decode site (per-row batched decode, fused batched
//! decode, single-step decode, prefill completion and the speculative burst
//! stream) runs [`mlxcel_core::finish_step`] through [`run_finish_step`]. The
//! [`FinishHooks`] here stream the token's text through the request's stop
//! matcher, read the b10621 generation bounds (#1477) and apply the context
//! bound (#1472). [`apply_finish_cause`] then maps the [`FinishCause`] onto the
//! sequence's [`FinishReason`] in one place.

use std::sync::mpsc;

use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::{FinishCause, FinishHooks, FinishInput, finish_step};

use super::generation_bounds::GenerationBounds;
use super::sequence::{FinishReason, SequenceInfo, SequenceState, emit_decoded_piece};
use super::stop_matcher::StopMatcher;
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::tokenizer::MlxcelTokenizer;

/// The scheduler's KV bound and context-shift setting, which decide the
/// context-bound stop (#1472). Copied into the speculative burst stream so a
/// burst applies the same bound as classic decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ContextBound {
    /// The KV window bound (`--max-kv-size`, or the context size), if any.
    pub(crate) max_kv_size: Option<usize>,
    /// Whether context shifting trims the window instead of stopping.
    pub(crate) context_shift: bool,
}

impl ContextBound {
    /// Whether a sequence must stop now because its next token would not fit
    /// the KV window with context shifting disabled.
    ///
    /// Token-count based (`prompt + generated + 1 >= bound`), mirroring
    /// upstream's `slot.prompt.n_tokens() + 1 >= slot.n_ctx`: with shifting
    /// disabled nothing ever trims, so the token count IS the live KV window,
    /// including for the paged and Turbo cache modes whose trim operation is
    /// a recorded no-op. VLM sequences are exempt, as upstream exempts
    /// multimodal from the context machinery (their KV length is the
    /// embedding count, not the text token count).
    pub(crate) fn stop_due(self, prompt_len: usize, generated_len: usize, is_vlm: bool) -> bool {
        !self.context_shift
            && !is_vlm
            && self
                .max_kv_size
                .is_some_and(|max| prompt_len + generated_len + 1 >= max)
    }
}

/// Where a finish step reads its EOS set, budget and history from.
pub(crate) enum StepLimits<'a> {
    /// Classic decode and prefill: the sequence's merged EOS, `max_tokens`,
    /// and its token history when the sampler reads one.
    Sequence,
    /// The speculative burst stream: the stream's merged EOS set and its
    /// emission budget (`max_tokens.max(1)`). The burst keeps no history.
    Burst { eos: &'a [i32], max_tokens: usize },
}

/// [`FinishHooks`] over the disjoint `SequenceInfo` fields the stop matcher,
/// the generation bounds and the context bound need, so the finish step can
/// hold `generated_tokens` mutably at the same time.
struct SequenceFinishHooks<'s> {
    decode_state: &'s mut StreamingDecodeState,
    stop_matcher: &'s mut StopMatcher,
    bounds: &'s mut GenerationBounds,
    response_tx: &'s mpsc::Sender<GenerateEvent>,
    tokenizer: &'s MlxcelTokenizer,
    logprobs: Option<TokenLogprobData>,
    prompt_len: usize,
    is_vlm: bool,
    context: ContextBound,
}

impl FinishHooks for SequenceFinishHooks<'_> {
    fn stop_text(&mut self, token: i32, generated_len: usize) -> bool {
        // Stream through the request's stop matcher (#1466): text that could
        // still become a stop string is held back, and a completed stop string
        // ends the sequence with the match excluded.
        match self.decode_state.on_token(token, self.tokenizer) {
            Some(text) => emit_decoded_piece(
                self.stop_matcher,
                self.bounds,
                self.response_tx,
                generated_len,
                text,
                Some(token),
                self.logprobs.take(),
            )
            .is_some(),
            None => false,
        }
    }

    fn bound_stopped(&self) -> bool {
        self.bounds.fired().is_some()
    }

    fn context_bound_due(&self, generated_len: usize) -> bool {
        self.context
            .stop_due(self.prompt_len, generated_len, self.is_vlm)
    }
}

/// Run the shared finish step for `token` on `seq` without touching its state.
///
/// `logprobs` rides the token's stream event. The caller maps a returned cause
/// with [`apply_finish_cause`] (or [`finish_reason_for`] when it transitions
/// the state itself).
pub(crate) fn run_finish_step(
    seq: &mut SequenceInfo,
    tokenizer: &MlxcelTokenizer,
    token: i32,
    logprobs: Option<TokenLogprobData>,
    structured_stopped: bool,
    context: ContextBound,
    limits: StepLimits<'_>,
) -> Option<FinishCause> {
    let SequenceInfo {
        decode_state,
        stop_matcher,
        bounds,
        response_tx,
        generated_tokens,
        token_history,
        merged_eos,
        sampling,
        max_tokens,
        prompt_tokens,
        vlm_embeddings,
        ..
    } = seq;
    let (eos, max_tokens, history): (&[i32], usize, Option<&mut Vec<i32>>) = match limits {
        StepLimits::Sequence => (
            merged_eos.as_slice(),
            *max_tokens,
            sampling.needs_token_history().then_some(token_history),
        ),
        StepLimits::Burst { eos, max_tokens } => (eos, max_tokens, None),
    };
    let mut hooks = SequenceFinishHooks {
        decode_state,
        stop_matcher,
        bounds,
        response_tx,
        tokenizer,
        logprobs,
        prompt_len: prompt_tokens.len(),
        is_vlm: vlm_embeddings.is_some(),
        context,
    };
    finish_step(
        FinishInput {
            token,
            eos,
            generated: generated_tokens,
            history,
            max_tokens,
            structured_stopped,
            loop_detection: &sampling.loop_detection,
        },
        &mut hooks,
    )
}

/// The [`FinishReason`] a cause reports, applying the cause's side effects on
/// `seq`: an EOS stop marks [`SequenceInfo::eos_terminated`] (#1754), and a
/// context-bound stop marks the response truncated (#1472).
pub(crate) fn finish_reason_for(seq: &mut SequenceInfo, cause: FinishCause) -> FinishReason {
    match cause {
        FinishCause::Eos => {
            seq.eos_terminated = true;
            FinishReason::Stop
        }
        FinishCause::StructuredStop => FinishReason::Stop,
        FinishCause::StopSequence => FinishReason::StopSequence,
        FinishCause::Length => FinishReason::Length,
        FinishCause::ContextExhausted => {
            seq.retention.context_exhausted = true;
            FinishReason::Length
        }
        FinishCause::RepetitionLoop => FinishReason::RepetitionLoop,
    }
}

/// Finish `seq` for `cause`. A failed state transition is logged and never
/// aborts the request.
pub(crate) fn apply_finish_cause(seq: &mut SequenceInfo, cause: FinishCause) {
    let reason = finish_reason_for(seq, cause);
    match seq.state.transition_to(SequenceState::Finished(reason)) {
        Ok(()) if cause == FinishCause::RepetitionLoop => tracing::info!(
            generated = seq.generated_tokens.len(),
            "loop detection: ending generation early (repetition loop)"
        ),
        Ok(()) => {}
        Err(err) => tracing::error!("State transition error: {err}"),
    }
}

/// The classic decode and prefill finish: run the shared finish step on the
/// sequence's own limits and, when it fires, finish the sequence.
pub(crate) fn finish_decode_token(
    seq: &mut SequenceInfo,
    tokenizer: &MlxcelTokenizer,
    token: i32,
    logprobs: Option<TokenLogprobData>,
    structured_stopped: bool,
    context: ContextBound,
) -> Option<FinishCause> {
    let cause = run_finish_step(
        seq,
        tokenizer,
        token,
        logprobs,
        structured_stopped,
        context,
        StepLimits::Sequence,
    );
    if let Some(cause) = cause {
        apply_finish_cause(seq, cause);
    }
    cause
}
