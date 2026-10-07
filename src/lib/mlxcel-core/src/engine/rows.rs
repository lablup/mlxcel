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

//! The per-row half of a step: sampling through [`RowSampler`] and finishing
//! through [`finish_step`], composed once for every row count.
//!
//! A [`StepRow`] is what the engine needs to turn one row of logits into an
//! emitted token and a finish verdict: the row's sampler and config, its
//! generated stream and penalty history, its EOS set and budget, and a
//! [`StepRowHooks`] through which the caller supplies the pieces that live
//! outside the engine (the structured-output mask and matcher, the
//! thinking-budget override, the stop-string matcher and generation bounds
//! behind [`FinishHooks`]). Every row reports its own [`RowOutcome`]; a row's
//! failure is a [`RowError`] on that outcome and never touches its neighbours.

use crate::cache::SequenceId;
use crate::decode_finish::{FinishCause, FinishHooks, FinishInput, finish_step};
use crate::ffi;
use crate::generate::SamplingConfig;
use crate::sampling::{
    FusedSampleParams, LogprobSource, LogprobsConfig, TokenBiasMap, TokenLogprobData,
    batched_fused_sample_tokens, compute_logprobs, compute_post_sampling_probs,
    row_supports_fused_batch_except_bias,
};
use crate::sampling_row_step::{LogitMask, RowSampler, TokenDraw};
use crate::{MlxArray, UniquePtr};

/// The caller-side hooks one row's sampling and finishing need.
///
/// The [`FinishHooks`] half streams the token's text, applies the stop
/// matcher and reads the generation and context bounds; this half covers
/// what happens between the forward and the finish step.
pub trait StepRowHooks: FinishHooks {
    /// The structured-output mask for this row, run on the raw logits before
    /// the sampling chain. `None` when the row carries no constraint.
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask>;

    /// The token to emit for `sampled` (the thinking-budget override); the
    /// identity when nothing overrides the draw.
    fn override_token(&mut self, sampled: i32) -> i32;

    /// Advance the structured-output matcher with the pre-override `sampled`
    /// token. `Ok(true)` means the matcher reached its stop state; `Err` is a
    /// matcher failure that finishes the row with an error.
    fn consume_sampled(&mut self, sampled: i32) -> Result<bool, String>;

    /// The emitted token is known and the matcher accepted the sampled one
    /// (after the override and [`StepRowHooks::consume_sampled`], before the
    /// finish step). A prefill completion stamps its first-token time here,
    /// so a matcher failure reports its error without a first-token event.
    fn resolved(&mut self, _sampled: i32, _token: i32) {}

    /// The per-token logprob payload that rides the token's stream event,
    /// handed over before the finish step emits the token.
    fn set_logprobs(&mut self, _logprobs: Option<TokenLogprobData>) {}
}

/// One row of a step: everything the engine samples and finishes with.
pub struct StepRow<'a, H: StepRowHooks> {
    pub seq_id: SequenceId,
    pub sampler: &'a mut RowSampler,
    pub sampling: &'a SamplingConfig,
    /// Penalty history; pushed by the finish step when the config reads one.
    pub token_history: &'a mut Vec<i32>,
    /// The generated stream; pushed by the finish step unless EOS.
    pub generated: &'a mut Vec<i32>,
    /// Merged EOS ids (model EOS plus per-request stop tokens).
    pub eos: &'a [i32],
    pub max_tokens: usize,
    pub logprobs: &'a LogprobsConfig,
    /// Whether [`StepRowHooks::logit_mask`] returns a mask (keeps the row off
    /// the fused batch path).
    pub needs_mask: bool,
    /// Whether [`StepRowHooks::override_token`] may change the draw (keeps the
    /// row off the fused batch path).
    pub needs_override: bool,
    pub hooks: H,
}

/// Why a row failed its step. The batch continues; the row finishes with an
/// error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowError {
    /// The structured-output mask or matcher failed.
    Structured(String),
    /// Evaluating the row's sampled token threw at the MLX boundary (#822).
    Eval(String),
}

impl RowError {
    pub fn message(&self) -> &str {
        match self {
            RowError::Structured(msg) | RowError::Eval(msg) => msg,
        }
    }
}

/// One row's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowOutcome {
    pub seq_id: SequenceId,
    /// The token the sampler drew (before any override).
    pub sampled: i32,
    /// The token emitted to the stream.
    pub token: i32,
    /// The finish verdict, `None` while the row continues.
    pub finish: Option<FinishCause>,
    /// Set when the row failed before it could finish; `token` and `finish`
    /// are then meaningless.
    pub error: Option<RowError>,
}

impl RowOutcome {
    fn failed(seq_id: SequenceId, error: RowError) -> Self {
        Self {
            seq_id,
            sampled: 0,
            token: 0,
            finish: None,
            error: Some(error),
        }
    }

    /// Whether an eval ran for this row and succeeded, for the caller's
    /// backend health counter: every row that was not failed by an eval.
    pub fn eval_ok(&self) -> bool {
        !matches!(self.error, Some(RowError::Eval(_)))
    }
}

/// What the fused-eligibility rule reads from one row: its sampling config
/// and the per-row obligations that keep it on the per-row chain.
#[derive(Debug, Clone, Copy)]
pub struct FusedRowGate<'c> {
    pub sampling: &'c SamplingConfig,
    /// A structured-output mask runs on the row's logits.
    pub needs_mask: bool,
    /// A thinking-budget override may change the draw.
    pub needs_override: bool,
    /// The row streams a per-token logprobs payload.
    pub logprobs_enabled: bool,
}

/// The one fused-eligibility rule, shared by [`Engine::step`](super::Engine::step)
/// (through [`fused_params`]) and the scheduler's lookahead gate: the shared
/// fused parameters when every row can be sampled in one `[B, vocab] -> [B]`
/// dispatch, i.e. all rows share the scalar parameters and none needs a
/// history penalty, a mask, an override or a per-token payload. Token bias is
/// folded into the logits, so a biased row stays eligible. A `None` item (a
/// row the caller could not resolve) or an empty iterator returns `None`.
pub fn shared_fused_params<'c>(
    rows: impl IntoIterator<Item = Option<FusedRowGate<'c>>>,
) -> Option<FusedSampleParams> {
    let mut shared: Option<FusedSampleParams> = None;
    for gate in rows {
        let gate = gate?;
        if !row_supports_fused_batch_except_bias(
            gate.sampling,
            gate.needs_mask,
            gate.needs_override,
            gate.logprobs_enabled,
        ) {
            return None;
        }
        let params = FusedSampleParams::from_config(gate.sampling);
        match shared {
            None => shared = Some(params),
            Some(first) if !first.matches(&params) => return None,
            Some(_) => {}
        }
    }
    shared
}

impl<H: StepRowHooks> StepRow<'_, H> {
    /// This row's input to the fused-eligibility rule.
    pub fn fused_gate(&self) -> FusedRowGate<'_> {
        FusedRowGate {
            sampling: self.sampling,
            needs_mask: self.needs_mask,
            needs_override: self.needs_override,
            logprobs_enabled: self.logprobs.enabled,
        }
    }
}

/// [`shared_fused_params`] over a step's rows.
pub fn fused_params<H: StepRowHooks>(rows: &[StepRow<'_, H>]) -> Option<FusedSampleParams> {
    shared_fused_params(rows.iter().map(|row| Some(row.fused_gate())))
}

/// Sample and finish every row of `logits` (`[B, 1, vocab]`, one row per
/// `rows` entry, in order).
///
/// A batch of more than one row takes the fused path when [`fused_params`]
/// admits it, else the per-row chain: mask, draw, eval, override, logprobs,
/// matcher, finish. A batch of one always runs the per-row chain, the path
/// the single-sequence decode has always taken.
///
/// Both paths evaluate the sampled tokens through the fallible boundary
/// before reading them back (#822). The fused draw is one evaluation for the
/// whole batch, so a throw there fails every row of it with
/// [`RowError::Eval`]; the per-row chain fails only the row that threw.
pub fn sample_and_finish<H: StepRowHooks>(
    logits: &MlxArray,
    rows: &mut [StepRow<'_, H>],
) -> Vec<RowOutcome> {
    if rows.len() > 1
        && let Some(params) = fused_params(rows)
    {
        let tokens = batched_fused_sample_tokens(logits, &params, &row_biases(rows));
        if let Err(err) = crate::try_eval(&tokens) {
            let msg = err.to_string();
            return rows
                .iter()
                .map(|row| RowOutcome::failed(row.seq_id, RowError::Eval(msg.clone())))
                .collect();
        }
        return rows
            .iter_mut()
            .zip(tokens_to_host(&tokens))
            .map(|(row, token)| finish_row(row, token, token, None, false))
            .collect();
    }
    rows.iter_mut()
        .enumerate()
        .map(|(i, row)| sample_and_finish_row(&row_logits(logits, i), row))
        .collect()
}

/// The per-row chain for one row's `[1, 1, vocab]` logits.
///
/// Used by the per-row branch of [`sample_and_finish`] and by a prefill
/// completion, whose first token takes the same chain over the prefill's
/// last-position logits.
pub fn sample_and_finish_row<H: StepRowHooks>(
    logits: &MlxArray,
    row: &mut StepRow<'_, H>,
) -> RowOutcome {
    let want_distribution =
        row.logprobs.enabled && row.logprobs.source == LogprobSource::PostSampling;
    // The one per-row sampling step (#2169): the structured-output mask runs
    // on the row's logits before the chain; a mask failure is a clean row
    // error rather than silent non-conforming output.
    let draw = row.sampler.draw(
        logits,
        row.sampling,
        row.token_history,
        row.hooks.logit_mask(),
        want_distribution,
    );
    let TokenDraw {
        token: token_arr,
        adjusted_logits,
        distribution,
    } = match draw {
        Ok(draw) => draw,
        Err(msg) => return RowOutcome::failed(row.seq_id, RowError::Structured(msg)),
    };
    // #822: force-evaluate the sampled token through the fallible boundary.
    // On an MLX throw only this row fails; the infallible readback below
    // would otherwise re-trigger the throw and abort the process.
    if let Err(err) = crate::try_eval(&token_arr) {
        return RowOutcome::failed(row.seq_id, RowError::Eval(err.to_string()));
    }
    let sampled = ffi::item_i32(&token_arr);
    // The override first, so an override skips the log-softmax work: the
    // payload for the sampled token would be dropped anyway because the
    // emitted token differs. `resolve` confirms the emitted token with the
    // sampler feedback state (#1485).
    let (sampled, token) = row
        .sampler
        .resolve(sampled, |t| row.hooks.override_token(t));
    let logprobs = if token == sampled {
        match row.logprobs.source {
            LogprobSource::PostSampling => distribution
                .as_ref()
                .map(|p| compute_post_sampling_probs(p, sampled, row.logprobs.top_k)),
            LogprobSource::RawModel if row.logprobs.enabled => {
                let raw_row = ffi::slice_last_logits(logits);
                compute_logprobs(&raw_row, sampled, row.logprobs)
            }
            _ => compute_logprobs(&adjusted_logits, sampled, row.logprobs),
        }
    } else {
        None
    };
    // The matcher advances with the pre-override token: its mask described
    // the unaltered logits, and a forced close tag would be outside its set.
    let structured_stopped = match row.hooks.consume_sampled(sampled) {
        Ok(stopped) => stopped,
        Err(msg) => return RowOutcome::failed(row.seq_id, RowError::Structured(msg)),
    };
    row.hooks.resolved(sampled, token);
    finish_row(row, sampled, token, logprobs, structured_stopped)
}

/// Run the shared finish step (#2168) for an already sampled `token`.
///
/// Also the collect half of a pipelined step: the lookahead's device-side
/// fused draw is read back on the host and each row finishes here.
pub fn finish_row<H: StepRowHooks>(
    row: &mut StepRow<'_, H>,
    sampled: i32,
    token: i32,
    logprobs: Option<TokenLogprobData>,
    structured_stopped: bool,
) -> RowOutcome {
    row.hooks.set_logprobs(logprobs);
    let history = row
        .sampling
        .needs_token_history()
        .then_some(&mut *row.token_history);
    let finish = finish_step(
        FinishInput {
            token,
            eos: row.eos,
            generated: row.generated,
            history,
            max_tokens: row.max_tokens,
            structured_stopped,
            loop_detection: &row.sampling.loop_detection,
        },
        &mut row.hooks,
    );
    RowOutcome {
        seq_id: row.seq_id,
        sampled,
        token,
        finish,
        error: None,
    }
}

/// Every row's token bias in row order, for the fused dispatch points.
pub fn row_biases<'r, H: StepRowHooks>(rows: &'r [StepRow<'_, H>]) -> Vec<&'r TokenBiasMap> {
    rows.iter().map(|r| &r.sampling.token_bias).collect()
}

/// The `[B]` host tokens of a device-side fused draw: `fused_sample`
/// returns a row-contiguous `uint32` array whose raw bytes are read as `i32`,
/// exact for any token id in `0..vocab_size`.
///
/// Uses [`ffi::array_evaluated_bytes`] (a per-array `eval`, no `contiguous()`
/// op) rather than `array_to_raw_bytes`: in the steady lookahead pipeline the
/// next forward is already scheduled on the same stream before this read,
/// and a fresh `contiguous()` op would queue behind it and collapse the
/// overlap. This reader waits only on the token array's own completion.
pub fn tokens_to_host(tokens: &MlxArray) -> Vec<i32> {
    ffi::array_evaluated_bytes(tokens)
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `logits` sliced to a `[1, 1, vocab]` row, the shape the per-row chain
/// expects (a prefill's last-position logits already have it).
pub fn row_logits(logits: &MlxArray, i: usize) -> UniquePtr<MlxArray> {
    ffi::slice(logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX])
}
