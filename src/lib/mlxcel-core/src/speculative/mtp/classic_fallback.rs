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

//! Acceptance-aware fallback to classic decode in the B = 1 MTP round loop
//! (issue #3425).
//!
//! On Inkling-Small the native MTP head emits 1.43 to 1.51 tokens per round,
//! and an exact 2-row round costs about 36 ms against a 22 ms decode step,
//! so every drafted round loses to classic decode. The fallback reuses the
//! DFlash width policy ([`ExpectedThroughputWidth`]): each round picks the
//! width in `1..=block_size` with the most expected tokens per measured
//! millisecond, width 1 being classic decode. When classic decode wins, the
//! round loop runs a pipelined run of classic steps (step `i + 1` is queued
//! on the lazy token of step `i` before step `i` is read, as classic decode
//! does), and drafts again after the policy's re-probe period, which doubles
//! after each re-try that loses ([`PROBE_BACKOFF_CAP`]).
//!
//! The drafter does not see the classic steps as they run. Their hidden
//! rows and tokens are kept, and the next drafted round first hands them to
//! [`Drafter::extend_history`] in one forward, so a long classic run costs
//! the drafter nothing until it drafts again.
//!
//! It runs only for a target that declares
//! [`MtpTarget::exact_at_every_verify_width`] (a block of any width up to
//! the requested one, and a single classic step, reproduce classic decode
//! bit for bit) with a drafter that declares
//! [`Drafter::extends_history_without_draft`], under the default
//! `MLXCEL_MTP_BLOCK_CONTROLLER` and without per-token logprobs.
//! `MLXCEL_MTP_BLOCK_CONTROLLER=requested` keeps the requested width every
//! round, as before.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cxx::UniquePtr;

use super::target::{MtpClassicStep, MtpTarget};
use crate::drafter::Drafter;
use crate::drafter::dflash::width_policy::ExpectedThroughputWidth;
use crate::ffi::{self, MlxArray};
use crate::generate::SamplingConfig;

/// Largest classic run between two re-tries of the drafter. A re-try is a
/// full drafted round (about 14 ms above a decode step on Inkling-Small)
/// plus the drafter's catch-up forward, so at the cap the re-tries cost
/// under 1% of a classic run.
pub(crate) const PROBE_BACKOFF_CAP: usize = 64;

/// Classic steps one call of the round loop runs at most, so a server tick
/// that runs one session step returns to the scheduler about as often as a
/// drafted round does.
pub(crate) const MAX_CLASSIC_STEPS_PER_CALL: usize = 16;

/// What a run of classic steps produced.
pub(crate) struct ClassicRun {
    /// One token per completed step, in order.
    pub(crate) tokens: Vec<i32>,
    /// The hidden row of each completed step's input position.
    pub(crate) hidden: Vec<UniquePtr<MlxArray>>,
    /// The last token is an EOS id.
    pub(crate) hit_eos: bool,
    /// Steady per-step time from the first read-back on, so the first
    /// step's warm-up and missing overlap are left out; `None` for a run of
    /// one step.
    pub(crate) step_ms: Option<f64>,
    /// Host time spent building and queuing the steps, ms.
    pub(crate) build_ms: f64,
    /// Host time spent waiting for the tokens, ms.
    pub(crate) wait_ms: f64,
}

/// Per-session state of the fallback.
pub(crate) struct ClassicFallback {
    policy: ExpectedThroughputWidth,
    /// Classic-step tokens the drafter has not seen yet.
    pending_tokens: Vec<i32>,
    /// Their hidden rows, aligned with `pending_tokens`.
    pending_hidden: Vec<UniquePtr<MlxArray>>,
    /// Classic steps this session ran, for the round-loop diagnostics.
    pub(crate) classic_steps: usize,
    /// Wall time of the drafter catch-up forwards, ms.
    pub(crate) catch_up_ms: f64,
    /// Host time building classic steps and waiting for their tokens, ms.
    pub(crate) classic_build_ms: f64,
    pub(crate) classic_wait_ms: f64,
    warned_catch_up: bool,
}

impl std::fmt::Debug for ClassicFallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassicFallback")
            .field("policy", &self.policy)
            .field("pending_tokens", &self.pending_tokens.len())
            .field("classic_steps", &self.classic_steps)
            .finish()
    }
}

impl ClassicFallback {
    /// The fallback for one session, when the target, the drafter and the
    /// request admit it.
    pub(crate) fn for_session<T: MtpTarget>(
        target: &T,
        drafter: &dyn Drafter,
        block_size: usize,
        allowed: bool,
    ) -> Option<Self> {
        (allowed && target.exact_at_every_verify_width() && drafter.extends_history_without_draft())
            .then(|| Self {
                policy: ExpectedThroughputWidth::new(block_size)
                    .with_probe_backoff(PROBE_BACKOFF_CAP),
                pending_tokens: Vec::new(),
                pending_hidden: Vec::new(),
                classic_steps: 0,
                catch_up_ms: 0.0,
                classic_build_ms: 0.0,
                classic_wait_ms: 0.0,
                warned_catch_up: false,
            })
    }

    /// The width of the next round, 1 meaning classic steps. `remaining`
    /// counts the bonus position, as the round loop passes it.
    pub(crate) fn decide(&self, remaining: usize) -> usize {
        self.policy.decide(remaining)
    }

    /// How many classic steps to run now, at most `budget`.
    pub(crate) fn classic_run_len(&self, budget: usize) -> usize {
        self.policy
            .classic_run_len()
            .min(MAX_CLASSIC_STEPS_PER_CALL)
            .min(budget)
    }

    /// Feed a drafted round: its width, accepted drafts and wall time.
    pub(crate) fn record_round(&mut self, width: usize, accepted: usize, round_ms: f64) {
        self.policy.record_round(width, accepted, round_ms);
    }

    /// Feed a classic run and keep its rows for the drafter.
    pub(crate) fn record_classic_run(&mut self, run: ClassicRun) {
        self.classic_build_ms += run.build_ms;
        self.classic_wait_ms += run.wait_ms;
        self.policy
            .record_classic_steps(run.tokens.len(), run.step_ms);
        self.classic_steps += run.tokens.len();
        self.pending_tokens.extend(run.tokens);
        self.pending_hidden.extend(run.hidden);
    }

    /// Hand the classic steps the drafter has not seen to it, before a
    /// drafted round. A failure is logged once and leaves the drafter with
    /// a shorter history, which costs acceptance only.
    pub(crate) fn catch_up(&mut self, drafter: &mut dyn Drafter, sampling: &SamplingConfig) {
        if self.pending_tokens.is_empty() {
            return;
        }
        let started = Instant::now();
        let rows: Vec<&MlxArray> = self.pending_hidden.iter().map(|row| &**row).collect();
        let hidden = crate::ops::concatenate_many(&rows, 1);
        let outcome = drafter.extend_history(&hidden, &self.pending_tokens, sampling);
        self.pending_tokens.clear();
        self.pending_hidden.clear();
        self.catch_up_ms += started.elapsed().as_secs_f64() * 1000.0;
        if let Err(err) = outcome
            && !std::mem::replace(&mut self.warned_catch_up, true)
        {
            tracing::warn!(
                "MTP drafter could not take the classic steps: {err}; drafting from a shorter history"
            );
        }
    }

    /// `width:rounds@cost_ms` per width and the acceptance estimates.
    pub(crate) fn summary(&self) -> String {
        format!(
            "{} probe_period {}",
            self.policy.summary(),
            self.policy.probe_period()
        )
    }
}

/// Run up to `steps` classic steps from `bonus`, stopping after an EOS token
/// or once `cancel` is set. Step `i + 1` is queued on the lazy token of step
/// `i` before step `i` is read, as classic decode does. A step queued past an
/// EOS is dropped unread; the session ends there, so the row it appended is
/// never used. `None` when the target has no classic step.
pub(crate) fn run_classic_steps<T: MtpTarget>(
    target: &T,
    bonus: i32,
    steps: usize,
    eos_token_ids: &[i32],
    cancel: &AtomicBool,
    sampling: &SamplingConfig,
) -> Option<ClassicRun> {
    let mut run = ClassicRun {
        tokens: Vec::with_capacity(steps),
        hidden: Vec::with_capacity(steps),
        hit_eos: false,
        step_ms: None,
        build_ms: 0.0,
        wait_ms: 0.0,
    };
    let mut first_read: Option<Instant> = None;
    let mut input = ffi::from_slice_i32(&[bonus], &[1, 1]);
    let mut in_flight: Option<MtpClassicStep> = None;
    for step in 0..=steps {
        let launched = if step < steps && !cancel.load(Ordering::Relaxed) {
            let started = Instant::now();
            // A target without a classic step builds nothing, so refusing on
            // the first step leaves its cache untouched; a later refusal
            // (not one the trait allows) ends the run after the steps
            // already queued.
            let launched = match target.classic_step(&input, sampling) {
                Some(next) => {
                    // SAFETY: both arrays live until the end of this block,
                    // and the bridge consumes the pointer slice
                    // synchronously.
                    unsafe {
                        let evals: [*const MlxArray; 2] = [&*next.token, &*next.hidden];
                        ffi::async_eval_all(&evals);
                    }
                    input = ffi::reshape(&next.token, &[1, 1]);
                    Some(next)
                }
                None if step == 0 => return None,
                None => None,
            };
            run.build_ms += started.elapsed().as_secs_f64() * 1000.0;
            launched
        } else {
            None
        };
        if let Some(previous) = in_flight.take() {
            let waited = Instant::now();
            let value = crate::drafter::dflash::materialize_argmax_i32_vec(&previous.token, 1)[0];
            run.wait_ms += waited.elapsed().as_secs_f64() * 1000.0;
            first_read.get_or_insert_with(Instant::now);
            run.tokens.push(value);
            run.hidden.push(previous.hidden);
            if eos_token_ids.contains(&value) {
                run.hit_eos = true;
                break;
            }
        }
        match launched {
            Some(next) => in_flight = Some(next),
            None => break,
        }
    }
    if let Some(first) = first_read
        && run.tokens.len() > 1
    {
        run.step_ms = Some(first.elapsed().as_secs_f64() * 1000.0 / (run.tokens.len() - 1) as f64);
    }
    Some(run)
}
