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

//! Classic decode steps inside the B = 1 round loop (issue #3410).
//!
//! When [`super::width_policy::ExpectedThroughputWidth`] finds that no draft
//! width pays for itself, the round loop runs one-row verify forwards with
//! no draft. Run one at a time, each would wait for its own result before
//! the next graph is built; on Inkling 2 Small that made a step 58 ms
//! against classic decode's 52. So a run is pipelined the way classic
//! decode is: step `i + 1` is built on the lazy argmax of step `i` and
//! queued before step `i` is read back. Each step's captured hidden row is
//! kept for the next draft.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cxx::UniquePtr;

use super::round_loop::{SpeculativeTarget, argmax_logits_to_array, per_position_logprobs};
use crate::ffi::{self, MlxArray};
use crate::sampling::{LogprobsConfig, TokenLogprobData};

/// What a run of classic steps produced.
pub(super) struct ClassicRun {
    /// One token per completed step, in order.
    pub(super) tokens: Vec<i32>,
    /// Aligned with `tokens` when logprobs are enabled; empty otherwise.
    pub(super) logprobs: Vec<Option<TokenLogprobData>>,
    /// The captured hidden row `[1, 1, dim]` of each completed step.
    pub(super) hidden: Vec<UniquePtr<MlxArray>>,
    /// The last token is an EOS id.
    pub(super) hit_eos: bool,
    /// Wall time of the whole run.
    pub(super) elapsed_ms: f64,
    /// Steady per-step time, measured from the first read-back so the first
    /// step's warm-up and missing overlap are left out; `None` for a run of
    /// one step.
    pub(super) step_ms: Option<f64>,
}

/// Run up to `steps` classic steps from `bonus`, stopping after an EOS
/// token or once `cancel` is set. A step queued past an EOS is dropped
/// unread; generation ends there, so the row it appended is never used.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_classic_steps<T: SpeculativeTarget>(
    target: &T,
    caches: &mut [T::Cache],
    capture_layer_ids: &[usize],
    bonus: i32,
    steps: usize,
    eos_token_ids: &[i32],
    cancel: &AtomicBool,
    logprobs_config: &LogprobsConfig,
) -> ClassicRun {
    let started = Instant::now();
    let mut run = ClassicRun {
        tokens: Vec::with_capacity(steps),
        logprobs: Vec::new(),
        hidden: Vec::with_capacity(steps),
        hit_eos: false,
        elapsed_ms: 0.0,
        step_ms: None,
    };
    let mut first_read: Option<Instant> = None;
    let mut input = ffi::from_slice_i32(&[bonus], &[1, 1]);
    let mut in_flight: Option<(T::VerifyOut, UniquePtr<MlxArray>, UniquePtr<MlxArray>)> = None;
    for step in 0..=steps {
        let launched = (step < steps && !cancel.load(Ordering::Relaxed)).then(|| {
            let out = target.verify_forward_with_capture_layers(&input, caches, capture_layer_ids);
            let token = ffi::astype(
                &argmax_logits_to_array(target.verify_logits(&out), 1),
                crate::dtype::INT32,
            );
            let hidden = target.concat_hidden_for_drafter(&out);
            // SAFETY: both arrays live until this closure returns, and the
            // bridge consumes the pointer slice synchronously.
            unsafe {
                let evals: [*const MlxArray; 2] = [&*token, &*hidden];
                ffi::async_eval_all(&evals);
            }
            input = ffi::reshape(&token, &[1, 1]);
            (out, token, hidden)
        });
        if let Some((out, token, hidden)) = in_flight.take() {
            let value = super::materialize_argmax_i32_vec(&token, 1)[0];
            first_read.get_or_insert_with(Instant::now);
            if logprobs_config.enabled {
                let lp =
                    per_position_logprobs(target.verify_logits(&out), &[value], logprobs_config);
                run.logprobs.push(lp.and_then(|v| v.into_iter().next()));
            }
            run.tokens.push(value);
            run.hidden.push(hidden);
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
    run.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Some(first) = first_read
        && run.tokens.len() > 1
    {
        run.step_ms = Some(first.elapsed().as_secs_f64() * 1000.0 / (run.tokens.len() - 1) as f64);
    }
    run
}
