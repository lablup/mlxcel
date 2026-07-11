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

//! Tick-cooperative B=1 MTP speculative slices (issue #734).
//!
//! The run-to-completion burst ([`super::speculative_burst`]) serves a
//! speculative request's entire prefill + decode lifecycle inside ONE
//! scheduler tick, so every concurrent classic-decode row stalls for the
//! whole burst (the head-of-line block measured by
//! `BurstFinalized::burst_wall_ms`, issue #638 / PR #733). This module
//! removes that block for the B=1 MTP arm on the Gemma 4 family: the
//! request is served as a sequence of SLICES, one per scheduler tick:
//! slice 0 is the prefill + seed
//! ([`mlxcel_core::speculative::mtp::MtpGenerator::begin_session`]), and
//! every later slice is exactly one speculative round
//! ([`mlxcel_core::speculative::mtp::MtpGenerator::step_session`]). The
//! scheduler re-enters its tick loop between slices, so classic rows
//! advance between rounds and the HOL stall drops to about one round's
//! worker occupancy.
//!
//! ## Why the generator is reconstructed every tick
//!
//! `MtpGenerator<T>` owns its target adapter, and the Gemma 4 adapters
//! (`Gemma4MtpTargetAdapter` and friends) BORROW the model, so the
//! generator is self-referential with respect to the scheduler and cannot
//! be stashed across ticks (the blocker PR #733 documented). What CAN be
//! stashed is everything the round loop actually carries between rounds:
//!
//! - [`MtpSliceJob`] owns the request's `SequenceInfo`, the generator's
//!   [`MtpSessionState`] (whose `MtpVerifyOutput` holds owned MLX array
//!   handles plus plain integers, no model borrows), the cross-slice
//!   [`BurstStreamState`], and the bound drafter handle.
//! - The per-sequence KV cache lives in the model's own sequence slot
//!   (`ModelOwnedSequenceState[seq_id]`), untouched between ticks.
//! - The B=1 adapters are stateless views (wrapper ref + seq_id + config
//!   scalars), so reconstructing one per tick is free.
//!
//! On resume, `step_session` re-arms the drafter's shared K/V from the
//! stored `MtpVerifyOutput` at the top of the round, which is why the
//! drafter is held here WITHOUT [`Drafter::reset`] between slices; the
//! resetting [`super::speculative_burst::WorkerDrafterSlot::return_drafter`]
//! runs only once, when the whole session finishes.
//!
//! ## Streaming
//!
//! Each slice streams its round's accepted tokens immediately through the
//! shared [`stream_burst_tokens`] helper (the same per-token
//! thinking-budget / EOS / logprobs pipeline the run-to-completion burst
//! drives at finalize), so the tick-cooperative path streams per round
//! instead of emitting the whole request in one lump, while keeping the
//! client-visible event stream byte-identical.
//!
//! ## Scope
//!
//! B=1 MTP on the Gemma 4 family only. The DFlash arm and the B>1 batched
//! arms keep the legacy run-to-completion burst (their generators do not
//! expose a resumable step API yet), and `MLXCEL_MTP_TICK_SLICE=0` forces
//! the MTP arm back onto the legacy burst as an operator escape hatch.

use std::time::{Duration, Instant};

use mlxcel_core::drafter::Drafter;
use mlxcel_core::generate::GenerationStats;
use mlxcel_core::speculative::mtp::target::MtpTarget;
use mlxcel_core::speculative::mtp::{MtpAcceptanceSummary, MtpGenerator, MtpSessionState};

use super::sequence::SequenceInfo;
use super::speculative_burst::{BurstStreamState, begin_burst_stream, stream_burst_tokens};

/// Whether the tick-cooperative MTP slice is enabled. Default on;
/// `MLXCEL_MTP_TICK_SLICE=0|false|no|off` forces the B=1 MTP arm back onto
/// the legacy run-to-completion burst (operator escape hatch; every other
/// env gate keeps its existing meaning).
pub(crate) fn mtp_tick_slice_enabled() -> bool {
    mtp_tick_slice_default(std::env::var("MLXCEL_MTP_TICK_SLICE").ok().as_deref())
}

/// Pure decision core of [`mtp_tick_slice_enabled`], separated for unit
/// testing.
pub(crate) fn mtp_tick_slice_default(env_override: Option<&str>) -> bool {
    match env_override {
        Some(v) => !matches!(v, "0" | "false" | "FALSE" | "no" | "off"),
        None => true,
    }
}

/// Tick-arbitration between an in-flight speculative slice and the classic
/// scheduler actions (decode / prefill / chunked prefill).
///
/// `yielded_last_tick` is the scheduler's fairness flag: `true` when the
/// previous executed action was a speculative round. `others_have_work` is
/// whether any classic action has work this tick (active rows, queued
/// prefills, or an in-progress chunked prefill).
///
/// Returns `true` when this tick should run one speculative round. The
/// scheme is strict alternation under contention: round, classic action,
/// round, classic action, and so on. A classic row's inter-token gap is
/// bounded by about one speculative round, and the speculative request's
/// inter-round gap is bounded by about one classic action. Without
/// contention the slice takes every tick (and, symmetrically, never blocks
/// an Idle wait, because the slice IS work).
pub(crate) fn slice_takes_tick(yielded_last_tick: bool, others_have_work: bool) -> bool {
    !others_have_work || !yielded_last_tick
}

/// Terminal accounting of a finished slice session.
pub(crate) struct MtpSliceFinish {
    /// Generation stats from `finish_session` (prefill + summed per-slice
    /// decode wall; cross-tick gaps excluded).
    pub(crate) stats: GenerationStats,
    /// Acceptance summary for the adaptive MTP policy, identical in its
    /// count fields to what a run-to-completion `generate` call would have
    /// produced for the same rounds.
    pub(crate) summary: Option<MtpAcceptanceSummary>,
}

/// One in-flight tick-cooperative B=1 MTP request.
///
/// Owned by the scheduler (`BatchScheduler::speculative_slice`) between
/// ticks. Every field is owned (nothing borrows the model), which is the
/// property that lets the job live across ticks while the borrowing
/// generator is reconstructed per tick.
pub(crate) struct MtpSliceJob {
    /// The request. Its `SequenceInfo::cancelled` flag is an
    /// `Arc<AtomicBool>` shared with the connection handler, so client
    /// disconnects reach the session between (and within) slices.
    pub(crate) seq: SequenceInfo,
    /// Resumable generator session. `None` only after the session finished
    /// (it is consumed by `finish_session`).
    session: Option<MtpSessionState>,
    /// Cross-slice streaming state (EOS classification, budget, thinking
    /// budget interplay).
    pub(crate) stream: BurstStreamState,
    /// The bound drafter, held WITHOUT reset between slices. `None` only
    /// transiently inside a slice (while the generator owns it) and after
    /// [`Self::take_drafter`] at finalize.
    drafter: Option<Box<dyn Drafter>>,
    /// Requested draft block size (K), fixed for the session.
    pub(crate) block_size: usize,
    /// Adopted prompt-cache prefix length (issue #518), kept so the
    /// per-tick adapter reconstruction is bit-identical to slice 0's.
    /// Only `prefill_and_seed` consumes it, so this is documentation-grade
    /// state after slice 0.
    pub(crate) prefill_start_offset: usize,
    /// Slices executed so far (slice 0 included).
    pub(crate) slices: usize,
    /// Maximum single-slice wall-clock (ms): the realized per-tick HOL
    /// bound this request imposed on concurrent rows.
    pub(crate) max_slice_wall_ms: f64,
    /// Cumulative slice wall-clock (ms): total worker occupancy.
    pub(crate) total_slice_wall_ms: f64,
    /// Terminal accounting; `Some` once the session (or the stream layer)
    /// finished. [`Self::finished`] gates the scheduler's finalize.
    pub(crate) finish: Option<MtpSliceFinish>,
}

impl MtpSliceJob {
    /// Whether the request finished and must be finalized.
    pub(crate) fn finished(&self) -> bool {
        self.finish.is_some()
    }

    /// Take the drafter for the end-of-session return to the
    /// [`super::speculative_burst::WorkerDrafterSlot`] (which resets it).
    pub(crate) fn take_drafter(&mut self) -> Option<Box<dyn Drafter>> {
        self.drafter.take()
    }

    fn record_slice(&mut self, wall: Duration) {
        let wall_ms = wall.as_secs_f64() * 1000.0;
        self.slices += 1;
        self.total_slice_wall_ms += wall_ms;
        if wall_ms > self.max_slice_wall_ms {
            self.max_slice_wall_ms = wall_ms;
        }
    }
}

/// Run slice 0 (prefill + seed + first bonus) and build the cross-tick job.
///
/// The caller has already gated the request (variant, adopted-prefix,
/// drafter load + compat + bind, the same gates `run_mtp_burst` applies)
/// and transitioned it to `Prefilling`. Generic over `T: MtpTarget` so the
/// scheduler can pass any Gemma 4 adapter and the unit tests can pass the
/// mock target.
///
/// `token_history` is the history-dependent-penalty context for the first
/// bonus (the caller computes `initial_token_history(&prompt, ..)`), and
/// `model_eos_token_ids` seeds the stream layer's merged EOS set, both
/// exactly as on the run-to-completion path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn begin_slice_session<T: MtpTarget>(
    target: T,
    drafter: Box<dyn Drafter>,
    mut seq: SequenceInfo,
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
    model_eos_token_ids: Vec<i32>,
    block_size: usize,
    profile_probe_rounds: usize,
    prefill_start_offset: usize,
    token_history: &[i32],
) -> MtpSliceJob {
    let slice_start = Instant::now();
    let max_tokens = seq.max_tokens.max(1);
    let prompt = seq.prompt_tokens.clone();

    let mut generator = MtpGenerator::new(target, drafter, block_size)
        .with_profile_probe_rounds(profile_probe_rounds);
    let (session, first) = generator.begin_session(
        &prompt,
        max_tokens,
        &seq.sampling,
        token_history,
        &seq.logprobs_config,
    );

    // Stream the first bonus immediately: per-slice streaming starts at
    // slice 0, unlike the legacy burst which lumps every token at finalize.
    let mut stream = begin_burst_stream(model_eos_token_ids, &seq);
    let stream_done = stream_burst_tokens(
        tokenizer,
        &mut seq,
        &mut stream,
        &first.new_tokens,
        &first.new_logprobs,
    );

    let mut job = MtpSliceJob {
        seq,
        session: Some(session),
        stream,
        drafter: None,
        block_size,
        prefill_start_offset,
        slices: 0,
        max_slice_wall_ms: 0.0,
        total_slice_wall_ms: 0.0,
        finish: None,
    };
    settle_slice(&mut job, generator, first.finished || stream_done);
    job.record_slice(slice_start.elapsed());
    job
}

/// Run exactly ONE speculative round of an in-flight job.
///
/// The caller reconstructs the target adapter for this tick and passes it
/// in; the drafter comes out of the job (held unreset since the previous
/// slice) and goes back in afterwards. `step_session` re-arms the
/// drafter's shared K/V from the stored verify output at the top of the
/// round, so no drafter-side state needs to have survived the gap.
pub(crate) fn step_slice_session<T: MtpTarget>(
    target: T,
    job: &mut MtpSliceJob,
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
) {
    debug_assert!(!job.finished(), "step_slice_session on a finished job");
    let slice_start = Instant::now();
    // Invariant: while `finish` is `None`, the job holds both the session
    // and the drafter (the only code that takes them is this function and
    // the finalize path, and both restore or finish).
    let drafter = job
        .drafter
        .take()
        .expect("MtpSliceJob invariant: drafter present while in flight");
    let mut session = job
        .session
        .take()
        .expect("MtpSliceJob invariant: session present while in flight");

    let mut generator = MtpGenerator::new(target, drafter, job.block_size);
    let out = generator.step_session(
        &mut session,
        &job.seq.sampling,
        &job.seq.cancelled,
        &job.seq.logprobs_config,
    );
    let stream_done = stream_burst_tokens(
        tokenizer,
        &mut job.seq,
        &mut job.stream,
        &out.new_tokens,
        &out.new_logprobs,
    );

    job.session = Some(session);
    settle_slice(job, generator, out.finished || stream_done);
    job.record_slice(slice_start.elapsed());
}

/// Common tail of a slice: either finish the session (stamping the stats +
/// acceptance summary) or park the session + drafter for the next tick.
///
/// A `finished` signal can come from the generator (EOS / budget / cancel /
/// drafter failure) or from the stream layer (e.g. a thinking-budget forced
/// `</think>` that is an EOS id); in the latter case the session is
/// finished eagerly so its diagnostics are logged and the remaining rounds
/// never run, mirroring the legacy burst's client-side truncation.
fn settle_slice<T: MtpTarget>(
    job: &mut MtpSliceJob,
    mut generator: MtpGenerator<T>,
    finished: bool,
) {
    if finished {
        let session = job
            .session
            .take()
            .expect("settle_slice: session present when finishing");
        let (stats, summary) = generator.finish_session(session);
        job.finish = Some(MtpSliceFinish { stats, summary });
    }
    let (_target, drafter) = generator.into_parts();
    job.drafter = Some(drafter);
}

#[cfg(test)]
mod tests {
    use super::{mtp_tick_slice_default, slice_takes_tick};

    #[test]
    fn tick_slice_env_gate_default_on_and_off_values() {
        assert!(mtp_tick_slice_default(None), "default must be on");
        for on in ["1", "true", "yes", "on", "anything-else"] {
            assert!(mtp_tick_slice_default(Some(on)), "{on} must enable");
        }
        for off in ["0", "false", "FALSE", "no", "off"] {
            assert!(!mtp_tick_slice_default(Some(off)), "{off} must disable");
        }
    }

    /// Scheduler-level interleaving regression (issue #734 acceptance):
    /// with one speculative slice in flight and classic work present, the
    /// tick arbitration must alternate strictly, so (a) the classic rows'
    /// inter-token gap is bounded by one speculative round and (b) the
    /// speculative request's inter-round gap is bounded by one classic
    /// action. This simulates the scheduler's `run()` flag protocol over
    /// the REAL `slice_takes_tick` policy (the same function
    /// `decide_action` consults), not a reimplementation.
    #[test]
    fn slice_alternates_strictly_with_classic_work() {
        let mut yielded = false;
        let mut history: Vec<&'static str> = Vec::new();
        for _ in 0..20 {
            if slice_takes_tick(yielded, /* others_have_work */ true) {
                history.push("round");
                yielded = true; // run(): SpeculativeRound arm sets the flag
            } else {
                history.push("classic");
                yielded = false; // run(): Prefill/Decode arms clear the flag
            }
        }
        // Strict alternation: no two consecutive ticks of the same kind.
        for pair in history.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "ticks must alternate under contention: {history:?}"
            );
        }
        // Bounded gaps in both directions.
        let max_gap = |kind: &str| {
            let mut gap = 0usize;
            let mut max = 0usize;
            for t in &history {
                if *t == kind {
                    gap = 0;
                } else {
                    gap += 1;
                    max = max.max(gap);
                }
            }
            max
        };
        assert!(
            max_gap("classic") <= 1,
            "classic rows must wait at most one speculative round: {history:?}"
        );
        assert!(
            max_gap("round") <= 1,
            "the speculative request must wait at most one classic action: {history:?}"
        );
    }

    /// Without classic work the slice takes every tick (never Idle-blocks),
    /// regardless of the fairness flag.
    #[test]
    fn slice_takes_every_tick_without_classic_work() {
        assert!(slice_takes_tick(false, false));
        assert!(slice_takes_tick(true, false));
    }

    /// Right after a classic action the slice always gets the next tick.
    #[test]
    fn slice_takes_tick_after_classic_action() {
        assert!(slice_takes_tick(false, true));
        assert!(!slice_takes_tick(true, true));
    }
}
