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
//! stored `MtpVerifyOutput` at the top of the round, so no drafter-side
//! state needs to survive between rounds. While a job holds the slot, the
//! drafter is held in the job WITHOUT [`Drafter::reset`] between slices;
//! the resetting
//! [`super::speculative_burst::WorkerDrafterSlot::return_drafter`] runs
//! when the whole session finishes, and (since issue #746) at every
//! park boundary when the slot rotates to another grantee, which is safe
//! because the MTP assistant drafter's reset is the trait default no-op
//! (see [`MtpSliceJob::attach_drafter`]).
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
//!
//! ## Slot rotation (issue #746)
//!
//! A slice holds the worker's single speculative slot for its whole
//! generation, which spans many ticks. Pre-#746, a second
//! speculative-eligible request arriving mid-slice fell back to classic
//! decode permanently, so one long stream monopolized speculative
//! acceleration. Now the slot is GRANTED in bounded turns: the scheduler
//! keeps a small backlog ([`SliceBacklogEntry`]) of parked in-flight jobs
//! and admitted waiters, and under contention the active job is parked at
//! a round boundary once its grant budget
//! (`MLXCEL_MTP_SLICE_GRANT_ROUNDS`, [`mtp_slice_grant_rounds`]) is spent,
//! releasing the drafter for the next grantee. Without contention the
//! budget never binds and behavior is identical to #734. The drafter
//! handoff is routed through the existing
//! [`super::speculative_burst::WorkerDrafterSlot`] return/take plumbing;
//! see [`MtpSliceJob::attach_drafter`] for why that is correct.

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

/// Default grant budget for one hold of the speculative slice slot under
/// contention (issue #746), counted in executed slices (slice 0, the
/// prefill + seed, counts as the first slice of an admission grant).
pub(crate) const MTP_SLICE_GRANT_ROUNDS_DEFAULT: usize = 8;

/// Cap on the slice backlog (parked jobs + waiters) queued behind the
/// active job (issue #746).
///
/// Rationale: a waiter streams nothing until its slice 0 runs, so its
/// worst-case pre-first-token wait is about `cap x budget` speculative
/// slices (each grant ahead of it can spend the full budget) plus the
/// strict-alternation classic ticks interleaved between them. With the
/// defaults (cap 2, budget 8) that is ~16 rounds of wall clock, small
/// against a long generation and repaid by speculative acceleration over
/// the rest of the stream. Beyond the cap, a request falls back to
/// classic decode exactly as pre-#746, so the bound cannot grow with
/// offered load.
pub(crate) const MTP_SLICE_BACKLOG_CAP: usize = 2;

/// Grant budget for one hold of the speculative slice slot (issue #746):
/// `MLXCEL_MTP_SLICE_GRANT_ROUNDS`, default
/// [`MTP_SLICE_GRANT_ROUNDS_DEFAULT`]. `0` means unbounded (the legacy
/// #734 whole-generation hold, as an operator escape hatch; admission
/// then also stops parking waiters, restoring the pre-#746 classic
/// fallback for concurrent speculative requests).
///
/// Read cadence: once per GRANT (at the two grant-start sites, cached in
/// the scheduler's `speculative_slice_grant_budget`) and once per
/// admission decision, never per round; env access takes std's
/// process-wide ENV lock and allocates, so the per-round expiry check
/// compares against the cached value instead.
pub(crate) fn mtp_slice_grant_rounds() -> usize {
    mtp_slice_grant_rounds_default(
        std::env::var("MLXCEL_MTP_SLICE_GRANT_ROUNDS")
            .ok()
            .as_deref(),
    )
}

/// Pure decision core of [`mtp_slice_grant_rounds`], separated for unit
/// testing. Unparseable values fall back to the default.
pub(crate) fn mtp_slice_grant_rounds_default(env_override: Option<&str>) -> usize {
    env_override
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(MTP_SLICE_GRANT_ROUNDS_DEFAULT)
}

/// Whether the active job's slot grant has expired at a round boundary
/// (issue #746). Binds ONLY under contention: with an empty backlog the
/// job keeps the slot and behavior is byte-identical to #734 (no
/// rotation, the slice takes every tick [`slice_takes_tick`] gives it).
/// A `budget` of 0 never expires (legacy hold escape hatch).
///
/// Free fast-handoff property: the counter keeps counting while the job
/// runs uncontended, so if contention appears after the budget is
/// already spent, the FIRST round boundary rotates immediately and a
/// newcomer waits at most about one round.
pub(crate) fn slice_grant_expired(
    slices_in_grant: usize,
    budget: usize,
    backlog_pending: bool,
) -> bool {
    budget > 0 && backlog_pending && slices_in_grant >= budget
}

/// Whether a speculative-eligible request arriving while the slice slot
/// is busy may join the backlog as a waiter (issue #746) instead of
/// falling back to classic decode. Beyond [`MTP_SLICE_BACKLOG_CAP`], or
/// with rotation disabled (`budget == 0`, under which a waiter could
/// starve for the active job's whole generation), the request takes the
/// pre-#746 classic fallback.
pub(crate) fn slice_backlog_admits(backlog_len: usize, budget: usize) -> bool {
    budget > 0 && backlog_len < MTP_SLICE_BACKLOG_CAP
}

/// Anti-starvation skip cap for the grant order (issue #746 security
/// hardening). A backlog entry that `MTP_SLICE_GRANT_SKIP_CAP` grant
/// decisions have passed over becomes OVERDUE and must be granted next,
/// regardless of priority lane. Without this floor, strict lane
/// precedence would let a sustained stream of higher-lane arrivals
/// starve a lower-lane PARKED job indefinitely; a parked job is not in
/// the active batch, so starving it stalls its client stream completely,
/// losing the #745 "everyone still progresses" floor, and priorities are
/// client-assignable (the `X-Priority` header), so the starvation would
/// be remotely triggerable. With the cap, every entry is granted within
/// a bounded number of grant boundaries.
pub(crate) const MTP_SLICE_GRANT_SKIP_CAP: usize = 2;

/// Index of the next slot grantee in the backlog (issue #746). Each
/// element is the entry's `(priority lane, skipped_grants)` pair in ring
/// order.
///
/// Selection: if any entry is OVERDUE (skipped by at least
/// [`MTP_SLICE_GRANT_SKIP_CAP`] grant decisions), the most-skipped entry
/// wins, earliest in the ring on ties, regardless of lane; this is the
/// anti-starvation floor. Otherwise: priority lane first, then FIFO;
/// parked jobs and waiters share the one ring in park/arrival order, so
/// neither kind can starve the other within a lane, and a lower-lane
/// entry can be overtaken by later higher-lane arrivals only until its
/// skip counter reaches the cap.
pub(crate) fn next_grant_index(
    entries: &[(super::sequence::RequestPriority, usize)],
) -> Option<usize> {
    // Anti-starvation escalation: an overdue entry preempts lane order.
    let mut overdue: Option<(usize, usize)> = None;
    for (idx, &(_, skips)) in entries.iter().enumerate() {
        if skips >= MTP_SLICE_GRANT_SKIP_CAP
            && overdue.is_none_or(|(best_skips, _)| skips > best_skips)
        {
            // Strictly greater keeps the earliest overdue entry on ties.
            overdue = Some((skips, idx));
        }
    }
    if let Some((_, idx)) = overdue {
        return Some(idx);
    }
    // Un-escalated order: priority lane first, FIFO within a lane.
    let mut best: Option<(super::sequence::RequestPriority, usize)> = None;
    for (idx, &(lane, _)) in entries.iter().enumerate() {
        // Strictly greater keeps the earliest entry within a lane.
        if best.is_none_or(|(best_lane, _)| lane > best_lane) {
            best = Some((lane, idx));
        }
    }
    best.map(|(_, idx)| idx)
}

/// One backlog entry waiting for a speculative slice slot grant (issue
/// #746). Owned by the scheduler (`BatchScheduler::speculative_slice_backlog`).
pub(crate) struct SliceBacklogEntry {
    pub(crate) kind: SliceBacklogKind,
    /// Grant decisions that selected some OTHER entry while this one
    /// waited in the ring. Incremented by the scheduler for every
    /// non-selected entry when a grantee is popped; at
    /// [`MTP_SLICE_GRANT_SKIP_CAP`] the entry becomes overdue and
    /// [`next_grant_index`] must select it next. A granted entry leaves
    /// the ring, so re-parking re-enters with a fresh counter of 0 (the
    /// "reset on grant").
    pub(crate) skipped_grants: usize,
}

/// The payload of a [`SliceBacklogEntry`].
pub(crate) enum SliceBacklogKind {
    /// An in-flight job parked at a grant boundary. Its drafter was
    /// returned to the worker slot at park time; promotion re-acquires
    /// and re-attaches it ([`MtpSliceJob::attach_drafter`]).
    Parked(Box<MtpSliceJob>),
    /// An admitted speculative-eligible request whose slice 0 has not
    /// run yet. Promotion runs slice 0 through the same begin machinery
    /// as direct admission.
    Waiter(Box<SequenceInfo>),
}

impl SliceBacklogEntry {
    /// A freshly parked in-flight job (skip counter starts at 0).
    pub(crate) fn parked(job: Box<MtpSliceJob>) -> Self {
        Self {
            kind: SliceBacklogKind::Parked(job),
            skipped_grants: 0,
        }
    }

    /// A freshly admitted waiter (skip counter starts at 0).
    pub(crate) fn waiter(seq: Box<SequenceInfo>) -> Self {
        Self {
            kind: SliceBacklogKind::Waiter(seq),
            skipped_grants: 0,
        }
    }

    /// The entry's request priority lane, for [`next_grant_index`].
    pub(crate) fn priority(&self) -> super::sequence::RequestPriority {
        match &self.kind {
            SliceBacklogKind::Parked(job) => job.seq.priority,
            SliceBacklogKind::Waiter(seq) => seq.priority,
        }
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
    /// The bound drafter, held WITHOUT reset between slices while the job
    /// is ACTIVE (holds the slot). `None` transiently inside a slice
    /// (while the generator owns it), after [`Self::take_drafter`] at
    /// finalize, and for the whole time the job is PARKED in the
    /// scheduler's slice backlog (issue #746): the park boundary returns
    /// the drafter to the worker slot for the next grantee, and promotion
    /// re-attaches it via [`Self::attach_drafter`].
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

    /// Take the drafter for the return to the
    /// [`super::speculative_burst::WorkerDrafterSlot`] (which resets it):
    /// at end of session, and at a park boundary when the slot rotates to
    /// another grantee (issue #746).
    pub(crate) fn take_drafter(&mut self) -> Option<Box<dyn Drafter>> {
        self.drafter.take()
    }

    /// Re-attach the worker drafter at grant-promotion time (issue #746),
    /// the counterpart of [`Self::take_drafter`] at the park boundary.
    ///
    /// Correctness of the round-trip through
    /// [`super::speculative_burst::WorkerDrafterSlot::return_drafter`]
    /// (which calls [`Drafter::reset`]) and back: `step_session`'s
    /// resumability contract requires that nothing the shared-KV re-arm
    /// does not rebuild is destroyed between rounds. For the MTP arm this
    /// holds because (a) the Gemma 4 assistant drafter does not override
    /// `reset`, so its reset is the trait default no-op (verified: it
    /// unloads no weights and unbinds nothing), (b) its bind state
    /// (captured target embedding + resolved LM head) is derived from the
    /// worker's one target model and is recomputed identically by the
    /// promotion-time re-bind, and (c) every per-round value the drafter
    /// consumes (`shared_kv`, offsets, RoPE position) is a stateless
    /// overwrite performed by `set_shared_kv` at the top of EVERY round
    /// from the session's own stored `MtpVerifyOutput`, so rounds of
    /// another session in between cannot leak state into this one.
    pub(crate) fn attach_drafter(&mut self, drafter: Box<dyn Drafter>) {
        debug_assert!(
            self.drafter.is_none(),
            "attach_drafter over a still-held drafter"
        );
        self.drafter = Some(drafter);
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
    // Invariant: while `finish` is `None` AND the job holds the slot, the
    // job holds both the session and the drafter (a PARKED job holds only
    // the session; the promotion path re-attaches the drafter before any
    // step, see `attach_drafter`). The only code that takes them is this
    // function, the park boundary, and the finalize path, and each
    // restores, re-attaches, or finishes.
    let drafter = job
        .drafter
        .take()
        .expect("MtpSliceJob invariant: drafter present while holding the slot");
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
    use super::super::sequence::RequestPriority;
    use super::{
        MTP_SLICE_GRANT_ROUNDS_DEFAULT, mtp_slice_grant_rounds_default, mtp_tick_slice_default,
        next_grant_index, slice_backlog_admits, slice_grant_expired, slice_takes_tick,
    };

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

    /// Grant-budget env parsing (issue #746): default 8, custom values,
    /// 0 = unbounded legacy hold, garbage falls back to the default.
    #[test]
    fn grant_rounds_env_default_custom_zero_and_garbage() {
        assert_eq!(
            mtp_slice_grant_rounds_default(None),
            MTP_SLICE_GRANT_ROUNDS_DEFAULT
        );
        assert_eq!(mtp_slice_grant_rounds_default(Some("4")), 4);
        assert_eq!(mtp_slice_grant_rounds_default(Some(" 12 ")), 12);
        assert_eq!(mtp_slice_grant_rounds_default(Some("0")), 0);
        for garbage in ["", "abc", "-3", "8.5", "eight"] {
            assert_eq!(
                mtp_slice_grant_rounds_default(Some(garbage)),
                MTP_SLICE_GRANT_ROUNDS_DEFAULT,
                "{garbage:?} must fall back to the default"
            );
        }
    }

    /// The grant budget binds ONLY under contention (issue #746): an
    /// uncontended job never rotates regardless of how long it has held
    /// the slot, and budget 0 never expires (legacy hold escape hatch).
    #[test]
    fn grant_expiry_binds_only_under_contention() {
        assert!(slice_grant_expired(8, 8, true));
        assert!(
            slice_grant_expired(100, 8, true),
            "fast handoff after idle overrun"
        );
        assert!(!slice_grant_expired(7, 8, true), "budget not yet spent");
        assert!(
            !slice_grant_expired(100, 8, false),
            "no contention, no rotation"
        );
        assert!(!slice_grant_expired(100, 0, true), "budget 0 is unbounded");
    }

    /// Backlog admission (issue #746): admits below the cap, rejects at
    /// and beyond it (those requests keep the pre-#746 classic-decode
    /// fallback), and rejects everything when rotation is disabled
    /// (budget 0), under which a waiter could starve for the active
    /// job's whole generation.
    #[test]
    fn backlog_admission_cap_and_legacy_escape_hatch() {
        assert!(slice_backlog_admits(0, 8));
        assert!(slice_backlog_admits(1, 8));
        assert!(!slice_backlog_admits(2, 8), "cap reached: classic fallback");
        assert!(!slice_backlog_admits(5, 8));
        assert!(!slice_backlog_admits(0, 0), "budget 0 disables waiting");
    }

    /// Grant order, un-escalated case (issue #746): priority lane first,
    /// FIFO within a lane (parked jobs and waiters share the one ring in
    /// park/arrival order, so the earliest entry of the highest lane
    /// present wins while no entry is overdue).
    #[test]
    fn grant_order_is_priority_lane_first_then_fifo() {
        use RequestPriority::{High, Low, Normal};
        assert_eq!(next_grant_index(&[]), None);
        assert_eq!(next_grant_index(&[(Normal, 0)]), Some(0));
        assert_eq!(
            next_grant_index(&[(Normal, 0), (Normal, 0)]),
            Some(0),
            "FIFO"
        );
        assert_eq!(
            next_grant_index(&[(Normal, 0), (High, 0)]),
            Some(1),
            "lane first"
        );
        assert_eq!(
            next_grant_index(&[(High, 0), (Normal, 0), (High, 0)]),
            Some(0)
        );
        assert_eq!(next_grant_index(&[(Low, 0), (Normal, 0)]), Some(1));
        assert_eq!(
            next_grant_index(&[(Low, 0), (Low, 0), (Normal, 0)]),
            Some(2)
        );
    }

    /// Anti-starvation escalation (issue #746): an entry skipped by
    /// [`super::MTP_SLICE_GRANT_SKIP_CAP`] grant decisions is overdue and
    /// preempts lane order; the most-skipped overdue entry wins, earliest
    /// in ring order on ties; below the cap, lane order still rules.
    #[test]
    fn grant_order_escalates_overdue_entries_over_lanes() {
        use super::MTP_SLICE_GRANT_SKIP_CAP;
        use RequestPriority::{High, Low, Normal};
        let cap = MTP_SLICE_GRANT_SKIP_CAP;
        assert_eq!(next_grant_index(&[(High, 0), (Normal, cap)]), Some(1));
        assert_eq!(next_grant_index(&[(High, 0), (Low, cap)]), Some(1));
        assert_eq!(
            next_grant_index(&[(High, 0), (Normal, cap - 1)]),
            Some(0),
            "below the cap lane order rules"
        );
        assert_eq!(
            next_grant_index(&[(Normal, cap), (High, cap + 1)]),
            Some(1),
            "most-skipped overdue entry wins"
        );
        assert_eq!(
            next_grant_index(&[(Normal, cap), (Low, cap)]),
            Some(0),
            "tie on skips: earliest in ring order"
        );
    }

    /// The escalation loop bounds starvation (issue #746): a Normal
    /// entry behind an endless supply of FRESH High entries (each
    /// granted entry leaves the ring; a new arrival or a re-park enters
    /// with a counter of 0, the reset-on-grant) would never be selected
    /// under strict lane order, but with the skip counter incremented by
    /// every decision that passes it over, it must be selected within
    /// `MTP_SLICE_GRANT_SKIP_CAP + 1` grant decisions.
    #[test]
    fn grant_order_skip_cap_bounds_starvation_under_high_lane_stream() {
        use super::MTP_SLICE_GRANT_SKIP_CAP;
        use RequestPriority::{High, Normal};
        let mut normal_skips = 0usize;
        let mut decisions = 0usize;
        loop {
            decisions += 1;
            assert!(
                decisions <= MTP_SLICE_GRANT_SKIP_CAP + 1,
                "starvation must be bounded"
            );
            // Ring: a fresh High entry (skips 0) plus the waiting Normal.
            let entries = [(High, 0), (Normal, normal_skips)];
            let idx = next_grant_index(&entries).expect("non-empty ring");
            if idx == 1 {
                break;
            }
            // The High entry was granted; the Normal entry was skipped
            // and the next decision sees another fresh High arrival.
            normal_skips += 1;
        }
        assert_eq!(
            decisions,
            MTP_SLICE_GRANT_SKIP_CAP + 1,
            "selected exactly when overdue"
        );
    }

    /// Issue #746 tick arbitration: the SpeculativeRound action is taken
    /// on "speculative work pending" (active job OR non-empty backlog),
    /// not on "active job present", and the SAME `slice_takes_tick`
    /// alternation governs promotion ticks (empty slot, backlog pending)
    /// as governs round ticks. So the #734 HOL bound is preserved across
    /// a rotation: at most one speculative action (a round, a waiter's
    /// slice 0, or a promotion + round) per tick, alternating strictly
    /// with classic work.
    #[test]
    fn backlog_grant_ticks_obey_the_same_alternation() {
        // Simulate the scheduler's flag protocol across a rotation: ticks
        // 0..3 have an active job, tick 4 parks it (slot empty, backlog
        // pending), later ticks promote and continue. The arbitration
        // input is identical in every phase.
        let mut yielded = false;
        let mut history: Vec<&'static str> = Vec::new();
        for _ in 0..12 {
            // "Speculative work pending" covers both phases; classic work
            // is always present in this scenario.
            if slice_takes_tick(yielded, /* others_have_work */ true) {
                history.push("speculative");
                yielded = true;
            } else {
                history.push("classic");
                yielded = false;
            }
        }
        for pair in history.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "promotion ticks must alternate exactly like round ticks: {history:?}"
            );
        }
    }
}
