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

//! Pure tick-arbitration policy for the batch scheduler (issues #908, #1011).
//!
//! [`BatchScheduler::decide_action`](super::scheduler::BatchScheduler) used to
//! carry its whole policy inline, and the unit tests re-implemented that policy
//! in a local helper so they could run without a model. A mirrored policy is
//! only as truthful as the copy: `scheduler_tests.rs` grew a test literally
//! named `chunked_prefill_interleaving_pattern` whose assertions record that
//! chunked prefill does *not* interleave, because the copy was updated from the
//! code while the name and comments kept describing the intent. Issue #908 hit
//! exactly that gap, so the policy now lives here as one pure function that
//! both the scheduler and the tests call.
//!
//! # The starvation this module used to pin, and now forbids
//!
//! The pre-#908 policy resolved a tick with a chunked prefill in progress as
//! follows: if any sequence is decoding, decode; only when the active batch is
//! empty does the chunked prefill advance. Nothing in that policy alternates,
//! and it is a pure function of scheduler state, so the state that selects
//! `Decode` is unchanged by running a decode. A long prompt admitted next to a
//! live decode batch therefore ran chunk 0 and then made no further progress
//! until every decoding sequence had finished. Measured on an M1 Ultra
//! (`docs/benchmark_results/mixed-step-prototype-m1ultra-2026-08-03.md`), the
//! prefill held at chunk 1 of 19 for 20 s while decode advanced 138 to 394
//! steps. With streams arriving continuously that wait has no ceiling, so the
//! admitted request's time to first token has none either.
//!
//! That is the opposite of the behaviour the surrounding comments used to
//! claim, and it inverts the latency problem issue #908 set out to solve:
//! decode streams are never blocked by a chunked prefill, the chunked prefill
//! is blocked by them.
//!
//! # The fairness policy (issue #1011)
//!
//! Alternation was not merely absent before #1011, it was *inexpressible*:
//! [`TickState`] carried no history, so no pure function of it could give the
//! parked prefill a turn. The fix is one counter,
//! [`TickState::decode_ticks_since_prefill_grant`], which
//! [`BatchScheduler`](super::scheduler::BatchScheduler) carries across ticks
//! and this module both reads and advances. The policy stays a pure function,
//! which is what lets the tests exercise the shipped policy instead of a copy
//! of it.
//!
//! **The rule.** While a chunked prefill is parked next to a live decode batch,
//! decode wins the tick until it has won [`TickState::prefill_grant_interval`]
//! of them in a row; the next such tick is GRANTED to the prefill, which runs
//! one chunk and resets the counter.
//!
//! **The bound this guarantees.** The parked prefill advances by at least one
//! chunk every `interval + 1` classic ticks, so a prompt of `C` chunks reaches
//! its first token within `C * (interval + 1)` classic ticks of admission
//! regardless of how long the decode batch keeps running, versus no bound at
//! all before. In wall-clock terms that is `C * (interval * D + P)` for a
//! decode step `D` and a chunk forward `P`. The counterpart cost is exact and
//! falls out of the same arithmetic: over one grant cycle the decoding streams
//! get `interval` tokens per `interval * D + P` of wall clock, so their mean
//! inter-token latency during the admission window is `D + P / interval`. The
//! interval is the ITL/TTFT dial, and `--prefill-grant-interval` exposes it.
//!
//! `interval = 0` disables the grant and restores the pre-#1011 starvation
//! exactly, as an operator escape hatch for deployments that would rather have
//! an unbounded TTFT than any admission-window ITL cost.
//!
//! **Why a grant counter rather than a wall-clock deadline.** A deadline needs
//! a clock read on the tick hot path and ties the policy to hardware speed, so
//! the same configuration means different things on different machines and the
//! unit tests cannot pin it without faking time. A tick counter is
//! hardware-independent, costs one comparison, and states its bound in the unit
//! the scheduler actually schedules in. See ADR 0005.
//!
//! **Prefill versus prefill does not arise.** `BatchScheduler` holds the parked
//! prompt in a single `Option<SequenceInfo>` (`chunked_prefill_seq`), and the
//! chunked branch below short-circuits above the admission branch, so while one
//! prompt is parked no second request can be admitted at all, whatever the
//! batch's occupancy. At most one chunked prefill exists at any instant, so
//! there is no second parked prefill for a grant to be unfair to. The grant
//! therefore cannot relocate the starvation from prefill-versus-decode to
//! prefill-versus-prefill; it can only shorten the head-of-line wait of the
//! queue behind the parked prompt, because the parked prompt finishes sooner.
//!
//! [`TickChoice::MixedStep`] remains the opt-in `MLXCEL_MIXED_STEP` prototype
//! from #908 (both workloads every tick, i.e. the `interval = 1` corner of this
//! frontier plus a shared tick); see
//! `docs/adr/0005-mixed-prefill-decode-step-execution.md`.

/// Whether the mixed prefill/decode step prototype is enabled (issue #908).
///
/// Default **off**: with `MLXCEL_MIXED_STEP` unset the mixed-step branch is
/// unreachable. `MLXCEL_MIXED_STEP=1|true|yes|on` opts in.
pub(crate) fn mixed_step_enabled() -> bool {
    mixed_step_default(std::env::var("MLXCEL_MIXED_STEP").ok().as_deref())
}

/// Pure decision core of [`mixed_step_enabled`], separated for unit testing.
///
/// Unset, empty, or unrecognised means off. Only the explicit affirmative set
/// turns the prototype on, which is the conservative direction for a flag that
/// changes scheduler arbitration. The value is trimmed and lowercased first, so
/// `On`, ` yes `, and `TrUe` behave like their canonical spellings; this is an
/// operator-facing variable and a surprising rejection of `On` would read as
/// the prototype silently not engaging.
pub(crate) fn mixed_step_default(env_override: Option<&str>) -> bool {
    match env_override {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Environment override for [`resolve_prefill_grant_interval`].
pub(crate) const PREFILL_GRANT_INTERVAL_ENV: &str = "MLXCEL_PREFILL_GRANT_INTERVAL";

/// Default value of `--prefill-grant-interval` (issue #1011).
///
/// Chosen from measurement, not from taste. On a quiet M1 Ultra serving
/// llama-3.1-8b-4bit at `--parallel 8 --prefill-chunk-size 512`, a 512-token
/// chunk costs 775 ms against a 58 ms decode step, so `P / D` is about 13 and
/// the frontier over a 14-chunk prompt measures (median of three interleaved
/// repeats, admitted TTFT / mean admission-window ITL inflation):
///
/// | interval | TTFT | mean ITL |
/// |---|---|---|
/// | 0 (disabled) | 50.5 s, and unbounded in general | 1.0x |
/// | 4 | 14.7 s | 3.9x |
/// | 8 | 17.6 s | 2.3x |
/// | **16** | **23.8 s** | **1.6x** |
/// | 32 | 29.9 s | 1.2x |
///
/// 16 is the knee: each halving below it buys much less TTFT than the
/// inter-token latency it costs, and each doubling above it buys much less
/// latency than the TTFT it costs. An operator who wants the admitted request's
/// first token sooner lowers it; one who wants the decoding streams undisturbed
/// raises it, or sets `0` to opt back into the unbounded wait entirely. Full
/// table, dispersion, and the reason the p95 column must not be used to pick
/// this number: `docs/benchmark_results/prefill-fairness-m1ultra-2026-08-03.md`.
pub(crate) const PREFILL_GRANT_INTERVAL_DEFAULT: u32 = 16;

/// Resolve the prefill grant interval: explicit CLI value, else
/// [`PREFILL_GRANT_INTERVAL_ENV`], else [`PREFILL_GRANT_INTERVAL_DEFAULT`].
///
/// Read once at scheduler construction, never on the tick path, for the same
/// reason `MLXCEL_MIXED_STEP` is: `std::env::var` takes a process-wide lock and
/// allocates.
pub(crate) fn resolve_prefill_grant_interval(configured: Option<usize>) -> u32 {
    if let Some(value) = configured {
        return u32::try_from(value).unwrap_or(u32::MAX);
    }
    prefill_grant_interval_default(std::env::var(PREFILL_GRANT_INTERVAL_ENV).ok().as_deref())
}

/// Pure decision core of the environment half of
/// [`resolve_prefill_grant_interval`], separated for unit testing.
///
/// Unparseable values fall back to the default rather than to 0: a typo must
/// not silently reinstate the unbounded-TTFT starvation this policy exists to
/// remove. `0` still disables the grant, but only when spelled exactly.
pub(crate) fn prefill_grant_interval_default(env_override: Option<&str>) -> u32 {
    env_override
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(PREFILL_GRANT_INTERVAL_DEFAULT)
}

/// The scheduler state the tick policy reads, flattened to plain data.
///
/// Flattening is deliberate: it keeps the policy independent of
/// `BatchScheduler`, which owns a model and cannot be built in a unit test, so
/// the tests exercise the real policy instead of a copy of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct TickState {
    /// An MTP slice job is active, or the grant backlog is non-empty (#734/#746).
    pub speculative_pending: bool,
    /// The previous tick was a speculative round, so this one yields to a
    /// classic action when one has work.
    pub speculative_yielded: bool,
    /// A chunked prefill is parked mid-prompt awaiting continuation.
    pub chunked_prefill_in_progress: bool,
    /// No sequence is currently decoding.
    pub active_is_empty: bool,
    /// The active batch has no free slot (`len == max_batch_size`).
    pub active_is_full: bool,
    /// No request is waiting to be admitted.
    pub queue_is_empty: bool,
    /// Preemption is enabled and a higher-priority request is waiting.
    pub should_preempt: bool,
    /// `MLXCEL_MIXED_STEP` is set (issue #908 prototype).
    pub mixed_step_enabled: bool,
    /// Consecutive contended ticks that resolved to `Decode` since the parked
    /// chunked prefill last ran a chunk (issue #1011).
    ///
    /// This is the only piece of history the policy carries, and it is what
    /// makes alternation expressible at all. `BatchScheduler` owns it and can
    /// only set it from [`TickDecision::decode_ticks_since_prefill_grant`], so
    /// no call site can advance the tick without advancing the counter.
    pub decode_ticks_since_prefill_grant: u32,
    /// How many consecutive decode ticks a parked chunked prefill yields before
    /// it is granted one (`--prefill-grant-interval`, issue #1011). `0`
    /// disables the grant and restores the pre-#1011 starvation.
    pub prefill_grant_interval: u32,
}

/// The action the policy selects for one tick.
///
/// This is the model-free twin of
/// [`BatchSchedulerAction`](super::sequence::BatchSchedulerAction): the
/// scheduler maps each variant onto the real action, attaching the active
/// sequence ids that `Decode` and `MixedStep` carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TickChoice {
    /// Advance the in-flight speculative slice by one round.
    SpeculativeRound,
    /// Decode every active sequence one token.
    Decode,
    /// Decode every active sequence one token **and** advance the parked
    /// chunked prefill by one chunk, in that order, within this tick
    /// (issue #908 prototype; requires `mixed_step_enabled`).
    MixedStep,
    /// Admit a queued request, or continue the parked chunked prefill.
    Prefill,
    /// Nothing to do; block on the request channel.
    Idle,
}

/// One tick's arbitration result: what to run, and the fairness counter the
/// scheduler must carry into the next tick.
///
/// The counter rides along with the choice on purpose. The alternative shape,
/// a `decide_tick` that returns only a choice plus a `self.counter = 0` at
/// whichever call site happens to run a prefill, is exactly the kind of update
/// a later edit forgets, and forgetting it here silently reinstates the
/// unbounded starvation. Making the new counter part of the return value means
/// the single place that applies a decision
/// ([`BatchScheduler::decide_action`](super::scheduler::BatchScheduler)) cannot
/// take the choice without also taking the counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TickDecision {
    /// The action to run this tick.
    pub choice: TickChoice,
    /// The value [`TickState::decode_ticks_since_prefill_grant`] must hold on
    /// the next tick.
    pub decode_ticks_since_prefill_grant: u32,
}

/// Whether a parked chunked prefill is owed its fairness tick (issue #1011).
///
/// Pure and public to the module so the tests can characterise divergence from
/// the pre-#908 policy in terms of the grant firing, rather than by
/// transcribing the branch structure a second time.
#[inline]
pub(crate) fn prefill_grant_due(state: &TickState) -> bool {
    state.prefill_grant_interval > 0
        && state.decode_ticks_since_prefill_grant >= state.prefill_grant_interval
}

/// Select this tick's action and the fairness counter that follows it.
///
/// The pre-#908 policy is preserved everywhere except the one branch where a
/// chunked prefill is parked next to a live decode batch. There, the
/// `MLXCEL_MIXED_STEP` prototype (#908) still wins when enabled, and otherwise
/// decode wins until the #1011 grant is due.
pub(crate) fn decide_tick(state: &TickState) -> TickDecision {
    let choice = select_choice(state);
    TickDecision {
        choice,
        decode_ticks_since_prefill_grant: next_grant_counter(state, choice),
    }
}

/// The choice half of [`decide_tick`].
fn select_choice(state: &TickState) -> TickChoice {
    // Tick-cooperative speculative slice work pending (issue #734): run one
    // speculative action per tick, alternating strictly with the classic
    // actions when they have work, so concurrent classic rows advance between
    // rounds and the speculative request never starves. When nothing else has
    // work the slice takes every tick.
    //
    // The #1011 grant does not disturb that alternation, because it changes
    // only WHICH classic action runs on a tick the slice has already yielded,
    // never whether a classic action runs. Symmetrically, a speculative slice
    // cannot starve the parked prefill: strict alternation hands the classic
    // arm every other tick, and only those ticks move the grant counter, so the
    // grant still fires, at half the wall-clock rate. The bound doubles under
    // speculative contention; it does not disappear.
    if state.speculative_pending {
        let others_have_work =
            state.chunked_prefill_in_progress || !state.active_is_empty || !state.queue_is_empty;
        if super::speculative_slice::slice_takes_tick(state.speculative_yielded, others_have_work) {
            return TickChoice::SpeculativeRound;
        }
        // Fall through: grant this tick to a classic action; its arm clears
        // `speculative_slice_yielded` so the next tick returns to the slice.
    }

    // A chunked prefill is parked mid-prompt. At most one can be, because the
    // scheduler holds it in a single `Option` and this branch short-circuits
    // above the admission branch below, so nothing else is admitted while it is
    // parked; see the module docs.
    if state.chunked_prefill_in_progress {
        if !state.active_is_empty {
            // `MLXCEL_MIXED_STEP` (issue #908 prototype) advances both
            // workloads on this tick. It outranks the grant because it is
            // strictly more aggressive: every tick is a prefill tick, so a
            // grant would have nothing left to add.
            if state.mixed_step_enabled {
                return TickChoice::MixedStep;
            }
            // Issue #1011: decode has won `prefill_grant_interval` contended
            // ticks in a row, so this one belongs to the parked prefill. Before
            // #1011 this returned `Decode` unconditionally and, because the
            // policy is a pure function of state that a decode does not change,
            // kept returning it until the batch drained.
            if prefill_grant_due(state) {
                return TickChoice::Prefill;
            }
            return TickChoice::Decode;
        }
        // No active sequences: continue the prefill.
        return TickChoice::Prefill;
    }

    if state.active_is_empty && state.queue_is_empty {
        return TickChoice::Idle;
    }

    // When active sequences exist:
    // 1. Preemption overrides when enabled and a higher-priority request waits.
    // 2. If the batch is NOT full and the queue has work, admit one new
    //    sequence (larger batches amortize weight-loading bandwidth).
    // 3. Otherwise decode the existing sequences.
    if !state.active_is_empty {
        if state.should_preempt {
            return TickChoice::Prefill;
        }
        if !state.active_is_full && !state.queue_is_empty {
            return TickChoice::Prefill;
        }
        return TickChoice::Decode;
    }

    // Batch is empty but the queue has work.
    TickChoice::Prefill
}

/// Advance the fairness counter for the action this tick selected.
///
/// The ledger being kept is "contended decode ticks the parked prefill has
/// yielded since it last ran", so only ticks that actually resolve that
/// contention move it.
fn next_grant_counter(state: &TickState, choice: TickChoice) -> u32 {
    if !state.chunked_prefill_in_progress {
        // Nothing is parked, so there is no wait to account for. A prompt that
        // parks on a later tick therefore starts its wait at zero rather than
        // inheriting a stale count and being granted immediately.
        return 0;
    }
    match choice {
        // The parked chunk ran this tick. `Prefill` with a chunked prefill in
        // progress is always the continuation, never an admission:
        // `execute_prefill` short-circuits into `continue_chunked_prefill`
        // whenever `chunked_prefill_seq` is set. `MixedStep` runs the chunk too.
        TickChoice::Prefill | TickChoice::MixedStep => 0,
        // Decode won a contended tick; the prefill is one tick closer to its
        // grant. Saturating rather than wrapping, so a pathologically long wait
        // pins the grant on instead of silently rolling back under it.
        TickChoice::Decode => state.decode_ticks_since_prefill_grant.saturating_add(1),
        // The speculative slice took the tick (issue #734). Neither the decode
        // batch nor the prefill advanced, so the decode-versus-prefill ledger is
        // unchanged and the grant neither approaches nor recedes.
        TickChoice::SpeculativeRound => state.decode_ticks_since_prefill_grant,
        // Unreachable while a chunked prefill is parked (the chunked branch
        // returns above the idle arm). Kept total rather than `unreachable!`,
        // since a panic in tick arbitration would take the server down.
        TickChoice::Idle => state.decode_ticks_since_prefill_grant,
    }
}

#[cfg(test)]
#[path = "tick_policy_tests.rs"]
mod tick_policy_tests;
