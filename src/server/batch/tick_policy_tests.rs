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

//! Unit tests for the pure tick-arbitration policy (issues #908, #1011).
//!
//! These call [`decide_tick`] itself, not a copy of it. The whole reason this
//! module exists is that the previous tests re-implemented the policy locally
//! and drifted from it in their naming, which hid the chunked-prefill
//! starvation that issue #908 turned up and issue #1011 fixed.

use super::*;

/// One boolean state of the policy, for a given counter and configuration.
///
/// Seven booleans, so the boolean space is 128 states and the parity tests
/// below are exhaustive over it rather than sampled. The two numeric fields
/// (#1011) are swept separately by the callers, which is what keeps the
/// enumeration finite while still covering both sides of the grant threshold.
fn all_states(mixed_step_enabled: bool, counter: u32, interval: u32) -> Vec<TickState> {
    let mut out = Vec::with_capacity(128);
    for bits in 0u8..128 {
        out.push(TickState {
            speculative_pending: bits & 1 != 0,
            speculative_yielded: bits & 2 != 0,
            chunked_prefill_in_progress: bits & 4 != 0,
            active_is_empty: bits & 8 != 0,
            active_is_full: bits & 16 != 0,
            queue_is_empty: bits & 32 != 0,
            should_preempt: bits & 64 != 0,
            mixed_step_enabled,
            decode_ticks_since_prefill_grant: counter,
            prefill_grant_interval: interval,
        });
    }
    out
}

/// The scheduler policy exactly as it stood before issue #908, transcribed from
/// `BatchScheduler::decide_action` at commit 230163db.
///
/// This is a deliberate mirror, which the module docs otherwise warn against.
/// It is safe here for one reason: it is compared against [`decide_tick`] over
/// the complete boolean state space, so any drift fails a test instead of
/// hiding. Since #1011 its job is no longer "prove nothing changed" (the
/// default policy deliberately changed) but "prove exactly what changed": the
/// only inputs on which the shipped policy diverges from this transcription are
/// the ones where the fairness grant fires.
fn pre_908_policy(state: &TickState) -> TickChoice {
    if state.speculative_pending {
        let others_have_work =
            state.chunked_prefill_in_progress || !state.active_is_empty || !state.queue_is_empty;
        if super::super::speculative_slice::slice_takes_tick(
            state.speculative_yielded,
            others_have_work,
        ) {
            return TickChoice::SpeculativeRound;
        }
    }
    if state.chunked_prefill_in_progress {
        if !state.active_is_empty {
            return TickChoice::Decode;
        }
        return TickChoice::Prefill;
    }
    if state.active_is_empty && state.queue_is_empty {
        return TickChoice::Idle;
    }
    if !state.active_is_empty {
        if state.should_preempt {
            return TickChoice::Prefill;
        }
        if !state.active_is_full && !state.queue_is_empty {
            return TickChoice::Prefill;
        }
        return TickChoice::Decode;
    }
    TickChoice::Prefill
}

/// Counter values swept by the exhaustive tests. Spans both sides of every
/// interval used below, including well past it, so the "grant stays due once it
/// is due" direction is covered too.
const COUNTERS: [u32; 8] = [0, 1, 2, 3, 4, 5, 8, 17];

// -------------------------------------------------------------------
// Divergence from the pre-#908 policy: exhaustive, and characterised
// -------------------------------------------------------------------

/// The escape hatch is byte-identical to the pre-#908 policy.
///
/// `--prefill-grant-interval 0` disables the fairness grant, and an operator
/// who sets it is asking for exactly the old arbitration back, including its
/// unbounded parked-prefill wait. This is the original #908 parity proof,
/// unweakened, re-aimed at the configuration it still holds for; the counter is
/// swept because a disabled grant must ignore it at every value.
#[test]
fn grant_disabled_is_identical_to_the_pre_908_policy() {
    for counter in COUNTERS {
        for state in all_states(false, counter, 0) {
            assert_eq!(
                decide_tick(&state).choice,
                pre_908_policy(&state),
                "--prefill-grant-interval 0 must restore the pre-#908 arbitration exactly; \
                 diverged at {state:?}"
            );
        }
    }
}

/// The shipped default diverges from the pre-#908 policy on exactly the inputs
/// where the fairness grant fires, and nowhere else.
///
/// This is the successor to `mixed_step_off_is_identical_to_the_pre_908_policy`
/// (issue #908), whose premise #1011 deliberately invalidated: the default
/// policy is no longer identical to the old one. Deleting it would have thrown
/// away the strongest guard in the file, so it is re-stated as a two-directional
/// characterisation over the same complete state space:
///
/// - **Soundness.** Wherever the two differ, the difference is a fired grant:
///   the new policy says `Prefill`, the old one said `Decode`, and the state is
///   a parked chunked prefill next to a live batch with the grant due.
/// - **Completeness.** Wherever the grant is due on a tick the old policy gave
///   to decode, the new policy really does hand it to the prefill. Without this
///   direction the test would still pass if the grant never fired at all, which
///   is precisely the defect being fixed.
///
/// Neither direction re-derives the branch structure: the "old policy said
/// `Decode`" clause is read from the transcription, which is what excludes the
/// speculative-round states without this test having to know why.
#[test]
fn default_policy_differs_from_pre_908_only_where_the_grant_fires() {
    let mut fired = 0usize;
    for interval in [1u32, 2, 3, 4, 16] {
        for counter in COUNTERS {
            for state in all_states(false, counter, interval) {
                let now = decide_tick(&state).choice;
                let before = pre_908_policy(&state);
                let grant_state = state.chunked_prefill_in_progress
                    && !state.active_is_empty
                    && prefill_grant_due(&state)
                    && before == TickChoice::Decode;

                if now != before {
                    // Soundness: the only licensed divergence.
                    assert_eq!(
                        now,
                        TickChoice::Prefill,
                        "unexpected divergence at {state:?}"
                    );
                    assert_eq!(
                        before,
                        TickChoice::Decode,
                        "unexpected divergence at {state:?}"
                    );
                    assert!(
                        grant_state,
                        "diverged somewhere other than a fired grant: {state:?}"
                    );
                    fired += 1;
                } else {
                    // Completeness: a due grant must not be quietly skipped.
                    assert!(
                        !grant_state,
                        "the grant was due but decode still took the tick: {state:?}"
                    );
                }
            }
        }
    }
    assert!(
        fired > 0,
        "no state fired the grant; the sweep no longer covers the branch it exists to check"
    );
}

#[test]
fn mixed_step_off_never_selects_mixed_step() {
    for interval in [0u32, 1, 4, 16] {
        for counter in COUNTERS {
            for state in all_states(false, counter, interval) {
                assert_ne!(
                    decide_tick(&state).choice,
                    TickChoice::MixedStep,
                    "MixedStep must be unreachable with the flag off; reached at {state:?}"
                );
            }
        }
    }
}

/// The `MLXCEL_MIXED_STEP` prototype still differs from the flag-off policy
/// only on the chunked interleave branch, and it outranks the grant there: a
/// mixed tick already runs a chunk every tick, so a grant has nothing to add.
#[test]
fn mixed_step_on_differs_only_on_the_chunked_interleave_branch() {
    for interval in [0u32, 1, 4, 16] {
        for counter in COUNTERS {
            for state in all_states(true, counter, interval) {
                let off = TickState {
                    mixed_step_enabled: false,
                    ..state
                };
                let with = decide_tick(&state).choice;
                let without = decide_tick(&off).choice;
                if with == without {
                    continue;
                }
                // The only divergence the prototype is allowed to introduce.
                assert_eq!(
                    with,
                    TickChoice::MixedStep,
                    "unexpected divergence {state:?}"
                );
                assert!(
                    // Without the prototype this tick is a decode, or the #1011
                    // grant when it happened to be due.
                    without == TickChoice::Decode || without == TickChoice::Prefill,
                    "unexpected divergence {state:?}"
                );
                assert!(state.chunked_prefill_in_progress, "{state:?}");
                assert!(!state.active_is_empty, "{state:?}");
            }
        }
    }
}

// -------------------------------------------------------------------
// The fairness counter (issue #1011)
// -------------------------------------------------------------------

/// The counter the policy hands back is a total function of the state and the
/// choice, checked exhaustively so no branch can quietly stop accounting.
///
/// The scheduler cannot apply a choice without applying this value, so these
/// four rules are the whole of the counter's lifecycle.
#[test]
fn grant_counter_transitions_are_exhaustively_accounted() {
    for mixed in [false, true] {
        for interval in [0u32, 1, 4, 16] {
            for counter in COUNTERS {
                for state in all_states(mixed, counter, interval) {
                    let decision = decide_tick(&state);
                    let next = decision.decode_ticks_since_prefill_grant;
                    if !state.chunked_prefill_in_progress {
                        assert_eq!(next, 0, "no parked prefill must mean no wait: {state:?}");
                        continue;
                    }
                    match decision.choice {
                        // The chunk ran.
                        TickChoice::Prefill | TickChoice::MixedStep => {
                            assert_eq!(next, 0, "a chunk ran, so the wait resets: {state:?}");
                        }
                        // Decode won a contended tick.
                        TickChoice::Decode => {
                            assert_eq!(next, counter + 1, "decode must be charged: {state:?}");
                        }
                        // The speculative slice took the tick; ledger unchanged.
                        TickChoice::SpeculativeRound => {
                            assert_eq!(
                                next, counter,
                                "a speculative round moves neither workload: {state:?}"
                            );
                        }
                        TickChoice::Idle => {
                            unreachable!("Idle is unreachable with a prefill parked: {state:?}")
                        }
                    }
                }
            }
        }
    }
}

/// A prefill that parks after a long decode run starts its wait at zero, so it
/// is not granted a tick the instant it arrives.
#[test]
fn a_freshly_parked_prefill_starts_its_wait_at_zero() {
    let decoding = TickState {
        chunked_prefill_in_progress: false,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        decode_ticks_since_prefill_grant: 999,
        prefill_grant_interval: PREFILL_GRANT_INTERVAL_DEFAULT,
        ..TickState::default()
    };
    let decision = decide_tick(&decoding);
    assert_eq!(decision.choice, TickChoice::Decode);
    assert_eq!(decision.decode_ticks_since_prefill_grant, 0);
}

// -------------------------------------------------------------------
// The starvation this issue forbids (issue #1011)
// -------------------------------------------------------------------

/// The defect, forbidden rather than pinned.
///
/// Before #1011 this state was a fixed point: decoding alters no field the
/// policy reads, so `Decode` was selected forever and the parked prompt made no
/// progress until the batch drained (measured at 20 s of zero progress on an M1
/// Ultra). The predecessor of this test, `chunked_prefill_starves_until_
/// active_batch_drains`, asserted that fixed point as correct behaviour.
///
/// Now the same state must yield a grant, and it must do so on a bounded tick:
/// exactly `interval + 1` ticks per chunk, with `interval` decode ticks in
/// between and never more.
#[test]
fn a_parked_chunked_prefill_is_never_starved_by_a_live_decode_batch() {
    for interval in [1u32, 2, 4, 8, 16, PREFILL_GRANT_INTERVAL_DEFAULT] {
        let mut state = TickState {
            chunked_prefill_in_progress: true,
            active_is_empty: false,
            active_is_full: true,
            queue_is_empty: true,
            prefill_grant_interval: interval,
            ..TickState::default()
        };

        const TICKS: u32 = 1000;
        let mut grants = 0u32;
        let mut decode_run = 0u32;
        let mut longest_decode_run = 0u32;
        let mut first_grant_tick = None;
        for tick in 0..TICKS {
            let decision = decide_tick(&state);
            match decision.choice {
                TickChoice::Prefill => {
                    grants += 1;
                    first_grant_tick.get_or_insert(tick);
                    decode_run = 0;
                }
                TickChoice::Decode => {
                    decode_run += 1;
                    longest_decode_run = longest_decode_run.max(decode_run);
                }
                other => panic!("unexpected choice {other:?} at interval {interval}"),
            }
            // The scheduler's one application site, in miniature.
            state.decode_ticks_since_prefill_grant = decision.decode_ticks_since_prefill_grant;
        }

        assert_eq!(
            first_grant_tick,
            Some(interval),
            "interval {interval}: the first grant must land on tick {interval} (0-based)"
        );
        assert_eq!(
            longest_decode_run, interval,
            "interval {interval}: decode must never win more than {interval} ticks in a row"
        );
        // One chunk per `interval + 1` ticks, so a C-chunk prompt finishes
        // within C * (interval + 1) ticks however long the batch decodes.
        assert_eq!(
            grants,
            TICKS / (interval + 1),
            "interval {interval}: grant rate must be exactly one tick in {}",
            interval + 1
        );
    }
}

/// The escape hatch reproduces the pre-#1011 fixed point, on purpose.
///
/// This is the one place the old behaviour is still asserted, and it earns its
/// keep as a regression boundary: `--prefill-grant-interval 0` is documented as
/// "restore the previous arbitration", and a change that made the grant fire
/// anyway would break that promise silently. Note what it demonstrates about
/// the default: the loop below is what every default-configured server did
/// before this issue.
#[test]
fn grant_interval_zero_restores_the_pre_1011_fixed_point() {
    let mut state = TickState {
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        prefill_grant_interval: 0,
        ..TickState::default()
    };
    for tick in 0..1000 {
        let decision = decide_tick(&state);
        assert_eq!(
            decision.choice,
            TickChoice::Decode,
            "tick {tick}: with the grant disabled the prefill must stay starved"
        );
        state.decode_ticks_since_prefill_grant = decision.decode_ticks_since_prefill_grant;
    }

    // As before, the prefill resumes only once the last decoding sequence
    // completes.
    let drained = TickState {
        active_is_empty: true,
        active_is_full: false,
        ..state
    };
    assert_eq!(decide_tick(&drained).choice, TickChoice::Prefill);
}

/// The prototype converts the same fixed point into a full interleave: every
/// tick advances both workloads.
#[test]
fn mixed_step_advances_both_workloads_every_tick() {
    let mut state = TickState {
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        mixed_step_enabled: true,
        prefill_grant_interval: PREFILL_GRANT_INTERVAL_DEFAULT,
        ..TickState::default()
    };
    for tick in 0..1000 {
        let decision = decide_tick(&state);
        assert_eq!(
            decision.choice,
            TickChoice::MixedStep,
            "tick {tick}: mixed step should advance prefill and decode together"
        );
        // Every tick runs a chunk, so the grant counter can never build up and
        // the grant never has anything to add.
        assert_eq!(decision.decode_ticks_since_prefill_grant, 0);
        state.decode_ticks_since_prefill_grant = decision.decode_ticks_since_prefill_grant;
    }
}

/// With the batch empty there is no decode work to mix in, so the prototype
/// must fall through to the ordinary prefill continuation rather than
/// dispatching a mixed step over an empty id list.
#[test]
fn mixed_step_falls_back_to_prefill_when_batch_is_empty() {
    let state = TickState {
        chunked_prefill_in_progress: true,
        active_is_empty: true,
        queue_is_empty: true,
        mixed_step_enabled: true,
        prefill_grant_interval: PREFILL_GRANT_INTERVAL_DEFAULT,
        ..TickState::default()
    };
    assert_eq!(decide_tick(&state).choice, TickChoice::Prefill);
}

// -------------------------------------------------------------------
// Speculative slice arbitration (issue #734) versus the grant
// -------------------------------------------------------------------

/// A pending speculative slice that has not yielded still takes the tick, with
/// the prototype on. Mixed steps and speculative rounds therefore exclude each
/// other, which ADR 0005 records as a prototype limitation rather than a
/// design decision.
#[test]
fn speculative_round_outranks_mixed_step() {
    let state = TickState {
        speculative_pending: true,
        speculative_yielded: false,
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        mixed_step_enabled: true,
        ..TickState::default()
    };
    assert_eq!(decide_tick(&state).choice, TickChoice::SpeculativeRound);
}

/// After a speculative round yields, the classic arm it falls through to is the
/// mixed step (with the flag on), so the prototype participates in the #734
/// alternation exactly where `Decode` used to.
#[test]
fn yielded_speculative_slice_falls_through_to_mixed_step() {
    let state = TickState {
        speculative_pending: true,
        speculative_yielded: true,
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        queue_is_empty: true,
        mixed_step_enabled: true,
        ..TickState::default()
    };
    assert_eq!(decide_tick(&state).choice, TickChoice::MixedStep);
}

/// A pending speculative slice outranks a *due* grant, exactly as it outranks a
/// decode. The grant changes which classic action runs, never whether the
/// classic arm runs at all, so the #734 alternation is untouched.
#[test]
fn speculative_round_outranks_a_due_grant() {
    let state = TickState {
        speculative_pending: true,
        speculative_yielded: false,
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        decode_ticks_since_prefill_grant: 999,
        prefill_grant_interval: 4,
        ..TickState::default()
    };
    let decision = decide_tick(&state);
    assert_eq!(decision.choice, TickChoice::SpeculativeRound);
    // The ledger is untouched, so the grant is still due on the next classic
    // tick rather than having been consumed by a round that ran no chunk.
    assert_eq!(decision.decode_ticks_since_prefill_grant, 999);
}

/// A speculative slice cannot starve the parked prefill either (issue #1011
/// interaction with #734).
///
/// Under contention #734 hands the classic arm every other tick, and only
/// classic ticks move the grant ledger, so the grant still fires, at half the
/// wall-clock rate: the parked prefill runs a chunk at least once per
/// `2 * (interval + 1)` wall ticks. This drives the scheduler's real flag
/// protocol (`run()` sets `speculative_slice_yielded` after a round and clears
/// it after any classic arm) over the real policy.
#[test]
fn a_speculative_slice_cannot_starve_the_parked_prefill() {
    const INTERVAL: u32 = 4;
    let mut state = TickState {
        speculative_pending: true,
        speculative_yielded: false,
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        prefill_grant_interval: INTERVAL,
        ..TickState::default()
    };

    let mut history: Vec<TickChoice> = Vec::new();
    for _ in 0..60 {
        let decision = decide_tick(&state);
        history.push(decision.choice);
        // Exactly what `run()` does with the outcome.
        state.speculative_yielded = decision.choice == TickChoice::SpeculativeRound;
        state.decode_ticks_since_prefill_grant = decision.decode_ticks_since_prefill_grant;
    }

    // #734 strict alternation survives: no two consecutive speculative rounds,
    // and no two consecutive classic ticks.
    for pair in history.windows(2) {
        let spec = |c: &TickChoice| *c == TickChoice::SpeculativeRound;
        assert_ne!(
            spec(&pair[0]),
            spec(&pair[1]),
            "the grant must not break #734 alternation: {history:?}"
        );
    }

    // The prefill still gets its chunks, and the gap between them is bounded by
    // twice the classic bound because half the ticks belong to the slice.
    let grant_ticks: Vec<usize> = history
        .iter()
        .enumerate()
        .filter(|(_, c)| **c == TickChoice::Prefill)
        .map(|(i, _)| i)
        .collect();
    assert!(
        grant_ticks.len() >= 5,
        "the parked prefill must keep advancing under speculative contention: {history:?}"
    );
    for pair in grant_ticks.windows(2) {
        assert!(
            pair[1] - pair[0] <= 2 * (INTERVAL as usize + 1),
            "grant gap {} exceeds the doubled bound: {history:?}",
            pair[1] - pair[0]
        );
    }
}

// -------------------------------------------------------------------
// Env gates
// -------------------------------------------------------------------

#[test]
fn mixed_step_defaults_off() {
    assert!(!mixed_step_default(None));
    assert!(!mixed_step_default(Some("")));
    assert!(!mixed_step_default(Some("0")));
    assert!(!mixed_step_default(Some("false")));
    assert!(!mixed_step_default(Some("off")));
    // Unrecognised values stay off: this flag changes scheduler arbitration,
    // so an ambiguous value must not opt in.
    assert!(!mixed_step_default(Some("maybe")));
    assert!(!mixed_step_default(Some("2")));
}

#[test]
fn mixed_step_opt_in_values() {
    for v in [
        "1", "true", "TRUE", "True", "TrUe", "yes", "YES", "on", "ON", "On", " on ", "\ttrue\n",
    ] {
        assert!(
            mixed_step_default(Some(v)),
            "{v:?} should enable the prototype"
        );
    }
}

/// `MLXCEL_PREFILL_GRANT_INTERVAL` parsing (issue #1011).
///
/// Note the asymmetry with `MLXCEL_MIXED_STEP`: an unparseable value falls back
/// to the shipped default, not to 0. The conservative direction for a
/// prototype gate is "stay off"; the conservative direction for a fairness
/// policy is "stay fair", because falling back to 0 would silently reinstate
/// the unbounded TTFT on a typo.
#[test]
fn prefill_grant_interval_env_parsing() {
    assert_eq!(
        prefill_grant_interval_default(None),
        PREFILL_GRANT_INTERVAL_DEFAULT
    );
    assert_eq!(prefill_grant_interval_default(Some("4")), 4);
    assert_eq!(prefill_grant_interval_default(Some(" 12 ")), 12);
    assert_eq!(prefill_grant_interval_default(Some("1")), 1);
    // Only an exact 0 disables the grant.
    assert_eq!(prefill_grant_interval_default(Some("0")), 0);
    for garbage in ["", "abc", "-3", "8.5", "eight", "16x"] {
        assert_eq!(
            prefill_grant_interval_default(Some(garbage)),
            PREFILL_GRANT_INTERVAL_DEFAULT,
            "{garbage:?} must fall back to the default rather than disabling fairness"
        );
    }
}

/// The CLI value wins over the environment, and both reach the policy.
#[test]
fn prefill_grant_interval_cli_overrides_env() {
    // An explicit CLI value short-circuits before the env is read, so this is
    // deterministic regardless of the ambient environment.
    assert_eq!(resolve_prefill_grant_interval(Some(3)), 3);
    assert_eq!(resolve_prefill_grant_interval(Some(0)), 0);
    // Absurd values saturate rather than wrapping into a small interval.
    assert_eq!(resolve_prefill_grant_interval(Some(usize::MAX)), u32::MAX);
}

/// The mixed-step counter is the prototype's dispatch proof, so it has to be
/// reachable from a snapshot. A benchmark reading a counter that nothing
/// increments cannot tell "the arm did not engage" from "the wiring is broken".
/// The same argument applies to the #1011 grant counter, which is the only
/// counter that can distinguish a fairness-enabled server from a disabled one.
#[test]
fn tick_policy_counters_reach_the_snapshot() {
    use crate::server::batch::observability::BatchObservability;

    let obs = BatchObservability::new();
    assert_eq!(obs.snapshot().mixed_steps_processed, 0);
    obs.record_mixed_step();
    obs.record_mixed_step();
    assert_eq!(obs.snapshot().mixed_steps_processed, 2);

    assert_eq!(obs.snapshot().prefill_grants_processed, 0);
    obs.record_prefill_grant();
    assert_eq!(obs.snapshot().prefill_grants_processed, 1);

    // Neither counter is a side effect of the other, and nothing else moves
    // them, so a zero delta in a benchmark means the arm did not engage rather
    // than that the counter is inert.
    assert_eq!(obs.snapshot().mixed_steps_processed, 2);
    assert_eq!(obs.snapshot().decode_steps_processed, 0);
}
