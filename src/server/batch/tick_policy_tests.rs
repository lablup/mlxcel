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

//! Unit tests for the pure tick-arbitration policy (issue #908).
//!
//! These call [`decide_tick`] itself, not a copy of it. The whole reason this
//! module exists is that the previous tests re-implemented the policy locally
//! and drifted from it in their naming, which hid the chunked-prefill
//! starvation that issue #908 turned up.

use super::*;

/// Every state the policy reads, enumerated. Seven booleans, so the full space
/// is 128 states and the parity tests below are exhaustive rather than sampled.
fn all_states(mixed_step_enabled: bool) -> Vec<TickState> {
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
        });
    }
    out
}

/// The scheduler policy exactly as it stood before issue #908, transcribed from
/// `BatchScheduler::decide_action` at commit 230163db.
///
/// This is a deliberate mirror, which the module docs otherwise warn against.
/// It is safe here for one reason: it is compared against [`decide_tick`] over
/// the complete 128-state space, so any drift fails a test instead of hiding.
/// Its only job is to prove that `MLXCEL_MIXED_STEP` unset changes nothing.
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

// -------------------------------------------------------------------
// Default-off parity (mandatory regression guard from issue #908)
// -------------------------------------------------------------------

#[test]
fn mixed_step_off_is_identical_to_the_pre_908_policy() {
    for state in all_states(false) {
        assert_eq!(
            decide_tick(&state),
            pre_908_policy(&state),
            "MLXCEL_MIXED_STEP unset must not change arbitration; diverged at {state:?}"
        );
    }
}

#[test]
fn mixed_step_off_never_selects_mixed_step() {
    for state in all_states(false) {
        assert_ne!(
            decide_tick(&state),
            TickChoice::MixedStep,
            "MixedStep must be unreachable with the flag off; reached at {state:?}"
        );
    }
}

#[test]
fn mixed_step_on_differs_only_on_the_chunked_interleave_branch() {
    for state in all_states(true) {
        let off = TickState {
            mixed_step_enabled: false,
            ..state
        };
        let with = decide_tick(&state);
        let without = decide_tick(&off);
        if with == without {
            continue;
        }
        // The only divergence the prototype is allowed to introduce.
        assert_eq!(
            with,
            TickChoice::MixedStep,
            "unexpected divergence {state:?}"
        );
        assert_eq!(
            without,
            TickChoice::Decode,
            "unexpected divergence {state:?}"
        );
        assert!(state.chunked_prefill_in_progress, "{state:?}");
        assert!(!state.active_is_empty, "{state:?}");
    }
}

// -------------------------------------------------------------------
// The starvation this issue found (issue #908)
// -------------------------------------------------------------------

/// The finding behind ADR 0005: with a chunked prefill parked and any sequence
/// decoding, the policy selects `Decode`, and because it is a pure function of
/// state that a decode does not change, it selects `Decode` again next tick,
/// forever. The parked prompt makes no progress until the batch drains.
///
/// This is the inverse of the premise issue #908 opened with (decode streams
/// stalling behind prefill chunks). Decode never stalls here; the long prompt
/// does.
#[test]
fn chunked_prefill_starves_until_active_batch_drains() {
    let state = TickState {
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        ..TickState::default()
    };

    // Decoding does not alter any field the policy reads, so the tick sequence
    // is a fixed point. 1000 ticks stands in for "unbounded".
    for tick in 0..1000 {
        assert_eq!(
            decide_tick(&state),
            TickChoice::Decode,
            "tick {tick}: parked chunked prefill should still be starved"
        );
    }

    // The prefill resumes only once the last decoding sequence completes.
    let drained = TickState {
        active_is_empty: true,
        active_is_full: false,
        ..state
    };
    assert_eq!(decide_tick(&drained), TickChoice::Prefill);
}

/// The prototype converts that fixed point into a real interleave: every tick
/// advances both workloads, so the parked prompt progresses while the batch is
/// still decoding.
#[test]
fn mixed_step_advances_both_workloads_every_tick() {
    let state = TickState {
        chunked_prefill_in_progress: true,
        active_is_empty: false,
        active_is_full: true,
        queue_is_empty: true,
        mixed_step_enabled: true,
        ..TickState::default()
    };
    for tick in 0..1000 {
        assert_eq!(
            decide_tick(&state),
            TickChoice::MixedStep,
            "tick {tick}: mixed step should advance prefill and decode together"
        );
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
        ..TickState::default()
    };
    assert_eq!(decide_tick(&state), TickChoice::Prefill);
}

// -------------------------------------------------------------------
// Speculative slice arbitration keeps priority over the prototype
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
    assert_eq!(decide_tick(&state), TickChoice::SpeculativeRound);
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
    assert_eq!(decide_tick(&state), TickChoice::MixedStep);
}

// -------------------------------------------------------------------
// Env gate
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
    for v in ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"] {
        assert!(
            mixed_step_default(Some(v)),
            "{v} should enable the prototype"
        );
    }
}
