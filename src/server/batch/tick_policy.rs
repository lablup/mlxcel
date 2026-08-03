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

//! Pure tick-arbitration policy for the batch scheduler (issue #908).
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
//! # The starvation this module pins
//!
//! The pre-#908 policy resolves a tick with a chunked prefill in progress as
//! follows: if any sequence is decoding, decode; only when the active batch is
//! empty does the chunked prefill advance. Nothing in the policy alternates,
//! and the policy is a pure function of scheduler state, so the state that
//! selects `Decode` is unchanged by running a decode. A long prompt admitted
//! next to a live decode batch therefore runs chunk 0 and then makes no further
//! progress until every decoding sequence has finished.
//!
//! That is the opposite of the behaviour the surrounding comments claim, and it
//! inverts the latency problem issue #908 set out to solve: decode streams are
//! never blocked by a chunked prefill, the chunked prefill is blocked by them.
//! [`TickChoice::MixedStep`] is the opt-in prototype that makes the interleave
//! real; see `docs/adr/0005-mixed-prefill-decode-step-execution.md`.

/// Whether the mixed prefill/decode step prototype is enabled (issue #908).
///
/// Default **off**: with `MLXCEL_MIXED_STEP` unset the policy is byte-identical
/// to the pre-#908 scheduler. `MLXCEL_MIXED_STEP=1|true|yes|on` opts in.
pub(crate) fn mixed_step_enabled() -> bool {
    mixed_step_default(std::env::var("MLXCEL_MIXED_STEP").ok().as_deref())
}

/// Pure decision core of [`mixed_step_enabled`], separated for unit testing.
///
/// Unset, empty, or unrecognised means off. Only the explicit affirmative set
/// turns the prototype on, which is the conservative direction for a flag that
/// changes scheduler arbitration.
pub(crate) fn mixed_step_default(env_override: Option<&str>) -> bool {
    match env_override {
        Some(v) => matches!(
            v,
            "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "on" | "ON"
        ),
        None => false,
    }
}

/// The scheduler state the tick policy reads, flattened to plain booleans.
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

/// Select this tick's action.
///
/// Preserved from the pre-#908 scheduler exactly, with one addition: the
/// `MixedStep` branch, reachable only when `state.mixed_step_enabled` is true.
/// With that flag false every input maps to the same choice the pre-#908
/// policy made.
pub(crate) fn decide_tick(state: &TickState) -> TickChoice {
    // Tick-cooperative speculative slice work pending (issue #734): run one
    // speculative action per tick, alternating strictly with the classic
    // actions when they have work, so concurrent classic rows advance between
    // rounds and the speculative request never starves. When nothing else has
    // work the slice takes every tick.
    if state.speculative_pending {
        let others_have_work =
            state.chunked_prefill_in_progress || !state.active_is_empty || !state.queue_is_empty;
        if super::speculative_slice::slice_takes_tick(state.speculative_yielded, others_have_work) {
            return TickChoice::SpeculativeRound;
        }
        // Fall through: grant this tick to a classic action; its arm clears
        // `speculative_slice_yielded` so the next tick returns to the slice.
    }

    // A chunked prefill is parked mid-prompt.
    if state.chunked_prefill_in_progress {
        if !state.active_is_empty {
            // Default (`MLXCEL_MIXED_STEP` unset): decode wins every tick, and
            // because this policy is a pure function of state, it keeps winning
            // until the batch drains. The parked prefill makes no progress in
            // the meantime. See the module docs; this is pinned by
            // `chunked_prefill_starves_until_active_batch_drains`.
            if state.mixed_step_enabled {
                return TickChoice::MixedStep;
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

#[cfg(test)]
#[path = "tick_policy_tests.rs"]
mod tick_policy_tests;
