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

//! Unit tests for the disaggregated router front-end's usage accounting
//! (issue #387).
//!
//! Model-free: exercise `resolve_completion_tokens`, the pure function that
//! turns the worker's authoritative wire-carried token count (or the emitted-
//! piece fallback) into the reported `usage.completion_tokens`.

use super::*;

/// When the worker reports an authoritative count, it is used verbatim even if
/// it differs from the emitted-piece count. This is the byte-fallback case: a
/// Gemma `<0xXX>` sequence produces fewer text pieces than model tokens, so the
/// frame count under-counts and the authoritative count corrects it.
#[test]
fn prefers_authoritative_count_over_frame_count() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(10),
    };
    // Three multi-byte characters surfaced as 3 text pieces, but the worker
    // generated 10 model tokens (byte-fallback expansion).
    assert_eq!(resolve_completion_tokens(&outcome, 3, 16), 10);
}

/// With no authoritative count on the wire (a mixed-version cluster running an
/// older prefill or decode node), the router falls back to the emitted-piece
/// count, preserving the prior behavior.
#[test]
fn falls_back_to_frame_count_when_no_authoritative_count() {
    let outcome = HandoffOutcome {
        generated_tokens: None,
    };
    assert_eq!(resolve_completion_tokens(&outcome, 7, 16), 7);
}

/// When no authoritative count is available and the emitted-piece count exceeds
/// `max_tokens` (which can happen for a hostile or buggy node in a mixed-version
/// cluster), the fallback is clamped to `max_tokens` for uniform defensiveness.
#[test]
fn fallback_frame_count_clamped_to_max_tokens() {
    let outcome = HandoffOutcome {
        generated_tokens: None,
    };
    // Frame count exceeds the budget; the fallback must not report more than allowed.
    assert_eq!(resolve_completion_tokens(&outcome, 9999, 16), 16);
}

/// The authoritative count is clamped to `max_tokens`: the router bounds
/// generation to its own budget, so a larger reported count (a buggy or hostile
/// node) must not inflate the usage figure.
#[test]
fn clamps_authoritative_count_to_max_tokens() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(9999),
    };
    assert_eq!(resolve_completion_tokens(&outcome, 4, 16), 16);
}

/// An authoritative count exactly at the budget is reported as-is and drives the
/// "length" finish_reason (`count >= max_tokens`), matching single-node.
#[test]
fn authoritative_count_at_budget_is_length() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(16),
    };
    let max_tokens = 16;
    let completion_tokens = resolve_completion_tokens(&outcome, 16, max_tokens);
    assert_eq!(completion_tokens, 16);
    assert!(
        completion_tokens >= max_tokens,
        "should map to finish=length"
    );
}

/// A byte-fallback under-count that would have flipped the finish_reason: the
/// frame count (15) is below the budget (16) and would report "stop", but the
/// authoritative count (16) correctly reports "length".
#[test]
fn authoritative_count_fixes_finish_reason_flip() {
    let max_tokens = 16;
    let frame_counted = 15; // emitted-piece under-count
    let authoritative = HandoffOutcome {
        generated_tokens: Some(16),
    };
    let no_count = HandoffOutcome {
        generated_tokens: None,
    };

    let fixed = resolve_completion_tokens(&authoritative, frame_counted, max_tokens);
    let legacy = resolve_completion_tokens(&no_count, frame_counted, max_tokens);

    assert!(fixed >= max_tokens, "authoritative count reports length");
    assert!(
        legacy < max_tokens,
        "frame-count fallback would have reported stop"
    );
}
