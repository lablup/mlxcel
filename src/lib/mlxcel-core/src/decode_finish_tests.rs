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

//! Order and edge cases of [`finish_step`] (#2168).

use super::*;

const EOS: i32 = 2;

/// Hooks that fire on chosen tokens and record what they were asked.
#[derive(Default)]
struct ScriptedHooks {
    stop_on: Option<i32>,
    bound: bool,
    context_at: Option<usize>,
    stop_calls: Vec<(i32, usize)>,
}

impl FinishHooks for ScriptedHooks {
    fn stop_text(&mut self, token: i32, generated_len: usize) -> bool {
        self.stop_calls.push((token, generated_len));
        self.stop_on == Some(token)
    }

    fn bound_stopped(&self) -> bool {
        self.bound
    }

    fn context_bound_due(&self, generated_len: usize) -> bool {
        self.context_at.is_some_and(|at| generated_len >= at)
    }
}

fn step(
    token: i32,
    generated: &mut Vec<i32>,
    max_tokens: usize,
    structured_stopped: bool,
    loop_detection: &LoopDetectionConfig,
    hooks: &mut dyn FinishHooks,
) -> Option<FinishCause> {
    finish_step(
        FinishInput {
            token,
            eos: &[EOS],
            generated,
            history: None,
            max_tokens,
            structured_stopped,
            loop_detection,
        },
        hooks,
    )
}

#[test]
fn eos_on_the_first_token_is_not_pushed_and_skips_every_hook() {
    let mut generated = Vec::new();
    let mut history = vec![9];
    let mut hooks = ScriptedHooks {
        stop_on: Some(EOS),
        bound: true,
        ..Default::default()
    };
    let cause = finish_step(
        FinishInput {
            token: EOS,
            eos: &[EOS],
            generated: &mut generated,
            history: Some(&mut history),
            max_tokens: 1,
            structured_stopped: true,
            loop_detection: &LoopDetectionConfig::disabled(),
        },
        &mut hooks,
    );
    assert_eq!(cause, Some(FinishCause::Eos));
    assert!(generated.is_empty());
    assert_eq!(history, vec![9]);
    assert!(hooks.stop_calls.is_empty());
}

#[test]
fn a_non_eos_token_is_pushed_to_generated_and_history() {
    let mut generated = vec![5];
    let mut history = vec![1, 5];
    let mut hooks = ScriptedHooks::default();
    let cause = finish_step(
        FinishInput {
            token: 7,
            eos: &[EOS],
            generated: &mut generated,
            history: Some(&mut history),
            max_tokens: 10,
            structured_stopped: false,
            loop_detection: &LoopDetectionConfig::disabled(),
        },
        &mut hooks,
    );
    assert_eq!(cause, None);
    assert_eq!(generated, vec![5, 7]);
    assert_eq!(history, vec![1, 5, 7]);
    // The stop matcher sees the token after the push.
    assert_eq!(hooks.stop_calls, vec![(7, 2)]);
}

#[test]
fn max_tokens_one_finishes_length_on_the_prefill_token() {
    let mut generated = Vec::new();
    let cause = step(
        7,
        &mut generated,
        1,
        false,
        &LoopDetectionConfig::disabled(),
        &mut NoStopHooks,
    );
    assert_eq!(cause, Some(FinishCause::Length));
    assert_eq!(generated, vec![7]);
}

#[test]
fn a_stop_string_and_max_tokens_on_the_same_token_report_stop_sequence() {
    let mut generated = Vec::new();
    let mut hooks = ScriptedHooks {
        stop_on: Some(7),
        ..Default::default()
    };
    let cause = step(
        7,
        &mut generated,
        1,
        false,
        &LoopDetectionConfig::disabled(),
        &mut hooks,
    );
    assert_eq!(cause, Some(FinishCause::StopSequence));
}

#[test]
fn checks_run_in_the_reference_order() {
    let looping = LoopDetectionConfig::new(1, 1, 2);
    // Every condition holds on the same token: the stop string wins.
    let mut hooks = ScriptedHooks {
        stop_on: Some(7),
        bound: true,
        context_at: Some(0),
        ..Default::default()
    };
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 1, true, &looping, &mut hooks),
        Some(FinishCause::StopSequence)
    );

    // Without the stop string, the generation bound wins (reported `Length`).
    hooks.stop_on = None;
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 100, true, &looping, &mut hooks),
        Some(FinishCause::Length)
    );

    // Without the bound, the structured stop wins over the budget.
    hooks.bound = false;
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 1, true, &looping, &mut hooks),
        Some(FinishCause::StructuredStop)
    );

    // Without the structured stop, the budget wins over the context bound.
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 1, false, &looping, &mut hooks),
        Some(FinishCause::Length)
    );

    // Without the budget, the context bound wins over the repetition loop.
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 100, false, &looping, &mut hooks),
        Some(FinishCause::ContextExhausted)
    );

    // Last, the repetition loop.
    hooks.context_at = None;
    let mut generated = vec![7];
    assert_eq!(
        step(7, &mut generated, 100, false, &looping, &mut hooks),
        Some(FinishCause::RepetitionLoop)
    );
}

#[test]
fn the_context_bound_sees_the_post_push_length() {
    let mut hooks = ScriptedHooks {
        context_at: Some(3),
        ..Default::default()
    };
    let disabled = LoopDetectionConfig::disabled();
    let mut generated = vec![4];
    assert_eq!(
        step(5, &mut generated, 100, false, &disabled, &mut hooks),
        None
    );
    assert_eq!(
        step(6, &mut generated, 100, false, &disabled, &mut hooks),
        Some(FinishCause::ContextExhausted)
    );
}

#[test]
fn disabled_loop_detection_never_ends_a_repeating_stream() {
    let disabled = LoopDetectionConfig::disabled();
    let mut generated = Vec::new();
    for _ in 0..256 {
        assert_eq!(
            step(
                7,
                &mut generated,
                usize::MAX,
                false,
                &disabled,
                &mut NoStopHooks
            ),
            None
        );
    }
    assert_eq!(generated.len(), 256);
}

#[test]
fn enabled_loop_detection_ends_on_the_completing_repeat() {
    // A 2-token block repeated 3 times: the sixth token completes the loop.
    let cfg = LoopDetectionConfig::new(1, 4, 3);
    let mut generated = Vec::new();
    let stream = [10, 11, 10, 11, 10, 11];
    for (i, &token) in stream.iter().enumerate() {
        let cause = step(
            token,
            &mut generated,
            usize::MAX,
            false,
            &cfg,
            &mut NoStopHooks,
        );
        if i + 1 < stream.len() {
            assert_eq!(cause, None, "token {i} must not finish");
        } else {
            assert_eq!(cause, Some(FinishCause::RepetitionLoop));
        }
    }
    // The looping token is pushed: the caller decides whether it is emitted.
    assert_eq!(generated, stream);
}

#[test]
fn no_stop_hooks_never_fire() {
    let mut hooks = NoStopHooks;
    assert!(!hooks.stop_text(1, 1));
    assert!(!hooks.bound_stopped());
    assert!(!hooks.context_bound_due(usize::MAX));
}
