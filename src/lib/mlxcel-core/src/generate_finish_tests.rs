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

//! The CLI decode-loop site class of the shared finish step (#2168).
//!
//! The token streams are the ones `src/server/batch/finish_step_tests.rs`
//! feeds the server site classes, so the CLI reports the same
//! [`FinishCause`] for EOS, the budget and a repetition loop, while stop
//! strings, generation bounds and the context bound never fire
//! ([`NoStopHooks`]).

use super::{cli_emits_final_token, cli_finish_step};
use crate::decode_finish::FinishCause;
use crate::loop_detection::LoopDetectionConfig;

const EOS: i32 = 1;

fn bytes(s: &str) -> Vec<i32> {
    s.bytes().map(i32::from).collect()
}

/// What a streaming `CxxGenerator` loop produced.
struct CliRun {
    emitted: Vec<i32>,
    generated: Vec<i32>,
    finish: Option<FinishCause>,
}

/// The per-token tail of `generate_streaming`: the zero-budget guard, the
/// shared finish step, and the callback hand-off.
fn run_cli_stream(
    tokens: &[i32],
    max_tokens: usize,
    loop_detection: LoopDetectionConfig,
) -> CliRun {
    let mut emitted = Vec::new();
    let mut generated = Vec::new();
    let mut finish = None;
    for (n, &token) in tokens.iter().enumerate() {
        if n >= max_tokens {
            break;
        }
        match cli_finish_step(
            token,
            &[EOS],
            &mut generated,
            None,
            max_tokens,
            &loop_detection,
        ) {
            None => emitted.push(token),
            Some(cause) => {
                if cli_emits_final_token(cause) {
                    emitted.push(token);
                }
                finish = Some(cause);
                break;
            }
        }
    }
    CliRun {
        emitted,
        generated,
        finish,
    }
}

#[test]
fn eos_ends_the_cli_loop_without_storing_or_emitting_it() {
    let run = run_cli_stream(&[97, 98, EOS, 99], 16, LoopDetectionConfig::disabled());
    assert_eq!(run.finish, Some(FinishCause::Eos));
    assert_eq!(run.generated, bytes("ab"));
    assert_eq!(run.emitted, bytes("ab"));
}

#[test]
fn the_token_that_spends_the_budget_is_emitted_then_the_loop_ends() {
    let run = run_cli_stream(&bytes("abcd"), 3, LoopDetectionConfig::disabled());
    assert_eq!(run.finish, Some(FinishCause::Length));
    assert_eq!(run.generated, bytes("abc"));
    assert_eq!(run.emitted, bytes("abc"));
}

#[test]
fn a_zero_budget_emits_nothing() {
    let run = run_cli_stream(&bytes("abc"), 0, LoopDetectionConfig::disabled());
    assert_eq!(run.finish, None);
    assert!(run.generated.is_empty());
    assert!(run.emitted.is_empty());
}

#[test]
fn a_repetition_loop_withholds_its_looping_token() {
    let run = run_cli_stream(&bytes("abababab"), 16, LoopDetectionConfig::new(1, 4, 3));
    assert_eq!(run.finish, Some(FinishCause::RepetitionLoop));
    assert_eq!(run.generated, bytes("ababab"));
    assert_eq!(run.emitted, bytes("ababa"));
}

#[test]
fn stop_strings_bounds_and_the_context_bound_never_fire_on_the_cli() {
    // The server's stop-string, generation-bound and context-bound streams
    // run to their end on the CLI.
    for stream in ["xySTOPz", "x\nyz", "abcd"] {
        let tokens = bytes(stream);
        let run = run_cli_stream(&tokens, 16, LoopDetectionConfig::disabled());
        assert_eq!(run.finish, None, "{stream:?} must not finish on the CLI");
        assert_eq!(run.emitted, tokens);
        assert_eq!(run.generated, tokens);
    }
}

#[test]
fn the_cli_records_history_only_when_the_loop_owns_it() {
    let disabled = LoopDetectionConfig::disabled();
    let mut generated = Vec::new();
    let mut history = vec![7];
    assert_eq!(
        cli_finish_step(5, &[EOS], &mut generated, Some(&mut history), 4, &disabled),
        None
    );
    assert_eq!(
        cli_finish_step(6, &[EOS], &mut generated, None, 4, &disabled),
        None
    );
    assert_eq!(generated, vec![5, 6]);
    assert_eq!(history, vec![7, 5]);
}
