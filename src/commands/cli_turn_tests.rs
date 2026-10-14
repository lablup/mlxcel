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

//! Terminal-side helpers of one in-process server turn (issue #2173).

use super::*;

#[test]
fn a_delta_is_written_verbatim_when_not_dimmed() {
    let mut out = Vec::new();
    write_delta(&mut out, "hello \u{1f600}", false).expect("write");
    assert_eq!(out, "hello \u{1f600}".as_bytes());
}

#[test]
fn a_dimmed_delta_is_wrapped_in_the_dim_escape() {
    let mut out = Vec::new();
    write_delta(&mut out, "thinking", true).expect("write");
    assert_eq!(out, format!("{DIM}thinking{RESET}").into_bytes());
}

#[test]
fn the_first_interrupt_cancels_and_the_second_terminates() {
    let flag = AtomicBool::new(false);
    assert_eq!(interrupt_action(Some(&flag)), InterruptAction::Cancel);
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(interrupt_action(Some(&flag)), InterruptAction::Terminate);
    assert!(flag.load(Ordering::Acquire));
}

#[test]
fn an_interrupt_outside_a_turn_terminates() {
    assert_eq!(interrupt_action(None), InterruptAction::Terminate);
}

#[test]
fn the_base_model_notice_asks_about_the_template_only_when_it_may_print() {
    assert!(base_model_notice_due(false, || true));
    assert!(!base_model_notice_due(false, || false));
    // `--no-chat-template` never asks.
    assert!(!base_model_notice_due(true, || {
        panic!("the template question must not be asked")
    }));
}
