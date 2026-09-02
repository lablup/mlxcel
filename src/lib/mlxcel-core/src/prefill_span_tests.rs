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

//! Unit tests for the prefill-span announcement (#1358).

use super::{announce, current};

#[test]
fn nothing_is_announced_outside_a_prefill() {
    assert_eq!(current(), None);
}

#[test]
fn an_announcement_is_visible_and_ends_with_its_guard() {
    {
        let _span = announce(5136);
        assert_eq!(current(), Some(5136));
    }
    assert_eq!(current(), None);
}

#[test]
fn a_nested_announcement_restores_the_outer_one() {
    let _outer = announce(9000);
    {
        let _inner = announce(120);
        assert_eq!(current(), Some(120));
    }
    assert_eq!(
        current(),
        Some(9000),
        "the outer prefill must not inherit the inner prompt's length"
    );
}

#[test]
fn a_non_positive_length_announces_nothing() {
    let _span = announce(0);
    assert_eq!(current(), None);
    let _negative = announce(-1);
    assert_eq!(current(), None);
}

#[test]
fn an_announcement_does_not_reach_another_thread() {
    // The server drives prefill and decode from the scheduler thread, so an
    // announcement must never be visible to a thread that did not make it.
    let _span = announce(5136);
    let seen = std::thread::spawn(current).join().expect("thread joins");
    assert_eq!(seen, None);
    assert_eq!(current(), Some(5136));
}
