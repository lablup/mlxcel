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

use super::*;

fn qwen() -> ThinkingMarkers {
    ThinkingMarkers {
        think_start: Some("<think>".to_string()),
        think_end: Some("</think>".to_string()),
        ..ThinkingMarkers::default()
    }
}

#[test]
fn primed_prompt_detection_tolerates_trailing_whitespace() {
    assert!(prompt_primed_open_thinking(
        &qwen(),
        "<|im_start|>assistant\n<think>\n"
    ));
    assert!(!prompt_primed_open_thinking(
        &qwen(),
        "<|im_start|>assistant\n"
    ));
    assert!(!prompt_primed_open_thinking(
        &ThinkingMarkers::default(),
        "<think>\n"
    ));
    assert_eq!(
        prompt_primed_open_close_marker(&qwen(), "x<think>\n").as_deref(),
        Some("</think>")
    );
    assert_eq!(prompt_primed_open_close_marker(&qwen(), "x"), None);
}

#[test]
fn render_full_hides_reasoning_unless_asked() {
    let text = "<think>plan</think>answer";
    assert_eq!(render_full(text, false, false, false), "answer");
    assert_eq!(render_full(text, false, true, false), "plananswer");
    assert_eq!(
        render_full(text, false, true, true),
        format!("{DIM}plan{RESET}answer")
    );
}

#[test]
fn render_full_starts_inside_a_primed_block() {
    assert_eq!(
        render_full("plan</think>answer", true, false, false),
        "answer"
    );
    // A block the model never closed stays hidden.
    assert_eq!(render_full("still thinking", true, false, false), "");
}

#[test]
fn plain_text_passes_through() {
    assert_eq!(
        render_full("hello world", false, false, false),
        "hello world"
    );
}

#[test]
fn reasoning_only_needs_hidden_text_and_no_visible_output() {
    assert!(is_reasoning_only("<think>x</think>", false, false));
    assert!(!is_reasoning_only("<think>x</think>", true, false));
    assert!(!is_reasoning_only("<think>x</think>", false, true));
    assert!(!is_reasoning_only("  ", false, false));
}
