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
    for text in [
        "hello world",
        "answer  \n",
        " \t\r\n",
        "안녕\u{2003}\n",
        "answer <thi  \n",
    ] {
        for show_reasoning in [false, true] {
            assert_eq!(render_full(text, false, show_reasoning, false), text);
        }
    }
}

#[test]
fn preserving_plain_whitespace_keeps_structured_cleanup() {
    for (text, primed, want) in [
        ("<think>plan</think>answer  \n", false, "answer"),
        ("plan</think>answer  \n", true, "answer"),
        ("unclosed thought  \n", true, ""),
        ("answer \n<tool_call>{}</tool_call>  \n", false, "answer"),
        ("answer \n<tool_call>unfinished  \n", false, "answer"),
        ("<|channel|>final<|message|> answer  \n", false, "answer"),
        ("to=user<|message|>answer  \n", false, "answer"),
        ("answer <|channel>", false, "answer"),
    ] {
        assert_eq!(render_full(text, primed, false, false), want);
        let mut filter = if primed {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        };
        let mut streamed = String::new();
        for ch in text.chars() {
            streamed.push_str(
                filter
                    .feed(&ch.to_string())
                    .content
                    .as_deref()
                    .unwrap_or_default(),
            );
        }
        streamed.push_str(
            filter
                .flush_preserving_plain_whitespace()
                .content
                .as_deref()
                .unwrap_or_default(),
        );
        assert_eq!(streamed, want, "one character per feed: {text:?}");
    }
}

#[test]
fn plain_whitespace_flush_is_chunk_independent_and_accounts_positions() {
    let text = "answer  \n";
    for chunks in [vec![text], vec!["answer", " ", " ", "\n"]] {
        let mut filter = StreamFilter::new();
        let mut content = String::new();
        let mut consumed = 0;
        let mut suppressed = 0;
        for chunk in &chunks {
            let out = filter.feed(chunk);
            content.push_str(out.content.as_deref().unwrap_or_default());
            consumed += out.consumed_positions;
            suppressed += out.suppressed_positions;
        }
        let tail = filter.flush_preserving_plain_whitespace();
        content.push_str(tail.content.as_deref().unwrap_or_default());
        consumed += tail.consumed_positions;
        suppressed += tail.suppressed_positions;
        assert_eq!(content, text);
        assert_eq!(consumed, chunks.len());
        assert_eq!(suppressed, 0);
        assert_eq!(
            filter.flush_preserving_plain_whitespace(),
            FilterOutput::default()
        );
    }
    // Raw-display opt-in must not change the established chat default.
    let mut filter = StreamFilter::new();
    let first = filter.feed(text);
    let last = filter.flush();
    assert_eq!(
        format!(
            "{}{}",
            first.content.unwrap_or_default(),
            last.content.unwrap_or_default()
        ),
        "answer"
    );
}

#[test]
fn reasoning_only_needs_hidden_text_and_no_visible_output() {
    assert!(is_reasoning_only("<think>x</think>", false, false));
    assert!(!is_reasoning_only("<think>x</think>", true, false));
    assert!(!is_reasoning_only("<think>x</think>", false, true));
    assert!(!is_reasoning_only("  ", false, false));
}
