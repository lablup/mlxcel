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

//! Tool calls emitted inside a reasoning block (#3209 item b).
//!
//! mlx-lm#1973 executes a call the model wrote inside a reasoning block it
//! closes later; mlx-vlm#2613 never does. mlxcel takes the mlx-vlm rule for a
//! closed block, and the mlx-lm one only for the unambiguous case: a complete
//! call that ends an unclosed block with nothing after it.
//!
//! Every case runs through the streaming filter (whole, and in chunks of 1, 5
//! and 17 bytes) and through the non-streamed cleanup the routes use, and the
//! two must agree on content, reasoning and calls.

use super::stream_filter::StreamFilter;
use super::{ParsedToolCall, parse_tool_calls_with_context};
use crate::server::routes::chat::extract_reasoning_content;
use crate::server::types::request::{FunctionDefinition, Tool};

fn tools() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            })),
        },
    }]
}

/// One tool-call syntax: how a `get_weather(city)` call is written, and the
/// thinking pair the family uses.
struct Syntax {
    open: &'static str,
    close: &'static str,
    call: fn(&str) -> String,
}

fn hermes_call(city: &str) -> String {
    format!(r#"<tool_call>{{"name":"get_weather","arguments":{{"city":"{city}"}}}}</tool_call>"#)
}

fn gemma4_call(city: &str) -> String {
    format!(r#"<|tool_call>call:get_weather{{city:<|"|>{city}<|"|>}}<tool_call|>"#)
}

const HERMES: Syntax = Syntax {
    open: "<think>",
    close: "</think>",
    call: hermes_call,
};

const GEMMA4: Syntax = Syntax {
    open: "<|channel>thought",
    close: "<channel|>",
    call: gemma4_call,
};

impl Syntax {
    fn call(&self, city: &str) -> String {
        (self.call)(city)
    }

    /// The call with its end marker cut off.
    fn unterminated(&self, city: &str) -> String {
        let call = self.call(city);
        let end = ["</tool_call>", "<tool_call|>"]
            .iter()
            .find_map(|end| call.strip_suffix(end))
            .expect("call ends with an end marker");
        end.to_string()
    }
}

/// What the client sees.
#[derive(Debug, PartialEq)]
struct Reply {
    content: String,
    reasoning: String,
    calls: Vec<ParsedToolCall>,
}

impl Reply {
    fn cities(&self) -> Vec<String> {
        self.calls
            .iter()
            .map(|call| {
                let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
                args["city"].as_str().unwrap().to_string()
            })
            .collect()
    }
}

/// The streaming route: `delta.content` / `delta.reasoning_content` from the
/// filter, calls from the end-of-stream parse.
fn streamed(pieces: &[&str], primed: bool) -> Reply {
    let mut filter = if primed {
        StreamFilter::new_primed_open_thinking()
    } else {
        StreamFilter::new()
    };
    let (mut content, mut reasoning, mut raw) = (String::new(), String::new(), String::new());
    let mut consumed = 0usize;
    for piece in pieces {
        raw.push_str(piece);
        let out = filter.feed(piece);
        consumed += out.consumed_positions;
        content.push_str(out.content.as_deref().unwrap_or(""));
        reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
    }
    let out = filter.flush();
    consumed += out.consumed_positions;
    content.push_str(out.content.as_deref().unwrap_or(""));
    reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
    assert_eq!(
        consumed,
        pieces.len(),
        "positions not reported exactly once for {pieces:?}"
    );
    let tools = tools();
    let calls = parse_tool_calls_with_context(&raw, Some(&tools), primed).tool_calls;
    Reply {
        content,
        reasoning,
        calls,
    }
}

/// The non-streamed route: content and calls from the parser, reasoning from
/// the filter replay (`extract_reasoning_content`).
fn non_streamed(raw: &str, primed: bool) -> Reply {
    let tools = tools();
    let parsed = parse_tool_calls_with_context(raw, Some(&tools), primed);
    Reply {
        content: parsed.content,
        reasoning: extract_reasoning_content(raw, primed).unwrap_or_default(),
        calls: parsed.tool_calls,
    }
}

fn pieces(raw: &str, width: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < raw.len() {
        let mut end = (start + width).min(raw.len());
        while !raw.is_char_boundary(end) {
            end += 1;
        }
        out.push(&raw[start..end]);
        start = end;
    }
    out
}

/// Both paths, every chunking; returns the agreed reply.
fn reply(raw: &str, primed: bool) -> Reply {
    let expected = non_streamed(raw, primed);
    for width in [raw.len().max(1), 1, 5, 17] {
        let chunks = pieces(raw, width);
        let got = streamed(&chunks, primed);
        assert_eq!(
            got, expected,
            "streamed (width {width}) != non-streamed for {raw:?}"
        );
    }
    expected
}

#[test]
fn closed_reasoning_example_is_not_executed() {
    for syntax in [HERMES, GEMMA4] {
        let example = syntax.call("Berlin");
        let real = syntax.call("Warsaw");
        let (open, close) = (syntax.open, syntax.close);

        let raw = format!("{open}example {example} then{close}{real}");
        let got = reply(&raw, false);
        assert_eq!(got.cities(), ["Warsaw"], "{raw:?}");
        assert!(got.reasoning.contains(&example), "{got:?}");
        assert!(got.reasoning.ends_with("then"), "{got:?}");
        assert_eq!(got.content, "", "{got:?}");

        let raw = format!("{open}example {example} then{close}Done.");
        let got = reply(&raw, false);
        assert!(got.calls.is_empty(), "{got:?}");
        assert!(got.reasoning.contains(&example), "{got:?}");
        assert_eq!(got.content, "Done.", "{got:?}");

        // A primed open (no open marker in the output) behaves the same.
        let raw = format!("example {example} then{close}{real}");
        let got = reply(&raw, true);
        assert_eq!(got.cities(), ["Warsaw"], "{raw:?}");
        assert!(got.reasoning.contains(&example), "{got:?}");
    }
}

#[test]
fn unterminated_example_does_not_swallow_real_call() {
    for syntax in [HERMES, GEMMA4] {
        let example = syntax.unterminated("Berlin");
        let real = syntax.call("Warsaw");
        let (open, close) = (syntax.open, syntax.close);
        let raw = format!("{open}example {example}{close}{real}");
        let got = reply(&raw, false);
        assert_eq!(got.cities(), ["Warsaw"], "{raw:?}");
        assert!(got.reasoning.contains(&example), "{got:?}");
        assert_eq!(got.content, "", "{got:?}");

        let raw = format!("{open}example {example}{close}Done.");
        let got = reply(&raw, false);
        assert!(got.calls.is_empty(), "{got:?}");
        assert_eq!(got.content, "Done.", "{got:?}");
    }
}

#[test]
fn tail_call_in_unclosed_reasoning_executes() {
    for syntax in [HERMES, GEMMA4] {
        let real = syntax.call("Warsaw");
        let open = syntax.open;
        for (raw, primed) in [
            (format!("{open}need weather{real}"), false),
            (format!("{open}need weather{real}\n "), false),
            (format!("need weather{real}"), true),
            (format!("need weather{real}\n"), true),
        ] {
            let got = reply(&raw, primed);
            assert_eq!(got.cities(), ["Warsaw"], "{raw:?}");
            assert_eq!(got.reasoning, "need weather", "{raw:?}");
            assert_eq!(got.content, "", "{raw:?}");
        }
    }
}

#[test]
fn non_tail_call_in_unclosed_reasoning_is_reasoning() {
    for syntax in [HERMES, GEMMA4] {
        let call = syntax.call("Warsaw");
        let open = syntax.open;
        for (raw, primed) in [
            (format!("{open}a{call}b"), false),
            (format!("a{call}b"), true),
            // A truncated call at the end is reasoning too.
            (format!("{open}a{}", syntax.unterminated("Warsaw")), false),
            (format!("a{}", syntax.unterminated("Warsaw")), true),
            // A call followed by another one: only the last can be the tail.
            (format!("{open}a{call} {}", syntax.call("Oslo")), false),
        ] {
            let got = reply(&raw, primed);
            let body = raw.strip_prefix(open).unwrap_or(&raw);
            if raw.ends_with("Oslo<|\"|>}<tool_call|>") || raw.ends_with("Oslo\"}}</tool_call>") {
                assert_eq!(got.cities(), ["Oslo"], "{raw:?}");
                assert_eq!(got.reasoning, format!("a{call} "), "{raw:?}");
            } else {
                assert!(got.calls.is_empty(), "{raw:?}: {got:?}");
                assert_eq!(got.reasoning, body, "{raw:?}");
            }
            assert_eq!(got.content, "", "{raw:?}");
        }
    }
}

#[test]
fn call_after_the_reasoning_close_is_unchanged() {
    // The ordinary case: reasoning, then a call outside it.
    for syntax in [HERMES, GEMMA4] {
        let real = syntax.call("Warsaw");
        let (open, close) = (syntax.open, syntax.close);
        let raw = format!("{open}plan{close}\n{real}");
        let got = reply(&raw, false);
        assert_eq!(got.cities(), ["Warsaw"]);
        assert_eq!(got.reasoning, "plan");
        assert_eq!(got.content, "");
    }
}

#[test]
fn oversized_in_reasoning_block_is_released_as_reasoning() {
    // Past the hold bound the block is an example: it streams as reasoning,
    // and the reasoning close still ends the block.
    let body = "x".repeat((1 << 20) + 16);
    let raw = format!("<think>a<tool_call>{body}</tool_call>b</think>Done.");
    let mut filter = StreamFilter::new();
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut fed = 0usize;
    for piece in pieces(&raw, 4096) {
        fed += piece.len();
        let out = filter.feed(piece);
        reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
        content.push_str(out.content.as_deref().unwrap_or(""));
        // What the filter still holds stays under the bound plus one chunk
        // and the markers.
        let held = fed - reasoning.len() - content.len();
        assert!(held <= (1 << 20) + 4096 + 64, "holding {held} bytes");
    }
    let out = filter.flush();
    reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
    content.push_str(out.content.as_deref().unwrap_or(""));
    assert_eq!(reasoning, format!("a<tool_call>{body}</tool_call>b"));
    assert_eq!(content, "Done.");
}

#[test]
fn many_reasoning_blocks_are_scanned_in_linear_time() {
    // Every closed block sends the unclosed-reasoning scan back to look for
    // the next open marker of each family. Absent families must not be
    // re-searched to the end of the text for every block, or a long
    // generation of short blocks costs blocks x length x families.
    let blocks = 200_000;
    let mut raw = "<think>a</think>b".repeat(blocks);
    raw.push_str("<think>need weather");
    raw.push_str(&hermes_call("Warsaw"));
    let started = std::time::Instant::now();
    let parsed = parse_tool_calls_with_context(&raw, Some(&tools()), false);
    let elapsed = started.elapsed();
    let cities: Vec<_> = parsed
        .tool_calls
        .iter()
        .map(|call| {
            serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap()["city"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(cities, ["Warsaw"]);
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "{blocks} blocks took {elapsed:?}"
    );
}
