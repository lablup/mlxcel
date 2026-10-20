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

//! Incremental tool-call streaming (#2396): every case feeds one model output
//! one character at a time (and under other chunkings) and checks the
//! streamed calls against the authoritative `parse_tool_calls` result.

use std::collections::BTreeMap;

use serde_json::json;

use super::{IncrementalToolCallStreamer, ToolCallStreamEvent};
use crate::server::tool_calls::parse_tool_calls;
use crate::server::types::request::{FunctionDefinition, Tool};

fn tool(name: &str, parameters: serde_json::Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
        },
    }
}

fn tools() -> Vec<Tool> {
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"}
            }}),
        ),
        tool(
            "get_time",
            json!({"type": "object", "properties": {"tz": {"type": "string"}}}),
        ),
    ]
}

/// What a client accumulates from the events, per call index.
#[derive(Debug, Default, Clone, PartialEq)]
struct Call {
    name: String,
    id: String,
    arguments: String,
    deltas: usize,
}

fn collect(events: &[ToolCallStreamEvent]) -> BTreeMap<usize, Call> {
    let mut calls: BTreeMap<usize, Call> = BTreeMap::new();
    for event in events {
        match event {
            ToolCallStreamEvent::Start { index, name, id } => {
                assert!(
                    !calls.contains_key(index),
                    "index {index} started twice: {events:?}"
                );
                calls.insert(
                    *index,
                    Call {
                        name: name.clone(),
                        id: id.clone(),
                        ..Call::default()
                    },
                );
            }
            ToolCallStreamEvent::ArgsDelta { index, delta } => {
                let call = calls
                    .get_mut(index)
                    .unwrap_or_else(|| panic!("delta before start for {index}: {events:?}"));
                assert!(!delta.is_empty(), "empty delta");
                call.arguments.push_str(delta);
                call.deltas += 1;
            }
        }
    }
    calls
}

fn feed_pieces(
    pieces: &[&str],
    tools: &[Tool],
    choice: Option<&str>,
    primed: bool,
) -> (IncrementalToolCallStreamer, Vec<ToolCallStreamEvent>) {
    let mut streamer = IncrementalToolCallStreamer::new(tools, choice, primed);
    let mut events = Vec::new();
    for piece in pieces {
        events.extend(streamer.feed(piece));
    }
    (streamer, events)
}

fn per_char(text: &str) -> Vec<&str> {
    text.char_indices()
        .map(|(i, c)| &text[i..i + c.len_utf8()])
        .collect()
}

fn chunkings(text: &str) -> Vec<Vec<&str>> {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let mut out = vec![vec![text], per_char(text)];
    for width in [2usize, 3, 7, 13] {
        let mut pieces = Vec::new();
        let mut i = 0;
        while i + 1 < bounds.len() {
            let j = (i + width).min(bounds.len() - 1);
            pieces.push(&text[bounds[i]..bounds[j]]);
            i = j;
        }
        out.push(pieces);
    }
    out
}

/// Feed `text` per character and require every streamed call to match the
/// final parse exactly: same index, same name, and the concatenated deltas
/// equal to the final `arguments`.
fn assert_streams_like_final(
    text: &str,
    choice: Option<&str>,
    primed: bool,
) -> BTreeMap<usize, Call> {
    let tools = tools();
    let parsed = parse_tool_calls(text, Some(&tools));
    let mut first: Option<BTreeMap<usize, Call>> = None;
    for pieces in chunkings(text) {
        let (_, events) = feed_pieces(&pieces, &tools, choice, primed);
        let calls = collect(&events);
        for (index, call) in &calls {
            let final_call = parsed
                .tool_calls
                .get(*index)
                .unwrap_or_else(|| panic!("streamed index {index} not in {parsed:?}"));
            assert_eq!(call.name, final_call.name, "name of call {index}");
            assert_eq!(call.arguments, final_call.arguments, "arguments of {index}");
        }
        let shape: BTreeMap<usize, (String, String)> = calls
            .iter()
            .map(|(i, c)| (*i, (c.name.clone(), c.arguments.clone())))
            .collect();
        if let Some(first) = &first {
            let first_shape: BTreeMap<usize, (String, String)> = first
                .iter()
                .map(|(i, c)| (*i, (c.name.clone(), c.arguments.clone())))
                .collect();
            assert_eq!(shape, first_shape, "chunking changed the result");
        } else {
            first = Some(calls);
        }
    }
    let (_, events) = feed_pieces(&per_char(text), &tools, choice, primed);
    collect(&events)
}

const HERMES: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";

#[test]
fn hermes_call_streams_its_raw_arguments_per_character() {
    let calls = assert_streams_like_final(HERMES, None, false);
    assert_eq!(calls.len(), 1);
    let call = &calls[&0];
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, r#"{"city": "Paris", "days": 3}"#);
    assert!(call.deltas > 1, "{call:?}");
    assert!(call.id.starts_with("call_"));
}

#[test]
fn two_call_mistral_bracket_reply_streams_both_calls() {
    let text = r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}get_time[ARGS] {"tz": "UTC"}"#;
    let calls = assert_streams_like_final(text, None, false);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[&0].name, "get_weather");
    assert_eq!(calls[&0].arguments, r#"{"city": "Paris"}"#);
    assert_eq!(calls[&1].name, "get_time");
    assert_eq!(calls[&1].arguments, r#"{"tz": "UTC"}"#);
    assert!(calls.values().all(|c| c.deltas > 1));
}

const QWEN: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco \"Bay\"\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

#[test]
fn qwen3_coder_string_and_integer_parameters_stream_as_compact_json() {
    let calls = assert_streams_like_final(QWEN, None, false);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[&0].arguments,
        r#"{"city":"San Francisco \"Bay\"","days":3}"#
    );

    // The string parameter streams before its `</parameter>` arrives.
    let tools = tools();
    let mut streamer = IncrementalToolCallStreamer::new(&tools, None, false);
    let close = QWEN.find("</parameter>").expect("close tag");
    let mut deltas_before_close = 0usize;
    let mut fed = 0usize;
    for piece in per_char(QWEN) {
        let events = streamer.feed(piece);
        fed += piece.len();
        if fed <= close {
            deltas_before_close += events
                .iter()
                .filter(|e| matches!(e, ToolCallStreamEvent::ArgsDelta { .. }))
                .count();
        }
    }
    // `{` and `"city":"` open the object; the value itself adds more.
    assert!(
        deltas_before_close >= 4,
        "only {deltas_before_close} deltas before </parameter>"
    );
}

#[test]
fn named_tool_choice_never_streams_the_other_call() {
    let text = concat!(
        "<tool_call>{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}</tool_call>\n",
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>"
    );
    let calls = assert_streams_like_final(text, Some("get_weather"), false);
    assert_eq!(calls.keys().copied().collect::<Vec<_>>(), vec![1]);
    assert_eq!(calls[&1].name, "get_weather");
    assert_eq!(calls[&1].arguments, r#"{"city": "Oslo"}"#);
}

#[test]
fn calls_inside_thinking_are_not_streamed() {
    let call = r#"<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>"#;
    let text = format!("<think>maybe {call}</think>\n{HERMES}");
    let calls = assert_streams_like_final(&text, None, false);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[&0].name, "get_weather");

    // A primed prompt: the output opens inside the thinking block.
    let primed = format!("maybe {call}</think>\n{HERMES}");
    let calls = assert_streams_like_final(&primed, None, true);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[&0].name, "get_weather");
}

#[test]
fn unknown_tools_take_no_index() {
    let text = concat!(
        "<tool_call>{\"name\": \"launch\", \"arguments\": {}}</tool_call>",
        "<tool_call>{\"name\": \"functions.get_time\", \"arguments\": {\"tz\": \"UTC\"}}</tool_call>"
    );
    let calls = assert_streams_like_final(text, None, false);
    assert_eq!(calls.len(), 1);
    // The namespace is stripped as the final tool filter strips it.
    assert_eq!(calls[&0].name, "get_time");
}

#[test]
fn mistral_list_uses_an_id_that_arrives_before_the_name() {
    let text = concat!(
        "[TOOL_CALLS] [{\"id\": \"abc123XYZ\", \"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, ",
        "{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}, \"id\": \"late00001\"}]"
    );
    let calls = assert_streams_like_final(text, None, false);
    assert_eq!(calls[&0].id, "abc123XYZ");
    // An id after the name arrives once the call already started: ignored.
    assert_ne!(calls[&1].id, "late00001");
    assert!(calls[&1].id.starts_with("call_"));
}

#[test]
fn llama3_python_tag_and_bare_json_stream() {
    let tagged =
        r#"Let me check.<|python_tag|>{"name": "get_weather", "parameters": {"city": "Seoul"}}"#;
    let calls = assert_streams_like_final(tagged, None, false);
    assert_eq!(calls[&0].arguments, r#"{"city": "Seoul"}"#);

    let bare = r#"{"name": "get_weather", "parameters": {"city": "Seoul"}}"#;
    let calls = assert_streams_like_final(bare, None, false);
    assert_eq!(calls[&0].arguments, r#"{"city": "Seoul"}"#);
}

#[test]
fn bare_json_without_an_arguments_member_never_starts() {
    let tools = tools();
    let answer = r#"{"name": "get_weather", "description": "a JSON answer"}"#;
    let (_, events) = feed_pieces(&per_char(answer), &tools, None, false);
    assert!(events.is_empty(), "{events:?}");
}

#[test]
fn string_encoded_arguments_are_sent_decoded_in_one_delta() {
    let text =
        r#"<tool_call>{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}</tool_call>"#;
    let calls = assert_streams_like_final(text, None, false);
    assert_eq!(calls[&0].arguments, r#"{"city": "Paris"}"#);
    assert_eq!(calls[&0].deltas, 1);
}

#[test]
fn arguments_before_the_name_are_held_until_start() {
    let text = r#"<tool_call>{"arguments": {"city": "Rome"}, "name": "get_weather"}</tool_call>"#;
    let calls = assert_streams_like_final(text, None, false);
    assert_eq!(calls[&0].arguments, r#"{"city": "Rome"}"#);
}

#[test]
fn other_formats_produce_no_events() {
    let tools = tools();
    for text in [
        "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        "<tool_call><function=get_weather>{\"city\": \"Paris\"}</function></tool_call>",
        "Plain [TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}",
    ] {
        let (_, events) = feed_pieces(&per_char(text), &tools, None, false);
        assert!(events.is_empty(), "{text}: {events:?}");
    }
}

#[test]
fn held_input_stays_bounded_on_long_values() {
    let tools = tools();
    let long = "x".repeat(50_000);
    let qwen = format!(
        "<tool_call><function=get_weather><parameter=city>{long}   \n</parameter></function></tool_call>"
    );
    let hermes = format!(
        "<tool_call>{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"{long}\"}}}}</tool_call>"
    );
    for text in [qwen, hermes] {
        let mut streamer = IncrementalToolCallStreamer::new(&tools, None, false);
        let mut sent = String::new();
        for piece in per_char(&text) {
            for event in streamer.feed(piece) {
                if let ToolCallStreamEvent::ArgsDelta { delta, .. } = event {
                    sent.push_str(&delta);
                }
            }
            assert!(
                streamer.pending.len() <= 32,
                "held {}",
                streamer.pending.len()
            );
        }
        let parsed = parse_tool_calls(&text, Some(&tools));
        assert_eq!(sent, parsed.tool_calls[0].arguments);
    }
}

#[test]
fn plan_final_reuses_ids_and_sends_the_unsent_rest() {
    let tools = tools();
    // Truncated mid-value: the final parse still yields the call.
    let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
    let (streamer, events) = feed_pieces(&per_char(text), &tools, None, false);
    let streamed = collect(&events);
    let mut parsed = parse_tool_calls(text, Some(&tools));
    let plan = streamer.plan_final(&mut parsed.tool_calls, "req-1");
    assert_eq!(plan.calls.len(), 1);
    assert_eq!(plan.calls[0].streamed, Some(0));
    assert_eq!(
        parsed.tool_calls[0].id.as_deref(),
        Some(streamed[&0].id.as_str())
    );
    let rest = plan.calls[0].remainder.clone().expect("prefix agrees");
    assert_eq!(
        format!("{}{rest}", streamed[&0].arguments),
        r#"{"city":"Par"}"#
    );
    assert!(plan.orphans.is_empty());
}

#[test]
fn plan_final_reports_a_disagreeing_stream() {
    let tools = tools();
    let (streamer, _) = feed_pieces(&per_char(HERMES), &tools, None, false);
    let mut parsed = parse_tool_calls(HERMES, Some(&tools));
    parsed.tool_calls[0].arguments = r#"{"city":"Lyon"}"#.to_string();
    let plan = streamer.plan_final(&mut parsed.tool_calls, "req-2");
    assert_eq!(plan.calls[0].remainder, None);

    // A name mismatch moves the parsed call to a fresh index.
    let mut parsed = parse_tool_calls(HERMES, Some(&tools));
    parsed.tool_calls[0].name = "get_time".to_string();
    let plan = streamer.plan_final(&mut parsed.tool_calls, "req-3");
    assert_eq!(plan.calls[0].streamed, None);
    assert_eq!(plan.calls[0].index, 1);
    assert_eq!(plan.orphans, vec![0]);

    // No parsed call at all: the streamed call is an orphan.
    let plan = streamer.plan_final(&mut [], "req-4");
    assert!(plan.calls.is_empty());
    assert_eq!(plan.orphans, vec![0]);
}

#[test]
fn raw_string_predicate_matches_coerce_xml_param() {
    use crate::server::tool_calls::formats::{coerce_xml_param, xml_param_keeps_raw_string};
    let schemas = [
        json!({"type": "string"}),
        json!({"type": ["string"]}),
        json!({"type": "STRING"}),
        json!({"type": ["string", "null"]}),
        json!({"type": "string", "enum": ["a"]}),
        json!({"anyOf": [{"type": "string"}]}),
        json!({"type": "integer"}),
        json!({}),
    ];
    for schema in &schemas {
        if xml_param_keeps_raw_string(Some(schema)) {
            for raw in ["null", "true", "12", "[1]", "{\"a\":1}", "text", "yes"] {
                assert_eq!(
                    coerce_xml_param(raw, Some(schema)),
                    serde_json::Value::String(raw.to_string()),
                    "{schema} / {raw}"
                );
            }
        }
    }
    assert!(xml_param_keeps_raw_string(Some(&schemas[0])));
    assert!(!xml_param_keeps_raw_string(Some(&schemas[3])));
    assert!(!xml_param_keeps_raw_string(None));
}
