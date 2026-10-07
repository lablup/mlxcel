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

//! The in-process chat turn's request and bookkeeping helpers (issue #2173).
//! None of these need a checkpoint: the request shapes, the option build of
//! the raw completion path, the cancel and tool-call gates, and how streamed
//! text reaches the caller.

use serde_json::json;

use super::*;
use crate::cli::in_process_client::{
    CliServerSettings, chat_request_body, completion_request_body,
};

fn settings_with_stop_and_p_less() -> CliServerSettings {
    let mut settings = CliServerSettings {
        p_less: true,
        ..CliServerSettings::default()
    };
    settings.extra.stop = vec!["END".to_string()];
    settings
}

#[test]
fn the_cli_chat_body_parses_as_a_streaming_chat_request() {
    let body = chat_request_body(
        json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "hello" },
            { "role": "user", "content": "again" },
        ]),
        &settings_with_stop_and_p_less(),
    );
    let request = chat_request_from_json(body).expect("parses");
    assert!(request.stream);
    assert_eq!(request.messages.len(), 3);
    assert_eq!(request.params.p_less, Some(true));
    assert_eq!(request.params.stop, Some(vec!["END".to_string()]));
    assert_eq!(request.params.max_tokens, None, "`-n` is a server default");
}

#[test]
fn a_chat_body_without_messages_is_an_invalid_chat_request() {
    let err = chat_request_from_json(json!({ "model": "m" })).expect_err("no messages");
    assert!(err.to_string().starts_with("invalid chat request"), "{err}");
}

#[test]
fn the_cli_completion_body_carries_the_prompt_verbatim() {
    let body = completion_request_body("raw text\n", &settings_with_stop_and_p_less());
    let request = completion_request_from_json(body).expect("parses");
    assert_eq!(request.prompt, "raw text\n");
    assert!(request.stream);
    assert_eq!(request.params.p_less, Some(true));
    assert_eq!(request.params.stop, Some(vec!["END".to_string()]));
}

#[test]
fn a_completion_body_without_a_prompt_is_an_invalid_completion_request() {
    let err = completion_request_from_json(json!({ "model": "m" })).expect_err("no prompt");
    assert!(
        err.to_string().starts_with("invalid completion request"),
        "{err}"
    );
}

#[test]
fn completion_options_follow_the_completions_route() {
    let config = ServerConfig::default();
    let live = config.live_settings();
    let request = completion_request_from_json(json!({
        "model": "m",
        "prompt": "p",
        "max_tokens": 7,
        "temperature": 0.5,
        "stop": ["\n\n"],
    }))
    .expect("parses");
    let options = completion_options(&request, &config, &live, None);
    assert_eq!(options.max_tokens, 7);
    assert_eq!(options.sampling.temperature, 0.5);
    assert_eq!(options.stop_sequences, Some(vec!["\n\n".to_string()]));
    assert!(
        !options.thinking_enter_block_on_start,
        "a raw prompt is not primed with an open thinking block"
    );
    assert!(options.prompt_cache_ctx.is_none());
}

#[test]
fn completion_options_keep_server_defaults_for_absent_fields() {
    let config = ServerConfig::default();
    let live = config.live_settings();
    let request =
        completion_request_from_json(json!({ "model": "m", "prompt": "p" })).expect("parses");
    let options = completion_options(&request, &config, &live, None);
    assert_eq!(options.sampling.temperature, config.default_temperature);
    assert_eq!(options.sampling.top_k, config.default_top_k);
}

#[test]
fn a_cancelled_or_tool_calling_turn_leaves_no_next_turn_state() {
    assert!(records_turn_end(false, "stop"));
    assert!(records_turn_end(false, "length"));
    assert!(!records_turn_end(true, "stop"), "cancelled");
    assert!(!records_turn_end(false, "tool_calls"));
    assert!(!records_turn_end(true, "tool_calls"));
}

#[test]
fn reasoning_is_echoed_only_when_the_format_emits_reasoning_content() {
    assert_eq!(echoed_reasoning(ReasoningFormat::Auto, "why"), "why");
    assert_eq!(echoed_reasoning(ReasoningFormat::DeepSeek, "why"), "why");
    assert_eq!(
        echoed_reasoning(ReasoningFormat::DeepSeekLegacy, "why"),
        "why"
    );
    assert_eq!(echoed_reasoning(ReasoningFormat::None, "why"), "");
}

fn parsed(name: &str) -> tool_calls::ParsedToolCall {
    tool_calls::ParsedToolCall {
        name: name.to_string(),
        arguments: "{}".to_string(),
    }
}

#[test]
fn tool_calls_are_all_reported_without_a_specific_choice() {
    let calls = select_tool_calls(vec![parsed("a"), parsed("b")], None);
    let names: Vec<_> = calls.iter().map(|call| call.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(calls[0].arguments, "{}");
}

#[test]
fn a_specific_tool_choice_keeps_only_that_function() {
    let calls = select_tool_calls(vec![parsed("a"), parsed("b")], Some("b"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "b");
    assert!(select_tool_calls(vec![parsed("a")], Some("z")).is_empty());
}

#[test]
fn delivered_text_reaches_the_caller_reasoning_first_and_skips_empty_parts() {
    let mut text = TurnText {
        raw: String::new(),
        content: String::new(),
        reasoning: String::new(),
    };
    let mut seen = Vec::new();
    let mut record = |delta: ChatDelta<'_>| {
        seen.push(match delta {
            ChatDelta::Content(t) => format!("c:{t}"),
            ChatDelta::Reasoning(t) => format!("r:{t}"),
        });
    };
    let mut both = passthrough("answer");
    both.reasoning = Some("think".to_string());
    text.deliver(both, &mut record);
    let mut empty = passthrough("");
    empty.reasoning = Some(String::new());
    text.deliver(empty, &mut record);
    text.deliver(passthrough(" more"), &mut record);
    assert_eq!(seen, ["r:think", "c:answer", "c: more"]);
    assert_eq!(text.content, "answer more");
    assert_eq!(text.reasoning, "think");
}

#[test]
fn passthrough_consumes_one_position_as_content() {
    let out = passthrough("tok");
    assert_eq!(out.content.as_deref(), Some("tok"));
    assert_eq!(out.reasoning, None);
    assert_eq!(out.consumed_positions, 1);
    assert_eq!(out.suppressed_positions, 0);
}
