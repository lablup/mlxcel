use std::collections::BTreeMap;

use serde_json::json;

use super::muse_atem_stream_support as support;
use super::routes::chat;
use super::tool_calls::{self, ToolCallFormat};
use super::types::request::{ChatCompletionRequest, Message, MessageContent, Role};
use super::types::response::ChatCompletionResponse;
use super::types::stream::ChatCompletionChunk;

#[derive(Debug, Default)]
struct StreamedCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default)]
struct ChatStreamResult {
    chunks: Vec<ChatCompletionChunk>,
    content: String,
    reasoning: String,
    calls: Vec<StreamedCall>,
    finish_reason: Option<String>,
}

fn stream_chat(request: &ChatCompletionRequest, chunks: Vec<String>) -> ChatStreamResult {
    let request_id = "chatcmpl_stream".to_string();
    let model = support::MODEL.to_string();
    let mut out = Vec::new();
    let mut filter = tool_calls::stream_filter::StreamFilter::new();
    let mut accumulated = String::new();
    out.push(ChatCompletionChunk::initial(
        request_id.clone(),
        model.clone(),
    ));

    for chunk in chunks {
        accumulated.push_str(&chunk);
        let emit = filter.feed(&chunk);
        if let Some(reasoning) = emit.reasoning.filter(|s| !s.is_empty()) {
            out.push(ChatCompletionChunk::reasoning_content(
                request_id.clone(),
                model.clone(),
                reasoning,
            ));
        }
        if let Some(content) = emit.content.filter(|s| !s.is_empty()) {
            out.push(ChatCompletionChunk::content(
                request_id.clone(),
                model.clone(),
                content,
            ));
        }
    }
    let trailing = filter.flush();
    if let Some(reasoning) = trailing.reasoning.filter(|s| !s.is_empty()) {
        out.push(ChatCompletionChunk::reasoning_content(
            request_id.clone(),
            model.clone(),
            reasoning,
        ));
    }
    if let Some(content) = trailing.content.filter(|s| !s.is_empty()) {
        out.push(ChatCompletionChunk::content(
            request_id.clone(),
            model.clone(),
            content,
        ));
    }

    let mut finish_reason = "stop".to_string();
    if tool_calls::should_parse_tool_calls(request) {
        let parsed = tool_calls::parse_tool_calls(&accumulated, request.tools.as_deref());
        if parsed.has_tool_calls() {
            let specific = request
                .tool_choice
                .as_ref()
                .and_then(|choice| choice.specific_function());
            for (idx, call) in parsed.tool_calls.iter().enumerate() {
                if specific.is_some_and(|name| name != call.name) {
                    continue;
                }
                out.push(ChatCompletionChunk::tool_call_start(
                    request_id.clone(),
                    model.clone(),
                    idx,
                    format!("call_stream_{idx}"),
                    call.name.clone(),
                ));
                out.push(ChatCompletionChunk::tool_call_arguments(
                    request_id.clone(),
                    model.clone(),
                    idx,
                    call.arguments.clone(),
                ));
            }
            finish_reason = "tool_calls".to_string();
        }
    }
    out.push(ChatCompletionChunk::finish(
        request_id,
        model,
        finish_reason.clone(),
    ));
    reconstruct(out)
}

fn reconstruct(chunks: Vec<ChatCompletionChunk>) -> ChatStreamResult {
    let mut result = ChatStreamResult {
        chunks,
        ..Default::default()
    };
    let mut calls: BTreeMap<usize, StreamedCall> = BTreeMap::new();
    for chunk in &result.chunks {
        support::assert_no_atem_json(chunk);
        let Some(choice) = chunk.choices.first() else {
            continue;
        };
        if let Some(text) = choice.delta.content.as_ref() {
            result.content.push_str(text);
        }
        if let Some(text) = choice.delta.reasoning_content.as_ref() {
            result.reasoning.push_str(text);
        }
        if let Some(deltas) = choice.delta.tool_calls.as_ref() {
            for delta in deltas {
                let entry = calls.entry(delta.index).or_default();
                if let Some(id) = delta.id.as_ref() {
                    entry.id = Some(id.clone());
                }
                if let Some(function) = delta.function.as_ref() {
                    if let Some(name) = function.name.as_ref() {
                        entry.name = Some(name.clone());
                    }
                    if let Some(arguments) = function.arguments.as_ref() {
                        entry.arguments.push_str(arguments);
                    }
                }
            }
        }
        if choice.finish_reason.is_some() {
            result.finish_reason = choice.finish_reason.clone();
        }
    }
    result.calls = calls.into_values().collect();
    result
}

#[test]
fn chat_streaming_atem_parallel_calls_match_non_streaming_and_replay() {
    let mut request = support::chat_request();
    let streamed = stream_chat(&request, support::one_byte_chunks(support::RAW_ATEM));
    let parsed = tool_calls::parse_tool_calls(support::RAW_ATEM, request.tools.as_deref());
    assert_eq!(parsed.format, Some(ToolCallFormat::Atem));
    let non_streaming = ChatCompletionResponse::new_with_tool_calls(
        "chatcmpl_nonstream".to_string(),
        support::MODEL.to_string(),
        parsed.content.clone(),
        tool_calls::build_tool_call_responses(&parsed, &request),
        32,
        18,
        None,
    )
    .with_reasoning_content(chat::extract_reasoning_content(support::RAW_ATEM, false));

    assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(streamed.content, parsed.content);
    assert_eq!(
        Some(streamed.reasoning.as_str()),
        non_streaming.choices[0]
            .message
            .reasoning_content
            .as_deref()
    );
    assert_eq!(streamed.calls.len(), 2);
    for (idx, (actual, expected)) in streamed
        .calls
        .iter()
        .zip(parsed.tool_calls.iter())
        .enumerate()
    {
        assert_eq!(
            actual.id.as_deref(),
            Some(format!("call_stream_{idx}").as_str())
        );
        assert_eq!(actual.name.as_deref(), Some(expected.name.as_str()));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&actual.arguments).expect("stream args"),
            support::parsed_args(expected)
        );
    }

    let calls = streamed
        .calls
        .iter()
        .zip(parsed.tool_calls.iter())
        .map(|(wire, parsed)| support::tool_call_message(wire.id.clone().expect("call id"), parsed))
        .collect();
    request.messages.push(Message {
        role: Role::Assistant,
        content: MessageContent::Text(parsed.content.clone()),
        name: None,
        tool_call_id: None,
        reasoning: Some(streamed.reasoning.clone()),
        tool_calls: Some(calls),
    });
    for (call, output) in streamed
        .calls
        .iter()
        .zip(["{\"temp_c\":29}", "{\"id\":\"evt_1\"}"])
    {
        request.messages.push(Message {
            role: Role::Tool,
            content: MessageContent::Text(output.to_string()),
            name: None,
            tool_call_id: call.id.clone(),
            reasoning: None,
            tool_calls: None,
        });
    }
    request.messages.push(Message {
        role: Role::User,
        content: MessageContent::Text("Now answer.".to_string()),
        name: None,
        tool_call_id: None,
        reasoning: None,
        tool_calls: None,
    });
    let replay_prompt = support::render_chat_request(&request);
    assert!(replay_prompt.contains("<atem:invoke name=\"weather.get_current\">"));
    assert!(replay_prompt.contains("<tool_output name=\"calendar.create_event\">"));

    let final_stream = stream_chat(
        &request,
        vec![
            "It is ".to_string(),
            "29 C and lunch is scheduled.".to_string(),
        ],
    );
    assert_eq!(final_stream.content, "It is 29 C and lunch is scheduled.");
    assert!(final_stream.calls.is_empty());
    assert_eq!(final_stream.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn chat_streaming_atem_allowlist_and_malformed_eof_do_not_leak() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": support::MODEL,
        "messages": [{"role": "user", "content": "Use allowed tools only."}],
        "tools": [support::tools()[0]]
    }))
    .expect("chat request");
    let mixed = concat!(
        "<atem:function_calls>",
        "<atem:invoke name=\"weather.get_current\"><atem:parameter name=\"city\">Seoul</atem:parameter></atem:invoke>",
        "<atem:invoke name=\"calendar.create_event\"><atem:parameter name=\"title\">Lunch</atem:parameter></atem:invoke>",
        "</atem:function_calls>"
    );
    let streamed = stream_chat(&request, support::one_byte_chunks(mixed));
    assert_eq!(streamed.calls.len(), 1);
    assert_eq!(
        streamed.calls[0].name.as_deref(),
        Some("weather.get_current")
    );
    assert_eq!(streamed.finish_reason.as_deref(), Some("tool_calls"));

    let malformed = "<atem:function_calls><atem:invoke bad=\"weather.get_current\">";
    let streamed = stream_chat(&request, support::one_byte_chunks(malformed));
    assert!(streamed.calls.is_empty());
    assert!(streamed.content.is_empty());
    assert!(streamed.reasoning.is_empty());
    assert_eq!(streamed.finish_reason.as_deref(), Some("stop"));
}
