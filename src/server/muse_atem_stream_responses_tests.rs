use std::collections::BTreeMap;

use serde_json::json;

use super::muse_atem_stream_support as support;
use super::responses_translator::{
    OutboundContext, build_response_object, responses_request_to_chat,
};
use super::routes::responses;
use super::streaming_responses::ResponseStreamEmitter;
use super::tool_calls::{self, ToolCallFormat};
use super::types::responses_request::CreateResponseRequest;
use super::types::responses_response::{
    ResponseFunctionCallOutput, ResponseItemStatus, ResponseObject, ResponseOutputContent,
    ResponseOutputItem, ResponseOutputMessage, ResponseReasoningOutput, ResponseReasoningPart,
    ResponseStatus,
};
use super::types::responses_stream::ResponseStreamEvent;

#[derive(Debug, Default)]
struct RespCall {
    output_index: usize,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct RespStreamSummary {
    events: Vec<ResponseStreamEvent>,
    text: String,
    reasoning: String,
    calls: Vec<RespCall>,
    completed: Option<ResponseObject>,
}

fn make_request(input: serde_json::Value) -> CreateResponseRequest {
    serde_json::from_value(json!({
        "model": support::MODEL,
        "input": input,
        "tools": [
            {"type": "function", "name": "weather.get_current", "description": "Get weather", "parameters": {"type": "object"}},
            {"type": "function", "name": "calendar.create_event", "description": "Create event", "parameters": {"type": "object"}}
        ],
        "parallel_tool_calls": true
    }))
    .expect("responses request")
}

fn emit_reasoning(
    em: &mut ResponseStreamEmitter,
    events: &mut Vec<ResponseStreamEvent>,
    text: String,
) {
    let r_id = if let Some(id) = em.active_reasoning_id.clone() {
        id
    } else {
        let new_id = "rs_stream".to_string();
        em.open_reasoning(new_id.clone());
        events.push(ResponseStreamEvent::OutputItemAdded {
            sequence_number: em.next_seq(),
            output_index: em.output_index(),
            item: ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                id: new_id.clone(),
                status: ResponseItemStatus::InProgress,
                content: vec![],
            }),
        });
        new_id
    };
    em.reasoning_text_acc.push_str(&text);
    events.push(ResponseStreamEvent::ReasoningTextDelta {
        sequence_number: em.next_seq(),
        item_id: r_id,
        output_index: em.output_index(),
        content_index: 0,
        delta: text,
    });
}

fn emit_text(em: &mut ResponseStreamEmitter, events: &mut Vec<ResponseStreamEvent>, text: String) {
    if let Some((r_id, r_text)) = em.close_reasoning() {
        events.push(ResponseStreamEvent::ReasoningTextDone {
            sequence_number: em.next_seq(),
            item_id: r_id.clone(),
            output_index: em.output_index(),
            content_index: 0,
            text: r_text.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            sequence_number: em.next_seq(),
            output_index: em.output_index(),
            item: ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                id: r_id,
                status: ResponseItemStatus::Completed,
                content: vec![ResponseReasoningPart::ReasoningText { text: r_text }],
            }),
        });
        em.advance_output_index();
    }
    let msg_id = if let Some(id) = em.active_message_id.clone() {
        id
    } else {
        let new_id = "msg_stream".to_string();
        em.open_message(new_id.clone());
        events.push(ResponseStreamEvent::OutputItemAdded {
            sequence_number: em.next_seq(),
            output_index: em.output_index(),
            item: ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
                new_id.clone(),
                vec![],
            )),
        });
        events.push(ResponseStreamEvent::ContentPartAdded {
            sequence_number: em.next_seq(),
            item_id: new_id.clone(),
            output_index: em.output_index(),
            content_index: 0,
            part: ResponseOutputContent::output_text(String::new()),
        });
        new_id
    };
    em.message_text_acc.push_str(&text);
    events.push(ResponseStreamEvent::OutputTextDelta {
        sequence_number: em.next_seq(),
        item_id: msg_id,
        output_index: em.output_index(),
        content_index: 0,
        delta: text,
    });
}

fn stream_responses(request: &CreateResponseRequest, chunks: Vec<String>) -> RespStreamSummary {
    let translated = responses_request_to_chat(request, None, None).expect("translate");
    let chat_request = &translated.chat_request;
    let mut events = Vec::new();
    let mut em = ResponseStreamEmitter::new();
    let mut filter = tool_calls::stream_filter::StreamFilter::new();
    let mut accumulated = String::new();
    let initial = build_response_object(OutboundContext {
        response_id: "resp_stream".to_string(),
        model_id: support::MODEL.to_string(),
        created_at: 1.0,
        completed_at: 1.0,
        status: ResponseStatus::InProgress,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: String::new(),
        reasoning_text: None,
        parsed_tool_calls: None,
        max_tool_calls: request.max_tool_calls,
        request,
        error: None,
        incomplete_reason: None,
        finish_reason: "in_progress".to_string(),
    });
    events.push(ResponseStreamEvent::Created {
        sequence_number: em.next_seq(),
        response: initial.clone(),
    });
    events.push(ResponseStreamEvent::InProgress {
        sequence_number: em.next_seq(),
        response: initial,
    });

    for chunk in chunks {
        accumulated.push_str(&chunk);
        let emit = filter.feed(&chunk);
        if let Some(text) = emit.reasoning.filter(|s| !s.is_empty()) {
            emit_reasoning(&mut em, &mut events, text);
        }
        if let Some(text) = emit.content.filter(|s| !s.is_empty()) {
            emit_text(&mut em, &mut events, text);
        }
    }
    let trailing = filter.flush();
    if let Some(text) = trailing.reasoning.filter(|s| !s.is_empty()) {
        emit_reasoning(&mut em, &mut events, text);
    }
    if let Some(text) = trailing.content.filter(|s| !s.is_empty()) {
        emit_text(&mut em, &mut events, text);
    }

    let parsed_tools = if tool_calls::should_parse_tool_calls(chat_request) {
        Some(tool_calls::parse_tool_calls(
            &accumulated,
            chat_request.tools.as_deref(),
        ))
    } else {
        None
    };
    if let Some((r_id, r_text)) = em.close_reasoning() {
        events.push(ResponseStreamEvent::ReasoningTextDone {
            sequence_number: em.next_seq(),
            item_id: r_id.clone(),
            output_index: em.output_index(),
            content_index: 0,
            text: r_text.clone(),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            sequence_number: em.next_seq(),
            output_index: em.output_index(),
            item: ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                id: r_id,
                status: ResponseItemStatus::Completed,
                content: vec![ResponseReasoningPart::ReasoningText { text: r_text }],
            }),
        });
        em.advance_output_index();
    }
    let (_, message_text) = em.close_message().unwrap_or_default();
    if !message_text.is_empty() {
        events.push(ResponseStreamEvent::OutputTextDone {
            sequence_number: em.next_seq(),
            item_id: "msg_stream".to_string(),
            output_index: em.output_index(),
            content_index: 0,
            text: message_text.clone(),
        });
        events.push(ResponseStreamEvent::ContentPartDone {
            sequence_number: em.next_seq(),
            item_id: "msg_stream".to_string(),
            output_index: em.output_index(),
            content_index: 0,
            part: ResponseOutputContent::output_text(message_text.clone()),
        });
        events.push(ResponseStreamEvent::OutputItemDone {
            sequence_number: em.next_seq(),
            output_index: em.output_index(),
            item: ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
                "msg_stream".to_string(),
                vec![ResponseOutputContent::output_text(message_text.clone())],
            )),
        });
        em.advance_output_index();
    }
    if let Some(parsed) = parsed_tools.as_ref() {
        for (idx, call) in parsed.tool_calls.iter().enumerate() {
            let fc = ResponseFunctionCallOutput {
                id: format!("fc_stream_{idx}"),
                call_id: format!("call_stream_{idx}"),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                status: ResponseItemStatus::Completed,
            };
            let item = ResponseOutputItem::FunctionCall(fc);
            events.push(ResponseStreamEvent::OutputItemAdded {
                sequence_number: em.next_seq(),
                output_index: em.output_index(),
                item: item.clone(),
            });
            events.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                sequence_number: em.next_seq(),
                item_id: format!("fc_stream_{idx}"),
                output_index: em.output_index(),
                delta: call.arguments.clone(),
            });
            events.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                sequence_number: em.next_seq(),
                item_id: format!("fc_stream_{idx}"),
                output_index: em.output_index(),
                arguments: call.arguments.clone(),
            });
            events.push(ResponseStreamEvent::OutputItemDone {
                sequence_number: em.next_seq(),
                output_index: em.output_index(),
                item,
            });
            em.advance_output_index();
        }
    }
    let completed = build_response_object(OutboundContext {
        response_id: "resp_stream".to_string(),
        model_id: support::MODEL.to_string(),
        created_at: 1.0,
        completed_at: 2.0,
        status: ResponseStatus::Completed,
        prompt_tokens: 32,
        completion_tokens: 18,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: message_text,
        reasoning_text: em.completed_reasoning_text(),
        parsed_tool_calls: parsed_tools.as_ref(),
        max_tool_calls: request.max_tool_calls,
        request,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    });
    events.push(ResponseStreamEvent::Completed {
        sequence_number: em.next_seq(),
        response: completed,
    });
    summarize(events)
}

fn summarize(events: Vec<ResponseStreamEvent>) -> RespStreamSummary {
    let mut out = RespStreamSummary {
        events,
        ..Default::default()
    };
    let mut calls: BTreeMap<String, RespCall> = BTreeMap::new();
    for (expected_seq, event) in out.events.iter().enumerate() {
        support::assert_no_atem_json(event);
        assert_eq!(event.sequence_number(), expected_seq as u64);
        match event {
            ResponseStreamEvent::ReasoningTextDelta { delta, .. } => out.reasoning.push_str(delta),
            ResponseStreamEvent::OutputTextDelta { delta, .. } => out.text.push_str(delta),
            ResponseStreamEvent::OutputItemAdded {
                output_index,
                item: ResponseOutputItem::FunctionCall(fc),
                ..
            } => {
                calls.insert(
                    fc.id.clone(),
                    RespCall {
                        output_index: *output_index,
                        name: fc.name.clone(),
                        arguments: String::new(),
                    },
                );
            }
            ResponseStreamEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
                calls
                    .entry(item_id.clone())
                    .or_default()
                    .arguments
                    .push_str(delta);
            }
            ResponseStreamEvent::Completed { response, .. } => {
                out.completed = Some(response.clone());
            }
            _ => {}
        }
    }
    out.calls = calls.into_values().collect();
    out
}

#[test]
fn responses_streaming_atem_parallel_calls_match_non_streaming() {
    let request = make_request(json!("Check weather and schedule lunch."));
    let streamed = stream_responses(&request, support::one_byte_chunks(support::RAW_ATEM));
    let translated = responses_request_to_chat(&request, None, None).expect("translate");
    let parsed =
        tool_calls::parse_tool_calls(support::RAW_ATEM, translated.chat_request.tools.as_deref());
    assert_eq!(parsed.format, Some(ToolCallFormat::Atem));
    let (visible, reasoning) = responses::split_reasoning(support::RAW_ATEM, Some(&parsed));
    let non_streaming = build_response_object(OutboundContext {
        response_id: "resp_nonstream".to_string(),
        model_id: support::MODEL.to_string(),
        created_at: 1.0,
        completed_at: 2.0,
        status: ResponseStatus::Completed,
        prompt_tokens: 32,
        completion_tokens: 18,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: visible,
        reasoning_text: reasoning,
        parsed_tool_calls: Some(&parsed),
        max_tool_calls: request.max_tool_calls,
        request: &request,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    });

    assert_eq!(streamed.text, non_streaming.output_text);
    assert_eq!(streamed.reasoning, "Need current facts.");
    assert_eq!(streamed.calls.len(), 2);
    assert_eq!(streamed.calls[0].output_index, 1);
    assert_eq!(streamed.calls[1].output_index, 2);
    for (actual, expected) in streamed.calls.iter().zip(parsed.tool_calls.iter()) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&actual.arguments).expect("stream args"),
            support::parsed_args(expected)
        );
    }
    let completed = streamed.completed.expect("completed response");
    assert_eq!(completed.status, ResponseStatus::Completed);
    assert!(
        completed
            .output
            .iter()
            .any(|item| matches!(item, ResponseOutputItem::Reasoning(_)))
    );
    assert_eq!(
        completed
            .output
            .iter()
            .filter(|item| matches!(item, ResponseOutputItem::FunctionCall(_)))
            .count(),
        2
    );
}

#[test]
fn responses_streaming_replay_final_answer_and_malformed_eof() {
    let replay = make_request(json!([
        {"type": "message", "role": "user", "content": "Check weather."},
        {"type": "function_call", "call_id": "call_weather", "name": "weather.get_current", "arguments": "{\"city\":\"Seoul\"}"},
        {"type": "function_call_output", "call_id": "call_weather", "output": "{\"temp_c\":29}"},
        {"type": "message", "role": "user", "content": "Now answer."}
    ]));
    let translated = responses_request_to_chat(&replay, None, None).expect("translate");
    let prompt = support::render_chat_request(&translated.chat_request);
    assert!(prompt.contains("<atem:invoke name=\"weather.get_current\">"));
    assert!(prompt.contains("<tool_output name=\"weather.get_current\">"));

    let final_stream = stream_responses(&replay, support::one_byte_chunks("It is 29 C."));
    assert_eq!(final_stream.text, "It is 29 C.");
    assert!(final_stream.calls.is_empty());
    assert_eq!(
        final_stream.completed.expect("completed").output_text,
        "It is 29 C."
    );

    let malformed = "<atem:function_calls><atem:invoke bad=\"weather.get_current\">";
    let malformed_stream = stream_responses(&replay, support::one_byte_chunks(malformed));
    assert!(malformed_stream.text.is_empty());
    assert!(malformed_stream.reasoning.is_empty());
    assert!(malformed_stream.calls.is_empty());
}
