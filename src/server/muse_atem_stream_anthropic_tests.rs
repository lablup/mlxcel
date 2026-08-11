use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::anthropic_translator::{
    anthropic_request_to_chat, anthropic_stop_reason, build_content_blocks, parsed_call_to_tool_use,
};
use super::muse_atem_stream_support as support;
use super::routes::anthropic;
use super::tool_calls::{self, ToolCallFormat};
use super::types::anthropic_request::AnthropicRequest;
use super::types::anthropic_response::{
    AnthropicMessageResponse, AnthropicResponseBlock, AnthropicUsage,
};
use super::types::anthropic_stream::{
    AnthropicBlockDelta, AnthropicMessageDeltaBody, AnthropicMessageDeltaUsage,
    AnthropicStreamEvent,
};

#[derive(Debug, Default)]
struct AnthropicCall {
    id: String,
    index: usize,
    name: String,
    input_json: String,
}

#[derive(Debug, Default)]
struct AnthropicStreamSummary {
    events: Vec<AnthropicStreamEvent>,
    text: String,
    thinking: String,
    calls: Vec<AnthropicCall>,
    stop_reason: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenBlock {
    Text,
    Thinking,
}

fn make_request(messages: Value) -> AnthropicRequest {
    serde_json::from_value(json!({
        "model": support::MODEL,
        "max_tokens": 64,
        "messages": messages,
        "tools": [
            {"name": "weather.get_current", "description": "Get weather", "input_schema": {"type": "object"}},
            {"name": "calendar.create_event", "description": "Create event", "input_schema": {"type": "object"}}
        ],
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    }))
    .expect("anthropic request")
}

fn close_open(
    events: &mut Vec<AnthropicStreamEvent>,
    index: &mut usize,
    open: &mut Option<OpenBlock>,
) {
    if open.take().is_some() {
        events.push(AnthropicStreamEvent::ContentBlockStop { index: *index });
        *index += 1;
    }
}

fn open_block(
    events: &mut Vec<AnthropicStreamEvent>,
    index: &mut usize,
    open: &mut Option<OpenBlock>,
    kind: OpenBlock,
) {
    if *open == Some(kind) {
        return;
    }
    close_open(events, index, open);
    let content_block = match kind {
        OpenBlock::Text => AnthropicResponseBlock::Text {
            text: String::new(),
        },
        OpenBlock::Thinking => AnthropicResponseBlock::Thinking {
            thinking: String::new(),
            signature: String::new(),
        },
    };
    events.push(AnthropicStreamEvent::ContentBlockStart {
        index: *index,
        content_block,
    });
    *open = Some(kind);
}

fn stream_anthropic(request: &AnthropicRequest, chunks: Vec<String>) -> AnthropicStreamSummary {
    let translated = anthropic_request_to_chat(request);
    let mut events = vec![AnthropicStreamEvent::MessageStart {
        message: AnthropicMessageResponse::new(
            "msg_stream".to_string(),
            vec![],
            support::MODEL.to_string(),
            None,
            None,
            AnthropicUsage {
                input_tokens: 32,
                output_tokens: 0,
            },
        ),
    }];
    let mut filter = tool_calls::stream_filter::StreamFilter::new();
    let mut accumulated = String::new();
    let mut index = 0usize;
    let mut open = None;

    for chunk in chunks {
        accumulated.push_str(&chunk);
        let emit = filter.feed(&chunk);
        if let Some(thinking) = emit.reasoning.filter(|s| !s.is_empty()) {
            open_block(&mut events, &mut index, &mut open, OpenBlock::Thinking);
            events.push(AnthropicStreamEvent::ContentBlockDelta {
                index,
                delta: AnthropicBlockDelta::ThinkingDelta { thinking },
            });
        }
        if let Some(text) = emit.content.filter(|s| !s.is_empty()) {
            open_block(&mut events, &mut index, &mut open, OpenBlock::Text);
            events.push(AnthropicStreamEvent::ContentBlockDelta {
                index,
                delta: AnthropicBlockDelta::TextDelta { text },
            });
        }
    }
    let trailing = filter.flush();
    if let Some(thinking) = trailing.reasoning.filter(|s| !s.is_empty()) {
        open_block(&mut events, &mut index, &mut open, OpenBlock::Thinking);
        events.push(AnthropicStreamEvent::ContentBlockDelta {
            index,
            delta: AnthropicBlockDelta::ThinkingDelta { thinking },
        });
    }
    if let Some(text) = trailing.content.filter(|s| !s.is_empty()) {
        open_block(&mut events, &mut index, &mut open, OpenBlock::Text);
        events.push(AnthropicStreamEvent::ContentBlockDelta {
            index,
            delta: AnthropicBlockDelta::TextDelta { text },
        });
    }
    close_open(&mut events, &mut index, &mut open);

    let parsed = if tool_calls::should_parse_tool_calls(&translated.chat_request) {
        Some(tool_calls::parse_tool_calls(
            &accumulated,
            translated.chat_request.tools.as_deref(),
        ))
    } else {
        None
    };
    let parsed_calls = parsed
        .as_ref()
        .filter(|parsed| parsed.has_tool_calls())
        .map(|parsed| parsed.tool_calls.clone());
    if let Some(calls) = parsed_calls.as_ref() {
        for call in calls {
            if let AnthropicResponseBlock::ToolUse { id, name, input } =
                parsed_call_to_tool_use(call)
            {
                events.push(AnthropicStreamEvent::ContentBlockStart {
                    index,
                    content_block: AnthropicResponseBlock::ToolUse {
                        id,
                        name,
                        input: json!({}),
                    },
                });
                events.push(AnthropicStreamEvent::ContentBlockDelta {
                    index,
                    delta: AnthropicBlockDelta::InputJsonDelta {
                        partial_json: serde_json::to_string(&input).expect("tool input JSON"),
                    },
                });
                events.push(AnthropicStreamEvent::ContentBlockStop { index });
                index += 1;
            }
        }
    }
    events.push(AnthropicStreamEvent::MessageDelta {
        delta: AnthropicMessageDeltaBody {
            stop_reason: Some(anthropic_stop_reason("stop", parsed_calls.is_some(), None)),
            stop_sequence: None,
        },
        usage: AnthropicMessageDeltaUsage { output_tokens: 18 },
    });
    events.push(AnthropicStreamEvent::MessageStop);
    summarize(events)
}

fn summarize(events: Vec<AnthropicStreamEvent>) -> AnthropicStreamSummary {
    let mut out = AnthropicStreamSummary {
        events,
        ..Default::default()
    };
    let mut calls: BTreeMap<usize, AnthropicCall> = BTreeMap::new();
    for event in &out.events {
        support::assert_no_atem_json(event);
        match event {
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block: AnthropicResponseBlock::ToolUse { id, name, .. },
            } => {
                calls.insert(
                    *index,
                    AnthropicCall {
                        id: id.clone(),
                        index: *index,
                        name: name.clone(),
                        input_json: String::new(),
                    },
                );
            }
            AnthropicStreamEvent::ContentBlockDelta {
                index,
                delta: AnthropicBlockDelta::InputJsonDelta { partial_json },
            } => {
                calls
                    .entry(*index)
                    .or_default()
                    .input_json
                    .push_str(partial_json);
            }
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicBlockDelta::TextDelta { text },
                ..
            } => out.text.push_str(text),
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicBlockDelta::ThinkingDelta { thinking },
                ..
            } => out.thinking.push_str(thinking),
            AnthropicStreamEvent::MessageDelta { delta, .. } => {
                out.stop_reason = delta.stop_reason.clone();
            }
            _ => {}
        }
    }
    out.calls = calls.into_values().collect();
    out
}

#[test]
fn anthropic_streaming_atem_parallel_calls_match_non_streaming() {
    let request = make_request(json!([
        {"role": "user", "content": "Check weather and schedule lunch."}
    ]));
    let streamed = stream_anthropic(&request, support::one_byte_chunks(support::RAW_ATEM));
    let translated = anthropic_request_to_chat(&request);
    let parsed =
        tool_calls::parse_tool_calls(support::RAW_ATEM, translated.chat_request.tools.as_deref());
    assert_eq!(parsed.format, Some(ToolCallFormat::Atem));
    let (visible, reasoning) = anthropic::split_visible_reasoning(support::RAW_ATEM, Some(&parsed));
    let non_streaming = build_content_blocks(
        &visible,
        reasoning.as_deref(),
        Some(&parsed.tool_calls),
        true,
    );

    assert_eq!(streamed.text, "");
    assert_eq!(streamed.thinking, "Need current facts.");
    assert_eq!(streamed.stop_reason.as_deref(), Some("tool_use"));
    assert_eq!(streamed.calls.len(), 2);
    assert_eq!(streamed.calls[0].index, 1);
    assert_eq!(streamed.calls[1].index, 2);
    for (actual, expected) in streamed.calls.iter().zip(parsed.tool_calls.iter()) {
        assert!(actual.id.starts_with("toolu_"));
        assert_eq!(actual.name, expected.name);
        assert_eq!(
            serde_json::from_str::<Value>(&actual.input_json).expect("stream input"),
            support::parsed_args(expected)
        );
    }
    assert_eq!(
        non_streaming
            .iter()
            .filter(|block| matches!(block, AnthropicResponseBlock::ToolUse { .. }))
            .count(),
        2
    );
}

#[test]
fn anthropic_streaming_replay_final_answer_and_malformed_eof() {
    let replay = make_request(json!([
        {"role": "user", "content": "Check weather."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_weather", "name": "weather.get_current", "input": {"city": "Seoul"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_weather", "content": "{\"temp_c\":29}"}
        ]},
        {"role": "user", "content": "Now answer."}
    ]));
    let translated = anthropic_request_to_chat(&replay);
    let prompt = support::render_chat_request(&translated.chat_request);
    assert!(prompt.contains("<atem:invoke name=\"weather.get_current\">"));
    assert!(prompt.contains("<tool_output name=\"weather.get_current\">"));

    let final_stream = stream_anthropic(&replay, support::one_byte_chunks("It is 29 C."));
    assert_eq!(final_stream.text, "It is 29 C.");
    assert!(final_stream.calls.is_empty());
    assert_eq!(final_stream.stop_reason.as_deref(), Some("end_turn"));

    let malformed = "<atem:function_calls><atem:invoke bad=\"weather.get_current\">";
    let malformed_stream = stream_anthropic(&replay, support::one_byte_chunks(malformed));
    assert!(malformed_stream.text.is_empty());
    assert!(malformed_stream.thinking.is_empty());
    assert!(malformed_stream.calls.is_empty());
    assert_eq!(malformed_stream.stop_reason.as_deref(), Some("end_turn"));
}
