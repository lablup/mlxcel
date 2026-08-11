use serde_json::{Value, json};

use super::anthropic_translator::{
    anthropic_request_to_chat, anthropic_stop_reason, build_content_blocks,
};
use super::chat_request::build_raw_json_messages;
use super::chat_template::ChatTemplateProcessor;
use super::responses_translator::{
    OutboundContext, build_response_object, responses_request_to_chat,
};
use super::routes::{anthropic, chat, responses};
use super::tool_calls::{self, ParsedToolCall, ToolCallFormat};
use super::types::anthropic_request::AnthropicRequest;
use super::types::anthropic_response::{
    AnthropicMessageResponse, AnthropicResponseBlock, AnthropicUsage,
};
use super::types::request::{
    ChatCompletionRequest, FunctionDefinition, Message, MessageContent, Role, Tool,
    ToolCallFunction, ToolCallInMessage,
};
use super::types::response::ChatCompletionResponse;
use super::types::responses_request::CreateResponseRequest;
use super::types::responses_response::{ResponseOutputItem, ResponseStatus};

const TEMPLATE: &str = include_str!("../../tests/fixtures/muse_glimmer/chat_template.jinja");
const MODEL: &str = "muse-glimmer-30b";
const RAW_ATEM: &str = concat!(
    "<think>Need current facts.</think>",
    "I need to use tools.\n",
    "<atem:function_calls>\n",
    "<atem:invoke name=\"weather.get_current\">\n",
    "<atem:parameter name=\"city\">Seoul</atem:parameter>\n",
    "<atem:parameter name=\"units\">celsius</atem:parameter>\n",
    "<atem:parameter name=\"options\">",
    "{\"alerts\":true,\"days\":[1,2],\"coords\":{\"lat\":37.56,\"lon\":126.97}}",
    "</atem:parameter>\n",
    "</atem:invoke>\n",
    "<atem:invoke name=\"calendar.create_event\">\n",
    "<atem:parameter name=\"title\">Lunch</atem:parameter>\n",
    "<atem:parameter name=\"attendees\">[\"Ada\",\"Grace\"]</atem:parameter>\n",
    "</atem:invoke>\n",
    "</atem:function_calls>"
);

fn processor() -> ChatTemplateProcessor {
    ChatTemplateProcessor::with_template(TEMPLATE.to_string())
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "weather.get_current".to_string(),
                description: Some("Get current weather for a city.".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "units": {"type": "string"},
                        "options": {"type": "object"}
                    },
                    "required": ["city"]
                })),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "calendar.create_event".to_string(),
                description: Some("Create a calendar event.".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "attendees": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title"]
                })),
            },
        },
    ]
}

fn render_chat_request(request: &ChatCompletionRequest) -> String {
    let messages = build_raw_json_messages(request);
    processor()
        .apply_raw(&messages, request.tools.as_deref())
        .expect("render Muse Glimmer request")
}

fn parse_with_tools(request: &ChatCompletionRequest, raw: &str) -> tool_calls::ToolCallParseResult {
    let parsed = tool_calls::parse_tool_calls(raw, request.tools.as_deref());
    assert_eq!(parsed.format, Some(ToolCallFormat::Atem));
    assert_eq!(parsed.tool_calls.len(), 2);
    assert!(!parsed.content.contains("<atem:"));
    parsed
}

fn parsed_args(call: &ParsedToolCall) -> Value {
    serde_json::from_str(&call.arguments).expect("parsed ATEM arguments must be JSON")
}

fn assert_rendered_replay(prompt: &str) {
    assert!(prompt.contains("<atem:invoke name=\"weather.get_current\">"));
    assert!(prompt.contains("<atem:invoke name=\"calendar.create_event\">"));
    assert!(prompt.contains("<tool_output name=\"weather.get_current\">"));
    assert!(prompt.contains("<tool_output name=\"calendar.create_event\">"));
}

fn assert_no_atem(value: &Value) {
    let serialized = value.to_string();
    assert!(
        !serialized.contains("<atem:"),
        "ATEM tags leaked into response JSON: {serialized}"
    );
}

fn tool_call_message(id: String, call: &ParsedToolCall) -> ToolCallInMessage {
    ToolCallInMessage {
        id,
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    }
}

#[test]
fn chat_completions_atem_non_streaming_round_trip() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Check weather and schedule lunch."}],
        "tools": tools(),
        "parallel_tool_calls": true
    }))
    .expect("chat request");

    let initial_prompt = render_chat_request(&request);
    assert!(initial_prompt.contains("Check weather and schedule lunch."));
    assert!(initial_prompt.contains("You can invoke a function"));

    let parsed = parse_with_tools(&request, RAW_ATEM);
    assert_eq!(
        parsed_args(&parsed.tool_calls[0])["options"]["coords"]["lon"],
        126.97
    );
    assert_eq!(parsed_args(&parsed.tool_calls[1])["attendees"][1], "Grace");
    let reasoning = chat::extract_reasoning_content(RAW_ATEM, false);
    let response_calls = tool_calls::build_tool_call_responses(&parsed, &request);
    let response = ChatCompletionResponse::new_with_tool_calls(
        "chatcmpl_muse".to_string(),
        MODEL.to_string(),
        parsed.content.clone(),
        response_calls.clone(),
        32,
        18,
        None,
    )
    .with_reasoning_content(reasoning);
    let response_json = serde_json::to_value(&response).expect("chat response JSON");
    assert_eq!(response_json["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        response_json["choices"][0]["message"]["reasoning_content"],
        "Need current facts."
    );
    assert_no_atem(&response_json);

    let calls = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("chat tool calls");
    request.messages.push(Message {
        role: Role::Assistant,
        content: MessageContent::Text(parsed.content.clone()),
        name: None,
        tool_call_id: None,
        reasoning: Some("Need current facts.".to_string()),
        tool_calls: Some(
            calls
                .iter()
                .zip(parsed.tool_calls.iter())
                .map(|(wire, parsed)| tool_call_message(wire.id.clone(), parsed))
                .collect(),
        ),
    });
    for (call, output) in calls.iter().zip(["{\"temp_c\":29}", "{\"id\":\"evt_1\"}"]) {
        request.messages.push(Message {
            role: Role::Tool,
            content: MessageContent::Text(output.to_string()),
            name: None,
            tool_call_id: Some(call.id.clone()),
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
    assert_rendered_replay(&render_chat_request(&request));

    let final_parsed = tool_calls::parse_tool_calls("It is 29 C and lunch is scheduled.", None);
    let final_response = ChatCompletionResponse::new(
        "chatcmpl_final".to_string(),
        MODEL.to_string(),
        final_parsed.content,
        40,
        9,
        Some("stop".to_string()),
    );
    assert!(final_response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        final_response.choices[0].finish_reason.as_deref(),
        Some("stop")
    );
}

#[test]
fn responses_api_atem_non_streaming_round_trip() {
    let request: CreateResponseRequest = serde_json::from_value(json!({
        "model": MODEL,
        "input": "Check weather and schedule lunch.",
        "tools": [
            {"type": "function", "name": "weather.get_current", "description": "Get weather", "parameters": {"type": "object"}},
            {"type": "function", "name": "calendar.create_event", "description": "Create event", "parameters": {"type": "object"}}
        ],
        "parallel_tool_calls": true
    }))
    .expect("responses request");
    let translated = responses_request_to_chat(&request, None, None).expect("responses translate");
    assert!(render_chat_request(&translated.chat_request).contains("Check weather"));

    let parsed = parse_with_tools(&translated.chat_request, RAW_ATEM);
    let (visible, reasoning) = responses::split_reasoning(RAW_ATEM, Some(&parsed));
    let response = build_response_object(OutboundContext {
        response_id: "resp_muse".to_string(),
        model_id: MODEL.to_string(),
        created_at: 1.0,
        completed_at: 2.0,
        status: ResponseStatus::Completed,
        prompt_tokens: 28,
        completion_tokens: 18,
        cached_tokens: 0,
        reasoning_tokens: 3,
        text: visible,
        reasoning_text: reasoning,
        parsed_tool_calls: Some(&parsed),
        max_tool_calls: request.max_tool_calls,
        request: &request,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    });
    let response_json = serde_json::to_value(&response).expect("responses JSON");
    assert_eq!(response.status, ResponseStatus::Completed);
    assert_no_atem(&response_json);
    assert_eq!(
        response
            .output
            .iter()
            .filter(|item| matches!(item, ResponseOutputItem::FunctionCall(_)))
            .count(),
        2
    );

    let replay: CreateResponseRequest = serde_json::from_value(json!({
        "model": MODEL,
        "input": [
            {"type": "message", "role": "user", "content": "Check weather and schedule lunch."},
            {"type": "function_call", "call_id": "call_weather", "name": parsed.tool_calls[0].name, "arguments": parsed.tool_calls[0].arguments},
            {"type": "function_call", "call_id": "call_calendar", "name": parsed.tool_calls[1].name, "arguments": parsed.tool_calls[1].arguments},
            {"type": "function_call_output", "call_id": "call_weather", "output": "{\"temp_c\":29}"},
            {"type": "function_call_output", "call_id": "call_calendar", "output": "{\"id\":\"evt_1\"}"},
            {"type": "message", "role": "user", "content": "Now answer."}
        ],
        "tools": [
            {"type": "function", "name": "weather.get_current", "description": "Get weather", "parameters": {"type": "object"}},
            {"type": "function", "name": "calendar.create_event", "description": "Create event", "parameters": {"type": "object"}}
        ]
    }))
    .expect("responses replay request");
    let replay_chat = responses_request_to_chat(&replay, None, None).expect("responses replay");
    assert_rendered_replay(&render_chat_request(&replay_chat.chat_request));

    let final_parsed = tool_calls::parse_tool_calls(
        "It is 29 C and lunch is scheduled.",
        replay_chat.chat_request.tools.as_deref(),
    );
    assert!(!final_parsed.has_tool_calls());
    let final_response = build_response_object(OutboundContext {
        response_id: "resp_final".to_string(),
        model_id: MODEL.to_string(),
        created_at: 3.0,
        completed_at: 4.0,
        status: ResponseStatus::Completed,
        prompt_tokens: 45,
        completion_tokens: 9,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: final_parsed.content.clone(),
        reasoning_text: None,
        parsed_tool_calls: Some(&final_parsed),
        max_tool_calls: replay.max_tool_calls,
        request: &replay,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    });
    assert_eq!(
        final_response.output_text,
        "It is 29 C and lunch is scheduled."
    );
}

#[test]
fn anthropic_api_atem_non_streaming_round_trip() {
    let request: AnthropicRequest = serde_json::from_value(json!({
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Check weather and schedule lunch."}],
        "tools": [
            {"name": "weather.get_current", "description": "Get weather", "input_schema": {"type": "object"}},
            {"name": "calendar.create_event", "description": "Create event", "input_schema": {"type": "object"}}
        ]
    }))
    .expect("anthropic request");
    let translated = anthropic_request_to_chat(&request);
    assert!(render_chat_request(&translated.chat_request).contains("Check weather"));

    let parsed = parse_with_tools(&translated.chat_request, RAW_ATEM);
    let (visible, reasoning) = anthropic::split_visible_reasoning(RAW_ATEM, Some(&parsed));
    let blocks = build_content_blocks(
        &visible,
        reasoning.as_deref(),
        Some(&parsed.tool_calls),
        true,
    );
    let response = AnthropicMessageResponse::new(
        "msg_muse".to_string(),
        blocks,
        MODEL.to_string(),
        Some(anthropic_stop_reason("stop", parsed.has_tool_calls(), None)),
        None,
        AnthropicUsage {
            input_tokens: 28,
            output_tokens: 18,
        },
    );
    let response_json = serde_json::to_value(&response).expect("anthropic JSON");
    assert_eq!(response.stop_reason.as_deref(), Some("tool_use"));
    assert_no_atem(&response_json);
    assert_eq!(
        response
            .content
            .iter()
            .filter(|item| matches!(item, AnthropicResponseBlock::ToolUse { .. }))
            .count(),
        2
    );

    let replay: AnthropicRequest = serde_json::from_value(json!({
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Check weather and schedule lunch."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_weather", "name": parsed.tool_calls[0].name, "input": parsed_args(&parsed.tool_calls[0])},
                {"type": "tool_use", "id": "toolu_calendar", "name": parsed.tool_calls[1].name, "input": parsed_args(&parsed.tool_calls[1])}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_weather", "content": "{\"temp_c\":29}"},
                {"type": "tool_result", "tool_use_id": "toolu_calendar", "content": "{\"id\":\"evt_1\"}"}
            ]},
            {"role": "user", "content": "Now answer."}
        ],
        "tools": [
            {"name": "weather.get_current", "description": "Get weather", "input_schema": {"type": "object"}},
            {"name": "calendar.create_event", "description": "Create event", "input_schema": {"type": "object"}}
        ]
    }))
    .expect("anthropic replay request");
    let replay_chat = anthropic_request_to_chat(&replay);
    assert_rendered_replay(&render_chat_request(&replay_chat.chat_request));

    let final_blocks =
        build_content_blocks("It is 29 C and lunch is scheduled.", None, None, false);
    let final_response = AnthropicMessageResponse::new(
        "msg_final".to_string(),
        final_blocks,
        MODEL.to_string(),
        Some(anthropic_stop_reason("stop", false, None)),
        None,
        AnthropicUsage {
            input_tokens: 45,
            output_tokens: 9,
        },
    );
    assert_eq!(final_response.stop_reason.as_deref(), Some("end_turn"));
}

#[test]
fn atem_unknown_tool_is_dropped_before_route_responses_are_built() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the allowed tool."}],
        "tools": [tools()[0]]
    }))
    .expect("chat request");
    let raw = concat!(
        "visible ",
        "<atem:function_calls><atem:invoke name=\"calendar.create_event\">",
        "<atem:parameter name=\"title\">Lunch</atem:parameter>",
        "</atem:invoke></atem:function_calls>"
    );
    let parsed = tool_calls::parse_tool_calls(raw, request.tools.as_deref());
    assert!(!parsed.has_tool_calls());
    assert!(!parsed.content.contains("<atem:"));

    let response = ChatCompletionResponse::new(
        "chatcmpl_unknown".to_string(),
        MODEL.to_string(),
        parsed.content,
        12,
        4,
        Some("stop".to_string()),
    );
    let response_json = serde_json::to_value(response).expect("chat response JSON");
    assert_eq!(response_json["choices"][0]["finish_reason"], "stop");
    assert_no_atem(&response_json);
}
