use serde::Serialize;
use serde_json::{Value, json};

use super::chat_request::build_raw_json_messages;
use super::chat_template::ChatTemplateProcessor;
use super::tool_calls::ParsedToolCall;
use super::types::request::{
    ChatCompletionRequest, FunctionDefinition, Tool, ToolCallFunction, ToolCallInMessage,
};

pub const TEMPLATE: &str = include_str!("../../tests/fixtures/muse_glimmer/chat_template.jinja");
pub const MODEL: &str = "muse-glimmer-30b";
pub const RAW_ATEM: &str = concat!(
    "<think>Need current facts.</think>",
    "<atem:function_calls>\n",
    "<atem:invoke name=\"weather.get_current\">\n",
    "<atem:parameter name=\"city\">Seoul</atem:parameter>\n",
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

pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "weather.get_current".to_string(),
                description: Some("Get current weather.".to_string()),
                parameters: Some(json!({"type": "object"})),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "calendar.create_event".to_string(),
                description: Some("Create an event.".to_string()),
                parameters: Some(json!({"type": "object"})),
            },
        },
    ]
}

pub fn chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Check weather and schedule lunch."}],
        "tools": tools(),
        "parallel_tool_calls": true
    }))
    .expect("chat request")
}

pub fn render_chat_request(request: &ChatCompletionRequest) -> String {
    let messages = build_raw_json_messages(request);
    ChatTemplateProcessor::with_template(TEMPLATE.to_string())
        .apply_raw(&messages, request.tools.as_deref())
        .expect("render Muse Glimmer chat template")
}

pub fn one_byte_chunks(text: &str) -> Vec<String> {
    text.chars().map(|ch| ch.to_string()).collect()
}

pub fn parsed_args(call: &ParsedToolCall) -> Value {
    serde_json::from_str(&call.arguments).expect("ATEM arguments must be JSON")
}

pub fn assert_no_atem_json<T: Serialize>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize response");
    assert!(!json.contains("<atem:"), "ATEM leaked into JSON: {json}");
}

pub fn tool_call_message(id: String, call: &ParsedToolCall) -> ToolCallInMessage {
    ToolCallInMessage {
        id,
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    }
}
