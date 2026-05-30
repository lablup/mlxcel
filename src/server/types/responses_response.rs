// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! OpenAI Responses API response types.
//!
//! Wire shape mirrors `openai-python` `response.py`. Phase 1 emits the
//! discriminated union variants exercised by the acceptance criteria —
//! `message` (with `output_text` content), `function_call`, and
//! `reasoning`. Other variants (`web_search_call`, `file_search_call`,
//! `code_interpreter_call`, `image_generation_call`, MCP, etc.) are
//! reserved for Phase 2/3 and the corresponding tool types are rejected
//! at the request boundary.

use std::collections::HashMap;

use serde::Serialize;

use super::request::ToolChoice;
use super::responses_request::{ResponseReasoningConfig, ResponseTextConfig};

/// Top-level response object returned by `POST /v1/responses` and
/// `GET /v1/responses/:id`.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<f64>,
    pub status: ResponseStatus,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    /// Convenience aggregator: concatenation of every `output[].content[].text`
    /// where the part is `output_text`.
    pub output_text: String,
    pub usage: ResponseUsage,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseErrorBody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<ResponseIncompleteDetails>,

    // -- Echoed request fields ------------------------------------------------
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceEchoed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfigEchoed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationRefEchoed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// `status` enum.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
}

impl ResponseStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseStatus::Queued => "queued",
            ResponseStatus::InProgress => "in_progress",
            ResponseStatus::Completed => "completed",
            ResponseStatus::Failed => "failed",
            ResponseStatus::Cancelled => "cancelled",
            ResponseStatus::Incomplete => "incomplete",
        }
    }
}

/// Discriminated union of output items. Phase 1 emits `Message`,
/// `FunctionCall`, and `Reasoning`. The serde tag matches the OpenAI
/// discriminator field.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message(ResponseOutputMessage),
    FunctionCall(ResponseFunctionCallOutput),
    Reasoning(ResponseReasoningOutput),
}

impl ResponseOutputItem {
    /// Stable item id used by the streaming envelope and rehydration.
    pub fn item_id(&self) -> &str {
        match self {
            ResponseOutputItem::Message(m) => m.id.as_str(),
            ResponseOutputItem::FunctionCall(f) => f.id.as_str(),
            ResponseOutputItem::Reasoning(r) => r.id.as_str(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputMessage {
    pub id: String,
    pub role: String,
    pub status: ResponseItemStatus,
    pub content: Vec<ResponseOutputContent>,
}

impl ResponseOutputMessage {
    pub fn new_assistant(id: String, content: Vec<ResponseOutputContent>) -> Self {
        Self {
            id,
            role: "assistant".to_string(),
            status: ResponseItemStatus::Completed,
            content,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFunctionCallOutput {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: ResponseItemStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseReasoningOutput {
    pub id: String,
    pub status: ResponseItemStatus,
    pub content: Vec<ResponseReasoningPart>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseReasoningPart {
    ReasoningText { text: String },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// Content parts under a [`ResponseOutputMessage`].
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputContent {
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<serde_json::Value>,
    },
    Refusal {
        refusal: String,
    },
}

impl ResponseOutputContent {
    pub fn output_text(text: String) -> Self {
        ResponseOutputContent::OutputText {
            text,
            annotations: Vec::new(),
            logprobs: None,
        }
    }
}

/// Token-usage summary. The breakdown structs mirror the OpenAI naming.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponseInputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponseOutputTokensDetails>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseInputTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputTokensDetails {
    pub reasoning_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseIncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseErrorBody {
    pub code: String,
    pub message: String,
}

/// Echoed `tool_choice`. Round-trip via JSON so we don't need a
/// dedicated Serialize impl on the request enum.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolChoiceEchoed {
    Mode(String),
    Object(serde_json::Value),
}

impl ToolChoiceEchoed {
    pub fn from_choice(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Mode(s) => ToolChoiceEchoed::Mode(s.clone()),
            ToolChoice::Specific(_) => ToolChoiceEchoed::Object(
                serde_json::to_value(EchoSpecific {
                    choice_type: "function".to_string(),
                    function: EchoSpecificFn {
                        name: choice.specific_function().unwrap_or("").to_string(),
                    },
                })
                .unwrap_or(serde_json::Value::Null),
            ),
        }
    }
}

#[derive(Serialize)]
struct EchoSpecific {
    #[serde(rename = "type")]
    choice_type: String,
    function: EchoSpecificFn,
}

#[derive(Serialize)]
struct EchoSpecificFn {
    name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseTextConfigEchoed {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

impl ResponseTextConfigEchoed {
    pub fn from_config(cfg: &ResponseTextConfig) -> Self {
        let format = cfg.format.as_ref().map(|f| {
            serde_json::to_value(EchoTextFormat::from_format(f)).unwrap_or(serde_json::Value::Null)
        });
        Self { format }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum EchoTextFormat {
    Text {
        #[serde(rename = "type")]
        kind: String,
    },
    JsonSchema {
        #[serde(rename = "type")]
        kind: String,
        json_schema: serde_json::Value,
    },
    Unsupported {
        #[serde(rename = "type")]
        kind: String,
    },
}

impl EchoTextFormat {
    fn from_format(format: &super::responses_request::ResponseTextFormat) -> Self {
        use super::responses_request::ResponseTextFormat;
        match format {
            ResponseTextFormat::Text => EchoTextFormat::Text {
                kind: "text".to_string(),
            },
            ResponseTextFormat::JsonSchema { schema } => EchoTextFormat::JsonSchema {
                kind: "json_schema".to_string(),
                json_schema: schema
                    .get("json_schema")
                    .cloned()
                    .unwrap_or_else(|| schema.clone()),
            },
            ResponseTextFormat::Unsupported => EchoTextFormat::Unsupported {
                kind: "unsupported".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ConversationRefEchoed {
    Id(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_text_aggregator_is_serialized() {
        let resp = ResponseObject {
            id: "resp_1".to_string(),
            object: "response".to_string(),
            created_at: 0.0,
            completed_at: None,
            status: ResponseStatus::Completed,
            model: "m".to_string(),
            output: vec![],
            output_text: "hello".to_string(),
            usage: ResponseUsage {
                input_tokens: 1,
                output_tokens: 1,
                total_tokens: 2,
                input_tokens_details: None,
                output_tokens_details: None,
            },
            error: None,
            incomplete_details: None,
            instructions: None,
            tools: None,
            tool_choice: None,
            text: None,
            reasoning: None,
            metadata: None,
            temperature: None,
            top_p: None,
            parallel_tool_calls: None,
            truncation: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            previous_response_id: None,
            conversation: None,
            prompt_cache_key: None,
            service_tier: None,
            user: None,
            store: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["output_text"], "hello");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["object"], "response");
    }

    #[test]
    fn response_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(ResponseStatus::InProgress).unwrap(),
            serde_json::Value::String("in_progress".to_string())
        );
    }

    #[test]
    fn output_message_serializes_with_assistant_role() {
        let item = ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
            "msg_1".to_string(),
            vec![ResponseOutputContent::output_text("hi".to_string())],
        ));
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "output_text");
        assert_eq!(json["content"][0]["text"], "hi");
    }

    #[test]
    fn function_call_output_serializes_with_required_fields() {
        let item = ResponseOutputItem::FunctionCall(ResponseFunctionCallOutput {
            id: "fc_1".to_string(),
            call_id: "call_abc".to_string(),
            name: "do_thing".to_string(),
            arguments: r#"{"x":1}"#.to_string(),
            status: ResponseItemStatus::Completed,
        });
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["type"], "function_call");
        assert_eq!(json["call_id"], "call_abc");
        assert_eq!(json["status"], "completed");
    }
}
