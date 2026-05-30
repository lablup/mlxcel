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

//! OpenAI Responses API request types.
//!
//! Wire shape mirrors `openai-python` `response_create_params.py`. Phase 1
//! supports the subset documented in the acceptance criteria
//! text input (string or typed item array), function tools, structured
//! output via `text.format.json_schema`, `previous_response_id` and
//! string-form `conversation` chaining, plus standard sampling overrides.
//! Built-in tool types (`web_search`, `file_search`, `computer_use_preview`,
//! `code_interpreter`, `image_generation`, `mcp`, `custom`, `apply_patch`,
//! `function_shell`) are deserialized into the [`ResponseTool::Unsupported`]
//! catch-all so the route layer can reject them with a clean 400 listing
//! the unsupported tool type rather than silently ignoring them.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::request::{ContentPart, FunctionDefinition, SamplingParams, ToolChoice};

/// `POST /v1/responses` request body.
///
/// Optional fields stay [`Option`] so the inbound translator can detect
/// "not supplied" and apply Responses-specific defaults (e.g. `store=true`
/// is the OpenAI default). Sampling overrides flatten through
/// [`SamplingParams`] for parity with `/v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateResponseRequest {
    /// Model identifier — must match the loaded mlxcel model alias or path.
    pub model: String,

    /// Input items or a plain string. Plain string is treated as a single
    /// user message; an array follows the typed-item discriminator in
    /// [`ResponseInputItem`].
    pub input: ResponseInput,

    /// System-like prompt prepended to the rendered conversation. Per the
    /// OpenAI rule, `instructions` from a referenced response (via
    /// `previous_response_id`) is **not** carried forward.
    #[serde(default)]
    pub instructions: Option<String>,

    /// Tool definitions. Only `{"type": "function", ...}` is accepted in
    /// Phase 1; every other variant maps to [`ResponseTool::Unsupported`].
    #[serde(default)]
    pub tools: Option<Vec<ResponseTool>>,

    /// Tool-selection strategy. Mirrors the chat-completions shape — the
    /// Responses-specific variants (`ToolChoiceMcp`, `ToolChoiceCustom`,
    /// etc.) require built-in tools that Phase 1 rejects, so the simple
    /// string/named-function form here is sufficient.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,

    /// Whether the model may issue multiple function calls in parallel.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Structured-output spec. Replaces chat completions' `response_format`
    /// — the JSON schema lives under `text.format.json_schema`.
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,

    /// Reasoning controls. Phase 1 honours `summary` only when the model's
    /// thinking-token budget machinery activates; `effort` is recorded
    /// (echoed back on the response) but otherwise treated as advisory.
    #[serde(default)]
    pub reasoning: Option<ResponseReasoningConfig>,

    /// Conversation reference. Accepts a bare id string or
    /// `{"id": "..."}`. Mutually exclusive with `previous_response_id`.
    #[serde(default)]
    pub conversation: Option<ConversationRef>,

    /// Prior response to chain off of. Mutually exclusive with
    /// `conversation`. The translator pulls the stored response's input
    /// and output items, appends them as history, then appends the
    /// current request's input.
    #[serde(default)]
    pub previous_response_id: Option<String>,

    /// Whether the server should persist the response for later retrieval
    /// via `GET /v1/responses/:id`. Defaults to `true` per OpenAI.
    #[serde(default)]
    pub store: Option<bool>,

    /// Whether to stream the response as SSE events.
    #[serde(default)]
    pub stream: bool,

    /// Streaming options.
    #[serde(default)]
    pub stream_options: Option<ResponseStreamOptions>,

    /// Cap on generated tokens. Mirrors `max_tokens` from chat completions.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,

    /// Cap on the number of tool calls in a single response. Phase 1
    /// honours this as a soft cap by truncating the parsed function-call
    /// output items.
    #[serde(default)]
    pub max_tool_calls: Option<usize>,

    /// Truncation policy. `"auto"` is recorded but treated as `disabled`
    /// in Phase 1 — oversize inputs are rejected with a 400.
    #[serde(default)]
    pub truncation: Option<String>,

    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling threshold.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Number of top log-probability alternatives to return per token.
    /// Forwarded to the chat-completions logprobs machinery.
    #[serde(default)]
    pub top_logprobs: Option<u8>,

    /// Free-form metadata. Capped at 16 entries by the route layer.
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,

    /// Prompt-prefix cache hint.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,

    /// OpenAI-deprecated end-user identifier.
    #[serde(default)]
    pub user: Option<String>,

    /// New name for `user`. Either field is accepted.
    #[serde(default)]
    pub safety_identifier: Option<String>,

    /// Background mode. Phase 1 rejects `background=true` with 400 since
    /// async response polling is deferred to Phase 3.
    #[serde(default)]
    pub background: Option<bool>,

    /// Service-tier hint. Accepted but ignored.
    #[serde(default)]
    pub service_tier: Option<String>,

    /// Sampling parameters that overlap with chat completions
    /// (`top_k`, `min_p`, repetition penalties, DRY, frequency/presence,
    /// `seed`, `stop`, `thinking_budget*`).
    #[serde(default, flatten)]
    pub sampling: SamplingParams,
}

/// `input` field discriminator: a bare string or an array of typed items.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Bare string — treated as a single user message.
    Text(String),
    /// Array of typed input items.
    Items(Vec<ResponseInputItem>),
}

impl ResponseInput {
    /// Materialise the input as a vector of items so downstream code can
    /// iterate uniformly. A bare string yields a single user message.
    pub fn into_items(self) -> Vec<ResponseInputItem> {
        match self {
            ResponseInput::Text(s) => vec![ResponseInputItem::Message {
                role: ResponseInputRole::User,
                content: ResponseInputContent::Text(s),
                name: None,
            }],
            ResponseInput::Items(items) => items,
        }
    }
}

/// Conversation reference — bare id or `{id: ...}` envelope.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ConversationRef {
    /// Bare id string.
    Id(String),
    /// `{"id": "conv_..."}` form.
    Object(ConversationRefObject),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConversationRefObject {
    pub id: String,
}

impl ConversationRef {
    pub fn id(&self) -> &str {
        match self {
            ConversationRef::Id(s) => s.as_str(),
            ConversationRef::Object(o) => o.id.as_str(),
        }
    }
}

/// Typed input item — `type` discriminator. Mirrors the OpenAI
/// `ResponseInputItem` union covered by Phase 1.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    /// Plain message — content is a string or content-part array.
    Message {
        role: ResponseInputRole,
        content: ResponseInputContent,
        #[serde(default)]
        name: Option<String>,
    },
    /// Prior function call (typically rehydrated from a stored response).
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    /// Result of a prior function call.
    FunctionCallOutput { call_id: String, output: String },
    /// Reasoning trace from a prior turn (rehydrated). Phase 1 records
    /// the text but does not push it through the thinking-token machinery.
    Reasoning { content: Vec<ReasoningContentPart> },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningContentPart {
    #[serde(rename = "type", default = "default_reasoning_type")]
    pub part_type: String,
    pub text: String,
}

fn default_reasoning_type() -> String {
    "reasoning_text".to_string()
}

/// Sender of an input message item.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseInputRole {
    User,
    Assistant,
    System,
    Developer,
}

/// Content of an input message — bare string or typed parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInputContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl ResponseInputContent {
    pub fn as_text(&self) -> String {
        match self {
            ResponseInputContent::Text(s) => s.clone(),
            ResponseInputContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Tool declaration. Only `function` is supported in Phase 1.
///
/// `Unsupported` is the catch-all the route layer rejects with a clean
/// 400; the deserialiser keeps the original `type` string in
/// [`UnsupportedToolType`] so the error message can name what was sent.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    /// `{"type": "function", "name": ..., "description": ..., "parameters": ...}`
    Function(FunctionToolDefinition),
    /// Catch-all for unsupported built-in tools. The deserialiser captures
    /// the raw object so the route can echo back the offending `type`.
    #[serde(other)]
    Unsupported,
}

/// Function-tool definition. The Responses API flattens the fields under
/// the tool object directly (unlike chat completions, which nests them
/// under `{"function": {...}}`).
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

impl FunctionToolDefinition {
    /// Convert to the chat-completions `FunctionDefinition` shape so the
    /// existing tool-call machinery can consume it without changes.
    pub fn to_function_definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

/// `text` config (replaces chat completions' `response_format`).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: Option<ResponseTextFormat>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    /// Plain text (no constraint).
    Text,
    /// JSON-schema constrained output. The schema payload mirrors the
    /// chat-completions shape so [`crate::server::structured`] can consume
    /// it unmodified after rehydration.
    JsonSchema {
        #[serde(flatten)]
        schema: serde_json::Value,
    },
    /// JSON-object loose mode. Not supported in Phase 1.
    #[serde(other)]
    Unsupported,
}

impl ResponseTextFormat {
    /// Re-encode this format as the `response_format` value expected by
    /// [`crate::server::structured::build_constraint_from_response_format`].
    /// Returns `None` when no constraint is needed.
    pub fn to_response_format_value(&self) -> Option<serde_json::Value> {
        match self {
            ResponseTextFormat::Text => None,
            ResponseTextFormat::JsonSchema { schema } => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "type".to_string(),
                    serde_json::Value::String("json_schema".to_string()),
                );
                if let Some(json_schema) = schema.get("json_schema") {
                    obj.insert("json_schema".to_string(), json_schema.clone());
                } else {
                    obj.insert("json_schema".to_string(), schema.clone());
                }
                Some(serde_json::Value::Object(obj))
            }
            ResponseTextFormat::Unsupported => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseStreamOptions {
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_as_string_yields_single_user_message() {
        let req: CreateResponseRequest =
            serde_json::from_str(r#"{"model":"m","input":"hello"}"#).unwrap();
        let items = req.input.into_items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseInputItem::Message { role, content, .. } => {
                assert_eq!(*role, ResponseInputRole::User);
                assert_eq!(content.as_text(), "hello");
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn input_as_items_round_trips() {
        let req: CreateResponseRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "input":[
                    {"type":"message","role":"system","content":"sys"},
                    {"type":"message","role":"user","content":[{"type":"text","text":"u"}]}
                ]
            }"#,
        )
        .unwrap();
        let items = req.input.into_items();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn conversation_ref_accepts_string_and_object() {
        let bare: ConversationRef = serde_json::from_str(r#""conv_abc""#).unwrap();
        assert_eq!(bare.id(), "conv_abc");
        let obj: ConversationRef = serde_json::from_str(r#"{"id":"conv_xyz"}"#).unwrap();
        assert_eq!(obj.id(), "conv_xyz");
    }

    #[test]
    fn unsupported_tool_type_falls_through_to_catch_all() {
        let tool: ResponseTool = serde_json::from_str(r#"{"type":"web_search"}"#).unwrap();
        matches!(tool, ResponseTool::Unsupported);
    }

    #[test]
    fn function_tool_parses_with_parameters() {
        let tool: ResponseTool = serde_json::from_str(
            r#"{"type":"function","name":"add","parameters":{"type":"object"}}"#,
        )
        .unwrap();
        match tool {
            ResponseTool::Function(f) => {
                assert_eq!(f.name, "add");
                assert!(f.parameters.is_some());
            }
            _ => panic!("expected function tool"),
        }
    }

    #[test]
    fn text_format_json_schema_round_trips() {
        let fmt: ResponseTextFormat = serde_json::from_str(
            r#"{"type":"json_schema","json_schema":{"name":"r","schema":{"type":"object"}}}"#,
        )
        .unwrap();
        let v = fmt.to_response_format_value().expect("constraint");
        assert_eq!(v["type"], "json_schema");
        assert!(v["json_schema"]["schema"].is_object());
    }

    #[test]
    fn text_format_plain_text_yields_no_constraint() {
        let fmt: ResponseTextFormat = serde_json::from_str(r#"{"type":"text"}"#).unwrap();
        assert!(fmt.to_response_format_value().is_none());
    }
}
