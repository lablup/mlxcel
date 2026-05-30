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

//! OpenAI and llama-server compatible request types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

// ---------------------------------------------------------------------------
// Tool calling types (OpenAI-compatible)
// Used by: ChatCompletionRequest, chat_template, routes/chat
// ---------------------------------------------------------------------------

/// A tool definition (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Human-readable description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool choice specification
///
/// Can be a string ("auto", "none", "required") or an object specifying a
/// particular function.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode: "auto", "none", or "required"
    Mode(String),
    /// Specific function: {"type": "function", "function": {"name": "X"}}
    Specific(ToolChoiceFunction),
}

impl ToolChoice {
    /// Returns the mode string for simple string choices, or "specific" for
    /// named function selections.
    pub fn mode(&self) -> &str {
        match self {
            ToolChoice::Mode(s) => s.as_str(),
            ToolChoice::Specific(_) => "specific",
        }
    }

    /// Returns true if tool calling is effectively disabled.
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "none")
    }

    /// Returns the specific function name if this is a named choice.
    pub fn specific_function(&self) -> Option<&str> {
        match self {
            ToolChoice::Specific(f) => Some(&f.function.name),
            _ => None,
        }
    }
}

/// Named function tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunctionName,
}

/// Function name within a tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

/// A tool call within an assistant message (multi-turn history)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInMessage {
    /// Unique tool call ID
    pub id: String,
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// Function call details
    pub function: ToolCallFunction,
}

/// Function name + arguments within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name
    pub name: String,
    /// Stringified JSON arguments
    pub arguments: String,
}

impl Role {
    /// Convert role to lowercase string for chat templates
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Content part for multimodal messages (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (supports base64 data URIs, `file://`, local paths,
    /// and `http(s)` URLs)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Video URL content for VLMs that support video inputs.
    /// Mirrors the `image_url` shape for symmetry. Accepts local paths,
    /// `file://...`, and (where the model supports it) `http(s)://...`.
    /// Frame extraction relies on `ffmpeg` being available on the server
    /// host's PATH; missing `ffmpeg` produces a clean 4xx response rather
    /// than a crash.
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
    /// Audio input content (base64-encoded audio data)
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

/// Image URL reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL: `data:image/...;base64,...`, `file://...`, bare local path, or
    /// `http(s)://...`
    pub url: String,
}

/// Video URL reference. Same wire shape as [`ImageUrl`] for
/// symmetry with the OpenAI vision content blocks.
///
/// `fps` is an optional sampling rate. When omitted, the server falls
/// back to [`mlxcel::video::DEFAULT_FPS`] (2.0 fps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    /// URL: `data:video/...;base64,...` (where supported), `file://...`,
    /// bare local path, or `http(s)://...`.
    pub url: String,
    /// Optional sampling rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

/// Audio input reference (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data, or a URL/file path
    pub data: String,
    /// Audio format: "wav", "mp3", etc.
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

/// Message content: either a plain string or multimodal array
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content
    Text(String),
    /// Multimodal content parts (text + images)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content from the message
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image data URIs/paths from multimodal content
    pub fn image_urls(&self) -> Vec<String> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract audio input data from multimodal content
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::InputAudio { input_audio } => Some(input_audio.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract video URL references from multimodal content.
    pub fn video_urls(&self) -> Vec<VideoUrl> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::VideoUrl { video_url } => Some(video_url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl Default for MessageContent {
    /// Empty text — the canonical "no content" value.
    ///
    /// Assistant messages whose payload is a `tool_calls` array legitimately
    /// omit `content` or send `null` (issue #89); both map to this.
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

/// Deserialize [`MessageContent`], tolerating an explicit JSON `null`.
///
/// Paired with `#[serde(default)]` on the field, this lets `content` accept
/// the three shapes OpenAI-compatible clients emit on assistant messages that
/// carry `tool_calls` (issue #89): the key is absent (handled by `default`),
/// the value is `null` (mapped to empty content here), or the value is a
/// normal string / multimodal array. Without it, axum's `Json` extractor
/// rejects the follow-up request of a tool-calling loop with HTTP 422
/// (`missing field 'content'`).
fn deserialize_message_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<MessageContent>::deserialize(deserializer)?.unwrap_or_default())
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// Message content.
    ///
    /// Optional and nullable per the OpenAI Chat Completions spec: assistant
    /// messages that carry `tool_calls` may omit `content` or send `null`. A
    /// missing or `null` value deserializes to empty text (issue #89), keeping
    /// the `content` field present for Jinja chat templates that read
    /// `message.content`.
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID for `role: "tool"` messages (references a previous tool call)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant (multi-turn history)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInMessage>>,
}

/// Sampling parameters shared across endpoints
///
/// All parameters are optional. When not specified in the request,
/// server defaults (from CLI arguments) will be used.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random)
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold
    pub top_p: Option<f32>,
    /// Top-k sampling (0 = disabled)
    pub top_k: Option<usize>,
    /// Min-p sampling threshold (0.0 = disabled)
    pub min_p: Option<f32>,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: Option<f32>,
    /// Context size for repetition penalty
    pub repetition_context_size: Option<usize>,
    /// Logit bias for specific tokens
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    // DRY (Don't Repeat Yourself) sampling parameters
    /// DRY penalty multiplier (0.0 = disabled, typical: 0.8-1.3)
    pub dry_multiplier: Option<f32>,
    /// DRY penalty base for exponential scaling (typical: 1.75)
    pub dry_base: Option<f32>,
    /// Minimum sequence length before DRY penalties apply (typical: 2)
    pub dry_allowed_length: Option<usize>,
    /// Number of recent tokens to scan for DRY (0 = entire context)
    pub dry_penalty_last_n: Option<usize>,
    /// Sequence breaker tokens for DRY (resets matching)
    pub dry_sequence_breakers: Option<Vec<i32>>,

    // XTC (Exclude Top Choices) sampling parameters
    /// XTC probability (0.0 = disabled)
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold
    pub xtc_threshold: Option<f32>,

    // OpenAI-compatible frequency/presence penalties
    /// Frequency penalty (0.0 = disabled) - penalizes based on frequency
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (0.0 = disabled) - penalizes based on presence
    pub presence_penalty: Option<f32>,

    // thinking-token budget (Qwen3-family reasoning cap).
    //
    // Three aliases accepted, first non-None wins (see
    // `thinking_budget::pick_budget_alias`). llama.cpp-compatible primary
    // name, vLLM alias, and Qwen alias. Value semantics: -1 unrestricted,
    // 0 immediate close, N > 0 cap at N tokens inside the `<think>` block.
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,
}

/// Stream options for controlling streaming behavior
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    /// Include token usage statistics in the final streaming chunk
    #[serde(default)]
    pub include_usage: bool,
}

/// Chat completion request (POST /v1/chat/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier
    pub model: String,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether to return log probabilities of output tokens
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probability alternatives to return per token (0–20)
    #[serde(default)]
    pub top_logprobs: Option<u8>,

    // Tool calling fields (OpenAI-compatible)
    /// Tool definitions available for the model to call
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Controls how the model selects tools: "auto", "none", "required", or a
    /// specific function object
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue multiple tool calls in parallel
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// top-level `chat_template_kwargs` (llama.cpp shape).
    ///
    /// A JSON object whose keys are forwarded as Jinja template kwargs when
    /// rendering the conversation. Primary shape; wins over nested
    /// `extra_body.chat_template_kwargs`, flattened OpenAI-SDK `extra_body`
    /// aliases, and DashScope flat `extra_body.preserve_thinking`. See
    /// [`crate::server::chat_template_kwargs::extract_request_kwargs`] for
    /// the full precedence chain.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,

    /// nested `extra_body` compatibility (vLLM / manual callers).
    ///
    /// Some callers send an actual top-level `extra_body` object. Only the
    /// keys we currently recognize are read back out; unknown keys are
    /// silently ignored to match llama.cpp's lenient behavior.
    #[serde(default)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,

    /// OpenAI-compatible prompt-cache key hint.
    ///
    /// Clients can send this to pin a conversation to a specific prompt-cache
    /// session bucket. When present it wins over the standard OpenAI `user`
    /// field in [`crate::server::prompt_cache::key::resolve_session_key`];
    /// when absent the server falls back to `user`, then to an anonymous
    /// bucket sentinel. The string is never echoed back to the client — it is
    /// only used as a session-bucket discriminator inside the cache key hash.
    ///
    /// Also round-trips through the flattened OpenAI-SDK `extra_body`
    /// mechanism ([`Self::extra_body_fields`]): SDKs that pass `extra_body=
    /// {"prompt_cache_key": "..."}` land here via the flatten; SDKs that send
    /// it at the request root land here directly.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,

    /// OpenAI-standard stable end-user identifier.
    ///
    /// Used as a session-bucket fallback for the prompt-prefix cache when
    /// `prompt_cache_key` is not supplied. See
    /// [`crate::server::prompt_cache::key::resolve_session_key`] for the
    /// full precedence chain. The value is treated as opaque bytes; the
    /// server never attempts to interpret it as an identity or access control
    /// token.
    #[serde(default)]
    pub user: Option<String>,

    /// OpenAI SDK `extra_body={...}` flattened into the request root.
    ///
    /// The official OpenAI Python client merges `extra_body` into the top-level
    /// JSON object instead of emitting a nested `"extra_body": {...}` wrapper.
    /// Capture those unknown root keys here so request-kwarg extraction can
    /// treat them the same as nested `extra_body` aliases.
    #[serde(default, flatten)]
    pub extra_body_fields: serde_json::Map<String, serde_json::Value>,

    /// OpenAI-compatible structured-output spec.
    ///
    /// Accepts the OpenAI Chat Completions shape:
    ///
    /// ```json
    /// {
    ///   "response_format": {
    ///     "type": "json_schema",
    ///     "json_schema": {
    ///       "name": "result",
    ///       "strict": true,
    ///       "schema": { "type": "object", ... }
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// When set with `type: "json_schema"`, generation is constrained via
    /// [`crate::server::structured`] so emitted tokens always conform to the
    /// supplied schema. Other types (`text`, `null`) are no-ops; `json_object`
    /// is rejected as unsupported in the MVP — see `extract_json_schema_from_response_format`.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,

    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

/// Text completion request (POST /v1/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Number of top log-probability alternatives to return (legacy format: 0–5)
    #[serde(default)]
    pub logprobs: Option<u8>,
    /// OpenAI-compatible structured-output spec; see the
    /// matching field on [`ChatCompletionRequest`] for shape details.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

impl ChatCompletionRequest {
    /// Merge nested `extra_body` with flattened OpenAI-SDK root fields.
    ///
    /// Flattened root keys win over the nested object on collision because the
    /// request body already exposed them at the higher-precedence top level.
    pub fn merged_extra_body(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        match (&self.extra_body, self.extra_body_fields.is_empty()) {
            (None, true) => None,
            (Some(extra), true) => Some(extra.clone()),
            (None, false) => Some(self.extra_body_fields.clone()),
            (Some(extra), false) => {
                let mut merged = extra.clone();
                for (key, value) in &self.extra_body_fields {
                    merged.insert(key.clone(), value.clone());
                }
                Some(merged)
            }
        }
    }

    /// Resolve the request-level `prompt_cache_key`.
    ///
    /// Precedence (first non-empty wins):
    ///   1. Top-level `prompt_cache_key`.
    ///   2. Flattened OpenAI-SDK `extra_body` field of the same name.
    ///   3. Nested `extra_body.prompt_cache_key`.
    ///
    /// Empty strings are treated as "not supplied" so a caller can't
    /// accidentally smuggle themselves into an empty-string bucket.
    pub fn resolve_prompt_cache_key(&self) -> Option<&str> {
        if let Some(k) = self.prompt_cache_key.as_deref()
            && !k.is_empty()
        {
            return Some(k);
        }
        if let Some(s) = self
            .extra_body_fields
            .get("prompt_cache_key")
            .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(s) = body
                .get("prompt_cache_key")
                .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        None
    }

    /// Resolve the request-level OpenAI-standard `user` identifier.
    ///
    /// Same precedence rules as [`Self::resolve_prompt_cache_key`]: top-level
    /// field, then flattened `extra_body`, then nested `extra_body`. Empty
    /// strings are ignored.
    pub fn resolve_user(&self) -> Option<&str> {
        if let Some(u) = self.user.as_deref()
            && !u.is_empty()
        {
            return Some(u);
        }
        if let Some(s) = self
            .extra_body_fields
            .get("user")
            .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(s) = body.get("user").and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        None
    }

    /// Convert messages to a prompt string using a simple format
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            let text = msg.content.text();
            match msg.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n\n", text));
                }
                Role::User => {
                    prompt.push_str(&format!("User: {}\n\n", text));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n\n", text));
                }
                Role::Tool => {
                    prompt.push_str(&format!("Tool: {}\n\n", text));
                }
            }
        }
        prompt.push_str("Assistant: ");
        prompt
    }

    /// Extract all image URLs from messages
    pub fn image_urls(&self) -> Vec<String> {
        self.messages
            .iter()
            .flat_map(|m| m.content.image_urls())
            .collect()
    }

    /// Extract all audio inputs from messages
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        self.messages
            .iter()
            .flat_map(|m| m.content.audio_inputs())
            .collect()
    }

    /// Extract all video URL references from messages.
    pub fn video_urls(&self) -> Vec<VideoUrl> {
        self.messages
            .iter()
            .flat_map(|m| m.content.video_urls())
            .collect()
    }
}

/// Native llama-server completion request (POST /completion)
#[derive(Debug, Clone, Deserialize)]
pub struct NativeCompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum number of tokens to predict
    pub n_predict: Option<usize>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<i32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Min-p sampling
    pub min_p: Option<f32>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty last N tokens
    pub repeat_last_n: Option<usize>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed
    pub seed: Option<u64>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// DRY penalty multiplier (0.0 = disabled)
    pub dry_multiplier: Option<f32>,
    /// DRY exponential base
    pub dry_base: Option<f32>,
    /// DRY minimum match length before penalty
    pub dry_allowed_length: Option<usize>,
    /// DRY lookback window (-1 = full context)
    pub dry_penalty_last_n: Option<usize>,
    /// DRY sequence breaker token IDs
    pub dry_sequence_breakers: Option<Vec<i32>>,

    // thinking-token budget (Qwen3-family reasoning cap).
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,

    /// structured-output `response_format` is **not** supported
    /// on the native llama-server `/completion` endpoint. The field is
    /// captured here only so the route can reject the request with a clear
    /// 400 instead of silently ignoring the schema and emitting
    /// non-conforming output. Use `/v1/chat/completions` for constrained
    /// decoding.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
}

/// Tokenize request (POST /tokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// Text content to tokenize
    pub content: String,
    /// Whether to add special tokens (BOS/EOS)
    pub add_special: Option<bool>,
}

/// Detokenize request (POST /detokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    /// Token IDs to decode
    pub tokens: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_content_default_is_empty_text() {
        let c = MessageContent::default();
        assert_eq!(c.text(), "");
        assert!(matches!(c, MessageContent::Text(_)));
    }

    #[test]
    fn message_without_content_field_deserializes_to_empty() {
        // OpenAI-compatible clients omit `content` on assistant messages whose
        // payload is a `tool_calls` array (issue #89).
        let json = r#"{
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_system_info", "arguments": "{}"}}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).expect("missing content must deserialize");
        assert_eq!(msg.content.text(), "");
        assert!(msg.tool_calls.is_some());
    }

    #[test]
    fn message_with_null_content_deserializes_to_empty() {
        // Some clients send `"content": null` rather than omitting the key.
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: Message = serde_json::from_str(json).expect("null content must deserialize");
        assert_eq!(msg.content.text(), "");
        assert!(matches!(msg.content, MessageContent::Text(_)));
    }

    #[test]
    fn message_with_string_content_deserializes() {
        let json = r#"{"role": "user", "content": "hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.text(), "hello");
        assert!(matches!(msg.content, MessageContent::Text(_)));
    }

    #[test]
    fn message_with_multimodal_content_deserializes() {
        let json = r#"{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.content, MessageContent::Parts(_)));
        assert_eq!(msg.content.text(), "describe");
        assert_eq!(
            msg.content.image_urls(),
            vec!["data:image/png;base64,AAAA".to_string()]
        );
    }
}
