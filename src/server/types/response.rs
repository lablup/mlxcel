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

//! OpenAI and llama-server compatible response types

use serde::Serialize;

// ---------------------------------------------------------------------------
// Logprobs structures
// ---------------------------------------------------------------------------

/// Top-logprob alternative for a single token position (chat completions)
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    /// Text representation of the token
    pub token: String,
    /// Log probability of this token
    pub logprob: f32,
    /// UTF-8 byte encoding of the token text (`null` for non-UTF-8 tokens)
    pub bytes: Option<Vec<u8>>,
}

/// Log probability data for a single generated token (chat completions)
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    /// Text representation of the token
    pub token: String,
    /// Log probability of this token
    pub logprob: f32,
    /// UTF-8 byte encoding of the token text (`null` for non-UTF-8 tokens)
    pub bytes: Option<Vec<u8>>,
    /// Top-k alternative tokens at this position (empty when `top_logprobs` is 0)
    pub top_logprobs: Vec<TopLogprob>,
}

/// Logprobs container for chat completion choices
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    /// Per-token log probability data for the generated content
    pub content: Option<Vec<TokenLogprob>>,
}

/// Logprobs container for legacy text completion choices
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    /// Decoded token strings
    pub tokens: Vec<String>,
    /// Log probability of each token
    pub token_logprobs: Vec<f32>,
    /// Character offset of each token within the completion text
    pub text_offset: Vec<usize>,
    /// Top-k alternatives at each position (each entry is `null` or a map of
    /// token→logprob)
    pub top_logprobs: Vec<Option<std::collections::HashMap<String, f32>>>,
}

/// Breakdown of prompt tokens by origin.
///
/// Present in the response body as `usage.prompt_tokens_details` when the
/// prompt-prefix cache is active. Omitted (`null` / absent) when the cache
/// feature is disabled so that clients relying on the `null` sentinel do not
/// need any changes (wire compatibility for disabled mode).
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    /// Number of prompt tokens that were served directly from the KV prefix
    /// cache, bypassing recomputation in the prefill stage.
    pub cached_tokens: u64,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Breakdown of prompt-token origins. `None` when the prompt-prefix cache
    /// feature is disabled (preserves wire compatibility for disabled mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    /// Log probabilities for output tokens; `null` when not requested
    pub logprobs: Option<ChatLogprobs>,
}

/// Chat message in response.
///
/// `content` is always serialized (as a string or JSON `null`) so the shape
/// matches the OpenAI / llama.cpp / vLLM convention: when the assistant
/// turn is a tool call with no accompanying text, `content` comes back as
/// explicit `null` rather than `""`. Routers that key on `content.is_null()`
/// to distinguish tool-only turns rely on this distinction
/// (`continuum-router/src/core/tool_calling/response.rs:21`,
/// `src/core/tool_calling/streaming.rs:258`).
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    /// Reasoning / thinking scratchpad surfaced for thinking models (Qwen-style
    /// `<think>…</think>`, Gemma 4 `<|channel>thought…<channel|>`). Present only
    /// when the model produced reasoning; omitted otherwise so non-thinking
    /// responses keep the existing wire shape. Mirrors the OpenAI / vLLM
    /// `reasoning_content` convention and matches the streaming path's
    /// `delta.reasoning_content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls made by the assistant (present when finish_reason is "tool_calls")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

/// A single tool call in the response (OpenAI format)
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallResponse {
    /// Unique tool call ID (format: "call_" + random string)
    pub id: String,
    /// Always "function"
    #[serde(rename = "type")]
    pub call_type: String,
    /// Function call details
    pub function: ToolCallFunctionResponse,
}

/// Function name + arguments in a tool call response
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionResponse {
    /// Function name
    pub name: String,
    /// Stringified JSON arguments
    pub arguments: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(
        id: String,
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
    ) -> Self {
        Self::new_with_logprobs(
            id,
            model,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            None,
        )
    }

    pub fn new_with_logprobs(
        id: String,
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
        logprobs: Option<ChatLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason,
                logprobs,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            },
        }
    }

    /// Create a response with tool calls.
    ///
    /// When `content` is empty, the serialized message sets `content` to
    /// JSON `null` (per OpenAI / llama.cpp / vLLM convention for tool-only
    /// assistant turns). A non-empty `content` is preserved verbatim so
    /// mixed-content responses (text preamble + tool call) still round-trip.
    /// `finish_reason` is automatically set to "tool_calls".
    pub fn new_with_tool_calls(
        id: String,
        model: String,
        content: String,
        tool_calls: Vec<ToolCallResponse>,
        prompt_tokens: usize,
        completion_tokens: usize,
        logprobs: Option<ChatLogprobs>,
    ) -> Self {
        let message_content = if content.is_empty() {
            None
        } else {
            Some(content)
        };
        Self {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: message_content,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls),
                },
                finish_reason: Some("tool_calls".to_string()),
                logprobs,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            },
        }
    }

    /// Populate `usage.prompt_tokens_details.cached_tokens` from the
    /// generation result's `cached_tokens` field. When `cached_tokens == 0`
    /// and `cache_enabled == false`, the details field is left as `None` to
    /// preserve wire compatibility. When `cache_enabled == true`, the field is
    /// always set (possibly to zero for a cold-miss request) so clients can
    /// distinguish "cache off" from "cache on but no hit".
    ///
    /// Used by: chat.rs (both streaming and non-streaming paths)
    #[must_use]
    pub fn with_cached_tokens(mut self, cached_tokens: usize, cache_enabled: bool) -> Self {
        if cache_enabled {
            self.usage.prompt_tokens_details = Some(PromptTokensDetails {
                cached_tokens: cached_tokens as u64,
            });
        }
        self
    }

    /// Attach reasoning / thinking scratchpad text to the first choice's
    /// message as `reasoning_content`. `None` leaves the field absent (it is
    /// `#[serde(skip_serializing_if = "Option::is_none")]`), so non-thinking
    /// responses keep the existing wire shape. This is the non-streaming
    /// counterpart to the streaming `delta.reasoning_content` chunks; both are
    /// derived from the same `StreamFilter` so the two endpoints surface
    /// identical reasoning. Chaining mirrors `with_cached_tokens`.
    ///
    /// Used by: chat.rs (non-streaming path)
    #[must_use]
    pub fn with_reasoning_content(mut self, reasoning: Option<String>) -> Self {
        if let Some(choice) = self.choices.first_mut() {
            choice.message.reasoning_content = reasoning;
        }
        self
    }
}

/// Text completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
    /// Log probabilities for output tokens; `null` when not requested
    pub logprobs: Option<CompletionLogprobs>,
}

/// Text completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(
        id: String,
        model: String,
        text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
    ) -> Self {
        Self::new_with_logprobs(
            id,
            model,
            text,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            None,
        )
    }

    pub fn new_with_logprobs(
        id: String,
        model: String,
        text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: Option<String>,
        logprobs: Option<CompletionLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason,
                logprobs,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            },
        }
    }
}

/// Model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchStatusInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability: Option<crate::server::batch::ObservabilitySnapshot>,
    /// Effective context window size in tokens.
    ///
    /// Reports the effective per-slot `--ctx-size` value. When the server was
    /// started without an explicit `--ctx-size` override this is 0, which means
    /// the model's own `max_position_embeddings` applies. Monitoring tools may
    /// use this as a hint; `0` should be treated as "model default / unknown".
    ///
    /// Present only when the model is loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<usize>,
    /// Name of the active tool-call parser, or `null` when the loaded chat
    /// template does not support tool calls.
    ///
    /// mlxcel's parser is output-format auto-detection (tries Hermes, Gemma 4,
    /// Mistral Nemo, Functionary, Llama 3, Command-R, and others in sequence).
    /// The field is always present once a model is loaded so that monitoring
    /// tools can distinguish "template does not support tools" (`null`) from
    /// "field missing because model has not finished loading" (field absent).
    ///
    /// Present only when the model is loaded.
    pub tool_call_parser: Option<String>,
}

/// Batch status included in the health check response.
#[derive(Debug, Clone, Serialize)]
pub struct BatchStatusInfo {
    pub active_sequences: usize,
    pub queue_depth: usize,
    pub max_batch_size: usize,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
    /// HTTP status code (not serialized — used by IntoResponse)
    #[serde(skip)]
    pub status: axum::http::StatusCode,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
            status: axum::http::StatusCode::BAD_REQUEST,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "server_busy".into(),
                code: None,
            },
            status: axum::http::StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl axum::response::IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (self.status, axum::Json(self)).into_response()
    }
}

/// Native llama-server completion response (POST /completion)
#[derive(Debug, Clone, Serialize)]
pub struct NativeCompletionResponse {
    pub content: String,
    pub stop: bool,
    pub generation_settings: serde_json::Value,
    pub model: String,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    pub timings: TimingInfo,
}

/// Timing information for generation (llama-server compatible)
#[derive(Debug, Clone, Serialize)]
pub struct TimingInfo {
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

/// Tokenize response (POST /tokenize)
#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i32>,
}

/// Detokenize response (POST /detokenize)
#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

/// Server properties response (GET /props)
#[derive(Debug, Clone, Serialize)]
pub struct PropsResponse {
    pub default_generation_settings: serde_json::Value,
    pub total_slots: usize,
}

/// Slot information (GET /slots)
#[derive(Debug, Clone, Serialize)]
pub struct SlotInfo {
    pub id: usize,
    pub state: String,
    pub model: String,
    /// Effective per-slot context window in tokens (`0` = model default).
    pub context_size: usize,
    pub is_processing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Usage serialization ------------------------------------

    /// When `prompt_tokens_details` is `None` the field must be omitted from
    /// the JSON output entirely (wire compatibility for disabled-cache mode).
    #[test]
    fn usage_without_cache_omits_prompt_tokens_details() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert!(
            !json
                .as_object()
                .unwrap()
                .contains_key("prompt_tokens_details"),
            "prompt_tokens_details must be absent when None"
        );
        assert_eq!(json["prompt_tokens"], 10);
        assert_eq!(json["completion_tokens"], 5);
        assert_eq!(json["total_tokens"], 15);
    }

    /// When `prompt_tokens_details` is `Some`, the field must be present and
    /// carry the correct `cached_tokens` value.
    #[test]
    fn usage_with_cache_includes_prompt_tokens_details() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 64 }),
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["prompt_tokens_details"]["cached_tokens"], 64);
    }

    /// `with_cached_tokens` must populate the field when `cache_enabled=true`.
    #[test]
    fn with_cached_tokens_sets_details_when_cache_enabled() {
        let resp = ChatCompletionResponse::new(
            "id".to_string(),
            "model".to_string(),
            "hi".to_string(),
            50,
            10,
            Some("stop".to_string()),
        )
        .with_cached_tokens(32, true);

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 32);
    }

    /// `with_cached_tokens` must leave the field absent when `cache_enabled=false`.
    #[test]
    fn with_cached_tokens_omits_details_when_cache_disabled() {
        let resp = ChatCompletionResponse::new(
            "id".to_string(),
            "model".to_string(),
            "hi".to_string(),
            50,
            10,
            Some("stop".to_string()),
        )
        .with_cached_tokens(99, false);

        let json = serde_json::to_value(&resp).unwrap();
        assert!(
            !json["usage"]
                .as_object()
                .unwrap()
                .contains_key("prompt_tokens_details"),
            "prompt_tokens_details must be absent when cache is disabled"
        );
    }

    /// `with_cached_tokens` with zero and `cache_enabled=true` should still
    /// emit the field (cold miss on a cache-enabled server).
    #[test]
    fn with_cached_tokens_zero_with_cache_enabled_emits_field() {
        let resp = ChatCompletionResponse::new(
            "id".to_string(),
            "model".to_string(),
            "hi".to_string(),
            50,
            10,
            Some("stop".to_string()),
        )
        .with_cached_tokens(0, true);

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    }

    // -- reasoning_content (thinking-scratchpad surface) --------

    /// A plain response (no reasoning) must omit `reasoning_content` so the
    /// wire shape is unchanged for non-thinking models.
    #[test]
    fn reasoning_content_absent_when_none() {
        let resp = ChatCompletionResponse::new(
            "id".to_string(),
            "model".to_string(),
            "the answer".to_string(),
            10,
            5,
            Some("stop".to_string()),
        )
        .with_reasoning_content(None);

        let json = serde_json::to_value(&resp).unwrap();
        let message = &json["choices"][0]["message"];
        assert!(
            !message
                .as_object()
                .unwrap()
                .contains_key("reasoning_content"),
            "reasoning_content must be absent when None, got: {message}"
        );
        assert_eq!(message["content"], "the answer");
    }

    /// When reasoning is present, `with_reasoning_content` must surface it on
    /// the first choice's message as `reasoning_content`.
    #[test]
    fn reasoning_content_present_when_some() {
        let resp = ChatCompletionResponse::new(
            "id".to_string(),
            "model".to_string(),
            "the answer".to_string(),
            10,
            5,
            Some("stop".to_string()),
        )
        .with_reasoning_content(Some("let me think about hash tables".to_string()));

        let json = serde_json::to_value(&resp).unwrap();
        let message = &json["choices"][0]["message"];
        assert_eq!(
            message["reasoning_content"],
            "let me think about hash tables"
        );
        assert_eq!(message["content"], "the answer");
    }

    #[test]
    fn slot_info_serializes_effective_context_size() {
        let slot = SlotInfo {
            id: 0,
            state: "idle".to_string(),
            model: "model".to_string(),
            context_size: 2048,
            is_processing: false,
            prompt_tokens: None,
            generated_tokens: None,
            elapsed_ms: None,
        };

        let json = serde_json::to_value(&slot).unwrap();
        assert_eq!(json["context_size"], 2048);
    }

    // -- OpenAI/llama.cpp/vLLM tool-call content=null shape --
    //
    // continuum-router documents the target OpenAI/llama.cpp format with
    // `"content": null` for tool-only assistant turns
    // (`src/core/tool_calling/response.rs:21`). These tests lock the wire
    // shape so the router's null-check consumers keep working.

    #[test]
    fn tool_only_turn_serializes_content_as_null() {
        let tool_call = ToolCallResponse {
            id: "call_abc".to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunctionResponse {
                name: "get_current_time".to_string(),
                arguments: "{\"format\":\"human\"}".to_string(),
            },
        };
        let resp = ChatCompletionResponse::new_with_tool_calls(
            "chatcmpl-1".to_string(),
            "g4".to_string(),
            String::new(),
            vec![tool_call],
            10,
            5,
            None,
        );
        let json = serde_json::to_value(&resp).unwrap();
        let message = &json["choices"][0]["message"];
        assert!(
            message["content"].is_null(),
            "tool-only turn must serialize content as JSON null, got: {}",
            message
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            message["tool_calls"][0]["function"]["name"],
            "get_current_time"
        );
    }

    #[test]
    fn tool_call_with_text_preamble_preserves_content() {
        // Mixed responses (text + tool call) must keep the preamble string
        // intact; the null treatment is reserved for empty-content tool-only
        // turns. Round-trip the OpenAI "content alongside tool_calls" shape.
        let tool_call = ToolCallResponse {
            id: "call_xyz".to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunctionResponse {
                name: "calculator".to_string(),
                arguments: "{\"expression\":\"2+2\"}".to_string(),
            },
        };
        let resp = ChatCompletionResponse::new_with_tool_calls(
            "chatcmpl-2".to_string(),
            "g4".to_string(),
            "Let me compute that.".to_string(),
            vec![tool_call],
            10,
            8,
            None,
        );
        let json = serde_json::to_value(&resp).unwrap();
        let message = &json["choices"][0]["message"];
        assert_eq!(message["content"], "Let me compute that.");
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn regular_response_serializes_content_as_string() {
        let resp = ChatCompletionResponse::new(
            "chatcmpl-3".to_string(),
            "g4".to_string(),
            "hello world".to_string(),
            5,
            3,
            Some("stop".to_string()),
        );
        let json = serde_json::to_value(&resp).unwrap();
        let message = &json["choices"][0]["message"];
        assert_eq!(message["content"], "hello world");
        assert!(
            message.get("tool_calls").is_none() || message["tool_calls"].is_null(),
            "non-tool response must omit tool_calls field"
        );
    }

    #[test]
    fn regular_response_with_empty_content_serializes_as_empty_string() {
        // A regular (non-tool) response with explicitly empty text stays a
        // string — the null treatment is specific to tool-only turns. This
        // prevents a refactor from accidentally broadening the null rule.
        let resp = ChatCompletionResponse::new(
            "chatcmpl-4".to_string(),
            "g4".to_string(),
            String::new(),
            5,
            0,
            Some("length".to_string()),
        );
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "");
    }
}
