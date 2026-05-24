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

//! Anthropic Messages API response types (`POST /v1/messages`).
//!
//! Ported from upstream mlx-vlm `server/anthropic.py` (PR #1196). These
//! types model the outbound Anthropic
//! [Messages](https://docs.anthropic.com/en/api/messages) response and the
//! error envelope. Outbound assembly lives in
//! [`crate::server::anthropic_translator`].

use serde::Serialize;

/// Top-level non-streaming `POST /v1/messages` response.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageResponse {
    /// `msg_<uuid>`.
    pub id: String,
    /// Always `"message"`.
    #[serde(rename = "type")]
    pub message_type: String,
    /// Always `"assistant"`.
    pub role: String,
    /// Ordered output content blocks.
    pub content: Vec<AnthropicResponseBlock>,
    /// Echoed model identifier.
    pub model: String,
    /// Why generation stopped: `end_turn`, `max_tokens`, `stop_sequence`,
    /// `tool_use`, or `null` (streaming `message_start`).
    pub stop_reason: Option<String>,
    /// The matched stop sequence (when `stop_reason == "stop_sequence"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: AnthropicUsage,
}

impl AnthropicMessageResponse {
    /// Construct a response with the standard `type`/`role` constants.
    pub fn new(
        id: String,
        content: Vec<AnthropicResponseBlock>,
        model: String,
        stop_reason: Option<String>,
        stop_sequence: Option<String>,
        usage: AnthropicUsage,
    ) -> Self {
        Self {
            id,
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model,
            stop_reason,
            stop_sequence,
            usage,
        }
    }
}

/// An output content block in a response.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseBlock {
    /// Visible text.
    #[serde(rename = "text")]
    Text { text: String },
    /// Extended-thinking trace.
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Tool-call request emitted by the model.
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Token usage counters.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    /// Prompt tokens.
    pub input_tokens: usize,
    /// Generated tokens.
    pub output_tokens: usize,
}

/// Anthropic error envelope: `{ "type": "error", "error": {...} }`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub envelope_type: String,
    pub error: AnthropicErrorBody,
    /// HTTP status code (not serialised — used by IntoResponse).
    #[serde(skip)]
    pub status: axum::http::StatusCode,
}

/// Inner error object.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl AnthropicErrorResponse {
    /// Build an error with an explicit HTTP status and Anthropic error type.
    pub fn new(
        status: axum::http::StatusCode,
        message: impl Into<String>,
        error_type: impl Into<String>,
    ) -> Self {
        Self {
            envelope_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: error_type.into(),
                message: message.into(),
            },
            status,
        }
    }

    /// 400 `invalid_request_error`.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(
            axum::http::StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
        )
    }

    /// 503 `overloaded_error` — all generation slots busy.
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            message,
            "overloaded_error",
        )
    }

    /// 500 `api_error` — generation failure.
    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "api_error",
        )
    }
}

impl axum::response::IntoResponse for AnthropicErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (self.status, axum::Json(self)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_response_serializes_with_constants() {
        let resp = AnthropicMessageResponse::new(
            "msg_1".to_string(),
            vec![AnthropicResponseBlock::Text {
                text: "hi".to_string(),
            }],
            "m".to_string(),
            Some("end_turn".to_string()),
            None,
            AnthropicUsage {
                input_tokens: 5,
                output_tokens: 3,
            },
        );
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hi");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"]["input_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens"], 3);
        // stop_sequence omitted when None
        assert!(v.get("stop_sequence").is_none());
    }

    #[test]
    fn tool_use_block_serializes() {
        let block = AnthropicResponseBlock::ToolUse {
            id: "toolu_1".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({"location": "SF"}),
        };
        let v = serde_json::to_value(&block).unwrap();
        assert_eq!(v["type"], "tool_use");
        assert_eq!(v["id"], "toolu_1");
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["input"]["location"], "SF");
    }

    #[test]
    fn error_envelope_shape() {
        let err = AnthropicErrorResponse::bad_request("nope");
        let v = serde_json::to_value(&err).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "nope");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }
}
