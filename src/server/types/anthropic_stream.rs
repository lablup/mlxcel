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

//! Anthropic Messages API streaming SSE event types.
//!
//! Ported from upstream mlx-vlm `server/anthropic.py` (PR #1196). The
//! Anthropic streaming protocol uses named SSE events, each carrying a typed
//! `data:` payload whose `type` field matches the `event:` name. This module
//! owns the typed event enum; the SSE encoder lives in
//! [`crate::server::streaming_anthropic`].
//!
//! ## Envelope ordering
//!
//! For a typical text-only generation, the encoder emits:
//!
//! 1. `message_start` (empty `content`, `usage.input_tokens` populated)
//! 2. `content_block_start` (index 0, empty `text` block)
//! 3. N × `content_block_delta` (`text_delta`)
//! 4. `content_block_stop` (index 0)
//! 5. `message_delta` (`stop_reason`, `usage.output_tokens`)
//! 6. `message_stop`
//!
//! Thinking blocks open a `thinking` content block (with `thinking_delta` /
//! `signature_delta`); tool calls open `tool_use` blocks with
//! `input_json_delta`.

use serde::Serialize;

use super::anthropic_response::{AnthropicMessageResponse, AnthropicResponseBlock};

/// A typed streaming event. The `event:` SSE header name is returned by
/// [`AnthropicStreamEvent::event_name`]; the variant serialises to the
/// `data:` payload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    /// `message_start`: opening envelope with the partial message object.
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageResponse },
    /// `content_block_start`: a new content block opened at `index`.
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicResponseBlock,
    },
    /// `content_block_delta`: an incremental update to the block at `index`.
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: AnthropicBlockDelta,
    },
    /// `content_block_stop`: the block at `index` is complete.
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    /// `message_delta`: top-level updates (stop reason, cumulative usage).
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaBody,
        usage: AnthropicMessageDeltaUsage,
    },
    /// `message_stop`: terminal event.
    #[serde(rename = "message_stop")]
    MessageStop,
    /// `error`: mid-stream failure envelope.
    #[serde(rename = "error")]
    Error { error: AnthropicStreamError },
    /// `ping`: keep-alive (Anthropic sends these; we map keepalive to it).
    #[serde(rename = "ping")]
    Ping,
}

impl AnthropicStreamEvent {
    /// The SSE `event:` header value.
    pub fn event_name(&self) -> &'static str {
        match self {
            AnthropicStreamEvent::MessageStart { .. } => "message_start",
            AnthropicStreamEvent::ContentBlockStart { .. } => "content_block_start",
            AnthropicStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            AnthropicStreamEvent::ContentBlockStop { .. } => "content_block_stop",
            AnthropicStreamEvent::MessageDelta { .. } => "message_delta",
            AnthropicStreamEvent::MessageStop => "message_stop",
            AnthropicStreamEvent::Error { .. } => "error",
            AnthropicStreamEvent::Ping => "ping",
        }
    }
}

/// Incremental delta payload for a content block.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicBlockDelta {
    /// Append text to a `text` block.
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    /// Append reasoning to a `thinking` block.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// Finalise a `thinking` block's cryptographic signature.
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    /// Append partial JSON to a `tool_use` block's input.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// `message_delta.delta` body.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// `message_delta.usage` body (cumulative output tokens).
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageDeltaUsage {
    pub output_tokens: usize,
}

/// Mid-stream error body.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::anthropic_response::AnthropicUsage;

    #[test]
    fn event_names_match_type_tags() {
        let cases: Vec<(AnthropicStreamEvent, &str)> = vec![
            (
                AnthropicStreamEvent::ContentBlockStop { index: 0 },
                "content_block_stop",
            ),
            (AnthropicStreamEvent::MessageStop, "message_stop"),
            (AnthropicStreamEvent::Ping, "ping"),
        ];
        for (ev, name) in cases {
            assert_eq!(ev.event_name(), name);
            let v = serde_json::to_value(&ev).unwrap();
            assert_eq!(v["type"], name);
        }
    }

    #[test]
    fn message_start_serializes_message() {
        let ev = AnthropicStreamEvent::MessageStart {
            message: AnthropicMessageResponse::new(
                "msg_1".to_string(),
                vec![],
                "m".to_string(),
                None,
                None,
                AnthropicUsage {
                    input_tokens: 7,
                    output_tokens: 0,
                },
            ),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "message_start");
        assert_eq!(v["message"]["type"], "message");
        assert_eq!(v["message"]["usage"]["input_tokens"], 7);
        assert!(v["message"]["stop_reason"].is_null());
    }

    #[test]
    fn text_delta_shape() {
        let ev = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicBlockDelta::TextDelta {
                text: "hi".to_string(),
            },
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "content_block_delta");
        assert_eq!(v["index"], 0);
        assert_eq!(v["delta"]["type"], "text_delta");
        assert_eq!(v["delta"]["text"], "hi");
    }

    #[test]
    fn input_json_delta_shape() {
        let ev = AnthropicStreamEvent::ContentBlockDelta {
            index: 2,
            delta: AnthropicBlockDelta::InputJsonDelta {
                partial_json: "{\"a\":1}".to_string(),
            },
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["delta"]["type"], "input_json_delta");
        assert_eq!(v["delta"]["partial_json"], "{\"a\":1}");
    }

    #[test]
    fn message_delta_shape() {
        let ev = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: AnthropicMessageDeltaUsage { output_tokens: 12 },
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "message_delta");
        assert_eq!(v["delta"]["stop_reason"], "end_turn");
        assert!(v["delta"]["stop_sequence"].is_null());
        assert_eq!(v["usage"]["output_tokens"], 12);
    }
}
