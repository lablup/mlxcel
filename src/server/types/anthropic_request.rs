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

//! Anthropic Messages API request types (`POST /v1/messages`).
//!
//! Ported from upstream mlx-vlm `server/anthropic.py` (PR #1196, commit
//! `313ad22`) plus the tool-call image-handling fix (PR #1200, commit
//! `f2e19de`). These types model the inbound Anthropic
//! [Messages](https://docs.anthropic.com/en/api/messages) schema; the
//! translation into the internal chat-completions shape lives in
//! [`crate::server::anthropic_translator`].

use serde::Deserialize;

/// Top-level `POST /v1/messages` request body.
///
/// Mirrors the Anthropic Messages request. Fields the local server cannot
/// honour (e.g. `metadata`, server-side tools) are accepted and ignored so
/// SDK clients keep working.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicRequest {
    /// Model identifier (must match the loaded model alias or path).
    pub model: String,
    /// Conversation turns.
    pub messages: Vec<AnthropicMessage>,
    /// Optional system prompt: a bare string or an array of text blocks.
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    /// Maximum number of tokens to generate. Required by the Anthropic API;
    /// optional here so callers that omit it fall back to the server default.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Up to N custom stop sequences. Generation text is truncated at the
    /// first matching sequence and `stop_reason` becomes `stop_sequence`.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response as SSE events.
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature (0.0 = greedy).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Tool definitions available for the model.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Controls how the model selects tools.
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Anthropic extended-thinking configuration. Read for `enable_thinking`
    /// and `budget_tokens` derivation.
    #[serde(default)]
    pub thinking: Option<AnthropicThinking>,
    /// Extra fields are tolerated (metadata, container, etc.) so SDK callers
    /// that send richer payloads do not get 400s.
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `system` may be a plain string or an array of typed content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    /// Bare string system prompt.
    Text(String),
    /// Array of blocks (only `text` blocks contribute text).
    Blocks(Vec<AnthropicSystemBlock>),
}

impl AnthropicSystem {
    /// Flatten the system prompt into a single string. Returns `None` when
    /// the prompt is empty after trimming.
    pub fn to_text(&self) -> Option<String> {
        match self {
            AnthropicSystem::Text(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(s.clone())
                }
            }
            AnthropicSystem::Blocks(blocks) => {
                let joined = blocks
                    .iter()
                    .filter_map(|b| b.text.as_deref())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                let trimmed = joined.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
        }
    }
}

/// A single block within an array-form `system` prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicSystemBlock {
    /// Block type (only `text` is interpreted).
    #[serde(rename = "type", default)]
    pub block_type: Option<String>,
    /// Text payload for `text` blocks.
    #[serde(default)]
    pub text: Option<String>,
}

/// A single conversation turn.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    /// `user`, `assistant`, or `system`.
    pub role: AnthropicRole,
    /// Content: a bare string or an array of typed content blocks.
    pub content: AnthropicMessageContent,
}

/// Message role.
///
/// The Anthropic spec only allows `user`/`assistant` inside `messages` (system
/// text belongs in the top-level `system` field), but Claude Code >= 2.1.156
/// interleaves `{"role":"system", ...}` turns inside the array for
/// mid-conversation system reminders. Modeling `System` here lets those
/// requests deserialize instead of failing the `Json` extractor with HTTP 422;
/// the translator then folds the text into an adjacent user turn (see
/// [`crate::server::anthropic_translator`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
    System,
}

impl AnthropicRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnthropicRole::User => "user",
            AnthropicRole::Assistant => "assistant",
            AnthropicRole::System => "system",
        }
    }
}

/// Message content: a bare string or an array of typed content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Plain text content.
    Text(String),
    /// Array of content blocks.
    Blocks(Vec<AnthropicContentBlock>),
}

/// A typed content block inside a message.
///
/// `Unknown` is the catch-all for block types the server does not model
/// (e.g. `redacted_thinking`); they are tolerated and dropped during
/// translation.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Plain text block.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image block with a `source` (base64 or url).
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    /// Document block; only the `text`-typed source contributes text.
    #[serde(rename = "document")]
    Document {
        #[serde(default)]
        source: Option<serde_json::Value>,
    },
    /// Assistant tool-call request.
    #[serde(rename = "tool_use")]
    ToolUse {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        input: Option<serde_json::Value>,
    },
    /// User-supplied tool result. `content` may itself contain image blocks
    /// (the PR #1200 fix).
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        tool_use_id: Option<String>,
        #[serde(default)]
        content: Option<AnthropicToolResultContent>,
        #[serde(default)]
        name: Option<String>,
    },
    /// Extended-thinking block (dropped during inbound translation).
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: Option<String>,
    },
    /// Any other block type — tolerated and ignored.
    #[serde(other)]
    Unknown,
}

/// `tool_result.content` may be a bare string or an array of blocks
/// (text and/or image).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    /// Bare string content.
    Text(String),
    /// Array of nested blocks (text and image supported).
    Blocks(Vec<AnthropicToolResultBlock>),
}

/// A nested block inside `tool_result.content`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolResultBlock {
    /// Text block.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image block (PR #1200: images inside tool results).
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    /// Tolerated unknown block.
    #[serde(other)]
    Unknown,
}

/// Image source: a base64 payload or a URL.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicImageSource {
    /// Base64-encoded image data with a media type.
    #[serde(rename = "base64")]
    Base64 {
        #[serde(default)]
        media_type: Option<String>,
        data: String,
    },
    /// Remote image URL.
    #[serde(rename = "url")]
    Url { url: String },
    /// Tolerated unknown source type.
    #[serde(other)]
    Unknown,
}

impl AnthropicImageSource {
    /// Convert the source into an `image_url`-style reference understood by
    /// the internal media plumbing: a `data:<media_type>;base64,<data>` URI
    /// for base64 sources, or the bare URL for url sources. Returns `None`
    /// for unknown/empty sources.
    pub fn to_image_ref(&self) -> Option<String> {
        match self {
            AnthropicImageSource::Base64 { media_type, data } => {
                if data.is_empty() {
                    return None;
                }
                let media = media_type.as_deref().unwrap_or("image/png");
                Some(format!("data:{media};base64,{data}"))
            }
            AnthropicImageSource::Url { url } => {
                if url.is_empty() {
                    None
                } else {
                    Some(url.clone())
                }
            }
            AnthropicImageSource::Unknown => None,
        }
    }
}

/// Tool definition (Anthropic shape).
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    /// Tool name. Server tools (no `input_schema`) are dropped.
    #[serde(default)]
    pub name: Option<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the tool's input.
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

/// Tool-choice spec (Anthropic shape).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
    /// Bare string form (rare; tolerated).
    Mode(String),
    /// Object form: `{ "type": "auto" | "any" | "none" | "tool", ... }`.
    Spec(AnthropicToolChoiceSpec),
}

/// Object-form tool choice.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoiceSpec {
    #[serde(rename = "type")]
    pub choice_type: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// Extended-thinking configuration block.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicThinking {
    /// `enabled`, `adaptive`, or `disabled`.
    #[serde(rename = "type", default)]
    pub thinking_type: Option<String>,
    /// Token budget for thinking.
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_string_round_trips() {
        let sys: AnthropicSystem = serde_json::from_str("\"be terse\"").unwrap();
        assert_eq!(sys.to_text().as_deref(), Some("be terse"));
    }

    #[test]
    fn system_blocks_join_text() {
        let sys: AnthropicSystem =
            serde_json::from_str(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#)
                .unwrap();
        assert_eq!(sys.to_text().as_deref(), Some("a\nb"));
    }

    #[test]
    fn system_empty_is_none() {
        let sys: AnthropicSystem = serde_json::from_str("\"   \"").unwrap();
        assert_eq!(sys.to_text(), None);
    }

    #[test]
    fn base64_image_source_builds_data_uri() {
        let src = AnthropicImageSource::Base64 {
            media_type: Some("image/jpeg".to_string()),
            data: "QUJD".to_string(),
        };
        assert_eq!(
            src.to_image_ref().as_deref(),
            Some("data:image/jpeg;base64,QUJD")
        );
    }

    #[test]
    fn base64_image_default_media_type() {
        let src = AnthropicImageSource::Base64 {
            media_type: None,
            data: "QUJD".to_string(),
        };
        assert_eq!(
            src.to_image_ref().as_deref(),
            Some("data:image/png;base64,QUJD")
        );
    }

    #[test]
    fn url_image_source_passes_through() {
        let src = AnthropicImageSource::Url {
            url: "https://example.com/x.png".to_string(),
        };
        assert_eq!(
            src.to_image_ref().as_deref(),
            Some("https://example.com/x.png")
        );
    }

    #[test]
    fn parse_full_request_with_tool_result_image() {
        let body = r#"{
            "model": "m",
            "max_tokens": 64,
            "system": "sys",
            "stop_sequences": ["STOP"],
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_image", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                        {"type": "text", "text": "here"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
                    ]}
                ]}
            ],
            "tools": [
                {"name": "get_image", "description": "d", "input_schema": {"type": "object"}}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.model, "m");
        assert_eq!(req.max_tokens, Some(64));
        assert_eq!(
            req.stop_sequences.as_deref(),
            Some(&["STOP".to_string()][..])
        );
        assert_eq!(req.messages.len(), 3);
        assert!(req.tools.is_some());
    }

    #[test]
    fn unknown_content_block_is_tolerated() {
        let body = r#"{"role":"user","content":[{"type":"redacted_thinking","data":"x"}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(body).unwrap();
        match msg.content {
            AnthropicMessageContent::Blocks(b) => {
                assert!(matches!(b[0], AnthropicContentBlock::Unknown));
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn extra_top_level_fields_tolerated() {
        let body = r#"{"model":"m","messages":[],"metadata":{"user_id":"u"},"container":"c"}"#;
        let req: AnthropicRequest = serde_json::from_str(body).unwrap();
        assert!(req.extra.contains_key("metadata"));
        assert!(req.extra.contains_key("container"));
    }

    #[test]
    fn system_role_turn_inside_messages_deserializes() {
        // Regression for #349: Claude Code >= 2.1.156 sends `role:"system"`
        // turns inside `messages`. Without the `System` variant this fails the
        // `Json` extractor with HTTP 422 before any generation.
        let body = r#"{"model":"m","max_tokens":32,"messages":[
            {"role":"user","content":"hi"},
            {"role":"system","content":"be terse"},
            {"role":"user","content":"greet me"}
        ]}"#;
        let req: AnthropicRequest = serde_json::from_str(body).unwrap();
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, AnthropicRole::User);
        assert_eq!(req.messages[1].role, AnthropicRole::System);
        assert_eq!(req.messages[2].role, AnthropicRole::User);
        assert_eq!(AnthropicRole::System.as_str(), "system");
    }
}
