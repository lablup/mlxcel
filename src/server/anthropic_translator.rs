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

//! Translator between the Anthropic Messages API and the internal
//! chat-completions request/response pipeline.
//!
//! Ported from upstream mlx-vlm `server/anthropic.py` (PR #1196, commit
//! `313ad22`) and the tool-call image-handling fix (PR #1200, commit
//! `f2e19de`).
//!
//! ## Inbound
//!
//! [`anthropic_request_to_chat`] flattens an [`AnthropicRequest`] into the
//! [`ChatCompletionRequest`] shape consumed by
//! [`crate::server::chat_request::prepare_chat_request_with_cache`]:
//!
//! 1. Render `system` (string or text-block array) as a leading system turn.
//! 2. Walk each message's content blocks:
//!    - `text` → message text
//!    - `image` (user role) → an `image_url` content part (media plumbing)
//!    - `tool_use` (assistant role) → `tool_calls` on the assistant turn
//!    - `tool_result` (user role) → a `role: tool` message; **image blocks
//!      inside the tool result become `image_url` parts** (the #1200 fix)
//!    - `thinking` / unknown → dropped
//! 3. Convert `tools` / `tool_choice` into the chat-request fields. Server
//!    tools (no `input_schema`) are dropped because the local server cannot
//!    execute them.
//!
//! ## Outbound
//!
//! [`build_content_blocks`] turns the generation result (visible text,
//! optional reasoning, parsed tool calls) into the ordered Anthropic
//! response `content[]`. [`anthropic_stop_reason`] and
//! [`apply_stop_sequences`] mirror the upstream stop-reason mapping.

use crate::server::tool_calls::types::ParsedToolCall;
use crate::server::types::anthropic_request::{
    AnthropicContentBlock, AnthropicMessage, AnthropicMessageContent, AnthropicRequest,
    AnthropicRole, AnthropicToolChoice, AnthropicToolResultBlock, AnthropicToolResultContent,
};
use crate::server::types::anthropic_response::AnthropicResponseBlock;
use crate::server::types::request::{
    ChatCompletionRequest, ContentPart, FunctionDefinition, ImageUrl, Message, MessageContent,
    Role, SamplingParams, Tool, ToolCallFunction, ToolCallInMessage, ToolChoice,
    ToolChoiceFunction, ToolChoiceFunctionName,
};

/// Generate a short hex id (16 chars) for synthetic tool-call ids.
pub fn short_uuid() -> String {
    let full = uuid::Uuid::new_v4().simple().to_string();
    full[..16].to_string()
}

/// Result of flattening an Anthropic request into the internal chat shape.
#[derive(Debug)]
pub struct AnthropicTranslated {
    /// Synthetic chat request driving the generation pipeline.
    pub chat_request: ChatCompletionRequest,
}

/// Flatten an [`AnthropicRequest`] into a [`ChatCompletionRequest`].
///
/// The route layer applies `state.config.default_max_tokens` when the request
/// omits `max_tokens`; this function performs only the structural translation.
pub fn anthropic_request_to_chat(request: &AnthropicRequest) -> AnthropicTranslated {
    let mut messages: Vec<Message> = Vec::new();

    // 1. System prompt (string or text-block array) → leading system turn.
    if let Some(system) = request.system.as_ref()
        && let Some(text) = system.to_text()
    {
        messages.push(Message {
            role: Role::System,
            content: MessageContent::Text(text),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        });
    }

    // 2. Walk conversation turns.
    for message in &request.messages {
        append_message(&mut messages, message);
    }

    // 3. Tools / tool_choice.
    let tools = convert_tools(request);
    let tool_choice = convert_tool_choice(request.tool_choice.as_ref());

    // 4. Sampling params.
    //
    // Anthropic's extended-thinking `budget_tokens` maps onto the internal
    // `thinking_budget_tokens` alias so the shared budget resolution
    // (`thinking_budget::pick_budget_alias` / `resolve_request_budget`) used
    // by the chat and responses routes applies uniformly.
    let thinking_budget_tokens = request
        .thinking
        .as_ref()
        .and_then(|t| t.budget_tokens)
        .and_then(|b| i32::try_from(b).ok());
    let params = SamplingParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        // Anthropic `stop_sequences` are applied as native stop sequences so
        // the scheduler halts generation early; the outbound assembly also
        // truncates defensively via `apply_stop_sequences`.
        stop: request.stop_sequences.clone(),
        thinking_budget_tokens,
        ..Default::default()
    };

    let chat_request = ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        stream: request.stream,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools,
        tool_choice,
        parallel_tool_calls: None,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: None,
        // Map the Anthropic `metadata.user_id` (stored untyped in `extra`) to
        // the chat request's `user` so per-user prompt-cache isolation works on
        // the Messages endpoint. Absent it, the session key falls back to the
        // shared anonymous bucket, which still yields within-endpoint reuse.
        user: request
            .extra
            .get("metadata")
            .and_then(|m| m.get("user_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        extra_body_fields: serde_json::Map::new(),
        response_format: None,
        params,
    };

    AnthropicTranslated { chat_request }
}

/// Append a single Anthropic message (and any tool-result turns it implies)
/// to the running internal message list.
fn append_message(out: &mut Vec<Message>, message: &AnthropicMessage) {
    let role = match message.role {
        AnthropicRole::User => Role::User,
        AnthropicRole::Assistant => Role::Assistant,
    };

    match &message.content {
        AnthropicMessageContent::Text(text) => {
            out.push(Message {
                role,
                content: MessageContent::Text(text.clone()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            });
        }
        AnthropicMessageContent::Blocks(blocks) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut image_parts: Vec<ContentPart> = Vec::new();
            let mut tool_calls: Vec<ToolCallInMessage> = Vec::new();
            let mut tool_results: Vec<Message> = Vec::new();

            for block in blocks {
                match block {
                    AnthropicContentBlock::Text { text } => {
                        if !text.is_empty() {
                            text_parts.push(text.clone());
                        }
                    }
                    AnthropicContentBlock::Document { source } => {
                        if let Some(text) = document_text(source.as_ref()) {
                            text_parts.push(text);
                        }
                    }
                    AnthropicContentBlock::Image { source } => {
                        // Images only carry meaning on user turns.
                        if message.role == AnthropicRole::User
                            && let Some(url) = source.to_image_ref()
                        {
                            image_parts.push(ContentPart::ImageUrl {
                                image_url: ImageUrl { url },
                            });
                        }
                    }
                    AnthropicContentBlock::ToolUse { id, name, input } => {
                        if message.role == AnthropicRole::Assistant {
                            tool_calls.push(tool_use_to_call(
                                id.as_deref(),
                                name.as_deref(),
                                input.as_ref(),
                            ));
                        }
                    }
                    AnthropicContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        if message.role == AnthropicRole::User {
                            tool_results.push(tool_result_message(
                                tool_use_id.as_deref(),
                                content.as_ref(),
                            ));
                        }
                    }
                    AnthropicContentBlock::Thinking { .. } | AnthropicContentBlock::Unknown => {
                        // Dropped: thinking traces and unknown blocks do not
                        // feed back into the next prompt.
                    }
                }
            }

            let joined_text = text_parts.join("\n");
            let has_text = !joined_text.trim().is_empty();

            // Emit the primary turn when it carries text, tool calls, or
            // images, OR when there are no tool results (mirror upstream:
            // a message that is *only* tool_results does not emit an empty
            // assistant/user turn). This keeps an empty user turn from being
            // injected ahead of the tool result messages.
            let has_images = !image_parts.is_empty();
            if has_text || !tool_calls.is_empty() || has_images || tool_results.is_empty() {
                let content = if has_images {
                    // Multimodal: text first (if any), then image parts.
                    let mut parts: Vec<ContentPart> = Vec::new();
                    if has_text {
                        parts.push(ContentPart::Text { text: joined_text });
                    }
                    parts.extend(image_parts);
                    MessageContent::Parts(parts)
                } else {
                    MessageContent::Text(joined_text)
                };
                out.push(Message {
                    role,
                    content,
                    name: None,
                    tool_call_id: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                });
            }

            out.extend(tool_results);
        }
    }
}

/// Convert an Anthropic `tool_use` block into an internal assistant
/// `tool_calls` entry.
fn tool_use_to_call(
    id: Option<&str>,
    name: Option<&str>,
    input: Option<&serde_json::Value>,
) -> ToolCallInMessage {
    let arguments = match input {
        Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    };
    ToolCallInMessage {
        id: id
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("toolu_{}", short_uuid())),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: name.unwrap_or("").to_string(),
            arguments,
        },
    }
}

/// Convert an Anthropic `tool_result` block into a `role: tool` message.
///
/// Per PR #1200, image blocks inside the tool result are surfaced as
/// `image_url` content parts so they flow through the media plumbing; text
/// blocks are concatenated. When no images are present, the content is a
/// plain string (the common case).
fn tool_result_message(
    tool_use_id: Option<&str>,
    content: Option<&AnthropicToolResultContent>,
) -> Message {
    let (text, images) = tool_result_to_text_and_images(content);

    let message_content = if images.is_empty() {
        MessageContent::Text(text)
    } else {
        let mut parts: Vec<ContentPart> = Vec::new();
        if !text.is_empty() {
            parts.push(ContentPart::Text { text });
        }
        for url in images {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl { url },
            });
        }
        MessageContent::Parts(parts)
    };

    Message {
        role: Role::Tool,
        content: message_content,
        name: None,
        tool_call_id: tool_use_id.map(|s| s.to_string()),
        tool_calls: None,
    }
}

/// Flatten `tool_result.content` into `(joined_text, image_refs)`.
fn tool_result_to_text_and_images(
    content: Option<&AnthropicToolResultContent>,
) -> (String, Vec<String>) {
    match content {
        None => (String::new(), Vec::new()),
        Some(AnthropicToolResultContent::Text(s)) => (s.clone(), Vec::new()),
        Some(AnthropicToolResultContent::Blocks(blocks)) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut images: Vec<String> = Vec::new();
            for block in blocks {
                match block {
                    AnthropicToolResultBlock::Text { text } => {
                        if !text.is_empty() {
                            text_parts.push(text.clone());
                        }
                    }
                    AnthropicToolResultBlock::Image { source } => {
                        if let Some(url) = source.to_image_ref() {
                            images.push(url);
                        }
                    }
                    AnthropicToolResultBlock::Unknown => {}
                }
            }
            (text_parts.join("\n"), images)
        }
    }
}

/// Extract text from a `document` block's `source` when it is a text source.
fn document_text(source: Option<&serde_json::Value>) -> Option<String> {
    let source = source?;
    let obj = source.as_object()?;
    if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
        obj.get("data")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
    } else {
        None
    }
}

/// Convert Anthropic tools into internal `function` tools, dropping server
/// tools (those without an `input_schema`).
fn convert_tools(request: &AnthropicRequest) -> Option<Vec<Tool>> {
    let tools = request.tools.as_ref()?;
    let out: Vec<Tool> = tools
        .iter()
        .filter_map(|t| {
            let name = t.name.as_ref()?;
            let input_schema = t.input_schema.as_ref()?;
            if name.is_empty() {
                return None;
            }
            Some(Tool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: name.clone(),
                    description: t.description.clone(),
                    parameters: Some(input_schema.clone()),
                },
            })
        })
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

/// Map Anthropic `tool_choice` onto the internal [`ToolChoice`].
fn convert_tool_choice(choice: Option<&AnthropicToolChoice>) -> Option<ToolChoice> {
    let choice = choice?;
    match choice {
        AnthropicToolChoice::Mode(s) => Some(ToolChoice::Mode(s.clone())),
        AnthropicToolChoice::Spec(spec) => match spec.choice_type.as_str() {
            "auto" => Some(ToolChoice::Mode("auto".to_string())),
            "none" => Some(ToolChoice::Mode("none".to_string())),
            "any" => Some(ToolChoice::Mode("required".to_string())),
            "tool" => spec.name.as_ref().map(|name| {
                ToolChoice::Specific(ToolChoiceFunction {
                    choice_type: "function".to_string(),
                    function: ToolChoiceFunctionName { name: name.clone() },
                })
            }),
            _ => None,
        },
    }
}

/// Map an internal finish reason / tool-call presence / stop-sequence match
/// onto an Anthropic `stop_reason`.
pub fn anthropic_stop_reason(
    finish_reason: &str,
    tool_calls: bool,
    stop_sequence: Option<&str>,
) -> String {
    if tool_calls {
        return "tool_use".to_string();
    }
    if stop_sequence.is_some() {
        return "stop_sequence".to_string();
    }
    match finish_reason {
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        _ => "end_turn".to_string(),
    }
}

/// Truncate `text` at the first matching stop sequence. Returns the
/// truncated text and the matched sequence (if any), mirroring upstream
/// `_apply_stop_sequences`.
pub fn apply_stop_sequences(
    text: &str,
    stop_sequences: Option<&[String]>,
) -> (String, Option<String>) {
    let Some(seqs) = stop_sequences else {
        return (text.to_string(), None);
    };
    if text.is_empty() || seqs.is_empty() {
        return (text.to_string(), None);
    }
    let mut best_index: Option<usize> = None;
    let mut best_sequence: Option<&str> = None;
    for seq in seqs {
        if seq.is_empty() {
            continue;
        }
        if let Some(idx) = text.find(seq.as_str())
            && best_index.map(|b| idx < b).unwrap_or(true)
        {
            best_index = Some(idx);
            best_sequence = Some(seq.as_str());
        }
    }
    match best_index {
        Some(idx) => (
            text[..idx].to_string(),
            best_sequence.map(|s| s.to_string()),
        ),
        None => (text.to_string(), None),
    }
}

/// Assemble the ordered Anthropic response `content[]` from a generation
/// result: optional thinking block, visible text block, then tool_use
/// blocks. Always returns at least one block (an empty text block when the
/// model produced nothing) to satisfy the Anthropic schema.
pub fn build_content_blocks(
    visible_text: &str,
    reasoning_text: Option<&str>,
    parsed_tool_calls: Option<&[ParsedToolCall]>,
    include_thinking: bool,
) -> Vec<AnthropicResponseBlock> {
    let mut blocks: Vec<AnthropicResponseBlock> = Vec::new();

    if include_thinking
        && let Some(reasoning) = reasoning_text
        && !reasoning.is_empty()
    {
        blocks.push(AnthropicResponseBlock::Thinking {
            thinking: reasoning.to_string(),
            signature: String::new(),
        });
    }

    if !visible_text.is_empty() {
        blocks.push(AnthropicResponseBlock::Text {
            text: visible_text.to_string(),
        });
    }

    if let Some(calls) = parsed_tool_calls {
        for call in calls {
            blocks.push(parsed_call_to_tool_use(call));
        }
    }

    if blocks.is_empty() {
        blocks.push(AnthropicResponseBlock::Text {
            text: String::new(),
        });
    }

    blocks
}

/// Convert an internal [`ParsedToolCall`] into an Anthropic `tool_use`
/// response block. The stringified arguments are parsed back into a JSON
/// object; a parse failure yields an empty object (never a crash).
pub fn parsed_call_to_tool_use(call: &ParsedToolCall) -> AnthropicResponseBlock {
    let input = if call.arguments.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str::<serde_json::Value>(&call.arguments)
            .ok()
            .filter(|v| v.is_object())
            .unwrap_or_else(|| serde_json::json!({}))
    };
    AnthropicResponseBlock::ToolUse {
        id: format!("toolu_{}", short_uuid()),
        name: call.name.clone(),
        input,
    }
}

/// Whether the request's `thinking` config enables extended thinking output.
pub fn thinking_enabled(request: &AnthropicRequest) -> bool {
    request
        .thinking
        .as_ref()
        .and_then(|t| t.thinking_type.as_deref())
        .map(|t| matches!(t, "enabled" | "adaptive"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::anthropic_request::AnthropicRequest;

    fn parse_req(body: &str) -> AnthropicRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn system_string_becomes_leading_system_message() {
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"system":"be terse","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.messages.len(), 2);
        assert!(matches!(t.chat_request.messages[0].role, Role::System));
        assert_eq!(t.chat_request.messages[0].content.text(), "be terse");
        assert!(matches!(t.chat_request.messages[1].role, Role::User));
        assert_eq!(t.chat_request.messages[1].content.text(), "hi");
        assert_eq!(t.chat_request.params.max_tokens, Some(8));
    }

    #[test]
    fn user_image_block_becomes_image_url_part() {
        let req = parse_req(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}}
            ]}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let urls = t.chat_request.image_urls();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "data:image/png;base64,QUJD");
        // Text preserved alongside image.
        assert_eq!(t.chat_request.messages[0].content.text(), "look");
    }

    #[test]
    fn assistant_tool_use_becomes_tool_calls() {
        let req = parse_req(
            r#"{"model":"m","messages":[{"role":"assistant","content":[
                {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}
            ]}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msg = &t.chat_request.messages[0];
        assert!(matches!(msg.role, Role::Assistant));
        let calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"SF"}"#);
    }

    #[test]
    fn tool_result_text_becomes_tool_message() {
        let req = parse_req(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}
            ]}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        // Only the tool message — no empty user turn ahead of it.
        assert_eq!(t.chat_request.messages.len(), 1);
        let msg = &t.chat_request.messages[0];
        assert!(matches!(msg.role, Role::Tool));
        assert_eq!(msg.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(msg.content.text(), "sunny");
    }

    #[test]
    fn tool_result_with_image_surfaces_image_url_part() {
        // PR #1200: image content inside a tool result must flow through.
        let req = parse_req(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":[
                    {"type":"text","text":"chart"},
                    {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"WFla"}}
                ]}
            ]}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.messages.len(), 1);
        let msg = &t.chat_request.messages[0];
        assert!(matches!(msg.role, Role::Tool));
        assert_eq!(msg.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(msg.content.text(), "chart");
        let urls = msg.content.image_urls();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "data:image/jpeg;base64,WFla");
        // And the whole-request image extractor sees it too.
        assert_eq!(t.chat_request.image_urls().len(), 1);
    }

    #[test]
    fn tools_convert_and_drop_server_tools() {
        let req = parse_req(
            r#"{"model":"m","messages":[],"tools":[
                {"name":"f","description":"d","input_schema":{"type":"object"}},
                {"name":"web_search"},
                {"type":"web_search_20250305","name":"web_search","max_uses":3}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let tools = t.chat_request.tools.as_ref().unwrap();
        // Only the function tool with input_schema survives.
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "f");
        assert!(tools[0].function.parameters.is_some());
    }

    #[test]
    fn all_valid_function_tools_preserved_for_max_tools_guard() {
        // The route-layer MAX_TOOLS guard inspects the converted tool count,
        // so the translator must preserve every valid function tool.
        let tools: Vec<String> = (0..200)
            .map(|i| {
                format!(r#"{{"name":"f{i}","description":"d","input_schema":{{"type":"object"}}}}"#)
            })
            .collect();
        let body = format!(
            r#"{{"model":"m","messages":[],"tools":[{}]}}"#,
            tools.join(",")
        );
        let req = parse_req(&body);
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.tools.as_ref().unwrap().len(), 200);
    }

    #[test]
    fn tool_choice_any_maps_to_required() {
        let req = parse_req(r#"{"model":"m","messages":[],"tool_choice":{"type":"any"}}"#);
        let t = anthropic_request_to_chat(&req);
        assert_eq!(
            t.chat_request.tool_choice.as_ref().unwrap().mode(),
            "required"
        );
    }

    #[test]
    fn tool_choice_named_tool_maps_to_specific() {
        let req =
            parse_req(r#"{"model":"m","messages":[],"tool_choice":{"type":"tool","name":"f"}}"#);
        let t = anthropic_request_to_chat(&req);
        let tc = t.chat_request.tool_choice.as_ref().unwrap();
        assert_eq!(tc.specific_function(), Some("f"));
    }

    #[test]
    fn stop_sequences_propagate_to_sampling_params() {
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"stop_sequences":["END","STOP"],"messages":[{"role":"user","content":"x"}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        assert_eq!(
            t.chat_request.params.stop.as_deref(),
            Some(&["END".to_string(), "STOP".to_string()][..])
        );
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(anthropic_stop_reason("stop", false, None), "end_turn");
        assert_eq!(anthropic_stop_reason("length", false, None), "max_tokens");
        assert_eq!(anthropic_stop_reason("tool_calls", false, None), "tool_use");
        assert_eq!(anthropic_stop_reason("stop", true, None), "tool_use");
        assert_eq!(
            anthropic_stop_reason("stop", false, Some("END")),
            "stop_sequence"
        );
        // tool_use wins over stop_sequence.
        assert_eq!(anthropic_stop_reason("stop", true, Some("END")), "tool_use");
    }

    #[test]
    fn apply_stop_sequences_truncates_at_first_match() {
        let (text, seq) = apply_stop_sequences(
            "hello END world STOP",
            Some(&["STOP".to_string(), "END".to_string()]),
        );
        assert_eq!(text, "hello ");
        assert_eq!(seq.as_deref(), Some("END"));
    }

    #[test]
    fn apply_stop_sequences_no_match_returns_input() {
        let (text, seq) = apply_stop_sequences("hello", Some(&["X".to_string()]));
        assert_eq!(text, "hello");
        assert_eq!(seq, None);
    }

    #[test]
    fn content_blocks_text_only() {
        let blocks = build_content_blocks("hi", None, None, false);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], AnthropicResponseBlock::Text { text } if text == "hi"));
    }

    #[test]
    fn content_blocks_empty_yields_empty_text() {
        let blocks = build_content_blocks("", None, None, false);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], AnthropicResponseBlock::Text { text } if text.is_empty()));
    }

    #[test]
    fn content_blocks_thinking_then_text() {
        let blocks = build_content_blocks("answer", Some("reasoning"), None, true);
        assert_eq!(blocks.len(), 2);
        assert!(
            matches!(&blocks[0], AnthropicResponseBlock::Thinking { thinking, .. } if thinking == "reasoning")
        );
        assert!(matches!(&blocks[1], AnthropicResponseBlock::Text { text } if text == "answer"));
    }

    #[test]
    fn content_blocks_thinking_suppressed_when_disabled() {
        let blocks = build_content_blocks("answer", Some("reasoning"), None, false);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], AnthropicResponseBlock::Text { .. }));
    }

    #[test]
    fn content_blocks_with_tool_calls() {
        let calls = vec![ParsedToolCall {
            name: "f".to_string(),
            arguments: r#"{"a":1}"#.to_string(),
        }];
        let blocks = build_content_blocks("", None, Some(&calls), false);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            AnthropicResponseBlock::ToolUse { name, input, id } => {
                assert_eq!(name, "f");
                assert_eq!(input["a"], 1);
                assert!(id.starts_with("toolu_"));
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn parsed_call_invalid_args_become_empty_object() {
        let call = ParsedToolCall {
            name: "f".to_string(),
            arguments: "not json".to_string(),
        };
        let block = parsed_call_to_tool_use(&call);
        match block {
            AnthropicResponseBlock::ToolUse { input, .. } => {
                assert!(input.is_object());
                assert_eq!(input.as_object().unwrap().len(), 0);
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn thinking_enabled_detection() {
        let req = parse_req(
            r#"{"model":"m","messages":[],"thinking":{"type":"enabled","budget_tokens":1024}}"#,
        );
        assert!(thinking_enabled(&req));
        let req2 = parse_req(r#"{"model":"m","messages":[],"thinking":{"type":"disabled"}}"#);
        assert!(!thinking_enabled(&req2));
        let req3 = parse_req(r#"{"model":"m","messages":[]}"#);
        assert!(!thinking_enabled(&req3));
    }

    #[test]
    fn metadata_user_id_maps_to_chat_user() {
        let req = parse_req(r#"{"model":"m","messages":[],"metadata":{"user_id":"u"}}"#);
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.user.as_deref(), Some("u"));
    }

    #[test]
    fn missing_metadata_user_id_leaves_chat_user_none() {
        let req = parse_req(r#"{"model":"m","messages":[]}"#);
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.user, None);
        // metadata present but without user_id also yields None.
        let req2 = parse_req(r#"{"model":"m","messages":[],"metadata":{"other":"x"}}"#);
        let t2 = anthropic_request_to_chat(&req2);
        assert_eq!(t2.chat_request.user, None);
    }
}
