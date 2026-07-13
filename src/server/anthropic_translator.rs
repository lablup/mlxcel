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
            reasoning: None,
            tool_calls: None,
        });
    }

    // 2. Walk conversation turns.
    for message in &request.messages {
        append_message(&mut messages, message);
    }

    // 2b. Relocate any `system`-role turn that landed mid-conversation so its
    // text actually reaches the model regardless of the chat template.
    fold_system_messages(&mut messages);

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
        // Claude Code >= 2.1.156 interleaves `system` turns inside `messages`.
        // Map them to the internal `System` role here; `fold_system_messages`
        // then relocates any that are not at the head of the conversation.
        AnthropicRole::System => Role::System,
    };

    match &message.content {
        AnthropicMessageContent::Text(text) => {
            out.push(Message {
                role,
                content: MessageContent::Text(text.clone()),
                name: None,
                tool_call_id: None,
                reasoning: None,
                tool_calls: None,
            });
        }
        AnthropicMessageContent::Blocks(blocks) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut image_parts: Vec<ContentPart> = Vec::new();
            let mut tool_calls: Vec<ToolCallInMessage> = Vec::new();
            let mut tool_results: Vec<Message> = Vec::new();
            let mut reasoning_parts: Vec<String> = Vec::new();

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
                                image_url: ImageUrl::new(url),
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
                    AnthropicContentBlock::Thinking { thinking } => {
                        // Forward extended-thinking traces from assistant turns
                        // into the parallel `reasoning` field (issue #362) so
                        // templates that render `message.get('reasoning')` see
                        // prior thinking across turns. Thinking on non-assistant
                        // turns is meaningless and ignored.
                        if message.role == AnthropicRole::Assistant
                            && let Some(text) = thinking.as_ref()
                            && !text.is_empty()
                        {
                            reasoning_parts.push(text.clone());
                        }
                    }
                    AnthropicContentBlock::Unknown => {
                        // Dropped: unknown blocks do not feed back into the
                        // next prompt.
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
                let reasoning = if reasoning_parts.is_empty() {
                    None
                } else {
                    Some(reasoning_parts.join("\n"))
                };
                out.push(Message {
                    role,
                    content,
                    name: None,
                    tool_call_id: None,
                    reasoning,
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

/// Relocate `system`-role turns that landed mid-conversation so their text is
/// guaranteed to reach the model.
///
/// WHY this exists (issue #349): Claude Code >= 2.1.156 interleaves
/// `{"role":"system", ...}` reminders inside the `messages` array (the pattern
/// is `[user, system, user, ...]`). Mapping the role is enough to stop the 422,
/// but it does NOT guarantee the text renders: many production chat templates
/// (Qwen-family, Llama 3, ...) special-case `messages[0]` for the system block
/// and their `{% for message in messages %}` loop only handles `user` /
/// `assistant` / `tool`. A `system` turn at `messages[1]` is then SILENTLY
/// DROPPED, so the reminder never influences generation.
///
/// Strategy, chosen to render correctly under any template:
///   * Leading `system` turns (before the first user/assistant/tool turn) are
///     merged into the single head system block — together with the top-level
///     `system` field when present, so neither is dropped.
///   * A mid-conversation `system` turn is folded (prepended) into the FOLLOWING
///     user turn. Claude Code emits the reminder right before a user turn, so
///     this keeps its positional/contextual meaning, and folding into an
///     existing turn (rather than emitting a fresh `system` message) means we
///     never rely on a template rendering a non-head `system` role. No new
///     `user` turn is created, so user/assistant alternation is unchanged.
///   * A trailing `system` turn with no following user turn is appended to the
///     preceding user turn, or merged into the head block as a last resort.
///
/// The pass is a no-op (early return) for the common case of a single head
/// system block produced by the top-level `system` field.
fn fold_system_messages(messages: &mut Vec<Message>) {
    // Action is needed only when a `system` message exists somewhere other than
    // the head (index 0). A lone head system block is left exactly as-is.
    let needs_fold = messages
        .iter()
        .enumerate()
        .any(|(i, m)| i != 0 && m.role == Role::System);
    if !needs_fold {
        return;
    }

    let original = std::mem::take(messages);
    let mut head_system: Option<String> = None;
    let mut body: Vec<Message> = Vec::with_capacity(original.len());
    let mut pending_system: Vec<String> = Vec::new();
    let mut seen_non_system = false;

    for message in original {
        match message.role {
            Role::System if !seen_non_system => {
                // Head block: top-level system and/or leading system turns.
                merge_head_system(&mut head_system, message.content.text());
            }
            Role::System => {
                // Mid-conversation reminder: buffer for the next user turn.
                let text = message.content.text();
                if !text.trim().is_empty() {
                    pending_system.push(text);
                }
            }
            Role::User => {
                seen_non_system = true;
                let mut message = message;
                if !pending_system.is_empty() {
                    combine_user_text(&mut message, &pending_system.join("\n\n"), true);
                    pending_system.clear();
                }
                body.push(message);
            }
            _ => {
                seen_non_system = true;
                body.push(message);
            }
        }
    }

    // Trailing reminders with no following user turn: append to the preceding
    // user turn when possible, otherwise fall back to the head block.
    if !pending_system.is_empty() {
        let preamble = pending_system.join("\n\n");
        match body.last_mut() {
            Some(last) if last.role == Role::User => combine_user_text(last, &preamble, false),
            _ => merge_head_system(&mut head_system, preamble),
        }
    }

    if let Some(text) = head_system {
        messages.push(Message {
            role: Role::System,
            content: MessageContent::Text(text),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        });
    }
    messages.extend(body);
}

/// Merge `text` into the accumulating head system block, joining with a blank
/// line. Empty/whitespace-only text is ignored.
fn merge_head_system(head: &mut Option<String>, text: String) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    match head {
        Some(existing) => {
            existing.push_str("\n\n");
            existing.push_str(trimmed);
        }
        None => *head = Some(trimmed.to_string()),
    }
}

/// Fold `addition` into a user message's text, keeping any image/other parts.
/// When `prepend` is true the addition leads (a reminder that precedes the
/// user's words); otherwise it trails.
fn combine_user_text(message: &mut Message, addition: &str, prepend: bool) {
    if addition.is_empty() {
        return;
    }
    let content = std::mem::take(&mut message.content);
    message.content = match content {
        MessageContent::Text(existing) => {
            if existing.trim().is_empty() {
                MessageContent::Text(addition.to_string())
            } else if prepend {
                MessageContent::Text(format!("{addition}\n\n{existing}"))
            } else {
                MessageContent::Text(format!("{existing}\n\n{addition}"))
            }
        }
        MessageContent::Parts(mut parts) => {
            // Merge into the existing text part so the reminder is separated
            // from the user's words with a blank line, matching the Text branch
            // (a bare extra text part would be concatenated with no separator,
            // since `MessageContent::text()` joins parts with an empty string).
            // Fall back to a fresh text part only when the turn is image-only.
            match parts
                .iter_mut()
                .find(|p| matches!(p, ContentPart::Text { .. }))
            {
                Some(ContentPart::Text { text }) => {
                    let existing = std::mem::take(text);
                    *text = if existing.trim().is_empty() {
                        addition.to_string()
                    } else if prepend {
                        format!("{addition}\n\n{existing}")
                    } else {
                        format!("{existing}\n\n{addition}")
                    };
                }
                _ => {
                    let part = ContentPart::Text {
                        text: addition.to_string(),
                    };
                    if prepend {
                        parts.insert(0, part);
                    } else {
                        parts.push(part);
                    }
                }
            }
            MessageContent::Parts(parts)
        }
    };
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
                image_url: ImageUrl::new(url),
            });
        }
        MessageContent::Parts(parts)
    };

    Message {
        role: Role::Tool,
        content: message_content,
        name: None,
        tool_call_id: tool_use_id.map(|s| s.to_string()),
        reasoning: None,
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
    fn assistant_thinking_block_becomes_reasoning_field() {
        // Issue #362: an Anthropic assistant `thinking` block is forwarded onto
        // the internal message's parallel `reasoning` field for cross-API
        // consistency, instead of being dropped.
        let req = parse_req(
            r#"{"model":"m","messages":[{"role":"assistant","content":[
                {"type":"thinking","thinking":"step by step"},
                {"type":"text","text":"the answer"}
            ]}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msg = &t.chat_request.messages[0];
        assert!(matches!(msg.role, Role::Assistant));
        assert_eq!(msg.content.text(), "the answer");
        assert_eq!(msg.reasoning.as_deref(), Some("step by step"));
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

    /// Render internal messages through a Qwen-style head-only template: the
    /// system block is only emitted for `messages[0]`, and the per-turn loop
    /// handles user/assistant exclusively. Any `system` turn left mid-array is
    /// SILENTLY DROPPED by this template, so a substring assertion on its output
    /// proves the reminder was actually folded into a rendered turn (issue #349).
    fn render_head_only(messages: &[Message]) -> String {
        use crate::server::chat_template::{ChatMessage, ChatTemplateProcessor};
        let template = r#"{%- if messages[0].role == 'system' -%}
SYS:{{ messages[0].content }}
{% endif -%}
{%- for m in messages -%}
{%- if m.role == 'user' -%}
U:{{ m.content }}
{% elif m.role == 'assistant' -%}
A:{{ m.content }}
{% endif -%}
{%- endfor -%}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let chat: Vec<ChatMessage> = messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.as_str().to_string(),
                content: m.content.text(),
            })
            .collect();
        processor
            .apply(&chat, None)
            .expect("head-only template must render")
    }

    #[test]
    fn system_role_turn_in_messages_deserializes_no_422() {
        // The Claude Code >= 2.1.156 shape parses instead of 422-ing.
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"messages":[
                {"role":"user","content":"hi"},
                {"role":"system","content":"be terse"},
                {"role":"user","content":"greet me"}
            ]}"#,
        );
        assert_eq!(req.messages.len(), 3);
    }

    #[test]
    fn mid_conversation_system_turn_folds_into_following_user() {
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"messages":[
                {"role":"user","content":"hi"},
                {"role":"system","content":"be terse"},
                {"role":"user","content":"greet me"}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        // No orphan mid-array system turn survives (a head-only template would
        // drop it); the reminder rode into the following user turn.
        assert!(msgs.iter().all(|m| m.role != Role::System));
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, Role::User));
        assert_eq!(msgs[0].content.text(), "hi");
        assert!(matches!(msgs[1].role, Role::User));
        assert_eq!(msgs[1].content.text(), "be terse\n\ngreet me");
        // Empirical proof: the system text survives a head-only Qwen-style
        // template render.
        let rendered = render_head_only(msgs);
        assert!(
            rendered.contains("be terse"),
            "system reminder must reach the prompt: {rendered}"
        );
    }

    #[test]
    fn top_level_system_and_mid_array_system_both_survive() {
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"system":"global rule","messages":[
                {"role":"user","content":"hi"},
                {"role":"system","content":"mid reminder"},
                {"role":"user","content":"go"}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        // Exactly one system block, at the head, carrying the top-level system.
        assert!(matches!(msgs[0].role, Role::System));
        assert_eq!(msgs[0].content.text(), "global rule");
        assert_eq!(
            msgs.iter().filter(|m| m.role == Role::System).count(),
            1,
            "no second head system block (some templates only read messages[0])"
        );
        // The mid-array reminder folded into the trailing user turn.
        let user_text: String = msgs
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.text())
            .collect::<Vec<_>>()
            .join("|");
        assert!(user_text.contains("mid reminder"));
        // Both reach the prompt under a head-only template.
        let rendered = render_head_only(msgs);
        assert!(rendered.contains("global rule"), "{rendered}");
        assert!(rendered.contains("mid reminder"), "{rendered}");
    }

    #[test]
    fn leading_array_system_merges_with_top_level_system() {
        // A `system` turn at messages[0] alongside a top-level `system` must
        // merge into one head block, dropping neither.
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"system":"top","messages":[
                {"role":"system","content":"lead"},
                {"role":"user","content":"hi"}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        assert_eq!(msgs.iter().filter(|m| m.role == Role::System).count(), 1);
        assert!(matches!(msgs[0].role, Role::System));
        let head = msgs[0].content.text();
        assert!(head.contains("top"), "{head}");
        assert!(head.contains("lead"), "{head}");
        assert!(matches!(msgs[1].role, Role::User));
        assert_eq!(msgs[1].content.text(), "hi");
    }

    #[test]
    fn trailing_system_turn_with_no_following_user_is_preserved() {
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"messages":[
                {"role":"user","content":"hi"},
                {"role":"system","content":"bye reminder"}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        assert!(msgs.iter().all(|m| m.role != Role::System));
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0].role, Role::User));
        assert_eq!(msgs[0].content.text(), "hi\n\nbye reminder");
        assert!(render_head_only(msgs).contains("bye reminder"));
    }

    #[test]
    fn mid_array_system_folds_into_multimodal_user_turn() {
        // The following user turn carries an image; the reminder must not clobber
        // the image part and must still surface as text.
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"messages":[
                {"role":"user","content":"first"},
                {"role":"system","content":"watch out"},
                {"role":"user","content":[
                    {"type":"text","text":"see this"},
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}}
                ]}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        assert!(msgs.iter().all(|m| m.role != Role::System));
        // Image preserved through the fold.
        assert_eq!(t.chat_request.image_urls().len(), 1);
        let folded = msgs.last().unwrap();
        // Reminder leads the user's words, separated by a blank line (not
        // concatenated): `text()` joins parts with an empty string, so a bare
        // extra text part would render "watch outsee this".
        assert_eq!(folded.content.text(), "watch out\n\nsee this");
        // Folded turn keeps exactly one text part plus the image part.
        match &folded.content {
            MessageContent::Parts(parts) => assert_eq!(parts.len(), 2),
            other => panic!("expected multimodal parts, got {other:?}"),
        }
    }

    #[test]
    fn consecutive_mid_array_system_turns_all_fold() {
        // Multiple back-to-back system reminders between user turns must all
        // survive, joined together, folded into the following user turn.
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"messages":[
                {"role":"user","content":"hi"},
                {"role":"system","content":"rule one"},
                {"role":"system","content":"rule two"},
                {"role":"user","content":"go"}
            ]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        let msgs = &t.chat_request.messages;
        assert!(msgs.iter().all(|m| m.role != Role::System));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].content.text(), "rule one\n\nrule two\n\ngo");
        let rendered = render_head_only(msgs);
        assert!(rendered.contains("rule one"), "{rendered}");
        assert!(rendered.contains("rule two"), "{rendered}");
    }

    #[test]
    fn system_string_only_path_unchanged_by_fold() {
        // Regression guard: the plain top-level-system case is a no-op for the
        // fold pass and keeps its exact prior shape.
        let req = parse_req(
            r#"{"model":"m","max_tokens":8,"system":"be terse","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let t = anthropic_request_to_chat(&req);
        assert_eq!(t.chat_request.messages.len(), 2);
        assert!(matches!(t.chat_request.messages[0].role, Role::System));
        assert_eq!(t.chat_request.messages[0].content.text(), "be terse");
        assert!(matches!(t.chat_request.messages[1].role, Role::User));
        assert_eq!(t.chat_request.messages[1].content.text(), "hi");
    }
}
