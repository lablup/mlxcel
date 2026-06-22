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

//! Bidirectional translator between the OpenAI Responses API and the
//! internal chat-completions request/response pipeline.
//!
//! ## Inbound
//!
//! [`responses_request_to_chat`] flattens a [`CreateResponseRequest`]
//! into the [`ChatCompletionRequest`] shape consumed by
//! [`crate::server::chat_request::prepare_chat_request_with_cache`]:
//!
//! 1. Resolve prior history. If `previous_response_id` is set, load the
//!    stored response and rehydrate its input + output as a transcript.
//!    Otherwise, if `conversation` is set, fetch the existing transcript.
//! 2. Render `instructions` (when present) as a leading system message.
//!    Per the OpenAI rule, instructions from a chained-from response are
//!    **not** carried — only the current request's instructions apply.
//! 3. Append the current request's inputs (string or item array).
//! 4. Convert tools/tool_choice/text.format into the chat-request fields.
//!
//! ## Outbound
//!
//! [`build_response_object`] takes the chat-completion generation result
//! plus any parsed tool calls and emits a [`ResponseObject`] with the
//! discriminated `output[]` array and the `output_text` aggregator.

use std::sync::Arc;

use crate::server::types::request::{
    ChatCompletionRequest, Message, MessageContent, Role, Tool, ToolCallFunction, ToolCallInMessage,
};
use crate::server::types::responses_request::{
    CreateResponseRequest, ResponseInputContent, ResponseInputItem, ResponseInputRole, ResponseTool,
};
use crate::server::types::responses_response::{
    ConversationRefEchoed, ResponseErrorBody, ResponseFunctionCallOutput,
    ResponseIncompleteDetails, ResponseInputTokensDetails, ResponseItemStatus, ResponseObject,
    ResponseOutputContent, ResponseOutputItem, ResponseOutputMessage, ResponseOutputTokensDetails,
    ResponseReasoningOutput, ResponseReasoningPart, ResponseStatus, ResponseTextConfigEchoed,
    ResponseUsage, ToolChoiceEchoed,
};

use super::conversation_store::{ConversationItem, ConversationStore};
use super::responses_store::{ResponsesStore, StoredResponse};
use super::tool_calls::types::ToolCallParseResult;

/// Maximum unsupported-tool-types catch-all label seen in error messages.
const UNSUPPORTED_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "mcp",
    "custom",
    "apply_patch",
    "function_shell",
];

/// Translation errors surfaced as 400 by the route layer.
#[derive(Debug, thiserror::Error)]
pub enum ResponsesTranslateError {
    #[error("previous_response_id and conversation are mutually exclusive")]
    PreviousAndConversation,
    #[error("previous_response_id '{0}' was not found or has expired")]
    PreviousNotFound(String),
    #[error(
        "conversation '{0}' was referenced but the conversation store is disabled on this server (--conversation-store-max-entries 0)"
    )]
    ConversationStoreDisabled(String),
    #[error("tool type '{0}' is not supported by mlxcel; Phase 1 supports only 'function' tools")]
    UnsupportedToolType(String),
    #[error("background=true is not supported in Phase 1; omit the field or set background=false")]
    BackgroundUnsupported,
    #[error("metadata is limited to 16 entries; received {0}")]
    MetadataTooLarge(usize),
    #[error("truncation='{0}' is not supported in Phase 1; only 'disabled' is accepted")]
    TruncationUnsupported(String),
    #[error("max_output_tokens must be > 0")]
    MaxOutputTokensInvalid,
}

/// Inbound translation result. Carries the synthetic
/// [`ChatCompletionRequest`] plus the canonicalised input items the
/// route handler needs to persist for chain rehydration.
#[derive(Debug)]
pub struct TranslatedRequest {
    pub chat_request: ChatCompletionRequest,
    pub canonical_input_items: Vec<ResponseInputItem>,
    pub effective_store: bool,
    pub conversation_id: Option<String>,
}

/// Flatten a Responses-API request into the chat-completions shape.
///
/// `responses_store` and `conversation_store` are consulted for
/// `previous_response_id` / `conversation` resolution. Either may be
/// `None` when the corresponding store is disabled; in that case the
/// caller is responsible for rejecting requests that reference those
/// stores before reaching this function.
pub fn responses_request_to_chat(
    request: &CreateResponseRequest,
    responses_store: Option<&Arc<ResponsesStore>>,
    conversation_store: Option<&Arc<ConversationStore>>,
) -> Result<TranslatedRequest, ResponsesTranslateError> {
    if request.previous_response_id.is_some() && request.conversation.is_some() {
        return Err(ResponsesTranslateError::PreviousAndConversation);
    }
    if request.background == Some(true) {
        return Err(ResponsesTranslateError::BackgroundUnsupported);
    }
    if let Some(meta) = request.metadata.as_ref()
        && meta.len() > 16
    {
        return Err(ResponsesTranslateError::MetadataTooLarge(meta.len()));
    }
    if let Some(trunc) = request.truncation.as_deref()
        && !trunc.is_empty()
        && trunc != "disabled"
    {
        return Err(ResponsesTranslateError::TruncationUnsupported(
            trunc.to_string(),
        ));
    }
    if let Some(cap) = request.max_output_tokens
        && cap == 0
    {
        return Err(ResponsesTranslateError::MaxOutputTokensInvalid);
    }
    reject_unsupported_tools(request.tools.as_deref())?;

    // Resolve prior items (history) from the chained reference.
    let mut prior_items: Vec<ResponseInputItem> = Vec::new();
    if let Some(prev_id) = request.previous_response_id.as_deref() {
        let store = responses_store
            .ok_or_else(|| ResponsesTranslateError::PreviousNotFound(prev_id.to_string()))?;
        let stored = store
            .get(prev_id)
            .ok_or_else(|| ResponsesTranslateError::PreviousNotFound(prev_id.to_string()))?;
        prior_items.extend(stored.input_items.clone());
        prior_items.extend(stored_outputs_as_input_items(&stored));
    } else if let Some(conv_ref) = request.conversation.as_ref() {
        // Reject when the conversation store is disabled — silently
        // ignoring the reference would leave the client thinking they
        // had multi-turn context when they don't (review H3).
        let store = conversation_store.ok_or_else(|| {
            ResponsesTranslateError::ConversationStoreDisabled(conv_ref.id().to_string())
        })?;
        if let Some(transcript) = store.get(conv_ref.id()) {
            for item in transcript.items {
                match item {
                    ConversationItem::Input(input) => prior_items.push(input),
                    ConversationItem::Output(output) => {
                        prior_items.extend(output_to_input_items(&output));
                    }
                }
            }
        }
    }

    // Append the current request's input items.
    let current_items = request.input.clone().into_items();
    let canonical_input_items = current_items.clone();
    let mut all_items = prior_items;
    all_items.extend(current_items);

    // Build the chat-completion messages. Instructions go in as a
    // leading system message so the existing template renderer treats
    // them like any other system turn.
    let mut messages: Vec<Message> = Vec::with_capacity(all_items.len() + 1);
    if let Some(ref instr) = request.instructions
        && !instr.is_empty()
    {
        messages.push(Message {
            role: Role::System,
            content: MessageContent::Text(instr.clone()),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        });
    }
    messages.extend(input_items_to_messages(&all_items));

    let tools = function_tools(request.tools.as_deref());
    let tool_choice = request.tool_choice.clone();
    let response_format = request
        .text
        .as_ref()
        .and_then(|t| t.format.as_ref())
        .and_then(|f| f.to_response_format_value());

    let mut sampling = request.sampling.clone();
    if sampling.max_tokens.is_none() {
        sampling.max_tokens = request.max_output_tokens;
    }
    if sampling.temperature.is_none() {
        sampling.temperature = request.temperature;
    }
    if sampling.top_p.is_none() {
        sampling.top_p = request.top_p;
    }

    let stream_options = if request.stream {
        Some(crate::server::types::request::StreamOptions {
            include_usage: true,
        })
    } else {
        None
    };

    let user = request
        .user
        .clone()
        .or_else(|| request.safety_identifier.clone());

    let chat_request = ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        stream: request.stream,
        stream_options,
        logprobs: None,
        top_logprobs: request.top_logprobs,
        tools,
        tool_choice,
        parallel_tool_calls: request.parallel_tool_calls,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: request.prompt_cache_key.clone(),
        user,
        extra_body_fields: serde_json::Map::new(),
        response_format,
        params: sampling,
    };

    let effective_store = request.store.unwrap_or(true);
    let conversation_id = request.conversation.as_ref().map(|c| c.id().to_string());

    Ok(TranslatedRequest {
        chat_request,
        canonical_input_items,
        effective_store,
        conversation_id,
    })
}

fn reject_unsupported_tools(tools: Option<&[ResponseTool]>) -> Result<(), ResponsesTranslateError> {
    let Some(tools) = tools else {
        return Ok(());
    };
    for tool in tools {
        if let ResponseTool::Unsupported = tool {
            // Pick the most informative label we know about; the
            // deserialiser stripped the type tag so we surface the
            // canonical list in the error message.
            return Err(ResponsesTranslateError::UnsupportedToolType(format!(
                "<unsupported> (one of: {})",
                UNSUPPORTED_TOOL_TYPES.join(", ")
            )));
        }
    }
    Ok(())
}

fn function_tools(tools: Option<&[ResponseTool]>) -> Option<Vec<Tool>> {
    let tools = tools?;
    let out: Vec<Tool> = tools
        .iter()
        .filter_map(|t| match t {
            ResponseTool::Function(f) => Some(Tool {
                tool_type: "function".to_string(),
                function: f.to_function_definition(),
            }),
            ResponseTool::Unsupported => None,
        })
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

/// Convert a flat list of input items into chat-completion messages.
///
/// `function_call` items become assistant messages with `tool_calls`;
/// `function_call_output` items become tool messages with the matching
/// `tool_call_id`. A `reasoning` item is not emitted as its own turn; instead
/// its text is buffered and attached to the parallel `reasoning` field of the
/// following assistant turn (issue #362) so templates that read
/// `message.get('reasoning')` see prior thinking. A reasoning item that is not
/// followed by an assistant turn before the next turn boundary is dropped.
fn input_items_to_messages(items: &[ResponseInputItem]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    let mut pending_tool_calls: Vec<ToolCallInMessage> = Vec::new();
    let mut pending_reasoning: Option<String> = None;

    for item in items {
        match item {
            ResponseInputItem::Message {
                role,
                content,
                name,
            } => {
                let converted_role = convert_role(*role);
                // Flush any pending tool calls as an assistant turn
                // before emitting the next role message.
                if !pending_tool_calls.is_empty() {
                    out.push(Message {
                        role: Role::Assistant,
                        content: MessageContent::Text(String::new()),
                        name: None,
                        tool_call_id: None,
                        reasoning: pending_reasoning.take(),
                        tool_calls: Some(std::mem::take(&mut pending_tool_calls)),
                    });
                }
                let reasoning = if converted_role == Role::Assistant {
                    pending_reasoning.take()
                } else {
                    None
                };
                out.push(Message {
                    role: converted_role,
                    content: convert_content(content),
                    name: name.clone(),
                    tool_call_id: None,
                    reasoning,
                    tool_calls: None,
                });
                // A turn boundary consumes any leftover reasoning so it cannot
                // leak onto a later, unrelated assistant turn.
                pending_reasoning = None;
            }
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                pending_tool_calls.push(ToolCallInMessage {
                    id: call_id.clone(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                if !pending_tool_calls.is_empty() {
                    out.push(Message {
                        role: Role::Assistant,
                        content: MessageContent::Text(String::new()),
                        name: None,
                        tool_call_id: None,
                        reasoning: pending_reasoning.take(),
                        tool_calls: Some(std::mem::take(&mut pending_tool_calls)),
                    });
                }
                out.push(Message {
                    role: Role::Tool,
                    content: MessageContent::Text(output.clone()),
                    name: None,
                    tool_call_id: Some(call_id.clone()),
                    reasoning: None,
                    tool_calls: None,
                });
                // A tool-output turn is not an assistant turn, so any buffered
                // reasoning that was not consumed by the flush above (e.g.
                // malformed input: Reasoning immediately followed by
                // FunctionCallOutput with no preceding FunctionCall) must be
                // cleared here. Without this, the buffered reasoning leaks onto
                // the next assistant turn, violating the invariant that a
                // reasoning item not followed by an assistant turn before the
                // next turn boundary is dropped.
                pending_reasoning = None;
            }
            ResponseInputItem::Reasoning { content } => {
                // Buffer the reasoning text and attach it to the following
                // assistant turn (issue #362) so templates that render
                // `message.get('reasoning')` see the model's prior thinking.
                let text = content
                    .iter()
                    .map(|p| p.text.as_str())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    pending_reasoning = Some(text);
                }
            }
        }
    }

    if !pending_tool_calls.is_empty() {
        out.push(Message {
            role: Role::Assistant,
            content: MessageContent::Text(String::new()),
            name: None,
            tool_call_id: None,
            reasoning: pending_reasoning.take(),
            tool_calls: Some(pending_tool_calls),
        });
    }

    out
}

fn convert_role(role: ResponseInputRole) -> Role {
    match role {
        ResponseInputRole::User => Role::User,
        // `developer` is a refinement on `system` from the OpenAI API;
        // mlxcel treats both as system turns.
        ResponseInputRole::Assistant => Role::Assistant,
        ResponseInputRole::System | ResponseInputRole::Developer => Role::System,
    }
}

fn convert_content(content: &ResponseInputContent) -> MessageContent {
    match content {
        ResponseInputContent::Text(s) => MessageContent::Text(s.clone()),
        ResponseInputContent::Parts(parts) => MessageContent::Parts(parts.to_vec()),
    }
}

/// Render a stored response's outputs as input items so they can be
/// replayed as conversation history on a chained call.
fn stored_outputs_as_input_items(stored: &StoredResponse) -> Vec<ResponseInputItem> {
    let mut out: Vec<ResponseInputItem> = Vec::new();
    for item in &stored.response.output {
        out.extend(output_to_input_items(item));
    }
    out
}

fn output_to_input_items(output: &ResponseOutputItem) -> Vec<ResponseInputItem> {
    match output {
        ResponseOutputItem::Message(msg) => {
            let role = if msg.role == "assistant" {
                ResponseInputRole::Assistant
            } else {
                ResponseInputRole::User
            };
            let text: String = msg
                .content
                .iter()
                .map(|c| match c {
                    ResponseOutputContent::OutputText { text, .. } => text.as_str(),
                    ResponseOutputContent::Refusal { refusal } => refusal.as_str(),
                })
                .collect::<Vec<_>>()
                .join("");
            vec![ResponseInputItem::Message {
                role,
                content: ResponseInputContent::Text(text),
                name: None,
            }]
        }
        ResponseOutputItem::FunctionCall(f) => vec![ResponseInputItem::FunctionCall {
            call_id: f.call_id.clone(),
            name: f.name.clone(),
            arguments: f.arguments.clone(),
        }],
        // Reasoning output items are not replayed — the model can
        // regenerate reasoning when prompted again.
        ResponseOutputItem::Reasoning(_) => vec![],
    }
}

/// Outbound assembly state.
///
/// Built once per request and passed into [`build_response_object`] to
/// avoid threading dozens of fields through the call site.
pub struct OutboundContext<'a> {
    pub response_id: String,
    pub model_id: String,
    pub created_at: f64,
    pub completed_at: f64,
    pub status: ResponseStatus,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub reasoning_tokens: usize,
    pub text: String,
    pub reasoning_text: Option<String>,
    pub parsed_tool_calls: Option<&'a ToolCallParseResult>,
    pub max_tool_calls: Option<usize>,
    pub request: &'a CreateResponseRequest,
    pub error: Option<ResponseErrorBody>,
    pub incomplete_reason: Option<String>,
    pub finish_reason: String,
}

/// Build the full [`ResponseObject`] from the generation outcome and
/// the original request (used for echoing fields back to the client).
pub fn build_response_object(ctx: OutboundContext<'_>) -> ResponseObject {
    let mut output: Vec<ResponseOutputItem> = Vec::new();
    let mut output_text_acc = String::new();

    // Reasoning item first (matches OpenAI ordering — reasoning before
    // the final message when both are emitted).
    if let Some(reasoning) = ctx.reasoning_text.as_ref().filter(|s| !s.is_empty()) {
        output.push(ResponseOutputItem::Reasoning(ResponseReasoningOutput {
            id: format!("rs_{}", short_uuid()),
            status: ResponseItemStatus::Completed,
            content: vec![ResponseReasoningPart::ReasoningText {
                text: reasoning.clone(),
            }],
        }));
    }

    // Function-call items.
    let mut emitted_tool_calls = 0usize;
    if let Some(parsed) = ctx.parsed_tool_calls {
        for call in &parsed.tool_calls {
            if let Some(max) = ctx.max_tool_calls
                && emitted_tool_calls >= max
            {
                break;
            }
            output.push(ResponseOutputItem::FunctionCall(
                ResponseFunctionCallOutput {
                    id: format!("fc_{}", short_uuid()),
                    call_id: format!("call_{}", short_uuid()),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    status: ResponseItemStatus::Completed,
                },
            ));
            emitted_tool_calls += 1;
        }
    }

    // Message item (only when there is text to emit; tool-only turns
    // omit the message item per the OpenAI shape).
    let has_text_content = !ctx.text.is_empty();
    let emitted_tools = emitted_tool_calls > 0;
    if has_text_content || !emitted_tools {
        let content = ResponseOutputContent::output_text(ctx.text.clone());
        output_text_acc.push_str(&ctx.text);
        output.push(ResponseOutputItem::Message(
            ResponseOutputMessage::new_assistant(format!("msg_{}", short_uuid()), vec![content]),
        ));
    }

    let usage = ResponseUsage {
        input_tokens: ctx.prompt_tokens,
        output_tokens: ctx.completion_tokens,
        total_tokens: ctx.prompt_tokens + ctx.completion_tokens,
        input_tokens_details: Some(ResponseInputTokensDetails {
            cached_tokens: ctx.cached_tokens,
        }),
        output_tokens_details: Some(ResponseOutputTokensDetails {
            reasoning_tokens: ctx.reasoning_tokens,
        }),
    };

    let metadata = ctx.request.metadata.clone();
    let tools_echo = ctx.request.tools.as_ref().map(|tools| echo_tools(tools));
    let tool_choice_echo = ctx
        .request
        .tool_choice
        .as_ref()
        .map(ToolChoiceEchoed::from_choice);
    let text_echo = ctx
        .request
        .text
        .as_ref()
        .map(ResponseTextConfigEchoed::from_config);
    let conversation_echo = ctx
        .request
        .conversation
        .as_ref()
        .map(|c| ConversationRefEchoed::Id(c.id().to_string()));
    let incomplete_details = ctx
        .incomplete_reason
        .map(|r| ResponseIncompleteDetails { reason: r });

    ResponseObject {
        id: ctx.response_id,
        object: "response".to_string(),
        created_at: ctx.created_at,
        completed_at: Some(ctx.completed_at),
        status: ctx.status,
        model: ctx.model_id,
        output,
        output_text: output_text_acc,
        usage,
        error: ctx.error,
        incomplete_details,
        instructions: ctx.request.instructions.clone(),
        tools: tools_echo,
        tool_choice: tool_choice_echo,
        text: text_echo,
        reasoning: ctx.request.reasoning.clone(),
        metadata,
        temperature: ctx.request.temperature,
        top_p: ctx.request.top_p,
        parallel_tool_calls: ctx.request.parallel_tool_calls,
        truncation: ctx.request.truncation.clone(),
        max_output_tokens: ctx.request.max_output_tokens,
        max_tool_calls: ctx.request.max_tool_calls,
        top_logprobs: ctx.request.top_logprobs,
        previous_response_id: ctx.request.previous_response_id.clone(),
        conversation: conversation_echo,
        prompt_cache_key: ctx.request.prompt_cache_key.clone(),
        service_tier: ctx.request.service_tier.clone(),
        user: ctx
            .request
            .user
            .clone()
            .or_else(|| ctx.request.safety_identifier.clone()),
        store: ctx.request.store,
    }
}

fn echo_tools(tools: &[ResponseTool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .filter_map(|t| match t {
            ResponseTool::Function(f) => Some(serde_json::json!({
                "type": "function",
                "name": f.name,
                "description": f.description,
                "parameters": f.parameters,
                "strict": f.strict,
            })),
            ResponseTool::Unsupported => None,
        })
        .collect()
}

pub(crate) fn short_uuid() -> String {
    let raw = uuid::Uuid::new_v4().simple().to_string();
    raw.chars().take(16).collect()
}

/// Convenience entry point for the route layer: build a response object
/// representing an error that occurred mid-generation.
#[allow(dead_code)]
pub fn build_failed_response(
    id: String,
    model: String,
    created_at: f64,
    error_code: &str,
    error_message: &str,
    request: &CreateResponseRequest,
) -> ResponseObject {
    let ctx = OutboundContext {
        response_id: id,
        model_id: model,
        created_at,
        completed_at: created_at,
        status: ResponseStatus::Failed,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: String::new(),
        reasoning_text: None,
        parsed_tool_calls: None,
        max_tool_calls: None,
        request,
        error: Some(ResponseErrorBody {
            code: error_code.to_string(),
            message: error_message.to_string(),
        }),
        incomplete_reason: None,
        finish_reason: "error".to_string(),
    };
    build_response_object(ctx)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::server::types::request::SamplingParams;
    use crate::server::types::responses_request::{
        ConversationRef, ResponseInput, ResponseTextConfig, ResponseTextFormat,
    };

    fn make_request(input: ResponseInput) -> CreateResponseRequest {
        CreateResponseRequest {
            model: "m".to_string(),
            input,
            instructions: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            text: None,
            reasoning: None,
            conversation: None,
            previous_response_id: None,
            store: None,
            stream: false,
            stream_options: None,
            max_output_tokens: None,
            max_tool_calls: None,
            truncation: None,
            temperature: None,
            top_p: None,
            top_logprobs: None,
            metadata: None,
            prompt_cache_key: None,
            user: None,
            safety_identifier: None,
            background: None,
            service_tier: None,
            sampling: SamplingParams::default(),
        }
    }

    #[test]
    fn rejects_previous_id_with_conversation() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.previous_response_id = Some("resp_x".to_string());
        req.conversation = Some(ConversationRef::Id("conv_y".to_string()));
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::PreviousAndConversation);
    }

    #[test]
    fn rejects_background_true() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.background = Some(true);
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::BackgroundUnsupported);
    }

    #[test]
    fn rejects_truncation_auto() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.truncation = Some("auto".to_string());
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::TruncationUnsupported(_));
    }

    #[test]
    fn rejects_metadata_over_16() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        let mut m = HashMap::new();
        for i in 0..17 {
            m.insert(format!("k{i}"), "v".to_string());
        }
        req.metadata = Some(m);
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::MetadataTooLarge(17));
    }

    #[test]
    fn rejects_unsupported_tool_type() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.tools = Some(vec![ResponseTool::Unsupported]);
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::UnsupportedToolType(_));
    }

    #[test]
    fn maps_string_input_to_single_user_message() {
        let req = make_request(ResponseInput::Text("hello".to_string()));
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        assert_eq!(translated.chat_request.messages.len(), 1);
        assert!(matches!(
            translated.chat_request.messages[0].role,
            Role::User
        ));
        assert_eq!(translated.chat_request.messages[0].content.text(), "hello");
    }

    #[test]
    fn reasoning_item_attaches_to_following_assistant_message() {
        // Issue #362: a Responses `reasoning` item is not its own turn; its
        // text rides on the parallel `reasoning` field of the next assistant
        // message so reasoning-aware chat templates can render it.
        use crate::server::types::responses_request::{
            ReasoningContentPart, ResponseInputContent, ResponseInputItem, ResponseInputRole,
        };
        let items = vec![
            ResponseInputItem::Message {
                role: ResponseInputRole::User,
                content: ResponseInputContent::Text("what is 2+2?".to_string()),
                name: None,
            },
            ResponseInputItem::Reasoning {
                content: vec![ReasoningContentPart {
                    part_type: "reasoning_text".to_string(),
                    text: "add 2 and 2".to_string(),
                }],
            },
            ResponseInputItem::Message {
                role: ResponseInputRole::Assistant,
                content: ResponseInputContent::Text("4".to_string()),
                name: None,
            },
        ];
        let req = make_request(ResponseInput::Items(items));
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        let msgs = &translated.chat_request.messages;
        let assistant = msgs
            .iter()
            .find(|m| matches!(m.role, Role::Assistant))
            .expect("assistant turn present");
        assert_eq!(assistant.reasoning.as_deref(), Some("add 2 and 2"));
        assert_eq!(assistant.content.text(), "4");
        let user = msgs
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .expect("user turn present");
        assert_eq!(
            user.reasoning, None,
            "reasoning must not leak onto user turn"
        );
    }

    #[test]
    fn reasoning_does_not_leak_when_function_call_output_has_no_preceding_function_call() {
        // Regression for the MEDIUM bug: a Reasoning item immediately followed
        // by a FunctionCallOutput with no preceding FunctionCall (malformed
        // input) must not attach the buffered reasoning to the next assistant
        // turn. The FunctionCallOutput arm must clear pending_reasoning even
        // when the tool-call flush is skipped.
        use crate::server::types::responses_request::{
            ReasoningContentPart, ResponseInputContent, ResponseInputItem, ResponseInputRole,
        };
        let items = vec![
            ResponseInputItem::Reasoning {
                content: vec![ReasoningContentPart {
                    part_type: "reasoning_text".to_string(),
                    text: "orphaned reasoning".to_string(),
                }],
            },
            // No FunctionCall precedes this output, so pending_tool_calls is
            // empty and the flush block that would consume pending_reasoning is
            // skipped. The arm must still clear pending_reasoning.
            ResponseInputItem::FunctionCallOutput {
                call_id: "call_orphan".to_string(),
                output: "result".to_string(),
            },
            ResponseInputItem::Message {
                role: ResponseInputRole::Assistant,
                content: ResponseInputContent::Text("answer".to_string()),
                name: None,
            },
        ];
        let req = make_request(ResponseInput::Items(items));
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        let assistant = translated
            .chat_request
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::Assistant))
            .expect("assistant turn present");
        assert_eq!(
            assistant.reasoning, None,
            "reasoning must not leak onto the assistant turn after a bare FunctionCallOutput"
        );
    }

    #[test]
    fn reasoning_attaches_to_function_call_turn_in_normal_tool_flow() {
        // Confirms the normal Reasoning -> FunctionCall -> FunctionCallOutput
        // flow is unaffected by the pending_reasoning = None fix: reasoning
        // still attaches to the function-call assistant turn, not the tool
        // turn, and the following message sees no reasoning.
        use crate::server::types::responses_request::{
            ReasoningContentPart, ResponseInputContent, ResponseInputItem, ResponseInputRole,
        };
        let items = vec![
            ResponseInputItem::Message {
                role: ResponseInputRole::User,
                content: ResponseInputContent::Text("call a tool".to_string()),
                name: None,
            },
            ResponseInputItem::Reasoning {
                content: vec![ReasoningContentPart {
                    part_type: "reasoning_text".to_string(),
                    text: "I should call do_thing".to_string(),
                }],
            },
            ResponseInputItem::FunctionCall {
                call_id: "call_1".to_string(),
                name: "do_thing".to_string(),
                arguments: "{}".to_string(),
            },
            ResponseInputItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: "done".to_string(),
            },
            ResponseInputItem::Message {
                role: ResponseInputRole::Assistant,
                content: ResponseInputContent::Text("all done".to_string()),
                name: None,
            },
        ];
        let req = make_request(ResponseInput::Items(items));
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        let msgs = &translated.chat_request.messages;

        // The function-call flush produces an assistant turn that carries the
        // reasoning; the later text assistant turn must not carry it.
        let function_call_turn = msgs
            .iter()
            .find(|m| matches!(m.role, Role::Assistant) && m.tool_calls.is_some())
            .expect("function-call assistant turn present");
        assert_eq!(
            function_call_turn.reasoning.as_deref(),
            Some("I should call do_thing"),
            "reasoning must attach to the function-call assistant turn"
        );

        let text_assistant_turn = msgs
            .iter()
            .find(|m| matches!(m.role, Role::Assistant) && m.tool_calls.is_none())
            .expect("text assistant turn present");
        assert_eq!(
            text_assistant_turn.reasoning, None,
            "reasoning must not leak onto the later text assistant turn"
        );
    }

    #[test]
    fn instructions_become_leading_system_message() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.instructions = Some("you are helpful".to_string());
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        assert!(matches!(
            translated.chat_request.messages[0].role,
            Role::System
        ));
        assert_eq!(
            translated.chat_request.messages[0].content.text(),
            "you are helpful"
        );
    }

    #[test]
    fn text_json_schema_format_maps_to_response_format() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.text = Some(ResponseTextConfig {
            format: Some(ResponseTextFormat::JsonSchema {
                schema: serde_json::json!({"name":"r","schema":{"type":"object"}}),
            }),
        });
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        let rf = translated.chat_request.response_format.expect("set");
        assert_eq!(rf["type"], "json_schema");
    }

    #[test]
    fn store_defaults_to_true_when_omitted() {
        let req = make_request(ResponseInput::Text("hi".to_string()));
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        assert!(translated.effective_store);
    }

    #[test]
    fn store_false_round_trips_to_translated_request() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.store = Some(false);
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        assert!(!translated.effective_store);
    }

    #[test]
    fn previous_response_id_without_store_returns_400() {
        let mut req = make_request(ResponseInput::Text("turn 2".to_string()));
        req.previous_response_id = Some("resp_doesnt_exist".to_string());
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::PreviousNotFound(_));
    }

    #[test]
    fn conversation_string_form_records_id() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.conversation = Some(ConversationRef::Id("conv_42".to_string()));
        // Conversation referenced with an empty conversation store is
        // accepted — the translator records the id so the route can
        // append the post-completion transcript.
        let conv_store = Arc::new(ConversationStore::new(
            crate::server::conversation_store::ConversationStoreConfig::default(),
        ));
        let translated = responses_request_to_chat(&req, None, Some(&conv_store)).unwrap();
        assert_eq!(translated.conversation_id, Some("conv_42".to_string()));
    }

    #[test]
    fn conversation_referenced_without_store_returns_400() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.conversation = Some(ConversationRef::Id("conv_99".to_string()));
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::ConversationStoreDisabled(_));
    }

    #[test]
    fn function_tool_propagates_to_chat_tools() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.tools = Some(vec![ResponseTool::Function(
            crate::server::types::responses_request::FunctionToolDefinition {
                name: "do_thing".to_string(),
                description: Some("does it".to_string()),
                parameters: Some(serde_json::json!({"type":"object"})),
                strict: None,
            },
        )]);
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        let tools = translated.chat_request.tools.expect("set");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "do_thing");
        assert_eq!(tools[0].tool_type, "function");
    }

    #[test]
    fn safety_identifier_falls_back_to_user_for_chat_user_field() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.safety_identifier = Some("sid_1".to_string());
        let translated = responses_request_to_chat(&req, None, None).unwrap();
        assert_eq!(translated.chat_request.user, Some("sid_1".to_string()));
    }

    #[test]
    fn max_output_tokens_zero_is_rejected() {
        let mut req = make_request(ResponseInput::Text("hi".to_string()));
        req.max_output_tokens = Some(0);
        let err = responses_request_to_chat(&req, None, None).unwrap_err();
        matches!(err, ResponsesTranslateError::MaxOutputTokensInvalid);
    }

    #[test]
    fn outbound_build_includes_output_text_aggregator() {
        let req = make_request(ResponseInput::Text("hi".to_string()));
        let ctx = OutboundContext {
            response_id: "resp_test".to_string(),
            model_id: "m".to_string(),
            created_at: 0.0,
            completed_at: 1.0,
            status: ResponseStatus::Completed,
            prompt_tokens: 5,
            completion_tokens: 3,
            cached_tokens: 1,
            reasoning_tokens: 0,
            text: "hello world".to_string(),
            reasoning_text: None,
            parsed_tool_calls: None,
            max_tool_calls: None,
            request: &req,
            error: None,
            incomplete_reason: None,
            finish_reason: "stop".to_string(),
        };
        let resp = build_response_object(ctx);
        assert_eq!(resp.output_text, "hello world");
        assert_eq!(resp.usage.total_tokens, 8);
        assert_eq!(
            resp.usage
                .input_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            Some(1)
        );
    }
}
