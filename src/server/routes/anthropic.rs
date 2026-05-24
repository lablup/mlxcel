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

//! HTTP handlers for the Anthropic Messages API.
//!
//! Ported from upstream mlx-vlm `server/anthropic.py` (PR #1196, commit
//! `313ad22`) plus the tool-call image-handling fix (PR #1200, commit
//! `f2e19de`).
//!
//! Endpoints:
//!
//! - `POST /v1/messages`               — create a message (sync or streaming)
//! - `POST /v1/messages/count_tokens`  — count prompt tokens for a request
//!
//! The handlers are intentionally thin: the inbound/outbound translation
//! lives in [`crate::server::anthropic_translator`], the SSE encoding in
//! [`crate::server::streaming_anthropic`], and the actual generation reuses
//! the existing chat building blocks (`prepare_chat_request_with_cache`,
//! `build_generate_options`, `tool_calls::*`). Only the Anthropic envelope
//! sequencing is new here.

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::Sse},
};

use crate::server::AppState;
use crate::server::anthropic_translator::{
    AnthropicTranslated, anthropic_request_to_chat, anthropic_stop_reason, apply_stop_sequences,
    build_content_blocks, parsed_call_to_tool_use, short_uuid, thinking_enabled,
};
use crate::server::chat_request::prepare_chat_request_with_cache;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::streaming_anthropic::{AnthropicBlockEmitter, anthropic_sse_channel};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::anthropic_request::AnthropicRequest;
use crate::server::types::anthropic_response::{
    AnthropicErrorResponse, AnthropicMessageResponse, AnthropicResponseBlock, AnthropicUsage,
};
use crate::server::types::anthropic_stream::{
    AnthropicMessageDeltaBody, AnthropicMessageDeltaUsage, AnthropicStreamError,
    AnthropicStreamEvent,
};

use super::chat::{MAX_TOOLS, build_generate_options, parse_priority_header};

/// POST /v1/messages
pub async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    let translated = anthropic_request_to_chat(&request);

    // Enforce the tools array size limit to prevent DoS via template
    // rendering, matching the chat-completions route. The check runs on the
    // *converted* function tools (server tools without `input_schema` are
    // already dropped) since that is what reaches the chat template.
    if let Some(ref tools) = translated.chat_request.tools
        && tools.len() > MAX_TOOLS
    {
        return AnthropicErrorResponse::bad_request(format!(
            "Too many tools: {}. Maximum allowed is {MAX_TOOLS}.",
            tools.len()
        ))
        .into_response();
    }

    // Resolve the thinking budget exactly as the chat / responses routes do.
    let effective_max_tokens = translated
        .chat_request
        .params
        .max_tokens
        .unwrap_or(state.config.default_max_tokens);
    let raw_budget = pick_budget_alias(
        translated.chat_request.params.thinking_budget_tokens,
        translated.chat_request.params.thinking_token_budget,
        translated.chat_request.params.thinking_budget,
    );
    let budget_override = match resolve_request_budget(
        raw_budget,
        state.config.reasoning_budget,
        effective_max_tokens,
    ) {
        Ok(eff) => {
            if raw_budget.is_some() {
                ReasoningBudgetOverride::Explicit(eff)
            } else {
                ReasoningBudgetOverride::InheritServerDefault
            }
        }
        Err(err) => return AnthropicErrorResponse::bad_request(err.to_string()).into_response(),
    };

    let include_thinking = thinking_enabled(&request);
    let priority = parse_priority_header(&headers);

    if request.stream {
        stream_messages(
            state,
            request,
            translated,
            priority,
            budget_override,
            include_thinking,
        )
        .await
    } else {
        non_stream_messages(
            state,
            request,
            translated,
            priority,
            budget_override,
            include_thinking,
        )
        .await
    }
}

async fn non_stream_messages(
    state: AppState,
    request: AnthropicRequest,
    translated: AnthropicTranslated,
    priority: crate::server::batch::RequestPriority,
    budget_override: ReasoningBudgetOverride,
    include_thinking: bool,
) -> Response {
    if !state.can_accept_request() {
        return AnthropicErrorResponse::overloaded("All slots are busy. Please try again later.")
            .into_response();
    }

    let model_id = state.display_model_id().to_string();
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = match prepare_chat_request_with_cache(
        &state.chat_template,
        &translated.chat_request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(err) => return AnthropicErrorResponse::bad_request(err.to_string()).into_response(),
    };

    let mut options = build_generate_options(&translated.chat_request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;

    let result = match state.model_provider.generate_with_media_and_videos(
        prepared.prompt,
        options,
        prepared.image_data,
        prepared.audio_data,
        prepared.videos,
    ) {
        Ok(r) => r,
        Err(e) => {
            return AnthropicErrorResponse::api_error(format!("Generation failed: {e}"))
                .into_response();
        }
    };

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    // Parse tool calls and recover the visible/reasoning split using the
    // shared chat helpers.
    let parsed_tools = if tool_calls::should_parse_tool_calls(&translated.chat_request) {
        Some(tool_calls::parse_tool_calls(
            &result.text,
            translated.chat_request.tools.as_deref(),
        ))
    } else {
        None
    };
    let parsed_tool_calls = parsed_tools
        .as_ref()
        .filter(|p| p.has_tool_calls())
        .map(|p| p.tool_calls.clone());

    let (visible_text, reasoning_text) =
        split_visible_reasoning(&result.text, parsed_tools.as_ref());

    // Anthropic stop sequences: truncate visible text defensively (the
    // scheduler already halts on native stop sequences, but a match may
    // land inside a buffered chunk). Skip when tool calls were produced.
    let (final_text, stop_sequence) = if parsed_tool_calls.is_some() {
        (visible_text, None)
    } else {
        apply_stop_sequences(&visible_text, request.stop_sequences.as_deref())
    };

    let content_blocks = build_content_blocks(
        &final_text,
        reasoning_text.as_deref(),
        parsed_tool_calls.as_deref(),
        include_thinking,
    );
    let stop_reason = anthropic_stop_reason(
        &result.finish_reason,
        parsed_tool_calls.is_some(),
        stop_sequence.as_deref(),
    );

    let response = AnthropicMessageResponse::new(
        format!("msg_{}", short_uuid()),
        content_blocks,
        model_id,
        Some(stop_reason),
        stop_sequence,
        AnthropicUsage {
            input_tokens: result.prompt_tokens,
            output_tokens: result.completion_tokens,
        },
    );

    Json(response).into_response()
}

async fn stream_messages(
    state: AppState,
    request: AnthropicRequest,
    translated: AnthropicTranslated,
    priority: crate::server::batch::RequestPriority,
    budget_override: ReasoningBudgetOverride,
    include_thinking: bool,
) -> Response {
    if !state.can_accept_request() {
        return AnthropicErrorResponse::overloaded("All slots are busy. Please try again later.")
            .into_response();
    }

    let model_id = state.display_model_id().to_string();
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = match prepare_chat_request_with_cache(
        &state.chat_template,
        &translated.chat_request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(err) => return AnthropicErrorResponse::bad_request(err.to_string()).into_response(),
    };

    let mut options = build_generate_options(&translated.chat_request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;

    let (sender, stream, cancelled, keepalive) = anthropic_sse_channel(128);

    let model_id_for_task = model_id.clone();
    let stop_sequences = request.stop_sequences.clone();
    let parse_tools = tool_calls::should_parse_tool_calls(&translated.chat_request);
    let tools_for_parser = if parse_tools {
        translated.chat_request.tools.clone()
    } else {
        None
    };

    // Encode the rendered prompt once so the streaming `message_start` can
    // report the real prompt `input_tokens` up front (matches upstream; the
    // prior hardcoded 0 left strict Anthropic SDK clients without prompt
    // usage). Falls back to 0 if tokenization fails.
    let prompt_tokens = state
        .tokenizer
        .encode(&prepared.prompt, true)
        .map(|ids| ids.len())
        .unwrap_or(0);

    tokio::task::spawn_blocking(move || {
        let message_id = format!("msg_{}", short_uuid());

        // message_start carries the real prompt token count up front (matches
        // upstream); output_tokens accrues over the stream and is finalized in
        // the closing message_delta.
        let start_message = AnthropicMessageResponse::new(
            message_id.clone(),
            vec![],
            model_id_for_task.clone(),
            None,
            None,
            AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens: 0,
            },
        );
        let _ = sender.send_event(&AnthropicStreamEvent::MessageStart {
            message: start_message,
        });

        // Shared streaming state.
        let accumulated_raw = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let acc_clone = accumulated_raw.clone();
        let stream_filter = std::sync::Arc::new(std::sync::Mutex::new(StreamFilter::new()));
        let filter_for_callback = stream_filter.clone();
        let emitter = std::sync::Arc::new(std::sync::Mutex::new(AnthropicBlockEmitter::new()));
        let emitter_for_callback = emitter.clone();
        let sender_clone = sender.clone();
        // Accumulated visible text (post-filter) for the stop-sequence scan.
        let visible_acc = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let visible_for_callback = visible_acc.clone();

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                prepared.videos,
                cancelled,
                |token, _lp| {
                    if let Ok(mut acc) = acc_clone.lock() {
                        acc.push_str(&token);
                    }
                    let emit = filter_for_callback
                        .lock()
                        .ok()
                        .map(|mut f| f.feed(&token))
                        .unwrap_or_default();

                    if let Ok(mut em) = emitter_for_callback.lock() {
                        // Reasoning chunk → thinking block (only when extended
                        // thinking was requested; otherwise the reasoning is
                        // dropped from the visible stream, matching upstream).
                        if include_thinking
                            && let Some(reasoning) = emit.reasoning.filter(|s| !s.is_empty())
                        {
                            em.open_thinking(&sender_clone);
                            em.emit_thinking_delta(&sender_clone, reasoning);
                        }

                        if let Some(text) = emit.content.filter(|s| !s.is_empty()) {
                            if let Ok(mut v) = visible_for_callback.lock() {
                                v.push_str(&text);
                            }
                            em.open_text(&sender_clone);
                            em.emit_text_delta(&sender_clone, text);
                        }
                    }
                },
            );

        // Flush any buffered content out of the stream filter.
        let trailing = stream_filter
            .lock()
            .ok()
            .map(|mut f| f.flush())
            .unwrap_or_default();
        if let Some(text) = trailing.content.filter(|s| !s.is_empty())
            && let Ok(mut em) = emitter.lock()
        {
            if let Ok(mut v) = visible_acc.lock() {
                v.push_str(&text);
            }
            em.open_text(&sender);
            em.emit_text_delta(&sender, text);
        }

        let result = match result {
            Ok(r) => r,
            Err(err) => {
                if let Ok(mut em) = emitter.lock() {
                    em.close_open_block(&sender);
                }
                let _ = sender.send_event(&AnthropicStreamEvent::Error {
                    error: AnthropicStreamError {
                        error_type: "api_error".to_string(),
                        message: err.to_string(),
                    },
                });
                return;
            }
        };

        state.metrics.record_request(
            result.prompt_tokens,
            result.completion_tokens,
            result.generation_time_ms,
        );

        // Close any still-open text/thinking block before tool_use blocks.
        if let Ok(mut em) = emitter.lock() {
            em.close_open_block(&sender);
        }

        // Parse tool calls from the full raw output.
        let full_raw = accumulated_raw
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let parsed_tools = if parse_tools {
            Some(tool_calls::parse_tool_calls(
                &full_raw,
                tools_for_parser.as_deref(),
            ))
        } else {
            None
        };
        let parsed_calls = parsed_tools
            .as_ref()
            .filter(|p| p.has_tool_calls())
            .map(|p| p.tool_calls.clone());

        if let Some(calls) = parsed_calls.as_ref()
            && let Ok(mut em) = emitter.lock()
        {
            for call in calls {
                let block = parsed_call_to_tool_use(call);
                if let AnthropicResponseBlock::ToolUse { id, name, input } = block {
                    let input_json = serde_json::to_string(&input).unwrap_or_default();
                    em.emit_tool_use(&sender, id, name, &input_json);
                }
            }
        }

        // Stop-sequence detection on the visible text (skip when tool calls
        // were produced, mirroring upstream).
        let stop_sequence = if parsed_calls.is_some() {
            None
        } else {
            let visible = visible_acc.lock().map(|g| g.clone()).unwrap_or_default();
            apply_stop_sequences(&visible, stop_sequences.as_deref()).1
        };

        let stop_reason = anthropic_stop_reason(
            &result.finish_reason,
            parsed_calls.is_some(),
            stop_sequence.as_deref(),
        );

        let _ = sender.send_event(&AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: Some(stop_reason),
                stop_sequence,
            },
            usage: AnthropicMessageDeltaUsage {
                output_tokens: result.completion_tokens,
            },
        });
        let _ = sender.send_event(&AnthropicStreamEvent::MessageStop);
    });

    Sse::new(stream)
        .keep_alive(keepalive.into_inner())
        .into_response()
}

/// POST /v1/messages/count_tokens
///
/// Renders the prompt through the chat template and returns the prompt token
/// count. Mirrors the upstream `anthropic_count_tokens_endpoint`.
pub async fn anthropic_count_tokens(
    State(state): State<AppState>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    let translated = anthropic_request_to_chat(&request);

    // Enforce the tools array size limit before rendering the chat template.
    // count_tokens expands `tools` into the template just like the main
    // endpoint, so it needs the same MAX_TOOLS guard to avoid a
    // template-rendering DoS (the main `anthropic_messages` handler applies
    // the identical check).
    if let Some(ref tools) = translated.chat_request.tools
        && tools.len() > MAX_TOOLS
    {
        return AnthropicErrorResponse::bad_request(format!(
            "Too many tools: {}. Maximum allowed is {MAX_TOOLS}.",
            tools.len()
        ))
        .into_response();
    }

    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = match prepare_chat_request_with_cache(
        &state.chat_template,
        &translated.chat_request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(err) => return AnthropicErrorResponse::bad_request(err.to_string()).into_response(),
    };

    let token_count = match state.tokenizer.encode(&prepared.prompt, true) {
        Ok(ids) => ids.len(),
        Err(e) => {
            return AnthropicErrorResponse::bad_request(format!("Tokenization error: {e}"))
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({ "input_tokens": token_count })),
    )
        .into_response()
}

/// Split the raw generation output into `(visible_text, reasoning?)`.
///
/// Mirrors the chat/responses split: when tool parsing ran, the parser's
/// `content` is the visible text; otherwise structural tokens are stripped.
/// Reasoning is recovered from a `<think>...</think>` block (inline or
/// open-primed close-only) so it can populate a dedicated `thinking` block.
fn split_visible_reasoning(
    raw: &str,
    parsed: Option<&tool_calls::ToolCallParseResult>,
) -> (String, Option<String>) {
    let reasoning = extract_reasoning_from_raw(raw);
    let visible_source = match parsed {
        Some(p) => p.content.clone(),
        None => tool_calls::clean_structural_tokens(raw),
    };
    let visible = if let Some((_, after_close)) = visible_source.split_once("</think>") {
        after_close.trim_start().to_string()
    } else {
        visible_source.trim_start().to_string()
    };
    (visible, reasoning)
}

/// Recover the reasoning chunk from raw output (inline `<think>...</think>`
/// or open-primed close-only `...</think>`). Returns `None` when empty.
fn extract_reasoning_from_raw(raw: &str) -> Option<String> {
    if let Some(rest) = raw.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        let reason = rest[..end].trim();
        return if reason.is_empty() {
            None
        } else {
            Some(reason.to_string())
        };
    }
    if let Some(end) = raw.find("</think>") {
        let reason = raw[..end].trim();
        return if reason.is_empty() {
            None
        } else {
            Some(reason.to_string())
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_no_think_returns_cleaned() {
        let (visible, reasoning) = split_visible_reasoning("hello world", None);
        assert_eq!(visible, "hello world");
        assert_eq!(reasoning, None);
    }

    #[test]
    fn split_inline_think_block() {
        let (visible, reasoning) = split_visible_reasoning("<think>reasoning</think>answer", None);
        assert_eq!(visible, "answer");
        assert_eq!(reasoning.as_deref(), Some("reasoning"));
    }

    #[test]
    fn split_open_primed_close_only() {
        let (visible, reasoning) = split_visible_reasoning("trace text</think>final", None);
        assert_eq!(visible, "final");
        assert_eq!(reasoning.as_deref(), Some("trace text"));
    }

    #[test]
    fn extract_reasoning_empty_yields_none() {
        assert_eq!(extract_reasoning_from_raw("</think>x"), None);
        assert_eq!(extract_reasoning_from_raw("<think></think>x"), None);
    }
}
