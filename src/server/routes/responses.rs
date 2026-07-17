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

//! HTTP handlers for the OpenAI Responses API.
//!
//! Phase 1 endpoints:
//!
//! - `POST /v1/responses`        — create a response (sync or streaming)
//! - `GET /v1/responses/:id`     — retrieve a stored response
//! - `DELETE /v1/responses/:id`  — delete a stored response
//! - `POST /v1/responses/:id/cancel` — best-effort cancellation
//!
//! The handlers are intentionally thin: every meaningful decision lives
//! in [`crate::server::responses_translator`] (request flattening +
//! outbound assembly) and in the existing chat-completions building
//! blocks (`prepare_chat_request_with_cache`, `build_generate_options`,
//! `tool_calls::*`, `structured::*`). Only the streaming envelope
//! sequencing and the in-memory store wiring is new here.

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::Sse},
};

use crate::server::AppState;
use crate::server::chat_request::{prepare_chat_request_with_cache, request_has_effective_input};
use crate::server::config::ReasoningBudgetOverride;
use crate::server::conversation_store::ConversationItem;
use crate::server::responses_store::StoredResponse;
use crate::server::responses_translator::{
    OutboundContext, ResponsesTranslateError, build_response_object, responses_request_to_chat,
    short_uuid,
};
use crate::server::streaming_responses::{ResponseStreamEmitter, responses_sse_channel};
use crate::server::structured::build_constraint_from_response_format;
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{
    ResponseFunctionCallOutput, ResponseItemStatus, ResponseObject, ResponseOutputContent,
    ResponseOutputItem, ResponseOutputMessage, ResponseReasoningOutput, ResponseReasoningPart,
    ResponseStatus,
};
use crate::server::types::responses_stream::ResponseStreamEvent;

use super::chat::{
    build_generate_options, build_prompt_cache_request_context, parse_priority_header,
    validate_xtc_params,
};

/// POST /v1/responses
pub async fn create_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateResponseRequest>,
) -> Response {
    // -- Translate Responses request → ChatCompletionRequest ---------------
    let translated = match responses_request_to_chat(
        &request,
        state.responses_store.as_ref(),
        state.conversation_store.as_ref(),
    ) {
        Ok(t) => t,
        Err(err) => return translate_error_to_response(err).into_response(),
    };

    // Reject requests with no effective input before any model dispatch
    // (issue #773), mirroring the chat-completions check. Runs on the
    // translated `ChatCompletionRequest` so history pulled in from
    // `previous_response_id` / `conversation` and the `instructions`
    // system-message counts toward "effective input" the same way a
    // legitimate chat-completions history turn would.
    if !request_has_effective_input(&translated.chat_request) {
        return ErrorResponse::new(
            "Request must include at least one non-empty message content or media input.",
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate XTC (Exclude Top Choices) sampling parameter ranges before any
    // generation work begins.
    if let Err(message) = validate_xtc_params(
        translated.chat_request.params.xtc_threshold,
        translated.chat_request.params.xtc_probability,
    ) {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    // Build the structured-output constraint (json_schema) up front.
    let structured = {
        let tokenizer = state.tokenizer.clone();
        let response_format = translated.chat_request.response_format.clone();
        match tokio::task::spawn_blocking(move || {
            build_constraint_from_response_format(tokenizer.as_ref(), response_format.as_ref())
        })
        .await
        {
            Ok(Ok(opt)) => opt,
            Ok(Err(err)) => {
                return ErrorResponse::new(err.to_string(), "invalid_request_error")
                    .into_response();
            }
            Err(_) => {
                return ErrorResponse::new("structured-output preparation failed", "server_error")
                    .into_response();
            }
        }
    };

    // Resolve thinking budget the same way the chat path does.
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
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };

    let priority = parse_priority_header(&headers);
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp() as f64;

    if request.stream {
        stream_create_response(
            state,
            request,
            translated,
            response_id,
            created_at,
            priority,
            budget_override,
            structured,
        )
        .await
    } else {
        non_stream_create_response(
            state,
            request,
            translated,
            response_id,
            created_at,
            priority,
            budget_override,
            structured,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn non_stream_create_response(
    state: AppState,
    request: CreateResponseRequest,
    translated: crate::server::responses_translator::TranslatedRequest,
    response_id: String,
    created_at: f64,
    priority: crate::server::batch::RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
) -> Response {
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let model_id = state.display_model_id().to_string();
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = prepare_chat_request_with_cache(
        &state.chat_template,
        &translated.chat_request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await;
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    let mut options = build_generate_options(&translated.chat_request.params, &state.config);
    options.priority = priority;
    // per-request Gemma 4 image soft-token budget, resolved and validated from
    // the translated `image_url` content parts. `None` when unset.
    options.image_soft_tokens = prepared.image_soft_tokens;
    options.reasoning_budget = budget_override;
    options.structured = structured;
    // Wire the cross-request prompt-prefix KV cache (epic #116) into the
    // Responses path, mirroring the chat-completions handler. Built after
    // `prepare_chat_request_with_cache` so the digest sees the resolved
    // multimodal bytes; text-only requests yield a key byte-identical to the
    // chat path.
    options.prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &translated.chat_request,
        &prepared.image_data,
        &prepared.audio_data,
    );

    let result = state.model_provider.generate_with_media_and_videos(
        prepared.prompt,
        options,
        prepared.image_data,
        prepared.audio_data,
        prepared.videos,
    );

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            return ErrorResponse::new(format!("Generation error: {e}"), "server_error")
                .into_response();
        }
    };

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    // Parse tool calls (reuse chat-path helpers).
    let parsed_tools = if tool_calls::should_parse_tool_calls(&translated.chat_request) {
        Some(tool_calls::parse_tool_calls(
            &result.text,
            translated.chat_request.tools.as_deref(),
        ))
    } else {
        None
    };
    let (visible_text, _reasoning_text) = split_reasoning(&result.text, parsed_tools.as_ref());

    let completed_at = chrono::Utc::now().timestamp() as f64;
    let response = build_response_object(OutboundContext {
        response_id: response_id.clone(),
        model_id: model_id.clone(),
        created_at,
        completed_at,
        status: ResponseStatus::Completed,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        cached_tokens: result.cached_tokens,
        reasoning_tokens: 0,
        text: visible_text,
        reasoning_text: _reasoning_text,
        parsed_tool_calls: parsed_tools.as_ref(),
        max_tool_calls: request.max_tool_calls,
        request: &request,
        error: None,
        incomplete_reason: None,
        finish_reason: result.finish_reason.clone(),
    });

    persist_response(&state, &request, &translated, &response);

    Json(response).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn stream_create_response(
    state: AppState,
    request: CreateResponseRequest,
    translated: crate::server::responses_translator::TranslatedRequest,
    response_id: String,
    created_at: f64,
    priority: crate::server::batch::RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
) -> Response {
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let model_id = state.display_model_id().to_string();
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = prepare_chat_request_with_cache(
        &state.chat_template,
        &translated.chat_request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await;
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    let mut options = build_generate_options(&translated.chat_request.params, &state.config);
    options.priority = priority;
    // per-request Gemma 4 image soft-token budget, resolved and validated from
    // the translated `image_url` content parts. `None` when unset.
    options.image_soft_tokens = prepared.image_soft_tokens;
    options.reasoning_budget = budget_override;
    options.structured = structured;
    // Wire the prompt-prefix KV cache (epic #116) into the streaming Responses
    // path too, before `options`/`translated` move into the spawned task.
    options.prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &translated.chat_request,
        &prepared.image_data,
        &prepared.audio_data,
    );

    let (sender, stream, cancelled, keepalive) = responses_sse_channel(128);

    // Review H2: register this response in the in-flight registry so
    // POST /v1/responses/:id/cancel can abort it from a different
    // request thread. We OR the existing client-disconnect token with
    // the in-flight token by sharing the same Arc<AtomicBool>: the
    // SSE channel already owns one (`cancelled`), and we additionally
    // expose it via the store under the response id. When the cancel
    // handler flips it, the scheduler observes the abort on its next
    // poll, just like the client-disconnect case.
    if let Some(store) = state.responses_store.as_ref() {
        match store.in_flight_write() {
            Ok(mut g) => {
                g.insert(response_id.clone(), cancelled.clone());
            }
            Err(_) => {
                tracing::warn!(
                    "responses_store in-flight registry is poisoned; cancellation API will not work for {response_id}"
                );
            }
        }
    }

    let response_id_for_task = response_id.clone();
    let model_id_for_task = model_id.clone();
    let request_for_task = request.clone();
    let translated_for_task = translated;
    let state_for_task = state.clone();
    let request_for_persist = request.clone();
    let parse_tools = tool_calls::should_parse_tool_calls(&translated_for_task.chat_request);
    let tools_for_parser = if parse_tools {
        translated_for_task.chat_request.tools.clone()
    } else {
        None
    };

    tokio::task::spawn_blocking(move || {
        let mut emitter = ResponseStreamEmitter::new();
        let initial_response = build_response_object(OutboundContext {
            response_id: response_id_for_task.clone(),
            model_id: model_id_for_task.clone(),
            created_at,
            completed_at: created_at,
            status: ResponseStatus::InProgress,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            reasoning_tokens: 0,
            text: String::new(),
            reasoning_text: None,
            parsed_tool_calls: None,
            max_tool_calls: request_for_task.max_tool_calls,
            request: &request_for_task,
            error: None,
            incomplete_reason: None,
            finish_reason: "in_progress".to_string(),
        });

        let _ = sender.send_event(&ResponseStreamEvent::Created {
            sequence_number: emitter.next_seq(),
            response: initial_response.clone(),
        });
        let _ = sender.send_event(&ResponseStreamEvent::InProgress {
            sequence_number: emitter.next_seq(),
            response: initial_response,
        });

        // Lazy envelope state: we don't open the message or reasoning
        // item until the first matching chunk arrives, so we can emit
        // them in the order the model produces them (reasoning first
        // for Qwen3/DeepSeek priming, content first otherwise).
        let accumulated_raw = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let acc_clone = accumulated_raw.clone();
        let stream_filter = std::sync::Arc::new(std::sync::Mutex::new(StreamFilter::new()));
        let filter_for_callback = stream_filter.clone();
        let sender_clone = sender.clone();
        let emitter_arc = std::sync::Arc::new(std::sync::Mutex::new(emitter));
        let emitter_for_callback = emitter_arc.clone();

        let result = state_for_task
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

                    // Reasoning chunk: lazily open the reasoning item
                    // envelope, then emit the delta.
                    if let Some(text) = emit.reasoning.filter(|s| !s.is_empty())
                        && let Ok(mut em) = emitter_for_callback.lock()
                    {
                        let r_id = if let Some(id) = em.active_reasoning_id.clone() {
                            id
                        } else {
                            let new_id = format!("rs_{}", short_uuid());
                            em.open_reasoning(new_id.clone());
                            let placeholder =
                                ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                                    id: new_id.clone(),
                                    status: ResponseItemStatus::InProgress,
                                    content: vec![],
                                });
                            let seq = em.next_seq();
                            let _ =
                                sender_clone.send_event(&ResponseStreamEvent::OutputItemAdded {
                                    sequence_number: seq,
                                    output_index: em.output_index(),
                                    item: placeholder,
                                });
                            new_id
                        };
                        em.reasoning_text_acc.push_str(&text);
                        let seq = em.next_seq();
                        let out_idx = em.output_index();
                        let _ = sender_clone.send_event(&ResponseStreamEvent::ReasoningTextDelta {
                            sequence_number: seq,
                            item_id: r_id,
                            output_index: out_idx,
                            content_index: 0,
                            delta: text,
                        });
                    }

                    // Content chunk: close any active reasoning envelope
                    // first, then lazily open the message envelope and
                    // emit the delta.
                    if let Some(text) = emit.content.filter(|s| !s.is_empty())
                        && let Ok(mut em) = emitter_for_callback.lock()
                    {
                        if let Some((r_id, r_text)) = em.close_reasoning() {
                            let seq = em.next_seq();
                            let _ =
                                sender_clone.send_event(&ResponseStreamEvent::ReasoningTextDone {
                                    sequence_number: seq,
                                    item_id: r_id.clone(),
                                    output_index: em.output_index(),
                                    content_index: 0,
                                    text: r_text.clone(),
                                });
                            let reasoning_item =
                                ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                                    id: r_id.clone(),
                                    status: ResponseItemStatus::Completed,
                                    content: vec![ResponseReasoningPart::ReasoningText {
                                        text: r_text,
                                    }],
                                });
                            let seq = em.next_seq();
                            let _ = sender_clone.send_event(&ResponseStreamEvent::OutputItemDone {
                                sequence_number: seq,
                                output_index: em.output_index(),
                                item: reasoning_item,
                            });
                            em.advance_output_index();
                        }

                        let msg_id = if let Some(id) = em.active_message_id.clone() {
                            id
                        } else {
                            let new_id = format!("msg_{}", short_uuid());
                            em.open_message(new_id.clone());
                            let placeholder = ResponseOutputItem::Message(
                                ResponseOutputMessage::new_assistant(new_id.clone(), vec![]),
                            );
                            let seq = em.next_seq();
                            let _ =
                                sender_clone.send_event(&ResponseStreamEvent::OutputItemAdded {
                                    sequence_number: seq,
                                    output_index: em.output_index(),
                                    item: placeholder,
                                });
                            let seq = em.next_seq();
                            let _ =
                                sender_clone.send_event(&ResponseStreamEvent::ContentPartAdded {
                                    sequence_number: seq,
                                    item_id: new_id.clone(),
                                    output_index: em.output_index(),
                                    content_index: 0,
                                    part: ResponseOutputContent::output_text(String::new()),
                                });
                            new_id
                        };
                        em.message_text_acc.push_str(&text);
                        let seq = em.next_seq();
                        let out_idx = em.output_index();
                        let _ = sender_clone.send_event(&ResponseStreamEvent::OutputTextDelta {
                            sequence_number: seq,
                            item_id: msg_id,
                            output_index: out_idx,
                            content_index: 0,
                            delta: text,
                        });
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
            && let Ok(mut em) = emitter_for_callback.lock()
            && let Some(msg_id) = em.active_message_id.clone()
        {
            em.message_text_acc.push_str(&text);
            let seq = em.next_seq();
            let out_idx = em.output_index();
            let _ = sender.send_event(&ResponseStreamEvent::OutputTextDelta {
                sequence_number: seq,
                item_id: msg_id,
                output_index: out_idx,
                content_index: 0,
                delta: text,
            });
        }

        let result = match result {
            Ok(r) => r,
            Err(err) => {
                if let Ok(mut em) = emitter_for_callback.lock() {
                    let seq = em.next_seq();
                    let _ = sender.send_event(&ResponseStreamEvent::Error {
                        sequence_number: seq,
                        code: "server_error".to_string(),
                        message: err.to_string(),
                    });
                }
                if let Some(store) = state_for_task.responses_store.as_ref() {
                    store.unregister_in_flight(&response_id_for_task);
                }
                return;
            }
        };

        state_for_task.metrics.record_request(
            result.prompt_tokens,
            result.completion_tokens,
            result.generation_time_ms,
        );

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

        let mut em = emitter_for_callback.lock().expect("emitter mutex");

        // Close any still-open reasoning envelope (model produced
        // reasoning but never transitioned to content — e.g. budget
        // hit mid-thinking).
        if let Some((r_id, r_text)) = em.close_reasoning() {
            let seq = em.next_seq();
            let _ = sender.send_event(&ResponseStreamEvent::ReasoningTextDone {
                sequence_number: seq,
                item_id: r_id.clone(),
                output_index: em.output_index(),
                content_index: 0,
                text: r_text.clone(),
            });
            let item = ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                id: r_id,
                status: ResponseItemStatus::Completed,
                content: vec![ResponseReasoningPart::ReasoningText { text: r_text }],
            });
            let seq = em.next_seq();
            let _ = sender.send_event(&ResponseStreamEvent::OutputItemDone {
                sequence_number: seq,
                output_index: em.output_index(),
                item,
            });
            em.advance_output_index();
        }

        // Close the message envelope (if one was opened). Reasoning-
        // only responses with no content skip this block entirely.
        let (closed_msg_id, message_text) = em
            .close_message()
            .unwrap_or_else(|| (String::new(), String::new()));
        if !closed_msg_id.is_empty() {
            let seq = em.next_seq();
            let _ = sender.send_event(&ResponseStreamEvent::OutputTextDone {
                sequence_number: seq,
                item_id: closed_msg_id.clone(),
                output_index: em.output_index(),
                content_index: 0,
                text: message_text.clone(),
            });
            let seq = em.next_seq();
            let _ = sender.send_event(&ResponseStreamEvent::ContentPartDone {
                sequence_number: seq,
                item_id: closed_msg_id.clone(),
                output_index: em.output_index(),
                content_index: 0,
                part: ResponseOutputContent::output_text(message_text.clone()),
            });
            let msg_item = ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
                closed_msg_id.clone(),
                vec![ResponseOutputContent::output_text(message_text.clone())],
            ));
            let seq = em.next_seq();
            let _ = sender.send_event(&ResponseStreamEvent::OutputItemDone {
                sequence_number: seq,
                output_index: em.output_index(),
                item: msg_item,
            });
            em.advance_output_index();
        }

        let reasoning_text_for_response = if em.reasoning_text_acc.is_empty() {
            None
        } else {
            Some(em.reasoning_text_acc.clone())
        };

        if let Some(parsed) = parsed_tools.as_ref() {
            for call in &parsed.tool_calls {
                let fc_id = format!("fc_{}", short_uuid());
                let call_id = format!("call_{}", short_uuid());
                let fc = ResponseFunctionCallOutput {
                    id: fc_id.clone(),
                    call_id: call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    status: ResponseItemStatus::Completed,
                };
                let item = ResponseOutputItem::FunctionCall(fc);
                let seq = em.next_seq();
                let _ = sender.send_event(&ResponseStreamEvent::OutputItemAdded {
                    sequence_number: seq,
                    output_index: em.output_index(),
                    item: item.clone(),
                });
                let seq = em.next_seq();
                let _ = sender.send_event(&ResponseStreamEvent::FunctionCallArgumentsDelta {
                    sequence_number: seq,
                    item_id: fc_id.clone(),
                    output_index: em.output_index(),
                    delta: call.arguments.clone(),
                });
                let seq = em.next_seq();
                let _ = sender.send_event(&ResponseStreamEvent::FunctionCallArgumentsDone {
                    sequence_number: seq,
                    item_id: fc_id.clone(),
                    output_index: em.output_index(),
                    arguments: call.arguments.clone(),
                });
                let seq = em.next_seq();
                let _ = sender.send_event(&ResponseStreamEvent::OutputItemDone {
                    sequence_number: seq,
                    output_index: em.output_index(),
                    item,
                });
                em.advance_output_index();
            }
        }

        // Final response object. The text is the message accumulator; the
        // outbound assembly fills the canonical output[] for the
        // "completed" event payload.
        let completed_at_now = chrono::Utc::now().timestamp() as f64;
        let final_response = build_response_object(OutboundContext {
            response_id: response_id_for_task.clone(),
            model_id: model_id_for_task,
            created_at,
            completed_at: completed_at_now,
            status: ResponseStatus::Completed,
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            cached_tokens: result.cached_tokens,
            reasoning_tokens: 0,
            text: message_text,
            reasoning_text: reasoning_text_for_response,
            parsed_tool_calls: parsed_tools.as_ref(),
            max_tool_calls: request_for_task.max_tool_calls,
            request: &request_for_task,
            error: None,
            incomplete_reason: None,
            finish_reason: result.finish_reason.clone(),
        });

        let seq = em.next_seq();
        let _ = sender.send_event(&ResponseStreamEvent::Completed {
            sequence_number: seq,
            response: final_response.clone(),
        });

        persist_response(
            &state_for_task,
            &request_for_persist,
            &translated_for_task,
            &final_response,
        );

        // Review H2: drop the in-flight registry entry now that the
        // streaming task is done. Subsequent cancel calls for this id
        // will fall through to the persisted-response path.
        if let Some(store) = state_for_task.responses_store.as_ref() {
            store.unregister_in_flight(&response_id_for_task);
        }
    });

    Sse::new(stream)
        .keep_alive(keepalive.into_inner())
        .into_response()
}

/// GET /v1/responses/:id
pub async fn retrieve_response(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(store) = state.responses_store.as_ref() else {
        return ErrorResponse::new(
            "response storage is disabled on this server; restart with --responses-store-max-entries > 0",
            "invalid_request_error",
        )
        .into_response();
    };
    match store.get(&id) {
        Some(stored) => Json(stored.response).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("response '{id}' not found"),
                "not_found",
            )),
        )
            .into_response(),
    }
}

/// DELETE /v1/responses/:id
pub async fn delete_response(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(store) = state.responses_store.as_ref() else {
        return ErrorResponse::new(
            "response storage is disabled on this server",
            "invalid_request_error",
        )
        .into_response();
    };
    let deleted = store.remove(&id).is_some();
    Json(serde_json::json!({
        "id": id,
        "object": "response.deleted",
        "deleted": deleted,
    }))
    .into_response()
}

/// POST /v1/responses/:id/cancel
pub async fn cancel_response(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(store) = state.responses_store.as_ref() else {
        return ErrorResponse::new(
            "response storage is disabled on this server",
            "invalid_request_error",
        )
        .into_response();
    };

    // Review H2: first check the in-flight registry — if the response
    // is still streaming, flip its cancellation token so the scheduler
    // aborts the underlying sequence and the streaming task emits
    // `response.failed` / `response.error` on its way out.
    let in_flight_aborted = store.cancel_in_flight(&id);

    if let Some(mut stored) = store.get(&id) {
        // Already-persisted response: just mark cancelled.
        stored.response.status = ResponseStatus::Cancelled;
        let stored_for_insert = StoredResponse {
            response: stored.response.clone(),
            input_items: stored.input_items.clone(),
        };
        store.insert(id.clone(), stored_for_insert);
        return Json(stored.response).into_response();
    }

    if in_flight_aborted {
        // The streaming task is still mid-flight; return a minimal
        // "cancelled" envelope so the client knows the cancel call was
        // honoured. The persisted entry (when `store=true`) will be
        // written by the streaming task on its way out.
        return Json(serde_json::json!({
            "id": id,
            "object": "response",
            "status": "cancelled",
        }))
        .into_response();
    }

    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(
            format!("response '{id}' not found"),
            "not_found",
        )),
    )
        .into_response()
}

// -- Helpers ------------------------------------------------------------------

fn translate_error_to_response(err: ResponsesTranslateError) -> ErrorResponse {
    let msg = err.to_string();
    let code = match err {
        ResponsesTranslateError::PreviousNotFound(_) => "invalid_request_error",
        _ => "invalid_request_error",
    };
    ErrorResponse::new(msg, code)
}

/// Split the raw generation output into a `(visible_text, reasoning?)`
/// pair for Responses-API output assembly.
///
/// The strategy mirrors the chat-completions path but operates without
/// access to the chat-template prompt suffix: we read the raw output
/// directly so we can recover the reasoning chunk for the dedicated
/// `reasoning` output item.
///
/// Three priming conventions are handled:
/// - **Inline `<think>...</think>`**: classic Qwen3/DeepSeek-R1 shape
///   when the template does not pre-open the block.
/// - **Open-primed close-only `...</think>...`**: the chat template
///   injected `<think>\n` into the prompt, so the model's first emitted
///   tokens are reasoning, followed by `</think>` and the visible
///   reply.
/// - **No `<think>` at all**: visible text is the cleaned raw output;
///   no reasoning is emitted.
fn split_reasoning(
    raw: &str,
    parsed: Option<&crate::server::tool_calls::types::ToolCallParseResult>,
) -> (String, Option<String>) {
    // Recover the reasoning chunk from the *raw* output before any
    // thinking-block stripping. `clean_structural_tokens` (and the tool
    // parser's internal `strip_thinking`) erase `<think>...</think>`
    // entirely, which would otherwise mask reasoning from this helper.
    let reasoning = extract_reasoning_from_raw(raw);
    let visible_source = match parsed {
        Some(p) => p.content.clone(),
        None => tool_calls::clean_structural_tokens(raw),
    };
    // If the visible source still carries a `</think>` close tag (the
    // open-primed-close-only case — `clean_structural_tokens` does not
    // strip a bare close marker), keep only the chunk after it. Any
    // reasoning before the close tag was already captured by
    // `extract_reasoning_from_raw`.
    let visible = if let Some(after_close) = visible_source.split_once("</think>") {
        after_close.1.trim_start().to_string()
    } else {
        visible_source.trim_start().to_string()
    };
    (visible, reasoning)
}

/// Extract reasoning from the raw generation output, handling both the
/// inline `<think>...</think>` form and the open-primed close-only
/// `<reasoning text></think>...` form. Returns `None` if no reasoning
/// can be recovered.
fn extract_reasoning_from_raw(raw: &str) -> Option<String> {
    // Inline `<think>...</think>` block.
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
    // Open-primed close-only: the prompt injected `<think>\n` so the
    // emitted text begins with the reasoning trace followed by
    // `</think>`. Anything before the close tag is the reasoning chunk;
    // we ignore content past the tag (the visible text path handles it).
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

fn persist_response(
    state: &AppState,
    request: &CreateResponseRequest,
    translated: &crate::server::responses_translator::TranslatedRequest,
    response: &ResponseObject,
) {
    if !translated.effective_store {
        return;
    }
    if let Some(store) = state.responses_store.as_ref() {
        store.insert(
            response.id.clone(),
            StoredResponse {
                response: response.clone(),
                input_items: translated.canonical_input_items.clone(),
            },
        );
    }
    if let Some(conv_id) = translated.conversation_id.as_ref()
        && let Some(conv_store) = state.conversation_store.as_ref()
    {
        let mut items: Vec<ConversationItem> = translated
            .canonical_input_items
            .iter()
            .cloned()
            .map(ConversationItem::Input)
            .collect();
        items.extend(
            response
                .output
                .iter()
                .cloned()
                .map(ConversationItem::Output),
        );
        conv_store.append(conv_id, items);
    }
    let _ = request; // silence unused on minimal-feature builds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::ChatCompletionRequest;

    #[test]
    fn split_reasoning_inline_block_separates_content() {
        let (visible, reasoning) = split_reasoning("<think>some thoughts</think>hi there", None);
        assert_eq!(visible, "hi there");
        assert_eq!(reasoning.as_deref(), Some("some thoughts"));
    }

    #[test]
    fn split_reasoning_open_primed_close_only_separates_content() {
        // Qwen3 primed-open scenario: prompt injected `<think>\n`, the
        // model's emitted text begins with the reasoning trace followed
        // by `</think>` and the real reply.
        let (visible, reasoning) = split_reasoning("reasoning text</think>\n\nHello!", None);
        assert_eq!(visible, "Hello!");
        assert_eq!(reasoning.as_deref(), Some("reasoning text"));
    }

    #[test]
    fn split_reasoning_close_only_empty_reasoning_yields_none() {
        let (visible, reasoning) = split_reasoning("</think>\n\nGreetings.", None);
        assert_eq!(visible, "Greetings.");
        assert!(reasoning.is_none());
    }

    #[test]
    fn split_reasoning_no_think_block_returns_raw() {
        let (visible, reasoning) = split_reasoning("just text", None);
        assert_eq!(visible, "just text");
        assert!(reasoning.is_none());
    }

    // -- no-effective-input rejection (issue #773), exercised on the
    // translated request the create_response handler actually checks.

    fn translate(body: &str) -> ChatCompletionRequest {
        let request: CreateResponseRequest = serde_json::from_str(body).unwrap();
        responses_request_to_chat(&request, None, None)
            .expect("translation should succeed")
            .chat_request
    }

    #[test]
    fn responses_route_rejects_empty_string_input() {
        let chat_request = translate(r#"{"model":"m","input":""}"#);
        assert!(!request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_rejects_whitespace_only_input() {
        let chat_request = translate(r#"{"model":"m","input":"   "}"#);
        assert!(!request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_rejects_content_list_with_only_empty_text() {
        let chat_request = translate(
            r#"{"model":"m","input":[
                {"type":"message","role":"user","content":[{"type":"text","text":""}]}
            ]}"#,
        );
        assert!(!request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_accepts_image_only_input() {
        let chat_request = translate(
            r#"{"model":"m","input":[
                {"type":"message","role":"user","content":[
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}
                ]}
            ]}"#,
        );
        assert!(request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_accepts_normal_text_input() {
        let chat_request = translate(r#"{"model":"m","input":"Hello there"}"#);
        assert!(request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_history_with_real_content_passes_despite_empty_last_message() {
        let chat_request = translate(
            r#"{"model":"m","input":[
                {"type":"message","role":"user","content":"What is the capital of France?"},
                {"type":"message","role":"assistant","content":"Paris."},
                {"type":"message","role":"user","content":""}
            ]}"#,
        );
        assert!(request_has_effective_input(&chat_request));
    }

    #[test]
    fn responses_route_non_empty_instructions_count_as_effective_input() {
        // `instructions` is translated into a leading system message (see
        // `responses_request_to_chat`), so a request with blank `input` but
        // real `instructions` is not degenerate — documented behavior, not
        // an oversight.
        let chat_request =
            translate(r#"{"model":"m","input":"","instructions":"You are a helpful assistant."}"#);
        assert!(request_has_effective_input(&chat_request));
    }

    // -- XTC (Exclude Top Choices) request validation on /v1/responses --
    //
    // `CreateResponseRequest` flattens `SamplingParams`, and the translator
    // forwards it into `translated.chat_request.params`. These tests walk the
    // same translate-then-validate path as `create_response` and confirm the
    // shared `validate_xtc_params` gate rejects out-of-range values.

    #[test]
    fn responses_rejects_out_of_range_xtc_probability() {
        let request: CreateResponseRequest =
            serde_json::from_str(r#"{"model":"m","input":"hi","xtc_probability":2.0}"#).unwrap();
        let translated = responses_request_to_chat(&request, None, None).unwrap();
        assert_eq!(
            validate_xtc_params(
                translated.chat_request.params.xtc_threshold,
                translated.chat_request.params.xtc_probability,
            ),
            Err("xtc_probability must be between 0.0 and 1.0")
        );
    }

    #[test]
    fn responses_rejects_out_of_range_xtc_threshold() {
        let request: CreateResponseRequest =
            serde_json::from_str(r#"{"model":"m","input":"hi","xtc_threshold":-0.1}"#).unwrap();
        let translated = responses_request_to_chat(&request, None, None).unwrap();
        assert_eq!(
            validate_xtc_params(
                translated.chat_request.params.xtc_threshold,
                translated.chat_request.params.xtc_probability,
            ),
            Err("xtc_threshold must be between 0.0 and 0.5")
        );
    }

    #[test]
    fn responses_accepts_in_range_xtc() {
        let request: CreateResponseRequest = serde_json::from_str(
            r#"{"model":"m","input":"hi","xtc_probability":0.5,"xtc_threshold":0.1}"#,
        )
        .unwrap();
        let translated = responses_request_to_chat(&request, None, None).unwrap();
        assert!(
            validate_xtc_params(
                translated.chat_request.params.xtc_threshold,
                translated.chat_request.params.xtc_probability,
            )
            .is_ok()
        );
    }
}
