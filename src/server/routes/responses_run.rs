// Copyright 2025-2026 Lablup Inc.
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

//! One streamed `/v1/responses` generation, independent of the HTTP
//! connection that asked for it (issue #2896).
//!
//! [`prepare_response_run`] renders the prompt, builds the generate options
//! and the prompt-cache context, and reserves the queue slot, producing a
//! [`PreparedRun`] that owns everything the blocking task needs.
//! [`drive_response_run`] runs that generation on the calling (blocking)
//! thread and delivers every Responses event through a
//! [`ResponseEventSink`]. The foreground SSE route uses [`SseResponseSink`];
//! a sink that is not tied to a client connection can drive the same
//! generation, which is what background mode builds on.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::server::chat_request::{
    PreparedChatRequest, prepare_chat_request_with_cache_and_video_allowlist,
};
use crate::server::config::ReasoningBudgetOverride;
use crate::server::model_provider::{
    AdmissionWaiter, QueueFullError, SingleStreamQueueReservation,
};
use crate::server::reasoning_echo::ReasoningEchoScope;
use crate::server::request_options::chat_carries_loop_amplifier;
use crate::server::responses_translator::{
    OutboundContext, TranslatedRequest, build_response_object, short_uuid,
};
use crate::server::streaming_responses::{
    CancellationToken, ResponseStreamEmitter, ResponseStreamSender,
};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{
    ResponseItemStatus, ResponseObject, ResponseOutputContent, ResponseOutputItem,
    ResponseOutputMessage, ResponseReasoningOutput, ResponseReasoningPart, ResponseStatus,
};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, LiveSettings, ServerGenerateOptions};

use super::chat::{
    build_generate_options_with_live, build_prompt_cache_request_context,
    is_prompt_primed_open_thinking,
};
use super::responses::{count_reasoning_tokens, persist_response};
use super::responses_auto_compact::{
    AutoCompaction, cap_output_tokens, clamp_reasoning_budget, final_status,
    mark_incomplete_output, prepend_compaction_item, send_compaction_events, stopped_at_limit,
};

/// Where one Responses generation delivers its events and its outcome.
///
/// The foreground SSE route implements it with [`SseResponseSink`]; every
/// caller of [`drive_response_run`] supplies one.
pub(crate) trait ResponseEventSink: Send + Sync {
    /// Deliver one event. Must not block on a slow reader unless it is the foreground SSE sink.
    fn emit(&self, event: &ResponseStreamEvent);
    /// First prefill-progress or token callback; fires at most once.
    fn on_started(&self);
    /// Called once before the terminal event, replacing the direct persist_response call.
    fn record_terminal(&self, response: &ResponseObject);
    /// Retire cancel registration; true when a cancel was acknowledged (replaces finish_in_flight).
    fn finish_registration(&self) -> bool;
    /// Generation is over: a reader leaving from here on is not a cancel.
    fn generation_finished(&self);
}

/// Everything the blocking generation task needs, produced by
/// [`prepare_response_run`] and consumed by [`drive_response_run`].
pub(crate) struct PreparedRun {
    pub(crate) state: AppState,
    pub(crate) live: Arc<LiveSettings>,
    pub(crate) request: Arc<CreateResponseRequest>,
    pub(crate) translated: Arc<TranslatedRequest>,
    pub(crate) response_id: String,
    pub(crate) model_id: String,
    pub(crate) created_at: f64,
    pub(crate) prepared: PreparedChatRequest,
    pub(crate) options: ServerGenerateOptions,
    pub(crate) queue_reservation: Option<SingleStreamQueueReservation>,
    pub(crate) echo_scope: Option<ReasoningEchoScope>,
    pub(crate) primed_open_thinking: bool,
    pub(crate) auto: Option<AutoCompaction>,
}

impl PreparedRun {
    /// Arm the dispatch-admission signal on this run's options. The returned
    /// waiter flips `cancelled` when dropped before admission, so it must be
    /// armed with the token later handed to [`drive_response_run`].
    pub(crate) fn arm_admission(&mut self, cancelled: CancellationToken) -> AdmissionWaiter {
        AdmissionWaiter::arm(&mut self.options, cancelled)
    }
}

/// Render, build options and the cache context, and reserve the queue slot
/// for one streamed Responses request.
///
/// Used by: the streaming `/v1/responses` handler.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn prepare_response_run(
    state: AppState,
    live: Arc<LiveSettings>,
    request: CreateResponseRequest,
    translated: TranslatedRequest,
    response_id: String,
    created_at: f64,
    priority: crate::server::batch::RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
    auto: Option<AutoCompaction>,
) -> Result<PreparedRun, ErrorResponse> {
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let model_id = state.display_model_id().to_string();
    let prompt_cache_enabled = state.prompt_cache.is_some();
    // Reasoning re-injection for content-only history (#2110, #2118), as on
    // `/v1/chat/completions`: only the render sees the filled copy; the cache
    // context, tool parsing and the record read the translated request as the
    // client sent it. A `reasoning` input item maps to `Message.reasoning`, so
    // client-sent reasoning still wins.
    let echo_scope =
        crate::server::reasoning_echo::chat_scope(&state, &live, &translated.chat_request);
    let render_request = crate::server::reasoning_echo::render_request(
        &state,
        echo_scope.as_ref(),
        &translated.chat_request,
    );
    let prepared = prepare_chat_request_with_cache_and_video_allowlist(
        &state.chat_template,
        &render_request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
        &state.video_dir_allowlist,
    )
    .await;
    let mut prepared = match prepared {
        Ok(prepared) => prepared,
        Err(err) => {
            return Err(ErrorResponse::new(err.to_string(), "invalid_request_error"));
        }
    };
    // Loop-detection amplifier signal (issues #967 and #977): the Responses
    // translator maps the tool fields onto the chat request. Structured output
    // alone no longer arms the family default.
    let amplified = chat_carries_loop_amplifier(&translated.chat_request);
    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    let mut options = build_generate_options_with_live(
        &translated.chat_request.params,
        &state.config,
        &live,
        amplified,
    );
    // Output cap at the remaining context, for `context_management`
    // requests only (issue #2397, mlx-vlm#2458).
    if let Some(auto) = auto.as_ref() {
        cap_output_tokens(&state, auto, &prepared, &mut options.max_tokens)?;
    }
    options.priority = priority;
    // per-request Gemma 4 image soft-token budget, resolved and validated from
    // the translated `image_url` content parts. `None` when unset.
    options.image_soft_tokens = prepared.image_soft_tokens;
    // A native chat renderer (Kimi K3's XTML format, #1338) produced the
    // prompt as token ids. Handing them to the provider is what keeps the
    // rendered structure intact: re-tokenizing `prepared.prompt` would have to
    // re-recognize control-token spellings, which is the injection surface the
    // native renderer closes. `None` for every template-rendered request.
    options.pre_rendered_prompt_tokens = prepared.prompt_token_ids.take();
    options.reasoning_budget = budget_override;
    // The budget was validated against the uncapped limit; keep it below the
    // capped one (issue #2397).
    if auto.is_some() {
        clamp_reasoning_budget(&mut options.reasoning_budget, options.max_tokens);
    }
    // The prompt may already leave generation inside an open thinking block
    // (a template that primes `<think>\n`, #2123): the scheduler's
    // `ThinkingState` then counts reasoning from the first token, as on
    // `/v1/chat/completions`.
    options.thinking_enter_block_on_start = primed_open_thinking;
    options.structured = structured;
    // Wire the prompt-prefix KV cache (epic #116) into the streaming Responses
    // path too, before `options`/`translated` move into the spawned task.
    options.prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &live,
        &translated.chat_request,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );

    let queue_reservation = match state.model_provider.reserve_single_stream_queue_slot() {
        Ok(reservation) => reservation,
        Err(err) => return Err(super::generation_error_to_response(err)),
    };

    Ok(PreparedRun {
        state,
        live,
        request: Arc::new(request),
        translated: Arc::new(translated),
        response_id,
        model_id,
        created_at,
        prepared,
        options,
        queue_reservation,
        echo_scope,
        primed_open_thinking,
        auto,
    })
}

/// The driver's own view of the sink: it forwards everything to the caller's
/// sink and keeps every `response.output_item.done` item with its
/// `output_index` (#2396), so the final response object is built from exactly
/// the items the run delivered whichever sink ran.
struct RunEvents {
    sink: Arc<dyn ResponseEventSink>,
    done_items: std::sync::Mutex<Vec<(usize, ResponseOutputItem)>>,
    started: AtomicBool,
}

impl RunEvents {
    fn new(sink: Arc<dyn ResponseEventSink>) -> Self {
        Self {
            sink,
            done_items: std::sync::Mutex::new(Vec::new()),
            started: AtomicBool::new(false),
        }
    }

    /// The completed output items this run delivered, in `output_index`
    /// order (#2396). Empty when no item was completed.
    fn streamed_output(&self) -> Vec<ResponseOutputItem> {
        let mut done = self
            .done_items
            .lock()
            .map(|done| done.clone())
            .unwrap_or_default();
        done.sort_by_key(|(index, _)| *index);
        done.into_iter().map(|(_, item)| item).collect()
    }

    /// Report the first prefill-progress or token callback to the sink once.
    fn mark_started(&self) {
        if !self.started.swap(true, Ordering::AcqRel) {
            self.sink.on_started();
        }
    }
}

impl ResponseEventSink for RunEvents {
    fn emit(&self, event: &ResponseStreamEvent) {
        if let ResponseStreamEvent::OutputItemDone {
            output_index, item, ..
        } = event
            && let Ok(mut done) = self.done_items.lock()
        {
            done.push((*output_index, item.clone()));
        }
        self.sink.emit(event);
    }

    fn on_started(&self) {
        self.mark_started();
    }

    fn record_terminal(&self, response: &ResponseObject) {
        self.sink.record_terminal(response);
    }

    fn finish_registration(&self) -> bool {
        self.sink.finish_registration()
    }

    fn generation_finished(&self) {
        self.sink.generation_finished();
    }
}

/// Run one prepared Responses generation to its terminal event on the
/// calling thread, delivering every event through `sink`.
///
/// Blocking: call it from `spawn_blocking`. `cancelled` aborts the
/// generation; a stop it caused is reported as `cancelled`.
///
/// Used by: the streaming `/v1/responses` handler.
pub(crate) fn drive_response_run(
    run: PreparedRun,
    cancelled: CancellationToken,
    sink: Arc<dyn ResponseEventSink>,
) {
    let PreparedRun {
        state,
        live,
        request,
        translated,
        response_id,
        model_id,
        created_at,
        prepared,
        options,
        queue_reservation,
        echo_scope,
        primed_open_thinking,
        auto,
    } = run;
    let events = RunEvents::new(sink);
    let parse_tools = tool_calls::should_parse_tool_calls(&translated.chat_request);
    let tools_for_parser = if parse_tools {
        translated.chat_request.tools.clone()
    } else {
        None
    };
    // The generation call consumes `cancelled`; keep a handle so the final
    // status can report a stop the token caused (a cancel call, a client
    // disconnect, or `DELETE /v1/stream`).
    let cancelled_for_status = cancelled.clone();

    let mut emitter = ResponseStreamEmitter::new();
    let initial_response = build_response_object(OutboundContext {
        response_id: response_id.clone(),
        model_id: model_id.clone(),
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
        max_tool_calls: request.max_tool_calls,
        request: &request,
        error: None,
        incomplete_reason: None,
        finish_reason: "in_progress".to_string(),
    });

    events.emit(&ResponseStreamEvent::Created {
        sequence_number: emitter.next_seq(),
        response: initial_response.clone(),
    });
    events.emit(&ResponseStreamEvent::InProgress {
        sequence_number: emitter.next_seq(),
        response: initial_response,
    });
    // Automatic compaction ran before the stream opened (issue #2397):
    // report it and emit its item at output_index 0.
    send_compaction_events(&events, &mut emitter, &response_id, auto.as_ref());

    // Lazy envelope state: we don't open the message or reasoning
    // item until the first matching chunk arrives, so we can emit
    // them in the order the model produces them (reasoning first
    // for Qwen3/DeepSeek priming, content first otherwise).
    // Slot accounting (#1440); see the non-streaming handler.
    let slot = state.slots.begin(
        &prepared.prompt,
        super::slots::slot_params_json(&options, true),
        Some(options.max_tokens as i64),
    );
    let accumulated_raw = Arc::new(std::sync::Mutex::new(String::new()));
    let acc_clone = accumulated_raw.clone();
    // A primed prompt starts the filter inside the thinking block, so the
    // close-only trace streams as reasoning rather than answer text
    // (#2123), matching the non-streaming split.
    let stream_filter = Arc::new(std::sync::Mutex::new(
        if primed_open_thinking {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        }
        // #2394: GLM-4.1V-Thinking's `<answer>` wrapper, as non-streamed.
        .with_answer_wrapper(tool_calls::answer_wrapper_enabled(&state.tokenizer)),
    ));
    let filter_for_callback = stream_filter.clone();
    // Incremental tool-call argument streaming (#2396), when tool calls
    // are parsed for this request. The end-of-stream parse stays
    // authoritative.
    let tool_stream = tools_for_parser.as_deref().map(|tools| {
        Arc::new(std::sync::Mutex::new(
            super::tool_call_stream::ResponsesToolCallStream::new(
                tools,
                translated
                    .chat_request
                    .tool_choice
                    .as_ref()
                    .and_then(|choice| choice.specific_function()),
                primed_open_thinking,
            ),
        ))
    });
    let tool_stream_for_callback = tool_stream.clone();
    let emitter_arc = Arc::new(std::sync::Mutex::new(emitter));
    let emitter_for_callback = emitter_arc.clone();

    let result = state
        .model_provider
        .generate_streaming_with_logprobs_cancellable_videos_declared_reserved_live_with_prefill(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
            prepared.media,
            queue_reservation,
            cancelled,
            &live,
            |token, _lp| {
                events.mark_started();
                slot.on_token(&token);
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
                        let placeholder = ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                            id: new_id.clone(),
                            status: ResponseItemStatus::InProgress,
                            content: vec![],
                        });
                        let seq = em.next_seq();
                        events.emit(&ResponseStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index: em.output_index(),
                            item: placeholder,
                        });
                        new_id
                    };
                    em.reasoning_text_acc.push_str(&text);
                    let seq = em.next_seq();
                    let out_idx = em.output_index();
                    events.emit(&ResponseStreamEvent::ReasoningTextDelta {
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
                        events.emit(&ResponseStreamEvent::ReasoningTextDone {
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
                        events.emit(&ResponseStreamEvent::OutputItemDone {
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
                        events.emit(&ResponseStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index: em.output_index(),
                            item: placeholder,
                        });
                        let seq = em.next_seq();
                        events.emit(&ResponseStreamEvent::ContentPartAdded {
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
                    events.emit(&ResponseStreamEvent::OutputTextDelta {
                        sequence_number: seq,
                        item_id: msg_id,
                        output_index: out_idx,
                        content_index: 0,
                        delta: text,
                    });
                }

                // Tool-call items and argument fragments (#2396).
                if let Some(tool_stream) = tool_stream_for_callback.as_ref()
                    && let Ok(mut tool_stream) = tool_stream.lock()
                    && let Ok(mut em) = emitter_for_callback.lock()
                {
                    for event in tool_stream.feed(&token, &mut em) {
                        events.emit(&event);
                    }
                }
            },
            |stats| {
                events.mark_started();
                slot.on_prefill_progress(stats.prompt_tokens, stats.cached_tokens, stats.processed);
            },
        );
    // Generation is over: from here a client disconnect is not a cancel.
    events.generation_finished();

    // Flush any buffered content out of the stream filter.
    let trailing = stream_filter
        .lock()
        .ok()
        .map(|mut f| f.flush())
        .unwrap_or_default();
    if let Some(text) = trailing.reasoning.filter(|s| !s.is_empty())
        && let Ok(mut em) = emitter_for_callback.lock()
    {
        let r_id = if let Some(id) = em.active_reasoning_id.clone() {
            id
        } else {
            let new_id = format!("rs_{}", short_uuid());
            em.open_reasoning(new_id.clone());
            let placeholder = ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                id: new_id.clone(),
                status: ResponseItemStatus::InProgress,
                content: vec![],
            });
            let seq = em.next_seq();
            events.emit(&ResponseStreamEvent::OutputItemAdded {
                sequence_number: seq,
                output_index: em.output_index(),
                item: placeholder,
            });
            new_id
        };
        em.reasoning_text_acc.push_str(&text);
        let seq = em.next_seq();
        let out_idx = em.output_index();
        events.emit(&ResponseStreamEvent::ReasoningTextDelta {
            sequence_number: seq,
            item_id: r_id,
            output_index: out_idx,
            content_index: 0,
            delta: text,
        });
    }
    if let Some(text) = trailing.content.filter(|s| !s.is_empty())
        && let Ok(mut em) = emitter_for_callback.lock()
    {
        let msg_id = if let Some(id) = em.active_message_id.clone() {
            id
        } else {
            let new_id = format!("msg_{}", short_uuid());
            em.open_message(new_id.clone());
            let placeholder = ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
                new_id.clone(),
                vec![],
            ));
            let seq = em.next_seq();
            events.emit(&ResponseStreamEvent::OutputItemAdded {
                sequence_number: seq,
                output_index: em.output_index(),
                item: placeholder,
            });
            let seq = em.next_seq();
            events.emit(&ResponseStreamEvent::ContentPartAdded {
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
        events.emit(&ResponseStreamEvent::OutputTextDelta {
            sequence_number: seq,
            item_id: msg_id,
            output_index: out_idx,
            content_index: 0,
            delta: text,
        });
    }

    if let Ok(r) = &result {
        slot.finish(
            r.prompt_tokens,
            r.cached_tokens,
            r.completion_tokens,
            &r.text,
        );
    }
    let result = match result {
        Ok(r) => r,
        Err(err) => {
            let cancel_acknowledged = events.finish_registration();
            // A cancel that lands before the first token ends generation
            // with an error. The client was told `cancelled`, so report
            // and keep exactly that. The conversation gets nothing: this
            // turn produced no output.
            if cancel_acknowledged {
                let mut response = build_response_object(OutboundContext {
                    response_id: response_id.clone(),
                    model_id,
                    created_at,
                    completed_at: created_at,
                    status: ResponseStatus::Cancelled,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    cached_tokens: 0,
                    reasoning_tokens: 0,
                    text: String::new(),
                    reasoning_text: None,
                    parsed_tool_calls: None,
                    max_tool_calls: request.max_tool_calls,
                    request: &request,
                    error: None,
                    incomplete_reason: None,
                    finish_reason: "cancelled".to_string(),
                });
                response.completed_at = None;
                // The assembly adds a message item even for empty text;
                // no item was generated here.
                response.output.clear();
                // Recorded before the terminal event, as on the normal path.
                events.record_terminal(&response);
                if let Ok(mut em) = emitter_for_callback.lock() {
                    let sequence_number = em.next_seq();
                    events.emit(&ResponseStreamEvent::Completed {
                        sequence_number,
                        response,
                    });
                }
                return;
            }
            let queue_full = err.downcast_ref::<QueueFullError>().is_some();
            if let Ok(mut em) = emitter_for_callback.lock() {
                let seq = em.next_seq();
                events.emit(&ResponseStreamEvent::Error {
                    sequence_number: seq,
                    code: if queue_full {
                        "server_overloaded".to_string()
                    } else {
                        "server_error".to_string()
                    },
                    message: if queue_full {
                        "All slots are busy. Please try again later.".to_string()
                    } else {
                        err.to_string()
                    },
                });
            }
            return;
        }
    };

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    let full_raw = accumulated_raw
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    let mut parsed_tools = if parse_tools {
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
        events.emit(&ResponseStreamEvent::ReasoningTextDone {
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
        events.emit(&ResponseStreamEvent::OutputItemDone {
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
        events.emit(&ResponseStreamEvent::OutputTextDone {
            sequence_number: seq,
            item_id: closed_msg_id.clone(),
            output_index: em.output_index(),
            content_index: 0,
            text: message_text.clone(),
        });
        let seq = em.next_seq();
        events.emit(&ResponseStreamEvent::ContentPartDone {
            sequence_number: seq,
            item_id: closed_msg_id.clone(),
            output_index: em.output_index(),
            content_index: 0,
            part: ResponseOutputContent::output_text(message_text.clone()),
        });
        let mut message = ResponseOutputMessage::new_assistant(
            closed_msg_id.clone(),
            vec![ResponseOutputContent::output_text(message_text.clone())],
        );
        // A `context_management` reply cut off at its output limit
        // (issue #2397).
        if stopped_at_limit(auto.as_ref(), &result.finish_reason) {
            message.status = ResponseItemStatus::Incomplete;
        }
        let msg_item = ResponseOutputItem::Message(message);
        let seq = em.next_seq();
        events.emit(&ResponseStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index: em.output_index(),
            item: msg_item,
        });
        em.advance_output_index();
    }

    let reasoning_text_for_response = em.completed_reasoning_text();
    // A message item a streamed tool call closed early (#2396) still
    // belongs to the response text.
    let message_text = match tool_stream.as_ref().and_then(|t| t.lock().ok()) {
        Some(tool_stream) => tool_stream.message_text(&message_text),
        None => message_text,
    };

    // Remember the streamed `reasoning` item for a follow-up that echoes
    // only the message item (#2118), from exactly the text the stream
    // delivered. Skipped for a tool-calling turn, as above.
    if !parsed_tools.as_ref().is_some_and(|p| p.has_tool_calls()) {
        crate::server::reasoning_echo::record_reply(
            &state,
            echo_scope.as_ref(),
            &translated.chat_request.messages,
            &message_text,
            reasoning_text_for_response.as_deref(),
        );
    }

    // #2396: streamed calls end with their parsed arguments under the
    // ids they streamed with; every other parsed call is emitted in one
    // shot. The ids are written back so the final object agrees.
    if let (Some(parsed), Some(tool_stream)) = (parsed_tools.as_mut(), tool_stream.as_ref())
        && let Ok(tool_stream) = tool_stream.lock()
    {
        for event in tool_stream.finish(&mut parsed.tool_calls, &mut em, &response_id) {
            events.emit(&event);
        }
    }

    // Final response object. The text is the message accumulator; the
    // outbound assembly fills the canonical output[] for the
    // "completed" event payload.
    let completed_at_now = chrono::Utc::now().timestamp() as f64;
    // Retiring the registry entry first closes the window where a cancel
    // would be answered `cancelled` after this status was decided.
    let cancel_acknowledged = events.finish_registration();
    // A cancelled stream ends through the normal path, so `final_status`
    // alone would record it as `completed` and contradict the cancel
    // reply. An acknowledged cancel wins even over a natural finish: the
    // client was already told `cancelled`.
    let (status, incomplete_reason) =
        if cancel_acknowledged || cancelled_for_status.load(Ordering::Relaxed) {
            (ResponseStatus::Cancelled, None)
        } else {
            final_status(auto.as_ref(), &result.finish_reason)
        };
    let mut final_response = build_response_object(OutboundContext {
        response_id: response_id.clone(),
        model_id,
        created_at,
        completed_at: completed_at_now,
        status,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        cached_tokens: result.cached_tokens,
        reasoning_tokens: count_reasoning_tokens(
            &state.tokenizer,
            reasoning_text_for_response.as_deref(),
        ),
        text: message_text,
        reasoning_text: reasoning_text_for_response,
        parsed_tool_calls: parsed_tools.as_ref(),
        max_tool_calls: request.max_tool_calls,
        request: &request,
        error: None,
        incomplete_reason,
        finish_reason: result.finish_reason.clone(),
    });

    // #2396: with tool-call streaming a call can split the message into
    // segments and its item is added mid-stream, so the completed object
    // (and the stored copy `previous_response_id` reads) lists exactly
    // the items the stream delivered: same ids, order, `output_index`
    // and content, with the parsed arguments the `.done` events carried.
    if tool_stream.is_some() {
        let streamed = events.streamed_output();
        if !streamed.is_empty() {
            final_response.output_text = streamed_output_text(&streamed);
            final_response.output = streamed;
        }
    }

    prepend_compaction_item(auto.as_ref(), &mut final_response);
    mark_incomplete_output(&mut final_response);
    if status == ResponseStatus::Cancelled {
        final_response.completed_at = None;
    }

    // Stored before the terminal event: a slow client must not delay the
    // record, and whoever reads the event can already retrieve it.
    events.record_terminal(&final_response);

    // The OpenAI event set has no cancelled terminal event: a cancelled
    // stream ends with `response.completed` and `status: cancelled`.
    let seq = em.next_seq();
    events.emit(&match status {
        ResponseStatus::Incomplete => ResponseStreamEvent::Incomplete {
            sequence_number: seq,
            response: final_response,
        },
        _ => ResponseStreamEvent::Completed {
            sequence_number: seq,
            response: final_response,
        },
    });
}

/// The `output_text` aggregate of `items`: every message's `output_text`
/// parts, in order.
fn streamed_output_text(items: &[ResponseOutputItem]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            ResponseOutputItem::Message(message) => Some(&message.content),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            ResponseOutputContent::OutputText { text, .. } => Some(text.as_str()),
            ResponseOutputContent::Refusal { .. } => None,
        })
        .collect()
}

/// The foreground sink: events go to the client's SSE channel, the terminal
/// response is persisted the way the streaming route always has, and the
/// in-flight cancel registration is the response store's.
pub(crate) struct SseResponseSink {
    /// The sender used while generation runs: a closed channel flips the
    /// request's cancellation token, so a disconnect aborts generation.
    sender: ResponseStreamSender,
    /// The sender for the events after generation, whose closed channel
    /// cancels nothing ([`ResponseStreamSender::after_generation`]).
    after_generation: ResponseStreamSender,
    generation_over: AtomicBool,
    state: AppState,
    request: Arc<CreateResponseRequest>,
    translated: Arc<TranslatedRequest>,
    response_id: String,
}

impl SseResponseSink {
    /// Wrap `sender` for `run`, and register `cancelled` in the response
    /// store's in-flight registry so `POST /v1/responses/:id/cancel` can
    /// abort this response from another request.
    ///
    /// Used by: the streaming `/v1/responses` handler.
    pub(crate) fn new(
        run: &PreparedRun,
        sender: ResponseStreamSender,
        cancelled: &CancellationToken,
    ) -> Self {
        // Review H2: register this response in the in-flight registry so
        // POST /v1/responses/:id/cancel can abort it from a different
        // request thread. We OR the existing client-disconnect token with
        // the in-flight token by sharing the same Arc<AtomicBool>: the
        // SSE channel already owns one (`cancelled`), and we additionally
        // expose it via the store under the response id. When the cancel
        // handler flips it, the scheduler observes the abort on its next
        // poll, just like the client-disconnect case.
        if let Some(store) = run.state.responses_store.as_ref() {
            match store.in_flight_write() {
                Ok(mut g) => {
                    g.insert(run.response_id.clone(), cancelled.clone());
                }
                Err(_) => {
                    let response_id = &run.response_id;
                    tracing::warn!(
                        "responses_store in-flight registry is poisoned; cancellation API will not work for {response_id}"
                    );
                }
            }
        }
        Self {
            after_generation: sender.after_generation(),
            sender,
            generation_over: AtomicBool::new(false),
            state: run.state.clone(),
            request: Arc::clone(&run.request),
            translated: Arc::clone(&run.translated),
            response_id: run.response_id.clone(),
        }
    }
}

impl ResponseEventSink for SseResponseSink {
    fn emit(&self, event: &ResponseStreamEvent) {
        let sender = if self.generation_over.load(Ordering::Acquire) {
            &self.after_generation
        } else {
            &self.sender
        };
        let _ = sender.send_event(event);
    }

    fn on_started(&self) {}

    fn record_terminal(&self, response: &ResponseObject) {
        if response.output.is_empty() {
            // A cancel acknowledged before the first token: the stored
            // record is kept, but the conversation gets nothing, since this
            // turn produced no output. Every generated response carries at
            // least one output item, so only that path reaches here.
            if self.translated.effective_store
                && let Some(store) = self.state.responses_store.as_ref()
            {
                store.insert(
                    response.id.clone(),
                    crate::server::responses_store::StoredResponse {
                        response: response.clone(),
                        input_items: self.translated.canonical_input_items.clone(),
                    },
                );
            }
            return;
        }
        persist_response(&self.state, &self.request, &self.translated, response);
    }

    fn finish_registration(&self) -> bool {
        self.state
            .responses_store
            .as_ref()
            .is_some_and(|store| store.finish_in_flight(&self.response_id))
    }

    fn generation_finished(&self) {
        self.generation_over.store(true, Ordering::Release);
    }
}

#[cfg(test)]
#[path = "responses_run_tests.rs"]
mod responses_run_tests;
