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

//! One chat turn through the in-process server (issue #2173).
//!
//! [`InProcessServer::chat`] is the streaming `/v1/chat/completions` handler
//! without the SSE writer: it admits and renders the request with
//! [`admit_chat_request`] and [`prepare_chat_generation`], submits it to the
//! model worker with a cancellation flag, splits the token stream into content
//! and reasoning through the server's [`StreamFilter`], parses tool calls at
//! the end with the server's parsers, records the reasoning re-echo entry
//! (#2110) and submits the next-turn prompt-cache warm-up (#1144). What it
//! hands back instead of SSE frames is [`ChatDelta`] callbacks and a
//! [`ChatTurn`] summary.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};

use super::InProcessServer;
use crate::server::GenerationResult;
use crate::server::batch::RequestPriority;
use crate::server::routes::chat::submit_next_turn_warmup;
use crate::server::routes::chat_generation::{
    ChatGeneration, admit_chat_request, prepare_chat_generation,
};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::{FilterOutput, StreamFilter};
use crate::server::types::{ChatCompletionRequest, CompletionRequest, ErrorResponse};

/// A piece of a streamed assistant turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatDelta<'a> {
    /// Text for the content channel.
    Content(&'a str),
    /// Text for the reasoning channel (the model's thinking block).
    Reasoning(&'a str),
}

/// A tool call the server's parsers found in the turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurnToolCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
}

/// What one chat turn produced.
#[derive(Debug)]
pub struct ChatTurn {
    /// Everything the content channel received, as streamed.
    pub content: String,
    /// Everything the reasoning channel received, or `None` when empty.
    pub reasoning: Option<String>,
    /// Tool calls parsed from the turn (only when the request declared tools).
    pub tool_calls: Vec<ChatTurnToolCall>,
    /// OpenAI finish reason (`stop`, `length`, `tool_calls`, ...).
    pub finish_reason: String,
    /// Whether the caller's cancellation flag stopped the turn early.
    pub cancelled: bool,
    /// The worker's own result: token ids, counts and timings.
    pub result: GenerationResult,
}

/// Parse a chat request from its `/v1/chat/completions` JSON body. The CLI
/// builds its requests as JSON so every flag it maps lands on the request
/// field a client would set, through the same deserializer.
pub fn chat_request_from_json(body: serde_json::Value) -> Result<ChatCompletionRequest> {
    serde_json::from_value(body).map_err(|e| anyhow!("invalid chat request: {e}"))
}

/// Parse a raw completion request from its `/v1/completions` JSON body, the
/// form `--no-chat-template` sends.
pub fn completion_request_from_json(body: serde_json::Value) -> Result<CompletionRequest> {
    serde_json::from_value(body).map_err(|e| anyhow!("invalid completion request: {e}"))
}

fn error_to_anyhow(err: ErrorResponse) -> anyhow::Error {
    anyhow!("{}", err.error.message)
}

/// Content and reasoning collected while a turn streams.
struct TurnText {
    raw: String,
    content: String,
    reasoning: String,
}

impl TurnText {
    fn deliver<F: FnMut(ChatDelta<'_>)>(&mut self, emit: FilterOutput, on_delta: &mut F) {
        if let Some(text) = emit.reasoning.as_deref()
            && !text.is_empty()
        {
            self.reasoning.push_str(text);
            on_delta(ChatDelta::Reasoning(text));
        }
        if let Some(text) = emit.content.as_deref()
            && !text.is_empty()
        {
            self.content.push_str(text);
            on_delta(ChatDelta::Content(text));
        }
    }
}

fn passthrough(token: &str) -> FilterOutput {
    FilterOutput {
        content: Some(token.to_string()),
        reasoning: None,
        suppressed_positions: 0,
        consumed_positions: 1,
        thinking_open: None,
        thinking_close: None,
    }
}

impl InProcessServer {
    /// Run one chat turn and stream its text through `on_delta`.
    ///
    /// Setting `cancel` stops the turn through the worker's cancellation path
    /// (the one a disconnected HTTP client triggers); the turn then returns
    /// with [`ChatTurn::cancelled`] set and whatever text it produced.
    pub fn chat<F>(
        &self,
        mut request: ChatCompletionRequest,
        cancel: Arc<AtomicBool>,
        mut on_delta: F,
    ) -> Result<ChatTurn>
    where
        F: FnMut(ChatDelta<'_>),
    {
        let state = self.state();
        if let Some(reason) = self.chat_unavailable_reason() {
            anyhow::bail!(reason);
        }
        let admission = self
            .runtime()
            .block_on(admit_chat_request(state, &mut request))
            .map_err(error_to_anyhow)?;
        let live = admission.live.clone();
        let generation = self
            .runtime()
            .block_on(prepare_chat_generation(
                state,
                &request,
                RequestPriority::Normal,
                admission,
            ))
            .map_err(error_to_anyhow)?;
        let ChatGeneration {
            render_request,
            echo_scope,
            prepared,
            options,
            warmup_ctx,
            primed_open_thinking,
            echo_prefill,
            ..
        } = generation;

        let skip_chat_parsing = state.config.skip_chat_parsing;
        let parse_tools = !skip_chat_parsing && tool_calls::should_parse_tool_calls(&request);
        let mut filter = if primed_open_thinking {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        };
        let mut text = TurnText {
            raw: String::new(),
            content: String::new(),
            reasoning: String::new(),
        };
        if let Some(prefill) = echo_prefill.as_deref() {
            text.deliver(passthrough(prefill), &mut on_delta);
        }
        let reservation = state.model_provider.reserve_single_stream_queue_slot()?;
        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos_declared_reserved_live_with_prefill(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                prepared.videos,
                prepared.media,
                reservation,
                cancel.clone(),
                &live,
                |token, _logprobs| {
                    if parse_tools {
                        text.raw.push_str(&token);
                    }
                    let emit = if skip_chat_parsing {
                        passthrough(&token)
                    } else {
                        filter.feed(&token)
                    };
                    text.deliver(emit, &mut on_delta);
                },
                |_prefill| {},
            )?;
        if !skip_chat_parsing {
            text.deliver(filter.flush(), &mut on_delta);
        }

        let mut finish_reason = result.finish_reason.clone();
        let mut calls = Vec::new();
        if parse_tools {
            let parsed = tool_calls::parse_tool_calls(&text.raw, request.tools.as_deref());
            if parsed.has_tool_calls() {
                let specific = request
                    .tool_choice
                    .as_ref()
                    .and_then(|choice| choice.specific_function())
                    .map(str::to_string);
                calls = parsed
                    .tool_calls
                    .into_iter()
                    .filter(|call| specific.as_deref().is_none_or(|name| call.name == name))
                    .map(|call| ChatTurnToolCall {
                        name: call.name,
                        arguments: call.arguments,
                    })
                    .collect();
                if !calls.is_empty() {
                    finish_reason = "tool_calls".to_string();
                }
            }
        }

        let cancelled = cancel.load(Ordering::Acquire);
        // The reasoning re-echo record (#2110) and the next-turn warm-up
        // (#1144) follow the streaming handler: a finished, non-tool turn only.
        if !cancelled && finish_reason != "tool_calls" {
            let mut stored_reasoning = None;
            if let Some(scope) = echo_scope.as_ref()
                && state.reasoning_echo.record(
                    scope,
                    &request.messages,
                    &text.content,
                    &text.reasoning,
                )
            {
                stored_reasoning = Some(text.reasoning.clone());
            }
            if let Some(ctx) = warmup_ctx.as_ref()
                && !crate::server::prompt_cache::boundary_snapshot_disabled()
                && !tool_calls::should_parse_tool_calls(&request)
            {
                submit_next_turn_warmup(
                    state,
                    &live,
                    &render_request,
                    ctx,
                    &text.content,
                    stored_reasoning.as_deref(),
                );
            }
        }

        Ok(ChatTurn {
            reasoning: (!text.reasoning.is_empty()).then_some(text.reasoning),
            content: text.content,
            tool_calls: calls,
            finish_reason,
            cancelled,
            result,
        })
    }

    /// Run one raw completion (`/v1/completions`, no chat template) and
    /// stream its text through `on_delta` as content. Used by
    /// `--no-chat-template`, which asks for the prompt verbatim.
    pub fn complete<F>(
        &self,
        request: CompletionRequest,
        cancel: Arc<AtomicBool>,
        mut on_delta: F,
    ) -> Result<ChatTurn>
    where
        F: FnMut(ChatDelta<'_>),
    {
        let state = self.state();
        if let Some(reason) = self.chat_unavailable_reason() {
            anyhow::bail!(reason);
        }
        let live = state.live();
        // The completions route's option build: no tool-shaped prompt, no
        // thinking priming, and the server-wide prompt-cache switch.
        let mut options = crate::server::routes::chat::build_generate_options_with_live(
            &request.params,
            &state.config,
            &live,
            false,
        );
        options.thinking_enter_block_on_start = false;
        options.prompt_cache_ctx =
            crate::server::routes::chat::build_raw_prompt_cache_context(state, None);
        let mut text = TurnText {
            raw: String::new(),
            content: String::new(),
            reasoning: String::new(),
        };
        let reservation = state.model_provider.reserve_single_stream_queue_slot()?;
        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos_declared_reserved_live_with_prefill(
                request.prompt,
                options,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Default::default(),
                reservation,
                cancel.clone(),
                &live,
                |token, _logprobs| text.deliver(passthrough(&token), &mut on_delta),
                |_prefill| {},
            )?;
        Ok(ChatTurn {
            content: text.content,
            reasoning: None,
            tool_calls: Vec::new(),
            finish_reason: result.finish_reason.clone(),
            cancelled: cancel.load(Ordering::Acquire),
            result,
        })
    }
}
