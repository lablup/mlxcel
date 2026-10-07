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

//! The request half of `/v1/chat/completions`, shared by the HTTP handler and
//! the in-process server `mlxcel run` drives (issue #2173).
//!
//! [`admit_chat_request`] is the handler's validation prefix (empty input,
//! logprob and sampler ranges, media capability, video expansion, tool inputs,
//! the thinking budget and the structured-output constraint).
//! [`prepare_chat_generation`] renders the conversation through the chat
//! template and builds the [`ServerGenerateOptions`] the model worker runs.
//! The streaming and non-streaming handlers and
//! [`crate::server::in_process::InProcessServer::chat`] all call both, so a
//! chat turn typed into `mlxcel run` reaches the worker as the same request a
//! client posting the same messages would send.

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use mlxcel_core::sampling::LogprobsConfig;

use super::chat::{
    build_chat_constraint, build_prompt_cache_request_context, is_prompt_primed_open_thinking,
    primed_open_thinking_close_marker, validate_chat_tool_inputs, validate_top_n_sigma,
    validate_typical_p, validate_xtc_params,
};
use crate::server::batch::RequestPriority;
use crate::server::chat_request::{
    PreparedChatRequest, prepare_chat_request_with_cache, request_has_effective_input,
};
use crate::server::config::{PromptCacheRequestContext, ReasoningBudgetOverride};
use crate::server::reasoning_echo::ReasoningEchoScope;
use crate::server::request_options::{
    chat_carries_loop_amplifier, resolve_server_max_tokens_with_live,
};
use crate::server::structured::StructuredOutputConstraint;
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::{ChatCompletionRequest, ErrorResponse};
use crate::server::{AppState, LiveSettings, ServerGenerateOptions};

/// What [`admit_chat_request`] resolved for a request it accepted.
pub(crate) struct ChatAdmission {
    /// The live-settings snapshot the request runs under.
    pub(crate) live: Arc<LiveSettings>,
    /// The request's thinking budget, validated against its `max_tokens`.
    pub(crate) budget_override: ReasoningBudgetOverride,
    /// The structured-output constraint (`response_format` or a forced tool
    /// call), or `None` for unconstrained generation.
    pub(crate) structured: Option<Arc<Mutex<StructuredOutputConstraint>>>,
}

/// Validate a chat request and resolve its per-request constraints, in the
/// order `/v1/chat/completions` always has. A video block on a checkpoint
/// without a temporal tower is expanded into stills here, so the request the
/// caller renders afterwards is an ordinary multi-image one.
pub(crate) async fn admit_chat_request(
    state: &AppState,
    request: &mut ChatCompletionRequest,
) -> Result<ChatAdmission, ErrorResponse> {
    let live = state.live();
    // Reject requests with no effective input before any other validation or
    // model dispatch (issue #773).
    if !request_has_effective_input(request) {
        return Err(ErrorResponse::new(
            "Request must include at least one non-empty message content or media input.",
            "invalid_request_error",
        ));
    }
    if let Some(top) = request.top_logprobs
        && top > 20
    {
        return Err(ErrorResponse::new(
            "top_logprobs must be between 0 and 20",
            "invalid_request_error",
        ));
    }
    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return Err(ErrorResponse::new(
            "top_logprobs requires logprobs to be set to true",
            "invalid_request_error",
        ));
    }
    if let Err(message) =
        validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability)
    {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    if let Err(message) = validate_top_n_sigma(request.params.top_n_sigma) {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    if let Err(message) = validate_typical_p(request.params.typical_p) {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    // Media the checkpoint cannot consume is refused before any referenced URL
    // or file is read (issue #1451).
    if let Some(rejection) = crate::server::media_capability_rejection(
        request,
        state.media_support,
        state.display_model_id(),
    ) {
        return Err(rejection);
    }
    // Video read as ordered stills on a checkpoint with a vision tower only
    // (issue #1322), before the template renders.
    if let Err(err) = crate::server::chat_request::expand_request_video_parts(state, request).await
    {
        return Err(err.into_error_response());
    }
    if let Err(message) = validate_chat_tool_inputs(request) {
        return Err(ErrorResponse::new(message, "invalid_request_error"));
    }
    let effective_max_tokens =
        resolve_server_max_tokens_with_live(&state.config, &live, request.params.max_tokens);
    let raw_budget = pick_budget_alias(
        request.params.thinking_budget_tokens,
        request.params.thinking_token_budget,
        request.params.thinking_budget,
    );
    let budget_override =
        match resolve_request_budget(raw_budget, live.reasoning_budget, effective_max_tokens) {
            Ok(effective) => ReasoningBudgetOverride::Explicit(effective),
            Err(err) => {
                return Err(ErrorResponse::new(err.to_string(), "invalid_request_error"));
            }
        };
    // Grammar compilation runs on the blocking pool inside.
    let structured = build_chat_constraint(state, request).await?;
    Ok(ChatAdmission {
        live,
        budget_override,
        structured,
    })
}

/// A rendered chat request ready for the model worker.
pub(crate) struct ChatGeneration<'r> {
    /// The request as the template saw it: the client's request, or a copy
    /// with stored reasoning re-injected into content-only history (#2110).
    pub(crate) render_request: Cow<'r, ChatCompletionRequest>,
    /// The reasoning re-echo scope, when the store applies to this request.
    pub(crate) echo_scope: Option<ReasoningEchoScope>,
    /// The rendered prompt and the resolved media. `prompt_token_ids` has
    /// already moved into [`Self::options`].
    pub(crate) prepared: PreparedChatRequest,
    /// The worker options, with the prompt-cache context, thinking state,
    /// constraint and logprob settings attached.
    pub(crate) options: ServerGenerateOptions,
    /// The prompt-cache context without its history string, kept for the
    /// post-completion next-turn warm-up (issue #1144).
    pub(crate) warmup_ctx: Option<PromptCacheRequestContext>,
    /// Whether the rendered prompt left the model inside an open thinking
    /// block, so the stream filter starts in the reasoning channel.
    pub(crate) primed_open_thinking: bool,
    /// The close marker of the block the prompt primed open, if any (#1470).
    pub(crate) primed_close_marker: Option<String>,
    /// b10621 `echo` (#1470): the prefilled assistant text to lead the reply.
    pub(crate) echo_prefill: Option<String>,
}

/// Render `request` and build its worker options, the way both chat handlers
/// do. `admission` is what [`admit_chat_request`] returned for the request.
pub(crate) async fn prepare_chat_generation<'r>(
    state: &AppState,
    request: &'r ChatCompletionRequest,
    priority: RequestPriority,
    admission: ChatAdmission,
) -> Result<ChatGeneration<'r>, ErrorResponse> {
    let ChatAdmission {
        live,
        budget_override,
        structured,
    } = admission;
    // The prefix-stable rendering path (unset `preserve_thinking` defaults to
    // true) follows whether the prompt-prefix cache is installed.
    let prompt_cache_enabled = state.prompt_cache.is_some();
    // Re-inject reasoning this server generated for content-only assistant
    // turns (issue #2110). Only the render sees the filled copy: the cache
    // context, tool parsing and the echo record read the request as sent.
    let echo_scope = crate::server::reasoning_echo::chat_scope(state, &live, request);
    let render_request =
        crate::server::reasoning_echo::render_request(state, echo_scope.as_ref(), request);
    let mut prepared = prepare_chat_request_with_cache(
        &state.chat_template,
        &render_request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
    )
    .await
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))?;
    let echo_prefill = request
        .resolve_echo()
        .then(|| prepared.assistant_prefill.clone())
        .flatten();
    // Built after preparation so the multimodal digest sees the resolved
    // image and audio bytes.
    let prompt_cache_ctx = build_prompt_cache_request_context(
        state,
        &live,
        request,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );
    let warmup_ctx = prompt_cache_ctx.as_ref().map(|ctx| {
        let mut c = ctx.clone();
        c.history_prompt = None;
        c
    });
    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    let primed_close_marker =
        primed_open_thinking_close_marker(&state.thinking_markers, &prepared.prompt);
    // Loop-detection amplifier (issues #967 and #977): only tool-shaped
    // prompts arm the family default.
    let amplified = chat_carries_loop_amplifier(request);
    let mut options = super::chat::build_generate_options_with_live(
        &request.params,
        &state.config,
        &live,
        amplified,
    );
    options.priority = priority;
    options.reasoning_budget = budget_override;
    options.prompt_cache_ctx = prompt_cache_ctx;
    // Per-request Gemma 4 image soft-token budget, validated by preparation.
    options.image_soft_tokens = prepared.image_soft_tokens;
    // A native chat renderer (Kimi K3, #1338) produced token ids; handing
    // them over keeps control-token structure out of re-tokenization.
    options.pre_rendered_prompt_tokens = prepared.prompt_token_ids.take();
    // `ThinkingState` counts reasoning from the first decoded token only when
    // the prompt left the model inside an open thinking block.
    options.thinking_enter_block_on_start = primed_open_thinking;
    options.structured = structured;
    if request.logprobs == Some(true) {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k: request.top_logprobs.unwrap_or(0) as usize,
            source: Default::default(),
        };
    }
    Ok(ChatGeneration {
        render_request,
        echo_scope,
        prepared,
        options,
        warmup_ctx,
        primed_open_thinking,
        primed_close_marker,
        echo_prefill,
    })
}
