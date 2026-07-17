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

//! OpenAI-compatible chat completions adapter.
//!
//! This file should stay thin: it flattens the HTTP request, delegates prompt
//! preparation and option merging to shared helpers, and streams chunk payloads
//! back through `server/streaming.rs`.

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};

use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};

use crate::server::batch::RequestPriority;
use crate::server::chat_request::{prepare_chat_request_with_cache, request_has_effective_input};
use crate::server::chat_template_kwargs::{extract_request_kwargs, merge_server_and_request};
use crate::server::config::{PromptCacheRequestContext, ReasoningBudgetOverride};
use crate::server::prompt_cache::key::{
    multimodal_digest_from_vecs, resolve_session_key, template_sig,
};
use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::streaming::sse_channel;
use crate::server::structured::{StructuredOutputError, build_constraint_from_response_format};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::response::{ChatLogprobs, TokenLogprob, TopLogprob};
use crate::server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse,
    SamplingParams,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};
use crate::tokenizer::MlxcelTokenizer;

/// Map a [`StructuredOutputError`] to an HTTP error response.
///
/// Schema-shape problems and unsupported variants are 400; anything else
/// (matcher build failures, tokenizer adaptation issues) is 500.
///
/// `SchemaTooLarge` is treated as 400 (`invalid_request_error`) because the
/// fault is in the user-supplied schema. The error message — which already
/// avoids leaking llguidance internals — is safe to surface to the client
/// directly so they can correct their input.
pub(crate) fn structured_error_to_response(err: StructuredOutputError) -> ErrorResponse {
    match err {
        StructuredOutputError::InvalidRequest(_)
        | StructuredOutputError::InvalidSchema(_)
        | StructuredOutputError::SchemaTooLarge(_) => {
            ErrorResponse::new(err.to_string(), "invalid_request_error")
        }
        StructuredOutputError::UnsupportedTokenizer(_) | StructuredOutputError::Matcher(_) => {
            ErrorResponse::new(err.to_string(), "server_error")
        }
    }
}

/// Build the per-request prompt-cache context.
///
/// Returns `None` when the store is not installed on `state`, signalling to
/// the scheduler that no cache lookup or donate-back should run for this
/// request. When `Some`, the caller fills `options.prompt_cache_ctx` with
/// the returned value so the scheduler can compose a stable
/// [`crate::server::prompt_cache::key::PromptCacheKey`] on its own thread.
///
/// `image_data` / `audio_data` are the **resolved** multimodal byte payloads
/// (post base64-decode / file-read / URL-fetch) from
/// [`prepare_chat_request_with_cache`]. They are folded into the key via a
/// [`crate::server::prompt_cache::key::MultimodalDigest`] so a text-only prefix
/// can never collide with an image/audio one. Text-only requests pass empty
/// slices, which yields `MultimodalDigest::empty()` and a key byte-identical to
/// the pre-#124 path. Callers must therefore build the context **after**
/// preparing the request so the resolved bytes are available.
pub(crate) fn build_prompt_cache_request_context(
    state: &AppState,
    request: &ChatCompletionRequest,
    image_data: &[Vec<u8>],
    audio_data: &[Vec<u8>],
) -> Option<PromptCacheRequestContext> {
    state.prompt_cache.as_ref()?;
    // Mirror the kwargs merge that `prepare_chat_request_with_cache` performs
    // so the digest sees the same canonicalized map as the rendering pipeline.
    let merged_extra_body = request.merged_extra_body();
    let per_request_kwargs = extract_request_kwargs(
        request.chat_template_kwargs.as_ref(),
        merged_extra_body.as_ref(),
    );
    let merged_kwargs = merge_server_and_request(
        state.config.chat_template_kwargs.as_ref(),
        &per_request_kwargs,
    );

    let template_signature = template_sig(
        state.chat_template.template_source(),
        &merged_kwargs,
        request.tool_choice.as_ref(),
        request.tools.as_deref(),
    );
    let session_key =
        resolve_session_key(request.resolve_prompt_cache_key(), request.resolve_user()).to_string();
    // Digest the resolved multimodal payload. Empty slices (text-only) hash to
    // `MultimodalDigest::empty()`, leaving the composed key unchanged.
    let mm_digest = multimodal_digest_from_vecs(image_data, audio_data);
    Some(PromptCacheRequestContext {
        model_id: state.display_model_id().to_string(),
        lora_id: None,
        template_sig: template_signature,
        session_key,
        mm_digest,
    })
}

/// Decode a single token ID to its text representation using the tokenizer.
pub(crate) fn decode_token(tokenizer: &MlxcelTokenizer, token_id: i32) -> String {
    tokenizer
        .decode(&[token_id as u32], false)
        .unwrap_or_default()
}

/// Convert a `TokenLogprobData` to a `TokenLogprob` response struct.
///
/// `top_k` controls how many top alternatives to include. Pass 0 to include
/// none (only the selected token's logprob will be in the response).
pub(crate) fn token_lp_to_response(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
) -> TokenLogprob {
    let token_text = decode_token(tokenizer, lp.token_id);
    let bytes = token_text.as_bytes().to_vec();

    let top_logprobs: Vec<TopLogprob> = lp
        .top_alternatives
        .iter()
        .take(top_k)
        .map(|&(alt_id, alt_lp)| {
            let alt_text = decode_token(tokenizer, alt_id);
            let alt_bytes = alt_text.as_bytes().to_vec();
            TopLogprob {
                token: alt_text,
                logprob: alt_lp,
                bytes: Some(alt_bytes),
            }
        })
        .collect();

    TokenLogprob {
        token: token_text,
        logprob: lp.logprob,
        bytes: Some(bytes),
        top_logprobs,
    }
}

/// Build a `ChatLogprobs` from a list of `TokenLogprobData`.
pub(crate) fn build_chat_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp_data: &[TokenLogprobData],
    top_k: usize,
) -> ChatLogprobs {
    let content = lp_data
        .iter()
        .map(|lp| token_lp_to_response(tokenizer, lp, top_k))
        .collect();
    ChatLogprobs {
        content: Some(content),
    }
}

/// Build a single-token `ChatLogprobs` for streaming chunks.
pub(crate) fn build_single_token_chat_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
) -> ChatLogprobs {
    ChatLogprobs {
        content: Some(vec![token_lp_to_response(tokenizer, lp, top_k)]),
    }
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Reject requests with no effective input before any other validation or
    // model dispatch (issue #773): an empty `messages` array, or messages
    // whose content is empty/whitespace-only with no media/tool/reasoning
    // payload, would otherwise reach the model worker and waste a prefill.
    if !request_has_effective_input(&request) {
        return ErrorResponse::new(
            "Request must include at least one non-empty message content or media input.",
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate top_logprobs range per OpenAI spec (0-20)
    if let Some(top) = request.top_logprobs
        && top > 20
    {
        return ErrorResponse::new(
            "top_logprobs must be between 0 and 20",
            "invalid_request_error",
        )
        .into_response();
    }
    // top_logprobs requires logprobs: true
    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return ErrorResponse::new(
            "top_logprobs requires logprobs to be set to true",
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate XTC (Exclude Top Choices) sampling parameter ranges before any
    // generation work begins.
    if let Err(message) =
        validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability)
    {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    // reject `video_url` content blocks early for models that do
    // not support video. Detected once at startup from `config.json` and
    // cached on `AppState.media_support`. Returning a 400 keeps the client
    // contract honest — silently dropping video frames would still consume
    // tokens and produce a confusing reply.
    if !state.media_support.video && request_has_video_blocks(&request) {
        return ErrorResponse::new(
            format!(
                "video_url content blocks are not supported by model '{}'",
                state.display_model_id()
            ),
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate tool_choice values
    if let Some(ref tc) = request.tool_choice {
        match tc {
            crate::server::types::request::ToolChoice::Mode(mode) => {
                if !["auto", "none", "required"].contains(&mode.as_str()) {
                    return ErrorResponse::new(
                        format!("Invalid tool_choice value: '{mode}'. Must be 'auto', 'none', 'required', or a function object."),
                        "invalid_request_error",
                    )
                    .into_response();
                }
            }
            crate::server::types::request::ToolChoice::Specific(_) => {}
        }
    }

    // Enforce tools array size limit to prevent DoS via template rendering
    if let Some(ref tools) = request.tools
        && tools.len() > MAX_TOOLS
    {
        return ErrorResponse::new(
            format!(
                "Too many tools: {}. Maximum allowed is {MAX_TOOLS}.",
                tools.len()
            ),
            "invalid_request_error",
        )
        .into_response();
    }

    // validate thinking_budget_tokens early so malformed values
    // surface as 400 before any generation work begins.
    let effective_max_tokens = request
        .params
        .max_tokens
        .unwrap_or(state.config.default_max_tokens);
    let raw_budget = pick_budget_alias(
        request.params.thinking_budget_tokens,
        request.params.thinking_token_budget,
        request.params.thinking_budget,
    );
    let budget_override = match resolve_request_budget(
        raw_budget,
        state.config.reasoning_budget,
        effective_max_tokens,
    ) {
        Ok(effective) => {
            if raw_budget.is_some() {
                ReasoningBudgetOverride::Explicit(effective)
            } else {
                ReasoningBudgetOverride::InheritServerDefault
            }
        }
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };

    // + H2: build the structured-output constraint up
    // front so any schema validation error surfaces as a 400 before
    // generation work starts. Grammar compilation can be ~hundreds of ms
    // and (worst case, before the size guard) hundreds of MB — running
    // it directly on the Tokio runtime worker thread would block other
    // in-flight requests. We move it onto a blocking task and await the
    // join handle. Returns `None` when the request did not ask for
    // structured output, in which case the rest of the pipeline behaves
    // identically to before this issue.
    let structured = {
        let tokenizer = state.tokenizer.clone();
        let response_format = request.response_format.clone();
        match tokio::task::spawn_blocking(move || {
            build_constraint_from_response_format(tokenizer.as_ref(), response_format.as_ref())
        })
        .await
        {
            Ok(Ok(opt)) => opt,
            Ok(Err(err)) => return structured_error_to_response(err).into_response(),
            Err(join_err) => {
                tracing::error!("structured-output build task panicked: {join_err}");
                return ErrorResponse::new("structured-output preparation failed", "server_error")
                    .into_response();
            }
        }
    };

    let priority = parse_priority_header(&headers);
    if request.stream {
        stream_chat_completion(state, request, priority, budget_override, structured).await
    } else {
        non_stream_chat_completion(state, request, priority, budget_override, structured)
            .await
            .into_response()
    }
}

/// Extract the `X-Priority` header value, defaulting to `Normal`.
pub(crate) fn parse_priority_header(headers: &HeaderMap) -> RequestPriority {
    headers
        .get("x-priority")
        .and_then(|v| v.to_str().ok())
        .and_then(RequestPriority::from_header)
        .unwrap_or_default()
}

async fn non_stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
) -> Result<Json<ChatCompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    // when the prompt-prefix cache is installed, enter the
    // prefix-stable rendering path so unset preserve_thinking defaults to
    // true. The store is built by startup.rs only when configured, so
    // `state.prompt_cache.is_some()` is the operator-visible flag here.
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = prepare_chat_request_with_cache(
        &state.chat_template,
        &request,
        state.config.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
    )
    .await
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))?;
    // Build the prompt-cache context AFTER preparation so the multimodal
    // digest sees the resolved image/audio bytes.
    let prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &request,
        &prepared.image_data,
        &prepared.audio_data,
    );
    let primed_open_thinking = is_prompt_primed_open_thinking(&prepared.prompt);
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    options.prompt_cache_ctx = prompt_cache_ctx;
    // per-request Gemma 4 image soft-token budget, already validated against the
    // supported ladder by `prepare_chat_request_with_cache`. `None` for every
    // request that did not set `detail` / `max_soft_tokens`.
    options.image_soft_tokens = prepared.image_soft_tokens;
    // `ThinkingState` counts reasoning tokens from the first decoded token
    // only when the prompt already left the model inside an open thinking
    // block. The chat template decides this at render time (Qwen primes
    // `<think>\n`; Gemma 4's enable_thinking=true path primes
    // `<|channel>thought\n`; every other path leaves generation starting
    // outside any block). Setting this per-request keeps
    // `thinking_budget_tokens` functional for both families and avoids
    // counting ordinary content tokens as reasoning when the prompt
    // wasn't primed.
    options.thinking_enter_block_on_start = primed_open_thinking;
    // attach the structured-output constraint built at the
    // request boundary so the scheduler runs constrained sampling for this
    // sequence.
    options.structured = structured;

    // Set logprobs configuration when requested
    let top_k = request.top_logprobs.unwrap_or(0) as usize;
    if request.logprobs == Some(true) {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    // Generate (blocking call handled by model provider's worker thread).
    // forward resolved video paths alongside images and audio.
    // For non-video models the route guard above already rejected the
    // request, so `prepared.videos` is always empty here unless the
    // model supports video.
    let result = state
        .model_provider
        .generate_with_media_and_videos(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
        )
        .map_err(|e| ErrorResponse::new(format!("Generation error: {e}"), "server_error"))?;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    // Build logprobs for the response if requested
    let logprobs = result.logprobs.as_deref().and_then(|lp_data| {
        if lp_data.is_empty() {
            None
        } else {
            Some(build_chat_logprobs(&state.tokenizer, lp_data, top_k))
        }
    });

    let cached_tokens = result.cached_tokens;

    // Surface the thinking scratchpad as `reasoning_content`. This is additive:
    // the `content` computation below (strip_unclosed_primed_thinking /
    // clean_structural_tokens / tool-call parsing) is unchanged. Reusing the
    // streaming `StreamFilter` here means streaming and non-streaming responses
    // split reasoning from content identically. `None` for non-thinking models
    // leaves the field absent, closing the dropped-reasoning gap for every
    // thinking family at once (Qwen `<think>`, Gemma 4 `<|channel>`).
    let reasoning = extract_reasoning_content(&result.text, primed_open_thinking);

    // Issue #467: when the prompt primed an open thinking channel and the model
    // never emitted its close marker, the whole generation routes to
    // `reasoning_content` and the user-facing `content` is emptied below.
    // Surface that here so a broken or degenerate decode (e.g. an unsupported
    // quantization collapsing into repeating tokens) does not masquerade as a
    // clean, intentionally-empty response.
    if primed_thinking_unclosed(&result.text, primed_open_thinking) {
        tracing::warn!(
            target: "mlxcel::thinking",
            completion_tokens = result.completion_tokens as u64,
            finish_reason = %result.finish_reason,
            "primed thinking channel never closed: `content` is empty and all output \
             routed to `reasoning_content`; the decode may be truncated or degenerate"
        );
    }

    // Try to parse tool calls from the output
    if tool_calls::should_parse_tool_calls(&request) {
        let tools = request.tools.as_deref();
        let parsed = tool_calls::parse_tool_calls(&result.text, tools);

        // Harmony (GPT-OSS) carries its `analysis` channel as reasoning inside
        // the parse result; prefer it over the StreamFilter-derived `reasoning`,
        // which does not recognise Harmony's `<|channel|>` markers. Every other
        // family leaves `parsed.reasoning_content` `None` and keeps the
        // StreamFilter value.
        let reasoning = parsed
            .reasoning_content
            .clone()
            .or_else(|| reasoning.clone());

        if parsed.has_tool_calls() {
            let tool_call_responses = tool_calls::build_tool_call_responses(&parsed, &request);
            if !tool_call_responses.is_empty() {
                let content = strip_unclosed_primed_thinking(
                    parsed.content.clone(),
                    &result.text,
                    primed_open_thinking,
                );
                return Ok(Json(
                    ChatCompletionResponse::new_with_tool_calls(
                        request_id,
                        model_id,
                        content,
                        tool_call_responses,
                        result.prompt_tokens,
                        result.completion_tokens,
                        logprobs,
                    )
                    .with_cached_tokens(cached_tokens, prompt_cache_enabled)
                    .with_reasoning_content(reasoning.clone()),
                ));
            }
        }

        // No tool calls found, but tool parsing was enabled — use the cleaned
        // content from the parser (thinking blocks and structural markers
        // stripped) instead of the raw generation output.
        let content =
            strip_unclosed_primed_thinking(parsed.content, &result.text, primed_open_thinking);
        return Ok(Json(
            ChatCompletionResponse::new_with_logprobs(
                request_id,
                model_id,
                content,
                result.prompt_tokens,
                result.completion_tokens,
                Some(result.finish_reason),
                logprobs,
            )
            .with_cached_tokens(cached_tokens, prompt_cache_enabled)
            .with_reasoning_content(reasoning.clone()),
        ));
    }

    // Even without tool-call parsing, strip structural tokens so Gemma 4
    // (and similar) markers like `<channel|>` / `<turn|>` never leak into
    // plain chat responses.
    let cleaned_text = strip_unclosed_primed_thinking(
        tool_calls::clean_structural_tokens(&result.text),
        &result.text,
        primed_open_thinking,
    );

    Ok(Json(
        ChatCompletionResponse::new_with_logprobs(
            request_id,
            model_id,
            cleaned_text,
            result.prompt_tokens,
            result.completion_tokens,
            Some(result.finish_reason),
            logprobs,
        )
        .with_cached_tokens(cached_tokens, prompt_cache_enabled)
        .with_reasoning_content(reasoning),
    ))
}

/// Per-request streaming callback state (issue #633).
///
/// Collapses the three previously-separate `Arc<Mutex<…>>` values (tool-call
/// accumulator, logprobs buffer, stream filter) into a single lock. The
/// streaming callback runs on one blocking thread and the post-generation flush
/// runs on that same thread afterwards, so the lock exists only to satisfy the
/// `Send` bound on the `spawn_blocking` closure (an `Arc<RefCell<…>>` would not
/// be `Send`); it is never contended. Combining them turns three lock/unlock
/// pairs per token into one.
struct StreamCallbackState {
    /// Raw generated text accumulated for end-of-stream tool-call parsing. Only
    /// appended to when tool-call parsing is enabled.
    accumulated: String,
    /// Stream filter that splits reasoning/content and strips structural tokens.
    stream_filter: StreamFilter,
    /// Per-`feed()` logprob buffer, drained in lockstep with the filter's
    /// consumed/suppressed positions. Only used when logprobs are enabled.
    lp_buffer: std::collections::VecDeque<Option<TokenLogprobData>>,
}

async fn stream_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    // same prompt-cache flag as the non-streaming path so that
    // both endpoints default preserve_thinking=true identically when the
    // cache is installed.
    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = prepare_chat_request_with_cache(
        &state.chat_template,
        &request,
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
    // Build the prompt-cache context AFTER preparation so the multimodal
    // digest sees the resolved image/audio bytes.
    let prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &request,
        &prepared.image_data,
        &prepared.audio_data,
    );
    let primed_open_thinking = is_prompt_primed_open_thinking(&prepared.prompt);
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    options.prompt_cache_ctx = prompt_cache_ctx;
    // per-request Gemma 4 image soft-token budget, already validated against the
    // supported ladder by `prepare_chat_request_with_cache`. `None` for every
    // request that did not set `detail` / `max_soft_tokens`.
    options.image_soft_tokens = prepared.image_soft_tokens;
    // `ThinkingState` counts reasoning tokens from the first decoded token
    // only when the prompt already left the model inside an open thinking
    // block. The chat template decides this at render time (Qwen primes
    // `<think>\n`; Gemma 4's enable_thinking=true path primes
    // `<|channel>thought\n`; every other path leaves generation starting
    // outside any block). Setting this per-request keeps
    // `thinking_budget_tokens` functional for both families and avoids
    // counting ordinary content tokens as reasoning when the prompt
    // wasn't primed.
    options.thinking_enter_block_on_start = primed_open_thinking;
    // forward the constraint built at the request boundary so
    // streamed generation is also constrained.
    options.structured = structured;

    // Extract include_usage before request is moved into the closure
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // Set logprobs configuration when requested
    let top_k = request.top_logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs == Some(true);
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    let parse_tools = tool_calls::should_parse_tool_calls(&request);
    let tools_for_parser = if parse_tools {
        request.tools.clone()
    } else {
        None
    };
    let tool_choice = request.tool_choice.clone();

    // sse_channel also returns an SseKeepAlive that sends periodic
    // SSE comment events. This prevents proxy/client idle-timeout disconnects
    // during long prefill phases (32k+ token prompts) where no token event is
    // emitted until the first generated token arrives.
    let (events, stream, cancelled, keepalive) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();
    let tokenizer = state.tokenizer.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        let _ = finish_events.json(&initial);

        // Accumulate full output for tool call parsing at the end
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();

        // Single per-token lock (issue #633): the tool-call accumulator, the
        // stream filter, and the parallel logprob buffer live behind one Mutex
        // instead of three, so the hot callback locks once per token.
        //
        // Stream filter: strips model-specific structural tokens from content
        // deltas so clients never see <|channel>, <|tool_call>, <think>, etc.
        // Always on — even non-tool chat requests need to suppress thinking-
        // channel markers and stray turn tokens emitted by Gemma 4 and Qwen-
        // style reasoning models.
        //
        // When the generation prompt primed an open thinking marker — either
        // `<|channel>thought\n` (Gemma 4 enable_thinking=true) or `<think>\n`
        // (Qwen-style enable_thinking=true via OPEN_THINKING_SUFFIXES) — the
        // model's first emitted tokens are already reasoning content. Start
        // the filter in `Thinking` state so those tokens route to
        // `reasoning_content` until the model emits the matching close marker
        // (`<channel|>` or `</think>`); otherwise the scratchpad leaks to the
        // client when max_tokens is reached mid-reasoning.
        //
        // The logprob buffer holds one entry per `feed()` call. The stream
        // filter buffers incoming text fragments to handle delimiter matching at
        // token boundaries; when it later drains buffered bytes, the original
        // per-token `lp_data` is drained in lockstep with the filter's
        // `consumed_positions` output (drop `consumed - suppressed` emitted
        // entries; pop `suppressed` entries for placeholder chunks). This
        // preserves the upstream mlx-lm `replace(t, text="")` semantics so OpenAI
        // clients aligning by `choices[].logprobs.content` keep position info.
        let cb_state = std::sync::Arc::new(std::sync::Mutex::new(StreamCallbackState {
            accumulated: String::new(),
            stream_filter: if primed_open_thinking {
                StreamFilter::new_primed_open_thinking()
            } else {
                StreamFilter::new()
            },
            lp_buffer: std::collections::VecDeque::new(),
        }));
        let cb_state_for_callback = cb_state.clone();

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                prepared.videos,
                cancelled,
                |token, lp_data| {
                    // Single lock per token (issue #633): accumulate raw text,
                    // push this token's lp_data, run the stream filter, and drain
                    // the lp buffer under one lock. `lp_data` is pushed before
                    // `feed()` because the filter may buffer the token internally
                    // until a partial-match ambiguity resolves; the original
                    // lp_data must stay available for placeholder chunks it later
                    // drains. The emitted chunks are collected and sent after the
                    // lock is released so the (uncontended) lock never spans a
                    // channel send.
                    let mut pending: Vec<ChatCompletionChunk> = Vec::new();
                    {
                        let Ok(mut cb) = cb_state_for_callback.lock() else {
                            return;
                        };
                        let cb = &mut *cb;

                        if parse_tools {
                            cb.accumulated.push_str(&token);
                        }
                        if logprobs_enabled {
                            cb.lp_buffer.push_back(lp_data.clone());
                        }

                        // Split thinking scratchpad from user-facing content.
                        // Thinking goes out as `delta.reasoning_content`; regular
                        // text goes as `delta.content`.
                        let emit = cb.stream_filter.feed(&token);

                        // Drain the parallel lp_data buffer in lockstep with the
                        // filter's consumed_positions output:
                        //   - Emitted positions (consumed, not suppressed): drop.
                        //   - Suppressed positions: collect for placeholder chunks.
                        let suppressed_lp: Vec<Option<TokenLogprobData>> = if logprobs_enabled
                            && emit.consumed_positions > 0
                        {
                            let emitted = emit.consumed_positions - emit.suppressed_positions;
                            let mut suppressed_out = Vec::with_capacity(emit.suppressed_positions);
                            for _ in 0..emitted {
                                cb.lp_buffer.pop_front();
                            }
                            for _ in 0..emit.suppressed_positions {
                                suppressed_out.push(cb.lp_buffer.pop_front().flatten());
                            }
                            suppressed_out
                        } else {
                            Vec::new()
                        };

                        if let Some(reasoning_text) = emit.reasoning
                            && !reasoning_text.is_empty()
                        {
                            pending.push(ChatCompletionChunk::reasoning_content(
                                request_id_inner.clone(),
                                model_id_inner.clone(),
                                reasoning_text,
                            ));
                        }

                        if let Some(text) = emit.content
                            && !text.is_empty()
                        {
                            let logprobs = if logprobs_enabled {
                                lp_data.as_ref().map(|lp| {
                                    build_single_token_chat_logprobs(&tokenizer, lp, top_k)
                                })
                            } else {
                                None
                            };
                            pending.push(ChatCompletionChunk::content_with_logprobs(
                                request_id_inner.clone(),
                                model_id_inner.clone(),
                                text,
                                logprobs,
                            ));
                        }

                        // Preserve token-position alignment for parallel tool
                        // calls (upstream ml-explore/mlx-lm#1170, commit aa4f880). When
                        // the stream filter consumed a control-token delimiter
                        // (e.g. `<tool_call>`) it drained those bytes without
                        // producing output; with logprobs enabled, downstream
                        // consumers expect one event per token position, so emit
                        // an empty-content placeholder carrying the original
                        // per-token lp_data.
                        if logprobs_enabled && emit.suppressed_positions > 0 {
                            for slot_lp in suppressed_lp {
                                let logprobs = slot_lp.as_ref().map(|lp| {
                                    build_single_token_chat_logprobs(&tokenizer, lp, top_k)
                                });
                                pending.push(ChatCompletionChunk::content_with_logprobs(
                                    request_id_inner.clone(),
                                    model_id_inner.clone(),
                                    String::new(),
                                    logprobs,
                                ));
                            }
                        }
                    }

                    for chunk in &pending {
                        let _ = token_events.json(chunk);
                    }
                },
            );

        // Flush any remaining buffered content from the stream filter
        let remaining = cb_state
            .lock()
            .ok()
            .map(|mut cb| cb.stream_filter.flush())
            .unwrap_or_default();
        if let Some(text) = remaining.reasoning
            && !text.is_empty()
        {
            let chunk = ChatCompletionChunk::reasoning_content(
                request_id_clone.clone(),
                model_id_clone.clone(),
                text,
            );
            let _ = finish_events.json(&chunk);
        }
        if let Some(text) = remaining.content
            && !text.is_empty()
        {
            let chunk = ChatCompletionChunk::content_with_logprobs(
                request_id_clone.clone(),
                model_id_clone.clone(),
                text,
                None,
            );
            let _ = finish_events.json(&chunk);
        }

        // Check for tool calls in accumulated output
        let mut finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };

        if parse_tools && let Ok(cb) = cb_state.lock() {
            let tools_ref = tools_for_parser.as_deref();
            let parsed = tool_calls::parse_tool_calls(&cb.accumulated, tools_ref);

            if parsed.has_tool_calls() {
                // Emit tool call deltas
                let specific_fn = tool_choice
                    .as_ref()
                    .and_then(|tc| tc.specific_function())
                    .map(|s| s.to_string());

                for (idx, call) in parsed.tool_calls.iter().enumerate() {
                    // Filter by specific function if applicable
                    if let Some(ref fn_name) = specific_fn
                        && call.name != *fn_name
                    {
                        continue;
                    }

                    let call_id = tool_calls::generate_tool_call_id();

                    // Send tool call start delta
                    let start_chunk = ChatCompletionChunk::tool_call_start(
                        request_id_clone.clone(),
                        model_id_clone.clone(),
                        idx,
                        call_id,
                        call.name.clone(),
                    );
                    let _ = finish_events.json(&start_chunk);

                    // Send arguments as a single chunk
                    let args_chunk = ChatCompletionChunk::tool_call_arguments(
                        request_id_clone.clone(),
                        model_id_clone.clone(),
                        idx,
                        call.arguments.clone(),
                    );
                    let _ = finish_events.json(&args_chunk);
                }

                finish_reason = "tool_calls".to_string();
            }
        }

        // Send finish chunk
        let finish = ChatCompletionChunk::finish(
            request_id_clone.clone(),
            model_id_clone.clone(),
            finish_reason,
        );
        let _ = finish_events.json(&finish);

        // Send usage chunk if requested (stream_options.include_usage)
        if include_usage && let Ok(ref r) = result {
            let usage_chunk = ChatCompletionChunk::usage_with_cache(
                request_id_clone.clone(),
                model_id_clone.clone(),
                r.prompt_tokens,
                r.completion_tokens,
                r.cached_tokens,
                prompt_cache_enabled,
            );
            let _ = finish_events.json(&usage_chunk);
        }

        finish_events.done();
    });

    Sse::new(stream)
        .keep_alive(keepalive.into_inner())
        .into_response()
}

/// Returns `true` when the request body carries at least one `video_url`
/// content part anywhere in `messages`. The check matches before the heavier
/// `extract_chat_video_paths` resolution step so non-video-capable models
/// can refuse the request without paying canonicalisation / disk-I/O cost.
fn request_has_video_blocks(request: &ChatCompletionRequest) -> bool {
    !request.video_urls().is_empty()
}

/// Validate the XTC (Exclude Top Choices) sampling parameter ranges.
///
/// `xtc_threshold` must be `0.0..=0.5` and `xtc_probability` must be
/// `0.0..=1.0` when set; an absent field (`None`) is always valid since the
/// server default then applies. Returns `Err` with a client-facing message
/// so the caller can surface a 400 `invalid_request_error` before any
/// generation work begins.
pub(crate) fn validate_xtc_params(
    xtc_threshold: Option<f32>,
    xtc_probability: Option<f32>,
) -> Result<(), &'static str> {
    if let Some(threshold) = xtc_threshold
        && !(0.0..=0.5).contains(&threshold)
    {
        return Err("xtc_threshold must be between 0.0 and 0.5");
    }
    if let Some(probability) = xtc_probability
        && !(0.0..=1.0).contains(&probability)
    {
        return Err("xtc_probability must be between 0.0 and 1.0");
    }
    Ok(())
}

/// Maximum number of tools allowed in a single request.
///
/// Enforced in `chat_completions()` to prevent DoS via large tool definitions
/// being rendered through the Jinja2 chat template.
pub(crate) const MAX_TOOLS: usize = 128;

/// Suffixes that a rendered chat prompt uses to leave the model inside an
/// open thinking block. Each corresponds to a family-specific reasoning
/// convention:
///
/// * `<think>\n` — Qwen3 / Qwen3.5 / Exaone4 / Jamba and similar templates
///   that emit `<think>\n` at the end of the generation prompt to prime
///   reasoning. The close marker in generated text is `</think>`.
/// * `<|channel>thought\n` is the Gemma 4 open reasoning channel. Its close
///   marker in generated text is `<channel|>`. Since issue #686 the Gemma 4
///   generation prompt is rendered faithfully to the template (no post-render
///   priming patch): the interactive default ends with the CLOSED
///   `<|channel>thought\n<channel|>` scaffold (not primed for open thinking),
///   so this suffix only matches a prompt that genuinely leaves the channel
///   open (e.g. a caller-crafted continuation).
///
/// Ordered longest-first so the match is unambiguous.
const OPEN_THINKING_SUFFIXES: &[&str] = &["<|channel>thought\n", "<think>\n"];

/// Whether the rendered chat prompt primed an open thinking block whose
/// close marker the model is expected to emit (`</think>` for Qwen-style,
/// `<channel|>` for Gemma 4). Callers use this to:
///
/// * initialize the streaming filter and non-streaming thinking stripper
///   so the first generated tokens surface as reasoning rather than
///   assistant content,
/// * choose `thinking_enter_block_on_start` for the scheduler's
///   `ThinkingState` so per-request `thinking_budget_tokens` counts every
///   emitted token from the start (otherwise the state would wait for an
///   opening token that never appears because the prompt already contains
///   it).
pub(crate) fn is_prompt_primed_open_thinking(prompt: &str) -> bool {
    OPEN_THINKING_SUFFIXES
        .iter()
        .any(|suffix| prompt.ends_with(suffix))
}

/// Close markers for each supported open-thinking priming convention. Paired
/// by family with [`OPEN_THINKING_SUFFIXES`] — either one closes "a block
/// the prompt opened", so the post-processor treats the generation as
/// closed when any of them appears in the raw output.
const OPEN_THINKING_CLOSE_MARKERS: &[&str] = &["<channel|>", "</think>"];

/// Whether the prompt primed an open thinking block that the raw output never
/// closed.
///
/// True exactly when `primed` is set (the generation prompt ended with an open
/// thinking marker) and `raw_output` contains none of the close markers
/// (`<channel|>` for Gemma 4, `</think>` for Qwen-style). In that state the
/// whole generation is reasoning and the non-streaming `content` is emptied by
/// [`strip_unclosed_primed_thinking`].
///
/// Callers surface this condition (a `tracing::warn!`) instead of returning a
/// silently-empty `content`, so a broken or degenerate decode that never emits
/// a close marker (issue #467: an unsupported quantization collapsing into
/// repeating tokens) does not masquerade as a clean, intentionally-empty
/// response.
fn primed_thinking_unclosed(raw_output: &str, primed: bool) -> bool {
    primed
        && !OPEN_THINKING_CLOSE_MARKERS
            .iter()
            .any(|m| raw_output.contains(m))
}

/// Strip reasoning content that would otherwise leak when the prompt primed
/// an open thinking block and the model never emitted its close marker.
///
/// * Returns `content` unchanged when the prompt did not prime open thinking.
/// * Returns `content` unchanged when `raw_output` contains any known close
///   marker (`<channel|>` for Gemma 4, `</think>` for Qwen-style) — the
///   regular parsers already handle that case.
/// * Returns an empty string when the whole generation was unclosed thinking
///   (see [`primed_thinking_unclosed`], the shared predicate the caller also
///   uses to emit the unclosed-thinking warning).
fn strip_unclosed_primed_thinking(content: String, raw_output: &str, primed: bool) -> String {
    if primed_thinking_unclosed(raw_output, primed) {
        String::new()
    } else {
        content
    }
}

/// Extract the reasoning / thinking scratchpad from a completed generation by
/// replaying the raw text through the same [`StreamFilter`] the streaming path
/// uses (`stream_chat_completion` builds the identical filter at the SSE
/// construction site). Reusing the filter is what guarantees streaming and
/// non-streaming surface the same `reasoning_content`: the filter's state
/// machine is deterministic over the concatenated input, so feeding the whole
/// string in one `feed()` accumulates the same reasoning/content split as
/// feeding it token-by-token (the parity is locked by a unit test).
///
/// `primed_open_thinking` selects the filter's start state. A prompt that
/// primed an open thinking block (`<think>\n` for Qwen-style, or
/// `<|channel>thought\n` for Gemma 4 `enable_thinking=true`) starts the filter
/// inside `Thinking` so the leading generated tokens route to reasoning even
/// though the opening marker lives in the prompt, not the output. This mirrors
/// the non-streaming content path in [`strip_unclosed_primed_thinking`]: when
/// the whole window is unclosed thinking, all of it becomes `reasoning` and the
/// user-facing `content` is empty.
///
/// Returns `Some(reasoning)` when any reasoning text was captured, `None`
/// otherwise (so the response omits `reasoning_content` for non-thinking
/// models). Tool-call blocks are suppressed by the filter and never leak into
/// reasoning; they are materialized by the parser path instead.
fn extract_reasoning_content(raw_text: &str, primed_open_thinking: bool) -> Option<String> {
    let mut filter = if primed_open_thinking {
        StreamFilter::new_primed_open_thinking()
    } else {
        StreamFilter::new()
    };
    let mut reasoning = String::new();
    if let Some(r) = filter.feed(raw_text).reasoning {
        reasoning.push_str(&r);
    }
    if let Some(r) = filter.flush().reasoning {
        reasoning.push_str(&r);
    }
    if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

/// Build ServerGenerateOptions using request params with server config as defaults
///
/// The explicit per-request loop-detection override is read from `params` (the
/// vLLM `max_pattern_size` / `min_pattern_size` / `min_count` fields, issue
/// #432). The Gemma 4 family default-on is applied engine-side from the loaded
/// model type in
/// [`crate::server::request_options::build_server_generate_options`], so callers
/// pass no amplifier signal.
pub(crate) fn build_generate_options(
    params: &SamplingParams,
    config: &ServerConfig,
) -> ServerGenerateOptions {
    build_server_generate_options(
        config,
        RequestOptionOverrides {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_k: params.top_k.map(|k| k as i32),
            top_p: params.top_p,
            min_p: params.min_p,
            repetition_penalty: params.repetition_penalty,
            seed: params.seed,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            dry_sequence_breakers: params.dry_sequence_breakers.clone(),
            xtc_probability: params.xtc_probability,
            xtc_threshold: params.xtc_threshold,
            stop_sequences: params.stop.clone(),
            priority: RequestPriority::default(),
            // the caller (non_stream_chat_completion /
            // stream_chat_completion) sets `options.reasoning_budget`
            // explicitly after `build_generate_options` returns, so the
            // default here is just a placeholder.
            reasoning_budget: ReasoningBudgetOverride::default(),
            // Placeholder: the caller overrides this with
            // `is_prompt_primed_open_thinking(&prepared.prompt)` once the
            // chat template has rendered, so the value picked up by the
            // scheduler matches the actual prompt tail (Qwen `<think>\n`,
            // Gemma 4 `<|channel>thought\n`, or neither).
            thinking_enter_block_on_start: false,
            loop_detection_request: crate::server::request_options::loop_detection_from_request(
                params.max_pattern_size,
                params.min_pattern_size,
                params.min_count,
            ),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::request::{FunctionDefinition, Tool};

    fn make_tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters: None,
            },
        }
    }

    #[test]
    fn max_tools_constant_is_128() {
        assert_eq!(MAX_TOOLS, 128);
    }

    // -- XTC (Exclude Top Choices) request validation --

    #[test]
    fn validate_xtc_params_accepts_unset_fields() {
        assert!(validate_xtc_params(None, None).is_ok());
    }

    #[test]
    fn validate_xtc_params_accepts_in_range_boundaries() {
        assert!(validate_xtc_params(Some(0.0), Some(0.0)).is_ok());
        assert!(validate_xtc_params(Some(0.5), Some(1.0)).is_ok());
        assert!(validate_xtc_params(Some(0.1), Some(0.4)).is_ok());
    }

    #[test]
    fn validate_xtc_params_rejects_out_of_range_threshold() {
        // Above the 0.5 upper bound.
        assert_eq!(
            validate_xtc_params(Some(0.6), None),
            Err("xtc_threshold must be between 0.0 and 0.5")
        );
        // Below the 0.0 lower bound.
        assert_eq!(
            validate_xtc_params(Some(-0.1), None),
            Err("xtc_threshold must be between 0.0 and 0.5")
        );
    }

    #[test]
    fn validate_xtc_params_rejects_out_of_range_probability() {
        // Above the 1.0 upper bound.
        assert_eq!(
            validate_xtc_params(None, Some(1.1)),
            Err("xtc_probability must be between 0.0 and 1.0")
        );
        // Below the 0.0 lower bound.
        assert_eq!(
            validate_xtc_params(None, Some(-0.5)),
            Err("xtc_probability must be between 0.0 and 1.0")
        );
    }

    #[test]
    fn validate_xtc_params_checks_threshold_before_probability() {
        // Both fields invalid: the threshold check runs first.
        assert_eq!(
            validate_xtc_params(Some(0.9), Some(2.0)),
            Err("xtc_threshold must be between 0.0 and 0.5")
        );
    }

    #[test]
    fn tools_below_limit_accepted() {
        // Build a vec of exactly MAX_TOOLS tools — should not exceed the limit
        let tools: Vec<Tool> = (0..MAX_TOOLS)
            .map(|i| make_tool(&format!("fn_{i}")))
            .collect();
        assert!(tools.len() <= MAX_TOOLS);
    }

    #[test]
    fn tools_above_limit_detected() {
        // Build a vec of MAX_TOOLS + 1 tools — must exceed the limit
        let tools: Vec<Tool> = (0..=MAX_TOOLS)
            .map(|i| make_tool(&format!("fn_{i}")))
            .collect();
        assert!(tools.len() > MAX_TOOLS);
    }

    // -- Gemma 4 open-thinking prompt-primed detection --

    #[test]
    fn prompt_primed_detection_matches_exact_suffix() {
        // A prompt that genuinely ends in exactly `<|channel>thought\n`
        // (open channel) is primed for open thinking. Only that exact suffix
        // counts.
        assert!(is_prompt_primed_open_thinking(
            "<|turn>model\n<|channel>thought\n"
        ));
    }

    #[test]
    fn prompt_primed_detection_rejects_closed_priming() {
        // `enable_thinking=false` templates end with the CLOSED priming.
        // Those are NOT prompt-primed open-thinking and must not trigger.
        assert!(!is_prompt_primed_open_thinking(
            "<|turn>model\n<|channel>thought\n<channel|>\n"
        ));
    }

    #[test]
    fn prompt_primed_detection_rejects_unrelated_endings() {
        assert!(!is_prompt_primed_open_thinking("<|turn>model\n"));
        assert!(!is_prompt_primed_open_thinking(""));
        assert!(!is_prompt_primed_open_thinking("some content"));
    }

    // -- strip_unclosed_primed_thinking --

    #[test]
    fn strip_unclosed_primed_thinking_empties_when_primed_and_no_close() {
        // Classic failure mode: model hit max_tokens inside the primed
        // channel, raw output contains no `<channel|>`. Return empty content.
        let content = "reasoning overflow".to_string();
        let raw = "reasoning overflow";
        assert_eq!(
            strip_unclosed_primed_thinking(content, raw, true),
            String::new()
        );
    }

    #[test]
    fn strip_unclosed_primed_thinking_preserves_when_close_present() {
        // Primed but model DID close the channel. The parser already
        // stripped the thinking block; whatever remains in `content` is
        // real user-visible text and must pass through.
        let content = "the answer".to_string();
        let raw = "thinking<channel|>the answer";
        assert_eq!(
            strip_unclosed_primed_thinking(content.clone(), raw, true),
            content
        );
    }

    #[test]
    fn strip_unclosed_primed_thinking_noop_when_not_primed() {
        // Non-primed requests (enable_thinking=false, or non-Gemma model)
        // must never be touched by this helper — preserves backward compat
        // for every other template.
        let content = "whatever the parser returned".to_string();
        let raw = "whatever the parser returned";
        assert_eq!(
            strip_unclosed_primed_thinking(content.clone(), raw, false),
            content
        );
    }

    // -- primed_thinking_unclosed (issue #467 unclosed-thinking surface) --

    #[test]
    fn primed_thinking_unclosed_true_when_primed_and_no_close_marker() {
        // The reported failure shape: primed Gemma 4 channel, degenerate output
        // with no `<channel|>` close marker anywhere. This is the condition the
        // non-streaming path warns on instead of silently emptying content.
        assert!(primed_thinking_unclosed(
            "1\n//\n same////\n1 uma\n//\n//",
            true
        ));
    }

    #[test]
    fn primed_thinking_unclosed_false_when_close_marker_present() {
        // A real close marker (either family) means the block closed normally.
        assert!(!primed_thinking_unclosed(
            "thinking<channel|>the answer",
            true
        ));
        assert!(!primed_thinking_unclosed("reasoning</think>done", true));
    }

    #[test]
    fn primed_thinking_unclosed_false_when_not_primed() {
        // Non-primed requests are never flagged, even without a close marker.
        assert!(!primed_thinking_unclosed(
            "plain answer with no markers",
            false
        ));
    }

    #[test]
    fn primed_thinking_unclosed_is_the_content_emptying_predicate() {
        // The predicate is the single source of truth: it is true exactly when
        // strip_unclosed_primed_thinking empties content, so the warning and the
        // emptying can never disagree.
        let unclosed = "all reasoning, never closed";
        assert!(primed_thinking_unclosed(unclosed, true));
        assert_eq!(
            strip_unclosed_primed_thinking("x".to_string(), unclosed, true),
            String::new()
        );

        let closed = "reasoning<channel|>answer";
        assert!(!primed_thinking_unclosed(closed, true));
        assert_eq!(
            strip_unclosed_primed_thinking("answer".to_string(), closed, true),
            "answer"
        );
    }

    // -- extract_reasoning_content (non-streaming reasoning surface) --

    #[test]
    fn extract_reasoning_qwen_think_block() {
        // Qwen-style: a closed <think>…</think> block before the answer. The
        // reasoning is captured; the answer stays out of reasoning.
        let raw = "<think>reasoning</think>the answer";
        assert_eq!(
            extract_reasoning_content(raw, false),
            Some("reasoning".to_string())
        );
    }

    #[test]
    fn extract_reasoning_qwen_primed_open() {
        // enable_thinking=true primes `<think>\n` in the prompt, so the raw
        // output starts mid-think with no opening marker. The filter must
        // start in Thinking and route the leading tokens to reasoning until
        // the close marker.
        let raw = "reasoning</think>the answer";
        assert_eq!(
            extract_reasoning_content(raw, true),
            Some("reasoning".to_string())
        );
    }

    #[test]
    fn extract_reasoning_gemma4_channel_block() {
        // Gemma 4: a full <|channel>…<channel|> block before the answer.
        let raw = "<|channel>thought\ndeliberating<channel|>the answer";
        assert_eq!(
            extract_reasoning_content(raw, false),
            Some("thought\ndeliberating".to_string())
        );
    }

    #[test]
    fn extract_reasoning_gemma4_primed_all_thinking() {
        // Gemma 4 enable_thinking=true primes `<|channel>thought\n`; if the
        // model fills the whole window without ever emitting `<channel|>`, the
        // entire generation is reasoning and the user-facing content is empty.
        // Mirrors the content side (strip_unclosed_primed_thinking ->
        // String::new()) and the parser's all-thinking branch.
        let raw = "still deliberating about hash tables with no close marker";
        assert_eq!(
            extract_reasoning_content(raw, true),
            Some("still deliberating about hash tables with no close marker".to_string())
        );
    }

    #[test]
    fn extract_reasoning_none_for_plain_content() {
        // A non-thinking model (no markers, not primed) yields no reasoning,
        // so the response omits `reasoning_content`.
        assert_eq!(
            extract_reasoning_content("just a normal answer", false),
            None
        );
    }

    // -- streaming / non-streaming parity ---------------------------
    //
    // The non-streaming extractor feeds the whole generated string through the
    // same StreamFilter the streaming path feeds token-by-token. This locks
    // the equivalence: the accumulated (content, reasoning) split must be
    // identical regardless of how the input is fragmented.

    fn absorb(
        out: crate::server::tool_calls::stream_filter::FilterOutput,
        content: &mut String,
        reasoning: &mut String,
    ) {
        if let Some(c) = out.content {
            content.push_str(&c);
        }
        if let Some(r) = out.reasoning {
            reasoning.push_str(&r);
        }
    }

    fn split_whole(text: &str, primed: bool) -> (String, String) {
        let mut f = if primed {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        };
        let (mut content, mut reasoning) = (String::new(), String::new());
        absorb(f.feed(text), &mut content, &mut reasoning);
        absorb(f.flush(), &mut content, &mut reasoning);
        (content, reasoning)
    }

    fn split_chunked(text: &str, primed: bool) -> (String, String) {
        let mut f = if primed {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        };
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            absorb(
                f.feed(ch.encode_utf8(&mut buf)),
                &mut content,
                &mut reasoning,
            );
        }
        absorb(f.flush(), &mut content, &mut reasoning);
        (content, reasoning)
    }

    #[test]
    fn reasoning_split_identical_whole_vs_chunked() {
        let samples = [
            ("<think>reasoning here</think>the answer", false),
            ("<|channel>thought\ndeliberate<channel|>final answer", false),
            ("still reasoning</think>then content", true),
            ("unclosed thinking forever", true),
        ];
        for (text, primed) in samples {
            assert_eq!(
                split_whole(text, primed),
                split_chunked(text, primed),
                "whole-vs-chunked split must match for {text:?} (primed={primed})"
            );
        }
    }

    // -- video_url block detection ---------------------------

    use crate::server::types::request::{ImageUrl, VideoUrl};
    use crate::server::types::{ContentPart, Message, MessageContent, Role, SamplingParams};

    fn build_request(parts: Vec<ContentPart>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Parts(parts),
                name: None,
                tool_call_id: None,
                reasoning: None,
                tool_calls: None,
            }],
            stream: false,
            stream_options: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            chat_template_kwargs: None,
            extra_body: None,
            prompt_cache_key: None,
            user: None,
            extra_body_fields: serde_json::Map::new(),
            response_format: None,
            params: SamplingParams::default(),
        }
    }

    #[test]
    fn request_has_video_blocks_returns_true_when_video_url_present() {
        let req = build_request(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "file:///tmp/clip.mp4".to_string(),
                    fps: None,
                },
            },
        ]);
        assert!(request_has_video_blocks(&req));
    }

    #[test]
    fn request_has_video_blocks_returns_false_for_text_and_image_only() {
        let req = build_request(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl::new("data:image/png;base64,abc".to_string()),
            },
        ]);
        assert!(!request_has_video_blocks(&req));
    }

    #[test]
    fn request_has_video_blocks_returns_false_for_plain_text() {
        let mut req = build_request(vec![ContentPart::Text {
            text: "hi".to_string(),
        }]);
        // Replace with a plain-string MessageContent to cover the
        // `MessageContent::Text` branch.
        req.messages[0].content = MessageContent::Text("hi".to_string());
        assert!(!request_has_video_blocks(&req));
    }

    // -- no-effective-input rejection (issue #773), exercised at the
    // handler-check boundary rather than the shared helper's own unit tests
    // (see chat_request_tests.rs for the full helper matrix). ------------

    #[test]
    fn chat_completions_handler_check_rejects_empty_messages() {
        let mut req = build_request(vec![]);
        req.messages.clear();
        assert!(!request_has_effective_input(&req));
    }

    #[test]
    fn chat_completions_handler_check_rejects_empty_string_content() {
        let mut req = build_request(vec![]);
        req.messages[0].content = MessageContent::Text(String::new());
        assert!(!request_has_effective_input(&req));
    }

    #[test]
    fn chat_completions_handler_check_accepts_image_only_request() {
        let req = build_request(vec![ContentPart::ImageUrl {
            image_url: ImageUrl::new("data:image/png;base64,abc".to_string()),
        }]);
        assert!(request_has_effective_input(&req));
    }

    #[test]
    fn chat_completions_no_effective_input_error_matches_issue_773_spec() {
        // The handler's early-reject branch must surface HTTP 400,
        // `invalid_request_error`, and this exact message.
        let response = ErrorResponse::new(
            "Request must include at least one non-empty message content or media input.",
            "invalid_request_error",
        );
        assert_eq!(response.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(
            response.error.message,
            "Request must include at least one non-empty message content or media input."
        );
    }
}
