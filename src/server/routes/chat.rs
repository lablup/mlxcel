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
    response::{IntoResponse, Response},
};

use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};

use crate::server::batch::RequestPriority;
use crate::server::chat_request::{
    prepare_chat_request_with_cache, request_has_effective_input, resolve_effective_kwargs,
};
use crate::server::config::{PromptCacheRequestContext, ReasoningBudgetOverride};
use crate::server::prompt_cache::key::{
    MultimodalDigest, multimodal_digest_from_vecs, resolve_session_key, template_sig,
};
use crate::server::request_options::{
    RequestOptionOverrides, build_server_generate_options_with_live, chat_carries_loop_amplifier,
    resolve_server_max_tokens_with_live,
};
use crate::server::streaming::{sse_channel_resumable, sse_response};
use crate::server::structured::{StructuredOutputError, build_constraint_from_response_format};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::{FilterOutput, StreamFilter};
use crate::server::types::response::{ChatLogprobs, TokenLogprob, TopLogprob};
use crate::server::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse,
    SamplingParams,
};
use crate::server::{AppState, LiveSettings, ServerConfig, ServerGenerateOptions};
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
        | StructuredOutputError::SchemaTooLarge(_)
        | StructuredOutputError::InvalidGrammar(_) => {
            ErrorResponse::new(err.to_string(), "invalid_request_error")
        }
        StructuredOutputError::UnsupportedTokenizer(_) | StructuredOutputError::Matcher(_) => {
            ErrorResponse::new(err.to_string(), "server_error")
        }
    }
}

fn generation_error_to_response(err: anyhow::Error) -> ErrorResponse {
    super::generation_error_to_response(err)
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
///
/// `history_prompt` is the request re-rendered with `add_generation_prompt =
/// false` (see [`crate::server::chat_request::PreparedChatRequest::history_prompt`]).
/// It travels with the context so the scheduler can capture a history-boundary
/// snapshot during prefill for snapshot-only families (issue #1143). Pass
/// `None` to opt this request out.
pub(crate) fn build_prompt_cache_request_context(
    state: &AppState,
    live: &LiveSettings,
    request: &ChatCompletionRequest,
    image_data: &[Vec<u8>],
    audio_data: &[Vec<u8>],
    history_prompt: Option<&str>,
) -> Option<PromptCacheRequestContext> {
    state.prompt_cache.as_ref()?;
    // b10621 `cache_prompt: false` opts this one request out. Returning `None`
    // here is the whole implementation, and deliberately so: the scheduler
    // reaches the store only through this context, for the lookup
    // (`try_adopt_cached_prefix`) and for the donate-back
    // (`donate_finished_sequence_cache`) alike. Withholding it therefore
    // forces a cold prefill AND leaves every entry another request might reuse
    // exactly as it was, which is the half of the contract a lookup-only
    // opt-out would miss.
    if request.resolve_cache_prompt() == Some(false) {
        return None;
    }
    // Share the kwargs resolution with `prepare_chat_request_with_cache` so the
    // digest sees the same canonicalized map as the rendering pipeline. Calling
    // the helper rather than re-deriving the merge here keeps a mapped
    // `reasoning_effort` (issue #1164) inside `template_sig`, so two requests
    // that differ only in reasoning effort land in different cache buckets
    // instead of sharing one and re-prefilling past the divergence point.
    let merged_kwargs = resolve_effective_kwargs(
        &state.chat_template,
        request,
        live.chat_template_kwargs.as_ref(),
        &request.merged_extra_body(),
    );

    let template_signature = template_sig(
        state.chat_template.template_source(),
        &merged_kwargs,
        request.tool_choice.as_ref(),
        request.tools.as_deref(),
        // The dimension only matters when this request actually has a trailing
        // assistant message: hashing the flag unconditionally would split every
        // bucket in the store the first time an operator passes
        // `--no-prefill-assistant`.
        state.prefill_assistant()
            && crate::server::assistant_prefill::resolve(request, true)
                .is_ok_and(|prefill| prefill.is_some()),
    );
    let session_key =
        resolve_session_key(request.resolve_prompt_cache_key(), request.resolve_user()).to_string();
    // Digest the resolved multimodal payload. Empty slices (text-only) hash to
    // `MultimodalDigest::empty()`, leaving the composed key unchanged.
    let mm_digest = multimodal_digest_from_vecs(image_data, audio_data);
    Some(PromptCacheRequestContext {
        model_id: state.display_model_id().to_string(),
        // Key cache entries by the effective adapter scales (#1439) so a
        // POST /lora-adapters swap can never resurrect KV computed under a
        // different configuration. Chat requests always carry the server
        // default (the native `lora` field exists on /completion only).
        lora_id: state
            .config
            .lora_runtime
            .as_ref()
            .map(|set| crate::lora::RuntimeLoraSet::scales_digest(&set.server_scales())),
        template_sig: template_signature,
        session_key,
        mm_digest,
        history_prompt: history_prompt.map(str::to_string),
        history_prefix_tokens: None,
    })
}

/// Cache-key template dimension for the two raw-prompt routes (#1473).
///
/// `/v1/completions` and `/completion` take a prompt string that is never run
/// through the chat template, so there is no template source, kwargs map,
/// tool list or tool choice to digest. A fixed sentinel keeps them in their
/// own bucket: a raw prompt that happens to equal a rendered conversation
/// must not adopt that conversation's KV, because the two would diverge the
/// moment the template changed underneath them.
pub(crate) const RAW_PROMPT_TEMPLATE_SIG: &str = "mlxcel:prompt-cache:raw-prompt:v1";

/// Build the prompt-cache context for a raw-prompt route (#1473).
///
/// The sibling of [`build_prompt_cache_request_context`] for `/v1/completions`
/// and native `/completion`. Same contract, and deliberately the same single
/// seam: returning `None` is the whole implementation of a per-request
/// opt-out, because the scheduler reaches the store only through this context
/// for both the prefix lookup and the donate-back, so withholding it forces a
/// cold prefill AND leaves every entry another request might reuse untouched.
///
/// `cache_prompt` is the b10621 per-request field, `None` when the route or
/// the request did not express one. Raw-prompt routes carry no session key,
/// no multimodal payload and no history render, so the remaining key
/// dimensions are the model id and [`RAW_PROMPT_TEMPLATE_SIG`].
pub(crate) fn build_raw_prompt_cache_context(
    state: &AppState,
    cache_prompt: Option<bool>,
) -> Option<PromptCacheRequestContext> {
    state.prompt_cache.as_ref()?;
    if cache_prompt == Some(false) {
        return None;
    }
    Some(PromptCacheRequestContext {
        model_id: state.display_model_id().to_string(),
        lora_id: None,
        template_sig: RAW_PROMPT_TEMPLATE_SIG.to_string(),
        session_key: resolve_session_key(None, None).to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: None,
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
    let live = state.live();
    if let Some(response) = super::chat_not_available(&state) {
        return response.into_response();
    }

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
    if let Err(message) = validate_top_n_sigma(request.params.top_n_sigma) {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }
    if let Err(message) = validate_typical_p(request.params.typical_p) {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    // Reject image, audio and video content blocks the loaded checkpoint
    // cannot consume, before any referenced URL or file is read. Capability is
    // detected once at startup from `config.json` and cached on
    // `AppState.media_support`; the refusal carries b10621's own
    // `<kind> input is not supported` wording (issue #1451). Silently dropping
    // an image would still consume tokens and produce a reply the caller could
    // not tell apart from one that saw the picture.
    if let Some(rejection) = crate::server::media_capability_rejection(
        &request,
        state.media_support,
        state.display_model_id(),
    ) {
        return rejection.into_response();
    }

    // Keep tool validation shared with the disaggregated router front so both
    // paths reject invalid and oversized requests before template rendering.
    if let Err(message) = validate_chat_tool_inputs(&request) {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    // validate thinking_budget_tokens early so malformed values
    // surface as 400 before any generation work begins.
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
                return ErrorResponse::new(err.to_string(), "invalid_request_error")
                    .into_response();
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
        // b10621 resumable streams (#1444): a streaming request carrying
        // `X-Conversation-Id` buffers its SSE bytes in a session replayable
        // via `GET /v1/stream`, scoped to the API key that created it.
        let resumable = ResumableStreamContext {
            conversation_id: super::stream::conversation_id_from_headers(&headers),
            owner: super::stream::request_stream_owner(&state, &headers),
        };
        stream_chat_completion(
            state,
            live,
            request,
            priority,
            budget_override,
            structured,
            resumable,
        )
        .await
    } else {
        non_stream_chat_completion(state, live, request, priority, budget_override, structured)
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

/// `pub(crate)` so the b10621 transcription-compatibility route can dispatch
/// through the same chat pipeline upstream's own handler does, without a JSON
/// round-trip: it returns the typed `ChatCompletionResponse` (issue #1446).
pub(crate) async fn non_stream_chat_completion(
    state: AppState,
    live: std::sync::Arc<LiveSettings>,
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
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
    )
    .await
    .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))?;
    // Build the prompt-cache context AFTER preparation so the multimodal
    // digest sees the resolved image/audio bytes.
    let prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &live,
        &request,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );
    // Retained for the post-completion warm-up (issue #1144); the context
    // itself moves into `options` below. Cheap: the history string has already
    // been handed to the context and the token vector is not filled until the
    // dispatch thread.
    let warmup_ctx = prompt_cache_ctx.as_ref().map(|ctx| {
        let mut c = ctx.clone();
        c.history_prompt = None;
        c
    });
    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    // Loop-detection amplifier signal (issues #967 and #977): only tool-shaped
    // prompts arm the family default. Grammar-only requests stay disabled.
    let amplified = chat_carries_loop_amplifier(&request);
    let mut options =
        build_generate_options_with_live(&request.params, &state.config, &live, amplified);
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
            source: Default::default(),
        };
    }

    // Generate (blocking call handled by model provider's worker thread).
    // forward resolved video paths alongside images and audio.
    // For non-video models the route guard above already rejected the
    // request, so `prepared.videos` is always empty here unless the
    // model supports video.
    // Slot accounting (#1440): ties this request to a numbered slot for
    // GET /slots. Taken before `options` moves into the provider call.
    let slot = state.slots.begin(
        &prepared.prompt,
        super::slots::slot_params_json(&options, false),
        Some(options.max_tokens as i64),
    );
    // b10621 `echo` (#1470): with a prefilled assistant message and `echo`
    // set, the prefill leads the response. Upstream reaches the same shape by
    // NOT priming its chat parser, so the first diff carries the continuation
    // text; prepending it to the generated text here feeds the identical
    // string through tool-call parsing, the reasoning split and the
    // `--reasoning-format` placement.
    let echo_prefill = request
        .resolve_echo()
        .then(|| prepared.assistant_prefill.clone())
        .flatten();
    let mut result = state
        .model_provider
        .generate_with_media_and_videos_declared_live(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
            prepared.media,
            &live,
        )
        .map_err(generation_error_to_response)?;

    if let Some(prefill) = echo_prefill.as_deref() {
        result.text.insert_str(0, prefill);
    }

    // Structured Florence-2 task output (issue #1073): produced only by the
    // seq2seq worker, `None` for every other family. Attached below as the
    // assistant message's `florence2_result` extension field, next to the
    // human-readable `content` that carries the same answer as text.
    let florence2_result = result.structured_output.take();

    slot.finish(
        result.prompt_tokens,
        result.cached_tokens,
        result.completion_tokens,
        &result.text,
    );
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

    // b10621 `--skip-chat-parsing` (issue #1447): force a pure content parser.
    // Everything the model emitted goes to `content` verbatim, reasoning and
    // tool-call syntax included, and no `reasoning_content` or `tool_calls` is
    // produced. Placed before every other shaping step so nothing downstream
    // can re-introduce a parser.
    if state.config.skip_chat_parsing {
        return Ok(Json(
            ChatCompletionResponse::new_with_logprobs(
                request_id,
                model_id,
                result.text,
                result.prompt_tokens,
                result.completion_tokens,
                Some(result.finish_reason),
                logprobs,
            )
            .with_cached_tokens(cached_tokens, prompt_cache_enabled)
            .with_florence2_result(florence2_result),
        ));
    }

    // Where the thoughts end up: b10621 `--reasoning-format`
    // (`LLAMA_ARG_THINK`). mlxcel's historical behavior is `deepseek`, which is
    // also what `auto` resolves to here; `none` and `deepseek-legacy` keep the
    // thinking block in `content`.
    let reasoning_format = state.config.reasoning_format;
    let reasoning_alias_field = state.config.reasoning_alias_field;

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
                let answer = strip_unclosed_primed_thinking(
                    parsed.content.clone(),
                    &result.text,
                    primed_open_thinking,
                );
                let shaped = crate::server::shape_response(
                    reasoning_format,
                    answer.clone(),
                    || {
                        tool_calls::content_with_thinking_block(
                            &result.text,
                            &answer,
                            reasoning.as_deref(),
                        )
                    },
                    reasoning.clone(),
                );
                return Ok(Json(
                    ChatCompletionResponse::new_with_tool_calls(
                        request_id,
                        model_id,
                        shaped.content,
                        tool_call_responses,
                        result.prompt_tokens,
                        result.completion_tokens,
                        logprobs,
                    )
                    .with_cached_tokens(cached_tokens, prompt_cache_enabled)
                    .with_reasoning_content_alias_field(
                        shaped.reasoning_content,
                        reasoning_alias_field,
                    ),
                ));
            }
        }

        // No tool calls found, but tool parsing was enabled — use the cleaned
        // content from the parser (thinking blocks and structural markers
        // stripped) instead of the raw generation output.
        let answer =
            strip_unclosed_primed_thinking(parsed.content, &result.text, primed_open_thinking);
        let shaped = crate::server::shape_response(
            reasoning_format,
            answer.clone(),
            || tool_calls::content_with_thinking_block(&result.text, &answer, reasoning.as_deref()),
            reasoning.clone(),
        );
        return Ok(Json(
            ChatCompletionResponse::new_with_logprobs(
                request_id,
                model_id,
                shaped.content,
                result.prompt_tokens,
                result.completion_tokens,
                Some(result.finish_reason),
                logprobs,
            )
            .with_cached_tokens(cached_tokens, prompt_cache_enabled)
            .with_reasoning_content_alias_field(shaped.reasoning_content, reasoning_alias_field)
            .with_florence2_result(florence2_result),
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
    let shaped = crate::server::shape_response(
        reasoning_format,
        cleaned_text.clone(),
        || {
            tool_calls::content_with_thinking_block(
                &result.text,
                &cleaned_text,
                reasoning.as_deref(),
            )
        },
        reasoning.clone(),
    );

    // Warm the next turn's history prefix in the background (issue #1144).
    // Only the plain chat path does this: a tool-calling turn is echoed back as
    // a tool result rather than as assistant content, so the reply text is not
    // a reliable guess at the next prompt there.
    if let Some(ref ctx) = warmup_ctx {
        submit_next_turn_warmup(&state, &live, &request, ctx, &cleaned_text);
    }

    Ok(Json(
        ChatCompletionResponse::new_with_logprobs(
            request_id,
            model_id,
            shaped.content,
            result.prompt_tokens,
            result.completion_tokens,
            Some(result.finish_reason),
            logprobs,
        )
        .with_cached_tokens(cached_tokens, prompt_cache_enabled)
        .with_reasoning_content_alias_field(shaped.reasoning_content, reasoning_alias_field)
        .with_florence2_result(florence2_result),
    ))
}

/// Submit a background prompt-cache warm-up for the next turn (issue #1144).
///
/// Called after a healthy completion, once the reply text the client will echo
/// back is known. Renders the next turn's history prefix, tokenizes it, and
/// hands it to the worker; every step is best-effort and a failure at any point
/// simply leaves the conversation with its #1143 boundary snapshot, which is
/// still a hit on the next turn.
///
/// `ctx` is the same prompt-cache context the request carried, so the warm-up
/// lands in the bucket the next turn will look in.
fn submit_next_turn_warmup(
    state: &AppState,
    live: &LiveSettings,
    request: &ChatCompletionRequest,
    ctx: &PromptCacheRequestContext,
    reply: &str,
) {
    if state.prompt_cache.is_none()
        || crate::server::prompt_cache::boundary_snapshot_disabled()
        || crate::server::prompt_cache::cache_warmup_disabled()
    {
        return;
    }
    let Some(history) = crate::server::chat_request::render_next_turn_history(
        &state.chat_template,
        request,
        live.chat_template_kwargs.as_ref(),
        reply,
    ) else {
        return;
    };
    let tokenize = |text: &str| {
        crate::server::model_provider::tokenize_prompt_for_generation(&state.tokenizer, text).ok()
    };
    let (Some(probe_a), Some(probe_b)) = (tokenize(&history.probe_a), tokenize(&history.probe_b))
    else {
        return;
    };
    // Keep only the head both renders agree on. `probe` is a real generation
    // prompt for a hypothetical next turn, so the agreeing head is precisely
    // what any next turn reproduces: everything past it depends on how the
    // template renders the turn that follows, which this request cannot know.
    let Some(keep) = crate::server::chat_request::clip_warmup_target(&probe_a, &probe_b) else {
        return;
    };
    let tokens: Vec<i32> = probe_a[..keep].to_vec();
    tracing::debug!(
        probe = probe_a.len(),
        kept = tokens.len(),
        "prompt-cache warm-up submitted"
    );
    let mut warm_ctx = ctx.clone();
    // The vector travels as the job payload; the context only carries key
    // identity from here on.
    warm_ctx.history_prompt = None;
    warm_ctx.history_prefix_tokens = None;
    state
        .model_provider
        .submit_prompt_cache_warmup(tokens, warm_ctx);
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
    /// User-facing content accumulated for the post-completion prompt-cache
    /// warm-up (issue #1144). This is the filtered `delta.content` stream, not
    /// the raw generation, because it is what the client receives and therefore
    /// what it echoes back as the assistant turn. Only appended to when a
    /// warm-up is actually possible for this request.
    warmup_content: String,
    /// Stream filter that splits reasoning/content and strips structural tokens.
    stream_filter: StreamFilter,
    /// Per-`feed()` logprob buffer, drained in lockstep with the filter's
    /// consumed/suppressed positions. Only used when logprobs are enabled.
    lp_buffer: std::collections::VecDeque<Option<TokenLogprobData>>,
    /// Thinking-delimiter echo for `--reasoning-format none` /
    /// `deepseek-legacy` (#1470).
    thinking_echo: ThinkingDelimiterEcho,
}

/// Re-emits the literal thinking delimiters into `delta.content` under
/// `--reasoning-format none` and `deepseek-legacy` (#1470).
///
/// Both placements keep the thoughts in `message.content` **with their tags**,
/// and the non-streaming path rebuilds `{open}{thoughts}{close}{answer}` from
/// the raw text. The streaming path has no raw text: [`StreamFilter`] consumes
/// a delimiter as it matches it, and a generation prompt that primed the block
/// open means no open marker is ever generated at all. This carries the two
/// halves of that problem: the marker the filter reports for the delimiters it
/// did match, and the family resolved from the primed close marker for the
/// open it will never see.
pub(crate) struct ThinkingDelimiterEcho {
    /// The canonical open marker to synthesize before the first reasoning
    /// fragment when the prompt primed the block open; `None` when it did not,
    /// in which case the model generates its own open marker and the filter
    /// reports it.
    primed_open: Option<&'static str>,
    /// Whether an open marker has already reached `delta.content`, so a second
    /// `<think>` inside one generation cannot double it.
    opened: bool,
}

impl ThinkingDelimiterEcho {
    /// `primed_close` is [`primed_open_thinking_close_marker`] for this
    /// request's prompt.
    pub(crate) fn new(primed_close: Option<&str>) -> Self {
        Self {
            primed_open: primed_close
                .and_then(tool_calls::thinking_marker_pair_for_close)
                .map(|(open, _)| open),
            opened: false,
        }
    }

    /// The open marker to write ahead of this call's reasoning text, if any.
    pub(crate) fn open(&mut self, emit: &FilterOutput) -> Option<&'static str> {
        if self.opened {
            return None;
        }
        if let Some(open) = emit.thinking_open {
            self.opened = true;
            return Some(open);
        }
        // Primed case: the block was opened by the prompt, so the first
        // reasoning fragment is where the marker belongs.
        if emit.reasoning.is_some()
            && let Some(open) = self.primed_open
        {
            self.opened = true;
            return Some(open);
        }
        None
    }

    /// The close marker to write after this call's reasoning text, if any.
    /// Independent of there being reasoning text: a close marker can arrive on
    /// its own fragment.
    pub(crate) fn close(&self, emit: &FilterOutput) -> Option<&'static str> {
        emit.thinking_close
    }
}

/// Resumable-stream request context (#1444): the `X-Conversation-Id` value,
/// if any, and the API-key identity that owns the session and the
/// completion-control entry.
pub(crate) struct ResumableStreamContext {
    pub(crate) conversation_id: Option<String>,
    pub(crate) owner: crate::server::stream_session::StreamOwner,
}

/// Stream one ASR completion as b10621's transcript events (issue #1446).
///
/// b10621's transcription route is a translation layer over
/// `/v1/chat/completions`, so its streamed response is the same generation as
/// the chat stream with a different envelope: one `transcript.text.delta` per
/// decoded token, then `transcript.text.done` carrying the whole transcript
/// and the usage block, then `data: [DONE]`. mlxcel used to run the
/// non-streaming completion and emit the finished transcript as a single
/// delta, which matched the frame shapes but not the delivery granularity.
///
/// This is deliberately not [`stream_chat_completion`] with a different
/// serializer. That function owns tool-call accumulation, logprob alignment,
/// reasoning placement, resumable sessions and completion control, none of
/// which a transcript event can carry. What the two share is the part that
/// matters for equality with the non-streaming answer: the same request
/// preparation, the same generate options, and the same [`StreamFilter`], so
/// the concatenated deltas are the text `non_stream_chat_completion` would
/// have returned rather than the raw token stream with the model's structural
/// and thinking tokens left in.
pub(crate) async fn stream_asr_completion(
    state: AppState,
    live: std::sync::Arc<LiveSettings>,
    request: ChatCompletionRequest,
    priority: RequestPriority,
) -> Response {
    use super::transcription_compat::{
        AsrUsage, transcription_delta_event, transcription_done_event,
    };

    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let prompt_cache_enabled = state.prompt_cache.is_some();
    let prepared = match prepare_chat_request_with_cache(
        &state.chat_template,
        &request,
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    let prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &live,
        &request,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );
    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    // An ASR request carries no tools and no grammar, so the loop-detection
    // amplifier is off for the same reason it is off for a plain completion.
    let mut options =
        build_generate_options_with_live(&request.params, &state.config, &live, false);
    options.priority = priority;
    options.prompt_cache_ctx = prompt_cache_ctx;
    options.image_soft_tokens = prepared.image_soft_tokens;
    options.thinking_enter_block_on_start = primed_open_thinking;

    let queue_reservation = match state.model_provider.reserve_single_stream_queue_slot() {
        Ok(reservation) => reservation,
        Err(err) => return generation_error_to_response(err).into_response(),
    };

    // A transcript stream is not resumable: b10621 mounts no conversation id
    // on this route, so there is no session to tee into.
    let (events, stream, cancelled, keepalive) =
        sse_channel_resumable(100, state.config.sse_ping_interval, None);

    let skip_chat_parsing = state.config.skip_chat_parsing;
    let token_events = events.clone();

    tokio::task::spawn_blocking(move || {
        let slot = state.slots.begin(
            &prepared.prompt,
            super::slots::slot_params_json(&options, true),
            Some(options.max_tokens as i64),
        );
        let filter = std::sync::Mutex::new(if primed_open_thinking {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        });
        // The transcript as the client will have seen it, assembled from the
        // deltas actually sent so the terminal `done` event cannot disagree
        // with their concatenation.
        let transcript = std::sync::Mutex::new(String::new());

        let emit = |text: &str| {
            if text.is_empty() {
                return;
            }
            if let Ok(mut acc) = transcript.lock() {
                acc.push_str(text);
            }
            let _ = token_events.json(&transcription_delta_event(text));
        };

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos_declared_reserved(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                prepared.videos,
                prepared.media,
                queue_reservation,
                cancelled,
                |token, _lp_data| {
                    slot.on_token(&token);
                    // `--skip-chat-parsing` (#1447) turns every parser off on
                    // this surface too, so the token goes out verbatim.
                    if skip_chat_parsing {
                        emit(&token);
                        return;
                    }
                    // Only `content` is transcript: a reasoning fragment is
                    // the model's scratchpad, which the non-streaming answer
                    // does not put in `content` either.
                    let piece = match filter.lock() {
                        Ok(mut f) => f.feed(&token).content,
                        Err(_) => return,
                    };
                    if let Some(text) = piece {
                        emit(&text);
                    }
                },
            );

        // Whatever the filter was still holding for delimiter matching.
        if !skip_chat_parsing
            && let Ok(mut f) = filter.lock()
            && let Some(text) = f.flush().content
        {
            drop(f);
            emit(&text);
        }

        match result {
            Ok(r) => {
                slot.finish(
                    r.prompt_tokens,
                    r.cached_tokens,
                    r.completion_tokens,
                    &r.text,
                );
                state.metrics.record_request(
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.generation_time_ms,
                );
                let usage = AsrUsage {
                    input_tokens: r.prompt_tokens as u32,
                    output_tokens: r.completion_tokens as u32,
                    cached_tokens: r.cached_tokens as u32,
                };
                let text = transcript
                    .lock()
                    .map(|acc| acc.clone())
                    .unwrap_or_else(|_| r.text.clone());
                let _ = events.json(&transcription_done_event(&text, usage));
            }
            Err(err) => {
                // The stream is already open, so the failure has to travel as
                // a frame. Upstream shapes an in-stream failure as an `error`
                // object on the event stream, before its own terminator.
                let _ = events.json(&serde_json::json!({
                    "error": { "message": err.to_string(), "type": "server_error" }
                }));
            }
        }
        events.done();
    });

    sse_response(stream, keepalive)
}

async fn stream_chat_completion(
    state: AppState,
    live: std::sync::Arc<LiveSettings>,
    request: ChatCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
    resumable: ResumableStreamContext,
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
        live.chat_template_kwargs.as_ref(),
        prompt_cache_enabled,
        state.should_render_history_boundary_snapshot(),
        state.prefill_assistant(),
        &state.thinking_markers,
    )
    .await;
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    // b10621 `echo` (#1470), captured before `prepared` is consumed.
    let echo_prefill = request
        .resolve_echo()
        .then(|| prepared.assistant_prefill.clone())
        .flatten();
    // Build the prompt-cache context AFTER preparation so the multimodal
    // digest sees the resolved image/audio bytes.
    let prompt_cache_ctx = build_prompt_cache_request_context(
        &state,
        &live,
        &request,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );
    // Retained for the post-completion warm-up (issue #1144), same as the
    // non-streaming path. `warmup_enabled` gates the per-token accumulation in
    // the callback so a request that can never warm up pays nothing for it.
    let warmup_ctx = prompt_cache_ctx.as_ref().map(|ctx| {
        let mut c = ctx.clone();
        c.history_prompt = None;
        c
    });
    let warmup_enabled = warmup_ctx.is_some()
        && !crate::server::prompt_cache::boundary_snapshot_disabled()
        && !tool_calls::should_parse_tool_calls(&request);
    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    // The family whose block the prompt primed open, for the #1470 delimiter
    // echo: with the open marker in the prompt rather than the generation, the
    // streamed content has to synthesize it, exactly as the non-streaming
    // `content_with_thinking_block` does from the close marker in the raw text.
    let primed_close_marker =
        primed_open_thinking_close_marker(&state.thinking_markers, &prepared.prompt);
    // Loop-detection amplifier signal (issue #967): same derivation as the
    // non-streaming path, so both chat surfaces resolve identically.
    let amplified = chat_carries_loop_amplifier(&request);
    let mut options =
        build_generate_options_with_live(&request.params, &state.config, &live, amplified);
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
            source: Default::default(),
        };
    }

    let queue_reservation = match state.model_provider.reserve_single_stream_queue_slot() {
        Ok(reservation) => reservation,
        Err(err) => return generation_error_to_response(err).into_response(),
    };

    // b10621 `--skip-chat-parsing` forces a pure content parser, so tool calls
    // are NOT extracted (issue #1447). Gating here rather than at the emission
    // site turns off the accumulator, the end-of-stream parse, and the
    // `finish_reason` override together: leaving any of them on would report
    // the same call twice, once as the raw syntax in `delta.content` and once
    // as a tool-call delta.
    let parse_tools =
        !state.config.skip_chat_parsing && tool_calls::should_parse_tool_calls(&request);
    let tools_for_parser = if parse_tools {
        request.tools.clone()
    } else {
        None
    };
    let tool_choice = request.tool_choice.clone();

    // b10621 `reasoning_control` (#1444): arm realtime reasoning control for
    // this completion. The shared flag travels into the scheduler's thinking
    // tracker, and the registration makes the completion addressable by
    // `POST /v1/chat/completions/control` under its `chatcmpl-...` id for
    // exactly the lifetime of the generation task (the guard moves into the
    // blocking task below and drops when it exits, on every path).
    let control_force = (request.params.reasoning_control == Some(true))
        .then(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
    options.reasoning_control = control_force.clone();
    let control_registration = state.completion_controls.register(
        request_id.clone(),
        control_force,
        resumable.owner.clone(),
    );

    // b10621 resumable stream (#1444): with `X-Conversation-Id` present,
    // create (or replace) the conversation's session and tee every SSE
    // payload into it. The session's cancellation token replaces the
    // per-connection one, so a client disconnect no longer aborts the
    // generation; `DELETE /v1/stream` and session replacement do.
    let session = resumable.conversation_id.as_deref().map(|cid| {
        state
            .stream_sessions
            .create_or_replace(cid, resumable.owner.clone())
    });

    // sse_channel also returns an SseKeepAlive that sends periodic
    // SSE comment events. This prevents proxy/client idle-timeout disconnects
    // during long prefill phases (32k+ token prompts) where no token event is
    // emitted until the first generated token arrives.
    let (events, stream, cancelled, keepalive) =
        sse_channel_resumable(100, state.config.sse_ping_interval, session);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();
    let tokenizer = state.tokenizer.clone();

    // Clones for the post-completion warm-up (issue #1144), taken only when a
    // warm-up is actually possible so the ordinary streaming path allocates
    // nothing extra.
    let warmup_state = warmup_enabled.then(|| state.clone());
    let warmup_request = warmup_enabled.then(|| request.clone());

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Keep the completion controllable for exactly the lifetime of this
        // task; the guard unregisters the id on drop, whatever exit path
        // the task takes (#1444).
        let _control_registration = control_registration;
        // Slot accounting (#1440): ties this request to a numbered slot for
        // GET /slots. Taken before `options` moves into the provider call.
        let slot = state.slots.begin(
            &prepared.prompt,
            super::slots::slot_params_json(&options, true),
            Some(options.max_tokens as i64),
        );
        // Send initial chunk with role
        let initial =
            ChatCompletionChunk::initial(request_id_clone.clone(), model_id_clone.clone());
        let _ = finish_events.json(&initial);

        // b10621 `echo` (#1470): a prefilled assistant message leads the
        // response when `echo` is set. Upstream reaches the same shape by not
        // priming its chat parser, so the first diff already carries the
        // continuation text; here it is one content delta ahead of the token
        // stream, which is byte-identical once the deltas are concatenated.
        if let Some(prefill) = echo_prefill.as_deref() {
            let chunk = ChatCompletionChunk::content_with_logprobs(
                request_id_clone.clone(),
                model_id_clone.clone(),
                prefill.to_string(),
                None,
            );
            let _ = finish_events.json(&chunk);
        }

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
        // b10621 chat-parsing and reasoning-placement settings (issue #1447),
        // read once so both the per-token callback and the flush below see the
        // same values without reaching back into the shared state.
        let skip_chat_parsing = state.config.skip_chat_parsing;
        let reasoning_format = state.config.reasoning_format;
        let reasoning_alias_field = state.config.reasoning_alias_field;

        let cb_state = std::sync::Arc::new(std::sync::Mutex::new(StreamCallbackState {
            accumulated: String::new(),
            warmup_content: String::new(),
            stream_filter: if primed_open_thinking {
                StreamFilter::new_primed_open_thinking()
            } else {
                StreamFilter::new()
            },
            lp_buffer: std::collections::VecDeque::new(),
            thinking_echo: ThinkingDelimiterEcho::new(primed_close_marker.as_deref()),
        }));
        let cb_state_for_callback = cb_state.clone();

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos_declared_reserved_live(
                prepared.prompt,
                options,
                prepared.image_data,
                prepared.audio_data,
                prepared.videos,
                prepared.media,
                queue_reservation,
                cancelled,
                &live,
                |token, lp_data| {
                    slot.on_token(&token);
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
                        //
                        // b10621 `--skip-chat-parsing` (issue #1447) turns every
                        // parser off, so the filter is bypassed and the token is
                        // emitted verbatim. Synthesizing the pass-through output
                        // rather than skipping the block keeps the logprob
                        // bookkeeping below on one code path: one position in,
                        // one position emitted, none suppressed.
                        let emit = if skip_chat_parsing {
                            FilterOutput {
                                content: Some(token.clone()),
                                reasoning: None,
                                suppressed_positions: 0,
                                consumed_positions: 1,
                                thinking_open: None,
                                thinking_close: None,
                            }
                        } else {
                            cb.stream_filter.feed(&token)
                        };

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

                        // Where the thoughts go is `--reasoning-format`'s
                        // decision: `deepseek` / `auto` route them to
                        // `delta.reasoning_content` only, `deepseek-legacy` to
                        // both, and `none` to `delta.content` only.
                        // #1470: under `none` / `deepseek-legacy` the thoughts
                        // reach `delta.content` WITH their literal delimiters,
                        // so the concatenated stream equals the non-streaming
                        // `message.content` byte for byte. Resolved before the
                        // reasoning text is moved, and outside the
                        // non-empty-reasoning guard, because a close marker can
                        // arrive on a fragment that carries no reasoning.
                        let (echo_open, echo_close) =
                            if reasoning_format.keeps_thoughts_in_content() {
                                (cb.thinking_echo.open(&emit), cb.thinking_echo.close(&emit))
                            } else {
                                (None, None)
                            };
                        if let Some(reasoning_text) = emit.reasoning
                            && !reasoning_text.is_empty()
                        {
                            if reasoning_format.emits_reasoning_content() {
                                pending.push(
                                    ChatCompletionChunk::reasoning_content_with_alias_field(
                                        request_id_inner.clone(),
                                        model_id_inner.clone(),
                                        reasoning_text.clone(),
                                        reasoning_alias_field,
                                    ),
                                );
                            }
                            if reasoning_format.keeps_thoughts_in_content() {
                                let mut text = String::with_capacity(reasoning_text.len() + 24);
                                if let Some(open) = echo_open {
                                    text.push_str(open);
                                }
                                text.push_str(&reasoning_text);
                                pending.push(ChatCompletionChunk::content_with_logprobs(
                                    request_id_inner.clone(),
                                    model_id_inner.clone(),
                                    text,
                                    None,
                                ));
                            }
                        } else if let Some(open) = echo_open {
                            // An open marker whose fragment carried no thoughts
                            // yet still belongs in the stream.
                            pending.push(ChatCompletionChunk::content_with_logprobs(
                                request_id_inner.clone(),
                                model_id_inner.clone(),
                                open.to_string(),
                                None,
                            ));
                        }
                        if let Some(close) = echo_close {
                            pending.push(ChatCompletionChunk::content_with_logprobs(
                                request_id_inner.clone(),
                                model_id_inner.clone(),
                                close.to_string(),
                                None,
                            ));
                        }

                        if let Some(text) = emit.content
                            && !text.is_empty()
                        {
                            if warmup_enabled {
                                cb.warmup_content.push_str(&text);
                            }
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
            if reasoning_format.emits_reasoning_content() {
                let chunk = ChatCompletionChunk::reasoning_content_with_alias_field(
                    request_id_clone.clone(),
                    model_id_clone.clone(),
                    text.clone(),
                    reasoning_alias_field,
                );
                let _ = finish_events.json(&chunk);
            }
            if reasoning_format.keeps_thoughts_in_content() {
                let chunk = ChatCompletionChunk::content_with_logprobs(
                    request_id_clone.clone(),
                    model_id_clone.clone(),
                    text,
                    None,
                );
                let _ = finish_events.json(&chunk);
            }
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

        if let Ok(r) = &result {
            slot.finish(
                r.prompt_tokens,
                r.cached_tokens,
                r.completion_tokens,
                &r.text,
            );
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

        // Warm the next turn's history prefix in the background (issue #1144).
        // Submitted after the stream's content is complete and before the
        // terminal chunks, on the generation thread that is about to go idle.
        // Skipped when the turn produced tool calls: that reply is echoed back
        // as a tool result, not as assistant content.
        if finish_reason != "tool_calls"
            && let (Some(state), Some(request), Some(ctx)) =
                (&warmup_state, &warmup_request, &warmup_ctx)
        {
            let reply = cb_state
                .lock()
                .map(|cb| cb.warmup_content.clone())
                .unwrap_or_default();
            submit_next_turn_warmup(state, &live, request, ctx, &reply);
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

    sse_response(stream, keepalive)
}

/// Returns `true` when the request body carries at least one `video_url`
/// content part anywhere in `messages`. The check matches before the heavier
/// `extract_chat_video_paths` resolution step so non-video-capable models
/// can refuse the request without paying canonicalisation / disk-I/O cost.
/// Retained as the narrow `video_url`-only predicate the modality gate
/// subsumes; the shared `media_capability_rejection` covers image, audio and
/// video together (issue #1451).
#[cfg_attr(not(test), allow(dead_code))]
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

/// Validate the top-n-sigma sampling parameter.
///
/// `top_n_sigma` must be finite and `>= 0.0` when set (`0.0` disables the
/// filter); an absent field (`None`) is always valid and resolves to the
/// disabled baseline. Returns `Err` with a client-facing message so the
/// caller can surface a 400 `invalid_request_error` before any generation
/// work begins.
pub(crate) fn validate_top_n_sigma(top_n_sigma: Option<f32>) -> Result<(), &'static str> {
    if let Some(value) = top_n_sigma
        && !(value.is_finite() && value >= 0.0)
    {
        return Err("top_n_sigma must be >= 0.0");
    }
    Ok(())
}

/// Validate the locally-typical-sampling parameter.
///
/// `typical_p` must be finite and in `(0.0, 1.0]` when set (`1.0` disables
/// the filter); an absent field (`None`) is always valid and lets the
/// server-wide `--typical` default apply. `0.0` is rejected rather than
/// treated as "keep one token": b10621's schema leaves the low end of the
/// range undeclared, and a zero cutoff would otherwise mask every token.
pub(crate) fn validate_typical_p(typical_p: Option<f32>) -> Result<(), &'static str> {
    if let Some(value) = typical_p
        && !(value.is_finite() && value > 0.0 && value <= 1.0)
    {
        return Err("typical_p must be in (0.0, 1.0]");
    }
    Ok(())
}

/// Maximum number of tools allowed in a single request.
pub(crate) const MAX_TOOLS: usize = 128;

/// Validate the chat tool fields shared by single-node and router fronts.
///
/// Both callers run this before chat-template rendering so an invalid
/// `tool_choice` or oversized tool list cannot reach Jinja2. Error strings are
/// intentionally kept byte-identical to the original single-node guards.
pub(crate) fn validate_chat_tool_inputs(request: &ChatCompletionRequest) -> Result<(), String> {
    if let Some(crate::server::types::request::ToolChoice::Mode(mode)) = &request.tool_choice
        && !["auto", "none", "required"].contains(&mode.as_str())
    {
        return Err(format!(
            "Invalid tool_choice value: '{mode}'. Must be 'auto', 'none', 'required', or a function object."
        ));
    }

    if let Some(tools) = &request.tools
        && tools.len() > MAX_TOOLS
    {
        return Err(format!(
            "Too many tools: {}. Maximum allowed is {MAX_TOOLS}.",
            tools.len()
        ));
    }

    Ok(())
}

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
pub(crate) fn is_prompt_primed_open_thinking(
    markers: &crate::tokenizer::ThinkingMarkers,
    prompt: &str,
) -> bool {
    if markers.has_thinking() {
        return crate::reasoning_stream::prompt_primed_open_thinking(markers, prompt);
    }
    legacy_primed_close_marker(prompt).is_some()
}

/// Suffix table this check used before it became marker-driven, kept as a
/// compatibility fallback for a tokenizer that declares no thinking markers.
///
/// The marker-driven path is strictly better when markers resolve, because it
/// reads the model's own spelling and tolerates the trailing whitespace a
/// template may strip. But it can only run when
/// [`MlxcelTokenizer::infer_thinking_markers`] finds its pair in the vocab, and
/// a model whose template primes `<think>` without declaring the token would
/// have gone from "primed" to "not primed" across this change. That is a
/// regression in the budget-accounting and unclosed-stripping paths even though
/// the reasoning filter itself is inert without markers, so the old behaviour is
/// preserved rather than assumed unreachable.
///
/// Every checkpoint on hand resolves markers (Gemma 4 through
/// `<|channel>` / `<channel|>`, Qwen and DeepSeek-V4 through
/// `<think>` / `</think>`), so this path is expected to be dead. It exists
/// because "expected to be dead" is not the same as "cannot be reached".
const LEGACY_OPEN_THINKING_SUFFIXES: &[(&str, &str)] = &[
    ("<|channel>thought\n", "<channel|>"),
    ("<think>\n", "</think>"),
];

fn legacy_primed_close_marker(prompt: &str) -> Option<&'static str> {
    LEGACY_OPEN_THINKING_SUFFIXES
        .iter()
        .find(|(suffix, _)| prompt.ends_with(*suffix))
        .map(|(_, close)| *close)
}

/// The close marker the model is expected to emit for the thinking block this
/// generation prompt primed, or `None` when it primed none.
///
/// Resolved from the tokenizer's own markers rather than a table of known
/// prompt suffixes (issue #1554). The table carried its trailing newline as a
/// literal, so a family whose template strips that newline (DeepSeek-V4 renders
/// a bare `<think>`) read as "not primed": the reasoning filter never entered
/// its thinking state, and the trace was dropped on a completed generation or
/// leaked into `content` when `max_tokens` cut it short. Deriving from markers
/// covers every family whose tokenizer declares them, and shares one
/// implementation with the CLI check that always did it this way.
///
/// Used by: server::chat_request (b10621 assistant prefill, #1470)
pub(crate) fn primed_open_thinking_close_marker(
    markers: &crate::tokenizer::ThinkingMarkers,
    prompt: &str,
) -> Option<String> {
    if markers.has_thinking() {
        return crate::reasoning_stream::prompt_primed_open_close_marker(markers, prompt);
    }
    legacy_primed_close_marker(prompt).map(str::to_string)
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
pub(crate) fn extract_reasoning_content(
    raw_text: &str,
    primed_open_thinking: bool,
) -> Option<String> {
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
/// [`crate::server::request_options::build_server_generate_options`], but since
/// issue #967 it also requires the request to carry an amplifier. `params`
/// carries no tools and no `response_format`, so that signal cannot be derived
/// here and every caller must pass it explicitly as
/// `request_carries_loop_amplifier`. Chat-shaped callers compute it with
/// [`crate::server::request_options::chat_carries_loop_amplifier`]; raw-prompt
/// endpoints that accept neither tools nor a schema pass `false`.
#[allow(dead_code)]
pub(crate) fn build_generate_options(
    params: &SamplingParams,
    config: &ServerConfig,
    request_carries_loop_amplifier: bool,
) -> ServerGenerateOptions {
    build_generate_options_with_live(
        params,
        config,
        &config.live_settings(),
        request_carries_loop_amplifier,
    )
}

/// Build chat generation options from one captured settings snapshot.
pub(crate) fn build_generate_options_with_live(
    params: &SamplingParams,
    config: &ServerConfig,
    live: &LiveSettings,
    request_carries_loop_amplifier: bool,
) -> ServerGenerateOptions {
    // OpenAI `logit_bias` (#1485): keys are token ids as strings; a key
    // that is not an integer is tokenized at enqueue time and the bias
    // applies to every resulting token, the same string-key form b10621's
    // schema handler accepts. A present (even empty) map replaces the
    // server-wide `--logit-bias` set, as upstream's handler clears and
    // rebuilds the list.
    let (logit_bias, logit_bias_texts) = match &params.logit_bias {
        Some(map) => {
            let mut nums = Vec::new();
            let mut texts = Vec::new();
            for (key, &bias) in map {
                match key.parse::<i32>() {
                    Ok(id) => nums.push((id, bias)),
                    Err(_) => texts.push((key.clone(), bias)),
                }
            }
            // Deterministic application order across the HashMap.
            nums.sort_by_key(|&(id, _)| id);
            texts.sort_by(|a, b| a.0.cmp(&b.0));
            (Some(nums), texts)
        }
        None => (None, Vec::new()),
    };

    build_server_generate_options_with_live(
        config,
        live,
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
            top_n_sigma: params.top_n_sigma,
            typical_p: params.typical_p,
            // b10621 repeat_last_n / mlx-lm repetition_context_size: the
            // usize deserialization already rejects negatives (upstream's
            // schema floor is 0), so only the i32 clamp remains.
            penalty_last_n: params
                .repetition_context_size
                .map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
            // The OpenAI schema has no ignore_eos; the server-wide
            // --ignore-eos default still applies through the resolution.
            ignore_eos: None,
            stop_sequences: params.stop.clone(),
            priority: RequestPriority::default(),
            // the caller (non_stream_chat_completion /
            // stream_chat_completion) sets `options.reasoning_budget`
            // explicitly after `build_generate_options` returns, so the
            // default here is just a placeholder.
            reasoning_budget: ReasoningBudgetOverride::default(),
            // Placeholder: the caller overrides this with
            // `is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt)` once the
            // chat template has rendered, so the value picked up by the
            // scheduler matches the actual prompt tail (Qwen `<think>\n`,
            // Gemma 4 `<|channel>thought\n`, or neither).
            thinking_enter_block_on_start: false,
            loop_detection_request: crate::server::request_options::loop_detection_from_request(
                params.max_pattern_size,
                params.min_pattern_size,
                params.min_count,
            ),
            request_carries_loop_amplifier,
            // #1485: the OpenAI-shaped surface exposes only logit_bias; the
            // mirostat / dynatemp / adaptive-p / min_keep / n_probs fields
            // are native-`/completion` schema, so the server-wide defaults
            // resolve for them.
            logit_bias,
            logit_bias_texts,
            dry_sequence_breaker_strings: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            adaptive_target: None,
            adaptive_decay: None,
            adaptive_p_named: None,
            min_keep: None,
            n_probs: None,
            post_sampling_probs: None,
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
    fn validate_top_n_sigma_accepts_unset_zero_and_positive() {
        assert!(validate_top_n_sigma(None).is_ok());
        assert!(validate_top_n_sigma(Some(0.0)).is_ok());
        assert!(validate_top_n_sigma(Some(1.0)).is_ok());
        assert!(validate_top_n_sigma(Some(100.0)).is_ok());
    }

    #[test]
    fn chat_rejects_negative_top_n_sigma() {
        assert_eq!(
            validate_top_n_sigma(Some(-1.0)),
            Err("top_n_sigma must be >= 0.0")
        );
        assert_eq!(
            validate_top_n_sigma(Some(-0.001)),
            Err("top_n_sigma must be >= 0.0")
        );
    }

    #[test]
    fn validate_top_n_sigma_rejects_non_finite() {
        assert_eq!(
            validate_top_n_sigma(Some(f32::NAN)),
            Err("top_n_sigma must be >= 0.0")
        );
        assert_eq!(
            validate_top_n_sigma(Some(f32::INFINITY)),
            Err("top_n_sigma must be >= 0.0")
        );
    }

    #[test]
    fn validate_typical_p_accepts_unset_disabled_and_in_range() {
        assert!(validate_typical_p(None).is_ok());
        assert!(validate_typical_p(Some(1.0)).is_ok());
        assert!(validate_typical_p(Some(0.5)).is_ok());
        assert!(validate_typical_p(Some(0.001)).is_ok());
    }

    #[test]
    fn chat_rejects_out_of_domain_typical_p() {
        for bad in [0.0f32, -0.5, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                validate_typical_p(Some(bad)),
                Err("typical_p must be in (0.0, 1.0]"),
                "typical_p={bad} must be rejected"
            );
        }
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

    // -- prompt-primed open-thinking detection (marker-driven, issue #1554) --

    fn markers_for(open: &str, close: &str) -> crate::tokenizer::ThinkingMarkers {
        crate::tokenizer::ThinkingMarkers {
            think_start: Some(open.to_string()),
            think_end: Some(close.to_string()),
            think_start_tokens: Some(vec![1]),
            think_end_tokens: Some(vec![2]),
            ..Default::default()
        }
    }

    #[test]
    fn prompt_primed_detection_matches_exact_suffix() {
        let m = markers_for("<|channel>thought", "<channel|>");
        assert!(is_prompt_primed_open_thinking(
            &m,
            "<|turn>model\n<|channel>thought\n"
        ));
    }

    #[test]
    fn prompt_primed_detection_rejects_closed_priming() {
        // `enable_thinking=false` templates end with the CLOSED priming.
        let m = markers_for("<|channel>thought", "<channel|>");
        assert!(!is_prompt_primed_open_thinking(
            &m,
            "<|turn>model\n<|channel>thought\n<channel|>\n"
        ));
    }

    #[test]
    fn prompt_primed_detection_rejects_unrelated_endings() {
        let m = markers_for("<|channel>thought", "<channel|>");
        assert!(!is_prompt_primed_open_thinking(&m, "<|turn>model\n"));
        assert!(!is_prompt_primed_open_thinking(&m, ""));
        assert!(!is_prompt_primed_open_thinking(&m, "some content"));
    }

    /// Issue #1554: DeepSeek-V4's template strips the newline after its open
    /// marker, so the prompt ends with a bare `<think>`. The previous
    /// implementation matched a table of literal suffixes that each carried a
    /// trailing `\n`, so this read as "not primed": the reasoning filter never
    /// entered its thinking state, and the trace was dropped on a completed
    /// generation or leaked into `content` when `max_tokens` cut it short.
    #[test]
    fn prompt_primed_detection_accepts_a_newline_free_open_marker() {
        let m = markers_for("<think>", "</think>");
        // The real DeepSeek-V4 render, tail-exact.
        assert!(is_prompt_primed_open_thinking(
            &m,
            "<\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}><think>"
        ));
        assert_eq!(
            primed_open_thinking_close_marker(
                &m,
                "<\u{ff5c}User\u{ff5c}>hi<\u{ff5c}Assistant\u{ff5c}><think>"
            )
            .as_deref(),
            Some("</think>")
        );
        // The Qwen-style newline form still works: whitespace is trimmed, not
        // required.
        assert!(is_prompt_primed_open_thinking(&m, "<think>\n"));
        // A closed block is not primed.
        assert!(!is_prompt_primed_open_thinking(&m, "<think>\n</think>"));
    }

    /// Without markers the check falls back to the pre-#1554 suffix table, so
    /// no prompt shape that was primed before this change stops being primed.
    /// The marker-driven path can only run when the tokenizer declares its
    /// pair, and a model that primes `<think>` without declaring the token
    /// would otherwise have silently lost its budget accounting and unclosed
    /// stripping.
    #[test]
    fn prompt_primed_detection_falls_back_to_the_legacy_table_without_markers() {
        let none = crate::tokenizer::ThinkingMarkers::default();

        // Both shapes the old table covered still read as primed.
        assert!(is_prompt_primed_open_thinking(&none, "<think>\n"));
        assert_eq!(
            primed_open_thinking_close_marker(&none, "<think>\n").as_deref(),
            Some("</think>")
        );
        assert!(is_prompt_primed_open_thinking(
            &none,
            "<|turn>model\n<|channel>thought\n"
        ));
        assert_eq!(
            primed_open_thinking_close_marker(&none, "<|turn>model\n<|channel>thought\n")
                .as_deref(),
            Some("<channel|>")
        );

        // And what the old table rejected is still rejected: the fallback is
        // the old behaviour exactly, not a looser version of it.
        assert!(!is_prompt_primed_open_thinking(&none, "<think>"));
        assert!(!is_prompt_primed_open_thinking(&none, ""));
        assert!(!is_prompt_primed_open_thinking(&none, "some content"));
        assert!(!is_prompt_primed_open_thinking(
            &none,
            "<|turn>model\n<|channel>thought\n<channel|>\n"
        ));
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
            Some("\ndeliberating".to_string())
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
            cache_prompt: None,
            user: None,
            reasoning_effort: None,
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

#[cfg(test)]
mod reasoning_format_route_tests {
    use crate::server::ReasoningFormat;
    use crate::server::tool_calls::{clean_structural_tokens, content_with_thinking_block};

    /// One Qwen-style generation with a thinking block, as the model emits it.
    const RAW: &str = "<think>Let me count.</think>The answer is 42.";

    /// The two content forms the route hands to `shape_response`, built the
    /// same way the route builds them so a change to either shows up here
    /// rather than only in a live request.
    fn content_forms(raw: &str) -> (String, String) {
        let answer = clean_structural_tokens(raw);
        let reasoning = super::extract_reasoning_content(raw, false);
        let with_thoughts = content_with_thinking_block(raw, &answer, reasoning.as_deref());
        (answer, with_thoughts)
    }

    #[test]
    fn the_thoughts_form_is_the_answer_with_its_block_restored() {
        let (answer, with_thoughts) = content_forms(RAW);
        assert_eq!(answer, "The answer is 42.");
        assert_eq!(with_thoughts, RAW, "byte-exact for a Qwen-style block");
    }

    #[test]
    fn each_format_places_the_thoughts_where_b10621_does() {
        let (answer, with_thoughts) = content_forms(RAW);
        for (format, expected_content, expected_reasoning) in [
            (
                ReasoningFormat::Auto,
                answer.as_str(),
                Some("Let me count."),
            ),
            (
                ReasoningFormat::DeepSeek,
                answer.as_str(),
                Some("Let me count."),
            ),
            (ReasoningFormat::None, with_thoughts.as_str(), None),
            (
                ReasoningFormat::DeepSeekLegacy,
                with_thoughts.as_str(),
                Some("Let me count."),
            ),
        ] {
            let shaped = crate::server::shape_response(
                format,
                answer.clone(),
                || with_thoughts.clone(),
                super::extract_reasoning_content(RAW, false),
            );
            assert_eq!(shaped.content, expected_content, "{format} content");
            assert_eq!(
                shaped.reasoning_content.as_deref(),
                expected_reasoning,
                "{format} reasoning_content"
            );
        }
    }

    /// Drive a whole generation through the streaming filter one fragment at a
    /// time and rebuild what a `--reasoning-format none` client would see in
    /// `delta.content`, the way the route builds it (#1470).
    fn streamed_content(raw: &str, fragments: usize, primed_close: Option<&'static str>) -> String {
        use crate::server::tool_calls::stream_filter::StreamFilter;
        let mut filter = if primed_close.is_some() {
            StreamFilter::new_primed_open_thinking()
        } else {
            StreamFilter::new()
        };
        let mut echo = super::ThinkingDelimiterEcho::new(primed_close);
        let mut out = String::new();
        let step = raw.len().div_ceil(fragments.max(1));
        let mut start = 0;
        let mut pieces: Vec<&str> = Vec::new();
        while start < raw.len() {
            let mut end = (start + step).min(raw.len());
            while end < raw.len() && !raw.is_char_boundary(end) {
                end += 1;
            }
            pieces.push(&raw[start..end]);
            start = end;
        }
        for piece in pieces {
            let emit = filter.feed(piece);
            let open = echo.open(&emit);
            let close = echo.close(&emit);
            match emit.reasoning.as_deref() {
                Some(reasoning) if !reasoning.is_empty() => {
                    if let Some(open) = open {
                        out.push_str(open);
                    }
                    out.push_str(reasoning);
                }
                _ => {
                    if let Some(open) = open {
                        out.push_str(open);
                    }
                }
            }
            if let Some(close) = close {
                out.push_str(close);
            }
            if let Some(content) = emit.content.as_deref() {
                out.push_str(content);
            }
        }
        let emit = filter.flush();
        if let Some(reasoning) = emit.reasoning.as_deref() {
            out.push_str(reasoning);
        }
        if let Some(content) = emit.content.as_deref() {
            out.push_str(content);
        }
        out
    }

    /// The `--reasoning-format none` stream carries the thinking delimiters,
    /// byte-identically to the non-streaming `message.content` (#1470).
    ///
    /// Before #1470 the `StreamFilter` consumed each delimiter as it matched
    /// it, so `delta.content` carried the thoughts without their tags, which is
    /// the very thing `none` asks for.
    #[test]
    fn a_streamed_none_response_carries_the_thinking_delimiters() {
        let (_, with_thoughts) = content_forms(RAW);
        for fragments in 1..=RAW.len().min(12) {
            assert_eq!(
                streamed_content(RAW, fragments, None),
                with_thoughts,
                "fragments = {fragments}"
            );
        }
    }

    /// The whitespace a model emits between its close marker and its answer is
    /// part of what `--reasoning-format none` keeps, and the streamed form
    /// passes it through verbatim, so the rebuilt non-streaming form has to
    /// carry it too (#1470).
    ///
    /// Caught by real-checkpoint validation: Qwen3 answering `</think>\n\n2
    /// plus 2 equals 4.` streamed 608 bytes against a 606-byte
    /// `message.content`, the two newlines the parser's trim had removed.
    #[test]
    fn the_gap_between_the_close_marker_and_the_answer_survives() {
        const GAPPED: &str = "<think>Let me count.</think>\n\nThe answer is 42.";
        let (answer, with_thoughts) = content_forms(GAPPED);
        assert_eq!(
            answer, "The answer is 42.",
            "the answer itself stays trimmed"
        );
        assert_eq!(with_thoughts, GAPPED, "byte-exact against the generation");
        for fragments in 1..=10 {
            assert_eq!(
                streamed_content(GAPPED, fragments, None),
                with_thoughts,
                "fragments = {fragments}"
            );
        }
    }

    /// The Gemma 4 family streams its own canonical pair, not the literal
    /// `<|channel>thought` opener the filter consumes (#1470).
    #[test]
    fn a_streamed_gemma_channel_block_carries_the_canonical_markers() {
        const GEMMA: &str = "<|channel>thinking<channel|>the answer";
        let (_, with_thoughts) = content_forms(GEMMA);
        for fragments in 1..=6 {
            assert_eq!(
                streamed_content(GEMMA, fragments, None),
                with_thoughts,
                "fragments = {fragments}"
            );
        }
    }

    /// A prompt-primed block never generates its open marker, so the stream
    /// synthesizes it, exactly as `content_with_thinking_block` does from the
    /// close marker present in the raw text (#1470).
    #[test]
    fn a_primed_block_synthesizes_its_open_marker_in_the_stream() {
        const PRIMED: &str = "Let me count.</think>The answer is 42.";
        let answer = clean_structural_tokens(PRIMED);
        let reasoning = super::extract_reasoning_content(PRIMED, true);
        let with_thoughts = content_with_thinking_block(PRIMED, &answer, reasoning.as_deref());
        for fragments in 1..=8 {
            assert_eq!(
                streamed_content(PRIMED, fragments, Some("</think>")),
                with_thoughts,
                "fragments = {fragments}"
            );
        }
    }

    /// Text with no thinking block gains no delimiters (#1470).
    #[test]
    fn a_streamed_plain_response_gains_no_delimiters() {
        const PLAIN: &str = "just an answer";
        assert_eq!(streamed_content(PLAIN, 3, None), PLAIN);
    }

    #[test]
    fn a_gemma_style_channel_block_keeps_its_delimiters_too() {
        // Reconstructing from the extracted reasoning preserves the Gemma
        // markers, which the raw-text cleaning pass strips out.
        const GEMMA: &str = "<|channel>thinking<channel|>the answer";
        let (answer, with_thoughts) = content_forms(GEMMA);
        assert_eq!(answer, "the answer");
        assert_eq!(with_thoughts, GEMMA);
    }

    #[test]
    fn the_thoughts_form_never_restores_tool_call_syntax() {
        // The whole reason the form is built from the parser's content rather
        // than from the raw text: restoring the raw text would report the same
        // call twice, once as `content` and once as `tool_calls`.
        const WITH_CALL: &str = concat!(
            "<think>I should call it.</think>",
            r#"<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>"#
        );
        let parsed = crate::server::tool_calls::parse_tool_calls(WITH_CALL, None);
        let reasoning = super::extract_reasoning_content(WITH_CALL, false);
        let with_thoughts =
            content_with_thinking_block(WITH_CALL, &parsed.content, reasoning.as_deref());
        assert!(
            with_thoughts.contains("I should call it."),
            "the thoughts must be kept: {with_thoughts:?}"
        );
        assert!(
            !with_thoughts.contains("get_weather"),
            "the tool call must NOT be restored into content: {with_thoughts:?}"
        );
    }

    #[test]
    fn text_without_a_thinking_block_is_identical_under_every_format() {
        const PLAIN: &str = "just an answer";
        let (answer, with_thoughts) = content_forms(PLAIN);
        assert_eq!(answer, with_thoughts);
        for format in [
            ReasoningFormat::Auto,
            ReasoningFormat::None,
            ReasoningFormat::DeepSeek,
            ReasoningFormat::DeepSeekLegacy,
        ] {
            let shaped = crate::server::shape_response(
                format,
                answer.clone(),
                || with_thoughts.clone(),
                super::extract_reasoning_content(PLAIN, false),
            );
            assert_eq!(shaped.content, PLAIN, "{format}");
            assert_eq!(shaped.reasoning_content, None, "{format}");
        }
    }
}
