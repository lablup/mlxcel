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

//! Native completion endpoint (llama-server /completion format)
//!
//! Different from /v1/completions — uses `n_predict` instead of `max_tokens`,
//! returns `{"content": "...", "stop": true, "timings": {...}}`.
//!
//! Like the OpenAI-compatible routes, this file stays as an HTTP adapter while
//! generation policy and SSE plumbing live in shared server modules.

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::media::MediaRequestMetadata;
use crate::server::model_provider::StopKind;
use crate::server::request_options::{
    RequestOptionOverrides, build_server_generate_options, resolve_server_max_tokens,
};
use crate::server::streaming::{sse_channel, sse_response};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::{
    ErrorResponse, NativeCompletionChunk, NativeCompletionRequest, NativeCompletionResponse,
    NativeTimings, StopType,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};

fn generation_error_to_response(err: anyhow::Error) -> ErrorResponse {
    super::generation_error_to_response(err)
}

/// POST /completion
pub async fn native_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<NativeCompletionRequest>,
) -> Response {
    if let Some(response) = super::chat_not_available(&state) {
        return response.into_response();
    }

    // the native `/completion` endpoint does not support
    // `response_format`. Reject up front with a clear 400 so the client
    // does not assume their schema was honored — the chat-completions
    // route is the supported path for constrained decoding.
    if let Some(value) = request.response_format.as_ref() {
        // Treat `{"type": "text"}` (and `null`) as a no-op rather than an
        // error since they explicitly disable structured output.
        let format_type = value
            .as_object()
            .and_then(|m| m.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if format_type != "text" {
            return ErrorResponse::new(
                "response_format is not supported on /completion; use /v1/chat/completions \
                 for structured-output / JSON-Schema constrained decoding"
                    .to_string(),
                "invalid_request_error",
            )
            .into_response();
        }
    }

    // validate thinking_budget_tokens early (semantics match
    // /v1/chat/completions but the cap is checked against n_predict).
    let effective_n_predict = resolve_server_max_tokens(&state.config, request.n_predict);
    let raw_budget = pick_budget_alias(
        request.thinking_budget_tokens,
        request.thinking_token_budget,
        request.thinking_budget,
    );
    let budget_override = match resolve_request_budget(
        raw_budget,
        state.config.reasoning_budget,
        effective_n_predict,
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

    // Native fields mlxcel cannot honor are rejected with a diagnostic naming
    // the field, rather than accepted and ignored (#1441). A value that cannot
    // change behavior (the upstream default, or an explicit `false`) is inert
    // and passes, so a client that sends the whole schema with defaults is not
    // turned away.
    if let Some(response) = reject_unsupported_native_fields(&request) {
        return response;
    }

    // b10621 declares `sse_ping_interval` with hard limits (-1, INT32_MAX) at
    // schema-parse time, so an out-of-domain value is rejected whether or not
    // the request streams (#1432).
    let ping_interval = match request.sse_ping_interval {
        Some(secs) => match crate::server::transport::resolve_sse_ping_interval(secs) {
            Ok(interval) => interval,
            Err(err) => {
                return ErrorResponse::new(err.to_string(), "invalid_request_error")
                    .into_response();
            }
        },
        None => state.config.sse_ping_interval,
    };

    let priority = parse_priority_header(&headers);
    if request.stream.unwrap_or(false) {
        stream_native_completion(state, request, priority, budget_override, ping_interval).await
    } else {
        non_stream_native_completion(state, request, priority, budget_override)
            .await
            .into_response()
    }
}

/// Extract the `X-Priority` header value, defaulting to `Normal`.
///
/// Delegated to the shared implementation in `super::chat`.
fn parse_priority_header(headers: &HeaderMap) -> RequestPriority {
    super::chat::parse_priority_header(headers)
}

async fn non_stream_native_completion(
    state: AppState,
    request: NativeCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
) -> Result<Json<NativeCompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let mut options = build_native_options(&request, &state);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    // The response echoes the settings that were actually used, so keep a copy
    // before the options are moved into the generator.
    let options_snapshot = options.clone();

    let result = state
        .model_provider
        .generate(request.prompt.clone(), options)
        .map_err(generation_error_to_response)?;

    let prompt_ms = result.prompt_eval_ms as f64;
    let gen_ms = result.generation_only_ms as f64;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    Ok(Json(build_native_response(
        &state,
        &request,
        &options_snapshot,
        NativeOutcome {
            content: result.text,
            tokens: Vec::new(),
            tokens_predicted: result.completion_tokens,
            tokens_evaluated: result.prompt_tokens,
            cached_tokens: result.cached_tokens,
            stop_kind: &result.stop_kind,
            prompt_ms,
            predicted_ms: gen_ms,
        },
    )))
}

/// The parts of a finished generation the native response is built from.
///
/// Grouped into one struct so the streaming and non-streaming paths build the
/// final object through exactly one function and cannot drift apart.
struct NativeOutcome<'a> {
    content: String,
    tokens: Vec<i32>,
    tokens_predicted: usize,
    tokens_evaluated: usize,
    cached_tokens: usize,
    /// Why generation ended, at b10621's granularity: the `n_predict` budget, an
    /// EOS token, or a string stop sequence, in which case it carries the
    /// matched string (issue #1466). `GenerationResult::finish_reason` is the
    /// OpenAI wire string and collapses the last two into one `"stop"`, so it
    /// cannot drive `stop_type` and is not carried here.
    stop_kind: &'a StopKind,
    prompt_ms: f64,
    predicted_ms: f64,
}

/// Assemble the b10621 native completion object.
///
/// `stop_type` and `stopping_word` come from the scheduler's `StopKind`, which
/// distinguishes a string stop-sequence match from an EOS token and from the
/// `n_predict` budget. The matched string cannot be recovered from `content`,
/// because b10621 (and mlxcel, since #1466) excludes it from the returned text.
fn build_native_response(
    state: &AppState,
    request: &NativeCompletionRequest,
    options: &ServerGenerateOptions,
    outcome: NativeOutcome<'_>,
) -> NativeCompletionResponse {
    let (stop_type, stopping_word) = match outcome.stop_kind {
        StopKind::Word(word) => (StopType::Word, word.clone()),
        StopKind::Limit => (StopType::Limit, String::new()),
        StopKind::Eos => (StopType::Eos, String::new()),
    };
    NativeCompletionResponse {
        index: 0,
        has_new_line: outcome.content.contains('\n'),
        content: outcome.content,
        tokens: outcome.tokens,
        // mlxcel's continuous-batching scheduler does not expose a stable
        // per-request slot number, so the "no numbered slot" sentinel upstream
        // uses on its own streaming frames is reported here as well.
        id_slot: -1,
        stop: true,
        model: state.display_model_id().to_string(),
        tokens_predicted: outcome.tokens_predicted,
        tokens_evaluated: outcome.tokens_evaluated,
        generation_settings: native_generation_settings(request, options),
        prompt: request.prompt.clone(),
        // The server clamps an over-long request rather than truncating the
        // prompt, so no request reaches here with a dropped prefix.
        truncated: false,
        stop_type,
        stopping_word,
        tokens_cached: outcome.cached_tokens,
        timings: NativeTimings::new(
            outcome.cached_tokens,
            outcome.tokens_evaluated,
            outcome.prompt_ms,
            outcome.tokens_predicted,
            outcome.predicted_ms,
        ),
    }
}

/// The resolved generation settings echoed back on the response.
///
/// b10621 echoes its whole `task_params` here (49 keys). mlxcel reports the
/// settings it actually resolved and acts on; a key mlxcel has no analogue for
/// is omitted rather than reported with an invented value, which would be a
/// worse answer than its absence.
fn native_generation_settings(
    request: &NativeCompletionRequest,
    options: &ServerGenerateOptions,
) -> serde_json::Value {
    let sampling = &options.sampling;
    serde_json::json!({
        "seed": sampling.seed.unwrap_or(u64::MAX),
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "min_p": sampling.min_p,
        "xtc_probability": sampling.xtc_probability,
        "xtc_threshold": sampling.xtc_threshold,
        "repeat_penalty": sampling.repetition_penalty,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "dry_multiplier": sampling.dry_multiplier,
        "dry_base": sampling.dry_base,
        "dry_allowed_length": sampling.dry_allowed_length,
        "dry_penalty_last_n": sampling.dry_penalty_last_n,
        "dry_sequence_breakers": sampling.dry_sequence_breakers,
        "stop": options.stop_sequences.clone().unwrap_or_default(),
        "max_tokens": options.max_tokens,
        "n_predict": options.max_tokens,
        "stream": request.stream.unwrap_or(false),
    })
}

async fn stream_native_completion(
    state: AppState,
    request: NativeCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    ping_interval: Option<std::time::Duration>,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let mut options = build_native_options(&request, &state);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    let options_snapshot = options.clone();
    let prompt = request.prompt.clone();

    let queue_reservation = match state.model_provider.reserve_single_stream_queue_slot() {
        Ok(reservation) => reservation,
        Err(err) => return generation_error_to_response(err).into_response(),
    };

    // sse_channel also returns an SseKeepAlive for proxy idle-timeout
    // prevention during long prefill phases. The interval is the per-request
    // `sse_ping_interval` when the client sent one, otherwise the server's
    // `--sse-ping-interval` (#1432); the caller resolved and validated it.
    let (events, stream, cancelled, keepalive) = sse_channel(100, ping_interval);
    let finish_events = events.clone();
    let timings_per_token = request.timings_per_token.unwrap_or(false);
    let started = std::time::Instant::now();

    tokio::task::spawn_blocking(move || {
        let token_events = finish_events.clone();
        let emitted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let emitted_for_tokens = emitted.clone();

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable_videos_declared_reserved(
                prompt,
                options,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                MediaRequestMetadata::default(),
                queue_reservation,
                cancelled,
                |token, _lp| {
                    let n =
                        emitted_for_tokens.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    // Upstream's per-token frame is small: the full metadata
                    // block belongs to the final frame alone. `timings` is
                    // attached here only under `timings_per_token`.
                    let chunk = NativeCompletionChunk {
                        index: 0,
                        content: token.to_string(),
                        tokens: Vec::new(),
                        stop: false,
                        id_slot: -1,
                        tokens_predicted: n,
                        tokens_evaluated: 0,
                        timings: timings_per_token.then(|| {
                            NativeTimings::new(0, 0, 0.0, n, started.elapsed().as_millis() as f64)
                        }),
                        prompt_progress: None,
                    };
                    let _ = token_events.json(&chunk);
                },
            );

        // The final frame is the whole non-streaming object with `stop: true`,
        // built through the same function so the two shapes cannot drift.
        // There is no `[DONE]` sentinel: b10621's native stream ends with this
        // frame, and emitting one would be an extra event no llama-server
        // client expects.
        match result {
            Ok(result) => {
                let final_frame = build_native_response(
                    &state,
                    &request,
                    &options_snapshot,
                    NativeOutcome {
                        content: String::new(),
                        tokens: Vec::new(),
                        tokens_predicted: result.completion_tokens,
                        tokens_evaluated: result.prompt_tokens,
                        cached_tokens: result.cached_tokens,
                        stop_kind: &result.stop_kind,
                        prompt_ms: result.prompt_eval_ms as f64,
                        predicted_ms: result.generation_only_ms as f64,
                    },
                );
                let _ = finish_events.json(&final_frame);
            }
            Err(err) => {
                // A failed generation still terminates the stream with a frame
                // the client can act on rather than a silent close.
                let _ = finish_events.json(&serde_json::json!({
                    "error": {
                        "code": 500,
                        "message": err.to_string(),
                        "type": "server_error",
                    }
                }));
            }
        }
    });

    sse_response(stream, keepalive)
}

/// Reject the native fields mlxcel has no equivalent for.
///
/// The epic's policy is that a flag or field whose value has observable
/// semantics must not be silently ignored. Each field below is accepted at its
/// inert value and refused otherwise, with a message naming the field and what
/// to use instead, so a `llama-server` request that depends on one fails
/// loudly here instead of returning a plausible-looking wrong answer.
fn reject_unsupported_native_fields(request: &NativeCompletionRequest) -> Option<Response> {
    let refuse = |message: String| -> Option<Response> {
        Some(ErrorResponse::new(message, "invalid_request_error").into_response())
    };

    if request.n_cmpl.is_some_and(|n| n != 1) {
        return refuse(
            "n_cmpl (alias n) above 1 is not supported on /completion; mlxcel serves one \
             completion per request. Send one request per completion"
                .to_string(),
        );
    }
    if request.n_indent.is_some_and(|n| n != 0) {
        return refuse(
            "n_indent is not supported on /completion; mlxcel has no minimum-indentation \
             stop rule for fill-in-the-middle completions"
                .to_string(),
        );
    }
    if request.t_max_predict_ms.is_some_and(|t| t > 0) {
        return refuse(
            "t_max_predict_ms is not supported per request; bound a stalled decode with the \
             server-wide --decode-timeout / MLXCEL_DECODE_TIMEOUT instead"
                .to_string(),
        );
    }
    if request.return_progress.unwrap_or(false) {
        return refuse(
            "return_progress is not supported on /completion; the mlxcel scheduler emits no \
             prompt-processing progress events on this path"
                .to_string(),
        );
    }
    if request.verbose.unwrap_or(false) {
        return refuse(
            "verbose is not supported on /completion; mlxcel has no __verbose debug block. \
             Use GET /slots and GET /metrics for per-request observability"
                .to_string(),
        );
    }
    if request.return_tokens.unwrap_or(false) {
        return refuse(
            "return_tokens is not supported on /completion; mlxcel's streaming decoder emits \
             detokenized text and does not surface the raw token ids on this path. Use POST \
             /tokenize on the returned content"
                .to_string(),
        );
    }
    None
}

fn build_native_options(
    request: &NativeCompletionRequest,
    state: &AppState,
) -> ServerGenerateOptions {
    build_native_generate_options(&state.config, request)
}

fn build_native_generate_options(
    config: &ServerConfig,
    request: &NativeCompletionRequest,
) -> ServerGenerateOptions {
    build_server_generate_options(
        config,
        RequestOptionOverrides {
            max_tokens: request.n_predict,
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: request.min_p,
            repetition_penalty: request.repeat_penalty,
            seed: request.seed,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            dry_multiplier: request.dry_multiplier,
            dry_base: request.dry_base,
            dry_allowed_length: request.dry_allowed_length,
            dry_penalty_last_n: request.dry_penalty_last_n,
            dry_sequence_breakers: request.dry_sequence_breakers.clone(),
            // The native `/completion` endpoint has no per-request XTC
            // fields (llama-server's `/completion` schema does not define
            // them), so XTC always resolves to the disabled baseline here.
            xtc_probability: None,
            xtc_threshold: None,
            stop_sequences: request.stop.clone(),
            priority: RequestPriority::default(),
            // the caller fills this from the validated request
            // body + server default after `build_native_options` returns.
            reasoning_budget: ReasoningBudgetOverride::default(),
            // `/completion` takes a raw prompt; the caller is
            // responsible for priming `<think>` in the prompt if they want
            // in-block counting to start at the first decoded token.
            thinking_enter_block_on_start: false,
            // The native `/completion` endpoint has no per-request
            // loop-detection field, so the policy is resolved engine-side by
            // `resolve_loop_detection`.
            loop_detection_request: None,
            // The endpoint takes a raw prompt with no `tools`, and while it
            // does accept `response_format`, the guard at the top of this
            // module rejects anything other than `{"type": "text"}` with a 400,
            // so no grammar constraint can ever be active here. Neither half of
            // the amplifier signal is reachable. Since issue #967 the Gemma 4
            // family default-on is gated on that signal and therefore no longer
            // applies here; a global `MLXCEL_LOOP_DETECTION` override still
            // does, and remains the way to enable detection on this endpoint.
            request_carries_loop_amplifier: false,
        },
    )
}
