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
    response::{IntoResponse, Response, sse::Sse},
};

use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::request_options::{RequestOptionOverrides, build_server_generate_options};
use crate::server::streaming::sse_channel;
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::{
    ErrorResponse, NativeCompletionRequest, NativeCompletionResponse, TimingInfo,
};
use crate::server::{AppState, ServerConfig, ServerGenerateOptions};

/// POST /completion
pub async fn native_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<NativeCompletionRequest>,
) -> Response {
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
    let effective_n_predict = request.n_predict.unwrap_or(state.config.default_max_tokens);
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

    let priority = parse_priority_header(&headers);
    if request.stream.unwrap_or(false) {
        stream_native_completion(state, request, priority, budget_override).await
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

    let result = state
        .model_provider
        .generate(request.prompt.clone(), options)
        .map_err(|e| ErrorResponse::new(format!("Generation error: {}", e), "server_error"))?;

    let prompt_ms = result.prompt_eval_ms as f64;
    let gen_ms = result.generation_only_ms as f64;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    Ok(Json(NativeCompletionResponse {
        content: result.text,
        stop: result.finish_reason == "stop",
        generation_settings: serde_json::json!({}),
        model: state.display_model_id().to_string(),
        tokens_predicted: result.completion_tokens,
        tokens_evaluated: result.prompt_tokens,
        timings: TimingInfo {
            prompt_n: result.prompt_tokens,
            prompt_ms,
            prompt_per_token_ms: if result.prompt_tokens > 0 {
                prompt_ms / result.prompt_tokens as f64
            } else {
                0.0
            },
            prompt_per_second: if prompt_ms > 0.0 {
                result.prompt_tokens as f64 / (prompt_ms / 1000.0)
            } else {
                0.0
            },
            predicted_n: result.completion_tokens,
            predicted_ms: gen_ms,
            predicted_per_token_ms: if result.completion_tokens > 0 {
                gen_ms / result.completion_tokens as f64
            } else {
                0.0
            },
            predicted_per_second: if gen_ms > 0.0 {
                result.completion_tokens as f64 / (gen_ms / 1000.0)
            } else {
                0.0
            },
        },
    }))
}

async fn stream_native_completion(
    state: AppState,
    request: NativeCompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let mut options = build_native_options(&request, &state);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    let prompt = request.prompt.clone();

    // sse_channel also returns an SseKeepAlive for proxy
    // idle-timeout prevention during long prefill phases.
    let (events, stream, cancelled, keepalive) = sse_channel(100);
    let finish_events = events.clone();

    tokio::task::spawn_blocking(move || {
        let token_events = finish_events.clone();

        let result = state.model_provider.generate_streaming_cancellable(
            prompt,
            options,
            cancelled,
            |token| {
                let chunk = serde_json::json!({
                    "content": token,
                    "stop": false,
                });
                let _ = token_events.json(&chunk);
            },
        );

        // Send final chunk
        let stop = match &result {
            Ok(r) => r.finish_reason == "stop",
            Err(_) => true,
        };
        let final_chunk = serde_json::json!({
            "content": "",
            "stop": true,
            "stop_type": if stop { "stop" } else { "limit" },
        });
        let _ = finish_events.json(&final_chunk);
    });

    Sse::new(stream)
        .keep_alive(keepalive.into_inner())
        .into_response()
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
            // loop-detection field. The Gemma 4 family default-on and the
            // global `MLXCEL_LOOP_DETECTION` override still apply engine-side
            // via `resolve_loop_detection`.
            loop_detection_request: None,
        },
    )
}
