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

//! OpenAI-compatible text completions adapter.
//!
//! Policy for defaults, sampling, and streaming stays in shared server helpers;
//! this module only translates the HTTP request/response shape.

use std::collections::HashMap;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response, sse::Sse},
};

use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};

use crate::server::AppState;
use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::streaming::sse_channel;
use crate::server::structured::build_constraint_from_response_format;
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::response::CompletionLogprobs;
use crate::server::types::{CompletionChunk, CompletionRequest, CompletionResponse, ErrorResponse};
use crate::tokenizer::MlxcelTokenizer;

use super::chat::{
    build_generate_options, decode_token, parse_priority_header, structured_error_to_response,
    validate_xtc_params,
};

/// Build a `CompletionLogprobs` from a list of `TokenLogprobData` (legacy format).
fn build_completion_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp_data: &[TokenLogprobData],
    top_k: usize,
) -> CompletionLogprobs {
    let mut tokens = Vec::with_capacity(lp_data.len());
    let mut token_logprobs = Vec::with_capacity(lp_data.len());
    let mut text_offset = Vec::with_capacity(lp_data.len());
    let mut top_logprobs_list = Vec::with_capacity(lp_data.len());

    let mut char_offset: usize = 0;

    for lp in lp_data {
        let token_text = decode_token(tokenizer, lp.token_id);
        text_offset.push(char_offset);
        char_offset += token_text.len();
        tokens.push(token_text);
        token_logprobs.push(lp.logprob);

        let top_map = if top_k > 0 && !lp.top_alternatives.is_empty() {
            let mut map = HashMap::new();
            for &(alt_id, alt_lp) in lp.top_alternatives.iter().take(top_k) {
                let alt_text = decode_token(tokenizer, alt_id);
                map.insert(alt_text, alt_lp);
            }
            Some(map)
        } else {
            None
        };
        top_logprobs_list.push(top_map);
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        text_offset,
        top_logprobs: top_logprobs_list,
    }
}

/// Build a streaming `CompletionLogprobs` for a single token chunk.
fn build_single_token_completion_logprobs(
    tokenizer: &MlxcelTokenizer,
    lp: &TokenLogprobData,
    top_k: usize,
    char_offset: usize,
) -> CompletionLogprobs {
    let token_text = decode_token(tokenizer, lp.token_id);
    let top_map = if top_k > 0 && !lp.top_alternatives.is_empty() {
        let mut map = HashMap::new();
        for &(alt_id, alt_lp) in lp.top_alternatives.iter().take(top_k) {
            let alt_text = decode_token(tokenizer, alt_id);
            map.insert(alt_text, alt_lp);
        }
        Some(map)
    } else {
        None
    };
    CompletionLogprobs {
        tokens: vec![token_text],
        token_logprobs: vec![lp.logprob],
        text_offset: vec![char_offset],
        top_logprobs: vec![top_map],
    }
}

/// Returns `true` when `prompt` is whitespace-only but non-empty (issue
/// #806): a string that survives deserialization as a `String` but has no
/// non-whitespace character.
///
/// A fully empty prompt (`""`) is deliberately *not* flagged by this
/// predicate: unlike the chat-shaped routes (`/v1/chat/completions`,
/// `/v1/responses`, `/v1/messages`), which always have template scaffolding
/// around user content, `/v1/completions` takes a raw prompt with no
/// scaffolding, so an empty prompt is a legitimate request for unconditional
/// generation from BOS on a base/text-completion model. `"   \n\t"`-shaped
/// prompts have no such legitimate reading, so they are treated the same as
/// no effective input.
fn prompt_is_whitespace_only(prompt: &str) -> bool {
    !prompt.is_empty() && prompt.trim().is_empty()
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Response {
    // Reject whitespace-only-but-nonempty prompts before any model dispatch,
    // alongside the route's other pre-dispatch validations (issue #806).
    // See `prompt_is_whitespace_only` for why an empty prompt is allowed
    // through instead of rejected here.
    if prompt_is_whitespace_only(&request.prompt) {
        return ErrorResponse::new(
            "Request must include at least one non-empty message content or media input.",
            "invalid_request_error",
        )
        .into_response();
    }

    // Validate logprobs range per OpenAI spec (0-5 for legacy completions)
    if let Some(top) = request.logprobs
        && top > 5
    {
        return ErrorResponse::new("logprobs must be between 0 and 5", "invalid_request_error")
            .into_response();
    }

    // Validate XTC (Exclude Top Choices) sampling parameter ranges before any
    // generation work begins.
    if let Err(message) =
        validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability)
    {
        return ErrorResponse::new(message, "invalid_request_error").into_response();
    }

    // validate thinking_budget_tokens early.
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
    // front. Grammar compilation can be slow on adversarial schemas, so
    // we run it on `spawn_blocking` rather than blocking the Tokio
    // runtime thread. Returns `None` when no `response_format` was
    // supplied.
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
        stream_completion(state, request, priority, budget_override, structured).await
    } else {
        non_stream_completion(state, request, priority, budget_override, structured)
            .await
            .into_response()
    }
}

async fn non_stream_completion(
    state: AppState,
    request: CompletionRequest,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    structured: Option<
        std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
    >,
) -> Result<Json<CompletionResponse>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();

    let prompt = request.prompt.clone();
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    // `/v1/completions` takes a raw prompt just like `/completion`;
    // the request body is not routed through the chat template so the prompt
    // is not primed with `<think>\n`. Override the chat-oriented default from
    // `build_generate_options` so the scheduler waits for the model to emit
    // `<think>` itself before counting reasoning tokens.
    options.thinking_enter_block_on_start = false;
    // forward structured-output constraint into the worker.
    options.structured = structured;

    // In the legacy format, `logprobs` is a number (top-k); 0 means return only
    // the selected token's log-prob, None means don't return logprobs at all.
    let top_k = request.logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs.is_some();
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    // Generate (blocking call handled by model provider's worker thread)
    let result = state
        .model_provider
        .generate(prompt, options)
        .map_err(|e| ErrorResponse::new(format!("Generation error: {}", e), "server_error"))?;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    // Build legacy-format logprobs if requested
    let logprobs = result.logprobs.as_deref().and_then(|lp_data| {
        if lp_data.is_empty() || !logprobs_enabled {
            None
        } else {
            Some(build_completion_logprobs(&state.tokenizer, lp_data, top_k))
        }
    });

    Ok(Json(CompletionResponse::new_with_logprobs(
        request_id,
        model_id,
        result.text,
        result.prompt_tokens,
        result.completion_tokens,
        Some(result.finish_reason),
        logprobs,
    )))
}

async fn stream_completion(
    state: AppState,
    request: CompletionRequest,
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

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = state.display_model_id().to_string();
    let prompt = request.prompt.clone();
    let mut options = build_generate_options(&request.params, &state.config);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    // see non_stream_completion — raw-text endpoint, no
    // `<think>\n` priming, so the scheduler must wait for the model to
    // emit `<think>` itself before counting reasoning tokens.
    options.thinking_enter_block_on_start = false;
    // forward structured-output constraint into the worker.
    options.structured = structured;

    // Extract include_usage before request is moved into the closure
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // In the legacy format, `logprobs` is a number (top-k); 0 means return only
    // the selected token's log-prob, None means don't return logprobs at all.
    let top_k = request.logprobs.unwrap_or(0) as usize;
    let logprobs_enabled = request.logprobs.is_some();
    if logprobs_enabled {
        options.logprobs = LogprobsConfig {
            enabled: true,
            top_k,
        };
    }

    // sse_channel also returns an SseKeepAlive that sends periodic
    // SSE comment events to prevent proxy/client idle-timeout disconnects
    // during long prefill phases.
    let (events, stream, cancelled, keepalive) = sse_channel(100);

    // Clone for the spawned task
    let request_id_clone = request_id.clone();
    let model_id_clone = model_id.clone();
    let finish_events = events.clone();
    let tokenizer = state.tokenizer.clone();

    // Spawn a blocking task to handle generation
    tokio::task::spawn_blocking(move || {
        // Use logprobs-aware streaming
        let token_events = finish_events.clone();
        let request_id_inner = request_id_clone.clone();
        let model_id_inner = model_id_clone.clone();
        let mut char_offset: usize = 0;

        let result = state
            .model_provider
            .generate_streaming_with_logprobs_cancellable(
                prompt,
                options,
                Vec::new(),
                Vec::new(),
                cancelled,
                |token, lp_data| {
                    let logprobs = if logprobs_enabled {
                        lp_data.as_ref().map(|lp| {
                            let chunk_lp = build_single_token_completion_logprobs(
                                &tokenizer,
                                lp,
                                top_k,
                                char_offset,
                            );
                            char_offset += token.len();
                            chunk_lp
                        })
                    } else {
                        None
                    };
                    let chunk = CompletionChunk::content_with_logprobs(
                        request_id_inner.clone(),
                        model_id_inner.clone(),
                        token,
                        logprobs,
                    );
                    let _ = token_events.json(&chunk);
                },
            );

        // Send finish chunk
        let finish_reason = match &result {
            Ok(r) => r.finish_reason.clone(),
            Err(_) => "error".to_string(),
        };
        let finish = CompletionChunk::finish(
            request_id_clone.clone(),
            model_id_clone.clone(),
            finish_reason,
        );
        let _ = finish_events.json(&finish);

        // Send usage chunk if requested (stream_options.include_usage)
        if include_usage && let Ok(ref r) = result {
            let usage_chunk = CompletionChunk::usage(
                request_id_clone.clone(),
                model_id_clone.clone(),
                r.prompt_tokens,
                r.completion_tokens,
            );
            let _ = finish_events.json(&usage_chunk);
        }

        finish_events.done();
    });

    Sse::new(stream)
        .keep_alive(keepalive.into_inner())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- XTC (Exclude Top Choices) request validation on /v1/completions --
    //
    // `CompletionRequest` flattens `SamplingParams`, so `xtc_probability` /
    // `xtc_threshold` are reachable off `request.params`. These tests exercise
    // the same field access the `completions` handler performs and confirm the
    // shared `validate_xtc_params` gate rejects out-of-range values.

    #[test]
    fn completions_rejects_out_of_range_xtc_probability() {
        let request: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi","xtc_probability":2.0}"#).unwrap();
        assert_eq!(
            validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability),
            Err("xtc_probability must be between 0.0 and 1.0")
        );
    }

    #[test]
    fn completions_rejects_out_of_range_xtc_threshold() {
        let request: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi","xtc_threshold":-0.1}"#).unwrap();
        assert_eq!(
            validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability),
            Err("xtc_threshold must be between 0.0 and 0.5")
        );
    }

    #[test]
    fn completions_accepts_in_range_xtc() {
        let request: CompletionRequest = serde_json::from_str(
            r#"{"model":"m","prompt":"hi","xtc_probability":0.5,"xtc_threshold":0.1}"#,
        )
        .unwrap();
        assert!(
            validate_xtc_params(request.params.xtc_threshold, request.params.xtc_probability)
                .is_ok()
        );
    }

    // -- no-effective-input decision for /v1/completions (issue #806) --
    //
    // Unlike the chat-shaped routes, `/v1/completions` deliberately allows a
    // fully empty prompt through (unconditional generation from BOS is a
    // legitimate base-model use case) while still rejecting
    // whitespace-only-but-nonempty prompts, which have no such legitimate
    // reading. These tests exercise `prompt_is_whitespace_only`, the same
    // predicate the `completions` handler calls before dispatch.

    #[test]
    fn completions_rejects_whitespace_only_spaces() {
        assert!(prompt_is_whitespace_only("   "));
    }

    #[test]
    fn completions_rejects_whitespace_only_newline() {
        assert!(prompt_is_whitespace_only("\n"));
    }

    #[test]
    fn completions_rejects_whitespace_only_tabs_and_mixed() {
        assert!(prompt_is_whitespace_only("\t \n\t"));
    }

    /// Pinned regression test: an empty prompt (`""`) is an intentional
    /// unconditional-generation request on `/v1/completions` (issue #806)
    /// and must NOT be rejected by the guard. This test exists specifically
    /// so a future change cannot silently regress that decision back to
    /// rejecting empty prompts.
    #[test]
    fn completions_allows_fully_empty_prompt() {
        assert!(!prompt_is_whitespace_only(""));
    }

    #[test]
    fn completions_allows_normal_prompt() {
        assert!(!prompt_is_whitespace_only("Hello, world"));
    }

    #[test]
    fn completions_no_effective_input_error_matches_issue_773_spec() {
        // The handler's whitespace-only-prompt rejection must surface the
        // same HTTP 400 `invalid_request_error` shape and message string as
        // the #773 guard used on the chat-shaped routes.
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

    #[test]
    fn completions_deserializes_empty_and_whitespace_prompts() {
        // Confirms `{"prompt":""}` and a whitespace-only prompt both pass
        // deserialization (required `String`, not `Option`), so the guard
        // is the only line of defense before dispatch.
        let empty: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":""}"#).unwrap();
        assert!(!prompt_is_whitespace_only(&empty.prompt));

        let whitespace: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"  \n\t"}"#).unwrap();
        assert!(prompt_is_whitespace_only(&whitespace.prompt));
    }
}
