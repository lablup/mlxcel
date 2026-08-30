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

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::media::MediaRequestMetadata;
use crate::server::model_provider::PrefillStats;
use crate::server::model_provider::StopKind;
use crate::server::request_options::{
    RequestOptionOverrides, build_server_generate_options_with_live,
    resolve_server_max_tokens_with_live,
};
use crate::server::streaming::{sse_channel_resumable, sse_response};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::{
    ErrorResponse, NativeCompletionChunk, NativeCompletionRequest, NativeCompletionResponse,
    NativeTimings, StopType, select_response_fields,
};
use crate::server::{AppState, LiveSettings, ServerConfig, ServerGenerateOptions};

fn generation_error_to_response(err: anyhow::Error) -> ErrorResponse {
    super::generation_error_to_response(err)
}

/// POST /completion
pub async fn native_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<NativeCompletionRequest>,
) -> Response {
    let live = state.live();
    serve_native_completion(state, live, &headers, request).await
}

/// The native completion path, entered with an already-parsed request.
///
/// `POST /infill` reaches generation through here after it has replaced the
/// request's `prompt` with the FIM-formatted one, exactly as b10621's own
/// infill handler falls through to its shared completion implementation. Every
/// validation, option-resolution and response-shaping step below therefore
/// applies to both routes rather than being duplicated per route (#1442).
pub(crate) async fn serve_native_completion(
    state: AppState,
    live: Arc<LiveSettings>,
    headers: &HeaderMap,
    request: NativeCompletionRequest,
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

    // b10621's `n_predict` domain is [-1, INT32_MAX] with -1 meaning "as many
    // as the context allows", so the sign is resolved here rather than by
    // serde, which would answer 422 to a value upstream serves (#1441).
    let requested_n_predict = match request.resolve_n_predict() {
        Ok(resolved) => resolved,
        Err(err) => {
            return ErrorResponse::new(err, "invalid_request_error").into_response();
        }
    };

    // `stream_options` is tolerated at any shape but a present non-boolean
    // `include_usage` is refused, matching the pinned binary. The resolved
    // value is inert on this route for the same reason it is inert upstream:
    // the native final frame always carries the counts and the timing block.
    if let Err(err) = request.validate_stream_options() {
        return ErrorResponse::new(err, "invalid_request_error").into_response();
    }

    // validate thinking_budget_tokens early (semantics match
    // /v1/chat/completions but the cap is checked against n_predict).
    let effective_n_predict =
        resolve_server_max_tokens_with_live(&state.config, &live, requested_n_predict);
    let raw_budget = pick_budget_alias(
        request.thinking_budget_tokens,
        request.thinking_token_budget,
        request.thinking_budget,
    );
    let budget_override =
        match resolve_request_budget(raw_budget, live.reasoning_budget, effective_n_predict) {
            Ok(effective) => ReasoningBudgetOverride::Explicit(effective),
            Err(err) => {
                return ErrorResponse::new(err.to_string(), "invalid_request_error")
                    .into_response();
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
    if let Some(response) = reject_non_inert_lora_field(&state, &request) {
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

    let priority = parse_priority_header(headers);
    if request.stream.unwrap_or(false) {
        // b10621 resumable streams (#1444): `X-Conversation-Id` attaches a
        // replayable session on the native route too, scoped to the
        // presenting API key.
        let resumable = super::chat::ResumableStreamContext {
            conversation_id: super::stream::conversation_id_from_headers(headers),
            owner: super::stream::request_stream_owner(&state, headers),
        };
        stream_native_completion(
            state,
            live,
            request,
            requested_n_predict,
            priority,
            budget_override,
            ping_interval,
            resumable,
        )
        .await
    } else {
        non_stream_native_completion(
            state,
            live,
            request,
            requested_n_predict,
            priority,
            budget_override,
        )
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
    live: std::sync::Arc<LiveSettings>,
    request: NativeCompletionRequest,
    requested_n_predict: Option<usize>,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
) -> Result<Json<serde_json::Value>, ErrorResponse> {
    // Queue-depth admission control: reject when prefill queue is full
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }

    let mut options = build_native_options(&request, requested_n_predict, &state, &live);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    // The response echoes the settings that were actually used, so keep a copy
    // before the options are moved into the generator.
    let options_snapshot = options.clone();

    // Slot accounting (#1440): ties this request to a numbered slot for
    // GET /slots and the response's `id_slot` field.
    let slot = state.slots.begin(
        &request.prompt,
        super::slots::slot_params_json(&options, false),
        Some(options.max_tokens as i64),
    );

    let result = state
        .model_provider
        .generate_with_live(request.prompt.clone(), options, &live)
        .map_err(generation_error_to_response)?;
    slot.finish(
        result.prompt_tokens,
        result.cached_tokens,
        result.completion_tokens,
        &result.text,
    );

    let prompt_ms = result.prompt_eval_ms as f64;
    let gen_ms = result.generation_only_ms as f64;

    state.metrics.record_request(
        result.prompt_tokens,
        result.completion_tokens,
        result.generation_time_ms,
    );

    let response = build_native_response(
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
            id_slot: slot.id_slot(),
        },
    );
    Ok(Json(project_native_response(&request, &response)))
}

/// Serialize the native object and apply the request's `response_fields`
/// projection (#1441).
///
/// The projection is the last step on both the non-streaming body and the
/// final streaming frame, exactly as upstream applies it to its final result
/// only: the per-token frames upstream emits are unprojected, and so are
/// mlxcel's.
fn project_native_response(
    request: &NativeCompletionRequest,
    response: &NativeCompletionResponse,
) -> serde_json::Value {
    let body = serde_json::to_value(response).unwrap_or_else(|_| serde_json::json!({}));
    select_response_fields(body, &request.response_field_paths())
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
    /// The slot the request ran on, from the slot registry (#1440); `-1` when
    /// every slot was busy for the whole request, the sentinel upstream uses
    /// on frames that carry no slot.
    id_slot: i64,
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
        id_slot: outcome.id_slot,
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
        "typical_p": sampling.typical_p,
        "top_n_sigma": sampling.top_n_sigma,
        "repeat_last_n": sampling.penalty_last_n,
        "ignore_eos": options.ignore_eos,
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

/// The prefill snapshot as the streaming arm carries it (#1441).
///
/// `Default` is "not observed yet", which reports zeros exactly as the frames
/// did before the snapshot existed, so a backend that never emits one degrades
/// to the old shape rather than to a wrong one. Once observed, `predicted_ms`
/// is measured from the first token, which is what upstream's `predicted_ms`
/// counts: its first streaming frame reports `predicted_n: 1` against
/// `predicted_ms: 0.001`, not against the prefill duration.
#[derive(Debug, Clone, Copy, Default)]
struct StreamPrefill {
    stats: PrefillStats,
    first_token_at: Option<std::time::Instant>,
}

impl StreamPrefill {
    fn observed(stats: PrefillStats) -> Self {
        Self {
            stats,
            first_token_at: Some(std::time::Instant::now()),
        }
    }

    fn predicted_ms(&self) -> f64 {
        self.first_token_at
            .map(|t| t.elapsed().as_millis() as f64)
            .unwrap_or(0.0)
    }
}

async fn stream_native_completion(
    state: AppState,
    live: std::sync::Arc<LiveSettings>,
    request: NativeCompletionRequest,
    requested_n_predict: Option<usize>,
    priority: RequestPriority,
    budget_override: ReasoningBudgetOverride,
    ping_interval: Option<std::time::Duration>,
    resumable: super::chat::ResumableStreamContext,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let mut options = build_native_options(&request, requested_n_predict, &state, &live);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    // b10621 `reasoning_control` (#1444): arming creates the runtime
    // force-end flag, as upstream's on-demand budget sampler does. The
    // native response never exposes the internal completion id, so no
    // control request can address it; that matches b10621, whose native
    // route also never reveals `oaicompat_cmpl_id`.
    options.reasoning_control = (request.reasoning_control == Some(true))
        .then(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
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
    // b10621 resumable stream (#1444): tee the SSE payloads into the
    // conversation's session when `X-Conversation-Id` was sent.
    let session = resumable.conversation_id.as_deref().map(|cid| {
        state
            .stream_sessions
            .create_or_replace(cid, resumable.owner.clone())
    });
    let (events, stream, cancelled, keepalive) = sse_channel_resumable(100, ping_interval, session);
    let finish_events = events.clone();
    let timings_per_token = request.timings_per_token.unwrap_or(false);

    tokio::task::spawn_blocking(move || {
        // Slot accounting (#1440): ties this request to a numbered slot for
        // GET /slots and the frames' `id_slot` field.
        let slot = state.slots.begin(
            &prompt,
            super::slots::slot_params_json(&options, true),
            Some(options.max_tokens as i64),
        );
        let token_events = finish_events.clone();
        let emitted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let emitted_for_tokens = emitted.clone();
        // The prefill snapshot lands once, before the first token, and every
        // frame after it reports the real `prompt_n` / `prompt_ms` / `cache_n`
        // and `tokens_evaluated` the way b10621 does. Before #1441 those were
        // zeroed until the final frame (#1441).
        let prefill = std::sync::Arc::new(std::sync::Mutex::new(StreamPrefill::default()));
        let prefill_for_tokens = prefill.clone();
        let prefill_sink = prefill.clone();

        let result = state
            .model_provider
            .generate_streaming_native_reserved_live(
                prompt,
                options,
                MediaRequestMetadata::default(),
                queue_reservation,
                cancelled,
                &live,
                |token, _lp| {
                    slot.on_token(&token);
                    let n =
                        emitted_for_tokens.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let snapshot = prefill_for_tokens
                        .lock()
                        .map(|guard| *guard)
                        .unwrap_or_default();
                    // Upstream's per-token frame is small: the full metadata
                    // block belongs to the final frame alone. `timings` is
                    // attached here only under `timings_per_token`.
                    let chunk = NativeCompletionChunk {
                        index: 0,
                        content: token.to_string(),
                        tokens: Vec::new(),
                        stop: false,
                        id_slot: slot.id_slot(),
                        tokens_predicted: n,
                        tokens_evaluated: snapshot.stats.prompt_tokens,
                        timings: timings_per_token.then(|| {
                            NativeTimings::new(
                                snapshot.stats.cached_tokens,
                                snapshot.stats.prompt_tokens,
                                snapshot.stats.prompt_ms as f64,
                                n,
                                snapshot.predicted_ms(),
                            )
                        }),
                        prompt_progress: None,
                    };
                    let _ = token_events.json(&chunk);
                },
                |stats| {
                    slot.on_prefill(stats.prompt_tokens, stats.cached_tokens);
                    if let Ok(mut guard) = prefill_sink.lock() {
                        *guard = StreamPrefill::observed(stats);
                    }
                },
            );

        // The final frame is the whole non-streaming object with `stop: true`,
        // built through the same function so the two shapes cannot drift.
        // There is no `[DONE]` sentinel: b10621's native stream ends with this
        // frame, and emitting one would be an extra event no llama-server
        // client expects.
        match result {
            Ok(result) => {
                slot.finish(
                    result.prompt_tokens,
                    result.cached_tokens,
                    result.completion_tokens,
                    &result.text,
                );
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
                        id_slot: slot.id_slot(),
                    },
                );
                let _ = finish_events.json(&project_native_response(&request, &final_frame));
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

/// Whether a native `samplers` value spells exactly b10621's default chain
/// order, in either of the two upstream shapes: an array of stage names or a
/// single string of stage characters.
fn native_samplers_is_the_default_order(value: &serde_json::Value) -> bool {
    const DEFAULT_NAMES: [&str; 9] = [
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
    ];
    match value {
        serde_json::Value::Array(items) => {
            items.len() == DEFAULT_NAMES.len()
                && items
                    .iter()
                    .zip(DEFAULT_NAMES)
                    .all(|(item, name)| item.as_str() == Some(name))
        }
        serde_json::Value::String(chars) => chars == "edskypmxt",
        _ => false,
    }
}

/// Reject the native fields mlxcel has no equivalent for.
///
/// The epic's policy is that a flag or field whose value has observable
/// semantics must not be silently ignored. Each field below is accepted at its
/// inert value and refused otherwise, with a message naming the field and what
/// to use instead, so a `llama-server` request that depends on one fails
/// loudly here instead of returning a plausible-looking wrong answer.
/// Refuse a per-request `lora` value that asks for a different adapter
/// configuration than the one fused at load (#1439).
///
/// b10621 swaps adapter scales per request; mlxcel's adapters are fused into
/// the weights, so the only honest answers are accepting a value that
/// resolves to the configuration in force (inert, upstream's own semantics
/// for that value) and refusing anything else with a diagnostic, instead of
/// serving the request on weights the client did not ask for. The resolution
/// rule is upstream's: listed ids set their scale, unlisted adapters drop to
/// 0.0, unknown ids are ignored.
fn reject_non_inert_lora_field(
    state: &AppState,
    request: &NativeCompletionRequest,
) -> Option<Response> {
    let value = request.lora.as_ref()?;
    let Some(entries) = value.as_array() else {
        return Some(
            ErrorResponse::new(
                "lora must be an array of {id, scale} objects",
                "invalid_request_error",
            )
            .into_response(),
        );
    };
    let current: Vec<f32> = state
        .config
        .lora_adapters
        .iter()
        .map(|spec| spec.reported_scale())
        .collect();
    let requested = super::lora_adapters::requested_scales(entries, current.len());
    if requested == current {
        return None;
    }
    Some(
        ErrorResponse::new(
            "per-request LoRA adapter selection is not supported: mlxcel fuses adapters into the model weights at load time, so this request's `lora` value cannot be served. Send the server's current adapter configuration (GET /lora-adapters), or restart with --lora / --lora-scaled for a different one",
            "invalid_request_error",
        )
        .into_response(),
    )
}

fn reject_unsupported_native_fields(request: &NativeCompletionRequest) -> Option<Response> {
    let refuse = |message: String| -> Option<Response> {
        Some(ErrorResponse::new(message, "invalid_request_error").into_response())
    };

    // Sampler chain order (#1436): mlxcel's chain is fixed to b10621's
    // default order, so the field is accepted only when it spells exactly
    // that order (array-of-names or character form), which is inert; any
    // other order is refused rather than silently sampled in a different
    // order than the client asked for.
    if let Some(samplers) = request.samplers.as_ref()
        && !native_samplers_is_the_default_order(samplers)
    {
        return refuse(format!(
            "samplers {samplers} is not supported: mlxcel's sampler chain order is fixed to the b10621 default (penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature, character form edskypmxt); send that order or omit the field"
        ));
    }
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
    requested_n_predict: Option<usize>,
    state: &AppState,
    live: &LiveSettings,
) -> ServerGenerateOptions {
    build_native_generate_options_with_live(&state.config, live, request, requested_n_predict)
}

#[allow(dead_code)]
fn build_native_generate_options(
    config: &ServerConfig,
    request: &NativeCompletionRequest,
    requested_n_predict: Option<usize>,
) -> ServerGenerateOptions {
    build_native_generate_options_with_live(
        config,
        &config.live_settings(),
        request,
        requested_n_predict,
    )
}

fn build_native_generate_options_with_live(
    config: &ServerConfig,
    live: &LiveSettings,
    request: &NativeCompletionRequest,
    requested_n_predict: Option<usize>,
) -> ServerGenerateOptions {
    build_server_generate_options_with_live(
        config,
        live,
        RequestOptionOverrides {
            // Already through `resolve_n_predict`, so an upstream `-1` arrives
            // here as an unbounded budget the context clamp reduces.
            max_tokens: requested_n_predict,
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
            // b10621's /completion schema declares both XTC fields with
            // SOFT 0..=1 limits: an out-of-range value is CLAMPED into the
            // domain, not rejected (its field_num::eval only throws for hard
            // limits). A threshold above 0.5 is in range and inert, matching
            // upstream (#1436).
            xtc_probability: request.xtc_probability.map(|v| v.clamp(0.0, 1.0)),
            xtc_threshold: request.xtc_threshold.map(|v| v.clamp(0.0, 1.0)),
            // b10621 treats `top_n_sigma <= 0.0` (its default is `-1.0`) as
            // disabled, so non-positive and non-finite values map to the
            // explicit disabled form 0.0, which still overrides a
            // server-wide --top-nsigma default (#1436).
            top_n_sigma: request
                .top_n_sigma
                .map(|v| if v.is_finite() && v > 0.0 { v } else { 0.0 }),
            // b10621's repeat_last_n (#1436): schema floor is 0, so the
            // usize deserialization already rejects negatives; the value
            // maps straight onto the shared penalty window.
            penalty_last_n: request
                .repeat_last_n
                .map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
            // b10621 declares `typical_p` with no schema limits and treats
            // any value at or above 1.0 as disabled, so a present value
            // outside the enabled range (0.0, 1.0) maps to the explicit
            // disabled form 1.0. It still overrides a server-wide --typical
            // default, exactly like an upstream request value replaces the
            // server default; only an ABSENT field lets the default apply.
            typical_p: request.typical_p.map(|v| {
                if v.is_finite() && v > 0.0 && v < 1.0 {
                    v
                } else {
                    1.0
                }
            }),
            // b10621 ignore_eos (#1436): a bool field, absent falls back to
            // the server-wide --ignore-eos default.
            ignore_eos: request.ignore_eos,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerConfig;

    fn native_request(json: &str) -> NativeCompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn native_completion_maps_typical_p() {
        let request = native_request(r#"{"prompt":"hi","typical_p":0.5}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert_eq!(options.sampling.typical_p, 0.5);
    }

    #[test]
    fn native_completion_out_of_domain_typical_p_is_the_explicit_disable() {
        // b10621 declares the field with no schema limits and treats values
        // at or above 1.0 as disabled. A present out-of-domain value must
        // resolve to the explicit disabled form (1.0) and override the
        // server-wide default, not fall back to it.
        let config = ServerConfig {
            default_typical_p: 0.4,
            ..Default::default()
        };
        for body in [
            r#"{"prompt":"hi","typical_p":1.0}"#,
            r#"{"prompt":"hi","typical_p":0.0}"#,
            r#"{"prompt":"hi","typical_p":-1.0}"#,
            r#"{"prompt":"hi","typical_p":2.5}"#,
        ] {
            let request = native_request(body);
            let options = build_native_generate_options(&config, &request, None);
            assert_eq!(options.sampling.typical_p, 1.0, "body: {body}");
        }
    }

    #[test]
    fn native_completion_absent_typical_p_takes_the_server_default() {
        let config = ServerConfig {
            default_typical_p: 0.4,
            ..Default::default()
        };
        let request = native_request(r#"{"prompt":"hi"}"#);
        let options = build_native_generate_options(&config, &request, None);
        assert_eq!(options.sampling.typical_p, 0.4);
    }

    #[test]
    fn native_completion_maps_top_n_sigma_with_the_b10621_sentinel() {
        let config = ServerConfig {
            default_top_n_sigma: 1.5,
            ..Default::default()
        };
        let request = native_request(r#"{"prompt":"hi","top_n_sigma":1.0}"#);
        let options = build_native_generate_options(&config, &request, None);
        assert_eq!(options.sampling.top_n_sigma, 1.0);
        // b10621's -1.0 disabled sentinel maps to the explicit 0.0 disabled
        // form and still overrides the server-wide default.
        let request = native_request(r#"{"prompt":"hi","top_n_sigma":-1.0}"#);
        let options = build_native_generate_options(&config, &request, None);
        assert_eq!(options.sampling.top_n_sigma, 0.0);
        // Absent falls back to the (sanitized) server default.
        let request = native_request(r#"{"prompt":"hi"}"#);
        let options = build_native_generate_options(&config, &request, None);
        assert_eq!(options.sampling.top_n_sigma, 1.5);
    }

    #[test]
    fn native_completion_maps_xtc_fields_with_the_b10621_soft_clamp() {
        let request =
            native_request(r#"{"prompt":"hi","xtc_probability":0.4,"xtc_threshold":0.2}"#);
        assert!(reject_unsupported_native_fields(&request).is_none());
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert_eq!(options.sampling.xtc_probability, 0.4);
        assert_eq!(options.sampling.xtc_threshold, 0.2);
        // A threshold above 0.5 is in range (and inert), matching upstream.
        let inert = native_request(r#"{"prompt":"hi","xtc_threshold":0.9}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &inert, None);
        assert_eq!(options.sampling.xtc_threshold, 0.9);
        // b10621 declares SOFT 0..=1 limits: out-of-range values CLAMP into
        // the domain (a 200, not a 400), so the resolved values sit on the
        // boundary.
        for (body, expect_p, expect_t) in [
            (r#"{"prompt":"hi","xtc_probability":1.5}"#, 1.0_f32, 0.1_f32),
            (r#"{"prompt":"hi","xtc_probability":-0.1}"#, 0.0, 0.1),
            (r#"{"prompt":"hi","xtc_threshold":1.5}"#, 0.0, 1.0),
            (r#"{"prompt":"hi","xtc_threshold":-0.1}"#, 0.0, 0.0),
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "body {body} must be accepted (clamped, matching upstream's soft limits)"
            );
            let options = build_native_generate_options(&ServerConfig::default(), &request, None);
            assert_eq!(options.sampling.xtc_probability, expect_p, "body {body}");
            assert_eq!(options.sampling.xtc_threshold, expect_t, "body {body}");
        }
    }

    #[test]
    fn native_completion_maps_ignore_eos_and_repeat_last_n() {
        let request = native_request(r#"{"prompt":"hi","ignore_eos":true,"repeat_last_n":16}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert!(options.ignore_eos);
        assert_eq!(options.sampling.penalty_last_n, 16);
        // repeat_last_n: 0 disables the penalty stage (b10621 sentinel).
        let request = native_request(r#"{"prompt":"hi","repeat_last_n":0}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert_eq!(options.sampling.penalty_last_n, 0);
        // Absent falls back to the server-wide --repeat-last-n default.
        let request = native_request(r#"{"prompt":"hi"}"#);
        let config = ServerConfig::default();
        let options = build_native_generate_options(&config, &request, None);
        assert_eq!(
            options.sampling.penalty_last_n,
            config.default_repetition_context_size as i32
        );
        assert!(!options.ignore_eos);
    }

    #[test]
    fn native_samplers_accepts_only_the_default_order() {
        let default_array = r#"{"prompt":"hi","samplers":["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"]}"#;
        let default_chars = r#"{"prompt":"hi","samplers":"edskypmxt"}"#;
        for body in [default_array, default_chars] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "the default order is the inert configuration: {body}"
            );
        }
        for body in [
            r#"{"prompt":"hi","samplers":["top_k","temperature"]}"#,
            r#"{"prompt":"hi","samplers":"kt"}"#,
            r#"{"prompt":"hi","samplers":42}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_some(),
                "a non-default order must be refused: {body}"
            );
        }
    }

    #[test]
    fn native_speculative_dotted_fields_are_accepted_and_inert_like_b10621() {
        // Upstream registers these seven flat dotted keys behind a schema
        // block that is compiled out, so b10621 accepts and ignores them;
        // mlxcel must answer the same request the same way (a 400 would
        // refuse what upstream serves) and resolve exactly the same options
        // as if the fields were absent.
        let body = r#"{"prompt":"hi","speculative.n_max":8,"speculative.n_min":2,"speculative.p_min":0.5,"speculative.type":"ngram-simple","speculative.ngram_min_hits":2,"speculative.ngram_size_m":48,"speculative.ngram_size_n":12}"#;
        let request = native_request(body);
        assert!(reject_unsupported_native_fields(&request).is_none());
        assert!(
            request.speculative_n_max.is_some(),
            "the dotted key must be captured, not dropped"
        );
        let with_fields = build_native_generate_options(&ServerConfig::default(), &request, None);
        let bare = native_request(r#"{"prompt":"hi"}"#);
        let baseline = build_native_generate_options(&ServerConfig::default(), &bare, None);
        assert_eq!(
            with_fields.sampling.temperature, baseline.sampling.temperature,
            "inert fields must not perturb the resolved options"
        );
        assert_eq!(with_fields.max_tokens, baseline.max_tokens);
    }

    #[test]
    fn native_generation_settings_echo_typical_p() {
        let request = native_request(r#"{"prompt":"hi","typical_p":0.5}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        let settings = native_generation_settings(&request, &options);
        assert_eq!(settings["typical_p"], 0.5);
    }
}
