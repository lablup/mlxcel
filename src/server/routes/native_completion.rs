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
use crate::server::model_provider::{SpeculativeStats, StopKind};
use crate::server::request_options::{
    RequestOptionOverrides, build_server_generate_options_with_live,
    resolve_server_max_tokens_with_live,
};
use crate::server::streaming::{sse_channel_resumable, sse_response};
use crate::server::thinking_budget::{pick_budget_alias, resolve_request_budget};
use crate::server::types::{
    ErrorResponse, NativeCompletionChunk, NativeCompletionRequest, NativeCompletionResponse,
    NativeTimings, PromptProgress, StopType, select_response_fields,
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

    // b10621 `n_keep` / `n_discard` (#1472): validated against upstream's
    // schema floors before anything runs; the resolved values ride the
    // generation options into the scheduler's context-retention state.
    if let Err(err) = request.validate_retention() {
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

    // b10621 grammar surfaces (#1485). Resolution and compilation both happen
    // before the request is admitted, so a malformed grammar answers 400
    // instead of degrading silently into an unconstrained generation.
    let grammar_spec = match crate::server::grammar::resolve_native_grammar(
        state.tokenizer.as_ref(),
        request.json_schema.as_ref(),
        request.grammar.as_ref(),
        request.grammar_lazy,
        request.grammar_triggers.as_ref(),
        request.preserved_tokens.as_ref(),
        state.config.default_grammar.as_ref(),
    ) {
        Ok(spec) => spec,
        Err(err) => {
            return ErrorResponse::new(err.to_string(), "invalid_request_error").into_response();
        }
    };
    let grammar = match build_native_grammar(&state, grammar_spec).await {
        Ok(grammar) => grammar,
        Err(response) => return response,
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
            grammar,
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
            grammar,
        )
        .await
        .into_response()
    }
}

/// The compiled constraint plus the spec it came from, or `None` when the
/// request asked for no grammar.
type NativeGrammar = Option<(
    std::sync::Arc<crate::server::grammar::GrammarSpec>,
    std::sync::Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>,
)>;

/// Compile a resolved [`crate::server::grammar::GrammarSpec`] off the runtime
/// thread.
///
/// GBNF lowering plus `llguidance` compilation is CPU-bound and, on a large
/// grammar, slow enough to stall unrelated in-flight requests if it ran on a
/// Tokio worker, which is why the schema path already uses `spawn_blocking`.
async fn build_native_grammar(
    state: &AppState,
    spec: Option<crate::server::grammar::GrammarSpec>,
) -> Result<NativeGrammar, Response> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let spec = std::sync::Arc::new(spec);
    let tokenizer = state.tokenizer.clone();
    let build_spec = std::sync::Arc::clone(&spec);
    match tokio::task::spawn_blocking(move || {
        crate::server::structured::build_constraint_from_grammar_spec(
            tokenizer.as_ref(),
            &build_spec,
        )
    })
    .await
    {
        Ok(Ok(constraint)) => Ok(Some((spec, constraint))),
        Ok(Err(err)) => Err(super::chat::structured_error_to_response(err).into_response()),
        Err(join_err) => {
            tracing::error!("grammar compilation task panicked: {join_err}");
            Err(ErrorResponse::new("grammar preparation failed", "server_error").into_response())
        }
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
    grammar: NativeGrammar,
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
    if let Some((spec, constraint)) = grammar {
        options.grammar = Some(spec);
        options.structured = Some(constraint);
    }
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
            // b10621 `return_tokens` (#1477): the raw sampled ids, or an empty
            // array when the field is absent or false.
            tokens: native_returned_tokens(&request, result.generated_token_ids),
            tokens_predicted: result.completion_tokens,
            tokens_evaluated: result.prompt_tokens,
            cached_tokens: result.cached_tokens,
            stop_kind: &result.stop_kind,
            prompt_ms,
            predicted_ms: gen_ms,
            id_slot: slot.id_slot(),
            probs: result.logprobs,
            speculative: result.speculative,
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

/// The effective `n_probs` for a request (#1485): `n_probs` with the
/// `logprobs` alias, positive values only.
fn native_n_probs(request: &NativeCompletionRequest) -> usize {
    request
        .n_probs
        .or(request.logprobs)
        .map(|n| n.max(0) as usize)
        .unwrap_or(0)
}

/// Clamp a probability value for JSON (#1485): serde_json serializes a
/// non-finite float as `null`, so `-inf` (a zero probability in log space)
/// folds to the lowest finite f32, exactly upstream's `logarithm` guard.
fn finite_prob(value: f32) -> f32 {
    if value.is_finite() { value } else { f32::MIN }
}

/// The longest valid-UTF-8 prefix of a token's bytes, upstream's
/// `validate_utf8` truncation: a byte-level BPE token routinely ends inside
/// a multi-byte character, and the `bytes` array is how a client reassembles
/// the text across it.
fn utf8_prefix(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(err) => String::from_utf8_lossy(&bytes[..err.valid_up_to()]).into_owned(),
    }
}

/// One `{id, token, bytes, logprob|prob}` object of the native probability
/// report (#1485), b10621's `completion_token_output::to_json` shape.
fn native_prob_object(
    tokenizer: Option<&crate::tokenizer::MlxcelTokenizer>,
    id: i32,
    value: f32,
    post_sampling: bool,
) -> serde_json::Value {
    let bytes = tokenizer
        .and_then(|t| t.token_piece_bytes(id as u32))
        .unwrap_or_default();
    let key = if post_sampling { "prob" } else { "logprob" };
    serde_json::json!({
        "id": id,
        "token": utf8_prefix(&bytes),
        "bytes": bytes,
        key: finite_prob(value),
    })
}

/// The native `completion_probabilities` array (#1485): one entry per
/// generated token, each with its own probability and its `top_logprobs` /
/// `top_probs` alternatives, b10621's `probs_vector_to_json` shape.
fn native_completion_probabilities(
    tokenizer: Option<&crate::tokenizer::MlxcelTokenizer>,
    entries: &[mlxcel_core::sampling::TokenLogprobData],
    post_sampling: bool,
) -> serde_json::Value {
    let top_key = if post_sampling {
        "top_probs"
    } else {
        "top_logprobs"
    };
    serde_json::Value::Array(
        entries
            .iter()
            .map(|lp| {
                let mut obj = native_prob_object(tokenizer, lp.token_id, lp.logprob, post_sampling);
                let tops: Vec<serde_json::Value> = lp
                    .top_alternatives
                    .iter()
                    .map(|&(id, value)| native_prob_object(tokenizer, id, value, post_sampling))
                    .collect();
                obj[top_key] = serde_json::Value::Array(tops);
                obj
            })
            .collect(),
    )
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
    /// Per-token probability data accumulated over the generation (#1485);
    /// `None` unless the request set `n_probs`.
    probs: Option<Vec<mlxcel_core::sampling::TokenLogprobData>>,
    /// Drafter acceptance counters for the request (#1314); `None` unless a
    /// drafter executed at least one verify round, in which case `timings`
    /// carries none of the `draft_*` keys.
    speculative: Option<SpeculativeStats>,
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
        // A context-bound stop (#1472) is upstream's STOP_TYPE_LIMIT too; it
        // is separated from `Limit` only so `truncated` below can tell the
        // two apart.
        StopKind::Limit | StopKind::ContextExhausted => (StopType::Limit, String::new()),
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
        generation_settings: native_generation_settings(
            &state.config,
            state
                .model_provider
                .prompt_tokenizer()
                .map(std::sync::Arc::as_ref),
            request,
            options,
        ),
        prompt: request.prompt.clone(),
        // Set exactly when the generation stopped at the per-slot context
        // bound with context shifting disabled (#1472), which is when b10621
        // sets it; an over-long prompt never reaches here (it is refused at
        // admission), so a dropped-prefix truncation cannot occur.
        truncated: matches!(outcome.stop_kind, StopKind::ContextExhausted),
        stop_type,
        stopping_word,
        // b10621 reports the SLOT's cache occupancy after the request, not
        // what the prefix cache supplied for it (#1477). Six independent
        // measurements against the pinned binary agree on
        // `tokens_evaluated + tokens_predicted - 1`, including the
        // `n_predict: 0` prompt-only case (which upstream still answers with
        // `tokens_predicted: 1`) and a fully cache-hit request, where the
        // figure is unchanged by the hit. `timings.cache_n` below stays the
        // cache-supplied count, which is a different quantity and is what
        // `prompt_n` is derived from.
        tokens_cached: outcome
            .tokens_evaluated
            .saturating_add(outcome.tokens_predicted)
            .saturating_sub(1),
        timings: NativeTimings::new(
            outcome.cached_tokens,
            outcome.tokens_evaluated,
            outcome.prompt_ms,
            outcome.tokens_predicted,
            outcome.predicted_ms,
        )
        // b10621 appends `draft_n` / `draft_n_accepted` to `timings` only
        // when the request was drafted, and omits them otherwise (#1314).
        .with_speculative(outcome.speculative.as_ref()),
        completion_probabilities: match (native_n_probs(request), outcome.probs.as_ref()) {
            (n, Some(entries)) if n > 0 => Some(native_completion_probabilities(
                state
                    .model_provider
                    .prompt_tokenizer()
                    .map(std::sync::Arc::as_ref),
                entries,
                request.post_sampling_probs.unwrap_or(false),
            )),
            _ => None,
        },
    }
}

/// The resolved generation settings echoed back on the response.
///
/// b10621 echoes its whole `task_params` here (49 keys). mlxcel reports the
/// settings it actually resolved and acts on; a key mlxcel has no analogue for
/// is omitted rather than reported with an invented value, which would be a
/// worse answer than its absence.
fn native_generation_settings(
    config: &ServerConfig,
    tokenizer: Option<&crate::tokenizer::MlxcelTokenizer>,
    request: &NativeCompletionRequest,
    options: &ServerGenerateOptions,
) -> serde_json::Value {
    let sampling = &options.sampling;
    // Built in two halves: `serde_json::json!` hits its recursion limit around
    // forty keys, and b10621's block has forty-nine.
    let mut settings = serde_json::json!({
        // The random sentinel reports as b10621's uint32 LLAMA_DEFAULT_SEED
        // now that the seed domain folds into uint32 space (#1485).
        "seed": sampling.seed.unwrap_or(u64::from(u32::MAX)),
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
        // Reported as the breaker STRINGS since #1485, b10621's own value
        // domain for the key (the native field and the flag both carry
        // strings now).
        "dry_sequence_breakers": options.dry_breaker_strings.clone().unwrap_or_default(),
        "stop": options.stop_sequences.clone().unwrap_or_default(),
        "max_tokens": options.max_tokens,
        "n_predict": options.max_tokens,
        // The resolved context-retention pair (#1472): the request's value,
        // or the server-wide `--keep` / half-window default it fell back to.
        "n_keep": options.retention.n_keep.unwrap_or(0),
        "n_discard": options.retention.n_discard.unwrap_or(0),
        "stream": request.stream.unwrap_or(false),
        // #1485 sampling remainder.
        "mirostat": sampling.mirostat,
        "mirostat_tau": sampling.mirostat_tau,
        "mirostat_eta": sampling.mirostat_eta,
        "dynatemp_range": sampling.dynatemp_range,
        "dynatemp_exponent": sampling.dynatemp_exponent,
        "adaptive_target": sampling.adaptive_target,
        "adaptive_decay": sampling.adaptive_decay,
        "min_keep": sampling.min_keep,
        "n_probs": if options.logprobs.enabled { options.logprobs.top_k } else { 0 },
        "post_sampling_probs": options.post_sampling_probs,
        // #1485 grammar surfaces, reported from the SAME resolution the
        // constraint was compiled from, so the report cannot claim a grammar
        // the sampler is not running. `grammar` is empty for a schema-sourced
        // constraint: b10621 reports the GBNF its schema converter produced,
        // and mlxcel has no such intermediate text.
        "grammar": options
            .grammar
            .as_ref()
            .and_then(|g| g.gbnf.clone())
            .unwrap_or_default(),
        "grammar_lazy": options.grammar.as_ref().is_some_and(|g| g.lazy),
        "grammar_triggers": native_trigger_report(options.grammar.as_deref()),
        "preserved_tokens": options
            .grammar
            .as_ref()
            .map(|g| g.preserved.clone())
            .unwrap_or_default(),
    });
    // #1477 remainder. Each of these is a value mlxcel genuinely resolves and
    // acts on, so the omit-rather-than-invent policy (recorded on GET /props)
    // no longer applies to it.
    let rest = serde_json::json!({
        "samplers": native_samplers_report(options),
        "timings_per_token": request.timings_per_token.unwrap_or(false),
        "logit_bias": native_logit_bias_report(tokenizer, options),
        "lora": native_lora_report(config, options),
        // The native route is a raw-prompt endpoint with no chat template and
        // no chat parsing, which is upstream's state on this route too: it
        // reports the "Content-only" format and an empty generation prompt
        // there whatever the model's template is.
        "chat_format": "Content-only",
        "generation_prompt": "",
        "reasoning_format": native_reasoning_format_report(config),
        // Upstream computes this as `stream && reasoning_format ==
        // deepseek-legacy` and no parser reads the result (settled in #1470
        // against the full b10621 tree), so it is reported, not acted on.
        "reasoning_in_content": request.stream.unwrap_or(false)
            && matches!(
                config.reasoning_format,
                crate::server::ReasoningFormat::DeepSeekLegacy
            ),
    });
    if let (Some(target), Some(extra)) = (settings.as_object_mut(), rest.as_object()) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
    settings
}

/// The sampler chain this request runs, in b10621's `samplers` shape (#1477).
///
/// mlxcel's chain order is fixed to upstream's default and a request naming a
/// different order is refused rather than silently reordered
/// (`field:samplers`, by_design), so the reported list is that order plus the
/// `adaptive_p` stage when the request armed it. Upstream echoes the list it
/// was given verbatim and would therefore place `adaptive_p` wherever the
/// client wrote it; mlxcel reports its canonical position, which is the same
/// normalization the fixed chain already applies.
fn native_samplers_report(options: &ServerGenerateOptions) -> Vec<&'static str> {
    let mut names: Vec<&'static str> =
        crate::server::startup::b10621_default_sampler_names().collect();
    if options.sampling.adaptive_target >= 0.0 {
        names.push("adaptive_p");
    }
    names
}

/// The resolved token biases in b10621's `[{token, bias}]` shape (#1477).
///
/// Upstream tokenizes a string-keyed bias at schema time and reports the
/// resulting ids, so mlxcel resolves its own text-keyed biases through the
/// request-dispatch tokenizer with the same `common_tokenize(vocab, text,
/// false)` settings the scheduler uses, rather than reporting the numeric
/// half alone. Without a tokenizer only the numeric half can be named, which
/// is the honest answer for a backend that pre-tokenizes elsewhere.
fn native_logit_bias_report(
    tokenizer: Option<&crate::tokenizer::MlxcelTokenizer>,
    options: &ServerGenerateOptions,
) -> serde_json::Value {
    let mut out: Vec<serde_json::Value> = options
        .logit_bias
        .iter()
        .map(|&(token, bias)| serde_json::json!({"bias": bias, "token": token}))
        .collect();
    if !options.logit_bias_texts.is_empty()
        && let Some(tokenizer) = tokenizer
    {
        for (text, bias) in &options.logit_bias_texts {
            if let Ok(ids) = tokenizer.encode_with_special(text, false, false) {
                for id in ids {
                    out.push(serde_json::json!({"bias": bias, "token": id as i32}));
                }
            }
        }
    }
    serde_json::Value::Array(out)
}

/// The adapter scales in force for this request, in b10621's `[{id, scale}]`
/// shape (#1477).
///
/// `id` is the adapter's index in the server's `--lora` list, which is the
/// same identifier `GET /lora-adapters` and the request's own `lora` field
/// use (#1439). The per-request snapshot wins when the runtime path resolved
/// one; otherwise the server-wide configuration is reported.
fn native_lora_report(config: &ServerConfig, options: &ServerGenerateOptions) -> serde_json::Value {
    let scales: Vec<f32> = match options.lora_scales.as_deref() {
        Some(snapshot) => snapshot.clone(),
        None => config
            .lora_adapters
            .iter()
            .map(|spec| spec.reported_scale())
            .collect(),
    };
    serde_json::Value::Array(
        scales
            .iter()
            .enumerate()
            .map(|(id, scale)| serde_json::json!({"id": id, "scale": scale}))
            .collect(),
    )
}

/// `--reasoning-format` as b10621 reports it (#1477).
///
/// Upstream resolves `auto` before it echoes the value: the pinned binary
/// started with the default reports `deepseek`. mlxcel's `auto` behaves as
/// `deepseek` for every family it supports (the `--reasoning-format` entry
/// records why), so reporting the resolved name rather than the literal flag
/// is both what upstream does and what mlxcel acts on.
fn native_reasoning_format_report(config: &ServerConfig) -> &'static str {
    match config.reasoning_format {
        crate::server::ReasoningFormat::Auto => "deepseek",
        other => other.as_str(),
    }
}

/// `grammar_triggers` in b10621's own reporting shape: `{type, value}` plus a
/// `token` key on the token form only.
fn native_trigger_report(spec: Option<&crate::server::grammar::GrammarSpec>) -> serde_json::Value {
    use crate::server::grammar::GrammarTrigger;
    let Some(spec) = spec else {
        return serde_json::Value::Array(Vec::new());
    };
    serde_json::Value::Array(
        spec.triggers
            .iter()
            .map(|trigger| match trigger {
                GrammarTrigger::Token(id) => {
                    serde_json::json!({"type": 0, "value": "", "token": id})
                }
                GrammarTrigger::Word(value) => serde_json::json!({"type": 1, "value": value}),
                GrammarTrigger::Pattern(value) => serde_json::json!({"type": 2, "value": value}),
                GrammarTrigger::PatternFull(value) => {
                    serde_json::json!({"type": 3, "value": value})
                }
            })
            .collect(),
    )
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
    grammar: NativeGrammar,
) -> Response {
    // Queue-depth admission control: return 503 before opening SSE stream
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let mut options = build_native_options(&request, requested_n_predict, &state, &live);
    options.priority = priority;
    options.reasoning_budget = budget_override;
    if let Some((spec, constraint)) = grammar {
        options.grammar = Some(spec);
        options.structured = Some(constraint);
    }
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
    // #1485 n_probs streaming report inputs, captured before the request
    // moves into the worker closure.
    let stream_n_probs = native_n_probs(&request);
    let stream_post_sampling = request.post_sampling_probs.unwrap_or(false);
    let stream_tokenizer = state.model_provider.prompt_tokenizer().cloned();
    // b10621 emits `prompt_progress` frames only for a streaming request that
    // asked for them (#1477); its own gate is `params.stream &&
    // params.return_progress`, and this arm is the streaming one.
    let want_progress = request.return_progress.unwrap_or(false);
    let progress_events = events.clone();

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
                |token, meta, lp| {
                    slot.on_token(&token);
                    let fallback =
                        emitted_for_tokens.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    // b10621 reports the slot's generated-token count, not the
                    // number of frames it has sent: a piece held back by the
                    // stop matcher or by an incomplete UTF-8 sequence still
                    // advances it (#1477). The frame counter is the fallback
                    // for a backend that reports no count.
                    let n = meta.decoded.unwrap_or(fallback);
                    let snapshot = prefill_for_tokens
                        .lock()
                        .map(|guard| *guard)
                        .unwrap_or_default();
                    // Upstream's per-token frame is small: the full metadata
                    // block belongs to the final frame alone. `timings` is
                    // attached here only under `timings_per_token`, and
                    // without the `draft_*` keys (#1314): those counters are
                    // the run's totals and are only known once the round loop
                    // has finished, so a mid-stream frame carrying them would
                    // be reporting a total that is not one yet.
                    let chunk = NativeCompletionChunk {
                        index: 0,
                        content: token.to_string(),
                        // Upstream fills the per-frame array with the single id
                        // that produced the frame, unconditionally: it is NOT
                        // gated on `return_tokens`, which governs the
                        // non-streaming array only (measured on the pinned
                        // binary, #1477).
                        tokens: meta.token_id.map(|id| vec![id]).unwrap_or_default(),
                        stop: false,
                        // Upstream's `send_partial_response` never stamps a slot
                        // id, so every partial frame carries the -1 sentinel and
                        // only the final frame names the slot (#1477).
                        id_slot: -1,
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
                        // b10621 streams the probability report per token
                        // (#1485): a one-entry array in the final object's
                        // shape, present only under `n_probs`.
                        completion_probabilities: match (stream_n_probs, lp.as_ref()) {
                            (n, Some(lp)) if n > 0 => Some(native_completion_probabilities(
                                stream_tokenizer.as_deref(),
                                std::slice::from_ref(lp),
                                stream_post_sampling,
                            )),
                            _ => None,
                        },
                    };
                    let _ = token_events.json(&chunk);
                },
                |stats| {
                    // b10621 `return_progress` (#1477): every observation
                    // becomes a `prompt_progress` frame with empty content,
                    // ahead of the first content frame. Without the field the
                    // observations are internal bookkeeping only, exactly as
                    // upstream gates them on `stream && return_progress`.
                    if want_progress {
                        let frame = NativeCompletionChunk {
                            index: 0,
                            content: String::new(),
                            // Upstream builds the frame from a default-
                            // constructed token, so its `tokens` array is the
                            // one-element `[0]` rather than empty; measured on
                            // the pinned binary.
                            tokens: vec![0],
                            stop: false,
                            id_slot: -1,
                            tokens_predicted: 0,
                            tokens_evaluated: stats.prompt_tokens,
                            timings: None,
                            prompt_progress: Some(PromptProgress {
                                total: stats.prompt_tokens,
                                cache: stats.cached_tokens,
                                processed: stats.processed,
                                time_ms: stats.prompt_ms,
                            }),
                            completion_probabilities: None,
                        };
                        let _ = progress_events.json(&frame);
                    }
                    if !stats.first_token {
                        return;
                    }
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
                        // Upstream's final frame in stream mode carries neither
                        // content nor ids: both were already sent on the
                        // per-token frames (#1477).
                        tokens: Vec::new(),
                        tokens_predicted: result.completion_tokens,
                        tokens_evaluated: result.prompt_tokens,
                        cached_tokens: result.cached_tokens,
                        probs: result.logprobs.clone(),
                        stop_kind: &result.stop_kind,
                        prompt_ms: result.prompt_eval_ms as f64,
                        predicted_ms: result.generation_only_ms as f64,
                        id_slot: slot.id_slot(),
                        // The final frame is where the run's totals are
                        // known; the per-token frames above stay without
                        // them (#1314).
                        speculative: result.speculative,
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

/// Validate a native `samplers` value against mlxcel's fixed chain, in
/// either of the two upstream shapes (an array of stage names or a single
/// string of stage characters): the fixed b10621 default order is accepted,
/// optionally extended with the `adaptive_p` stage (#1485; upstream appends
/// that stage after the chain wherever the list names it). `Ok(named)`
/// reports whether `adaptive_p` was named; `Err(())` is an order mlxcel
/// cannot honor.
fn native_samplers_validate(value: &serde_json::Value) -> Result<bool, ()> {
    match value {
        serde_json::Value::Array(items) => {
            let names: Vec<&str> = items
                .iter()
                .map(|item| item.as_str().ok_or(()))
                .collect::<Result<_, _>>()?;
            crate::server::startup::validate_sampler_names(&names)
        }
        serde_json::Value::String(chars) => crate::server::startup::validate_sampler_seq(chars),
        _ => Err(()),
    }
}

/// Whether a native `samplers` value spells an accepted chain
/// ([`native_samplers_validate`]); kept as the boolean form the field
/// rejection uses.
fn native_samplers_is_the_default_order(value: &serde_json::Value) -> bool {
    native_samplers_validate(value).is_ok()
}

/// Reject the native fields mlxcel has no equivalent for.
///
/// The epic's policy is that a flag or field whose value has observable
/// semantics must not be silently ignored. Each field below is accepted at its
/// inert value and refused otherwise, with a message naming the field and what
/// to use instead, so a `llama-server` request that depends on one fails
/// loudly here instead of returning a plausible-looking wrong answer.
/// Validate the per-request `lora` field (#1439).
///
/// With the unfused runtime path (the default), the field carries upstream's
/// full semantics: listed ids set their scale, unlisted adapters drop to 0.0,
/// unknown ids are ignored, and the resolved snapshot applies to this
/// request's forwards only (see [`resolve_request_lora_scales`]). Under
/// `--lora-fuse`, adapters are baked into the weights, so the only honest
/// answers are accepting a value that resolves to the configuration in force
/// (inert, upstream's own semantics for that value) and refusing anything
/// else with a diagnostic, instead of serving the request on weights the
/// client did not ask for.
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
    if state.config.lora_runtime.is_some() {
        // Runtime path: every resolvable value is servable per request.
        return None;
    }
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
            "per-request LoRA adapter selection is not supported with --lora-fuse: the adapters are fused into the model weights at load time, so this request's `lora` value cannot be served. Send the server's current adapter configuration (GET /lora-adapters), or restart without --lora-fuse",
            "invalid_request_error",
        )
        .into_response(),
    )
}

/// Resolve the request's `lora` field into the per-request scale snapshot
/// (#1439), upstream's `construct_lora_list` rule. `None` keeps the
/// server-default snapshot the options builder took.
fn resolve_request_lora_scales(
    config: &ServerConfig,
    request: &NativeCompletionRequest,
) -> Option<std::sync::Arc<Vec<f32>>> {
    config.lora_runtime.as_ref()?;
    let entries = request.lora.as_ref()?.as_array()?;
    Some(std::sync::Arc::new(super::lora_adapters::requested_scales(
        entries,
        config.lora_adapters.len(),
    )))
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
            "samplers {samplers} is not supported: mlxcel's sampler chain order is fixed to the b10621 default (penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature, character form edskypmxt), optionally extended with adaptive_p / a (#1485); send that order or omit the field"
        ));
    }

    // #1485 value-domain validations mirroring b10621's schema handlers.
    if let Some(mode) = request.mirostat
        && !(0..=2).contains(&mode)
    {
        return refuse(format!(
            "mirostat {mode} is not supported: valid values are 0 (disabled), 1 (Mirostat) and 2 (Mirostat 2.0)"
        ));
    }
    // b10621 declares HARD limits 0.0..=0.99 on adaptive_decay.
    if let Some(decay) = request.adaptive_decay
        && !(decay.is_finite() && (0.0..=0.99).contains(&decay))
    {
        return refuse(format!(
            "adaptive_decay: Value must be between 0 <= value <= 0.99, but got {decay}"
        ));
    }
    // b10621 declares HARD limits 0..=2147483647 on min_keep.
    if let Some(min_keep) = request.min_keep
        && !(0..=i64::from(i32::MAX)).contains(&min_keep)
    {
        return refuse(format!(
            "min_keep: Value must be between 0 <= value <= 2147483647, but got {min_keep}"
        ));
    }
    // b10621 accepts only a non-empty array of STRINGS (its json_value
    // falls back to an empty vector on any other shape and then errors).
    if let Some(breakers) = request.dry_sequence_breakers.as_ref() {
        let ok = breakers
            .as_array()
            .is_some_and(|arr| !arr.is_empty() && arr.iter().all(serde_json::Value::is_string));
        if !ok {
            return refuse(
                "Error: dry_sequence_breakers must be a non-empty array of strings".to_string(),
            );
        }
    }
    if request.n_cmpl.is_some_and(|n| n != 1) {
        return refuse(
            "n_cmpl (alias n) above 1 is not supported on /completion; mlxcel serves one \
             completion per request. Send one request per completion"
                .to_string(),
        );
    }
    // b10621 declares HARD limits 0..=2147483647 on n_indent, so a negative
    // value is refused rather than clamped (#1477).
    if let Some(n) = request.n_indent
        && !(0..=i64::from(i32::MAX)).contains(&n)
    {
        return refuse(format!(
            "n_indent: Value must be between 0 <= value <= 2147483647, but got {n}"
        ));
    }
    // b10621 declares HARD limits -1..=INT64_MAX on t_max_predict_ms; only -1
    // and 0 disable it, and anything below -1 is refused.
    if let Some(ms) = request.t_max_predict_ms
        && ms < -1
    {
        return refuse(format!(
            "t_max_predict_ms: Value must be between -1 <= value <= 9223372036854775807, but got {ms}"
        ));
    }
    None
}

/// The top-level `tokens` array of a non-streaming native response (#1477).
///
/// b10621 accumulates the sampled ids only when `return_tokens` is set and
/// returns an empty array otherwise, so the field is a projection rather than
/// an extra generation cost. The matched token of a string stop sequence is
/// included even though its text is excluded from `content`, which is why the
/// ids come from the scheduler rather than from re-tokenizing the answer.
fn native_returned_tokens(request: &NativeCompletionRequest, ids: Vec<i32>) -> Vec<i32> {
    if request.return_tokens.unwrap_or(false) {
        ids
    } else {
        Vec::new()
    }
}

fn build_native_options(
    request: &NativeCompletionRequest,
    requested_n_predict: Option<usize>,
    state: &AppState,
    live: &LiveSettings,
) -> ServerGenerateOptions {
    let mut options =
        build_native_generate_options_with_live(&state.config, live, request, requested_n_predict);
    // b10621 `--cache-prompt` / `cache_prompt` coverage (#1473): this route
    // used to build no prompt-cache request context at all, so it never looked
    // a prefix up and never donated one back, whatever the flag said. The
    // per-request field opts one request out of both halves by withholding the
    // context, which is the same mechanism the chat routes use.
    options.prompt_cache_ctx =
        super::chat::build_raw_prompt_cache_context(state, request.cache_prompt);
    options
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
    // b10621 logit_bias (#1485): a present field replaces the server-wide
    // set wholesale (even when it parses to nothing), upstream's
    // clear-and-rebuild handler.
    let (logit_bias, logit_bias_texts) = match request.logit_bias.as_ref() {
        Some(value) => {
            let (nums, texts) = parse_native_logit_bias(value);
            (Some(nums), texts)
        }
        None => (None, Vec::new()),
    };

    let mut options = build_server_generate_options_with_live(
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
            // b10621 value domain (#1485): the native field carries breaker
            // STRINGS (validated non-empty-array-of-strings by the field
            // rejection); the exact-id channel serves only the
            // OpenAI-shaped surface.
            dry_sequence_breakers: None,
            dry_sequence_breaker_strings: request.dry_sequence_breakers.as_ref().map(|v| {
                v.as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default()
            }),
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
            // #1485 sampling remainder.
            mirostat: request.mirostat,
            mirostat_tau: request.mirostat_tau,
            mirostat_eta: request.mirostat_eta,
            dynatemp_range: request.dynatemp_range,
            dynatemp_exponent: request.dynatemp_exponent,
            // b10621 declares a SOFT upper limit of 1.0 (values above clamp
            // down); negative disables at the sampler.
            adaptive_target: request.adaptive_target.map(|v| v.min(1.0)),
            adaptive_decay: request.adaptive_decay,
            // A present `samplers` field replaces the server-wide list
            // wholesale, so its adaptive_p flag wins outright (the value was
            // already validated by the field rejection).
            adaptive_p_named: request
                .samplers
                .as_ref()
                .map(|v| native_samplers_validate(v).unwrap_or(false)),
            min_keep: request.min_keep.map(|v| v.max(0) as usize),
            // b10621 alias order: `logprobs` counts only when `n_probs`
            // itself is absent. Non-positive values disable the report.
            n_probs: request
                .n_probs
                .or(request.logprobs)
                .map(|n| n.max(0) as usize),
            post_sampling_probs: request.post_sampling_probs,
            logit_bias,
            logit_bias_texts,
        },
    );
    // b10621 `n_keep` / `n_discard` (#1472): the per-request retention
    // overrides, with `n_keep` resolved against the server-wide `--keep` here
    // so the echoed generation settings report the effective value.
    options.retention = crate::server::config::RetentionOverride {
        n_keep: Some(request.n_keep.unwrap_or(config.n_keep)),
        n_discard: request.n_discard,
    };
    // b10621 `n_indent` / `t_max_predict_ms` (#1477): the two per-request
    // generation bounds, resolved to the scheduler's inert forms. Both value
    // domains were checked before the request was admitted.
    options.n_indent = request.n_indent.unwrap_or(0).max(0) as usize;
    options.t_max_predict_ms = request
        .t_max_predict_ms
        .filter(|&ms| ms > 0)
        .map(|ms| ms as u64);
    // Per-request adapter scales (#1439): the request's own `lora` field
    // replaces the server-default snapshot the options builder took, for this
    // request's forwards only. Resolved in this funnel rather than at the
    // route, so the `#[allow(dead_code)]` test helper sees the same field the
    // live path does.
    if let Some(scales) = resolve_request_lora_scales(config, request) {
        options.lora_scales = Some(scales);
    }
    options
}

/// Parse the native `logit_bias` value (#1485), b10621's two shapes: an
/// array of `[token, bias]` pairs, or an object mapping token (an id, or a
/// string to tokenize) to bias. `false` as a bias bans the token (`-inf`);
/// `true` and malformed entries are skipped, exactly as upstream's handler
/// skips them. String keys come back separately for enqueue-time
/// tokenization.
fn parse_native_logit_bias(value: &serde_json::Value) -> (Vec<(i32, f32)>, Vec<(String, f32)>) {
    fn parse_bias(v: &serde_json::Value) -> Option<f32> {
        if let Some(n) = v.as_f64() {
            return Some(n as f32);
        }
        if v.as_bool() == Some(false) {
            return Some(f32::NEG_INFINITY);
        }
        None
    }
    let mut nums = Vec::new();
    let mut texts = Vec::new();
    match value {
        serde_json::Value::Array(entries) => {
            for el in entries {
                let Some(pair) = el.as_array() else { continue };
                if pair.len() != 2 {
                    continue;
                }
                let Some(bias) = parse_bias(&pair[1]) else {
                    continue;
                };
                if let Some(id) = pair[0].as_i64() {
                    nums.push((id as i32, bias));
                } else if let Some(text) = pair[0].as_str() {
                    texts.push((text.to_string(), bias));
                }
            }
        }
        serde_json::Value::Object(map) => {
            for (key, v) in map {
                let Some(bias) = parse_bias(v) else { continue };
                match key.parse::<i32>() {
                    Ok(id) => nums.push((id, bias)),
                    Err(_) => texts.push((key.clone(), bias)),
                }
            }
        }
        _ => {}
    }
    (nums, texts)
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
        let settings =
            native_generation_settings(&ServerConfig::default(), None, &request, &options);
        assert_eq!(settings["typical_p"], 0.5);
    }

    // -- #1485 sampling remainder --

    #[test]
    fn native_completion_maps_the_1485_sampler_fields() {
        let request = native_request(
            r#"{"prompt":"hi","mirostat":2,"mirostat_tau":4.0,"mirostat_eta":0.2,"dynatemp_range":0.4,"dynatemp_exponent":1.5,"min_keep":3}"#,
        );
        assert!(reject_unsupported_native_fields(&request).is_none());
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert_eq!(options.sampling.mirostat, 2);
        assert_eq!(options.sampling.mirostat_tau, 4.0);
        assert_eq!(options.sampling.mirostat_eta, 0.2);
        assert_eq!(options.sampling.dynatemp_range, 0.4);
        assert_eq!(options.sampling.dynatemp_exponent, 1.5);
        assert_eq!(options.sampling.min_keep, 3);
    }

    #[test]
    fn native_mirostat_outside_the_declared_domain_is_refused() {
        // b10621 would abort its own process in common_sampler_init; the
        // actionable form of the same refusal is a 400 naming the domain.
        for body in [
            r#"{"prompt":"hi","mirostat":3}"#,
            r#"{"prompt":"hi","mirostat":-1}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_some(),
                "{body}"
            );
        }
        assert!(
            reject_unsupported_native_fields(&native_request(r#"{"prompt":"hi","mirostat":0}"#))
                .is_none()
        );
    }

    #[test]
    fn native_adaptive_decay_and_min_keep_enforce_upstreams_hard_limits() {
        for body in [
            r#"{"prompt":"hi","adaptive_decay":1.0}"#,
            r#"{"prompt":"hi","adaptive_decay":-0.1}"#,
            r#"{"prompt":"hi","min_keep":-1}"#,
            r#"{"prompt":"hi","min_keep":2147483648}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_some(),
                "outside a hard schema limit must be a 400: {body}"
            );
        }
        for body in [
            r#"{"prompt":"hi","adaptive_decay":0.99}"#,
            r#"{"prompt":"hi","min_keep":0}"#,
            r#"{"prompt":"hi","min_keep":2147483647}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "{body}"
            );
        }
    }

    #[test]
    fn native_adaptive_p_activates_only_through_the_samplers_stage() {
        // b10621 runs adaptive-p solely when params.samplers names the
        // stage; a bare adaptive_target is inert there and here.
        let bare = native_request(r#"{"prompt":"hi","adaptive_target":0.3}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &bare, None);
        assert_eq!(
            options.sampling.adaptive_target, -1.0,
            "inert without the stage"
        );

        let named = native_request(
            r#"{"prompt":"hi","adaptive_target":0.3,"samplers":["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature","adaptive_p"]}"#,
        );
        assert!(reject_unsupported_native_fields(&named).is_none());
        let options = build_native_generate_options(&ServerConfig::default(), &named, None);
        assert_eq!(options.sampling.adaptive_target, 0.3);
        // The soft schema limit clamps values above 1.0 down, as upstream's
        // set_limits does.
        let clamped =
            native_request(r#"{"prompt":"hi","adaptive_target":2.0,"samplers":"edskypmxta"}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &clamped, None);
        assert_eq!(options.sampling.adaptive_target, 1.0);
    }

    #[test]
    fn native_samplers_accepts_the_adaptive_p_extension_only() {
        for body in [
            r#"{"prompt":"hi","samplers":["adaptive_p","penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"]}"#,
            r#"{"prompt":"hi","samplers":"aedskypmxt"}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "adaptive_p may sit anywhere in the fixed order: {body}"
            );
        }
        let request = native_request(r#"{"prompt":"hi","samplers":["adaptive_p","top_k"]}"#);
        assert!(
            reject_unsupported_native_fields(&request).is_some(),
            "the nine fixed stages must still all be present in order"
        );
    }

    #[test]
    fn native_dry_sequence_breakers_must_be_a_nonempty_string_array() {
        for body in [
            r#"{"prompt":"hi","dry_sequence_breakers":[]}"#,
            r#"{"prompt":"hi","dry_sequence_breakers":[198]}"#,
            r#"{"prompt":"hi","dry_sequence_breakers":"\n"}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_some(),
                "b10621 wording refuses anything but a non-empty string array: {body}"
            );
        }
        let request = native_request(r#"{"prompt":"hi","dry_sequence_breakers":["\n",":"]}"#);
        assert!(reject_unsupported_native_fields(&request).is_none());
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert_eq!(
            options.dry_breaker_strings,
            Some(vec!["\n".to_string(), ":".to_string()]),
            "the strings ride to the scheduler's vocabulary-scan derivation"
        );
        assert_eq!(options.sampling.dry_sequence_breakers, Vec::<i32>::new());
    }

    #[test]
    fn native_logit_bias_parses_both_upstream_shapes_and_false_bans() {
        let array = native_request(
            r#"{"prompt":"hi","logit_bias":[[15043,1.5],[7,false],["Hello",-2.0],[3,true],"bad"]}"#,
        );
        let options = build_native_generate_options(&ServerConfig::default(), &array, None);
        assert_eq!(options.logit_bias[0], (15043, 1.5));
        assert_eq!(options.logit_bias[1].0, 7);
        assert_eq!(
            options.logit_bias[1].1,
            f32::NEG_INFINITY,
            "false bans the token"
        );
        assert_eq!(
            options.logit_bias.len(),
            2,
            "true and malformed entries are skipped"
        );
        assert_eq!(options.logit_bias_texts, vec![("Hello".to_string(), -2.0)]);

        let object = native_request(r#"{"prompt":"hi","logit_bias":{"15043":1.0," Hi":false}}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &object, None);
        assert_eq!(options.logit_bias, vec![(15043, 1.0)]);
        assert_eq!(options.logit_bias_texts.len(), 1);
        assert_eq!(options.logit_bias_texts[0].0, " Hi");
        assert_eq!(options.logit_bias_texts[0].1, f32::NEG_INFINITY);
    }

    #[test]
    fn native_logit_bias_field_replaces_the_server_wide_set() {
        let config = ServerConfig {
            default_logit_bias: vec![(5, 3.0)],
            ..Default::default()
        };
        let absent = native_request(r#"{"prompt":"hi"}"#);
        let options = build_native_generate_options(&config, &absent, None);
        assert_eq!(
            options.logit_bias,
            vec![(5, 3.0)],
            "absent inherits the flag set"
        );

        let empty = native_request(r#"{"prompt":"hi","logit_bias":[]}"#);
        let options = build_native_generate_options(&config, &empty, None);
        assert_eq!(
            options.logit_bias,
            Vec::<(i32, f32)>::new(),
            "a present field clears and rebuilds, upstream's handler shape"
        );
    }

    #[test]
    fn native_n_probs_and_its_logprobs_alias_enable_the_report() {
        let request = native_request(r#"{"prompt":"hi","n_probs":5}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        assert!(options.logprobs.enabled);
        assert_eq!(options.logprobs.top_k, 5);
        assert_eq!(
            options.logprobs.source,
            mlxcel_core::sampling::LogprobSource::RawModel
        );

        let alias = native_request(r#"{"prompt":"hi","logprobs":3}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &alias, None);
        assert_eq!(options.logprobs.top_k, 3);

        // n_probs wins over the alias when both are present, upstream's
        // alias order.
        let both = native_request(r#"{"prompt":"hi","n_probs":2,"logprobs":9}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &both, None);
        assert_eq!(options.logprobs.top_k, 2);

        let post = native_request(r#"{"prompt":"hi","n_probs":4,"post_sampling_probs":true}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &post, None);
        assert_eq!(
            options.logprobs.source,
            mlxcel_core::sampling::LogprobSource::PostSampling
        );
        assert!(options.post_sampling_probs);

        let off = native_request(r#"{"prompt":"hi","n_probs":0}"#);
        let options = build_native_generate_options(&ServerConfig::default(), &off, None);
        assert!(!options.logprobs.enabled);
    }

    #[test]
    fn native_backend_sampling_is_accepted_and_inert_in_both_values() {
        for body in [
            r#"{"prompt":"hi","backend_sampling":true}"#,
            r#"{"prompt":"hi","backend_sampling":false}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "{body}"
            );
            let options = build_native_generate_options(&ServerConfig::default(), &request, None);
            // mlxcel's sampler IS the backend graph; nothing to switch.
            assert!(config_is_untouched_by_backend_sampling(&options));
        }
    }

    fn config_is_untouched_by_backend_sampling(options: &ServerGenerateOptions) -> bool {
        let baseline = build_native_generate_options(
            &ServerConfig::default(),
            &native_request(r#"{"prompt":"hi"}"#),
            None,
        );
        options.sampling.temperature == baseline.sampling.temperature
            && options.sampling.top_k == baseline.sampling.top_k
    }

    /// A tiny byte-level tokenizer with a single-token `<tool>` marker, so the
    /// preserved-token and trigger-word promotion rules can be exercised
    /// without loading a checkpoint.
    fn grammar_tokenizer() -> crate::tokenizer::MlxcelTokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 5, "content": "<tool>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {"a": 0, "b": 1, "c": 2, " ": 3, "{": 4, "<tool>": 5},
                "merges": []
            }
        }"#;
        crate::tokenizer::MlxcelTokenizer::HuggingFace(
            tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("fixture tokenizer builds"),
        )
    }

    fn resolve(body: &str) -> Result<Option<crate::server::grammar::GrammarSpec>, String> {
        let request = native_request(body);
        let tokenizer = grammar_tokenizer();
        crate::server::grammar::resolve_native_grammar(
            &tokenizer,
            request.json_schema.as_ref(),
            request.grammar.as_ref(),
            request.grammar_lazy,
            request.grammar_triggers.as_ref(),
            request.preserved_tokens.as_ref(),
            None,
        )
        .map_err(|e| e.to_string())
    }

    #[test]
    fn native_grammar_surfaces_no_longer_reject_a_constrained_request() {
        // Every one of these was a 400 before the grammar engine landed; the
        // route must now admit them and resolve a constraint instead.
        for body in [
            r#"{"prompt":"hi","json_schema":{}}"#,
            r#"{"prompt":"hi","grammar":"root ::= \"a\""}"#,
            r#"{"prompt":"hi","grammar_lazy":true}"#,
        ] {
            let request = native_request(body);
            assert!(
                reject_unsupported_native_fields(&request).is_none(),
                "grammar surfaces are implemented and must not be refused: {body}"
            );
        }
    }

    #[test]
    fn grammar_wins_over_json_schema_by_presence_not_by_emptiness() {
        // b10621 takes the schema branch only when `json_schema` is present
        // AND `grammar` is absent, which its own field description contradicts.
        let spec = resolve(
            r#"{"prompt":"hi","json_schema":{"type":"object"},"grammar":"root ::= \"a\""}"#,
        )
        .unwrap()
        .expect("a grammar is resolved");
        assert_eq!(spec.gbnf.as_deref(), Some("root ::= \"a\""));
        assert!(spec.schema.is_none());

        // An EMPTY grammar alongside a schema leaves no grammar at all.
        assert!(
            resolve(r#"{"prompt":"hi","json_schema":{"type":"object"},"grammar":""}"#)
                .unwrap()
                .is_none()
        );

        // The schema alone takes the schema branch.
        let spec = resolve(r#"{"prompt":"hi","json_schema":{"type":"object"}}"#)
            .unwrap()
            .expect("a schema is resolved");
        assert!(spec.schema.is_some());
        assert!(spec.gbnf.is_none());
    }

    #[test]
    fn a_non_string_grammar_is_ignored_the_way_upstreams_json_value_ignores_it() {
        assert!(
            resolve(r#"{"prompt":"hi","grammar":123}"#)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_lazy_grammar_needs_triggers_only_when_the_triggers_key_is_present() {
        // Upstream's check lives inside the `grammar_triggers` handler, so a
        // bare `grammar_lazy: true` is accepted and simply never triggers.
        assert!(
            resolve(r#"{"prompt":"hi","grammar_lazy":true}"#)
                .unwrap()
                .is_none()
        );
        let err =
            resolve(r#"{"prompt":"hi","grammar_lazy":true,"grammar_triggers":[]}"#).unwrap_err();
        assert_eq!(err, "Error: no triggers set for lazy grammar!");
    }

    #[test]
    fn a_single_token_trigger_word_must_be_a_preserved_token() {
        let body = r#"{"prompt":"hi","grammar":"root ::= \"a\"","grammar_lazy":true,"grammar_triggers":[{"type":1,"value":"<tool>"}]}"#;
        let err = resolve(body).unwrap_err();
        assert_eq!(
            err,
            "Grammar trigger word should be marked as preserved token: <tool>"
        );

        let body = r#"{"prompt":"hi","grammar":"root ::= \"a\"","grammar_lazy":true,"preserved_tokens":["<tool>"],"grammar_triggers":[{"type":1,"value":"<tool>"}]}"#;
        let spec = resolve(body).unwrap().expect("a grammar is resolved");
        assert_eq!(
            spec.triggers,
            vec![crate::server::grammar::GrammarTrigger::Token(5)]
        );
        assert_eq!(spec.preserved, vec![5]);

        // A multi-token word stays a WORD trigger and needs no preserved entry.
        let body = r#"{"prompt":"hi","grammar":"root ::= \"a\"","grammar_lazy":true,"grammar_triggers":[{"type":1,"value":"abc"}]}"#;
        let spec = resolve(body).unwrap().expect("a grammar is resolved");
        assert_eq!(
            spec.triggers,
            vec![crate::server::grammar::GrammarTrigger::Word(
                "abc".to_string()
            )]
        );
    }

    #[test]
    fn a_malformed_trigger_or_preserved_shape_is_refused() {
        assert!(resolve(r#"{"prompt":"hi","grammar_triggers":{}}"#).is_err());
        assert!(resolve(r#"{"prompt":"hi","preserved_tokens":{}}"#).is_err());
        assert!(resolve(r#"{"prompt":"hi","grammar_triggers":[{"type":9,"value":"x"}]}"#).is_err());
    }

    #[test]
    fn the_server_default_grammar_survives_a_request_that_sets_none() {
        let default = crate::server::grammar::GrammarSpec::from_gbnf("root ::= \"z\"".to_string());
        let request = native_request(r#"{"prompt":"hi","grammar":""}"#);
        let tokenizer = grammar_tokenizer();
        let spec = crate::server::grammar::resolve_native_grammar(
            &tokenizer,
            request.json_schema.as_ref(),
            request.grammar.as_ref(),
            request.grammar_lazy,
            request.grammar_triggers.as_ref(),
            request.preserved_tokens.as_ref(),
            Some(&default),
        )
        .unwrap()
        .expect("the server default is inherited");
        assert_eq!(spec.gbnf.as_deref(), Some("root ::= \"z\""));
    }

    #[test]
    fn native_seed_folds_into_uint32_space() {
        let request = native_request(r#"{"prompt":"hi","seed":-2}"#);
        assert_eq!(
            request.seed,
            Some(4_294_967_294),
            "b10621's unchecked uint32 cast makes -2 a deterministic seed"
        );
        let random = native_request(r#"{"prompt":"hi","seed":-1}"#);
        assert_eq!(random.seed, None);
    }

    #[test]
    fn native_generation_settings_echo_the_1485_keys() {
        let request = native_request(
            r#"{"prompt":"hi","mirostat":1,"dynatemp_range":0.5,"min_keep":2,"n_probs":3,"dry_sequence_breakers":["x"]}"#,
        );
        let options = build_native_generate_options(&ServerConfig::default(), &request, None);
        let settings =
            native_generation_settings(&ServerConfig::default(), None, &request, &options);
        assert_eq!(settings["mirostat"], 1);
        assert_eq!(settings["dynatemp_range"], 0.5);
        assert_eq!(settings["min_keep"], 2);
        assert_eq!(settings["n_probs"], 3);
        assert_eq!(settings["post_sampling_probs"], false);
        assert_eq!(settings["dry_sequence_breakers"], serde_json::json!(["x"]));
        assert_eq!(settings["adaptive_target"], -1.0);
    }

    #[test]
    fn native_probability_report_uses_upstreams_key_names() {
        use mlxcel_core::sampling::TokenLogprobData;
        let entries = [TokenLogprobData {
            token_id: 3,
            logprob: -0.5,
            top_alternatives: vec![(3, -0.5), (7, -1.5)],
        }];
        let pre = native_completion_probabilities(None, &entries, false);
        assert!(pre[0]["logprob"].is_number());
        assert!(pre[0]["top_logprobs"].is_array());
        assert_eq!(pre[0]["id"], 3);
        assert_eq!(pre[0]["top_logprobs"][1]["id"], 7);
        let post = native_completion_probabilities(None, &entries, true);
        assert!(post[0]["prob"].is_number());
        assert!(post[0]["top_probs"].is_array());
    }
}
