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

//! `POST /v1/responses/compact` and `POST /responses/compact` (issue #2397).
//!
//! The handler expands any capsule already in `input`, then hands the history
//! to [`compact_history`] with a [`ResponsesCompactionBackend`]: token counts
//! run the `/v1/responses/input_tokens` render and count, and the summary is
//! one non-streaming generation on the worker path `/v1/responses` uses. The
//! answer is `{id, object: "response.compaction", output, usage}`, where
//! `output` is the kept instructions, the retained items and one `compaction`
//! item, and `usage` counts the summary pass. Nothing is stored.

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use super::chat::{
    build_generate_options_with_live, build_prompt_cache_request_context,
    is_prompt_primed_open_thinking,
};
use super::prompt_inspection::{count_prompt_tokens, count_text_prompt_tokens, render_chat_prompt};
use super::responses_compact_media::MediaCache;
use crate::server::request_options::resolve_server_max_tokens_with_live;
use crate::server::responses_compaction::{
    CompactionBackend, CompactionError, CompactionLimits, DEFAULT_SUMMARY_TOKENS,
    MAX_SUMMARY_TOKENS, SUMMARY_INSTRUCTION, SummaryPass, compact_history, compaction_item,
    default_keep_tokens, encode_capsule, input_item_to_json, resolve_compaction,
};
use crate::server::responses_translator::responses_request_to_chat;
use crate::server::tool_calls;
use crate::server::types::ErrorResponse;
use crate::server::types::request::ChatCompletionRequest;
use crate::server::types::responses_request::{
    CreateResponseRequest, ResponseInput, ResponseInputContent, ResponseInputItem,
    ResponseInputRole, ResponseReasoningConfig,
};
use crate::server::{AppState, LiveSettings};

/// POST /v1/responses/compact and POST /responses/compact
pub async fn compact_response(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    if let Some(response) = super::chat_not_available(&state) {
        return response.into_response();
    }
    match compact(&state, body).await {
        Ok(body) => Json(body).into_response(),
        Err(err) => err.into_error_response().into_response(),
    }
}

async fn compact(state: &AppState, body: Value) -> Result<Value, CompactionError> {
    let invalid = |message: String| CompactionError::InvalidRequest(message);
    // `keep_tokens` is the one compact-only field; the rest of the body parses
    // as a Responses request. Read it alone so a large body is not cloned.
    let keep_tokens: Option<usize> = match body.get("keep_tokens") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            serde_json::from_value(value.clone())
                .map_err(|err| invalid(format!("Invalid request: keep_tokens: {err}")))?,
        ),
    };
    let mut request: CreateResponseRequest =
        serde_json::from_value(body).map_err(|err| invalid(format!("Invalid request: {err}")))?;
    if request.stream {
        return Err(invalid(
            "compaction is not streamed; omit stream".to_string(),
        ));
    }
    if request.previous_response_id.is_some() || request.conversation.is_some() {
        return Err(invalid(
            "compaction takes the full history in input; previous_response_id and \
             conversation are not accepted here"
                .to_string(),
        ));
    }
    let summary_tokens = request.max_output_tokens.unwrap_or(DEFAULT_SUMMARY_TOKENS);
    if !(1..=MAX_SUMMARY_TOKENS).contains(&summary_tokens) {
        return Err(invalid(format!(
            "max_output_tokens must be between 1 and {MAX_SUMMARY_TOKENS}"
        )));
    }
    if keep_tokens == Some(0) {
        return Err(invalid("keep_tokens must be > 0".to_string()));
    }

    let input = std::mem::replace(&mut request.input, ResponseInput::Items(Vec::new()));
    let items = resolve_compaction(input.into_items())
        .map_err(|err| invalid(format!("invalid compaction item: {err}")))?;
    let context_tokens = state.effective_context_size();
    let limits = CompactionLimits {
        context_tokens,
        keep_tokens: keep_tokens.unwrap_or_else(|| default_keep_tokens(context_tokens)),
    };
    let live = state.live();
    let backend = ResponsesCompactionBackend {
        state,
        live: &live,
        template: &request,
        summary_tokens,
        media: MediaCache::default(),
    };
    let outcome = compact_history(&backend, items, limits).await?;

    let summary = outcome.summary.text.trim();
    let capsule = compaction_item(encode_capsule(summary, &outcome.retained));
    let mut output: Vec<Value> = outcome
        .instructions
        .iter()
        .chain(&outcome.retained)
        .map(input_item_to_json)
        .collect();
    output.push(json!({
        "type": "compaction",
        "id": capsule.id,
        "encrypted_content": capsule.encrypted_content,
    }));
    validate_replay_size(
        &output,
        state.display_model_id(),
        crate::server::app::main_json_body_limit_bytes(),
    )?;
    let pass = &outcome.summary;
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    tracing::info!(
        response_id = %id,
        input_tokens_before = outcome.tokens_before,
        input_tokens_after = outcome.tokens_after,
        "Responses compaction answered"
    );
    Ok(json!({
        "id": id,
        "object": "response.compaction",
        "created_at": chrono::Utc::now().timestamp(),
        "model": state.display_model_id(),
        "output": output,
        "usage": {
            "input_tokens": pass.prompt_tokens,
            "input_tokens_details": {"cached_tokens": pass.cached_tokens},
            "output_tokens": pass.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": pass.prompt_tokens + pass.completion_tokens,
        },
    }))
}

/// The retained items appear both in plain output and in the version-1
/// capsule. Refuse an output the same server would reject on immediate replay.
pub(crate) fn validate_replay_size(
    output: &[Value],
    model: &str,
    limit: usize,
) -> Result<(), CompactionError> {
    #[derive(serde::Serialize)]
    struct Replay<'a> {
        model: &'a str,
        input: &'a [Value],
    }
    let bytes = crate::server::store_budget::serialized_json_len_saturating(&Replay {
        model,
        input: output,
    });
    if bytes > limit {
        return Err(CompactionError::InvalidRequest(format!(
            "compacted output needs {bytes} bytes to replay, above the {limit}-byte request limit; reduce retained media or keep_tokens. Original context preserved"
        )));
    }
    Ok(())
}

/// [`CompactionBackend`] on the `/v1/responses` render, count and worker path.
///
/// `template` supplies everything but `input` (model, `instructions`, tools,
/// sampling), so a count renders exactly what `/v1/responses` would.
///
/// Used by: `POST /v1/responses/compact`; the phase-B automatic compaction
/// hook on `/v1/responses` reuses it.
pub(crate) struct ResponsesCompactionBackend<'a> {
    pub state: &'a AppState,
    pub live: &'a std::sync::Arc<LiveSettings>,
    pub template: &'a CreateResponseRequest,
    /// Requested summary output budget.
    pub summary_tokens: usize,
    /// Images read once for every render of this compaction.
    pub media: MediaCache,
}

impl ResponsesCompactionBackend<'_> {
    fn translate(
        &self,
        request: &CreateResponseRequest,
    ) -> Result<ChatCompletionRequest, CompactionError> {
        responses_request_to_chat(request, None, None)
            .map(|translated| translated.chat_request)
            .map_err(|err| CompactionError::InvalidRequest(err.to_string()))
    }

    async fn localized_items(
        &self,
        items: &[ResponseInputItem],
    ) -> Result<Vec<ResponseInputItem>, CompactionError> {
        // Admission must precede the cache's first fetch, just as it precedes
        // media acquisition in the ordinary input-token route.
        let chat = self.translate(&self.request_with(items.to_vec()))?;
        if let Some(error) = crate::server::media_capability_rejection(
            &chat,
            self.state.media_support,
            self.state.display_model_id(),
        ) {
            return Err(CompactionError::Response(error));
        }
        super::chat::validate_chat_tool_inputs(&chat).map_err(CompactionError::InvalidRequest)?;
        crate::server::media::validate_image_count(
            chat.image_urls().len(),
            crate::server::media::current_image_input_limits(),
        )
        .map_err(|error| CompactionError::InvalidRequest(error.to_string()))?;
        Ok(self.media.localize(items).await)
    }

    fn request_with(&self, items: Vec<ResponseInputItem>) -> CreateResponseRequest {
        let mut request = self.template.clone();
        request.input = ResponseInput::Items(items);
        request.store = Some(false);
        request
    }
}

impl CompactionBackend for ResponsesCompactionBackend<'_> {
    async fn count(&self, items: &[ResponseInputItem]) -> Result<usize, CompactionError> {
        let items = self.localized_items(items).await?;
        let chat = self.translate(&self.request_with(items))?;
        let prepared = render_chat_prompt(self.state, self.live, chat)
            .await
            .map_err(CompactionError::Response)?;
        count_prompt_tokens(self.state, self.live, prepared)
            .await
            .map_err(|err| CompactionError::Response(err.into_error_response()))
    }

    async fn summarize(&self, head: &[ResponseInputItem]) -> Result<SummaryPass, CompactionError> {
        let mut items = self.localized_items(head).await?;
        items.push(ResponseInputItem::Message {
            role: ResponseInputRole::User,
            content: ResponseInputContent::Text(SUMMARY_INSTRUCTION.to_string()),
            name: None,
        });
        let mut request = self.request_with(items);
        // A deterministic, tool-free, thinking-free pass: the instruction says
        // not to call tools, and no grammar or tool parser should run.
        request.tools = None;
        request.tool_choice = None;
        request.text = None;
        request.max_tool_calls = None;
        request.max_output_tokens = Some(self.summary_tokens);
        request.sampling.max_tokens = Some(self.summary_tokens);
        request.temperature = Some(0.0);
        request.sampling.temperature = Some(0.0);
        request.reasoning = Some(ResponseReasoningConfig {
            effort: Some("none".to_string()),
            summary: None,
        });
        let chat = self.translate(&request)?;
        generate_summary(self.state, self.live, chat).await
    }
}

/// One non-streaming generation through the provider call
/// `non_stream_create_response` makes. It neither records a reasoning-echo
/// reply nor stores a response: the pass is not a turn of the conversation.
async fn generate_summary(
    state: &AppState,
    live: &std::sync::Arc<LiveSettings>,
    chat: ChatCompletionRequest,
) -> Result<SummaryPass, CompactionError> {
    if !state.can_accept_request() {
        return Err(CompactionError::Response(
            ErrorResponse::service_unavailable("All slots are busy. Please try again later."),
        ));
    }
    let max_tokens =
        resolve_server_max_tokens_with_live(&state.config, live, chat.params.max_tokens);
    let mut prepared = render_chat_prompt(state, live, chat.clone())
        .await
        .map_err(CompactionError::Response)?;
    // A media prompt's length is known only on the worker; the scheduler's
    // admission guard refuses it there with the exceed-context wording.
    if !prepared.carries_media() {
        let prompt_tokens = count_text_prompt_tokens(
            state,
            &prepared.prompt,
            prepared.prompt_token_ids.as_deref(),
        )
        .map_err(|err| {
            CompactionError::Response(ErrorResponse::new(
                format!("Tokenization error: {err}"),
                "invalid_request_error",
            ))
        })?;
        let context = state.effective_context_size();
        if prompt_tokens + max_tokens > context {
            return Err(CompactionError::ContextLengthExceeded(format!(
                "the summary request needs {prompt_tokens} prompt tokens plus {max_tokens} output \
                 tokens, above the {context}-token context window; original context preserved"
            )));
        }
    }

    let primed_open_thinking =
        is_prompt_primed_open_thinking(&state.thinking_markers, &prepared.prompt);
    let mut options = build_generate_options_with_live(&chat.params, &state.config, live, false);
    options.image_soft_tokens = prepared.image_soft_tokens;
    options.pre_rendered_prompt_tokens = prepared.prompt_token_ids.take();
    options.thinking_enter_block_on_start = primed_open_thinking;
    // The summarized prefix is the prompt of earlier turns, so the prompt
    // cache can serve it.
    options.prompt_cache_ctx = build_prompt_cache_request_context(
        state,
        live,
        &chat,
        &prepared.image_data,
        &prepared.audio_data,
        prepared.history_prompt.as_deref(),
    );
    let slot = state.slots.begin(
        &prepared.prompt,
        super::slots::slot_params_json(&options, false),
        Some(options.max_tokens as i64),
    );
    let result = state
        .model_provider
        .generate_with_media_and_videos_declared_live_with_prefill(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
            prepared.media,
            live,
            |stats| {
                slot.on_prefill_progress(stats.prompt_tokens, stats.cached_tokens, stats.processed);
            },
        )
        .map_err(|err| {
            let response = super::generation_error_to_response(err);
            if response.error.error_type == "exceed_context_size_error" {
                CompactionError::ContextLengthExceeded(response.error.message)
            } else {
                CompactionError::Response(response)
            }
        })?;
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
    if result.finish_reason == "length" {
        return Err(CompactionError::SummaryFailed(format!(
            "the summary reached max_output_tokens ({max_tokens}) before it finished; raise \
             max_output_tokens. Original context preserved."
        )));
    }
    let (text, _reasoning) =
        super::responses::split_reasoning(&result.text, None, primed_open_thinking);
    let text = if tool_calls::answer_wrapper_enabled(&state.tokenizer) {
        tool_calls::strip_answer_wrapper(&text)
    } else {
        text
    };
    Ok(SummaryPass {
        text,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        cached_tokens: result.cached_tokens,
    })
}

#[cfg(test)]
#[path = "responses_compact_route_tests.rs"]
mod responses_compact_route_tests;
