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

//! Automatic compaction on `POST /v1/responses` (issue #2397, phase B).
//!
//! A request with `context_management: [{"type": "compaction",
//! "compact_threshold": N}]` has its rendered input counted on the
//! `/v1/responses/input_tokens` path before generation. When the count
//! reaches `N`, or leaves no room for output, the history goes through the
//! same [`compact_history_counted`] and [`ResponsesCompactionBackend`] as
//! `POST /v1/responses/compact`, and generation runs on the compacted
//! context. The response output then starts with the `compaction` item.
//!
//! For every `context_management` request, compacted or not, generation is
//! capped at the context left after the prompt (Blaizzy/mlx-vlm#2458), and a
//! `length` finish reports `status: "incomplete"` with reason
//! `max_output_tokens`. Requests without the field are untouched.
//!
//! Compaction finishes before the SSE stream opens, so a failure is an HTTP
//! error on both arms, as a context overflow is today. A streamed request
//! then reports it with `mlx.compaction.started` and
//! `mlx.compaction.completed` right after `response.in_progress`
//! (Blaizzy/mlx-vlm#2418).

use serde_json::{Value, json};

use super::prompt_inspection::count_text_prompt_tokens;
use super::responses_compact::{ResponsesCompactionBackend, validate_replay_size};
use super::responses_compact_media::MediaCache;
use crate::server::chat_request::PreparedChatRequest;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::responses_compaction::{
    CompactionBackend, CompactionError, CompactionLimits, compact_history_counted, compaction_item,
    default_keep_tokens, encode_capsule, summary_message,
};
use crate::server::responses_translator::{TranslatedRequest, responses_request_to_chat};
use crate::server::streaming_responses::{ResponseStreamEmitter, ResponseStreamSender};
use crate::server::thinking_budget::ThinkingBudget;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::{CreateResponseRequest, ResponseInput};
use crate::server::types::responses_response::{
    ResponseCompactionItem, ResponseItemStatus, ResponseObject, ResponseOutputItem, ResponseStatus,
};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, LiveSettings};

/// Largest `compact_threshold` accepted. Anything above it is a client bug,
/// not a context size.
pub(crate) const MAX_COMPACT_THRESHOLD: u64 = 100_000_000;
/// `incomplete_details.reason` for a reply that stopped at its output limit.
pub(crate) const INCOMPLETE_MAX_OUTPUT_TOKENS: &str = "max_output_tokens";

/// Parse `context_management` into its compaction threshold.
///
/// `None`, `null` and `[]` disable automatic compaction. More than one entry
/// (duplicates included), a `type` other than `compaction`, or a
/// `compact_threshold` that is missing, not an integer, below 1 or above
/// [`MAX_COMPACT_THRESHOLD`] is 400 `invalid_request_error`.
pub(crate) fn parse_context_management(
    value: Option<&Value>,
) -> Result<Option<usize>, ErrorResponse> {
    let invalid = |message: String| ErrorResponse::new(message, "invalid_request_error");
    let entries = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            return Err(invalid(
                "context_management must be an array of {\"type\": \"compaction\", \
                 \"compact_threshold\": N} entries"
                    .to_string(),
            ));
        }
    };
    let entry = match entries.as_slice() {
        [] => return Ok(None),
        [entry] => entry,
        _ => {
            return Err(invalid(format!(
                "context_management accepts one entry, got {}",
                entries.len()
            )));
        }
    };
    match entry.get("type").and_then(Value::as_str) {
        Some("compaction") => {}
        Some(other) => {
            return Err(invalid(format!(
                "unsupported context_management type '{other}'; only 'compaction' is supported"
            )));
        }
        None => {
            return Err(invalid(
                "context_management entry needs \"type\": \"compaction\"".to_string(),
            ));
        }
    }
    let threshold = entry
        .get("compact_threshold")
        .and_then(Value::as_u64)
        .filter(|threshold| (1..=MAX_COMPACT_THRESHOLD).contains(threshold))
        .ok_or_else(|| {
            invalid(format!(
                "context_management compact_threshold must be an integer between 1 and \
                 {MAX_COMPACT_THRESHOLD}"
            ))
        })?;
    // Bounded by MAX_COMPACT_THRESHOLD, so this cannot truncate.
    Ok(Some(threshold as usize))
}

/// The outcome of the pre-generation check for one `context_management`
/// request.
#[derive(Debug, Clone)]
pub(crate) struct AutoCompaction {
    /// Rendered input tokens before compaction.
    pub tokens_before: usize,
    /// Rendered input tokens of the context generation runs on.
    pub tokens_after: usize,
    /// The capsule, when compaction ran.
    pub item: Option<ResponseCompactionItem>,
}

impl AutoCompaction {
    pub(crate) fn compacted(&self) -> bool {
        self.item.is_some()
    }
}

/// Summary output budget of an automatic pass: 1/16 of the context, between
/// 128 and 1024 tokens (upstream's automatic default).
fn auto_summary_tokens(context_tokens: usize) -> usize {
    (context_tokens / 16).clamp(128, 1024)
}

fn context_length_exceeded(message: String) -> ErrorResponse {
    CompactionError::ContextLengthExceeded(message).into_error_response()
}

/// Count the rendered input of `translated` and compact it when it reaches
/// `threshold` or fills the context window.
///
/// On compaction, `translated.chat_request` is rebuilt from the instructions,
/// the summary and the retained turns, and `canonical_input_items` becomes
/// the instructions alone: the stored response carries the capsule as its
/// first output item, so a `previous_response_id` hop replays instructions,
/// capsule and tail. `effective_store` and `conversation_id` are kept.
///
/// A history with no safe split point proceeds uncompacted while it still
/// fits; every other compaction failure is returned, and nothing is replaced.
pub(crate) async fn apply_auto_compaction(
    state: &AppState,
    live: &std::sync::Arc<LiveSettings>,
    request: &CreateResponseRequest,
    translated: &mut TranslatedRequest,
    threshold: usize,
) -> Result<AutoCompaction, ErrorResponse> {
    // Counting can take worker slots (media prompts are counted on the
    // worker), so a busy server refuses before any count, with the 503 the
    // generation arms return.
    if !state.can_accept_request() {
        return Err(ErrorResponse::service_unavailable(
            "All slots are busy. Please try again later.",
        ));
    }
    let Some(items) = translated.resolved_items.take() else {
        return Err(ErrorResponse::internal_server_error(
            "automatic compaction: resolved history missing",
        ));
    };
    let context_tokens = state.effective_context_size();
    // Counts render the full history as one request, as the compact route
    // does, so the chain references are cleared; `input` is replaced on
    // every render.
    let mut template = request.clone();
    template.input = ResponseInput::Items(Vec::new());
    template.previous_response_id = None;
    template.conversation = None;
    template.context_management = None;
    // The summary pass is one non-streamed generation.
    template.stream = false;
    let backend = ResponsesCompactionBackend {
        state,
        live,
        template: &template,
        summary_tokens: auto_summary_tokens(context_tokens),
        media: MediaCache::default(),
    };
    let tokens_before = backend
        .count(&items)
        .await
        .map_err(CompactionError::into_error_response)?;
    let mut auto = AutoCompaction {
        tokens_before,
        tokens_after: tokens_before,
        item: None,
    };
    if tokens_before < threshold && tokens_before < context_tokens {
        tracing::debug!(
            tokens_before,
            threshold,
            "Responses automatic compaction not needed"
        );
        return Ok(auto);
    }
    tracing::info!(
        tokens_before,
        threshold,
        context_tokens,
        "Responses automatic compaction triggered"
    );
    let limits = CompactionLimits {
        context_tokens,
        keep_tokens: default_keep_tokens(context_tokens),
    };
    let outcome = match compact_history_counted(&backend, items, limits, tokens_before).await {
        Ok(outcome) => outcome,
        // Upstream proceeds unchanged when nothing can be split; that is
        // only safe while the input leaves room for output.
        Err(CompactionError::NoSplitPoint(reason)) if tokens_before < context_tokens => {
            tracing::info!(
                tokens_before,
                reason,
                "Responses automatic compaction skipped; generating uncompacted"
            );
            return Ok(auto);
        }
        Err(CompactionError::NoSplitPoint(_)) => {
            return Err(context_length_exceeded(format!(
                "input is {tokens_before} tokens and fills the {context_tokens}-token context \
                 window, and it has no completed turn to compact; original context preserved"
            )));
        }
        Err(err) => return Err(err.into_error_response()),
    };

    let summary = outcome.summary.text.trim();
    let item = compaction_item(encode_capsule(summary, &outcome.retained));
    // The next hop resends this capsule; refuse one the server would reject.
    validate_replay_size(
        &[json!({
            "type": "compaction",
            "id": item.id,
            "encrypted_content": item.encrypted_content,
        })],
        state.display_model_id(),
        crate::server::app::main_json_body_limit_bytes(),
    )
    .map_err(CompactionError::into_error_response)?;

    drop(backend);
    let mut compacted = outcome.instructions.clone();
    compacted.extend(summary_message(summary));
    compacted.extend(outcome.retained);
    template.input = ResponseInput::Items(compacted);
    template.stream = request.stream;
    let rebuilt = responses_request_to_chat(&template, None, None)
        .map_err(|err| ErrorResponse::new(err.to_string(), "invalid_request_error"))?;
    translated.chat_request = rebuilt.chat_request;
    translated.canonical_input_items = outcome.instructions;
    auto.tokens_after = outcome.tokens_after;
    auto.item = Some(item);
    Ok(auto)
}

/// Cap `max_tokens` at the context left after the rendered prompt
/// (Blaizzy/mlx-vlm#2458). A text prompt is counted exactly; a media prompt,
/// whose length is known only on the worker, uses the count-path figure.
pub(crate) fn cap_output_tokens(
    state: &AppState,
    auto: &AutoCompaction,
    prepared: &PreparedChatRequest,
    max_tokens: &mut usize,
) -> Result<(), ErrorResponse> {
    let prompt_tokens = if prepared.carries_media() {
        auto.tokens_after
    } else {
        count_text_prompt_tokens(
            state,
            &prepared.prompt,
            prepared.prompt_token_ids.as_deref(),
        )
        .unwrap_or(auto.tokens_after)
    };
    let context = state.effective_context_size();
    let remaining = context.saturating_sub(prompt_tokens);
    if remaining == 0 {
        return Err(context_length_exceeded(format!(
            "input is {prompt_tokens} tokens, which leaves no room for output in the \
             {context}-token context window"
        )));
    }
    if remaining < *max_tokens {
        tracing::debug!(
            requested = *max_tokens,
            remaining,
            "Responses output capped at the remaining context"
        );
        *max_tokens = remaining;
    }
    Ok(())
}

/// Keep a limited reasoning budget below the capped `max_tokens`.
///
/// `resolve_request_budget` validated the budget against the uncapped limit
/// (budget <= max_tokens). After [`cap_output_tokens`] lowers the limit, the
/// same relation is restored with one token kept back, so the forced
/// end-of-thinking still leaves room for visible output. A budget of 0 or an
/// unbounded one is left as is.
pub(crate) fn clamp_reasoning_budget(budget: &mut ReasoningBudgetOverride, max_tokens: usize) {
    let ReasoningBudgetOverride::Explicit(Some(ThinkingBudget::Limited(limit))) = *budget else {
        return;
    };
    let ceiling = max_tokens.saturating_sub(1);
    if (limit.get() as usize) <= ceiling {
        return;
    }
    let clamped = u32::try_from(ceiling)
        .ok()
        .and_then(std::num::NonZeroU32::new)
        .map_or(ThinkingBudget::ImmediateClose, ThinkingBudget::Limited);
    tracing::debug!(
        requested = limit.get(),
        max_tokens,
        "reasoning budget clamped to the capped output limit"
    );
    *budget = ReasoningBudgetOverride::Explicit(Some(clamped));
}

/// Whether a `context_management` reply stopped at its output limit.
pub(crate) fn stopped_at_limit(auto: Option<&AutoCompaction>, finish_reason: &str) -> bool {
    auto.is_some() && finish_reason == "length"
}

/// An incomplete reply has no `completed_at`, and its last message item,
/// the one the limit cut off, is `incomplete` too (OpenAI's shape). Only
/// [`final_status`] sets `incomplete`, so other responses are untouched.
pub(crate) fn mark_incomplete_output(response: &mut ResponseObject) {
    if response.status != ResponseStatus::Incomplete {
        return;
    }
    response.completed_at = None;
    if let Some(ResponseOutputItem::Message(message)) = response
        .output
        .iter_mut()
        .rev()
        .find(|item| matches!(item, ResponseOutputItem::Message(_)))
    {
        message.status = ResponseItemStatus::Incomplete;
    }
}

/// Final `status` and `incomplete_details.reason`. Only a
/// `context_management` request reports a `length` finish as incomplete;
/// every other request keeps `completed`.
pub(crate) fn final_status(
    auto: Option<&AutoCompaction>,
    finish_reason: &str,
) -> (ResponseStatus, Option<String>) {
    if stopped_at_limit(auto, finish_reason) {
        (
            ResponseStatus::Incomplete,
            Some(INCOMPLETE_MAX_OUTPUT_TOKENS.to_string()),
        )
    } else {
        (ResponseStatus::Completed, None)
    }
}

/// Put the compaction item first in `response.output`, unless the streamed
/// items already carry it there.
pub(crate) fn prepend_compaction_item(
    auto: Option<&AutoCompaction>,
    response: &mut ResponseObject,
) {
    let Some(item) = auto.and_then(|auto| auto.item.as_ref()) else {
        return;
    };
    if response
        .output
        .first()
        .is_some_and(|first| first.item_id() == item.id)
    {
        return;
    }
    response
        .output
        .insert(0, ResponseOutputItem::Compaction(item.clone()));
}

/// After `response.in_progress`: the two `mlx.compaction.*` events, then the
/// compaction item at the current `output_index`, which is then advanced so
/// the generated items follow it. Sends nothing when no compaction ran.
pub(crate) fn send_compaction_events(
    sender: &ResponseStreamSender,
    emitter: &mut ResponseStreamEmitter,
    response_id: &str,
    auto: Option<&AutoCompaction>,
) {
    let Some((auto, item)) = auto.and_then(|auto| auto.item.as_ref().map(|item| (auto, item)))
    else {
        return;
    };
    let item = ResponseOutputItem::Compaction(item.clone());
    let events = [
        ResponseStreamEvent::CompactionStarted {
            sequence_number: emitter.next_seq(),
            response_id: response_id.to_string(),
            input_tokens: auto.tokens_before,
        },
        ResponseStreamEvent::CompactionCompleted {
            sequence_number: emitter.next_seq(),
            response_id: response_id.to_string(),
            input_tokens_before: auto.tokens_before,
            input_tokens_after: auto.tokens_after,
        },
        ResponseStreamEvent::OutputItemAdded {
            sequence_number: emitter.next_seq(),
            output_index: emitter.output_index(),
            item: item.clone(),
        },
        ResponseStreamEvent::OutputItemDone {
            sequence_number: emitter.next_seq(),
            output_index: emitter.output_index(),
            item,
        },
    ];
    for event in &events {
        let _ = sender.send_event(event);
    }
    emitter.advance_output_index();
}

#[cfg(test)]
#[path = "responses_auto_compact_tests.rs"]
mod responses_auto_compact_tests;
