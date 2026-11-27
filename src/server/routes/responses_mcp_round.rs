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

//! Event bookkeeping and one model turn of the MCP tool loop (issue #1457).
//!
//! [`LoopEvents`] numbers every event of a multi-turn response and keeps the
//! output items in the order they were delivered, so the final response
//! object lists exactly what the stream carried. [`run_round`] generates one
//! model turn into it: reasoning and message text stream as they are
//! produced, and tool-call markup is held back by the same [`StreamFilter`]
//! the chat routes use, to be parsed once the turn ends.

use crate::server::model_provider::GenerationResult;
use crate::server::responses_translator::short_uuid;
use crate::server::streaming_responses::{CancellationToken, ResponseStreamEmitter};
use crate::server::tool_calls;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::types::responses_response::{
    ResponseItemStatus, ResponseOutputContent, ResponseOutputItem, ResponseOutputMessage,
    ResponseReasoningOutput, ResponseReasoningPart,
};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, LiveSettings};

use super::responses_run::{PreparedGeneration, ResponseEventSink};

/// The event stream of one tool-loop response.
pub(super) struct LoopEvents<'a> {
    sink: &'a dyn ResponseEventSink,
    emitter: ResponseStreamEmitter,
    /// Completed output items, in `output_index` order.
    pub(super) output: Vec<ResponseOutputItem>,
}

impl<'a> LoopEvents<'a> {
    pub(super) fn new(sink: &'a dyn ResponseEventSink) -> Self {
        Self {
            sink,
            emitter: ResponseStreamEmitter::new(),
            output: Vec::new(),
        }
    }

    /// Emit the event `build` makes from the next sequence number and the
    /// current output index.
    pub(super) fn event(&mut self, build: impl FnOnce(u64, usize) -> ResponseStreamEvent) {
        let sequence_number = self.emitter.next_seq();
        let event = build(sequence_number, self.emitter.output_index());
        self.sink.emit(&event);
    }

    /// `response.output_item.added` for the item now being produced.
    pub(super) fn item_added(&mut self, item: ResponseOutputItem) {
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::OutputItemAdded {
                sequence_number,
                output_index,
                item,
            },
        );
    }

    /// `response.output_item.done`: the item is final and the next one takes
    /// the following index.
    pub(super) fn item_done(&mut self, item: ResponseOutputItem) {
        let done = item.clone();
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::OutputItemDone {
                sequence_number,
                output_index,
                item: done,
            },
        );
        self.output.push(item);
        self.emitter.advance_output_index();
    }

    /// All reasoning text delivered so far, across turns.
    pub(super) fn reasoning_text(&self) -> Option<String> {
        self.emitter.completed_reasoning_text()
    }

    fn reasoning_delta(&mut self, text: String) {
        let item_id = match self.emitter.active_reasoning_id.clone() {
            Some(id) => id,
            None => {
                let id = format!("rs_{}", short_uuid());
                self.emitter.open_reasoning(id.clone());
                self.item_added(ResponseOutputItem::Reasoning(ResponseReasoningOutput {
                    id: id.clone(),
                    status: ResponseItemStatus::InProgress,
                    content: vec![],
                }));
                id
            }
        };
        self.emitter.reasoning_text_acc.push_str(&text);
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::ReasoningTextDelta {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                delta: text,
            },
        );
    }

    /// Close the open reasoning item, returning its text.
    fn close_reasoning(&mut self) -> Option<String> {
        let (item_id, text) = self.emitter.close_reasoning()?;
        let (done_id, done_text) = (item_id.clone(), text.clone());
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::ReasoningTextDone {
                sequence_number,
                item_id: done_id,
                output_index,
                content_index: 0,
                text: done_text,
            },
        );
        self.item_done(ResponseOutputItem::Reasoning(ResponseReasoningOutput {
            id: item_id,
            status: ResponseItemStatus::Completed,
            content: vec![ResponseReasoningPart::ReasoningText { text: text.clone() }],
        }));
        Some(text)
    }

    fn content_delta(&mut self, text: String, reasoning: &mut String) {
        if let Some(closed) = self.close_reasoning() {
            reasoning.push_str(&closed);
        }
        let item_id = match self.emitter.active_message_id.clone() {
            Some(id) => id,
            None => {
                let id = format!("msg_{}", short_uuid());
                self.emitter.open_message(id.clone());
                let mut message = ResponseOutputMessage::new_assistant(id.clone(), vec![]);
                message.status = ResponseItemStatus::InProgress;
                self.item_added(ResponseOutputItem::Message(message));
                let part_id = id.clone();
                self.event(
                    |sequence_number, output_index| ResponseStreamEvent::ContentPartAdded {
                        sequence_number,
                        item_id: part_id,
                        output_index,
                        content_index: 0,
                        part: ResponseOutputContent::output_text(String::new()),
                    },
                );
                id
            }
        };
        self.emitter.message_text_acc.push_str(&text);
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::OutputTextDelta {
                sequence_number,
                item_id,
                output_index,
                content_index: 0,
                delta: text,
            },
        );
    }

    /// Close the open message item, returning its text. `truncated` marks a
    /// message the output limit cut off.
    fn close_message(&mut self, truncated: bool) -> String {
        let Some((item_id, text)) = self.emitter.close_message() else {
            return String::new();
        };
        let (text_id, text_done) = (item_id.clone(), text.clone());
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::OutputTextDone {
                sequence_number,
                item_id: text_id,
                output_index,
                content_index: 0,
                text: text_done,
            },
        );
        let (part_id, part_text) = (item_id.clone(), text.clone());
        self.event(
            |sequence_number, output_index| ResponseStreamEvent::ContentPartDone {
                sequence_number,
                item_id: part_id,
                output_index,
                content_index: 0,
                part: ResponseOutputContent::output_text(part_text),
            },
        );
        let mut message = ResponseOutputMessage::new_assistant(
            item_id,
            vec![ResponseOutputContent::output_text(text.clone())],
        );
        if truncated {
            message.status = ResponseItemStatus::Incomplete;
        }
        self.item_done(ResponseOutputItem::Message(message));
        text
    }
}

/// What one model turn produced.
pub(super) struct RoundOutput {
    pub(super) result: GenerationResult,
    /// Everything the model emitted, tool-call markup included.
    pub(super) raw: String,
    /// The visible message text of this turn.
    pub(super) message_text: String,
    /// The reasoning text of this turn.
    pub(super) reasoning_text: Option<String>,
    /// The turn's prompt primed an open reasoning block, so the tool-call
    /// parse starts inside it (#3209).
    pub(super) primed_open_thinking: bool,
}

/// Generate one model turn on the calling (blocking) thread, streaming its
/// reasoning and message items through `events`.
pub(super) fn run_round(
    state: &AppState,
    live: &LiveSettings,
    generation: PreparedGeneration,
    cancelled: CancellationToken,
    events: &mut LoopEvents<'_>,
) -> anyhow::Result<RoundOutput> {
    let PreparedGeneration {
        prepared,
        options,
        queue_reservation,
        primed_open_thinking,
        ..
    } = generation;
    let slot = state.slots.begin(
        &prepared.prompt,
        super::slots::slot_params_json(&options, true),
        Some(options.max_tokens as i64),
    );
    let mut raw = String::new();
    let mut reasoning = String::new();
    let mut filter = if primed_open_thinking {
        StreamFilter::new_primed_open_thinking()
    } else {
        StreamFilter::new()
    }
    .with_answer_wrapper(tool_calls::answer_wrapper_enabled(&state.tokenizer));

    let result = state
        .model_provider
        .generate_streaming_with_logprobs_cancellable_videos_declared_reserved_live_with_prefill(
            prepared.prompt,
            options,
            prepared.image_data,
            prepared.audio_data,
            prepared.videos,
            prepared.media,
            queue_reservation,
            cancelled,
            live,
            |token, _logprobs| {
                slot.on_token(&token);
                raw.push_str(&token);
                let out = filter.feed(&token);
                if let Some(text) = out.reasoning.filter(|text| !text.is_empty()) {
                    events.reasoning_delta(text);
                }
                if let Some(text) = out.content.filter(|text| !text.is_empty()) {
                    events.content_delta(text, &mut reasoning);
                }
            },
            |stats| {
                slot.on_prefill_progress(stats.prompt_tokens, stats.cached_tokens, stats.processed);
            },
        )?;

    let trailing = filter.flush();
    if let Some(text) = trailing.reasoning.filter(|text| !text.is_empty()) {
        events.reasoning_delta(text);
    }
    if let Some(text) = trailing.content.filter(|text| !text.is_empty()) {
        events.content_delta(text, &mut reasoning);
    }
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

    if let Some(closed) = events.close_reasoning() {
        reasoning.push_str(&closed);
    }
    let message_text = events.close_message(result.finish_reason == "length");
    Ok(RoundOutput {
        result,
        raw,
        message_text,
        reasoning_text: Some(reasoning).filter(|text| !text.trim().is_empty()),
        primed_open_thinking,
    })
}
