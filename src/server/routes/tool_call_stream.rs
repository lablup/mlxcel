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

//! Incremental tool-call emission for the streaming Chat Completions and
//! Responses routes (#2396).
//!
//! Both wrap an [`IncrementalToolCallStreamer`]: while tokens arrive they turn
//! its events into wire chunks, and at the end of generation they reconcile
//! the streamed calls with the authoritative `parse_tool_calls` result through
//! [`IncrementalToolCallStreamer::plan_final`]. A call that was streamed keeps
//! its id and receives only the unsent rest of its arguments; every other
//! parsed call is emitted in one shot, exactly as before this change.
//!
//! Used by: routes/chat, routes/responses

use crate::server::responses_translator::short_uuid;
use crate::server::streaming_responses::ResponseStreamEmitter;
use crate::server::tool_calls::incremental::{IncrementalToolCallStreamer, ToolCallStreamEvent};
use crate::server::tool_calls::{self, ParsedToolCall};
use crate::server::types::request::Tool;
use crate::server::types::responses_response::{
    ResponseFunctionCallOutput, ResponseItemStatus, ResponseOutputContent, ResponseOutputItem,
    ResponseOutputMessage, ResponseReasoningOutput, ResponseReasoningPart,
};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::types::stream::ChatCompletionChunk;

/// Chat Completions: tool-call deltas in `choices[0].delta.tool_calls`.
#[derive(Debug)]
pub(crate) struct ChatToolCallStream {
    streamer: IncrementalToolCallStreamer,
    specific_function: Option<String>,
}

impl ChatToolCallStream {
    pub(crate) fn new(
        tools: &[Tool],
        specific_function: Option<&str>,
        primed_open_thinking: bool,
    ) -> Self {
        Self {
            streamer: IncrementalToolCallStreamer::new(
                tools,
                specific_function,
                primed_open_thinking,
            ),
            specific_function: specific_function.map(str::to_string),
        }
    }

    /// Feed one decoded token; appends the tool-call chunks it produced.
    pub(crate) fn feed(
        &mut self,
        token: &str,
        request_id: &str,
        model: &str,
        out: &mut Vec<ChatCompletionChunk>,
    ) {
        for event in self.streamer.feed(token) {
            out.push(match event {
                ToolCallStreamEvent::Start { index, name, id } => {
                    ChatCompletionChunk::tool_call_start(
                        request_id.to_string(),
                        model.to_string(),
                        index,
                        id,
                        name,
                    )
                }
                ToolCallStreamEvent::ArgsDelta { index, delta } => {
                    ChatCompletionChunk::tool_call_arguments(
                        request_id.to_string(),
                        model.to_string(),
                        index,
                        delta,
                    )
                }
            });
        }
    }

    /// The end-of-generation chunks for `parsed` (the authoritative calls).
    ///
    /// The named `tool_choice` filter applies to the one-shot calls exactly as
    /// before; a streamed call already passed it. The concatenated deltas of
    /// an agreeing streamed call equal its final `arguments`.
    pub(crate) fn finish(
        &self,
        parsed: &mut [ParsedToolCall],
        request_id: &str,
        model: &str,
    ) -> Vec<ChatCompletionChunk> {
        let plan = self.streamer.plan_final(parsed, request_id);
        let mut out = Vec::new();
        for entry in plan.calls {
            let call = &parsed[entry.parsed_index];
            if self
                .specific_function
                .as_deref()
                .is_some_and(|name| name != call.name)
            {
                continue;
            }
            if entry.streamed.is_some() {
                if let Some(rest) = entry.remainder.filter(|rest| !rest.is_empty()) {
                    out.push(ChatCompletionChunk::tool_call_arguments(
                        request_id.to_string(),
                        model.to_string(),
                        entry.index,
                        rest,
                    ));
                }
                continue;
            }
            let call_id = call
                .id
                .clone()
                .unwrap_or_else(tool_calls::generate_tool_call_id);
            out.push(ChatCompletionChunk::tool_call_start(
                request_id.to_string(),
                model.to_string(),
                entry.index,
                call_id,
                call.name.clone(),
            ));
            out.push(ChatCompletionChunk::tool_call_arguments(
                request_id.to_string(),
                model.to_string(),
                entry.index,
                call.arguments.clone(),
            ));
        }
        out
    }
}

/// Where a streamed `function_call` item lives in the Responses output.
#[derive(Debug, Clone)]
struct StreamedItem {
    fc_id: String,
    output_index: usize,
}

/// Responses: `function_call` output items with
/// `response.function_call_arguments.delta` events.
#[derive(Debug)]
pub(crate) struct ResponsesToolCallStream {
    streamer: IncrementalToolCallStreamer,
    /// One entry per streamed call, aligned with `streamer.calls()`.
    items: Vec<StreamedItem>,
    /// Text of message items closed because a call started after them.
    closed_message_text: String,
}

impl ResponsesToolCallStream {
    pub(crate) fn new(
        tools: &[Tool],
        specific_function: Option<&str>,
        primed_open_thinking: bool,
    ) -> Self {
        Self {
            streamer: IncrementalToolCallStreamer::new(
                tools,
                specific_function,
                primed_open_thinking,
            ),
            items: Vec::new(),
            closed_message_text: String::new(),
        }
    }

    /// The whole visible text: messages closed by a tool call, then `tail`
    /// (the message still open at the end).
    pub(crate) fn message_text(&self, tail: &str) -> String {
        format!("{}{tail}", self.closed_message_text)
    }

    /// Feed one decoded token; returns the events to send, in order.
    pub(crate) fn feed(
        &mut self,
        token: &str,
        em: &mut ResponseStreamEmitter,
    ) -> Vec<ResponseStreamEvent> {
        let mut out = Vec::new();
        for event in self.streamer.feed(token) {
            match event {
                ToolCallStreamEvent::Start { name, id, .. } => {
                    // A function-call item follows the items before it: close
                    // the open reasoning and message items first, as the
                    // content path closes reasoning.
                    close_reasoning(em, &mut out);
                    if let Some(text) = close_message(em, &mut out) {
                        self.closed_message_text.push_str(&text);
                    }
                    let fc_id = format!("fc_{}", short_uuid());
                    let output_index = em.output_index();
                    em.advance_output_index();
                    out.push(ResponseStreamEvent::OutputItemAdded {
                        sequence_number: em.next_seq(),
                        output_index,
                        item: function_call_item(
                            &fc_id,
                            id,
                            name,
                            String::new(),
                            ResponseItemStatus::InProgress,
                        ),
                    });
                    self.items.push(StreamedItem {
                        fc_id,
                        output_index,
                    });
                }
                ToolCallStreamEvent::ArgsDelta { index, delta } => {
                    let Some(item) = self.item_for(index) else {
                        continue;
                    };
                    out.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                        sequence_number: em.next_seq(),
                        item_id: item.fc_id.clone(),
                        output_index: item.output_index,
                        delta,
                    });
                }
            }
        }
        out
    }

    fn item_for(&self, index: usize) -> Option<&StreamedItem> {
        let pos = self
            .streamer
            .calls()
            .iter()
            .position(|c| c.index == index)?;
        self.items.get(pos)
    }

    /// The end-of-generation events for `parsed`. Every parsed call ends with
    /// `function_call_arguments.done` and `output_item.done` carrying its
    /// parsed arguments; a streamed call the parse does not confirm is closed
    /// as `incomplete` with the text it streamed. Each call's id is written
    /// back into `parsed` so the final response object carries the same ids.
    pub(crate) fn finish(
        &self,
        parsed: &mut [ParsedToolCall],
        em: &mut ResponseStreamEmitter,
        response_id: &str,
    ) -> Vec<ResponseStreamEvent> {
        let plan = self.streamer.plan_final(parsed, response_id);
        let mut out = Vec::new();
        for entry in plan.calls {
            let call = &mut parsed[entry.parsed_index];
            let call_id = call
                .id
                .get_or_insert_with(|| format!("call_{}", short_uuid()))
                .clone();
            let (fc_id, output_index) = match entry.streamed.and_then(|pos| self.items.get(pos)) {
                Some(item) => {
                    if let Some(rest) = entry.remainder.filter(|rest| !rest.is_empty()) {
                        out.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                            sequence_number: em.next_seq(),
                            item_id: item.fc_id.clone(),
                            output_index: item.output_index,
                            delta: rest,
                        });
                    }
                    (item.fc_id.clone(), item.output_index)
                }
                None => {
                    let fc_id = format!("fc_{}", short_uuid());
                    let output_index = em.output_index();
                    em.advance_output_index();
                    out.push(ResponseStreamEvent::OutputItemAdded {
                        sequence_number: em.next_seq(),
                        output_index,
                        item: function_call_item(
                            &fc_id,
                            call_id.clone(),
                            call.name.clone(),
                            call.arguments.clone(),
                            ResponseItemStatus::Completed,
                        ),
                    });
                    out.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                        sequence_number: em.next_seq(),
                        item_id: fc_id.clone(),
                        output_index,
                        delta: call.arguments.clone(),
                    });
                    (fc_id, output_index)
                }
            };
            out.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                sequence_number: em.next_seq(),
                item_id: fc_id.clone(),
                output_index,
                arguments: call.arguments.clone(),
            });
            out.push(ResponseStreamEvent::OutputItemDone {
                sequence_number: em.next_seq(),
                output_index,
                item: function_call_item(
                    &fc_id,
                    call_id,
                    call.name.clone(),
                    call.arguments.clone(),
                    ResponseItemStatus::Completed,
                ),
            });
        }
        for pos in plan.orphans {
            let (Some(item), Some(call)) = (self.items.get(pos), self.streamer.calls().get(pos))
            else {
                continue;
            };
            out.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                sequence_number: em.next_seq(),
                item_id: item.fc_id.clone(),
                output_index: item.output_index,
                arguments: call.sent.clone(),
            });
            out.push(ResponseStreamEvent::OutputItemDone {
                sequence_number: em.next_seq(),
                output_index: item.output_index,
                item: function_call_item(
                    &item.fc_id,
                    call.id.clone(),
                    call.name.clone(),
                    call.sent.clone(),
                    ResponseItemStatus::Incomplete,
                ),
            });
        }
        out
    }
}

fn function_call_item(
    fc_id: &str,
    call_id: String,
    name: String,
    arguments: String,
    status: ResponseItemStatus,
) -> ResponseOutputItem {
    ResponseOutputItem::FunctionCall(ResponseFunctionCallOutput {
        id: fc_id.to_string(),
        call_id,
        name,
        arguments,
        status,
    })
}

/// Close the open reasoning item, as the content path does.
fn close_reasoning(em: &mut ResponseStreamEmitter, out: &mut Vec<ResponseStreamEvent>) {
    let Some((id, text)) = em.close_reasoning() else {
        return;
    };
    out.push(ResponseStreamEvent::ReasoningTextDone {
        sequence_number: em.next_seq(),
        item_id: id.clone(),
        output_index: em.output_index(),
        content_index: 0,
        text: text.clone(),
    });
    out.push(ResponseStreamEvent::OutputItemDone {
        sequence_number: em.next_seq(),
        output_index: em.output_index(),
        item: ResponseOutputItem::Reasoning(ResponseReasoningOutput {
            id,
            status: ResponseItemStatus::Completed,
            content: vec![ResponseReasoningPart::ReasoningText { text }],
        }),
    });
    em.advance_output_index();
}

/// Close the open message item, as the end of the stream does; returns its
/// text.
fn close_message(
    em: &mut ResponseStreamEmitter,
    out: &mut Vec<ResponseStreamEvent>,
) -> Option<String> {
    let (id, text) = em.close_message()?;
    out.push(ResponseStreamEvent::OutputTextDone {
        sequence_number: em.next_seq(),
        item_id: id.clone(),
        output_index: em.output_index(),
        content_index: 0,
        text: text.clone(),
    });
    out.push(ResponseStreamEvent::ContentPartDone {
        sequence_number: em.next_seq(),
        item_id: id.clone(),
        output_index: em.output_index(),
        content_index: 0,
        part: ResponseOutputContent::output_text(text.clone()),
    });
    out.push(ResponseStreamEvent::OutputItemDone {
        sequence_number: em.next_seq(),
        output_index: em.output_index(),
        item: ResponseOutputItem::Message(ResponseOutputMessage::new_assistant(
            id,
            vec![ResponseOutputContent::output_text(text.clone())],
        )),
    });
    em.advance_output_index();
    Some(text)
}

#[cfg(test)]
#[path = "tool_call_stream_route_tests.rs"]
mod tool_call_stream_route_tests;
