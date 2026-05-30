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

//! Streaming events for the OpenAI Responses API.
//!
//! Wire shape:
//!
//! ```text
//! event: response.output_text.delta
//! data: {"type":"response.output_text.delta","sequence_number":N, ...}
//!
//! ```
//!
//! The `event:` field equals the inner `type` discriminator. Every event
//! carries a monotonically increasing `sequence_number` (per response) so
//! clients can reorder or detect gaps.
//!
//! Phase 1 emits the following events:
//! - `response.created`, `response.in_progress`, `response.completed`,
//!   `response.failed`, `response.incomplete`, `response.error`
//! - `response.output_item.added`, `response.output_item.done`
//! - `response.content_part.added`, `response.content_part.done`
//! - `response.output_text.delta`, `response.output_text.done`
//! - `response.function_call_arguments.delta`,
//!   `response.function_call_arguments.done`
//! - `response.reasoning_text.delta`, `response.reasoning_text.done`
//!
//! Reserved-for-Phase-2/3 events (audio, refusal, MCP, web/file search,
//! code interpreter, image gen, custom tool) are not emitted.

use serde::Serialize;

use super::responses_response::{ResponseObject, ResponseOutputContent, ResponseOutputItem};

/// Top-level SSE event. The `type` tag round-trips as both the SSE
/// `event:` header (via [`Self::event_name`]) and the payload's
/// `"type"` field.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.incomplete")]
    Incomplete {
        sequence_number: u64,
        response: ResponseObject,
    },
    #[serde(rename = "response.error")]
    Error {
        sequence_number: u64,
        code: String,
        message: String,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: ResponseOutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: ResponseOutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponseOutputContent,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponseOutputContent,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
}

impl ResponseStreamEvent {
    /// SSE `event:` header value.
    pub fn event_name(&self) -> &'static str {
        match self {
            ResponseStreamEvent::Created { .. } => "response.created",
            ResponseStreamEvent::InProgress { .. } => "response.in_progress",
            ResponseStreamEvent::Completed { .. } => "response.completed",
            ResponseStreamEvent::Failed { .. } => "response.failed",
            ResponseStreamEvent::Incomplete { .. } => "response.incomplete",
            ResponseStreamEvent::Error { .. } => "response.error",
            ResponseStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
            ResponseStreamEvent::OutputItemDone { .. } => "response.output_item.done",
            ResponseStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
            ResponseStreamEvent::ContentPartDone { .. } => "response.content_part.done",
            ResponseStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
            ResponseStreamEvent::OutputTextDone { .. } => "response.output_text.done",
            ResponseStreamEvent::FunctionCallArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            ResponseStreamEvent::FunctionCallArgumentsDone { .. } => {
                "response.function_call_arguments.done"
            }
            ResponseStreamEvent::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            ResponseStreamEvent::ReasoningTextDone { .. } => "response.reasoning_text.done",
        }
    }

    /// Monotonic counter on the event.
    pub fn sequence_number(&self) -> u64 {
        match self {
            ResponseStreamEvent::Created {
                sequence_number, ..
            }
            | ResponseStreamEvent::InProgress {
                sequence_number, ..
            }
            | ResponseStreamEvent::Completed {
                sequence_number, ..
            }
            | ResponseStreamEvent::Failed {
                sequence_number, ..
            }
            | ResponseStreamEvent::Incomplete {
                sequence_number, ..
            }
            | ResponseStreamEvent::Error {
                sequence_number, ..
            }
            | ResponseStreamEvent::OutputItemAdded {
                sequence_number, ..
            }
            | ResponseStreamEvent::OutputItemDone {
                sequence_number, ..
            }
            | ResponseStreamEvent::ContentPartAdded {
                sequence_number, ..
            }
            | ResponseStreamEvent::ContentPartDone {
                sequence_number, ..
            }
            | ResponseStreamEvent::OutputTextDelta {
                sequence_number, ..
            }
            | ResponseStreamEvent::OutputTextDone {
                sequence_number, ..
            }
            | ResponseStreamEvent::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | ResponseStreamEvent::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | ResponseStreamEvent::ReasoningTextDelta {
                sequence_number, ..
            }
            | ResponseStreamEvent::ReasoningTextDone {
                sequence_number, ..
            } => *sequence_number,
        }
    }
}

/// Monotonic counter helper for emitting events. Holds the next
/// sequence number to use.
#[derive(Debug, Default)]
pub struct SequenceCounter {
    next: u64,
}

impl SequenceCounter {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn next(&mut self) -> u64 {
        let n = self.next;
        self.next += 1;
        n
    }

    pub fn current(&self) -> u64 {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::responses_response::{ResponseStatus, ResponseUsage};

    fn empty_response() -> ResponseObject {
        ResponseObject {
            id: "resp_1".to_string(),
            object: "response".to_string(),
            created_at: 0.0,
            completed_at: None,
            status: ResponseStatus::InProgress,
            model: "m".to_string(),
            output: vec![],
            output_text: String::new(),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
                input_tokens_details: None,
                output_tokens_details: None,
            },
            error: None,
            incomplete_details: None,
            instructions: None,
            tools: None,
            tool_choice: None,
            text: None,
            reasoning: None,
            metadata: None,
            temperature: None,
            top_p: None,
            parallel_tool_calls: None,
            truncation: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            previous_response_id: None,
            conversation: None,
            prompt_cache_key: None,
            service_tier: None,
            user: None,
            store: None,
        }
    }

    #[test]
    fn created_event_serializes_with_type_field() {
        let ev = ResponseStreamEvent::Created {
            sequence_number: 0,
            response: empty_response(),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "response.created");
        assert_eq!(v["sequence_number"], 0);
        assert_eq!(ev.event_name(), "response.created");
    }

    #[test]
    fn output_text_delta_carries_required_fields() {
        let ev = ResponseStreamEvent::OutputTextDelta {
            sequence_number: 5,
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "hi".to_string(),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "response.output_text.delta");
        assert_eq!(v["item_id"], "msg_1");
        assert_eq!(v["delta"], "hi");
    }

    #[test]
    fn sequence_counter_is_monotonic() {
        let mut c = SequenceCounter::new();
        assert_eq!(c.next(), 0);
        assert_eq!(c.next(), 1);
        assert_eq!(c.next(), 2);
        assert_eq!(c.current(), 3);
    }

    #[test]
    fn function_call_arguments_done_serializes() {
        let ev = ResponseStreamEvent::FunctionCallArgumentsDone {
            sequence_number: 10,
            item_id: "fc_1".to_string(),
            output_index: 1,
            arguments: r#"{"x":1}"#.to_string(),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"], "response.function_call_arguments.done");
        assert_eq!(v["arguments"], r#"{"x":1}"#);
    }
}
