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

//! The output items of the MCP tool loop and the events that deliver them
//! (issue #1457): `mcp_list_tools`, `mcp_call`, and the `function_call` of a
//! client-declared tool. The loop that decides which of them to produce is
//! [`super::responses_mcp_loop`].

use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::Value;

use crate::server::mcp::McpClientError;
use crate::server::mcp::request::PlannedServer;
use crate::server::responses_translator::short_uuid;
use crate::server::streaming_responses::CancellationToken;
use crate::server::types::responses_mcp::{
    McpCallItem, McpCallStatus, McpListToolsItem, McpListedTool, mcp_result_text,
};
use crate::server::types::responses_response::{
    ResponseFunctionCallOutput, ResponseItemStatus, ResponseOutputItem,
};
use crate::server::types::responses_stream::ResponseStreamEvent;

use super::responses_mcp_loop::{Refusal, ToolLoop};

impl ToolLoop<'_> {
    /// Deliver one server's `mcp_list_tools` item: the tools the request may
    /// use from it, or why it could not list any.
    pub(super) fn list_tools(&mut self, server: &PlannedServer) {
        let mut item = McpListToolsItem {
            id: format!("mcpl_{}", short_uuid()),
            server_label: server.request.server_label.clone(),
            tools: Vec::new(),
            error: None,
        };
        let item_id = item.id.clone();
        self.events
            .item_added(ResponseOutputItem::McpListTools(item.clone()));
        let in_progress_id = item_id.clone();
        self.events.event(|sequence_number, output_index| {
            ResponseStreamEvent::McpListToolsInProgress {
                sequence_number,
                item_id: in_progress_id,
                output_index,
            }
        });
        match &server.tools {
            Ok(tools) => {
                item.tools = tools
                    .iter()
                    .map(|tool| McpListedTool {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        input_schema: tool.input_schema.clone(),
                        annotations: tool.annotations.clone(),
                    })
                    .collect();
                self.events.event(|sequence_number, output_index| {
                    ResponseStreamEvent::McpListToolsCompleted {
                        sequence_number,
                        item_id,
                        output_index,
                    }
                });
            }
            Err(error) => {
                item.error = Some(error.clone());
                self.events.event(|sequence_number, output_index| {
                    ResponseStreamEvent::McpListToolsFailed {
                        sequence_number,
                        item_id,
                        output_index,
                    }
                });
            }
        }
        self.events
            .item_done(ResponseOutputItem::McpListTools(item));
    }

    /// Deliver one `mcp_call` item: run the call, or report `refusal` in its
    /// place. Returns the item id and the text the model sees as the result.
    pub(super) fn mcp_call(
        &mut self,
        server_label: &str,
        name: &str,
        arguments: &str,
        approval_request_id: Option<String>,
        refusal: Option<Refusal>,
    ) -> (String, String) {
        let arguments = if arguments.trim().is_empty() {
            "{}".to_string()
        } else {
            arguments.to_string()
        };
        let mut item = McpCallItem {
            id: format!("mcp_{}", short_uuid()),
            server_label: server_label.to_string(),
            name: name.to_string(),
            arguments: arguments.clone(),
            output: None,
            error: None,
            approval_request_id,
            status: McpCallStatus::InProgress,
        };
        let item_id = item.id.clone();
        self.events
            .item_added(ResponseOutputItem::McpCall(item.clone()));
        if refusal.is_none() {
            let in_progress_id = item_id.clone();
            self.events.event(|sequence_number, output_index| {
                ResponseStreamEvent::McpCallInProgress {
                    sequence_number,
                    item_id: in_progress_id,
                    output_index,
                }
            });
        }
        let (delta_id, delta) = (item_id.clone(), arguments.clone());
        self.events.event(|sequence_number, output_index| {
            ResponseStreamEvent::McpCallArgumentsDelta {
                sequence_number,
                item_id: delta_id,
                output_index,
                delta,
            }
        });
        let (done_id, done_arguments) = (item_id.clone(), arguments.clone());
        self.events.event(|sequence_number, output_index| {
            ResponseStreamEvent::McpCallArgumentsDone {
                sequence_number,
                item_id: done_id,
                output_index,
                arguments: done_arguments,
            }
        });

        match refusal {
            Some(refusal) => {
                item.error = Some(refusal.error);
                item.status = refusal.status;
            }
            None => {
                self.usage.tool_calls += 1;
                match self.execute(server_label, name, &arguments) {
                    Ok(output) if !output.is_error => {
                        if output.truncated {
                            self.usage.truncated_outputs += 1;
                        }
                        item.output = Some(output.text);
                        item.status = McpCallStatus::Completed;
                    }
                    Ok(output) => {
                        // The tool reported its own failure; its text is the
                        // error the model reads.
                        item.error = Some(if output.text.is_empty() {
                            "the tool reported an error".to_string()
                        } else {
                            output.text
                        });
                        item.status = McpCallStatus::Failed;
                    }
                    Err(error) => {
                        item.error = Some(error);
                        item.status = McpCallStatus::Failed;
                    }
                }
            }
        }
        if item.error.is_some() {
            self.usage.failed_tool_calls += 1;
        }
        let terminal_id = item_id.clone();
        let completed = item.status == McpCallStatus::Completed;
        self.events.event(|sequence_number, output_index| {
            if completed {
                ResponseStreamEvent::McpCallCompleted {
                    sequence_number,
                    item_id: terminal_id,
                    output_index,
                }
            } else {
                ResponseStreamEvent::McpCallFailed {
                    sequence_number,
                    item_id: terminal_id,
                    output_index,
                }
            }
        });
        let text = mcp_result_text(item.output.as_deref(), item.error.as_deref());
        self.events.item_done(ResponseOutputItem::McpCall(item));
        (item_id, text)
    }

    /// Run one `tools/call`. Any failure is the call's own error text.
    fn execute(
        &self,
        server_label: &str,
        name: &str,
        arguments: &str,
    ) -> Result<crate::server::mcp::McpToolOutput, String> {
        let arguments: Value = match serde_json::from_str(arguments) {
            Ok(value @ Value::Object(_)) => value,
            Ok(_) | Err(_) => {
                return Err("the model's arguments are not a JSON object".to_string());
            }
        };
        let runtime = &self.run.setup.runtime;
        let cancelled = self.cancelled;
        let started = std::time::Instant::now();
        let result: Result<_, McpClientError> = self.handle.block_on(async {
            tokio::select! {
                result = runtime.call_tool(server_label, name, &arguments) => result,
                () = wait_cancelled(cancelled) => {
                    Err(McpClientError::Io("the request was cancelled".to_string()))
                }
            }
        });
        // Names, timing and outcome only: arguments and output may hold
        // secrets and stay out of the log.
        tracing::debug!(
            server = %server_label,
            tool = %name,
            elapsed_ms = started.elapsed().as_millis() as u64,
            ok = result.is_ok(),
            "MCP tool call finished"
        );
        result.map_err(|error| error.to_string())
    }

    /// A call to a client-declared function tool, delivered whole.
    pub(super) fn function_call(&mut self, call_id: &str, name: &str, arguments: &str) {
        let mut item = ResponseFunctionCallOutput {
            id: format!("fc_{}", short_uuid()),
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
            status: ResponseItemStatus::InProgress,
        };
        let item_id = item.id.clone();
        self.events
            .item_added(ResponseOutputItem::FunctionCall(item.clone()));
        let (delta_id, delta) = (item_id.clone(), arguments.to_string());
        self.events.event(|sequence_number, output_index| {
            ResponseStreamEvent::FunctionCallArgumentsDelta {
                sequence_number,
                item_id: delta_id,
                output_index,
                delta,
            }
        });
        let done_arguments = arguments.to_string();
        self.events.event(|sequence_number, output_index| {
            ResponseStreamEvent::FunctionCallArgumentsDone {
                sequence_number,
                item_id,
                output_index,
                arguments: done_arguments,
            }
        });
        item.arguments = arguments.to_string();
        item.status = ResponseItemStatus::Completed;
        self.events
            .item_done(ResponseOutputItem::FunctionCall(item));
    }
}

async fn wait_cancelled(cancelled: &CancellationToken) {
    while !cancelled.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
