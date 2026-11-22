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

//! Lower MCP history items to the function-call items the chat renderer
//! already understands (issue #1457).
//!
//! A chained or replayed transcript can hold `mcp_list_tools`, `mcp_call`,
//! `mcp_approval_request` and `mcp_approval_response` items. To the model an
//! MCP tool is a function tool, so its history is a function call followed by
//! its output:
//!
//! - `mcp_call` becomes a `function_call` plus a `function_call_output` with
//!   the stored result. It is history only and is never run again.
//! - `mcp_approval_request` becomes a `function_call` whose output is still
//!   open.
//! - `mcp_approval_response` closes that call: a denial with a fixed refusal
//!   text, an approval with an empty output that the `mcp_call` item resolving
//!   the request fills in. For the request being served that item does not
//!   exist yet; the tool loop runs the call and fills the output itself.
//! - `mcp_list_tools` contributes nothing.
//!
//! Consecutive call items form one run, emitted as all calls and then all
//! outputs, so one model turn that made several calls renders as one
//! assistant turn followed by its tool results.

use std::collections::{HashMap, HashSet};

use crate::server::types::responses_mcp::{mcp_denial_text, mcp_result_text};
use crate::server::types::responses_request::{ResponseInputItem, ResponseToolOutput};

/// Whether `items` holds anything [`lower_mcp_items`] would rewrite.
pub(crate) fn has_mcp_items(items: &[ResponseInputItem]) -> bool {
    items.iter().any(|item| {
        matches!(
            item,
            ResponseInputItem::McpListTools { .. }
                | ResponseInputItem::McpCall { .. }
                | ResponseInputItem::McpApprovalRequest { .. }
                | ResponseInputItem::McpApprovalResponse { .. }
        )
    })
}

/// Server labels whose tools a transcript already lists.
pub(crate) fn listed_mcp_servers(items: &[ResponseInputItem]) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();
    for item in items {
        if let ResponseInputItem::McpListTools { server_label } = item
            && !labels.contains(server_label)
        {
            labels.push(server_label.clone());
        }
    }
    labels
}

fn function_call(call_id: &str, name: &str, arguments: &str) -> ResponseInputItem {
    ResponseInputItem::FunctionCall {
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments: arguments.to_string(),
    }
}

fn function_output(call_id: &str, output: String) -> ResponseInputItem {
    ResponseInputItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: ResponseToolOutput::Text(output),
    }
}

#[derive(Default)]
struct Lowering {
    out: Vec<ResponseInputItem>,
    run_calls: Vec<ResponseInputItem>,
    run_outputs: Vec<ResponseInputItem>,
    /// Approval requests rendered as a call whose output is still open.
    open_requests: HashSet<String>,
    /// Approved requests: index in `out` of the output waiting to be filled.
    approved_slots: HashMap<String, usize>,
}

impl Lowering {
    fn flush_run(&mut self) {
        self.out.append(&mut self.run_calls);
        self.out.append(&mut self.run_outputs);
    }

    fn push(&mut self, item: ResponseInputItem) {
        match item {
            ResponseInputItem::FunctionCall { .. } => self.run_calls.push(item),
            ResponseInputItem::McpListTools { .. } => {}
            ResponseInputItem::McpCall {
                id,
                name,
                arguments,
                output,
                error,
                approval_request_id,
                ..
            } => {
                let text = mcp_result_text(output.as_deref(), error.as_deref());
                let resolved = approval_request_id.filter(|approval_id| {
                    self.approved_slots.contains_key(approval_id)
                        || self.open_requests.contains(approval_id)
                });
                match resolved {
                    Some(approval_id) => {
                        self.flush_run();
                        if let Some(slot) = self.approved_slots.remove(&approval_id) {
                            self.out[slot] = function_output(&approval_id, text);
                        } else {
                            self.open_requests.remove(&approval_id);
                            self.out.push(function_output(&approval_id, text));
                        }
                    }
                    None => {
                        self.run_calls.push(function_call(&id, &name, &arguments));
                        self.run_outputs.push(function_output(&id, text));
                    }
                }
            }
            ResponseInputItem::McpApprovalRequest {
                id,
                name,
                arguments,
                ..
            } => {
                self.run_calls.push(function_call(&id, &name, &arguments));
                self.open_requests.insert(id);
            }
            ResponseInputItem::McpApprovalResponse {
                approval_request_id,
                approve,
                reason,
            } => {
                self.flush_run();
                // An answer to a request this transcript does not hold has
                // nothing to close.
                if !self.open_requests.remove(&approval_request_id) {
                    return;
                }
                if approve {
                    self.out
                        .push(function_output(&approval_request_id, String::new()));
                    self.approved_slots
                        .insert(approval_request_id, self.out.len() - 1);
                } else {
                    self.out.push(function_output(
                        &approval_request_id,
                        mcp_denial_text(reason.as_deref()),
                    ));
                }
            }
            other => {
                self.flush_run();
                self.out.push(other);
            }
        }
    }
}

/// Rewrite every MCP item of `items` as described in the module docs. A
/// transcript without MCP items is returned unchanged.
///
/// Used by: the Responses request translator.
pub(crate) fn lower_mcp_items(items: Vec<ResponseInputItem>) -> Vec<ResponseInputItem> {
    if !has_mcp_items(&items) {
        return items;
    }
    let mut lowering = Lowering::default();
    for item in items {
        lowering.push(item);
    }
    lowering.flush_run();
    lowering.out
}

#[cfg(test)]
#[path = "responses_mcp_lowering_tests.rs"]
mod tests;
