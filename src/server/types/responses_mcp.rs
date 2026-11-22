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

//! Wire types of the Responses API built-in `mcp` tool (issue #1457).
//!
//! The request side is the `{"type": "mcp", ...}` tool declaration; the
//! response side is the three output items a server-side tool loop produces:
//! `mcp_list_tools`, `mcp_call` and `mcp_approval_request`.

use serde::{Deserialize, Serialize};

/// `{"type": "mcp", "server_label": ..., "allowed_tools": ..., "require_approval": ...}`.
///
/// `allowed_tools` and `require_approval` stay raw JSON so a malformed value
/// is a 400 that names the field, not an extractor rejection. The fields that
/// describe a remote server are kept only to be refused by name.
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolDefinition {
    pub server_label: String,
    #[serde(default)]
    pub server_url: Option<String>,
    #[serde(default)]
    pub connector_id: Option<String>,
    #[serde(default)]
    pub server_description: Option<String>,
    #[serde(default)]
    pub headers: Option<serde_json::Value>,
    #[serde(default)]
    pub authorization: Option<String>,
    #[serde(default)]
    pub allowed_tools: Option<serde_json::Value>,
    #[serde(default)]
    pub require_approval: Option<serde_json::Value>,
}

impl McpToolDefinition {
    /// The declaration as echoed in the response's `tools` array.
    pub fn echo(&self) -> serde_json::Value {
        let mut echoed = serde_json::json!({
            "type": "mcp",
            "server_label": self.server_label,
            "require_approval": self
                .require_approval
                .clone()
                .unwrap_or_else(|| serde_json::json!("always")),
        });
        if let Some(allowed) = self.allowed_tools.as_ref() {
            echoed["allowed_tools"] = allowed.clone();
        }
        if let Some(description) = self.server_description.as_ref() {
            echoed["server_description"] = serde_json::json!(description);
        }
        echoed
    }
}

/// `mcp_list_tools`: the tools one server offers to this response.
#[derive(Debug, Clone, Serialize)]
pub struct McpListToolsItem {
    pub id: String,
    pub server_label: String,
    pub tools: Vec<McpListedTool>,
    /// Set when the server could not be started or did not list its tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpListedTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
}

/// `mcp_call`: one tool call the server ran, or refused to run.
#[derive(Debug, Clone, Serialize)]
pub struct McpCallItem {
    pub id: String,
    pub server_label: String,
    pub name: String,
    /// The arguments as a JSON string, as the model produced them.
    pub arguments: String,
    pub output: Option<String>,
    pub error: Option<String>,
    /// The approval request this call resolved, when it needed one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval_request_id: Option<String>,
    pub status: McpCallStatus,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpCallStatus {
    InProgress,
    Completed,
    Failed,
    /// The call was not run because a bound was reached.
    Incomplete,
}

/// `mcp_approval_request`: a call that waits for the caller's approval.
#[derive(Debug, Clone, Serialize)]
pub struct McpApprovalRequestItem {
    pub id: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
}

/// `usage.mcp`, an mlxcel extension: what the tool loop did and the bounds
/// it ran under, so a caller can see which bound ended a response.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct McpUsage {
    /// Model turns whose MCP calls were run or refused.
    pub tool_rounds: usize,
    /// Tool calls sent to an MCP server.
    pub tool_calls: usize,
    /// Calls that ended with an `error` (failed, refused or not run).
    pub failed_tool_calls: usize,
    /// Calls whose output was cut at `max_output_bytes`.
    pub truncated_outputs: usize,
    /// `--mcp-max-tool-rounds`.
    pub max_tool_rounds: usize,
    /// `--mcp-tool-timeout`, in milliseconds.
    pub tool_timeout_ms: u64,
    /// `--mcp-max-output-bytes`.
    pub max_output_bytes: usize,
}

/// The text a tool result contributes to the model's context: the output, or
/// the failure when the call did not produce one.
///
/// Used by: the tool loop and the history replay of stored `mcp_call` items.
pub fn mcp_result_text(output: Option<&str>, error: Option<&str>) -> String {
    match (output, error) {
        (Some(output), None) => output.to_string(),
        (output, Some(error)) => match output.filter(|output| !output.is_empty()) {
            Some(output) => format!("Tool call failed: {error}\n{output}"),
            None => format!("Tool call failed: {error}"),
        },
        (None, None) => String::new(),
    }
}

/// The tool result a denied approval contributes to the model's context.
pub fn mcp_denial_text(reason: Option<&str>) -> String {
    match reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        Some(reason) => {
            format!("The user denied this tool call, so it was not run. Reason: {reason}")
        }
        None => "The user denied this tool call, so it was not run.".to_string(),
    }
}
