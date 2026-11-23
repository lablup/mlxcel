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

//! What the `mcp` tool declarations of one `/v1/responses` request resolve
//! to (issue #1457): which configured servers it may use, which of their
//! tools the model sees, and which calls need the caller's approval.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use super::client::McpToolInfo;
use super::runtime::McpRuntime;
use crate::server::types::request::{FunctionDefinition, Tool};
use crate::server::types::responses_mcp::McpToolDefinition;

/// When a tool call needs the caller's approval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalPolicy {
    Always,
    Never,
    /// The object form: the named tools run without approval, every other
    /// tool needs it.
    NeverFor(HashSet<String>),
}

impl ApprovalPolicy {
    pub fn requires_approval(&self, tool: &str) -> bool {
        match self {
            ApprovalPolicy::Always => true,
            ApprovalPolicy::Never => false,
            ApprovalPolicy::NeverFor(tools) => !tools.contains(tool),
        }
    }
}

/// One validated `mcp` tool declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolRequest {
    pub server_label: String,
    /// `None` allows every tool the server lists.
    pub allowed_tools: Option<HashSet<String>>,
    pub approval: ApprovalPolicy,
}

impl McpToolRequest {
    pub fn allows(&self, tool: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.contains(tool))
    }
}

fn tool_names(value: &Value, field: &str) -> Result<HashSet<String>, String> {
    let Some(names) = value.as_array() else {
        return Err(format!("mcp tool '{field}' must be an array of tool names"));
    };
    names
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("mcp tool '{field}' must be an array of tool names"))
        })
        .collect()
}

/// `{"tool_names": [...]}`, the only filter form served.
fn tool_name_filter(value: &Value, field: &str) -> Result<HashSet<String>, String> {
    let Some(filter) = value.as_object() else {
        return Err(format!(
            "mcp tool '{field}' must be an object with 'tool_names'"
        ));
    };
    if let Some(key) = filter.keys().find(|key| key.as_str() != "tool_names") {
        return Err(format!(
            "mcp tool '{field}.{key}' is not supported; only 'tool_names' is"
        ));
    }
    match filter.get("tool_names") {
        Some(names) => tool_names(names, &format!("{field}.tool_names")),
        None => Err(format!("mcp tool '{field}' needs 'tool_names'")),
    }
}

/// Validate one declaration. Remote-server fields are refused by name:
/// only servers the operator configured at startup can be used.
pub fn parse_mcp_tool(definition: &McpToolDefinition) -> Result<McpToolRequest, String> {
    let label = definition.server_label.as_str();
    if definition.server_url.is_some() {
        return Err(format!(
            "mcp tool '{label}': 'server_url' is not supported. This server runs only the MCP \
             servers configured at startup; name one of them with 'server_label'"
        ));
    }
    if definition.connector_id.is_some() {
        return Err(format!(
            "mcp tool '{label}': 'connector_id' is not supported"
        ));
    }
    if definition.headers.is_some() || definition.authorization.is_some() {
        return Err(format!(
            "mcp tool '{label}': 'headers' and 'authorization' apply to remote MCP servers and \
             are not supported; a configured stdio server takes its credentials from the \
             server's MCP configuration"
        ));
    }
    let allowed_tools = match definition.allowed_tools.as_ref() {
        None | Some(Value::Null) => None,
        Some(names @ Value::Array(_)) => Some(tool_names(names, "allowed_tools")?),
        Some(filter) => Some(tool_name_filter(filter, "allowed_tools")?),
    };
    let approval = match definition.require_approval.as_ref() {
        // OpenAI's default: a call needs approval unless the request says
        // otherwise.
        None | Some(Value::Null) => ApprovalPolicy::Always,
        Some(Value::String(mode)) => match mode.as_str() {
            "always" => ApprovalPolicy::Always,
            "never" => ApprovalPolicy::Never,
            other => {
                return Err(format!(
                    "mcp tool '{label}': require_approval '{other}' is not supported; use \
                     'always', 'never', or an object with 'always' / 'never' tool_names"
                ));
            }
        },
        Some(Value::Object(filter)) => {
            if let Some(key) = filter
                .keys()
                .find(|key| !matches!(key.as_str(), "always" | "never"))
            {
                return Err(format!(
                    "mcp tool '{label}': require_approval.{key} is not supported"
                ));
            }
            let mut never = match filter.get("never") {
                Some(never) => tool_name_filter(never, "require_approval.never")?,
                None => HashSet::new(),
            };
            // A tool named on both sides needs approval.
            if let Some(always) = filter.get("always") {
                for tool in tool_name_filter(always, "require_approval.always")? {
                    never.remove(&tool);
                }
            }
            ApprovalPolicy::NeverFor(never)
        }
        Some(_) => {
            return Err(format!(
                "mcp tool '{label}': require_approval must be a string or an object"
            ));
        }
    };
    Ok(McpToolRequest {
        server_label: label.to_string(),
        allowed_tools,
        approval,
    })
}

/// One server a request named, with the tools it may use from it.
#[derive(Debug, Clone)]
pub struct PlannedServer {
    pub request: McpToolRequest,
    /// The allowed tools, or why the server could not list any.
    pub tools: Result<Vec<McpToolInfo>, String>,
}

/// Where a tool name the model calls leads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTool {
    pub server_label: String,
    pub requires_approval: bool,
}

/// The MCP side of one request.
#[derive(Debug, Clone, Default)]
pub struct McpRequestPlan {
    /// Servers in declaration order.
    pub servers: Vec<PlannedServer>,
    /// Tool name to its server and approval rule.
    pub tools: HashMap<String, PlannedTool>,
    /// The same tools as chat function tools, in listing order.
    pub function_tools: Vec<Tool>,
}

impl McpRequestPlan {
    pub fn server(&self, label: &str) -> Option<&McpToolRequest> {
        self.servers
            .iter()
            .map(|server| &server.request)
            .find(|request| request.server_label == label)
    }
}

/// Resolve the declarations against the configured servers.
///
/// A declaration that names no configured server, or names one twice, is an
/// error. So is a tool name that two servers, or a server and a client
/// function tool, would both answer to: the model addresses a tool by name
/// alone, and a silent pick could run the wrong one. A server that cannot
/// list its tools is not an error here; it is reported in its
/// `mcp_list_tools` item and contributes no tools.
pub async fn plan_request(
    runtime: &McpRuntime,
    declarations: &[&McpToolDefinition],
    client_function_names: &HashSet<String>,
) -> Result<McpRequestPlan, String> {
    let mut plan = McpRequestPlan::default();
    for definition in declarations {
        let request = parse_mcp_tool(definition)?;
        let label = request.server_label.clone();
        if !runtime.has_server(&label) {
            return Err(format!(
                "mcp tool: server_label '{label}' is not a configured MCP server (configured: {})",
                runtime.server_names().join(", ")
            ));
        }
        if plan.server(&label).is_some() {
            return Err(format!(
                "mcp tool: server_label '{label}' is declared more than once"
            ));
        }
        let tools = match runtime.list_tools(&label).await {
            Ok(tools) => Ok(tools
                .iter()
                .filter(|tool| request.allows(&tool.name))
                .cloned()
                .collect::<Vec<_>>()),
            Err(error) => Err(error.to_string()),
        };
        for tool in tools.iter().flatten() {
            if client_function_names.contains(&tool.name) {
                return Err(format!(
                    "tool name '{}' is declared as a function tool and is also a tool of MCP \
                     server '{label}'; rename the function tool or exclude the MCP tool with \
                     'allowed_tools'",
                    tool.name
                ));
            }
            if let Some(other) = plan.tools.get(&tool.name) {
                return Err(format!(
                    "tool name '{}' is offered by MCP servers '{}' and '{label}'; restrict one \
                     of them with 'allowed_tools'",
                    tool.name, other.server_label
                ));
            }
            plan.tools.insert(
                tool.name.clone(),
                PlannedTool {
                    server_label: label.clone(),
                    requires_approval: request.approval.requires_approval(&tool.name),
                },
            );
            plan.function_tools.push(Tool {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: Some(tool.input_schema.clone()),
                },
            });
        }
        plan.servers.push(PlannedServer { request, tools });
    }
    Ok(plan)
}

#[cfg(all(test, unix))]
#[path = "request_tests.rs"]
mod tests;
