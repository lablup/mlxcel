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

//! Startup configuration of the server-side MCP tool loop (issue #1457).
//!
//! The server set comes from a Cursor-compatible `mcpServers` JSON document,
//! the shape llama-server b10621 accepts for `--mcp-servers-config` and
//! `--mcp-servers-json`. Only the stdio transport is served: an entry names a
//! `command` with its `args` and `env`. Parsing is strict, so an entry that
//! asks for a transport this server does not implement fails startup instead
//! of being silently dropped.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use crate::cli::ui_compat_args::UiCompatArgs;

/// Default cap on tool rounds in one response.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 8;
/// Default per-call timeout, in seconds.
pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 30;
/// Default cap on the text one tool call may return, in bytes.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Upper bound on configured servers: each one is a child process.
pub const MAX_SERVERS: usize = 32;

/// One stdio MCP server: the argv to spawn and the environment it receives
/// on top of the minimal inherited set (see [`super::client`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// Bounds applied to every response that runs MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpLimits {
    /// Model turns that may be followed by tool execution in one response.
    pub max_tool_rounds: usize,
    /// Wall-clock limit of one `tools/call`, and of one server start.
    pub tool_timeout: Duration,
    /// Cap on the text of one tool result handed to the model.
    pub max_output_bytes: usize,
}

impl Default for McpLimits {
    fn default() -> Self {
        Self {
            max_tool_rounds: DEFAULT_MAX_TOOL_ROUNDS,
            tool_timeout: Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }
}

impl McpLimits {
    /// Cap on one JSON-RPC line read from a child. A tool result is JSON
    /// text, so its wire form is larger than the text it carries; anything
    /// beyond this is refused without being buffered.
    pub fn max_message_bytes(&self) -> usize {
        self.max_output_bytes
            .saturating_mul(4)
            .saturating_add(64 * 1024)
    }
}

/// The resolved feature configuration. Its presence on the startup config
/// is what enables the tool loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStartupConfig {
    pub servers: Vec<McpServerSpec>,
    pub limits: McpLimits,
}

/// Resolve the MCP flags of one command line.
///
/// `Ok(None)` means the feature is off. The server set without `--mcp-tools`
/// is an error, so a `LLAMA_ARG_MCP_SERVERS_*` variable inherited from a
/// llama.cpp deployment never spawns a process on its own.
///
/// Used by: mlxcel serve, mlxcel-server.
pub fn resolve_mcp_startup(args: &UiCompatArgs) -> Result<Option<McpStartupConfig>, String> {
    if !args.mcp_tools {
        // Refuses a server set that arrives without the opt-in.
        args.ensure_supported()?;
        return Ok(None);
    }
    let document = match (&args.mcp_servers_config, &args.mcp_servers_json) {
        (Some(_), Some(_)) => {
            return Err(
                "--mcp-servers-config and --mcp-servers-json both name an MCP server set; \
                 pass exactly one of them."
                    .to_string(),
            );
        }
        (Some(path), None) => std::fs::read_to_string(path)
            .map_err(|err| format!("--mcp-servers-config {path}: cannot read the file: {err}"))?,
        (None, Some(json)) => json.clone(),
        (None, None) => {
            return Err(
                "--mcp-tools needs an MCP server set: pass --mcp-servers-config <PATH> or \
                 --mcp-servers-json <JSON> with a Cursor-compatible {\"mcpServers\": {...}} \
                 document."
                    .to_string(),
            );
        }
    };
    let servers = parse_mcp_servers(&document)?;
    if args.mcp_max_tool_rounds == 0 {
        return Err("--mcp-max-tool-rounds must be at least 1.".to_string());
    }
    if args.mcp_tool_timeout == 0 {
        return Err("--mcp-tool-timeout must be at least 1 second.".to_string());
    }
    if args.mcp_max_output_bytes < 64 {
        return Err("--mcp-max-output-bytes must be at least 64.".to_string());
    }
    Ok(Some(McpStartupConfig {
        servers,
        limits: McpLimits {
            max_tool_rounds: args.mcp_max_tool_rounds,
            tool_timeout: Duration::from_secs(args.mcp_tool_timeout),
            max_output_bytes: args.mcp_max_output_bytes,
        },
    }))
}

/// Parse a `{"mcpServers": {"<name>": {"command", "args", "env"}}}` document.
pub fn parse_mcp_servers(document: &str) -> Result<Vec<McpServerSpec>, String> {
    let root: Value = serde_json::from_str(document)
        .map_err(|err| format!("MCP server configuration is not valid JSON: {err}"))?;
    let Some(root) = root.as_object() else {
        return Err("MCP server configuration must be a JSON object.".to_string());
    };
    if let Some(key) = root.keys().find(|key| key.as_str() != "mcpServers") {
        return Err(format!(
            "MCP server configuration has an unknown top-level key '{key}'; only 'mcpServers' \
             is accepted."
        ));
    }
    let Some(servers) = root.get("mcpServers").and_then(Value::as_object) else {
        return Err(
            "MCP server configuration must contain an 'mcpServers' object keyed by server name."
                .to_string(),
        );
    };
    if servers.is_empty() {
        return Err("MCP server configuration lists no servers under 'mcpServers'.".to_string());
    }
    if servers.len() > MAX_SERVERS {
        return Err(format!(
            "MCP server configuration lists {} servers; at most {MAX_SERVERS} are accepted.",
            servers.len()
        ));
    }
    servers
        .iter()
        .map(|(name, entry)| parse_server(name, entry))
        .collect()
}

fn parse_server(name: &str, entry: &Value) -> Result<McpServerSpec, String> {
    if !valid_server_name(name) {
        return Err(format!(
            "MCP server name {name:?} is invalid: use 1 to 64 characters from A-Z, a-z, 0-9, \
             '_', '-' and '.'."
        ));
    }
    let Some(entry) = entry.as_object() else {
        return Err(format!("MCP server '{name}' must be a JSON object."));
    };
    for key in entry.keys() {
        match key.as_str() {
            "command" | "args" | "env" | "type" => {}
            "url" | "serverUrl" | "headers" | "transport" => {
                return Err(format!(
                    "MCP server '{name}' sets '{key}': only stdio servers (a 'command' with \
                     'args' and 'env') are supported; HTTP and SSE transports are not."
                ));
            }
            other => {
                return Err(format!(
                    "MCP server '{name}' has an unknown key '{other}'; accepted keys are \
                     'command', 'args', 'env' and 'type'."
                ));
            }
        }
    }
    if let Some(kind) = entry.get("type")
        && kind.as_str() != Some("stdio")
    {
        return Err(format!(
            "MCP server '{name}' has type {kind}: only \"stdio\" is supported."
        ));
    }
    let command = entry
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|command| !command.is_empty())
        .ok_or_else(|| format!("MCP server '{name}' needs a non-empty string 'command'."))?
        .to_string();
    let args = match entry.get("args") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    format!("MCP server '{name}': every 'args' entry must be a string.")
                })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(format!(
                "MCP server '{name}': 'args' must be an array of strings."
            ));
        }
    };
    let env = match entry.get("env") {
        None | Some(Value::Null) => BTreeMap::new(),
        Some(Value::Object(map)) => map
            .iter()
            .map(|(key, value)| {
                if key.is_empty() || key.contains('=') || key.contains('\0') {
                    return Err(format!(
                        "MCP server '{name}': {key:?} is not a valid environment variable name."
                    ));
                }
                value
                    .as_str()
                    .filter(|value| !value.contains('\0'))
                    .map(|value| (key.clone(), value.to_string()))
                    .ok_or_else(|| {
                        format!("MCP server '{name}': 'env' value of '{key}' must be a string.")
                    })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(format!(
                "MCP server '{name}': 'env' must be an object of string values."
            ));
        }
    };
    Ok(McpServerSpec {
        name: name.to_string(),
        command,
        args,
        env,
    })
}

fn valid_server_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

/// Whether a listener is reachable only from this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenerScope {
    /// Loopback TCP or a Unix domain socket.
    LocalOnly,
    /// Any other TCP bind address.
    Network,
}

/// Refuse the tool loop on a network listener that authenticates nobody.
///
/// MCP children run with the server's privileges, so every caller that can
/// reach `/v1/responses` can make them act. Called before any child is
/// spawned and before the model loads.
pub fn validate_mcp_exposure(
    scope: ListenerScope,
    host: &str,
    has_api_keys: bool,
) -> Result<(), String> {
    if has_api_keys || scope == ListenerScope::LocalOnly {
        return Ok(());
    }
    Err(format!(
        "--mcp-tools lets API callers run MCP child processes with this server's privileges \
         and requires at least one resolved API key when --host {host:?} is not loopback. \
         Configure --api-key or --api-key-file, bind to 127.0.0.1, ::1, or localhost, or use \
         a Unix domain socket"
    ))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
