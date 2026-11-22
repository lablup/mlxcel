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

//! Server-side MCP tool loop support (issue #1457).
//!
//! `mlxcel-server --mcp-tools` starts the stdio MCP servers named by
//! `--mcp-servers-config` / `--mcp-servers-json` and lets a `/v1/responses`
//! request use their tools through the OpenAI built-in `mcp` tool contract.
//! This module owns everything below the HTTP route:
//!
//! - [`config`]: the `mcpServers` document, the bounds, and the startup
//!   exposure gate.
//! - [`client`]: one child process speaking newline-delimited JSON-RPC, read
//!   under byte and time limits.
//! - [`runtime`]: the configured servers, tool discovery, and lazy restart.
//! - [`approvals`]: tool calls waiting for the caller's approval.
//! - [`request`]: what one request's `mcp` tool declarations resolve to.
//!
//! The loop itself lives in `routes::responses_mcp`. The feature is off unless
//! the operator passes `--mcp-tools`: MCP children run with the server's
//! privileges.

pub mod approvals;
pub mod client;
pub mod config;
pub mod request;
pub mod runtime;
#[cfg(all(test, unix))]
pub(crate) mod test_support;

pub use client::{McpClientError, McpToolInfo, McpToolOutput};
pub use config::{
    ListenerScope, McpLimits, McpServerSpec, McpStartupConfig, parse_mcp_servers,
    resolve_mcp_startup, validate_mcp_exposure,
};
pub use runtime::McpRuntime;
