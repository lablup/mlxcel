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

//! The set of configured MCP servers and the calls made to them (issue
//! #1457).
//!
//! Each server is one child process behind an async mutex, so calls to the
//! same server run one at a time, matching the sequential stdio transport.
//! Tools are discovered at startup. A server whose child failed (crash, hang,
//! oversized or malformed message) is discarded and started again by the next
//! call that needs it; a failed tool call is never retried, because a tool may
//! not be idempotent.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use super::approvals::PendingApprovals;
use super::client::{McpClientError, McpToolInfo, McpToolOutput, StdioMcpClient};
use super::config::{McpLimits, McpServerSpec, McpStartupConfig};

struct McpServer {
    spec: McpServerSpec,
    /// The child, behind the lock that serializes exchanges with it.
    client: tokio::sync::Mutex<Option<StdioMcpClient>>,
    /// The discovered tools, readable without waiting for a call in flight:
    /// every request that names the server reads them while planning.
    tools: std::sync::RwLock<Option<Arc<Vec<McpToolInfo>>>>,
}

impl McpServer {
    fn cached_tools(&self) -> Option<Arc<Vec<McpToolInfo>>> {
        self.tools.read().ok().and_then(|tools| tools.clone())
    }
}

/// Shared handle to the MCP servers of one `mlxcel-server` process.
pub struct McpRuntime {
    servers: BTreeMap<String, McpServer>,
    limits: McpLimits,
    approvals: PendingApprovals,
    /// Tool-loop responses running now (see [`Self::admit_run`]).
    active_runs: AtomicUsize,
}

/// One admitted tool-loop response. Dropping it frees the place.
pub(crate) struct McpRunPermit(Arc<McpRuntime>);

impl Drop for McpRunPermit {
    fn drop(&mut self) {
        self.0.active_runs.fetch_sub(1, Ordering::AcqRel);
    }
}

impl McpRuntime {
    /// Start every configured server and discover its tools.
    ///
    /// A server that fails to start or to list its tools does not stop the
    /// others or the process: it is logged, reported as unavailable to the
    /// requests that name it, and tried again by the next such request.
    ///
    /// Used by: `start_server`.
    pub async fn start(config: McpStartupConfig) -> Arc<Self> {
        let runtime = Arc::new(Self::new(config));
        for (name, server) in &runtime.servers {
            match runtime.list_tools(name).await {
                Ok(tools) => tracing::info!(
                    server = %name,
                    command = %server.spec.command,
                    tools = tools.len(),
                    "MCP server ready"
                ),
                Err(error) => tracing::warn!(
                    server = %name,
                    command = %server.spec.command,
                    %error,
                    "MCP server is unavailable; it will be started again on the next request that names it"
                ),
            }
        }
        runtime
    }

    /// Build the runtime without starting any child.
    pub fn new(config: McpStartupConfig) -> Self {
        let servers = config
            .servers
            .into_iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    McpServer {
                        spec,
                        client: tokio::sync::Mutex::new(None),
                        tools: std::sync::RwLock::new(None),
                    },
                )
            })
            .collect();
        Self {
            servers,
            limits: config.limits,
            approvals: PendingApprovals::default(),
            active_runs: AtomicUsize::new(0),
        }
    }

    pub fn limits(&self) -> McpLimits {
        self.limits
    }

    pub(crate) fn approvals(&self) -> &PendingApprovals {
        &self.approvals
    }

    /// Admit one tool-loop response, or `None` when `max_runs` already run.
    ///
    /// A tool-loop response holds a blocking-pool thread for its whole life,
    /// including the time it waits on tool calls, and that time is not
    /// counted by the model queue. Without this cap enough concurrent
    /// requests against a slow tool would exhaust the blocking pool every
    /// other route depends on.
    pub(crate) fn admit_run(self: &Arc<Self>, max_runs: usize) -> Option<McpRunPermit> {
        let max_runs = max_runs.max(1);
        self.active_runs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |running| {
                (running < max_runs).then_some(running + 1)
            })
            .ok()?;
        Some(McpRunPermit(Arc::clone(self)))
    }

    /// Wait for the server's child, at most one tool timeout. Calls to one
    /// server run one at a time, so without this bound a queue of callers
    /// behind a slow tool would wait without limit.
    async fn lock_client<'a>(
        &self,
        server: &'a McpServer,
    ) -> Result<tokio::sync::MutexGuard<'a, Option<StdioMcpClient>>, McpClientError> {
        tokio::time::timeout(self.limits.tool_timeout, server.client.lock())
            .await
            .map_err(|_| McpClientError::Busy(self.limits.tool_timeout))
    }

    pub fn has_server(&self, label: &str) -> bool {
        self.servers.contains_key(label)
    }

    /// Configured server names, for diagnostics.
    pub fn server_names(&self) -> Vec<&str> {
        self.servers.keys().map(String::as_str).collect()
    }

    /// The tools of `label`: the startup discovery when it succeeded,
    /// otherwise a fresh attempt.
    pub async fn list_tools(&self, label: &str) -> Result<Arc<Vec<McpToolInfo>>, McpClientError> {
        let server = self.server(label)?;
        if let Some(tools) = server.cached_tools() {
            return Ok(tools);
        }
        let mut slot = self.lock_client(server).await?;
        // Another request may have finished the discovery while this one
        // waited for the child.
        if let Some(tools) = server.cached_tools() {
            return Ok(tools);
        }
        let mut client = self.connected(server, &mut slot).await?;
        match client.list_tools(self.limits.tool_timeout).await {
            Ok(tools) => {
                let tools = Arc::new(tools);
                if let Ok(mut cached) = server.tools.write() {
                    *cached = Some(Arc::clone(&tools));
                }
                *slot = Some(client);
                Ok(tools)
            }
            Err(error) => {
                discard(label, client, &error).await;
                Err(error)
            }
        }
    }

    /// Run one tool call under the configured timeout and output caps.
    pub async fn call_tool(
        &self,
        label: &str,
        name: &str,
        arguments: &Value,
    ) -> Result<McpToolOutput, McpClientError> {
        let server = self.server(label)?;
        let mut slot = self.lock_client(server).await?;
        let mut client = self.connected(server, &mut slot).await?;
        let result = client
            .call_tool(
                name,
                arguments,
                self.limits.tool_timeout,
                self.limits.max_message_bytes(),
                self.limits.max_output_bytes,
            )
            .await;
        match &result {
            Err(error) if !error.keeps_connection() => discard(label, client, error).await,
            _ => *slot = Some(client),
        }
        result
    }

    fn server(&self, label: &str) -> Result<&McpServer, McpClientError> {
        self.servers
            .get(label)
            .ok_or_else(|| McpClientError::Spawn(format!("no MCP server named '{label}'")))
    }

    /// Take the server's live client, or start a new child when there is
    /// none or the previous one has exited.
    async fn connected(
        &self,
        server: &McpServer,
        slot: &mut Option<StdioMcpClient>,
    ) -> Result<StdioMcpClient, McpClientError> {
        if let Some(mut client) = slot.take() {
            if !client.has_exited() {
                return Ok(client);
            }
            tracing::debug!(server = %server.spec.name, "MCP server exited; starting it again");
            client.shutdown().await;
        }
        StdioMcpClient::connect(&server.spec, self.limits.tool_timeout).await
    }
}

/// Kill a client whose stream can no longer be trusted. The child's stderr
/// goes to the debug log only: it is untrusted and may carry secrets.
async fn discard(label: &str, client: StdioMcpClient, error: &McpClientError) {
    tracing::debug!(
        server = %label,
        %error,
        stderr_tail = %client.stderr_tail(),
        "discarding MCP server process after a failed exchange"
    );
    client.shutdown().await;
}

#[cfg(all(test, unix))]
#[path = "runtime_tests.rs"]
mod tests;
