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

//! Stdio MCP client: one child process speaking newline-delimited JSON-RPC
//! 2.0 (issue #1457).
//!
//! The child is untrusted as a peer even though the operator configured it:
//! every line it writes is read under a byte cap, its stderr is drained into
//! a small ring so a chatty process can neither block on a full pipe nor grow
//! server memory, and every exchange runs under the caller's deadline. A
//! client that failed in any way is discarded; the runtime starts a fresh
//! child for the next call.
//!
//! The child is started from an argv, never through a shell, with a cleared
//! environment plus a short allowlist of ordinary variables and the entries
//! the operator configured. The server's own secrets (API keys, access
//! tokens, any `MLXCEL_*` or `LLAMA_*` variable) are therefore not inherited.
//!
//! On Unix the child leads its own process group, and discarding the client
//! kills the whole group (issue #3296). A launcher such as `npx` or `uvx`
//! runs the real server as a grandchild, which killing the launcher alone
//! would leave running.

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::config::McpServerSpec;

/// MCP revision this client announces in `initialize`.
const PROTOCOL_VERSION: &str = "2025-06-18";
/// Bytes of child stderr kept for diagnostics.
const STDERR_TAIL_BYTES: usize = 4 * 1024;
/// Cap on one control message (`initialize`, `tools/list`).
const MAX_CONTROL_MESSAGE_BYTES: usize = 1024 * 1024;
/// Messages a child may interleave before the awaited reply.
const MAX_INTERLEAVED_MESSAGES: usize = 256;
/// `tools/list` pages followed for one server.
const MAX_TOOL_PAGES: usize = 16;
/// Tools kept from one server.
pub(crate) const MAX_TOOLS_PER_SERVER: usize = 256;

/// Variables a child inherits from the server. Everything else is cleared.
const INHERITED_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "TERM",
    "SystemRoot",
    "SYSTEMROOT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
    "PATHEXT",
    "COMSPEC",
];

/// Why an exchange with a child failed. Each one fails a single tool call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum McpClientError {
    #[error("MCP server could not be started: {0}")]
    Spawn(String),
    #[error("MCP server did not answer within {} ms", .0.as_millis())]
    Timeout(Duration),
    #[error(
        "MCP server stayed busy with other calls for {} ms; this call was not sent",
        .0.as_millis()
    )]
    Busy(Duration),
    #[error("MCP server exited or closed its output before answering")]
    Closed,
    #[error("MCP server sent a message larger than the {0}-byte limit")]
    Oversized(usize),
    #[error("MCP server sent an invalid JSON-RPC message: {0}")]
    InvalidJsonRpc(String),
    #[error("MCP server returned JSON-RPC error {code}: {message}")]
    Rpc { code: i64, message: String },
    #[error("MCP server I/O failed: {0}")]
    Io(String),
}

impl McpClientError {
    /// Whether the child can still be used. A JSON-RPC error is a normal
    /// reply; every other failure leaves the stream in an unknown state.
    pub(crate) fn keeps_connection(&self) -> bool {
        matches!(self, McpClientError::Rpc { .. })
    }
}

/// One tool a server advertises.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub annotations: Option<Value>,
}

/// The text of one finished `tools/call`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolOutput {
    pub text: String,
    /// The server flagged the result with `isError`.
    pub is_error: bool,
    /// The text was cut at the output cap.
    pub truncated: bool,
}

/// A running child and the pipes to it.
pub(crate) struct StdioMcpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    line: Vec<u8>,
    /// The child's process group, until it has been killed. Cleared after
    /// the one `killpg` so a later call cannot signal a reused group id.
    #[cfg(unix)]
    process_group: Option<i32>,
}

impl StdioMcpClient {
    /// Start the child and complete the MCP handshake within `deadline`.
    pub(crate) async fn connect(
        spec: &McpServerSpec,
        deadline: Duration,
    ) -> Result<Self, McpClientError> {
        let mut command = Command::new(&spec.command);
        command
            .args(&spec.args)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // A group of its own, so `kill_process_group` reaches grandchildren.
        #[cfg(unix)]
        command.process_group(0);
        for name in INHERITED_ENV {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command.envs(&spec.env);
        // The error reaches API callers in the `mcp_list_tools` item, so it
        // carries the OS reason only; the command stays in the server log.
        let mut child = command.spawn().map_err(|err| {
            tracing::warn!(
                server = %spec.name,
                command = %spec.command,
                error = %err,
                "MCP server process could not be spawned"
            );
            McpClientError::Spawn(err.to_string())
        })?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(McpClientError::Spawn(
                "child pipes were not created".to_string(),
            ));
        };
        // With `process_group(0)` the group id is the child's pid.
        #[cfg(unix)]
        let process_group = child
            .id()
            .and_then(|pid| i32::try_from(pid).ok())
            .filter(|pid| *pid > 0);
        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        tokio::spawn(drain_stderr(stderr, Arc::clone(&stderr_tail)));
        let mut client = Self {
            #[cfg(unix)]
            process_group,
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            stderr_tail,
            line: Vec::new(),
        };
        match tokio::time::timeout(deadline, client.handshake()).await {
            Ok(Ok(())) => Ok(client),
            Ok(Err(err)) => {
                client.shutdown().await;
                Err(err)
            }
            Err(_) => {
                client.shutdown().await;
                Err(McpClientError::Timeout(deadline))
            }
        }
    }

    async fn handshake(&mut self) -> Result<(), McpClientError> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mlxcel", "version": env!("CARGO_PKG_VERSION")},
            }),
            MAX_CONTROL_MESSAGE_BYTES,
        )
        .await?;
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .await
    }

    /// Whether the child has already exited.
    pub(crate) fn has_exited(&mut self) -> bool {
        let exited = !matches!(self.child.try_wait(), Ok(None));
        if exited {
            // The leader is reaped now; kill what is left of its group
            // before the group id can be reused.
            self.kill_process_group();
        }
        exited
    }

    /// Send SIGKILL to the child's whole process group, once. A no-op on
    /// other platforms, where only the child itself is killed.
    fn kill_process_group(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.process_group.take() {
            // SAFETY: `killpg` only sends a signal; `group` is the id of the
            // group this client created and has not signalled before.
            unsafe {
                libc::killpg(group, libc::SIGKILL);
            }
        }
    }

    /// Fetch the advertised tools within `deadline`.
    pub(crate) async fn list_tools(
        &mut self,
        deadline: Duration,
    ) -> Result<Vec<McpToolInfo>, McpClientError> {
        match tokio::time::timeout(deadline, self.list_tools_inner()).await {
            Ok(result) => result,
            Err(_) => Err(McpClientError::Timeout(deadline)),
        }
    }

    async fn list_tools_inner(&mut self) -> Result<Vec<McpToolInfo>, McpClientError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_TOOL_PAGES {
            let params = match cursor.as_ref() {
                Some(cursor) => json!({"cursor": cursor}),
                None => json!({}),
            };
            let result = self
                .request("tools/list", params, MAX_CONTROL_MESSAGE_BYTES)
                .await?;
            let Some(page) = result.get("tools").and_then(Value::as_array) else {
                return Err(McpClientError::InvalidJsonRpc(
                    "tools/list result has no 'tools' array".to_string(),
                ));
            };
            for tool in page {
                let Some(name) = tool.get("name").and_then(Value::as_str) else {
                    return Err(McpClientError::InvalidJsonRpc(
                        "tools/list entry has no string 'name'".to_string(),
                    ));
                };
                if tools.len() >= MAX_TOOLS_PER_SERVER {
                    return Ok(tools);
                }
                tools.push(McpToolInfo {
                    name: name.to_string(),
                    description: tool
                        .get("description")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    input_schema: tool
                        .get("inputSchema")
                        .filter(|schema| schema.is_object())
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                    annotations: tool.get("annotations").filter(|a| a.is_object()).cloned(),
                });
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|cursor| !cursor.is_empty())
                .map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    /// Run one `tools/call` within `deadline`. `max_message_bytes` caps the
    /// reply line as it is read; `max_output_bytes` caps the text kept.
    pub(crate) async fn call_tool(
        &mut self,
        name: &str,
        arguments: &Value,
        deadline: Duration,
        max_message_bytes: usize,
        max_output_bytes: usize,
    ) -> Result<McpToolOutput, McpClientError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = match tokio::time::timeout(
            deadline,
            self.request("tools/call", params, max_message_bytes),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => return Err(McpClientError::Timeout(deadline)),
        };
        Ok(tool_output_from_result(&result, max_output_bytes))
    }

    /// Kill the child and its process group, then reap the child, so a
    /// failed server leaves neither a zombie nor a running grandchild.
    pub(crate) async fn shutdown(mut self) {
        self.kill_process_group();
        let _ = self.child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
    }

    /// The last bytes the child wrote to stderr, for debug logs only.
    pub(crate) fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| String::from_utf8_lossy(&tail).into_owned())
            .unwrap_or_default()
    }

    async fn send(&mut self, message: &Value) -> Result<(), McpClientError> {
        let mut line = serde_json::to_vec(message)
            .map_err(|err| McpClientError::Io(format!("cannot encode request: {err}")))?;
        line.push(b'\n');
        let write = async {
            self.stdin.write_all(&line).await?;
            self.stdin.flush().await
        };
        write.await.map_err(|err| match err.kind() {
            std::io::ErrorKind::BrokenPipe => McpClientError::Closed,
            _ => McpClientError::Io(err.to_string()),
        })
    }

    /// Send one request and read lines until its reply arrives.
    async fn request(
        &mut self,
        method: &str,
        params: Value,
        max_message_bytes: usize,
    ) -> Result<Value, McpClientError> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        for _ in 0..MAX_INTERLEAVED_MESSAGES {
            let mut line = std::mem::take(&mut self.line);
            let read = read_line_bounded(&mut self.stdout, max_message_bytes, &mut line).await;
            let parsed = read.and_then(|()| parse_message(&line));
            self.line = line;
            match parsed? {
                Incoming::Blank | Incoming::Notification => {}
                Incoming::Request(request_id) => {
                    // This client offers no capabilities, so every request a
                    // server sends back (sampling, roots, elicitation) is
                    // answered as unsupported. `ping` is the one exception.
                    self.send(&json!({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "method not supported by this client"},
                    }))
                    .await?;
                }
                Incoming::Ping(request_id) => {
                    self.send(&json!({"jsonrpc": "2.0", "id": request_id, "result": {}}))
                        .await?;
                }
                Incoming::Response {
                    id: reply_id,
                    outcome,
                } => {
                    if reply_id != Some(id) {
                        return Err(McpClientError::InvalidJsonRpc(format!(
                            "reply id does not match request id {id}"
                        )));
                    }
                    return outcome;
                }
            }
        }
        Err(McpClientError::InvalidJsonRpc(format!(
            "no reply within {MAX_INTERLEAVED_MESSAGES} messages"
        )))
    }
}

impl Drop for StdioMcpClient {
    /// A client dropped without `shutdown` (a cancelled request, a server
    /// stopping) still takes its grandchildren down. `kill_on_drop` then
    /// kills and reaps the child itself.
    fn drop(&mut self) {
        self.kill_process_group();
    }
}

enum Incoming {
    Blank,
    Notification,
    Request(Value),
    Ping(Value),
    Response {
        id: Option<u64>,
        outcome: Result<Value, McpClientError>,
    },
}

fn parse_message(line: &[u8]) -> Result<Incoming, McpClientError> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(Incoming::Blank);
    }
    let value: Value = serde_json::from_slice(line)
        .map_err(|err| McpClientError::InvalidJsonRpc(format!("line is not JSON: {err}")))?;
    let Some(object) = value.as_object() else {
        return Err(McpClientError::InvalidJsonRpc(
            "message is not a JSON object".to_string(),
        ));
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(McpClientError::InvalidJsonRpc(
            "message does not declare jsonrpc 2.0".to_string(),
        ));
    }
    if let Some(method) = object.get("method").and_then(Value::as_str) {
        return Ok(match object.get("id").filter(|id| !id.is_null()) {
            Some(id) if method == "ping" => Incoming::Ping(id.clone()),
            Some(id) => Incoming::Request(id.clone()),
            None => Incoming::Notification,
        });
    }
    let id = object.get("id").and_then(Value::as_u64);
    if let Some(error) = object.get("error").filter(|error| !error.is_null()) {
        let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unspecified error")
            .chars()
            .take(512)
            .collect();
        return Ok(Incoming::Response {
            id,
            outcome: Err(McpClientError::Rpc { code, message }),
        });
    }
    match object.get("result") {
        Some(result) => Ok(Incoming::Response {
            id,
            outcome: Ok(result.clone()),
        }),
        None => Err(McpClientError::InvalidJsonRpc(
            "message has neither 'method', 'result' nor 'error'".to_string(),
        )),
    }
}

/// Read one `\n`-terminated line into `out`, refusing to buffer more than
/// `cap` bytes of it.
pub(crate) async fn read_line_bounded<R>(
    reader: &mut R,
    cap: usize,
    out: &mut Vec<u8>,
) -> Result<(), McpClientError>
where
    R: AsyncBufRead + Unpin,
{
    out.clear();
    loop {
        let buffer = reader
            .fill_buf()
            .await
            .map_err(|err| McpClientError::Io(err.to_string()))?;
        if buffer.is_empty() {
            return Err(McpClientError::Closed);
        }
        let (chunk, consumed, done) = match buffer.iter().position(|byte| *byte == b'\n') {
            Some(newline) => (&buffer[..newline], newline + 1, true),
            None => (buffer, buffer.len(), false),
        };
        if out.len().saturating_add(chunk.len()) > cap {
            return Err(McpClientError::Oversized(cap));
        }
        out.extend_from_slice(chunk);
        reader.consume(consumed);
        if done {
            return Ok(());
        }
    }
}

/// Keep the last [`STDERR_TAIL_BYTES`] of a child's stderr and discard the
/// rest, until the pipe closes.
async fn drain_stderr(mut stderr: tokio::process::ChildStderr, tail: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 2048];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(read) => {
                if let Ok(mut tail) = tail.lock() {
                    tail.extend_from_slice(&chunk[..read]);
                    if tail.len() > STDERR_TAIL_BYTES {
                        let excess = tail.len() - STDERR_TAIL_BYTES;
                        tail.drain(..excess);
                    }
                }
            }
        }
    }
}

/// Flatten a `tools/call` result into the text handed to the model. Text
/// parts are joined; other part kinds are named, not interpreted.
pub(crate) fn tool_output_from_result(result: &Value, max_output_bytes: usize) -> McpToolOutput {
    let mut text = String::new();
    if let Some(parts) = result.get("content").and_then(Value::as_array) {
        for part in parts {
            let piece = match part.get("type").and_then(Value::as_str) {
                Some("text") => part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                Some(kind) => format!("[{kind} content omitted]"),
                None => continue,
            };
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&piece);
        }
    }
    if text.is_empty()
        && let Some(structured) = result.get("structuredContent").filter(|v| !v.is_null())
    {
        text = structured.to_string();
    }
    let is_error = result.get("isError").and_then(Value::as_bool) == Some(true);
    let truncated = text.len() > max_output_bytes;
    if truncated {
        let mut end = max_output_bytes;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(&format!("\n[output truncated at {max_output_bytes} bytes]"));
    }
    McpToolOutput {
        text,
        is_error,
        truncated,
    }
}

// The fixture server is a Python script started as `python3`.
#[cfg(all(test, unix))]
#[path = "client_tests.rs"]
mod tests;
