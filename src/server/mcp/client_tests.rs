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

use std::time::Duration;

use serde_json::json;

use super::*;
use crate::server::mcp::test_support::fake_spec;

const DEADLINE: Duration = Duration::from_secs(10);
const CAP: usize = 64 * 1024;
/// A server-side secret in the `MLXCEL_*` namespace. Assembled from two
/// pieces so the environment-variable documentation check does not read it as
/// a variable mlxcel consumes.
const SERVER_SECRET: &str = concat!("MLXCEL", "_MCP_TEST_SECRET");

#[tokio::test]
async fn bounded_line_reader_refuses_a_line_over_the_cap() {
    let mut short = tokio::io::BufReader::new(&b"abc\ndef\n"[..]);
    let mut line = Vec::new();
    read_line_bounded(&mut short, 8, &mut line).await.unwrap();
    assert_eq!(line, b"abc");
    read_line_bounded(&mut short, 8, &mut line).await.unwrap();
    assert_eq!(line, b"def");
    assert_eq!(
        read_line_bounded(&mut short, 8, &mut line).await,
        Err(McpClientError::Closed)
    );

    let long = vec![b'x'; 10_000];
    // A small reader buffer forces the line across many fills.
    let mut reader = tokio::io::BufReader::with_capacity(64, &long[..]);
    assert_eq!(
        read_line_bounded(&mut reader, 1000, &mut line).await,
        Err(McpClientError::Oversized(1000))
    );
    assert!(line.len() <= 1000, "buffered {} bytes", line.len());
}

#[test]
fn tool_output_joins_text_parts_and_names_other_kinds() {
    let output = tool_output_from_result(
        &json!({"content": [
            {"type": "text", "text": "one"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            {"type": "text", "text": "two"}
        ]}),
        CAP,
    );
    assert_eq!(output.text, "one\n[image content omitted]\ntwo");
    assert!(!output.is_error && !output.truncated);
}

#[test]
fn tool_output_is_truncated_at_the_cap_on_a_char_boundary() {
    let output = tool_output_from_result(
        &json!({"content": [{"type": "text", "text": "가나다라마바사"}]}),
        10,
    );
    assert!(output.truncated);
    assert!(
        output
            .text
            .starts_with("가나다\n[output truncated at 10 bytes]"),
        "{}",
        output.text
    );
}

#[test]
fn tool_output_carries_the_error_flag_and_structured_fallback() {
    let failed = tool_output_from_result(
        &json!({"content": [{"type": "text", "text": "nope"}], "isError": true}),
        CAP,
    );
    assert!(failed.is_error);
    let structured = tool_output_from_result(&json!({"structuredContent": {"sum": 3}}), CAP);
    assert_eq!(structured.text, r#"{"sum":3}"#);
}

#[tokio::test]
async fn client_lists_and_calls_tools_of_the_scripted_server() {
    let mut client = StdioMcpClient::connect(&fake_spec("fake", &[]), DEADLINE)
        .await
        .expect("connects");
    let tools = client.list_tools(DEADLINE).await.expect("lists");
    let add = tools.iter().find(|tool| tool.name == "add").expect("add");
    assert_eq!(add.input_schema["required"], json!(["a", "b"]));
    // A tool without a schema gets an empty object schema.
    let pid = tools.iter().find(|tool| tool.name == "pid").expect("pid");
    assert_eq!(pid.input_schema["type"], "object");

    let output = client
        .call_tool("add", &json!({"a": 19, "b": 23}), DEADLINE, CAP, CAP)
        .await
        .expect("call succeeds");
    assert_eq!(output.text, "42");
    assert!(!output.is_error);

    // An unknown tool is a JSON-RPC error and leaves the connection usable.
    let error = client
        .call_tool("no_such_tool", &json!({}), DEADLINE, CAP, CAP)
        .await
        .expect_err("unknown tool");
    assert!(
        matches!(error, McpClientError::Rpc { code: -32602, .. }),
        "{error:?}"
    );
    assert!(error.keeps_connection());
    let again = client
        .call_tool("add", &json!({"a": 1, "b": 1}), DEADLINE, CAP, CAP)
        .await
        .expect("still usable");
    assert_eq!(again.text, "2");
    client.shutdown().await;
}

#[test]
fn child_environment_holds_the_configured_entries_but_not_server_secrets() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime builds");
    let connected = {
        let _env = crate::test_support::env_lock::env_lock();
        // SAFETY: serialized via the crate-wide env lock held above.
        unsafe {
            std::env::set_var(SERVER_SECRET, "s3cret");
            std::env::set_var("LLAMA_API_KEY", "s3cret");
            std::env::set_var("HF_TOKEN_MCP_TEST", "s3cret");
        }
        // The child copies its environment when it is spawned, inside this
        // block; the guard is not held across an await of the test itself.
        let connected =
            runtime.block_on(StdioMcpClient::connect(&fake_spec("fake", &[]), DEADLINE));
        // SAFETY: as above.
        unsafe {
            std::env::remove_var(SERVER_SECRET);
            std::env::remove_var("LLAMA_API_KEY");
            std::env::remove_var("HF_TOKEN_MCP_TEST");
        }
        connected
    };
    let mut client = connected.expect("connects");
    let names = runtime
        .block_on(client.call_tool("env_names", &json!({}), DEADLINE, CAP, CAP))
        .expect("call succeeds")
        .text;
    let names: Vec<&str> = names.lines().collect();
    assert!(names.contains(&"FAKE_MCP_VISIBLE"), "{names:?}");
    assert!(names.contains(&"PATH"), "{names:?}");
    let leaked: Vec<&&str> = names
        .iter()
        .filter(|name| {
            name.starts_with("MLXCEL_") || name.starts_with("LLAMA_") || name.contains("TOKEN")
        })
        .collect();
    assert!(leaked.is_empty(), "child inherited {leaked:?}");
    runtime.block_on(client.shutdown());
}

#[tokio::test]
async fn a_flood_on_stderr_does_not_block_the_client() {
    let mut client = StdioMcpClient::connect(&fake_spec("fake", &["--stderr-flood"]), DEADLINE)
        .await
        .expect("connects despite a megabyte on stderr");
    let output = client
        .call_tool("add", &json!({"a": 2, "b": 2}), DEADLINE, CAP, CAP)
        .await
        .expect("call succeeds");
    assert_eq!(output.text, "4");
    assert!(client.stderr_tail().len() <= STDERR_TAIL_BYTES);
    client.shutdown().await;
}

#[tokio::test]
async fn startup_failures_are_reported_within_the_deadline() {
    let short = Duration::from_secs(2);
    let started = std::time::Instant::now();
    assert_eq!(
        StdioMcpClient::connect(&fake_spec("fake", &["--startup", "hang"]), short)
            .await
            .err(),
        Some(McpClientError::Timeout(short))
    );
    assert!(started.elapsed() < Duration::from_secs(8));
    assert_eq!(
        StdioMcpClient::connect(&fake_spec("fake", &["--startup", "crash"]), DEADLINE)
            .await
            .err(),
        Some(McpClientError::Closed)
    );
    assert!(matches!(
        StdioMcpClient::connect(&fake_spec("fake", &["--startup", "garbage"]), DEADLINE)
            .await
            .err(),
        Some(McpClientError::InvalidJsonRpc(_))
    ));
    let mut missing = fake_spec("fake", &[]);
    missing.command = "mlxcel-no-such-mcp-command".to_string();
    assert!(matches!(
        StdioMcpClient::connect(&missing, DEADLINE).await.err(),
        Some(McpClientError::Spawn(_))
    ));
}

/// Start the scripted server with a sleeping grandchild and return the
/// connected client with the grandchild's pid (issue #3296).
#[cfg(unix)]
async fn connect_with_grandchild(dir: &std::path::Path) -> (StdioMcpClient, i32) {
    let pid_file = dir.join("grandchild.pid");
    let path = pid_file.to_str().expect("utf-8 temp path");
    let client = StdioMcpClient::connect(&fake_spec("fake", &["--grandchild", path]), DEADLINE)
        .await
        .expect("connects");
    // The server writes the file before it reads `initialize`.
    let pid = std::fs::read_to_string(&pid_file)
        .expect("grandchild pid written")
        .trim()
        .parse::<i32>()
        .expect("pid parses");
    assert!(process_exists(pid), "grandchild {pid} is running");
    (client, pid)
}

#[cfg(unix)]
fn process_exists(pid: i32) -> bool {
    // SAFETY: signal 0 only checks that the pid exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Wait up to five seconds for `pid` to disappear (launchd or init reaps the
/// orphan), then kill it so a failing run leaves nothing behind.
#[cfg(unix)]
async fn grandchild_is_gone(pid: i32) -> bool {
    for _ in 0..100 {
        if !process_exists(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // SAFETY: `pid` is the fixture's own grandchild, still alive.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
    false
}

#[cfg(unix)]
#[tokio::test]
async fn shutdown_kills_the_whole_process_group() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (client, grandchild) = connect_with_grandchild(dir.path()).await;
    client.shutdown().await;
    assert!(
        grandchild_is_gone(grandchild).await,
        "grandchild {grandchild} outlived shutdown"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn dropping_the_client_kills_the_whole_process_group() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (client, grandchild) = connect_with_grandchild(dir.path()).await;
    drop(client);
    assert!(
        grandchild_is_gone(grandchild).await,
        "grandchild {grandchild} outlived the dropped client"
    );
}
