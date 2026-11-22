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

use std::time::{Duration, Instant};

use serde_json::json;

use super::*;
use crate::server::mcp::test_support::{fake_config, fake_spec, test_limits};

async fn pid(runtime: &McpRuntime) -> String {
    runtime
        .call_tool("fake", "pid", &json!({}))
        .await
        .expect("pid call succeeds")
        .text
}

#[tokio::test]
async fn start_discovers_tools_and_calls_reuse_one_child() {
    let runtime = McpRuntime::start(fake_config()).await;
    let tools = runtime.list_tools("fake").await.expect("discovered");
    assert!(tools.iter().any(|tool| tool.name == "lookup"));
    let first = pid(&runtime).await;
    assert_eq!(pid(&runtime).await, first, "one child serves both calls");
    let sum = runtime
        .call_tool("fake", "add", &json!({"a": 17, "b": 25}))
        .await
        .expect("add succeeds");
    assert_eq!(sum.text, "42");
}

/// Each misbehaving tool fails its own call with its own error, inside the
/// time bound, and the next call gets a freshly started child.
#[tokio::test]
async fn a_misbehaving_call_fails_alone_and_the_server_restarts_lazily() {
    let runtime = McpRuntime::start(fake_config()).await;
    let limits = test_limits();
    let mut last_pid = pid(&runtime).await;
    for tool in ["hang", "crash", "big", "garbage"] {
        let started = Instant::now();
        let error = runtime
            .call_tool("fake", tool, &json!({}))
            .await
            .expect_err(tool);
        assert!(
            started.elapsed() < limits.tool_timeout + Duration::from_secs(5),
            "{tool} took {:?}",
            started.elapsed()
        );
        match tool {
            "hang" => assert_eq!(error, McpClientError::Timeout(limits.tool_timeout)),
            "crash" => assert_eq!(error, McpClientError::Closed),
            "big" => assert_eq!(error, McpClientError::Oversized(limits.max_message_bytes())),
            _ => assert!(
                matches!(error, McpClientError::InvalidJsonRpc(_)),
                "{error:?}"
            ),
        }
        // The failed child is gone; the next call starts another.
        let next_pid = pid(&runtime).await;
        assert_ne!(
            next_pid, last_pid,
            "{tool} must not leave its child serving"
        );
        last_pid = next_pid;
    }
    // Tools stay listed from the startup discovery across restarts.
    assert!(runtime.list_tools("fake").await.is_ok());
}

#[tokio::test]
async fn a_tool_error_and_a_long_output_are_results_not_failures() {
    let runtime = McpRuntime::start(fake_config()).await;
    let before = pid(&runtime).await;
    let failed = runtime
        .call_tool("fake", "fail", &json!({}))
        .await
        .expect("an isError result is still a result");
    assert!(failed.is_error);
    assert_eq!(failed.text, "the tool could not do that");

    let wordy = runtime
        .call_tool("fake", "wordy", &json!({"size": 6000}))
        .await
        .expect("over the text cap but under the message cap");
    assert!(wordy.truncated);
    assert!(wordy.text.len() < 4096 + 64, "{}", wordy.text.len());
    assert!(wordy.text.ends_with("[output truncated at 4096 bytes]"));
    assert_eq!(
        pid(&runtime).await,
        before,
        "neither result restarts the child"
    );
}

#[tokio::test]
async fn a_child_that_died_between_calls_is_replaced_by_the_next_call() {
    let runtime = McpRuntime::start(fake_config()).await;
    let first = pid(&runtime).await;
    // SAFETY-free kill: the pid is this test's own child.
    let status = std::process::Command::new("kill")
        .args(["-9", &first])
        .status()
        .expect("kill runs");
    assert!(status.success());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let second = pid(&runtime).await;
    assert_ne!(first, second);
}

#[tokio::test]
async fn a_server_that_cannot_start_is_unavailable_and_others_still_serve() {
    let mut config = fake_config();
    config.limits.tool_timeout = Duration::from_secs(2);
    config
        .servers
        .push(fake_spec("broken", &["--startup", "crash"]));
    let mut missing = fake_spec("missing", &[]);
    missing.command = "mlxcel-no-such-mcp-command".to_string();
    config.servers.push(missing);
    let runtime = McpRuntime::start(config).await;
    assert!(runtime.list_tools("fake").await.is_ok());
    assert_eq!(
        runtime.list_tools("broken").await.err(),
        Some(McpClientError::Closed)
    );
    assert!(matches!(
        runtime.list_tools("missing").await.err(),
        Some(McpClientError::Spawn(_))
    ));
    assert!(matches!(
        runtime.call_tool("missing", "add", &json!({})).await.err(),
        Some(McpClientError::Spawn(_))
    ));
    assert!(matches!(
        runtime.call_tool("nope", "add", &json!({})).await.err(),
        Some(McpClientError::Spawn(_))
    ));
    assert_eq!(runtime.server_names(), ["broken", "fake", "missing"]);
}

/// Planning a request reads the tool list, which must not wait behind a tool
/// call another request has in flight on the same server.
#[tokio::test]
async fn listing_tools_does_not_wait_for_a_call_in_flight() {
    let runtime = McpRuntime::start(fake_config()).await;
    let hanging = {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move { runtime.call_tool("fake", "hang", &json!({})).await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    let started = Instant::now();
    assert!(runtime.list_tools("fake").await.is_ok());
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "listing waited {:?}",
        started.elapsed()
    );
    assert_eq!(
        hanging.await.expect("task joins"),
        Err(McpClientError::Timeout(test_limits().tool_timeout))
    );
}
