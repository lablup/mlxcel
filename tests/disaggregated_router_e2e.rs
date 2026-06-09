// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Three-process disaggregated router E2E test (#126 B3b2b).
//!
//! Spawns three real `mlxcel-server` processes:
//! - A `--node-role decode` node
//! - A `--node-role prefill` node (wired to the decode node)
//! - A `--node-role router` front-end (wired to both)
//!
//! Then posts a streaming chat-completions request to the router's HTTP
//! endpoint and asserts the concatenated SSE delta content matches the
//! single-node greedy output.
//!
//! The expected text constant is left as a placeholder so the orchestrator
//! can fill it in from a reference hybrid-node run during the gate.
//!
//! Run with:
//!
//! ```text
//! cargo test --test disaggregated_router_e2e --release \
//!     --features metal,accelerate -- --ignored --nocapture
//! ```

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// qwen3 checkpoint directory name (pool-backed Fp16, the handoff scope).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";

/// Expected concatenated SSE content from the router for the test prompt.
///
/// This is the single-node greedy reference: a hybrid `mlxcel-server` answers
/// the same "What is 2 + 2?" chat request (temperature 0, enable_thinking
/// false) with exactly this content, stopping on EOS. The disaggregated router
/// path (router tokenizes -> prefill -> decode -> SSE) must reproduce it
/// byte-for-byte. Update if the model changes.
const EXPECTED_ROUTER_TEXT: &str = "2 + 2 = 4.";

/// Kills its child process on drop so a panicking assertion never leaks a
/// spawned `mlxcel-server` process.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Reserve `n` distinct ephemeral localhost ports atomically, then release
/// them. A small TOCTOU window remains before the child processes rebind.
fn reserve_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port"))
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().expect("local addr").port())
        .collect()
    // Listeners drop here, freeing the ports.
}

/// Spawn an `mlxcel-server` process with the given arguments.
fn spawn_role_server(args: &[&str]) -> ChildGuard {
    ChildGuard(
        Command::new(repo_binary_path("mlxcel-server"))
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn mlxcel-server"),
    )
}

/// Poll-connect `addr` until it accepts a TCP connection or the deadline passes.
async fn wait_for_tcp(addr: &str, deadline: Instant) -> bool {
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Poll the router's `/health` HTTP endpoint until it returns 200 or the
/// deadline passes.
async fn wait_for_http_health(url: &str, deadline: Instant) -> bool {
    let client = reqwest::Client::new();
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(url).send().await
            && resp.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns three real mlxcel-server processes loading qwen3-0.6b-4bit; run with --ignored"]
async fn disaggregated_router_streams_correct_output() {
    let model_dir = repo_model_dir(QWEN3_DIR);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping: mlxcel-server binary not found at {}.\n\
             Build with: cargo build --release --bin mlxcel-server --features metal,accelerate",
            binary.display()
        );
        return;
    }

    let ports = reserve_ports(6);
    let prefill_http = ports[0].to_string();
    let decode_http = ports[1].to_string();
    let router_http = ports[2].to_string();
    let prefill_serving_addr = format!("127.0.0.1:{}", ports[3]);
    let decode_serving_addr = format!("127.0.0.1:{}", ports[4]);
    let router_serving_addr = format!("127.0.0.1:{}", ports[5]);
    let model_arg = model_dir.to_string_lossy().to_string();

    // Spawn decode first so it is ready before prefill starts handing off.
    let _decode = spawn_role_server(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &decode_http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "decode",
        "--serving-bind",
        &decode_serving_addr,
        "--no-warmup",
    ]);
    let _prefill = spawn_role_server(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &prefill_http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "prefill",
        "--serving-bind",
        &prefill_serving_addr,
        "--decode-peers",
        &decode_serving_addr,
        "--no-warmup",
    ]);
    // The router needs a model path for the tokenizer and chat template but
    // does NOT load weights; --no-warmup is still needed so it does not try
    // to warm up a model.
    let _router = spawn_role_server(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &router_http,
        "--node-role",
        "router",
        "--serving-bind",
        &router_serving_addr,
        "--prefill-peers",
        &prefill_serving_addr,
        "--decode-peers",
        &decode_serving_addr,
        "--no-warmup",
    ]);

    // Wait for all three processes to be ready. The prefill and decode nodes
    // expose their role transports on the serving TCP ports; the router
    // exposes an HTTP /health endpoint.
    let deadline = Instant::now() + Duration::from_secs(240);
    assert!(
        wait_for_tcp(&decode_serving_addr, deadline).await,
        "decode node serving transport never came up at {decode_serving_addr}"
    );
    assert!(
        wait_for_tcp(&prefill_serving_addr, deadline).await,
        "prefill node serving transport never came up at {prefill_serving_addr}"
    );
    let health_url = format!("http://127.0.0.1:{router_http}/health");
    assert!(
        wait_for_http_health(&health_url, deadline).await,
        "router HTTP /health never returned 200 at {health_url}"
    );

    // POST a streaming chat-completions request to the router.
    let client = reqwest::Client::new();
    let chat_url = format!("http://127.0.0.1:{router_http}/v1/chat/completions");
    let response = client
        .post(&chat_url)
        .json(&serde_json::json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .send()
        .await
        .expect("POST /v1/chat/completions");
    assert!(
        response.status().is_success(),
        "router returned HTTP {}: expected 200",
        response.status()
    );

    // Read the SSE body and concatenate delta content across chunks.
    let body = response.text().await.expect("read SSE body");
    let mut content = String::new();
    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
                && let Some(text) = v
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|t| t.as_str())
            {
                content.push_str(text);
            }
        }
    }

    eprintln!("router SSE content: {content:?}");
    assert!(
        !content.is_empty(),
        "router returned an empty SSE content stream"
    );
    assert_eq!(
        content, EXPECTED_ROUTER_TEXT,
        "router SSE content does not match the expected single-node output\n\
         expected: {EXPECTED_ROUTER_TEXT:?}\n     got: {content:?}"
    );
    eprintln!("OK: three-process disaggregated router reproduced the expected output via SSE.");
}
