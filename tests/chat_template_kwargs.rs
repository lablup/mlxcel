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

//! Integration tests `chat_template_kwargs` / `preserve_thinking`.
//!
//! The end-to-end tests spin up `mlxcel-server` against a real Qwen3-family
//! model and exercise the HTTP `/v1/chat/completions` surface in all three
//! per-request shapes: top-level `chat_template_kwargs`, nested
//! `extra_body.chat_template_kwargs`, the OpenAI SDK's flattened
//! root-level `extra_body={"preserve_thinking": ...}` alias, and the
//! DashScope flat `extra_body.preserve_thinking`.
//!
//! All tests in this file require Qwen3 weights at
//! `models/qwen3-0.6b-4bit/` and are gated with
//! `#[ignore = "requires local model weights and the mlxcel-server binary"]`.
//!
//! To run the gated tests:
//! ```text
//! cargo test --test chat_template_kwargs --release -- --ignored
//! ```

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Qwen3 thinking model used as a stand-in for Qwen3.6. The chat template
/// rendering path we exercise here is identical between Qwen3, Qwen3.5, and
/// Qwen3.6 (the differences are the model weights / training, not the
/// template mechanics).
const QWEN3_MODEL: &str = "qwen3-0.6b-4bit";

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("mlxcel-server did not become healthy at {base_url}");
}

fn spawn_server(args: &[&str]) -> Child {
    Command::new(repo_binary_path("mlxcel-server"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mlxcel-server")
}

fn stop_server(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

async fn post_chat(
    client: &reqwest::Client,
    base_url: &str,
    body: serde_json::Value,
) -> reqwest::Response {
    client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send chat request")
}

/// Multi-turn conversation body with a planted `<think>...</think>` block in
/// each assistant reply, used by the preserve_thinking tests.
///
/// The final user turn drives the current request; prior assistant replies
/// have thinking traces that either survive (preserve=true) or are stripped
/// (preserve=false).
fn make_multi_turn_body(chat_template_kwargs: Option<serde_json::Value>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nSimple addition.\n</think>\n\nIt's 4."},
            {"role": "user", "content": "And 3+3?"}
        ],
        "max_tokens": 32,
        "temperature": 0.0
    });
    if let Some(k) = chat_template_kwargs {
        body["chat_template_kwargs"] = k;
    }
    body
}

/// integration: per-request top-level `chat_template_kwargs.preserve_thinking=true`
/// must produce a 200 response. The actual prompt-rendering correctness is
/// covered by unit tests — this test only confirms that the server accepts
/// the shape end-to-end without 4xx/5xx errors.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_accepts_top_level_chat_template_kwargs() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping: model not present at {}", model_dir.display());
        return;
    }
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    // Primary llama.cpp shape.
    let resp = post_chat(
        &client,
        &base_url,
        make_multi_turn_body(Some(serde_json::json!({"preserve_thinking": true}))),
    )
    .await;
    let status = resp.status();
    stop_server(&mut child);
    assert!(
        status.is_success(),
        "top-level chat_template_kwargs must be accepted; got {status}"
    );
}

/// vLLM/OpenAI-SDK shape: nested under `extra_body`.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_accepts_extra_body_nested_chat_template_kwargs() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping: model not present at {}", model_dir.display());
        return;
    }
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let body = serde_json::json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "extra_body": {
            "chat_template_kwargs": {"preserve_thinking": true}
        }
    });
    let resp = post_chat(&client, &base_url, body).await;
    let status = resp.status();
    stop_server(&mut child);
    assert!(
        status.is_success(),
        "extra_body.chat_template_kwargs must be accepted; got {status}"
    );
}

/// DashScope shape: flat `extra_body.preserve_thinking`.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_accepts_dashscope_flat_preserve_thinking() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping: model not present at {}", model_dir.display());
        return;
    }
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let body = serde_json::json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "extra_body": {"preserve_thinking": true}
    });
    let resp = post_chat(&client, &base_url, body).await;
    let status = resp.status();
    stop_server(&mut child);
    assert!(
        status.is_success(),
        "DashScope flat extra_body.preserve_thinking must be accepted; got {status}"
    );
}

/// OpenAI SDK shape: `extra_body={"preserve_thinking": true}` flattened into
/// the request root.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_accepts_flattened_openai_extra_body_preserve_thinking() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping: model not present at {}", model_dir.display());
        return;
    }
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let body = serde_json::json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "preserve_thinking": true
    });
    let resp = post_chat(&client, &base_url, body).await;
    let status = resp.status();
    stop_server(&mut child);
    assert!(
        status.is_success(),
        "flattened OpenAI extra_body preserve_thinking must be accepted; got {status}"
    );
}

/// Server-wide default via `--chat-template-kwargs` applies when the request
/// carries no per-request override.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn server_default_chat_template_kwargs_applies_without_override() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping: model not present at {}", model_dir.display());
        return;
    }
    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_str,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--no-warmup",
        "--chat-template-kwargs",
        r#"{"preserve_thinking": true}"#,
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    // Request does NOT set chat_template_kwargs; the server default must
    // still apply.
    let resp = post_chat(&client, &base_url, make_multi_turn_body(None)).await;
    let status = resp.status();
    stop_server(&mut child);
    assert!(
        status.is_success(),
        "server default should flow into request without error; got {status}"
    );
}

/// Malformed `--chat-template-kwargs` JSON must cause the server to exit
/// with a non-zero status before listening on any port.
#[tokio::test]
#[ignore = "requires the mlxcel-server binary"]
async fn server_refuses_malformed_chat_template_kwargs_json() {
    let port = reserve_port();
    let port_str = port.to_string();
    // Use a non-existent model path — if the JSON validation fires first
    // (as it must), the server will never reach the model-loading step.
    let output = Command::new(repo_binary_path("mlxcel-server"))
        .args([
            "--model",
            "/nonexistent-model",
            "--host",
            "127.0.0.1",
            "--port",
            &port_str,
            "--chat-template-kwargs",
            "not-json{",
        ])
        .output()
        .expect("spawn server for malformed-json check");

    assert!(
        !output.status.success(),
        "server must exit non-zero on malformed --chat-template-kwargs"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("chat-template-kwargs") && stderr.contains("JSON"),
        "stderr should explain the JSON error; got: {stderr}"
    );
}
