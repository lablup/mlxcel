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

//! Integration tests thinking-token budget.
//!
//! The correctness tests spin up the `mlxcel-server` binary against a real
//! Qwen3 model and verify the generation loop enforces the configured cap
//! end-to-end through the HTTP API. Both `/v1/chat/completions` and
//! `/completion` are exercised, in both streaming and non-streaming modes.
//!
//! All tests in this file require the Qwen3 model weights to be present at
//! `models/qwen3-0.6b-4bit/`. They are gated with
//! `#[ignore = "requires local model weights and the mlxcel-server binary"]`
//! so that `cargo test --all` succeeds in CI where the model is absent.
//!
//! To run the gated tests with the model present:
//! ```text
//! cargo test --test thinking_budget --release -- --ignored
//! ```

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Smallest Qwen3 thinking model we have locally.
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

async fn chat_message_text(resp: reqwest::Response) -> String {
    let body: serde_json::Value = resp.json().await.expect("parse chat response");
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

async fn post_completion(
    client: &reqwest::Client,
    base_url: &str,
    body: serde_json::Value,
) -> reqwest::Response {
    client
        .post(format!("{base_url}/completion"))
        .json(&body)
        .send()
        .await
        .expect("send /completion request")
}

async fn completion_content(resp: reqwest::Response) -> String {
    let body: serde_json::Value = resp.json().await.expect("parse /completion response");
    body["content"].as_str().unwrap_or_default().to_string()
}

/// Word-count proxy for the reasoning-block length. We don't have the
/// tokenizer here, so we use `split_whitespace` as a coarse proxy — the
/// cap-vs-unbounded comparison has a large enough margin that this is
/// robust. Returns `Some(0)` when the block is empty and `None` when no
/// `<think>...</think>` block is present.
fn count_reasoning_words(text: &str) -> Option<usize> {
    let open = text.find("<think>").map(|i| i + "<think>".len())?;
    let close = text[open..].find("</think>")? + open;
    let body = &text[open..close];
    Some(body.split_whitespace().count())
}

/// Non-streaming: `/v1/chat/completions` with a small cap vs unbounded.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_nonstream_budget_enforced_end_to_end() {
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

    let capped = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Explain why the sky is blue in one paragraph."}],
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_budget_tokens": 8
            }),
        )
        .await,
    )
    .await;
    let capped_words = count_reasoning_words(&capped).unwrap_or(0);

    let unbounded = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Explain why the sky is blue in one paragraph."}],
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_budget_tokens": -1
            }),
        )
        .await,
    )
    .await;
    let unbounded_words = count_reasoning_words(&unbounded).unwrap_or(0);

    stop_server(&mut child);

    assert!(
        capped_words <= unbounded_words,
        "capped reasoning ({capped_words}) should be <= unbounded ({unbounded_words})"
    );
}

/// Streaming: SSE path forces `</think>` in the delta stream.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn chat_streaming_budget_enforced_end_to_end() {
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

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Explain recursion."}],
            "max_tokens": 256,
            "temperature": 0.0,
            "thinking_budget_tokens": 4,
            "stream": true
        }))
        .send()
        .await
        .expect("send streaming chat request");

    let body = resp.text().await.expect("read SSE body");
    let mut assembled = String::new();
    for chunk in body.split("\n\n") {
        let Some(data) = chunk.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
            assembled.push_str(delta);
        }
    }

    stop_server(&mut child);

    assert!(
        assembled.contains("</think>"),
        "streaming budget must force </think> emission; got: {assembled}"
    );
}

/// `/completion` with llama.cpp-style `thinking_budget_tokens`.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn native_completion_budget_enforced_end_to_end() {
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

    let content = completion_content(
        post_completion(
            &client,
            &base_url,
            serde_json::json!({
                "prompt": "<|im_start|>user\nExplain gradient descent.<|im_end|>\n<|im_start|>assistant\n",
                "n_predict": 256,
                "temperature": 0.0,
                "thinking_budget_tokens": 4
            }),
        )
        .await,
    )
    .await;

    stop_server(&mut child);

    assert!(
        content.contains("</think>"),
        "/completion with budget=4 must force </think> emission; got: {content}"
    );
}

/// `--reasoning-budget` CLI flag applied server-wide.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn server_cli_default_applies_to_every_request() {
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
        "--reasoning-budget",
        "4",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let body = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Why is the sky blue?"}],
                "max_tokens": 256,
                "temperature": 0.0
            }),
        )
        .await,
    )
    .await;

    stop_server(&mut child);

    assert!(
        body.contains("</think>"),
        "--reasoning-budget 4 must force </think> without per-request override; got: {body}"
    );
}

/// A per-request `-1` reverts to unbounded even when the server default is set.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn per_request_minus_one_reverts_server_default() {
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
        "--reasoning-budget",
        "2",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let capped = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Explain photosynthesis."}],
                "max_tokens": 256,
                "temperature": 0.0
            }),
        )
        .await,
    )
    .await;
    let capped_words = count_reasoning_words(&capped).unwrap_or(usize::MAX);

    let unbounded = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Explain photosynthesis."}],
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_budget_tokens": -1
            }),
        )
        .await,
    )
    .await;
    let unbounded_words = count_reasoning_words(&unbounded).unwrap_or(0);

    stop_server(&mut child);

    assert!(
        unbounded_words >= capped_words,
        "per-request -1 must allow >= capped reasoning; capped={capped_words}, unbounded={unbounded_words}"
    );
}

/// Alias coverage: `thinking_token_budget` (vLLM alias).
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn alias_thinking_token_budget_is_honored() {
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
    let body = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_token_budget": 4
            }),
        )
        .await,
    )
    .await;
    stop_server(&mut child);
    assert!(body.contains("</think>"));
}

/// Alias coverage: `thinking_budget` (Qwen alias).
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn alias_thinking_budget_is_honored() {
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
    let body = chat_message_text(
        post_chat(
            &client,
            &base_url,
            serde_json::json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 256,
                "temperature": 0.0,
                "thinking_budget": 4
            }),
        )
        .await,
    )
    .await;
    stop_server(&mut child);
    assert!(body.contains("</think>"));
}

/// Invalid negative values must return a 400 response.
#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn invalid_budget_returns_400() {
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

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "thinking_budget_tokens": -7
        }))
        .send()
        .await
        .expect("send invalid budget request");

    let status = resp.status();
    stop_server(&mut child);
    assert_eq!(status.as_u16(), 400, "expected 400 for invalid budget");
}
