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

//! One request with `top_k` above the vocabulary must not take the server
//! down (#2247).
//!
//! The server forwards a request's `top_k` to the sampler unchanged, and the
//! stock sampling chain called `argpartition(-x, top_k - 1)`, which throws when
//! `top_k > vocab`. The throw crossed a non-`Result` cxx bridge function and
//! aborted the whole `mlxcel-server` process, dropping every in-flight request.
//! This test boots the real binary on Qwen3-0.6B-4bit and sends chat
//! completions with a huge `top_k`, with top-p inactive (the default path that
//! reaches the chain) and active, and asserts each returns a normal completion
//! and the server stays up. Before the fix the first request gets no response
//! because the server process died.
//!
//! Skips with a note when the checkpoint or the server binary is absent (set
//! `MLXCEL_REQUIRE_MODELS=1` to make a missing checkpoint fail instead).

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

const MODEL: &str = "Qwen3-0.6B-4bit";

/// Far above Qwen3's 151936-token vocabulary.
const HUGE_TOP_K: i64 = 1_000_000;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

/// Kills the server on every exit path, including a failed assertion.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str, server: &mut Child) -> bool {
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.try_wait() {
            panic!("mlxcel-server exited before becoming healthy: {status}");
        }
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    false
}

#[tokio::test]
async fn chat_completion_with_top_k_above_vocab_returns_a_normal_response() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("skipping: checkpoint {} not found", model_dir.display());
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "skipping: mlxcel-server binary not built at {}",
            binary.display()
        );
        return;
    }

    let port = reserve_port();
    let child = Command::new(&binary)
        .arg("-m")
        .arg(&model_dir)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn mlxcel-server");
    let mut server = ServerGuard(child);
    let base_url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("http client");
    assert!(
        wait_for_health(&client, &base_url, &mut server.0).await,
        "mlxcel-server did not become healthy within 180 s"
    );

    // top_p 1.0 leaves the rejection kernel unrouted, so the stock chain runs
    // the top-k filter: the case that aborted. 0.9 covers the routed case.
    for top_p in [1.0f64, 0.9] {
        let body = serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_k": HUGE_TOP_K,
            "top_p": top_p,
            "seed": 2247,
        });
        let response = client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&body)
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(err) => {
                // An aborting server drops the connection before the process
                // is reaped, so give it a moment before reading the status.
                tokio::time::sleep(Duration::from_secs(2)).await;
                let exit = server.0.try_wait().ok().flatten();
                panic!(
                    "top_k={HUGE_TOP_K} top_p={top_p}: no response ({err}); server exit: {exit:?}"
                );
            }
        };
        let status = response.status();
        let value: serde_json::Value = response.json().await.expect("parse chat response JSON");
        assert!(
            status.is_success(),
            "top_k={HUGE_TOP_K} top_p={top_p}: status {status}, body {value}"
        );
        assert!(
            value["choices"][0]["message"].is_object(),
            "top_k={HUGE_TOP_K} top_p={top_p}: no message in {value}"
        );
        let completion_tokens = value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        assert!(
            completion_tokens > 0,
            "top_k={HUGE_TOP_K} top_p={top_p}: no tokens sampled in {value}"
        );
    }

    assert!(
        server.0.try_wait().expect("poll server").is_none(),
        "mlxcel-server exited after the requests"
    );
    let health = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .expect("health after the requests");
    assert!(
        health.status().is_success(),
        "server unhealthy after the requests"
    );
}
