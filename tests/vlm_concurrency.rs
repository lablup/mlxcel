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

//! Ignored real-model VLM server concurrency smoke tests.
//!
//! Issue #731 fixed a Gemma 3n VLM race where the MobileNet/MSFA
//! `per_layer_inputs` tensor was parked in one model-wide fallback slot. A
//! burst of two server requests could have one row consume the other's tensor
//! or panic after the slot was drained. The test below runs the real HTTP
//! stack with two concurrent image requests so the scheduler exercises the
//! sequence-aware binding path.
//!
//! To run the gated test:
//! ```text
//! cargo test --release --test vlm_concurrency -- --ignored
//! ```

mod common;

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use common::{repo_binary_path, repo_model_dir};

const GEMMA3N_E2B_MODEL: &str = "gemma3n-e2b-4bit";
const GEMMA3N_E4B_MODEL: &str = "gemma3n-e4b-4bit";

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn fixture_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png")
}

fn fixture_image_data_uri() -> String {
    let bytes = std::fs::read(fixture_image_path()).expect("read VLM fixture image");
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:image/png;base64,{encoded}")
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

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("mlxcel-server did not become healthy at {base_url}");
}

async fn post_vlm_chat(
    client: reqwest::Client,
    base_url: String,
    image_data_uri: String,
    text: &'static str,
) -> serde_json::Value {
    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gemma3n-vl-test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": text}
                ]
            }],
            "max_tokens": 24,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("send VLM chat request");
    let status = response.status();
    let body = response
        .json::<serde_json::Value>()
        .await
        .expect("parse VLM chat response");
    assert!(
        status.is_success(),
        "VLM chat request returned non-success status {status}: {body}"
    );
    body
}

fn has_chat_choice(response: &serde_json::Value) -> bool {
    response["choices"][0]["message"]["content"].is_string()
}

#[test]
#[ignore = "requires local Gemma 3n VLM weights and the mlxcel-bench-decode binary"]
fn gemma3n_vlm_single_row_bench_prefill_accepts_m5_tile_padding() {
    let model_dir = repo_model_dir(GEMMA3N_E4B_MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let model_arg = model_dir.to_string_lossy().to_string();
    let image_arg = fixture_image_path().to_string_lossy().to_string();
    let output = Command::new(repo_binary_path("mlxcel-bench-decode"))
        .args([
            "-m",
            &model_arg,
            "-p",
            "What is in this image?",
            "--image",
            &image_arg,
            "-n",
            "4",
            "--warmup-tokens",
            "20",
        ])
        .output()
        .expect("run mlxcel-bench-decode");

    assert!(
        output.status.success(),
        "single-row Gemma 3n VLM prefill failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
#[ignore = "requires local Gemma 3n VLM weights and the mlxcel-server binary"]
async fn gemma3n_vlm_two_concurrent_image_requests_do_not_cross_contaminate_state() {
    let model_dir = repo_model_dir(GEMMA3N_E2B_MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let model_arg = model_dir.to_string_lossy().to_string();
    let port_arg = port.to_string();
    let mut child = spawn_server(&[
        "--model",
        &model_arg,
        "--alias",
        "gemma3n-vl-test",
        "--host",
        "127.0.0.1",
        "--port",
        &port_arg,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--max-queue-depth",
        "8",
        "--no-warmup",
    ]);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .expect("build reqwest client");
    wait_for_health(&client, &base_url).await;

    let image_a = fixture_image_data_uri();
    let image_b = fixture_image_data_uri();
    let handle_a = tokio::spawn(post_vlm_chat(
        client.clone(),
        base_url.clone(),
        image_a,
        "What is in this image? Answer in one short sentence.",
    ));
    let handle_b = tokio::spawn(post_vlm_chat(
        client.clone(),
        base_url.clone(),
        image_b,
        "Describe the main object in this image in one short sentence.",
    ));

    let response_a = handle_a.await.expect("join VLM request A");
    let response_b = handle_b.await.expect("join VLM request B");
    stop_server(&mut child);

    assert!(
        has_chat_choice(&response_a),
        "missing chat choice: {response_a}"
    );
    assert!(
        has_chat_choice(&response_b),
        "missing chat choice: {response_b}"
    );
}
