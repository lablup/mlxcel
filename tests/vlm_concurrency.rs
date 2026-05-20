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
const QWEN2_5_VL_MODEL: &str = "qwen2.5-vl-3b-4bit";
const QWEN2_VL_MODEL: &str = "qwen2-vl-2b-4bit";

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

/// Regression guard for the `qwen2.5-vl-3b-4bit` same-process warmup failure.
///
/// `qwen2.5-vl-3b-4bit` previously aborted during warmup on M5 Max when run
/// through the same-process bench harness (warmup pass followed immediately by
/// a measured pass in the same process). The root cause was a generator-owned
/// KV-cache rebuild missing for some VLM paths — the warmup pass left stale KV
/// state that caused a 4-D attention mask broadcast to abort on the subsequent
/// measured pass. PR #34 fixed this by rebuilding generator-owned KV caches
/// before each single-row run in `CxxGenerator::reset_with_model()`.
/// `qwen2_5_vl` has no model-owned fallback state, so the fix applied
/// transitively.
///
/// This test exercises exactly the `FAIL:warmup` repro path: image VLM with
/// `--warmup-tokens 20` so the harness performs both a warmup pass and a
/// measured pass in the same process.
#[test]
#[ignore = "requires local Qwen2.5-VL weights and the mlxcel-bench-decode binary"]
fn qwen2_5_vl_single_row_bench_prefill_succeeds_with_m5_warmup() {
    let model_dir = repo_model_dir(QWEN2_5_VL_MODEL);
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
        "same-process warmup+measure for qwen2.5-vl-3b-4bit VLM failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Regression guard for the qwen2-vl-2b-4bit zero-token image-mode bug.
///
/// `qwen2-vl-2b-4bit` previously generated 0 tokens when given an image input.
/// The root cause was `insert_qwen_vl_image_tokens` only inserting the image
/// placeholder once rather than expanding it to the correct grid-count of
/// vision tokens. This meant the image prompt was effectively empty after
/// tokenisation, and the model produced no output. The fix expands
/// one-placeholder-per-image prompts to the full grid count.
///
/// This test verifies two things: (a) the bench harness exits successfully
/// (no panic / abort), and (b) at least one token was actually generated.
/// A plain success-only assertion would miss the 0-token regression because
/// the harness exited with code 0 even in the broken state.
#[test]
#[ignore = "requires local Qwen2-VL weights and the mlxcel-bench-decode binary"]
fn qwen2_vl_single_row_bench_prefill_generates_nonempty_image_output() {
    let model_dir = repo_model_dir(QWEN2_VL_MODEL);
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
            "8",
            "--warmup-tokens",
            "20",
        ])
        .output()
        .expect("run mlxcel-bench-decode");

    assert!(
        output.status.success(),
        "qwen2-vl-2b-4bit image VLM run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse "Generated tokens: N" from the profile output and assert N > 0.
    // In the broken state the bench exited 0 but produced 0 tokens.
    let generated_tokens: u64 = stdout
        .lines()
        .find_map(|line| {
            let stripped = line.trim().strip_prefix("Generated tokens:")?;
            stripped.trim().parse().ok()
        })
        .unwrap_or(0);
    assert!(
        generated_tokens > 0,
        "qwen2-vl-2b-4bit image input produced 0 generated tokens\nfull stdout:\n{stdout}"
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
