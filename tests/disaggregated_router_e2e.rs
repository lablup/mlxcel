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

/// Parse an SSE body into concatenated `(content, reasoning_content)` deltas.
fn parse_sse_deltas(body: &str) -> (String, String) {
    let mut content = String::new();
    let mut reasoning = String::new();
    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                let delta = v
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"));
                if let Some(text) = delta
                    .and_then(|d| d.get("content"))
                    .and_then(|t| t.as_str())
                {
                    content.push_str(text);
                }
                if let Some(text) = delta
                    .and_then(|d| d.get("reasoning_content"))
                    .and_then(|t| t.as_str())
                {
                    reasoning.push_str(text);
                }
            }
        }
    }
    (content, reasoning)
}

/// Issue #198: a THINKING-ENABLED request through the disaggregated router
/// must produce the same `content` / `reasoning_content` split as the
/// single-node chat route, with no structural markers (`<think>`) leaking
/// into either field.
///
/// Runs the single-node hybrid reference first (then kills it), so at most
/// two model-loaded processes are resident at once alongside the router.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns four real mlxcel-server processes loading qwen3-0.6b-4bit; run with --ignored"]
async fn disaggregated_router_filters_thinking_output() {
    let model_dir = repo_model_dir(QWEN3_DIR);
    if !model_dir.exists() {
        eprintln!("Skipping {QWEN3_DIR}: model directory not found");
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    let request_body = serde_json::json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "stream": true,
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": true}
    });
    let client = reqwest::Client::new();

    // ---- Single-node hybrid reference ----
    let (ref_content, ref_reasoning) = {
        let ports = reserve_ports(1);
        let hybrid_http = ports[0].to_string();
        let _hybrid = spawn_role_server(&[
            "-m",
            &model_arg,
            "--host",
            "127.0.0.1",
            "--port",
            &hybrid_http,
            "--no-warmup",
        ]);
        let deadline = Instant::now() + Duration::from_secs(240);
        let health_url = format!("http://127.0.0.1:{hybrid_http}/health");
        assert!(
            wait_for_http_health(&health_url, deadline).await,
            "hybrid reference server never became healthy at {health_url}"
        );
        let response = client
            .post(format!(
                "http://127.0.0.1:{hybrid_http}/v1/chat/completions"
            ))
            .json(&request_body)
            .send()
            .await
            .expect("POST to hybrid reference");
        assert!(response.status().is_success(), "hybrid returned an error");
        let body = response.text().await.expect("read hybrid SSE body");
        parse_sse_deltas(&body)
        // _hybrid drops here, killing the reference server.
    };
    eprintln!(
        "hybrid reference: content={ref_content:?} reasoning ({} chars)",
        ref_reasoning.len()
    );
    assert!(
        !ref_reasoning.is_empty(),
        "thinking-enabled hybrid reference produced no reasoning_content; \
         the parity comparison would be vacuous"
    );

    // ---- Three-process disaggregated router run ----
    let ports = reserve_ports(6);
    let prefill_http = ports[0].to_string();
    let decode_http = ports[1].to_string();
    let router_http = ports[2].to_string();
    let prefill_serving_addr = format!("127.0.0.1:{}", ports[3]);
    let decode_serving_addr = format!("127.0.0.1:{}", ports[4]);
    let router_serving_addr = format!("127.0.0.1:{}", ports[5]);

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

    let deadline = Instant::now() + Duration::from_secs(240);
    assert!(
        wait_for_tcp(&decode_serving_addr, deadline).await,
        "decode node serving transport never came up"
    );
    assert!(
        wait_for_tcp(&prefill_serving_addr, deadline).await,
        "prefill node serving transport never came up"
    );
    let health_url = format!("http://127.0.0.1:{router_http}/health");
    assert!(
        wait_for_http_health(&health_url, deadline).await,
        "router HTTP /health never returned 200"
    );

    let response = client
        .post(format!(
            "http://127.0.0.1:{router_http}/v1/chat/completions"
        ))
        .json(&request_body)
        .send()
        .await
        .expect("POST to router");
    assert!(response.status().is_success(), "router returned an error");
    let body = response.text().await.expect("read router SSE body");
    let (content, reasoning) = parse_sse_deltas(&body);
    eprintln!(
        "router: content={content:?} reasoning ({} chars)",
        reasoning.len()
    );

    assert!(
        !content.contains("<think>") && !content.contains("</think>"),
        "thinking markers leaked into router content: {content:?}"
    );
    assert!(
        !reasoning.contains("<think>") && !reasoning.contains("</think>"),
        "thinking markers leaked into router reasoning_content"
    );
    assert_eq!(
        reasoning, ref_reasoning,
        "router reasoning_content does not match the single-node chat route"
    );
    assert_eq!(
        content, ref_content,
        "router content does not match the single-node chat route"
    );
    eprintln!("OK: router thinking-enabled output matches the single-node split.");
}

// ── /v1/completions parity (issue #200) ──────────────────────────────────

/// Model alias used for the completion-parity test. The single-node baseline
/// is started with `--alias <this>`, so its `display_model_id()` equals the
/// `model` string the router echoes back from the request body. With the model
/// field aligned, the only inherently volatile fields left are `id` and
/// `created`, which the comparison normalizes.
const COMPLETION_MODEL_ALIAS: &str = "qwen3-completions";

/// Normalize the volatile `id` and `created` fields of a completion body (or
/// chunk) so two independent runs can be compared for byte-identical structure.
fn normalize_completion(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        if obj.contains_key("id") {
            obj.insert("id".to_string(), serde_json::json!("<id>"));
        }
        if obj.contains_key("created") {
            obj.insert("created".to_string(), serde_json::json!(0));
        }
    }
    v
}

/// Parse an SSE completion stream into the ordered list of normalized chunk
/// objects. The `[DONE]` sentinel is dropped; keepalive comment lines (which do
/// not start with `data: `) are ignored.
fn parse_completion_chunks(body: &str) -> Vec<serde_json::Value> {
    let mut chunks = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                chunks.push(normalize_completion(v));
            }
        }
    }
    chunks
}

/// Issue #200: `POST /v1/completions` through the disaggregated router must
/// return output byte-identical to single-node `/v1/completions` for the same
/// prompt and sampling, in both the non-streaming (single JSON) and streaming
/// (SSE) shapes.
///
/// The single-node baseline runs first (started with `--alias` so its model id
/// matches the router's echoed `model`), is captured, then killed before the
/// three-process router stack starts, so at most three model-loaded processes
/// are resident at once. The comparison normalizes only the inherently volatile
/// `id` / `created` fields; everything else (object, model, system_fingerprint,
/// choices[index, text, finish_reason, logprobs], usage[...], the per-chunk
/// stream shapes, and the `[DONE]` sentinel) must match exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns four real mlxcel-server processes loading qwen3-0.6b-4bit; run with --ignored"]
async fn disaggregated_router_completions_match_single_node() {
    let model_dir = repo_model_dir(QWEN3_DIR);
    if !model_dir.exists() {
        eprintln!("Skipping {QWEN3_DIR}: model directory not found");
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    let prompt = "The capital of France is";
    let nonstream_body = serde_json::json!({
        "model": COMPLETION_MODEL_ALIAS,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0
    });
    let stream_body = serde_json::json!({
        "model": COMPLETION_MODEL_ALIAS,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": true
    });
    let client = reqwest::Client::new();

    // ---- Single-node reference (started with --alias for model parity) ----
    let (ref_nonstream, ref_stream_chunks) = {
        let ports = reserve_ports(1);
        let http = ports[0].to_string();
        let _single = spawn_role_server(&[
            "-m",
            &model_arg,
            "--host",
            "127.0.0.1",
            "--port",
            &http,
            "--alias",
            COMPLETION_MODEL_ALIAS,
            "--no-warmup",
        ]);
        let deadline = Instant::now() + Duration::from_secs(240);
        let health_url = format!("http://127.0.0.1:{http}/health");
        assert!(
            wait_for_http_health(&health_url, deadline).await,
            "single-node reference never became healthy at {health_url}"
        );
        let completions_url = format!("http://127.0.0.1:{http}/v1/completions");

        let ns_resp = client
            .post(&completions_url)
            .json(&nonstream_body)
            .send()
            .await
            .expect("POST single-node /v1/completions (non-stream)");
        assert!(
            ns_resp.status().is_success(),
            "single-node non-stream completion returned HTTP {}",
            ns_resp.status()
        );
        let ns_json = ns_resp
            .json::<serde_json::Value>()
            .await
            .expect("parse single-node non-stream completion JSON");

        let st_resp = client
            .post(&completions_url)
            .json(&stream_body)
            .send()
            .await
            .expect("POST single-node /v1/completions (stream)");
        assert!(
            st_resp.status().is_success(),
            "single-node stream completion returned HTTP {}",
            st_resp.status()
        );
        let st_body = st_resp.text().await.expect("read single-node SSE body");

        (
            normalize_completion(ns_json),
            parse_completion_chunks(&st_body),
        )
        // _single drops here, killing the reference server.
    };
    eprintln!("single-node non-stream reference: {ref_nonstream}");
    assert!(
        ref_nonstream["choices"][0]["text"]
            .as_str()
            .is_some_and(|t| !t.is_empty()),
        "single-node reference produced empty completion text; parity check would be vacuous"
    );
    assert!(
        !ref_stream_chunks.is_empty(),
        "single-node reference produced no SSE chunks; parity check would be vacuous"
    );

    // ---- Three-process disaggregated router run ----
    let ports = reserve_ports(6);
    let prefill_http = ports[0].to_string();
    let decode_http = ports[1].to_string();
    let router_http = ports[2].to_string();
    let prefill_serving_addr = format!("127.0.0.1:{}", ports[3]);
    let decode_serving_addr = format!("127.0.0.1:{}", ports[4]);
    let router_serving_addr = format!("127.0.0.1:{}", ports[5]);

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

    let deadline = Instant::now() + Duration::from_secs(240);
    assert!(
        wait_for_tcp(&decode_serving_addr, deadline).await,
        "decode node serving transport never came up"
    );
    assert!(
        wait_for_tcp(&prefill_serving_addr, deadline).await,
        "prefill node serving transport never came up"
    );
    let health_url = format!("http://127.0.0.1:{router_http}/health");
    assert!(
        wait_for_http_health(&health_url, deadline).await,
        "router HTTP /health never returned 200"
    );
    let router_completions_url = format!("http://127.0.0.1:{router_http}/v1/completions");

    // Non-streaming parity.
    let router_ns_resp = client
        .post(&router_completions_url)
        .json(&nonstream_body)
        .send()
        .await
        .expect("POST router /v1/completions (non-stream)");
    assert!(
        router_ns_resp.status().is_success(),
        "router non-stream completion returned HTTP {}",
        router_ns_resp.status()
    );
    let router_ns = normalize_completion(
        router_ns_resp
            .json::<serde_json::Value>()
            .await
            .expect("parse router non-stream completion JSON"),
    );
    eprintln!("router non-stream: {router_ns}");
    assert_eq!(
        router_ns, ref_nonstream,
        "router non-stream /v1/completions body is not byte-identical to single-node"
    );

    // Streaming parity.
    let router_st_resp = client
        .post(&router_completions_url)
        .json(&stream_body)
        .send()
        .await
        .expect("POST router /v1/completions (stream)");
    assert!(
        router_st_resp.status().is_success(),
        "router stream completion returned HTTP {}",
        router_st_resp.status()
    );
    let router_st_body = router_st_resp.text().await.expect("read router SSE body");
    let router_st_chunks = parse_completion_chunks(&router_st_body);
    assert_eq!(
        router_st_chunks, ref_stream_chunks,
        "router stream /v1/completions chunks are not byte-identical to single-node"
    );
    eprintln!("OK: router /v1/completions matches single-node (non-stream + stream).");
}

// ── Multi-node routing, balancing, and failover (issue #201) ─────────────

/// Spawn a `--node-role decode` server wired to its own serving address.
fn spawn_decode(model_arg: &str, http: &str, serving_addr: &str) -> ChildGuard {
    spawn_role_server(&[
        "-m",
        model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "decode",
        "--serving-bind",
        serving_addr,
        "--no-warmup",
    ])
}

/// Spawn a `--node-role prefill` server wired to its serving address and a
/// single configured decode peer (the router overrides the decode target per
/// request via `decode_target`; the config peer is only the fallback).
fn spawn_prefill(model_arg: &str, http: &str, serving_addr: &str, decode_peer: &str) -> ChildGuard {
    spawn_role_server(&[
        "-m",
        model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        http,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--decode-storage-backend",
        "paged",
        "--node-role",
        "prefill",
        "--serving-bind",
        serving_addr,
        "--decode-peers",
        decode_peer,
        "--no-warmup",
    ])
}

/// POST a non-streaming completion to the router and return `(status, text)`.
async fn post_completion(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
) -> (reqwest::StatusCode, String) {
    let resp = client
        .post(url)
        .json(&serde_json::json!({
            "model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0.0
        }))
        .send()
        .await
        .expect("POST /v1/completions");
    let status = resp.status();
    let text = if status.is_success() {
        resp.json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v["choices"][0]["text"].as_str().map(str::to_string))
            .unwrap_or_default()
    } else {
        String::new()
    };
    (status, text)
}

/// Issue #201, AC1 + AC2: with two prefill AND two decode nodes the router
/// spreads requests across both pools (asserted via `GET /router/stats`
/// per-node hit counters), and killing one prefill node does not wedge the
/// router: subsequent requests still succeed by routing to the survivor.
///
/// Five processes: 2 decode + 2 prefill + 1 router. qwen3-0.6b-4bit is tiny, but
/// this still loads four model copies, so it is `#[ignore]` and run explicitly
/// by the gate, not in the default unit run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns five real mlxcel-server processes (2 prefill + 2 decode + router); run with --ignored"]
async fn disaggregated_router_balances_and_survives_node_failure() {
    let model_dir = repo_model_dir(QWEN3_DIR);
    if !model_dir.exists() {
        eprintln!("Skipping {QWEN3_DIR}: model directory not found");
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();

    // Ports: [prefill0_http, prefill1_http, decode0_http, decode1_http,
    //         router_http, prefill0_srv, prefill1_srv, decode0_srv,
    //         decode1_srv, router_srv].
    let ports = reserve_ports(10);
    let prefill0_http = ports[0].to_string();
    let prefill1_http = ports[1].to_string();
    let decode0_http = ports[2].to_string();
    let decode1_http = ports[3].to_string();
    let router_http = ports[4].to_string();
    let prefill0_srv = format!("127.0.0.1:{}", ports[5]);
    let prefill1_srv = format!("127.0.0.1:{}", ports[6]);
    let decode0_srv = format!("127.0.0.1:{}", ports[7]);
    let decode1_srv = format!("127.0.0.1:{}", ports[8]);
    let router_srv = format!("127.0.0.1:{}", ports[9]);

    // Decode nodes first so they are ready before prefill hands off.
    let _decode0 = spawn_decode(&model_arg, &decode0_http, &decode0_srv);
    let _decode1 = spawn_decode(&model_arg, &decode1_http, &decode1_srv);
    // Each prefill node configures decode0 as its static fallback; the router
    // overrides the decode target per request, balancing both decode nodes.
    let mut prefill0 = Some(spawn_prefill(
        &model_arg,
        &prefill0_http,
        &prefill0_srv,
        &decode0_srv,
    ));
    let _prefill1 = spawn_prefill(&model_arg, &prefill1_http, &prefill1_srv, &decode0_srv);
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
        &router_srv,
        "--prefill-peers",
        &format!("{prefill0_srv},{prefill1_srv}"),
        "--decode-peers",
        &format!("{decode0_srv},{decode1_srv}"),
        "--no-warmup",
    ]);

    let deadline = Instant::now() + Duration::from_secs(240);
    for addr in [&decode0_srv, &decode1_srv, &prefill0_srv, &prefill1_srv] {
        assert!(
            wait_for_tcp(addr, deadline).await,
            "serving transport never came up at {addr}"
        );
    }
    let health_url = format!("http://127.0.0.1:{router_http}/health");
    assert!(
        wait_for_http_health(&health_url, deadline).await,
        "router HTTP /health never returned 200"
    );

    let client = reqwest::Client::new();
    let completions_url = format!("http://127.0.0.1:{router_http}/v1/completions");
    let stats_url = format!("http://127.0.0.1:{router_http}/router/stats");
    let prompt = "The capital of France is";

    // ---- AC1: distribution across both pools ----
    for _ in 0..8 {
        let (status, text) = post_completion(&client, &completions_url, "qwen3", prompt).await;
        assert!(
            status.is_success(),
            "pre-failure request failed: HTTP {status}"
        );
        assert!(!text.is_empty(), "pre-failure request returned empty text");
    }
    let stats: serde_json::Value = client
        .get(&stats_url)
        .send()
        .await
        .expect("GET /router/stats")
        .json()
        .await
        .expect("parse /router/stats");
    eprintln!("router stats before failure: {stats}");
    let prefill_hits = stats["prefill_hits"]
        .as_object()
        .expect("prefill_hits object");
    let decode_hits = stats["decode_hits"]
        .as_object()
        .expect("decode_hits object");
    let nonzero = |m: &serde_json::Map<String, serde_json::Value>| {
        m.values().filter(|v| v.as_u64().unwrap_or(0) > 0).count()
    };
    assert_eq!(
        nonzero(prefill_hits),
        2,
        "expected both prefill nodes to receive requests, got {prefill_hits:?}"
    );
    assert_eq!(
        nonzero(decode_hits),
        2,
        "expected both decode nodes to receive requests, got {decode_hits:?}"
    );

    // ---- AC2: kill one prefill node; subsequent requests still succeed ----
    drop(prefill0.take()); // ChildGuard::drop kills the process.
    tokio::time::sleep(Duration::from_secs(1)).await;
    for i in 0..6 {
        let (status, text) = post_completion(&client, &completions_url, "qwen3", prompt).await;
        assert!(
            status.is_success(),
            "post-failure request {i} failed: HTTP {status} (router wedged after node death)"
        );
        assert!(
            !text.is_empty(),
            "post-failure request {i} returned empty text"
        );
    }
    eprintln!("OK: router balanced both pools and survived a prefill-node failure.");
}
