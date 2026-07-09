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

/// Gemma checkpoint directory name. gemma3 is a MODEL-OWNED paged family: the
/// disaggregated pool-block handoff (#125) does not support it, so a prefill
/// request for it must be rejected per-request with a clean error while the
/// prefill node keeps serving (issue #708). Used by
/// [`disaggregated_router_rejects_model_owned_family_and_keeps_serving`] as the
/// model-owned fixture (it is NOT a byte-fallback parity fixture; see
/// [`MINICPM_DIR`], which is both byte-fallback AND pool-backed). Fetch with:
/// `./target/release/mlxcel download mlx-community/gemma-3-1b-it-4bit`.
const GEMMA_DIR: &str = "gemma-3-1b-it-4bit";

/// Model alias for the model-owned rejection test (single-node baseline started
/// with `--alias <this>` so its `display_model_id()` matches the router-echoed
/// `model`).
const GEMMA_MODEL_ALIAS: &str = "gemma-completions";

/// MiniCPM-2B (llama-format export) checkpoint directory name (issue #398).
/// Two properties make it the chat-usage parity fixture:
///
/// * its Llama-style SentencePiece tokenizer has `byte_fallback = true`
///   (`<0xXX>` byte pieces in-vocab), satisfying the byte-fallback tokenizer
///   requirement of issue #398, and
/// * the llama-format export runs as a plain dense Llama model, so it is
///   pool-backed Fp16 — inside the disaggregated handoff scope (#125).
///
/// Gemma, the natural byte-fallback reference used by [`GEMMA_DIR`], cannot be
/// used here: gemma3 is a model-owned paged family and the prefill handoff
/// does not support it — the request crashes the prefill node's serving loop
/// (`PagedBlockPool::read_block_contents: layer 0 has no pool tensors`,
/// issue #708).
/// Fetch with:
/// `./target/release/mlxcel download mlx-community/MiniCPM-2B-sft-4bit-llama-format-mlx`
/// and place/symlink it at `models/minicpm-2b-4bit`.
const MINICPM_DIR: &str = "minicpm-2b-4bit";

/// Model alias for the MiniCPM byte-fallback CHAT usage parity test (issue
/// #398). A distinct alias from [`GEMMA_MODEL_ALIAS`] keeps the
/// completions-path and chat-path fixtures independent even though each test
/// spawns its own isolated set of processes.
const MINICPM_CHAT_MODEL_ALIAS: &str = "minicpm-chat";

/// Model alias for the MiniCPM byte-fallback COMPLETIONS parity test (issues
/// #387 / #708). Distinct from [`MINICPM_CHAT_MODEL_ALIAS`] so the
/// completions-path and chat-path fixtures stay independent.
const MINICPM_COMPLETIONS_MODEL_ALIAS: &str = "minicpm-completions";

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

/// Issue #387: `POST /v1/completions` through the disaggregated router must
/// report `usage.completion_tokens` and `finish_reason` identical to single-node
/// for a BYTE-FALLBACK tokenizer (MiniCPM-2B, llama-format export), not just
/// byte-level-BPE (Qwen).
///
/// The MiniCPM llama-format SentencePiece tokenizer has `byte_fallback = true`,
/// so a multi-byte character (e.g. the `é` in "café") is emitted as `<0xXX>`
/// byte pieces: several model tokens that surface as a single detokenized text
/// piece. The previous router counted emitted pieces, which under-counted those
/// tokens and could flip `finish_reason` between "length" and "stop". With the
/// worker's authoritative token count carried over the wire
/// (`ResultFrame.generated_tokens`), the router reports the exact count.
///
/// The prompt forces the multi-byte `é` in "café" into the output so the byte-
/// fallback path is actually exercised; the full body is compared byte-for-byte
/// (normalizing only the volatile `id` / `created`), and `usage.completion_tokens`
/// plus `finish_reason` are asserted explicitly for a clear failure message.
///
/// Retargeted from Gemma to MiniCPM-2B (issue #708): the original fixture used
/// [`GEMMA_DIR`], but gemma3 is a model-owned paged family the disaggregated
/// handoff cannot serve (the request crashed the prefill node), so the test was
/// unrunnable. MiniCPM-2B is both byte-fallback AND pool-backed dense Llama, so
/// it exercises the same authoritative-count path inside the handoff scope. See
/// [`MINICPM_DIR`].
///
/// Gated like the other real-model tests: `#[ignore]` plus a checkpoint-presence
/// guard that skips cleanly when the MiniCPM checkpoint is absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns four real mlxcel-server processes loading minicpm-2b-4bit; run with --ignored"]
async fn disaggregated_router_completions_match_single_node_byte_fallback() {
    let model_dir = repo_model_dir(MINICPM_DIR);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {MINICPM_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download \
             mlx-community/MiniCPM-2B-sft-4bit-llama-format-mlx (place it at \
             models/{MINICPM_DIR})",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    // A prompt that forces the multi-byte `é` (a byte-fallback `<0xXX>`
    // sequence) into the greedy continuation so the count fix is exercised.
    let prompt = "Spell the French word for coffee. It is café. Repeat it:";
    let nonstream_body = serde_json::json!({
        "model": MINICPM_COMPLETIONS_MODEL_ALIAS,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0
    });
    let stream_body = serde_json::json!({
        "model": MINICPM_COMPLETIONS_MODEL_ALIAS,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": true,
        "stream_options": {"include_usage": true}
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
            MINICPM_COMPLETIONS_MODEL_ALIAS,
            "--no-warmup",
        ]);
        let deadline = Instant::now() + Duration::from_secs(240);
        let health_url = format!("http://127.0.0.1:{http}/health");
        assert!(
            wait_for_http_health(&health_url, deadline).await,
            "single-node MiniCPM reference never became healthy at {health_url}"
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
    eprintln!("single-node MiniCPM non-stream reference: {ref_nonstream}");
    let ref_text = ref_nonstream["choices"][0]["text"].as_str().unwrap_or("");
    assert!(
        !ref_text.is_empty(),
        "single-node MiniCPM reference produced empty completion text; parity check would be vacuous"
    );
    // Guard the test's premise: the byte-fallback path is only exercised if the
    // greedy output actually contains a multi-byte character.
    assert!(
        !ref_text.is_ascii(),
        "single-node MiniCPM output {ref_text:?} is pure ASCII; the byte-fallback \
         count path is not exercised. Adjust the prompt so the output contains a \
         multi-byte character."
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

    // Non-streaming parity: usage.completion_tokens + finish_reason must match.
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
    eprintln!("router MiniCPM non-stream: {router_ns}");
    assert_eq!(
        router_ns["usage"]["completion_tokens"], ref_nonstream["usage"]["completion_tokens"],
        "router usage.completion_tokens diverges from single-node for a byte-fallback tokenizer"
    );
    assert_eq!(
        router_ns["choices"][0]["finish_reason"], ref_nonstream["choices"][0]["finish_reason"],
        "router finish_reason diverges from single-node for a byte-fallback tokenizer"
    );
    assert_eq!(
        router_ns, ref_nonstream,
        "router non-stream /v1/completions body is not byte-identical to single-node (MiniCPM)"
    );

    // Streaming parity: the usage chunk and finish chunk must match too.
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
        "router stream /v1/completions chunks are not byte-identical to single-node (MiniCPM)"
    );
    eprintln!(
        "OK: router /v1/completions matches single-node for a byte-fallback tokenizer \
         (usage + finish_reason)."
    );
}

/// Issue #398: `POST /v1/chat/completions` through the disaggregated router must
/// report a `usage` object (`prompt_tokens`, `completion_tokens`, `total_tokens`)
/// identical to single-node, in both the non-streaming response and the
/// streaming final usage chunk (gated on `stream_options.include_usage`), for a
/// BYTE-FALLBACK tokenizer (MiniCPM-2B, llama-format export). Before this fix
/// the router chat path never emitted `usage` at all.
///
/// Reuses the same byte-fallback prompt idea as
/// `disaggregated_router_completions_match_single_node_byte_fallback` (issue
/// #387) so the authoritative wire-carried token count is exercised end to
/// end: with a byte-fallback SentencePiece tokenizer, a multi-byte character
/// missing from the vocab is emitted as several `<0xXX>` byte-fallback model
/// tokens that surface as a single detokenized text piece, so a naive
/// emitted-piece count would under-count `completion_tokens` (the divergence
/// itself is pinned by the `resolve_completion_tokens` unit tests). Gemma
/// cannot serve as the fixture — see [`MINICPM_DIR`].
///
/// Gated like the other real-model tests: `#[ignore]` plus a checkpoint-presence
/// guard that skips cleanly when the MiniCPM checkpoint is absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns four real mlxcel-server processes loading minicpm-2b-4bit; run with --ignored"]
async fn disaggregated_router_chat_usage_matches_single_node_byte_fallback() {
    let model_dir = repo_model_dir(MINICPM_DIR);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {MINICPM_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download \
             mlx-community/MiniCPM-2B-sft-4bit-llama-format-mlx (place it at \
             models/{MINICPM_DIR})",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    // Same byte-fallback-forcing prompt as the completions parity test above,
    // phrased as a chat turn.
    let messages = serde_json::json!([{
        "role": "user",
        "content": "Spell the French word for coffee. It is café. Repeat it:"
    }]);
    let nonstream_body = serde_json::json!({
        "model": MINICPM_CHAT_MODEL_ALIAS,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0
    });
    let stream_body = serde_json::json!({
        "model": MINICPM_CHAT_MODEL_ALIAS,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let client = reqwest::Client::new();

    // ---- Single-node reference (started with --alias for model parity) ----
    let (ref_nonstream, ref_usage_chunk) = {
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
            MINICPM_CHAT_MODEL_ALIAS,
            "--no-warmup",
        ]);
        let deadline = Instant::now() + Duration::from_secs(240);
        let health_url = format!("http://127.0.0.1:{http}/health");
        assert!(
            wait_for_http_health(&health_url, deadline).await,
            "single-node MiniCPM reference never became healthy at {health_url}"
        );
        let chat_url = format!("http://127.0.0.1:{http}/v1/chat/completions");

        let ns_resp = client
            .post(&chat_url)
            .json(&nonstream_body)
            .send()
            .await
            .expect("POST single-node /v1/chat/completions (non-stream)");
        assert!(
            ns_resp.status().is_success(),
            "single-node non-stream chat completion returned HTTP {}",
            ns_resp.status()
        );
        let ns_json = ns_resp
            .json::<serde_json::Value>()
            .await
            .expect("parse single-node non-stream chat completion JSON");

        let st_resp = client
            .post(&chat_url)
            .json(&stream_body)
            .send()
            .await
            .expect("POST single-node /v1/chat/completions (stream)");
        assert!(
            st_resp.status().is_success(),
            "single-node stream chat completion returned HTTP {}",
            st_resp.status()
        );
        let st_body = st_resp.text().await.expect("read single-node SSE body");
        let st_chunks = parse_completion_chunks(&st_body);
        let usage_chunk = st_chunks
            .iter()
            .find(|c| !c["usage"].is_null())
            .cloned()
            .expect("single-node chat stream must carry a final usage chunk");

        (ns_json, usage_chunk)
        // _single drops here, killing the reference server.
    };
    eprintln!("single-node MiniCPM chat non-stream reference: {ref_nonstream}");
    let ref_text = ref_nonstream["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !ref_text.is_empty(),
        "single-node MiniCPM chat reference produced empty content; parity check would be vacuous"
    );
    // Guard the test's premise: the byte-fallback path is only exercised if the
    // greedy output actually contains a multi-byte character.
    assert!(
        !ref_text.is_ascii(),
        "single-node MiniCPM chat output {ref_text:?} is pure ASCII; the byte-fallback \
         count path is not exercised. Adjust the prompt so the output contains a \
         multi-byte character."
    );
    assert!(
        !ref_nonstream["usage"].is_null(),
        "single-node chat non-stream response must carry a usage object; \
         parity check would be vacuous"
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
    let router_chat_url = format!("http://127.0.0.1:{router_http}/v1/chat/completions");

    // Non-streaming parity: the router must now emit a `usage` object at all,
    // and it must match single-node exactly for the byte-fallback tokenizer.
    let router_ns_resp = client
        .post(&router_chat_url)
        .json(&nonstream_body)
        .send()
        .await
        .expect("POST router /v1/chat/completions (non-stream)");
    assert!(
        router_ns_resp.status().is_success(),
        "router non-stream chat completion returned HTTP {}",
        router_ns_resp.status()
    );
    let router_ns = router_ns_resp
        .json::<serde_json::Value>()
        .await
        .expect("parse router non-stream chat completion JSON");
    eprintln!("router MiniCPM chat non-stream: {router_ns}");
    assert!(
        !router_ns["usage"].is_null(),
        "router non-stream /v1/chat/completions must carry a usage object (issue #398)"
    );
    assert_eq!(
        router_ns["usage"]["prompt_tokens"], ref_nonstream["usage"]["prompt_tokens"],
        "router usage.prompt_tokens diverges from single-node"
    );
    assert_eq!(
        router_ns["usage"]["completion_tokens"], ref_nonstream["usage"]["completion_tokens"],
        "router usage.completion_tokens diverges from single-node for a byte-fallback tokenizer"
    );
    assert_eq!(
        router_ns["usage"]["total_tokens"], ref_nonstream["usage"]["total_tokens"],
        "router usage.total_tokens diverges from single-node"
    );
    assert_eq!(
        router_ns["choices"][0]["finish_reason"], ref_nonstream["choices"][0]["finish_reason"],
        "router finish_reason diverges from single-node for a byte-fallback tokenizer"
    );

    // Streaming parity: the final usage chunk (stream_options.include_usage)
    // must also match single-node exactly.
    let router_st_resp = client
        .post(&router_chat_url)
        .json(&stream_body)
        .send()
        .await
        .expect("POST router /v1/chat/completions (stream)");
    assert!(
        router_st_resp.status().is_success(),
        "router stream chat completion returned HTTP {}",
        router_st_resp.status()
    );
    let router_st_body = router_st_resp.text().await.expect("read router SSE body");
    let router_st_chunks = parse_completion_chunks(&router_st_body);
    let router_usage_chunk = router_st_chunks
        .iter()
        .find(|c| !c["usage"].is_null())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "router chat stream carried no usage chunk despite \
                 stream_options.include_usage=true (issue #398); chunks: {router_st_chunks:?}"
            )
        });
    assert_eq!(
        router_usage_chunk["usage"]["prompt_tokens"], ref_usage_chunk["usage"]["prompt_tokens"],
        "router stream usage.prompt_tokens diverges from single-node"
    );
    assert_eq!(
        router_usage_chunk["usage"]["completion_tokens"],
        ref_usage_chunk["usage"]["completion_tokens"],
        "router stream usage.completion_tokens diverges from single-node for a \
         byte-fallback tokenizer"
    );
    assert_eq!(
        router_usage_chunk["usage"]["total_tokens"], ref_usage_chunk["usage"]["total_tokens"],
        "router stream usage.total_tokens diverges from single-node"
    );
    // The usage chunk itself carries no choices, per the OpenAI streaming-usage
    // convention (matches the router's own /v1/completions usage chunk).
    assert_eq!(
        router_usage_chunk["choices"],
        serde_json::json!([]),
        "router chat usage chunk must carry an empty choices array"
    );
    eprintln!(
        "OK: router /v1/chat/completions matches single-node usage for a byte-fallback \
         tokenizer (non-stream + stream)."
    );
}

/// Issue #708: a `POST /v1/completions` for a MODEL-OWNED paged family (gemma3)
/// through the 3-node disaggregated stack must fail with a CLEAN per-request
/// error, and the prefill node must KEEP SERVING afterward: it must not crash
/// the serving-role loop and take the node down (which would surface as a 503
/// peer-down on every subsequent request).
///
/// Before the fix, `extract_paged_blocks` read the model-owned shadow block
/// table's unwritten pool tensors (`PagedBlockPool::read_block_contents: layer 0
/// has no pool tensors to read from`), the error propagated out of the prefill
/// serving-role loop, the loop exited, the router health monitor marked the peer
/// down, and every later request returned 503. After the fix the prefill node
/// rejects the request up front (a node-level `handoff_supported` check) with a
/// per-request error frame and stays alive.
///
/// The availability assertion: one model is served per node, so a follow-up
/// request on a SUPPORTED model is impossible on the same stack. "Keeps serving"
/// is therefore asserted by a SECOND identical request returning the SAME clean
/// per-request error (an error status that is NOT 503 peer-down, with the
/// model-owned rejection message in the body) rather than a wedged node.
///
/// Gated like the other real-model tests: `#[ignore]` plus a checkpoint-presence
/// guard that skips cleanly when the Gemma checkpoint is absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "spawns three real mlxcel-server processes loading gemma-3-1b-it-4bit; run with --ignored"]
async fn disaggregated_router_rejects_model_owned_family_and_keeps_serving() {
    let model_dir = repo_model_dir(GEMMA_DIR);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {GEMMA_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/gemma-3-1b-it-4bit",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping: mlxcel-server binary not found");
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    let client = reqwest::Client::new();

    // ---- Three-process disaggregated router run ----
    let ports = reserve_ports(6);
    let prefill_http = ports[0].to_string();
    let decode_http = ports[1].to_string();
    let router_http = ports[2].to_string();
    let prefill_serving_addr = format!("127.0.0.1:{}", ports[3]);
    let decode_serving_addr = format!("127.0.0.1:{}", ports[4]);
    let router_serving_addr = format!("127.0.0.1:{}", ports[5]);

    let _decode = spawn_decode(&model_arg, &decode_http, &decode_serving_addr);
    let _prefill = spawn_prefill(
        &model_arg,
        &prefill_http,
        &prefill_serving_addr,
        &decode_serving_addr,
    );
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
    let completions_url = format!("http://127.0.0.1:{router_http}/v1/completions");
    let body = serde_json::json!({
        "model": GEMMA_MODEL_ALIAS,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0
    });

    // ---- Request #1: a clean per-request error (not a hang, not success) ----
    let resp1 = client
        .post(&completions_url)
        .json(&body)
        .send()
        .await
        .expect("POST router /v1/completions (#1)");
    let status1 = resp1.status();
    let body1 = resp1.text().await.unwrap_or_default();
    eprintln!("model-owned request #1 -> HTTP {status1}: {body1}");
    assert!(
        status1.is_client_error() || status1.is_server_error(),
        "model-owned request should return a clean 4xx/5xx error, got HTTP {status1}"
    );
    assert_ne!(
        status1,
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "first model-owned request returned 503 (peer-down); the prefill node should reject \
         per-request, not be marked down"
    );
    assert!(
        body1.contains("model-owned") || body1.contains("handoff"),
        "error body should explain the model-owned handoff rejection, got: {body1}"
    );

    // Give a hypothetically-crashed node time to be marked down by the router's
    // health monitor, so request #2 is a genuine "is the node still serving?"
    // probe (mirrors the failover test's post-kill settle delay).
    tokio::time::sleep(Duration::from_secs(2)).await;

    // ---- Request #2: proves the prefill node kept serving ----
    // A crashed node would be marked down by now, wedging this into a 503
    // peer-down (or a transport error) with no model-owned rejection body. A
    // live node returns the SAME clean per-request rejection.
    let resp2 = client
        .post(&completions_url)
        .json(&body)
        .send()
        .await
        .expect("POST router /v1/completions (#2)");
    let status2 = resp2.status();
    let body2 = resp2.text().await.unwrap_or_default();
    eprintln!("model-owned request #2 -> HTTP {status2}: {body2}");
    assert_ne!(
        status2,
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "second model-owned request returned 503 (peer-down): the prefill node did not keep \
         serving after the first rejected request (issue #708 regression)"
    );
    assert!(
        body2.contains("model-owned") || body2.contains("handoff"),
        "second request must carry the same clean model-owned rejection, proving the node is \
         still serving; got HTTP {status2}: {body2}"
    );
    assert_eq!(
        status1, status2,
        "the prefill node must reject both requests identically; got {status1} then {status2}"
    );
    eprintln!(
        "OK: model-owned family rejected per-request with a clean error and the prefill node \
         kept serving (no 503 peer-down)."
    );
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
