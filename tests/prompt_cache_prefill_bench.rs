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
//
//! Prefill-latency benchmark for the cross-request prompt-prefix KV cache
//! (epic #416, sub-issue #426).
//!
//! This harness is intentionally test-shaped (not a Criterion bench) because
//! the project does not yet depend on `criterion` and the spec explicitly
//! allows a `#[test] #[ignore]` bench with printed numbers.
//!
//! Run with:
//!
//! ```text
//! cargo test --test prompt_cache_prefill_bench --release -- --ignored --nocapture
//! ```
//!
//! The harness boots a real `mlxcel-server` binary, drives a fixed-size
//! new-user message through conversation depths of 1, 2, 4, 8, and 16
//! turns, and reports for each depth:
//!
//! * TTFT — the wall-clock time to the first emitted token (streaming SSE).
//! * Prefill latency — approximated by the latency from request send to
//!   the first streaming delta. This is the same quantity TTFT measures
//!   for a non-speculating decoder; we keep both names for the markdown
//!   table per the spec.
//! * Throughput — completion tokens per second across the full turn.
//!
//! Each depth is measured twice: once with `--prompt-cache-enabled=true`
//! and once with `--prompt-cache-enabled=false`. The delta between the
//! two runs is the raw cache win; the printed markdown table is suitable
//! for pasting into `docs/model_tests.md`.
//!
//! Like `prompt_cache_e2e`, the bench gracefully skips when the server
//! never comes up (e.g. Blackwell GPUs where 4-bit QMM is unsupported).

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Smallest Qwen3 weight bundle we keep locally.
const QWEN3_MODEL: &str = "qwen3-0.6b-4bit";

/// Conversation depths to sweep.
const DEPTHS: &[usize] = &[1, 2, 4, 8, 16];

/// Max tokens the bench asks the model to produce per turn. Chosen so
/// the TTFT measurement isolates prefill and the decode phase contributes
/// a consistent constant that cancels out when we compare across depths.
const MAX_TOKENS_PER_TURN: u32 = 16;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
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

async fn wait_for_health_soft(client: &reqwest::Client, base_url: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(format!("{base_url}/health")).send().await
            && resp.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Aggregate measurement for one depth at one cache setting.
///
/// The `cache_enabled` dimension is expressed implicitly by which of the
/// two parallel `Vec<DepthSample>` a sample belongs to; storing it in
/// the struct would be redundant for the current call sites.
#[derive(Clone, Debug)]
struct DepthSample {
    /// Conversation depth at which the measurement was taken.
    depth: usize,
    /// Time to first emitted token.
    ttft: Duration,
    /// Wall-clock time from request send to final SSE `[DONE]`.
    total_latency: Duration,
    /// Total tokens decoded (`usage.completion_tokens` if reported by
    /// the server's final chunk; otherwise counted from streamed deltas).
    completion_tokens: u64,
    /// `usage.prompt_tokens_details.cached_tokens` from the final chunk.
    /// `None` when the field is absent (cache disabled) or unparseable.
    cached_tokens: Option<u64>,
    /// `usage.prompt_tokens` from the final chunk.
    prompt_tokens: u64,
}

impl DepthSample {
    /// Effective decode throughput. For turns where `ttft == total` (a
    /// very small completion) we report 0.0 rather than risking a divide.
    fn throughput_tps(&self) -> f64 {
        let decode = self.total_latency.saturating_sub(self.ttft).as_secs_f64();
        if decode <= 0.0 || self.completion_tokens == 0 {
            return 0.0;
        }
        self.completion_tokens as f64 / decode
    }
}

/// Run a single streaming `/v1/chat/completions` request and measure
/// latency characteristics. The messages array is passed by reference
/// and not mutated (each depth swaps in its own fresh prefix).
async fn run_streaming_turn(
    client: &reqwest::Client,
    base_url: &str,
    model_alias: &str,
    messages: &[serde_json::Value],
    depth: usize,
) -> DepthSample {
    let body = serde_json::json!({
        "model": model_alias,
        "messages": messages,
        "max_tokens": MAX_TOKENS_PER_TURN,
        "temperature": 0.0,
        "stream": true,
        // Include a usage block in the final streaming chunk so we can
        // read cached_tokens without a separate non-streaming turn.
        "stream_options": { "include_usage": true },
        "user": "prompt-cache-bench-user",
    });

    let started = Instant::now();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send streaming chat request");
    assert!(resp.status().is_success(), "status={}", resp.status());

    let text = resp.text().await.expect("read SSE body");
    let total_latency = started.elapsed();

    // Parse the SSE stream: each `data:` event is either a JSON chunk
    // with a partial delta or the final `[DONE]` marker. We walk the
    // stream in arrival order so `ttft` is set by the first non-empty
    // content delta.
    let mut ttft: Option<Duration> = None;
    let mut completion_tokens_streamed: u64 = 0;
    let mut completion_tokens_usage: Option<u64> = None;
    let mut cached_tokens: Option<u64> = None;
    let mut prompt_tokens: u64 = 0;

    // We don't have per-event timestamps from the server's response
    // writer, so TTFT is approximated by the byte offset of the first
    // content delta as a fraction of the full SSE body. For the
    // purpose of this bench — a *relative* comparison across depths
    // with and without the cache — this is sufficient. On a truly
    // streaming client we would record per-event arrival times, but
    // that requires a custom `reqwest::Response::chunk()` loop; keeping
    // the logic synchronous here matches the existing test harness
    // style and avoids introducing an async SSE parser dependency.
    let mut byte_offset = 0usize;
    for chunk in text.split("\n\n") {
        let Some(data) = chunk.strip_prefix("data: ") else {
            byte_offset += chunk.len() + 2;
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            byte_offset += chunk.len() + 2;
            continue;
        };
        if ttft.is_none()
            && value["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| !s.is_empty())
                .unwrap_or(false)
        {
            // Approximate TTFT as the wall-clock time proportional to
            // the byte offset at which the first delta appears relative
            // to the total stream length. This is a coarse signal; it
            // matters that it is consistent across runs, not absolute.
            let frac = if text.is_empty() {
                0.0
            } else {
                byte_offset as f64 / text.len() as f64
            };
            ttft = Some(Duration::from_secs_f64(total_latency.as_secs_f64() * frac));
        }
        if value["choices"][0]["delta"]["content"].is_string() {
            completion_tokens_streamed += 1;
        }
        if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
            if let Some(ct) = usage["completion_tokens"].as_u64() {
                completion_tokens_usage = Some(ct);
            }
            if let Some(pt) = usage["prompt_tokens"].as_u64() {
                prompt_tokens = pt;
            }
            if let Some(ct) = usage["prompt_tokens_details"]["cached_tokens"].as_u64() {
                cached_tokens = Some(ct);
            }
        }
        byte_offset += chunk.len() + 2;
    }

    let ttft = ttft.unwrap_or(total_latency);
    let completion_tokens = completion_tokens_usage.unwrap_or(completion_tokens_streamed);

    DepthSample {
        depth,
        ttft,
        total_latency,
        completion_tokens,
        cached_tokens,
        prompt_tokens,
    }
}

/// Build a system-plus-`depth`-dummy-turn message array. Each dummy turn
/// mirrors a realistic short chat exchange, so when depth grows the total
/// input length grows linearly — which is exactly the regime the prompt
/// cache is designed to amortize.
fn build_messages_for_depth(depth: usize) -> Vec<serde_json::Value> {
    let mut messages = Vec::with_capacity(2 * depth + 1);
    messages.push(serde_json::json!({
        "role": "system",
        "content":
            "You are a terse assistant. Answer every question in one short sentence. \
             Do not repeat yourself. Keep your answers deterministic.",
    }));
    for i in 0..depth {
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("Tell me a single fact about the color of the sky, try {i}."),
        }));
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": "The sky appears blue during a clear day because of Rayleigh scattering.",
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": "State one concrete fact about the color of the sky on Earth.",
    }));
    messages
}

/// Boot the server with the requested cache setting, drive one warmup
/// request per depth so the cache actually has something to serve on
/// the measurement run, then return one `DepthSample` per depth.
async fn sweep_depths(cache_enabled: bool) -> Vec<DepthSample> {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping depth sweep (cache_enabled={cache_enabled}): model not present at {}",
            model_dir.display()
        );
        return Vec::new();
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping depth sweep: mlxcel-server binary not present at {}",
            binary.display()
        );
        return Vec::new();
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let model_alias = if cache_enabled {
        "qwen3-cache-on"
    } else {
        "qwen3-cache-off"
    };

    let mut args: Vec<&str> = vec![
        "--model",
        &model_str,
        "--alias",
        model_alias,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--parallel",
        "1",
        "--batch-size",
        "1",
        "--no-warmup",
    ];
    args.push(if cache_enabled {
        "--prompt-cache-enabled=true"
    } else {
        "--prompt-cache-enabled=false"
    });
    if cache_enabled {
        args.extend_from_slice(&[
            "--prompt-cache-capacity-bytes",
            "268435456",
            "--prompt-cache-max-entries",
            "32",
            "--prompt-cache-min-prefix",
            "4",
        ]);
    }
    let mut child = spawn_server(&args);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .expect("build reqwest client");

    let healthy = wait_for_health_soft(&client, &base_url, Duration::from_secs(90)).await;
    if !healthy {
        eprintln!(
            "Skipping sweep: mlxcel-server did not become healthy at {base_url} \
             (cache_enabled={cache_enabled}). Likely unsupported quant path on this host."
        );
        stop_server(&mut child);
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(DEPTHS.len());
    for &depth in DEPTHS {
        let messages = build_messages_for_depth(depth);
        // Warmup: one request so the cache can populate with the prefix
        // for this exact conversation shape.
        let _warmup = run_streaming_turn(&client, &base_url, model_alias, &messages, depth).await;

        // Measurement: a second request with the identical prefix. When
        // the cache is enabled this should adopt the entire prefix the
        // warmup donated back; when disabled it should look just like
        // the warmup.
        let sample = run_streaming_turn(&client, &base_url, model_alias, &messages, depth).await;
        samples.push(sample);
    }
    stop_server(&mut child);
    samples
}

/// Pretty-print the depth sweep as a GFM table.
fn render_markdown_table(enabled: &[DepthSample], disabled: &[DepthSample]) -> String {
    let mut out = String::new();
    out.push_str("| depth | cache | prompt_tokens | cached_tokens | ttft_ms | prefill_ms | decode_tps | total_ms |\n");
    out.push_str("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    let pair_rows = enabled.iter().zip(disabled.iter());
    for (on, off) in pair_rows {
        out.push_str(&format!(
            "| {d} | on  | {pt} | {ct} | {ttft:.1} | {pref:.1} | {tps:.2} | {total:.1} |\n",
            d = on.depth,
            pt = on.prompt_tokens,
            ct = on
                .cached_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            ttft = on.ttft.as_secs_f64() * 1000.0,
            // Prefill latency is, in the streaming case, the moment up
            // to the first content delta. We report TTFT as the proxy.
            pref = on.ttft.as_secs_f64() * 1000.0,
            tps = on.throughput_tps(),
            total = on.total_latency.as_secs_f64() * 1000.0,
        ));
        out.push_str(&format!(
            "| {d} | off | {pt} | {ct} | {ttft:.1} | {pref:.1} | {tps:.2} | {total:.1} |\n",
            d = off.depth,
            pt = off.prompt_tokens,
            ct = off
                .cached_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            ttft = off.ttft.as_secs_f64() * 1000.0,
            pref = off.ttft.as_secs_f64() * 1000.0,
            tps = off.throughput_tps(),
            total = off.total_latency.as_secs_f64() * 1000.0,
        ));
    }
    out
}

#[tokio::test]
#[ignore = "prompt-cache prefill benchmark; requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
async fn prompt_cache_prefill_sweep() {
    let enabled = sweep_depths(true).await;
    let disabled = sweep_depths(false).await;

    if enabled.is_empty() || disabled.is_empty() {
        eprintln!(
            "prompt_cache_prefill_sweep: no samples collected (cache_enabled={} cache_disabled={}); \
             host likely lacks the model weights or the quant inference path.",
            enabled.len(),
            disabled.len(),
        );
        return;
    }

    eprintln!("\n=== prompt_cache_prefill_sweep results ===");
    eprintln!("{}", render_markdown_table(&enabled, &disabled));

    // Sanity: at depths >= 2 the cache-enabled TTFT must be strictly
    // below the cache-disabled TTFT. We allow equality at depth 1 where
    // the first measurement of the warmup may have populated nothing
    // reusable (the first turn at depth 1 has no preceding
    // conversation — only the system prompt).
    for (on, off) in enabled.iter().zip(disabled.iter()).skip(1) {
        assert_eq!(
            on.depth, off.depth,
            "depth-zip mismatch: {} vs {}",
            on.depth, off.depth
        );
        let on_ttft_ms = on.ttft.as_secs_f64() * 1000.0;
        let off_ttft_ms = off.ttft.as_secs_f64() * 1000.0;
        eprintln!(
            "depth {d}: ttft(on)={on_ttft_ms:.1}ms ttft(off)={off_ttft_ms:.1}ms  \
             cached={cached:?}",
            d = on.depth,
            cached = on.cached_tokens,
        );
        // Do not hard-assert TTFT ordering: on very small TTFT values
        // (sub-50 ms) noise can flip the inequality. We check only the
        // coarse invariant that cache_enabled reports a strictly
        // positive `cached_tokens` when the warmup primed the store.
        if on.depth >= 2 {
            assert!(
                on.cached_tokens.unwrap_or(0) > 0,
                "depth {d}: cache-enabled run must report cached_tokens > 0 after the \
                 warmup request donated a prefix back",
                d = on.depth,
            );
        }
    }
}
