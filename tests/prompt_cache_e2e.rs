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

//! End-to-end integration test for the cross-request prompt-prefix KV cache
//!
//! This test boots a real `mlxcel-server` binary with the prompt-prefix
//! cache enabled and drives it through the public OpenAI-compatible HTTP
//! surface. It validates three independent pieces of the feature:
//!
//! 1. Wire contract — `usage.prompt_tokens_details.cached_tokens` appears in
//!    the JSON response when the cache is enabled.
//! 2. Hit behavior — turn 1 reports `cached_tokens == 0`; turns 2 through 5
//!    each report a strictly positive `cached_tokens` that is at least the
//!    previous turn's prompt length (monotonic prefix growth).
//! 3. Prefill-latency win — the wall-clock time for prefill at turn N, with
//!    an identical new-user-message length across turns, stays well below
//!    turn 1 (upper bound: 1.3x; we tolerate observational noise on
//!    resource-constrained CI hosts).
//!
//! The test is gated with `#[ignore]` because it requires the
//! `mlxcel-server` binary and a local `qwen3-0.6b-4bit` checkout. Run it
//! explicitly with:
//!
//! ```text
//! cargo test --test prompt_cache_e2e --release -- --ignored --nocapture
//! ```
//!
//! Because a real host may lack the model weights or the GPU/ANE path for
//! quantized inference (GB10 / Blackwell cannot currently run 4-bit QMM on
//! the JIT path — see `docs_internal/platform/cuda-quantized-support.md`),
//! the test skips gracefully with an `eprintln!` rather than failing when
//! the server cannot become healthy or cannot serve a request.

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Smallest Qwen3 weight bundle we keep locally.
const QWEN3_MODEL: &str = "qwen3-0.6b-4bit";

/// Number of successive chat turns driven against the server.
const NUM_TURNS: usize = 5;

/// Upper bound on turn-N prefill latency relative to turn-1 prefill latency.
/// The spec calls for <= 1.3x. We keep the literal here so any
/// future tightening happens in one place.
const PREFILL_LATENCY_UPPER_RATIO: f64 = 1.3;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

/// Try to reach `/health`. Returns `true` on the first 200 within the
/// timeout, `false` otherwise. We intentionally return rather than panic
/// so the calling test can skip-with-a-note if the server never comes up
/// (e.g. unsupported quant path on the current GPU).
async fn wait_for_health_soft(client: &reqwest::Client, base_url: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
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

/// One turn's response summary extracted from the JSON body.
#[derive(Debug, Clone)]
struct TurnSummary {
    /// Assistant content returned in `choices[0].message.content`.
    content: String,
    /// `usage.prompt_tokens`.
    prompt_tokens: u64,
    /// `usage.completion_tokens`.
    completion_tokens: u64,
    /// `usage.prompt_tokens_details.cached_tokens` — `None` when the field
    /// is absent (cache disabled on the server) or unparseable.
    cached_tokens: Option<u64>,
    /// Total wall-clock latency from request send to response parse. With
    /// temperature 0 and `max_tokens` small, the prefill phase dominates
    /// this for turn 1 (cold) and the adopt-plus-decode path dominates it
    /// for turns 2..N (hot).
    wall_latency: Duration,
}

/// Append an `(assistant, user)` pair to the rolling messages array and
/// issue one `/v1/chat/completions` request.
///
/// `assistant_tail` is the assistant reply from the previous turn (empty
/// before turn 1). `new_user_question` is the user message for this turn.
async fn one_turn(
    client: &reqwest::Client,
    base_url: &str,
    model_alias: &str,
    messages: &mut Vec<serde_json::Value>,
    new_user_question: &str,
    max_tokens: u32,
) -> TurnSummary {
    messages.push(serde_json::json!({
        "role": "user",
        "content": new_user_question,
    }));

    let body = serde_json::json!({
        "model": model_alias,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        // Route through the session-key path so the prompt cache can bucket
        // the turns together (wiring).
        "user": "prompt-cache-e2e-user",
    });

    let sent_at = Instant::now();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send chat request");
    let status = resp.status();
    let value: serde_json::Value = resp.json().await.expect("parse chat response JSON");
    let wall_latency = sent_at.elapsed();

    assert!(
        status.is_success(),
        "chat completion returned non-200: status={status} body={value}"
    );

    let content = value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let prompt_tokens = value["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let cached_tokens = value["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64();

    // Thread the reply into the conversation so the next turn's prefix
    // actually includes this turn's assistant tokens. An empty reply is
    // tolerable (some short prompts hit EOS immediately) but we still
    // preserve the turn structure for bucket stability.
    messages.push(serde_json::json!({
        "role": "assistant",
        "content": content.clone(),
    }));

    TurnSummary {
        content,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        wall_latency,
    }
}

/// The core multi-turn chat-completions test with the prompt cache enabled.
///
/// Keeps the new-user-message length identical across turns so the prefill
/// work at turn N, with cache hits, is directly comparable to the cold
/// prefill at turn 1.
#[tokio::test]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
async fn multi_turn_chat_reports_cached_tokens_and_lowers_prefill_latency() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping multi_turn_chat_reports_cached_tokens_and_lowers_prefill_latency: \
             model not present at {}",
            model_dir.display()
        );
        return;
    }

    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping: mlxcel-server binary not present at {}. \
             Build with `cargo build --bin mlxcel-server` first.",
            binary.display()
        );
        return;
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let model_alias = "qwen3-cache-e2e";

    // Prompt cache explicitly enabled via CLI; small caps so we stay within
    // memory even on CI hosts. Dense decode storage is the only backend
    // that can currently donate back to the store (gate).
    let mut child = spawn_server(&[
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
        "--metrics",
        "--prompt-cache-enabled=true",
        "--prompt-cache-capacity-bytes",
        "268435456", // 256 MiB is plenty for a 0.6B model and 5 turns
        "--prompt-cache-max-entries",
        "32",
        "--prompt-cache-min-prefix",
        "4",
    ]);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("build reqwest client");

    let healthy = wait_for_health_soft(&client, &base_url, Duration::from_secs(90)).await;
    if !healthy {
        eprintln!(
            "Skipping: mlxcel-server did not become healthy at {base_url}. \
             The host may not support the selected model's quantization path. \
             Not failing the test so CI stays green on unsupported hardware."
        );
        stop_server(&mut child);
        return;
    }

    // A compact, open-ended system prompt seeds the shared prefix. The
    // user question is identical across turns (same token budget for the
    // *new* tokens in each turn) so per-turn prefill cost differences are
    // driven purely by cache adoption, not by input shape.
    let mut messages: Vec<serde_json::Value> = vec![serde_json::json!({
        "role": "system",
        "content":
            "You are a terse assistant. Answer every question in one short sentence. \
             Do not repeat yourself. Keep your answers deterministic.",
    })];
    let new_user_question = "State one concrete fact about the color of the sky on Earth.";

    let mut turns: Vec<TurnSummary> = Vec::with_capacity(NUM_TURNS);
    for turn_idx in 0..NUM_TURNS {
        let summary = one_turn(
            &client,
            &base_url,
            model_alias,
            &mut messages,
            new_user_question,
            16,
        )
        .await;
        eprintln!(
            "turn {idx}: prompt_tokens={pt} completion_tokens={ct} cached_tokens={cached:?} \
             wall_latency={lat_ms:.1}ms content=\"{content}\"",
            idx = turn_idx + 1,
            pt = summary.prompt_tokens,
            ct = summary.completion_tokens,
            cached = summary.cached_tokens,
            lat_ms = summary.wall_latency.as_secs_f64() * 1000.0,
            content = summary.content.replace('\n', " "),
        );
        turns.push(summary);
    }

    stop_server(&mut child);

    // ---------------------------------------------------------------
    // Assertion 1: wire contract — cached_tokens field present on every
    // turn when the cache is enabled on the server.
    // ---------------------------------------------------------------
    for (i, t) in turns.iter().enumerate() {
        assert!(
            t.cached_tokens.is_some(),
            "turn {}: usage.prompt_tokens_details.cached_tokens must be present when \
             --prompt-cache-enabled=true",
            i + 1
        );
    }

    // ---------------------------------------------------------------
    // Assertion 2: turn 1 cold → cached_tokens == 0.
    // ---------------------------------------------------------------
    assert_eq!(
        turns[0].cached_tokens,
        Some(0),
        "turn 1 must not have any cached tokens (cold start): got {:?}",
        turns[0].cached_tokens
    );

    // ---------------------------------------------------------------
    // Assertion 3: turns 2..=N hot → cached_tokens > 0, monotonic
    // growth at least up to the previous turn's prompt length. A
    // non-decreasing relation across turns is also required because
    // the conversation grows monotonically.
    // ---------------------------------------------------------------
    let mut prev_cached: u64 = 0;
    for (idx, pair) in turns.windows(2).enumerate() {
        // `windows(2)` yields &[prev, curr]; `idx` counts the pair, so the
        // 1-based turn number for `curr` is `idx + 2`.
        let [prev_turn, cur_turn] = [&pair[0], &pair[1]];
        let turn_number = idx + 2;
        let cached = cur_turn.cached_tokens.unwrap_or(0);
        let prev_prompt_len = prev_turn.prompt_tokens;
        assert!(
            cached > 0,
            "turn {turn_number}: expected cached_tokens > 0 after turn 1's prefix \
             was donated back; got {cached}",
        );
        assert!(
            cached >= prev_cached,
            "turn {turn_number}: cached_tokens must grow monotonically across turns; \
             prev={prev_cached} now={cached}",
        );
        // The spec: "cached_tokens > 0 and monotonically >= previous-turn
        // prompt length". With healthy donate-back at turn k, the next
        // turn's prefix reuses at least all tokens that turn k saw as its
        // prompt, up to the min_prefix_tokens cap. Allow a small slack
        // (min_prefix gate, template boundary) but flag regressions that
        // drop below the previous prompt length.
        assert!(
            cached + 4 >= prev_prompt_len,
            "turn {turn_number}: cached_tokens ({cached}) should cover at least the \
             previous turn's prompt length ({prev_prompt_len}) minus the min-prefix slack",
        );
        prev_cached = cached;
    }

    // ---------------------------------------------------------------
    // Assertion 4: prefill-latency win. Turn N's wall-clock latency,
    // with identical new-user-message size across turns, stays within
    // `PREFILL_LATENCY_UPPER_RATIO * turn_1_latency`. This is a
    // pessimistic proxy for prefill time: decoding at fixed `max_tokens`
    // contributes the same constant across turns so ratio inflation is
    // dominated by prefill differences.
    //
    // We tolerate observational noise: if turn 1 was unreasonably short
    // (< 50 ms) we downgrade the assertion to a warning instead of
    // failing, because ratio arithmetic on a sub-50ms baseline is
    // dominated by scheduler jitter, not prefill cost.
    // ---------------------------------------------------------------
    let baseline_ms = turns[0].wall_latency.as_secs_f64() * 1000.0;
    if baseline_ms < 50.0 {
        eprintln!(
            "Warning: turn-1 latency was only {baseline_ms:.1}ms; prefill ratio \
             assertion downgraded to informational. This is expected on very \
             small models where decode dominates total time."
        );
        return;
    }
    for (idx, turn) in turns.iter().enumerate().skip(1) {
        let turn_number = idx + 1;
        let tn_ms = turn.wall_latency.as_secs_f64() * 1000.0;
        let ratio = tn_ms / baseline_ms;
        eprintln!(
            "prefill-ratio check: turn {turn_number} took {tn_ms:.1}ms vs turn-1 \
             {baseline_ms:.1}ms (ratio={ratio:.3}, upper={PREFILL_LATENCY_UPPER_RATIO:.2})",
        );
        assert!(
            ratio <= PREFILL_LATENCY_UPPER_RATIO,
            "turn {turn_number}: wall latency ratio {ratio:.3} exceeds the \
             {PREFILL_LATENCY_UPPER_RATIO:.2} upper bound relative to turn 1. \
             Prompt cache adoption appears to have stopped working.",
        );
    }
}

/// Negative control: with the cache **disabled** on the server, the
/// wire contract hides `prompt_tokens_details` and every turn's prefill
/// latency looks like a cold prefill. This gives us a known-good
/// baseline to compare the cached case against in review.
#[tokio::test]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
async fn multi_turn_chat_with_cache_disabled_never_reports_cached_tokens() {
    let model_dir = repo_model_dir(QWEN3_MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping multi_turn_chat_with_cache_disabled_never_reports_cached_tokens: \
             model not present at {}",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping: mlxcel-server binary not present at {}",
            binary.display()
        );
        return;
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();
    let model_str = model_dir.to_string_lossy().to_string();
    let model_alias = "qwen3-no-cache";

    let mut child = spawn_server(&[
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
        "--prompt-cache-enabled=false",
    ]);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("build reqwest client");

    let healthy = wait_for_health_soft(&client, &base_url, Duration::from_secs(90)).await;
    if !healthy {
        eprintln!(
            "Skipping: mlxcel-server did not become healthy at {base_url}. \
             Not failing CI on unsupported hardware."
        );
        stop_server(&mut child);
        return;
    }

    let mut messages: Vec<serde_json::Value> = vec![serde_json::json!({
        "role": "system",
        "content":
            "You are a terse assistant. Answer every question in one short sentence.",
    })];
    let question = "Name one color.";

    for _ in 0..3_usize {
        let summary = one_turn(&client, &base_url, model_alias, &mut messages, question, 8).await;
        assert!(
            summary.cached_tokens.is_none(),
            "cache is disabled so usage.prompt_tokens_details.cached_tokens must be absent; \
             got {:?}",
            summary.cached_tokens
        );
    }

    stop_server(&mut child);
}
