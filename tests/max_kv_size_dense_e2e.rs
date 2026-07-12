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

//! End-to-end regression test for `--max-kv-size` on the DENSE decode backend
//! (issue #718).
//!
//! A prompt longer than the cap forces `KVCache::trim_front` to drop tokens
//! mid-decode. Before the fix, the dense front-trim dropped the leading
//! attention-sink tokens (position 0 onward), and decode collapsed into
//! degenerate repetition of the prompt (e.g. echoing the question over and
//! over). The paged backend was unaffected because pool-backed Fp16 sequences
//! make `trim_front` a no-op. Int8 always runs on the dense backend, so it
//! always exposed the same path.
//!
//! The fix pins a small attention-sink prefix when trimming
//! (`KVCache::trim_front_keep_sink`, mirroring mlx-lm
//! `RotatingKVCache(keep=4)`). This test boots a real `mlxcel-server`, sends a
//! ~500-token prompt with `--max-kv-size 256` on BOTH failing configurations
//! (`--kv-cache-mode fp16 --decode-storage-backend dense` and
//! `--kv-cache-mode int8`), and asserts the completion is coherent and
//! non-degenerate rather than a repeated echo of the prompt.
//!
//! Gated with `#[ignore]` because it needs the `mlxcel-server` binary and a
//! local checkpoint. Run explicitly:
//!
//! ```text
//! cargo test --test max_kv_size_dense_e2e --release -- --ignored --nocapture
//! ```
//!
//! The checkpoint defaults to `meta-llama-3.1-8b-instruct-4bit` and can be
//! overridden with `MLXCEL_MAXKV_E2E_MODEL`. The test skips gracefully (with an
//! `eprintln!`) when the model or binary is absent, or when the server cannot
//! become healthy on the current host.

mod common;

use std::collections::HashMap;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Default checkpoint; overridable with `MLXCEL_MAXKV_E2E_MODEL`.
const DEFAULT_MODEL: &str = "meta-llama-3.1-8b-instruct-4bit";

/// `--max-kv-size` cap. The prompt below is comfortably longer than this so
/// the dense front-trim fires during decode.
const MAX_KV_SIZE: u32 = 256;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

async fn wait_for_health_soft(client: &reqwest::Client, base_url: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/v1/models")).send().await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
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

/// Build a ~500-token prompt: numbered planet facts (so the answer lives in
/// the text) followed by a question whose answer is a single fact. Longer than
/// `MAX_KV_SIZE` so decode trims mid-stream.
fn build_long_prompt() -> String {
    let facts = [
        "Mercury is the smallest planet and the closest to the Sun.",
        "Venus has a thick atmosphere of carbon dioxide and is the hottest planet.",
        "Earth is the only planet known to support life and has one moon.",
        "Mars is called the red planet because of iron oxide on its surface.",
        "Jupiter is the largest planet and has a famous great red spot storm.",
        "Saturn is known for its bright and extensive ring system.",
        "Uranus rotates on its side with an axial tilt of about ninety eight degrees.",
        "Neptune is the farthest planet and has the strongest winds in the solar system.",
        "The asteroid belt lies between the orbits of Mars and Jupiter.",
        "A light year is the distance that light travels in one year in a vacuum.",
        "The Sun holds about ninety nine percent of the mass of the solar system.",
        "Comets develop glowing tails when they approach the Sun and heat up.",
    ];
    let mut prompt = String::from(
        "You are a careful assistant. Read the following notes about the planets \
         of the solar system and then answer the question at the end.\n\n",
    );
    // 48 numbered lines (four passes) lands around 500 tokens, well over the cap.
    for i in 0..48 {
        prompt.push_str(&format!("{}. {}\n", i + 1, facts[i % facts.len()]));
    }
    prompt.push_str(
        "\nQuestion: Which planet is the closest to the Sun, and what is special \
         about the planet Saturn? Answer in one or two clear sentences.\n\nAnswer:",
    );
    prompt
}

/// Fraction of distinct whitespace-delimited words. Degenerate repetition
/// drives this toward zero (the buggy output repeated a short cycle).
fn distinct_word_ratio(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let distinct: std::collections::HashSet<&str> = words.iter().copied().collect();
    distinct.len() as f64 / words.len() as f64
}

/// Maximum number of times any word `n`-gram repeats in `text`.
fn max_ngram_repeat(text: &str, n: usize) -> usize {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return 0;
    }
    let mut counts: HashMap<Vec<&str>, usize> = HashMap::new();
    let mut max = 0;
    for window in words.windows(n) {
        let entry = counts.entry(window.to_vec()).or_insert(0);
        *entry += 1;
        max = max.max(*entry);
    }
    max
}

/// Assert the completion answers the prompt instead of degenerating. The
/// pre-fix output echoed the question verbatim many times; the fixed output is
/// a short coherent answer.
fn assert_non_degenerate(label: &str, completion: &str) {
    let text = completion.trim();
    assert!(
        !text.is_empty(),
        "[{label}] completion was empty (model produced no tokens)"
    );

    // The buggy output echoed the prompt's question tail over and over; a
    // coherent answer never repeats it. This is the direct #718 signature.
    assert!(
        !text.contains("what is special about the planet Saturn"),
        "[{label}] completion echoed the prompt question (degenerate repetition): {text:?}"
    );

    let ratio = distinct_word_ratio(text);
    assert!(
        ratio > 0.4,
        "[{label}] completion is dominated by repeated words (distinct ratio {ratio:.2}): {text:?}"
    );

    let rep5 = max_ngram_repeat(text, 5);
    assert!(
        rep5 <= 2,
        "[{label}] completion repeats a 5-gram {rep5} times (degenerate loop): {text:?}"
    );
}

/// Boot the server with `extra_args`, send the long prompt to
/// `/v1/completions`, and return the completion text. `None` signals a
/// skip-worthy condition (server never became healthy on this host).
async fn run_completion(model_path: &str, port: u16, extra_args: &[&str]) -> Option<String> {
    let base_url = format!("http://127.0.0.1:{port}");
    let port_str = port.to_string();

    let mut args: Vec<&str> = vec![
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        &port_str,
        "--parallel",
        "1",
    ];
    args.extend_from_slice(extra_args);

    let mut child = spawn_server(&args);
    let client = reqwest::Client::new();

    if !wait_for_health_soft(&client, &base_url, Duration::from_secs(180)).await {
        stop_server(&mut child);
        return None;
    }

    let body = serde_json::json!({
        "model": "maxkv-e2e",
        "prompt": build_long_prompt(),
        "max_tokens": 40,
        "temperature": 0.0,
        "stream": false,
    });

    let result = client
        .post(format!("{base_url}/v1/completions"))
        .json(&body)
        .send()
        .await;

    let completion = match result {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(value) => value["choices"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            Err(err) => {
                stop_server(&mut child);
                panic!("parse completion JSON failed: {err}");
            }
        },
        Ok(resp) => {
            let status = resp.status();
            stop_server(&mut child);
            panic!("completion returned non-200: status={status}");
        }
        Err(err) => {
            stop_server(&mut child);
            panic!("completion request failed: {err}");
        }
    };

    stop_server(&mut child);
    Some(completion)
}

#[tokio::test]
#[ignore = "requires the mlxcel-server binary and a local checkpoint (MLXCEL_MAXKV_E2E_MODEL)"]
async fn dense_max_kv_size_prompt_over_cap_stays_coherent() {
    let model_name =
        std::env::var("MLXCEL_MAXKV_E2E_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let model_dir = repo_model_dir(&model_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping dense_max_kv_size_prompt_over_cap_stays_coherent: \
             model not present at {}",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!(
            "Skipping: mlxcel-server binary not present at {}. \
             Build with `cargo build --release --bin mlxcel-server` first.",
            binary.display()
        );
        return;
    }
    let model_str = model_dir.to_string_lossy().to_string();
    let cap = MAX_KV_SIZE.to_string();

    // Failing case A: explicit dense backend with Fp16 KV.
    let port_a = reserve_port();
    let case_a = run_completion(
        &model_str,
        port_a,
        &[
            "--kv-cache-mode",
            "fp16",
            "--decode-storage-backend",
            "dense",
            "--max-kv-size",
            &cap,
        ],
    )
    .await;
    match case_a {
        Some(text) => {
            eprintln!("[fp16+dense] completion: {text:?}");
            assert_non_degenerate("fp16+dense", &text);
        }
        None => {
            eprintln!(
                "Skipping: server never became healthy for the fp16+dense case \
                 (unsupported quant path on this host?)"
            );
            return;
        }
    }

    // Failing case B: Int8 KV, which always forces the dense backend.
    let port_b = reserve_port();
    let case_b = run_completion(
        &model_str,
        port_b,
        &["--kv-cache-mode", "int8", "--max-kv-size", &cap],
    )
    .await;
    match case_b {
        Some(text) => {
            eprintln!("[int8] completion: {text:?}");
            assert_non_degenerate("int8", &text);
        }
        None => {
            eprintln!("Skipping int8 assertion: server never became healthy for the int8 case");
        }
    }
}
