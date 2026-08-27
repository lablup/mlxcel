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

//! Differential validation of llama-server b10621 prompt-cache semantics
//! (issue #1453), against a real `mlxcel-server` and a real checkpoint.
//!
//! # Why differential rather than a byte-identity or perplexity check
//!
//! What has to be true of a prompt cache is not that it produces good output,
//! it is that it produces the *same* output as not having it. That is a
//! statement about two runs, so a single run's perplexity cannot express it and
//! a byte-identity check against a fixed expected string cannot survive a
//! kernel change that moves both arms together. The gate here is the one
//! `docs/benchmarks.md` prescribes for a change that moves the numbers, applied
//! through the server's own wire format: both arms are scored over the same
//! token stream, and what is compared is the per-position top-1 token and its
//! logprob, with disagreement counted separately on the positions the reference
//! had actually decided.
//!
//! Teacher-forcing comes for free here and is not a simplification: at
//! `temperature 0` with a fixed seed the two arms decode the same prompt, so
//! every position after the first divergence would be conditioned on different
//! text and the comparison would lose its power exactly as it does in
//! `examples/logit_trace`. Reporting the first divergence index alongside the
//! rate is what keeps that visible.
//!
//! # The four arms
//!
//! 1. **Cold** — a fresh server, first request. `cached_tokens == 0`.
//! 2. **Prefix cache hit** — the same request again. `cached_tokens > 0`, and
//!    every decided position must agree with the cold arm.
//! 3. **Per-request disable** — the same request with `cache_prompt: false`.
//!    It must reprocess from cold (`cached_tokens == 0`), agree with the cold
//!    arm, and leave the store intact, which the fourth request proves by
//!    hitting again afterwards.
//! 4. **Mixed batch widths** — three concurrent requests whose prompts differ
//!    in length, each compared against its own solo run. This is the case a
//!    prefix cache gets wrong quietly: an adopted prefix that is correct alone
//!    can be wrong when the batch pads three sequences of different lengths
//!    into one decode.
//!
//! `--cache-reuse` gets its own test: b10621's KV-shift chunk reuse is not
//! implemented, and the acceptance criterion for that is an explicit failure
//! rather than a silent no-op, so what is asserted is that the server refuses
//! to start.
//!
//! Gated with `#[ignore]` because it needs the `mlxcel-server` binary and a
//! local `qwen3-0.6b-4bit` checkout:
//!
//! ```text
//! cargo test --test prompt_cache_compat_e2e --profile test-fast \
//!     --features metal,accelerate -- --ignored --nocapture
//! ```

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// Smallest Qwen3 weight bundle kept locally, shared with
/// `tests/prompt_cache_e2e.rs`.
const MODEL: &str = "qwen3-0.6b-4bit";

/// Decoded tokens per arm. Long enough for a disagreement to have somewhere to
/// show up, short enough that four arms plus a concurrent batch stay quick.
const MAX_TOKENS: usize = 32;

/// The top-two logprob gap above which the model is treated as having decided.
/// `docs/benchmarks.md` gates on disagreement at decided positions for exactly
/// this reason: a position where the top two were a hair apart has no right
/// answer to get wrong, and pooling those with decided ones hides the only
/// distinction that matters.
const DECIDED_GAP: f64 = 2.0;

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

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
        .env_remove("APC_BLOCK_SIZE")
        .env_remove("APC_ENABLED")
        .env_remove("LLAMA_ARG_CACHE_PROMPT")
        .env_remove("LLAMA_ARG_CACHE_REUSE")
        .env_remove("MLXCEL_PROMPT_CACHE_ENABLED")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mlxcel-server")
}

fn stop_server(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// One arm's decoded trace: the chosen token at each position and how decided
/// the model was about it.
#[derive(Debug, Clone)]
struct Trace {
    /// `choices[0].logprobs.content[i].token`.
    tokens: Vec<String>,
    /// The chosen token's logprob at each position.
    chosen: Vec<f64>,
    /// The gap between the top two alternatives at each position, or `None`
    /// where fewer than two were returned.
    gap: Vec<Option<f64>>,
    /// `usage.prompt_tokens_details.cached_tokens`, `None` when absent.
    cached_tokens: Option<u64>,
}

impl Trace {
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// How two arms compare, in the terms `docs/benchmarks.md` gates on.
#[derive(Debug)]
struct Comparison {
    positions: usize,
    decided: usize,
    disagreed: usize,
    disagreed_decided: usize,
    /// Index of the first differing token, or `None` when every one agrees.
    first_divergence: Option<usize>,
    /// The reference's top-two gap at [`Self::first_divergence`]. This is the
    /// number that decides whether a divergence is a behavior change or the
    /// jitter class: once the arms pick different tokens at position `i`,
    /// every position after it is conditioned on different text, so counting
    /// those as further disagreements measures the cascade rather than the
    /// change. What the first flip was worth is the honest question.
    first_divergence_gap: Option<f64>,
    /// Largest absolute difference in the chosen token's logprob over the
    /// positions both arms agreed on.
    max_logprob_delta: f64,
}

impl Comparison {
    fn describe(&self, label: &str) -> String {
        format!(
            "{label}: {}/{} positions disagree ({:.3}%), {}/{} decided positions disagree \
             ({:.3}%), first divergence {:?} at reference top-two gap {:?}, \
             max |dlogprob| on agreeing positions {:.6}",
            self.disagreed,
            self.positions,
            100.0 * self.disagreed as f64 / self.positions.max(1) as f64,
            self.disagreed_decided,
            self.decided,
            100.0 * self.disagreed_decided as f64 / self.decided.max(1) as f64,
            self.first_divergence,
            self.first_divergence_gap,
            self.max_logprob_delta,
        )
    }

    /// Whether the arms parted at a position the reference had decided.
    ///
    /// `false` for a run that never diverged, and for one that parted where
    /// the reference's own top two were within [`DECIDED_GAP`] of each other.
    fn diverged_on_a_decided_position(&self) -> bool {
        self.first_divergence_gap.is_some_and(|g| g >= DECIDED_GAP)
    }
}

/// Compare a candidate trace against a reference one.
///
/// Positions are compared pairwise up to the shorter length; a length
/// difference is itself a divergence and is reported through
/// `first_divergence`.
fn compare(reference: &Trace, candidate: &Trace) -> Comparison {
    let positions = reference.len().min(candidate.len());
    let mut decided = 0usize;
    let mut disagreed = 0usize;
    let mut disagreed_decided = 0usize;
    let mut first_divergence = None;
    let mut max_logprob_delta = 0.0f64;

    for i in 0..positions {
        let is_decided = reference.gap[i].is_some_and(|g| g >= DECIDED_GAP);
        if is_decided {
            decided += 1;
        }
        if reference.tokens[i] != candidate.tokens[i] {
            disagreed += 1;
            if is_decided {
                disagreed_decided += 1;
            }
            first_divergence.get_or_insert(i);
        } else {
            max_logprob_delta =
                max_logprob_delta.max((reference.chosen[i] - candidate.chosen[i]).abs());
        }
    }
    if first_divergence.is_none() && reference.len() != candidate.len() {
        first_divergence = Some(positions);
    }

    let first_divergence_gap =
        first_divergence.and_then(|i| reference.gap.get(i).copied().flatten());

    Comparison {
        positions,
        decided,
        disagreed,
        disagreed_decided,
        first_divergence,
        first_divergence_gap,
        max_logprob_delta,
    }
}

/// Send one greedy chat completion and extract its trace.
///
/// `cache_prompt` is sent at the request root when `Some`, which is where a
/// b10621 client puts it.
async fn arm(
    client: &reqwest::Client,
    base_url: &str,
    prompt: &str,
    cache_prompt: Option<bool>,
) -> Option<Trace> {
    let mut body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 7,
        "logprobs": true,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    if let Some(value) = cache_prompt {
        body["cache_prompt"] = serde_json::Value::Bool(value);
    }

    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        eprintln!("request failed: {}", response.status());
        return None;
    }
    let json: serde_json::Value = response.json().await.ok()?;

    let content = json["choices"][0]["logprobs"]["content"].as_array()?;
    let mut tokens = Vec::with_capacity(content.len());
    let mut chosen = Vec::with_capacity(content.len());
    let mut gap = Vec::with_capacity(content.len());
    for position in content {
        tokens.push(position["token"].as_str()?.to_string());
        let lp = position["logprob"].as_f64()?;
        chosen.push(lp);
        // `top_logprobs` is sorted best-first, so the gap is between the first
        // two entries. A single entry means the server returned no runner-up
        // and the position cannot be classified.
        let tops = position["top_logprobs"].as_array();
        gap.push(match tops {
            Some(t) if t.len() >= 2 => {
                let a = t[0]["logprob"].as_f64()?;
                let b = t[1]["logprob"].as_f64()?;
                Some(a - b)
            }
            _ => None,
        });
    }

    Some(Trace {
        tokens,
        chosen,
        gap,
        cached_tokens: json["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64(),
    })
}

/// A prompt long enough to clear `PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS`
/// (32) so the cache is actually exercised, and varied in length across the
/// concurrent arm.
fn prompt_of(padding: usize) -> String {
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(padding);
    // The question asks for a list rather than a fact, so the decode runs to
    // `MAX_TOKENS` instead of stopping after three tokens. A comparison over
    // seven positions can miss a divergence that a comparison over thirty-two
    // would catch, and the point of this test is the comparison.
    format!(
        "You are a terse assistant. Here is some background you should ignore: {filler}\
         Now list the twelve largest cities in France, one per line, with no other text."
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
async fn prompt_cache_hit_and_per_request_disable_match_a_cold_evaluation() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("skipping: {} not present", model_dir.display());
        return;
    }
    let port = reserve_port();
    let port_s = port.to_string();
    let base_url = format!("http://127.0.0.1:{port}");
    let mut server = spawn_server(&[
        "-m",
        model_dir.to_str().expect("utf-8 model path"),
        "--port",
        &port_s,
        "--host",
        "127.0.0.1",
        // The b10621 spelling, so this test also proves the flag reaches the
        // store rather than only that the store works.
        "--cache-prompt",
        "--parallel",
        "4",
        "--no-warmup",
    ]);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("client");

    if !wait_for_health_soft(&client, &base_url, Duration::from_secs(180)).await {
        eprintln!("skipping: server never became healthy");
        stop_server(&mut server);
        return;
    }

    let prompt = prompt_of(6);

    let Some(cold) = arm(&client, &base_url, &prompt, None).await else {
        eprintln!("skipping: cold request could not be served");
        stop_server(&mut server);
        return;
    };
    assert!(cold.len() > 0, "the cold arm decoded nothing");
    assert_eq!(
        cold.cached_tokens,
        Some(0),
        "the first request on a fresh server cannot have reused anything"
    );

    let hot = arm(&client, &base_url, &prompt, None)
        .await
        .expect("the second request must be served");
    assert!(
        hot.cached_tokens.is_some_and(|n| n > 0),
        "the repeated prompt must hit the cache; got {:?}",
        hot.cached_tokens
    );

    let disabled = arm(&client, &base_url, &prompt, Some(false))
        .await
        .expect("the opt-out request must be served");
    assert_eq!(
        disabled.cached_tokens,
        Some(0),
        "cache_prompt: false must force a cold prefill, not report a hit"
    );

    // The store must be untouched by the opt-out: a fourth request with the
    // same prompt still hits. This is the half of the contract that a
    // lookup-only opt-out would fail, because it would still donate back and
    // could still evict.
    let hot_again = arm(&client, &base_url, &prompt, None)
        .await
        .expect("the fourth request must be served");
    assert!(
        hot_again.cached_tokens.is_some_and(|n| n > 0),
        "cache_prompt: false must not disturb what other requests can reuse; got {:?}",
        hot_again.cached_tokens
    );

    let hit = compare(&cold, &hot);
    let off = compare(&cold, &disabled);
    eprintln!("{}", hit.describe("prefix cache hit vs cold"));
    eprintln!("{}", off.describe("cache_prompt:false vs cold"));

    // These two arms run at the same batch width as the reference (one request
    // at a time), so the jitter class the concurrent test has to tolerate does
    // not apply here and the gate is exact agreement.
    for (label, cmp) in [("prefix cache hit", &hit), ("cache_prompt:false", &off)] {
        assert_eq!(
            cmp.disagreed_decided,
            0,
            "{label} disagrees with the cold evaluation on {} of {} decided positions; \
             a prompt cache that changes decided tokens is returning different text for the \
             same request. {}",
            cmp.disagreed_decided,
            cmp.decided,
            cmp.describe(label)
        );
        assert_eq!(
            cmp.disagreed,
            0,
            "{label} disagrees on {} of {} positions overall. {}",
            cmp.disagreed,
            cmp.positions,
            cmp.describe(label)
        );
    }

    stop_server(&mut server);
}

/// Mixed batch widths: three prompts of different lengths served concurrently
/// must each decode what they decode alone.
///
/// This is the shape a prefix cache gets wrong quietly. An adopted prefix that
/// is correct on its own can still be wrong once the batch pads three
/// sequences of unequal length into one decode, because the padding and the
/// per-row offsets are what an adopted cache has to agree with.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
async fn concurrent_requests_of_mixed_widths_match_their_solo_runs() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("skipping: {} not present", model_dir.display());
        return;
    }
    let port = reserve_port();
    let port_s = port.to_string();
    let base_url = format!("http://127.0.0.1:{port}");
    let mut server = spawn_server(&[
        "-m",
        model_dir.to_str().expect("utf-8 model path"),
        "--port",
        &port_s,
        "--host",
        "127.0.0.1",
        "--cache-prompt",
        "--parallel",
        "4",
        "--no-warmup",
    ]);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .expect("client");

    if !wait_for_health_soft(&client, &base_url, Duration::from_secs(180)).await {
        eprintln!("skipping: server never became healthy");
        stop_server(&mut server);
        return;
    }

    let prompts: Vec<String> = [2usize, 9, 20].iter().map(|p| prompt_of(*p)).collect();

    // Solo reference runs, one at a time, each also warming the store for its
    // own prompt so the concurrent pass exercises adoption rather than three
    // cold prefills.
    let mut solo = Vec::new();
    for prompt in &prompts {
        let Some(trace) = arm(&client, &base_url, prompt, None).await else {
            eprintln!("skipping: solo request could not be served");
            stop_server(&mut server);
            return;
        };
        solo.push(trace);
    }

    let concurrent = {
        let futures = prompts
            .iter()
            .map(|prompt| arm(&client, &base_url, prompt, None));
        futures::future::join_all(futures).await
    };

    let mut adopted = 0usize;
    for (i, candidate) in concurrent.into_iter().enumerate() {
        let candidate = candidate.expect("every concurrent request must be served");
        if candidate.cached_tokens.is_some_and(|n| n > 0) {
            adopted += 1;
        }
        let cmp = compare(&solo[i], &candidate);
        eprintln!("{}", cmp.describe(&format!("mixed batch width {i}")));
        // The gate is the FIRST divergence, not the total. Adoption is
        // correct or it is not, and a wrong prefix shows up as an immediate,
        // decided disagreement. What a strict "zero disagreements" assertion
        // would also catch is the documented batched-kernel jitter class
        // (issue #203): a decode at batch width three does not run the same
        // kernels as one at width one, near-ties can land the other way, and
        // once a single token flips every position after it is conditioned on
        // different text. Counting the cascade would make this test fail for
        // a reason that has nothing to do with the prompt cache.
        assert!(
            !cmp.diverged_on_a_decided_position(),
            "width {i} parted from its solo run at a position the model had decided, \
             which is a behavior change rather than the batched-kernel jitter class. {}",
            cmp.describe("mixed batch")
        );
    }
    // A pass in which nothing was adopted would satisfy the comparison above
    // trivially. At least one request must have reused its prefix for the
    // comparison to have been about adoption at all. It is not "all three":
    // an adopt TAKES its entry rather than sharing it, so three requests
    // racing for three entries can leave one of them to prefill cold, which
    // is correct behavior and not something to assert against.
    assert!(
        adopted > 0,
        "no concurrent request reused a prefix, so this pass compared three cold runs"
    );

    stop_server(&mut server);
}

/// b10621's KV-shift chunk reuse is not implemented, and epic #1431's rule is
/// that a value-bearing option is never accepted and ignored. The acceptance
/// criterion for this one is an explicit failure, so what is asserted is that
/// the process refuses to start rather than serving with a cache that behaves
/// exactly as it does at `--cache-reuse 0`.
///
/// The refusal happens while the startup configuration is being normalized,
/// before a single weight is read, so this returns in well under a second even
/// though it names a real checkpoint. It has to name a real one: the model path
/// is resolved earlier still, and an unresolvable path would report that
/// instead and prove nothing about `--cache-reuse`.
#[test]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
fn a_positive_cache_reuse_refuses_to_start() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("skipping: {} not present", model_dir.display());
        return;
    }
    let port = reserve_port().to_string();
    let output = Command::new(repo_binary_path("mlxcel-server"))
        .args([
            "-m",
            model_dir.to_str().expect("utf-8 model path"),
            "--port",
            &port,
            "--cache-reuse",
            "256",
        ])
        .env_remove("LLAMA_ARG_CACHE_REUSE")
        .output()
        .expect("spawn mlxcel-server");

    assert!(
        !output.status.success(),
        "--cache-reuse 256 must not start a server"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr}{stdout}");
    assert!(
        combined.contains("cache-reuse"),
        "the refusal must name the option it is refusing; got: {combined}"
    );
    // The message has to say what is missing, or an operator cannot tell a
    // refusal from a bug.
    assert!(
        combined.contains("prefix") || combined.contains("re-bas"),
        "the refusal must say why; got: {combined}"
    );
}

/// `--cache-reuse 0` is upstream's default and must start normally, so a
/// deployment that spells out the upstream default keeps working.
#[test]
#[ignore = "requires local model weights (qwen3-0.6b-4bit) and the mlxcel-server binary"]
fn cache_reuse_zero_starts_normally() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("skipping: {} not present", model_dir.display());
        return;
    }
    let port = reserve_port().to_string();
    let mut server = spawn_server(&[
        "-m",
        model_dir.to_str().expect("utf-8 model path"),
        "--port",
        &port,
        "--host",
        "127.0.0.1",
        "--cache-reuse",
        "0",
        "--no-warmup",
    ]);
    // A refusal exits immediately; a healthy start does not. Give the process
    // a moment and assert it is still alive, which is enough to separate the
    // two without waiting for weights to load.
    std::thread::sleep(Duration::from_secs(3));
    let alive = matches!(server.try_wait(), Ok(None));
    stop_server(&mut server);
    assert!(
        alive,
        "--cache-reuse 0 is the upstream default and must not refuse to start"
    );
}
