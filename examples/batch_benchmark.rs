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

//! Continuous batching benchmark tool.
//!
//! Sends concurrent chat completion requests to a running mlxcel-server and
//! measures throughput, TTFT, and per-request latency across different
//! concurrency levels.
//!
//! # Usage
//!
//! ```bash
//! # Start the server first:
//! mlxcel-server -m models/Meta-Llama-3.1-8B-Instruct-4bit --batch-size 8
//!
//! # Run the benchmark:
//! cargo run --release --example batch_benchmark -- \
//!   --server http://localhost:8080 \
//!   --concurrent 1,2,4,8 \
//!   --prompt "Explain quantum computing in detail" \
//!   --max-tokens 200 \
//!   --runs 3
//! ```

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use tokio::sync::Barrier;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

/// Benchmark tool for continuous batching throughput measurement.
#[derive(Parser, Debug)]
#[command(name = "batch_benchmark")]
#[command(about = "Benchmark continuous batching throughput on mlxcel-server")]
struct Args {
    /// Server base URL (e.g., http://localhost:8080).
    #[arg(long, default_value = "http://localhost:8080")]
    server: String,

    /// Comma-separated concurrency levels to test.
    #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
    concurrent: Vec<usize>,

    /// Prompt text to send in each request.
    #[arg(long, default_value = "Explain quantum computing in simple terms.")]
    prompt: String,

    /// Maximum tokens to generate per request.
    #[arg(long, default_value = "200")]
    max_tokens: usize,

    /// Number of runs per concurrency level (reports median).
    #[arg(long, default_value = "3")]
    runs: usize,

    /// Output format: "table" or "json".
    #[arg(long, default_value = "table")]
    format: String,

    /// Warmup requests before benchmarking (per concurrency level).
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Model name override for the request body (uses server default if empty).
    #[arg(long, default_value = "")]
    model: String,
}

// ---------------------------------------------------------------------------
// Metrics types
// ---------------------------------------------------------------------------

/// Metrics for a single request.
#[derive(Debug, Clone)]
struct RequestMetrics {
    /// Estimated time to first token (ms). Approximation from non-streaming response.
    ttft_ms: f64,
    /// Total request latency (ms).
    total_ms: f64,
    /// Number of completion tokens received.
    completion_tokens: usize,
    /// Tokens per second for this request.
    tps: f64,
}

/// Aggregated metrics for one concurrency level.
#[derive(Debug, Clone, Serialize)]
struct ConcurrencyResult {
    concurrency: usize,
    /// Median total throughput across all concurrent requests (tok/s).
    total_tps: f64,
    /// Median per-request TPS.
    per_request_tps: f64,
    /// Median estimated TTFT across requests (ms). Approximated, not measured.
    median_ttft_ms: f64,
    /// P99 estimated TTFT (ms). Approximated, not measured.
    p99_ttft_ms: f64,
    /// Median per-request latency (ms).
    median_latency_ms: f64,
    /// Total tokens generated across all concurrent requests.
    total_tokens: usize,
    /// Wall-clock time for the run (ms).
    wall_clock_ms: f64,
}

/// Full benchmark output.
#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    server: String,
    prompt: String,
    max_tokens: usize,
    runs_per_level: usize,
    results: Vec<ConcurrencyResult>,
}

// ---------------------------------------------------------------------------
// Request execution
// ---------------------------------------------------------------------------

/// Send one chat completion request (non-streaming) and measure timing.
async fn run_single_request(
    client: &reqwest::Client,
    server_url: &str,
    prompt: &str,
    max_tokens: usize,
    model: &str,
    barrier: Arc<Barrier>,
) -> Result<RequestMetrics, String> {
    // Wait at the barrier so all concurrent requests start simultaneously.
    barrier.wait().await;

    let start = Instant::now();

    let model_field = if model.is_empty() {
        "default".to_string()
    } else {
        model.to_string()
    };

    let body = serde_json::json!({
        "model": model_field,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "stream": false,
        "temperature": 0.0,
    });

    let resp = client
        .post(format!("{server_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {text}"));
    }

    let total_elapsed = start.elapsed();

    let resp_json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("json parse error: {e}"))?;

    let completion_tokens = resp_json["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0) as usize;

    // Estimate TTFT from server timing if available, otherwise use a fraction
    // of total time as approximation (first token is produced after prefill).
    let prompt_tokens = resp_json["usage"]["prompt_tokens"].as_u64().unwrap_or(1) as f64;
    let total_ms = total_elapsed.as_secs_f64() * 1000.0;

    // Rough TTFT estimate: assume prefill takes proportional time to prompt size
    // relative to total generation. A more accurate measurement would require
    // streaming mode. For non-streaming, we approximate as:
    //   TTFT ~ total_time * (prompt_tokens / (prompt_tokens + completion_tokens))
    let total_tokens_f = prompt_tokens + completion_tokens as f64;
    let ttft_ms = if total_tokens_f > 0.0 {
        total_ms * (prompt_tokens / total_tokens_f)
    } else {
        total_ms
    };

    let tps = if total_ms > 0.0 {
        completion_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };

    Ok(RequestMetrics {
        ttft_ms,
        total_ms,
        completion_tokens,
        tps,
    })
}

/// Run one benchmark at a given concurrency level.
async fn run_benchmark_round(
    client: &reqwest::Client,
    server_url: &str,
    prompt: &str,
    max_tokens: usize,
    concurrency: usize,
    model: &str,
) -> Result<Vec<RequestMetrics>, String> {
    let barrier = Arc::new(Barrier::new(concurrency));

    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let url = server_url.to_string();
        let p = prompt.to_string();
        let m = model.to_string();
        let b = barrier.clone();

        handles.push(tokio::spawn(async move {
            run_single_request(&client, &url, &p, max_tokens, &m, b).await
        }));
    }

    let mut results = Vec::with_capacity(concurrency);
    for handle in handles {
        match handle.await {
            Ok(Ok(metrics)) => results.push(metrics),
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(format!("task join error: {e}")),
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn percentile(values: &mut [f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (values.len() as f64 - 1.0)).ceil() as usize;
    values[idx.min(values.len() - 1)]
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async fn wait_for_server(client: &reqwest::Client, server_url: &str) -> Result<(), String> {
    let url = format!("{server_url}/health");
    for attempt in 1..=30 {
        match client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let body: serde_json::Value = resp.json().await.unwrap_or_default();
                let status = body["status"].as_str().unwrap_or("unknown");
                if status == "ok" {
                    return Ok(());
                }
                eprintln!(
                    "  Server responded but status is '{status}', waiting... (attempt {attempt}/30)"
                );
            }
            Ok(resp) => {
                eprintln!(
                    "  Server returned {}, waiting... (attempt {attempt}/30)",
                    resp.status()
                );
            }
            Err(_) => {
                eprintln!("  Server not reachable, waiting... (attempt {attempt}/30)");
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    Err("Server did not become ready within 60 seconds".to_string())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .expect("failed to create HTTP client");

    eprintln!("Batch Benchmark Tool");
    eprintln!("====================");
    eprintln!("Server:     {}", args.server);
    eprintln!("Prompt:     {:?}", args.prompt);
    eprintln!("Max tokens: {}", args.max_tokens);
    eprintln!("Levels:     {:?}", args.concurrent);
    eprintln!("Runs:       {}", args.runs);
    eprintln!();

    // Wait for server readiness
    eprint!("Checking server health... ");
    match wait_for_server(&client, &args.server).await {
        Ok(()) => eprintln!("OK"),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }

    let mut all_results = Vec::new();

    for &concurrency in &args.concurrent {
        eprintln!();
        eprintln!("--- Concurrency: {concurrency} ---");

        // Warmup
        if args.warmup > 0 {
            eprint!("  Warmup ({} request(s))... ", args.warmup);
            std::io::stderr().flush().ok();
            for _ in 0..args.warmup {
                let _ = run_benchmark_round(
                    &client,
                    &args.server,
                    &args.prompt,
                    args.max_tokens.min(10), // short warmup
                    1,
                    &args.model,
                )
                .await;
            }
            eprintln!("done");
        }

        // Actual runs
        let mut run_results: Vec<Vec<RequestMetrics>> = Vec::with_capacity(args.runs);

        for run in 1..=args.runs {
            eprint!("  Run {run}/{}... ", args.runs);
            std::io::stderr().flush().ok();

            match run_benchmark_round(
                &client,
                &args.server,
                &args.prompt,
                args.max_tokens,
                concurrency,
                &args.model,
            )
            .await
            {
                Ok(metrics) => {
                    let total_tokens: usize = metrics.iter().map(|m| m.completion_tokens).sum();
                    let max_latency = metrics.iter().map(|m| m.total_ms).fold(0.0f64, f64::max);
                    let total_tps = if max_latency > 0.0 {
                        total_tokens as f64 / (max_latency / 1000.0)
                    } else {
                        0.0
                    };
                    eprintln!(
                        "total_tps={total_tps:.1}, tokens={total_tokens}, wall={max_latency:.0}ms"
                    );
                    run_results.push(metrics);
                }
                Err(e) => {
                    eprintln!("ERROR: {e}");
                    continue;
                }
            }
        }

        if run_results.is_empty() {
            eprintln!("  All runs failed for concurrency={concurrency}, skipping");
            continue;
        }

        // Aggregate: take median across runs
        let mut total_tps_values: Vec<f64> = Vec::new();
        let mut per_req_tps_values: Vec<f64> = Vec::new();
        let mut ttft_values: Vec<f64> = Vec::new();
        let mut latency_values: Vec<f64> = Vec::new();
        let mut total_tokens_sum = 0usize;
        let mut wall_clock_values: Vec<f64> = Vec::new();

        for run_metrics in &run_results {
            let total_tokens: usize = run_metrics.iter().map(|m| m.completion_tokens).sum();
            let max_latency = run_metrics
                .iter()
                .map(|m| m.total_ms)
                .fold(0.0f64, f64::max);

            let total_tps = if max_latency > 0.0 {
                total_tokens as f64 / (max_latency / 1000.0)
            } else {
                0.0
            };

            total_tps_values.push(total_tps);
            wall_clock_values.push(max_latency);
            total_tokens_sum += total_tokens;

            for m in run_metrics {
                per_req_tps_values.push(m.tps);
                ttft_values.push(m.ttft_ms);
                latency_values.push(m.total_ms);
            }
        }

        let result = ConcurrencyResult {
            concurrency,
            total_tps: median(&mut total_tps_values),
            per_request_tps: median(&mut per_req_tps_values),
            median_ttft_ms: median(&mut ttft_values),
            p99_ttft_ms: percentile(&mut ttft_values, 99.0),
            median_latency_ms: median(&mut latency_values),
            total_tokens: total_tokens_sum / run_results.len().max(1),
            wall_clock_ms: median(&mut wall_clock_values),
        };

        all_results.push(result);
    }

    // Output
    let output = BenchmarkOutput {
        server: args.server.clone(),
        prompt: args.prompt.clone(),
        max_tokens: args.max_tokens,
        runs_per_level: args.runs,
        results: all_results,
    };

    eprintln!();

    if args.format == "json" {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        print_table(&output);
    }
}

fn print_table(output: &BenchmarkOutput) {
    println!();
    println!("Continuous Batching Benchmark Results");
    println!("=====================================");
    println!("Server:     {}", output.server);
    println!("Prompt:     {:?}", output.prompt);
    println!("Max tokens: {}", output.max_tokens);
    println!("Runs:       {}", output.runs_per_level);
    println!();

    // Header
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Concurrency",
        "Total TPS",
        "Per-req TPS",
        "~TTFT (ms)",
        "~P99 TTFT",
        "Latency (ms)",
        "Gain"
    );
    println!("{}", "-".repeat(84));

    let baseline_tps = output.results.first().map(|r| r.total_tps).unwrap_or(1.0);

    for result in &output.results {
        let gain = if baseline_tps > 0.0 {
            result.total_tps / baseline_tps
        } else {
            0.0
        };

        println!(
            "{:<12} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.0} {:>11.2}x",
            result.concurrency,
            result.total_tps,
            result.per_request_tps,
            result.median_ttft_ms,
            result.p99_ttft_ms,
            result.median_latency_ms,
            gain,
        );
    }

    println!();
}
