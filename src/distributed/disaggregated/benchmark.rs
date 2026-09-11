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

//! Disaggregated inference benchmark harness.
//!
//! Provides configurable benchmarks for measuring disaggregated inference
//! performance: TTFT, TPOT, throughput, cache transfer metrics, and
//! crossover analysis. Benchmarks run using simulated scheduling and
//! metrics infrastructure -- no real model or hardware required.
//!
//! Key types:
//!
//! - [`DIBenchmarkConfig`] -- benchmark parameters (node counts, prompt lengths)
//! - [`DIBenchmarkResult`] -- throughput, TTFT, TPOT, cache transfer metrics
//! - [`CacheTransferProfile`] -- breakdown of serialization, transfer, deserialization
//! - [`PromptLengthAnalysis`] -- how performance scales with prompt length
//! - [`DICrossoverAnalysis`] -- when DI outperforms single-node serving
//! - [`run_di_benchmark`] -- execute a single benchmark scenario
//! - [`run_prompt_length_analysis`] -- test across prompt lengths
//! - [`run_di_crossover_analysis`] -- find the DI advantage point
//! - [`format_di_report`] -- human-readable report
//!
//! Used by: integration tests, CI benchmarks

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a disaggregated inference benchmark run.
#[derive(Debug, Clone)]
pub struct DIBenchmarkConfig {
    /// Number of prefill nodes in the cluster.
    pub prefill_nodes: usize,
    /// Number of decode nodes in the cluster.
    pub decode_nodes: usize,
    /// Prompt lengths to test (in tokens).
    pub prompt_lengths: Vec<usize>,
    /// Number of concurrent requests per benchmark iteration.
    pub concurrency: usize,
    /// Simulated prefill compute time per token.
    pub prefill_time_per_token: Duration,
    /// Simulated decode compute time per token.
    pub decode_time_per_token: Duration,
    /// Simulated KV cache serialization time per layer.
    pub cache_serialize_time_per_layer: Duration,
    /// Simulated network transfer time per megabyte of cache data.
    pub cache_transfer_time_per_mb: Duration,
    /// Simulated KV cache deserialization time per layer.
    pub cache_deserialize_time_per_layer: Duration,
    /// Number of model layers (affects cache transfer cost).
    pub num_layers: usize,
    /// Bytes per layer per token in KV cache (affects transfer volume).
    pub bytes_per_layer_per_token: usize,
    /// Number of decode tokens to generate per request.
    pub decode_tokens: usize,
    /// Number of warmup iterations before measurement.
    pub warmup_iterations: usize,
}

impl Default for DIBenchmarkConfig {
    fn default() -> Self {
        Self {
            prefill_nodes: 1,
            decode_nodes: 1,
            prompt_lengths: vec![128],
            concurrency: 1,
            prefill_time_per_token: Duration::from_micros(10),
            decode_time_per_token: Duration::from_micros(50),
            cache_serialize_time_per_layer: Duration::from_micros(20),
            cache_transfer_time_per_mb: Duration::from_micros(500),
            cache_deserialize_time_per_layer: Duration::from_micros(15),
            num_layers: 32,
            bytes_per_layer_per_token: 256,
            decode_tokens: 32,
            warmup_iterations: 1,
        }
    }
}

impl DIBenchmarkConfig {
    /// Create a config for a specific prefill+decode node configuration.
    pub fn new(prefill_nodes: usize, decode_nodes: usize) -> Self {
        Self {
            prefill_nodes,
            decode_nodes,
            ..Default::default()
        }
    }

    /// Set the prompt lengths to test.
    #[must_use]
    pub fn with_prompt_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.prompt_lengths = lengths;
        self
    }

    /// Set the number of concurrent requests.
    #[must_use]
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Set the number of decode tokens.
    #[must_use]
    pub fn with_decode_tokens(mut self, n: usize) -> Self {
        self.decode_tokens = n;
        self
    }

    /// Set the number of model layers.
    #[must_use]
    pub fn with_num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set the prefill compute time per token.
    #[must_use]
    pub fn with_prefill_time(mut self, t: Duration) -> Self {
        self.prefill_time_per_token = t;
        self
    }

    /// Set the decode compute time per token.
    #[must_use]
    pub fn with_decode_time(mut self, t: Duration) -> Self {
        self.decode_time_per_token = t;
        self
    }

    /// Set the cache transfer time per megabyte.
    #[must_use]
    pub fn with_cache_transfer_time(mut self, t: Duration) -> Self {
        self.cache_transfer_time_per_mb = t;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.prefill_nodes > 0, "prefill_nodes must be > 0");
        ensure!(self.decode_nodes > 0, "decode_nodes must be > 0");
        ensure!(
            !self.prompt_lengths.is_empty(),
            "prompt_lengths must not be empty"
        );
        ensure!(self.concurrency > 0, "concurrency must be > 0");
        ensure!(self.decode_tokens > 0, "decode_tokens must be > 0");
        ensure!(self.num_layers > 0, "num_layers must be > 0");
        ensure!(
            self.bytes_per_layer_per_token > 0,
            "bytes_per_layer_per_token must be > 0"
        );
        Ok(())
    }

    /// Compute the KV cache size in bytes for a given prompt length.
    fn cache_size_bytes(&self, prompt_len: usize) -> usize {
        self.num_layers * self.bytes_per_layer_per_token * prompt_len
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Profile of KV cache transfer for a single request.
#[derive(Debug, Clone)]
pub struct CacheTransferProfile {
    /// Time to serialize the KV cache on the prefill node.
    pub serialization_time: Duration,
    /// Time to transfer the serialized cache over the network.
    pub transfer_time: Duration,
    /// Time to deserialize the KV cache on the decode node.
    pub deserialization_time: Duration,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Prompt length that produced this cache.
    pub prompt_len: usize,
}

impl CacheTransferProfile {
    /// Total handoff time (serialize + transfer + deserialize).
    pub fn total_handoff_time(&self) -> Duration {
        self.serialization_time + self.transfer_time + self.deserialization_time
    }

    /// Transfer throughput in MB/s.
    pub fn throughput_mb_per_sec(&self) -> f64 {
        let transfer_secs = self.transfer_time.as_secs_f64();
        if transfer_secs == 0.0 {
            return 0.0;
        }
        (self.bytes_transferred as f64 / (1024.0 * 1024.0)) / transfer_secs
    }
}

/// Results from a single disaggregated inference benchmark run.
#[derive(Debug, Clone)]
pub struct DIBenchmarkResult {
    /// Configuration used for this run.
    pub config: DIBenchmarkConfig,
    /// Prompt length tested in this run.
    pub prompt_len: usize,
    /// Time to first token (prefill + cache transfer).
    pub ttft: Duration,
    /// Average time per output token during decode.
    pub tpot: Duration,
    /// Aggregate throughput in tokens per second.
    pub throughput_tok_per_sec: f64,
    /// Cache transfer profile.
    pub cache_transfer: CacheTransferProfile,
    /// Total wall-clock time for the benchmark.
    pub total_time: Duration,
    /// Total tokens generated.
    pub total_tokens: u64,
    /// Single-node baseline TTFT for comparison.
    pub baseline_ttft: Duration,
    /// Single-node baseline throughput for comparison.
    pub baseline_throughput: f64,
    /// Speedup vs single-node (throughput ratio).
    pub speedup_vs_single: f64,
}

impl fmt::Display for DIBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DI Benchmark: {}P+{}D, prompt_len={}, concurrency={}",
            self.config.prefill_nodes,
            self.config.decode_nodes,
            self.prompt_len,
            self.config.concurrency,
        )?;
        writeln!(
            f,
            "  TTFT: {:.3}ms (baseline: {:.3}ms)",
            self.ttft.as_secs_f64() * 1000.0,
            self.baseline_ttft.as_secs_f64() * 1000.0,
        )?;
        writeln!(f, "  TPOT: {:.3}ms", self.tpot.as_secs_f64() * 1000.0,)?;
        writeln!(
            f,
            "  Throughput: {:.1} tok/s (baseline: {:.1}, speedup: {:.2}x)",
            self.throughput_tok_per_sec, self.baseline_throughput, self.speedup_vs_single,
        )?;
        writeln!(
            f,
            "  Cache transfer: {:.3}ms ({:.1} MB, {:.1} MB/s)",
            self.cache_transfer.total_handoff_time().as_secs_f64() * 1000.0,
            self.cache_transfer.bytes_transferred as f64 / (1024.0 * 1024.0),
            self.cache_transfer.throughput_mb_per_sec(),
        )?;
        write!(
            f,
            "  Total tokens: {} in {:.2}ms",
            self.total_tokens,
            self.total_time.as_secs_f64() * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Prompt length analysis
// ---------------------------------------------------------------------------

/// Analysis of how performance scales with prompt length.
#[derive(Debug, Clone)]
pub struct PromptLengthAnalysis {
    /// Results for each prompt length tested.
    pub results: Vec<DIBenchmarkResult>,
}

impl PromptLengthAnalysis {
    /// Find the prompt length where cache transfer dominates TTFT.
    ///
    /// Returns the shortest prompt length where cache transfer time
    /// exceeds 50% of TTFT.
    pub fn transfer_dominance_threshold(&self) -> Option<usize> {
        for r in &self.results {
            let handoff = r.cache_transfer.total_handoff_time().as_secs_f64();
            let ttft = r.ttft.as_secs_f64();
            if ttft > 0.0 && handoff / ttft > 0.5 {
                return Some(r.prompt_len);
            }
        }
        None
    }

    /// Compute the linear regression slope of transfer time vs prompt length.
    ///
    /// Returns bytes-per-second growth rate, or None if insufficient data.
    pub fn transfer_time_slope(&self) -> Option<f64> {
        if self.results.len() < 2 {
            return None;
        }
        let n = self.results.len() as f64;
        let sum_x: f64 = self.results.iter().map(|r| r.prompt_len as f64).sum();
        let sum_y: f64 = self
            .results
            .iter()
            .map(|r| r.cache_transfer.total_handoff_time().as_secs_f64())
            .sum();
        let sum_xy: f64 = self
            .results
            .iter()
            .map(|r| r.prompt_len as f64 * r.cache_transfer.total_handoff_time().as_secs_f64())
            .sum();
        let sum_xx: f64 = self
            .results
            .iter()
            .map(|r| (r.prompt_len as f64).powi(2))
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        Some((n * sum_xy - sum_x * sum_y) / denom)
    }
}

impl fmt::Display for PromptLengthAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Prompt Length Analysis")?;
        writeln!(f, "{:-<60}", "")?;
        writeln!(
            f,
            "  {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Prompt", "TTFT(ms)", "TPOT(ms)", "Transfer", "Speedup"
        )?;
        writeln!(
            f,
            "  {:->8}-+-{:->10}-+-{:->10}-+-{:->10}-+-{:->10}",
            "", "", "", "", ""
        )?;
        for r in &self.results {
            writeln!(
                f,
                "  {:>8} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10.2}x",
                r.prompt_len,
                r.ttft.as_secs_f64() * 1000.0,
                r.tpot.as_secs_f64() * 1000.0,
                r.cache_transfer.total_handoff_time().as_secs_f64() * 1000.0,
                r.speedup_vs_single,
            )?;
        }
        if let Some(threshold) = self.transfer_dominance_threshold() {
            writeln!(
                f,
                "\n  Transfer dominates TTFT at prompt_len >= {threshold}"
            )?;
        }
        if let Some(slope) = self.transfer_time_slope() {
            writeln!(f, "  Transfer time slope: {:.6} ms/token", slope * 1000.0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Crossover analysis
// ---------------------------------------------------------------------------

/// Entry in the crossover analysis comparing DI vs single-node.
#[derive(Debug, Clone)]
pub struct DICrossoverEntry {
    /// Cluster configuration description (e.g., "1P+1D").
    pub config_desc: String,
    /// Prompt length tested.
    pub prompt_len: usize,
    /// Number of concurrent requests.
    pub concurrency: usize,
    /// DI throughput (tok/s).
    pub di_throughput: f64,
    /// Single-node throughput (tok/s).
    pub baseline_throughput: f64,
    /// Speedup (DI / baseline).
    pub speedup: f64,
    /// Whether DI outperforms single-node.
    pub di_wins: bool,
}

/// Analysis of when disaggregated inference outperforms single-node.
#[derive(Debug, Clone)]
pub struct DICrossoverAnalysis {
    /// All tested configurations and their results.
    pub entries: Vec<DICrossoverEntry>,
}

impl DICrossoverAnalysis {
    /// Find the minimum concurrency at which DI outperforms single-node
    /// for the given cluster configuration.
    pub fn crossover_concurrency(&self, config_desc: &str) -> Option<usize> {
        self.entries
            .iter()
            .filter(|e| e.config_desc == config_desc && e.di_wins)
            .map(|e| e.concurrency)
            .min()
    }

    /// Find the minimum prompt length at which DI outperforms single-node
    /// for a given concurrency level.
    pub fn crossover_prompt_len(&self, concurrency: usize) -> Option<usize> {
        self.entries
            .iter()
            .filter(|e| e.concurrency == concurrency && e.di_wins)
            .map(|e| e.prompt_len)
            .min()
    }
}

impl fmt::Display for DICrossoverAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Crossover Analysis: DI vs Single-Node")?;
        writeln!(f, "{:-<70}", "")?;
        writeln!(
            f,
            "  {:>6} | {:>8} | {:>5} | {:>10} | {:>10} | {:>8}",
            "Config", "Prompt", "Conc.", "DI tok/s", "Base tok/s", "Speedup"
        )?;
        writeln!(
            f,
            "  {:->6}-+-{:->8}-+-{:->5}-+-{:->10}-+-{:->10}-+-{:->8}",
            "", "", "", "", "", ""
        )?;
        for e in &self.entries {
            let marker = if e.di_wins { " *" } else { "" };
            writeln!(
                f,
                "  {:>6} | {:>8} | {:>5} | {:>10.1} | {:>10.1} | {:>7.2}x{}",
                e.config_desc,
                e.prompt_len,
                e.concurrency,
                e.di_throughput,
                e.baseline_throughput,
                e.speedup,
                marker,
            )?;
        }
        writeln!(f, "\n  * = DI outperforms single-node")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

/// Run a single disaggregated inference benchmark for one prompt length.
///
/// Simulates the full DI pipeline: prefill compute, cache serialization,
/// network transfer, cache deserialization, and decode. Compares against
/// a single-node baseline that skips cache transfer.
pub fn run_di_benchmark(
    config: &DIBenchmarkConfig,
    prompt_len: usize,
) -> Result<DIBenchmarkResult> {
    config.validate()?;

    // Warmup.
    for _ in 0..config.warmup_iterations {
        simulate_di_request(config, prompt_len);
    }

    // Measure.
    let start = Instant::now();
    let sim = simulate_di_request(config, prompt_len);
    let total_time = start.elapsed();

    // Baseline: single-node (no cache transfer, sequential prefill+decode).
    let baseline = simulate_baseline_request(config, prompt_len);

    let total_tokens = config.concurrency as u64 * config.decode_tokens as u64;
    let throughput = if total_time.is_zero() {
        0.0
    } else {
        total_tokens as f64 / total_time.as_secs_f64()
    };

    let baseline_throughput = if baseline.total_time.is_zero() {
        0.0
    } else {
        config.decode_tokens as f64 / baseline.total_time.as_secs_f64()
    };

    let speedup = if baseline_throughput == 0.0 {
        0.0
    } else {
        throughput / baseline_throughput
    };

    Ok(DIBenchmarkResult {
        config: config.clone(),
        prompt_len,
        ttft: sim.ttft,
        tpot: sim.tpot,
        throughput_tok_per_sec: throughput,
        cache_transfer: sim.cache_transfer,
        total_time,
        total_tokens,
        baseline_ttft: baseline.ttft,
        baseline_throughput,
        speedup_vs_single: speedup,
    })
}

/// Run prompt length analysis across multiple prompt lengths.
pub fn run_prompt_length_analysis(config: &DIBenchmarkConfig) -> Result<PromptLengthAnalysis> {
    config.validate()?;
    let mut results = Vec::with_capacity(config.prompt_lengths.len());
    for &len in &config.prompt_lengths {
        results.push(run_di_benchmark(config, len)?);
    }
    Ok(PromptLengthAnalysis { results })
}

/// Run crossover analysis comparing DI vs single-node across configurations.
pub fn run_di_crossover_analysis(
    configs: &[(usize, usize)],
    prompt_lengths: &[usize],
    concurrency_levels: &[usize],
) -> Result<DICrossoverAnalysis> {
    ensure!(!configs.is_empty(), "configs must not be empty");
    ensure!(
        !prompt_lengths.is_empty(),
        "prompt_lengths must not be empty"
    );
    ensure!(
        !concurrency_levels.is_empty(),
        "concurrency_levels must not be empty"
    );

    let mut entries = Vec::new();

    for &(prefill, decode) in configs {
        let config_desc = format!("{prefill}P+{decode}D");
        for &prompt_len in prompt_lengths {
            for &conc in concurrency_levels {
                let config = DIBenchmarkConfig::new(prefill, decode)
                    .with_concurrency(conc)
                    .with_prompt_lengths(vec![prompt_len]);

                let result = run_di_benchmark(&config, prompt_len)?;

                entries.push(DICrossoverEntry {
                    config_desc: config_desc.clone(),
                    prompt_len,
                    concurrency: conc,
                    di_throughput: result.throughput_tok_per_sec,
                    baseline_throughput: result.baseline_throughput,
                    speedup: result.speedup_vs_single,
                    di_wins: result.speedup_vs_single > 1.0,
                });
            }
        }
    }

    Ok(DICrossoverAnalysis { entries })
}

/// Format a benchmark report as a human-readable string.
pub fn format_di_report(results: &[DIBenchmarkResult]) -> String {
    let mut report = String::new();
    report.push_str("Disaggregated Inference Benchmark Report\n");
    report.push_str(&"=".repeat(65));
    report.push('\n');

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            report.push_str(&"-".repeat(65));
            report.push('\n');
        }
        report.push_str(&format!("{result}\n"));
    }

    report.push_str(&"=".repeat(65));
    report.push('\n');

    // Summary table.
    report.push_str("\nSummary:\n");
    report.push_str("  Config | Prompt | TTFT(ms)  | TPOT(ms)  | Throughput | Speedup\n");
    report.push_str("  -------+--------+-----------+-----------+-----------+--------\n");
    for r in results {
        report.push_str(&format!(
            "  {:>2}P+{:<2}D | {:>6} | {:>9.3} | {:>9.3} | {:>7.1} t/s | {:>6.2}x\n",
            r.config.prefill_nodes,
            r.config.decode_nodes,
            r.prompt_len,
            r.ttft.as_secs_f64() * 1000.0,
            r.tpot.as_secs_f64() * 1000.0,
            r.throughput_tok_per_sec,
            r.speedup_vs_single,
        ));
    }

    report
}

// ---------------------------------------------------------------------------
// Internal simulation
// ---------------------------------------------------------------------------

/// Simulated request outcome.
struct SimulatedRequest {
    ttft: Duration,
    tpot: Duration,
    cache_transfer: CacheTransferProfile,
    #[allow(dead_code)]
    total_time: Duration,
}

/// Simulate a disaggregated inference request.
///
/// Models the full pipeline: prefill -> serialize -> transfer -> deserialize -> decode.
/// For concurrent requests with multiple nodes, prefill and decode are parallelized.
fn simulate_di_request(config: &DIBenchmarkConfig, prompt_len: usize) -> SimulatedRequest {
    // Prefill time (parallelized across prefill nodes).
    let prefill_per_node = config.concurrency.div_ceil(config.prefill_nodes);
    let prefill_time = config.prefill_time_per_token * prompt_len as u32 * prefill_per_node as u32;

    // Cache transfer: serialize + network + deserialize.
    let cache_bytes = config.cache_size_bytes(prompt_len) as u64;
    let cache_mb = cache_bytes as f64 / (1024.0 * 1024.0);

    let serialize_time = config.cache_serialize_time_per_layer * config.num_layers as u32;
    let transfer_time =
        Duration::from_secs_f64(config.cache_transfer_time_per_mb.as_secs_f64() * cache_mb);
    let deserialize_time = config.cache_deserialize_time_per_layer * config.num_layers as u32;

    let handoff_time = serialize_time + transfer_time + deserialize_time;

    // TTFT = prefill + handoff.
    let ttft = prefill_time + handoff_time;

    // Decode time (parallelized across decode nodes).
    let decode_per_node = config.concurrency.div_ceil(config.decode_nodes);
    let decode_time =
        config.decode_time_per_token * config.decode_tokens as u32 * decode_per_node as u32;

    // TPOT = decode_time / decode_tokens (per-sequence).
    let tpot = if config.decode_tokens > 0 {
        decode_time / config.decode_tokens as u32
    } else {
        Duration::ZERO
    };

    let total_time = ttft + decode_time;

    // Spin-wait to simulate wall-clock time for benchmark measurement.
    spin_wait(total_time);

    SimulatedRequest {
        ttft,
        tpot,
        cache_transfer: CacheTransferProfile {
            serialization_time: serialize_time,
            transfer_time,
            deserialization_time: deserialize_time,
            bytes_transferred: cache_bytes,
            prompt_len,
        },
        total_time,
    }
}

/// Simulate a single-node baseline request (no cache transfer).
fn simulate_baseline_request(config: &DIBenchmarkConfig, prompt_len: usize) -> SimulatedRequest {
    // Single-node: prefill + decode, no transfer overhead.
    let prefill_time =
        config.prefill_time_per_token * prompt_len as u32 * config.concurrency as u32;
    let decode_time =
        config.decode_time_per_token * config.decode_tokens as u32 * config.concurrency as u32;

    let ttft = prefill_time;
    let tpot = if config.decode_tokens > 0 {
        decode_time / config.decode_tokens as u32
    } else {
        Duration::ZERO
    };

    let total_time = prefill_time + decode_time;

    SimulatedRequest {
        ttft,
        tpot,
        cache_transfer: CacheTransferProfile {
            serialization_time: Duration::ZERO,
            transfer_time: Duration::ZERO,
            deserialization_time: Duration::ZERO,
            bytes_transferred: 0,
            prompt_len,
        },
        total_time,
    }
}

/// Busy-wait for the given duration to simulate compute/transfer time.
fn spin_wait(duration: Duration) {
    if duration.is_zero() {
        return;
    }
    let start = Instant::now();
    while start.elapsed() < duration {
        std::hint::spin_loop();
    }
}

#[cfg(test)]
#[path = "benchmark_tests.rs"]
mod tests;
