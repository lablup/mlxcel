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

//! Tensor parallelism benchmark harness.
//!
//! Provides configurable benchmarks for measuring TP throughput, latency,
//! all-reduce overhead, and scaling characteristics. Benchmarks run using
//! simulated collectives and schedulers -- no real multi-device hardware
//! required.
//!
//! Key types:
//!
//! - [`TPBenchmarkConfig`] -- benchmark parameters (tp_sizes, model sizes, etc.)
//! - [`TPBenchmarkResult`] -- throughput, TTFT, ITL, all-reduce overhead, scaling efficiency
//! - [`AllReduceProfile`] -- communication overhead breakdown
//! - [`CrossoverAnalysis`] -- model size vs TP benefit analysis
//! - [`run_tp_benchmark`] -- execute a single benchmark scenario
//! - [`run_scaling_analysis`] -- compare across model sizes and tp_sizes
//! - [`format_tp_benchmark_report`] -- human-readable report
//!
//! Used by: integration tests, CI benchmarks

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};

use super::collective::ring_allreduce_data_volume;
use super::synchronized::{SampledTokens, SamplingMode, TPExecutionConfig};
use super::tp_executor::{StepOutcome, TPExecutor};
use super::tp_scheduler::TPScheduler;
use crate::distributed::tensor_protocol::TensorDtype;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a TP benchmark run.
#[derive(Debug, Clone)]
pub struct TPBenchmarkConfig {
    /// TP sizes to benchmark (e.g., [1, 2, 4]).
    pub tp_sizes: Vec<usize>,
    /// Model hidden sizes to benchmark (e.g., [2048, 4096, 8192]).
    pub model_hidden_sizes: Vec<usize>,
    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,
    /// Sequence lengths to test.
    pub seq_lengths: Vec<usize>,
    /// Number of attention heads in the model.
    pub num_heads: usize,
    /// Number of KV heads (for GQA; equal to num_heads for MHA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// Simulated compute time per layer per token.
    pub layer_compute_time: Duration,
    /// Simulated all-reduce time per operation.
    pub allreduce_time: Duration,
    /// Number of decode steps to simulate.
    pub decode_steps: u32,
    /// Number of warmup steps before measurement.
    pub warmup_steps: u32,
    /// Dtype for collective benchmarks.
    pub dtype: TensorDtype,
}

impl Default for TPBenchmarkConfig {
    fn default() -> Self {
        Self {
            tp_sizes: vec![1, 2, 4],
            model_hidden_sizes: vec![4096],
            batch_sizes: vec![1],
            seq_lengths: vec![128],
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            layer_compute_time: Duration::from_micros(200),
            allreduce_time: Duration::from_micros(50),
            decode_steps: 16,
            warmup_steps: 2,
            dtype: TensorDtype::Float32,
        }
    }
}

impl TPBenchmarkConfig {
    /// Create a config for the given TP sizes.
    pub fn with_tp_sizes(tp_sizes: Vec<usize>) -> Self {
        Self {
            tp_sizes,
            ..Default::default()
        }
    }

    /// Set the number of decode steps.
    #[must_use]
    pub fn with_decode_steps(mut self, n: u32) -> Self {
        self.decode_steps = n;
        self
    }

    /// Set compute time per layer.
    #[must_use]
    pub fn with_layer_compute_time(mut self, t: Duration) -> Self {
        self.layer_compute_time = t;
        self
    }

    /// Set all-reduce time.
    #[must_use]
    pub fn with_allreduce_time(mut self, t: Duration) -> Self {
        self.allreduce_time = t;
        self
    }

    /// Set model hidden sizes for crossover analysis.
    #[must_use]
    pub fn with_model_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.model_hidden_sizes = sizes;
        self
    }

    /// Set batch sizes.
    #[must_use]
    pub fn with_batch_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.batch_sizes = sizes;
        self
    }

    /// Set sequence lengths.
    #[must_use]
    pub fn with_seq_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.seq_lengths = lengths;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.tp_sizes.is_empty(), "tp_sizes must not be empty");
        ensure!(self.decode_steps > 0, "decode_steps must be > 0");
        ensure!(self.num_heads > 0, "num_heads must be > 0");
        ensure!(self.head_dim > 0, "head_dim must be > 0");
        for &tp in &self.tp_sizes {
            ensure!(tp >= 1, "tp_size must be >= 1, got {tp}");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Results from a single TP benchmark run.
#[derive(Debug, Clone)]
pub struct TPBenchmarkResult {
    /// TP size used for this run.
    pub tp_size: usize,
    /// Total wall-clock time for the benchmark (excluding warmup).
    pub total_time: Duration,
    /// Tokens generated per second.
    pub throughput_tok_per_sec: f64,
    /// Time to first token.
    pub ttft: Duration,
    /// Average inter-token latency during decode.
    pub itl: Duration,
    /// All-reduce overhead as a fraction of total time (0.0 to 1.0).
    pub allreduce_overhead: f64,
    /// Scaling efficiency: throughput relative to tp_size=1 as a fraction
    /// of ideal linear scaling (1.0 = perfect).
    pub scaling_efficiency: f64,
    /// Estimated per-rank memory as a fraction of single-device memory.
    pub per_rank_memory_fraction: f64,
    /// Number of decode steps simulated.
    pub decode_steps: u32,
    /// Total tokens generated.
    pub total_tokens: u64,
    /// All-reduce profile breakdown.
    pub allreduce_profile: AllReduceProfile,
}

impl fmt::Display for TPBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TP Benchmark: tp_size={}", self.tp_size)?;
        writeln!(
            f,
            "  Throughput: {:.1} tok/s ({} tokens in {:.2}ms)",
            self.throughput_tok_per_sec,
            self.total_tokens,
            self.total_time.as_secs_f64() * 1000.0,
        )?;
        writeln!(
            f,
            "  TTFT: {:.3}ms  ITL: {:.3}ms",
            self.ttft.as_secs_f64() * 1000.0,
            self.itl.as_secs_f64() * 1000.0,
        )?;
        writeln!(
            f,
            "  All-reduce overhead: {:.1}%  Scaling efficiency: {:.1}%",
            self.allreduce_overhead * 100.0,
            self.scaling_efficiency * 100.0,
        )?;
        write!(
            f,
            "  Per-rank memory: {:.1}% of single-device",
            self.per_rank_memory_fraction * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// All-reduce profile
// ---------------------------------------------------------------------------

/// Breakdown of all-reduce communication overhead.
#[derive(Debug, Clone, Default)]
pub struct AllReduceProfile {
    /// Time spent serializing data for transfer.
    pub serialization_time: Duration,
    /// Time spent in network transfer.
    pub transfer_time: Duration,
    /// Time spent waiting for synchronization.
    pub sync_wait_time: Duration,
    /// Total all-reduce operations performed.
    pub total_operations: u64,
    /// Total bytes transferred across all operations.
    pub total_bytes_transferred: u64,
    /// Average bandwidth achieved (bytes/sec).
    pub average_bandwidth: f64,
}

impl AllReduceProfile {
    /// Total communication time.
    pub fn total_time(&self) -> Duration {
        self.serialization_time + self.transfer_time + self.sync_wait_time
    }
}

impl fmt::Display for AllReduceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "All-Reduce Profile:")?;
        writeln!(
            f,
            "  Serialization: {:.3}ms",
            self.serialization_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Transfer: {:.3}ms",
            self.transfer_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Sync wait: {:.3}ms",
            self.sync_wait_time.as_secs_f64() * 1000.0
        )?;
        writeln!(f, "  Operations: {}", self.total_operations)?;
        let bw_label = if self.average_bandwidth >= 1e9 {
            format!("{:.2} GB/s", self.average_bandwidth / 1e9)
        } else if self.average_bandwidth >= 1e6 {
            format!("{:.2} MB/s", self.average_bandwidth / 1e6)
        } else {
            format!("{:.2} KB/s", self.average_bandwidth / 1e3)
        };
        write!(f, "  Avg bandwidth: {bw_label}")
    }
}

// ---------------------------------------------------------------------------
// Crossover analysis
// ---------------------------------------------------------------------------

/// Analysis of when TP becomes beneficial for a given model size.
#[derive(Debug, Clone)]
pub struct CrossoverAnalysis {
    /// Results for each (model_hidden_size, tp_size) combination.
    pub entries: Vec<CrossoverEntry>,
}

/// A single data point in the crossover analysis.
#[derive(Debug, Clone)]
pub struct CrossoverEntry {
    /// Model hidden dimension used.
    pub model_hidden_size: usize,
    /// TP size used.
    pub tp_size: usize,
    /// Throughput achieved.
    pub throughput_tok_per_sec: f64,
    /// Scaling efficiency relative to single-device.
    pub scaling_efficiency: f64,
    /// All-reduce overhead fraction.
    pub allreduce_overhead: f64,
    /// Whether TP was beneficial (throughput > single-device).
    pub is_beneficial: bool,
}

impl CrossoverAnalysis {
    /// Find the smallest model size where TP becomes beneficial for a given tp_size.
    pub fn crossover_point(&self, tp_size: usize) -> Option<usize> {
        self.entries
            .iter()
            .filter(|e| e.tp_size == tp_size && e.is_beneficial)
            .map(|e| e.model_hidden_size)
            .min()
    }
}

impl fmt::Display for CrossoverAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Crossover Analysis")?;
        writeln!(f, "{:-<70}", "")?;
        writeln!(
            f,
            "  {:>10} | {:>4} | {:>10} | {:>10} | {:>10} | Beneficial",
            "Hidden", "TP", "Tok/s", "Scaling%", "AR Over%"
        )?;
        writeln!(
            f,
            "  {:-<10}-+-{:-<4}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+----------",
            "", "", "", "", ""
        )?;
        for entry in &self.entries {
            writeln!(
                f,
                "  {:>10} | {:>4} | {:>10.1} | {:>9.1}% | {:>9.1}% | {}",
                entry.model_hidden_size,
                entry.tp_size,
                entry.throughput_tok_per_sec,
                entry.scaling_efficiency * 100.0,
                entry.allreduce_overhead * 100.0,
                if entry.is_beneficial { "yes" } else { "no" },
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scaling analysis
// ---------------------------------------------------------------------------

/// Results of a scaling analysis across multiple TP sizes.
#[derive(Debug, Clone)]
pub struct ScalingAnalysis {
    /// Benchmark results for each TP size tested.
    pub results: Vec<TPBenchmarkResult>,
}

impl ScalingAnalysis {
    /// Get the throughput scaling factor between two TP sizes.
    pub fn scaling_factor(&self, from_idx: usize, to_idx: usize) -> Option<f64> {
        let from = self.results.get(from_idx)?;
        let to = self.results.get(to_idx)?;
        if from.throughput_tok_per_sec == 0.0 {
            return None;
        }
        Some(to.throughput_tok_per_sec / from.throughput_tok_per_sec)
    }
}

impl fmt::Display for ScalingAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TP Scaling Analysis")?;
        writeln!(f, "{:-<60}", "")?;
        for result in &self.results {
            writeln!(
                f,
                "  tp_size={}: {:.1} tok/s  AR overhead={:.1}%  scaling={:.1}%",
                result.tp_size,
                result.throughput_tok_per_sec,
                result.allreduce_overhead * 100.0,
                result.scaling_efficiency * 100.0,
            )?;
        }
        if self.results.len() >= 2
            && let Some(factor) = self.scaling_factor(0, 1)
        {
            writeln!(
                f,
                "  Scaling tp_size={} -> {}: {:.2}x",
                self.results[0].tp_size, self.results[1].tp_size, factor,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

trait BenchmarkTimer {
    fn wait(&mut self, duration: Duration);

    fn measure<F>(&mut self, body: F) -> Duration
    where
        F: FnOnce(&mut Self);
}

#[derive(Default)]
struct RealBenchmarkTimer;

impl BenchmarkTimer for RealBenchmarkTimer {
    fn wait(&mut self, duration: Duration) {
        spin_wait(duration);
    }

    fn measure<F>(&mut self, body: F) -> Duration
    where
        F: FnOnce(&mut Self),
    {
        let start = Instant::now();
        body(self);
        start.elapsed()
    }
}

#[derive(Default)]
struct NominalBenchmarkTimer {
    elapsed: Duration,
}

impl BenchmarkTimer for NominalBenchmarkTimer {
    fn wait(&mut self, duration: Duration) {
        self.elapsed += duration;
    }

    fn measure<F>(&mut self, body: F) -> Duration
    where
        F: FnOnce(&mut Self),
    {
        let start = self.elapsed;
        body(self);
        self.elapsed - start
    }
}

/// Run a single TP benchmark scenario.
///
/// Simulates TP inference with the given configuration by running scheduler
/// and executor in lockstep, measuring throughput, latency, and all-reduce
/// overhead using simulated collectives.
pub fn run_tp_benchmark(config: &TPBenchmarkConfig, tp_size: usize) -> Result<TPBenchmarkResult> {
    let mut timer = RealBenchmarkTimer;
    run_tp_benchmark_with_timer(config, tp_size, &mut timer)
}

fn run_tp_benchmark_with_timer<T: BenchmarkTimer>(
    config: &TPBenchmarkConfig,
    tp_size: usize,
    timer: &mut T,
) -> Result<TPBenchmarkResult> {
    config.validate()?;

    let batch_size = config.batch_sizes.first().copied().unwrap_or(1);
    let seq_len = config.seq_lengths.first().copied().unwrap_or(128);

    // Simulated per-step compute time scales inversely with TP size
    // (each rank handles fewer heads/dimensions).
    let compute_per_step = if tp_size > 1 {
        config.layer_compute_time / tp_size as u32
    } else {
        config.layer_compute_time
    };

    // All-reduce overhead only applies when tp_size > 1.
    let ar_overhead_per_step = if tp_size > 1 {
        config.allreduce_time
    } else {
        Duration::ZERO
    };

    let total_step_time = compute_per_step + ar_overhead_per_step;

    // Warmup.
    for _ in 0..config.warmup_steps {
        timer.wait(total_step_time);
    }

    // Measure TTFT: time for the first token (prefill of seq_len tokens).
    // Prefill simulates processing seq_len tokens through the model.
    let prefill_steps = (seq_len + batch_size - 1) / batch_size.max(1);
    let ttft = timer.measure(|timer| {
        for _ in 0..prefill_steps {
            timer.wait(total_step_time);
        }
    });

    // Measure decode steps.
    let mut total_ar_time = Duration::ZERO;
    let mut ar_ops = 0u64;

    let decode_time = timer.measure(|timer| {
        for _ in 0..config.decode_steps {
            timer.wait(compute_per_step);
            if tp_size > 1 {
                total_ar_time += timer.measure(|timer| {
                    timer.wait(ar_overhead_per_step);
                });
                ar_ops += 2; // attention all-reduce + FFN all-reduce
            }
        }
    });

    let total_tokens = batch_size as u64 * config.decode_steps as u64;
    let total_time = ttft + decode_time;

    let throughput = if total_time.is_zero() {
        0.0
    } else {
        total_tokens as f64 / total_time.as_secs_f64()
    };

    let itl = if config.decode_steps <= 1 {
        Duration::ZERO
    } else {
        decode_time / (config.decode_steps - 1)
    };

    let allreduce_overhead = if total_time.is_zero() {
        0.0
    } else {
        total_ar_time.as_secs_f64() / total_time.as_secs_f64()
    };

    // Estimate per-rank memory: weight memory is roughly 1/tp_size,
    // plus replicated components (embeddings, norms) which stay at ~10%.
    let per_rank_memory_fraction = if tp_size > 1 {
        let sharded_fraction = 0.9 / tp_size as f64;
        let replicated_fraction = 0.1;
        sharded_fraction + replicated_fraction
    } else {
        1.0
    };

    // Compute data volume for all-reduce profile.
    let hidden_size = config.num_heads * config.head_dim;
    let tensor_bytes = hidden_size * batch_size * 4; // float32
    let total_bytes = ring_allreduce_data_volume(tensor_bytes, tp_size) as u64 * ar_ops;

    let average_bandwidth = if total_ar_time.is_zero() {
        0.0
    } else {
        total_bytes as f64 / total_ar_time.as_secs_f64()
    };

    let allreduce_profile = AllReduceProfile {
        serialization_time: total_ar_time / 10, // ~10% serialization
        transfer_time: total_ar_time * 7 / 10,  // ~70% transfer
        sync_wait_time: total_ar_time * 2 / 10, // ~20% sync
        total_operations: ar_ops,
        total_bytes_transferred: total_bytes,
        average_bandwidth,
    };

    Ok(TPBenchmarkResult {
        tp_size,
        total_time,
        throughput_tok_per_sec: throughput,
        ttft,
        itl,
        allreduce_overhead,
        scaling_efficiency: 1.0, // Will be updated by scaling analysis
        per_rank_memory_fraction,
        decode_steps: config.decode_steps,
        total_tokens,
        allreduce_profile,
    })
}

/// Run scaling analysis across multiple TP sizes.
///
/// Benchmarks the same workload with each TP size and computes relative
/// scaling efficiency.
pub fn run_scaling_analysis(config: &TPBenchmarkConfig) -> Result<ScalingAnalysis> {
    config.validate()?;

    let mut results = Vec::with_capacity(config.tp_sizes.len());
    let mut baseline_throughput = None;

    for &tp_size in &config.tp_sizes {
        let mut result = run_tp_benchmark(config, tp_size)?;

        if baseline_throughput.is_none() {
            baseline_throughput = Some(result.throughput_tok_per_sec);
        }

        // Scaling efficiency: (actual_throughput) / (baseline * tp_size / baseline_tp)
        if let Some(baseline) = baseline_throughput {
            let first_tp = config.tp_sizes[0] as f64;
            let ideal = baseline * tp_size as f64 / first_tp;
            result.scaling_efficiency = if ideal > 0.0 {
                result.throughput_tok_per_sec / ideal
            } else {
                0.0
            };
        }

        results.push(result);
    }

    Ok(ScalingAnalysis { results })
}

/// Run crossover analysis to determine when TP becomes beneficial.
///
/// Tests each combination of model hidden size and TP size, comparing
/// throughput against the single-device baseline.
/// Uses nominal simulated durations rather than host wall-clock measurements
/// so model-size crossover decisions stay deterministic under runner load.
pub fn run_crossover_analysis(
    base_config: &TPBenchmarkConfig,
    model_hidden_sizes: &[usize],
    tp_sizes: &[usize],
) -> Result<CrossoverAnalysis> {
    run_crossover_analysis_with_timer::<NominalBenchmarkTimer>(
        base_config,
        model_hidden_sizes,
        tp_sizes,
    )
}

fn run_crossover_analysis_with_timer<T>(
    base_config: &TPBenchmarkConfig,
    model_hidden_sizes: &[usize],
    tp_sizes: &[usize],
) -> Result<CrossoverAnalysis>
where
    T: BenchmarkTimer + Default,
{
    base_config.validate()?;

    let mut entries = Vec::new();

    for &hidden_size in model_hidden_sizes {
        let num_heads = hidden_size / base_config.head_dim;
        let num_kv_heads = (num_heads / 4).max(1); // GQA ratio of 4
        let intermediate_size = (hidden_size * 8 / 3 + 255) & !255; // SwiGLU ~8/3x

        // Compute and all-reduce times scale with model size.
        let size_factor = hidden_size as f64 / 4096.0;
        let compute_time = Duration::from_micros((200.0 * size_factor * size_factor) as u64);
        let ar_time = Duration::from_micros((50.0 * size_factor) as u64);

        let mut config = base_config.clone();
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.intermediate_size = intermediate_size;
        config.layer_compute_time = compute_time;
        config.allreduce_time = ar_time;

        // Baseline: tp_size=1. Crossover analysis is a model comparison,
        // so it uses injected nominal time instead of wall-clock measurements.
        // The general TP benchmark path still uses `RealBenchmarkTimer`.
        let mut baseline_timer = T::default();
        let baseline = run_tp_benchmark_with_timer(&config, 1, &mut baseline_timer)?;

        for &tp_size in tp_sizes {
            if tp_size == 1 {
                entries.push(CrossoverEntry {
                    model_hidden_size: hidden_size,
                    tp_size: 1,
                    throughput_tok_per_sec: baseline.throughput_tok_per_sec,
                    scaling_efficiency: 1.0,
                    allreduce_overhead: 0.0,
                    is_beneficial: false, // Baseline is not "beneficial" by definition.
                });
                continue;
            }

            // Check if num_heads is divisible by tp_size.
            if !num_heads.is_multiple_of(tp_size) {
                continue;
            }

            let mut result_timer = T::default();
            let result = run_tp_benchmark_with_timer(&config, tp_size, &mut result_timer)?;
            let ideal_throughput = baseline.throughput_tok_per_sec * tp_size as f64;
            let scaling_efficiency = if ideal_throughput > 0.0 {
                result.throughput_tok_per_sec / ideal_throughput
            } else {
                0.0
            };

            let is_beneficial = result.throughput_tok_per_sec > baseline.throughput_tok_per_sec;

            entries.push(CrossoverEntry {
                model_hidden_size: hidden_size,
                tp_size,
                throughput_tok_per_sec: result.throughput_tok_per_sec,
                scaling_efficiency,
                allreduce_overhead: result.allreduce_overhead,
                is_beneficial,
            });
        }
    }

    Ok(CrossoverAnalysis { entries })
}

/// Format a TP benchmark report as a human-readable string.
pub fn format_tp_benchmark_report(results: &[TPBenchmarkResult]) -> String {
    let mut report = String::new();
    report.push_str("Tensor Parallelism Benchmark Report\n");
    report.push_str(&"=".repeat(70));
    report.push('\n');

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            report.push_str(&"-".repeat(70));
            report.push('\n');
        }
        report.push_str(&format!("{result}\n"));
    }

    report.push_str(&"=".repeat(70));
    report.push('\n');

    // Summary table.
    report.push_str("\nSummary:\n");
    report.push_str("  TP | Throughput  | TTFT     | ITL      | AR Over% | Scaling% | Mem%\n");
    report.push_str("  ---+------------+----------+----------+----------+----------+-----\n");
    for result in results {
        report.push_str(&format!(
            "  {:>2} | {:>8.1} t/s | {:>6.2}ms | {:>6.2}ms | {:>7.1}% | {:>7.1}% | {:>3.0}%\n",
            result.tp_size,
            result.throughput_tok_per_sec,
            result.ttft.as_secs_f64() * 1000.0,
            result.itl.as_secs_f64() * 1000.0,
            result.allreduce_overhead * 100.0,
            result.scaling_efficiency * 100.0,
            result.per_rank_memory_fraction * 100.0,
        ));
    }

    report
}

// ---------------------------------------------------------------------------
// Scheduler/executor correctness benchmark
// ---------------------------------------------------------------------------

/// Run a scheduler-executor lockstep benchmark to verify correctness
/// under simulated TP execution.
///
/// Creates a scheduler (rank 0) and multiple executors (all ranks),
/// processes sequences through prefill and decode, and verifies
/// all executors stay synchronized.
pub fn run_lockstep_benchmark(
    tp_size: usize,
    num_sequences: usize,
    max_tokens: usize,
) -> Result<LockstepBenchmarkResult> {
    ensure!(tp_size >= 1, "tp_size must be >= 1");
    ensure!(num_sequences >= 1, "num_sequences must be >= 1");

    let exec_config = TPExecutionConfig::new(0, tp_size);
    let sampling_mode = SamplingMode::ReplicatedLmHead;

    let mut scheduler = TPScheduler::new(exec_config.clone(), sampling_mode)?;

    // Create executors for all ranks.
    let mut executors: Vec<TPExecutor> = Vec::with_capacity(tp_size);
    for rank in 0..tp_size {
        let rank_config = TPExecutionConfig::new(rank, tp_size);
        executors.push(TPExecutor::new(rank_config, sampling_mode)?);
    }

    // Submit sequences.
    for seq_id in 0..num_sequences as u64 {
        scheduler.submit_sequence(seq_id, 10, max_tokens, 0)?;
    }

    let start = Instant::now();
    let mut total_steps = 0u64;
    let mut total_tokens_generated = 0u64;

    // Run until no more work.
    let max_iterations = num_sequences * max_tokens * 10;
    for _ in 0..max_iterations {
        if !scheduler.has_work() {
            break;
        }

        let decisions = scheduler.schedule_step()?;

        for decision in &decisions {
            for executor in &mut executors {
                let outcome = executor.execute_step(decision)?;
                match &outcome {
                    StepOutcome::Executed { .. }
                    | StepOutcome::SequenceAdmitted { .. }
                    | StepOutcome::SequenceEvicted { .. } => {}
                    StepOutcome::Error { message, .. } => {
                        anyhow::bail!("executor error: {message}");
                    }
                    _ => {}
                }
            }
            total_steps += 1;
        }

        // Simulate decode: produce tokens for active decoding sequences.
        if scheduler.active_count() > 0 {
            let active = scheduler.active_count();
            let tokens: Vec<u32> = (0..active).map(|i| 100 + i as u32).collect();

            // Check which sequences are complete (generated enough tokens).
            let mut completed = Vec::new();
            for seq_id in 0..num_sequences as u64 {
                if let Some(info) = scheduler.sequence_info(seq_id)
                    && info.generated_tokens + 1 >= max_tokens
                {
                    completed.push(seq_id);
                }
            }

            if tokens.is_empty() {
                continue;
            }

            let sampled = SampledTokens {
                step_id: 0,
                tokens,
                completed_seq_ids: completed,
            };

            let evictions = scheduler.record_sampled_tokens(&sampled)?;
            for executor in &mut executors {
                executor.apply_sampled_tokens(&sampled)?;
            }

            total_tokens_generated += active as u64;

            // Broadcast eviction decisions.
            for decision in &evictions {
                for executor in &mut executors {
                    executor.execute_step(decision)?;
                }
            }
        }
    }

    let elapsed = start.elapsed();

    // Verify all executors are synchronized.
    let all_synchronized = executors
        .windows(2)
        .all(|w| w[0].active_count() == w[1].active_count());

    Ok(LockstepBenchmarkResult {
        tp_size,
        num_sequences,
        max_tokens,
        total_steps,
        total_tokens_generated,
        elapsed,
        all_synchronized,
        errors: executors.iter().map(|e| e.error_count()).sum(),
    })
}

/// Result of a lockstep scheduler/executor benchmark.
#[derive(Debug, Clone)]
pub struct LockstepBenchmarkResult {
    /// TP size used.
    pub tp_size: usize,
    /// Number of sequences processed.
    pub num_sequences: usize,
    /// Maximum tokens per sequence.
    pub max_tokens: usize,
    /// Total scheduler steps executed.
    pub total_steps: u64,
    /// Total tokens generated across all sequences.
    pub total_tokens_generated: u64,
    /// Wall-clock time.
    pub elapsed: Duration,
    /// Whether all executors stayed synchronized.
    pub all_synchronized: bool,
    /// Total errors across all executors.
    pub errors: u64,
}

impl fmt::Display for LockstepBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Lockstep Benchmark: tp_size={}, seqs={}, max_tokens={}",
            self.tp_size, self.num_sequences, self.max_tokens
        )?;
        writeln!(
            f,
            "  Steps: {}  Tokens: {}  Time: {:.2}ms",
            self.total_steps,
            self.total_tokens_generated,
            self.elapsed.as_secs_f64() * 1000.0,
        )?;
        write!(
            f,
            "  Synchronized: {}  Errors: {}",
            self.all_synchronized, self.errors
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Busy-wait for the given duration (sub-millisecond accuracy).
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
