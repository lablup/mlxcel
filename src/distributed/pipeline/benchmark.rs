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

//! Pipeline parallelism benchmark harness.
//!
//! Provides configurable benchmarks for measuring pipeline throughput,
//! latency, bubble ratio, and scaling characteristics. Benchmarks run
//! using the simulated pipeline schedule and metrics infrastructure --
//! no real model or hardware required.
//!
//! Key types:
//!
//! - [`PipelineBenchmarkConfig`] -- benchmark parameters
//! - [`PipelineBenchmarkResult`] -- throughput, TTFT, TPOT, bubble ratio
//! - [`ScalingResult`] -- multi-stage scaling comparison
//! - [`run_pipeline_benchmark`] -- execute a single benchmark scenario
//! - [`run_scaling_benchmark`] -- compare across 2/4 stage counts
//! - [`format_benchmark_report`] -- human-readable report
//!
//! Used by: integration tests, CI benchmarks

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};

use super::metrics::{MetricsCollector, PipelineMetrics};
use super::schedule::{GPipeSchedule, PipelineConfig, PipelineSchedule, ScheduleAction};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a pipeline benchmark run.
#[derive(Debug, Clone)]
pub struct PipelineBenchmarkConfig {
    /// Number of pipeline stages.
    pub num_stages: u32,
    /// Number of micro-batches per step.
    pub num_micro_batches: u32,
    /// Simulated compute time per stage per micro-batch.
    pub stage_compute_time: Duration,
    /// Simulated activation transfer time between stages.
    pub transfer_time: Duration,
    /// Number of decode steps (tokens) to simulate per sequence.
    pub decode_steps: u32,
    /// Sequence length (prompt tokens) for TTFT measurement.
    pub sequence_length: usize,
    /// Number of warmup steps before measurement.
    pub warmup_steps: u32,
}

impl Default for PipelineBenchmarkConfig {
    fn default() -> Self {
        Self {
            num_stages: 2,
            num_micro_batches: 4,
            stage_compute_time: Duration::from_micros(500),
            transfer_time: Duration::from_micros(50),
            decode_steps: 32,
            sequence_length: 128,
            warmup_steps: 2,
        }
    }
}

impl PipelineBenchmarkConfig {
    /// Create a config with the given stage count and micro-batch count.
    pub fn new(num_stages: u32, num_micro_batches: u32) -> Self {
        Self {
            num_stages,
            num_micro_batches,
            ..Default::default()
        }
    }

    /// Set simulated compute time per stage.
    #[must_use]
    pub fn with_compute_time(mut self, t: Duration) -> Self {
        self.stage_compute_time = t;
        self
    }

    /// Set simulated transfer time between stages.
    #[must_use]
    pub fn with_transfer_time(mut self, t: Duration) -> Self {
        self.transfer_time = t;
        self
    }

    /// Set number of decode steps.
    #[must_use]
    pub fn with_decode_steps(mut self, n: u32) -> Self {
        self.decode_steps = n;
        self
    }

    /// Set sequence length.
    #[must_use]
    pub fn with_sequence_length(mut self, n: usize) -> Self {
        self.sequence_length = n;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_stages >= 2, "benchmark requires at least 2 stages");
        ensure!(self.num_micro_batches > 0, "num_micro_batches must be > 0");
        ensure!(self.decode_steps > 0, "decode_steps must be > 0");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Results from a single pipeline benchmark run.
#[derive(Debug, Clone)]
pub struct PipelineBenchmarkResult {
    /// Configuration used for this run.
    pub config: PipelineBenchmarkConfig,
    /// Total wall-clock time for the benchmark (excluding warmup).
    pub total_time: Duration,
    /// Tokens generated per second (across all sequences).
    pub throughput_tok_per_sec: f64,
    /// Time to first token (pipeline fill latency).
    pub ttft: Duration,
    /// Average time per output token during decode.
    pub tpot: Duration,
    /// Measured bubble ratio (from pipeline metrics).
    pub bubble_ratio: f64,
    /// Theoretical optimal bubble ratio for GPipe.
    pub theoretical_bubble_ratio: f64,
    /// Average stage utilization.
    pub average_utilization: f64,
    /// Number of decode steps simulated.
    pub decode_steps: u32,
    /// Total tokens generated.
    pub total_tokens: u64,
}

impl PipelineBenchmarkResult {
    /// Pipeline efficiency: ratio of theoretical to measured bubble ratio.
    /// Values close to 1.0 indicate the schedule is operating near the
    /// theoretical optimum. Values > 1.0 mean measured is better than
    /// theoretical (possible with measurement noise).
    pub fn bubble_efficiency(&self) -> f64 {
        if self.theoretical_bubble_ratio == 0.0 || self.bubble_ratio == 0.0 {
            return 1.0;
        }
        // Ratio of theoretical/measured: values near 1.0 = good.
        self.theoretical_bubble_ratio / self.bubble_ratio
    }
}

impl fmt::Display for PipelineBenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Pipeline Benchmark: {} stages, {} micro-batches",
            self.config.num_stages, self.config.num_micro_batches
        )?;
        writeln!(
            f,
            "  Throughput: {:.1} tok/s ({} tokens in {:.2}ms)",
            self.throughput_tok_per_sec,
            self.total_tokens,
            self.total_time.as_secs_f64() * 1000.0,
        )?;
        writeln!(
            f,
            "  TTFT: {:.3}ms  TPOT: {:.3}ms",
            self.ttft.as_secs_f64() * 1000.0,
            self.tpot.as_secs_f64() * 1000.0,
        )?;
        writeln!(
            f,
            "  Bubble ratio: {:.1}% (theoretical: {:.1}%, efficiency: {:.2}x)",
            self.bubble_ratio * 100.0,
            self.theoretical_bubble_ratio * 100.0,
            self.bubble_efficiency(),
        )?;
        write!(
            f,
            "  Avg utilization: {:.1}%",
            self.average_utilization * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// Scaling results
// ---------------------------------------------------------------------------

/// Comparison of benchmark results across different stage counts.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    /// Results for each stage configuration tested.
    pub results: Vec<PipelineBenchmarkResult>,
}

impl ScalingResult {
    /// Compute the scaling efficiency between two stage counts.
    ///
    /// Returns the throughput ratio: `result[to_idx] / result[from_idx]`.
    pub fn scaling_factor(&self, from_idx: usize, to_idx: usize) -> Option<f64> {
        let from = self.results.get(from_idx)?;
        let to = self.results.get(to_idx)?;
        if from.throughput_tok_per_sec == 0.0 {
            return None;
        }
        Some(to.throughput_tok_per_sec / from.throughput_tok_per_sec)
    }
}

impl fmt::Display for ScalingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Scaling Analysis")?;
        writeln!(f, "{:-<60}", "")?;
        for result in &self.results {
            writeln!(
                f,
                "  {} stages: {:.1} tok/s  bubble={:.1}%  util={:.1}%",
                result.config.num_stages,
                result.throughput_tok_per_sec,
                result.bubble_ratio * 100.0,
                result.average_utilization * 100.0,
            )?;
        }
        if self.results.len() >= 2
            && let Some(factor) = self.scaling_factor(0, 1)
        {
            writeln!(
                f,
                "  Scaling {} -> {} stages: {:.2}x",
                self.results[0].config.num_stages, self.results[1].config.num_stages, factor,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

/// Run a single pipeline benchmark scenario.
///
/// Simulates a GPipe schedule with the given configuration, measuring
/// throughput, latency, and pipeline utilization metrics.
pub fn run_pipeline_benchmark(config: &PipelineBenchmarkConfig) -> Result<PipelineBenchmarkResult> {
    config.validate()?;

    let pipe_config = PipelineConfig::new(config.num_stages, 1)?;
    let mut collector = MetricsCollector::new(config.num_stages);

    // Warmup: run schedule steps without measurement.
    for _ in 0..config.warmup_steps {
        let _ = simulate_pipeline_step(
            &pipe_config,
            config.num_micro_batches,
            config.stage_compute_time,
            config.transfer_time,
        )?;
    }

    // Measure TTFT: time for the first micro-batch to complete all stages.
    let ttft_start = Instant::now();
    let first_step_metrics = simulate_pipeline_step(
        &pipe_config,
        config.num_micro_batches,
        config.stage_compute_time,
        config.transfer_time,
    )?;
    let ttft = ttft_start.elapsed();

    collector.begin_step();
    collector.end_step(first_step_metrics);

    // Measure decode steps.
    let decode_start = Instant::now();
    for _ in 1..config.decode_steps {
        collector.begin_step();
        let step_metrics = simulate_pipeline_step(
            &pipe_config,
            config.num_micro_batches,
            config.stage_compute_time,
            config.transfer_time,
        )?;
        collector.end_step(step_metrics);
    }
    let decode_time = decode_start.elapsed();

    // Compute totals.
    let total_tokens = config.num_micro_batches as u64 * config.decode_steps as u64;
    let total_time = ttft + decode_time;

    let throughput = if total_time.is_zero() {
        0.0
    } else {
        total_tokens as f64 / total_time.as_secs_f64()
    };

    let tpot = if config.decode_steps <= 1 {
        Duration::ZERO
    } else {
        decode_time / (config.decode_steps - 1)
    };

    let summary = collector.summary();

    // Theoretical bubble ratio for GPipe: (S-1) / (S-1+M).
    let theoretical = if config.num_stages > 1 {
        let s = config.num_stages as f64;
        let m = config.num_micro_batches as f64;
        (s - 1.0) / (s - 1.0 + m)
    } else {
        0.0
    };

    Ok(PipelineBenchmarkResult {
        config: config.clone(),
        total_time,
        throughput_tok_per_sec: throughput,
        ttft,
        tpot,
        bubble_ratio: summary.average_bubble_ratio,
        theoretical_bubble_ratio: theoretical,
        average_utilization: summary.average_utilization,
        decode_steps: config.decode_steps,
        total_tokens,
    })
}

/// Run scaling benchmarks across multiple stage configurations.
///
/// Tests the same workload with each stage count in `stage_counts` and
/// returns comparative results.
pub fn run_scaling_benchmark(
    stage_counts: &[u32],
    num_micro_batches: u32,
    compute_time: Duration,
    transfer_time: Duration,
    decode_steps: u32,
) -> Result<ScalingResult> {
    ensure!(!stage_counts.is_empty(), "stage_counts must not be empty");

    let mut results = Vec::with_capacity(stage_counts.len());

    for &stages in stage_counts {
        let config = PipelineBenchmarkConfig {
            num_stages: stages,
            num_micro_batches,
            stage_compute_time: compute_time,
            transfer_time,
            decode_steps,
            ..Default::default()
        };
        let result = run_pipeline_benchmark(&config)?;
        results.push(result);
    }

    Ok(ScalingResult { results })
}

/// Format a benchmark report as a human-readable string.
pub fn format_benchmark_report(results: &[PipelineBenchmarkResult]) -> String {
    let mut report = String::new();
    report.push_str("Pipeline Parallelism Benchmark Report\n");
    report.push_str(&"=".repeat(60));
    report.push('\n');

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            report.push_str(&"-".repeat(60));
            report.push('\n');
        }
        report.push_str(&format!("{result}\n"));
    }

    report.push_str(&"=".repeat(60));
    report.push('\n');

    // Summary table.
    report.push_str("\nSummary:\n");
    report.push_str("  Stages | MBs | Throughput  | Bubble | Utilization\n");
    report.push_str("  -------+-----+------------+--------+------------\n");
    for result in results {
        report.push_str(&format!(
            "  {:>6} | {:>3} | {:>8.1} t/s | {:>5.1}% | {:>9.1}%\n",
            result.config.num_stages,
            result.config.num_micro_batches,
            result.throughput_tok_per_sec,
            result.bubble_ratio * 100.0,
            result.average_utilization * 100.0,
        ));
    }

    report
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Simulate one pipeline step through the GPipe schedule.
///
/// Drives the schedule to completion, simulating compute and transfer
/// times, and returns the collected metrics.
fn simulate_pipeline_step(
    pipe_config: &PipelineConfig,
    num_micro_batches: u32,
    compute_time: Duration,
    transfer_time: Duration,
) -> Result<PipelineMetrics> {
    let mut schedule = GPipeSchedule::new(pipe_config.clone(), num_micro_batches)?;
    let mut metrics = PipelineMetrics::new(pipe_config.num_stages, num_micro_batches);

    let step_start = Instant::now();

    // Drive the schedule to completion.
    let mut iterations = 0;
    let max_iterations = (num_micro_batches * pipe_config.num_stages * 4 + 100) as usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        let action = schedule.next_action();
        match action {
            ScheduleAction::Forward {
                stage_index,
                micro_batch_id,
            } => {
                // Simulate compute time.
                spin_wait(compute_time);
                metrics.record_compute(stage_index, compute_time);
                metrics.record_micro_batch_processed(stage_index);

                schedule.notify_forward_complete(stage_index, micro_batch_id);
            }
            ScheduleAction::Receive {
                stage_index,
                micro_batch_id: _,
            } => {
                // Simulate transfer wait.
                spin_wait(transfer_time);
                metrics.record_transfer(stage_index, transfer_time);
            }
            ScheduleAction::Flush { micro_batch_id } => {
                schedule.notify_flush_complete(micro_batch_id);
                metrics.sequences_completed += 1;
            }
            ScheduleAction::Idle => {
                // In simulation, Idle after all actions means we are waiting
                // for something that won't come -- break to avoid infinite loop.
                if schedule.is_complete() {
                    break;
                }
                // The schedule may have new actions after prior notifications.
                continue;
            }
            ScheduleAction::Done => break,
        }
    }

    metrics.wall_time = step_start.elapsed();
    Ok(metrics)
}

/// Busy-wait for the given duration to simulate compute/transfer time.
///
/// Uses a spin loop instead of thread::sleep for sub-millisecond accuracy.
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
