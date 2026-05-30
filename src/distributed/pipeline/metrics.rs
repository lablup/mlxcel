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

//! Pipeline parallelism metrics collection and reporting.
//!
//! Tracks bubble ratio, per-stage utilization, and latency breakdown for
//! the pipeline schedule. Metrics are collected per-step and can be
//! aggregated for reporting.
//!
//! Also exposes counters/histograms for elastic repartition events. The sink API is intentionally transport-agnostic — the Prometheus
//! endpoint reads from [`RepartitionMetricsSnapshot`] while unit tests
//! inspect the raw [`RepartitionMetrics`] struct.
//!
//! Used by: pipeline schedule, pipeline execution loop, server metrics endpoint

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::elastic::{RepartitionEvent, RepartitionEventSink, RepartitionOutcome};

/// Per-stage timing breakdown for a single pipeline step.
#[derive(Debug, Clone)]
pub struct StageMetrics {
    /// Index of this stage (0-based).
    pub stage_index: u32,
    /// Time spent in forward computation (model forward pass).
    pub compute_time: Duration,
    /// Time spent waiting for input activations from the previous stage.
    pub wait_time: Duration,
    /// Time spent transferring activations to the next stage.
    pub transfer_time: Duration,
    /// Number of micro-batches processed in this step.
    pub micro_batches_processed: u32,
}

impl StageMetrics {
    /// Create empty metrics for a stage.
    pub fn new(stage_index: u32) -> Self {
        Self {
            stage_index,
            compute_time: Duration::ZERO,
            wait_time: Duration::ZERO,
            transfer_time: Duration::ZERO,
            micro_batches_processed: 0,
        }
    }

    /// Total time this stage was active (compute + wait + transfer).
    pub fn total_time(&self) -> Duration {
        self.compute_time + self.wait_time + self.transfer_time
    }

    /// Stage utilization: fraction of total time spent computing.
    /// Returns 0.0 if total time is zero.
    pub fn utilization(&self) -> f64 {
        let total = self.total_time();
        if total.is_zero() {
            return 0.0;
        }
        self.compute_time.as_secs_f64() / total.as_secs_f64()
    }
}

impl fmt::Display for StageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stage {} | compute={:.2}ms wait={:.2}ms transfer={:.2}ms util={:.1}% mbs={}",
            self.stage_index,
            self.compute_time.as_secs_f64() * 1000.0,
            self.wait_time.as_secs_f64() * 1000.0,
            self.transfer_time.as_secs_f64() * 1000.0,
            self.utilization() * 100.0,
            self.micro_batches_processed,
        )
    }
}

/// Aggregated pipeline metrics for a single step or across steps.
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Per-stage metrics, keyed by stage index.
    pub stages: HashMap<u32, StageMetrics>,
    /// Total wall-clock time for this pipeline step.
    pub wall_time: Duration,
    /// Number of micro-batches in this step.
    pub num_micro_batches: u32,
    /// Number of pipeline stages.
    pub num_stages: u32,
    /// Number of completed (flushed) sequences in this step.
    pub sequences_completed: u32,
}

impl PipelineMetrics {
    /// Create empty pipeline metrics.
    pub fn new(num_stages: u32, num_micro_batches: u32) -> Self {
        let stages = (0..num_stages).map(|i| (i, StageMetrics::new(i))).collect();
        Self {
            stages,
            wall_time: Duration::ZERO,
            num_micro_batches,
            num_stages,
            sequences_completed: 0,
        }
    }

    /// Compute the pipeline bubble ratio.
    ///
    /// Bubble ratio = 1 - (sum of compute times) / (num_stages * wall_time).
    /// A perfect pipeline has bubble ratio 0; a fully serial pipeline has
    /// bubble ratio (N-1)/N.
    ///
    /// Returns 0.0 if wall_time is zero.
    pub fn bubble_ratio(&self) -> f64 {
        if self.wall_time.is_zero() || self.num_stages == 0 {
            return 0.0;
        }
        let total_compute: f64 = self
            .stages
            .values()
            .map(|s| s.compute_time.as_secs_f64())
            .sum();
        let ideal_total = self.num_stages as f64 * self.wall_time.as_secs_f64();
        1.0 - (total_compute / ideal_total)
    }

    /// Average stage utilization across all stages.
    pub fn average_utilization(&self) -> f64 {
        if self.stages.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.stages.values().map(|s| s.utilization()).sum();
        sum / self.stages.len() as f64
    }

    /// Theoretical optimal bubble ratio for GPipe schedule.
    ///
    /// For GPipe with S stages and M micro-batches:
    /// bubble_fraction = (S - 1) / (S - 1 + M)
    pub fn theoretical_bubble_ratio(&self) -> f64 {
        if self.num_stages <= 1 || self.num_micro_batches == 0 {
            return 0.0;
        }
        let s = self.num_stages as f64;
        let m = self.num_micro_batches as f64;
        (s - 1.0) / (s - 1.0 + m)
    }

    /// Record stage compute time.
    pub fn record_compute(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.compute_time += duration;
        }
    }

    /// Record stage wait time.
    pub fn record_wait(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.wait_time += duration;
        }
    }

    /// Record stage transfer time.
    pub fn record_transfer(&mut self, stage_index: u32, duration: Duration) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.transfer_time += duration;
        }
    }

    /// Increment the micro-batch count for a stage.
    pub fn record_micro_batch_processed(&mut self, stage_index: u32) {
        if let Some(stage) = self.stages.get_mut(&stage_index) {
            stage.micro_batches_processed += 1;
        }
    }
}

impl fmt::Display for PipelineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Pipeline | stages={} micro_batches={} wall={:.2}ms bubble={:.1}% avg_util={:.1}%",
            self.num_stages,
            self.num_micro_batches,
            self.wall_time.as_secs_f64() * 1000.0,
            self.bubble_ratio() * 100.0,
            self.average_utilization() * 100.0,
        )?;
        let mut stage_indices: Vec<u32> = self.stages.keys().copied().collect();
        stage_indices.sort();
        for idx in stage_indices {
            if let Some(stage) = self.stages.get(&idx) {
                writeln!(f, "  {stage}")?;
            }
        }
        Ok(())
    }
}

/// Accumulates pipeline metrics across multiple steps for aggregate reporting.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Number of pipeline stages (fixed for the lifetime of the pipeline).
    num_stages: u32,
    /// All recorded step metrics.
    step_metrics: Vec<PipelineMetrics>,
    /// Running wall-clock start time for the current step.
    step_start: Option<Instant>,
    /// Total sequences completed across all steps.
    total_sequences_completed: u64,
    /// Total micro-batches processed across all steps.
    total_micro_batches: u64,
}

impl MetricsCollector {
    /// Create a new collector for a pipeline with `num_stages` stages.
    pub fn new(num_stages: u32) -> Self {
        Self {
            num_stages,
            step_metrics: Vec::new(),
            step_start: None,
            total_sequences_completed: 0,
            total_micro_batches: 0,
        }
    }

    /// Mark the start of a new pipeline step.
    pub fn begin_step(&mut self) {
        self.step_start = Some(Instant::now());
    }

    /// Finalize the current step with the given metrics.
    pub fn end_step(&mut self, mut metrics: PipelineMetrics) {
        if let Some(start) = self.step_start.take() {
            metrics.wall_time = start.elapsed();
        }
        self.total_sequences_completed += metrics.sequences_completed as u64;
        self.total_micro_batches += metrics.num_micro_batches as u64;
        self.step_metrics.push(metrics);
    }

    /// Number of steps recorded so far.
    pub fn num_steps(&self) -> usize {
        self.step_metrics.len()
    }

    /// Total sequences completed across all steps.
    pub fn total_sequences_completed(&self) -> u64 {
        self.total_sequences_completed
    }

    /// Compute the average bubble ratio across all recorded steps.
    pub fn average_bubble_ratio(&self) -> f64 {
        if self.step_metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.step_metrics.iter().map(|m| m.bubble_ratio()).sum();
        sum / self.step_metrics.len() as f64
    }

    /// Compute the average stage utilization across all steps.
    pub fn average_utilization(&self) -> f64 {
        if self.step_metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .step_metrics
            .iter()
            .map(|m| m.average_utilization())
            .sum();
        sum / self.step_metrics.len() as f64
    }

    /// Get a snapshot summary of collected metrics.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            num_stages: self.num_stages,
            num_steps: self.step_metrics.len() as u64,
            total_sequences_completed: self.total_sequences_completed,
            total_micro_batches: self.total_micro_batches,
            average_bubble_ratio: self.average_bubble_ratio(),
            average_utilization: self.average_utilization(),
        }
    }
}

/// Summary snapshot of pipeline metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub num_stages: u32,
    pub num_steps: u64,
    pub total_sequences_completed: u64,
    pub total_micro_batches: u64,
    pub average_bubble_ratio: f64,
    pub average_utilization: f64,
}

impl fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pipeline Summary | stages={} steps={} seqs={} mbs={} bubble={:.1}% util={:.1}%",
            self.num_stages,
            self.num_steps,
            self.total_sequences_completed,
            self.total_micro_batches,
            self.average_bubble_ratio * 100.0,
            self.average_utilization * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Activation transfer latency histogram
// ---------------------------------------------------------------------------

/// Monotonic-merge histogram for activation transfer latency samples.
///
/// Samples are partitioned by `(src_stage, dst_stage)` so the Prometheus
/// renderer can emit per-pair p50/p95/p99 quantile buckets. The histogram
/// is approximate: we keep a fixed-size log-linear bucket layout rather
/// than a full sample array so memory stays bounded under sustained traffic.
///
/// Used by: stage workers (sample producers), `/metrics` endpoint (renderer).
#[derive(Debug, Default)]
pub struct ActivationLatencyHistogram {
    inner: Mutex<HashMap<(u32, u32), LatencyBuckets>>,
}

impl ActivationLatencyHistogram {
    /// Construct an empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single observation.
    pub fn observe(&self, src_stage: u32, dst_stage: u32, latency: Duration) {
        let us = latency.as_micros().min(u128::from(u64::MAX)) as u64;
        let mut map = self.inner.lock().expect("activation latency poisoned");
        map.entry((src_stage, dst_stage)).or_default().record(us);
    }

    /// Snapshot for rendering. Emits one entry per stage pair ordered
    /// `(src_stage, dst_stage)` ascending so output is stable.
    pub fn snapshot(&self) -> Vec<ActivationLatencyPair> {
        let map = self.inner.lock().expect("activation latency poisoned");
        let mut pairs: Vec<_> = map
            .iter()
            .map(|(&(src, dst), buckets)| ActivationLatencyPair {
                src_stage: src,
                dst_stage: dst,
                count: buckets.count,
                p50_us: buckets.quantile(0.50),
                p95_us: buckets.quantile(0.95),
                p99_us: buckets.quantile(0.99),
                max_us: buckets.max,
            })
            .collect();
        pairs.sort_by_key(|a| (a.src_stage, a.dst_stage));
        pairs
    }
}

/// Rendered per-stage-pair latency summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationLatencyPair {
    pub src_stage: u32,
    pub dst_stage: u32,
    pub count: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

/// Log-linear histogram buckets.
///
/// Covers 1 μs to ~1 s with 24 exponentially spaced bins; that is the
/// working range for activation transfer (single-digit μs on a local link,
/// hundreds of ms on a slow WAN link). Observations above the ceiling pin
/// to the top bucket, which is what we want for the p99 tail indicator.
#[derive(Debug, Clone, Default)]
struct LatencyBuckets {
    buckets: [u64; Self::BUCKET_COUNT],
    count: u64,
    max: u64,
}

impl LatencyBuckets {
    const BUCKET_COUNT: usize = 24;

    fn bucket_upper_bound(idx: usize) -> u64 {
        // Powers of two, starting at 1 microsecond: 1, 2, 4, ..., 2^(N-1).
        1u64 << idx.min(63)
    }

    fn record(&mut self, us: u64) {
        self.count = self.count.saturating_add(1);
        if us > self.max {
            self.max = us;
        }
        let idx = Self::bucket_index(us);
        self.buckets[idx] = self.buckets[idx].saturating_add(1);
    }

    fn bucket_index(us: u64) -> usize {
        if us == 0 {
            return 0;
        }
        let highest_bit = 63 - us.leading_zeros() as usize;
        (highest_bit + 1).min(Self::BUCKET_COUNT - 1)
    }

    fn quantile(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((self.count as f64) * q).ceil() as u64;
        let mut seen = 0u64;
        for (idx, &count) in self.buckets.iter().enumerate() {
            seen += count;
            if seen >= target {
                return Self::bucket_upper_bound(idx);
            }
        }
        self.max
    }
}

// ---------------------------------------------------------------------------
// KV cache admission rejection counters
// ---------------------------------------------------------------------------

/// Counters for KV cache admission rejections, keyed by
/// `(stage_index, reason)`.
///
/// Used by: cache admission path (`cache_manager.rs`), `/metrics` endpoint.
#[derive(Debug, Default)]
pub struct AdmissionRejectionCounters {
    inner: Mutex<HashMap<(u32, &'static str), u64>>,
}

impl AdmissionRejectionCounters {
    /// Construct an empty counter set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a rejection on `stage_index` with the given short reason label.
    pub fn record(&self, stage_index: u32, reason: &'static str) {
        let mut map = self.inner.lock().expect("admission counters poisoned");
        *map.entry((stage_index, reason)).or_insert(0) += 1;
    }

    /// Snapshot sorted `(stage, reason)` ascending for stable output.
    pub fn snapshot(&self) -> Vec<AdmissionRejectionEntry> {
        let map = self.inner.lock().expect("admission counters poisoned");
        let mut entries: Vec<_> = map
            .iter()
            .map(|(&(stage, reason), &count)| AdmissionRejectionEntry {
                stage_index: stage,
                reason,
                count,
            })
            .collect();
        entries.sort_by(|a, b| (a.stage_index, a.reason).cmp(&(b.stage_index, b.reason)));
        entries
    }
}

/// Rendered rejection counter entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionRejectionEntry {
    pub stage_index: u32,
    pub reason: &'static str,
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Stage utilization snapshot
// ---------------------------------------------------------------------------

/// Compact per-stage utilization snapshot intended for Prometheus rendering.
///
/// The full [`MetricsCollector`] tracks rich per-step breakdowns; this
/// container captures the fraction actually needed by operators watching
/// dashboards (busy fraction, bubble contribution) without the serialization
/// overhead of shipping an entire `PipelineMetrics` snapshot per scrape.
#[derive(Debug, Default)]
pub struct StageUtilizationRegistry {
    inner: Mutex<HashMap<u32, StageUtilizationAccumulator>>,
}

impl StageUtilizationRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sample for `stage_index`. `busy` is time spent on compute
    /// plus transfer; `total` includes wait/bubble time.
    pub fn record(&self, stage_index: u32, busy: Duration, total: Duration) {
        let mut map = self.inner.lock().expect("stage utilization poisoned");
        map.entry(stage_index).or_default().add(busy, total);
    }

    /// Snapshot ordered by stage index.
    pub fn snapshot(&self) -> Vec<StageUtilizationSnapshot> {
        let map = self.inner.lock().expect("stage utilization poisoned");
        let mut entries: Vec<_> = map
            .iter()
            .map(|(&stage, acc)| StageUtilizationSnapshot {
                stage_index: stage,
                busy_us: acc.busy_us,
                total_us: acc.total_us,
            })
            .collect();
        entries.sort_by_key(|e| e.stage_index);
        entries
    }
}

/// Rendered per-stage utilization sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageUtilizationSnapshot {
    pub stage_index: u32,
    pub busy_us: u64,
    pub total_us: u64,
}

impl StageUtilizationSnapshot {
    /// Busy fraction `[0.0, 1.0]`, or `0.0` if no samples have been recorded.
    pub fn busy_fraction(&self) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            self.busy_us as f64 / self.total_us as f64
        }
    }

    /// Bubble contribution: `1.0 - busy_fraction`.
    pub fn bubble_fraction(&self) -> f64 {
        1.0 - self.busy_fraction()
    }
}

#[derive(Debug, Clone, Default)]
struct StageUtilizationAccumulator {
    busy_us: u64,
    total_us: u64,
}

impl StageUtilizationAccumulator {
    fn add(&mut self, busy: Duration, total: Duration) {
        let b = busy.as_micros().min(u128::from(u64::MAX)) as u64;
        let t = total.as_micros().min(u128::from(u64::MAX)) as u64;
        self.busy_us = self.busy_us.saturating_add(b);
        self.total_us = self.total_us.saturating_add(t);
    }
}

// ---------------------------------------------------------------------------
// Pipeline observability aggregator
// ---------------------------------------------------------------------------

/// Top-level container for every pipeline-parallel observability surface.
///
/// Held by the server's `AppState` so route handlers render metrics without
/// knowing about the individual trackers.
#[derive(Debug, Default)]
pub struct PipelineObservability {
    /// Per-stage utilization aggregator.
    pub stage_utilization: StageUtilizationRegistry,
    /// Per-stage-pair activation transfer latency histogram.
    pub activation_latency: ActivationLatencyHistogram,
    /// KV cache admission rejection counters.
    pub admission_rejections: AdmissionRejectionCounters,
    /// Elastic repartition event counters (emission path).
    pub repartition: RepartitionMetrics,
    /// Rolling bubble ratio sample (best-effort; scheduler updates this when
    /// it closes out a pipeline step).
    bubble_ratio_bp: AtomicU64,
    bubble_ratio_samples: AtomicU64,
}

impl PipelineObservability {
    /// Construct an empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a bubble-ratio observation (fraction in `[0.0, 1.0]`). We
    /// store as basis points (×10_000) to keep arithmetic integer-atomic.
    pub fn record_bubble_ratio(&self, fraction: f64) {
        let bp = (fraction.clamp(0.0, 1.0) * 10_000.0).round() as u64;
        self.bubble_ratio_bp.fetch_add(bp, Ordering::Relaxed);
        self.bubble_ratio_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Mean bubble ratio across recorded samples. Zero when no samples.
    pub fn mean_bubble_ratio(&self) -> f64 {
        let samples = self.bubble_ratio_samples.load(Ordering::Relaxed);
        if samples == 0 {
            return 0.0;
        }
        let total = self.bubble_ratio_bp.load(Ordering::Relaxed);
        (total as f64 / samples as f64) / 10_000.0
    }

    /// Render every sub-metric into a single snapshot struct suitable for
    /// the Prometheus endpoint.
    pub fn snapshot(&self) -> PipelineObservabilitySnapshot {
        PipelineObservabilitySnapshot {
            stage_utilization: self.stage_utilization.snapshot(),
            activation_latency: self.activation_latency.snapshot(),
            admission_rejections: self.admission_rejections.snapshot(),
            repartition: self.repartition.snapshot(),
            mean_bubble_ratio: self.mean_bubble_ratio(),
        }
    }
}

/// Rendered snapshot returned by [`PipelineObservability::snapshot`].
#[derive(Debug, Clone, Default)]
pub struct PipelineObservabilitySnapshot {
    pub stage_utilization: Vec<StageUtilizationSnapshot>,
    pub activation_latency: Vec<ActivationLatencyPair>,
    pub admission_rejections: Vec<AdmissionRejectionEntry>,
    pub repartition: RepartitionMetricsSnapshot,
    pub mean_bubble_ratio: f64,
}

// ---------------------------------------------------------------------------
// Elastic repartition counters (consumed by the /metrics endpoint)
// ---------------------------------------------------------------------------

/// Atomic counters + histogram totals for elastic repartition events.
///
/// This is the low-level container the coordinator writes to; the
/// Prometheus endpoint reads a [`RepartitionMetricsSnapshot`] to render
/// stable text output. Used by: elastic coordinator (writer),
/// Prometheus endpoint (reader).
#[derive(Debug, Default)]
pub struct RepartitionMetrics {
    /// Total repartition events, partitioned by trigger kind + outcome.
    ///
    /// Key format: `(trigger_kind, outcome_label)`.
    ///
    /// `trigger_kind` is one of: "explicit", "memory_pressure".
    /// `outcome_label` is one of: "completed", "aborted", "failed", "progress"
    /// (the latter is emitted for intermediate state transitions).
    counters: Mutex<HashMap<(&'static str, &'static str), u64>>,
    /// Sum of drain durations observed on `completed` outcomes, in
    /// microseconds.
    drain_us_total: AtomicU64,
    /// Number of completed drains (the histogram count).
    drain_count: AtomicU64,
    /// Sum of total durations observed across *all* terminal events, in
    /// microseconds. Useful for computing average repartition wall time.
    total_us_total: AtomicU64,
    /// Number of terminal events observed across all outcomes.
    total_count: AtomicU64,
    /// Maximum drain duration observed, microseconds. Simple O(1) proxy for
    /// a p99 until the full histogram lands.
    drain_us_max: AtomicU64,
}

impl RepartitionMetrics {
    /// Create a fresh metrics container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot for the Prometheus endpoint.
    pub fn snapshot(&self) -> RepartitionMetricsSnapshot {
        let counters: HashMap<String, u64> = self
            .counters
            .lock()
            .expect("repartition metrics poisoned")
            .iter()
            .map(|((k, v), count)| (format!("{k}:{v}"), *count))
            .collect();
        RepartitionMetricsSnapshot {
            counters,
            drain_us_total: self.drain_us_total.load(Ordering::Relaxed),
            drain_count: self.drain_count.load(Ordering::Relaxed),
            total_us_total: self.total_us_total.load(Ordering::Relaxed),
            total_count: self.total_count.load(Ordering::Relaxed),
            drain_us_max: self.drain_us_max.load(Ordering::Relaxed),
        }
    }

    fn bump(&self, key: (&'static str, &'static str)) {
        let mut map = self.counters.lock().expect("repartition metrics poisoned");
        *map.entry(key).or_insert(0) += 1;
    }

    fn record_drain(&self, drain_us: u64) {
        self.drain_us_total.fetch_add(drain_us, Ordering::Relaxed);
        self.drain_count.fetch_add(1, Ordering::Relaxed);
        let mut current = self.drain_us_max.load(Ordering::Relaxed);
        while drain_us > current {
            match self.drain_us_max.compare_exchange_weak(
                current,
                drain_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

impl RepartitionEventSink for RepartitionMetrics {
    fn record_event(&self, event: &RepartitionEvent) {
        let trigger_kind = event.trigger.kind_label();
        let outcome_label = match event.outcome {
            Some(RepartitionOutcome::Completed) => "completed",
            Some(RepartitionOutcome::Aborted) => "aborted",
            Some(RepartitionOutcome::Failed) => "failed",
            None => "progress",
        };
        self.bump((trigger_kind, outcome_label));
        if event.outcome.is_some() {
            self.total_us_total.fetch_add(
                event.total_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            self.total_count.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(event.outcome, Some(RepartitionOutcome::Completed)) {
            self.record_drain(event.drain_duration.as_micros().min(u128::from(u64::MAX)) as u64);
        }
    }
}

/// Snapshot returned by [`RepartitionMetrics::snapshot`] for rendering.
#[derive(Debug, Clone, Default)]
pub struct RepartitionMetricsSnapshot {
    /// Counter values keyed by "trigger_kind:outcome".
    pub counters: HashMap<String, u64>,
    /// Sum of drain durations across completed events, microseconds.
    pub drain_us_total: u64,
    /// Number of completed drain observations.
    pub drain_count: u64,
    /// Sum of total repartition durations across terminal events, microseconds.
    pub total_us_total: u64,
    /// Number of terminal events observed.
    pub total_count: u64,
    /// Largest drain duration observed so far, microseconds.
    pub drain_us_max: u64,
}

impl RepartitionMetricsSnapshot {
    /// Total events counted, regardless of outcome.
    pub fn total_events(&self) -> u64 {
        self.counters.values().copied().sum()
    }

    /// Mean drain duration (microseconds) across completed events, or zero
    /// when no completed drain has been observed yet.
    pub fn mean_drain_us(&self) -> u64 {
        self.drain_us_total
            .checked_div(self.drain_count)
            .unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
