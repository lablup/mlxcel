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

use std::time::Duration;

use super::*;

#[test]
fn stage_utilization_all_compute() {
    let mut s = StageMetrics::new(0);
    s.compute_time = Duration::from_millis(100);
    assert!((s.utilization() - 1.0).abs() < 1e-9);
}

#[test]
fn stage_utilization_half_compute() {
    let mut s = StageMetrics::new(0);
    s.compute_time = Duration::from_millis(50);
    s.wait_time = Duration::from_millis(50);
    assert!((s.utilization() - 0.5).abs() < 1e-9);
}

#[test]
fn stage_utilization_zero_time() {
    let s = StageMetrics::new(0);
    assert!((s.utilization()).abs() < 1e-9);
}

#[test]
fn bubble_ratio_perfect_pipeline() {
    // All stages compute for the full wall time -> bubble = 0.
    let mut m = PipelineMetrics::new(2, 4);
    m.wall_time = Duration::from_millis(100);
    m.stages.get_mut(&0).unwrap().compute_time = Duration::from_millis(100);
    m.stages.get_mut(&1).unwrap().compute_time = Duration::from_millis(100);
    assert!((m.bubble_ratio()).abs() < 1e-9);
}

#[test]
fn bubble_ratio_serial_pipeline() {
    // 2 stages, only one computes at a time -> bubble = 0.5.
    let mut m = PipelineMetrics::new(2, 1);
    m.wall_time = Duration::from_millis(100);
    m.stages.get_mut(&0).unwrap().compute_time = Duration::from_millis(50);
    m.stages.get_mut(&1).unwrap().compute_time = Duration::from_millis(50);
    assert!((m.bubble_ratio() - 0.5).abs() < 1e-9);
}

#[test]
fn bubble_ratio_zero_wall_time() {
    let m = PipelineMetrics::new(2, 4);
    assert!((m.bubble_ratio()).abs() < 1e-9);
}

#[test]
fn theoretical_bubble_ratio_gpipe() {
    // S=4, M=4: theoretical = 3 / (3 + 4) = 3/7 ~ 0.4286
    let m = PipelineMetrics::new(4, 4);
    let expected = 3.0 / 7.0;
    assert!((m.theoretical_bubble_ratio() - expected).abs() < 1e-6);
}

#[test]
fn theoretical_bubble_ratio_many_micro_batches() {
    // S=2, M=8: theoretical = 1 / (1 + 8) = 1/9 ~ 0.111
    let m = PipelineMetrics::new(2, 8);
    let expected = 1.0 / 9.0;
    assert!((m.theoretical_bubble_ratio() - expected).abs() < 1e-6);
}

#[test]
fn theoretical_bubble_ratio_single_stage() {
    let m = PipelineMetrics::new(1, 4);
    assert!((m.theoretical_bubble_ratio()).abs() < 1e-9);
}

#[test]
fn record_operations() {
    let mut m = PipelineMetrics::new(2, 2);
    m.record_compute(0, Duration::from_millis(10));
    m.record_wait(0, Duration::from_millis(5));
    m.record_transfer(0, Duration::from_millis(2));
    m.record_micro_batch_processed(0);

    let s0 = m.stages.get(&0).unwrap();
    assert_eq!(s0.compute_time, Duration::from_millis(10));
    assert_eq!(s0.wait_time, Duration::from_millis(5));
    assert_eq!(s0.transfer_time, Duration::from_millis(2));
    assert_eq!(s0.micro_batches_processed, 1);
}

#[test]
fn metrics_collector_accumulation() {
    let mut collector = MetricsCollector::new(2);

    // Step 1.
    collector.begin_step();
    let mut m1 = PipelineMetrics::new(2, 4);
    m1.sequences_completed = 2;
    collector.end_step(m1);

    // Step 2.
    collector.begin_step();
    let mut m2 = PipelineMetrics::new(2, 4);
    m2.sequences_completed = 3;
    collector.end_step(m2);

    assert_eq!(collector.num_steps(), 2);
    assert_eq!(collector.total_sequences_completed(), 5);

    let summary = collector.summary();
    assert_eq!(summary.num_stages, 2);
    assert_eq!(summary.num_steps, 2);
    assert_eq!(summary.total_sequences_completed, 5);
    assert_eq!(summary.total_micro_batches, 8);
}

#[test]
fn pipeline_metrics_display() {
    let mut m = PipelineMetrics::new(2, 2);
    m.wall_time = Duration::from_millis(50);
    m.record_compute(0, Duration::from_millis(20));
    m.record_compute(1, Duration::from_millis(15));
    let display = format!("{m}");
    assert!(display.contains("Pipeline"));
    assert!(display.contains("Stage 0"));
    assert!(display.contains("Stage 1"));
}

#[test]
fn metrics_summary_display() {
    let collector = MetricsCollector::new(4);
    let summary = collector.summary();
    let display = format!("{summary}");
    assert!(display.contains("stages=4"));
}

// -----------------------------------------------------------------------------
// Elastic repartition counters (emission path read path).
// -----------------------------------------------------------------------------

mod repartition {
    use super::super::RepartitionMetrics;
    use crate::distributed::pipeline::elastic::{
        RepartitionEvent, RepartitionEventSink, RepartitionOutcome, RepartitionState,
        RepartitionTrigger,
    };
    use std::time::Duration;

    fn completed_event(trigger: RepartitionTrigger) -> RepartitionEvent {
        RepartitionEvent {
            trigger,
            to_state: RepartitionState::Idle,
            drain_duration: Duration::from_millis(200),
            total_duration: Duration::from_millis(500),
            outcome: Some(RepartitionOutcome::Completed),
            ranges_before: vec![0..8, 8..16],
            ranges_after: vec![0..10, 10..16],
        }
    }

    fn failed_event(trigger: RepartitionTrigger) -> RepartitionEvent {
        RepartitionEvent {
            trigger,
            to_state: RepartitionState::Failed,
            drain_duration: Duration::from_millis(50),
            total_duration: Duration::from_millis(120),
            outcome: Some(RepartitionOutcome::Failed),
            ranges_before: vec![0..8, 8..16],
            ranges_after: Vec::new(),
        }
    }

    fn progress_event(trigger: RepartitionTrigger) -> RepartitionEvent {
        RepartitionEvent {
            trigger,
            to_state: RepartitionState::Draining,
            drain_duration: Duration::ZERO,
            total_duration: Duration::from_millis(10),
            outcome: None,
            ranges_before: vec![0..8, 8..16],
            ranges_after: Vec::new(),
        }
    }

    #[test]
    fn sink_counts_events_by_trigger_and_outcome() {
        let metrics = RepartitionMetrics::new();

        metrics.record_event(&completed_event(RepartitionTrigger::Explicit));
        metrics.record_event(&failed_event(RepartitionTrigger::MemoryPressure {
            stage_index: 1,
            fraction: 0.97,
        }));
        metrics.record_event(&progress_event(RepartitionTrigger::Explicit));

        let snap = metrics.snapshot();
        assert_eq!(snap.counters.get("explicit:completed"), Some(&1));
        assert_eq!(snap.counters.get("memory_pressure:failed"), Some(&1));
        assert_eq!(snap.counters.get("explicit:progress"), Some(&1));
        assert_eq!(snap.total_events(), 3);
    }

    #[test]
    fn completed_events_drive_drain_histogram_totals() {
        let metrics = RepartitionMetrics::new();
        metrics.record_event(&completed_event(RepartitionTrigger::Explicit));
        metrics.record_event(&completed_event(RepartitionTrigger::Explicit));
        let snap = metrics.snapshot();
        assert_eq!(snap.drain_count, 2);
        // Two 200ms drains -> 400_000us total, mean = 200_000us.
        assert_eq!(snap.drain_us_total, 400_000);
        assert_eq!(snap.mean_drain_us(), 200_000);
        assert_eq!(snap.drain_us_max, 200_000);
    }

    #[test]
    fn failed_events_are_counted_but_not_in_drain_histogram() {
        let metrics = RepartitionMetrics::new();
        metrics.record_event(&failed_event(RepartitionTrigger::Explicit));
        let snap = metrics.snapshot();
        assert_eq!(
            snap.drain_count, 0,
            "failed events must not pollute drain p50"
        );
        // Failed events still count toward total repartition observations.
        assert_eq!(snap.total_count, 1);
        assert!(snap.total_us_total >= 120_000);
    }

    #[test]
    fn progress_events_do_not_affect_terminal_totals() {
        let metrics = RepartitionMetrics::new();
        metrics.record_event(&progress_event(RepartitionTrigger::Explicit));
        let snap = metrics.snapshot();
        assert_eq!(snap.total_count, 0, "progress events are not terminal");
        assert_eq!(snap.drain_count, 0);
        assert_eq!(snap.mean_drain_us(), 0);
    }
}

// -----------------------------------------------------------------------------
// activation latency histogram, admission rejection counters,
// stage utilization registry, and the top-level PipelineObservability
// aggregator.
// -----------------------------------------------------------------------------

mod observability {
    use super::super::{
        ActivationLatencyHistogram, AdmissionRejectionCounters, PipelineObservability,
        StageUtilizationRegistry,
    };
    use std::time::Duration;

    #[test]
    fn latency_histogram_reports_stable_ordering() {
        let hist = ActivationLatencyHistogram::new();
        // Record a mix of pairs.
        for _ in 0..10 {
            hist.observe(0, 1, Duration::from_micros(50));
        }
        for _ in 0..10 {
            hist.observe(1, 2, Duration::from_micros(200));
        }
        let snap = hist.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!((snap[0].src_stage, snap[0].dst_stage), (0, 1));
        assert_eq!((snap[1].src_stage, snap[1].dst_stage), (1, 2));
        // Every sample lived in the "0..=50us" bucket -> p95 for stage 0->1
        // should be bounded by the 2^N boundary that covers 50us (64us).
        assert!(snap[0].p95_us <= 64, "p95={}", snap[0].p95_us);
        // 200us samples should land at or above 256us boundary.
        assert!(snap[1].p95_us >= 128, "p95={}", snap[1].p95_us);
        assert_eq!(snap[0].count, 10);
        assert_eq!(snap[1].count, 10);
    }

    #[test]
    fn latency_histogram_reports_max_regardless_of_bucket_ceiling() {
        let hist = ActivationLatencyHistogram::new();
        hist.observe(0, 1, Duration::from_millis(250));
        let snap = hist.snapshot();
        assert_eq!(snap.len(), 1);
        // 250 ms = 250_000 us -> recorded verbatim in `max_us` even if
        // the bucket layout saturates.
        assert_eq!(snap[0].max_us, 250_000);
    }

    #[test]
    fn admission_rejection_counters_aggregate_by_stage_and_reason() {
        let counters = AdmissionRejectionCounters::new();
        counters.record(0, "memory");
        counters.record(0, "memory");
        counters.record(1, "sequence_cap");
        let snap = counters.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(
            (snap[0].stage_index, snap[0].reason, snap[0].count),
            (0, "memory", 2)
        );
        assert_eq!(
            (snap[1].stage_index, snap[1].reason, snap[1].count),
            (1, "sequence_cap", 1)
        );
    }

    #[test]
    fn stage_utilization_busy_fraction_rolls_up_samples() {
        let reg = StageUtilizationRegistry::new();
        reg.record(0, Duration::from_millis(80), Duration::from_millis(100));
        reg.record(0, Duration::from_millis(100), Duration::from_millis(100));
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].stage_index, 0);
        // Combined busy 180 / 200 = 0.9.
        assert!((snap[0].busy_fraction() - 0.9).abs() < 1e-6);
        assert!((snap[0].bubble_fraction() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn observability_snapshot_bundles_every_family() {
        let obs = PipelineObservability::new();
        obs.stage_utilization
            .record(0, Duration::from_millis(10), Duration::from_millis(20));
        obs.activation_latency
            .observe(0, 1, Duration::from_micros(100));
        obs.admission_rejections.record(0, "memory");
        obs.record_bubble_ratio(0.4);
        obs.record_bubble_ratio(0.6);
        let snap = obs.snapshot();
        assert_eq!(snap.stage_utilization.len(), 1);
        assert_eq!(snap.activation_latency.len(), 1);
        assert_eq!(snap.admission_rejections.len(), 1);
        assert!(
            (snap.mean_bubble_ratio - 0.5).abs() < 1e-6,
            "mean = {}",
            snap.mean_bubble_ratio
        );
    }
}
