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

//! End-to-end tests for pipeline parallelism.
//!
//! Exercises the full pipeline stack -- schedule, coordinator, metrics,
//! micro-batching, and chunked prefill -- in simulated multi-stage
//! configurations. All tests run in-process, no real hardware needed.

use std::time::Duration;

use mlxcel::distributed::pipeline::benchmark::{
    PipelineBenchmarkConfig, format_benchmark_report, run_pipeline_benchmark, run_scaling_benchmark,
};
use mlxcel::distributed::pipeline::cache_manager::PipelineCacheConfig;
use mlxcel::distributed::pipeline::metrics::{MetricsCollector, PipelineMetrics};
use mlxcel::distributed::pipeline::micro_batch::{
    split_into_micro_batches, suggested_micro_batch_size,
};
use mlxcel::distributed::pipeline::schedule::{
    GPipeSchedule, PipelineConfig, PipelineSchedule, ScheduleAction,
};
use mlxcel::distributed::pipeline::serving::{
    ChunkedPrefillPipeline, PipelineCoordinator, PipelineRequest, PipelineServingConfig,
    StageHealth, StageRole,
};
use mlxcel::distributed::request_tracker::RequestId;

// ---------------------------------------------------------------------------
// Helper: drive a schedule to completion and collect metrics
// ---------------------------------------------------------------------------

/// Drive a GPipe schedule to completion, returning step metrics.
fn drive_schedule_to_completion(schedule: &mut GPipeSchedule) -> (u32, u32) {
    let mut forward_count = 0u32;
    let mut flush_count = 0u32;
    let max_iterations = 10_000;

    for _ in 0..max_iterations {
        let action = schedule.next_action();
        match action {
            ScheduleAction::Forward {
                stage_index,
                micro_batch_id,
            } => {
                forward_count += 1;
                schedule.notify_forward_complete(stage_index, micro_batch_id);
            }
            ScheduleAction::Receive { .. } => {
                // Transfer received; no notification needed.
            }
            ScheduleAction::Flush { micro_batch_id } => {
                flush_count += 1;
                schedule.notify_flush_complete(micro_batch_id);
            }
            ScheduleAction::Idle => {
                if schedule.is_complete() {
                    break;
                }
            }
            ScheduleAction::Done => break,
            _ => {}
        }
    }

    (forward_count, flush_count)
}

// ===========================================================================
// 2-stage correctness tests
// ===========================================================================

#[test]
fn e2e_two_stage_single_micro_batch() {
    let config = PipelineConfig::new(2, 1).unwrap();
    let mut schedule = GPipeSchedule::new(config, 1).unwrap();

    let (forwards, flushes) = drive_schedule_to_completion(&mut schedule);

    assert!(schedule.is_complete());
    // 1 micro-batch * 2 stages = 2 forwards.
    assert_eq!(forwards, 2);
    assert_eq!(flushes, 1);
}

#[test]
fn e2e_two_stage_four_micro_batches() {
    let config = PipelineConfig::new(2, 1).unwrap();
    let mut schedule = GPipeSchedule::new(config, 4).unwrap();

    let (forwards, flushes) = drive_schedule_to_completion(&mut schedule);

    assert!(schedule.is_complete());
    // 4 micro-batches * 2 stages = 8 forwards.
    assert_eq!(forwards, 8);
    assert_eq!(flushes, 4);
}

#[test]
fn e2e_two_stage_coordinator_full_flow() {
    let serving_config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(serving_config).unwrap();

    // Submit 4 requests.
    let mut receivers = Vec::new();
    let mut request_ids = Vec::new();
    for _ in 0..4 {
        let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
        request_ids.push(req.request_id.clone());
        receivers.push(coord.submit_request(req).unwrap());
    }

    assert_eq!(coord.in_flight_count(), 4);

    // Simulate each request going through stage 0, then stage 1 with token.
    for (i, rid) in request_ids.iter().enumerate() {
        coord.process_stage_output(rid, 0, None, false).unwrap();
        coord
            .process_stage_output(rid, 1, Some(100 + i as u32), true)
            .unwrap();
    }

    // All requests should be delivered.
    assert_eq!(coord.in_flight_count(), 0);
    for (i, mut rx) in receivers.into_iter().enumerate() {
        let resp = rx.try_recv().unwrap();
        assert!(!resp.is_error());
        assert!(resp.is_finished);
        assert_eq!(resp.generated_tokens, vec![100 + i as u32]);
    }
}

#[test]
fn e2e_two_stage_multi_token_decode() {
    let serving_config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(serving_config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let rid = req.request_id.clone();
    let mut rx = coord.submit_request(req).unwrap();

    // Simulate 5 decode steps.
    for step in 0..5u32 {
        coord.process_stage_output(&rid, 0, None, false).unwrap();
        let is_last = step == 4;
        coord
            .process_stage_output(&rid, 1, Some(10 + step), is_last)
            .unwrap();
    }

    let resp = rx.try_recv().unwrap();
    assert_eq!(resp.generated_tokens, vec![10, 11, 12, 13, 14]);
    assert!(resp.is_finished);
}

// ===========================================================================
// 4-stage correctness tests
// ===========================================================================

#[test]
fn e2e_four_stage_single_micro_batch() {
    let config = PipelineConfig::new(4, 1).unwrap();
    let mut schedule = GPipeSchedule::new(config, 1).unwrap();

    let (forwards, flushes) = drive_schedule_to_completion(&mut schedule);

    assert!(schedule.is_complete());
    // 1 micro-batch * 4 stages = 4 forwards.
    assert_eq!(forwards, 4);
    assert_eq!(flushes, 1);
}

#[test]
fn e2e_four_stage_eight_micro_batches() {
    let config = PipelineConfig::new(4, 1).unwrap();
    let mut schedule = GPipeSchedule::new(config, 8).unwrap();

    let (forwards, flushes) = drive_schedule_to_completion(&mut schedule);

    assert!(schedule.is_complete());
    assert_eq!(forwards, 32); // 8 * 4
    assert_eq!(flushes, 8);
}

#[test]
fn e2e_four_stage_coordinator_flow() {
    let serving_config = PipelineServingConfig::new(4, 0).unwrap();
    let mut coord = PipelineCoordinator::new(serving_config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3, 4, 5], 20);
    let rid = req.request_id.clone();
    let mut rx = coord.submit_request(req).unwrap();

    // Process through 4 stages, 3 decode steps.
    for step in 0..3u32 {
        coord.process_stage_output(&rid, 0, None, false).unwrap();
        coord.process_stage_output(&rid, 1, None, false).unwrap();
        coord.process_stage_output(&rid, 2, None, false).unwrap();
        let is_last = step == 2;
        coord
            .process_stage_output(&rid, 3, Some(50 + step), is_last)
            .unwrap();
    }

    let resp = rx.try_recv().unwrap();
    assert!(!resp.is_error());
    assert!(resp.is_finished);
    assert_eq!(resp.generated_tokens, vec![50, 51, 52]);
}

#[test]
fn e2e_four_stage_stage_roles_correct() {
    assert_eq!(StageRole::from_index(0, 4), StageRole::First);
    assert_eq!(StageRole::from_index(1, 4), StageRole::Middle);
    assert_eq!(StageRole::from_index(2, 4), StageRole::Middle);
    assert_eq!(StageRole::from_index(3, 4), StageRole::Last);

    assert!(StageRole::First.is_entry_point());
    assert!(!StageRole::Middle.is_entry_point());
    assert!(StageRole::Last.produces_tokens());
    assert!(!StageRole::Middle.produces_tokens());
}

// ===========================================================================
// Determinism tests
// ===========================================================================

#[test]
fn e2e_determinism_schedule_actions_repeatable() {
    // Run the same schedule twice and verify identical action sequences.
    fn collect_actions(num_stages: u32, num_mbs: u32) -> Vec<ScheduleAction> {
        let config = PipelineConfig::new(num_stages, 1).unwrap();
        let mut schedule = GPipeSchedule::new(config, num_mbs).unwrap();
        let mut actions = Vec::new();

        for _ in 0..10_000 {
            let action = schedule.next_action();
            match &action {
                ScheduleAction::Forward {
                    stage_index,
                    micro_batch_id,
                } => {
                    let s = *stage_index;
                    let m = *micro_batch_id;
                    actions.push(action);
                    schedule.notify_forward_complete(s, m);
                }
                ScheduleAction::Receive { .. } => {
                    actions.push(action);
                }
                ScheduleAction::Flush { micro_batch_id } => {
                    let m = *micro_batch_id;
                    actions.push(action);
                    schedule.notify_flush_complete(m);
                }
                ScheduleAction::Idle => {
                    if schedule.is_complete() {
                        break;
                    }
                }
                ScheduleAction::Done => {
                    actions.push(action);
                    break;
                }
                _ => {}
            }
        }
        actions
    }

    let run1 = collect_actions(3, 4);
    let run2 = collect_actions(3, 4);

    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn e2e_determinism_coordinator_token_order() {
    // Verify token accumulation order is deterministic.
    fn run_coordinator() -> Vec<u32> {
        let config = PipelineServingConfig::new(2, 0).unwrap();
        let mut coord = PipelineCoordinator::new(config).unwrap();

        let req = PipelineRequest::new(
            RequestId::from_string("determinism-test".to_string()).unwrap(),
            0,
            vec![1, 2, 3],
            10,
        );
        let rid = req.request_id.clone();
        let mut rx = coord.submit_request(req).unwrap();

        for token in [10, 20, 30, 40, 50] {
            coord.process_stage_output(&rid, 0, None, false).unwrap();
            let is_last = token == 50;
            coord
                .process_stage_output(&rid, 1, Some(token), is_last)
                .unwrap();
        }

        rx.try_recv().unwrap().generated_tokens
    }

    let tokens1 = run_coordinator();
    let tokens2 = run_coordinator();
    assert_eq!(tokens1, tokens2);
    assert_eq!(tokens1, vec![10, 20, 30, 40, 50]);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn e2e_edge_single_token_generation() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1], 1);
    let rid = req.request_id.clone();
    let mut rx = coord.submit_request(req).unwrap();

    coord.process_stage_output(&rid, 0, None, false).unwrap();
    coord.process_stage_output(&rid, 1, Some(99), true).unwrap();

    let resp = rx.try_recv().unwrap();
    assert_eq!(resp.generated_tokens, vec![99]);
    assert!(resp.is_finished);
}

#[test]
fn e2e_edge_maximum_in_flight() {
    let config = PipelineServingConfig::new(2, 0)
        .unwrap()
        .with_max_in_flight(3);
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let mut receivers = Vec::new();
    for _ in 0..3 {
        let req = PipelineRequest::new(RequestId::new(), 0, vec![1], 10);
        receivers.push(coord.submit_request(req).unwrap());
    }

    // 4th request should be rejected.
    let overflow = coord.submit_request(PipelineRequest::new(RequestId::new(), 0, vec![1], 10));
    assert!(overflow.is_err());
    assert!(!coord.can_accept());
}

#[test]
fn e2e_edge_empty_token_ids() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![], 10);
    let rid = req.request_id.clone();
    assert!(req.is_prefill_complete()); // No tokens to prefill.

    let mut rx = coord.submit_request(req).unwrap();

    // Directly go to decode.
    coord.process_stage_output(&rid, 1, Some(42), true).unwrap();

    let resp = rx.try_recv().unwrap();
    assert_eq!(resp.generated_tokens, vec![42]);
}

#[test]
fn e2e_edge_stage_failure_during_processing() {
    let config = PipelineServingConfig::new(4, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let _rid = req.request_id.clone();
    let mut rx = coord.submit_request(req).unwrap();

    // Fail stage 0 before any processing.
    let failed = coord.handle_stage_failure(0);
    assert_eq!(failed.len(), 1);

    let resp = rx.try_recv().unwrap();
    assert!(resp.is_error());
    assert_eq!(coord.stage_health(0), StageHealth::Failed);
}

#[test]
fn e2e_edge_timeout_enforcement() {
    let config = PipelineServingConfig::new(2, 0)
        .unwrap()
        .with_timeout(Duration::from_millis(1));
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1], 10);
    let mut rx = coord.submit_request(req).unwrap();

    // Wait for timeout.
    std::thread::sleep(Duration::from_millis(5));

    let timed_out = coord.enforce_timeouts();
    assert_eq!(timed_out.len(), 1);

    let resp = rx.try_recv().unwrap();
    assert!(resp.is_error());
    assert_eq!(coord.in_flight_count(), 0);
}

#[test]
fn e2e_edge_chunked_prefill_long_prompt() {
    let mut prefill = ChunkedPrefillPipeline::new(4);
    let long_prompt: Vec<u32> = (0..100).collect();
    let req = PipelineRequest::new(RequestId::new(), 1, long_prompt, 50);

    let (start, end) = prefill.begin_prefill(&req);
    assert_eq!((start, end), (0, 4));

    // Process all chunks.
    let mut chunks_processed = 1;
    let mut current_start = start;
    let mut current_end = end;

    loop {
        let tokens_in_chunk = current_end - current_start;
        match prefill.advance_prefill(&req.request_id, tokens_in_chunk) {
            Some((s, e)) => {
                current_start = s;
                current_end = e;
                chunks_processed += 1;
            }
            None => break,
        }
    }

    assert_eq!(chunks_processed, 25); // 100 / 4 = 25 chunks
    assert!(!prefill.is_prefilling(&req.request_id));
}

// ===========================================================================
// Micro-batch splitting edge cases
// ===========================================================================

#[test]
fn e2e_micro_batch_split_exact() {
    let specs = split_into_micro_batches(8, 2).unwrap();
    assert_eq!(specs.len(), 4);
    for (i, spec) in specs.iter().enumerate() {
        assert_eq!(spec.id, i as u32);
        assert_eq!(spec.size, 2);
    }
}

#[test]
fn e2e_micro_batch_split_remainder() {
    let specs = split_into_micro_batches(7, 3).unwrap();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].size, 3);
    assert_eq!(specs[1].size, 3);
    assert_eq!(specs[2].size, 1); // Remainder.
}

#[test]
fn e2e_micro_batch_suggested_size() {
    // For 16 sequences and 4 stages, target = 4*2 = 8 micro-batches.
    // Suggested size = 16/8 = 2.
    let suggested = suggested_micro_batch_size(16, 4);
    assert_eq!(suggested, 2);
}

// ===========================================================================
// Pipeline utilization measurement
// ===========================================================================

#[test]
fn e2e_utilization_metrics_collection() {
    let mut collector = MetricsCollector::new(2);

    for step in 0..10 {
        collector.begin_step();

        let mut metrics = PipelineMetrics::new(2, 4);
        // Simulate 80% utilization: 80ms compute out of 100ms total.
        metrics.record_compute(0, Duration::from_millis(80));
        metrics.record_wait(0, Duration::from_millis(20));
        metrics.record_compute(1, Duration::from_millis(80));
        metrics.record_wait(1, Duration::from_millis(20));
        metrics.wall_time = Duration::from_millis(100);
        metrics.sequences_completed = if step == 9 { 4 } else { 0 };

        collector.end_step(metrics);
    }

    let summary = collector.summary();
    assert_eq!(summary.num_steps, 10);
    assert_eq!(summary.total_sequences_completed, 4);

    // Utilization should be around 80%.
    assert!(
        summary.average_utilization > 0.7,
        "utilization {:.2} should be > 0.7",
        summary.average_utilization
    );
}

#[test]
fn e2e_utilization_bubble_ratio_decreases_with_more_micro_batches() {
    // Theoretical: bubble = (S-1) / (S-1 + M).
    // 2-stage, 2 mb: 1/3 = 0.333
    // 2-stage, 8 mb: 1/9 = 0.111

    let metrics_few = PipelineMetrics::new(2, 2);
    let metrics_many = PipelineMetrics::new(2, 8);

    assert!(metrics_few.theoretical_bubble_ratio() > metrics_many.theoretical_bubble_ratio());
}

#[test]
fn e2e_utilization_four_plus_micro_batches_good_utilization() {
    // GPipe with 2 stages, 4+ micro-batches should have theoretical bubble < 25%.
    let metrics = PipelineMetrics::new(2, 4);
    let theoretical = metrics.theoretical_bubble_ratio();
    // (2-1) / (2-1+4) = 1/5 = 0.2
    assert!(
        theoretical < 0.25,
        "theoretical bubble {theoretical} should be < 0.25 with 4+ micro-batches"
    );

    let metrics_8 = PipelineMetrics::new(2, 8);
    let theoretical_8 = metrics_8.theoretical_bubble_ratio();
    // (2-1) / (2-1+8) = 1/9 = 0.111
    assert!(
        theoretical_8 < 0.15,
        "theoretical bubble {theoretical_8} should be < 0.15 with 8 micro-batches"
    );
}

// ===========================================================================
// Performance benchmarks (simulated)
// ===========================================================================

#[test]
fn e2e_benchmark_two_stage_throughput() {
    let config = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(Duration::from_micros(200))
        .with_transfer_time(Duration::from_micros(20))
        .with_decode_steps(16);

    let result = run_pipeline_benchmark(&config).unwrap();

    assert!(
        result.throughput_tok_per_sec > 0.0,
        "throughput should be positive"
    );
    assert!(!result.ttft.is_zero(), "TTFT should be non-zero");
    assert!(!result.tpot.is_zero(), "TPOT should be non-zero");
    assert_eq!(result.total_tokens, 4 * 16);
}

#[test]
fn e2e_benchmark_four_stage_throughput() {
    let config = PipelineBenchmarkConfig::new(4, 8)
        .with_compute_time(Duration::from_micros(200))
        .with_transfer_time(Duration::from_micros(20))
        .with_decode_steps(16);

    let result = run_pipeline_benchmark(&config).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert_eq!(result.total_tokens, 8 * 16);

    // Bubble ratio should be reasonable.
    assert!(
        result.bubble_ratio < 1.0,
        "bubble ratio should be < 1.0, got {}",
        result.bubble_ratio
    );
}

#[test]
fn e2e_benchmark_ttft_scales_with_stages() {
    // TTFT should increase with more stages (pipeline fill time).
    let config_2 = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(Duration::from_micros(500))
        .with_transfer_time(Duration::from_micros(50))
        .with_decode_steps(8);

    let config_4 = PipelineBenchmarkConfig::new(4, 4)
        .with_compute_time(Duration::from_micros(500))
        .with_transfer_time(Duration::from_micros(50))
        .with_decode_steps(8);

    let result_2 = run_pipeline_benchmark(&config_2).unwrap();
    let result_4 = run_pipeline_benchmark(&config_4).unwrap();

    assert!(
        result_4.ttft > result_2.ttft,
        "4-stage TTFT ({:?}) should be > 2-stage TTFT ({:?})",
        result_4.ttft,
        result_2.ttft,
    );
}

#[test]
fn e2e_benchmark_scaling_analysis() {
    let result = run_scaling_benchmark(
        &[2, 4],
        4,
        Duration::from_micros(200),
        Duration::from_micros(20),
        16,
    )
    .unwrap();

    assert_eq!(result.results.len(), 2);

    // Both configurations should produce positive throughput.
    for r in &result.results {
        assert!(r.throughput_tok_per_sec > 0.0);
    }

    // Scaling factor should be computable.
    let factor = result.scaling_factor(0, 1).unwrap();
    assert!(factor > 0.0);

    // Print the scaling report (for CI visibility).
    let report = format!("{result}");
    assert!(!report.is_empty());
}

#[test]
fn e2e_benchmark_report_generation() {
    let configs = [
        PipelineBenchmarkConfig::new(2, 2)
            .with_compute_time(Duration::from_micros(100))
            .with_decode_steps(8),
        PipelineBenchmarkConfig::new(2, 4)
            .with_compute_time(Duration::from_micros(100))
            .with_decode_steps(8),
        PipelineBenchmarkConfig::new(2, 8)
            .with_compute_time(Duration::from_micros(100))
            .with_decode_steps(8),
    ];

    let results: Vec<_> = configs
        .iter()
        .map(|c| run_pipeline_benchmark(c).unwrap())
        .collect();

    let report = format_benchmark_report(&results);

    // Report should contain all three configurations.
    assert!(report.contains("2 stages"));
    assert!(report.contains("Summary"));
}

// ===========================================================================
// Cache manager integration
// ===========================================================================

#[test]
fn e2e_cache_config_per_stage() {
    // Verify cache config works for each pipeline stage.
    let layers_per_stage = 8;
    for stage in 0..4u32 {
        let start = stage as usize * layers_per_stage;
        let end = start + layers_per_stage;
        let config = PipelineCacheConfig {
            stage_index: stage,
            num_stages: 4,
            layer_range: start..end,
            max_sequences: 64,
            memory_budget_bytes: 1024 * 1024 * 100,
            bytes_per_layer_per_token: 256,
            pressure_threshold: 0.9,
        };
        assert_eq!(config.stage_index, stage);
        assert_eq!(config.num_stages, 4);
        assert_eq!(config.num_layers(), layers_per_stage);
    }
}

// ===========================================================================
// CI compatibility
// ===========================================================================

#[test]
fn e2e_ci_no_external_dependencies() {
    // Verify all tests can run without network, GPU, or model files.
    // This test is a meta-check: if it compiles and runs, the test
    // suite has no hidden external dependencies.
    let config = PipelineBenchmarkConfig::default();
    let result = run_pipeline_benchmark(&config);
    assert!(result.is_ok());
}

#[test]
fn e2e_ci_benchmark_completes_quickly() {
    // Ensure benchmarks complete within a reasonable CI time budget.
    let start = std::time::Instant::now();

    let config = PipelineBenchmarkConfig::new(4, 8)
        .with_compute_time(Duration::from_micros(50))
        .with_transfer_time(Duration::from_micros(5))
        .with_decode_steps(16);

    let _ = run_pipeline_benchmark(&config).unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "benchmark took too long for CI: {elapsed:?}"
    );
}
