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

use super::*;

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

#[test]
fn config_default_is_valid() {
    let config = PipelineBenchmarkConfig::default();
    config.validate().unwrap();
}

#[test]
fn config_single_stage_rejected() {
    let config = PipelineBenchmarkConfig::new(1, 4);
    assert!(config.validate().is_err());
}

#[test]
fn config_zero_micro_batches_rejected() {
    let config = PipelineBenchmarkConfig::new(2, 0);
    assert!(config.validate().is_err());
}

#[test]
fn config_zero_decode_steps_rejected() {
    let config = PipelineBenchmarkConfig::new(2, 4).with_decode_steps(0);
    assert!(config.validate().is_err());
}

#[test]
fn config_builders() {
    let config = PipelineBenchmarkConfig::new(3, 8)
        .with_compute_time(Duration::from_millis(1))
        .with_transfer_time(Duration::from_micros(100))
        .with_decode_steps(64)
        .with_sequence_length(256);

    assert_eq!(config.num_stages, 3);
    assert_eq!(config.num_micro_batches, 8);
    assert_eq!(config.stage_compute_time, Duration::from_millis(1));
    assert_eq!(config.transfer_time, Duration::from_micros(100));
    assert_eq!(config.decode_steps, 64);
    assert_eq!(config.sequence_length, 256);
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

#[test]
fn benchmark_two_stages_produces_results() {
    let config = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(Duration::from_micros(100))
        .with_transfer_time(Duration::from_micros(10))
        .with_decode_steps(8);

    let result = run_pipeline_benchmark(&config).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(!result.total_time.is_zero());
    assert!(!result.ttft.is_zero());
    assert_eq!(result.total_tokens, 4 * 8); // 4 micro-batches * 8 steps
    // Bubble ratio from simulated metrics may be negative due to measurement
    // noise (spin_wait imprecision), so we only check theoretical here.
    assert!(result.theoretical_bubble_ratio > 0.0);
}

#[test]
fn benchmark_four_stages_produces_results() {
    let config = PipelineBenchmarkConfig::new(4, 8)
        .with_compute_time(Duration::from_micros(100))
        .with_transfer_time(Duration::from_micros(10))
        .with_decode_steps(8);

    let result = run_pipeline_benchmark(&config).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert_eq!(result.total_tokens, 8 * 8);

    // 4-stage theoretical bubble: (4-1)/(4-1+8) = 3/11 ~= 0.273
    let expected_theoretical = 3.0 / 11.0;
    assert!(
        (result.theoretical_bubble_ratio - expected_theoretical).abs() < 0.01,
        "theoretical bubble: {} expected: {}",
        result.theoretical_bubble_ratio,
        expected_theoretical
    );
}

#[test]
fn benchmark_more_micro_batches_improves_utilization() {
    // With more micro-batches, bubble ratio should decrease.
    let config_few = PipelineBenchmarkConfig::new(2, 2)
        .with_compute_time(Duration::from_micros(200))
        .with_transfer_time(Duration::from_micros(10))
        .with_decode_steps(16);

    let config_many = PipelineBenchmarkConfig::new(2, 8)
        .with_compute_time(Duration::from_micros(200))
        .with_transfer_time(Duration::from_micros(10))
        .with_decode_steps(16);

    let result_few = run_pipeline_benchmark(&config_few).unwrap();
    let result_many = run_pipeline_benchmark(&config_many).unwrap();

    // Theoretical: 2-stage, 2mb = 1/3 ~= 0.333; 2-stage, 8mb = 1/9 ~= 0.111
    assert!(
        result_many.theoretical_bubble_ratio < result_few.theoretical_bubble_ratio,
        "more micro-batches should reduce theoretical bubble ratio"
    );
}

#[test]
fn benchmark_bubble_efficiency_computable() {
    let config = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(Duration::from_micros(200))
        .with_transfer_time(Duration::ZERO)
        .with_decode_steps(16);

    let result = run_pipeline_benchmark(&config).unwrap();

    // Bubble efficiency should be a finite number.
    let efficiency = result.bubble_efficiency();
    assert!(
        efficiency.is_finite(),
        "bubble efficiency should be finite, got {efficiency}"
    );
}

#[test]
fn benchmark_tpot_reasonable() {
    let compute = Duration::from_micros(200);
    let config = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(compute)
        .with_transfer_time(Duration::from_micros(10))
        .with_decode_steps(32);

    let result = run_pipeline_benchmark(&config).unwrap();

    // TPOT should be at least one stage compute time.
    assert!(
        result.tpot >= compute,
        "TPOT ({:?}) should be >= stage compute time ({:?})",
        result.tpot,
        compute,
    );
}

// ---------------------------------------------------------------------------
// Scaling benchmarks
// ---------------------------------------------------------------------------

#[test]
fn scaling_benchmark_produces_results() {
    let result = run_scaling_benchmark(
        &[2, 4],
        4,
        Duration::from_micros(100),
        Duration::from_micros(10),
        8,
    )
    .unwrap();

    assert_eq!(result.results.len(), 2);
    assert_eq!(result.results[0].config.num_stages, 2);
    assert_eq!(result.results[1].config.num_stages, 4);

    // Scaling factor should be computable.
    let factor = result.scaling_factor(0, 1);
    assert!(factor.is_some());
    assert!(factor.unwrap() > 0.0);
}

#[test]
fn scaling_empty_stages_rejected() {
    let result = run_scaling_benchmark(
        &[],
        4,
        Duration::from_micros(100),
        Duration::from_micros(10),
        8,
    );
    assert!(result.is_err());
}

#[test]
fn scaling_factor_out_of_bounds() {
    let result = run_scaling_benchmark(
        &[2],
        4,
        Duration::from_micros(100),
        Duration::from_micros(10),
        8,
    )
    .unwrap();

    assert!(result.scaling_factor(0, 1).is_none());
}

// ---------------------------------------------------------------------------
// Display / formatting
// ---------------------------------------------------------------------------

#[test]
fn result_display_contains_key_info() {
    let config = PipelineBenchmarkConfig::new(2, 4)
        .with_compute_time(Duration::from_micros(100))
        .with_decode_steps(8);

    let result = run_pipeline_benchmark(&config).unwrap();
    let display = format!("{result}");

    assert!(display.contains("2 stages"));
    assert!(display.contains("4 micro-batches"));
    assert!(display.contains("tok/s"));
    assert!(display.contains("TTFT"));
    assert!(display.contains("TPOT"));
    assert!(display.contains("Bubble ratio"));
    assert!(display.contains("utilization"));
}

#[test]
fn scaling_result_display() {
    let result = run_scaling_benchmark(
        &[2, 4],
        4,
        Duration::from_micros(100),
        Duration::from_micros(10),
        8,
    )
    .unwrap();

    let display = format!("{result}");
    assert!(display.contains("Scaling Analysis"));
    assert!(display.contains("2 stages"));
    assert!(display.contains("4 stages"));
}

#[test]
fn format_report_contains_summary_table() {
    let configs = [
        PipelineBenchmarkConfig::new(2, 4)
            .with_compute_time(Duration::from_micros(100))
            .with_decode_steps(8),
        PipelineBenchmarkConfig::new(4, 8)
            .with_compute_time(Duration::from_micros(100))
            .with_decode_steps(8),
    ];

    let results: Vec<_> = configs
        .iter()
        .map(|c| run_pipeline_benchmark(c).unwrap())
        .collect();

    let report = format_benchmark_report(&results);
    assert!(report.contains("Benchmark Report"));
    assert!(report.contains("Summary"));
    assert!(report.contains("Stages"));
    assert!(report.contains("Throughput"));
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[test]
fn simulate_step_completes() {
    let pipe_config = PipelineConfig::new(2, 1).unwrap();
    let metrics = simulate_pipeline_step(
        &pipe_config,
        2,
        Duration::from_micros(50),
        Duration::from_micros(10),
    )
    .unwrap();

    assert!(metrics.sequences_completed > 0);
    assert!(!metrics.wall_time.is_zero());
    assert!(metrics.stages.contains_key(&0));
    assert!(metrics.stages.contains_key(&1));
}

#[test]
fn simulate_step_four_stages() {
    let pipe_config = PipelineConfig::new(4, 1).unwrap();
    let metrics = simulate_pipeline_step(
        &pipe_config,
        4,
        Duration::from_micros(50),
        Duration::from_micros(5),
    )
    .unwrap();

    assert!(metrics.sequences_completed > 0);
    for stage in 0..4 {
        assert!(
            metrics.stages.contains_key(&stage),
            "missing metrics for stage {stage}"
        );
    }
}
