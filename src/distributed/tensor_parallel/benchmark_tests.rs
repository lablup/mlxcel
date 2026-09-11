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

//! Unit tests for the TP benchmark module.

use std::time::Duration;

use super::*;

#[test]
fn benchmark_config_default_valid() {
    let config = TPBenchmarkConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn benchmark_config_empty_tp_sizes_invalid() {
    let config = TPBenchmarkConfig {
        tp_sizes: vec![],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn benchmark_config_zero_decode_steps_invalid() {
    let config = TPBenchmarkConfig {
        decode_steps: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn run_tp_benchmark_single_device() {
    let config = TPBenchmarkConfig::default().with_decode_steps(8);
    let result = run_tp_benchmark(&config, 1).unwrap();

    assert_eq!(result.tp_size, 1);
    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(!result.ttft.is_zero());
    assert_eq!(result.allreduce_overhead, 0.0);
    assert_eq!(result.per_rank_memory_fraction, 1.0);
    assert_eq!(result.total_tokens, 8);
}

#[test]
fn run_tp_benchmark_tp2() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(8)
        .with_layer_compute_time(Duration::from_micros(100))
        .with_allreduce_time(Duration::from_micros(20));

    let result = run_tp_benchmark(&config, 2).unwrap();

    assert_eq!(result.tp_size, 2);
    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(result.allreduce_overhead > 0.0);
    assert!(result.per_rank_memory_fraction < 1.0);
    assert!(result.allreduce_profile.total_operations > 0);
}

#[test]
fn run_tp_benchmark_tp4() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(8)
        .with_layer_compute_time(Duration::from_micros(100))
        .with_allreduce_time(Duration::from_micros(20));

    let result = run_tp_benchmark(&config, 4).unwrap();

    assert_eq!(result.tp_size, 4);
    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(result.per_rank_memory_fraction < 0.5);
}

#[test]
fn scaling_analysis_basic() {
    let config = TPBenchmarkConfig {
        tp_sizes: vec![1, 2],
        decode_steps: 8,
        layer_compute_time: Duration::from_micros(100),
        allreduce_time: Duration::from_micros(10),
        ..Default::default()
    };

    let analysis = run_scaling_analysis(&config).unwrap();

    assert_eq!(analysis.results.len(), 2);
    assert_eq!(analysis.results[0].tp_size, 1);
    assert_eq!(analysis.results[1].tp_size, 2);

    // TP=2 should have some throughput improvement.
    assert!(analysis.results[1].throughput_tok_per_sec > 0.0);

    // Scaling factor should be computable.
    let factor = analysis.scaling_factor(0, 1).unwrap();
    assert!(factor > 0.0);
}

#[test]
fn crossover_analysis_basic() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(4)
        .with_layer_compute_time(Duration::from_micros(50))
        .with_allreduce_time(Duration::from_micros(10));

    let analysis = run_crossover_analysis(&config, &[2048, 4096], &[1, 2]).unwrap();

    assert!(!analysis.entries.is_empty());

    // Every entry should have positive throughput.
    for entry in &analysis.entries {
        assert!(entry.throughput_tok_per_sec > 0.0);
    }
}

#[test]
fn format_report_non_empty() {
    let config = TPBenchmarkConfig::default().with_decode_steps(4);
    let r1 = run_tp_benchmark(&config, 1).unwrap();
    let r2 = run_tp_benchmark(&config, 2).unwrap();

    let report = format_tp_benchmark_report(&[r1, r2]);

    assert!(report.contains("Tensor Parallelism Benchmark Report"));
    assert!(report.contains("Summary"));
    assert!(report.contains("tp_size=1"));
    assert!(report.contains("tp_size=2"));
}

#[test]
fn allreduce_profile_total_time() {
    let profile = AllReduceProfile {
        serialization_time: Duration::from_millis(1),
        transfer_time: Duration::from_millis(7),
        sync_wait_time: Duration::from_millis(2),
        total_operations: 10,
        total_bytes_transferred: 1024,
        average_bandwidth: 1e6,
    };
    assert_eq!(profile.total_time(), Duration::from_millis(10));
}

#[test]
fn lockstep_benchmark_tp1() {
    let result = run_lockstep_benchmark(1, 2, 3).unwrap();

    assert!(result.all_synchronized);
    assert_eq!(result.errors, 0);
    assert!(result.total_tokens_generated > 0);
}

#[test]
fn lockstep_benchmark_tp2() {
    let result = run_lockstep_benchmark(2, 2, 3).unwrap();

    assert!(result.all_synchronized);
    assert_eq!(result.errors, 0);
    assert!(result.total_tokens_generated > 0);
}

#[test]
fn lockstep_benchmark_tp4() {
    let result = run_lockstep_benchmark(4, 3, 5).unwrap();

    assert!(result.all_synchronized);
    assert_eq!(result.errors, 0);
    assert!(result.total_tokens_generated > 0);
}

#[test]
fn display_impls_non_empty() {
    let config = TPBenchmarkConfig::default().with_decode_steps(4);
    let result = run_tp_benchmark(&config, 2).unwrap();
    let display = format!("{result}");
    assert!(display.contains("tp_size=2"));

    let profile = AllReduceProfile::default();
    let display = format!("{profile}");
    assert!(display.contains("All-Reduce Profile"));

    let lockstep = run_lockstep_benchmark(2, 1, 2).unwrap();
    let display = format!("{lockstep}");
    assert!(display.contains("Lockstep Benchmark"));
}
