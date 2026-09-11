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

#[test]
fn config_default_is_valid() {
    let config = DIBenchmarkConfig::default();
    config.validate().unwrap();
}

#[test]
fn config_builder_chain() {
    let config = DIBenchmarkConfig::new(2, 2)
        .with_prompt_lengths(vec![128, 1024])
        .with_concurrency(4)
        .with_decode_tokens(64);
    config.validate().unwrap();
    assert_eq!(config.prefill_nodes, 2);
    assert_eq!(config.decode_nodes, 2);
    assert_eq!(config.concurrency, 4);
    assert_eq!(config.decode_tokens, 64);
}

#[test]
fn config_validation_rejects_zero_prefill_nodes() {
    let config = DIBenchmarkConfig::new(0, 1);
    assert!(config.validate().is_err());
}

#[test]
fn config_validation_rejects_zero_decode_nodes() {
    let config = DIBenchmarkConfig::new(1, 0);
    assert!(config.validate().is_err());
}

#[test]
fn config_validation_rejects_empty_prompt_lengths() {
    let config = DIBenchmarkConfig::new(1, 1).with_prompt_lengths(vec![]);
    assert!(config.validate().is_err());
}

#[test]
fn benchmark_1p1d_produces_positive_throughput() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128])
        .with_decode_tokens(8);

    let result = run_di_benchmark(&config, 128).unwrap();
    assert!(
        result.throughput_tok_per_sec > 0.0,
        "throughput must be positive"
    );
    assert!(!result.ttft.is_zero(), "TTFT must be non-zero");
    assert!(!result.tpot.is_zero(), "TPOT must be non-zero");
}

#[test]
fn benchmark_ttft_includes_cache_transfer() {
    let config = DIBenchmarkConfig::new(1, 1).with_decode_tokens(4);

    let result = run_di_benchmark(&config, 128).unwrap();

    // TTFT should be greater than the cache transfer handoff time.
    let handoff = result.cache_transfer.total_handoff_time();
    assert!(
        result.ttft >= handoff,
        "TTFT ({:?}) should include cache handoff ({:?})",
        result.ttft,
        handoff,
    );
}

#[test]
fn benchmark_baseline_has_no_transfer_overhead() {
    let config = DIBenchmarkConfig::new(1, 1).with_decode_tokens(4);

    let result = run_di_benchmark(&config, 128).unwrap();

    // Baseline TTFT should be less than DI TTFT (no transfer overhead).
    assert!(
        result.baseline_ttft < result.ttft,
        "baseline TTFT ({:?}) should be less than DI TTFT ({:?})",
        result.baseline_ttft,
        result.ttft,
    );
}

#[test]
fn benchmark_cache_transfer_bytes_scale_with_prompt() {
    let config = DIBenchmarkConfig::new(1, 1).with_decode_tokens(4);

    let short = run_di_benchmark(&config, 128).unwrap();
    let long = run_di_benchmark(&config, 1024).unwrap();

    assert!(
        long.cache_transfer.bytes_transferred > short.cache_transfer.bytes_transferred,
        "longer prompt should transfer more bytes"
    );
}

#[test]
fn benchmark_cache_transfer_profile_has_all_phases() {
    let config = DIBenchmarkConfig::new(1, 1).with_decode_tokens(4);

    let result = run_di_benchmark(&config, 256).unwrap();
    let ct = &result.cache_transfer;

    assert!(!ct.serialization_time.is_zero());
    assert!(!ct.transfer_time.is_zero());
    assert!(!ct.deserialization_time.is_zero());
    assert!(ct.bytes_transferred > 0);
    assert_eq!(ct.prompt_len, 256);
    assert!(ct.throughput_mb_per_sec() > 0.0);
}

#[test]
fn prompt_length_analysis_has_correct_count() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128, 512, 1024])
        .with_decode_tokens(4);

    let analysis = run_prompt_length_analysis(&config).unwrap();
    assert_eq!(analysis.results.len(), 3);
}

#[test]
fn prompt_length_analysis_transfer_time_increases() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128, 1024, 4096])
        .with_decode_tokens(4);

    let analysis = run_prompt_length_analysis(&config).unwrap();

    for pair in analysis.results.windows(2) {
        assert!(
            pair[1].cache_transfer.total_handoff_time()
                >= pair[0].cache_transfer.total_handoff_time(),
            "transfer time should increase with prompt length"
        );
    }
}

#[test]
fn prompt_length_analysis_slope_is_positive() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128, 512, 1024, 4096])
        .with_decode_tokens(4);

    let analysis = run_prompt_length_analysis(&config).unwrap();
    let slope = analysis.transfer_time_slope();
    assert!(slope.is_some());
    assert!(
        slope.unwrap() > 0.0,
        "slope should be positive (more tokens = more transfer time)"
    );
}

#[test]
fn crossover_analysis_produces_entries() {
    let configs = vec![(1, 1), (1, 2)];
    let prompt_lengths = vec![128, 1024];
    let concurrency = vec![1, 4];

    let analysis = run_di_crossover_analysis(&configs, &prompt_lengths, &concurrency).unwrap();

    // 2 configs * 2 prompts * 2 concurrency = 8 entries.
    assert_eq!(analysis.entries.len(), 8);

    for entry in &analysis.entries {
        assert!(entry.di_throughput > 0.0);
        assert!(entry.speedup > 0.0);
    }
}

#[test]
fn crossover_analysis_display_works() {
    let configs = vec![(1, 1)];
    let prompt_lengths = vec![128];
    let concurrency = vec![1];

    let analysis = run_di_crossover_analysis(&configs, &prompt_lengths, &concurrency).unwrap();

    let display = format!("{analysis}");
    assert!(display.contains("Crossover Analysis"));
    assert!(display.contains("1P+1D"));
}

#[test]
fn format_di_report_contains_summary() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128])
        .with_decode_tokens(4);

    let results: Vec<_> = config
        .prompt_lengths
        .iter()
        .map(|&len| run_di_benchmark(&config, len).unwrap())
        .collect();

    let report = format_di_report(&results);
    assert!(report.contains("Disaggregated Inference"));
    assert!(report.contains("Summary"));
}
