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

use super::*;

// --- TransferBenchConfig ---

#[test]
fn bench_config_default() {
    let config = TransferBenchConfig::default();
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.seq_len, 2048);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn bench_config_bytes_per_layer() {
    let config = TransferBenchConfig {
        num_layers: 1,
        num_kv_heads: 8,
        head_dim: 128,
        seq_len: 2048,
        batch_size: 1,
        ..Default::default()
    };
    // 1 * 8 * 2048 * 128 = 2,097,152 elements per tensor.
    // * 2 bytes (float16) * 2 tensors (K+V) = 8,388,608 bytes.
    assert_eq!(config.bytes_per_layer(), 8_388_608);
}

#[test]
fn bench_config_total_bytes() {
    let config = TransferBenchConfig::for_model(32, 8, 128, 2048);
    assert_eq!(config.total_bytes(), config.bytes_per_layer() * 32);
}

#[test]
fn bench_config_total_size_str() {
    let config = TransferBenchConfig::for_model(32, 8, 128, 2048);
    let size = config.total_size_str();
    assert!(size.contains("MB") || size.contains("GB"));
}

// --- generate_synthetic_entries ---

#[test]
fn synthetic_entries_correct_count() {
    let config = TransferBenchConfig {
        num_layers: 4,
        num_kv_heads: 2,
        head_dim: 64,
        seq_len: 128,
        batch_size: 1,
        ..Default::default()
    };
    let entries = generate_synthetic_entries(&config);
    assert_eq!(entries.len(), 4);

    for entry in &entries {
        assert!(entry.keys.is_some());
        assert!(entry.values.is_some());
        let keys = entry.keys.as_ref().unwrap();
        assert_eq!(keys.shape, vec![1, 2, 128, 64]);
        assert_eq!(keys.dtype, 9); // FLOAT16
        // 1 * 2 * 128 * 64 * 2 bytes = 32,768 bytes
        assert_eq!(keys.data.len(), 32_768);
    }
}

#[test]
fn synthetic_entries_deterministic() {
    let config = TransferBenchConfig {
        num_layers: 2,
        num_kv_heads: 1,
        head_dim: 32,
        seq_len: 64,
        batch_size: 1,
        ..Default::default()
    };
    let entries1 = generate_synthetic_entries(&config);
    let entries2 = generate_synthetic_entries(&config);
    assert_eq!(
        entries1[0].keys.as_ref().unwrap().data,
        entries2[0].keys.as_ref().unwrap().data
    );
}

// --- TransferBenchmark ---

#[test]
fn benchmark_run_all() {
    let config = TransferBenchConfig {
        num_layers: 2,
        num_kv_heads: 1,
        head_dim: 32,
        seq_len: 64,
        batch_size: 1,
        warmup_iterations: 1,
        measure_iterations: 2,
        strategies: vec![TransferStrategy::Full],
        quantization_levels: vec![CacheQuantizationLevel::None, CacheQuantizationLevel::Int8],
    };

    let bench = TransferBenchmark::new(config);
    let results = bench.run_all();

    assert_eq!(results.len(), 2); // None + Int8

    // None should have ratio ~1.0.
    let none_result = &results[0];
    assert!(none_result.compression_ratio > 0.9);
    assert!(none_result.compression_ratio <= 1.01);

    // Int8 should have ratio <0.7.
    let int8_result = &results[1];
    assert!(int8_result.compression_ratio < 0.7);
}

#[test]
fn benchmark_result_summary_format() {
    let result = TransferBenchResult {
        strategy: TransferStrategy::Streamed,
        quantization: CacheQuantizationLevel::Int8,
        original_bytes: 1_000_000,
        wire_bytes: 510_000,
        mean_prepare_time: std::time::Duration::from_millis(5),
        mean_total_time: std::time::Duration::from_millis(6),
        throughput_mbps: 158.7,
        compression_ratio: 0.51,
        iterations: 5,
    };
    let summary = result.summary();
    assert!(summary.contains("streamed"));
    assert!(summary.contains("int8"));
    assert!(summary.contains("MB/s"));
}

// --- format_bytes ---

#[test]
fn format_bytes_ranges() {
    assert!(format_bytes(500).contains("B"));
    assert!(format_bytes(2048).contains("KiB"));
    assert!(format_bytes(5 * 1024 * 1024).contains("MiB"));
    assert!(format_bytes(2 * 1024 * 1024 * 1024).contains("GiB"));
}
