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

use std::time::{Duration, Instant};

use super::*;

// --- TransferStrategy ---

#[test]
fn transfer_strategy_display() {
    assert_eq!(TransferStrategy::Full.to_string(), "full");
    assert_eq!(TransferStrategy::Streamed.to_string(), "streamed");
    assert_eq!(
        TransferStrategy::LayerParallel.to_string(),
        "layer-parallel"
    );
}

// --- CacheQuantizationLevel ---

#[test]
fn quantization_level_display() {
    assert_eq!(CacheQuantizationLevel::None.to_string(), "none");
    assert_eq!(CacheQuantizationLevel::Int8.to_string(), "int8");
    assert_eq!(CacheQuantizationLevel::Int4.to_string(), "int4");
}

#[test]
fn quantization_level_bandwidth_ratio() {
    assert_eq!(CacheQuantizationLevel::None.bandwidth_ratio(), 1.0);
    assert!(CacheQuantizationLevel::Int8.bandwidth_ratio() < 0.55);
    assert!(CacheQuantizationLevel::Int8.bandwidth_ratio() > 0.45);
    assert!(CacheQuantizationLevel::Int4.bandwidth_ratio() < 0.30);
    assert!(CacheQuantizationLevel::Int4.bandwidth_ratio() > 0.20);
}

#[test]
fn quantization_level_default_is_none() {
    let level = CacheQuantizationLevel::default();
    assert_eq!(level, CacheQuantizationLevel::None);
}

// --- TransferConfig ---

#[test]
fn transfer_config_default() {
    let config = TransferConfig::default();
    assert_eq!(config.strategy, TransferStrategy::Streamed);
    assert_eq!(config.quantization, CacheQuantizationLevel::None);
    assert!(!config.compress);
    assert_eq!(config.concurrency, 4);
    assert!(config.pipeline_overlap);
}

#[test]
fn transfer_config_presets() {
    let full = TransferConfig::full();
    assert_eq!(full.strategy, TransferStrategy::Full);

    let high = TransferConfig::high_bandwidth();
    assert_eq!(high.strategy, TransferStrategy::LayerParallel);
    assert_eq!(high.concurrency, 8);
    assert_eq!(high.quantization, CacheQuantizationLevel::None);

    let low = TransferConfig::low_bandwidth();
    assert_eq!(low.strategy, TransferStrategy::Streamed);
    assert_eq!(low.quantization, CacheQuantizationLevel::Int8);
    assert!(low.compress);
}

// --- BandwidthSample ---

#[test]
fn bandwidth_sample_throughput() {
    let sample = BandwidthSample {
        bytes: 1_000_000,
        duration: Duration::from_secs(1),
        timestamp: Instant::now(),
    };
    assert!((sample.throughput_bps() - 1_000_000.0).abs() < 1.0);
}

#[test]
fn bandwidth_sample_zero_duration() {
    let sample = BandwidthSample {
        bytes: 1_000_000,
        duration: Duration::ZERO,
        timestamp: Instant::now(),
    };
    assert_eq!(sample.throughput_bps(), 0.0);
}

// --- BandwidthEstimator ---

#[test]
fn bandwidth_estimator_initial() {
    let est = BandwidthEstimator::new(0.25);
    assert_eq!(est.estimated_bps(), 0.0);
    assert_eq!(est.sample_count(), 0);
}

#[test]
fn bandwidth_estimator_first_sample() {
    let mut est = BandwidthEstimator::new(0.25);
    est.record_transfer(1_000_000, Duration::from_secs(1));
    assert!((est.estimated_bps() - 1_000_000.0).abs() < 1.0);
    assert_eq!(est.sample_count(), 1);
}

#[test]
fn bandwidth_estimator_ewma() {
    let mut est = BandwidthEstimator::new(0.5);
    // First sample: 1 MB/s
    est.record_transfer(1_000_000, Duration::from_secs(1));
    // Second sample: 2 MB/s
    est.record_transfer(2_000_000, Duration::from_secs(1));
    // EWMA: 0.5 * 2M + 0.5 * 1M = 1.5M
    assert!((est.estimated_bps() - 1_500_000.0).abs() < 100.0);
}

#[test]
fn bandwidth_estimator_transfer_time() {
    let mut est = BandwidthEstimator::new(0.25);
    est.record_transfer(1_000_000, Duration::from_secs(1)); // 1 MB/s
    let time = est.estimate_transfer_time(10_000_000);
    assert!((time.as_secs_f64() - 10.0).abs() < 0.1);
}

#[test]
fn bandwidth_estimator_no_data_transfer_time() {
    let est = BandwidthEstimator::new(0.25);
    let time = est.estimate_transfer_time(1_000_000);
    assert_eq!(time, Duration::from_secs(u64::MAX));
}

#[test]
fn bandwidth_estimator_percentiles() {
    let mut est = BandwidthEstimator::new(0.1);
    // Add samples with varying throughput.
    for i in 1..=20 {
        est.record_transfer(i * 1_000_000, Duration::from_secs(1));
    }
    // p50 should be around the median.
    let p50 = est.p50_bps();
    assert!(p50 > 5_000_000.0);
    assert!(p50 < 15_000_000.0);
    // p5 should be lower (conservative).
    let p5 = est.p5_bps();
    assert!(p5 <= p50);
}

// --- AdaptiveSelector ---

#[test]
fn adaptive_selector_no_data() {
    let sel = AdaptiveSelector::default();
    let config = sel.select(100_000_000);
    // Should return safe defaults.
    assert_eq!(config.strategy, TransferStrategy::Streamed);
}

#[test]
fn adaptive_selector_high_bandwidth() {
    let mut est = BandwidthEstimator::new(0.9);
    // Record high bandwidth samples.
    for _ in 0..5 {
        est.record_transfer(2_000_000_000, Duration::from_secs(1)); // 2 GB/s
    }
    let sel = AdaptiveSelector::new(est);
    let config = sel.select(100_000_000);
    assert_eq!(config.strategy, TransferStrategy::LayerParallel);
    assert_eq!(config.quantization, CacheQuantizationLevel::None);
}

#[test]
fn adaptive_selector_medium_bandwidth_large_cache() {
    let mut est = BandwidthEstimator::new(0.9);
    for _ in 0..5 {
        est.record_transfer(500_000_000, Duration::from_secs(1)); // 500 MB/s
    }
    let sel = AdaptiveSelector::new(est);
    // Large cache should trigger int8 quantization.
    let config = sel.select(512 * 1024 * 1024);
    assert_eq!(config.strategy, TransferStrategy::Streamed);
    assert_eq!(config.quantization, CacheQuantizationLevel::Int8);
}

#[test]
fn adaptive_selector_low_bandwidth() {
    let mut est = BandwidthEstimator::new(0.9);
    for _ in 0..5 {
        est.record_transfer(50_000_000, Duration::from_secs(1)); // 50 MB/s
    }
    let sel = AdaptiveSelector::new(est);
    let config = sel.select(100_000_000);
    assert_eq!(config.strategy, TransferStrategy::Streamed);
    assert_eq!(config.quantization, CacheQuantizationLevel::Int8);
    assert!(config.compress);
}

// --- LayerTransferResult ---

#[test]
fn layer_transfer_result_ratio() {
    let result = LayerTransferResult {
        layer_index: 0,
        wire_bytes: 500,
        original_bytes: 1000,
        duration: Duration::from_millis(100),
    };
    assert!((result.compression_ratio() - 0.5).abs() < 0.01);
    assert!(result.effective_throughput_bps() > 0.0);
}

#[test]
fn layer_transfer_result_zero_original() {
    let result = LayerTransferResult {
        layer_index: 0,
        wire_bytes: 0,
        original_bytes: 0,
        duration: Duration::from_millis(100),
    };
    assert_eq!(result.compression_ratio(), 1.0);
}

// --- TransferResult ---

#[test]
fn transfer_result_improvement() {
    let result = TransferResult {
        strategy: TransferStrategy::Streamed,
        quantization: CacheQuantizationLevel::Int8,
        layer_results: Vec::new(),
        total_duration: Duration::from_secs(7),
        total_wire_bytes: 500,
        total_original_bytes: 1000,
    };

    let baseline = Duration::from_secs(10);
    let saved = result.time_saved_vs(baseline);
    assert_eq!(saved, Duration::from_secs(3));
    assert!((result.improvement_pct(baseline) - 30.0).abs() < 0.1);
}

// --- LayerTransferHeader serialization ---

#[test]
fn layer_transfer_header_roundtrip() {
    let header = LayerTransferHeader {
        sequence_id: 42,
        layer_index: 5,
        total_layers: 32,
        quantized: true,
        quantization_level: CacheQuantizationLevel::Int8,
        original_num_elements: 1024,
    };

    let json = serde_json::to_vec(&header).unwrap();
    let decoded: LayerTransferHeader = serde_json::from_slice(&json).unwrap();

    assert_eq!(decoded.sequence_id, 42);
    assert_eq!(decoded.layer_index, 5);
    assert_eq!(decoded.total_layers, 32);
    assert!(decoded.quantized);
    assert_eq!(decoded.quantization_level, CacheQuantizationLevel::Int8);
    assert_eq!(decoded.original_num_elements, 1024);
}
