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

//! Benchmarking infrastructure for KV cache transfer strategies.
//!
//! Generates synthetic cache data matching real model parameters and
//! measures transfer throughput for different strategies, quantization
//! levels, and concurrency settings.
//!
//! Used by: performance validation and strategy tuning

use std::time::{Duration, Instant};

use super::{CacheQuantizationLevel, TransferStrategy};

use super::streamed::prepare_layer_payload;
use crate::distributed::kv_cache_serde::types::{RawTensorData, SerializableCacheEntry};

/// Configuration for a transfer benchmark run.
#[derive(Debug, Clone)]
pub struct TransferBenchConfig {
    /// Number of layers in the synthetic model.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Sequence length (number of cached tokens).
    pub seq_len: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measure_iterations: usize,
    /// Strategies to benchmark.
    pub strategies: Vec<TransferStrategy>,
    /// Quantization levels to benchmark.
    pub quantization_levels: Vec<CacheQuantizationLevel>,
}

impl Default for TransferBenchConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            seq_len: 2048,
            batch_size: 1,
            warmup_iterations: 2,
            measure_iterations: 5,
            strategies: vec![
                TransferStrategy::Full,
                TransferStrategy::Streamed,
                TransferStrategy::LayerParallel,
            ],
            quantization_levels: vec![
                CacheQuantizationLevel::None,
                CacheQuantizationLevel::Int8,
                CacheQuantizationLevel::Int4,
            ],
        }
    }
}

impl TransferBenchConfig {
    /// Configure for a specific model profile.
    pub fn for_model(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            seq_len,
            ..Default::default()
        }
    }

    /// Total float16 bytes per layer (keys + values).
    pub fn bytes_per_layer(&self) -> usize {
        // Shape: [batch, num_kv_heads, seq_len, head_dim]
        // Two tensors (K, V), each float16 (2 bytes per element).
        let elements_per_tensor =
            self.batch_size * self.num_kv_heads * self.seq_len * self.head_dim;
        elements_per_tensor * 2 * 2 // * 2 bytes * 2 tensors (K+V)
    }

    /// Total float16 bytes for all layers.
    pub fn total_bytes(&self) -> usize {
        self.bytes_per_layer() * self.num_layers
    }

    /// Human-readable total size.
    pub fn total_size_str(&self) -> String {
        format_bytes(self.total_bytes())
    }
}

/// Result of a single benchmark configuration.
#[derive(Debug, Clone)]
pub struct TransferBenchResult {
    /// Strategy tested.
    pub strategy: TransferStrategy,
    /// Quantization level tested.
    pub quantization: CacheQuantizationLevel,
    /// Original data size (bytes).
    pub original_bytes: usize,
    /// Wire data size after quantization (bytes).
    pub wire_bytes: usize,
    /// Mean serialization/quantization time.
    pub mean_prepare_time: Duration,
    /// Mean total time (prepare + simulated transfer).
    pub mean_total_time: Duration,
    /// Effective throughput in MB/s (original data / total time).
    pub throughput_mbps: f64,
    /// Bandwidth reduction ratio (wire / original).
    pub compression_ratio: f64,
    /// Number of measurement iterations.
    pub iterations: usize,
}

impl TransferBenchResult {
    /// Format as a human-readable summary line.
    pub fn summary(&self) -> String {
        format!(
            "{:>14} | {:>5} | {:>8} | {:>8} | {:>7.1} MB/s | {:.2}x",
            self.strategy,
            self.quantization,
            format_bytes(self.original_bytes),
            format_bytes(self.wire_bytes),
            self.throughput_mbps,
            self.compression_ratio,
        )
    }
}

/// Transfer benchmark runner.
///
/// Generates synthetic cache data and measures the CPU-side cost of
/// serialization, quantization, and preparation for different strategies.
///
/// Note: This measures the data preparation pipeline only, not actual
/// network transfer. For end-to-end benchmarks, use a real transport
/// implementation.
pub struct TransferBenchmark {
    config: TransferBenchConfig,
    /// Pre-generated synthetic cache entries.
    entries: Vec<SerializableCacheEntry>,
}

impl TransferBenchmark {
    /// Create a new benchmark with the given config.
    pub fn new(config: TransferBenchConfig) -> Self {
        let entries = generate_synthetic_entries(&config);
        Self { config, entries }
    }

    /// Run all configured benchmarks.
    pub fn run_all(&self) -> Vec<TransferBenchResult> {
        let mut results = Vec::new();

        for &quant in &self.config.quantization_levels {
            // Measure preparation (quantization + serialization) cost.
            let result = self.bench_preparation(quant);
            results.push(result);
        }

        results
    }

    /// Benchmark the data preparation pipeline for a given quantization level.
    fn bench_preparation(&self, quantization: CacheQuantizationLevel) -> TransferBenchResult {
        let original_bytes = self.config.total_bytes();

        // Warmup.
        for _ in 0..self.config.warmup_iterations {
            let _ = self.prepare_all(quantization);
        }

        // Measure.
        let mut prepare_times = Vec::with_capacity(self.config.measure_iterations);
        let mut wire_bytes_total = 0usize;

        for _ in 0..self.config.measure_iterations {
            let start = Instant::now();
            let wire_bytes = self.prepare_all(quantization);
            prepare_times.push(start.elapsed());
            wire_bytes_total = wire_bytes;
        }

        let mean_prepare = mean_duration(&prepare_times);

        // Simulated total time (prepare + hypothetical transfer at 1 GB/s).
        let simulated_transfer = Duration::from_secs_f64(wire_bytes_total as f64 / 1_000_000_000.0);
        let mean_total = mean_prepare + simulated_transfer;

        let throughput_mbps = if mean_total.as_secs_f64() > 0.0 {
            original_bytes as f64 / mean_total.as_secs_f64() / (1024.0 * 1024.0)
        } else {
            0.0
        };

        let compression_ratio = if original_bytes > 0 {
            wire_bytes_total as f64 / original_bytes as f64
        } else {
            1.0
        };

        TransferBenchResult {
            strategy: TransferStrategy::Full,
            quantization,
            original_bytes,
            wire_bytes: wire_bytes_total,
            mean_prepare_time: mean_prepare,
            mean_total_time: mean_total,
            throughput_mbps,
            compression_ratio,
            iterations: self.config.measure_iterations,
        }
    }

    /// Prepare all layers and return total wire bytes.
    fn prepare_all(&self, quantization: CacheQuantizationLevel) -> usize {
        let mut total_wire = 0usize;
        for entry in &self.entries {
            if let Ok((wire_data, _, _)) = prepare_layer_payload(entry, quantization, false) {
                total_wire += wire_data.len();
            }
        }
        total_wire
    }

    /// Print a formatted benchmark report.
    pub fn print_report(results: &[TransferBenchResult]) {
        println!("KV Cache Transfer Benchmark Results");
        println!("{}", "=".repeat(75));
        println!(
            "{:>14} | {:>5} | {:>8} | {:>8} | {:>12} | {:>6}",
            "Strategy", "Quant", "Original", "Wire", "Throughput", "Ratio"
        );
        println!("{}", "-".repeat(75));
        for result in results {
            println!("{}", result.summary());
        }
        println!("{}", "=".repeat(75));
    }
}

/// Generate synthetic cache entries matching the config parameters.
fn generate_synthetic_entries(config: &TransferBenchConfig) -> Vec<SerializableCacheEntry> {
    let elements_per_tensor =
        config.batch_size * config.num_kv_heads * config.seq_len * config.head_dim;
    let bytes_per_tensor = elements_per_tensor * 2; // float16

    let shape = vec![
        config.batch_size as i32,
        config.num_kv_heads as i32,
        config.seq_len as i32,
        config.head_dim as i32,
    ];

    (0..config.num_layers)
        .map(|layer_idx| {
            // Generate deterministic pseudo-random float16 data.
            let key_data = generate_pseudo_f16_data(bytes_per_tensor, layer_idx * 2);
            let value_data = generate_pseudo_f16_data(bytes_per_tensor, layer_idx * 2 + 1);

            SerializableCacheEntry {
                keys: Some(RawTensorData {
                    data: key_data,
                    shape: shape.clone(),
                    dtype: 9, // FLOAT16
                }),
                values: Some(RawTensorData {
                    data: value_data,
                    shape: shape.clone(),
                    dtype: 9, // FLOAT16
                }),
            }
        })
        .collect()
}

/// Generate pseudo-random float16 byte data for benchmarking.
///
/// Uses a simple LCG to produce reproducible data that approximates
/// the statistical distribution of real KV cache activations.
fn generate_pseudo_f16_data(num_bytes: usize, seed: usize) -> Vec<u8> {
    let mut data = vec![0u8; num_bytes];
    let mut state = seed as u64 ^ 0xDEADBEEF;

    for chunk in data.chunks_exact_mut(2) {
        // LCG step.
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Generate a float16 in a reasonable range (-1.0 to 1.0).
        // Use upper bits for better distribution.
        let raw = ((state >> 48) & 0xFFFF) as u16;
        // Map to a small float16 value (exponent 14 = ~0.5, 15 = ~1.0).
        let sign = (raw >> 15) & 1;
        let mantissa = raw & 0x3FF;
        let f16_bits = (sign << 15) | (0x0E << 10) | mantissa; // exponent 14 => ~0.25-0.5
        chunk[0] = f16_bits as u8;
        chunk[1] = (f16_bits >> 8) as u8;
    }

    data
}

fn mean_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let total: Duration = durations.iter().sum();
    total / durations.len() as u32
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1}GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1}MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
#[path = "benchmark_tests.rs"]
mod tests;
