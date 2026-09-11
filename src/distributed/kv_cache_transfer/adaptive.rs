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

//! Runtime bandwidth estimation and adaptive transfer strategy selection.
//!
//! Uses exponentially weighted moving average (EWMA) to track bandwidth
//! and selects optimal transfer configuration based on measured throughput.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::{CacheQuantizationLevel, TransferConfig, TransferStrategy};

/// A single bandwidth measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthSample {
    /// Bytes transferred in this sample.
    pub bytes: usize,
    /// Time elapsed for the transfer.
    pub duration: Duration,
    /// Timestamp when this sample was taken.
    pub timestamp: Instant,
}

impl BandwidthSample {
    /// Throughput in bytes per second.
    pub fn throughput_bps(&self) -> f64 {
        if self.duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.bytes as f64 / self.duration.as_secs_f64()
    }
}

/// Runtime bandwidth estimator using exponentially weighted moving average.
///
/// Measures transfer throughput and provides estimates for adaptive strategy
/// selection. Uses EWMA to respond to recent bandwidth changes while
/// smoothing transient spikes.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// EWMA smoothing factor (0.0 = ignore new, 1.0 = ignore history).
    alpha: f64,
    /// Current EWMA estimate in bytes/second.
    estimated_bps: f64,
    /// Number of samples recorded.
    sample_count: u64,
    /// History window for percentile calculations.
    recent_samples: VecDeque<BandwidthSample>,
    /// Maximum samples to keep in history.
    max_history: usize,
}

impl BandwidthEstimator {
    /// Create a new estimator with the given EWMA smoothing factor.
    ///
    /// `alpha` controls responsiveness: higher values track recent changes
    /// faster, lower values provide more stability. Typical: 0.2-0.3.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 0.99),
            estimated_bps: 0.0,
            sample_count: 0,
            recent_samples: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Record a bandwidth measurement.
    pub fn record(&mut self, sample: BandwidthSample) {
        let bps = sample.throughput_bps();
        if bps <= 0.0 {
            return;
        }

        if self.sample_count == 0 {
            self.estimated_bps = bps;
        } else {
            self.estimated_bps = self.alpha * bps + (1.0 - self.alpha) * self.estimated_bps;
        }
        self.sample_count += 1;

        self.recent_samples.push_back(sample);
        if self.recent_samples.len() > self.max_history {
            self.recent_samples.pop_front();
        }
    }

    /// Record a transfer of `bytes` that took `duration`.
    pub fn record_transfer(&mut self, bytes: usize, duration: Duration) {
        self.record(BandwidthSample {
            bytes,
            duration,
            timestamp: Instant::now(),
        });
    }

    /// Current estimated bandwidth in bytes/second.
    pub fn estimated_bps(&self) -> f64 {
        self.estimated_bps
    }

    /// Current estimated bandwidth in megabytes/second.
    pub fn estimated_mbps(&self) -> f64 {
        self.estimated_bps / (1024.0 * 1024.0)
    }

    /// Number of samples recorded.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Estimate transfer time for the given number of bytes.
    pub fn estimate_transfer_time(&self, bytes: usize) -> Duration {
        if self.estimated_bps <= 0.0 {
            return Duration::from_secs(u64::MAX);
        }
        Duration::from_secs_f64(bytes as f64 / self.estimated_bps)
    }

    /// Return the p50 (median) throughput from recent samples.
    pub fn p50_bps(&self) -> f64 {
        self.percentile_bps(50)
    }

    /// Return the p5 (conservative) throughput from recent samples.
    pub fn p5_bps(&self) -> f64 {
        self.percentile_bps(5)
    }

    fn percentile_bps(&self, pct: usize) -> f64 {
        if self.recent_samples.is_empty() {
            return 0.0;
        }
        let mut throughputs: Vec<f64> = self
            .recent_samples
            .iter()
            .map(|s| s.throughput_bps())
            .collect();
        throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (pct * throughputs.len() / 100).min(throughputs.len() - 1);
        throughputs[idx]
    }
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        Self::new(0.25)
    }
}

/// Selects optimal transfer configuration based on runtime conditions.
///
/// Uses bandwidth estimates and cache size to choose between strategies:
/// - High bandwidth (>1 GB/s): LayerParallel, no quantization
/// - Medium bandwidth (100 MB/s - 1 GB/s): Streamed, optional int8
/// - Low bandwidth (<100 MB/s): Streamed, int8 quantization + compression
pub struct AdaptiveSelector {
    estimator: BandwidthEstimator,
    /// Bandwidth threshold for high-bandwidth strategy (bytes/sec).
    high_bw_threshold: f64,
    /// Bandwidth threshold for medium-bandwidth strategy (bytes/sec).
    medium_bw_threshold: f64,
}

impl AdaptiveSelector {
    /// Create a selector with the given bandwidth estimator.
    pub fn new(estimator: BandwidthEstimator) -> Self {
        Self {
            estimator,
            high_bw_threshold: 1_000_000_000.0, // 1 GB/s
            medium_bw_threshold: 100_000_000.0, // 100 MB/s
        }
    }

    /// Record a bandwidth measurement.
    pub fn record_transfer(&mut self, bytes: usize, duration: Duration) {
        self.estimator.record_transfer(bytes, duration);
    }

    /// Select the optimal transfer config for the given cache size.
    ///
    /// Uses the conservative p5 bandwidth estimate to avoid optimistic
    /// choices that would cause timeouts.
    pub fn select(&self, cache_size_bytes: usize) -> TransferConfig {
        let bw = if self.estimator.sample_count() >= 3 {
            self.estimator.p5_bps()
        } else {
            self.estimator.estimated_bps()
        };

        if bw <= 0.0 {
            // No bandwidth data yet; use safe defaults.
            return TransferConfig::default();
        }

        if bw >= self.high_bw_threshold {
            // High bandwidth: parallel layers, no compression needed.
            TransferConfig::high_bandwidth()
        } else if bw >= self.medium_bw_threshold {
            // Medium bandwidth: stream with optional quantization for large caches.
            let quant = if cache_size_bytes > 256 * 1024 * 1024 {
                CacheQuantizationLevel::Int8
            } else {
                CacheQuantizationLevel::None
            };
            TransferConfig {
                strategy: TransferStrategy::Streamed,
                quantization: quant,
                compress: false,
                concurrency: 4,
                pipeline_overlap: true,
            }
        } else {
            // Low bandwidth: maximize compression.
            TransferConfig::low_bandwidth()
        }
    }

    /// Return the current bandwidth estimate in MB/s.
    pub fn estimated_mbps(&self) -> f64 {
        self.estimator.estimated_mbps()
    }

    /// Return the underlying estimator (read-only).
    pub fn estimator(&self) -> &BandwidthEstimator {
        &self.estimator
    }
}

impl Default for AdaptiveSelector {
    fn default() -> Self {
        Self::new(BandwidthEstimator::default())
    }
}
