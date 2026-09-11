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

//! KV cache transfer optimization for disaggregated inference.
//!
//! Provides multiple strategies to reduce transfer latency between prefill
//! and decode nodes:
//!
//! - [`streamed`] — Layer-by-layer streaming with compute overlap
//! - [`quantized`] — On-the-fly float16-to-int8/int4 quantization
//! - [`parallel`] — Concurrent multi-layer transfer
//! - [`benchmark`] — Transfer performance measurement
//! - [`adaptive`] — Runtime bandwidth estimation and strategy selection
//!
//! The [`AdaptiveSelector`] measures runtime bandwidth and selects the
//! optimal transfer configuration automatically.
//!
//! # Architecture
//!
//! ```text
//! Prefill Node                          Decode Node
//! ┌──────────────┐                    ┌──────────────┐
//! │ Layer 0 done │──┐  quantize       │              │
//! │ Layer 1 done │──┼──(optional)──>  │  reassemble  │
//! │ Layer 2 ...  │──┘  + stream       │  + dequant   │
//! │   (still     │                    │              │
//! │  computing)  │                    │              │
//! └──────────────┘                    └──────────────┘
//! ```

pub mod adaptive;
pub mod benchmark;
pub mod parallel;
pub mod quantized;
pub mod streamed;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// Re-export adaptive types at module level.
pub use adaptive::{AdaptiveSelector, BandwidthEstimator, BandwidthSample};

/// Transfer strategy for KV cache data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TransferStrategy {
    /// Send the entire serialized cache in a single message.
    /// Simplest approach, highest latency for large caches.
    Full,
    /// Stream cache layer-by-layer as each layer completes prefill.
    /// Overlaps compute and transfer for lower TTFT.
    Streamed,
    /// Send multiple layers concurrently over parallel connections.
    /// Best for high-bandwidth links with available parallelism.
    LayerParallel,
}

impl fmt::Display for TransferStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Streamed => write!(f, "streamed"),
            Self::LayerParallel => write!(f, "layer-parallel"),
        }
    }
}

/// Quantization level applied to KV cache before transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum CacheQuantizationLevel {
    /// No quantization; transfer in original dtype.
    #[default]
    None,
    /// Quantize float16 to int8 (~50% bandwidth reduction).
    Int8,
    /// Quantize float16 to int4 (~75% bandwidth reduction, higher quality loss).
    Int4,
}

impl fmt::Display for CacheQuantizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Int8 => write!(f, "int8"),
            Self::Int4 => write!(f, "int4"),
        }
    }
}

impl CacheQuantizationLevel {
    /// Approximate bandwidth reduction ratio (quantized / original).
    pub fn bandwidth_ratio(self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Int8 => 0.51, // 1 byte/elem + scale overhead
            Self::Int4 => 0.26, // 0.5 bytes/elem + scale overhead
        }
    }
}

/// Configuration for KV cache transfer between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Transfer strategy to use.
    pub strategy: TransferStrategy,
    /// Quantization level for cache tensors.
    pub quantization: CacheQuantizationLevel,
    /// Whether to apply LZ4 compression after quantization.
    pub compress: bool,
    /// Maximum number of layers to transfer concurrently
    /// (only meaningful for `LayerParallel` strategy).
    pub concurrency: usize,
    /// Whether to enable pipeline overlap (prefill next request
    /// while transferring current cache).
    pub pipeline_overlap: bool,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            strategy: TransferStrategy::Streamed,
            quantization: CacheQuantizationLevel::None,
            compress: false,
            concurrency: 4,
            pipeline_overlap: true,
        }
    }
}

impl TransferConfig {
    /// Create a config for the simplest full-transfer approach.
    pub fn full() -> Self {
        Self {
            strategy: TransferStrategy::Full,
            ..Default::default()
        }
    }

    /// Create a config optimized for high-bandwidth links (e.g., Thunderbolt).
    pub fn high_bandwidth() -> Self {
        Self {
            strategy: TransferStrategy::LayerParallel,
            quantization: CacheQuantizationLevel::None,
            compress: false,
            concurrency: 8,
            pipeline_overlap: true,
        }
    }

    /// Create a config optimized for bandwidth-constrained links (e.g., WiFi).
    pub fn low_bandwidth() -> Self {
        Self {
            strategy: TransferStrategy::Streamed,
            quantization: CacheQuantizationLevel::Int8,
            compress: true,
            concurrency: 2,
            pipeline_overlap: true,
        }
    }
}

/// Metadata attached to a layer-level cache transfer chunk.
///
/// Used by streamed and parallel transfer to identify individual
/// layer data within a multi-message transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTransferHeader {
    /// Sequence ID this cache belongs to.
    pub sequence_id: u64,
    /// Layer index within the model.
    pub layer_index: usize,
    /// Total number of layers in this transfer.
    pub total_layers: usize,
    /// Whether this chunk's tensor data is quantized.
    pub quantized: bool,
    /// Quantization level if quantized.
    pub quantization_level: CacheQuantizationLevel,
    /// Original number of float16 elements (for dequantization sizing).
    pub original_num_elements: usize,
}

/// Result of a single layer transfer operation.
#[derive(Debug, Clone)]
pub struct LayerTransferResult {
    /// Layer index that was transferred.
    pub layer_index: usize,
    /// Bytes sent on the wire (after quantization/compression).
    pub wire_bytes: usize,
    /// Original data size before optimization.
    pub original_bytes: usize,
    /// Time taken for this layer's transfer.
    pub duration: Duration,
}

impl LayerTransferResult {
    /// Compression ratio (wire / original). Lower is better.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.wire_bytes as f64 / self.original_bytes as f64
    }

    /// Effective throughput in bytes/second (based on original size).
    pub fn effective_throughput_bps(&self) -> f64 {
        if self.duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.original_bytes as f64 / self.duration.as_secs_f64()
    }
}

/// Aggregate result of a complete cache transfer.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Strategy that was used.
    pub strategy: TransferStrategy,
    /// Quantization level that was used.
    pub quantization: CacheQuantizationLevel,
    /// Per-layer results.
    pub layer_results: Vec<LayerTransferResult>,
    /// Total wall-clock time for the entire transfer.
    pub total_duration: Duration,
    /// Total bytes sent on the wire.
    pub total_wire_bytes: usize,
    /// Total original data size.
    pub total_original_bytes: usize,
}

impl TransferResult {
    /// Overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_original_bytes == 0 {
            return 1.0;
        }
        self.total_wire_bytes as f64 / self.total_original_bytes as f64
    }

    /// Overall throughput in MB/s (based on original data size).
    pub fn throughput_mbps(&self) -> f64 {
        if self.total_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        let bytes_per_sec = self.total_original_bytes as f64 / self.total_duration.as_secs_f64();
        bytes_per_sec / (1024.0 * 1024.0)
    }

    /// Time saved compared to a baseline full transfer duration.
    pub fn time_saved_vs(&self, baseline_duration: Duration) -> Duration {
        baseline_duration.saturating_sub(self.total_duration)
    }

    /// Percentage improvement over a baseline duration.
    pub fn improvement_pct(&self, baseline_duration: Duration) -> f64 {
        if baseline_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        let saved = self.time_saved_vs(baseline_duration);
        saved.as_secs_f64() / baseline_duration.as_secs_f64() * 100.0
    }
}

// Re-exports for convenience.
pub use benchmark::{TransferBenchConfig, TransferBenchResult, TransferBenchmark};
pub use parallel::ParallelLayerTransfer;
pub use quantized::{CacheQuantizationConfig, QuantizedCacheTransfer};
pub use streamed::{LayerReadyNotifier, StreamedCacheTransfer};
