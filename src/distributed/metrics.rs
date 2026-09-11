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

//! Per-node metrics collection for distributed inference clusters.
//!
//! Tracks throughput (tokens/sec), latency percentiles (p50/p95/p99),
//! memory usage, and network utilization for each node. Metrics are stored
//! in a thread-safe registry that can be queried by the health monitoring
//! system or exposed through an API endpoint.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use mlxcel_core::cache::PagedCacheStats;
use serde::{Deserialize, Serialize};

/// Latency percentile snapshot computed from a sliding window of samples.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// Median latency.
    pub p50: Duration,
    /// 95th percentile latency.
    pub p95: Duration,
    /// 99th percentile latency.
    pub p99: Duration,
    /// Minimum observed latency in the window.
    pub min: Duration,
    /// Maximum observed latency in the window.
    pub max: Duration,
}

/// Snapshot of a single node's metrics at a point in time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Tokens generated per second (rolling average).
    pub throughput_tokens_per_sec: f64,
    /// Request latency percentiles.
    pub latency: LatencyPercentiles,
    /// Memory used in bytes.
    pub memory_used_bytes: u64,
    /// Total memory available in bytes.
    pub memory_total_bytes: u64,
    /// Network bytes sent since last reset.
    pub network_bytes_sent: u64,
    /// Network bytes received since last reset.
    pub network_bytes_recv: u64,
    /// Number of active inference requests.
    pub active_requests: u32,
    /// Total requests processed since startup.
    pub total_requests: u64,
    /// Total tokens generated since startup.
    pub total_tokens: u64,
    /// Paged KV allocator and usage snapshot when paged decode is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paged_kv: Option<PagedKvMetrics>,
    /// Number of times paged decode was requested but fell back to dense execution.
    pub paged_decode_fallbacks: u64,
}

/// Snapshot of paged KV allocator state for operations and routing decisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PagedKvMetrics {
    pub block_size: u32,
    pub allocated_blocks: u64,
    pub live_blocks: u64,
    pub free_blocks: u64,
    pub bytes_reserved: u64,
    pub bytes_in_use: u64,
}

/// Configuration for the metrics collector.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// How many latency samples to retain in the sliding window.
    pub latency_window_size: usize,
    /// How many throughput samples to retain for rolling average.
    pub throughput_window_size: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            latency_window_size: 1000,
            throughput_window_size: 100,
        }
    }
}

/// Collects and computes metrics for a single node.
///
/// Thread-safe: all methods take `&self` and use interior mutability so the
/// collector can be shared across request handlers and background tasks.
#[derive(Clone)]
pub struct MetricsCollector {
    inner: Arc<RwLock<MetricsInner>>,
    config: MetricsConfig,
}

struct MetricsInner {
    /// Sliding window of latency samples (most recent at back).
    /// Uses VecDeque for O(1) eviction from the front instead of Vec::remove(0)
    /// which is O(n).
    latency_samples: VecDeque<Duration>,
    /// Sliding window of (timestamp, token_count) for throughput calculation.
    /// Uses VecDeque for O(1) eviction from the front.
    throughput_samples: VecDeque<(Instant, u64)>,
    /// Cumulative network counters.
    network_bytes_sent: u64,
    network_bytes_recv: u64,
    /// Active request gauge.
    active_requests: u32,
    /// Lifetime counters.
    total_requests: u64,
    total_tokens: u64,
    /// Memory snapshot (updated externally).
    memory_used_bytes: u64,
    memory_total_bytes: u64,
    /// Latest paged KV allocator snapshot.
    paged_kv: Option<PagedKvMetrics>,
    /// Count of paged decode fallback events.
    paged_decode_fallbacks: u64,
}

impl MetricsCollector {
    /// Create a new metrics collector with the given configuration.
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsInner {
                latency_samples: VecDeque::with_capacity(config.latency_window_size),
                throughput_samples: VecDeque::with_capacity(config.throughput_window_size),
                network_bytes_sent: 0,
                network_bytes_recv: 0,
                active_requests: 0,
                total_requests: 0,
                total_tokens: 0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
                paged_kv: None,
                paged_decode_fallbacks: 0,
            })),
            config,
        }
    }

    /// Record a single request latency sample.
    pub fn record_latency(&self, duration: Duration) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        if inner.latency_samples.len() >= self.config.latency_window_size {
            inner.latency_samples.pop_front();
        }
        inner.latency_samples.push_back(duration);
    }

    /// Record tokens generated, updating throughput tracking.
    pub fn record_tokens(&self, count: u64) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        let now = Instant::now();
        if inner.throughput_samples.len() >= self.config.throughput_window_size {
            inner.throughput_samples.pop_front();
        }
        inner.throughput_samples.push_back((now, count));
        inner.total_tokens += count;
    }

    /// Record network bytes sent.
    pub fn record_bytes_sent(&self, bytes: u64) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.network_bytes_sent += bytes;
    }

    /// Record network bytes received.
    pub fn record_bytes_recv(&self, bytes: u64) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.network_bytes_recv += bytes;
    }

    /// Increment the active request counter (call when a request starts).
    pub fn request_started(&self) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.active_requests += 1;
        inner.total_requests += 1;
    }

    /// Decrement the active request counter (call when a request completes).
    pub fn request_completed(&self) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.active_requests = inner.active_requests.saturating_sub(1);
    }

    /// Update memory usage snapshot.
    pub fn update_memory(&self, used_bytes: u64, total_bytes: u64) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.memory_used_bytes = used_bytes;
        inner.memory_total_bytes = total_bytes;
    }

    /// Update the latest paged KV allocator snapshot.
    pub fn update_paged_kv(&self, block_size: usize, stats: PagedCacheStats) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.paged_kv = Some(PagedKvMetrics {
            block_size: block_size as u32,
            allocated_blocks: stats.allocated_blocks as u64,
            live_blocks: stats.live_blocks as u64,
            free_blocks: stats.free_blocks as u64,
            bytes_reserved: stats.bytes_reserved as u64,
            bytes_in_use: stats.bytes_in_use as u64,
        });
    }

    /// Clear paged KV metrics when a node is no longer using the paged backend.
    pub fn clear_paged_kv(&self) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.paged_kv = None;
    }

    /// Record that paged decode fell back to dense execution.
    pub fn record_paged_decode_fallback(&self) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.paged_decode_fallbacks += 1;
    }

    /// Compute and return a snapshot of the current node metrics.
    pub fn snapshot(&self) -> NodeMetrics {
        let inner = self.inner.read().expect("metrics lock poisoned");

        let latency = compute_percentiles(&inner.latency_samples);
        let throughput = compute_throughput(&inner.throughput_samples);

        NodeMetrics {
            throughput_tokens_per_sec: throughput,
            latency,
            memory_used_bytes: inner.memory_used_bytes,
            memory_total_bytes: inner.memory_total_bytes,
            network_bytes_sent: inner.network_bytes_sent,
            network_bytes_recv: inner.network_bytes_recv,
            active_requests: inner.active_requests,
            total_requests: inner.total_requests,
            total_tokens: inner.total_tokens,
            paged_kv: inner.paged_kv.clone(),
            paged_decode_fallbacks: inner.paged_decode_fallbacks,
        }
    }

    /// Reset all counters and sliding windows.
    pub fn reset(&self) {
        let mut inner = self.inner.write().expect("metrics lock poisoned");
        inner.latency_samples.clear();
        inner.throughput_samples.clear();
        inner.network_bytes_sent = 0;
        inner.network_bytes_recv = 0;
        inner.active_requests = 0;
        inner.total_requests = 0;
        inner.total_tokens = 0;
        inner.paged_kv = None;
        inner.paged_decode_fallbacks = 0;
    }
}

/// Thread-safe registry mapping node IDs to their latest metrics snapshot.
///
/// Used by the health monitoring system to store metrics from all cluster nodes.
#[derive(Clone)]
pub struct ClusterMetrics {
    inner: Arc<RwLock<HashMap<String, NodeMetrics>>>,
}

impl ClusterMetrics {
    /// Create an empty cluster metrics registry.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update (or insert) the metrics snapshot for a node.
    pub fn update(&self, node_id: &str, metrics: NodeMetrics) {
        let mut inner = self.inner.write().expect("cluster metrics lock poisoned");
        inner.insert(node_id.to_string(), metrics);
    }

    /// Retrieve the latest metrics for a specific node.
    pub fn get(&self, node_id: &str) -> Option<NodeMetrics> {
        let inner = self.inner.read().expect("cluster metrics lock poisoned");
        inner.get(node_id).cloned()
    }

    /// Retrieve a snapshot of all node metrics.
    pub fn all(&self) -> HashMap<String, NodeMetrics> {
        let inner = self.inner.read().expect("cluster metrics lock poisoned");
        inner.clone()
    }

    /// Remove metrics for a node (e.g., when it leaves the cluster).
    pub fn remove(&self, node_id: &str) -> Option<NodeMetrics> {
        let mut inner = self.inner.write().expect("cluster metrics lock poisoned");
        inner.remove(node_id)
    }
}

impl Default for ClusterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute latency percentiles from a sorted copy of the sample window.
fn compute_percentiles(samples: &VecDeque<Duration>) -> LatencyPercentiles {
    if samples.is_empty() {
        return LatencyPercentiles::default();
    }

    let mut sorted: Vec<Duration> = samples.iter().copied().collect();
    sorted.sort();

    let len = sorted.len();
    LatencyPercentiles {
        p50: sorted[len * 50 / 100],
        p95: sorted[(len * 95 / 100).min(len - 1)],
        p99: sorted[(len * 99 / 100).min(len - 1)],
        min: sorted[0],
        max: sorted[len - 1],
    }
}

/// Compute rolling throughput (tokens/sec) from the sample window.
fn compute_throughput(samples: &VecDeque<(Instant, u64)>) -> f64 {
    if samples.len() < 2 {
        return if samples.len() == 1 {
            samples[0].1 as f64
        } else {
            0.0
        };
    }

    let first = samples.front().unwrap();
    let last = samples.back().unwrap();
    let elapsed = last.0.duration_since(first.0);

    if elapsed.is_zero() {
        return 0.0;
    }

    let total_tokens: u64 = samples.iter().map(|(_, count)| count).sum();
    total_tokens as f64 / elapsed.as_secs_f64()
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
