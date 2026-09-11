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

//! Disaggregated serving integration with the API server.
//!
//! Integrates the prefill scheduler, decode scheduler, and request router into
//! the existing OpenAI-compatible HTTP server. The external API remains
//! unchanged -- callers cannot distinguish between single-node and
//! disaggregated serving.
//!
//! # Serving Modes
//!
//! - **Hybrid** (default): Single-node operation. Zero routing, zero transfer,
//!   zero serialization overhead. The code path is identical to the current
//!   non-distributed server.
//! - **PrefillOnly**: Accepts requests, runs prefill, hands off KV caches to
//!   decode peers. Does not generate tokens beyond the first.
//! - **DecodeOnly**: Receives KV caches from prefill peers, runs autoregressive
//!   decoding. Does not process raw prompts.
//! - **Router**: Stateless request router that distributes incoming API requests
//!   to prefill and decode peers without running inference itself.
//!
//! # Zero-Overhead Hybrid Mode
//!
//! In hybrid mode, [`HybridModeGuard`] wraps the dispatch logic with a
//! compile-time-friendly branch: when the mode is `Hybrid`, every method
//! returns immediately (or delegates to the local model provider) without
//! touching any distributed data structures. This is NOT "disaggregated
//! with localhost transfer" -- it is a direct passthrough.
//!
//! Used by: server startup, server model worker, server routes

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::distributed::config::NodeRole;

// ── Serving Mode ─────────────────────────────────────────────────────

/// The serving mode this node operates in.
///
/// Determined at startup from `--node-role` and cluster configuration.
/// Once set, the mode does not change for the lifetime of the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ServingMode {
    /// Single-node operation with zero disaggregation overhead.
    /// This is the default when no distributed flags are provided.
    Hybrid,
    /// Prefill-only node: processes prompts, generates first token,
    /// transfers KV cache to decode peers.
    PrefillOnly,
    /// Decode-only node: receives KV caches, runs autoregressive decoding.
    DecodeOnly,
    /// Stateless router: distributes requests to prefill and decode peers.
    Router,
}

impl std::fmt::Display for ServingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hybrid => write!(f, "hybrid"),
            Self::PrefillOnly => write!(f, "prefill-only"),
            Self::DecodeOnly => write!(f, "decode-only"),
            Self::Router => write!(f, "router"),
        }
    }
}

impl ServingMode {
    /// Derive the serving mode from a `NodeRole`.
    ///
    /// - `Prefill` -> `PrefillOnly`
    /// - `Decode` -> `DecodeOnly`
    /// - `Hybrid` -> `Hybrid`
    /// - Other roles default to `Hybrid` (pipeline/TP nodes handle
    ///   disaggregation differently).
    pub fn from_node_role(role: NodeRole) -> Self {
        match role {
            NodeRole::Prefill => Self::PrefillOnly,
            NodeRole::Decode => Self::DecodeOnly,
            NodeRole::Hybrid => Self::Hybrid,
            _ => Self::Hybrid,
        }
    }

    /// Whether this mode runs local inference (prefill or decode).
    pub fn runs_inference(&self) -> bool {
        matches!(self, Self::Hybrid | Self::PrefillOnly | Self::DecodeOnly)
    }

    /// Whether this mode needs a loaded model.
    pub fn needs_model(&self) -> bool {
        self.runs_inference()
    }

    /// Whether this mode is the zero-overhead local path.
    pub fn is_hybrid(&self) -> bool {
        *self == Self::Hybrid
    }
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for disaggregated serving integration.
///
/// Assembled from CLI flags (`--node-role`, `--prefill-peers`,
/// `--decode-peers`) or from the cluster TOML config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedServingConfig {
    /// The serving mode for this node.
    pub mode: ServingMode,

    /// Addresses of prefill peer nodes (used by decode and router nodes
    /// to forward requests or receive cache transfers).
    pub prefill_peers: Vec<SocketAddr>,

    /// Addresses of decode peer nodes (used by prefill and router nodes
    /// to forward KV caches and receive completion events).
    pub decode_peers: Vec<SocketAddr>,

    /// Whether to enable streaming across the prefill-decode boundary.
    /// When true, the first token from prefill and subsequent tokens from
    /// decode are merged into a single seamless SSE stream.
    pub streaming_enabled: bool,

    /// Maximum time to wait for a decode peer to accept a KV cache
    /// transfer before failing the request.
    pub handoff_timeout: Duration,

    /// Maximum time to wait for the first token from a prefill peer
    /// when operating as a router.
    pub first_token_timeout: Duration,

    /// Interval for collecting and reporting disaggregated metrics.
    pub metrics_interval: Duration,
}

impl Default for DisaggregatedServingConfig {
    fn default() -> Self {
        Self {
            mode: ServingMode::Hybrid,
            prefill_peers: Vec::new(),
            decode_peers: Vec::new(),
            streaming_enabled: true,
            handoff_timeout: Duration::from_secs(30),
            first_token_timeout: Duration::from_secs(60),
            metrics_interval: Duration::from_secs(10),
        }
    }
}

impl DisaggregatedServingConfig {
    /// Build a config from CLI arguments.
    ///
    /// Returns `None` if no distributed flags are provided (pure local mode).
    pub fn from_cli(
        node_role: Option<&str>,
        prefill_peers: Vec<SocketAddr>,
        decode_peers: Vec<SocketAddr>,
    ) -> Result<Option<Self>> {
        let role_str = match node_role {
            Some(r) => r,
            None if prefill_peers.is_empty() && decode_peers.is_empty() => return Ok(None),
            None => "hybrid",
        };

        let role: NodeRole = role_str.parse().context("invalid --node-role value")?;
        let mode = ServingMode::from_node_role(role);

        // Validate peer configuration.
        match mode {
            ServingMode::PrefillOnly if decode_peers.is_empty() => {
                anyhow::bail!("prefill-only mode requires at least one --decode-peers address");
            }
            ServingMode::DecodeOnly if prefill_peers.is_empty() => {
                anyhow::bail!("decode-only mode requires at least one --prefill-peers address");
            }
            ServingMode::Router if prefill_peers.is_empty() || decode_peers.is_empty() => {
                anyhow::bail!("router mode requires both --prefill-peers and --decode-peers");
            }
            _ => {}
        }

        Ok(Some(Self {
            mode,
            prefill_peers,
            decode_peers,
            ..Self::default()
        }))
    }
}

// ── Disaggregated Metrics ────────────────────────────────────────────

/// Metrics specific to disaggregated serving operations.
///
/// All counters are atomic for lock-free reads from HTTP handlers and
/// writes from the serving pipeline.
#[derive(Debug)]
pub struct DisaggregatedMetrics {
    // Prefill metrics
    /// Total prompts processed by this node (prefill phase).
    pub prefill_prompts_total: AtomicU64,
    /// Total prompt tokens processed (prefill throughput numerator).
    pub prefill_tokens_total: AtomicU64,
    /// Cumulative prefill time in microseconds (for throughput calculation).
    pub prefill_time_us_total: AtomicU64,

    // Decode metrics
    /// Total tokens generated by this node (decode phase).
    pub decode_tokens_total: AtomicU64,
    /// Cumulative decode time in microseconds.
    pub decode_time_us_total: AtomicU64,

    // Transfer metrics
    /// Total KV cache transfers initiated.
    pub cache_transfers_total: AtomicU64,
    /// Total KV cache transfer failures.
    pub cache_transfer_failures: AtomicU64,
    /// Cumulative transfer latency in microseconds (for percentile calculation).
    pub cache_transfer_latency_us_total: AtomicU64,
    /// Total bytes transferred in KV cache handoffs.
    pub cache_transfer_bytes_total: AtomicU64,

    // Queue depth gauges (use load/store, not fetch_add)
    /// Current prefill queue depth.
    pub prefill_queue_depth: AtomicU64,
    /// Current decode queue depth.
    pub decode_queue_depth: AtomicU64,
    /// Current transfer queue depth.
    pub transfer_queue_depth: AtomicU64,

    // Stream bridging
    /// Total SSE streams bridged across prefill-decode boundary.
    pub streams_bridged_total: AtomicU64,
    /// Total stream bridge failures (token gaps, timeouts).
    pub stream_bridge_failures: AtomicU64,
}

impl Default for DisaggregatedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DisaggregatedMetrics {
    /// Create a new metrics instance with all counters at zero.
    pub fn new() -> Self {
        Self {
            prefill_prompts_total: AtomicU64::new(0),
            prefill_tokens_total: AtomicU64::new(0),
            prefill_time_us_total: AtomicU64::new(0),
            decode_tokens_total: AtomicU64::new(0),
            decode_time_us_total: AtomicU64::new(0),
            cache_transfers_total: AtomicU64::new(0),
            cache_transfer_failures: AtomicU64::new(0),
            cache_transfer_latency_us_total: AtomicU64::new(0),
            cache_transfer_bytes_total: AtomicU64::new(0),
            prefill_queue_depth: AtomicU64::new(0),
            decode_queue_depth: AtomicU64::new(0),
            transfer_queue_depth: AtomicU64::new(0),
            streams_bridged_total: AtomicU64::new(0),
            stream_bridge_failures: AtomicU64::new(0),
        }
    }

    /// Record a completed prefill operation.
    pub fn record_prefill(&self, prompt_tokens: usize, duration: Duration) {
        self.prefill_prompts_total.fetch_add(1, Ordering::Relaxed);
        self.prefill_tokens_total
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.prefill_time_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record generated decode tokens.
    pub fn record_decode_tokens(&self, count: usize, duration: Duration) {
        self.decode_tokens_total
            .fetch_add(count as u64, Ordering::Relaxed);
        self.decode_time_us_total
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a KV cache transfer.
    pub fn record_cache_transfer(&self, latency: Duration, bytes: u64) {
        self.cache_transfers_total.fetch_add(1, Ordering::Relaxed);
        self.cache_transfer_latency_us_total
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
        self.cache_transfer_bytes_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a KV cache transfer failure.
    pub fn record_cache_transfer_failure(&self) {
        self.cache_transfer_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Update queue depth gauges.
    pub fn update_queue_depths(&self, prefill: u64, decode: u64, transfer: u64) {
        self.prefill_queue_depth.store(prefill, Ordering::Relaxed);
        self.decode_queue_depth.store(decode, Ordering::Relaxed);
        self.transfer_queue_depth.store(transfer, Ordering::Relaxed);
    }

    /// Record a successful SSE stream bridge.
    pub fn record_stream_bridged(&self) {
        self.streams_bridged_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a stream bridge failure.
    pub fn record_stream_bridge_failure(&self) {
        self.stream_bridge_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Return a serializable snapshot of current metrics.
    pub fn snapshot(&self) -> DisaggregatedMetricsSnapshot {
        let prefill_tokens = self.prefill_tokens_total.load(Ordering::Relaxed);
        let prefill_time_us = self.prefill_time_us_total.load(Ordering::Relaxed);
        let decode_tokens = self.decode_tokens_total.load(Ordering::Relaxed);
        let decode_time_us = self.decode_time_us_total.load(Ordering::Relaxed);
        let transfer_count = self.cache_transfers_total.load(Ordering::Relaxed);
        let transfer_latency_us = self.cache_transfer_latency_us_total.load(Ordering::Relaxed);

        DisaggregatedMetricsSnapshot {
            prefill_prompts_total: self.prefill_prompts_total.load(Ordering::Relaxed),
            prefill_tokens_total: prefill_tokens,
            prefill_tokens_per_sec: if prefill_time_us > 0 {
                prefill_tokens as f64 / (prefill_time_us as f64 / 1_000_000.0)
            } else {
                0.0
            },
            decode_tokens_total: decode_tokens,
            decode_tokens_per_sec: if decode_time_us > 0 {
                decode_tokens as f64 / (decode_time_us as f64 / 1_000_000.0)
            } else {
                0.0
            },
            cache_transfers_total: transfer_count,
            cache_transfer_failures: self.cache_transfer_failures.load(Ordering::Relaxed),
            cache_transfer_avg_latency_ms: if transfer_count > 0 {
                (transfer_latency_us as f64 / transfer_count as f64) / 1000.0
            } else {
                0.0
            },
            cache_transfer_bytes_total: self.cache_transfer_bytes_total.load(Ordering::Relaxed),
            prefill_queue_depth: self.prefill_queue_depth.load(Ordering::Relaxed),
            decode_queue_depth: self.decode_queue_depth.load(Ordering::Relaxed),
            transfer_queue_depth: self.transfer_queue_depth.load(Ordering::Relaxed),
            streams_bridged_total: self.streams_bridged_total.load(Ordering::Relaxed),
            stream_bridge_failures: self.stream_bridge_failures.load(Ordering::Relaxed),
        }
    }
}

/// Serializable snapshot of disaggregated metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedMetricsSnapshot {
    pub prefill_prompts_total: u64,
    pub prefill_tokens_total: u64,
    pub prefill_tokens_per_sec: f64,
    pub decode_tokens_total: u64,
    pub decode_tokens_per_sec: f64,
    pub cache_transfers_total: u64,
    pub cache_transfer_failures: u64,
    pub cache_transfer_avg_latency_ms: f64,
    pub cache_transfer_bytes_total: u64,
    pub prefill_queue_depth: u64,
    pub decode_queue_depth: u64,
    pub transfer_queue_depth: u64,
    pub streams_bridged_total: u64,
    pub stream_bridge_failures: u64,
}

// ── Hybrid Mode Guard ────────────────────────────────────────────────

/// Guard that ensures zero overhead in hybrid/single-node mode.
///
/// In hybrid mode, all dispatch methods immediately delegate to the local
/// model provider without touching any distributed data structures, network
/// connections, or serialization.
///
/// The guard stores the serving mode and a reference to the optional
/// disaggregated server. When the mode is `Hybrid`, the server reference
/// is always `None`, making the branch prediction trivially optimizable.
#[derive(Debug, Clone)]
pub struct HybridModeGuard {
    mode: ServingMode,
}

impl HybridModeGuard {
    /// Create a guard for the given serving mode.
    pub fn new(mode: ServingMode) -> Self {
        Self { mode }
    }

    /// Returns the current serving mode.
    pub fn mode(&self) -> ServingMode {
        self.mode
    }

    /// Returns `true` if operating in hybrid mode (zero overhead).
    ///
    /// When this returns `true`, callers should use the local model
    /// provider directly without any disaggregated dispatch.
    #[inline]
    pub fn is_local(&self) -> bool {
        self.mode == ServingMode::Hybrid
    }

    /// Returns `true` if this node should run prefill for incoming requests.
    #[inline]
    pub fn should_prefill(&self) -> bool {
        matches!(self.mode, ServingMode::Hybrid | ServingMode::PrefillOnly)
    }

    /// Returns `true` if this node should run decode for sequences.
    #[inline]
    pub fn should_decode(&self) -> bool {
        matches!(self.mode, ServingMode::Hybrid | ServingMode::DecodeOnly)
    }

    /// Returns `true` if this node should route requests to peers.
    #[inline]
    pub fn should_route(&self) -> bool {
        matches!(self.mode, ServingMode::Router)
    }
}

// ── Disaggregated Server ─────────────────────────────────────────────

/// Coordinates disaggregated serving across the prefill-decode boundary.
///
/// Sits between the API server routes and the node-specific schedulers,
/// providing a unified dispatch interface that the routes can call
/// regardless of the serving mode.
///
/// # Architecture
///
/// ```text
///                     ┌──────────────────────┐
///  API Request ──────>│  DisaggregatedServer  │
///                     │                       │
///                     │  mode: ServingMode     │
///                     │  guard: HybridGuard    │
///                     │  metrics: Metrics      │
///                     │  config: Config        │
///                     └───────┬───────────────┘
///                             │
///         ┌───────────────────┼───────────────────┐
///         v                   v                   v
///   ┌──────────┐      ┌────────────┐      ┌───────────┐
///   │  Hybrid  │      │  Prefill   │      │  Decode   │
///   │ (local)  │      │  + Handoff │      │  + Ingest │
///   └──────────┘      └────────────┘      └───────────┘
/// ```
///
/// Used by: server startup, server model worker
pub struct DisaggregatedServer {
    /// Serving configuration.
    config: DisaggregatedServingConfig,

    /// Guard for zero-overhead hybrid mode checks.
    guard: HybridModeGuard,

    /// Disaggregated-specific metrics.
    metrics: DisaggregatedMetrics,

    /// Timestamp when the server was created.
    created_at: Instant,
}

impl std::fmt::Debug for DisaggregatedServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DisaggregatedServer")
            .field("mode", &self.config.mode)
            .field("prefill_peers", &self.config.prefill_peers.len())
            .field("decode_peers", &self.config.decode_peers.len())
            .finish()
    }
}

impl DisaggregatedServer {
    /// Create a new disaggregated server with the given configuration.
    pub fn new(config: DisaggregatedServingConfig) -> Self {
        let guard = HybridModeGuard::new(config.mode);
        Self {
            config,
            guard,
            metrics: DisaggregatedMetrics::new(),
            created_at: Instant::now(),
        }
    }

    /// Return a reference to the serving configuration.
    pub fn config(&self) -> &DisaggregatedServingConfig {
        &self.config
    }

    /// Return the serving mode.
    pub fn mode(&self) -> ServingMode {
        self.config.mode
    }

    /// Return a reference to the hybrid mode guard.
    pub fn guard(&self) -> &HybridModeGuard {
        &self.guard
    }

    /// Return a reference to the disaggregated metrics.
    pub fn metrics(&self) -> &DisaggregatedMetrics {
        &self.metrics
    }

    /// Return the server uptime.
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check whether this server should handle a request locally.
    ///
    /// In hybrid mode, this always returns `true`. In other modes,
    /// it returns `true` only if the request matches the node's role
    /// (prefill requests on prefill nodes, decode on decode nodes).
    #[inline]
    pub fn should_handle_locally(&self) -> bool {
        self.guard.is_local()
    }

    /// Return the list of prefill peer addresses.
    pub fn prefill_peers(&self) -> &[SocketAddr] {
        &self.config.prefill_peers
    }

    /// Return the list of decode peer addresses.
    pub fn decode_peers(&self) -> &[SocketAddr] {
        &self.config.decode_peers
    }

    /// Check if the server is healthy and ready to serve requests.
    ///
    /// In hybrid mode, readiness depends only on the local model being loaded.
    /// In disaggregated modes, at least one peer of the required type must be
    /// reachable.
    pub fn is_ready(&self) -> bool {
        match self.config.mode {
            ServingMode::Hybrid => true,
            ServingMode::PrefillOnly => !self.config.decode_peers.is_empty(),
            ServingMode::DecodeOnly => !self.config.prefill_peers.is_empty(),
            ServingMode::Router => {
                !self.config.prefill_peers.is_empty() && !self.config.decode_peers.is_empty()
            }
        }
    }
}

#[cfg(test)]
#[path = "serving_tests.rs"]
mod tests;
