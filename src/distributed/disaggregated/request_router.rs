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

//! Request router and load balancer for disaggregated inference.
//!
//! Orchestrates the full request lifecycle across prefill and decode nodes:
//!
//! 1. Incoming API requests are routed to the best available prefill node
//!    using configurable load balancing strategies (round-robin, least-loaded,
//!    memory-aware, prompt-length-aware).
//! 2. Completed prefill results (KV cache + first token) are routed to the
//!    best available decode node.
//! 3. Request lifecycle is tracked across the prefill -> decode handoff with
//!    phase-level granularity.
//! 4. Backpressure is applied when nodes are at capacity (queue, reject, or
//!    return HTTP 429).
//! 5. Node failures trigger automatic re-routing of affected requests.
//!
//! Used by: disaggregated serving pipeline, server API layer

use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::distributed::backpressure::{BackpressureMonitor, LoadLevel};
use crate::distributed::config::NodeRole;
use crate::distributed::metrics::ClusterMetrics;
use crate::distributed::registry::{NodeRegistry, NodeStatus};
use crate::distributed::request_tracker::RequestId;

// ── Configuration ────────────────────────────────────────────────────

/// Load balancing strategy for routing requests to nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DisaggRoutingStrategy {
    /// Rotate requests evenly across available nodes in stable order.
    RoundRobin,
    /// Route to the node with the fewest active requests.
    LeastLoaded,
    /// Route to the node with the most free memory, avoiding OOM.
    MemoryAware,
    /// Route long prompts to nodes with more compute headroom, short
    /// prompts to nodes that can finish quickly.
    PromptLengthAware,
}

impl fmt::Display for DisaggRoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "round-robin"),
            Self::LeastLoaded => write!(f, "least-loaded"),
            Self::MemoryAware => write!(f, "memory-aware"),
            Self::PromptLengthAware => write!(f, "prompt-length-aware"),
        }
    }
}

/// Configuration for the disaggregated request router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Maximum number of requests waiting to be assigned a prefill node.
    pub prefill_queue_capacity: usize,
    /// Maximum number of completed prefills waiting to be assigned a decode node.
    pub decode_queue_capacity: usize,
    /// Strategy for selecting prefill nodes.
    pub prefill_strategy: DisaggRoutingStrategy,
    /// Strategy for selecting decode nodes.
    pub decode_strategy: DisaggRoutingStrategy,
    /// How long a request may remain in any single phase before timeout.
    pub phase_timeout: Duration,
    /// How long a request may exist in total before being considered stale.
    pub request_timeout: Duration,
    /// Prompt length threshold (in tokens) for the PromptLengthAware strategy.
    /// Prompts longer than this are considered "long" and routed to nodes with
    /// more headroom.
    pub long_prompt_threshold: usize,
    /// Maximum number of tracked requests before automatic purge of terminal
    /// entries. Set to 0 to disable automatic purging.
    pub max_tracked_requests: usize,
    /// Age threshold for automatic purge of terminal requests (Completed/Failed).
    pub auto_purge_age: Duration,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            prefill_queue_capacity: 256,
            decode_queue_capacity: 256,
            prefill_strategy: DisaggRoutingStrategy::LeastLoaded,
            decode_strategy: DisaggRoutingStrategy::MemoryAware,
            phase_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(300),
            long_prompt_threshold: 2048,
            max_tracked_requests: 10_000,
            auto_purge_age: Duration::from_secs(60),
        }
    }
}

// ── Request phases ───────────────────────────────────────────────────

/// Phase of a request in the disaggregated inference pipeline.
///
/// Tracks the fine-grained position of a request as it moves through
/// the prefill -> decode handoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RequestPhase {
    /// Request received but not yet assigned to a prefill node.
    Queued,
    /// Prefill is in progress on the assigned node.
    Prefilling { node_id: String },
    /// Prefill complete; KV cache is being transferred to the decode node.
    TransferringCache { from_node: String, to_node: String },
    /// Decode is in progress on the assigned node.
    Decoding { node_id: String },
    /// Request completed successfully.
    Completed,
    /// Request failed at some phase.
    Failed { reason: String },
}

impl fmt::Display for RequestPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Prefilling { node_id } => write!(f, "prefilling({node_id})"),
            Self::TransferringCache { from_node, to_node } => {
                write!(f, "transferring({from_node}->{to_node})")
            }
            Self::Decoding { node_id } => write!(f, "decoding({node_id})"),
            Self::Completed => write!(f, "completed"),
            Self::Failed { reason } => write!(f, "failed({reason})"),
        }
    }
}

impl RequestPhase {
    /// Returns true if the phase is terminal (Completed or Failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed { .. })
    }
}

// ── Per-request tracking ─────────────────────────────────────────────

/// A request being tracked through the disaggregated pipeline.
#[derive(Debug, Clone)]
pub struct TrackedRequest {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Current phase in the pipeline.
    pub phase: RequestPhase,
    /// Node assigned for prefill (set when routed).
    pub prefill_node: Option<String>,
    /// Node assigned for decode (set when routed after prefill).
    pub decode_node: Option<String>,
    /// When the request was received by the router.
    pub created_at: Instant,
    /// When the current phase began.
    pub phase_started_at: Instant,
    /// Prompt length in tokens (used by prompt-length-aware routing).
    pub prompt_len: usize,
    /// Number of times this request has been re-routed due to failures.
    pub retry_count: u32,
}

impl TrackedRequest {
    /// Create a new tracked request in the Queued phase.
    fn new(request_id: RequestId, prompt_len: usize) -> Self {
        let now = Instant::now();
        Self {
            request_id,
            phase: RequestPhase::Queued,
            prefill_node: None,
            decode_node: None,
            created_at: now,
            phase_started_at: now,
            prompt_len,
            retry_count: 0,
        }
    }

    /// Total elapsed time since the request was created.
    pub fn total_elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Time spent in the current phase.
    pub fn phase_elapsed(&self) -> Duration {
        self.phase_started_at.elapsed()
    }
}

// ── Backpressure action ──────────────────────────────────────────────

/// Action the router takes when backpressure is detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackpressureAction {
    /// Request can proceed normally.
    Accept,
    /// Request should be queued and retried later.
    Queue,
    /// Request should be rejected immediately with the given reason.
    Reject(String),
}

// ── Per-node load snapshot ───────────────────────────────────────────

/// Snapshot of a node's load used for routing decisions.
#[derive(Debug, Clone)]
pub struct NodeLoadInfo {
    /// Node identifier.
    pub node_id: String,
    /// Role (Prefill, Decode, or Hybrid).
    pub role: NodeRole,
    /// Number of active requests on this node.
    pub active_requests: u32,
    /// Memory used in bytes.
    pub memory_used_bytes: u64,
    /// Total memory available in bytes.
    pub memory_total_bytes: u64,
    /// Whether the node is online.
    pub is_online: bool,
    /// Current load level from backpressure monitor.
    pub load_level: Option<LoadLevel>,
}

impl NodeLoadInfo {
    /// Memory utilization as a fraction (0.0 to 1.0).
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            return 0.0;
        }
        self.memory_used_bytes as f64 / self.memory_total_bytes as f64
    }

    /// Free memory in bytes.
    pub fn free_memory(&self) -> u64 {
        self.memory_total_bytes
            .saturating_sub(self.memory_used_bytes)
    }
}

// ── Router metrics ───────────────────────────────────────────────────

/// Aggregated metrics for the request router.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterMetrics {
    /// Total requests received by the router.
    pub total_requests: u64,
    /// Total requests successfully completed.
    pub total_completed: u64,
    /// Total requests that failed.
    pub total_failed: u64,
    /// Requests currently in the prefill queue.
    pub prefill_queue_depth: usize,
    /// Requests currently in the decode queue (waiting for decode node).
    pub decode_queue_depth: usize,
    /// Requests currently being prefilled.
    pub active_prefills: usize,
    /// Requests currently being decoded.
    pub active_decodes: usize,
    /// Total routing decisions made.
    pub routing_decisions: u64,
    /// Total node failure events handled.
    pub failure_events: u64,
    /// Total requests re-routed due to failures.
    pub rerouted_requests: u64,
    /// Total requests rejected due to backpressure.
    pub rejected_requests: u64,
}

// ── Request router ───────────────────────────────────────────────────

/// Request router and load balancer for disaggregated inference.
///
/// Sits between the API server and the prefill/decode node schedulers,
/// managing the full request lifecycle and load distribution.
pub struct RequestRouter {
    config: RouterConfig,
    registry: NodeRegistry,
    cluster_metrics: ClusterMetrics,
    backpressure: BackpressureMonitor,
    /// All tracked requests keyed by request ID string.
    requests: RwLock<HashMap<String, TrackedRequest>>,
    /// Round-robin counter for prefill node selection.
    prefill_rr_counter: AtomicUsize,
    /// Round-robin counter for decode node selection.
    decode_rr_counter: AtomicUsize,
    /// Monotonic counters for metrics.
    total_requests: AtomicU64,
    total_completed: AtomicU64,
    total_failed: AtomicU64,
    routing_decisions: AtomicU64,
    failure_events: AtomicU64,
    rerouted_requests: AtomicU64,
    rejected_requests: AtomicU64,
}

impl fmt::Debug for RequestRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestRouter")
            .field("prefill_strategy", &self.config.prefill_strategy)
            .field("decode_strategy", &self.config.decode_strategy)
            .field(
                "total_requests",
                &self.total_requests.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl RequestRouter {
    /// Create a new request router with the given dependencies.
    pub fn new(
        config: RouterConfig,
        registry: NodeRegistry,
        cluster_metrics: ClusterMetrics,
        backpressure: BackpressureMonitor,
    ) -> Self {
        Self {
            config,
            registry,
            cluster_metrics,
            backpressure,
            requests: RwLock::new(HashMap::new()),
            prefill_rr_counter: AtomicUsize::new(0),
            decode_rr_counter: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            routing_decisions: AtomicU64::new(0),
            failure_events: AtomicU64::new(0),
            rerouted_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
        }
    }

    /// Return a reference to the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    // ── Backpressure check ───────────────────────────────────────────

    /// Evaluate whether the router should accept, queue, or reject a new
    /// request based on current queue depths and node availability.
    pub fn apply_backpressure(&self) -> BackpressureAction {
        let requests = self.requests.read().expect("router lock poisoned");

        // Count requests in each non-terminal phase.
        let queued = requests
            .values()
            .filter(|r| matches!(r.phase, RequestPhase::Queued))
            .count();

        if queued >= self.config.prefill_queue_capacity {
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return BackpressureAction::Reject("prefill queue at capacity".to_string());
        }

        // Check if any prefill node is available.
        let prefill_nodes = self.get_prefill_load_infos();
        let any_available = prefill_nodes
            .iter()
            .any(|n| n.is_online && n.load_level != Some(LoadLevel::Critical));

        if !any_available {
            return BackpressureAction::Queue;
        }

        BackpressureAction::Accept
    }

    // ── Route to prefill ─────────────────────────────────────────────

    /// Accept a new request and route it to the best available prefill node.
    ///
    /// Returns the request ID and the selected prefill node ID.
    /// The request is tracked in the `Prefilling` phase upon success.
    pub fn route_to_prefill(&self, request_id: RequestId, prompt_len: usize) -> Result<String> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Acquire write lock once to check backpressure and insert atomically,
        // avoiding a TOCTOU race where the queue could exceed capacity between
        // the check and the insert.
        let mut requests = self.requests.write().expect("router lock poisoned");

        // Check queue depth under the lock.
        let queued = requests
            .values()
            .filter(|r| matches!(r.phase, RequestPhase::Queued))
            .count();

        if queued >= self.config.prefill_queue_capacity {
            drop(requests);
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("request rejected: prefill queue at capacity");
        }

        // Check if any prefill node is available (does not need the requests lock).
        let prefill_nodes = self.get_prefill_load_infos();
        let any_available = prefill_nodes
            .iter()
            .any(|n| n.is_online && n.load_level != Some(LoadLevel::Critical));

        if !any_available {
            // Insert as Queued; caller must poll or wait.
            let tracked = TrackedRequest::new(request_id.clone(), prompt_len);
            requests.insert(request_id.as_str().to_string(), tracked);
            self.maybe_auto_purge(&mut requests);
            drop(requests);
            anyhow::bail!("all prefill nodes busy; request queued");
        }

        if prefill_nodes.is_empty() {
            drop(requests);
            anyhow::bail!("no prefill nodes registered in the cluster");
        }

        let selected = self.select_prefill_node(&prefill_nodes, prompt_len)?;
        self.routing_decisions.fetch_add(1, Ordering::Relaxed);

        // Track the request.
        let mut tracked = TrackedRequest::new(request_id.clone(), prompt_len);
        tracked.phase = RequestPhase::Prefilling {
            node_id: selected.clone(),
        };
        tracked.phase_started_at = Instant::now();
        tracked.prefill_node = Some(selected.clone());

        requests.insert(request_id.as_str().to_string(), tracked);
        self.maybe_auto_purge(&mut requests);

        Ok(selected)
    }

    /// Select the best prefill node according to the configured strategy.
    fn select_prefill_node(
        &self,
        candidates: &[NodeLoadInfo],
        prompt_len: usize,
    ) -> Result<String> {
        let online: Vec<&NodeLoadInfo> = candidates
            .iter()
            .filter(|n| n.is_online && n.load_level != Some(LoadLevel::Critical))
            .collect();

        if online.is_empty() {
            anyhow::bail!("no online, non-critical prefill nodes available");
        }

        match self.config.prefill_strategy {
            DisaggRoutingStrategy::RoundRobin => {
                let idx = self.prefill_rr_counter.fetch_add(1, Ordering::Relaxed) % online.len();
                Ok(online[idx].node_id.clone())
            }
            DisaggRoutingStrategy::LeastLoaded => {
                let best = online.iter().min_by_key(|n| n.active_requests).unwrap();
                Ok(best.node_id.clone())
            }
            DisaggRoutingStrategy::MemoryAware => {
                let best = online.iter().max_by_key(|n| n.free_memory()).unwrap();
                Ok(best.node_id.clone())
            }
            DisaggRoutingStrategy::PromptLengthAware => {
                if prompt_len >= self.config.long_prompt_threshold {
                    // Long prompt: pick node with most free memory.
                    let best = online.iter().max_by_key(|n| n.free_memory()).unwrap();
                    Ok(best.node_id.clone())
                } else {
                    // Short prompt: pick least-loaded node for fast turnaround.
                    let best = online.iter().min_by_key(|n| n.active_requests).unwrap();
                    Ok(best.node_id.clone())
                }
            }
        }
    }

    // ── Route to decode ──────────────────────────────────────────────

    /// Route a completed prefill to the best available decode node.
    ///
    /// Updates the tracked request phase from `Prefilling` to
    /// `TransferringCache` and returns the selected decode node ID.
    pub fn route_to_decode(&self, request_id: &RequestId) -> Result<String> {
        let decode_nodes = self.get_decode_load_infos();
        if decode_nodes.is_empty() {
            anyhow::bail!("no decode nodes registered in the cluster");
        }

        let selected = self.select_decode_node(&decode_nodes)?;
        self.routing_decisions.fetch_add(1, Ordering::Relaxed);

        // Update the tracked request.
        let mut requests = self.requests.write().expect("router lock poisoned");
        if let Some(tracked) = requests.get_mut(request_id.as_str()) {
            let from_node = tracked
                .prefill_node
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            tracked.phase = RequestPhase::TransferringCache {
                from_node,
                to_node: selected.clone(),
            };
            tracked.phase_started_at = Instant::now();
            tracked.decode_node = Some(selected.clone());
        } else {
            anyhow::bail!("request {request_id} not found in router");
        }

        Ok(selected)
    }

    /// Select the best decode node according to the configured strategy.
    fn select_decode_node(&self, candidates: &[NodeLoadInfo]) -> Result<String> {
        let online: Vec<&NodeLoadInfo> = candidates
            .iter()
            .filter(|n| n.is_online && n.load_level != Some(LoadLevel::Critical))
            .collect();

        if online.is_empty() {
            anyhow::bail!("no online, non-critical decode nodes available");
        }

        match self.config.decode_strategy {
            DisaggRoutingStrategy::RoundRobin => {
                let idx = self.decode_rr_counter.fetch_add(1, Ordering::Relaxed) % online.len();
                Ok(online[idx].node_id.clone())
            }
            DisaggRoutingStrategy::LeastLoaded => {
                let best = online.iter().min_by_key(|n| n.active_requests).unwrap();
                Ok(best.node_id.clone())
            }
            DisaggRoutingStrategy::MemoryAware => {
                let best = online.iter().max_by_key(|n| n.free_memory()).unwrap();
                Ok(best.node_id.clone())
            }
            DisaggRoutingStrategy::PromptLengthAware => {
                // For decode routing, prompt-length-aware behaves like
                // memory-aware since decode memory needs correlate with
                // sequence length (KV cache size).
                let best = online.iter().max_by_key(|n| n.free_memory()).unwrap();
                Ok(best.node_id.clone())
            }
        }
    }

    // ── Phase transitions ────────────────────────────────────────────

    /// Update the phase of a tracked request.
    ///
    /// Returns `true` if the transition was applied, `false` if the request
    /// was not found or is already terminal.
    #[must_use]
    pub fn update_phase(&self, request_id: &RequestId, new_phase: RequestPhase) -> bool {
        let mut requests = self.requests.write().expect("router lock poisoned");
        if let Some(tracked) = requests.get_mut(request_id.as_str()) {
            if tracked.phase.is_terminal() {
                return false;
            }

            // Update terminal counters.
            match &new_phase {
                RequestPhase::Completed => {
                    self.total_completed.fetch_add(1, Ordering::Relaxed);
                }
                RequestPhase::Failed { .. } => {
                    self.total_failed.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }

            tracked.phase = new_phase;
            tracked.phase_started_at = Instant::now();
            true
        } else {
            false
        }
    }

    /// Mark a request as having entered the decoding phase on the given node.
    #[must_use]
    pub fn mark_decoding(&self, request_id: &RequestId, node_id: &str) -> bool {
        self.update_phase(
            request_id,
            RequestPhase::Decoding {
                node_id: node_id.to_string(),
            },
        )
    }

    /// Mark a request as completed.
    #[must_use]
    pub fn mark_completed(&self, request_id: &RequestId) -> bool {
        self.update_phase(request_id, RequestPhase::Completed)
    }

    /// Mark a request as failed.
    #[must_use]
    pub fn mark_failed(&self, request_id: &RequestId, reason: &str) -> bool {
        self.update_phase(
            request_id,
            RequestPhase::Failed {
                reason: reason.to_string(),
            },
        )
    }

    // ── Node failure handling ────────────────────────────────────────

    /// Handle a node failure by re-routing all affected in-flight requests.
    ///
    /// Re-routed requests are distributed round-robin across available
    /// candidates to avoid thundering-herd on a single alternative node.
    ///
    /// Requests in `TransferringCache` are re-queued for prefill (not jumped
    /// to Decoding) because the KV cache transfer was interrupted and the
    /// cache state on the target is incomplete.
    ///
    /// Returns the number of requests that were re-routed, and the number
    /// that could not be re-routed (marked as failed).
    pub fn handle_node_failure(&self, failed_node_id: &str) -> (usize, usize) {
        self.failure_events.fetch_add(1, Ordering::Relaxed);

        let mut requests = self.requests.write().expect("router lock poisoned");
        let mut rerouted = 0usize;
        let mut failed = 0usize;

        // Collect request IDs affected by this node failure.
        let mut affected: Vec<String> = requests
            .iter()
            .filter(|(_, r)| {
                !r.phase.is_terminal()
                    && match &r.phase {
                        RequestPhase::Prefilling { node_id } => node_id == failed_node_id,
                        RequestPhase::TransferringCache { from_node, to_node } => {
                            from_node == failed_node_id || to_node == failed_node_id
                        }
                        RequestPhase::Decoding { node_id } => node_id == failed_node_id,
                        _ => false,
                    }
            })
            .map(|(k, _)| k.clone())
            .collect();
        // `requests` is a HashMap, so this comes out in an instance-specific
        // order. The loop below hands candidates out round-robin in exactly
        // this sequence, so sorting is what makes the request-to-node pairing
        // reproducible; sorting the candidate list alone is not enough.
        affected.sort();

        // Get available alternative nodes for re-routing.
        let prefill_candidates: Vec<String> = self
            .registry
            .nodes_with_role(NodeRole::Prefill)
            .into_iter()
            .chain(self.registry.nodes_with_role(NodeRole::Hybrid))
            .filter(|n| {
                n.status == NodeStatus::Online
                    && n.config.id != failed_node_id
                    && !self.backpressure.is_critical(&n.config.id)
            })
            .map(|n| n.config.id)
            .collect();

        let decode_candidates: Vec<String> = self
            .registry
            .nodes_with_role(NodeRole::Decode)
            .into_iter()
            .chain(self.registry.nodes_with_role(NodeRole::Hybrid))
            .filter(|n| {
                n.status == NodeStatus::Online
                    && n.config.id != failed_node_id
                    && !self.backpressure.is_critical(&n.config.id)
            })
            .map(|n| n.config.id)
            .collect();

        // Round-robin counters to distribute re-routed requests across candidates.
        let mut prefill_rr = 0usize;
        let mut decode_rr = 0usize;

        for key in affected {
            if let Some(tracked) = requests.get_mut(&key) {
                match &tracked.phase {
                    RequestPhase::Prefilling { .. } => {
                        // Re-route to another prefill node (round-robin) or re-queue.
                        if !prefill_candidates.is_empty() {
                            let alt = &prefill_candidates[prefill_rr % prefill_candidates.len()];
                            prefill_rr += 1;
                            tracked.phase = RequestPhase::Prefilling {
                                node_id: alt.clone(),
                            };
                            tracked.prefill_node = Some(alt.clone());
                            tracked.phase_started_at = Instant::now();
                            tracked.retry_count += 1;
                            rerouted += 1;
                        } else {
                            // No alternative: re-queue for later.
                            tracked.phase = RequestPhase::Queued;
                            tracked.phase_started_at = Instant::now();
                            tracked.retry_count += 1;
                            rerouted += 1;
                        }
                    }
                    RequestPhase::TransferringCache { .. } => {
                        // KV cache transfer was interrupted -- the cache is
                        // incomplete on the target. Re-queue for prefill so
                        // the prompt is re-processed from scratch.
                        if !prefill_candidates.is_empty() {
                            let alt = &prefill_candidates[prefill_rr % prefill_candidates.len()];
                            prefill_rr += 1;
                            tracked.phase = RequestPhase::Prefilling {
                                node_id: alt.clone(),
                            };
                            tracked.prefill_node = Some(alt.clone());
                            tracked.decode_node = None;
                            tracked.phase_started_at = Instant::now();
                            tracked.retry_count += 1;
                            rerouted += 1;
                        } else {
                            // No prefill candidates either: re-queue.
                            tracked.phase = RequestPhase::Queued;
                            tracked.prefill_node = None;
                            tracked.decode_node = None;
                            tracked.phase_started_at = Instant::now();
                            tracked.retry_count += 1;
                            rerouted += 1;
                        }
                    }
                    RequestPhase::Decoding { .. } => {
                        // Re-route to another decode node (round-robin).
                        if !decode_candidates.is_empty() {
                            let alt = &decode_candidates[decode_rr % decode_candidates.len()];
                            decode_rr += 1;
                            tracked.phase = RequestPhase::Decoding {
                                node_id: alt.clone(),
                            };
                            tracked.decode_node = Some(alt.clone());
                            tracked.phase_started_at = Instant::now();
                            tracked.retry_count += 1;
                            rerouted += 1;
                        } else {
                            tracked.phase = RequestPhase::Failed {
                                reason: format!(
                                    "node {failed_node_id} failed; no alternative decode node"
                                ),
                            };
                            failed += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        self.rerouted_requests
            .fetch_add(rerouted as u64, Ordering::Relaxed);
        self.total_failed
            .fetch_add(failed as u64, Ordering::Relaxed);

        (rerouted, failed)
    }

    // ── Query ────────────────────────────────────────────────────────

    /// Get the current phase of a tracked request.
    pub fn get_phase(&self, request_id: &RequestId) -> Option<RequestPhase> {
        let requests = self.requests.read().expect("router lock poisoned");
        requests.get(request_id.as_str()).map(|r| r.phase.clone())
    }

    /// Get a clone of a tracked request.
    pub fn get_tracked_request(&self, request_id: &RequestId) -> Option<TrackedRequest> {
        let requests = self.requests.read().expect("router lock poisoned");
        requests.get(request_id.as_str()).cloned()
    }

    /// Return the number of requests in each phase (non-terminal only).
    pub fn phase_counts(&self) -> HashMap<&'static str, usize> {
        let requests = self.requests.read().expect("router lock poisoned");
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for r in requests.values() {
            if !r.phase.is_terminal() {
                let key: &'static str = match &r.phase {
                    RequestPhase::Queued => "queued",
                    RequestPhase::Prefilling { .. } => "prefilling",
                    RequestPhase::TransferringCache { .. } => "transferring",
                    RequestPhase::Decoding { .. } => "decoding",
                    _ => continue,
                };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Collect requests that have timed out in their current phase.
    ///
    /// Returns the request IDs of timed-out requests (not yet marked failed).
    pub fn collect_timed_out(&self) -> Vec<RequestId> {
        let requests = self.requests.read().expect("router lock poisoned");
        let phase_timeout = self.config.phase_timeout;
        let request_timeout = self.config.request_timeout;

        requests
            .values()
            .filter(|r| {
                !r.phase.is_terminal()
                    && (r.phase_elapsed() > phase_timeout || r.total_elapsed() > request_timeout)
            })
            .map(|r| r.request_id.clone())
            .collect()
    }

    /// Remove completed and failed requests older than the given age.
    ///
    /// Returns the number of removed requests.
    pub fn purge_terminal(&self, max_age: Duration) -> usize {
        let mut requests = self.requests.write().expect("router lock poisoned");
        let before = requests.len();
        requests.retain(|_, r| {
            if r.phase.is_terminal() {
                r.total_elapsed() < max_age
            } else {
                true
            }
        });
        before - requests.len()
    }

    /// Return a snapshot of router metrics.
    pub fn metrics(&self) -> RouterMetrics {
        let counts = self.phase_counts();

        RouterMetrics {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            prefill_queue_depth: counts.get("queued").copied().unwrap_or(0),
            decode_queue_depth: counts.get("transferring").copied().unwrap_or(0),
            active_prefills: counts.get("prefilling").copied().unwrap_or(0),
            active_decodes: counts.get("decoding").copied().unwrap_or(0),
            routing_decisions: self.routing_decisions.load(Ordering::Relaxed),
            failure_events: self.failure_events.load(Ordering::Relaxed),
            rerouted_requests: self.rerouted_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Build load info snapshots for all prefill-capable nodes.
    fn get_prefill_load_infos(&self) -> Vec<NodeLoadInfo> {
        self.get_load_infos_for_roles(&[NodeRole::Prefill, NodeRole::Hybrid])
    }

    /// Build load info snapshots for all decode-capable nodes.
    fn get_decode_load_infos(&self) -> Vec<NodeLoadInfo> {
        self.get_load_infos_for_roles(&[NodeRole::Decode, NodeRole::Hybrid])
    }

    /// Automatically purge terminal requests when the map exceeds the
    /// configured capacity. Called while the write lock is already held.
    fn maybe_auto_purge(&self, requests: &mut HashMap<String, TrackedRequest>) {
        let max = self.config.max_tracked_requests;
        if max == 0 || requests.len() <= max {
            return;
        }
        let purge_age = self.config.auto_purge_age;
        requests.retain(|_, r| {
            if r.phase.is_terminal() {
                r.total_elapsed() < purge_age
            } else {
                true
            }
        });
    }

    /// Build load info snapshots for nodes matching any of the given roles.
    fn get_load_infos_for_roles(&self, roles: &[NodeRole]) -> Vec<NodeLoadInfo> {
        let all_nodes = self.registry.all_nodes();
        let all_metrics = self.cluster_metrics.all();

        all_nodes
            .into_iter()
            .filter(|n| roles.contains(&n.config.role))
            .map(|n| {
                let metrics = all_metrics.get(&n.config.id);
                let load_level = self.backpressure.get_load_level(&n.config.id);

                NodeLoadInfo {
                    node_id: n.config.id.clone(),
                    role: n.config.role,
                    active_requests: metrics.map(|m| m.active_requests).unwrap_or(0),
                    memory_used_bytes: metrics.map(|m| m.memory_used_bytes).unwrap_or(0),
                    memory_total_bytes: metrics.map(|m| m.memory_total_bytes).unwrap_or(0),
                    is_online: n.status == NodeStatus::Online,
                    load_level,
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "request_router_tests.rs"]
mod tests;
