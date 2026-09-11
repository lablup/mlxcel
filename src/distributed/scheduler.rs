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

//! Distributed scheduler coordinator for multi-node inference.
//!
//! The scheduler routes incoming requests to appropriate nodes based on their
//! role and current load. It supports two coordination modes:
//!
//! - **Centralized**: A single coordinator node makes all routing decisions.
//! - **Distributed**: Nodes coordinate via peer-to-peer consensus (each node
//!   can make local routing decisions based on shared cluster state).
//!
//! The scheduler integrates with:
//! - [`RequestTracker`] for end-to-end lifecycle tracking
//! - [`BackpressureMonitor`] for load-aware routing
//! - [`HandoffQueueManager`] for cross-node request handoffs
//! - Pluggable [`RoutingStrategy`] implementations for PP/DI/TP extensibility

use std::fmt;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::backpressure::{BackpressureConfig, BackpressureMonitor, BackpressurePolicy};
use super::config::NodeRole;
use super::handoff_queue::{HandoffItem, HandoffQueueConfig, HandoffQueueManager};
use super::metrics::ClusterMetrics;
use super::registry::{NodeRegistry, NodeStatus};
use super::request_tracker::{RequestId, RequestState, RequestTracker, RequestTrackerConfig};
use super::routing::{
    LoadBalancedRouter, NodeCandidate, RoleBasedRouter, RoutingDecision, RoutingRequest,
    RoutingStrategy,
};

/// Coordination mode for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CoordinationMode {
    /// Single coordinator node makes all routing decisions.
    /// Simpler but has a single point of failure.
    Centralized,
    /// Nodes make local routing decisions using shared cluster state.
    /// More resilient but requires consistent state propagation.
    Distributed,
}

impl fmt::Display for CoordinationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Centralized => write!(f, "centralized"),
            Self::Distributed => write!(f, "distributed"),
        }
    }
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Coordination mode.
    pub mode: CoordinationMode,
    /// Backpressure configuration.
    pub backpressure: BackpressureConfig,
    /// Default handoff queue configuration.
    pub handoff_queue: HandoffQueueConfig,
    /// Request tracker configuration.
    pub request_tracker: RequestTrackerConfig,
    /// Whether to skip nodes under backpressure during routing.
    pub skip_pressured_nodes: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            mode: CoordinationMode::Centralized,
            backpressure: BackpressureConfig::default(),
            handoff_queue: HandoffQueueConfig::default(),
            request_tracker: RequestTrackerConfig::default(),
            skip_pressured_nodes: true,
        }
    }
}

/// The distributed scheduler coordinator.
///
/// Routes requests to nodes, tracks request lifecycles, manages backpressure
/// signaling, and handles cross-node request handoffs.
///
/// # Extensibility
///
/// The routing strategy can be swapped at runtime to support different
/// distributed inference patterns:
/// - Pipeline Parallel (PP): stage-based routing
/// - Disaggregated Inference (DI): prefill/decode role-based routing
/// - Tensor Parallel (TP): synchronized execution triggers
///
/// Custom strategies implement the [`RoutingStrategy`] trait.
pub struct Scheduler {
    config: SchedulerConfig,
    registry: NodeRegistry,
    metrics: ClusterMetrics,
    tracker: RequestTracker,
    backpressure: BackpressureMonitor,
    queues: HandoffQueueManager,
    strategy: Arc<dyn RoutingStrategy>,
}

impl fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scheduler")
            .field("mode", &self.config.mode)
            .field("strategy", &self.strategy.strategy_name())
            .field("active_requests", &self.tracker.active_count())
            .finish()
    }
}

impl Scheduler {
    /// Create a new scheduler with the given configuration, registry, and metrics.
    ///
    /// Uses [`LoadBalancedRouter`] as the default routing strategy.
    pub fn new(config: SchedulerConfig, registry: NodeRegistry, metrics: ClusterMetrics) -> Self {
        let tracker = RequestTracker::new(config.request_tracker.clone());
        let backpressure = BackpressureMonitor::new(config.backpressure.clone());
        let queues = HandoffQueueManager::new(config.handoff_queue.clone());

        Self {
            config,
            registry,
            metrics,
            tracker,
            backpressure,
            queues,
            strategy: Arc::new(LoadBalancedRouter),
        }
    }

    /// Create a scheduler with a custom routing strategy.
    pub fn with_strategy(
        config: SchedulerConfig,
        registry: NodeRegistry,
        metrics: ClusterMetrics,
        strategy: Arc<dyn RoutingStrategy>,
    ) -> Self {
        let tracker = RequestTracker::new(config.request_tracker.clone());
        let backpressure = BackpressureMonitor::new(config.backpressure.clone());
        let queues = HandoffQueueManager::new(config.handoff_queue.clone());

        Self {
            config,
            registry,
            metrics,
            tracker,
            backpressure,
            queues,
            strategy,
        }
    }

    /// Replace the current routing strategy.
    pub fn set_strategy(&mut self, strategy: Arc<dyn RoutingStrategy>) {
        self.strategy = strategy;
    }

    /// Return the current coordination mode.
    pub fn mode(&self) -> CoordinationMode {
        self.config.mode
    }

    /// Return the name of the current routing strategy.
    pub fn strategy_name(&self) -> &str {
        self.strategy.strategy_name()
    }

    // ── Request submission and routing ──────────────────────────────

    /// Submit a new request and route it to an appropriate node.
    ///
    /// Returns the assigned request ID and the routing decision. The request
    /// lifecycle is tracked from submission through completion.
    pub fn submit_request(
        &self,
        preferred_role: Option<NodeRole>,
        preferred_stage: Option<u32>,
        affinity_node: Option<String>,
    ) -> Result<(RequestId, RoutingDecision)> {
        // Assign a request ID and begin tracking.
        let request_id = self.tracker.submit();

        // Transition to Routing (always succeeds for a newly submitted request).
        let _ = self.tracker.transition(&request_id, RequestState::Routing);

        // Build routing request.
        let routing_request = RoutingRequest {
            request_id: request_id.as_str().to_string(),
            preferred_role,
            preferred_stage,
            preferred_rank: None,
            traffic_class: super::routing::TrafficClass::Any,
            affinity_node,
        };

        // Gather candidates from the registry.
        let candidates = self.build_candidates();

        // Route the request.
        match self.strategy.select_node(&routing_request, &candidates) {
            Ok(decision) => {
                // Check backpressure on the selected node.
                if self.backpressure.is_critical(&decision.target_node) {
                    match self.backpressure.overflow_policy() {
                        BackpressurePolicy::Redirect => {
                            // Try to find an alternative non-pressured node.
                            if let Some(alt) =
                                self.find_alternative(&decision.target_node, &candidates)
                            {
                                let _ = self.tracker.transition(
                                    &request_id,
                                    RequestState::Processing {
                                        node_id: alt.target_node.clone(),
                                    },
                                );
                                return Ok((request_id, alt));
                            }
                            // No alternative available; use original despite pressure.
                        }
                        BackpressurePolicy::Drop => {
                            let _ = self.tracker.transition(
                                &request_id,
                                RequestState::Failed {
                                    reason: format!(
                                        "target node {} is at critical load",
                                        decision.target_node
                                    ),
                                },
                            );
                            anyhow::bail!(
                                "request dropped: target node {} is at critical load",
                                decision.target_node
                            );
                        }
                        BackpressurePolicy::Block => {
                            // In sync context, proceed anyway (async blocking
                            // would be handled at a higher level).
                        }
                    }
                }

                // Transition to Processing.
                let _ = self.tracker.transition(
                    &request_id,
                    RequestState::Processing {
                        node_id: decision.target_node.clone(),
                    },
                );

                Ok((request_id, decision))
            }
            Err(e) => {
                let _ = self.tracker.transition(
                    &request_id,
                    RequestState::Failed {
                        reason: e.to_string(),
                    },
                );
                Err(e)
            }
        }
    }

    /// Initiate a handoff of a request from one node to another.
    ///
    /// Enqueues the request in the appropriate handoff queue and updates the
    /// lifecycle tracker.
    pub fn initiate_handoff(
        &self,
        request_id: &RequestId,
        from_node: &str,
        to_node: &str,
        payload: Vec<u8>,
    ) -> Result<()> {
        // Update lifecycle.
        if !self.tracker.transition(
            request_id,
            RequestState::Handoff {
                from_node: from_node.to_string(),
                to_node: to_node.to_string(),
            },
        ) {
            anyhow::bail!("cannot initiate handoff: request not found or already terminal");
        }

        // Enqueue in the handoff queue for this node pair.
        let queue_name = format!("{from_node}->{to_node}");
        let queue = self.queues.get_or_create(&queue_name);

        let item = HandoffItem {
            request_id: request_id.clone(),
            from_node: from_node.to_string(),
            to_node: to_node.to_string(),
            payload,
            enqueued_at: std::time::Instant::now(),
        };

        let result = queue.enqueue(item);
        match result {
            super::handoff_queue::EnqueueResult::Success
            | super::handoff_queue::EnqueueResult::DroppedOldest => Ok(()),
            super::handoff_queue::EnqueueResult::Rejected => {
                let _ = self.tracker.transition(
                    request_id,
                    RequestState::Failed {
                        reason: format!("handoff queue {queue_name} is full"),
                    },
                );
                anyhow::bail!("handoff rejected: queue {queue_name} is full");
            }
        }
    }

    /// Complete the handoff: mark the request as processing on the destination node.
    #[must_use]
    pub fn complete_handoff(&self, request_id: &RequestId, destination_node: &str) -> bool {
        self.tracker.transition(
            request_id,
            RequestState::Processing {
                node_id: destination_node.to_string(),
            },
        )
    }

    /// Mark a request as completed.
    #[must_use]
    pub fn complete_request(&self, request_id: &RequestId) -> bool {
        self.tracker.transition(request_id, RequestState::Completed)
    }

    /// Mark a request as failed.
    #[must_use]
    pub fn fail_request(&self, request_id: &RequestId, reason: &str) -> bool {
        self.tracker.transition(
            request_id,
            RequestState::Failed {
                reason: reason.to_string(),
            },
        )
    }

    // ── Backpressure ────────────────────────────────────────────────

    /// Update the backpressure state for a node based on its current metrics.
    pub fn update_node_load(&self, node_id: &str, active_requests: u32, memory_utilization: f64) {
        self.backpressure
            .update_from_metrics(node_id, active_requests, memory_utilization);
    }

    /// Check whether a specific node is under backpressure.
    pub fn is_node_pressured(&self, node_id: &str) -> bool {
        self.backpressure.is_under_pressure(node_id)
    }

    // ── Query ───────────────────────────────────────────────────────

    /// Get the current state of a request.
    pub fn get_request_state(&self, request_id: &RequestId) -> Option<RequestState> {
        self.tracker.get_state(request_id)
    }

    /// Return the number of active (non-terminal) requests.
    pub fn active_request_count(&self) -> usize {
        self.tracker.active_count()
    }

    /// Return the total number of tracked requests.
    pub fn tracked_request_count(&self) -> usize {
        self.tracker.tracked_count()
    }

    /// Return a reference to the request tracker.
    pub fn tracker(&self) -> &RequestTracker {
        &self.tracker
    }

    /// Return a reference to the backpressure monitor.
    pub fn backpressure(&self) -> &BackpressureMonitor {
        &self.backpressure
    }

    /// Return a reference to the handoff queue manager.
    pub fn queues(&self) -> &HandoffQueueManager {
        &self.queues
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Build the list of candidate nodes from the registry and metrics.
    fn build_candidates(&self) -> Vec<NodeCandidate> {
        let nodes = self.registry.all_nodes();
        let all_metrics = self.metrics.all();

        nodes
            .into_iter()
            .filter(|n| {
                // In centralized mode, skip nodes under critical backpressure
                // if configured.
                if self.config.skip_pressured_nodes && self.backpressure.is_critical(&n.config.id) {
                    return false;
                }
                true
            })
            .map(|n| {
                let metrics = all_metrics.get(&n.config.id).cloned();
                NodeCandidate {
                    node_id: n.config.id.clone(),
                    role: n.config.role,
                    status: n.status,
                    metrics,
                    stage: n.config.stage,
                    rank: n.config.rank,
                }
            })
            .collect()
    }

    /// Try to find an alternative node that is not under critical backpressure.
    fn find_alternative(
        &self,
        exclude_node: &str,
        candidates: &[NodeCandidate],
    ) -> Option<RoutingDecision> {
        let filtered: Vec<NodeCandidate> = candidates
            .iter()
            .filter(|c| {
                c.node_id != exclude_node
                    && c.status == NodeStatus::Online
                    && !self.backpressure.is_critical(&c.node_id)
            })
            .cloned()
            .collect();

        if filtered.is_empty() {
            return None;
        }

        // Use a simple role-based fallback for the alternative.
        let fallback = RoleBasedRouter;
        let dummy_request = RoutingRequest::new(String::new());

        fallback.select_node(&dummy_request, &filtered).ok()
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
