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

//! Routing strategies for distributed request scheduling.
//!
//! Provides a trait-based extensibility point for routing decisions. Built-in
//! strategies include role-based routing, load-balanced routing, and
//! round-robin. Pipeline Parallel (PP), Disaggregated Inference (DI), and
//! Tensor Parallel (TP) strategies can be plugged in by implementing the
//! [`RoutingStrategy`] trait.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use super::config::NodeRole;
use super::metrics::NodeMetrics;
use super::registry::NodeStatus;

/// Information about a candidate node available for routing.
#[derive(Debug, Clone)]
pub struct NodeCandidate {
    /// Unique node identifier.
    pub node_id: String,
    /// Node role in the cluster.
    pub role: NodeRole,
    /// Current health status.
    pub status: NodeStatus,
    /// Latest metrics snapshot (if available).
    pub metrics: Option<NodeMetrics>,
    /// Pipeline stage index (for PP routing).
    pub stage: Option<u32>,
    /// Tensor-parallel rank (for TP routing).
    pub rank: Option<u32>,
}

/// A request to be routed to a node.
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// The request identifier.
    pub request_id: String,
    /// Preferred node role for this request (e.g., Prefill for DI).
    pub preferred_role: Option<NodeRole>,
    /// Preferred pipeline stage (for PP routing).
    pub preferred_stage: Option<u32>,
    /// Preferred tensor-parallel rank (for 2D PP×TP routing).
    pub preferred_rank: Option<u32>,
    /// Traffic class for 2D routing: intra-stage TP collective vs inter-stage
    /// PP activation transfer. Ignored for 1D topologies.
    pub traffic_class: TrafficClass,
    /// Optional hint for sticky routing (route to same node as before).
    pub affinity_node: Option<String>,
}

/// Traffic classification used by 2D (PP × TP) transport routing.
///
/// The connection pool is shared across classes, but the routing layer picks a
/// different peer set for each class:
///
/// - [`Self::TpCollective`] — intra-stage all-reduce / reduce-scatter /
///   all-gather traffic; peer set is all PPTP nodes on the same pipeline
///   stage.
/// - [`Self::PpActivation`] — inter-stage activation / boundary tensor
///   handoff; peer set is the adjacent-stage node at the same TP rank.
/// - [`Self::Any`] — legacy 1D topologies that do not need the distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TrafficClass {
    /// Unspecified (legacy / 1D topologies).
    #[default]
    Any,
    /// Intra-stage TP collective traffic (all-reduce family).
    TpCollective,
    /// Inter-stage PP activation handoff traffic.
    PpActivation,
}

impl std::fmt::Display for TrafficClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "any"),
            Self::TpCollective => write!(f, "tp_collective"),
            Self::PpActivation => write!(f, "pp_activation"),
        }
    }
}

impl RoutingRequest {
    /// Build a routing request with only the required fields populated.
    ///
    /// All 2D-specific fields default to their neutral values, preserving the
    /// existing 1D routing behavior for legacy callers.
    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            preferred_role: None,
            preferred_stage: None,
            preferred_rank: None,
            traffic_class: TrafficClass::Any,
            affinity_node: None,
        }
    }
}

/// Result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// ID of the selected target node.
    pub target_node: String,
    /// Name of the strategy that made this decision.
    pub strategy: String,
    /// Optional reason for the selection (useful for debugging).
    pub reason: Option<String>,
}

/// Trait for pluggable routing strategies.
///
/// Implementations decide which node should handle a given request based on
/// the request properties and available candidates. This is the main
/// extensibility point for PP, DI, and TP custom routing logic.
///
/// Used by: Scheduler
pub trait RoutingStrategy: Send + Sync {
    /// Select a target node for the given request from the candidates.
    ///
    /// Returns the node ID of the selected candidate, or an error if no
    /// suitable candidate is found.
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision>;

    /// Return the human-readable name of this strategy.
    fn strategy_name(&self) -> &str;
}

/// Routes requests to nodes based on their role.
///
/// If the request has a `preferred_role`, only nodes with that role (or Hybrid)
/// are considered. Among matching nodes, the first online node is selected.
///
/// Used by: Scheduler (default for DI routing)
pub struct RoleBasedRouter;

impl RoutingStrategy for RoleBasedRouter {
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision> {
        // If affinity is set, try that node first.
        if let Some(ref affinity) = request.affinity_node
            && let Some(c) = candidates
                .iter()
                .find(|c| c.node_id == *affinity && c.status == NodeStatus::Online)
        {
            return Ok(RoutingDecision {
                target_node: c.node_id.clone(),
                strategy: self.strategy_name().to_string(),
                reason: Some("affinity match".to_string()),
            });
        }

        let online: Vec<&NodeCandidate> = candidates
            .iter()
            .filter(|c| c.status == NodeStatus::Online)
            .collect();

        if online.is_empty() {
            anyhow::bail!("no online nodes available for routing");
        }

        // Filter by preferred role if specified.
        if let Some(role) = request.preferred_role {
            let matching: Vec<&&NodeCandidate> = online
                .iter()
                .filter(|c| c.role == role || c.role == NodeRole::Hybrid)
                .collect();

            if let Some(node) = matching.first() {
                return Ok(RoutingDecision {
                    target_node: node.node_id.clone(),
                    strategy: self.strategy_name().to_string(),
                    reason: Some(format!("role match: {role}")),
                });
            }
        }

        // Fallback: first online node.
        Ok(RoutingDecision {
            target_node: online[0].node_id.clone(),
            strategy: self.strategy_name().to_string(),
            reason: Some("fallback to first online".to_string()),
        })
    }

    fn strategy_name(&self) -> &str {
        "role-based"
    }
}

/// Routes requests to the least-loaded online node.
///
/// Load is measured by `active_requests` from the node metrics. Nodes without
/// metrics are treated as having zero load (optimistic assumption for new nodes).
///
/// Used by: Scheduler (default load balancing)
pub struct LoadBalancedRouter;

impl RoutingStrategy for LoadBalancedRouter {
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision> {
        let mut online: Vec<&NodeCandidate> = candidates
            .iter()
            .filter(|c| c.status == NodeStatus::Online)
            .collect();

        if online.is_empty() {
            anyhow::bail!("no online nodes available for routing");
        }

        // Filter by preferred role if specified.
        if let Some(role) = request.preferred_role {
            let matching: Vec<&NodeCandidate> = online
                .iter()
                .filter(|c| c.role == role || c.role == NodeRole::Hybrid)
                .copied()
                .collect();
            if !matching.is_empty() {
                online = matching;
            }
        }

        // Sort by active requests (ascending), with memory usage as tiebreaker.
        online.sort_by(|a, b| {
            let load_a = a.metrics.as_ref().map(|m| m.active_requests).unwrap_or(0);
            let load_b = b.metrics.as_ref().map(|m| m.active_requests).unwrap_or(0);
            load_a.cmp(&load_b).then_with(|| {
                let mem_a = a.metrics.as_ref().map(|m| m.memory_used_bytes).unwrap_or(0);
                let mem_b = b.metrics.as_ref().map(|m| m.memory_used_bytes).unwrap_or(0);
                mem_a.cmp(&mem_b)
            })
        });

        let selected = online[0];
        let active = selected
            .metrics
            .as_ref()
            .map(|m| m.active_requests)
            .unwrap_or(0);

        Ok(RoutingDecision {
            target_node: selected.node_id.clone(),
            strategy: self.strategy_name().to_string(),
            reason: Some(format!("least loaded (active_requests={active})")),
        })
    }

    fn strategy_name(&self) -> &str {
        "load-balanced"
    }
}

/// Routes requests in round-robin order among online nodes.
///
/// Maintains an atomic counter to distribute requests evenly. If a preferred
/// role is specified, only matching nodes participate in the rotation.
///
/// Used by: Scheduler (simple even distribution)
pub struct RoundRobinRouter {
    counter: AtomicUsize,
}

impl RoundRobinRouter {
    /// Create a new round-robin router.
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for RoundRobinRouter {
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision> {
        let mut online: Vec<&NodeCandidate> = candidates
            .iter()
            .filter(|c| c.status == NodeStatus::Online)
            .collect();

        if online.is_empty() {
            anyhow::bail!("no online nodes available for routing");
        }

        // Filter by preferred role if specified.
        if let Some(role) = request.preferred_role {
            let matching: Vec<&NodeCandidate> = online
                .iter()
                .filter(|c| c.role == role || c.role == NodeRole::Hybrid)
                .copied()
                .collect();
            if !matching.is_empty() {
                online = matching;
            }
        }

        // Sort by node_id for deterministic ordering.
        online.sort_by(|a, b| a.node_id.cmp(&b.node_id));

        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % online.len();
        let selected = online[idx];

        Ok(RoutingDecision {
            target_node: selected.node_id.clone(),
            strategy: self.strategy_name().to_string(),
            reason: Some(format!("round-robin index={idx}")),
        })
    }

    fn strategy_name(&self) -> &str {
        "round-robin"
    }
}

/// Routes requests to a specific pipeline stage.
///
/// Selects nodes with `PipelineStage` role matching the requested stage index.
/// Falls back to load-based selection if multiple nodes serve the same stage.
///
/// Used by: Pipeline Parallel scheduling
pub struct PipelineStageRouter;

impl RoutingStrategy for PipelineStageRouter {
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision> {
        let target_stage = request
            .preferred_stage
            .ok_or_else(|| anyhow::anyhow!("pipeline stage router requires preferred_stage"))?;

        let matching: Vec<&NodeCandidate> = candidates
            .iter()
            .filter(|c| {
                c.status == NodeStatus::Online
                    && c.role == NodeRole::PipelineStage
                    && c.stage == Some(target_stage)
            })
            .collect();

        if matching.is_empty() {
            anyhow::bail!("no online node found for pipeline stage {target_stage}");
        }

        // If multiple nodes serve the same stage, pick the least loaded.
        let selected = matching
            .iter()
            .min_by_key(|c| c.metrics.as_ref().map(|m| m.active_requests).unwrap_or(0))
            .unwrap();

        Ok(RoutingDecision {
            target_node: selected.node_id.clone(),
            strategy: self.strategy_name().to_string(),
            reason: Some(format!("pipeline stage {target_stage}")),
        })
    }

    fn strategy_name(&self) -> &str {
        "pipeline-stage"
    }
}

/// Routes requests on the 2D `(pp_stage, tp_rank)` mesh.
///
/// The strategy requires both `preferred_stage` and `preferred_rank` on the
/// routing request. The [`TrafficClass`] on the request disambiguates which
/// transport handler a caller is targeting, but does not change the selected
/// peer — it is purely advisory so the connection pool user can pick the
/// appropriate downstream callback. Callers that need a different peer set
/// (e.g., broadcast to every rank on a stage for all-reduce) should use the
/// registry's `nodes_at_stage` / `nodes_at_rank` helpers directly.
///
/// Used by: 2D PP × TP request routing.
pub struct PipelineTensorParallelRouter;

impl RoutingStrategy for PipelineTensorParallelRouter {
    fn select_node(
        &self,
        request: &RoutingRequest,
        candidates: &[NodeCandidate],
    ) -> Result<RoutingDecision> {
        let target_stage = request
            .preferred_stage
            .ok_or_else(|| anyhow::anyhow!("2D router requires preferred_stage"))?;
        let target_rank = request
            .preferred_rank
            .ok_or_else(|| anyhow::anyhow!("2D router requires preferred_rank"))?;

        let selected = candidates
            .iter()
            .find(|c| {
                c.status == NodeStatus::Online
                    && c.role.is_pp_tp()
                    && c.stage == Some(target_stage)
                    && c.rank == Some(target_rank)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no online 2D node found for (stage={target_stage}, rank={target_rank})"
                )
            })?;

        Ok(RoutingDecision {
            target_node: selected.node_id.clone(),
            strategy: self.strategy_name().to_string(),
            reason: Some(format!(
                "pp-tp stage={target_stage} rank={target_rank} class={}",
                request.traffic_class
            )),
        })
    }

    fn strategy_name(&self) -> &str {
        "pipeline-tensor-parallel"
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
