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

//! Thread-safe node registry for tracking cluster membership.
//!
//! The registry is the runtime source of truth for which nodes are currently
//! participating in the cluster, their roles, and their capabilities. It
//! supports dynamic updates so nodes can join and leave without a restart.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use super::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};

/// Health status of a registered node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is reachable and ready to serve.
    Online,
    /// Node has not responded to recent health checks.
    Unreachable,
    /// Node is in the process of joining (not yet ready).
    Joining,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::Joining => write!(f, "joining"),
        }
    }
}

/// Runtime entry for a registered node, combining static config with dynamic
/// state such as health status.
#[derive(Debug, Clone)]
pub struct RegisteredNode {
    /// Static configuration for this node.
    pub config: NodeConfig,
    /// Current health status.
    pub status: NodeStatus,
}

/// Thread-safe registry of nodes in the cluster.
///
/// Designed for concurrent reads from request handlers and infrequent writes
/// from the discovery / health-check subsystem.
#[derive(Clone)]
pub struct NodeRegistry {
    inner: Arc<RwLock<RegistryInner>>,
}

struct RegistryInner {
    /// Cluster-level metadata.
    meta: ClusterMeta,
    /// Node ID -> registered node.
    nodes: HashMap<String, RegisteredNode>,
    /// ID of the local node (the node running this process).
    local_node_id: String,
}

impl NodeRegistry {
    /// Create a new registry seeded from a [`ClusterConfig`].
    ///
    /// All nodes start in [`NodeStatus::Joining`] except the local node,
    /// which is set to [`NodeStatus::Online`].
    pub fn from_config(config: &ClusterConfig, local_node_id: &str) -> Self {
        let mut nodes = HashMap::with_capacity(config.nodes.len());
        for node_cfg in &config.nodes {
            let status = if node_cfg.id == local_node_id {
                NodeStatus::Online
            } else {
                NodeStatus::Joining
            };
            nodes.insert(
                node_cfg.id.clone(),
                RegisteredNode {
                    config: node_cfg.clone(),
                    status,
                },
            );
        }
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                meta: config.cluster.clone(),
                nodes,
                local_node_id: local_node_id.to_string(),
            })),
        }
    }

    /// Return the local node's ID.
    pub fn local_node_id(&self) -> String {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .local_node_id
            .clone()
    }

    /// Return the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .len()
    }

    /// Look up a node by ID.
    pub fn get_node(&self, id: &str) -> Option<RegisteredNode> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .get(id)
            .cloned()
    }

    /// Return a snapshot of all registered nodes, sorted by node ID.
    ///
    /// The sort is part of the contract, not a convenience. Callers select a
    /// node by position (round-robin) or by first / last extreme
    /// (`min_by_key` / `max_by_key`, which tie-break on input order), and an
    /// idle cluster ties on every load metric. Since `nodes` is a `HashMap`
    /// whose iteration order is seeded per instance, an unsorted snapshot
    /// makes those selections differ between processes for identical cluster
    /// state.
    pub fn all_nodes(&self) -> Vec<RegisteredNode> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut nodes: Vec<RegisteredNode> = inner.nodes.values().cloned().collect();
        nodes.sort_by(|a, b| a.config.id.cmp(&b.config.id));
        nodes
    }

    /// Return nodes filtered by role, sorted by node ID.
    ///
    /// Sorted for the same reason as [`Self::all_nodes`]: the failover path
    /// walks this list round-robin by position when re-routing the requests
    /// that a failed node was holding.
    pub fn nodes_with_role(&self, role: NodeRole) -> Vec<RegisteredNode> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut nodes: Vec<RegisteredNode> = inner
            .nodes
            .values()
            .filter(|n| n.config.role == role)
            .cloned()
            .collect();
        nodes.sort_by(|a, b| a.config.id.cmp(&b.config.id));
        nodes
    }

    /// Return the registered node that owns the 2D `(pp_stage, tp_rank)`
    /// intersection, if any.
    ///
    /// Used by: 2D parallelism transport routing (intra-stage TP collectives
    /// and inter-stage PP activation handoff both look up peers through this
    /// method).
    pub fn find_pp_tp_node(&self, stage: u32, rank: u32) -> Option<RegisteredNode> {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .nodes
            .values()
            .find(|n| {
                n.config.role.is_pp_tp()
                    && n.config.stage == Some(stage)
                    && n.config.rank == Some(rank)
            })
            .cloned()
    }

    /// Return all PPTP peers on the same pipeline stage, sorted by rank.
    ///
    /// These are the peers that participate in intra-stage all-reduce / TP
    /// collective communication.
    ///
    /// Used by: 2D parallelism transport routing (TP collective path).
    pub fn nodes_at_stage(&self, stage: u32) -> Vec<RegisteredNode> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut nodes: Vec<RegisteredNode> = inner
            .nodes
            .values()
            .filter(|n| n.config.role.is_pp_tp() && n.config.stage == Some(stage))
            .cloned()
            .collect();
        nodes.sort_by_key(|n| n.config.rank.unwrap_or(u32::MAX));
        nodes
    }

    /// Return all PPTP peers at the same TP rank, sorted by stage.
    ///
    /// These are the peers that participate in inter-stage PP activation
    /// handoff between corresponding ranks.
    ///
    /// Used by: 2D parallelism transport routing (PP activation path).
    pub fn nodes_at_rank(&self, rank: u32) -> Vec<RegisteredNode> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut nodes: Vec<RegisteredNode> = inner
            .nodes
            .values()
            .filter(|n| n.config.role.is_pp_tp() && n.config.rank == Some(rank))
            .cloned()
            .collect();
        nodes.sort_by_key(|n| n.config.stage.unwrap_or(u32::MAX));
        nodes
    }

    /// Return the `(stage, rank)` of the local node, if this cluster uses
    /// 2D parallelism and the local node carries the PPTP role.
    pub fn local_pp_tp_coords(&self) -> Option<(u32, u32)> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let local = inner.nodes.get(&inner.local_node_id)?;
        if local.config.role.is_pp_tp() {
            Some((local.config.stage?, local.config.rank?))
        } else {
            None
        }
    }

    /// Return the cluster metadata.
    pub fn cluster_meta(&self) -> ClusterMeta {
        self.inner
            .read()
            .expect("registry lock poisoned")
            .meta
            .clone()
    }

    /// Update the status of a node. Returns `true` if the node was found.
    pub fn set_node_status(&self, id: &str, status: NodeStatus) -> bool {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        if let Some(node) = inner.nodes.get_mut(id) {
            node.status = status;
            true
        } else {
            false
        }
    }

    /// Register a new node or update an existing one.
    pub fn upsert_node(&self, config: NodeConfig, status: NodeStatus) {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        inner
            .nodes
            .insert(config.id.clone(), RegisteredNode { config, status });
    }

    /// Remove a node from the registry. Returns the removed node if present.
    pub fn remove_node(&self, id: &str) -> Option<RegisteredNode> {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        inner.nodes.remove(id)
    }

    /// Return the address of each peer (all nodes except the local one),
    /// ordered by the owning node's ID.
    pub fn peer_addresses(&self) -> Vec<SocketAddr> {
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut peers: Vec<&RegisteredNode> = inner
            .nodes
            .values()
            .filter(|n| n.config.id != inner.local_node_id)
            .collect();
        peers.sort_by(|a, b| a.config.id.cmp(&b.config.id));
        peers.into_iter().map(|n| n.config.address).collect()
    }

    /// Return a human-readable cluster topology string.
    pub fn topology_summary(&self) -> String {
        use std::fmt::Write;
        let inner = self.inner.read().expect("registry lock poisoned");
        let mut out = String::new();
        let _ = writeln!(out, "Cluster: {}", inner.meta.name);
        let _ = writeln!(
            out,
            "  TP size: {}, PP size: {}",
            inner.meta.tensor_parallel_size, inner.meta.pipeline_parallel_size
        );
        let _ = writeln!(out, "  Local node: {}", inner.local_node_id);
        let _ = writeln!(out, "  Nodes ({}):", inner.nodes.len());
        // Listed in ID order so two snapshots of the same cluster diff cleanly.
        let mut nodes: Vec<&RegisteredNode> = inner.nodes.values().collect();
        nodes.sort_by(|a, b| a.config.id.cmp(&b.config.id));
        for node in nodes {
            let local_tag = if node.config.id == inner.local_node_id {
                " (local)"
            } else {
                ""
            };
            let coords = match (node.config.stage, node.config.rank) {
                (Some(s), Some(r)) => format!(" (stage={s}, rank={r})"),
                (Some(s), None) => format!(" (stage={s})"),
                (None, Some(r)) => format!(" (rank={r})"),
                (None, None) => String::new(),
            };
            let _ = writeln!(
                out,
                "    - {} @ {} [{}]{coords} status={}{local_tag}",
                node.config.id, node.config.address, node.config.role, node.status
            );
        }
        out
    }

    /// Update the resource capabilities for a node. Returns `true` if found.
    pub fn update_resources(&self, id: &str, resources: NodeResources) -> bool {
        let mut inner = self.inner.write().expect("registry lock poisoned");
        if let Some(node) = inner.nodes.get_mut(id) {
            node.config.resources = resources;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
