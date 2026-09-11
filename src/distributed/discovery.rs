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

//! Static peer discovery for cluster bootstrap.
//!
//! This module provides the initial peer discovery mechanism based on the
//! static peer list from the cluster configuration. Future versions may add
//! mDNS or gossip-based discovery.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;

use super::registry::{NodeRegistry, NodeStatus};

/// Result of a single peer health check.
#[derive(Debug)]
pub struct PeerProbeResult {
    /// Address that was probed.
    pub address: SocketAddr,
    /// Node ID if the peer identified itself.
    pub node_id: String,
    /// Whether the probe succeeded.
    pub reachable: bool,
}

/// Probe all peers in the registry by attempting a TCP connection.
///
/// This is a simple reachability check -- it does not perform an application-level
/// health check. Each peer that accepts a TCP connection within the timeout is
/// marked [`NodeStatus::Online`]; others are marked [`NodeStatus::Unreachable`].
///
/// All probes run concurrently so the total wall-clock time is bounded by the
/// timeout rather than scaling linearly with the number of peers.
///
/// Returns a summary of probe results.
pub async fn probe_peers(registry: &NodeRegistry, timeout: Duration) -> Vec<PeerProbeResult> {
    let peers = registry.all_nodes();
    let local_id = registry.local_node_id();

    // Spawn all probes concurrently.
    let mut handles = Vec::with_capacity(peers.len());
    for node in &peers {
        if node.config.id == local_id {
            continue;
        }
        let addr = node.config.address;
        let node_id = node.config.id.clone();
        handles.push(tokio::spawn(async move {
            let reachable = tokio::time::timeout(timeout, tokio::net::TcpStream::connect(addr))
                .await
                .map(|r| r.is_ok())
                .unwrap_or(false);
            PeerProbeResult {
                address: addr,
                node_id,
                reachable,
            }
        }));
    }

    // Collect results and update the registry.
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        // If a spawned task panics we treat the peer as unreachable.
        let probe = match handle.await {
            Ok(p) => p,
            Err(_) => continue,
        };
        let status = if probe.reachable {
            NodeStatus::Online
        } else {
            NodeStatus::Unreachable
        };
        registry.set_node_status(&probe.node_id, status);
        results.push(probe);
    }

    results
}

/// Log the cluster topology after discovery.
pub fn log_cluster_topology(registry: &NodeRegistry) {
    let summary = registry.topology_summary();
    for line in summary.lines() {
        tracing::info!("{line}");
    }
}

/// Initialize distributed mode: create a registry from the config, probe
/// peers, and log the topology.
///
/// Returns the fully initialized [`NodeRegistry`].
pub async fn initialize_distributed(
    config: &super::config::ClusterConfig,
    local_node_id: &str,
    probe_timeout: Duration,
) -> Result<NodeRegistry> {
    let registry = NodeRegistry::from_config(config, local_node_id);

    tracing::info!(
        "Distributed mode: node '{}' joining cluster '{}'",
        local_node_id,
        config.cluster.name
    );

    let results = probe_peers(&registry, probe_timeout).await;
    let reachable = results.iter().filter(|r| r.reachable).count();
    let total = results.len();
    tracing::info!("Peer discovery: {reachable}/{total} peers reachable");

    log_cluster_topology(&registry);

    Ok(registry)
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
