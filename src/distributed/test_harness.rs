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

//! Test harness for simulating multi-node distributed clusters.
//!
//! [`TestCluster`] creates an in-process cluster of simulated nodes, each
//! with its own [`MockTransport`], [`NodeRegistry`], [`HeartbeatService`],
//! and [`FailureDetector`]. It provides helper methods for:
//!
//! - Adding and removing nodes dynamically
//! - Injecting network partitions and failures
//! - Waiting for specific cluster states
//! - Running performance benchmarks
//!
//! All operations are fully async and designed for `#[tokio::test]`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};

use super::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
use super::heartbeat::{HeartbeatConfig, HeartbeatService};
use super::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use super::registry::{NodeRegistry, NodeStatus};
use super::transport::Transport;

/// A simulated node within a [`TestCluster`].
pub struct TestNode {
    /// Node identifier.
    pub id: String,
    /// Virtual address.
    pub address: String,
    /// Node role.
    pub role: NodeRole,
    /// Mock transport for this node.
    pub transport: Arc<MockTransport>,
    /// Node registry (shared view of cluster state).
    pub registry: NodeRegistry,
    /// Heartbeat service (optional, started by `start_heartbeats`).
    pub heartbeat: Option<HeartbeatService>,
}

/// Configuration for creating a test cluster.
#[derive(Debug, Clone)]
pub struct TestClusterConfig {
    /// Simulated latency for all mock transports.
    pub transport_latency: Duration,
    /// Heartbeat interval (if heartbeats are enabled).
    pub heartbeat_interval: Duration,
    /// Failure detection threshold (number of missed heartbeats).
    pub failure_threshold: u32,
    /// Failure detection check interval.
    pub failure_check_interval: Duration,
}

impl Default for TestClusterConfig {
    fn default() -> Self {
        Self {
            transport_latency: Duration::ZERO,
            heartbeat_interval: Duration::from_millis(50),
            failure_threshold: 3,
            failure_check_interval: Duration::from_millis(25),
        }
    }
}

/// Test harness that manages a simulated multi-node cluster.
///
/// Provides a high-level API for setting up test scenarios without
/// manually wiring transports, registries, and heartbeat services.
///
/// # Example
///
/// ```no_run
/// use mlxcel::distributed::test_harness::TestCluster;
/// use mlxcel::distributed::config::NodeRole;
///
/// #[tokio::test]
/// async fn test_node_join() {
///     let mut cluster = TestCluster::new(Default::default());
///     cluster.add_node("node-0", NodeRole::Prefill).await;
///     cluster.add_node("node-1", NodeRole::Decode).await;
///     assert_eq!(cluster.node_count(), 2);
/// }
/// ```
pub struct TestCluster {
    /// Shared mock router for all nodes.
    router: MockRouter,
    /// Active nodes in the cluster.
    nodes: HashMap<String, TestNode>,
    /// Cluster configuration.
    config: TestClusterConfig,
    /// Auto-incrementing port counter for unique addresses.
    next_port: u16,
}

impl TestCluster {
    /// Create a new empty test cluster.
    pub fn new(config: TestClusterConfig) -> Self {
        Self {
            router: MockRouter::new(),
            nodes: HashMap::new(),
            config,
            next_port: 10000,
        }
    }

    /// Add a new node to the cluster with the given ID and role.
    ///
    /// Creates a mock transport, registers it with the router, and updates
    /// all existing node registries to include the new node.
    pub async fn add_node(&mut self, id: &str, role: NodeRole) -> &TestNode {
        let port = self.next_port;
        self.next_port += 1;
        let address = format!("127.0.0.1:{port}");

        // Create transport.
        let transport_config = MockTransportConfig {
            latency: self.config.transport_latency,
            ..Default::default()
        };
        let transport = Arc::new(
            MockTransport::new(address.clone(), self.router.clone(), transport_config).await,
        );

        // Build a ClusterConfig snapshot including all existing nodes + this one.
        let mut all_node_configs: Vec<NodeConfig> = self
            .nodes
            .values()
            .map(|n| NodeConfig {
                id: n.id.clone(),
                address: n.address.parse().unwrap(),
                role: n.role,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            })
            .collect();

        let new_node_config = NodeConfig {
            id: id.to_string(),
            address: address.parse().unwrap(),
            role,
            stage: None,
            rank: None,
            resources: NodeResources::default(),
        };
        all_node_configs.push(new_node_config.clone());

        let cluster_config = ClusterConfig {
            cluster: ClusterMeta::default(),
            nodes: all_node_configs.clone(),
        };

        // Create registry for the new node.
        let registry = NodeRegistry::from_config(&cluster_config, id);
        // Mark all nodes as Online.
        for node_cfg in &all_node_configs {
            registry.set_node_status(&node_cfg.id, NodeStatus::Online);
        }

        // Update existing node registries to include the new node.
        for existing_node in self.nodes.values() {
            existing_node
                .registry
                .upsert_node(new_node_config.clone(), NodeStatus::Online);
        }

        let test_node = TestNode {
            id: id.to_string(),
            address,
            role,
            transport,
            registry,
            heartbeat: None,
        };

        self.nodes.insert(id.to_string(), test_node);
        self.nodes.get(id).unwrap()
    }

    /// Remove a node from the cluster, simulating a graceful departure.
    ///
    /// Shuts down the node's transport and removes it from all registries.
    pub async fn remove_node(&mut self, id: &str) -> Result<()> {
        let node = self
            .nodes
            .remove(id)
            .ok_or_else(|| anyhow::anyhow!("node {id} not found"))?;

        // Stop heartbeat if running.
        if let Some(ref hb) = node.heartbeat {
            hb.stop();
        }

        // Shut down transport.
        node.transport.shutdown().await?;

        // Remove from all other registries.
        for remaining in self.nodes.values() {
            remaining.registry.remove_node(id);
        }

        Ok(())
    }

    /// Simulate a network partition: isolate a node.
    pub async fn partition_node(&self, id: &str) -> Result<()> {
        let node = self
            .nodes
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("node {id} not found"))?;
        self.router.partition_node(&node.address).await;
        Ok(())
    }

    /// Heal a network partition for a node.
    pub async fn heal_node(&self, id: &str) -> Result<()> {
        let node = self
            .nodes
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("node {id} not found"))?;
        self.router.heal_node(&node.address).await;
        Ok(())
    }

    /// Start heartbeat services for all nodes in the cluster.
    pub fn start_heartbeats(&mut self) {
        // Collect the info we need first to avoid borrow conflicts.
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        for id in &node_ids {
            let node = self.nodes.get(id).unwrap();
            if node.heartbeat.is_some() {
                continue; // Already running.
            }

            let hb_config = HeartbeatConfig {
                interval: self.config.heartbeat_interval,
                failure_threshold: self.config.failure_threshold,
                check_interval: self.config.failure_check_interval,
                include_metrics: false,
            };

            let registry = node.registry.clone();
            let transport = node.transport.clone();
            let service = HeartbeatService::new(hb_config, registry, None);
            service.start(transport);

            // Now mutably access to store the service.
            self.nodes.get_mut(id).unwrap().heartbeat = Some(service);
        }
    }

    /// Get a reference to a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&TestNode> {
        self.nodes.get(id)
    }

    /// Return the number of nodes in the cluster.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return all node IDs.
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get the mock router for advanced test scenarios.
    pub fn router(&self) -> &MockRouter {
        &self.router
    }

    /// Wait until a node's status in another node's registry matches the
    /// expected value, or timeout.
    pub async fn wait_for_status(
        &self,
        observer_id: &str,
        target_id: &str,
        expected: NodeStatus,
        timeout: Duration,
    ) -> Result<()> {
        let observer = self
            .nodes
            .get(observer_id)
            .ok_or_else(|| anyhow::anyhow!("observer {observer_id} not found"))?;

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Some(node) = observer.registry.get_node(target_id)
                && node.status == expected
            {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                bail!(
                    "timeout waiting for {target_id} to become {expected} in {observer_id}'s view"
                );
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Shut down all nodes and their services.
    pub async fn shutdown(&mut self) {
        for (_, node) in self.nodes.drain() {
            if let Some(ref hb) = node.heartbeat {
                hb.stop();
            }
            let _ = node.transport.shutdown().await;
        }
    }
}

/// Performance baseline for regression detection.
#[derive(Debug, Clone)]
pub struct PerfBaseline {
    /// Description of the metric.
    pub metric: String,
    /// Baseline value (e.g., microseconds, bytes/sec).
    pub baseline_value: f64,
    /// Maximum allowed regression as a multiplier (e.g., 1.5 = 50% slower).
    pub regression_threshold: f64,
}

impl PerfBaseline {
    /// Check whether a measured value exceeds the regression threshold.
    /// Returns `Ok(())` if within threshold, `Err` with details if regressed.
    pub fn check(&self, measured: f64) -> Result<()> {
        let limit = self.baseline_value * self.regression_threshold;
        if measured > limit {
            bail!(
                "performance regression for '{}': measured={measured:.2}, \
                 baseline={:.2}, threshold={:.2}x, limit={limit:.2}",
                self.metric,
                self.baseline_value,
                self.regression_threshold,
            );
        }
        Ok(())
    }
}

/// Measure control-message serialization round-trip time.
///
/// Serializes and deserializes a [`HeartbeatPayload`]-sized JSON structure
/// with the given `payload_size` bytes of padding, simulating the cost of
/// encoding/decoding control messages on the wire.
///
/// Returns the mean duration in microseconds over `iterations` runs.
pub fn measure_serialization_roundtrip(payload_size: usize, iterations: usize) -> f64 {
    use super::heartbeat::HeartbeatPayload;

    let mut total_us = 0.0;

    for i in 0..iterations {
        let payload = HeartbeatPayload {
            node_id: format!("bench-node-{i:0>width$}", width = payload_size.min(64)),
            sequence: i as u64,
            metrics: None,
        };

        let start = std::time::Instant::now();
        // Serialize
        let serialized = serde_json::to_vec(&payload).unwrap();
        // Deserialize
        let _decoded: HeartbeatPayload = serde_json::from_slice(&serialized).unwrap();
        total_us += start.elapsed().as_micros() as f64;
    }

    total_us / iterations as f64
}

/// Measure mock transport message transfer latency.
///
/// Returns the mean duration in microseconds over `iterations` runs.
pub async fn measure_mock_transfer_latency(router: &MockRouter, iterations: usize) -> f64 {
    let config = MockTransportConfig::default();
    let sender =
        MockTransport::new("bench-sender:1".to_string(), router.clone(), config.clone()).await;
    let receiver = MockTransport::new("bench-receiver:2".to_string(), router.clone(), config).await;

    let mut total_us = 0.0;

    for i in 0..iterations {
        let msg = super::transport::TransportMessage::Control {
            operation: format!("bench-{i}"),
            payload: bytes::Bytes::from(vec![0u8; 64]),
        };

        let start = std::time::Instant::now();
        sender.send("bench-receiver:2", msg).await.unwrap();
        let _ = receiver.recv().await.unwrap();
        total_us += start.elapsed().as_micros() as f64;
    }

    total_us / iterations as f64
}

#[cfg(test)]
#[path = "test_harness_tests.rs"]
mod tests;
