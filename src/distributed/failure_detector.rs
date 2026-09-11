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

//! Threshold-based failure detector for distributed cluster nodes.
//!
//! Uses a simple "missed heartbeats" approach: each node is expected to send
//! a heartbeat within a configurable interval. After a configurable number of
//! missed heartbeats, the node is declared failed.
//!
//! Integrates with [`NodeRegistry`] to update node status on failure detection
//! and notifies subscribers through a broadcast channel.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::broadcast;

use super::registry::{NodeRegistry, NodeStatus};

/// Configuration for the failure detector.
#[derive(Debug, Clone)]
pub struct FailureDetectorConfig {
    /// Expected heartbeat interval. A node that has not sent a heartbeat
    /// within `heartbeat_interval * failure_threshold` is considered failed.
    pub heartbeat_interval: Duration,
    /// Number of consecutive missed heartbeats before declaring failure.
    pub failure_threshold: u32,
    /// How often the detector checks for missed heartbeats.
    pub check_interval: Duration,
}

impl Default for FailureDetectorConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            failure_threshold: 3,
            check_interval: Duration::from_secs(2),
        }
    }
}

impl FailureDetectorConfig {
    /// The maximum time a node can be silent before being declared failed.
    pub fn failure_timeout(&self) -> Duration {
        self.heartbeat_interval * self.failure_threshold
    }
}

/// Event emitted when the failure detector changes a node's status.
#[derive(Debug, Clone)]
pub struct FailureEvent {
    /// ID of the node whose status changed.
    pub node_id: String,
    /// Previous status.
    pub previous_status: NodeStatus,
    /// New status after the change.
    pub new_status: NodeStatus,
    /// When the event occurred.
    pub timestamp: Instant,
}

/// Per-node heartbeat tracking state.
#[derive(Debug, Clone)]
struct NodeHeartbeatState {
    /// Timestamp of the last received heartbeat.
    last_heartbeat: Instant,
    /// Number of consecutive missed heartbeat windows.
    missed_count: u32,
    /// Whether we have already notified about this node's failure.
    notified_failed: bool,
}

/// Threshold-based failure detector that monitors heartbeat liveness.
///
/// Call [`record_heartbeat`](Self::record_heartbeat) when a heartbeat is
/// received from a peer, and [`check_failures`](Self::check_failures)
/// periodically to detect nodes that have gone silent.
///
/// Used by: HeartbeatService
pub struct FailureDetector {
    config: FailureDetectorConfig,
    state: Arc<RwLock<HashMap<String, NodeHeartbeatState>>>,
    event_tx: broadcast::Sender<FailureEvent>,
}

impl FailureDetector {
    /// Create a new failure detector with the given configuration.
    ///
    /// Returns the detector and a receiver for failure events. Additional
    /// receivers can be obtained via [`subscribe`](Self::subscribe).
    pub fn new(config: FailureDetectorConfig) -> (Self, broadcast::Receiver<FailureEvent>) {
        let (tx, rx) = broadcast::channel(64);
        (
            Self {
                config,
                state: Arc::new(RwLock::new(HashMap::new())),
                event_tx: tx,
            },
            rx,
        )
    }

    /// Subscribe to failure events. Returns a new receiver.
    pub fn subscribe(&self) -> broadcast::Receiver<FailureEvent> {
        self.event_tx.subscribe()
    }

    /// Register a node to be monitored. Starts the heartbeat clock now.
    pub fn register_node(&self, node_id: &str) {
        let mut state = self.state.write().expect("failure detector lock poisoned");
        state.insert(
            node_id.to_string(),
            NodeHeartbeatState {
                last_heartbeat: Instant::now(),
                missed_count: 0,
                notified_failed: false,
            },
        );
    }

    /// Remove a node from monitoring (e.g., graceful departure).
    pub fn unregister_node(&self, node_id: &str) {
        let mut state = self.state.write().expect("failure detector lock poisoned");
        state.remove(node_id);
    }

    /// Record that a heartbeat was received from the given node.
    /// Resets the missed counter. If the node was previously marked failed,
    /// the `notified_failed` flag is preserved so that the next
    /// `check_failures` call can emit a recovery event.
    pub fn record_heartbeat(&self, node_id: &str) {
        let mut state = self.state.write().expect("failure detector lock poisoned");
        if let Some(entry) = state.get_mut(node_id) {
            entry.last_heartbeat = Instant::now();
            entry.missed_count = 0;
            // Do NOT reset notified_failed here; check_failures will detect
            // that elapsed < timeout while notified_failed is true and emit
            // a recovery event.
        }
    }

    /// Check all monitored nodes for missed heartbeats and update the registry.
    ///
    /// Returns a list of node IDs that were newly declared failed during this
    /// check cycle.
    pub fn check_failures(&self, registry: &NodeRegistry) -> Vec<String> {
        let now = Instant::now();
        let timeout = self.config.failure_timeout();
        let mut newly_failed = Vec::new();

        let mut state = self.state.write().expect("failure detector lock poisoned");
        for (node_id, entry) in state.iter_mut() {
            let elapsed = now.duration_since(entry.last_heartbeat);

            if elapsed > timeout {
                if !entry.notified_failed {
                    // Determine previous status from registry
                    let previous_status = registry
                        .get_node(node_id)
                        .map(|n| n.status)
                        .unwrap_or(NodeStatus::Online);

                    registry.set_node_status(node_id, NodeStatus::Unreachable);
                    entry.notified_failed = true;
                    newly_failed.push(node_id.clone());

                    let event = FailureEvent {
                        node_id: node_id.clone(),
                        previous_status,
                        new_status: NodeStatus::Unreachable,
                        timestamp: now,
                    };
                    // Best-effort: if no subscribers, the event is dropped.
                    let _ = self.event_tx.send(event);

                    tracing::warn!(
                        node_id = node_id.as_str(),
                        elapsed_ms = elapsed.as_millis() as u64,
                        "node declared unreachable after {:.1}s without heartbeat",
                        elapsed.as_secs_f64()
                    );
                }
                entry.missed_count += 1;
            } else if entry.notified_failed {
                // Node recovered: got a heartbeat after being declared failed.
                // This can happen if record_heartbeat was called between checks.
                let previous_status = NodeStatus::Unreachable;
                registry.set_node_status(node_id, NodeStatus::Online);
                entry.notified_failed = false;
                entry.missed_count = 0;

                let event = FailureEvent {
                    node_id: node_id.clone(),
                    previous_status,
                    new_status: NodeStatus::Online,
                    timestamp: now,
                };
                let _ = self.event_tx.send(event);

                tracing::info!(
                    node_id = node_id.as_str(),
                    "node recovered and marked online"
                );
            }
        }

        newly_failed
    }

    /// Return the number of currently monitored nodes.
    pub fn monitored_count(&self) -> usize {
        self.state
            .read()
            .expect("failure detector lock poisoned")
            .len()
    }

    /// Check whether a specific node is currently considered failed.
    pub fn is_node_failed(&self, node_id: &str) -> bool {
        self.state
            .read()
            .expect("failure detector lock poisoned")
            .get(node_id)
            .map(|s| s.notified_failed)
            .unwrap_or(false)
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &FailureDetectorConfig {
        &self.config
    }
}

#[cfg(test)]
#[path = "failure_detector_tests.rs"]
mod tests;
