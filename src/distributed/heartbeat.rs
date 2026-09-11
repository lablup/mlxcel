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

//! Heartbeat protocol for distributed cluster health monitoring.
//!
//! The [`HeartbeatService`] runs as a background tokio task that:
//!
//! 1. Periodically sends heartbeat messages to all peer nodes via the transport layer
//! 2. Processes incoming heartbeat messages and feeds them to the failure detector
//! 3. Runs the failure detector's check cycle to detect unresponsive nodes
//! 4. Collects and distributes per-node metrics alongside heartbeats
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │            HeartbeatService              │
//! │  ┌──────────┐  ┌──────────────────────┐ │
//! │  │  sender   │  │  failure_detector    │ │
//! │  │  (tick)   │  │  (check_failures)    │ │
//! │  └──────────┘  └──────────────────────┘ │
//! │        │                ▲                │
//! │        ▼                │                │
//! │  ┌──────────┐  ┌──────────────────────┐ │
//! │  │Transport  │  │  record_heartbeat    │ │
//! │  │  .send()  │  │  (from recv loop)    │ │
//! │  └──────────┘  └──────────────────────┘ │
//! └─────────────────────────────────────────┘
//! ```

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::failure_detector::{FailureDetector, FailureDetectorConfig, FailureEvent};
use super::metrics::{MetricsCollector, NodeMetrics};
use super::registry::NodeRegistry;
use super::transport::{Transport, TransportMessage};

/// Configuration for the heartbeat service.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval between sending heartbeat messages to each peer.
    pub interval: Duration,
    /// Number of missed heartbeats before declaring a node failed.
    pub failure_threshold: u32,
    /// How often to run the failure detection check.
    pub check_interval: Duration,
    /// Whether to include local metrics in heartbeat messages.
    pub include_metrics: bool,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            failure_threshold: 3,
            check_interval: Duration::from_secs(2),
            include_metrics: true,
        }
    }
}

/// Payload included in heartbeat messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    /// ID of the node sending the heartbeat.
    pub node_id: String,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Optional metrics snapshot from the sending node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<NodeMetrics>,
}

/// The heartbeat operation tag used in [`TransportMessage::Control`].
pub const HEARTBEAT_OPERATION: &str = "heartbeat";

/// Background service that manages heartbeat sending, receiving, and failure
/// detection for a distributed cluster.
///
/// Created via [`HeartbeatService::new`] and started with [`HeartbeatService::start`].
/// The service runs until the cancellation token is triggered or
/// [`HeartbeatService::stop`] is called.
pub struct HeartbeatService {
    config: HeartbeatConfig,
    registry: NodeRegistry,
    failure_detector: Arc<FailureDetector>,
    metrics_collector: Option<MetricsCollector>,
    cancel: CancellationToken,
    /// Initial event receiver; kept so the broadcast channel stays alive until
    /// the service is dropped. Consumers should use `subscribe_events()`.
    _event_rx: broadcast::Receiver<FailureEvent>,
}

impl HeartbeatService {
    /// Create a new heartbeat service.
    ///
    /// The service monitors all peers in the registry (excluding the local node).
    /// If a `MetricsCollector` is provided, local metrics are included in
    /// heartbeat payloads.
    pub fn new(
        config: HeartbeatConfig,
        registry: NodeRegistry,
        metrics_collector: Option<MetricsCollector>,
    ) -> Self {
        let detector_config = FailureDetectorConfig {
            heartbeat_interval: config.interval,
            failure_threshold: config.failure_threshold,
            check_interval: config.check_interval,
        };
        let (detector, event_rx) = FailureDetector::new(detector_config);

        // Register all peers for monitoring.
        let local_id = registry.local_node_id();
        for node in registry.all_nodes() {
            if node.config.id != local_id {
                detector.register_node(&node.config.id);
            }
        }

        Self {
            config,
            registry,
            failure_detector: Arc::new(detector),
            metrics_collector,
            cancel: CancellationToken::new(),
            _event_rx: event_rx,
        }
    }

    /// Subscribe to failure/recovery events from the underlying detector.
    pub fn subscribe_events(&self) -> broadcast::Receiver<FailureEvent> {
        self.failure_detector.subscribe()
    }

    /// Return a reference to the failure detector.
    pub fn failure_detector(&self) -> &FailureDetector {
        &self.failure_detector
    }

    /// Start the heartbeat service as background tokio tasks.
    ///
    /// Spawns two tasks:
    /// 1. **Sender**: Periodically sends heartbeats to all peers
    /// 2. **Checker**: Periodically runs the failure detector
    ///
    /// The `transport` is used to send heartbeat control messages. Incoming
    /// heartbeats should be fed via [`process_incoming_heartbeat`](Self::process_incoming_heartbeat)
    /// from whatever message receive loop the application runs.
    pub fn start(&self, transport: Arc<dyn Transport>) {
        let cancel = self.cancel.clone();
        let registry = self.registry.clone();
        let interval = self.config.interval;
        let include_metrics = self.config.include_metrics;
        let metrics_collector = self.metrics_collector.clone();

        // Sender task
        let cancel_send = cancel.clone();
        tokio::spawn(async move {
            let mut sequence: u64 = 0;
            let local_id = registry.local_node_id();
            let mut ticker = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = cancel_send.cancelled() => {
                        tracing::debug!("heartbeat sender shutting down");
                        break;
                    }
                    _ = ticker.tick() => {
                        sequence += 1;
                        let metrics = if include_metrics {
                            metrics_collector.as_ref().map(|c| c.snapshot())
                        } else {
                            None
                        };

                        let payload = HeartbeatPayload {
                            node_id: local_id.clone(),
                            sequence,
                            metrics,
                        };

                        let payload_bytes = match serde_json::to_vec(&payload) {
                            Ok(b) => b,
                            Err(e) => {
                                tracing::error!("failed to serialize heartbeat: {e}");
                                continue;
                            }
                        };

                        let msg = TransportMessage::Control {
                            operation: HEARTBEAT_OPERATION.to_string(),
                            payload: Bytes::from(payload_bytes),
                        };

                        // Send to all peers
                        let peers = registry.peer_addresses();
                        for peer_addr in &peers {
                            let peer_str = peer_addr.to_string();
                            if let Err(e) = transport.send(&peer_str, msg.clone()).await {
                                tracing::debug!(
                                    peer = peer_str.as_str(),
                                    "failed to send heartbeat: {e}"
                                );
                            }
                        }
                    }
                }
            }
        });

        // Checker task
        let cancel_check = cancel.clone();
        let detector = self.failure_detector.clone();
        let registry_check = self.registry.clone();
        let check_interval = self.config.check_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(check_interval);

            loop {
                tokio::select! {
                    _ = cancel_check.cancelled() => {
                        tracing::debug!("failure checker shutting down");
                        break;
                    }
                    _ = ticker.tick() => {
                        let failed = detector.check_failures(&registry_check);
                        if !failed.is_empty() {
                            tracing::warn!(
                                "failure detector: {} node(s) unreachable: {}",
                                failed.len(),
                                failed.join(", ")
                            );
                        }
                    }
                }
            }
        });
    }

    /// Maximum allowed length for a node_id in incoming heartbeats.
    /// Prevents memory abuse from untrusted or malformed payloads.
    const MAX_NODE_ID_LEN: usize = 256;

    /// Process an incoming heartbeat message received from a peer.
    ///
    /// Call this from the application's transport receive loop when a control
    /// message with operation [`HEARTBEAT_OPERATION`] is received.
    ///
    /// Validates that the `node_id` in the payload has a reasonable length,
    /// rejecting spoofed or malformed heartbeats.
    pub fn process_incoming_heartbeat(&self, payload_bytes: &[u8]) {
        let payload: HeartbeatPayload = match serde_json::from_slice(payload_bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!("invalid heartbeat payload: {e}");
                return;
            }
        };

        // Reject node IDs that are empty or excessively long to prevent
        // memory abuse from untrusted peers.
        if payload.node_id.is_empty() || payload.node_id.len() > Self::MAX_NODE_ID_LEN {
            tracing::debug!(
                node_id_len = payload.node_id.len(),
                "rejecting heartbeat with invalid node_id length"
            );
            return;
        }

        tracing::trace!(
            node_id = payload.node_id.as_str(),
            sequence = payload.sequence,
            "received heartbeat"
        );

        self.failure_detector.record_heartbeat(&payload.node_id);
    }

    /// Stop the heartbeat service by cancelling all background tasks.
    pub fn stop(&self) {
        self.cancel.cancel();
    }

    /// Check if the service has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

#[cfg(test)]
#[path = "heartbeat_tests.rs"]
mod tests;
