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

//! Backpressure signaling for distributed inference nodes.
//!
//! Nodes signal when they are overloaded, causing the scheduler to redirect
//! or queue requests. The backpressure system tracks per-node load levels
//! and applies configurable policies (drop, block, redirect) when thresholds
//! are exceeded.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Load level classification for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LoadLevel {
    /// Node is idle or lightly loaded.
    Low,
    /// Node is moderately loaded but can accept more work.
    Normal,
    /// Node is heavily loaded; new requests should be avoided if alternatives exist.
    High,
    /// Node is at capacity; new requests must be redirected or queued.
    Critical,
}

impl fmt::Display for LoadLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// A backpressure signal from a node indicating its current load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureSignal {
    /// ID of the signaling node.
    pub node_id: String,
    /// Current load level.
    pub load_level: LoadLevel,
    /// Number of active requests on the node.
    pub active_requests: u32,
    /// Memory utilization as a fraction (0.0 to 1.0).
    pub memory_utilization: f64,
    /// Optional message providing context.
    pub message: Option<String>,
}

/// Policy to apply when a node is under backpressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackpressurePolicy {
    /// Drop the request with an error.
    Drop,
    /// Block (queue) the request until the node recovers.
    Block,
    /// Redirect the request to a less-loaded node.
    Redirect,
}

impl fmt::Display for BackpressurePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Drop => write!(f, "drop"),
            Self::Block => write!(f, "block"),
            Self::Redirect => write!(f, "redirect"),
        }
    }
}

/// Configuration for the backpressure monitor.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Active request threshold for High load level.
    pub high_watermark: u32,
    /// Active request threshold for Critical load level.
    pub critical_watermark: u32,
    /// Memory utilization threshold (0.0-1.0) for High load level.
    pub memory_high_threshold: f64,
    /// Memory utilization threshold (0.0-1.0) for Critical load level.
    pub memory_critical_threshold: f64,
    /// Policy to apply when a node reaches Critical load.
    pub overflow_policy: BackpressurePolicy,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            high_watermark: 8,
            critical_watermark: 16,
            memory_high_threshold: 0.80,
            memory_critical_threshold: 0.95,
            overflow_policy: BackpressurePolicy::Redirect,
        }
    }
}

/// Per-node backpressure state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NodeLoadState {
    /// Current load level.
    level: LoadLevel,
    /// When the load level was last updated.
    updated_at: Instant,
    /// Number of active requests.
    active_requests: u32,
    /// Memory utilization (0.0 to 1.0).
    memory_utilization: f64,
}

/// Thread-safe backpressure monitor tracking load levels across all nodes.
///
/// Used by: Scheduler, routing strategies
#[derive(Clone)]
pub struct BackpressureMonitor {
    config: BackpressureConfig,
    inner: Arc<RwLock<HashMap<String, NodeLoadState>>>,
}

impl BackpressureMonitor {
    /// Create a new backpressure monitor with the given configuration.
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process a backpressure signal from a node.
    pub fn record_signal(&self, signal: &BackpressureSignal) {
        let mut inner = self.inner.write().expect("backpressure lock poisoned");
        inner.insert(
            signal.node_id.clone(),
            NodeLoadState {
                level: signal.load_level,
                updated_at: Instant::now(),
                active_requests: signal.active_requests,
                memory_utilization: signal.memory_utilization,
            },
        );
    }

    /// Update the load level for a node based on its current metrics.
    ///
    /// Computes the load level from active_requests and memory_utilization
    /// using the configured thresholds.
    pub fn update_from_metrics(
        &self,
        node_id: &str,
        active_requests: u32,
        memory_utilization: f64,
    ) {
        let level = self.compute_load_level(active_requests, memory_utilization);
        let mut inner = self.inner.write().expect("backpressure lock poisoned");
        inner.insert(
            node_id.to_string(),
            NodeLoadState {
                level,
                updated_at: Instant::now(),
                active_requests,
                memory_utilization,
            },
        );
    }

    /// Compute the load level from raw metrics.
    fn compute_load_level(&self, active_requests: u32, memory_utilization: f64) -> LoadLevel {
        if active_requests >= self.config.critical_watermark
            || memory_utilization >= self.config.memory_critical_threshold
        {
            LoadLevel::Critical
        } else if active_requests >= self.config.high_watermark
            || memory_utilization >= self.config.memory_high_threshold
        {
            LoadLevel::High
        } else if active_requests > 0 {
            LoadLevel::Normal
        } else {
            LoadLevel::Low
        }
    }

    /// Get the current load level for a node. Returns `None` if the node
    /// has not reported any load information.
    pub fn get_load_level(&self, node_id: &str) -> Option<LoadLevel> {
        let inner = self.inner.read().expect("backpressure lock poisoned");
        inner.get(node_id).map(|s| s.level)
    }

    /// Check whether a node is under backpressure (High or Critical).
    pub fn is_under_pressure(&self, node_id: &str) -> bool {
        self.get_load_level(node_id)
            .map(|l| l >= LoadLevel::High)
            .unwrap_or(false)
    }

    /// Check whether a node is at critical load.
    pub fn is_critical(&self, node_id: &str) -> bool {
        self.get_load_level(node_id)
            .map(|l| l == LoadLevel::Critical)
            .unwrap_or(false)
    }

    /// Return the overflow policy to apply for a node at critical load.
    pub fn overflow_policy(&self) -> BackpressurePolicy {
        self.config.overflow_policy
    }

    /// Return a snapshot of load levels for all tracked nodes.
    pub fn all_load_levels(&self) -> HashMap<String, LoadLevel> {
        let inner = self.inner.read().expect("backpressure lock poisoned");
        inner.iter().map(|(k, v)| (k.clone(), v.level)).collect()
    }

    /// Return the IDs of all nodes currently at Critical load.
    pub fn critical_nodes(&self) -> Vec<String> {
        let inner = self.inner.read().expect("backpressure lock poisoned");
        inner
            .iter()
            .filter(|(_, v)| v.level == LoadLevel::Critical)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Remove a node from tracking (e.g., when it leaves the cluster).
    pub fn remove_node(&self, node_id: &str) {
        let mut inner = self.inner.write().expect("backpressure lock poisoned");
        inner.remove(node_id);
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &BackpressureConfig {
        &self.config
    }
}

#[cfg(test)]
#[path = "backpressure_tests.rs"]
mod tests;
