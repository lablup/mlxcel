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

//! Request lifecycle tracking for distributed inference.
//!
//! Assigns unique request IDs and tracks state transitions as requests move
//! across node boundaries (e.g., prefill -> decode handoff in disaggregated
//! inference, stage 1 -> stage 2 in pipeline parallel).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a distributed inference request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(String);

impl RequestId {
    /// Generate a new random request ID (UUID v4).
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a request ID from an existing string.
    /// Returns `None` if the string is empty or exceeds 256 bytes.
    pub fn from_string(id: String) -> Option<Self> {
        if id.is_empty() || id.len() > 256 {
            return None;
        }
        Some(Self(id))
    }

    /// Return the string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Current state of a request in its lifecycle.
///
/// Tracks the progression from submission through routing, processing,
/// optional cross-node handoff, and final completion or failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RequestState {
    /// Request has been submitted but not yet routed.
    Submitted,
    /// Request is being routed to a target node.
    Routing,
    /// Request is being processed on the specified node.
    Processing {
        /// ID of the node currently processing the request.
        node_id: String,
    },
    /// Request is being handed off between nodes (e.g., prefill -> decode).
    Handoff {
        /// Node handing off the request.
        from_node: String,
        /// Node receiving the request.
        to_node: String,
    },
    /// Request completed successfully.
    Completed,
    /// Request failed with an error.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
}

impl fmt::Display for RequestState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Submitted => write!(f, "submitted"),
            Self::Routing => write!(f, "routing"),
            Self::Processing { node_id } => write!(f, "processing(node={node_id})"),
            Self::Handoff { from_node, to_node } => write!(f, "handoff({from_node}->{to_node})"),
            Self::Completed => write!(f, "completed"),
            Self::Failed { reason } => write!(f, "failed({reason})"),
        }
    }
}

/// A timestamped state transition in the request lifecycle.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// The state transitioned to.
    pub state: RequestState,
    /// When this transition occurred.
    pub timestamp: Instant,
}

/// Full lifecycle record for a single request.
#[derive(Debug, Clone)]
pub struct RequestLifecycle {
    /// Unique request identifier.
    pub id: RequestId,
    /// Current state.
    pub current_state: RequestState,
    /// Ordered list of all state transitions.
    pub transitions: Vec<StateTransition>,
    /// When the request was first submitted.
    pub created_at: Instant,
}

impl RequestLifecycle {
    /// Create a new lifecycle in the Submitted state.
    fn new(id: RequestId) -> Self {
        let now = Instant::now();
        Self {
            id,
            current_state: RequestState::Submitted,
            transitions: vec![StateTransition {
                state: RequestState::Submitted,
                timestamp: now,
            }],
            created_at: now,
        }
    }

    /// Transition to a new state.
    fn transition(&mut self, state: RequestState) {
        self.current_state = state.clone();
        self.transitions.push(StateTransition {
            state,
            timestamp: Instant::now(),
        });
    }

    /// Return the total elapsed time since request creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Return true if the request is in a terminal state (Completed or Failed).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.current_state,
            RequestState::Completed | RequestState::Failed { .. }
        )
    }
}

/// Check whether a state transition is valid according to the request
/// lifecycle state machine.
///
/// Valid transitions:
///   Submitted  -> Routing | Failed
///   Routing    -> Processing | Failed
///   Processing -> Handoff | Completed | Failed
///   Handoff    -> Processing | Failed
///   Completed  -> (terminal, no transitions)
///   Failed     -> (terminal, no transitions)
fn is_valid_transition(from: &RequestState, to: &RequestState) -> bool {
    matches!(
        (from, to),
        (RequestState::Submitted, RequestState::Routing)
            | (RequestState::Submitted, RequestState::Failed { .. })
            | (RequestState::Routing, RequestState::Processing { .. })
            | (RequestState::Routing, RequestState::Failed { .. })
            | (
                RequestState::Processing { .. },
                RequestState::Handoff { .. }
            )
            | (RequestState::Processing { .. }, RequestState::Completed)
            | (RequestState::Processing { .. }, RequestState::Failed { .. })
            | (
                RequestState::Handoff { .. },
                RequestState::Processing { .. }
            )
            | (RequestState::Handoff { .. }, RequestState::Failed { .. })
    )
}

/// Maximum number of tracked requests before oldest completed requests
/// are evicted to prevent unbounded memory growth.
const DEFAULT_MAX_TRACKED: usize = 10_000;

/// Configuration for the request tracker.
#[derive(Debug, Clone)]
pub struct RequestTrackerConfig {
    /// Maximum number of request lifecycles to retain.
    /// Oldest completed requests are evicted when this limit is exceeded.
    pub max_tracked: usize,
}

impl Default for RequestTrackerConfig {
    fn default() -> Self {
        Self {
            max_tracked: DEFAULT_MAX_TRACKED,
        }
    }
}

/// Thread-safe tracker for request lifecycles across the cluster.
///
/// Used by: Scheduler, routing strategies
#[derive(Clone)]
pub struct RequestTracker {
    inner: Arc<RwLock<TrackerInner>>,
    config: RequestTrackerConfig,
}

struct TrackerInner {
    /// Request ID -> lifecycle.
    requests: HashMap<String, RequestLifecycle>,
}

impl RequestTracker {
    /// Create a new request tracker with the given configuration.
    pub fn new(config: RequestTrackerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(TrackerInner {
                requests: HashMap::new(),
            })),
            config,
        }
    }

    /// Submit a new request and return its assigned ID.
    pub fn submit(&self) -> RequestId {
        let id = RequestId::new();
        let lifecycle = RequestLifecycle::new(id.clone());
        let mut inner = self.inner.write().expect("tracker lock poisoned");
        self.evict_if_needed(&mut inner);
        inner.requests.insert(id.as_str().to_string(), lifecycle);
        id
    }

    /// Submit a request with a pre-assigned ID.
    pub fn submit_with_id(&self, id: RequestId) {
        let lifecycle = RequestLifecycle::new(id.clone());
        let mut inner = self.inner.write().expect("tracker lock poisoned");
        self.evict_if_needed(&mut inner);
        inner.requests.insert(id.as_str().to_string(), lifecycle);
    }

    /// Transition a request to a new state.
    ///
    /// Returns `false` if the request is not found, is already in a terminal
    /// state, or the transition is invalid according to the state machine.
    #[must_use]
    pub fn transition(&self, id: &RequestId, state: RequestState) -> bool {
        let mut inner = self.inner.write().expect("tracker lock poisoned");
        if let Some(lifecycle) = inner.requests.get_mut(id.as_str()) {
            if lifecycle.is_terminal() {
                return false;
            }
            if !is_valid_transition(&lifecycle.current_state, &state) {
                return false;
            }
            lifecycle.transition(state);
            true
        } else {
            false
        }
    }

    /// Get the current state of a request.
    pub fn get_state(&self, id: &RequestId) -> Option<RequestState> {
        let inner = self.inner.read().expect("tracker lock poisoned");
        inner
            .requests
            .get(id.as_str())
            .map(|l| l.current_state.clone())
    }

    /// Get a clone of the full lifecycle for a request.
    pub fn get_lifecycle(&self, id: &RequestId) -> Option<RequestLifecycle> {
        let inner = self.inner.read().expect("tracker lock poisoned");
        inner.requests.get(id.as_str()).cloned()
    }

    /// Return the number of currently tracked requests.
    pub fn tracked_count(&self) -> usize {
        self.inner
            .read()
            .expect("tracker lock poisoned")
            .requests
            .len()
    }

    /// Return the number of active (non-terminal) requests.
    pub fn active_count(&self) -> usize {
        self.inner
            .read()
            .expect("tracker lock poisoned")
            .requests
            .values()
            .filter(|l| !l.is_terminal())
            .count()
    }

    /// Remove a specific request from tracking.
    pub fn remove(&self, id: &RequestId) -> Option<RequestLifecycle> {
        let mut inner = self.inner.write().expect("tracker lock poisoned");
        inner.requests.remove(id.as_str())
    }

    /// Evict oldest completed requests if we exceed the limit.
    fn evict_if_needed(&self, inner: &mut TrackerInner) {
        if inner.requests.len() < self.config.max_tracked {
            return;
        }

        // Collect completed request keys sorted by creation time (oldest first).
        let mut completed: Vec<(String, Instant)> = inner
            .requests
            .iter()
            .filter(|(_, l)| l.is_terminal())
            .map(|(k, l)| (k.clone(), l.created_at))
            .collect();

        // The key component is what makes the sort key a TOTAL order, and it
        // must stay. `sort_by_key` is stable, `completed` comes out of a
        // `HashMap` whose iteration order `RandomState` randomizes per
        // instance, and the `take(to_remove)` below consumes only a prefix.
        // Without the tie-break, requests sharing a `created_at` tick keep hash
        // order and which of them is dropped changes from run to run. The
        // comparator form avoids cloning the key that `sort_by_key` would need.
        completed.sort_by(|(key_a, t_a), (key_b, t_b)| t_a.cmp(t_b).then_with(|| key_a.cmp(key_b)));

        // Remove oldest completed until we are under the limit.
        let to_remove = inner.requests.len().saturating_sub(self.config.max_tracked) + 1;
        for (key, _) in completed.into_iter().take(to_remove) {
            inner.requests.remove(&key);
        }
    }
}

#[cfg(test)]
#[path = "request_tracker_tests.rs"]
mod tests;
