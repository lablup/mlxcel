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

//! Cluster-wide correlation IDs for cross-node request tracing.
//!
//! Each request entering the cluster is assigned a unique [`CorrelationId`] that
//! propagates through all inter-node messages. This enables distributed log
//! correlation without a centralized log aggregator.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let ctx = RequestContext::new("node-0");
//! tracing::info!(correlation_id = %ctx.correlation_id, "processing request");
//! // Pass ctx.correlation_id through TransportMessage metadata
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Maximum number of hops a request context can traverse before being rejected.
/// Prevents infinite routing loops in misconfigured clusters.
pub const MAX_HOP_COUNT: u32 = 64;

/// Unique identifier for tracing a request across node boundaries.
///
/// Generated once when a request enters the cluster and propagated through
/// all downstream inter-node messages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrelationId(String);

impl CorrelationId {
    /// Generate a new random correlation ID (UUID v4).
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a correlation ID from an existing string (e.g., received from
    /// another node). Returns `None` if the string is empty or exceeds 256
    /// bytes (a sanity bound to prevent memory abuse from untrusted peers).
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

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Context carried through a distributed request, combining the correlation ID
/// with the originating node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    /// Cluster-wide correlation ID for this request.
    pub correlation_id: CorrelationId,
    /// ID of the node that originated (or is forwarding) this request.
    pub origin_node: String,
    /// Monotonic hop counter incremented at each node boundary.
    pub hop_count: u32,
    /// Optional parent span ID for hierarchical tracing.
    pub parent_span: Option<String>,
}

impl RequestContext {
    /// Create a new request context originating at the given node.
    pub fn new(origin_node: &str) -> Self {
        Self {
            correlation_id: CorrelationId::new(),
            origin_node: origin_node.to_string(),
            hop_count: 0,
            parent_span: None,
        }
    }

    /// Create a context from an existing correlation ID (e.g., received from
    /// a peer node), incrementing the hop count.
    ///
    /// Returns `None` if the hop count would exceed [`MAX_HOP_COUNT`], which
    /// indicates a likely routing loop.
    pub fn from_incoming(
        correlation_id: CorrelationId,
        origin_node: &str,
        previous_hop_count: u32,
    ) -> Option<Self> {
        let next_hop = previous_hop_count.saturating_add(1);
        if next_hop > MAX_HOP_COUNT {
            tracing::warn!(
                correlation_id = %correlation_id,
                hop_count = next_hop,
                "request exceeded max hop count ({MAX_HOP_COUNT}), possible routing loop"
            );
            return None;
        }
        Some(Self {
            correlation_id,
            origin_node: origin_node.to_string(),
            hop_count: next_hop,
            parent_span: None,
        })
    }

    /// Attach a parent span ID for hierarchical tracing.
    pub fn with_parent_span(mut self, span_id: String) -> Self {
        self.parent_span = Some(span_id);
        self
    }

    /// Serialize the context to JSON bytes for embedding in transport messages.
    /// Returns `None` if serialization fails (should not happen with valid data).
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        match serde_json::to_vec(self) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                tracing::error!("failed to serialize RequestContext: {e}");
                None
            }
        }
    }

    /// Deserialize a context from JSON bytes received in a transport message.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        serde_json::from_slice(bytes).ok()
    }
}

#[cfg(test)]
#[path = "correlation_tests.rs"]
mod tests;
