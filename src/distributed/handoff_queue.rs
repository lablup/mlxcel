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

//! Cross-node request handoff queues for distributed inference.
//!
//! Manages bounded queues for handing off requests between nodes (e.g.,
//! prefill -> decode in disaggregated inference, stage N -> stage N+1 in
//! pipeline parallel). Supports configurable capacity and overflow behavior.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::request_tracker::RequestId;

/// Policy to apply when a handoff queue reaches capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OverflowPolicy {
    /// Reject the new item (caller receives an error).
    Reject,
    /// Drop the oldest item in the queue to make room.
    DropOldest,
    /// Block until space becomes available (async only).
    Block,
}

impl fmt::Display for OverflowPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reject => write!(f, "reject"),
            Self::DropOldest => write!(f, "drop_oldest"),
            Self::Block => write!(f, "block"),
        }
    }
}

/// Configuration for a handoff queue.
#[derive(Debug, Clone)]
pub struct HandoffQueueConfig {
    /// Maximum number of items the queue can hold.
    pub capacity: usize,
    /// Policy to apply when the queue is full.
    pub overflow_policy: OverflowPolicy,
}

impl Default for HandoffQueueConfig {
    fn default() -> Self {
        Self {
            capacity: 64,
            overflow_policy: OverflowPolicy::Reject,
        }
    }
}

/// An item in the handoff queue, representing a request being transferred
/// between nodes.
#[derive(Debug, Clone)]
pub struct HandoffItem {
    /// The request being handed off.
    pub request_id: RequestId,
    /// ID of the source node.
    pub from_node: String,
    /// ID of the destination node.
    pub to_node: String,
    /// Opaque payload associated with the handoff (e.g., serialized KV cache
    /// metadata, partial generation state).
    pub payload: Vec<u8>,
    /// When this item was enqueued.
    pub enqueued_at: Instant,
}

/// Result of an enqueue operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnqueueResult {
    /// Item was successfully enqueued.
    Success,
    /// Item was enqueued after dropping the oldest item.
    DroppedOldest,
    /// Item was rejected because the queue is full.
    Rejected,
}

/// A bounded queue for cross-node request handoffs.
///
/// Thread-safe: uses interior mutability so it can be shared across tasks.
///
/// Used by: Scheduler, pipeline-parallel and disaggregated-inference routing
#[derive(Clone)]
pub struct HandoffQueue {
    config: HandoffQueueConfig,
    inner: Arc<RwLock<VecDeque<HandoffItem>>>,
    /// Counter for total items ever enqueued (for stats).
    stats: Arc<RwLock<QueueStats>>,
}

/// Statistics for a handoff queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total items enqueued.
    pub total_enqueued: u64,
    /// Total items dequeued.
    pub total_dequeued: u64,
    /// Total items dropped due to overflow.
    pub total_dropped: u64,
    /// Total items rejected due to overflow.
    pub total_rejected: u64,
}

impl HandoffQueue {
    /// Create a new handoff queue with the given configuration.
    pub fn new(config: HandoffQueueConfig) -> Self {
        let capacity = config.capacity;
        Self {
            config,
            inner: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            stats: Arc::new(RwLock::new(QueueStats::default())),
        }
    }

    /// Enqueue a handoff item according to the configured overflow policy.
    pub fn enqueue(&self, item: HandoffItem) -> EnqueueResult {
        let mut queue = self.inner.write().expect("handoff queue lock poisoned");
        let mut stats = self.stats.write().expect("handoff stats lock poisoned");

        if queue.len() >= self.config.capacity {
            match self.config.overflow_policy {
                OverflowPolicy::Reject => {
                    stats.total_rejected += 1;
                    return EnqueueResult::Rejected;
                }
                OverflowPolicy::DropOldest => {
                    queue.pop_front();
                    stats.total_dropped += 1;
                    queue.push_back(item);
                    stats.total_enqueued += 1;
                    return EnqueueResult::DroppedOldest;
                }
                OverflowPolicy::Block => {
                    // Synchronous blocking is not supported; callers should use
                    // the async dequeue method. Fall back to reject.
                    stats.total_rejected += 1;
                    return EnqueueResult::Rejected;
                }
            }
        }

        queue.push_back(item);
        stats.total_enqueued += 1;
        EnqueueResult::Success
    }

    /// Dequeue the next handoff item (FIFO order).
    pub fn dequeue(&self) -> Option<HandoffItem> {
        let mut queue = self.inner.write().expect("handoff queue lock poisoned");
        let item = queue.pop_front();
        if item.is_some() {
            let mut stats = self.stats.write().expect("handoff stats lock poisoned");
            stats.total_dequeued += 1;
        }
        item
    }

    /// Peek at the next item without removing it.
    pub fn peek(&self) -> Option<HandoffItem> {
        let queue = self.inner.read().expect("handoff queue lock poisoned");
        queue.front().cloned()
    }

    /// Return the current number of items in the queue.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("handoff queue lock poisoned")
            .len()
    }

    /// Check whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check whether the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.len() >= self.config.capacity
    }

    /// Return the configured capacity.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Return a snapshot of queue statistics.
    pub fn stats(&self) -> QueueStats {
        self.stats
            .read()
            .expect("handoff stats lock poisoned")
            .clone()
    }

    /// Clear all items from the queue.
    pub fn clear(&self) {
        let mut queue = self.inner.write().expect("handoff queue lock poisoned");
        queue.clear();
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &HandoffQueueConfig {
        &self.config
    }
}

/// Manager for multiple named handoff queues.
///
/// Queues are identified by a string key (e.g., "prefill->decode",
/// "stage-0->stage-1"). New queues are created on demand with the default
/// configuration or a per-queue override.
///
/// Used by: Scheduler
#[derive(Clone)]
pub struct HandoffQueueManager {
    default_config: HandoffQueueConfig,
    queues: Arc<RwLock<HashMap<String, HandoffQueue>>>,
}

impl HandoffQueueManager {
    /// Create a new manager with the given default queue configuration.
    pub fn new(default_config: HandoffQueueConfig) -> Self {
        Self {
            default_config,
            queues: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a queue with the given name and default config.
    pub fn get_or_create(&self, name: &str) -> HandoffQueue {
        let queues = self.queues.read().expect("queue manager lock poisoned");
        if let Some(q) = queues.get(name) {
            return q.clone();
        }
        drop(queues);

        let mut queues = self.queues.write().expect("queue manager lock poisoned");
        queues
            .entry(name.to_string())
            .or_insert_with(|| HandoffQueue::new(self.default_config.clone()))
            .clone()
    }

    /// Get or create a queue with a specific configuration.
    pub fn get_or_create_with_config(
        &self,
        name: &str,
        config: HandoffQueueConfig,
    ) -> HandoffQueue {
        let mut queues = self.queues.write().expect("queue manager lock poisoned");
        queues
            .entry(name.to_string())
            .or_insert_with(|| HandoffQueue::new(config))
            .clone()
    }

    /// Get an existing queue by name.
    pub fn get(&self, name: &str) -> Option<HandoffQueue> {
        let queues = self.queues.read().expect("queue manager lock poisoned");
        queues.get(name).cloned()
    }

    /// Return the names of all managed queues.
    pub fn queue_names(&self) -> Vec<String> {
        let queues = self.queues.read().expect("queue manager lock poisoned");
        queues.keys().cloned().collect()
    }

    /// Remove a queue by name.
    pub fn remove(&self, name: &str) -> Option<HandoffQueue> {
        let mut queues = self.queues.write().expect("queue manager lock poisoned");
        queues.remove(name)
    }
}

#[cfg(test)]
#[path = "handoff_queue_tests.rs"]
mod tests;
