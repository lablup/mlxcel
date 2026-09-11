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

use super::*;

fn make_item(request_id: &str, from: &str, to: &str) -> HandoffItem {
    HandoffItem {
        request_id: RequestId::from_string(request_id.to_string()).unwrap(),
        from_node: from.to_string(),
        to_node: to.to_string(),
        payload: vec![1, 2, 3],
        enqueued_at: Instant::now(),
    }
}

// -- OverflowPolicy --

#[test]
fn overflow_policy_display() {
    assert_eq!(format!("{}", OverflowPolicy::Reject), "reject");
    assert_eq!(format!("{}", OverflowPolicy::DropOldest), "drop_oldest");
    assert_eq!(format!("{}", OverflowPolicy::Block), "block");
}

// -- HandoffQueue basic operations --

#[test]
fn enqueue_and_dequeue() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    let item = make_item("req-1", "node-0", "node-1");
    assert_eq!(queue.enqueue(item), EnqueueResult::Success);
    assert_eq!(queue.len(), 1);
    assert!(!queue.is_empty());

    let dequeued = queue.dequeue().unwrap();
    assert_eq!(dequeued.request_id.as_str(), "req-1");
    assert!(queue.is_empty());
}

#[test]
fn fifo_ordering() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    queue.enqueue(make_item("req-1", "a", "b"));
    queue.enqueue(make_item("req-2", "a", "b"));
    queue.enqueue(make_item("req-3", "a", "b"));

    assert_eq!(queue.dequeue().unwrap().request_id.as_str(), "req-1");
    assert_eq!(queue.dequeue().unwrap().request_id.as_str(), "req-2");
    assert_eq!(queue.dequeue().unwrap().request_id.as_str(), "req-3");
    assert!(queue.dequeue().is_none());
}

#[test]
fn peek_does_not_remove() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    queue.enqueue(make_item("req-1", "a", "b"));

    let peeked = queue.peek().unwrap();
    assert_eq!(peeked.request_id.as_str(), "req-1");
    assert_eq!(queue.len(), 1);
}

#[test]
fn dequeue_empty_returns_none() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    assert!(queue.dequeue().is_none());
}

// -- Capacity and overflow --

#[test]
fn reject_policy_when_full() {
    let config = HandoffQueueConfig {
        capacity: 2,
        overflow_policy: OverflowPolicy::Reject,
    };
    let queue = HandoffQueue::new(config);
    assert_eq!(
        queue.enqueue(make_item("req-1", "a", "b")),
        EnqueueResult::Success
    );
    assert_eq!(
        queue.enqueue(make_item("req-2", "a", "b")),
        EnqueueResult::Success
    );
    assert!(queue.is_full());

    assert_eq!(
        queue.enqueue(make_item("req-3", "a", "b")),
        EnqueueResult::Rejected
    );
    assert_eq!(queue.len(), 2);

    let stats = queue.stats();
    assert_eq!(stats.total_enqueued, 2);
    assert_eq!(stats.total_rejected, 1);
}

#[test]
fn drop_oldest_policy_when_full() {
    let config = HandoffQueueConfig {
        capacity: 2,
        overflow_policy: OverflowPolicy::DropOldest,
    };
    let queue = HandoffQueue::new(config);
    queue.enqueue(make_item("req-1", "a", "b"));
    queue.enqueue(make_item("req-2", "a", "b"));

    assert_eq!(
        queue.enqueue(make_item("req-3", "a", "b")),
        EnqueueResult::DroppedOldest
    );
    assert_eq!(queue.len(), 2);

    // req-1 was dropped; req-2 is now first.
    assert_eq!(queue.dequeue().unwrap().request_id.as_str(), "req-2");
    assert_eq!(queue.dequeue().unwrap().request_id.as_str(), "req-3");

    let stats = queue.stats();
    assert_eq!(stats.total_enqueued, 3);
    assert_eq!(stats.total_dropped, 1);
}

#[test]
fn block_policy_falls_back_to_reject_sync() {
    let config = HandoffQueueConfig {
        capacity: 1,
        overflow_policy: OverflowPolicy::Block,
    };
    let queue = HandoffQueue::new(config);
    queue.enqueue(make_item("req-1", "a", "b"));
    // Block policy in sync context falls back to reject.
    assert_eq!(
        queue.enqueue(make_item("req-2", "a", "b")),
        EnqueueResult::Rejected
    );
}

// -- Statistics --

#[test]
fn stats_track_operations() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    queue.enqueue(make_item("req-1", "a", "b"));
    queue.enqueue(make_item("req-2", "a", "b"));
    queue.dequeue();

    let stats = queue.stats();
    assert_eq!(stats.total_enqueued, 2);
    assert_eq!(stats.total_dequeued, 1);
    assert_eq!(stats.total_dropped, 0);
    assert_eq!(stats.total_rejected, 0);
}

// -- Clear --

#[test]
fn clear_empties_queue() {
    let queue = HandoffQueue::new(HandoffQueueConfig::default());
    queue.enqueue(make_item("req-1", "a", "b"));
    queue.enqueue(make_item("req-2", "a", "b"));
    assert_eq!(queue.len(), 2);

    queue.clear();
    assert!(queue.is_empty());
}

// -- HandoffQueueManager --

#[test]
fn manager_creates_queues_on_demand() {
    let manager = HandoffQueueManager::new(HandoffQueueConfig::default());
    let q = manager.get_or_create("prefill->decode");
    assert!(q.is_empty());

    // Same name returns same queue.
    let q2 = manager.get_or_create("prefill->decode");
    q.enqueue(make_item("req-1", "a", "b"));
    assert_eq!(q2.len(), 1);
}

#[test]
fn manager_get_nonexistent_returns_none() {
    let manager = HandoffQueueManager::new(HandoffQueueConfig::default());
    assert!(manager.get("nonexistent").is_none());
}

#[test]
fn manager_queue_names() {
    let manager = HandoffQueueManager::new(HandoffQueueConfig::default());
    manager.get_or_create("a->b");
    manager.get_or_create("c->d");

    let names = manager.queue_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a->b".to_string()));
    assert!(names.contains(&"c->d".to_string()));
}

#[test]
fn manager_remove_queue() {
    let manager = HandoffQueueManager::new(HandoffQueueConfig::default());
    manager.get_or_create("test-queue");
    assert!(manager.get("test-queue").is_some());

    manager.remove("test-queue");
    assert!(manager.get("test-queue").is_none());
}

#[test]
fn manager_custom_config_per_queue() {
    let manager = HandoffQueueManager::new(HandoffQueueConfig::default());
    let custom = HandoffQueueConfig {
        capacity: 128,
        overflow_policy: OverflowPolicy::DropOldest,
    };
    let q = manager.get_or_create_with_config("high-cap", custom);
    assert_eq!(q.capacity(), 128);
}
