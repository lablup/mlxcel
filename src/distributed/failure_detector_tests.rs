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

use std::time::Duration;

use super::*;
use crate::distributed::config::{ClusterConfig, NodeRole};

fn test_registry() -> NodeRegistry {
    let config = ClusterConfig::from_cli(
        "local".to_string(),
        "127.0.0.1:8080".parse().unwrap(),
        NodeRole::Hybrid,
        vec!["127.0.0.1:8081".parse().unwrap()],
    );
    let registry = NodeRegistry::from_config(&config, "local");
    // Mark peer as online initially
    registry.set_node_status("peer-0", NodeStatus::Online);
    registry
}

#[test]
fn failure_timeout_calculation() {
    let config = FailureDetectorConfig {
        heartbeat_interval: Duration::from_secs(5),
        failure_threshold: 3,
        check_interval: Duration::from_secs(2),
    };
    assert_eq!(config.failure_timeout(), Duration::from_secs(15));
}

#[test]
fn register_and_unregister() {
    let (detector, _rx) = FailureDetector::new(FailureDetectorConfig::default());

    detector.register_node("node-1");
    assert_eq!(detector.monitored_count(), 1);

    detector.register_node("node-2");
    assert_eq!(detector.monitored_count(), 2);

    detector.unregister_node("node-1");
    assert_eq!(detector.monitored_count(), 1);

    detector.unregister_node("nonexistent");
    assert_eq!(detector.monitored_count(), 1);
}

#[test]
fn no_failure_when_heartbeats_are_fresh() {
    let registry = test_registry();
    let (detector, _rx) = FailureDetector::new(FailureDetectorConfig {
        heartbeat_interval: Duration::from_secs(5),
        failure_threshold: 3,
        check_interval: Duration::from_secs(1),
    });

    detector.register_node("peer-0");
    detector.record_heartbeat("peer-0");

    // Immediately check: heartbeat is fresh, no failures.
    let failed = detector.check_failures(&registry);
    assert!(failed.is_empty());
    assert!(!detector.is_node_failed("peer-0"));

    let peer = registry.get_node("peer-0").unwrap();
    assert_eq!(peer.status, NodeStatus::Online);
}

#[test]
fn detect_failure_after_timeout() {
    let registry = test_registry();
    let config = FailureDetectorConfig {
        heartbeat_interval: Duration::from_millis(10),
        failure_threshold: 1,
        check_interval: Duration::from_millis(5),
    };
    let (detector, mut rx) = FailureDetector::new(config);

    detector.register_node("peer-0");

    // Manually set the last heartbeat to the past by sleeping past the timeout.
    std::thread::sleep(Duration::from_millis(20));

    let failed = detector.check_failures(&registry);
    assert_eq!(failed, vec!["peer-0"]);
    assert!(detector.is_node_failed("peer-0"));

    // Registry should be updated
    let peer = registry.get_node("peer-0").unwrap();
    assert_eq!(peer.status, NodeStatus::Unreachable);

    // Should have received a failure event
    let event = rx.try_recv().unwrap();
    assert_eq!(event.node_id, "peer-0");
    assert_eq!(event.new_status, NodeStatus::Unreachable);
}

#[test]
fn no_duplicate_failure_notifications() {
    let registry = test_registry();
    let config = FailureDetectorConfig {
        heartbeat_interval: Duration::from_millis(10),
        failure_threshold: 1,
        check_interval: Duration::from_millis(5),
    };
    let (detector, mut rx) = FailureDetector::new(config);

    detector.register_node("peer-0");
    std::thread::sleep(Duration::from_millis(20));

    // First check triggers notification
    let failed1 = detector.check_failures(&registry);
    assert_eq!(failed1.len(), 1);

    // Second check should not re-notify
    let failed2 = detector.check_failures(&registry);
    assert!(failed2.is_empty());

    // Only one event should be in the channel
    assert!(rx.try_recv().is_ok());
    assert!(rx.try_recv().is_err());
}

#[test]
fn recovery_after_heartbeat() {
    let registry = test_registry();
    let config = FailureDetectorConfig {
        heartbeat_interval: Duration::from_millis(10),
        failure_threshold: 1,
        check_interval: Duration::from_millis(5),
    };
    let (detector, mut rx) = FailureDetector::new(config);

    detector.register_node("peer-0");
    std::thread::sleep(Duration::from_millis(20));

    // Detect failure
    detector.check_failures(&registry);
    let fail_event = rx.try_recv().unwrap();
    assert_eq!(fail_event.new_status, NodeStatus::Unreachable);

    // Simulate recovery
    detector.record_heartbeat("peer-0");
    detector.check_failures(&registry);

    let recovery_event = rx.try_recv().unwrap();
    assert_eq!(recovery_event.new_status, NodeStatus::Online);
    assert!(!detector.is_node_failed("peer-0"));

    let peer = registry.get_node("peer-0").unwrap();
    assert_eq!(peer.status, NodeStatus::Online);
}

#[test]
fn record_heartbeat_for_unknown_node_is_noop() {
    let (detector, _rx) = FailureDetector::new(FailureDetectorConfig::default());
    // Should not panic
    detector.record_heartbeat("unknown-node");
}

#[test]
fn is_node_failed_unknown_returns_false() {
    let (detector, _rx) = FailureDetector::new(FailureDetectorConfig::default());
    assert!(!detector.is_node_failed("unknown"));
}

#[test]
fn subscribe_receives_events() {
    let registry = test_registry();
    let config = FailureDetectorConfig {
        heartbeat_interval: Duration::from_millis(10),
        failure_threshold: 1,
        check_interval: Duration::from_millis(5),
    };
    let (detector, _rx1) = FailureDetector::new(config);
    let mut rx2 = detector.subscribe();

    detector.register_node("peer-0");
    std::thread::sleep(Duration::from_millis(20));
    detector.check_failures(&registry);

    let event = rx2.try_recv().unwrap();
    assert_eq!(event.node_id, "peer-0");
}

#[test]
fn default_config() {
    let config = FailureDetectorConfig::default();
    assert_eq!(config.heartbeat_interval, Duration::from_secs(5));
    assert_eq!(config.failure_threshold, 3);
    assert_eq!(config.check_interval, Duration::from_secs(2));
}
