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
use crate::distributed::metrics::{MetricsConfig, PagedKvMetrics};
use crate::distributed::registry::NodeStatus;

fn test_config() -> HeartbeatConfig {
    HeartbeatConfig {
        interval: Duration::from_millis(50),
        failure_threshold: 3,
        check_interval: Duration::from_millis(25),
        include_metrics: true,
    }
}

fn test_registry() -> NodeRegistry {
    let config = ClusterConfig::from_cli(
        "local".to_string(),
        "127.0.0.1:8080".parse().unwrap(),
        NodeRole::Hybrid,
        vec![
            "127.0.0.1:8081".parse().unwrap(),
            "127.0.0.1:8082".parse().unwrap(),
        ],
    );
    let registry = NodeRegistry::from_config(&config, "local");
    registry.set_node_status("peer-0", NodeStatus::Online);
    registry.set_node_status("peer-1", NodeStatus::Online);
    registry
}

#[test]
fn heartbeat_payload_serialization() {
    let payload = HeartbeatPayload {
        node_id: "node-0".to_string(),
        sequence: 42,
        metrics: None,
    };
    let bytes = serde_json::to_vec(&payload).unwrap();
    let restored: HeartbeatPayload = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored.node_id, "node-0");
    assert_eq!(restored.sequence, 42);
    assert!(restored.metrics.is_none());
}

#[test]
fn heartbeat_payload_with_metrics() {
    let payload = HeartbeatPayload {
        node_id: "node-0".to_string(),
        sequence: 1,
        metrics: Some(NodeMetrics {
            throughput_tokens_per_sec: 100.0,
            total_tokens: 500,
            paged_kv: Some(PagedKvMetrics {
                block_size: 32,
                allocated_blocks: 12,
                live_blocks: 10,
                free_blocks: 2,
                bytes_reserved: 8192,
                bytes_in_use: 6144,
            }),
            paged_decode_fallbacks: 1,
            ..Default::default()
        }),
    };
    let bytes = serde_json::to_vec(&payload).unwrap();
    let restored: HeartbeatPayload = serde_json::from_slice(&bytes).unwrap();
    assert!(restored.metrics.is_some());
    let m = restored.metrics.unwrap();
    assert_eq!(m.throughput_tokens_per_sec, 100.0);
    assert_eq!(m.total_tokens, 500);
    assert_eq!(m.paged_kv.unwrap().block_size, 32);
    assert_eq!(m.paged_decode_fallbacks, 1);
}

#[test]
fn service_creation_registers_peers() {
    let registry = test_registry();
    let config = test_config();
    let service = HeartbeatService::new(config, registry, None);

    // Should have registered the two peers for monitoring
    assert_eq!(service.failure_detector().monitored_count(), 2);
}

#[test]
fn process_incoming_heartbeat_records() {
    let registry = test_registry();
    let config = test_config();
    let service = HeartbeatService::new(config, registry.clone(), None);

    let payload = HeartbeatPayload {
        node_id: "peer-0".to_string(),
        sequence: 1,
        metrics: None,
    };
    let bytes = serde_json::to_vec(&payload).unwrap();

    service.process_incoming_heartbeat(&bytes);
    assert!(!service.failure_detector().is_node_failed("peer-0"));
}

#[test]
fn process_invalid_heartbeat_does_not_panic() {
    let registry = test_registry();
    let service = HeartbeatService::new(test_config(), registry, None);
    // Should not panic on invalid payload
    service.process_incoming_heartbeat(b"not valid json");
}

#[test]
fn stop_cancels_service() {
    let registry = test_registry();
    let service = HeartbeatService::new(test_config(), registry, None);

    assert!(!service.is_stopped());
    service.stop();
    assert!(service.is_stopped());
}

#[test]
fn subscribe_events() {
    let registry = test_registry();
    let config = HeartbeatConfig {
        interval: Duration::from_millis(10),
        failure_threshold: 1,
        check_interval: Duration::from_millis(5),
        include_metrics: false,
    };
    let service = HeartbeatService::new(config, registry.clone(), None);
    let mut rx = service.subscribe_events();

    // Wait for heartbeat timeout
    std::thread::sleep(Duration::from_millis(20));
    service.failure_detector().check_failures(&registry);

    // Should receive failure event for at least one peer
    let event = rx.try_recv();
    assert!(event.is_ok());
}

#[test]
fn default_config_values() {
    let config = HeartbeatConfig::default();
    assert_eq!(config.interval, Duration::from_secs(5));
    assert_eq!(config.failure_threshold, 3);
    assert_eq!(config.check_interval, Duration::from_secs(2));
    assert!(config.include_metrics);
}

#[test]
fn service_with_metrics_collector() {
    let registry = test_registry();
    let collector = MetricsCollector::new(MetricsConfig::default());
    collector.record_tokens(100);

    let config = test_config();
    let service = HeartbeatService::new(config, registry, Some(collector));

    // The metrics collector should be wired up
    assert_eq!(service.failure_detector().monitored_count(), 2);
}
