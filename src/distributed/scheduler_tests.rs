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

use std::net::SocketAddr;
use std::sync::Arc;

use super::*;
use crate::distributed::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
use crate::distributed::handoff_queue::OverflowPolicy;
use crate::distributed::metrics::{ClusterMetrics, NodeMetrics};
use crate::distributed::registry::NodeRegistry;

fn test_cluster_config() -> ClusterConfig {
    ClusterConfig {
        cluster: ClusterMeta::default(),
        nodes: vec![
            NodeConfig {
                id: "node-0".to_string(),
                address: "127.0.0.1:8080".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "node-1".to_string(),
                address: "127.0.0.1:8081".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "node-2".to_string(),
                address: "127.0.0.1:8082".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Hybrid,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
        ],
    }
}

fn test_scheduler() -> Scheduler {
    let config = test_cluster_config();
    let registry = NodeRegistry::from_config(&config, "node-0");
    // Mark all nodes as Online for testing.
    registry.set_node_status("node-1", NodeStatus::Online);
    registry.set_node_status("node-2", NodeStatus::Online);

    let metrics = ClusterMetrics::new();
    Scheduler::new(SchedulerConfig::default(), registry, metrics)
}

#[test]
fn scheduler_default_mode_is_centralized() {
    let scheduler = test_scheduler();
    assert_eq!(scheduler.mode(), CoordinationMode::Centralized);
}

#[test]
fn scheduler_default_strategy_is_load_balanced() {
    let scheduler = test_scheduler();
    assert_eq!(scheduler.strategy_name(), "load-balanced");
}

#[test]
fn submit_request_assigns_id_and_routes() {
    let scheduler = test_scheduler();
    let (request_id, decision) = scheduler.submit_request(None, None, None).unwrap();

    assert!(!request_id.as_str().is_empty());
    assert!(!decision.target_node.is_empty());

    // Request should be in Processing state.
    let state = scheduler.get_request_state(&request_id).unwrap();
    assert!(matches!(state, RequestState::Processing { .. }));
}

#[test]
fn submit_request_with_role_preference() {
    let scheduler = test_scheduler();
    let (_, decision) = scheduler
        .submit_request(Some(NodeRole::Prefill), None, None)
        .unwrap();

    // Should route to node-0 (Prefill) or node-2 (Hybrid).
    assert!(
        decision.target_node == "node-0" || decision.target_node == "node-2",
        "expected prefill-capable node, got {}",
        decision.target_node
    );
}

#[test]
fn submit_request_with_decode_role() {
    let scheduler = test_scheduler();
    let (_, decision) = scheduler
        .submit_request(Some(NodeRole::Decode), None, None)
        .unwrap();

    assert!(
        decision.target_node == "node-1" || decision.target_node == "node-2",
        "expected decode-capable node, got {}",
        decision.target_node
    );
}

#[test]
fn complete_request_lifecycle() {
    let scheduler = test_scheduler();
    let (request_id, _) = scheduler.submit_request(None, None, None).unwrap();

    assert!(scheduler.complete_request(&request_id));
    let state = scheduler.get_request_state(&request_id).unwrap();
    assert_eq!(state, RequestState::Completed);
}

#[test]
fn fail_request_lifecycle() {
    let scheduler = test_scheduler();
    let (request_id, _) = scheduler.submit_request(None, None, None).unwrap();

    assert!(scheduler.fail_request(&request_id, "timeout"));
    let state = scheduler.get_request_state(&request_id).unwrap();
    assert!(matches!(state, RequestState::Failed { reason } if reason == "timeout"));
}

#[test]
fn handoff_lifecycle() {
    let scheduler = test_scheduler();
    let (request_id, decision) = scheduler.submit_request(None, None, None).unwrap();

    let from = &decision.target_node;
    let to = if from == "node-0" { "node-1" } else { "node-0" };

    // Initiate handoff.
    scheduler
        .initiate_handoff(&request_id, from, to, vec![42])
        .unwrap();

    let state = scheduler.get_request_state(&request_id).unwrap();
    assert!(matches!(state, RequestState::Handoff { .. }));

    // Complete handoff.
    assert!(scheduler.complete_handoff(&request_id, to));
    let state = scheduler.get_request_state(&request_id).unwrap();
    assert!(matches!(state, RequestState::Processing { node_id } if node_id == to));

    // Complete the request.
    assert!(scheduler.complete_request(&request_id));
}

#[test]
fn handoff_queue_receives_item() {
    let scheduler = test_scheduler();
    let (request_id, _) = scheduler.submit_request(None, None, None).unwrap();

    scheduler
        .initiate_handoff(&request_id, "node-0", "node-1", vec![1, 2, 3])
        .unwrap();

    let queue = scheduler.queues().get("node-0->node-1").unwrap();
    assert_eq!(queue.len(), 1);

    let item = queue.dequeue().unwrap();
    assert_eq!(item.request_id, request_id);
    assert_eq!(item.payload, vec![1, 2, 3]);
}

#[test]
fn backpressure_updates_and_queries() {
    let scheduler = test_scheduler();

    assert!(!scheduler.is_node_pressured("node-0"));

    // Push node-0 to critical load.
    scheduler.update_node_load("node-0", 20, 0.5);
    assert!(scheduler.is_node_pressured("node-0"));
}

#[test]
fn backpressure_redirect_avoids_critical_node() {
    let config = test_cluster_config();
    let registry = NodeRegistry::from_config(&config, "node-0");
    registry.set_node_status("node-1", NodeStatus::Online);
    registry.set_node_status("node-2", NodeStatus::Online);

    let metrics = ClusterMetrics::new();
    // Set node-0 and node-1 loads.
    metrics.update(
        "node-0",
        NodeMetrics {
            active_requests: 20,
            ..NodeMetrics::default()
        },
    );
    metrics.update(
        "node-1",
        NodeMetrics {
            active_requests: 1,
            ..NodeMetrics::default()
        },
    );

    let scheduler_config = SchedulerConfig {
        skip_pressured_nodes: true,
        ..SchedulerConfig::default()
    };
    let scheduler = Scheduler::new(scheduler_config, registry, metrics);

    // Mark node-0 as critical.
    scheduler.update_node_load("node-0", 20, 0.5);

    // Submit should route away from node-0.
    let (_, decision) = scheduler.submit_request(None, None, None).unwrap();
    assert_ne!(
        decision.target_node, "node-0",
        "should not route to node under critical load"
    );
}

#[test]
fn custom_routing_strategy() {
    let config = test_cluster_config();
    let registry = NodeRegistry::from_config(&config, "node-0");
    registry.set_node_status("node-1", NodeStatus::Online);
    registry.set_node_status("node-2", NodeStatus::Online);

    let metrics = ClusterMetrics::new();
    let strategy = Arc::new(RoleBasedRouter);

    let scheduler =
        Scheduler::with_strategy(SchedulerConfig::default(), registry, metrics, strategy);

    assert_eq!(scheduler.strategy_name(), "role-based");

    let (_, decision) = scheduler
        .submit_request(Some(NodeRole::Decode), None, None)
        .unwrap();

    assert!(decision.target_node == "node-1" || decision.target_node == "node-2",);
}

#[test]
fn active_request_count() {
    let scheduler = test_scheduler();
    assert_eq!(scheduler.active_request_count(), 0);

    let (id1, _) = scheduler.submit_request(None, None, None).unwrap();
    let (_id2, _) = scheduler.submit_request(None, None, None).unwrap();
    assert_eq!(scheduler.active_request_count(), 2);

    assert!(scheduler.complete_request(&id1));
    assert_eq!(scheduler.active_request_count(), 1);
    assert_eq!(scheduler.tracked_request_count(), 2);
}

#[test]
fn coordination_mode_display() {
    assert_eq!(format!("{}", CoordinationMode::Centralized), "centralized");
    assert_eq!(format!("{}", CoordinationMode::Distributed), "distributed");
}

#[test]
fn set_strategy_changes_behavior() {
    let config = test_cluster_config();
    let registry = NodeRegistry::from_config(&config, "node-0");
    registry.set_node_status("node-1", NodeStatus::Online);
    registry.set_node_status("node-2", NodeStatus::Online);

    let metrics = ClusterMetrics::new();
    let mut scheduler = Scheduler::new(SchedulerConfig::default(), registry, metrics);

    assert_eq!(scheduler.strategy_name(), "load-balanced");

    scheduler.set_strategy(Arc::new(
        crate::distributed::routing::RoundRobinRouter::new(),
    ));
    assert_eq!(scheduler.strategy_name(), "round-robin");
}

#[test]
fn handoff_rejected_when_queue_full() {
    let config = test_cluster_config();
    let registry = NodeRegistry::from_config(&config, "node-0");
    registry.set_node_status("node-1", NodeStatus::Online);
    registry.set_node_status("node-2", NodeStatus::Online);

    let metrics = ClusterMetrics::new();
    let scheduler_config = SchedulerConfig {
        handoff_queue: HandoffQueueConfig {
            capacity: 1,
            overflow_policy: OverflowPolicy::Reject,
        },
        ..SchedulerConfig::default()
    };
    let scheduler = Scheduler::new(scheduler_config, registry, metrics);

    // Submit two requests.
    let (id1, _) = scheduler.submit_request(None, None, None).unwrap();
    let (id2, _) = scheduler.submit_request(None, None, None).unwrap();

    // First handoff succeeds.
    scheduler
        .initiate_handoff(&id1, "node-0", "node-1", vec![])
        .unwrap();

    // Second handoff to same queue should be rejected.
    let result = scheduler.initiate_handoff(&id2, "node-0", "node-1", vec![]);
    assert!(result.is_err());
}
