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

use std::net::SocketAddr;
use std::time::Duration;

use super::*;
use crate::distributed::backpressure::{BackpressureConfig, BackpressureMonitor};
use crate::distributed::config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
use crate::distributed::metrics::{ClusterMetrics, NodeMetrics};
use crate::distributed::registry::{NodeRegistry, NodeStatus};
use crate::distributed::request_tracker::RequestId;

/// Helper: build a test cluster with 2 prefill nodes and 2 decode nodes.
fn test_cluster() -> (NodeRegistry, ClusterMetrics, BackpressureMonitor) {
    let config = ClusterConfig {
        cluster: ClusterMeta::default(),
        nodes: vec![
            NodeConfig {
                id: "prefill-0".to_string(),
                address: "127.0.0.1:9000".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources {
                    memory_bytes: 16_000_000_000,
                    compute_units: 8,
                },
            },
            NodeConfig {
                id: "prefill-1".to_string(),
                address: "127.0.0.1:9001".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources {
                    memory_bytes: 16_000_000_000,
                    compute_units: 8,
                },
            },
            NodeConfig {
                id: "decode-0".to_string(),
                address: "127.0.0.1:9002".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources {
                    memory_bytes: 16_000_000_000,
                    compute_units: 8,
                },
            },
            NodeConfig {
                id: "decode-1".to_string(),
                address: "127.0.0.1:9003".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources {
                    memory_bytes: 16_000_000_000,
                    compute_units: 8,
                },
            },
        ],
    };

    let registry = NodeRegistry::from_config(&config, "prefill-0");
    // Mark all nodes online.
    registry.set_node_status("prefill-0", NodeStatus::Online);
    registry.set_node_status("prefill-1", NodeStatus::Online);
    registry.set_node_status("decode-0", NodeStatus::Online);
    registry.set_node_status("decode-1", NodeStatus::Online);

    let cluster_metrics = ClusterMetrics::new();
    let backpressure = BackpressureMonitor::new(BackpressureConfig::default());

    (registry, cluster_metrics, backpressure)
}

/// Helper: create a router with default config.
fn test_router() -> RequestRouter {
    let (registry, metrics, bp) = test_cluster();
    RequestRouter::new(RouterConfig::default(), registry, metrics, bp)
}

// ── Basic routing ────────────────────────────────────────────────────

#[test]
fn route_to_prefill_selects_prefill_node() {
    let router = test_router();
    let req_id = RequestId::new();
    let node = router.route_to_prefill(req_id.clone(), 100).unwrap();

    assert!(
        node.starts_with("prefill-"),
        "expected prefill node, got {node}"
    );

    // Request should be tracked.
    let phase = router.get_phase(&req_id).unwrap();
    assert!(
        matches!(phase, RequestPhase::Prefilling { .. }),
        "expected Prefilling phase, got {phase}"
    );
}

#[test]
fn route_to_decode_selects_decode_node() {
    let router = test_router();
    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    let decode_node = router.route_to_decode(&req_id).unwrap();
    assert!(
        decode_node.starts_with("decode-"),
        "expected decode node, got {decode_node}"
    );

    let phase = router.get_phase(&req_id).unwrap();
    assert!(
        matches!(phase, RequestPhase::TransferringCache { .. }),
        "expected TransferringCache phase, got {phase}"
    );
}

// ── Round-robin ──────────────────────────────────────────────────────

#[test]
fn round_robin_distributes_evenly() {
    let (registry, metrics, bp) = test_cluster();
    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..10 {
        let req_id = RequestId::new();
        let node = router.route_to_prefill(req_id, 100).unwrap();
        *counts.entry(node).or_insert(0) += 1;
    }

    // Both prefill nodes should have been selected.
    assert_eq!(counts.len(), 2, "expected 2 distinct prefill nodes");
    assert_eq!(counts.get("prefill-0"), Some(&5));
    assert_eq!(counts.get("prefill-1"), Some(&5));
}

// ── Least-loaded ─────────────────────────────────────────────────────

#[test]
fn least_loaded_prefers_idle_node() {
    let (registry, cluster_metrics, bp) = test_cluster();

    // Give prefill-0 a high load, prefill-1 low load.
    cluster_metrics.update(
        "prefill-0",
        NodeMetrics {
            active_requests: 10,
            ..Default::default()
        },
    );
    cluster_metrics.update(
        "prefill-1",
        NodeMetrics {
            active_requests: 1,
            ..Default::default()
        },
    );

    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::LeastLoaded,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, cluster_metrics, bp);

    let req_id = RequestId::new();
    let node = router.route_to_prefill(req_id, 100).unwrap();
    assert_eq!(node, "prefill-1");
}

// ── Memory-aware ─────────────────────────────────────────────────────

#[test]
fn memory_aware_prefers_node_with_more_free_memory() {
    let (registry, cluster_metrics, bp) = test_cluster();

    cluster_metrics.update(
        "decode-0",
        NodeMetrics {
            memory_used_bytes: 14_000_000_000,
            memory_total_bytes: 16_000_000_000,
            ..Default::default()
        },
    );
    cluster_metrics.update(
        "decode-1",
        NodeMetrics {
            memory_used_bytes: 4_000_000_000,
            memory_total_bytes: 16_000_000_000,
            ..Default::default()
        },
    );

    let config = RouterConfig {
        decode_strategy: DisaggRoutingStrategy::MemoryAware,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, cluster_metrics, bp);

    // First route to prefill (needed for decode routing).
    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    let decode_node = router.route_to_decode(&req_id).unwrap();
    assert_eq!(decode_node, "decode-1");
}

// ── Prompt-length-aware ──────────────────────────────────────────────

#[test]
fn prompt_length_aware_routes_long_prompt_to_memory_rich_node() {
    let (registry, cluster_metrics, bp) = test_cluster();

    cluster_metrics.update(
        "prefill-0",
        NodeMetrics {
            active_requests: 2,
            memory_used_bytes: 14_000_000_000,
            memory_total_bytes: 16_000_000_000,
            ..Default::default()
        },
    );
    cluster_metrics.update(
        "prefill-1",
        NodeMetrics {
            active_requests: 5,
            memory_used_bytes: 4_000_000_000,
            memory_total_bytes: 16_000_000_000,
            ..Default::default()
        },
    );

    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::PromptLengthAware,
        long_prompt_threshold: 1024,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, cluster_metrics, bp);

    // Long prompt -> memory-aware selection.
    let req_id = RequestId::new();
    let node = router.route_to_prefill(req_id, 4096).unwrap();
    assert_eq!(
        node, "prefill-1",
        "long prompt should go to memory-rich node"
    );

    // Short prompt -> least-loaded selection.
    let req_id2 = RequestId::new();
    let node2 = router.route_to_prefill(req_id2, 100).unwrap();
    assert_eq!(
        node2, "prefill-0",
        "short prompt should go to least-loaded node"
    );
}

// ── Backpressure ─────────────────────────────────────────────────────

#[test]
fn backpressure_rejects_when_queue_full() {
    let (registry, metrics, bp) = test_cluster();

    let config = RouterConfig {
        prefill_queue_capacity: 2,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    // Mark all prefill nodes as critical so requests get queued.
    router
        .backpressure
        .update_from_metrics("prefill-0", 20, 0.99);
    router
        .backpressure
        .update_from_metrics("prefill-1", 20, 0.99);

    // First 2 requests get queued.
    let r1 = RequestId::new();
    let _ = router.route_to_prefill(r1, 100); // will queue or error
    let r2 = RequestId::new();
    let _ = router.route_to_prefill(r2, 100);

    // Third request should be rejected.
    let r3 = RequestId::new();
    let result = router.route_to_prefill(r3, 100);
    assert!(result.is_err(), "expected rejection when queue is full");
}

#[test]
fn apply_backpressure_returns_accept_when_nodes_available() {
    let router = test_router();
    assert_eq!(router.apply_backpressure(), BackpressureAction::Accept);
}

// ── Phase transitions ────────────────────────────────────────────────

#[test]
fn phase_transitions_work() {
    let router = test_router();
    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    // Move to decode.
    let decode_node = router.route_to_decode(&req_id).unwrap();

    // Mark decoding.
    assert!(router.mark_decoding(&req_id, &decode_node));

    let phase = router.get_phase(&req_id).unwrap();
    assert!(matches!(phase, RequestPhase::Decoding { .. }));

    // Mark completed.
    assert!(router.mark_completed(&req_id));

    let phase = router.get_phase(&req_id).unwrap();
    assert_eq!(phase, RequestPhase::Completed);

    // Cannot transition again after completion.
    assert!(!router.mark_failed(&req_id, "late error"));
}

// ── Node failure handling ────────────────────────────────────────────

#[test]
fn handle_node_failure_reroutes_prefilling_requests() {
    let router = test_router();

    // Route a request to prefill-0.
    let req_id = RequestId::new();
    let node = router.route_to_prefill(req_id.clone(), 100).unwrap();

    if node == "prefill-0" {
        let (rerouted, failed) = router.handle_node_failure("prefill-0");
        assert_eq!(rerouted, 1);
        assert_eq!(failed, 0);

        let tracked = router.get_tracked_request(&req_id).unwrap();
        // Should be re-routed to prefill-1 or re-queued.
        assert_ne!(
            tracked.prefill_node.as_deref(),
            Some("prefill-0"),
            "should not remain on failed node"
        );
        assert_eq!(tracked.retry_count, 1);
    }
}

#[test]
fn handle_node_failure_fails_request_when_no_alternative() {
    // Create a cluster with only one decode node.
    let config = ClusterConfig {
        cluster: ClusterMeta::default(),
        nodes: vec![
            NodeConfig {
                id: "prefill-0".to_string(),
                address: "127.0.0.1:9000".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "decode-0".to_string(),
                address: "127.0.0.1:9001".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
        ],
    };

    let registry = NodeRegistry::from_config(&config, "prefill-0");
    registry.set_node_status("prefill-0", NodeStatus::Online);
    registry.set_node_status("decode-0", NodeStatus::Online);

    let cluster_metrics = ClusterMetrics::new();
    let bp = BackpressureMonitor::new(BackpressureConfig::default());
    let router = RequestRouter::new(RouterConfig::default(), registry, cluster_metrics, bp);

    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();
    router.route_to_decode(&req_id).unwrap();
    assert!(router.mark_decoding(&req_id, "decode-0"));

    let (rerouted, failed) = router.handle_node_failure("decode-0");
    assert_eq!(rerouted, 0);
    assert_eq!(failed, 1);

    let phase = router.get_phase(&req_id).unwrap();
    assert!(matches!(phase, RequestPhase::Failed { .. }));
}

// ── Metrics ──────────────────────────────────────────────────────────

#[test]
fn metrics_are_updated() {
    let router = test_router();
    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    let m = router.metrics();
    assert_eq!(m.total_requests, 1);
    assert_eq!(m.routing_decisions, 1);
    assert_eq!(m.active_prefills, 1);

    router.route_to_decode(&req_id).unwrap();
    let m = router.metrics();
    assert_eq!(m.routing_decisions, 2);

    assert!(router.mark_decoding(&req_id, "decode-0"));
    assert!(router.mark_completed(&req_id));
    let m = router.metrics();
    assert_eq!(m.total_completed, 1);
}

// ── Timeout collection ───────────────────────────────────────────────

#[test]
fn collect_timed_out_finds_stale_requests() {
    let (registry, metrics, bp) = test_cluster();
    let config = RouterConfig {
        phase_timeout: Duration::from_millis(1),
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    // Wait for timeout.
    std::thread::sleep(Duration::from_millis(5));

    let timed_out = router.collect_timed_out();
    assert!(
        !timed_out.is_empty(),
        "expected at least one timed-out request"
    );
    assert!(timed_out.contains(&req_id));
}

// ── Purge terminal ───────────────────────────────────────────────────

#[test]
fn purge_terminal_removes_old_completed() {
    let router = test_router();
    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();
    router.route_to_decode(&req_id).unwrap();
    assert!(router.mark_decoding(&req_id, "decode-0"));
    assert!(router.mark_completed(&req_id));

    // Purge with zero max_age removes everything terminal.
    let removed = router.purge_terminal(Duration::ZERO);
    assert_eq!(removed, 1);

    assert!(router.get_phase(&req_id).is_none());
}

// ── Strategy display ─────────────────────────────────────────────────

#[test]
fn strategy_display() {
    assert_eq!(DisaggRoutingStrategy::RoundRobin.to_string(), "round-robin");
    assert_eq!(
        DisaggRoutingStrategy::LeastLoaded.to_string(),
        "least-loaded"
    );
    assert_eq!(
        DisaggRoutingStrategy::MemoryAware.to_string(),
        "memory-aware"
    );
    assert_eq!(
        DisaggRoutingStrategy::PromptLengthAware.to_string(),
        "prompt-length-aware"
    );
}

// ── Phase display ────────────────────────────────────────────────────

#[test]
fn phase_display_and_terminal() {
    assert_eq!(RequestPhase::Queued.to_string(), "queued");
    assert!(!RequestPhase::Queued.is_terminal());
    assert!(RequestPhase::Completed.is_terminal());
    assert!(
        RequestPhase::Failed {
            reason: "test".into()
        }
        .is_terminal()
    );
}

// ── NodeLoadInfo helpers ─────────────────────────────────────────────

#[test]
fn node_load_info_utilization() {
    let info = NodeLoadInfo {
        node_id: "n".to_string(),
        role: NodeRole::Prefill,
        active_requests: 0,
        memory_used_bytes: 8_000_000_000,
        memory_total_bytes: 16_000_000_000,
        is_online: true,
        load_level: None,
    };

    assert!((info.memory_utilization() - 0.5).abs() < f64::EPSILON);
    assert_eq!(info.free_memory(), 8_000_000_000);

    // Zero total memory.
    let zero = NodeLoadInfo {
        memory_total_bytes: 0,
        ..info.clone()
    };
    assert_eq!(zero.memory_utilization(), 0.0);
}

// ── TransferringCache failure re-queues for prefill ─────────────────

#[test]
fn handle_node_failure_during_transfer_requeues_for_prefill() {
    let router = test_router();

    let req_id = RequestId::new();
    router.route_to_prefill(req_id.clone(), 100).unwrap();

    // Move to TransferringCache (prefill complete, cache being sent).
    let decode_node = router.route_to_decode(&req_id).unwrap();

    let phase = router.get_phase(&req_id).unwrap();
    assert!(
        matches!(phase, RequestPhase::TransferringCache { .. }),
        "expected TransferringCache, got {phase}"
    );

    // Simulate decode node failure during transfer.
    let (rerouted, failed) = router.handle_node_failure(&decode_node);
    assert_eq!(rerouted, 1);
    assert_eq!(failed, 0);

    // Request should be re-queued for prefill (not jumped to Decoding),
    // because the KV cache transfer was incomplete.
    let tracked = router.get_tracked_request(&req_id).unwrap();
    assert!(
        matches!(
            tracked.phase,
            RequestPhase::Prefilling { .. } | RequestPhase::Queued
        ),
        "expected Prefilling or Queued after TransferringCache failure, got {}",
        tracked.phase
    );
    assert_eq!(tracked.retry_count, 1);
    // decode_node should be cleared since we're re-doing prefill.
    assert!(
        tracked.decode_node.is_none(),
        "decode_node should be cleared after re-queuing for prefill"
    );
}

// ── Round-robin distribution during node failure ────────────────────

#[test]
fn handle_node_failure_distributes_rerouted_requests() {
    // Create a cluster with 1 prefill node that will fail and 2 alternatives.
    let config = ClusterConfig {
        cluster: ClusterMeta::default(),
        nodes: vec![
            NodeConfig {
                id: "prefill-0".to_string(),
                address: "127.0.0.1:9000".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "prefill-1".to_string(),
                address: "127.0.0.1:9001".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "prefill-2".to_string(),
                address: "127.0.0.1:9002".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Prefill,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
            NodeConfig {
                id: "decode-0".to_string(),
                address: "127.0.0.1:9003".parse::<SocketAddr>().unwrap(),
                role: NodeRole::Decode,
                stage: None,
                rank: None,
                resources: NodeResources::default(),
            },
        ],
    };

    let registry = NodeRegistry::from_config(&config, "prefill-0");
    registry.set_node_status("prefill-0", NodeStatus::Online);
    registry.set_node_status("prefill-1", NodeStatus::Online);
    registry.set_node_status("prefill-2", NodeStatus::Online);
    registry.set_node_status("decode-0", NodeStatus::Online);

    let cluster_metrics = ClusterMetrics::new();
    let bp = BackpressureMonitor::new(BackpressureConfig::default());

    let router_config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(router_config, registry, cluster_metrics, bp);

    // Route 4 requests to prefill-0 by manually setting them.
    let mut req_ids = Vec::new();
    for _ in 0..4 {
        let req_id = RequestId::new();
        // Route normally first, then we only care about ones on prefill-0.
        let _ = router.route_to_prefill(req_id.clone(), 100);
        req_ids.push(req_id);
    }

    // Force all requests onto prefill-0 for the test.
    {
        let mut requests = router.requests.write().unwrap();
        for (_, tracked) in requests.iter_mut() {
            tracked.phase = RequestPhase::Prefilling {
                node_id: "prefill-0".to_string(),
            };
            tracked.prefill_node = Some("prefill-0".to_string());
        }
    }

    // Now fail prefill-0 -- requests should be distributed across prefill-1 and prefill-2.
    let (rerouted, failed) = router.handle_node_failure("prefill-0");
    assert_eq!(rerouted, 4);
    assert_eq!(failed, 0);

    // Verify distribution: should not all go to the same node.
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for req_id in &req_ids {
        let tracked = router.get_tracked_request(req_id).unwrap();
        if let RequestPhase::Prefilling { node_id } = &tracked.phase {
            *node_counts.entry(node_id.clone()).or_insert(0) += 1;
        }
    }

    assert!(
        node_counts.len() == 2,
        "expected requests distributed across 2 alternative nodes, got {node_counts:?}"
    );
    assert_eq!(
        node_counts.get("prefill-1"),
        Some(&2),
        "expected 2 on prefill-1, got {node_counts:?}"
    );
    assert_eq!(
        node_counts.get("prefill-2"),
        Some(&2),
        "expected 2 on prefill-2, got {node_counts:?}"
    );
}

// ── Router-driven decode selection (issue #201) ─────────────────────

/// With round-robin decode routing (the strategy the router front uses), the
/// router spreads decode selections across the decode pool instead of always
/// picking the first node, so a router-chosen `decode_target` balances the
/// decode pool just like the prefill pool.
#[test]
fn route_to_decode_round_robin_distributes_across_decode_nodes() {
    let (registry, metrics, bp) = test_cluster();
    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        decode_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..6 {
        let req_id = RequestId::new();
        router.route_to_prefill(req_id.clone(), 100).unwrap();
        let decode = router.route_to_decode(&req_id).unwrap();
        *counts.entry(decode).or_insert(0) += 1;
    }

    assert_eq!(
        counts.len(),
        2,
        "expected both decode nodes used, got {counts:?}"
    );
    assert_eq!(counts.get("decode-0"), Some(&3));
    assert_eq!(counts.get("decode-1"), Some(&3));
}

/// A prefill node marked `Unreachable` in the registry is skipped by
/// `route_to_prefill`: all requests go to the surviving node. This is the
/// registry-status mechanism the router front's transport-error path and health
/// monitor drive.
#[test]
fn unreachable_prefill_node_is_skipped() {
    let (registry, metrics, bp) = test_cluster();
    registry.set_node_status("prefill-0", NodeStatus::Unreachable);
    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    for _ in 0..4 {
        let req_id = RequestId::new();
        let node = router.route_to_prefill(req_id, 100).unwrap();
        assert_eq!(node, "prefill-1", "must skip the unreachable prefill node");
    }
}

/// A decode node marked `Unreachable` is skipped by `route_to_decode`.
#[test]
fn unreachable_decode_node_is_skipped() {
    let (registry, metrics, bp) = test_cluster();
    registry.set_node_status("decode-0", NodeStatus::Unreachable);
    let config = RouterConfig {
        decode_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    for _ in 0..4 {
        let req_id = RequestId::new();
        router.route_to_prefill(req_id.clone(), 100).unwrap();
        let decode = router.route_to_decode(&req_id).unwrap();
        assert_eq!(decode, "decode-1", "must skip the unreachable decode node");
    }
}

/// Admission control returns `Queue` when every prefill node is unreachable, so
/// the router front rejects with HTTP 503 instead of attempting a doomed
/// dispatch (issue #201). With at least one node available it returns `Accept`.
#[test]
fn apply_backpressure_queues_when_all_prefill_unreachable() {
    let (registry, metrics, bp) = test_cluster();
    let router = RequestRouter::new(RouterConfig::default(), registry.clone(), metrics, bp);

    assert_eq!(router.apply_backpressure(), BackpressureAction::Accept);

    registry.set_node_status("prefill-0", NodeStatus::Unreachable);
    registry.set_node_status("prefill-1", NodeStatus::Unreachable);
    assert_eq!(
        router.apply_backpressure(),
        BackpressureAction::Queue,
        "all prefill nodes down must not be Accept"
    );
}

/// The router-front failover sequence: send to a prefill node fails, so the
/// node is marked unreachable and `handle_node_failure` re-routes its in-flight
/// requests; a subsequent `route_to_prefill` then lands on the survivor. This
/// is the unit-level mirror of `start_handoff`'s retry loop.
#[test]
fn failover_reroutes_to_survivor_after_marking_unreachable() {
    let (registry, metrics, bp) = test_cluster();
    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry.clone(), metrics, bp);

    // First request lands on one of the two nodes (registry iteration order is
    // not deterministic, so capture whichever it was rather than assuming).
    let first = RequestId::new();
    let failed_node = router.route_to_prefill(first.clone(), 100).unwrap();
    let survivor = if failed_node == "prefill-0" {
        "prefill-1"
    } else {
        "prefill-0"
    };

    // Simulate the send to the chosen node failing: mark it down and re-route.
    registry.set_node_status(&failed_node, NodeStatus::Unreachable);
    let (rerouted, failed) = router.handle_node_failure(&failed_node);
    assert_eq!(
        failed, 0,
        "a healthy alternative exists, nothing should fail"
    );
    assert!(rerouted >= 1, "the in-flight request should be re-routed");

    // The in-flight request must no longer point at the dead node.
    let tracked = router.get_tracked_request(&first).unwrap();
    assert_ne!(tracked.prefill_node.as_deref(), Some(failed_node.as_str()));

    // Every fresh request now routes only to the survivor.
    for _ in 0..3 {
        let next = RequestId::new();
        let node = router.route_to_prefill(next, 100).unwrap();
        assert_eq!(node, survivor, "must route to the surviving node");
    }
}

// ── Auto-purge on capacity ──────────────────────────────────────────

#[test]
fn auto_purge_removes_old_terminal_on_overflow() {
    let (registry, metrics, bp) = test_cluster();
    let config = RouterConfig {
        max_tracked_requests: 3,
        auto_purge_age: Duration::ZERO, // purge immediately
        ..Default::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    // Route and complete 3 requests.
    for _ in 0..3 {
        let req_id = RequestId::new();
        router.route_to_prefill(req_id.clone(), 100).unwrap();
        assert!(router.mark_completed(&req_id));
    }

    // 4th request should trigger auto-purge of completed entries.
    let req_id4 = RequestId::new();
    let result = router.route_to_prefill(req_id4.clone(), 100);
    assert!(result.is_ok(), "should succeed after auto-purge");

    // Only the new active request should remain (completed ones purged).
    let m = router.metrics();
    assert_eq!(m.active_prefills, 1);
}
