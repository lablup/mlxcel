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

//! Integration tests for multi-node distributed scenarios.
//!
//! These tests exercise the full stack: mock transport, node registry,
//! heartbeat service, failure detection, and request routing.
//!
//! All tests run in-process using MockTransport -- no network, no special
//! hardware, fully CI-compatible.

use std::time::Duration;

use bytes::Bytes;

use mlxcel::distributed::config::NodeRole;
use mlxcel::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use mlxcel::distributed::registry::NodeStatus;
use mlxcel::distributed::test_harness::{
    PerfBaseline, TestCluster, TestClusterConfig, measure_mock_transfer_latency,
    measure_serialization_roundtrip,
};
use mlxcel::distributed::transport::{Transport, TransportMessage};

// ---------------------------------------------------------------------------
// Node join scenarios
// ---------------------------------------------------------------------------

#[tokio::test]
async fn node_join_two_node_cluster() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    // Start with one node.
    cluster.add_node("node-0", NodeRole::Prefill).await;
    assert_eq!(cluster.node_count(), 1);

    // Add a second node.
    cluster.add_node("node-1", NodeRole::Decode).await;
    assert_eq!(cluster.node_count(), 2);

    // Both nodes should see each other as Online.
    let n0 = cluster.get_node("node-0").unwrap();
    let n1 = cluster.get_node("node-1").unwrap();

    let n0_view = n0.registry.get_node("node-1").unwrap();
    assert_eq!(n0_view.status, NodeStatus::Online);
    assert_eq!(n0_view.config.role, NodeRole::Decode);

    let n1_view = n1.registry.get_node("node-0").unwrap();
    assert_eq!(n1_view.status, NodeStatus::Online);
    assert_eq!(n1_view.config.role, NodeRole::Prefill);

    cluster.shutdown().await;
}

#[tokio::test]
async fn node_join_third_node_discovers_existing() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("node-0", NodeRole::Prefill).await;
    cluster.add_node("node-1", NodeRole::Decode).await;

    // Add third node -- should see both existing nodes.
    cluster.add_node("node-2", NodeRole::Hybrid).await;
    assert_eq!(cluster.node_count(), 3);

    let n2 = cluster.get_node("node-2").unwrap();
    assert_eq!(n2.registry.node_count(), 3);
    assert!(n2.registry.get_node("node-0").is_some());
    assert!(n2.registry.get_node("node-1").is_some());
    assert!(n2.registry.get_node("node-2").is_some());

    // Existing nodes should also see the new node.
    let n0 = cluster.get_node("node-0").unwrap();
    assert!(n0.registry.get_node("node-2").is_some());

    cluster.shutdown().await;
}

#[tokio::test]
async fn node_join_message_delivery_works() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("sender", NodeRole::Prefill).await;
    cluster.add_node("receiver", NodeRole::Decode).await;

    let sender = cluster.get_node("sender").unwrap();
    let receiver = cluster.get_node("receiver").unwrap();
    let recv_addr = receiver.address.clone();

    // Send a control message between nodes.
    let msg = TransportMessage::Control {
        operation: "join_ack".to_string(),
        payload: Bytes::from("welcome"),
    };
    sender.transport.send(&recv_addr, msg).await.unwrap();

    let (from, received) = receiver.transport.recv().await.unwrap();
    assert_eq!(from, sender.address);
    match received {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "join_ack");
            assert_eq!(&payload[..], b"welcome");
        }
        _ => panic!("expected Control message"),
    }

    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Node failure scenarios
// ---------------------------------------------------------------------------

#[tokio::test]
async fn node_failure_partition_detected() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("healthy", NodeRole::Hybrid).await;
    cluster.add_node("failing", NodeRole::Hybrid).await;

    let failing_addr = cluster.get_node("failing").unwrap().address.clone();

    // Partition the failing node.
    cluster.partition_node("failing").await.unwrap();

    // Verify that sends to the partitioned node fail.
    let healthy = cluster.get_node("healthy").unwrap();
    let msg = TransportMessage::Control {
        operation: "ping".to_string(),
        payload: Bytes::new(),
    };
    let result = healthy.transport.send(&failing_addr, msg).await;
    assert!(result.is_err());

    cluster.shutdown().await;
}

#[tokio::test]
async fn node_failure_graceful_removal() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("node-0", NodeRole::Hybrid).await;
    cluster.add_node("node-1", NodeRole::Hybrid).await;
    cluster.add_node("node-2", NodeRole::Hybrid).await;

    // Gracefully remove node-1.
    cluster.remove_node("node-1").await.unwrap();
    assert_eq!(cluster.node_count(), 2);

    // Remaining nodes should no longer see node-1.
    let n0 = cluster.get_node("node-0").unwrap();
    assert!(n0.registry.get_node("node-1").is_none());

    let n2 = cluster.get_node("node-2").unwrap();
    assert!(n2.registry.get_node("node-1").is_none());

    cluster.shutdown().await;
}

#[tokio::test]
async fn node_failure_partition_then_heal() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("a", NodeRole::Hybrid).await;
    cluster.add_node("b", NodeRole::Hybrid).await;

    let b_addr = cluster.get_node("b").unwrap().address.clone();

    // Partition node B.
    cluster.partition_node("b").await.unwrap();

    let a = cluster.get_node("a").unwrap();
    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::new(),
    };
    assert!(a.transport.send(&b_addr, msg).await.is_err());

    // Heal and verify communication resumes.
    cluster.heal_node("b").await.unwrap();

    let msg = TransportMessage::Control {
        operation: "recovered".to_string(),
        payload: Bytes::from("back_online"),
    };
    a.transport.send(&b_addr, msg).await.unwrap();

    let b = cluster.get_node("b").unwrap();
    let (_, received) = b.transport.recv().await.unwrap();
    match received {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "recovered");
            assert_eq!(&payload[..], b"back_online");
        }
        _ => panic!("expected Control"),
    }

    cluster.shutdown().await;
}

#[tokio::test]
async fn node_failure_inflight_request_fails_gracefully() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("client", NodeRole::Hybrid).await;
    cluster.add_node("server", NodeRole::Hybrid).await;

    let server_addr = cluster.get_node("server").unwrap().address.clone();

    // Install RPC handler on server.
    let server = cluster.get_node("server").unwrap();
    server
        .transport
        .serve_rpc(Box::new(|req| req.to_vec()))
        .await
        .unwrap();

    // Verify RPC works before partition.
    let client = cluster.get_node("client").unwrap();
    let resp = client
        .transport
        .rpc_call(&server_addr, b"hello")
        .await
        .unwrap();
    assert_eq!(resp, b"hello");

    // Partition server -- RPC should fail.
    cluster.partition_node("server").await.unwrap();
    let result = client.transport.rpc_call(&server_addr, b"will_fail").await;
    assert!(result.is_err());

    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Request routing scenarios
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_routing_by_role() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("prefill-0", NodeRole::Prefill).await;
    cluster.add_node("decode-0", NodeRole::Decode).await;
    cluster.add_node("decode-1", NodeRole::Decode).await;

    // Use node-0's registry to find decode nodes.
    let n0 = cluster.get_node("prefill-0").unwrap();
    let decode_nodes = n0.registry.nodes_with_role(NodeRole::Decode);
    assert_eq!(decode_nodes.len(), 2);

    // Route a message to each decode node.
    for decode_node in &decode_nodes {
        let addr = decode_node.config.address.to_string();
        let msg = TransportMessage::Control {
            operation: "decode_request".to_string(),
            payload: Bytes::from(format!("to:{}", decode_node.config.id)),
        };
        n0.transport.send(&addr, msg).await.unwrap();
    }

    // Verify both decode nodes received their messages.
    for id in &["decode-0", "decode-1"] {
        let node = cluster.get_node(id).unwrap();
        let (_, received) = node.transport.recv().await.unwrap();
        match received {
            TransportMessage::Control { operation, payload } => {
                assert_eq!(operation, "decode_request");
                let expected = format!("to:{id}");
                assert_eq!(&payload[..], expected.as_bytes());
            }
            _ => panic!("expected Control message"),
        }
    }

    cluster.shutdown().await;
}

#[tokio::test]
async fn request_routing_rpc_to_specific_role() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("coordinator", NodeRole::Hybrid).await;
    cluster.add_node("worker-0", NodeRole::PipelineStage).await;
    cluster.add_node("worker-1", NodeRole::PipelineStage).await;

    // Install RPC handlers on workers that echo their node ID.
    for id in &["worker-0", "worker-1"] {
        let node = cluster.get_node(id).unwrap();
        let node_id = id.to_string();
        node.transport
            .serve_rpc(Box::new(move |_req| node_id.as_bytes().to_vec()))
            .await
            .unwrap();
    }

    // Coordinator sends RPC to each pipeline stage worker.
    let coord = cluster.get_node("coordinator").unwrap();
    let workers = coord.registry.nodes_with_role(NodeRole::PipelineStage);
    assert_eq!(workers.len(), 2);

    for worker in &workers {
        let addr = worker.config.address.to_string();
        let response = coord.transport.rpc_call(&addr, b"status").await.unwrap();
        let resp_id = String::from_utf8(response).unwrap();
        assert_eq!(resp_id, worker.config.id);
    }

    cluster.shutdown().await;
}

#[tokio::test]
async fn request_routing_tensor_data_transfer() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("stage-0", NodeRole::PipelineStage).await;
    cluster.add_node("stage-1", NodeRole::PipelineStage).await;

    let s0 = cluster.get_node("stage-0").unwrap();
    let s1 = cluster.get_node("stage-1").unwrap();
    let s1_addr = s1.address.clone();

    // Simulate activation transfer between pipeline stages.
    let activation_data = Bytes::from(vec![42u8; 4096]);
    let msg = TransportMessage::TensorData {
        tensor_id: "layer.5.activation".to_string(),
        shape: vec![1, 32, 128],
        data: activation_data.clone(),
    };
    s0.transport.send(&s1_addr, msg).await.unwrap();

    let (from, received) = s1.transport.recv().await.unwrap();
    assert_eq!(from, s0.address);
    match received {
        TransportMessage::TensorData {
            tensor_id,
            shape,
            data,
        } => {
            assert_eq!(tensor_id, "layer.5.activation");
            assert_eq!(shape, vec![1, 32, 128]);
            assert_eq!(data, activation_data);
        }
        _ => panic!("expected TensorData"),
    }

    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Performance regression benchmarks
// ---------------------------------------------------------------------------

#[test]
fn perf_serialization_no_regression() {
    let us = measure_serialization_roundtrip(1024, 200);

    let baseline = PerfBaseline {
        metric: "serialization_roundtrip_1kb".to_string(),
        baseline_value: 10.0,       // 10 us baseline
        regression_threshold: 10.0, // Generous for CI variability
    };

    baseline
        .check(us)
        .unwrap_or_else(|e| panic!("serialization regression: {e}"));
}

#[test]
fn perf_serialization_large_payload() {
    let us = measure_serialization_roundtrip(1024 * 1024, 50);

    let baseline = PerfBaseline {
        metric: "serialization_roundtrip_1mb".to_string(),
        baseline_value: 50.0, // 50 us baseline
        regression_threshold: 10.0,
    };

    baseline
        .check(us)
        .unwrap_or_else(|e| panic!("serialization regression (1MB): {e}"));
}

#[tokio::test]
async fn perf_mock_transport_latency() {
    let router = MockRouter::new();
    let us = measure_mock_transfer_latency(&router, 200).await;

    let baseline = PerfBaseline {
        metric: "mock_transport_latency".to_string(),
        baseline_value: 50.0,       // 50 us baseline
        regression_threshold: 20.0, // Generous for CI
    };

    baseline
        .check(us)
        .unwrap_or_else(|e| panic!("mock transport regression: {e}"));
}

#[tokio::test]
async fn perf_message_throughput() {
    let router = MockRouter::new();
    let config = MockTransportConfig::default();
    let sender =
        MockTransport::new("tp-sender:1".to_string(), router.clone(), config.clone()).await;
    let receiver = MockTransport::new("tp-receiver:2".to_string(), router, config).await;

    let num_messages: usize = 500;
    let payload_size: usize = 1024;
    let payload = Bytes::from(vec![0xABu8; payload_size]);

    let start = std::time::Instant::now();

    // Send and receive concurrently to avoid channel backpressure stalling.
    let sender = std::sync::Arc::new(sender);
    let receiver = std::sync::Arc::new(receiver);

    let send_task = {
        let sender = sender.clone();
        let payload = payload.clone();
        tokio::spawn(async move {
            for i in 0..num_messages {
                let msg = TransportMessage::TensorData {
                    tensor_id: format!("bench_{i}"),
                    shape: vec![payload_size],
                    data: payload.clone(),
                };
                sender.send("tp-receiver:2", msg).await.unwrap();
            }
        })
    };

    let recv_task = {
        let receiver = receiver.clone();
        tokio::spawn(async move {
            for _ in 0..num_messages {
                receiver.recv().await.unwrap();
            }
        })
    };

    send_task.await.unwrap();
    recv_task.await.unwrap();
    let elapsed = start.elapsed();

    let total_bytes = (num_messages * payload_size) as f64;
    let mb_per_sec = total_bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0);

    // Verify it completes in reasonable time.
    assert!(
        elapsed < Duration::from_secs(10),
        "throughput benchmark took too long: {elapsed:?}"
    );
    assert!(mb_per_sec > 0.1, "throughput too low: {mb_per_sec:.2} MB/s");
}

// ---------------------------------------------------------------------------
// CI compatibility checks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ci_no_port_conflicts() {
    // Verify that TestCluster auto-assigns unique ports.
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    let mut addresses = Vec::new();
    for i in 0..10 {
        let id = format!("node-{i}");
        cluster.add_node(&id, NodeRole::Hybrid).await;
        addresses.push(cluster.get_node(&id).unwrap().address.clone());
    }

    // All addresses should be unique.
    let unique: std::collections::HashSet<_> = addresses.iter().collect();
    assert_eq!(unique.len(), 10);

    cluster.shutdown().await;
}

#[tokio::test]
async fn ci_test_timeout_respected() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("a", NodeRole::Hybrid).await;
    cluster.add_node("b", NodeRole::Hybrid).await;

    // wait_for_status with short timeout should fail gracefully.
    let result = cluster
        .wait_for_status("a", "b", NodeStatus::Unreachable, Duration::from_millis(50))
        .await;
    assert!(result.is_err());

    cluster.shutdown().await;
}

#[tokio::test]
async fn ci_concurrent_operations() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("node-0", NodeRole::Hybrid).await;
    cluster.add_node("node-1", NodeRole::Hybrid).await;

    let n0_transport = cluster.get_node("node-0").unwrap().transport.clone();
    let n1_transport = cluster.get_node("node-1").unwrap().transport.clone();
    let n0_addr = cluster.get_node("node-0").unwrap().address.clone();
    let n1_addr = cluster.get_node("node-1").unwrap().address.clone();

    // Send messages concurrently in both directions.
    let send_0_to_1 = {
        let addr = n1_addr.clone();
        let transport = n0_transport.clone();
        tokio::spawn(async move {
            for i in 0..50u32 {
                let msg = TransportMessage::Control {
                    operation: format!("0to1-{i}"),
                    payload: Bytes::new(),
                };
                transport.send(&addr, msg).await.unwrap();
            }
        })
    };

    let send_1_to_0 = {
        let addr = n0_addr.clone();
        let transport = n1_transport.clone();
        tokio::spawn(async move {
            for i in 0..50u32 {
                let msg = TransportMessage::Control {
                    operation: format!("1to0-{i}"),
                    payload: Bytes::new(),
                };
                transport.send(&addr, msg).await.unwrap();
            }
        })
    };

    send_0_to_1.await.unwrap();
    send_1_to_0.await.unwrap();

    // Drain messages at both ends.
    let mut count_at_0 = 0;
    let mut count_at_1 = 0;

    for _ in 0..50 {
        n0_transport.recv().await.unwrap();
        count_at_0 += 1;
    }
    for _ in 0..50 {
        n1_transport.recv().await.unwrap();
        count_at_1 += 1;
    }

    assert_eq!(count_at_0, 50);
    assert_eq!(count_at_1, 50);

    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// 2D (PP x TP) parallelism plumbing
// ---------------------------------------------------------------------------
//
// These tests verify that the runtime plumbing for the 2D parallelism
// introduced is wired end-to-end: config parsing, registry
// lookups, 2D-aware routing, and the cache-manager admission grid.

use mlxcel::distributed::config::ClusterConfig;
use mlxcel::distributed::pipeline::{
    CacheAdmissionRequest, PipelineCacheConfig, PipelineCacheManager, PpTpAdmissionOutcome,
    PpTpCoord, coordinated_2d_admission,
};
use mlxcel::distributed::registry::NodeRegistry;
use mlxcel::distributed::routing::{
    NodeCandidate, PipelineTensorParallelRouter, RoutingRequest, RoutingStrategy, TrafficClass,
};

fn pp_tp_2x2_toml() -> &'static str {
    r#"
[cluster]
name = "pp-tp-2x2-test"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "s0-r0"
address = "127.0.0.1:18100"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "s0-r1"
address = "127.0.0.1:18101"
role = "pipeline_tensor_parallel"
stage = 0
rank = 1

[[nodes]]
id = "s1-r0"
address = "127.0.0.1:18102"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "s1-r1"
address = "127.0.0.1:18103"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#
}

#[test]
fn pp_tp_2x2_cluster_registers_every_intersection() {
    let config = ClusterConfig::from_toml(pp_tp_2x2_toml()).unwrap();
    assert!(config.is_pp_tp_2d());
    let registry = NodeRegistry::from_config(&config, "s0-r0");
    // Each (stage, rank) pair is reachable via the registry.
    for stage in 0..2 {
        for rank in 0..2 {
            let node = registry.find_pp_tp_node(stage, rank).unwrap_or_else(|| {
                panic!("missing node for (stage={stage}, rank={rank})");
            });
            assert_eq!(node.config.stage, Some(stage));
            assert_eq!(node.config.rank, Some(rank));
        }
    }
    assert_eq!(registry.local_pp_tp_coords(), Some((0, 0)));
    assert_eq!(registry.nodes_at_stage(0).len(), 2); // TP collective peers.
    assert_eq!(registry.nodes_at_rank(0).len(), 2); // PP activation peers.
}

#[test]
fn pp_tp_router_handles_tp_collective_and_pp_activation_classes() {
    let router = PipelineTensorParallelRouter;
    let candidates: Vec<NodeCandidate> = [
        ("s0-r0", 0, 0),
        ("s0-r1", 0, 1),
        ("s1-r0", 1, 0),
        ("s1-r1", 1, 1),
    ]
    .iter()
    .map(|(id, stage, rank)| NodeCandidate {
        node_id: id.to_string(),
        role: NodeRole::PipelineTensorParallel,
        status: NodeStatus::Online,
        metrics: None,
        stage: Some(*stage),
        rank: Some(*rank),
    })
    .collect();

    // TP collective: stage 0, rank 0 wants to communicate with rank 1 on the
    // same stage.
    let tp_collective = RoutingRequest {
        request_id: "tp".to_string(),
        preferred_role: None,
        preferred_stage: Some(0),
        preferred_rank: Some(1),
        traffic_class: TrafficClass::TpCollective,
        affinity_node: None,
    };
    let tp_decision = router.select_node(&tp_collective, &candidates).unwrap();
    assert_eq!(tp_decision.target_node, "s0-r1");
    assert!(
        tp_decision
            .reason
            .as_deref()
            .unwrap()
            .contains("tp_collective")
    );

    // PP activation: stage 0, rank 0 hands activations to stage 1 at the
    // same TP rank.
    let pp_activation = RoutingRequest {
        request_id: "pp".to_string(),
        preferred_role: None,
        preferred_stage: Some(1),
        preferred_rank: Some(0),
        traffic_class: TrafficClass::PpActivation,
        affinity_node: None,
    };
    let pp_decision = router.select_node(&pp_activation, &candidates).unwrap();
    assert_eq!(pp_decision.target_node, "s1-r0");
    assert!(
        pp_decision
            .reason
            .as_deref()
            .unwrap()
            .contains("pp_activation")
    );
}

#[test]
fn pp_tp_cache_admission_rolls_back_on_any_rank_rejection() {
    // Build a 2x2 grid with one slot that cannot hold the sequence.
    let mk = |stage, _rank, layer_range, cap| {
        PipelineCacheManager::new(PipelineCacheConfig {
            stage_index: stage,
            num_stages: 2,
            layer_range,
            max_sequences: cap,
            memory_budget_bytes: 500_000,
            bytes_per_layer_per_token: 256,
            pressure_threshold: 0.9,
        })
        .unwrap()
    };

    let mut m00 = mk(0, 0, 0..8, 4);
    let mut m01 = mk(0, 1, 0..8, 1); // Saturate this slot first.
    m01.request_admission(&CacheAdmissionRequest::new(100, 16));
    let mut m10 = mk(1, 0, 8..16, 4);
    let mut m11 = mk(1, 1, 8..16, 4);

    let mut grid: Vec<(PpTpCoord, &mut PipelineCacheManager)> = vec![
        (PpTpCoord::new(0, 0), &mut m00),
        (PpTpCoord::new(0, 1), &mut m01),
        (PpTpCoord::new(1, 0), &mut m10),
        (PpTpCoord::new(1, 1), &mut m11),
    ];
    let outcome = coordinated_2d_admission(&mut grid, &CacheAdmissionRequest::new(7, 16)).unwrap();
    assert!(matches!(outcome, PpTpAdmissionOutcome::Rejected { .. }));

    // Other slots must stay clean so the 2D stage remains coherent.
    assert!(m00.get_allocation(7).is_none());
    assert!(m10.get_allocation(7).is_none());
    assert!(m11.get_allocation(7).is_none());
}
