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

use bytes::Bytes;

use super::*;
use crate::distributed::transport::TransportMessage;

#[tokio::test]
async fn create_empty_cluster() {
    let cluster = TestCluster::new(TestClusterConfig::default());
    assert_eq!(cluster.node_count(), 0);
}

#[tokio::test]
async fn add_nodes_to_cluster() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());

    cluster.add_node("node-0", NodeRole::Prefill).await;
    assert_eq!(cluster.node_count(), 1);

    cluster.add_node("node-1", NodeRole::Decode).await;
    assert_eq!(cluster.node_count(), 2);

    // Both nodes should see each other.
    let n0 = cluster.get_node("node-0").unwrap();
    let n1 = cluster.get_node("node-1").unwrap();
    assert_eq!(n0.registry.node_count(), 2);
    assert_eq!(n1.registry.node_count(), 2);
}

#[tokio::test]
async fn nodes_can_communicate() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("sender", NodeRole::Hybrid).await;
    cluster.add_node("receiver", NodeRole::Hybrid).await;

    let sender_node = cluster.get_node("sender").unwrap();
    let receiver_node = cluster.get_node("receiver").unwrap();

    let receiver_addr = receiver_node.address.clone();
    let msg = TransportMessage::Control {
        operation: "hello".to_string(),
        payload: Bytes::from("world"),
    };

    sender_node
        .transport
        .send(&receiver_addr, msg)
        .await
        .unwrap();

    let (from, received) = receiver_node.transport.recv().await.unwrap();
    assert_eq!(from, sender_node.address);
    match received {
        TransportMessage::Control { operation, payload } => {
            assert_eq!(operation, "hello");
            assert_eq!(&payload[..], b"world");
        }
        _ => panic!("expected Control message"),
    }
}

#[tokio::test]
async fn remove_node_from_cluster() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("node-0", NodeRole::Hybrid).await;
    cluster.add_node("node-1", NodeRole::Hybrid).await;
    cluster.add_node("node-2", NodeRole::Hybrid).await;
    assert_eq!(cluster.node_count(), 3);

    cluster.remove_node("node-1").await.unwrap();
    assert_eq!(cluster.node_count(), 2);

    // Remaining nodes should no longer have node-1 in registry.
    let n0 = cluster.get_node("node-0").unwrap();
    assert!(n0.registry.get_node("node-1").is_none());
}

#[tokio::test]
async fn partition_prevents_communication() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("a", NodeRole::Hybrid).await;
    cluster.add_node("b", NodeRole::Hybrid).await;

    let b_addr = cluster.get_node("b").unwrap().address.clone();

    cluster.partition_node("b").await.unwrap();

    let a = cluster.get_node("a").unwrap();
    let msg = TransportMessage::Control {
        operation: "test".to_string(),
        payload: Bytes::new(),
    };
    let result = a.transport.send(&b_addr, msg).await;
    assert!(result.is_err());

    // Heal and verify.
    cluster.heal_node("b").await.unwrap();
    let msg = TransportMessage::Control {
        operation: "healed".to_string(),
        payload: Bytes::new(),
    };
    a.transport.send(&b_addr, msg).await.unwrap();
}

#[tokio::test]
async fn node_ids_returns_all_ids() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("alpha", NodeRole::Prefill).await;
    cluster.add_node("beta", NodeRole::Decode).await;

    let mut ids = cluster.node_ids();
    ids.sort();
    assert_eq!(ids, vec!["alpha", "beta"]);
}

#[tokio::test]
async fn shutdown_stops_all_nodes() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("node-0", NodeRole::Hybrid).await;
    cluster.add_node("node-1", NodeRole::Hybrid).await;
    cluster.shutdown().await;
    assert_eq!(cluster.node_count(), 0);
}

#[tokio::test]
async fn wait_for_status_succeeds() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("observer", NodeRole::Hybrid).await;
    cluster.add_node("target", NodeRole::Hybrid).await;

    // Status is already Online from add_node.
    cluster
        .wait_for_status(
            "observer",
            "target",
            NodeStatus::Online,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn wait_for_status_times_out() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("observer", NodeRole::Hybrid).await;
    cluster.add_node("target", NodeRole::Hybrid).await;

    // Target is Online, but we wait for Unreachable -- should timeout.
    let result = cluster
        .wait_for_status(
            "observer",
            "target",
            NodeStatus::Unreachable,
            Duration::from_millis(100),
        )
        .await;
    assert!(result.is_err());
}

#[test]
fn perf_baseline_passes_within_threshold() {
    let baseline = PerfBaseline {
        metric: "test_metric".to_string(),
        baseline_value: 100.0,
        regression_threshold: 2.0,
    };

    baseline.check(150.0).unwrap(); // 150 < 200 (100 * 2.0)
}

#[test]
fn perf_baseline_fails_on_regression() {
    let baseline = PerfBaseline {
        metric: "test_metric".to_string(),
        baseline_value: 100.0,
        regression_threshold: 1.5,
    };

    let result = baseline.check(200.0); // 200 > 150 (100 * 1.5)
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("regression"));
}

#[test]
fn serialization_roundtrip_benchmark() {
    let us = measure_serialization_roundtrip(1024, 100);
    // Should complete in reasonable time (< 1ms per iteration).
    assert!(us < 1000.0, "serialization too slow: {us:.2} us/iter");
}

#[tokio::test]
async fn mock_transfer_latency_benchmark() {
    let router = MockRouter::new();
    let us = measure_mock_transfer_latency(&router, 100).await;
    // Mock transport with zero latency should be very fast.
    assert!(us < 5000.0, "mock transfer too slow: {us:.2} us/iter");
}

#[tokio::test]
async fn roles_are_preserved() {
    let mut cluster = TestCluster::new(TestClusterConfig::default());
    cluster.add_node("prefill-0", NodeRole::Prefill).await;
    cluster.add_node("decode-0", NodeRole::Decode).await;
    cluster
        .add_node("pipeline-0", NodeRole::PipelineStage)
        .await;

    let n = cluster.get_node("prefill-0").unwrap();
    assert_eq!(n.role, NodeRole::Prefill);

    let n = cluster.get_node("decode-0").unwrap();
    assert_eq!(n.role, NodeRole::Decode);

    let n = cluster.get_node("pipeline-0").unwrap();
    assert_eq!(n.role, NodeRole::PipelineStage);
}
