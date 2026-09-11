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
use crate::distributed::config::{ClusterConfig, NodeRole};

#[tokio::test]
async fn probe_unreachable_peers() {
    // Use addresses that should not be listening
    let config = ClusterConfig::from_cli(
        "local".to_string(),
        "127.0.0.1:19999".parse().unwrap(),
        NodeRole::Hybrid,
        vec!["127.0.0.1:19998".parse().unwrap()],
    );
    let registry = NodeRegistry::from_config(&config, "local");

    let results = probe_peers(&registry, Duration::from_millis(100)).await;
    assert_eq!(results.len(), 1); // Only the peer, not the local node
    assert!(!results[0].reachable);

    let peer = registry.get_node("peer-0").unwrap();
    assert_eq!(peer.status, NodeStatus::Unreachable);
}

#[tokio::test]
async fn probe_reachable_peer() {
    // Start a real TCP listener to test reachability
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let config = ClusterConfig::from_cli(
        "local".to_string(),
        "127.0.0.1:19997".parse().unwrap(),
        NodeRole::Hybrid,
        vec![addr],
    );
    let registry = NodeRegistry::from_config(&config, "local");

    let results = probe_peers(&registry, Duration::from_secs(1)).await;
    assert_eq!(results.len(), 1);
    assert!(results[0].reachable);

    let peer = registry.get_node("peer-0").unwrap();
    assert_eq!(peer.status, NodeStatus::Online);
}

#[tokio::test]
async fn initialize_distributed_creates_registry() {
    let config = ClusterConfig::from_cli(
        "node-0".to_string(),
        "127.0.0.1:19996".parse().unwrap(),
        NodeRole::Prefill,
        vec![],
    );

    let registry = initialize_distributed(&config, "node-0", Duration::from_millis(50))
        .await
        .unwrap();

    assert_eq!(registry.node_count(), 1);
    assert_eq!(registry.local_node_id(), "node-0");
}
