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
use crate::distributed::TransportBackend;

#[test]
fn parse_node_role_from_str() {
    assert_eq!("prefill".parse::<NodeRole>().unwrap(), NodeRole::Prefill);
    assert_eq!("decode".parse::<NodeRole>().unwrap(), NodeRole::Decode);
    assert_eq!(
        "pipeline_stage".parse::<NodeRole>().unwrap(),
        NodeRole::PipelineStage
    );
    assert_eq!(
        "pipeline-stage".parse::<NodeRole>().unwrap(),
        NodeRole::PipelineStage
    );
    assert_eq!(
        "tensor_parallel_rank".parse::<NodeRole>().unwrap(),
        NodeRole::TensorParallelRank
    );
    assert_eq!(
        "tp".parse::<NodeRole>().unwrap(),
        NodeRole::TensorParallelRank
    );
    assert_eq!("hybrid".parse::<NodeRole>().unwrap(), NodeRole::Hybrid);
    assert!("unknown".parse::<NodeRole>().is_err());
}

#[test]
fn node_role_display_roundtrip() {
    let roles = [
        NodeRole::Prefill,
        NodeRole::Decode,
        NodeRole::PipelineStage,
        NodeRole::TensorParallelRank,
        NodeRole::Hybrid,
    ];
    for role in roles {
        let s = role.to_string();
        let parsed: NodeRole = s.parse().unwrap();
        assert_eq!(parsed, role);
    }
}

#[test]
fn node_role_serialization() {
    let role = NodeRole::TensorParallelRank;
    let json = serde_json::to_string(&role).unwrap();
    assert_eq!(json, "\"tensor_parallel_rank\"");
    let deserialized: NodeRole = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, role);
}

#[test]
fn parse_cluster_config_toml() {
    let toml_str = r#"
[cluster]
name = "test-cluster"
tensor_parallel_size = 2
pipeline_parallel_size = 1

[[nodes]]
id = "node-0"
address = "192.168.1.10:8080"
role = "tensor_parallel_rank"
rank = 0

[[nodes]]
id = "node-1"
address = "192.168.1.11:8080"
role = "tensor_parallel_rank"
rank = 1
"#;
    let config = ClusterConfig::from_toml(toml_str).unwrap();
    assert_eq!(config.cluster.name, "test-cluster");
    assert_eq!(config.cluster.tensor_parallel_size, 2);
    assert_eq!(config.cluster.transport_backend, TransportBackend::Tcp);
    assert_eq!(config.nodes.len(), 2);
    assert_eq!(config.nodes[0].role, NodeRole::TensorParallelRank);
    assert_eq!(config.nodes[0].rank, Some(0));
    assert_eq!(config.nodes[1].rank, Some(1));
}

#[test]
fn parse_pipeline_stage_cluster_with_transport_backend() {
    let toml_str = r#"
[cluster]
name = "pipeline"
pipeline_parallel_size = 2
transport_backend = "thunderbolt"

[[nodes]]
id = "coordinator"
address = "10.0.0.10:9000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "10.0.0.11:9000"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "10.0.0.12:9000"
role = "pipeline_stage"
stage = 1
"#;
    let config = ClusterConfig::from_toml(toml_str).unwrap();
    assert_eq!(
        config.cluster.transport_backend,
        TransportBackend::Thunderbolt
    );
    assert_eq!(
        config
            .pipeline_stage_nodes()
            .into_iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["stage-0", "stage-1"]
    );
}

#[test]
fn parse_pipeline_stage_cluster_with_rdma_backend() {
    let toml_str = r#"
[cluster]
name = "pipeline"
pipeline_parallel_size = 2
transport_backend = "rdma"

[[nodes]]
id = "stage-0"
address = "10.0.0.11:9000"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "10.0.0.12:9000"
role = "pipeline_stage"
stage = 1
"#;
    let config = ClusterConfig::from_toml(toml_str).unwrap();
    assert_eq!(config.cluster.transport_backend, TransportBackend::Rdma);
}

#[test]
fn reject_pipeline_config_with_missing_stage_index() {
    let toml_str = r#"
[cluster]
name = "pipeline"
pipeline_parallel_size = 2

[[nodes]]
id = "stage-0"
address = "10.0.0.11:9000"
role = "pipeline_stage"

[[nodes]]
id = "stage-1"
address = "10.0.0.12:9000"
role = "pipeline_stage"
stage = 1
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("missing required 'stage' index"));
}

#[test]
fn reject_pipeline_config_with_non_stage_node_stage_field() {
    let toml_str = r#"
[cluster]
name = "pipeline"

[[nodes]]
id = "coord"
address = "10.0.0.10:9000"
role = "hybrid"
stage = 0
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("only pipeline_stage")
            && msg.contains("pipeline_tensor_parallel")
            && msg.contains("may set stage"),
        "unexpected error: {msg}"
    );
}

#[test]
fn reject_duplicate_node_ids() {
    let toml_str = r#"
[cluster]
name = "dup"

[[nodes]]
id = "same"
address = "127.0.0.1:8080"
role = "hybrid"

[[nodes]]
id = "same"
address = "127.0.0.1:8081"
role = "hybrid"
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("duplicate node id"));
}

#[test]
fn reject_duplicate_addresses() {
    let toml_str = r#"
[cluster]
name = "dup"

[[nodes]]
id = "a"
address = "127.0.0.1:8080"
role = "hybrid"

[[nodes]]
id = "b"
address = "127.0.0.1:8080"
role = "hybrid"
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("duplicate address"));
}

#[test]
fn reject_empty_nodes() {
    let toml_str = r#"
[cluster]
name = "empty"
nodes = []
"#;
    // toml may parse this differently; either parse error or validation error is fine
    let result = ClusterConfig::from_toml(toml_str);
    assert!(result.is_err());
}

#[test]
fn from_cli_builds_single_node() {
    let addr: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let config = ClusterConfig::from_cli("local".to_string(), addr, NodeRole::Hybrid, vec![]);
    assert_eq!(config.nodes.len(), 1);
    assert_eq!(config.nodes[0].id, "local");
    assert_eq!(config.nodes[0].role, NodeRole::Hybrid);
}

#[test]
fn from_cli_with_peers() {
    let addr: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let peer1: std::net::SocketAddr = "192.168.1.2:9000".parse().unwrap();
    let peer2: std::net::SocketAddr = "192.168.1.3:9000".parse().unwrap();
    let config = ClusterConfig::from_cli(
        "node-0".to_string(),
        addr,
        NodeRole::Prefill,
        vec![peer1, peer2],
    );
    assert_eq!(config.nodes.len(), 3);
    assert_eq!(config.nodes[0].role, NodeRole::Prefill);
    assert_eq!(config.nodes[1].address, peer1);
    assert_eq!(config.nodes[2].address, peer2);
}

#[test]
fn find_node_by_id() {
    let addr: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let config = ClusterConfig::from_cli("local".to_string(), addr, NodeRole::Hybrid, vec![]);
    assert!(config.find_node("local").is_some());
    assert!(config.find_node("nonexistent").is_none());
}

#[test]
fn topology_summary_contains_all_nodes() {
    let toml_str = r#"
[cluster]
name = "summary-test"

[[nodes]]
id = "alpha"
address = "10.0.0.1:8080"
role = "prefill"

[[nodes]]
id = "beta"
address = "10.0.0.2:8080"
role = "decode"
"#;
    let config = ClusterConfig::from_toml(toml_str).unwrap();
    let summary = config.topology_summary();
    assert!(summary.contains("summary-test"));
    assert!(summary.contains("alpha"));
    assert!(summary.contains("beta"));
    assert!(summary.contains("prefill"));
    assert!(summary.contains("decode"));
}

#[test]
fn reject_empty_node_id() {
    let toml_str = r#"
[cluster]
name = "test"

[[nodes]]
id = ""
address = "127.0.0.1:8080"
role = "hybrid"
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("node id must not be empty"));
}

#[test]
fn reject_control_characters_in_node_id() {
    let toml_str = "
[cluster]
name = \"test\"

[[nodes]]
id = \"bad\\nid\"
address = \"127.0.0.1:8080\"
role = \"hybrid\"
";
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("control characters"));
}

#[test]
fn reject_control_characters_in_cluster_name() {
    let toml_str = "[cluster]\nname = \"bad\\nname\"\n\n[[nodes]]\nid = \"node-0\"\naddress = \"127.0.0.1:8080\"\nrole = \"hybrid\"";
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(err.to_string().contains("control characters"));
}

#[test]
fn from_cli_deduplicates_local_address_in_peers() {
    let addr: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    // Pass the local address as a peer — should be silently dropped.
    let config = ClusterConfig::from_cli(
        "local".to_string(),
        addr,
        NodeRole::Hybrid,
        vec![addr, "192.168.1.2:9000".parse().unwrap()],
    );
    // Only local + 1 real peer, the duplicate local address was filtered.
    assert_eq!(config.nodes.len(), 2);
    assert_eq!(config.nodes[0].address, addr);
    assert_eq!(
        config.nodes[1].address,
        "192.168.1.2:9000".parse::<std::net::SocketAddr>().unwrap()
    );
}

// -- 2D (PP x TP) parallelism ------------------------------------------------

#[test]
fn parse_pp_tp_role_variants() {
    for alias in ["pipeline_tensor_parallel", "pp_tp", "pptp", "PP-TP", "2d"] {
        assert_eq!(
            alias.parse::<NodeRole>().unwrap(),
            NodeRole::PipelineTensorParallel,
            "alias {alias} did not parse"
        );
    }
}

#[test]
fn pp_tp_role_predicates() {
    assert!(NodeRole::PipelineTensorParallel.is_pipeline());
    assert!(NodeRole::PipelineTensorParallel.is_tensor_parallel());
    assert!(NodeRole::PipelineTensorParallel.is_pp_tp());
    assert!(NodeRole::PipelineStage.is_pipeline());
    assert!(!NodeRole::PipelineStage.is_pp_tp());
    assert!(!NodeRole::TensorParallelRank.is_pipeline());
    assert!(NodeRole::TensorParallelRank.is_tensor_parallel());
}

fn build_2x2_cluster_toml() -> &'static str {
    r#"
[cluster]
name = "pp-tp-2x2"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "s0-r0"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "s0-r1"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 1

[[nodes]]
id = "s1-r0"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "s1-r1"
address = "10.0.0.21:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#
}

#[test]
fn parse_pp_tp_2x2_cluster_config() {
    let config = ClusterConfig::from_toml(build_2x2_cluster_toml()).unwrap();
    assert_eq!(config.cluster.pipeline_parallel_size, 2);
    assert_eq!(config.cluster.tensor_parallel_size, 2);
    assert!(config.is_pp_tp_2d());
    assert_eq!(config.pp_tp_nodes().len(), 4);
    assert_eq!(config.pp_tp_nodes_at_stage(0).len(), 2);
    assert_eq!(config.pp_tp_nodes_at_stage(1).len(), 2);
    assert_eq!(config.pp_tp_nodes_at_rank(0).len(), 2);
    assert_eq!(config.pp_tp_nodes_at_rank(1).len(), 2);
    assert_eq!(config.pp_tp_node(0, 0).unwrap().id, "s0-r0");
    assert_eq!(config.pp_tp_node(1, 1).unwrap().id, "s1-r1");
    // Sort order should be (stage, rank).
    let ordered = config.pp_tp_nodes();
    assert_eq!(ordered[0].id, "s0-r0");
    assert_eq!(ordered[1].id, "s0-r1");
    assert_eq!(ordered[2].id, "s1-r0");
    assert_eq!(ordered[3].id, "s1-r1");
}

#[test]
fn pp_tp_missing_pair_fails_validation() {
    let toml_str = r#"
[cluster]
name = "bad"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "s0-r0"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "s0-r1"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 1

[[nodes]]
id = "s1-r0"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    let msg = err.to_string();
    // Message should clearly identify the 2D grid size mismatch.
    assert!(
        msg.contains("2x2") || msg.contains("2x2") || msg.contains("requires 4"),
        "unexpected error: {msg}"
    );
}

#[test]
fn pp_tp_duplicate_pair_fails_validation() {
    let toml_str = r#"
[cluster]
name = "bad"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "a"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "b"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "c"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "d"
address = "10.0.0.21:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate") && msg.contains("stage=0") && msg.contains("rank=0"),
        "unexpected error: {msg}"
    );
}

#[test]
fn pp_tp_out_of_range_rank_fails_validation() {
    let toml_str = r#"
[cluster]
name = "bad"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "s0-r0"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 0

[[nodes]]
id = "s0-r1"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 5

[[nodes]]
id = "s1-r0"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "s1-r1"
address = "10.0.0.21:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(
        err.to_string().contains("out-of-range rank"),
        "unexpected error: {err}"
    );
}

#[test]
fn pp_tp_missing_stage_fails_validation() {
    let toml_str = r#"
[cluster]
name = "bad"
pipeline_parallel_size = 2
tensor_parallel_size = 2

[[nodes]]
id = "a"
address = "10.0.0.10:8080"
role = "pipeline_tensor_parallel"
rank = 0

[[nodes]]
id = "b"
address = "10.0.0.11:8080"
role = "pipeline_tensor_parallel"
stage = 0
rank = 1

[[nodes]]
id = "c"
address = "10.0.0.20:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 0

[[nodes]]
id = "d"
address = "10.0.0.21:8080"
role = "pipeline_tensor_parallel"
stage = 1
rank = 1
"#;
    let err = ClusterConfig::from_toml(toml_str).unwrap_err();
    assert!(
        err.to_string().contains("missing required 'stage'"),
        "unexpected error: {err}"
    );
}

#[test]
fn topology_summary_shows_stage_and_rank() {
    let config = ClusterConfig::from_toml(build_2x2_cluster_toml()).unwrap();
    let summary = config.topology_summary();
    assert!(summary.contains("(stage=0, rank=0)"));
    assert!(summary.contains("(stage=1, rank=1)"));
}
