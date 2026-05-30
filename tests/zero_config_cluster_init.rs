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

//! Integration tests for the zero-config multi-machine PP bring-up path
//! introduced.
//!
//! The `#[ignore]` test at the bottom drives the full coordinator +
//! remote-stage pipeline against a real Llama 3.2 1B-4bit checkpoint. It
//! mirrors the production startup path by calling [`plan_cluster`] and the
//! remote runtime primitives directly, so the zero-config code path is
//! exercised by the same types that production callers use.

mod common;

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::repo_model_dir;

use mlxcel::SamplingConfig;
use mlxcel::distributed::pipeline::{
    RemotePipelineRuntimeConfig, RemoteStageServiceConfig, RemoteStageServiceHandle,
    StageAssignment, resolve_in_process_pipeline_num_layers,
};
use mlxcel::distributed::{
    ClusterConfig, ClusterDiscoveryMode, ClusterInitRequest, NodeRole, TransportBackend,
    plan_cluster, render_deterministic_toml,
};
use mlxcel::server::batch::{BatchObservability, RequestPriority};
use mlxcel::server::{
    BatchMetrics, ModelProvider, PipelineParallelRuntimeConfig, ServerConfig, ServerGenerateOptions,
};

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

/// Build the zero-config request fixture used by all tests in this file.
fn sample_request(peers: Vec<SocketAddr>) -> ClusterInitRequest {
    ClusterInitRequest {
        pp_stages: peers.len() as u32,
        cluster_name: "integration-zero-config".to_string(),
        transport_backend: TransportBackend::Tcp,
        discovery: ClusterDiscoveryMode::Static,
        discovery_timeout: None,
        discovery_port: 0,
        coordinator_http_addr: loopback(18080),
        coordinator_control_addr: loopback(19000),
        static_peers: peers,
        data_port_base: 19001,
        output_toml_path: None,
    }
}

#[test]
fn zero_config_plan_is_consumable_by_existing_cluster_loader() {
    let peers = vec![loopback(19001), loopback(19002)];
    let plan = plan_cluster(&sample_request(peers)).expect("plan must succeed");
    let parsed = ClusterConfig::from_toml(&plan.toml).expect("TOML must reparse");
    assert_eq!(parsed.cluster.pipeline_parallel_size, 2);
    assert_eq!(parsed.nodes.len(), 3);
    // Coordinator has no stage; stage-0 / stage-1 carry PipelineStage roles.
    assert!(
        parsed
            .nodes
            .iter()
            .any(|n| n.role == NodeRole::PipelineStage && n.stage == Some(0))
    );
    assert!(
        parsed
            .nodes
            .iter()
            .any(|n| n.role == NodeRole::PipelineStage && n.stage == Some(1))
    );
}

#[test]
fn zero_config_plan_is_deterministic_across_rerender() {
    let peers = vec![loopback(19005), loopback(19003), loopback(19004)];
    let plan1 = plan_cluster(&sample_request(peers.clone())).unwrap();

    // Rerender from the parsed struct and confirm byte equality.
    let parsed = ClusterConfig::from_toml(&plan1.toml).unwrap();
    let plan2_toml = render_deterministic_toml(&parsed);
    assert_eq!(plan1.toml, plan2_toml);

    // A second plan with the same peers in a different order must also match.
    let mut reordered = peers;
    reordered.reverse();
    let plan3 = plan_cluster(&sample_request(reordered)).unwrap();
    assert_eq!(plan1.toml, plan3.toml);
}

fn wait_for_loaded(provider: &ModelProvider) {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if provider.is_loaded() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("model provider did not finish loading within 90s");
}

fn two_stage_assignments(num_layers: usize, split: usize) -> [StageAssignment; 2] {
    [
        StageAssignment {
            stage_index: 0,
            device_id: "stage-0".to_string(),
            layer_range: 0..split,
            has_embedding: true,
            has_lm_head: false,
            estimated_memory_bytes: 0,
        },
        StageAssignment {
            stage_index: 1,
            device_id: "stage-1".to_string(),
            layer_range: split..num_layers,
            has_embedding: false,
            has_lm_head: true,
            estimated_memory_bytes: 0,
        },
    ]
}

/// Full end-to-end zero-config bring-up: plan the cluster with real peer
/// addresses, start two remote stage services backed by the same model
/// checkpoint, then confirm the remote coordinator produces the same output
/// as the dense baseline. Verifies that the zero-config-planned cluster TOML
/// drives the production remote-pipeline runtime unchanged.
#[test]
#[ignore = "requires local model weights and TCP-bound remote stage services"]
fn zero_config_two_stage_cluster_matches_dense_baseline() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let stage0_port = reserve_port();
    let stage1_port = reserve_port();
    let coordinator_control_port = reserve_port();
    let coordinator_http_port = reserve_port();

    let peers = vec![loopback(stage0_port), loopback(stage1_port)];

    let request = ClusterInitRequest {
        pp_stages: 2,
        cluster_name: "zero-config-e2e".to_string(),
        transport_backend: TransportBackend::Tcp,
        discovery: ClusterDiscoveryMode::Static,
        discovery_timeout: None,
        discovery_port: 0,
        coordinator_http_addr: loopback(coordinator_http_port),
        coordinator_control_addr: loopback(coordinator_control_port),
        static_peers: peers.clone(),
        data_port_base: stage0_port,
        output_toml_path: None,
    };
    let plan = plan_cluster(&request).expect("zero-config plan succeeds");
    let cluster = plan.cluster;

    let num_layers = resolve_in_process_pipeline_num_layers(&model_dir).unwrap();
    let assignments = two_stage_assignments(num_layers, num_layers / 2);

    let stage0_addr = cluster.pipeline_stage_node(0).unwrap().address.to_string();
    let stage1_addr = cluster.pipeline_stage_node(1).unwrap().address.to_string();

    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage1_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage0_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
    })
    .unwrap();

    let dense_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig::default(),
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    let remote_provider = ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &ServerConfig {
            pipeline_parallel_runtime: Some(PipelineParallelRuntimeConfig::RemoteCoordinator(
                RemotePipelineRuntimeConfig {
                    stage_peers: vec![stage0_addr.clone(), stage1_addr.clone()],
                    transport_backend: TransportBackend::Tcp,
                    bind_address: loopback(coordinator_control_port).to_string(),
                    stage_timeout: Duration::from_secs(30),
                },
            )),
            ..ServerConfig::default()
        },
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
    )
    .unwrap();

    wait_for_loaded(&dense_provider);
    wait_for_loaded(&remote_provider);

    let options = ServerGenerateOptions {
        max_tokens: 8,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        priority: RequestPriority::Normal,
        logprobs: Default::default(),
        reasoning_budget: Default::default(),
        thinking_enter_block_on_start: false,
        prompt_cache_ctx: None,
        structured: None,
    };

    let dense = dense_provider
        .generate("Hello".to_string(), options.clone())
        .unwrap();
    let remote = remote_provider
        .generate("Hello".to_string(), options)
        .unwrap();

    drop(remote_provider);
    drop(dense_provider);
    stage0.shutdown().unwrap();
    stage1.shutdown().unwrap();

    assert_eq!(dense.text, remote.text);
    assert_eq!(dense.completion_tokens, remote.completion_tokens);
}
