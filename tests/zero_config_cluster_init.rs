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

//! Integration tests for the zero-config multi-machine PP bring-up path.
//!
//! The weightless tests drive the coordinator primitives (`discover_peers`,
//! `plan_cluster`, `push_stage_assignments`) against two `await_stage_assignment`
//! peers on loopback, in static and mdns mode, with a tempdir checkpoint that
//! holds only a `config.json` and an empty `model.safetensors` (#2658).
//!
//! The `#[ignore]` tests need a real checkpoint. `zero_config_cli_*` spawn
//! three real `mlxcel-server` processes (coordinator plus two `--pp-peer`)
//! on loopback and compare a greedy `/v1/chat/completions` answer with a
//! single-node server; `zero_config_two_stage_cluster_matches_dense_baseline`
//! drives the remote runtime primitives in-process.

mod common;

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{TestClusterSecretFile, repo_binary_path, repo_model_dir, test_cluster_auth};

use mlxcel::SamplingConfig;
use mlxcel::distributed::pipeline::{
    RemotePipelineRuntimeConfig, RemoteStageServiceConfig, RemoteStageServiceHandle,
    StageAssignment, resolve_in_process_pipeline_num_layers,
};
use mlxcel::distributed::{
    ClusterAuth, ClusterConfig, ClusterDiscoveryMode, ClusterInitRequest, CommittedAssignment,
    NodeRole, PeerJoinConfig, TransportBackend, await_stage_assignment, discover_peers,
    plan_cluster, push_stage_assignments, render_deterministic_toml,
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

fn reserve_udp_port() -> u16 {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind ephemeral udp port");
    socket.local_addr().expect("local addr").port()
}

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

/// A checkpoint directory without weights, enough for the model identity
/// (`config.json` hash, weight file list, layer count) and the partition.
fn weightless_model_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"llama","num_hidden_layers":4,"hidden_size":64,"vocab_size":128}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
    dir
}

fn peer_join_config(
    listen: SocketAddr,
    discovery: ClusterDiscoveryMode,
    discovery_port: u16,
    model_dir: &Path,
    auth: &Arc<ClusterAuth>,
) -> PeerJoinConfig {
    PeerJoinConfig {
        listen_addr: listen,
        discovery,
        discovery_port,
        discovery_target: IpAddr::V4(Ipv4Addr::LOCALHOST),
        cluster_name: "integration-zero-config".to_string(),
        node_id_hint: None,
        model_dir: model_dir.to_path_buf(),
        auth: Arc::clone(auth),
        assignment_timeout: Duration::from_secs(30),
        commit_timeout: Duration::from_secs(10),
    }
}

/// What both weightless bring-up tests assert about the committed peers.
fn assert_committed_pair(committed: &mut [CommittedAssignment], control: SocketAddr) {
    committed.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    assert_eq!(committed.len(), 2);
    assert_eq!(committed[0].node_id, "stage-0");
    assert_eq!(committed[0].stage_index, 0);
    assert_eq!(committed[1].node_id, "stage-1");
    assert_eq!(committed[1].stage_index, 1);
    for assignment in committed.iter() {
        let toml = render_deterministic_toml(&assignment.cluster);
        let parsed = ClusterConfig::from_toml(&toml).expect("received cluster TOML parses");
        assert!(parsed.cluster.secret_file.is_none(), "{toml}");
        assert!(!toml.contains("secret_file"), "{toml}");
        let coordinator = parsed.find_node("coordinator").expect("coordinator node");
        assert!(coordinator.address.ip().is_loopback(), "{toml}");
        assert_eq!(coordinator.address, control);
        assert_eq!(assignment.coordinator_identity, control.to_string());
        assert_eq!(parsed.cluster.pipeline_parallel_size, 2);
    }
}

/// Run the coordinator side exactly as `run_zero_config_bring_up` does:
/// discovery, plan, push.
async fn run_coordinator(
    mode: ClusterDiscoveryMode,
    discovery_port: u16,
    static_peers: &[SocketAddr],
    control: SocketAddr,
    model_dir: &Path,
    auth: &ClusterAuth,
) -> SocketAddr {
    let resolved = discover_peers(
        mode,
        "integration-zero-config",
        IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        discovery_port,
        static_peers,
        2,
        Duration::from_secs(15),
        Some(auth),
    )
    .await
    .expect("discovery resolves both peers");
    assert_eq!(resolved.len(), 2, "{resolved:?}");
    let mut request = sample_request(resolved);
    request.discovery = mode;
    request.discovery_port = discovery_port;
    request.coordinator_control_addr = control;
    let plan = plan_cluster(&request).expect("plan");
    push_stage_assignments(&plan, control, model_dir, auth, Duration::from_secs(30))
        .await
        .expect("push commits both peers");
    control
}

/// Static discovery: the coordinator knows the two peer listen addresses;
/// no TOML exists before the peers receive theirs.
#[tokio::test]
async fn zero_config_static_bring_up_without_toml() {
    let model = weightless_model_dir();
    let auth = test_cluster_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let control = loopback(reserve_port());
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| {
            tokio::spawn(await_stage_assignment(peer_join_config(
                *addr,
                ClusterDiscoveryMode::Static,
                0,
                model.path(),
                &auth,
            )))
        })
        .collect();

    run_coordinator(
        ClusterDiscoveryMode::Static,
        0,
        &peers,
        control,
        model.path(),
        &auth,
    )
    .await;

    let mut committed = Vec::new();
    for handle in handles {
        committed.push(handle.await.unwrap().expect("peer commits"));
    }
    assert_committed_pair(&mut committed, control);
}

/// mdns discovery: the peers announce themselves through
/// `broadcast_beacon_loop` with target `127.0.0.1`; the coordinator starts
/// with no peer list at all.
#[tokio::test]
async fn zero_config_mdns_bring_up_without_toml() {
    let model = weightless_model_dir();
    let auth = test_cluster_auth();
    let discovery_port = reserve_udp_port();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let control = loopback(reserve_port());
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| {
            tokio::spawn(await_stage_assignment(peer_join_config(
                *addr,
                ClusterDiscoveryMode::Mdns,
                discovery_port,
                model.path(),
                &auth,
            )))
        })
        .collect();

    run_coordinator(
        ClusterDiscoveryMode::Mdns,
        discovery_port,
        &[],
        control,
        model.path(),
        &auth,
    )
    .await;

    let mut committed = Vec::new();
    for handle in handles {
        committed.push(handle.await.unwrap().expect("peer commits"));
    }
    assert_committed_pair(&mut committed, control);
    let own: Vec<SocketAddr> = committed
        .iter()
        .map(|c| c.cluster.find_node(&c.node_id).unwrap().address)
        .collect();
    let mut expected = peers.to_vec();
    expected.sort_by_key(|a| a.to_string());
    assert_eq!(own, expected, "stage order follows the peer address order");
}

/// Loopback ports the real-binary tests bind. The block is the one the
/// zero-config end-to-end rehearsal on the GB10 host is allotted; a port that
/// is busy is skipped, and when the block is exhausted the OS picks one.
const CLI_TEST_PORT_BLOCK: std::ops::RangeInclusive<u16> = 19500..=19549;

fn free_loopback_ports(count: usize, udp: bool) -> Vec<u16> {
    let mut ports = Vec::with_capacity(count);
    for port in CLI_TEST_PORT_BLOCK {
        if ports.len() == count {
            break;
        }
        let free = if udp {
            std::net::UdpSocket::bind(("127.0.0.1", port)).is_ok()
        } else {
            TcpListener::bind(("127.0.0.1", port)).is_ok()
        };
        if free {
            ports.push(port);
        }
    }
    while ports.len() < count {
        ports.push(if udp {
            reserve_udp_port()
        } else {
            reserve_port()
        });
    }
    ports
}

/// Kills its child on drop so a failed assertion never leaks a server.
struct ChildGuard {
    child: Child,
    log: PathBuf,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ChildGuard {
    fn log_tail(&self) -> String {
        let text = std::fs::read_to_string(&self.log).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(40);
        lines[start..].join("\n")
    }
}

fn spawn_server(log_dir: &Path, name: &str, args: &[String]) -> ChildGuard {
    let log = log_dir.join(format!("{name}.log"));
    let file = std::fs::File::create(&log).expect("create server log");
    let child = Command::new(repo_binary_path("mlxcel-server"))
        .args(args)
        .current_dir(log_dir)
        .env("MLXCEL_SDPA_DETERMINISTIC", "1")
        .stdout(Stdio::from(file.try_clone().expect("clone log handle")))
        .stderr(Stdio::from(file))
        .spawn()
        .unwrap_or_else(|err| panic!("spawn mlxcel-server {name}: {err}"));
    ChildGuard { child, log }
}

async fn wait_for_models(client: &reqwest::Client, port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(response) = client
            .get(format!("http://127.0.0.1:{port}/v1/models"))
            .send()
            .await
            && response.status().is_success()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

async fn greedy_chat(client: &reqwest::Client, port: u16) -> String {
    let body = serde_json::json!({
        "model": "zero-config",
        "messages": [{"role": "user", "content": "Name three primary colors."}],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("chat completion request");
    assert!(
        response.status().is_success(),
        "chat completion on port {port} returned {}",
        response.status()
    );
    let value: serde_json::Value = response.json().await.expect("chat completion JSON");
    value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no message content in {value}"))
        .to_string()
}

/// Spawn two `--pp-peer` processes and a `--pp-auto 2` coordinator in
/// `mode`, plus a single-node server, and compare their greedy answers.
async fn zero_config_cli_matches_single_node(mode: ClusterDiscoveryMode) {
    let model_dir = repo_model_dir("qwen3-0.6b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }
    let binary = repo_binary_path("mlxcel-server");
    if !binary.exists() {
        eprintln!("Skipping test: {} is not built", binary.display());
        return;
    }
    let model = model_dir.to_string_lossy().into_owned();
    let secret = TestClusterSecretFile::new();
    let work = tempfile::tempdir().expect("work dir");
    let tcp = free_loopback_ports(7, false);
    let (stage0, stage1, control, http, single, peer_http0, peer_http1) =
        (tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5], tcp[6]);
    let discovery_port = free_loopback_ports(1, true)[0];
    let mode_arg = format!("--cluster-discovery={mode}");

    let peer_args = |stage_port: u16, http_port: u16, index: usize| -> Vec<String> {
        let mut args = vec![
            "-m".to_string(),
            model.clone(),
            "--pp-peer".to_string(),
            mode_arg.clone(),
            "--cluster-control-addr".to_string(),
            format!("127.0.0.1:{stage_port}"),
            "--cluster-secret-file".to_string(),
            secret.path_arg().to_string(),
            "--cluster-config-out".to_string(),
            work.path()
                .join(format!("peer{index}.toml"))
                .display()
                .to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            http_port.to_string(),
        ];
        if mode == ClusterDiscoveryMode::Mdns {
            args.extend([
                "--cluster-discovery-target".to_string(),
                "127.0.0.1".to_string(),
                "--cluster-discovery-port".to_string(),
                discovery_port.to_string(),
            ]);
        }
        args
    };
    let _peer0 = spawn_server(work.path(), "peer0", &peer_args(stage0, peer_http0, 0));
    let _peer1 = spawn_server(work.path(), "peer1", &peer_args(stage1, peer_http1, 1));

    let mut coordinator_args = vec![
        "-m".to_string(),
        model.clone(),
        "--pp-auto".to_string(),
        "2".to_string(),
        mode_arg,
        "--cluster-control-addr".to_string(),
        format!("127.0.0.1:{control}"),
        "--cluster-secret-file".to_string(),
        secret.path_arg().to_string(),
        "--cluster-config-out".to_string(),
        work.path().join("coordinator.toml").display().to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        http.to_string(),
    ];
    match mode {
        ClusterDiscoveryMode::Mdns => coordinator_args.extend([
            "--cluster-discovery-port".to_string(),
            discovery_port.to_string(),
        ]),
        _ => coordinator_args.extend([
            "--cluster-peers".to_string(),
            format!("127.0.0.1:{stage0},127.0.0.1:{stage1}"),
        ]),
    }
    let coordinator = spawn_server(work.path(), "coordinator", &coordinator_args);
    let single_node = spawn_server(
        work.path(),
        "single",
        &[
            "-m".to_string(),
            model,
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            single.to_string(),
        ],
    );

    let client = reqwest::Client::new();
    assert!(
        wait_for_models(&client, single, Duration::from_secs(240)).await,
        "single-node server did not come up:\n{}",
        single_node.log_tail()
    );
    assert!(
        wait_for_models(&client, http, Duration::from_secs(240)).await,
        "zero-config coordinator did not come up ({mode}):\ncoordinator:\n{}\npeer0:\n{}\npeer1:\n{}",
        coordinator.log_tail(),
        _peer0.log_tail(),
        _peer1.log_tail()
    );
    for index in 0..2 {
        let toml_path = work.path().join(format!("peer{index}.toml"));
        let parsed = ClusterConfig::from_file(&toml_path).expect("peer wrote its cluster TOML");
        assert!(parsed.cluster.secret_file.is_none());
    }

    let expected = greedy_chat(&client, single).await;
    let actual = greedy_chat(&client, http).await;
    assert!(!expected.trim().is_empty(), "single-node answer is empty");
    assert_eq!(
        actual,
        expected,
        "zero-config {mode} pipeline output differs from single-node\ncoordinator:\n{}",
        coordinator.log_tail()
    );
}

#[tokio::test]
#[ignore = "requires qwen3-0.6b-4bit weights and spawns four mlxcel-server processes"]
async fn zero_config_cli_static_matches_single_node() {
    zero_config_cli_matches_single_node(ClusterDiscoveryMode::Static).await;
}

#[tokio::test]
#[ignore = "requires qwen3-0.6b-4bit weights and spawns four mlxcel-server processes"]
async fn zero_config_cli_mdns_matches_single_node() {
    zero_config_cli_matches_single_node(ClusterDiscoveryMode::Mdns).await;
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
        secret_file: None,
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
    let model_dir = repo_model_dir("llama-3.2-1b-instruct-4bit");
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
        secret_file: None,
    };
    let plan = plan_cluster(&request).expect("zero-config plan succeeds");
    let cluster = plan.cluster;

    let num_layers = resolve_in_process_pipeline_num_layers(&model_dir).unwrap();
    let assignments = two_stage_assignments(num_layers, num_layers / 2);

    let stage0_addr = cluster.pipeline_stage_node(0).unwrap().address.to_string();
    let stage1_addr = cluster.pipeline_stage_node(1).unwrap().address.to_string();

    let auth = test_cluster_auth();
    let coordinator_addr = loopback(coordinator_control_port).to_string();
    let stage1 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage1_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[1].clone(),
        num_stages: 2,
        coordinator_peer: coordinator_addr.clone(),
        upstream_peer: Some(stage0_addr.clone()),
        downstream_peer: None,
        auth: Arc::clone(&auth),
    })
    .unwrap();
    let stage0 = RemoteStageServiceHandle::spawn(RemoteStageServiceConfig {
        model_dir: model_dir.clone(),
        bind_address: stage0_addr.clone(),
        transport_backend: TransportBackend::Tcp,
        stage_assignment: assignments[0].clone(),
        num_stages: 2,
        coordinator_peer: coordinator_addr.clone(),
        upstream_peer: None,
        downstream_peer: Some(stage1_addr.clone()),
        auth: Arc::clone(&auth),
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
                    bind_address: coordinator_addr,
                    stage_timeout: Duration::from_secs(30),
                    auth,
                    vocab_size: None,
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
        n_indent: 0,
        t_max_predict_ms: None,
        reasoning_budget_message: None,
        retention: Default::default(),
        dry_breaker_strings: None,
        logit_bias: Vec::new(),
        logit_bias_texts: Vec::new(),
        post_sampling_probs: false,
        max_tokens: 8,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        ignore_eos: false,
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs: Default::default(),
        reasoning_budget: Default::default(),
        thinking_enter_block_on_start: false,
        reasoning_control: None,
        prompt_cache_ctx: None,
        structured: None,
        grammar: None,
        image_soft_tokens: None,
        pre_rendered_prompt_tokens: None,
        admission_notification: None,
        count_prompt_only: false,
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
