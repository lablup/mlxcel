// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use super::*;

fn sample_request_for(stages: u32, peers: &[&str]) -> ClusterInitRequest {
    ClusterInitRequest {
        pp_stages: stages,
        cluster_name: "zero-config-test".to_string(),
        transport_backend: TransportBackend::Tcp,
        discovery: ClusterDiscoveryMode::Static,
        discovery_timeout: None,
        discovery_port: DEFAULT_DISCOVERY_PORT,
        coordinator_http_addr: "127.0.0.1:8080".parse().unwrap(),
        coordinator_control_addr: "127.0.0.1:19000".parse().unwrap(),
        static_peers: peers.iter().map(|p| p.parse().unwrap()).collect(),
        data_port_base: 19001,
        output_toml_path: None,
    }
}

#[test]
fn plan_cluster_2_stage_matches_expected_topology() {
    let request = sample_request_for(2, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    let plan = plan_cluster(&request).expect("plan succeeds");
    assert_eq!(plan.cluster.nodes.len(), 3);
    assert_eq!(plan.cluster.cluster.pipeline_parallel_size, 2);
    assert_eq!(plan.cluster.cluster.tensor_parallel_size, 1);

    // Coordinator is first (stage == None).
    assert_eq!(plan.cluster.nodes[0].id, "coordinator");
    assert!(plan.cluster.nodes[0].stage.is_none());

    // Stages sorted by stage index.
    assert_eq!(plan.cluster.nodes[1].id, "stage-0");
    assert_eq!(plan.cluster.nodes[1].stage, Some(0));
    assert_eq!(plan.cluster.nodes[2].id, "stage-1");
    assert_eq!(plan.cluster.nodes[2].stage, Some(1));
}

#[test]
fn plan_cluster_stage_count_mismatch_is_actionable_error() {
    // Request 3 stages, give 2 peers.
    let request = sample_request_for(3, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    let err = plan_cluster(&request).unwrap_err().to_string();
    assert!(
        err.contains("peer(s) but --pp-auto requested"),
        "error missing guidance: {err}"
    );
}

#[test]
fn plan_cluster_requires_pp_stages_at_least_two() {
    let request = sample_request_for(1, &["192.168.1.10:19001"]);
    let err = plan_cluster(&request).unwrap_err().to_string();
    assert!(
        err.contains("at least 2 pipeline stages"),
        "error should mention the >= 2 requirement: {err}"
    );
}

#[test]
fn plan_cluster_reordered_peers_produce_same_topology() {
    let forward = sample_request_for(2, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    let reversed = sample_request_for(2, &["192.168.1.11:19001", "192.168.1.10:19001"]);
    let fwd_plan = plan_cluster(&forward).unwrap();
    let rev_plan = plan_cluster(&reversed).unwrap();

    // Stage assignment is tied to sorted peer ordering, so both plans are
    // byte-identical.
    assert_eq!(fwd_plan.toml, rev_plan.toml);
}

#[test]
fn plan_cluster_emits_byte_identical_toml() {
    let request = sample_request_for(2, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    let plan1 = plan_cluster(&request).unwrap();
    let plan2 = plan_cluster(&request).unwrap();
    assert_eq!(plan1.toml, plan2.toml);

    // Also reparse to confirm the TOML validates against the existing ClusterConfig loader.
    let reloaded = ClusterConfig::from_toml(&plan1.toml).unwrap();
    assert_eq!(reloaded.cluster.pipeline_parallel_size, request.pp_stages);
}

#[test]
fn plan_cluster_rejects_control_http_port_collision() {
    let mut request = sample_request_for(2, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    request.coordinator_control_addr = request.coordinator_http_addr;
    let err = plan_cluster(&request).unwrap_err().to_string();
    assert!(
        err.contains("conflicts with the HTTP listen address"),
        "expected control/http conflict diagnostic: {err}"
    );
}

#[test]
fn render_deterministic_toml_is_stable() {
    let request = sample_request_for(3, &["10.0.0.1:19001", "10.0.0.3:19001", "10.0.0.2:19001"]);
    let plan = plan_cluster(&request).unwrap();
    let toml = plan.toml.clone();
    // Reparse and re-render: result should be byte-identical (deterministic).
    let reloaded = ClusterConfig::from_toml(&toml).unwrap();
    let toml2 = render_deterministic_toml(&reloaded);
    assert_eq!(toml, toml2);
    // Stage ordering is preserved in textual form.
    assert!(toml.contains("stage = 0"));
    assert!(toml.contains("stage = 1"));
    assert!(toml.contains("stage = 2"));
}

#[test]
fn toml_escape_string_handles_special_chars() {
    assert_eq!(super::toml_escape_string("plain"), "\"plain\"");
    assert_eq!(
        super::toml_escape_string("with\"quote"),
        "\"with\\\"quote\""
    );
    assert_eq!(
        super::toml_escape_string("back\\slash"),
        "\"back\\\\slash\""
    );
    // Newlines become \n escapes.
    assert_eq!(super::toml_escape_string("a\nb"), "\"a\\nb\"");
}

#[test]
fn discovery_mode_parses_common_aliases() {
    use std::str::FromStr;
    assert_eq!(
        ClusterDiscoveryMode::from_str("mdns").unwrap(),
        ClusterDiscoveryMode::Mdns
    );
    assert_eq!(
        ClusterDiscoveryMode::from_str("udp").unwrap(),
        ClusterDiscoveryMode::Mdns
    );
    assert_eq!(
        ClusterDiscoveryMode::from_str("broadcast").unwrap(),
        ClusterDiscoveryMode::Mdns
    );
    assert_eq!(
        ClusterDiscoveryMode::from_str("static").unwrap(),
        ClusterDiscoveryMode::Static
    );
    assert_eq!(
        ClusterDiscoveryMode::from_str("off").unwrap(),
        ClusterDiscoveryMode::Static
    );
    assert!(ClusterDiscoveryMode::from_str("quic").is_err());
}

#[test]
fn allocate_data_ports_returns_distinct_increasing_values() {
    let ports =
        allocate_data_ports(IpAddr::V4(Ipv4Addr::LOCALHOST), 41000, 3).expect("allocate ports");
    assert_eq!(ports.len(), 3);
    for pair in ports.windows(2) {
        assert!(pair[0] < pair[1], "ports must increase: {:?}", ports);
    }
}

#[test]
fn is_port_available_detects_bound_port() {
    // Bind an ephemeral port so we know it's in use.
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    assert!(
        !is_port_available(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
        "port {port} should be unavailable while the listener holds it"
    );
    drop(listener);
}

#[test]
fn write_plan_toml_persists_generated_bytes() {
    let tmp = std::env::temp_dir().join(format!(
        "mlxcel-cluster-init-test-{}.toml",
        std::process::id()
    ));
    let request = sample_request_for(2, &["192.168.1.10:19001", "192.168.1.11:19001"]);
    let plan = plan_cluster(&request).unwrap();
    write_plan_toml(&plan, &tmp).unwrap();
    let back = std::fs::read_to_string(&tmp).unwrap();
    assert_eq!(back, plan.toml);
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn discover_peers_static_mode_short_circuits() {
    let seeds: Vec<SocketAddr> = vec![
        "192.168.1.10:19001".parse().unwrap(),
        "192.168.1.11:19001".parse().unwrap(),
    ];
    let result = discover_peers(
        ClusterDiscoveryMode::Static,
        "test-cluster",
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        0,
        &seeds,
        2,
        Duration::from_millis(100),
    )
    .await
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].to_string(), "192.168.1.10:19001");
    assert_eq!(result[1].to_string(), "192.168.1.11:19001");
}

#[tokio::test]
async fn discover_peers_mdns_times_out_with_actionable_error() {
    // Use an unroutable target port for a guaranteed timeout.
    let err = discover_peers(
        ClusterDiscoveryMode::Mdns,
        "nonexistent-test-cluster",
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        0,
        &[],
        2,
        Duration::from_millis(150),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("cluster discovery timed out"),
        "error must mention timeout: {err}"
    );
    assert!(
        err.contains("--cluster-discovery=static"),
        "error must suggest fallback: {err}"
    );
}
