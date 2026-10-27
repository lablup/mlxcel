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

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpStream;

use super::*;
use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::cluster_init::{ClusterInitRequest, plan_cluster};
use crate::distributed::transport::TransportBackend;

fn test_auth() -> Arc<ClusterAuth> {
    Arc::new(ClusterAuth::new(&ClusterSecret::random()))
}

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

/// Reserve a loopback TCP port. The socket is dropped before the port is
/// handed back; the coordinator's connect retry covers the small window
/// before the peer rebinds it.
fn reserve_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

/// A checkpoint directory without weights: a `config.json` with
/// `num_hidden_layers` and an empty `model.safetensors`.
fn temp_model_dir(num_layers: usize, vocab: u32) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        format!(
            r#"{{"model_type":"llama","num_hidden_layers":{num_layers},"hidden_size":64,"vocab_size":{vocab}}}"#
        ),
    )
    .unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
    dir
}

fn plan_for(peers: &[SocketAddr], control: SocketAddr) -> ClusterInitPlan {
    plan_cluster(&ClusterInitRequest {
        pp_stages: peers.len() as u32,
        cluster_name: "assignment-test".to_string(),
        transport_backend: TransportBackend::Tcp,
        discovery: ClusterDiscoveryMode::Static,
        discovery_timeout: None,
        discovery_port: 0,
        coordinator_http_addr: loopback(18080),
        coordinator_control_addr: control,
        static_peers: peers.to_vec(),
        data_port_base: 19001,
        output_toml_path: None,
        secret_file: Some(std::path::PathBuf::from("/etc/mlxcel/cluster.secret")),
    })
    .expect("plan")
}

fn peer_config(listen: SocketAddr, model_dir: &Path, auth: &Arc<ClusterAuth>) -> PeerJoinConfig {
    PeerJoinConfig {
        listen_addr: listen,
        discovery: ClusterDiscoveryMode::Static,
        discovery_port: 0,
        discovery_target: IpAddr::V4(Ipv4Addr::LOCALHOST),
        cluster_name: "assignment-test".to_string(),
        node_id_hint: None,
        model_dir: model_dir.to_path_buf(),
        auth: Arc::clone(auth),
        assignment_timeout: Duration::from_secs(10),
        commit_timeout: Duration::from_secs(5),
    }
}

fn spawn_peer(config: PeerJoinConfig) -> tokio::task::JoinHandle<Result<CommittedAssignment>> {
    tokio::spawn(await_stage_assignment(config))
}

/// Wait until a TCP connect to `addr` succeeds, so a test's raw client does
/// not race the peer's bind.
async fn wait_for_listener(addr: SocketAddr) {
    for _ in 0..200 {
        if let Ok(stream) = TcpStream::connect(addr).await {
            drop(stream);
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("peer listener on {addr} did not come up");
}

fn sample_offer(
    node_id: &str,
    stage_index: u32,
    model_dir: &Path,
    plan: &ClusterInitPlan,
) -> StageAssignmentOffer {
    let model = ModelIdentity::from_model_dir(model_dir).unwrap();
    let num_stages = plan.cluster.cluster.pipeline_parallel_size;
    let (assignments, _) = resolve_in_process_stage_assignments_for_model(
        model_dir,
        model.num_layers,
        Some(num_stages as usize),
        None,
    )
    .unwrap();
    let mut cluster = plan.cluster.clone();
    cluster.cluster.secret_file = None;
    StageAssignmentOffer {
        protocol: ASSIGNMENT_PROTOCOL_VERSION,
        mlxcel_version: env!("CARGO_PKG_VERSION").to_string(),
        cluster_name: cluster.cluster.name.clone(),
        node_id: node_id.to_string(),
        stage_index,
        num_stages,
        layer_range: assignments[stage_index as usize].layer_range.clone(),
        coordinator_control_addr: plan
            .cluster
            .nodes
            .iter()
            .find(|n| n.role != NodeRole::PipelineStage)
            .unwrap()
            .address,
        model,
        cluster_toml: render_deterministic_toml(&cluster),
    }
}

#[test]
fn assignment_frames_round_trip_and_reject_oversize() {
    let dir = temp_model_dir(4, 100);
    let plan = plan_for(&[loopback(19001), loopback(19002)], loopback(19000));
    let frames = [
        AssignmentFrame::Offer(sample_offer("stage-0", 0, dir.path(), &plan)),
        AssignmentFrame::Accepted {
            node_id: "stage-0".to_string(),
        },
        AssignmentFrame::Rejected {
            reason: AssignmentRejectReason::ModelMismatch,
            detail: "different checkpoint".to_string(),
        },
        AssignmentFrame::Commit,
        AssignmentFrame::Committed {
            node_id: "stage-1".to_string(),
        },
    ];
    for frame in &frames {
        let bytes = encode_frame(frame).unwrap();
        assert_eq!(&decode_frame(&bytes).unwrap(), frame, "{}", frame.kind());
        let truncated = &bytes[..bytes.len() - 1];
        assert!(
            decode_frame(truncated).is_err(),
            "truncated {} decoded",
            frame.kind()
        );
    }

    // A payload one byte over the limit is refused by the encoder.
    let too_big = AssignmentFrame::Rejected {
        reason: AssignmentRejectReason::InvalidClusterConfig,
        detail: "x".repeat(MAX_ASSIGNMENT_FRAME_BYTES),
    };
    let err = encode_frame(&too_big).expect_err("oversize frame must not encode");
    assert!(err.to_string().contains("exceeds"), "{err}");

    // A prefix announcing more than the limit is refused before any body is
    // read, whatever follows it.
    let mut oversize_prefix = ((MAX_ASSIGNMENT_FRAME_BYTES + 1) as u32)
        .to_be_bytes()
        .to_vec();
    oversize_prefix.extend(std::iter::repeat_n(b'{', MAX_ASSIGNMENT_FRAME_BYTES + 1));
    let err = decode_frame(&oversize_prefix).expect_err("oversize prefix must be refused");
    assert!(err.to_string().contains("above the"), "{err}");
    assert!(decode_frame(&[0, 0]).is_err());
}

#[test]
fn model_identity_changes_with_config_or_weight_files() {
    let a = temp_model_dir(4, 100);
    let b = temp_model_dir(4, 100);
    let id_a = ModelIdentity::from_model_dir(a.path()).unwrap();
    assert_eq!(id_a, ModelIdentity::from_model_dir(b.path()).unwrap());
    assert_eq!(id_a.num_layers, 4);
    assert_eq!(
        id_a.weight_files,
        vec![("model.safetensors".to_string(), 0)]
    );

    // A different config changes the hash (and here the layer count).
    let other_config = temp_model_dir(6, 100);
    let id_c = ModelIdentity::from_model_dir(other_config.path()).unwrap();
    assert_ne!(id_a.config_sha256, id_c.config_sha256);
    assert_eq!(id_c.num_layers, 6);

    // An extra shard, or a shard of another size, changes the identity even
    // with the same config.
    std::fs::write(b.path().join("model-00002-of-00002.safetensors"), b"abc").unwrap();
    let id_b = ModelIdentity::from_model_dir(b.path()).unwrap();
    assert_eq!(id_b.config_sha256, id_a.config_sha256);
    assert_ne!(id_b, id_a);
    assert_eq!(
        id_b.weight_files,
        vec![
            ("model-00002-of-00002.safetensors".to_string(), 3),
            ("model.safetensors".to_string(), 0),
        ]
    );
    std::fs::write(a.path().join("model.safetensors"), b"zz").unwrap();
    assert_ne!(ModelIdentity::from_model_dir(a.path()).unwrap(), id_a);
    assert!(id_a.describe_difference(&id_c).contains("4 vs 6 layers"));
}

#[tokio::test]
async fn peer_commits_exactly_one_assignment() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let plan = plan_for(&peers, loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, dir.path(), &auth)))
        .collect();

    push_stage_assignments(&plan, control, dir.path(), &auth, Duration::from_secs(10))
        .await
        .expect("first push commits both peers");

    let mut committed: Vec<CommittedAssignment> = Vec::new();
    for handle in handles {
        committed.push(handle.await.unwrap().expect("peer commits"));
    }
    committed.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    assert_eq!(committed[0].node_id, "stage-0");
    assert_eq!(committed[0].stage_index, 0);
    assert_eq!(committed[1].node_id, "stage-1");
    assert_eq!(committed[1].stage_index, 1);
    for assignment in &committed {
        assert_eq!(assignment.coordinator_identity, control.to_string());
        assert!(assignment.cluster.cluster.secret_file.is_none());
        assert_eq!(
            assignment.cluster.find_node("coordinator").unwrap().address,
            control
        );
        let toml = render_deterministic_toml(&assignment.cluster);
        ClusterConfig::from_toml(&toml).expect("localized cluster reparses");
    }

    // The listeners are closed: a second push finds nobody and names both.
    let err = push_stage_assignments(
        &plan,
        control,
        dir.path(),
        &auth,
        Duration::from_millis(700),
    )
    .await
    .expect_err("a committed peer accepts no second assignment");
    let failed = err
        .downcast_ref::<AssignmentPushFailed>()
        .expect("typed push failure");
    assert_eq!(failed.failures.len(), 2, "{err}");
    assert!(err.to_string().contains("stage-0"), "{err}");
    assert!(err.to_string().contains("stage-1"), "{err}");
}

#[tokio::test]
async fn peer_refuses_model_mismatch_with_typed_error() {
    let peer_model = temp_model_dir(4, 100);
    let coordinator_model = temp_model_dir(4, 200);
    let auth = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let plan = plan_for(&peers, loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, peer_model.path(), &auth)))
        .collect();

    let err = push_stage_assignments(
        &plan,
        control,
        coordinator_model.path(),
        &auth,
        Duration::from_secs(10),
    )
    .await
    .expect_err("a different checkpoint must not be assigned");
    let failed = err.downcast_ref::<AssignmentPushFailed>().unwrap();
    assert_eq!(failed.failures.len(), 2);
    assert!(err.to_string().contains("stage-0"), "{err}");
    assert!(err.to_string().contains("model mismatch"), "{err}");

    for handle in handles {
        let peer_err = handle
            .await
            .unwrap()
            .expect_err("peer ends with a typed refusal");
        let refused = peer_err
            .downcast_ref::<AssignmentRefused>()
            .unwrap_or_else(|| panic!("expected AssignmentRefused, got {peer_err:#}"));
        assert_eq!(refused.reason, AssignmentRejectReason::ModelMismatch);
        assert!(
            refused.detail.contains("config.json sha256"),
            "{}",
            refused.detail
        );
    }
}

#[tokio::test]
async fn peer_refuses_version_mismatch() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peer_addr = loopback(reserve_port());
    let other = loopback(reserve_port());
    let plan = plan_for(&[peer_addr, other], loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let (node_id, stage_index) = {
        let node = plan
            .cluster
            .nodes
            .iter()
            .find(|n| n.address == peer_addr)
            .unwrap();
        (node.id.clone(), node.stage.unwrap())
    };
    let handle = spawn_peer(peer_config(peer_addr, dir.path(), &auth));
    wait_for_listener(peer_addr).await;

    let mut stream = TcpStream::connect(peer_addr).await.unwrap();
    let keys = client_handshake(
        &mut stream,
        &auth,
        &control.to_string(),
        FrameSuite::default(),
    )
    .await
    .expect("raw coordinator authenticates");
    let mut stream = ProtectedStream::protected(stream, keys);
    let mut offer = sample_offer(&node_id, stage_index, dir.path(), &plan);
    offer.mlxcel_version = "0.0.0".to_string();
    write_frame(&mut stream, &AssignmentFrame::Offer(offer))
        .await
        .unwrap();
    match read_frame(&mut stream).await.unwrap() {
        AssignmentFrame::Rejected { reason, detail } => {
            assert_eq!(reason, AssignmentRejectReason::VersionMismatch);
            assert!(detail.contains("0.0.0"), "{detail}");
        }
        other => panic!("expected Rejected, got {}", other.kind()),
    }

    let peer_err = handle
        .await
        .unwrap()
        .expect_err("version mismatch is fatal");
    let refused = peer_err.downcast_ref::<AssignmentRefused>().unwrap();
    assert_eq!(refused.reason, AssignmentRejectReason::VersionMismatch);
}

#[tokio::test]
async fn peer_discards_offer_without_commit() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let plan = plan_for(&peers, loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, dir.path(), &auth)))
        .collect();
    wait_for_listener(peers[0]).await;

    // A coordinator that offers, sees Accepted, and dies before Commit.
    {
        let node = plan
            .cluster
            .nodes
            .iter()
            .find(|n| n.address == peers[0])
            .unwrap();
        let mut stream = TcpStream::connect(peers[0]).await.unwrap();
        let keys = client_handshake(
            &mut stream,
            &auth,
            &control.to_string(),
            FrameSuite::default(),
        )
        .await
        .unwrap();
        let mut stream = ProtectedStream::protected(stream, keys);
        let offer = sample_offer(&node.id, node.stage.unwrap(), dir.path(), &plan);
        write_frame(&mut stream, &AssignmentFrame::Offer(offer))
            .await
            .unwrap();
        match read_frame(&mut stream).await.unwrap() {
            AssignmentFrame::Accepted { node_id } => assert_eq!(node_id, node.id),
            other => panic!("expected Accepted, got {}", other.kind()),
        }
        drop(stream);
    }

    // The peer discarded that offer and still takes a real push.
    push_stage_assignments(&plan, control, dir.path(), &auth, Duration::from_secs(10))
        .await
        .expect("push after a dropped offer commits");
    for handle in handles {
        handle.await.unwrap().expect("peer commits once");
    }
}

#[tokio::test]
async fn peer_times_out_without_assignment() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let mut config = peer_config(loopback(0), dir.path(), &auth);
    config.assignment_timeout = Duration::from_millis(200);
    let started = std::time::Instant::now();
    let err = await_stage_assignment(config)
        .await
        .expect_err("no coordinator means a timeout");
    assert!(started.elapsed() < Duration::from_secs(5));
    let message = err.to_string();
    assert!(
        message.contains("no stage assignment received within"),
        "{message}"
    );
    assert!(message.contains("assignment-test"), "{message}");
    assert!(message.contains("static"), "{message}");
}

#[tokio::test]
async fn peer_ignores_coordinator_with_wrong_secret() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let wrong = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let plan = plan_for(&peers, loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, dir.path(), &auth)))
        .collect();
    wait_for_listener(peers[0]).await;
    wait_for_listener(peers[1]).await;

    let err = push_stage_assignments(&plan, control, dir.path(), &wrong, Duration::from_secs(10))
        .await
        .expect_err("a coordinator with another secret assigns nothing");
    let failed = err.downcast_ref::<AssignmentPushFailed>().unwrap();
    assert_eq!(failed.failures.len(), 2, "{err}");
    assert!(err.to_string().contains("handshake failed"), "{err}");

    push_stage_assignments(&plan, control, dir.path(), &auth, Duration::from_secs(10))
        .await
        .expect("the real coordinator is still accepted afterwards");
    for handle in handles {
        handle
            .await
            .unwrap()
            .expect("peer commits to the real coordinator");
    }
}

#[tokio::test]
async fn coordinator_names_peers_that_did_not_commit() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let live = loopback(reserve_port());
    let dead = loopback(reserve_port());
    let plan = plan_for(&[live, dead], loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let dead_id = plan
        .cluster
        .nodes
        .iter()
        .find(|n| n.address == dead)
        .unwrap()
        .id
        .clone();
    let live_id = plan
        .cluster
        .nodes
        .iter()
        .find(|n| n.address == live)
        .unwrap()
        .id
        .clone();
    let mut config = peer_config(live, dir.path(), &auth);
    config.assignment_timeout = Duration::from_millis(1500);
    config.commit_timeout = Duration::from_millis(500);
    let handle = spawn_peer(config);
    wait_for_listener(live).await;

    let err = push_stage_assignments(
        &plan,
        control,
        dir.path(),
        &auth,
        Duration::from_millis(300),
    )
    .await
    .expect_err("a dead peer fails the push");
    let failed = err.downcast_ref::<AssignmentPushFailed>().unwrap();
    assert_eq!(failed.failures.len(), 1, "{err}");
    assert_eq!(failed.failures[0].node_id, dead_id);
    assert_eq!(failed.failures[0].addr, dead);
    assert!(failed.failures[0].reason.contains("timed out"), "{err}");
    assert!(
        failed.failures[0].reason.contains("never connected")
            && failed.failures[0].reason.contains("last connect error"),
        "{err}"
    );
    assert!(!err.to_string().contains(&format!("{live_id} at")), "{err}");

    // The live peer accepted but never saw a Commit, so it kept waiting
    // until its own timeout instead of committing.
    let peer_err = handle
        .await
        .unwrap()
        .expect_err("the live peer must not commit without Commit");
    assert!(
        peer_err
            .to_string()
            .contains("no stage assignment received within"),
        "{peer_err:#}"
    );
}

#[tokio::test]
async fn peer_substitutes_bound_identity_for_unspecified_coordinator() {
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let control_port = reserve_port();
    let unspecified_control = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), control_port);
    let plan = plan_for(&peers, unspecified_control);
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, dir.path(), &auth)))
        .collect();

    push_stage_assignments(
        &plan,
        unspecified_control,
        dir.path(),
        &auth,
        Duration::from_secs(10),
    )
    .await
    .expect("push with an unspecified control address");

    for (handle, listen) in handles.into_iter().zip(peers) {
        let committed = handle.await.unwrap().expect("peer commits");
        assert_eq!(
            committed.coordinator_identity,
            loopback(control_port).to_string()
        );
        let coordinator = committed.cluster.find_node("coordinator").unwrap();
        assert_eq!(coordinator.address, loopback(control_port));
        let own = committed.cluster.find_node(&committed.node_id).unwrap();
        assert_eq!(own.address, listen);
        assert!(
            render_deterministic_toml(&committed.cluster)
                .contains(&format!("address = \"{}\"", loopback(control_port)))
        );
    }
}

#[tokio::test]
async fn peer_waits_for_commit_beyond_frame_io_timeout() {
    // An accepted peer idles while the coordinator is still connecting to a
    // later peer (a static peer may start up to the 60 s push bound after the
    // first). Its wait for `Commit` is bounded by the commit timeout, not by
    // the 10 s per-frame bound, or such a bring-up fails on the early peer.
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peer_addr = loopback(reserve_port());
    let other = loopback(reserve_port());
    let plan = plan_for(&[peer_addr, other], loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let node = plan
        .cluster
        .nodes
        .iter()
        .find(|n| n.address == peer_addr)
        .unwrap()
        .clone();
    let mut config = peer_config(peer_addr, dir.path(), &auth);
    config.assignment_timeout = ASSIGNMENT_IO_TIMEOUT + Duration::from_secs(20);
    config.commit_timeout = ASSIGNMENT_IO_TIMEOUT + Duration::from_secs(10);
    let handle = spawn_peer(config);
    wait_for_listener(peer_addr).await;

    let mut stream = TcpStream::connect(peer_addr).await.unwrap();
    let keys = client_handshake(
        &mut stream,
        &auth,
        &control.to_string(),
        FrameSuite::default(),
    )
    .await
    .unwrap();
    let mut stream = ProtectedStream::protected(stream, keys);
    let offer = sample_offer(&node.id, node.stage.unwrap(), dir.path(), &plan);
    write_frame(&mut stream, &AssignmentFrame::Offer(offer))
        .await
        .unwrap();
    match read_frame(&mut stream).await.unwrap() {
        AssignmentFrame::Accepted { node_id } => assert_eq!(node_id, node.id),
        other => panic!("expected Accepted, got {}", other.kind()),
    }

    tokio::time::sleep(ASSIGNMENT_IO_TIMEOUT + Duration::from_secs(1)).await;
    write_frame(&mut stream, &AssignmentFrame::Commit)
        .await
        .expect("the accepted peer still holds the connection");
    match read_frame(&mut stream).await.unwrap() {
        AssignmentFrame::Committed { node_id } => assert_eq!(node_id, node.id),
        other => panic!("expected Committed, got {}", other.kind()),
    }
    let committed = handle
        .await
        .unwrap()
        .expect("peer commits after the long wait");
    assert_eq!(committed.node_id, node.id);
}

#[test]
fn verify_offer_rejects_foreign_node_id_coordinator_and_stage_count() {
    let dir = temp_model_dir(4, 100);
    let peer_addr = loopback(19510);
    let control = loopback(19500);
    let plan = plan_for(&[peer_addr, loopback(19511)], control);
    let node = plan
        .cluster
        .nodes
        .iter()
        .find(|n| n.address == peer_addr)
        .unwrap()
        .clone();
    let local = PeerLocalFacts {
        cluster_name: "assignment-test".to_string(),
        model: ModelIdentity::from_model_dir(dir.path()).unwrap(),
        model_dir: dir.path().to_path_buf(),
        listener_addr: peer_addr,
    };
    let good = sample_offer(&node.id, node.stage.unwrap(), dir.path(), &plan);
    verify_offer(&good, &local, control).expect("the planned offer verifies");

    // A node id that is not `stage-<i>` would name a file outside
    // `.mlxcel/` on the peer; it is refused even when the TOML agrees.
    let mut traversal = good.clone();
    let mut cluster = ClusterConfig::from_toml(&good.cluster_toml).unwrap();
    for n in &mut cluster.nodes {
        if n.id == node.id {
            n.id = "../../escape".to_string();
        }
    }
    traversal.node_id = "../../escape".to_string();
    traversal.cluster_toml = render_deterministic_toml(&cluster);
    let (reason, detail) = verify_offer(&traversal, &local, control).unwrap_err();
    assert_eq!(reason, AssignmentRejectReason::InvalidClusterConfig);
    assert!(detail.contains("escape"), "{detail}");

    // The TOML's coordinator node must be the authenticated coordinator,
    // not only the offer's `coordinator_control_addr` field.
    let mut foreign = good.clone();
    let mut cluster = ClusterConfig::from_toml(&good.cluster_toml).unwrap();
    for n in &mut cluster.nodes {
        if n.role != NodeRole::PipelineStage {
            n.address = loopback(19501);
        }
    }
    foreign.cluster_toml = render_deterministic_toml(&cluster);
    let (reason, detail) = verify_offer(&foreign, &local, control).unwrap_err();
    assert_eq!(reason, AssignmentRejectReason::AddressMismatch);
    assert!(detail.contains("19501"), "{detail}");

    // More stages than layers is refused before any per-stage work.
    let mut excess = good.clone();
    excess.num_stages = u32::MAX;
    let (reason, _) = verify_offer(&excess, &local, control).unwrap_err();
    assert_eq!(reason, AssignmentRejectReason::LayerRangeMismatch);
}

#[tokio::test]
async fn coordinator_retries_while_peer_pending_pool_is_full() {
    // Unauthenticated sockets holding the peer's pending-handshake slots make
    // the peer close the coordinator's socket at accept. The coordinator
    // retries until its deadline instead of failing the bring-up.
    let dir = temp_model_dir(4, 100);
    let auth = test_auth();
    let peers = [loopback(reserve_port()), loopback(reserve_port())];
    let plan = plan_for(&peers, loopback(reserve_port()));
    let control = plan.cluster.find_node("coordinator").unwrap().address;
    let handles: Vec<_> = peers
        .iter()
        .map(|addr| spawn_peer(peer_config(*addr, dir.path(), &auth)))
        .collect();
    wait_for_listener(peers[0]).await;
    wait_for_listener(peers[1]).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut idle = Vec::new();
    for _ in 0..MAX_PENDING_ASSIGNMENT_HANDSHAKES_PER_IP {
        idle.push(TcpStream::connect(peers[0]).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let release = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        drop(idle);
    });

    push_stage_assignments(&plan, control, dir.path(), &auth, Duration::from_secs(10))
        .await
        .expect("push succeeds once the pending slots free up");
    release.await.unwrap();
    for handle in handles {
        handle.await.unwrap().expect("peer commits");
    }
}

#[test]
fn commit_reserve_keeps_a_frame_window_inside_the_push_bound() {
    // Production: the offer phase ends one frame-I/O bound before the end.
    assert_eq!(
        commit_reserve(ASSIGNMENT_ACK_TIMEOUT),
        ASSIGNMENT_IO_TIMEOUT
    );
    assert_eq!(
        ASSIGNMENT_ACK_TIMEOUT - commit_reserve(ASSIGNMENT_ACK_TIMEOUT),
        Duration::from_secs(50)
    );
    // A short injected timeout keeps at most half for the commit phase.
    assert_eq!(
        commit_reserve(Duration::from_millis(300)),
        Duration::from_millis(150)
    );
    assert_eq!(commit_reserve(Duration::ZERO), Duration::ZERO);
}
