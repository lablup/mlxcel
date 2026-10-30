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

//! Control-plane hardening of the remote stage service (#2637, finding 2).
//!
//! The service runs on loopback TCP around a stub executor (no model
//! weights). An authenticated coordinator transport, an authenticated but
//! unrelated transport, and a raw unauthenticated socket each try to drive
//! it; only the coordinator's commands may have an effect, and no single
//! malformed frame may end the service thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::bail;
use bytes::Bytes;
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;
use crate::distributed::pipeline::partial_loading::LayerFilter;
use crate::distributed::transport::MessageKind;
use crate::distributed::{TcpTransport, TcpTransportConfig};

/// What the stub executor answers an execution with.
#[derive(Clone, Copy)]
enum StubOutput {
    /// Fail every execution (the control-plane tests never run a model).
    Error,
    /// Return `[1, 1, 4]` logits, as a final stage would (#2694).
    Logits,
    /// Return `[1, 2, 4]` hidden states, as a middle stage would; the
    /// harness configures no downstream peer, so forwarding them fails
    /// after a successful execution (#2714 item b).
    HiddenStates,
}

/// Executor without model weights: lifecycle commands need no model, and
/// every execution attempt is counted so a test can assert nothing ran.
struct StubExecutor {
    stage: StageAssignment,
    filter: LayerFilter,
    executions: Arc<AtomicUsize>,
    /// Cache sets handed back through `release_caches`.
    releases: Arc<AtomicUsize>,
    output: StubOutput,
}

/// How the harness wires the stage's upstream neighbour.
#[derive(Clone, Copy)]
enum UpstreamKind {
    /// No upstream (an entry stage).
    None,
    /// The coordinator transport doubles as the upstream, so a rogue
    /// authenticated peer is neither the coordinator nor a neighbour.
    Coordinator,
    /// A separate authenticated transport, so a test can tell frames sent
    /// to the upstream from frames sent to the coordinator.
    Separate,
}

impl StageExecutor for StubExecutor {
    fn stage_assignment(&self) -> &StageAssignment {
        &self.stage
    }

    fn layer_filter(&self) -> &LayerFilter {
        &self.filter
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn release_caches(&self, _caches: &[KVCache]) {
        self.releases.fetch_add(1, Ordering::SeqCst);
    }

    fn execute(
        &self,
        _input: StageExecutionInput<'_>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        match self.output {
            StubOutput::Error => bail!("stub executor does not run models"),
            StubOutput::Logits => Ok(StageExecutionOutput::Logits(mlxcel_core::from_slice_f32(
                &[0.25; 4],
                &[1, 1, 4],
            ))),
            StubOutput::HiddenStates => Ok(StageExecutionOutput::HiddenStates(
                mlxcel_core::from_slice_f32(&[0.5; 8], &[1, 2, 4]),
            )),
        }
    }
}

fn stage_assignment(stage_index: usize) -> StageAssignment {
    StageAssignment {
        stage_index,
        device_id: format!("stub-{stage_index}"),
        layer_range: (stage_index * 4)..(stage_index * 4 + 4),
        has_embedding: stage_index == 0,
        has_lm_head: true,
        estimated_memory_bytes: 0,
    }
}

struct Harness {
    auth: Arc<ClusterAuth>,
    coordinator: Arc<TcpTransport>,
    coordinator_addr: String,
    /// The separate upstream transport of [`UpstreamKind::Separate`].
    upstream: Option<Arc<TcpTransport>>,
    stage: Option<RemoteStageServiceHandle>,
    /// Where IPv4 peers (the upstream, a rogue) send frames to the stage.
    stage_addr: String,
    /// Where the coordinator sends frames to the stage; differs from
    /// `stage_addr` only for [`CoordinatorKind::WildcardV6DualStack`].
    stage_addr_for_coordinator: String,
    executions: Arc<AtomicUsize>,
    releases: Arc<AtomicUsize>,
}

async fn bind_authenticated(auth: &Arc<ClusterAuth>) -> Arc<TcpTransport> {
    bind_authenticated_at(auth, "127.0.0.1:0").await
}

async fn bind_authenticated_at(auth: &Arc<ClusterAuth>, bind_address: &str) -> Arc<TcpTransport> {
    Arc::new(
        TcpTransport::bind(TcpTransportConfig {
            bind_address: bind_address.to_string(),
            auth: Some(Arc::clone(auth)),
            ..Default::default()
        })
        .await
        .unwrap(),
    )
}

/// How the stage's config names the coordinator.
#[derive(Clone, Copy)]
enum CoordinatorKind {
    /// The coordinator's own listener address, as `--pp-auto` localizes it.
    Concrete,
    /// `[::]:<port>`: an unspecified IP (a hand-written config, or the
    /// `0.0.0.0:<port>` a `--pp-auto` coordinator writes), so the matcher
    /// accepts any authenticated identity on that port. The coordinator
    /// listens on IPv4 loopback, and `[::]` rather than `0.0.0.0` makes a
    /// dial of the configured address reach `::1`, where it does not listen.
    WildcardV6,
    /// `[::]:<port>` with the coordinator on `[::1]` and the stage dual-stack
    /// on `[::]`, so a second holder of the secret can take the coordinator's
    /// port on IPv4 loopback and pass the matcher as the coordinator
    /// (#2714 item 2). The issue's shape (rogue on `[::1]`) cannot be built
    /// on one host: a `[::1]` identity dialing an IPv4 stage arrives from
    /// `127.0.0.1` and is refused at the handshake before the matcher runs,
    /// so the roles are swapped and the same path is exercised.
    WildcardV6DualStack,
}

fn port_of(addr: &str) -> &str {
    addr.rsplit_once(':').unwrap().1
}

impl Harness {
    /// `upstream_is_coordinator` configures the coordinator transport as the
    /// stage's upstream neighbour, so a rogue authenticated peer is neither
    /// the coordinator nor a neighbour.
    async fn spawn(stage_index: usize, upstream_is_coordinator: bool) -> Self {
        let upstream = if upstream_is_coordinator {
            UpstreamKind::Coordinator
        } else {
            UpstreamKind::None
        };
        Self::spawn_with(stage_index, upstream, StubOutput::Error).await
    }

    async fn spawn_with(stage_index: usize, upstream: UpstreamKind, output: StubOutput) -> Self {
        Self::spawn_configured(stage_index, upstream, output, CoordinatorKind::Concrete).await
    }

    async fn spawn_configured(
        stage_index: usize,
        upstream: UpstreamKind,
        output: StubOutput,
        coordinator_kind: CoordinatorKind,
    ) -> Self {
        Self::spawn_full(stage_index, upstream, output, coordinator_kind, None).await
    }

    /// An entry stage whose sequence cap is lowered to `max_sequences`.
    async fn spawn_capped(max_sequences: usize) -> Self {
        Self::spawn_full(
            0,
            UpstreamKind::None,
            StubOutput::Error,
            CoordinatorKind::Concrete,
            Some(max_sequences),
        )
        .await
    }

    async fn spawn_full(
        stage_index: usize,
        upstream: UpstreamKind,
        output: StubOutput,
        coordinator_kind: CoordinatorKind,
        sequence_cap: Option<usize>,
    ) -> Self {
        let auth = Arc::new(ClusterAuth::new(&ClusterSecret::random()));
        let (coordinator_bind, stage_bind) = match coordinator_kind {
            CoordinatorKind::Concrete | CoordinatorKind::WildcardV6 => {
                ("127.0.0.1:0", "127.0.0.1:0")
            }
            CoordinatorKind::WildcardV6DualStack => ("[::1]:0", "[::]:0"),
        };
        let coordinator = bind_authenticated_at(&auth, coordinator_bind).await;
        let coordinator_addr = coordinator.local_addr().unwrap();
        let coordinator_peer = match coordinator_kind {
            CoordinatorKind::Concrete => coordinator_addr.clone(),
            CoordinatorKind::WildcardV6 | CoordinatorKind::WildcardV6DualStack => {
                format!("[::]:{}", port_of(&coordinator_addr))
            }
        };
        let (upstream, upstream_peer) = match upstream {
            UpstreamKind::None => (None, None),
            UpstreamKind::Coordinator => (None, Some(coordinator_addr.clone())),
            UpstreamKind::Separate => {
                let upstream = bind_authenticated(&auth).await;
                let addr = upstream.local_addr().unwrap();
                (Some(upstream), Some(addr))
            }
        };
        let executions = Arc::new(AtomicUsize::new(0));
        let assignment = stage_assignment(stage_index);
        let factory_assignment = assignment.clone();
        let factory_executions = Arc::clone(&executions);
        let releases = Arc::new(AtomicUsize::new(0));
        let factory_releases = Arc::clone(&releases);
        let config = RemoteStageServiceConfig {
            model_dir: std::path::PathBuf::from("/nonexistent/stub-model"),
            bind_address: stage_bind.to_string(),
            transport_backend: TransportBackend::Tcp,
            stage_assignment: assignment,
            num_stages: 2,
            coordinator_peer,
            upstream_peer,
            downstream_peer: None,
            auth: Arc::clone(&auth),
        };
        let factory: ExecutorFactory = Box::new(move || {
            Ok(Box::new(StubExecutor {
                filter: LayerFilter::from_stage(&factory_assignment),
                stage: factory_assignment,
                executions: factory_executions,
                releases: factory_releases,
                output,
            }) as Box<dyn StageExecutor>)
        });
        let stage = match sequence_cap {
            None => RemoteStageServiceHandle::spawn_with_executor(config, factory),
            Some(cap) => {
                RemoteStageServiceHandle::spawn_with_executor_and_sequence_cap(config, factory, cap)
            }
        }
        .unwrap();
        let (stage_addr, stage_addr_for_coordinator) = match coordinator_kind {
            CoordinatorKind::Concrete | CoordinatorKind::WildcardV6 => {
                let addr = stage.local_addr().to_string();
                (addr.clone(), addr)
            }
            CoordinatorKind::WildcardV6DualStack => {
                let port = port_of(stage.local_addr()).to_string();
                (format!("127.0.0.1:{port}"), format!("[::1]:{port}"))
            }
        };
        Self {
            auth,
            coordinator,
            coordinator_addr,
            upstream,
            stage: Some(stage),
            stage_addr,
            stage_addr_for_coordinator,
            executions,
            releases,
        }
    }

    /// The stage address `from` reaches the stage at.
    fn stage_addr_for(&self, from: &TcpTransport) -> &str {
        if std::ptr::eq(from, &*self.coordinator) {
            &self.stage_addr_for_coordinator
        } else {
            &self.stage_addr
        }
    }

    async fn authenticated_peer(&self) -> Arc<TcpTransport> {
        bind_authenticated(&self.auth).await
    }

    /// Send a forward activation for `pp-seq-1`, micro-batch 2, from `from`
    /// with a valid `[1, 2, 4]` hidden-state tensor.
    async fn send_forward_activation(&self, from: &TcpTransport, reply_nonce: Option<u64>) {
        let hidden =
            serialize_mlx_array(&mlxcel_core::from_slice_f32(&[0.0; 8], &[1, 2, 4])).unwrap();
        let activation = ActivationMessage::forward(
            RequestId::from_string("pp-seq-1".to_string()).unwrap(),
            2,
            0,
            2,
            hidden,
            None,
            None,
            2,
        )
        .with_reply_nonce(reply_nonce);
        from.send(
            &self.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                payload: Bytes::from(serde_json::to_vec(&activation).unwrap()),
            },
        )
        .await
        .unwrap();
    }

    /// Nothing may reach `on` within 300 ms.
    async fn expect_silence(on: &TcpTransport, what: &str) {
        let got = tokio::time::timeout(Duration::from_millis(300), on.recv()).await;
        assert!(got.is_err(), "{what}: {got:?}");
    }

    async fn send_command(
        &self,
        from: &TcpTransport,
        nonce: u64,
        reply_addr: &str,
        command: RemoteStageCommand,
    ) {
        let payload = serde_json::to_vec(&RemoteStageRequest {
            reply_addr: reply_addr.to_string(),
            nonce,
            command,
        })
        .unwrap();
        from.send(
            self.stage_addr_for(from),
            TransportMessage::Control {
                operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                payload: Bytes::from(payload),
            },
        )
        .await
        .unwrap();
    }

    /// The coordinator prepares `request_id` and gets its `Ack`, as every
    /// `RunEntry` is preceded by a `PrepareSequence` broadcast.
    async fn prepare(&self, nonce: u64, request_id: &str) {
        self.send_command(
            &self.coordinator,
            nonce,
            &self.coordinator_addr,
            RemoteStageCommand::PrepareSequence {
                request_id: request_id.to_string(),
            },
        )
        .await;
        let ack = self.recv_response(&self.coordinator).await;
        assert!(
            matches!(ack, RemoteStageResponse::Ack { nonce: got, .. } if got == nonce),
            "expected Ack for PrepareSequence, got {ack:?}"
        );
    }

    /// A second holder of the secret listening on the coordinator's port on
    /// IPv4 loopback, which a coordinator matcher configured as `[::]:<port>`
    /// accepts as the coordinator ([`CoordinatorKind::WildcardV6DualStack`]).
    async fn rogue_on_coordinator_port(&self) -> Arc<TcpTransport> {
        let port = port_of(&self.coordinator_addr);
        bind_authenticated_at(&self.auth, &format!("127.0.0.1:{port}")).await
    }

    /// Probe through the coordinator and return the stage's state.
    async fn probe(&self, nonce: u64) -> RemoteStageResponse {
        self.send_command(
            &self.coordinator,
            nonce,
            &self.coordinator_addr,
            RemoteStageCommand::ProbeState,
        )
        .await;
        self.recv_response(&self.coordinator).await
    }

    async fn recv_response(&self, on: &TcpTransport) -> RemoteStageResponse {
        let (_from, msg) = tokio::time::timeout(Duration::from_secs(5), on.recv())
            .await
            .expect("stage must answer within 5s (it may have died)")
            .unwrap();
        let TransportMessage::Control { operation, payload } = msg else {
            panic!("unexpected frame");
        };
        assert_eq!(operation, PIPELINE_STAGE_RESPONSE_OPERATION);
        serde_json::from_slice(&payload).unwrap()
    }

    fn finish(mut self) {
        self.stage.take().unwrap().shutdown().unwrap();
    }
}

fn in_flight(response: &RemoteStageResponse) -> usize {
    match response {
        RemoteStageResponse::State { state, .. } | RemoteStageResponse::Ack { state, .. } => {
            state.in_flight_requests
        }
        other => panic!("expected State/Ack, got {other:?}"),
    }
}

fn is_shutdown(response: &RemoteStageResponse) -> bool {
    match response {
        RemoteStageResponse::State { state, .. } => state.shutdown,
        other => panic!("expected State, got {other:?}"),
    }
}

/// Write one transport frame on a raw socket, skipping the handshake.
async fn raw_control_frame(addr: &str, operation: &str, payload: &[u8]) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut body = Vec::new();
    body.extend_from_slice(&(operation.len() as u32).to_be_bytes());
    body.extend_from_slice(operation.as_bytes());
    body.extend_from_slice(payload);
    let mut frame = vec![MessageKind::Control as u8];
    frame.extend_from_slice(&(body.len() as u64).to_be_bytes());
    frame.extend_from_slice(&body);
    // The peer may drop the socket mid-write once the handshake fails;
    // either outcome is fine for an attacker's frame.
    let _ = stream.write_all(&frame).await;
    let _ = stream.flush().await;
    tokio::time::sleep(Duration::from_millis(200)).await;
}

#[tokio::test]
async fn unauthenticated_control_frame_is_rejected_before_dispatch() {
    let harness = Harness::spawn(0, false).await;
    let shutdown = serde_json::to_vec(&RemoteStageRequest {
        reply_addr: "127.0.0.1:9".to_string(),
        nonce: 1,
        command: RemoteStageCommand::Shutdown,
    })
    .unwrap();
    raw_control_frame(
        &harness.stage_addr,
        PIPELINE_STAGE_COMMAND_OPERATION,
        &shutdown,
    )
    .await;

    let state = harness.probe(7).await;
    assert!(!is_shutdown(&state), "unauthenticated Shutdown took effect");
    assert_eq!(state.nonce(), 7);
    harness.finish();
}

#[tokio::test]
async fn malformed_json_frame_does_not_end_service() {
    let harness = Harness::spawn(0, false).await;
    harness
        .coordinator
        .send(
            &harness.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                payload: Bytes::from_static(b"{this is not json"),
            },
        )
        .await
        .unwrap();
    harness
        .coordinator
        .send(
            &harness.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                payload: Bytes::from_static(b"\xff\xfe"),
            },
        )
        .await
        .unwrap();

    let state = harness.probe(11).await;
    assert!(!is_shutdown(&state));
    assert_eq!(state.nonce(), 11);
    harness.finish();
}

#[tokio::test]
async fn commands_from_authenticated_non_coordinator_have_no_effect() {
    let harness = Harness::spawn(0, false).await;
    harness
        .send_command(
            &harness.coordinator,
            1,
            &harness.coordinator_addr,
            RemoteStageCommand::PrepareSequence {
                request_id: "pp-seq-1".to_string(),
            },
        )
        .await;
    let ack = harness.recv_response(&harness.coordinator).await;
    assert_eq!(in_flight(&ack), 1);

    let rogue = harness.authenticated_peer().await;
    let rogue_addr = rogue.local_addr().unwrap();
    for command in [
        RemoteStageCommand::CancelRequest {
            request_id: "pp-seq-1".to_string(),
        },
        RemoteStageCommand::BeginDrain,
        RemoteStageCommand::RunEntry {
            request_id: "pp-seq-1".to_string(),
            micro_batch_id: 0,
            input_ids: vec![1, 2, 3],
            attention_mask: None,
        },
        RemoteStageCommand::Shutdown,
    ] {
        harness.send_command(&rogue, 99, &rogue_addr, command).await;
    }
    // The rogue peer must get no reply at all.
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rogue.recv())
            .await
            .is_err(),
        "rogue peer received a reply"
    );

    let state = harness.probe(2).await;
    assert_eq!(in_flight(&state), 1, "rogue CancelRequest took effect");
    assert!(!is_shutdown(&state), "rogue Shutdown took effect");
    match &state {
        RemoteStageResponse::State { state, .. } => {
            assert!(!state.draining, "rogue BeginDrain took effect");
        }
        _ => unreachable!(),
    }
    assert_eq!(
        harness.executions.load(Ordering::SeqCst),
        0,
        "rogue RunEntry reached the executor"
    );
    harness.finish();
}

#[tokio::test]
async fn replies_go_to_authenticated_source_not_reply_addr() {
    let harness = Harness::spawn(0, false).await;
    let bystander = harness.authenticated_peer().await;
    let bystander_addr = bystander.local_addr().unwrap();
    harness
        .send_command(
            &harness.coordinator,
            5,
            &bystander_addr,
            RemoteStageCommand::ProbeState,
        )
        .await;
    let state = harness.recv_response(&harness.coordinator).await;
    assert_eq!(state.nonce(), 5);
    assert!(
        tokio::time::timeout(Duration::from_millis(300), bystander.recv())
            .await
            .is_err(),
        "reply was routed to the payload's reply_addr"
    );
    harness.finish();
}

#[tokio::test]
async fn activation_from_non_neighbour_is_ignored() {
    // Stage 1 whose configured upstream is the coordinator transport, not
    // the rogue peer.
    let harness = Harness::spawn(1, true).await;
    let rogue = harness.authenticated_peer().await;
    let activation = ActivationMessage::forward(
        RequestId::from_string("pp-seq-1".to_string()).unwrap(),
        0,
        0,
        2,
        vec![0u8; 8],
        None,
        None,
        1,
    );
    rogue
        .send(
            &harness.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                payload: Bytes::from(serde_json::to_vec(&activation).unwrap()),
            },
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let state = harness.probe(3).await;
    assert_eq!(in_flight(&state), 0);
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);
    harness.finish();
}

/// The final stage (index 1 of 2) answers a forward activation that carries
/// the coordinator's `RunEntry` nonce with a `RunResult` sent straight to
/// the coordinator; nothing travels back to the upstream stage (#2694).
#[tokio::test]
async fn final_stage_sends_run_result_to_coordinator_not_upstream() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Logits).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(7_777)).await;

    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::RunResult {
            stage_index,
            nonce,
            request_id,
            micro_batch_id,
            logits,
        } => {
            assert_eq!(stage_index, 1);
            assert_eq!(nonce, 7_777, "the RunResult must echo the RunEntry nonce");
            assert_eq!(request_id, "pp-seq-1");
            assert_eq!(micro_batch_id, 2);
            let logits = super::super::wire_tensor::deserialize_wire_tensor(&logits).unwrap();
            assert_eq!(
                mlxcel_core::array_shape(logits.as_ref().unwrap()),
                vec![1, 1, 4]
            );
        }
        other => panic!("expected RunResult at the coordinator, got {other:?}"),
    }
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(harness.executions.load(Ordering::SeqCst), 1);
    harness.finish();
}

/// A coordinator configured with an unspecified IP (`--pp-auto` on every
/// interface writes `0.0.0.0:19000`) names no host to dial: the final stage
/// must send its `RunResult` to the authenticated source of the
/// coordinator's commands, not to the configured address. `[::]` keeps the
/// test meaningful on one host: dialing it reaches `::1`, where the IPv4
/// coordinator does not listen, while `0.0.0.0` would reach `127.0.0.1`.
#[tokio::test]
async fn final_stage_sends_run_result_to_authenticated_coordinator_identity() {
    let harness = Harness::spawn_configured(
        1,
        UpstreamKind::Separate,
        StubOutput::Logits,
        CoordinatorKind::WildcardV6,
    )
    .await;
    // Every RunEntry is preceded by a PrepareSequence broadcast.
    harness.prepare(41, "pp-seq-1").await;

    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(4_242)).await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::RunResult {
            stage_index, nonce, ..
        } => {
            assert_eq!(stage_index, 1);
            assert_eq!(nonce, 4_242);
        }
        other => panic!("expected RunResult at the coordinator, got {other:?}"),
    }
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    harness.finish();
}

/// A forward activation that reaches the final stage without a reply nonce
/// cannot be correlated with any exchange: it is executed and dropped, and
/// neither the coordinator nor the upstream stage receives anything.
#[tokio::test]
async fn final_stage_drops_result_of_activation_without_reply_nonce() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Logits).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, None).await;

    Harness::expect_silence(&harness.coordinator, "the coordinator received a frame").await;
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(harness.executions.load(Ordering::SeqCst), 1);
    let state = harness.probe(31).await;
    assert!(!is_shutdown(&state), "the stage must keep serving");
    harness.finish();
}

#[tokio::test]
async fn run_entry_with_bad_tensor_answers_error_and_keeps_serving() {
    let harness = Harness::spawn(0, false).await;
    harness.prepare(20, "pp-seq-1").await;
    harness
        .send_command(
            &harness.coordinator,
            21,
            &harness.coordinator_addr,
            RemoteStageCommand::RunEntry {
                request_id: "pp-seq-1".to_string(),
                micro_batch_id: 0,
                input_ids: vec![0xde, 0xad],
                attention_mask: None,
            },
        )
        .await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error { nonce, message, .. } => {
            assert_eq!(nonce, 21);
            assert!(message.contains("invalid entry tensors"), "{message}");
        }
        other => panic!("expected Error, got {other:?}"),
    }
    let state = harness.probe(22).await;
    assert!(!is_shutdown(&state));
    harness.finish();
}

/// Issue #2716: entry tensors are decoded under the stage's wire limits, so
/// a well-formed but out-of-bounds `input_ids` (batch 2) is refused on its
/// header and never reaches the executor.
#[tokio::test]
async fn run_entry_with_out_of_bounds_ids_answers_limit_error() {
    let harness = Harness::spawn(0, false).await;
    harness.prepare(40, "pp-seq-1").await;
    let ids = serialize_mlx_array(&mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[2, 2])).unwrap();
    harness
        .send_command(
            &harness.coordinator,
            41,
            &harness.coordinator_addr,
            RemoteStageCommand::RunEntry {
                request_id: "pp-seq-1".to_string(),
                micro_batch_id: 0,
                input_ids: ids,
                attention_mask: None,
            },
        )
        .await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error { nonce, message, .. } => {
            assert_eq!(nonce, 41);
            assert!(
                message.contains("axis 0 is 2, expected exactly 1"),
                "{message}"
            );
        }
        other => panic!("expected Error, got {other:?}"),
    }
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);
    harness.finish();
}

/// Issue #2716: an upstream activation with a batch axis other than 1 is
/// refused before the executor runs, and the stage keeps serving. The
/// refusal is reported to the sequence's coordinator as the `Error` of the
/// `RunEntry` exchange whose nonce the activation carried (#2714 item b).
#[tokio::test]
async fn activation_outside_stage_limits_never_executes() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Logits).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    let hidden = serialize_mlx_array(&mlxcel_core::from_slice_f32(&[0.0; 16], &[2, 2, 4])).unwrap();
    let activation = ActivationMessage::forward(
        RequestId::from_string("pp-seq-1".to_string()).unwrap(),
        2,
        0,
        2,
        hidden,
        None,
        None,
        2,
    )
    .with_reply_nonce(Some(42));
    upstream
        .send(
            &harness.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                payload: Bytes::from(serde_json::to_vec(&activation).unwrap()),
            },
        )
        .await
        .unwrap();

    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error {
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(nonce, 42, "the Error must echo the RunEntry nonce");
            assert_eq!(request_id.as_deref(), Some("pp-seq-1"));
            assert!(
                message.contains("axis 0 is 2, expected exactly 1"),
                "{message}"
            );
        }
        other => panic!("expected Error at the coordinator, got {other:?}"),
    }
    let state = harness.probe(43).await;
    assert!(!is_shutdown(&state), "the stage must keep serving");
    assert_eq!(in_flight(&state), 0, "the refused request must be finished");
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);
    harness.finish();
}

/// An execution failure on the final stage (index 1 of 2) is reported to the
/// coordinator as the `Error` of the `RunEntry` exchange whose nonce the
/// activation carried, so `forward_sequence` fails at once instead of
/// waiting out `stage_timeout`; nothing travels back to the upstream stage
/// (#2712 item 1).
#[tokio::test]
async fn final_stage_execution_error_goes_to_coordinator() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Error).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(9_001)).await;

    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error {
            stage_index,
            nonce,
            request_id,
            message,
            state,
        } => {
            assert_eq!(stage_index, 1);
            assert_eq!(nonce, 9_001, "the Error must echo the RunEntry nonce");
            assert_eq!(request_id.as_deref(), Some("pp-seq-1"));
            assert!(message.contains("stub executor"), "{message}");
            assert!(state.is_some(), "the Error carries the stage snapshot");
        }
        other => panic!("expected Error at the coordinator, got {other:?}"),
    }
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(harness.executions.load(Ordering::SeqCst), 1);
    let state = harness.probe(9_002).await;
    assert_eq!(in_flight(&state), 0, "the failed request must be finished");
    harness.finish();
}

/// Without a reply nonce the failure cannot be matched to any exchange: it
/// is logged and dropped, and neither the coordinator nor the upstream stage
/// receives anything (the upstream `nonce: 0` Error of earlier builds was
/// never consumed).
#[tokio::test]
async fn final_stage_drops_execution_error_without_reply_nonce() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Error).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, None).await;

    Harness::expect_silence(&harness.coordinator, "the coordinator received a frame").await;
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(harness.executions.load(Ordering::SeqCst), 1);
    let state = harness.probe(32).await;
    assert!(!is_shutdown(&state), "the stage must keep serving");
    assert_eq!(in_flight(&state), 0);
    harness.finish();
}

// ---------------------------------------------------------------------------
// Per-sequence reply targets (#2714 item 2)
// ---------------------------------------------------------------------------

/// With the coordinator configured under an unspecified IP, a second holder
/// of the secret on the same port passes the coordinator matcher. It may
/// probe, but it cannot claim a sequence another identity prepared, and the
/// final stage's `RunResult` goes to the identity that prepared the
/// sequence, not to the source of the latest command.
#[tokio::test]
async fn second_authenticated_peer_cannot_redirect_run_result() {
    let harness = Harness::spawn_configured(
        1,
        UpstreamKind::Separate,
        StubOutput::Logits,
        CoordinatorKind::WildcardV6DualStack,
    )
    .await;
    harness.prepare(41, "pp-seq-1").await;

    let rogue = harness.rogue_on_coordinator_port().await;
    let rogue_addr = rogue.local_addr().unwrap();
    harness
        .send_command(&rogue, 90, &rogue_addr, RemoteStageCommand::ProbeState)
        .await;
    let state = harness.recv_response(&rogue).await;
    assert!(
        matches!(state, RemoteStageResponse::State { nonce: 90, .. }),
        "the rogue passes the coordinator matcher and may probe: {state:?}"
    );
    harness
        .send_command(
            &rogue,
            91,
            &rogue_addr,
            RemoteStageCommand::PrepareSequence {
                request_id: "pp-seq-1".to_string(),
            },
        )
        .await;
    match harness.recv_response(&rogue).await {
        RemoteStageResponse::Error {
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(nonce, 91);
            assert_eq!(request_id.as_deref(), Some("pp-seq-1"));
            assert!(
                message.contains("prepared by another coordinator identity"),
                "{message}"
            );
        }
        other => panic!("the rogue re-prepared an owned sequence: {other:?}"),
    }

    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(4_242)).await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::RunResult { nonce, .. } => assert_eq!(nonce, 4_242),
        other => panic!("expected RunResult at the coordinator, got {other:?}"),
    }
    Harness::expect_silence(&rogue, "the rogue received a frame").await;
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    harness.finish();
}

/// `CancelRequest` and `ReleaseSequence` for a sequence another identity
/// prepared are refused with an `Error` and change nothing; the owner's
/// `ReleaseSequence` still works.
#[tokio::test]
async fn non_owner_cannot_cancel_or_release_a_prepared_sequence() {
    let harness = Harness::spawn_configured(
        0,
        UpstreamKind::None,
        StubOutput::Error,
        CoordinatorKind::WildcardV6DualStack,
    )
    .await;
    harness.prepare(1, "pp-seq-1").await;

    let rogue = harness.rogue_on_coordinator_port().await;
    let rogue_addr = rogue.local_addr().unwrap();
    for (nonce, command) in [
        (
            61,
            RemoteStageCommand::CancelRequest {
                request_id: "pp-seq-1".to_string(),
            },
        ),
        (
            62,
            RemoteStageCommand::ReleaseSequence {
                request_id: "pp-seq-1".to_string(),
            },
        ),
        (
            63,
            RemoteStageCommand::RunEntry {
                request_id: "pp-seq-1".to_string(),
                micro_batch_id: 0,
                input_ids: vec![1, 2, 3],
                attention_mask: None,
            },
        ),
    ] {
        harness
            .send_command(&rogue, nonce, &rogue_addr, command)
            .await;
        match harness.recv_response(&rogue).await {
            RemoteStageResponse::Error {
                nonce: got,
                message,
                ..
            } => {
                assert_eq!(got, nonce);
                assert!(
                    message.contains("prepared by another coordinator identity"),
                    "{message}"
                );
            }
            other => panic!("non-owner command {nonce} was accepted: {other:?}"),
        }
    }
    let state = harness.probe(2).await;
    assert_eq!(in_flight(&state), 1, "a non-owner command took effect");
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);

    harness
        .send_command(
            &harness.coordinator,
            3,
            &harness.coordinator_addr,
            RemoteStageCommand::ReleaseSequence {
                request_id: "pp-seq-1".to_string(),
            },
        )
        .await;
    let ack = harness.recv_response(&harness.coordinator).await;
    assert!(
        matches!(ack, RemoteStageResponse::Ack { nonce: 3, .. }),
        "{ack:?}"
    );
    assert_eq!(in_flight(&ack), 0);
    harness.finish();
}

/// A final stage that gets an activation for a sequence nobody prepared has
/// no owner to answer: it never executes it (#2726), sends nothing (not even
/// to the configured coordinator address) and keeps serving.
#[tokio::test]
async fn final_stage_drops_result_of_unprepared_sequence() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::Logits).await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(5_555)).await;

    Harness::expect_silence(&harness.coordinator, "the coordinator received a frame").await;
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(
        harness.executions.load(Ordering::SeqCst),
        0,
        "an activation for an unprepared sequence must not execute"
    );
    assert_eq!(harness.releases.load(Ordering::SeqCst), 0);
    let state = harness.probe(56).await;
    assert!(!is_shutdown(&state), "the stage must keep serving");
    assert_eq!(
        in_flight(&state),
        0,
        "the dropped activation must leave no request behind"
    );
    harness.finish();
}

/// Rewritten for #2726 (was `first_prepare_discards_state_left_by_unowned_commands`):
/// a `RunEntry` for a not-yet-prepared id, here from a second holder of the
/// secret that passes the coordinator matcher, is refused before it decodes
/// or executes anything, so no state exists for the first `PrepareSequence`
/// to discard and a rogue cannot pre-seed the KV state of a request id the
/// coordinator will use next. The owner's prepare and re-prepare release
/// nothing.
#[tokio::test]
async fn unowned_run_entry_cannot_seed_state_before_first_prepare() {
    let harness = Harness::spawn_configured(
        0,
        UpstreamKind::None,
        StubOutput::Logits,
        CoordinatorKind::WildcardV6DualStack,
    )
    .await;
    let rogue = harness.rogue_on_coordinator_port().await;
    let rogue_addr = rogue.local_addr().unwrap();
    let input_ids = serialize_mlx_array(&mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3])).unwrap();
    harness
        .send_command(
            &rogue,
            71,
            &rogue_addr,
            RemoteStageCommand::RunEntry {
                request_id: "pp-seq-1".to_string(),
                micro_batch_id: 0,
                input_ids,
                attention_mask: None,
            },
        )
        .await;
    match harness.recv_response(&rogue).await {
        RemoteStageResponse::Error {
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(nonce, 71);
            assert_eq!(request_id.as_deref(), Some("pp-seq-1"));
            assert!(
                message.contains("sequence pp-seq-1 was not prepared on this stage"),
                "{message}"
            );
        }
        other => panic!("an unowned RunEntry must be refused: {other:?}"),
    }
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);

    harness.prepare(72, "pp-seq-1").await;
    assert_eq!(
        harness.releases.load(Ordering::SeqCst),
        0,
        "the refused RunEntry must have left no caches for the first PrepareSequence to discard"
    );
    let state = harness.probe(73).await;
    assert_eq!(in_flight(&state), 1);

    // A re-prepare by the owner keeps its own state.
    harness.prepare(74, "pp-seq-1").await;
    assert_eq!(harness.releases.load(Ordering::SeqCst), 0);
    harness.finish();
}

/// #2726 item 1: a `RunEntry` for an id no coordinator prepared on this
/// stage is answered with an `Error` of that exchange before its tensors are
/// decoded; the executor never runs and no request or cache is created.
#[tokio::test]
async fn unprepared_run_entry_is_refused_and_creates_no_state() {
    let harness = Harness::spawn_with(0, UpstreamKind::None, StubOutput::Logits).await;
    let input_ids = serialize_mlx_array(&mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3])).unwrap();
    harness
        .send_command(
            &harness.coordinator,
            81,
            &harness.coordinator_addr,
            RemoteStageCommand::RunEntry {
                request_id: "pp-seq-9".to_string(),
                micro_batch_id: 0,
                input_ids,
                attention_mask: None,
            },
        )
        .await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error {
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(nonce, 81);
            assert_eq!(request_id.as_deref(), Some("pp-seq-9"));
            assert!(
                message.contains("sequence pp-seq-9 was not prepared on this stage"),
                "{message}"
            );
        }
        other => panic!("an unprepared RunEntry must be refused: {other:?}"),
    }
    assert_eq!(
        harness.executions.load(Ordering::SeqCst),
        0,
        "the executor must not run for an unprepared sequence"
    );
    let state = harness.probe(82).await;
    assert_eq!(in_flight(&state), 0, "the refusal must create no request");

    // No cache set was built either: the first PrepareSequence has nothing
    // to discard.
    harness.prepare(83, "pp-seq-9").await;
    assert_eq!(harness.releases.load(Ordering::SeqCst), 0);
    harness.finish();
}

/// #2726 item 1: with the cap lowered to 2, a `PrepareSequence` for a third
/// id is refused and changes nothing, the owner's re-prepare of an id it
/// holds is still accepted at the cap, and a slot freed by a release can be
/// prepared again.
#[tokio::test]
async fn prepare_beyond_the_sequence_cap_is_refused() {
    let harness = Harness::spawn_capped(2).await;
    harness.prepare(1, "pp-seq-1").await;
    harness.prepare(2, "pp-seq-2").await;

    harness
        .send_command(
            &harness.coordinator,
            3,
            &harness.coordinator_addr,
            RemoteStageCommand::PrepareSequence {
                request_id: "pp-seq-3".to_string(),
            },
        )
        .await;
    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error {
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(nonce, 3);
            assert_eq!(request_id.as_deref(), Some("pp-seq-3"));
            assert!(
                message.contains("pipeline stage already holds 2 prepared sequences"),
                "{message}"
            );
        }
        other => panic!("a prepare beyond the cap must be refused: {other:?}"),
    }
    assert_eq!(
        in_flight(&harness.probe(4).await),
        2,
        "the refusal changed state"
    );

    // The owner's re-prepare at the cap is not a new entry.
    harness.prepare(5, "pp-seq-1").await;
    assert_eq!(in_flight(&harness.probe(6).await), 2);

    // A released slot is free again.
    harness
        .send_command(
            &harness.coordinator,
            7,
            &harness.coordinator_addr,
            RemoteStageCommand::ReleaseSequence {
                request_id: "pp-seq-2".to_string(),
            },
        )
        .await;
    let ack = harness.recv_response(&harness.coordinator).await;
    assert!(
        matches!(ack, RemoteStageResponse::Ack { nonce: 7, .. }),
        "{ack:?}"
    );
    harness.prepare(8, "pp-seq-3").await;
    assert_eq!(in_flight(&harness.probe(9).await), 2);
    assert_eq!(harness.executions.load(Ordering::SeqCst), 0);
    harness.finish();
}

// ---------------------------------------------------------------------------
// Every activation failure exit reports to the coordinator (#2714 item b)
// ---------------------------------------------------------------------------

/// A failure after a successful execution (here: hidden states to forward
/// but no downstream peer) is reported to the coordinator as the `Error` of
/// the activation's nonce, the request is finished, and the stage keeps
/// serving.
#[tokio::test]
async fn post_execution_failure_reports_error_to_coordinator() {
    let harness = Harness::spawn_with(1, UpstreamKind::Separate, StubOutput::HiddenStates).await;
    harness.prepare(1, "pp-seq-1").await;
    let upstream = harness.upstream.as_ref().unwrap();
    harness.send_forward_activation(upstream, Some(8_080)).await;

    match harness.recv_response(&harness.coordinator).await {
        RemoteStageResponse::Error {
            stage_index,
            nonce,
            request_id,
            message,
            ..
        } => {
            assert_eq!(stage_index, 1);
            assert_eq!(nonce, 8_080, "the Error must echo the RunEntry nonce");
            assert_eq!(request_id.as_deref(), Some("pp-seq-1"));
            assert!(message.contains("no downstream peer"), "{message}");
        }
        other => panic!("expected Error at the coordinator, got {other:?}"),
    }
    Harness::expect_silence(upstream, "the upstream stage received a frame").await;
    assert_eq!(harness.executions.load(Ordering::SeqCst), 1);
    let state = harness.probe(8_081).await;
    assert!(!is_shutdown(&state), "the stage must keep serving");
    assert_eq!(in_flight(&state), 0, "the failed request must be finished");
    harness.finish();
}
