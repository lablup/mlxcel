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

use bytes::Bytes;
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;
use crate::distributed::pipeline::partial_loading::LayerFilter;
use crate::distributed::transport::MessageKind;
use crate::distributed::{TcpTransport, TcpTransportConfig};

/// Executor that never touches MLX: lifecycle commands need no model, and
/// every execution attempt is counted so a test can assert nothing ran.
struct StubExecutor {
    stage: StageAssignment,
    filter: LayerFilter,
    executions: Arc<AtomicUsize>,
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

    fn release_caches(&self, _caches: &[KVCache]) {}

    fn execute(
        &self,
        _input: StageExecutionInput<'_>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        bail!("stub executor does not run models")
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
    stage: Option<RemoteStageServiceHandle>,
    stage_addr: String,
    executions: Arc<AtomicUsize>,
}

impl Harness {
    /// `upstream_is_coordinator` configures the coordinator transport as the
    /// stage's upstream neighbour, so a rogue authenticated peer is neither
    /// the coordinator nor a neighbour.
    async fn spawn(stage_index: usize, upstream_is_coordinator: bool) -> Self {
        let auth = Arc::new(ClusterAuth::new(&ClusterSecret::random()));
        let coordinator = Arc::new(
            TcpTransport::bind(TcpTransportConfig {
                bind_address: "127.0.0.1:0".to_string(),
                auth: Some(Arc::clone(&auth)),
                ..Default::default()
            })
            .await
            .unwrap(),
        );
        let coordinator_addr = coordinator.local_addr().unwrap();
        let upstream = upstream_is_coordinator.then(|| coordinator_addr.clone());
        let executions = Arc::new(AtomicUsize::new(0));
        let assignment = stage_assignment(stage_index);
        let factory_assignment = assignment.clone();
        let factory_executions = Arc::clone(&executions);
        let stage = RemoteStageServiceHandle::spawn_with_executor(
            RemoteStageServiceConfig {
                model_dir: std::path::PathBuf::from("/nonexistent/stub-model"),
                bind_address: "127.0.0.1:0".to_string(),
                transport_backend: TransportBackend::Tcp,
                stage_assignment: assignment,
                num_stages: 2,
                coordinator_peer: coordinator_addr.clone(),
                upstream_peer: upstream,
                downstream_peer: None,
                auth: Arc::clone(&auth),
            },
            Box::new(move || {
                Ok(Box::new(StubExecutor {
                    filter: LayerFilter::from_stage(&factory_assignment),
                    stage: factory_assignment,
                    executions: factory_executions,
                }) as Box<dyn StageExecutor>)
            }),
        )
        .unwrap();
        let stage_addr = stage.local_addr().to_string();
        Self {
            auth,
            coordinator,
            coordinator_addr,
            stage: Some(stage),
            stage_addr,
            executions,
        }
    }

    async fn authenticated_peer(&self) -> Arc<TcpTransport> {
        Arc::new(
            TcpTransport::bind(TcpTransportConfig {
                bind_address: "127.0.0.1:0".to_string(),
                auth: Some(Arc::clone(&self.auth)),
                ..Default::default()
            })
            .await
            .unwrap(),
        )
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
            &self.stage_addr,
            TransportMessage::Control {
                operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                payload: Bytes::from(payload),
            },
        )
        .await
        .unwrap();
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

#[tokio::test]
async fn run_entry_with_bad_tensor_answers_error_and_keeps_serving() {
    let harness = Harness::spawn(0, false).await;
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
