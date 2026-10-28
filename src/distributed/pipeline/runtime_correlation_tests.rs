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

//! Reply correlation in `RemotePipelineRuntime::exchange` (#2637, finding 1).
//!
//! A mock transport stands in for TCP so a second, "attacker" node can
//! inject frames that would never pass the transport handshake on the wire.
//! Each test drives the real `forward_sequence` / `probe_stages` paths and
//! checks that only the awaited reply (right source, right nonce, right
//! request) becomes output.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mlxcel_core::cache::SequenceId;

use super::*;
use crate::distributed::cluster_auth::ClusterSecret;
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::distributed::pipeline::activation_transfer::{ActivationMessage, StageLifecycleState};
use crate::distributed::tensor_protocol::TensorDtype;

const COORDINATOR: &str = "10.0.0.1:1";
const STAGE: &str = "10.1.1.1:1";
/// The final stage of the two-stage fixtures (#2694): it sends the
/// `RunResult` straight to the coordinator.
const LAST_STAGE: &str = "10.1.1.2:1";
const ATTACKER: &str = "10.9.9.9:1";
const VOCAB: usize = 4;
const LEGIT_FILL: f32 = 1.0;
const FORGED_FILL: f32 = 9.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Attacker sends a RunResult with the right nonce and request id first.
    ForgedSource,
    /// The stage itself answers with a wrong nonce first.
    WrongNonce,
    /// The stage answers for a different request id first.
    WrongRequestId,
    /// The stage answers with logits of the wrong vocab dimension.
    WrongShape,
    /// The stage and the attacker flood frames and never answer.
    Flood,
    /// Attacker forges a draining State for ProbeState first.
    ForgedState,
    /// Two stage peers: the last stage answers the entry stage's RunEntry
    /// with the RunResult (the production flow since #2694).
    ResultFromLastStage,
    /// Two stage peers: the entry stage itself sends the RunResult, which a
    /// multi-stage coordinator must not accept.
    ResultFromEntryStage,
}

fn logits_bytes(fill: f32, seq_len: u64, vocab: u64) -> Vec<u8> {
    let data: Vec<u8> = std::iter::repeat_n(fill, (seq_len * vocab) as usize)
        .flat_map(f32::to_le_bytes)
        .collect();
    ActivationMessage::serialize_activation(TensorDtype::Float32, &[1, seq_len, vocab], &data)
        .unwrap()
}

async fn send_response(transport: &MockTransport, to: &str, response: RemoteStageResponse) {
    let payload = serde_json::to_vec(&response).unwrap();
    transport
        .send(
            to,
            TransportMessage::Control {
                operation: PIPELINE_STAGE_RESPONSE_OPERATION.to_string(),
                payload: Bytes::from(payload),
            },
        )
        .await
        .unwrap();
}

fn run_result(
    nonce: u64,
    request_id: &str,
    fill: f32,
    seq_len: u64,
    vocab: u64,
) -> RemoteStageResponse {
    RemoteStageResponse::RunResult {
        stage_index: 0,
        nonce,
        request_id: request_id.to_string(),
        micro_batch_id: 0,
        logits: logits_bytes(fill, seq_len, vocab),
    }
}

fn state_response(nonce: u64, draining: bool) -> RemoteStageResponse {
    let lifecycle = StageLifecycleState {
        health: crate::distributed::StageHealth::Healthy,
        draining,
        ..Default::default()
    };
    RemoteStageResponse::State {
        stage_index: 0,
        nonce,
        state: lifecycle.snapshot(),
    }
}

/// A stage that only acknowledges lifecycle commands (the last stage of the
/// two-peer fixtures; its RunResult is sent by the scripted entry stage).
async fn run_ack_only_stage(stage: Arc<MockTransport>) {
    loop {
        let Ok((from, msg)) = stage.recv().await else {
            return;
        };
        let TransportMessage::Control { operation, payload } = msg else {
            continue;
        };
        if operation != PIPELINE_STAGE_COMMAND_OPERATION {
            continue;
        }
        let request: RemoteStageRequest = serde_json::from_slice(&payload).unwrap();
        let response = match request.command {
            RemoteStageCommand::ProbeState => state_response(request.nonce, false),
            RemoteStageCommand::RunEntry { .. } => continue,
            _ => RemoteStageResponse::Ack {
                stage_index: 1,
                nonce: request.nonce,
                state: StageLifecycleState::default().snapshot(),
            },
        };
        send_response(&stage, &from, response).await;
    }
}

/// The scripted stage: acknowledges lifecycle commands, and answers
/// `RunEntry` / `ProbeState` according to `mode`. `attacker` is the attacker
/// transport, or the last stage in the two-peer modes.
async fn run_scripted_stage(stage: Arc<MockTransport>, attacker: Arc<MockTransport>, mode: Mode) {
    loop {
        let Ok((from, msg)) = stage.recv().await else {
            return;
        };
        let TransportMessage::Control { operation, payload } = msg else {
            continue;
        };
        if operation != PIPELINE_STAGE_COMMAND_OPERATION {
            continue;
        }
        let request: RemoteStageRequest = serde_json::from_slice(&payload).unwrap();
        let nonce = request.nonce;
        let ack = RemoteStageResponse::Ack {
            stage_index: 0,
            nonce,
            state: StageLifecycleState::default().snapshot(),
        };
        match request.command {
            RemoteStageCommand::RunEntry {
                request_id,
                input_ids,
                ..
            } => {
                let seq_len = ActivationMessage::deserialize_activation(&input_ids)
                    .map(|t| t.shape[1])
                    .unwrap_or(1);
                let vocab = VOCAB as u64;
                match mode {
                    Mode::ForgedSource => {
                        send_response(
                            &attacker,
                            &from,
                            run_result(nonce, &request_id, FORGED_FILL, seq_len, vocab),
                        )
                        .await;
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, &request_id, LEGIT_FILL, seq_len, vocab),
                        )
                        .await;
                    }
                    Mode::WrongNonce => {
                        send_response(
                            &stage,
                            &from,
                            run_result(
                                nonce.wrapping_add(1),
                                &request_id,
                                FORGED_FILL,
                                seq_len,
                                vocab,
                            ),
                        )
                        .await;
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, &request_id, LEGIT_FILL, seq_len, vocab),
                        )
                        .await;
                    }
                    Mode::WrongRequestId => {
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, "pp-seq-999", FORGED_FILL, seq_len, vocab),
                        )
                        .await;
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, &request_id, LEGIT_FILL, seq_len, vocab),
                        )
                        .await;
                    }
                    Mode::WrongShape => {
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, &request_id, FORGED_FILL, seq_len, vocab - 1),
                        )
                        .await;
                    }
                    Mode::Flood => {
                        // Flood from its own task so this stage keeps
                        // acknowledging the cleanup commands that follow the
                        // coordinator's timeout; the flood outlives the
                        // 200ms exchange deadline several times over.
                        let flood_stage = Arc::clone(&stage);
                        let flood_attacker = Arc::clone(&attacker);
                        let to = from.clone();
                        tokio::spawn(async move {
                            for i in 0..300u64 {
                                if i % 2 == 0 {
                                    send_response(
                                        &flood_attacker,
                                        &to,
                                        run_result(i, &request_id, FORGED_FILL, seq_len, vocab),
                                    )
                                    .await;
                                } else {
                                    let _ = flood_stage
                                        .send(
                                            &to,
                                            TransportMessage::Control {
                                                operation: "noise".to_string(),
                                                payload: Bytes::from_static(b"x"),
                                            },
                                        )
                                        .await;
                                }
                                tokio::time::sleep(Duration::from_millis(5)).await;
                            }
                        });
                    }
                    Mode::ForgedState | Mode::ResultFromEntryStage => {
                        send_response(
                            &stage,
                            &from,
                            run_result(nonce, &request_id, LEGIT_FILL, seq_len, vocab),
                        )
                        .await;
                    }
                    Mode::ResultFromLastStage => {
                        // `attacker` is the last stage here.
                        send_response(
                            &attacker,
                            &from,
                            run_result(nonce, &request_id, LEGIT_FILL, seq_len, vocab),
                        )
                        .await;
                    }
                }
            }
            RemoteStageCommand::ProbeState => {
                if mode == Mode::ForgedState {
                    send_response(&attacker, &from, state_response(nonce, true)).await;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                send_response(&stage, &from, state_response(nonce, false)).await;
            }
            RemoteStageCommand::PrepareSequence { .. }
            | RemoteStageCommand::ReleaseSequence { .. }
            | RemoteStageCommand::BeginDrain
            | RemoteStageCommand::CancelRequest { .. }
            | RemoteStageCommand::Shutdown => {
                send_response(&stage, &from, ack).await;
            }
        }
    }
}

struct Fixture {
    runtime: RemotePipelineRuntime,
    _rt: tokio::runtime::Runtime,
}

fn fixture(mode: Mode, stage_timeout: Duration) -> Fixture {
    fixture_with(mode, stage_timeout, false)
}

/// Two stage peers (`STAGE`, `LAST_STAGE`); the last stage acknowledges
/// lifecycle commands and is the `attacker` argument of the scripted stage.
fn fixture_two_stage(mode: Mode, stage_timeout: Duration) -> Fixture {
    fixture_with(mode, stage_timeout, true)
}

fn fixture_with(mode: Mode, stage_timeout: Duration, two_stages: bool) -> Fixture {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let other_addr = if two_stages { LAST_STAGE } else { ATTACKER };
    let (coordinator, stage, other) = rt.block_on(async {
        let router = MockRouter::new();
        let coordinator = Arc::new(
            MockTransport::new(
                COORDINATOR.to_string(),
                router.clone(),
                MockTransportConfig::default(),
            )
            .await,
        );
        let stage = Arc::new(
            MockTransport::new(
                STAGE.to_string(),
                router.clone(),
                MockTransportConfig::default(),
            )
            .await,
        );
        let other = Arc::new(
            MockTransport::new(
                other_addr.to_string(),
                router,
                MockTransportConfig::default(),
            )
            .await,
        );
        (coordinator, stage, other)
    });
    let mut stage_peers = vec![STAGE.to_string()];
    if two_stages {
        rt.spawn(run_ack_only_stage(Arc::clone(&other)));
        stage_peers.push(LAST_STAGE.to_string());
    }
    rt.spawn(run_scripted_stage(stage, other, mode));
    let runtime = RemotePipelineRuntime::with_transport(
        RemotePipelineRuntimeConfig {
            stage_peers,
            transport_backend: TransportBackend::Tcp,
            bind_address: COORDINATOR.to_string(),
            stage_timeout,
            auth: Arc::new(ClusterAuth::new(&ClusterSecret::random())),
            vocab_size: Some(VOCAB),
        },
        coordinator,
    )
    .unwrap();
    Fixture { runtime, _rt: rt }
}

fn forward(fixture: &Fixture) -> Result<UniquePtr<MlxArray>> {
    let seq_id = SequenceId::from_raw(1);
    fixture.runtime.prepare_sequence_state(seq_id);
    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    fixture.runtime.forward_sequence(seq_id, &input_ids, None)
}

fn first_value(logits: &MlxArray) -> f32 {
    mlxcel_core::utils::array_to_vec_f32(logits)[0]
}

#[test]
fn exchange_drops_run_result_from_non_stage_source() {
    let fx = fixture(Mode::ForgedSource, Duration::from_secs(5));
    let logits = forward(&fx).expect("legitimate reply must still arrive");
    assert_eq!(
        first_value(logits.as_ref().unwrap()),
        LEGIT_FILL,
        "forged logits from a non-stage peer reached the forward path"
    );
}

#[test]
fn exchange_drops_run_result_with_wrong_nonce() {
    let fx = fixture(Mode::WrongNonce, Duration::from_secs(5));
    let logits = forward(&fx).expect("legitimate reply must still arrive");
    assert_eq!(first_value(logits.as_ref().unwrap()), LEGIT_FILL);
}

#[test]
fn exchange_drops_run_result_with_wrong_request_id() {
    let fx = fixture(Mode::WrongRequestId, Duration::from_secs(5));
    let logits = forward(&fx).expect("legitimate reply must still arrive");
    assert_eq!(first_value(logits.as_ref().unwrap()), LEGIT_FILL);
}

/// The vocab axis is refused on the frame header, before the logits are
/// decoded (#2716).
#[test]
fn forward_rejects_logits_with_wrong_vocab_dimension() {
    let fx = fixture(Mode::WrongShape, Duration::from_secs(5));
    let err = forward(&fx)
        .err()
        .expect("wrong-shape logits must be rejected");
    assert!(
        err.to_string().contains(&format!(
            "axis 2 is {}, expected exactly {VOCAB}",
            VOCAB - 1
        )),
        "unexpected error: {err}"
    );
}

#[test]
fn exchange_timeout_is_absolute_under_frame_flood() {
    let fx = fixture(Mode::Flood, Duration::from_millis(200));
    let started = Instant::now();
    let err = forward(&fx).err().expect("flooded exchange must time out");
    let elapsed = started.elapsed();
    assert!(err.to_string().contains("timed out"), "{err}");
    // The flood lasts about 1.5s. With a per-frame timeout the exchange
    // alone would outlive it (plus a 200ms idle gap); with the absolute
    // deadline it ends at 200ms, followed by two cleanup exchanges that the
    // scripted stage acknowledges at once.
    assert!(
        elapsed < Duration::from_millis(1000),
        "exchange deadline was extended by the flood: {elapsed:?}"
    );
}

/// With two stage peers the `RunEntry` goes to the entry stage and the
/// `RunResult` comes from the last stage (#2694).
#[test]
fn run_result_from_last_stage_completes_forward() {
    let fx = fixture_two_stage(Mode::ResultFromLastStage, Duration::from_secs(5));
    let logits = forward(&fx).expect("the last stage's RunResult must complete the exchange");
    assert_eq!(first_value(logits.as_ref().unwrap()), LEGIT_FILL);
}

/// The same `RunResult` from the entry stage is not the awaited reply in a
/// multi-stage pipeline: it is dropped and the exchange times out.
#[test]
fn run_result_from_entry_stage_is_dropped_in_multi_stage_pipeline() {
    let fx = fixture_two_stage(Mode::ResultFromEntryStage, Duration::from_millis(300));
    let err = forward(&fx)
        .err()
        .expect("a RunResult from a stage other than the last must not complete the exchange");
    assert!(err.to_string().contains("timed out"), "{err}");
}

#[test]
fn probe_drops_forged_state_from_non_stage_source() {
    let fx = fixture(Mode::ForgedState, Duration::from_secs(5));
    let states = fx.runtime.probe_stages().expect("probe");
    assert!(matches!(
        &states[0],
        RemoteStageResponse::State { state, .. } if !state.draining
    ));
}

#[test]
fn validate_logits_shape_accepts_model_output_and_rejects_everything_else() {
    let ok = mlxcel_core::from_slice_f32(&[0.0; 8], &[1, 2, 4]);
    validate_logits_shape(&ok, 2, Some(4)).unwrap();
    let last_only = mlxcel_core::from_slice_f32(&[0.0; 4], &[1, 1, 4]);
    validate_logits_shape(&last_only, 2, Some(4)).unwrap();
    validate_logits_shape(&ok, 2, None).unwrap();

    let wrong_vocab = mlxcel_core::from_slice_f32(&[0.0; 6], &[1, 2, 3]);
    assert!(validate_logits_shape(&wrong_vocab, 2, Some(4)).is_err());
    let wrong_batch = mlxcel_core::from_slice_f32(&[0.0; 16], &[2, 2, 4]);
    assert!(validate_logits_shape(&wrong_batch, 2, Some(4)).is_err());
    let wrong_rank = mlxcel_core::from_slice_f32(&[0.0; 4], &[1, 4]);
    assert!(validate_logits_shape(&wrong_rank, 1, Some(4)).is_err());
    let wrong_seq = mlxcel_core::from_slice_f32(&[0.0; 12], &[1, 3, 4]);
    assert!(validate_logits_shape(&wrong_seq, 2, Some(4)).is_err());
    // A length strictly between 1 and the input length is neither a full
    // nor a last-position-only output.
    let between = mlxcel_core::from_slice_f32(&[0.0; 8], &[1, 2, 4]);
    assert!(validate_logits_shape(&between, 4, Some(4)).is_err());
}
