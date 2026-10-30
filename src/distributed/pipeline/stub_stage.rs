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

//! A scripted remote pipeline stage for tests: a real authenticated
//! `TcpTransport` endpoint that answers the coordinator's commands without
//! model weights.
//!
//! Used by: the remote runtime tests (`runtime.rs`), the scheduler tests
//! that drive a `PipelineServerModel` over a real `RemotePipelineRuntime`
//! against a failing stage (#2727).

use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use tokio::sync::oneshot;

use super::activation_transfer::{ActivationMessage, StageLifecycleState};
use super::remote_service::{
    PIPELINE_STAGE_COMMAND_OPERATION, PIPELINE_STAGE_RESPONSE_OPERATION, RemoteStageCommand,
    RemoteStageRequest, RemoteStageResponse,
};
use crate::distributed::cluster_auth::ClusterAuth;
use crate::distributed::tensor_protocol::TensorDtype;
use crate::distributed::{
    StageHealth, TcpTransport, TcpTransportConfig, Transport, TransportMessage,
};

/// How the stub answers a `RunEntry`.
#[derive(Clone, Copy)]
pub(crate) enum StubRunBehavior {
    /// Never answer: the coordinator's exchange times out.
    Silent,
    /// Answer every `RunEntry` with an `Error`.
    Error,
    /// Answer with `[1, n, vocab]` float32 logits that put `token` on top of
    /// every position, except the `fault_on`-th `RunEntry` (1-based, counted
    /// over the stage's lifetime), which gets `fault` instead. This is the
    /// shape of a stage that serves requests and fails one forward mid-way.
    Logits {
        vocab: i32,
        token: i32,
        fault_on: Option<u32>,
        fault: RunFault,
    },
    /// Answer every `RunEntry` with `[1, n, w]` logits that put `token` on
    /// top, where the width `w` alternates between the two `vocabs` (the
    /// first `RunEntry` gets `vocabs[0]`): a stage whose rows disagree on
    /// the logits width, which only a coordinator without a configured
    /// vocabulary size accepts row by row.
    MixedVocab { vocabs: [i32; 2], token: i32 },
}

/// The one faulted `RunEntry` of [`StubRunBehavior::Logits`].
#[derive(Clone, Copy)]
pub(crate) enum RunFault {
    /// An `Error` reply, as a stage whose forward threw sends.
    Error,
    /// No reply, as a stage that died or a dropped frame leaves the
    /// coordinator; the exchange times out after `stage_timeout`.
    Silent,
}

pub(crate) struct StubStageHandle {
    addr: String,
    commands: Arc<Mutex<Vec<String>>>,
    /// The request id of every `PrepareSequence`, in arrival order.
    prepared_ids: Arc<Mutex<Vec<String>>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<thread::JoinHandle<Result<()>>>,
}

impl StubStageHandle {
    pub(crate) fn spawn(run_behavior: StubRunBehavior, auth: Arc<ClusterAuth>) -> Result<Self> {
        Self::spawn_with(run_behavior, false, auth)
    }

    /// `reject_prepare` answers every `PrepareSequence` with an `Error`,
    /// as a stage does for a sequence another identity already prepared
    /// (#2714 items 2 and a).
    pub(crate) fn spawn_with(
        run_behavior: StubRunBehavior,
        reject_prepare: bool,
        auth: Arc<ClusterAuth>,
    ) -> Result<Self> {
        Self::spawn_inner(run_behavior, reject_prepare, false, auth)
    }

    /// A stage that never answers `CancelRequest` or `ReleaseSequence`, as
    /// a stage behind a lossy link or a stalled stage leaves the teardown
    /// broadcast: the coordinator's exchange with it times out.
    pub(crate) fn spawn_silent_on_teardown(
        run_behavior: StubRunBehavior,
        auth: Arc<ClusterAuth>,
    ) -> Result<Self> {
        Self::spawn_inner(run_behavior, false, true, auth)
    }

    fn spawn_inner(
        run_behavior: StubRunBehavior,
        reject_prepare: bool,
        silent_teardown: bool,
        auth: Arc<ClusterAuth>,
    ) -> Result<Self> {
        let commands = Arc::new(Mutex::new(Vec::new()));
        let commands_for_thread = Arc::clone(&commands);
        let prepared_ids = Arc::new(Mutex::new(Vec::new()));
        let prepared_ids_for_thread = Arc::clone(&prepared_ids);
        let (startup_tx, startup_rx) = std::sync::mpsc::channel::<Result<String>>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let join_handle = thread::spawn(move || -> Result<()> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| anyhow!("failed to build stub stage runtime: {err}"))?;
            runtime.block_on(async move {
                let transport = Arc::new(
                    TcpTransport::bind(TcpTransportConfig {
                        bind_address: "127.0.0.1:0".to_string(),
                        auth: Some(auth),
                        ..Default::default()
                    })
                    .await?,
                );
                let local_addr = transport.local_addr()?;
                startup_tx
                    .send(Ok(local_addr.clone()))
                    .map_err(|_| anyhow!("failed to publish stub stage address"))?;

                let mut lifecycle = StageLifecycleState {
                    health: StageHealth::Healthy,
                    ..Default::default()
                };
                let mut run_entries: u32 = 0;
                let mut shutdown_rx = shutdown_rx;
                loop {
                    tokio::select! {
                        _ = &mut shutdown_rx => break,
                        recv = transport.recv() => {
                            let (from, msg) = recv?;
                            let TransportMessage::Control { operation, payload } = msg else {
                                continue;
                            };
                            if operation != PIPELINE_STAGE_COMMAND_OPERATION {
                                continue;
                            }
                            let request: RemoteStageRequest = serde_json::from_slice(&payload)?;
                            let nonce = request.nonce;
                            let command_name = match &request.command {
                                RemoteStageCommand::PrepareSequence { .. } => "prepare",
                                RemoteStageCommand::ReleaseSequence { .. } => "release",
                                RemoteStageCommand::BeginDrain => "begin_drain",
                                RemoteStageCommand::CancelRequest { .. } => "cancel",
                                RemoteStageCommand::Shutdown => "shutdown",
                                RemoteStageCommand::RunEntry { .. } => "run_entry",
                                RemoteStageCommand::ProbeState => "probe_state",
                            };
                            commands_for_thread
                                .lock()
                                .expect("stub command log poisoned")
                                .push(command_name.to_string());
                            if let RemoteStageCommand::PrepareSequence { request_id } =
                                &request.command
                            {
                                prepared_ids_for_thread
                                    .lock()
                                    .expect("stub prepare log poisoned")
                                    .push(request_id.clone());
                            }

                            let reply = match request.command {
                                RemoteStageCommand::PrepareSequence { request_id }
                                    if reject_prepare =>
                                {
                                    Some(RemoteStageResponse::Error {
                                        stage_index: 0,
                                        nonce,
                                        request_id: Some(request_id.clone()),
                                        message: format!(
                                            "sequence {request_id} was prepared by another \
                                             coordinator identity"
                                        ),
                                        state: Some(lifecycle.snapshot()),
                                    })
                                }
                                RemoteStageCommand::PrepareSequence { .. } => {
                                    lifecycle.mark_request_started();
                                    Some(RemoteStageResponse::Ack {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    })
                                }
                                RemoteStageCommand::ReleaseSequence { .. }
                                | RemoteStageCommand::CancelRequest { .. }
                                    if silent_teardown =>
                                {
                                    None
                                }
                                RemoteStageCommand::ReleaseSequence { .. } => {
                                    lifecycle.mark_request_finished();
                                    Some(RemoteStageResponse::Ack {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    })
                                }
                                RemoteStageCommand::BeginDrain => {
                                    lifecycle.draining = true;
                                    Some(RemoteStageResponse::Ack {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    })
                                }
                                RemoteStageCommand::CancelRequest { request_id: _ } => {
                                    lifecycle.mark_request_finished();
                                    Some(RemoteStageResponse::Ack {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    })
                                }
                                RemoteStageCommand::Shutdown => {
                                    lifecycle.draining = true;
                                    lifecycle.shutdown = true;
                                    lifecycle.health = StageHealth::Failed;
                                    lifecycle.in_flight_requests = 0;
                                    let response = RemoteStageResponse::Ack {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    };
                                    send_stub_response(&transport, &from, response).await?;
                                    break;
                                }
                                RemoteStageCommand::RunEntry {
                                    request_id,
                                    micro_batch_id,
                                    input_ids,
                                    attention_mask: _,
                                } => {
                                    run_entries += 1;
                                    let error = |message: &str| RemoteStageResponse::Error {
                                        stage_index: 0,
                                        nonce,
                                        request_id: Some(request_id.clone()),
                                        message: message.to_string(),
                                        state: Some(lifecycle.snapshot()),
                                    };
                                    match run_behavior {
                                        StubRunBehavior::Silent => None,
                                        StubRunBehavior::Error => {
                                            Some(error("stub stage forced error"))
                                        }
                                        StubRunBehavior::Logits {
                                            fault_on: Some(n),
                                            fault,
                                            ..
                                        } if n == run_entries => match fault {
                                            RunFault::Error => {
                                                Some(error("stub stage forced error"))
                                            }
                                            RunFault::Silent => None,
                                        },
                                        StubRunBehavior::Logits { vocab, token, .. }
                                        | StubRunBehavior::MixedVocab {
                                            vocabs: [vocab, _],
                                            token,
                                        } => {
                                            let vocab = match run_behavior {
                                                StubRunBehavior::MixedVocab { vocabs, .. }
                                                    if run_entries.is_multiple_of(2) =>
                                                {
                                                    vocabs[1]
                                                }
                                                _ => vocab,
                                            };
                                            let seq_len =
                                                ActivationMessage::deserialize_activation(
                                                    &input_ids,
                                                )
                                                .map(|t| t.shape[1])
                                                .unwrap_or(1);
                                            Some(RemoteStageResponse::RunResult {
                                                stage_index: 0,
                                                nonce,
                                                request_id: request_id.clone(),
                                                micro_batch_id,
                                                logits: logits_bytes(seq_len, vocab, token)?,
                                            })
                                        }
                                    }
                                }
                                RemoteStageCommand::ProbeState => {
                                    Some(RemoteStageResponse::State {
                                        stage_index: 0,
                                        nonce,
                                        state: lifecycle.snapshot(),
                                    })
                                }
                            };

                            if let Some(response) = reply {
                                // Reply to the authenticated source, which is
                                // the coordinator's listener identity.
                                assert_eq!(request.reply_addr, from);
                                send_stub_response(&transport, &from, response).await?;
                            }
                        }
                    }
                }
                transport.shutdown().await?;
                Ok(())
            })
        });
        let addr = startup_rx
            .recv()
            .map_err(|_| anyhow!("stub stage startup channel dropped"))??;
        Ok(Self {
            addr,
            commands,
            prepared_ids,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        })
    }

    pub(crate) fn addr(&self) -> &str {
        &self.addr
    }

    /// Every command received so far, by name, in arrival order.
    pub(crate) fn commands(&self) -> Vec<String> {
        self.commands
            .lock()
            .expect("stub command log poisoned")
            .clone()
    }

    pub(crate) fn prepared_ids(&self) -> Vec<String> {
        self.prepared_ids
            .lock()
            .expect("stub prepare log poisoned")
            .clone()
    }

    pub(crate) fn shutdown(mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("stub stage thread panicked"))??;
        }
        Ok(())
    }
}

/// `[1, seq_len, vocab]` float32 wire logits with `token` scored 10.0 at
/// every position and every other id 0.0, built without MLX so the stub
/// thread never touches the device.
fn logits_bytes(seq_len: u64, vocab: i32, token: i32) -> Result<Vec<u8>> {
    let vocab = u64::try_from(vocab).map_err(|_| anyhow!("negative vocab"))?;
    let mut data = Vec::with_capacity((seq_len * vocab) as usize * 4);
    for _ in 0..seq_len {
        for id in 0..vocab {
            let score: f32 = if i64::from(token) == id as i64 {
                10.0
            } else {
                0.0
            };
            data.extend_from_slice(&score.to_le_bytes());
        }
    }
    ActivationMessage::serialize_activation(TensorDtype::Float32, &[1, seq_len, vocab], &data)
}

async fn send_stub_response(
    transport: &Arc<TcpTransport>,
    peer: &str,
    response: RemoteStageResponse,
) -> Result<()> {
    let payload = serde_json::to_vec(&response)?;
    transport
        .send(
            peer,
            TransportMessage::Control {
                operation: PIPELINE_STAGE_RESPONSE_OPERATION.to_string(),
                payload: Bytes::from(payload),
            },
        )
        .await
}
