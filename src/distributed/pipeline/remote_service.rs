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

//! TCP-backed remote pipeline stage service.
//!
//! Used by: remote pipeline runtime tests, future cross-machine server startup

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use bytes::Bytes;
use mlxcel_core::layers::KVCache;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::distributed::pipeline::activation_transfer::{
    ActivationMessage, PIPELINE_ACTIVATION_OPERATION, StageLifecycleSnapshot, StageLifecycleState,
};
use crate::distributed::request_tracker::RequestId;
use crate::distributed::transport_factory::bind_transport;
use crate::distributed::{Transport, TransportBackend, TransportMessage};
use crate::worker_failfast::run_core_thread_or_abort;

use super::stage_executor::{LoadedStageExecutor, StageExecutionInput, StageExecutionOutput};
use super::wire_tensor::{deserialize_wire_tensor, sequence_length, serialize_mlx_array};
use super::{StageAssignment, StageExecutor};

pub(crate) const PIPELINE_STAGE_COMMAND_OPERATION: &str = "pipeline_stage_command";
pub(crate) const PIPELINE_STAGE_RESPONSE_OPERATION: &str = "pipeline_stage_response";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteStageCommand {
    PrepareSequence {
        request_id: String,
    },
    ReleaseSequence {
        request_id: String,
    },
    BeginDrain,
    CancelRequest {
        request_id: String,
    },
    Shutdown,
    RunEntry {
        request_id: String,
        micro_batch_id: u32,
        input_ids: Vec<u8>,
        attention_mask: Option<Vec<u8>>,
    },
    ProbeState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteStageRequest {
    pub reply_addr: String,
    pub command: RemoteStageCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteStageResponse {
    Ack {
        stage_index: u32,
        state: StageLifecycleSnapshot,
        pending_entry_replies: usize,
    },
    RunResult {
        stage_index: u32,
        request_id: String,
        micro_batch_id: u32,
        logits: Vec<u8>,
    },
    State {
        stage_index: u32,
        state: StageLifecycleSnapshot,
        pending_entry_replies: usize,
    },
    Error {
        stage_index: u32,
        request_id: Option<String>,
        message: String,
        state: Option<StageLifecycleSnapshot>,
    },
}

#[derive(Debug, Clone)]
pub struct RemoteStageServiceConfig {
    pub model_dir: PathBuf,
    pub bind_address: String,
    pub transport_backend: TransportBackend,
    pub stage_assignment: StageAssignment,
    pub num_stages: u32,
    pub upstream_peer: Option<String>,
    pub downstream_peer: Option<String>,
}

struct PendingEntryReply {
    reply_peer: String,
    request_id: String,
    micro_batch_id: u32,
}

struct RemoteStageService {
    stage_index: u32,
    num_stages: u32,
    transport: Arc<dyn Transport>,
    executor: LoadedStageExecutor,
    caches_by_request: HashMap<String, Vec<KVCache>>,
    active_requests: HashSet<String>,
    pending_entry_replies: HashMap<(String, u32), PendingEntryReply>,
    upstream_peer: Option<String>,
    downstream_peer: Option<String>,
    lifecycle: StageLifecycleState,
}

impl RemoteStageService {
    async fn load(config: RemoteStageServiceConfig, transport: Arc<dyn Transport>) -> Result<Self> {
        ensure!(
            config.stage_assignment.stage_index < config.num_stages as usize,
            "stage assignment index {} exceeds configured stage count {}",
            config.stage_assignment.stage_index,
            config.num_stages
        );
        let peers: Vec<String> = [
            config.upstream_peer.as_ref(),
            config.downstream_peer.as_ref(),
        ]
        .into_iter()
        .flatten()
        .cloned()
        .collect();
        transport.connect(&peers).await?;
        let executor = LoadedStageExecutor::load(&config.model_dir, &config.stage_assignment)
            .with_context(|| {
                format!(
                    "failed to load remote stage {} from {}",
                    config.stage_assignment.stage_index,
                    config.model_dir.display()
                )
            })?;
        Ok(Self {
            stage_index: config.stage_assignment.stage_index as u32,
            num_stages: config.num_stages,
            transport,
            executor,
            caches_by_request: HashMap::new(),
            active_requests: HashSet::new(),
            pending_entry_replies: HashMap::new(),
            upstream_peer: config.upstream_peer,
            downstream_peer: config.downstream_peer,
            lifecycle: StageLifecycleState {
                health: crate::distributed::pipeline::StageHealth::Healthy,
                ..Default::default()
            },
        })
    }

    async fn run(mut self, mut shutdown_rx: oneshot::Receiver<()>) -> Result<()> {
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => break,
                recv = self.transport.recv() => {
                    let (from, msg) = recv?;
                    let should_shutdown = self.handle_transport_message(&from, msg).await?;
                    if should_shutdown {
                        break;
                    }
                }
            }
        }
        self.transport.shutdown().await?;
        Ok(())
    }

    async fn handle_transport_message(
        &mut self,
        _from: &str,
        msg: TransportMessage,
    ) -> Result<bool> {
        match msg {
            TransportMessage::Control { operation, payload } => {
                if operation == PIPELINE_STAGE_COMMAND_OPERATION {
                    let request: RemoteStageRequest = serde_json::from_slice(&payload)?;
                    self.handle_command(&request.reply_addr, request.command)
                        .await
                } else if operation == PIPELINE_ACTIVATION_OPERATION {
                    let activation: ActivationMessage = serde_json::from_slice(&payload)?;
                    self.handle_activation(activation).await?;
                    Ok(false)
                } else {
                    Ok(false)
                }
            }
            TransportMessage::TensorData { .. } => Ok(false),
        }
    }

    async fn handle_command(
        &mut self,
        reply_addr: &str,
        request: RemoteStageCommand,
    ) -> Result<bool> {
        match request {
            RemoteStageCommand::PrepareSequence { request_id } => {
                if self.lifecycle.draining {
                    self.send_request_error(
                        reply_addr,
                        Some(request_id),
                        "pipeline stage is draining",
                    )
                    .await?;
                    return Ok(false);
                }
                self.mark_request_started(&request_id);
                self.caches_by_request
                    .entry(request_id)
                    .or_insert_with(|| self.executor.make_caches());
                self.send_response(reply_addr, self.ack_response()).await?;
                Ok(false)
            }
            RemoteStageCommand::ReleaseSequence { request_id } => {
                self.finish_request(&request_id);
                self.send_response(reply_addr, self.ack_response()).await?;
                Ok(false)
            }
            RemoteStageCommand::BeginDrain => {
                self.lifecycle.draining = true;
                self.send_response(reply_addr, self.ack_response()).await?;
                Ok(false)
            }
            RemoteStageCommand::CancelRequest { request_id } => {
                self.finish_request(&request_id);
                self.send_response(reply_addr, self.ack_response()).await?;
                Ok(false)
            }
            RemoteStageCommand::Shutdown => {
                self.lifecycle.draining = true;
                self.lifecycle.shutdown = true;
                self.lifecycle.health = crate::distributed::pipeline::StageHealth::Failed;
                let active: Vec<String> = self.active_requests.iter().cloned().collect();
                for request_id in active {
                    self.finish_request(&request_id);
                }
                self.send_response(reply_addr, self.ack_response()).await?;
                Ok(true)
            }
            RemoteStageCommand::RunEntry {
                request_id,
                micro_batch_id,
                input_ids,
                attention_mask,
            } => {
                if self.stage_index != 0 {
                    self.send_request_error(
                        reply_addr,
                        Some(request_id),
                        &format!(
                            "stage {} cannot accept entry execution requests",
                            self.stage_index
                        ),
                    )
                    .await?;
                    return Ok(false);
                }
                if self.lifecycle.draining && !self.active_requests.contains(&request_id) {
                    self.send_request_error(
                        reply_addr,
                        Some(request_id),
                        "pipeline stage is draining",
                    )
                    .await?;
                    return Ok(false);
                }
                let input_ids = deserialize_wire_tensor(&input_ids)?;
                let attention_mask = attention_mask
                    .as_deref()
                    .map(deserialize_wire_tensor)
                    .transpose()?;
                self.mark_request_started(&request_id);
                let output = {
                    let caches = self
                        .caches_by_request
                        .entry(request_id.clone())
                        .or_insert_with(|| self.executor.make_caches());
                    self.executor.execute(
                        StageExecutionInput::TokenIds(input_ids.as_ref().unwrap()),
                        caches,
                        attention_mask.as_ref().and_then(|mask| mask.as_ref()),
                    )
                };
                match output {
                    Err(err) => {
                        self.finish_request(&request_id);
                        self.send_request_error(reply_addr, Some(request_id), &err.to_string())
                            .await?;
                        Ok(false)
                    }
                    Ok(StageExecutionOutput::Logits(logits)) => {
                        self.send_response(
                            reply_addr,
                            RemoteStageResponse::RunResult {
                                stage_index: self.stage_index,
                                request_id,
                                micro_batch_id,
                                logits: serialize_mlx_array(logits.as_ref().unwrap())?,
                            },
                        )
                        .await?;
                        Ok(false)
                    }
                    Ok(StageExecutionOutput::HiddenStates(hidden_states)) => {
                        let downstream = self.downstream_peer.clone().ok_or_else(|| {
                            anyhow!("entry stage {} has no downstream peer", self.stage_index)
                        })?;
                        let seq_len = sequence_length(hidden_states.as_ref().unwrap())?;
                        self.pending_entry_replies.insert(
                            (request_id.clone(), micro_batch_id),
                            PendingEntryReply {
                                reply_peer: reply_addr.to_string(),
                                // reply path uses the caller's listening transport address.
                                request_id: request_id.clone(),
                                micro_batch_id,
                            },
                        );
                        if let Err(err) = self
                            .send_activation(
                            &downstream,
                            ActivationMessage::forward(
                                RequestId::from_string(request_id.clone()).ok_or_else(|| {
                                    anyhow!(
                                        "invalid request id for remote entry stage: {request_id}"
                                    )
                                })?,
                                micro_batch_id,
                                self.stage_index,
                                self.num_stages,
                                serialize_mlx_array(hidden_states.as_ref().unwrap())?,
                                attention_mask
                                    .as_ref()
                                    .and_then(|mask| mask.as_ref())
                                    .map(serialize_mlx_array)
                                    .transpose()?,
                                None,
                                seq_len,
                            ),
                        )
                        .await
                        {
                            self.finish_request(&request_id);
                            self.send_request_error(
                                reply_addr,
                                Some(request_id.clone()),
                                &err.to_string(),
                            )
                                .await?;
                        }
                        Ok(false)
                    }
                }
            }
            RemoteStageCommand::ProbeState => {
                self.send_response(
                    reply_addr,
                    RemoteStageResponse::State {
                        stage_index: self.stage_index,
                        state: self.lifecycle.snapshot(),
                        pending_entry_replies: self.pending_entry_replies.len(),
                    },
                )
                .await?;
                Ok(false)
            }
        }
    }

    async fn handle_activation(&mut self, activation: ActivationMessage) -> Result<()> {
        let request_key = activation.request_id.as_str().to_string();
        if activation.is_reverse_path {
            if self.stage_index == 0 {
                let Some(pending) = self
                    .pending_entry_replies
                    .remove(&(request_key.clone(), activation.micro_batch_id))
                else {
                    bail!(
                        "entry stage {} received reverse activation without pending reply for {}:{}",
                        self.stage_index,
                        request_key,
                        activation.micro_batch_id
                    );
                };
                return self
                    .send_response(
                        &pending.reply_peer,
                        RemoteStageResponse::RunResult {
                            stage_index: self.stage_index,
                            request_id: pending.request_id,
                            micro_batch_id: pending.micro_batch_id,
                            logits: activation.tensor_data,
                        },
                    )
                    .await;
            }

            let upstream = self.upstream_peer.clone().ok_or_else(|| {
                anyhow!(
                    "stage {} received reverse activation without upstream peer",
                    self.stage_index
                )
            })?;
            return self.send_activation(&upstream, activation).await;
        }

        ensure!(
            self.stage_index > 0,
            "entry stage {} unexpectedly received forward activation",
            self.stage_index
        );
        let hidden_states = deserialize_wire_tensor(&activation.tensor_data)?;
        let attention_mask = activation
            .attention_mask
            .as_deref()
            .map(deserialize_wire_tensor)
            .transpose()?;
        self.mark_request_started(&request_key);
        let output = {
            let caches = self
                .caches_by_request
                .entry(request_key.clone())
                .or_insert_with(|| self.executor.make_caches());
            self.executor.execute(
                StageExecutionInput::HiddenStates(hidden_states.as_ref().unwrap()),
                caches,
                attention_mask.as_ref().and_then(|mask| mask.as_ref()),
            )
        };
        match output {
            Err(err) => {
                self.finish_request(&request_key);
                if let Some(upstream) = self.upstream_peer.clone() {
                    self.send_response(
                        &upstream,
                        RemoteStageResponse::Error {
                            stage_index: self.stage_index,
                            request_id: Some(request_key),
                            message: err.to_string(),
                            state: Some(self.lifecycle.snapshot()),
                        },
                    )
                    .await?;
                }
                Ok(())
            }
            Ok(StageExecutionOutput::HiddenStates(hidden_states)) => {
                let downstream = self
                    .downstream_peer
                    .clone()
                    .ok_or_else(|| anyhow!("stage {} has no downstream peer", self.stage_index))?;
                let seq_len = sequence_length(hidden_states.as_ref().unwrap())?;
                self.send_activation(
                    &downstream,
                    ActivationMessage::forward(
                        RequestId::from_string(request_key.clone()).ok_or_else(|| {
                            anyhow!("invalid request id for remote stage: {request_key}")
                        })?,
                        activation.micro_batch_id,
                        self.stage_index,
                        self.num_stages,
                        serialize_mlx_array(hidden_states.as_ref().unwrap())?,
                        attention_mask
                            .as_ref()
                            .and_then(|mask| mask.as_ref())
                            .map(serialize_mlx_array)
                            .transpose()?,
                        None,
                        seq_len,
                    ),
                )
                .await
            }
            Ok(StageExecutionOutput::Logits(logits)) => {
                let upstream = self.upstream_peer.clone().ok_or_else(|| {
                    anyhow!(
                        "stage {} produced logits without upstream peer",
                        self.stage_index
                    )
                })?;
                let seq_len = sequence_length(logits.as_ref().unwrap())?;
                self.send_activation(
                    &upstream,
                    ActivationMessage::reverse(
                        RequestId::from_string(request_key.clone()).ok_or_else(|| {
                            anyhow!("invalid request id for remote stage: {request_key}")
                        })?,
                        activation.micro_batch_id,
                        self.stage_index,
                        self.num_stages,
                        serialize_mlx_array(logits.as_ref().unwrap())?,
                        seq_len,
                    ),
                )
                .await
            }
        }
    }

    async fn send_activation(&self, peer: &str, message: ActivationMessage) -> Result<()> {
        let payload = serde_json::to_vec(&message)?;
        self.transport
            .send(
                peer,
                TransportMessage::Control {
                    operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                    payload: Bytes::from(payload),
                },
            )
            .await
    }

    async fn send_response(&self, peer: &str, response: RemoteStageResponse) -> Result<()> {
        let payload = serde_json::to_vec(&response)?;
        self.transport
            .send(
                peer,
                TransportMessage::Control {
                    operation: PIPELINE_STAGE_RESPONSE_OPERATION.to_string(),
                    payload: Bytes::from(payload),
                },
            )
            .await
    }

    fn ack_response(&self) -> RemoteStageResponse {
        RemoteStageResponse::Ack {
            stage_index: self.stage_index,
            state: self.lifecycle.snapshot(),
            pending_entry_replies: self.pending_entry_replies.len(),
        }
    }

    fn mark_request_started(&mut self, request_id: &str) {
        if self.active_requests.insert(request_id.to_string()) {
            self.lifecycle.mark_request_started();
        }
    }

    fn finish_request(&mut self, request_id: &str) {
        if self.active_requests.remove(request_id) {
            self.lifecycle.mark_request_finished();
        }
        if let Some(caches) = self.caches_by_request.remove(request_id) {
            self.executor.release_caches(&caches);
        }
        self.pending_entry_replies
            .retain(|(pending_request_id, _), _| pending_request_id != request_id);
    }

    async fn send_request_error(
        &self,
        peer: &str,
        request_id: Option<String>,
        message: &str,
    ) -> Result<()> {
        self.send_response(
            peer,
            RemoteStageResponse::Error {
                stage_index: self.stage_index,
                request_id,
                message: message.to_string(),
                state: Some(self.lifecycle.snapshot()),
            },
        )
        .await
    }
}

pub struct RemoteStageServiceHandle {
    local_addr: String,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join_handle: Option<std::thread::JoinHandle<Result<()>>>,
}

impl RemoteStageServiceHandle {
    pub fn spawn(config: RemoteStageServiceConfig) -> Result<Self> {
        let (startup_tx, startup_rx) = std::sync::mpsc::channel::<Result<String>>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let join_handle = std::thread::spawn(move || -> Result<()> {
            // Re-impose fail-fast on this core inference thread under release
            // `panic = "unwind"` (issue #375): `RemoteStageService::run` executes
            // core model forward passes for this pipeline stage, so an uncaught
            // panic must abort the stage process for a supervised restart. Without
            // this, a panic would unwind out of the thread and leave
            // `serve_remote_pipeline_stage` (src/server/startup.rs) parked on
            // `ctrl_c` with a zombie stage that a supervisor will not restart. A
            // normal `Err` return is the contained path: the wrapper forwards it
            // to the join side unchanged, so only a panic aborts.
            run_core_thread_or_abort("remote-pipeline-stage", move || -> Result<()> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .context("failed to build remote stage runtime")?;
                runtime.block_on(async move {
                    let transport =
                        bind_transport(config.transport_backend, &config.bind_address).await?;
                    let local_addr = transport.local_addr()?;
                    startup_tx
                        .send(Ok(local_addr))
                        .map_err(|_| anyhow!("failed to publish remote stage local address"))?;
                    let service = RemoteStageService::load(config, transport).await?;
                    service.run(shutdown_rx).await
                })
            })
        });
        let local_addr = startup_rx
            .recv()
            .map_err(|_| anyhow!("remote stage startup channel dropped"))??;
        Ok(Self {
            local_addr,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        })
    }

    pub fn local_addr(&self) -> &str {
        &self.local_addr
    }

    pub fn shutdown(mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("remote stage thread panicked"))??;
        }
        Ok(())
    }
}
