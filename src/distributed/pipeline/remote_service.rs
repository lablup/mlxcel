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

//! TCP-backed remote pipeline stage service.
//!
//! Trust model (#2637): the transport only delivers frames from peers that
//! completed the cluster handshake, and on top of that the service accepts
//! commands only from the configured coordinator and forward activations
//! only from the configured upstream stage. Replies go to the authenticated
//! source of the command, never to an address named inside the payload. A
//! malformed frame (bad JSON, bad tensor, an activation the stage cannot
//! correlate) is a per-frame condition: it is logged and the loop keeps
//! serving. Only a closed transport channel, a `Shutdown` from the
//! coordinator, or the local shutdown signal ends the loop.
//!
//! Per-token flow (#2694): the coordinator sends `RunEntry` to the entry
//! stage, which carries the exchange nonce on the forward activation
//! (`ActivationMessage::reply_nonce`); middle stages copy it; the final stage
//! answers with `RunResult` sent straight to the coordinator's authenticated
//! identity (the source of its latest command). The logits therefore cross
//! the wire once (3 frames per decoded token: `RunEntry`, forward activation,
//! `RunResult`) instead of relaying back through every stage as a reverse
//! activation (4 frames). The service drops
//! a reverse activation; the in-process `stage_worker` keeps its own reverse
//! path. Execution errors on a non-entry stage still travel upstream with
//! nonce 0 and end in a coordinator timeout (unchanged).
//!
//! Used by: server startup (`serve_remote_pipeline_stage`), remote pipeline
//! runtime tests, real-model pipeline harnesses

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use bytes::Bytes;
use mlxcel_core::layers::KVCache;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::distributed::cluster_auth::ClusterAuth;
use crate::distributed::peer_identity::PeerMatcher;
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
        #[serde(with = "super::wire_bytes::base64_bytes")]
        input_ids: Vec<u8>,
        #[serde(with = "super::wire_bytes::base64_bytes_opt")]
        attention_mask: Option<Vec<u8>>,
    },
    ProbeState,
}

/// Envelope for every coordinator command.
///
/// `nonce` is a fresh random value per exchange; the stage echoes it in the
/// reply so the coordinator can tell this exchange's answer from a stale or
/// forged frame. `reply_addr` is informational: replies always go to the
/// authenticated source of the command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteStageRequest {
    pub reply_addr: String,
    #[serde(default)]
    pub nonce: u64,
    pub command: RemoteStageCommand,
}

/// Every variant echoes the request `nonce` so forged `Ack`, `Error`, and
/// `State` frames are rejected by the coordinator, not only `RunResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteStageResponse {
    Ack {
        stage_index: u32,
        #[serde(default)]
        nonce: u64,
        state: StageLifecycleSnapshot,
    },
    RunResult {
        stage_index: u32,
        #[serde(default)]
        nonce: u64,
        request_id: String,
        micro_batch_id: u32,
        /// Wire-format tensor bytes, base64 in JSON (#2656): the per-token
        /// logits of a 150k-vocabulary model are the largest frame of a
        /// decode step.
        #[serde(with = "super::wire_bytes::base64_bytes")]
        logits: Vec<u8>,
    },
    State {
        stage_index: u32,
        #[serde(default)]
        nonce: u64,
        state: StageLifecycleSnapshot,
    },
    Error {
        stage_index: u32,
        #[serde(default)]
        nonce: u64,
        request_id: Option<String>,
        message: String,
        state: Option<StageLifecycleSnapshot>,
    },
}

impl RemoteStageResponse {
    /// The echoed exchange nonce.
    pub fn nonce(&self) -> u64 {
        match self {
            Self::Ack { nonce, .. }
            | Self::RunResult { nonce, .. }
            | Self::State { nonce, .. }
            | Self::Error { nonce, .. } => *nonce,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RemoteStageServiceConfig {
    pub model_dir: PathBuf,
    pub bind_address: String,
    pub transport_backend: TransportBackend,
    pub stage_assignment: StageAssignment,
    pub num_stages: u32,
    /// The coordinator's control address. Commands from any other peer are
    /// dropped, authenticated or not.
    pub coordinator_peer: String,
    pub upstream_peer: Option<String>,
    pub downstream_peer: Option<String>,
    /// Cluster authentication; required, there is no unauthenticated mode.
    pub auth: Arc<ClusterAuth>,
}

/// Builds the stage executor inside the service thread, so executors that
/// are not `Send` (MLX-backed) never cross a thread boundary.
type ExecutorFactory = Box<dyn FnOnce() -> Result<Box<dyn StageExecutor>> + Send>;

struct RemoteStageService {
    stage_index: u32,
    num_stages: u32,
    transport: Arc<dyn Transport>,
    executor: Box<dyn StageExecutor>,
    caches_by_request: HashMap<String, Vec<KVCache>>,
    active_requests: HashSet<String>,
    coordinator: PeerMatcher,
    /// The authenticated identity of the coordinator's most recent command.
    /// The final stage sends its `RunResult` here rather than to the
    /// configured address, which may name no single host (`0.0.0.0:19000`
    /// from a `--pp-auto` coordinator on every interface); every `RunEntry`
    /// is preceded by a `PrepareSequence` broadcast that sets it (#2694).
    coordinator_identity: Option<String>,
    upstream: Option<PeerMatcher>,
    downstream: Option<PeerMatcher>,
    lifecycle: StageLifecycleState,
}

impl RemoteStageService {
    async fn load(
        config: RemoteStageServiceConfig,
        transport: Arc<dyn Transport>,
        executor_factory: ExecutorFactory,
    ) -> Result<Self> {
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
        let coordinator = PeerMatcher::resolve(&config.coordinator_peer).await;
        let upstream = match &config.upstream_peer {
            Some(peer) => Some(PeerMatcher::resolve(peer).await),
            None => None,
        };
        let downstream = match &config.downstream_peer {
            Some(peer) => Some(PeerMatcher::resolve(peer).await),
            None => None,
        };
        let executor = executor_factory()?;
        Ok(Self {
            stage_index: config.stage_assignment.stage_index as u32,
            num_stages: config.num_stages,
            transport,
            executor,
            caches_by_request: HashMap::new(),
            active_requests: HashSet::new(),
            coordinator,
            coordinator_identity: None,
            upstream,
            downstream,
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
                    // Only a closed inbound channel is an infrastructure
                    // failure that ends the service.
                    let (from, msg) = recv?;
                    match self.handle_transport_message(&from, msg).await {
                        Ok(true) => break,
                        Ok(false) => {}
                        Err(err) => {
                            // Per-frame condition: a malformed or
                            // unserviceable frame must not take the stage
                            // down. The coordinator times out that exchange
                            // and cleans up; the stage keeps serving.
                            tracing::warn!(
                                "remote stage {}: dropping frame from {from}: {err:#}",
                                self.stage_index
                            );
                        }
                    }
                }
            }
        }
        self.transport.shutdown().await?;
        Ok(())
    }

    async fn handle_transport_message(
        &mut self,
        from: &str,
        msg: TransportMessage,
    ) -> Result<bool> {
        match msg {
            TransportMessage::Control { operation, payload } => {
                if operation == PIPELINE_STAGE_COMMAND_OPERATION {
                    if !self.coordinator.matches(from) {
                        tracing::warn!(
                            "remote stage {}: ignoring command from {from}; only the coordinator \
                             {} may issue commands",
                            self.stage_index,
                            self.coordinator.configured()
                        );
                        return Ok(false);
                    }
                    if self.coordinator_identity.as_deref() != Some(from) {
                        self.coordinator_identity = Some(from.to_string());
                    }
                    let request: RemoteStageRequest = serde_json::from_slice(&payload)
                        .context("malformed pipeline_stage_command payload")?;
                    if request.reply_addr != from {
                        tracing::debug!(
                            "remote stage {}: command names reply_addr {} but came from {from}; \
                             replying to the authenticated source",
                            self.stage_index,
                            request.reply_addr
                        );
                    }
                    self.handle_command(from, request.nonce, request.command)
                        .await
                } else if operation == PIPELINE_ACTIVATION_OPERATION {
                    let activation: ActivationMessage = serde_json::from_slice(&payload)
                        .context("malformed pipeline_activation payload")?;
                    if activation.is_reverse_path {
                        // Logits travel straight from the final stage to the
                        // coordinator (#2694); nothing legitimate comes back
                        // along the stages.
                        tracing::warn!(
                            "remote stage {}: dropping reverse activation from {from} for {}:{}; \
                             the final stage sends its result to the coordinator directly",
                            self.stage_index,
                            activation.request_id,
                            activation.micro_batch_id
                        );
                        return Ok(false);
                    }
                    match self.upstream.as_ref() {
                        Some(peer) if peer.matches(from) => {}
                        Some(peer) => {
                            tracing::warn!(
                                "remote stage {}: ignoring forward activation from {from}; expected \
                                 it from {}",
                                self.stage_index,
                                peer.configured()
                            );
                            return Ok(false);
                        }
                        None => {
                            tracing::warn!(
                                "remote stage {}: ignoring forward activation from {from}; this \
                                 stage has no upstream neighbour",
                                self.stage_index
                            );
                            return Ok(false);
                        }
                    }
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
        nonce: u64,
        request: RemoteStageCommand,
    ) -> Result<bool> {
        match request {
            RemoteStageCommand::PrepareSequence { request_id } => {
                if self.lifecycle.draining {
                    self.send_request_error(
                        reply_addr,
                        nonce,
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
                self.send_response(reply_addr, self.ack_response(nonce))
                    .await?;
                Ok(false)
            }
            RemoteStageCommand::ReleaseSequence { request_id } => {
                self.finish_request(&request_id);
                self.send_response(reply_addr, self.ack_response(nonce))
                    .await?;
                Ok(false)
            }
            RemoteStageCommand::BeginDrain => {
                self.lifecycle.draining = true;
                self.send_response(reply_addr, self.ack_response(nonce))
                    .await?;
                Ok(false)
            }
            RemoteStageCommand::CancelRequest { request_id } => {
                self.finish_request(&request_id);
                self.send_response(reply_addr, self.ack_response(nonce))
                    .await?;
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
                self.send_response(reply_addr, self.ack_response(nonce))
                    .await?;
                Ok(true)
            }
            RemoteStageCommand::RunEntry {
                request_id,
                micro_batch_id,
                input_ids,
                attention_mask,
            } => {
                self.handle_run_entry(
                    reply_addr,
                    nonce,
                    request_id,
                    micro_batch_id,
                    &input_ids,
                    attention_mask.as_deref(),
                )
                .await
            }
            RemoteStageCommand::ProbeState => {
                self.send_response(
                    reply_addr,
                    RemoteStageResponse::State {
                        stage_index: self.stage_index,
                        nonce,
                        state: self.lifecycle.snapshot(),
                    },
                )
                .await?;
                Ok(false)
            }
        }
    }

    async fn handle_run_entry(
        &mut self,
        reply_addr: &str,
        nonce: u64,
        request_id: String,
        micro_batch_id: u32,
        input_ids: &[u8],
        attention_mask: Option<&[u8]>,
    ) -> Result<bool> {
        if self.stage_index != 0 {
            self.send_request_error(
                reply_addr,
                nonce,
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
                nonce,
                Some(request_id),
                "pipeline stage is draining",
            )
            .await?;
            return Ok(false);
        }
        // A bad wire tensor is the coordinator's (or a forger's) problem,
        // not a reason to stop serving: answer with an error and continue.
        let decoded = deserialize_wire_tensor(input_ids).and_then(|ids| {
            attention_mask
                .map(deserialize_wire_tensor)
                .transpose()
                .map(|mask| (ids, mask))
        });
        let (input_ids, attention_mask) = match decoded {
            Ok(decoded) => decoded,
            Err(err) => {
                self.send_request_error(
                    reply_addr,
                    nonce,
                    Some(request_id),
                    &format!("invalid entry tensors: {err}"),
                )
                .await?;
                return Ok(false);
            }
        };
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
                self.send_request_error(reply_addr, nonce, Some(request_id), &err.to_string())
                    .await?;
                Ok(false)
            }
            Ok(StageExecutionOutput::Logits(logits)) => {
                self.send_response(
                    reply_addr,
                    RemoteStageResponse::RunResult {
                        stage_index: self.stage_index,
                        nonce,
                        request_id,
                        micro_batch_id,
                        logits: serialize_mlx_array(logits.as_ref().unwrap())?,
                    },
                )
                .await?;
                Ok(false)
            }
            Ok(StageExecutionOutput::HiddenStates(hidden_states)) => {
                let downstream = self
                    .downstream
                    .as_ref()
                    .map(|peer| peer.configured().to_string())
                    .ok_or_else(|| {
                        anyhow!("entry stage {} has no downstream peer", self.stage_index)
                    })?;
                let seq_len = sequence_length(hidden_states.as_ref().unwrap())?;
                // The exchange nonce rides the forward activation; the final
                // stage echoes it in the RunResult it sends to the
                // coordinator (#2694), so no reply is recorded here.
                if let Err(err) = self
                    .send_activation(
                        &downstream,
                        ActivationMessage::forward(
                            RequestId::from_string(request_id.clone()).ok_or_else(|| {
                                anyhow!("invalid request id for remote entry stage: {request_id}")
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
                        )
                        .with_reply_nonce(Some(nonce)),
                    )
                    .await
                {
                    self.finish_request(&request_id);
                    self.send_request_error(
                        reply_addr,
                        nonce,
                        Some(request_id.clone()),
                        &err.to_string(),
                    )
                    .await?;
                }
                Ok(false)
            }
        }
    }

    /// Run a forward activation from the upstream stage: hand hidden states
    /// downstream with the same `reply_nonce`, or, on the final stage, send
    /// the logits to the coordinator as the `RunResult` of that nonce.
    async fn handle_activation(&mut self, activation: ActivationMessage) -> Result<()> {
        let request_key = activation.request_id.as_str().to_string();
        ensure!(
            !activation.is_reverse_path,
            "stage {} cannot handle a reverse activation",
            self.stage_index
        );
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
                if let Some(upstream) = self.upstream_addr() {
                    self.send_response(
                        &upstream,
                        RemoteStageResponse::Error {
                            stage_index: self.stage_index,
                            nonce: 0,
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
                    .downstream
                    .as_ref()
                    .map(|peer| peer.configured().to_string())
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
                    )
                    .with_reply_nonce(activation.reply_nonce),
                )
                .await
            }
            Ok(StageExecutionOutput::Logits(logits)) => {
                // Final stage: the logits go once, straight to the
                // coordinator, as the RunResult of the RunEntry exchange whose
                // nonce the activation carried (#2694). Without that nonce
                // the coordinator could not match the frame to an exchange,
                // so there is nothing useful to send.
                let Some(nonce) = activation.reply_nonce else {
                    tracing::warn!(
                        "remote stage {}: forward activation for {}:{} carries no reply nonce; \
                         dropping its logits (an entry stage older than #2694?)",
                        self.stage_index,
                        request_key,
                        activation.micro_batch_id
                    );
                    return Ok(());
                };
                // The coordinator's authenticated source, as for every other
                // reply; the configured address only before any command.
                let coordinator = self
                    .coordinator_identity
                    .clone()
                    .unwrap_or_else(|| self.coordinator.configured().to_string());
                self.send_response(
                    &coordinator,
                    RemoteStageResponse::RunResult {
                        stage_index: self.stage_index,
                        nonce,
                        request_id: request_key,
                        micro_batch_id: activation.micro_batch_id,
                        logits: serialize_mlx_array(logits.as_ref().unwrap())?,
                    },
                )
                .await
            }
        }
    }

    fn upstream_addr(&self) -> Option<String> {
        self.upstream
            .as_ref()
            .map(|peer| peer.configured().to_string())
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

    fn ack_response(&self, nonce: u64) -> RemoteStageResponse {
        RemoteStageResponse::Ack {
            stage_index: self.stage_index,
            nonce,
            state: self.lifecycle.snapshot(),
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
    }

    async fn send_request_error(
        &self,
        peer: &str,
        nonce: u64,
        request_id: Option<String>,
        message: &str,
    ) -> Result<()> {
        self.send_response(
            peer,
            RemoteStageResponse::Error {
                stage_index: self.stage_index,
                nonce,
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
    /// Resolves when the service thread has returned, so a supervisor can
    /// react to a service that died (for example a neighbour that rejected
    /// the cluster handshake) instead of waiting for a shutdown signal.
    exited_rx: Option<oneshot::Receiver<()>>,
}

impl RemoteStageServiceHandle {
    /// Spawn the service, loading the stage executor from `config.model_dir`.
    pub fn spawn(config: RemoteStageServiceConfig) -> Result<Self> {
        let model_dir = config.model_dir.clone();
        let stage_assignment = config.stage_assignment.clone();
        Self::spawn_with_executor(
            config,
            Box::new(move || {
                LoadedStageExecutor::load(&model_dir, &stage_assignment)
                    .with_context(|| {
                        format!(
                            "failed to load remote stage {} from {}",
                            stage_assignment.stage_index,
                            model_dir.display()
                        )
                    })
                    .map(|executor| Box::new(executor) as Box<dyn StageExecutor>)
            }),
        )
    }

    /// Spawn the service around an executor built by `executor_factory` on
    /// the service thread. Tests use this to drive the control plane without
    /// model weights.
    pub(crate) fn spawn_with_executor(
        config: RemoteStageServiceConfig,
        executor_factory: ExecutorFactory,
    ) -> Result<Self> {
        let (startup_tx, startup_rx) = std::sync::mpsc::channel::<Result<String>>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (exited_tx, exited_rx) = oneshot::channel();
        let join_handle = std::thread::spawn(move || -> Result<()> {
            // Re-impose fail-fast on this core inference thread under release
            // `panic = "unwind"` (issue #375): `RemoteStageService::run` executes
            // core model forward passes for this pipeline stage, so an uncaught
            // panic must abort the stage process for a supervised restart. Without
            // this, a panic would unwind out of the thread and leave
            // `serve_remote_pipeline_stage` (src/server/startup.rs) parked on
            // `ctrl_c` with a zombie stage that a supervisor will not restart. A
            // normal `Err` return is the contained path: the wrapper forwards it
            // to the join side unchanged, so only a panic aborts. Per-frame
            // errors (malformed or unauthorized frames) never reach here: the
            // run loop logs them and keeps serving (#2637); only a closed
            // transport channel returns `Err` from `run`.
            let result =
                run_core_thread_or_abort("remote-pipeline-stage", move || -> Result<()> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .context("failed to build remote stage runtime")?;
                    runtime.block_on(async move {
                        let transport = bind_transport(
                            config.transport_backend,
                            &config.bind_address,
                            Some(Arc::clone(&config.auth)),
                        )
                        .await?;
                        let local_addr = transport.local_addr()?;
                        startup_tx
                            .send(Ok(local_addr))
                            .map_err(|_| anyhow!("failed to publish remote stage local address"))?;
                        let service =
                            RemoteStageService::load(config, transport, executor_factory).await?;
                        service.run(shutdown_rx).await
                    })
                });
            let _ = exited_tx.send(());
            result
        });
        let local_addr = startup_rx
            .recv()
            .map_err(|_| anyhow!("remote stage startup channel dropped"))??;
        Ok(Self {
            local_addr,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
            exited_rx: Some(exited_rx),
        })
    }

    pub fn local_addr(&self) -> &str {
        &self.local_addr
    }

    /// Wait until the service thread has exited for any reason. Resolves
    /// immediately if it already has; `shutdown` then returns its result.
    pub async fn exited(&mut self) {
        if let Some(rx) = self.exited_rx.take() {
            let _ = rx.await;
        }
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

#[cfg(test)]
#[path = "remote_service_tests.rs"]
mod tests;
