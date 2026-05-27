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

//! Runtime backends for server-side pipeline execution.
//!
//! Used by: `server_runtime`, future remote PP runtime

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::concatenate;
use mlxcel_core::{MlxArray, UniquePtr, copy, slice};

use crate::distributed::RequestId;
use crate::distributed::transport_factory::bind_transport;
use crate::distributed::{Transport, TransportBackend, TransportMessage};

use super::remote_service::{
    PIPELINE_STAGE_COMMAND_OPERATION, PIPELINE_STAGE_RESPONSE_OPERATION, RemoteStageCommand,
    RemoteStageRequest, RemoteStageResponse,
};
use super::wire_tensor::{deserialize_wire_tensor, serialize_mlx_array};
use super::{
    InProcessStageWorkerLoop, PipelineWorkerInput, load_in_process_stage_worker_with_adapter,
    log_partition_quality, resolve_in_process_pipeline_num_layers,
    resolve_in_process_stage_assignments_for_model,
};

/// Server-side pipeline runtime abstraction.
///
/// Implementations may execute stages locally, dispatch them to remote peers,
/// or mix both strategies while keeping the `LanguageModel`-facing control
/// plane unchanged.
pub trait PipelineModelRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId);
    fn release_sequence_state_by_id(&self, seq_id: SequenceId);
    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>>;
    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>>;
}

/// Current local runtime backed by the shared in-process stage worker loop.
pub struct InProcessPipelineRuntime {
    worker_loop: Mutex<InProcessStageWorkerLoop>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
}

impl InProcessPipelineRuntime {
    pub fn load(
        model_dir: &Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
    ) -> Result<(usize, Vec<i32>, Self)> {
        Self::load_with_adapter(model_dir, pp_layers, pp_micro_batch_size, None)
    }

    pub fn load_with_adapter(
        model_dir: &Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
        adapter_path: Option<&Path>,
    ) -> Result<(usize, Vec<i32>, Self)> {
        let num_layers = resolve_in_process_pipeline_num_layers(model_dir)?;
        let (assignments, report) =
            resolve_in_process_stage_assignments_for_model(model_dir, num_layers, None, pp_layers)?;
        log_partition_quality(&report);
        let worker_loop = load_in_process_stage_worker_with_adapter(
            model_dir,
            &assignments,
            pp_micro_batch_size,
            adapter_path,
        )?;
        Ok((
            num_layers,
            crate::read_eos_token_ids(model_dir),
            Self {
                worker_loop: Mutex::new(worker_loop),
                request_ids: Mutex::new(HashMap::new()),
            },
        ))
    }

    fn request_id_for(&self, seq_id: SequenceId) -> RequestId {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .get(&seq_id)
            .cloned()
            .unwrap_or_else(|| panic!("pipeline request id missing for sequence {seq_id}"))
    }
}

impl PipelineModelRuntime for InProcessPipelineRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        let request_id = RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
            .expect("sequence-derived request id must be valid");
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        let request_id = self
            .request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .remove(&seq_id);
        if let Some(request_id) = request_id {
            self.worker_loop
                .lock()
                .expect("pipeline worker loop poisoned")
                .release_request(&request_id);
        }
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let request_id = self.request_id_for(seq_id);
        let mut input = PipelineWorkerInput::new(request_id, copy(input_ids));
        if let Some(mask) = mask {
            input = input.with_attention_mask(copy(mask));
        }
        let mut worker_loop = self
            .worker_loop
            .lock()
            .expect("pipeline worker loop poisoned");
        let mut outputs = worker_loop.run_to_completion(vec![input])?;
        let output = outputs
            .pop()
            .ok_or_else(|| anyhow!("pipeline worker loop returned no logits"))?;
        Ok(output.logits)
    }

    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape.get(1).copied().unwrap_or(1);
        let request_ids: Vec<RequestId> =
            seq_ids.iter().map(|&id| self.request_id_for(id)).collect();
        let inputs: Vec<PipelineWorkerInput> = request_ids
            .iter()
            .enumerate()
            .map(|(i, request_id)| {
                PipelineWorkerInput::new(
                    request_id.clone(),
                    slice(input_ids, &[i as i32, 0], &[i as i32 + 1, seq_len]),
                )
            })
            .collect();
        let mut worker_loop = self
            .worker_loop
            .lock()
            .expect("pipeline worker loop poisoned");
        let outputs = worker_loop.run_to_completion(inputs)?;
        let mut logits_by_request: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        for output in outputs {
            logits_by_request.insert(output.request_id.as_str().to_string(), output.logits);
        }

        let mut ordered = request_ids.into_iter();
        let first_request = ordered
            .next()
            .ok_or_else(|| anyhow!("pipeline batched decode received an empty request set"))?;
        let mut merged = logits_by_request
            .remove(first_request.as_str())
            .ok_or_else(|| anyhow!("missing logits for {}", first_request))?;
        for request_id in ordered {
            let logits = logits_by_request
                .remove(request_id.as_str())
                .ok_or_else(|| anyhow!("missing logits for {}", request_id))?;
            merged = concatenate(&merged, &logits, 0);
        }
        Ok(merged)
    }
}

/// Placeholder config for a future remote transport-backed runtime.
#[derive(Debug, Clone)]
pub struct RemotePipelineRuntimeConfig {
    pub stage_peers: Vec<String>,
    pub transport_backend: TransportBackend,
    pub bind_address: String,
    pub stage_timeout: Duration,
}

pub struct RemotePipelineRuntime {
    stage_peers: Vec<String>,
    transport: Arc<dyn Transport>,
    io_runtime: Mutex<tokio::runtime::Runtime>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
    stage_timeout: Duration,
    draining: AtomicBool,
}

impl RemotePipelineRuntime {
    pub fn new(config: RemotePipelineRuntimeConfig) -> Result<Self> {
        if config.stage_peers.is_empty() {
            return Err(anyhow!(
                "remote pipeline runtime requires at least one stage peer"
            ));
        }
        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| anyhow!("failed to build remote pipeline runtime: {err}"))?;
        let transport = io_runtime.block_on(async {
            let transport = bind_transport(config.transport_backend, &config.bind_address).await?;
            transport.connect(&config.stage_peers).await?;
            Ok::<_, anyhow::Error>(transport)
        })?;
        Ok(Self {
            stage_peers: config.stage_peers,
            transport,
            io_runtime: Mutex::new(io_runtime),
            request_ids: Mutex::new(HashMap::new()),
            stage_timeout: config.stage_timeout,
            draining: AtomicBool::new(false),
        })
    }

    fn request_id_for(&self, seq_id: SequenceId) -> RequestId {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .get(&seq_id)
            .cloned()
            .unwrap_or_else(|| panic!("pipeline request id missing for sequence {seq_id}"))
    }

    fn exchange(&self, peer: &str, request: RemoteStageCommand) -> Result<RemoteStageResponse> {
        let transport = Arc::clone(&self.transport);
        let peer = peer.to_string();
        let reply_addr = self.transport.local_addr()?;
        let timeout = self.stage_timeout;
        self.io_runtime
            .lock()
            .expect("remote pipeline runtime poisoned")
            .block_on(async move {
                let payload = serde_json::to_vec(&RemoteStageRequest {
                    reply_addr,
                    command: request,
                })?;
                transport
                    .send(
                        &peer,
                        TransportMessage::Control {
                            operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                            payload: payload.into(),
                        },
                    )
                    .await?;

                loop {
                    let (_from, msg) = tokio::time::timeout(timeout, transport.recv())
                        .await
                        .map_err(|_| {
                            anyhow!("remote stage {} timed out after {:?}", peer, timeout)
                        })??;
                    let TransportMessage::Control { operation, payload } = msg else {
                        continue;
                    };
                    if operation != PIPELINE_STAGE_RESPONSE_OPERATION {
                        continue;
                    }
                    return serde_json::from_slice::<RemoteStageResponse>(&payload)
                        .map_err(Into::into);
                }
            })
    }

    fn broadcast_ack(&self, request: RemoteStageCommand) -> Result<()> {
        for peer in &self.stage_peers {
            match self.exchange(peer, request.clone())? {
                RemoteStageResponse::Ack { .. } => {}
                RemoteStageResponse::Error { message, .. } => {
                    bail!("remote stage {} failed request: {}", peer, message);
                }
                other => {
                    bail!(
                        "unexpected remote stage response from {}: {:?}",
                        peer,
                        other
                    );
                }
            }
        }
        Ok(())
    }

    pub fn probe_stages(&self) -> Result<Vec<RemoteStageResponse>> {
        let mut states = Vec::with_capacity(self.stage_peers.len());
        for peer in &self.stage_peers {
            match self.exchange(peer, RemoteStageCommand::ProbeState)? {
                state @ RemoteStageResponse::State { .. } => states.push(state),
                RemoteStageResponse::Error { message, .. } => {
                    bail!("remote stage {} failed probe: {}", peer, message);
                }
                other => bail!("unexpected remote stage probe response: {:?}", other),
            }
        }
        Ok(states)
    }

    pub fn begin_drain(&self) -> Result<()> {
        self.draining.store(true, Ordering::Release);
        self.broadcast_ack(RemoteStageCommand::BeginDrain)
    }

    pub fn shutdown(&self) -> Result<()> {
        self.draining.store(true, Ordering::Release);
        self.broadcast_ack(RemoteStageCommand::Shutdown)
    }

    fn cleanup_request(&self, request_id: &RequestId) {
        let _ = self.broadcast_ack(RemoteStageCommand::CancelRequest {
            request_id: request_id.as_str().to_string(),
        });
        let _ = self.broadcast_ack(RemoteStageCommand::ReleaseSequence {
            request_id: request_id.as_str().to_string(),
        });
    }
}

impl PipelineModelRuntime for RemotePipelineRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        assert!(
            !self.draining.load(Ordering::Acquire),
            "remote pipeline runtime is draining; refusing to prepare new sequence state"
        );
        let request_id = RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
            .expect("sequence-derived request id must be valid");
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id.clone());
        self.broadcast_ack(RemoteStageCommand::PrepareSequence {
            request_id: request_id.as_str().to_string(),
        })
        .expect("failed to prepare remote pipeline sequence state");
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        let Some(request_id) = self
            .request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .remove(&seq_id)
        else {
            return;
        };
        let _ = self.broadcast_ack(RemoteStageCommand::ReleaseSequence {
            request_id: request_id.as_str().to_string(),
        });
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let request_id = self.request_id_for(seq_id);
        let response = match self.exchange(
            &self.stage_peers[0],
            RemoteStageCommand::RunEntry {
                request_id: request_id.as_str().to_string(),
                micro_batch_id: 0,
                input_ids: serialize_mlx_array(input_ids)?,
                attention_mask: mask.map(serialize_mlx_array).transpose()?,
            },
        ) {
            Ok(response) => response,
            Err(err) => {
                self.cleanup_request(&request_id);
                return Err(err);
            }
        };
        match response {
            RemoteStageResponse::RunResult { logits, .. } => deserialize_wire_tensor(&logits),
            RemoteStageResponse::Error { message, .. } => {
                self.cleanup_request(&request_id);
                Err(anyhow!(message))
            }
            other => {
                self.cleanup_request(&request_id);
                Err(anyhow!("unexpected remote forward response: {:?}", other))
            }
        }
    }

    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        input_ids: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape.get(1).copied().unwrap_or(1);
        let mut ordered = seq_ids.iter().enumerate();
        let Some((first_idx, first_seq_id)) = ordered.next() else {
            return Err(anyhow!(
                "pipeline batched decode received an empty request set"
            ));
        };
        let mut merged = self.forward_sequence(
            *first_seq_id,
            slice(
                input_ids,
                &[first_idx as i32, 0],
                &[first_idx as i32 + 1, seq_len],
            )
            .as_ref()
            .unwrap(),
            None,
        )?;
        for (idx, seq_id) in ordered {
            let logits = self.forward_sequence(
                *seq_id,
                slice(input_ids, &[idx as i32, 0], &[idx as i32 + 1, seq_len])
                    .as_ref()
                    .unwrap(),
                None,
            )?;
            merged = concatenate(&merged, &logits, 0);
        }
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;

    use bytes::Bytes;
    use tokio::sync::oneshot;

    use super::*;
    use crate::distributed::{StageHealth, TcpTransport, TcpTransportConfig};

    #[derive(Clone, Copy)]
    enum StubRunBehavior {
        Silent,
        Error,
    }

    struct StubStageHandle {
        addr: String,
        commands: Arc<Mutex<Vec<String>>>,
        shutdown_tx: Option<oneshot::Sender<()>>,
        join_handle: Option<thread::JoinHandle<Result<()>>>,
    }

    impl StubStageHandle {
        fn spawn(run_behavior: StubRunBehavior) -> Result<Self> {
            let commands = Arc::new(Mutex::new(Vec::new()));
            let commands_for_thread = Arc::clone(&commands);
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
                            ..Default::default()
                        })
                        .await?,
                    );
                    let local_addr = transport.local_addr()?;
                    startup_tx
                        .send(Ok(local_addr.clone()))
                        .map_err(|_| anyhow!("failed to publish stub stage address"))?;

                    let mut lifecycle = super::super::activation_transfer::StageLifecycleState {
                        health: StageHealth::Healthy,
                        ..Default::default()
                    };
                    let mut shutdown_rx = shutdown_rx;
                    loop {
                        tokio::select! {
                            _ = &mut shutdown_rx => break,
                            recv = transport.recv() => {
                                let (_from, msg) = recv?;
                                let TransportMessage::Control { operation, payload } = msg else {
                                    continue;
                                };
                                if operation != PIPELINE_STAGE_COMMAND_OPERATION {
                                    continue;
                                }
                                let request: RemoteStageRequest = serde_json::from_slice(&payload)?;
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

                                let reply = match request.command {
                                    RemoteStageCommand::PrepareSequence { .. } => {
                                        lifecycle.mark_request_started();
                                        Some(RemoteStageResponse::Ack {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        })
                                    }
                                    RemoteStageCommand::ReleaseSequence { .. } => {
                                        lifecycle.mark_request_finished();
                                        Some(RemoteStageResponse::Ack {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        })
                                    }
                                    RemoteStageCommand::BeginDrain => {
                                        lifecycle.draining = true;
                                        Some(RemoteStageResponse::Ack {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        })
                                    }
                                    RemoteStageCommand::CancelRequest { request_id: _ } => {
                                        lifecycle.mark_request_finished();
                                        Some(RemoteStageResponse::Ack {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        })
                                    }
                                    RemoteStageCommand::Shutdown => {
                                        lifecycle.draining = true;
                                        lifecycle.shutdown = true;
                                        lifecycle.health = StageHealth::Failed;
                                        lifecycle.in_flight_requests = 0;
                                        let response = RemoteStageResponse::Ack {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        };
                                        send_stub_response(&transport, &request.reply_addr, response)
                                            .await?;
                                        break;
                                    }
                                    RemoteStageCommand::RunEntry {
                                        request_id,
                                        micro_batch_id: _,
                                        input_ids: _,
                                        attention_mask: _,
                                    } => match run_behavior {
                                        StubRunBehavior::Silent => None,
                                        StubRunBehavior::Error => Some(RemoteStageResponse::Error {
                                            stage_index: 0,
                                            request_id: Some(request_id),
                                            message: "stub stage forced error".to_string(),
                                            state: Some(lifecycle.snapshot()),
                                        }),
                                    },
                                    RemoteStageCommand::ProbeState => {
                                        Some(RemoteStageResponse::State {
                                            stage_index: 0,
                                            state: lifecycle.snapshot(),
                                            pending_entry_replies: 0,
                                        })
                                    }
                                };

                                if let Some(response) = reply {
                                    send_stub_response(&transport, &request.reply_addr, response).await?;
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
                shutdown_tx: Some(shutdown_tx),
                join_handle: Some(join_handle),
            })
        }

        fn addr(&self) -> &str {
            &self.addr
        }

        fn commands(&self) -> Vec<String> {
            self.commands
                .lock()
                .expect("stub command log poisoned")
                .clone()
        }

        fn shutdown(mut self) -> Result<()> {
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

    #[test]
    fn remote_runtime_rejects_empty_peer_set() {
        let err = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: Vec::new(),
            transport_backend: TransportBackend::Tcp,
            bind_address: "127.0.0.1:0".to_string(),
            stage_timeout: Duration::from_secs(30),
        })
        .err()
        .expect("empty stage peer config must fail");
        assert!(err.to_string().contains("at least one stage peer"));
    }

    #[test]
    fn remote_runtime_rejects_non_thunderbolt_bind_address() {
        let err = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: vec!["127.0.0.1:20000".to_string()],
            transport_backend: TransportBackend::Thunderbolt,
            bind_address: "127.0.0.1:0".to_string(),
            stage_timeout: Duration::from_secs(30),
        })
        .err()
        .expect("invalid thunderbolt bind address must fail");
        assert!(
            err.to_string()
                .contains("does not belong to a Thunderbolt Bridge interface")
                || err
                    .to_string()
                    .contains("no Thunderbolt Bridge interface with an IP address was detected")
        );
    }

    #[test]
    fn remote_runtime_begin_drain_marks_stages_and_blocks_new_sequences() {
        let stage = StubStageHandle::spawn(StubRunBehavior::Error).expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: vec![stage.addr().to_string()],
            transport_backend: TransportBackend::Tcp,
            bind_address: "127.0.0.1:0".to_string(),
            stage_timeout: Duration::from_secs(1),
        })
        .expect("create runtime");

        runtime.begin_drain().expect("begin drain");
        let states = runtime.probe_stages().expect("probe stages");
        assert!(matches!(
            &states[0],
            RemoteStageResponse::State { state, .. } if state.draining
        ));

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.prepare_sequence_state(SequenceId::from_raw(7));
        }));
        assert!(panic.is_err(), "prepare should panic while draining");

        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    #[test]
    fn remote_runtime_timeout_cleans_up_sequence_state() {
        let stage = StubStageHandle::spawn(StubRunBehavior::Silent).expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
            stage_peers: vec![stage.addr().to_string()],
            transport_backend: TransportBackend::Tcp,
            bind_address: "127.0.0.1:0".to_string(),
            stage_timeout: Duration::from_millis(100),
        })
        .expect("create runtime");

        let seq_id = SequenceId::from_raw(11);
        runtime.prepare_sequence_state(seq_id);
        let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let err = runtime
            .forward_sequence(seq_id, &input_ids, None)
            .err()
            .expect("silent stage must time out");
        assert!(err.to_string().contains("timed out"));

        let commands = stage.commands();
        assert_eq!(
            commands,
            vec!["prepare", "run_entry", "cancel", "release"],
            "timeout path should cancel and release sequence state"
        );

        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }
}
