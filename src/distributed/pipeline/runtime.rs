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

//! Runtime backends for server-side pipeline execution.
//!
//! Used by: `server_runtime`, future remote PP runtime

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::concatenate;
use mlxcel_core::{MlxArray, UniquePtr, copy, slice};

use crate::distributed::RequestId;
use crate::distributed::cluster_auth::ClusterAuth;
use crate::distributed::handshake_warn::{HandshakeWarnClass, handshake_warn_allowed};
use crate::distributed::peer_identity::PeerMatcher;
use crate::distributed::peer_text::loggable_peer_text;
use crate::distributed::transport_factory::bind_transport;
use crate::distributed::{Transport, TransportBackend, TransportMessage};

use super::remote_service::{
    PIPELINE_STAGE_COMMAND_OPERATION, PIPELINE_STAGE_RESPONSE_OPERATION, RemoteStageCommand,
    RemoteStageRequest, RemoteStageResponse,
};
use super::stage_availability::{
    GateDecision, ProbeClose, STAGE_PROBE_INTERVAL, StageAvailability,
};
use super::wire_tensor::{
    StageWireLimits, deserialize_wire_tensor_with_limits, serialize_mlx_array,
};
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
    /// Prepare the runtime's state for `seq_id`. `Err` fails only this
    /// sequence (the caller releases what it allocated and answers the
    /// request with the error); a remote stage that refuses the
    /// `PrepareSequence`, for example because another identity already
    /// claimed the id, must not take the coordinator down (#2714 item a).
    fn prepare_sequence_state(&self, seq_id: SequenceId) -> Result<()>;
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

    fn request_id_for(&self, seq_id: SequenceId) -> Result<RequestId> {
        request_id_of(&self.request_ids, seq_id)
    }
}

/// The request id `prepare_sequence_state` recorded for `seq_id`. A missing
/// entry is an error, not a panic: the sequence was never prepared, or its
/// prepare failed and was rolled back (#2714 item a).
fn request_id_of(
    request_ids: &Mutex<HashMap<SequenceId, RequestId>>,
    seq_id: SequenceId,
) -> Result<RequestId> {
    request_ids
        .lock()
        .expect("pipeline request map poisoned")
        .get(&seq_id)
        .cloned()
        .ok_or_else(|| anyhow!("pipeline request id missing for sequence {seq_id}"))
}

/// The sequence-derived request id of the in-process runtime, which has no
/// peers; the remote runtime adds a random suffix (#2726).
fn request_id_for_sequence(seq_id: SequenceId) -> RequestId {
    RequestId::from_string(format!("pp-seq-{}", seq_id.as_u64()))
        .expect("sequence-derived request id must be valid")
}

impl PipelineModelRuntime for InProcessPipelineRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) -> Result<()> {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id_for_sequence(seq_id));
        Ok(())
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
        let request_id = self.request_id_for(seq_id)?;
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
        let request_ids: Vec<RequestId> = seq_ids
            .iter()
            .map(|&id| self.request_id_for(id))
            .collect::<Result<_>>()?;
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

/// Longest the coordinator waits for a stage to acknowledge a teardown
/// command (cancel, release). Teardown carries no result anyone waits for:
/// the frame is sent either way, a healthy but busy stage still processes
/// it, and a late ack is dropped by the nonce matcher. Waiting the full
/// `stage_timeout` for each of them let one silent stage cost the single
/// model worker about four stage timeouts per failed request (#2762).
pub(crate) const TEARDOWN_ACK_TIMEOUT: Duration = Duration::from_secs(2);

/// Configuration for the remote transport-backed coordinator runtime.
#[derive(Debug, Clone)]
pub struct RemotePipelineRuntimeConfig {
    pub stage_peers: Vec<String>,
    pub transport_backend: TransportBackend,
    pub bind_address: String,
    pub stage_timeout: Duration,
    /// Cluster authentication shared with every stage; required (#2637).
    pub auth: Arc<ClusterAuth>,
    /// Expected logits vocabulary dimension. `None` skips the vocab check
    /// (rank and batch are always checked); `PipelineServerModel::load_remote`
    /// fills it from the model's `config.json`.
    pub vocab_size: Option<usize>,
}

/// One configured stage: the address commands are sent to and the matcher
/// that recognizes its authenticated identity on replies.
struct StagePeer {
    addr: String,
    matcher: PeerMatcher,
}

/// What an exchange will accept as its reply.
struct ExchangeMatch<'a> {
    /// The stage the command went to; `Ack` and `State` must come from it.
    peer: &'a StagePeer,
    /// The stage a `RunResult` must come from: the last stage of the
    /// pipeline for `RunEntry` (it sends the logits straight to the
    /// coordinator, #2694), otherwise `peer`. A single-stage pipeline has
    /// the same stage in both roles.
    result_peer: &'a StagePeer,
    /// The stages an `Error` may come from: every configured stage for
    /// `RunEntry`, because a middle or final stage can fail the request it
    /// was forwarded and reports that straight to the coordinator (#2712),
    /// otherwise only `peer`. The per-exchange nonce, carried only on that
    /// request's activations, keeps this as tight as the `RunResult` rule.
    error_peers: &'a [StagePeer],
    nonce: u64,
    /// For `RunEntry`: the `(request_id, micro_batch_id)` the `RunResult`
    /// must carry; an `Error` must name the same `request_id`.
    run: Option<(&'a str, u32)>,
}

impl ExchangeMatch<'_> {
    /// `None` when the frame is the awaited reply, else why it is not.
    fn reject_reason(&self, from: &str, response: &RemoteStageResponse) -> Option<&'static str> {
        match response {
            RemoteStageResponse::RunResult { .. } => {
                if !self.result_peer.matcher.matches(from) {
                    return Some("RunResult source is not the final stage");
                }
            }
            RemoteStageResponse::Error { .. } => {
                if !self
                    .error_peers
                    .iter()
                    .any(|peer| peer.matcher.matches(from))
                {
                    return Some("Error source is not a stage this exchange accepts");
                }
            }
            _ => {
                if !self.peer.matcher.matches(from) {
                    return Some("source is not the stage this exchange was sent to");
                }
            }
        }
        if response.nonce() != self.nonce {
            return Some("nonce does not match this exchange");
        }
        if let Some((request_id, micro_batch_id)) = self.run {
            match response {
                RemoteStageResponse::RunResult {
                    request_id: got_request,
                    micro_batch_id: got_mb,
                    ..
                } if got_request != request_id || *got_mb != micro_batch_id => {
                    return Some("RunResult request_id or micro_batch_id does not match");
                }
                RemoteStageResponse::Error {
                    request_id: got_request,
                    ..
                } if got_request.as_deref() != Some(request_id) => {
                    return Some("Error request_id does not match this exchange");
                }
                _ => {}
            }
        }
        None
    }
}

/// Reject logits whose shape cannot be the model's output for `input_ids`:
/// anything but rank 3, batch 1, a sequence length equal to the input's (or
/// 1 for a last-position-only family), and the configured vocabulary size.
fn validate_logits_shape(
    logits: &MlxArray,
    input_seq_len: i32,
    vocab_size: Option<usize>,
) -> Result<()> {
    let shape = mlxcel_core::array_shape(logits);
    let ok = shape.len() == 3
        && shape[0] == 1
        && (shape[1] == input_seq_len || shape[1] == 1)
        && vocab_size.is_none_or(|vocab| i64::from(shape[2]) == vocab as i64);
    if !ok {
        bail!(
            "remote stage returned logits with shape {shape:?}; expected [1, {input_seq_len} or 1, {}]",
            vocab_size.map_or_else(|| "vocab".to_string(), |v| v.to_string())
        );
    }
    Ok(())
}

pub struct RemotePipelineRuntime {
    stage_peers: Vec<StagePeer>,
    transport: Arc<dyn Transport>,
    auth: Arc<ClusterAuth>,
    vocab_size: Option<usize>,
    io_runtime: Mutex<tokio::runtime::Runtime>,
    request_ids: Mutex<HashMap<SequenceId, RequestId>>,
    stage_timeout: Duration,
    /// Per-stage wait for a teardown ack: `stage_timeout` capped at
    /// [`TEARDOWN_ACK_TIMEOUT`].
    teardown_ack_timeout: Duration,
    draining: AtomicBool,
    /// The availability gate (#2782): open while a stage is known silent.
    availability: Mutex<StageAvailability>,
    /// Least time between two probes of an open gate.
    probe_interval: Duration,
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
            let transport = bind_transport(
                config.transport_backend,
                &config.bind_address,
                Some(Arc::clone(&config.auth)),
            )
            .await?;
            // A stage that rejects the cluster handshake fails this call,
            // so a secret mismatch surfaces at startup rather than on the
            // first request.
            transport.connect(&config.stage_peers).await?;
            Ok::<_, anyhow::Error>(transport)
        })?;
        Self::with_io_runtime(config, transport, io_runtime)
    }

    /// Build the runtime over an already bound transport. Tests inject a
    /// mock transport here to exercise reply correlation without TCP.
    #[cfg(test)]
    pub(crate) fn with_transport(
        config: RemotePipelineRuntimeConfig,
        transport: Arc<dyn Transport>,
    ) -> Result<Self> {
        if config.stage_peers.is_empty() {
            return Err(anyhow!(
                "remote pipeline runtime requires at least one stage peer"
            ));
        }
        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| anyhow!("failed to build remote pipeline runtime: {err}"))?;
        Self::with_io_runtime(config, transport, io_runtime)
    }

    fn with_io_runtime(
        config: RemotePipelineRuntimeConfig,
        transport: Arc<dyn Transport>,
        io_runtime: tokio::runtime::Runtime,
    ) -> Result<Self> {
        let stage_peers = io_runtime.block_on(async {
            let mut peers = Vec::with_capacity(config.stage_peers.len());
            for addr in &config.stage_peers {
                peers.push(StagePeer {
                    addr: addr.clone(),
                    matcher: PeerMatcher::resolve(addr).await,
                });
            }
            peers
        });
        Ok(Self {
            stage_peers,
            transport,
            auth: config.auth,
            vocab_size: config.vocab_size,
            io_runtime: Mutex::new(io_runtime),
            request_ids: Mutex::new(HashMap::new()),
            stage_timeout: config.stage_timeout,
            teardown_ack_timeout: config.stage_timeout.min(TEARDOWN_ACK_TIMEOUT),
            draining: AtomicBool::new(false),
            availability: Mutex::new(StageAvailability::default()),
            probe_interval: STAGE_PROBE_INTERVAL,
        })
    }

    /// Lower the teardown ack timeout; tests use it to keep a silent stage's
    /// teardown short next to a longer `stage_timeout`.
    #[cfg(test)]
    pub(crate) fn with_teardown_ack_timeout(mut self, timeout: Duration) -> Self {
        self.teardown_ack_timeout = timeout;
        self
    }

    /// Shorten the probe interval of the availability gate; tests use it
    /// to see the gate close again within a bounded run.
    #[cfg(test)]
    pub(crate) fn with_probe_interval(mut self, interval: Duration) -> Self {
        self.probe_interval = interval;
        self
    }

    fn request_id_for(&self, seq_id: SequenceId) -> Result<RequestId> {
        request_id_of(&self.request_ids, seq_id)
    }

    /// Send one command to `peer` and wait for its reply.
    ///
    /// Correlation (#2637): the reply must come from the authenticated
    /// identity of the stage the command went to (for the `RunResult` of a
    /// `RunEntry`, from the final stage, which sends it directly, #2694; for
    /// the `Error` of a `RunEntry`, from any stage of the pipeline, since
    /// the stage that fails the request reports it directly, #2712), echo
    /// this exchange's fresh nonce, and for `RunEntry` name the same
    /// `request_id` (and `micro_batch_id`). Anything else is dropped and the
    /// wait continues against an absolute deadline, so neither a forged frame
    /// nor a flood of them can produce output or extend the timeout, and a
    /// late reply to an earlier timed-out exchange cannot satisfy this one.
    fn exchange(
        &self,
        peer: &StagePeer,
        request: RemoteStageCommand,
    ) -> Result<RemoteStageResponse> {
        let result = self.exchange_within(peer, request, self.stage_timeout);
        if result.is_err() {
            // Only this full-`stage_timeout` path opens the availability
            // gate (#2782): a short teardown or probe wait that runs out
            // says nothing new about a stage already known silent, and an
            // `Error` reply is an answer.
            self.mark_unavailable(&peer.addr);
        }
        result
    }

    /// Record that `peer` did not answer an exchange within `stage_timeout`
    /// and open the availability gate, logging the closed-to-open
    /// transition once.
    fn mark_unavailable(&self, peer: &str) {
        let transition = self
            .availability
            .lock()
            .expect("stage availability poisoned")
            .mark_unavailable(peer, Instant::now());
        if transition {
            self.log_gate_opened(peer, self.stage_timeout);
        }
    }

    /// The availability gate's closed-to-open line, once per transition:
    /// `peer` did not answer within `waited`.
    fn log_gate_opened(&self, peer: &str, waited: Duration) {
        tracing::warn!(
            "remote stage {peer} did not answer within {waited:?}; refusing new pipeline requests \
             at once until every stage answers a probe (one probe per {:?} while it stays silent)",
            self.probe_interval
        );
    }

    /// The availability gate (#2782 item 1), checked before a prepare and
    /// before a forward's `RunEntry`. Closed: `Ok`. Open and not probed
    /// within `probe_interval`: probe every stage with the short teardown
    /// wait; when all answer `State` the gate closes and the request is
    /// served, otherwise it is refused. Open and probed recently: refused
    /// at once, nothing sent.
    ///
    /// Locking: the `availability` mutex is held only for the state
    /// transitions, never across the probe exchanges. `decide` records the
    /// probe time before the lock is dropped, so a concurrent request that
    /// arrives while this one is probing sees a recent probe and is
    /// refused instead of probing too; the exchanges themselves are
    /// serialized by the `io_runtime` mutex like every other exchange. The
    /// probe result is applied under the lock again with the generation
    /// `decide` handed out, so a timeout that opened a newer outage while
    /// the probe was in flight is not closed by this probe's success, and
    /// two overlapping probes (one that outlasts `probe_interval`) agree:
    /// see `StageAvailability::close_after_probe` and `probe_failed`.
    fn ensure_stages_available(&self) -> Result<()> {
        let decision = self
            .availability
            .lock()
            .expect("stage availability poisoned")
            .decide(Instant::now(), self.probe_interval);
        let (generation, stage) = match decision {
            GateDecision::Pass => return Ok(()),
            GateDecision::Refuse(err) => return Err(err.into()),
            GateDecision::Probe { generation, stage } => (generation, stage),
        };
        let mut silent: Option<String> = None;
        for peer in &self.stage_peers {
            match self.exchange_within(
                peer,
                RemoteStageCommand::ProbeState,
                self.teardown_ack_timeout,
            ) {
                Ok(RemoteStageResponse::State { .. }) => {}
                Ok(RemoteStageResponse::Error { message, .. }) => {
                    tracing::debug!(
                        "remote stage {} answered the availability probe with an error: {}",
                        peer.addr,
                        loggable_peer_text(&message)
                    );
                    silent = Some(peer.addr.clone());
                    break;
                }
                Ok(other) => {
                    tracing::debug!(
                        "remote stage {} answered the availability probe with an unexpected \
                         response: {other:?}",
                        peer.addr
                    );
                    silent = Some(peer.addr.clone());
                    break;
                }
                Err(err) => {
                    tracing::debug!(
                        "remote stage {} did not answer the availability probe: {err:#}",
                        peer.addr
                    );
                    silent = Some(peer.addr.clone());
                    break;
                }
            }
        }
        // The result is applied under the lock; the lines are logged after
        // it is dropped.
        let now = Instant::now();
        match silent {
            None => {
                let outcome = {
                    let mut availability = self
                        .availability
                        .lock()
                        .expect("stage availability poisoned");
                    match availability.close_after_probe(generation, now) {
                        ProbeClose::Superseded => Err(availability.refusal(now)),
                        other => Ok(other),
                    }
                };
                match outcome {
                    Ok(ProbeClose::Closed(outage)) => {
                        tracing::info!(
                            "every remote stage answered the availability probe; serving \
                             pipeline requests again after {outage:?} without {stage}"
                        );
                        Ok(())
                    }
                    // An overlapping probe for the same outage closed it.
                    Ok(_) => Ok(()),
                    // A newer timeout replaced the outage this probe was for.
                    Err(refusal) => Err(refusal.into()),
                }
            }
            Some(peer) => {
                let failure = self
                    .availability
                    .lock()
                    .expect("stage availability poisoned")
                    .probe_failed(generation, &peer, now);
                if failure.reopened {
                    self.log_gate_opened(&peer, self.teardown_ack_timeout);
                }
                Err(failure.refusal.into())
            }
        }
    }

    /// [`Self::exchange`] with an explicit wait for the reply. Teardown
    /// commands pass the shorter `teardown_ack_timeout` (#2762); everything
    /// else waits `stage_timeout`.
    fn exchange_within(
        &self,
        peer: &StagePeer,
        request: RemoteStageCommand,
        timeout: Duration,
    ) -> Result<RemoteStageResponse> {
        let transport = Arc::clone(&self.transport);
        let reply_addr = self.transport.local_addr()?;
        let nonce = self.auth.random_u64();
        let run_key = match &request {
            RemoteStageCommand::RunEntry {
                request_id,
                micro_batch_id,
                ..
            } => Some((request_id.clone(), *micro_batch_id)),
            _ => None,
        };
        let result_peer = if run_key.is_some() {
            self.stage_peers
                .last()
                .expect("the runtime refuses an empty stage peer set")
        } else {
            peer
        };
        let error_peers: &[StagePeer] = if run_key.is_some() {
            &self.stage_peers[..]
        } else {
            std::slice::from_ref(peer)
        };
        self.io_runtime
            .lock()
            .expect("remote pipeline runtime poisoned")
            .block_on(async move {
                let payload = serde_json::to_vec(&RemoteStageRequest {
                    reply_addr,
                    nonce,
                    command: request,
                })?;
                transport
                    .send(
                        &peer.addr,
                        TransportMessage::Control {
                            operation: PIPELINE_STAGE_COMMAND_OPERATION.to_string(),
                            payload: payload.into(),
                        },
                    )
                    .await?;

                let matcher = ExchangeMatch {
                    peer,
                    result_peer,
                    error_peers,
                    nonce,
                    run: run_key.as_ref().map(|(id, mb)| (id.as_str(), *mb)),
                };
                let deadline = tokio::time::Instant::now() + timeout;
                let mut dropped: u32 = 0;
                loop {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        bail!("remote stage {} timed out after {:?}", peer.addr, timeout);
                    }
                    let (from, msg) = tokio::time::timeout(remaining, transport.recv())
                        .await
                        .map_err(|_| {
                            anyhow!("remote stage {} timed out after {:?}", peer.addr, timeout)
                        })??;
                    let TransportMessage::Control { operation, payload } = msg else {
                        continue;
                    };
                    if operation != PIPELINE_STAGE_RESPONSE_OPERATION {
                        continue;
                    }
                    let response = match serde_json::from_slice::<RemoteStageResponse>(&payload) {
                        Ok(response) => response,
                        Err(err) => {
                            let reason = malformed_reply_reason(&err);
                            Self::note_dropped(&mut dropped, &from, &reason);
                            continue;
                        }
                    };
                    if let Some(reason) = matcher.reject_reason(&from, &response) {
                        Self::note_dropped(&mut dropped, &from, reason);
                        continue;
                    }
                    return Ok(response);
                }
            })
    }

    /// Log a dropped reply frame: the first few at warn, the rest at debug
    /// so a flood cannot drown the log.
    fn note_dropped(dropped: &mut u32, from: &str, reason: &str) {
        *dropped = dropped.saturating_add(1);
        if *dropped <= 3 {
            tracing::warn!(
                "remote pipeline coordinator dropped a stage response from {from}: {reason}"
            );
        } else {
            tracing::debug!(
                "remote pipeline coordinator dropped a stage response from {from}: {reason} \
                 ({dropped} dropped in this exchange)"
            );
        }
    }

    fn broadcast_ack(&self, request: RemoteStageCommand) -> Result<()> {
        for peer in &self.stage_peers {
            match self.exchange(peer, request.clone())? {
                RemoteStageResponse::Ack { .. } => {}
                RemoteStageResponse::Error { message, .. } => {
                    bail!(
                        "remote stage {} failed request: {}",
                        peer.addr,
                        loggable_peer_text(&message)
                    );
                }
                other => {
                    bail!(
                        "unexpected remote stage response from {}: {:?}",
                        peer.addr,
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
                    bail!(
                        "remote stage {} failed probe: {}",
                        peer.addr,
                        loggable_peer_text(&message)
                    );
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

    /// Send `request` to every stage, carrying on past a stage that fails
    /// it. A teardown command (cancel, release) that stopped at the first
    /// stage to time out would never reach the stages after it, and those
    /// would keep the sequence, its KV caches and a slot of their
    /// prepared-sequence cap, with nothing left to release them (#2727).
    /// Failures are logged, never returned. Each stage gets at most
    /// `teardown_ack_timeout` to acknowledge (#2762): nothing waits for the
    /// result, the frame is sent either way, and a late ack is dropped by
    /// the nonce matcher of whatever exchange is running then. While the
    /// availability gate is open, the stage it names gets no wait at all
    /// (#2782): the frame is still written, and a stage known silent would
    /// only cost the model worker the wait.
    fn broadcast_best_effort(&self, request: &RemoteStageCommand, what: &str) {
        let silent_peer = self
            .availability
            .lock()
            .expect("stage availability poisoned")
            .failed_peer()
            .map(str::to_string);
        for peer in &self.stage_peers {
            let timeout = if silent_peer.as_deref() == Some(peer.addr.as_str()) {
                Duration::ZERO
            } else {
                self.teardown_ack_timeout
            };
            match self.exchange_within(peer, request.clone(), timeout) {
                Ok(RemoteStageResponse::Ack { .. }) => {}
                // A stage that never prepared the sequence, or refused it,
                // may answer with an error; there is nothing to undo there.
                Ok(RemoteStageResponse::Error { message, .. }) => tracing::debug!(
                    "remote stage {} answered {what} with an error: {}",
                    peer.addr,
                    loggable_peer_text(&message)
                ),
                Ok(_) => tracing::debug!(
                    "remote stage {} answered {what} with an unexpected response",
                    peer.addr
                ),
                Err(err) if timeout.is_zero() => tracing::debug!(
                    "remote stage {} is silent; {what} was sent without waiting for its \
                     acknowledgement: {err:#}",
                    peer.addr
                ),
                Err(err) => Self::log_teardown_miss(&peer.addr, what, timeout, &err),
            }
        }
    }

    /// A stage missed the teardown ack wait. A healthy stage still finishing
    /// a long forward does this too and applies the frame when it gets to
    /// it, so the line draws from the `teardown-ack` warning budget
    /// (#2782 item 3) and goes to debug once the budget is spent.
    fn log_teardown_miss(addr: &str, what: &str, timeout: Duration, err: &anyhow::Error) {
        if handshake_warn_allowed(HandshakeWarnClass::TeardownAck) {
            tracing::warn!(
                "remote stage {addr} did not acknowledge {what} within {timeout:?}; the frame was \
                 sent and a slow stage still applies it: {err:#}"
            );
        } else {
            tracing::debug!(
                "remote stage {addr} did not acknowledge {what} within {timeout:?}; the frame was \
                 sent and a slow stage still applies it: {err:#}"
            );
        }
    }

    /// Discard the coordinator's state for `seq_id` after its forward
    /// failed, then cancel and release the sequence on every stage. The
    /// `request_ids` entry goes first so the scheduler's later close of the
    /// failed sequence is a no-op: every stage sees one cancel and one
    /// release, and a silent stage costs one teardown ack wait per command
    /// instead of a full `stage_timeout` for each of three (#2762). Every
    /// failure after the `RunEntry` exchange starts (transport error, stage
    /// error, undecodable or mis-shaped logits, unexpected reply) ends
    /// here, as does a refusal by the availability gate (#2782).
    /// The two `serialize_mlx_array` exits before the exchange return `Err`
    /// without calling this: nothing reached a stage, so the mapping stays
    /// and the scheduler's later close releases the sequence exactly once.
    fn fail_sequence(&self, seq_id: SequenceId, request_id: &RequestId) {
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .remove(&seq_id);
        self.broadcast_best_effort(
            &RemoteStageCommand::CancelRequest {
                request_id: request_id.as_str().to_string(),
            },
            "cancel",
        );
        self.release_on_every_stage(request_id);
    }

    fn release_on_every_stage(&self, request_id: &RequestId) {
        self.broadcast_best_effort(
            &RemoteStageCommand::ReleaseSequence {
                request_id: request_id.as_str().to_string(),
            },
            "release",
        );
    }
}

impl PipelineModelRuntime for RemotePipelineRuntime {
    /// Record the request id and broadcast `PrepareSequence` to every stage.
    /// A stage that answers with an `Error` (it is draining, or another
    /// identity prepared the same id, #2714) or times out fails only this
    /// sequence: the id mapping is removed, the stages that did prepare it
    /// are told to release it (best effort; a stage that never prepared it,
    /// or refused it, answers `Error` and is ignored), and the error is
    /// returned to the caller instead of aborting the model worker.
    fn prepare_sequence_state(&self, seq_id: SequenceId) -> Result<()> {
        if self.draining.load(Ordering::Acquire) {
            bail!("remote pipeline runtime is draining; refusing to prepare new sequence state");
        }
        self.ensure_stages_available()?;
        // A sequential `pp-seq-{N}` is predictable: another holder of the
        // secret could prepare the next one first and make this prepare fail
        // (#2726). A CSPRNG suffix makes the id unguessable; every later
        // command reads it back from `request_ids`.
        let request_id = RequestId::from_string(format!(
            "pp-seq-{}-{:016x}",
            seq_id.as_u64(),
            self.auth.random_u64()
        ))
        .ok_or_else(|| anyhow!("invalid remote request id for sequence {seq_id}"))?;
        self.request_ids
            .lock()
            .expect("pipeline request map poisoned")
            .insert(seq_id, request_id.clone());
        let prepared = self.broadcast_ack(RemoteStageCommand::PrepareSequence {
            request_id: request_id.as_str().to_string(),
        });
        if let Err(err) = prepared {
            self.request_ids
                .lock()
                .expect("pipeline request map poisoned")
                .remove(&seq_id);
            self.release_on_every_stage(&request_id);
            return Err(err.context(format!(
                "failed to prepare remote pipeline sequence state for {request_id}"
            )));
        }
        Ok(())
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
        self.release_on_every_stage(&request_id);
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        input_ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let request_id = self.request_id_for(seq_id)?;
        // A sequence prepared before a stage went silent is failed here at
        // once instead of waiting out the `RunEntry` (#2782); like every
        // other post-mapping failure it goes through `fail_sequence`, so
        // `Err` still means the coordinator's state was discarded (#2753).
        if let Err(err) = self.ensure_stages_available() {
            self.fail_sequence(seq_id, &request_id);
            return Err(err);
        }
        let input_seq_len = mlxcel_core::array_shape(input_ids)
            .get(1)
            .copied()
            .unwrap_or(1);
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
                self.fail_sequence(seq_id, &request_id);
                return Err(err);
            }
        };
        match response {
            RemoteStageResponse::RunResult { logits, .. } => {
                // Bound the frame by the expected logits shape before decoding
                // it (#2716); `validate_logits_shape` keeps the exact
                // "input length or 1" rule.
                let limits = StageWireLimits {
                    activation_width: None,
                    vocab_size: self.vocab_size.map(|v| v as u64),
                }
                .logits(u64::try_from(input_seq_len.max(1)).unwrap_or(1));
                let logits = match deserialize_wire_tensor_with_limits(&logits, &limits) {
                    Ok(logits) => logits,
                    Err(err) => {
                        self.fail_sequence(seq_id, &request_id);
                        return Err(err);
                    }
                };
                if let Err(err) =
                    validate_logits_shape(logits.as_ref().unwrap(), input_seq_len, self.vocab_size)
                {
                    self.fail_sequence(seq_id, &request_id);
                    return Err(err);
                }
                Ok(logits)
            }
            RemoteStageResponse::Error {
                stage_index,
                message,
                ..
            } => {
                self.fail_sequence(seq_id, &request_id);
                Err(anyhow!(
                    "remote stage {stage_index} failed the forward: {}",
                    loggable_peer_text(&message)
                ))
            }
            other => {
                self.fail_sequence(seq_id, &request_id);
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
        // Every row passed `validate_logits_shape`, which allows a sequence
        // axis of the input length or 1 and, without a configured vocabulary
        // size, any width. Rows that disagree cannot be concatenated, and
        // MLX throws that at graph construction, through an FFI call that
        // cannot report it (the process aborts). Refuse them here instead.
        let row_shape = mlxcel_core::array_shape(&merged);
        for (idx, seq_id) in ordered {
            let logits = self.forward_sequence(
                *seq_id,
                slice(input_ids, &[idx as i32, 0], &[idx as i32 + 1, seq_len])
                    .as_ref()
                    .unwrap(),
                None,
            )?;
            let shape = mlxcel_core::array_shape(&logits);
            if shape != row_shape {
                bail!(
                    "remote stage returned logits with shape {shape:?} for sequence {seq_id}; \
                     the batch's first row has {row_shape:?}"
                );
            }
            merged = concatenate(&merged, &logits, 0);
        }
        Ok(merged)
    }
}

/// Why a stage reply that did not decode was dropped, made safe to log:
/// serde_json quotes an unknown variant or field name from the payload
/// verbatim (#2782 item 2).
fn malformed_reply_reason(err: &serde_json::Error) -> String {
    format!("malformed: {}", loggable_peer_text(&err.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::stub_stage::{STUB_FORCED_ERROR, StubRunBehavior, StubStageHandle};
    use super::*;
    use crate::distributed::cluster_auth::ClusterSecret;

    fn test_auth() -> Arc<ClusterAuth> {
        Arc::new(ClusterAuth::new(&ClusterSecret::random()))
    }

    fn runtime_config(
        stage_peers: Vec<String>,
        transport_backend: TransportBackend,
        stage_timeout: Duration,
        auth: Arc<ClusterAuth>,
    ) -> RemotePipelineRuntimeConfig {
        RemotePipelineRuntimeConfig {
            stage_peers,
            transport_backend,
            bind_address: "127.0.0.1:0".to_string(),
            stage_timeout,
            auth,
            vocab_size: None,
        }
    }

    #[test]
    fn remote_runtime_rejects_empty_peer_set() {
        let err = RemotePipelineRuntime::new(runtime_config(
            Vec::new(),
            TransportBackend::Tcp,
            Duration::from_secs(30),
            test_auth(),
        ))
        .err()
        .expect("empty stage peer config must fail");
        assert!(err.to_string().contains("at least one stage peer"));
    }

    #[test]
    fn remote_runtime_rejects_non_thunderbolt_bind_address() {
        let err = RemotePipelineRuntime::new(runtime_config(
            vec!["127.0.0.1:20000".to_string()],
            TransportBackend::Thunderbolt,
            Duration::from_secs(30),
            test_auth(),
        ))
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
        let auth = test_auth();
        let stage = StubStageHandle::spawn(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            Arc::clone(&auth),
        )
        .expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_secs(1),
            auth,
        ))
        .expect("create runtime");

        runtime.begin_drain().expect("begin drain");
        let states = runtime.probe_stages().expect("probe stages");
        assert!(matches!(
            &states[0],
            RemoteStageResponse::State { state, .. } if state.draining
        ));

        let err = runtime
            .prepare_sequence_state(SequenceId::from_raw(7))
            .expect_err("prepare must fail while draining");
        assert!(err.to_string().contains("draining"), "{err:#}");

        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    /// A stage that refuses `PrepareSequence` (another identity claimed the
    /// id, #2714 item 2) fails only that sequence: `prepare_sequence_state`
    /// returns the stage's error instead of panicking on the model-worker
    /// thread, the id mapping is rolled back so a later forward is an error
    /// rather than a panic, and the stage is told to release the id.
    #[test]
    fn remote_runtime_prepare_failure_is_an_error_not_a_panic() {
        let auth = test_auth();
        let stage = StubStageHandle::spawn_with(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            true,
            Arc::clone(&auth),
        )
        .expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_secs(1),
            auth,
        ))
        .expect("create runtime");
        let seq_id = SequenceId::from_raw(21);
        let err = runtime
            .prepare_sequence_state(seq_id)
            .expect_err("a refused PrepareSequence must be an error");
        let text = format!("{err:#}");
        assert!(
            text.contains("prepared by another coordinator identity"),
            "{text}"
        );
        assert!(
            runtime.request_id_for(seq_id).is_err(),
            "the request id mapping must be rolled back"
        );
        let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let forward_err = runtime
            .forward_sequence(seq_id, &input_ids, None)
            .err()
            .expect("a forward for an unprepared sequence must fail, not panic");
        assert!(
            forward_err.to_string().contains("request id missing"),
            "{forward_err:#}"
        );
        assert_eq!(
            stage.commands(),
            vec!["prepare", "release"],
            "the failed prepare must be released on the stage"
        );
        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    /// `pp-seq-{seq}-` followed by 16 lowercase hex digits.
    fn is_random_sequence_id(id: &str, seq: u64) -> bool {
        id.strip_prefix(&format!("pp-seq-{seq}-"))
            .is_some_and(|hex| {
                hex.len() == 16
                    && hex
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
    }

    /// #2726 item 2: the remote request id carries a CSPRNG suffix, so a
    /// second holder of the secret cannot prepare the id the coordinator
    /// will use next. Preparing the same `SequenceId` twice (with a release
    /// in between) sends two different ids, both of the form
    /// `pp-seq-21-<16 hex>`, and later commands use the recorded one.
    #[test]
    fn remote_runtime_request_ids_are_unpredictable() {
        let auth = test_auth();
        let stage = StubStageHandle::spawn(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            Arc::clone(&auth),
        )
        .expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_secs(1),
            auth,
        ))
        .expect("create runtime");
        let seq_id = SequenceId::from_raw(21);
        runtime
            .prepare_sequence_state(seq_id)
            .expect("first prepare");
        let first = runtime.request_id_for(seq_id).expect("first request id");
        runtime.release_sequence_state_by_id(seq_id);
        runtime
            .prepare_sequence_state(seq_id)
            .expect("second prepare");
        let second = runtime.request_id_for(seq_id).expect("second request id");

        let prepared = stage.prepared_ids();
        assert_eq!(prepared.len(), 2, "{prepared:?}");
        for id in &prepared {
            assert!(
                is_random_sequence_id(id, 21),
                "expected pp-seq-21-<16 hex digits>, got {id}"
            );
        }
        assert_ne!(prepared[0], prepared[1], "the two ids must differ");
        assert_eq!(prepared[0], first.as_str());
        assert_eq!(prepared[1], second.as_str());
        assert_eq!(stage.commands(), vec!["prepare", "release", "prepare"]);
        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    #[test]
    fn remote_runtime_timeout_cleans_up_sequence_state() {
        let auth = test_auth();
        let stage = StubStageHandle::spawn(StubRunBehavior::Silent, Arc::clone(&auth))
            .expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_millis(100),
            auth,
        ))
        .expect("create runtime");

        let seq_id = SequenceId::from_raw(11);
        runtime.prepare_sequence_state(seq_id).expect("prepare");
        let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let err = runtime
            .forward_sequence(seq_id, &input_ids, None)
            .err()
            .expect("silent stage must time out");
        assert!(err.to_string().contains("timed out"));

        // The gate is open after the timeout, so the teardown goes out
        // without waiting for this stage's acknowledgement (#2782); the
        // stub may still be reading the frames.
        let commands = stage.wait_for_commands(4, Duration::from_secs(2));
        assert_eq!(
            commands,
            vec!["prepare", "run_entry", "cancel", "release"],
            "timeout path should cancel and release sequence state"
        );

        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    /// #2727: a stage that does not answer the teardown (it stalled, or the
    /// link to it drops frames) must not keep the cancel and release from
    /// the stages after it. Before, the broadcast stopped at the first
    /// stage that timed out, so every later stage kept the failed sequence
    /// and its KV caches with nothing left to release them, one more per
    /// failed request until their prepared-sequence cap refused every new
    /// request.
    #[test]
    fn remote_runtime_teardown_reaches_every_stage_past_a_silent_one() {
        let auth = test_auth();
        // The first stage fails the forward and then ignores the teardown;
        // the last stage is healthy.
        let first = StubStageHandle::spawn_silent_on_teardown(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            Arc::clone(&auth),
        )
        .expect("spawn first stub stage");
        let last = StubStageHandle::spawn(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            Arc::clone(&auth),
        )
        .expect("spawn last stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![first.addr().to_string(), last.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_millis(300),
            auth,
        ))
        .expect("create runtime");

        let seq_id = SequenceId::from_raw(5);
        runtime.prepare_sequence_state(seq_id).expect("prepare");
        let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        runtime
            .forward_sequence(seq_id, &input_ids, None)
            .err()
            .expect("the first stage fails the forward");
        // The scheduler's close of the failed sequence is a no-op: the
        // failed forward already discarded the sequence's state (#2762).
        runtime.release_sequence_state_by_id(seq_id);

        assert_eq!(
            first.commands(),
            vec!["prepare", "run_entry", "cancel", "release"],
            "a failed forward releases the sequence exactly once"
        );
        assert_eq!(
            last.commands(),
            vec!["prepare", "cancel", "release"],
            "the teardown must reach the stage after the silent one, once"
        );
        runtime.shutdown().expect("shutdown runtime");
        first.shutdown().expect("shutdown first stub stage");
        last.shutdown().expect("shutdown last stub stage");
    }

    /// #2762 item 1: one stage that stays silent on `RunEntry` and on the
    /// teardown costs the model worker one `stage_timeout` (to detect the
    /// failure) plus one teardown ack timeout per teardown command, not a
    /// full `stage_timeout` for each of cancel, release, and the scheduler's
    /// later release. With `stage_timeout` 1 s and a 100 ms teardown ack
    /// timeout the failed forward and the close must return within 1.7 s;
    /// before the fix they took about 4 s.
    #[test]
    fn silent_stage_failure_is_bounded_by_one_stage_timeout() {
        let auth = test_auth();
        let stage =
            StubStageHandle::spawn_silent_on_teardown(StubRunBehavior::Silent, Arc::clone(&auth))
                .expect("spawn stub stage");
        let stage_timeout = Duration::from_secs(1);
        let teardown_ack_timeout = Duration::from_millis(100);
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            stage_timeout,
            auth,
        ))
        .expect("create runtime")
        .with_teardown_ack_timeout(teardown_ack_timeout);

        let seq_id = SequenceId::from_raw(9);
        runtime.prepare_sequence_state(seq_id).expect("prepare");
        let input_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let started = std::time::Instant::now();
        let err = runtime
            .forward_sequence(seq_id, &input_ids, None)
            .err()
            .expect("the silent stage must time out");
        assert!(err.to_string().contains("timed out"), "{err:#}");
        // The scheduler's close of the failed sequence.
        runtime.release_sequence_state_by_id(seq_id);
        let elapsed = started.elapsed();

        let bound = stage_timeout + 2 * teardown_ack_timeout + Duration::from_millis(500);
        assert!(
            elapsed <= bound,
            "a silent stage cost {elapsed:?}; the failure path must return within {bound:?}"
        );
        assert_eq!(
            stage.wait_for_commands(4, Duration::from_secs(2)),
            vec!["prepare", "run_entry", "cancel", "release"],
            "the teardown goes out once, without waiting on the silent stage"
        );
        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    /// #2727: rows of a batched decode whose logits disagree on their shape
    /// (possible without a configured vocabulary size) fail the step with
    /// an error. Concatenating them made MLX throw through an FFI call that
    /// cannot report it, which aborted the coordinator.
    #[test]
    fn remote_runtime_refuses_a_batch_whose_rows_disagree_on_logits_shape() {
        let auth = test_auth();
        let stage = StubStageHandle::spawn(
            StubRunBehavior::MixedVocab {
                vocabs: [8, 9],
                token: 3,
            },
            Arc::clone(&auth),
        )
        .expect("spawn stub stage");
        let runtime = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_secs(5),
            auth,
        ))
        .expect("create runtime");
        let seq_ids = [SequenceId::from_raw(1), SequenceId::from_raw(2)];
        for &seq_id in &seq_ids {
            runtime.prepare_sequence_state(seq_id).expect("prepare");
        }
        let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
        let err = runtime
            .forward_batched(&seq_ids, &input_ids)
            .err()
            .expect("rows with different logits widths must fail the batch");
        assert!(format!("{err:#}").contains("logits with shape"), "{err:#}");
        for &seq_id in &seq_ids {
            runtime.release_sequence_state_by_id(seq_id);
        }
        runtime.shutdown().expect("shutdown runtime");
        stage.shutdown().expect("shutdown stub stage");
    }

    #[test]
    fn remote_runtime_refuses_to_start_against_stage_with_other_secret() {
        let stage = StubStageHandle::spawn(
            StubRunBehavior::Error {
                message: STUB_FORCED_ERROR,
            },
            test_auth(),
        )
        .expect("spawn stub stage");
        let err = RemotePipelineRuntime::new(runtime_config(
            vec![stage.addr().to_string()],
            TransportBackend::Tcp,
            Duration::from_secs(1),
            test_auth(),
        ))
        .err()
        .expect("coordinator with a different secret must fail at startup");
        let text = format!("{err:#}");
        assert!(text.contains("cluster authentication"), "{text}");
        assert!(text.contains("secret mismatch"), "{text}");
        stage.shutdown().expect("shutdown stub stage");
    }
    #[test]
    fn malformed_reply_reason_escapes_the_quoted_variant() {
        let err =
            serde_json::from_str::<RemoteStageResponse>("{\"Evil\\nINFO forged\\u001b[31m\": {}}")
                .expect_err("unknown variant");
        assert!(
            err.to_string().contains('\n'),
            "serde_json echoes it raw: {err}"
        );
        let reason = malformed_reply_reason(&err);
        assert!(!reason.contains('\n'), "{reason}");
        assert!(!reason.contains('\u{1b}'), "{reason}");
        assert!(reason.contains("Evil\\nINFO forged"), "{reason}");
    }
}

#[cfg(test)]
#[path = "runtime_correlation_tests.rs"]
mod correlation_tests;

#[cfg(test)]
#[path = "runtime_availability_tests.rs"]
mod availability_tests;
