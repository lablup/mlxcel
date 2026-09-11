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

//! Activation transfer between pipeline parallelism stages.
//!
//! Defines [`ActivationMessage`] and async channels ([`ActivationSender`] /
//! [`ActivationReceiver`]) with back-pressure for forwarding hidden-state
//! tensors between adjacent stages.
//!
//! - **Forward path**: layer output + mask + position IDs to the next stage.
//! - **Reverse path**: logits/tokens from last stage back to first stage.
//!
//! Used by: pipeline execution loop, distributed scheduler

use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::distributed::request_tracker::RequestId;
use crate::distributed::tensor_protocol::{TensorDtype, TensorKind};
use crate::distributed::tensor_serialize::{
    DeserializedTensor, SerializeOptions, deserialize_tensor, serialize_tensor,
};
use crate::distributed::{Transport, TransportMessage};

use super::serving::StageHealth;

/// Payload transferred between adjacent pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationMessage {
    /// Identifies the original inference request.
    pub request_id: RequestId,
    /// Micro-batch index within a pipeline schedule (enables 1F1B overlap).
    pub micro_batch_id: u32,
    /// Index of the stage that produced this activation.
    pub stage_index: u32,
    /// Total number of stages in the pipeline.
    pub num_stages: u32,
    /// Serialized hidden-state tensor (wire-format bytes from tensor_serialize).
    pub tensor_data: Vec<u8>,
    /// Serialized attention mask tensor (wire-format bytes), if present.
    pub attention_mask: Option<Vec<u8>>,
    /// Serialized position IDs tensor (wire-format bytes), if present.
    pub position_ids: Option<Vec<u8>>,
    /// True when this message travels the reverse path (last stage -> first).
    pub is_reverse_path: bool,
    /// Sequence length for the current micro-batch.
    pub seq_len: u32,
    /// Monotonic timestamp (nanos since an arbitrary epoch) for latency tracking.
    pub timestamp_ns: u64,
}

impl ActivationMessage {
    /// Create a forward-path activation message.
    pub fn forward(
        request_id: RequestId,
        micro_batch_id: u32,
        stage_index: u32,
        num_stages: u32,
        tensor_data: Vec<u8>,
        attention_mask: Option<Vec<u8>>,
        position_ids: Option<Vec<u8>>,
        seq_len: u32,
    ) -> Self {
        Self {
            request_id,
            micro_batch_id,
            stage_index,
            num_stages,
            tensor_data,
            attention_mask,
            position_ids,
            is_reverse_path: false,
            seq_len,
            timestamp_ns: timestamp_nanos(),
        }
    }

    /// Create a reverse-path message (logits/tokens from last stage).
    pub fn reverse(
        request_id: RequestId,
        micro_batch_id: u32,
        stage_index: u32,
        num_stages: u32,
        tensor_data: Vec<u8>,
        seq_len: u32,
    ) -> Self {
        Self {
            request_id,
            micro_batch_id,
            stage_index,
            num_stages,
            tensor_data,
            attention_mask: None,
            position_ids: None,
            is_reverse_path: true,
            seq_len,
            timestamp_ns: timestamp_nanos(),
        }
    }

    /// Total byte size of all tensor payloads in this message.
    pub fn payload_size(&self) -> usize {
        self.tensor_data.len()
            + self.attention_mask.as_ref().map_or(0, |v| v.len())
            + self.position_ids.as_ref().map_or(0, |v| v.len())
    }

    /// Serialize an activation tensor into wire-format bytes suitable for
    /// the `tensor_data` field.
    pub fn serialize_activation(dtype: TensorDtype, shape: &[u64], data: &[u8]) -> Result<Vec<u8>> {
        let opts = SerializeOptions {
            kind: TensorKind::Activation,
            ..Default::default()
        };
        serialize_tensor(dtype, shape, data, &opts)
    }

    /// Deserialize the `tensor_data` field back into a tensor.
    pub fn deserialize_activation(wire_bytes: &[u8]) -> Result<DeserializedTensor> {
        let (tensor, _consumed) = deserialize_tensor(wire_bytes)?;
        Ok(tensor)
    }
}

impl fmt::Display for ActivationMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.is_reverse_path {
            "reverse"
        } else {
            "forward"
        };
        write!(
            f,
            "Activation[{direction} stage={} mb={} seq_len={} payload={}B]",
            self.stage_index,
            self.micro_batch_id,
            self.seq_len,
            self.payload_size(),
        )
    }
}

/// Configuration for activation transfer channels.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Maximum number of messages that can be buffered before the sender
    /// blocks (providing back-pressure). Default: 4.
    pub capacity: usize,
    /// Timeout for send operations. `None` means wait indefinitely.
    pub send_timeout: Option<Duration>,
    /// Timeout for receive operations. `None` means wait indefinitely.
    pub recv_timeout: Option<Duration>,
}

pub(crate) const PIPELINE_ACTIVATION_OPERATION: &str = "pipeline_activation";

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            capacity: 4,
            send_timeout: None,
            recv_timeout: None,
        }
    }
}

/// Mutable lifecycle state for a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageLifecycleState {
    pub health: StageHealth,
    pub draining: bool,
    pub shutdown: bool,
    pub in_flight_requests: usize,
    pub last_barrier_id: Option<u64>,
}

impl Default for StageLifecycleState {
    fn default() -> Self {
        Self {
            health: StageHealth::Unknown,
            draining: false,
            shutdown: false,
            in_flight_requests: 0,
            last_barrier_id: None,
        }
    }
}

impl StageLifecycleState {
    #[must_use]
    pub fn snapshot(&self) -> StageLifecycleSnapshot {
        StageLifecycleSnapshot {
            health: self.health,
            draining: self.draining,
            shutdown: self.shutdown,
            in_flight_requests: self.in_flight_requests,
            last_barrier_id: self.last_barrier_id,
        }
    }

    pub fn mark_request_started(&mut self) {
        self.in_flight_requests += 1;
    }

    pub fn mark_request_finished(&mut self) {
        self.in_flight_requests = self.in_flight_requests.saturating_sub(1);
    }
}

/// Serializable view of stage lifecycle state returned by RPC probes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageLifecycleSnapshot {
    pub health: StageHealth,
    pub draining: bool,
    pub shutdown: bool,
    pub in_flight_requests: usize,
    pub last_barrier_id: Option<u64>,
}

/// Lifecycle RPC request for remote pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StageLifecycleRequest {
    ProbeHealth,
    BeginDrain,
    Barrier { barrier_id: u64 },
    CancelRequest { request_id: String },
    Shutdown,
}

/// Lifecycle RPC response for remote pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StageLifecycleResponse {
    State(StageLifecycleSnapshot),
    BarrierAck {
        barrier_id: u64,
        state: StageLifecycleSnapshot,
    },
    CancelAck {
        request_id: String,
        state: StageLifecycleSnapshot,
    },
}

/// Install an RPC control plane for a pipeline stage.
pub async fn install_stage_control_service(
    transport: Arc<dyn Transport>,
    state: Arc<Mutex<StageLifecycleState>>,
) -> Result<()> {
    transport
        .serve_rpc(Box::new(move |request| {
            let decoded: Result<StageLifecycleRequest> =
                serde_json::from_slice(request).map_err(Into::into);
            let response = match decoded {
                Ok(StageLifecycleRequest::ProbeHealth) => {
                    let state = state.lock().expect("stage lifecycle state poisoned");
                    StageLifecycleResponse::State(state.snapshot())
                }
                Ok(StageLifecycleRequest::BeginDrain) => {
                    let mut state = state.lock().expect("stage lifecycle state poisoned");
                    state.draining = true;
                    StageLifecycleResponse::State(state.snapshot())
                }
                Ok(StageLifecycleRequest::Barrier { barrier_id }) => {
                    let mut state = state.lock().expect("stage lifecycle state poisoned");
                    state.last_barrier_id = Some(barrier_id);
                    StageLifecycleResponse::BarrierAck {
                        barrier_id,
                        state: state.snapshot(),
                    }
                }
                Ok(StageLifecycleRequest::CancelRequest { request_id }) => {
                    let mut state = state.lock().expect("stage lifecycle state poisoned");
                    state.mark_request_finished();
                    StageLifecycleResponse::CancelAck {
                        request_id,
                        state: state.snapshot(),
                    }
                }
                Ok(StageLifecycleRequest::Shutdown) => {
                    let mut state = state.lock().expect("stage lifecycle state poisoned");
                    state.draining = true;
                    state.shutdown = true;
                    state.health = StageHealth::Failed;
                    state.in_flight_requests = 0;
                    StageLifecycleResponse::State(state.snapshot())
                }
                Err(err) => StageLifecycleResponse::State(StageLifecycleSnapshot {
                    health: StageHealth::Failed,
                    draining: true,
                    shutdown: true,
                    in_flight_requests: 0,
                    last_barrier_id: Some(hash_error(&err.to_string())),
                }),
            };
            serde_json::to_vec(&response)
                .unwrap_or_else(|err| panic!("failed to serialize lifecycle response: {err}"))
        }))
        .await
}

/// One side of a transport-backed pipeline link.
#[derive(Clone)]
pub struct TransportStageEndpoint {
    transport: Arc<dyn Transport>,
    peer_addr: String,
    label: String,
}

impl TransportStageEndpoint {
    pub fn new(
        transport: Arc<dyn Transport>,
        peer_addr: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            peer_addr: peer_addr.into(),
            label: label.into(),
        }
    }

    pub async fn send_activation(&self, msg: ActivationMessage) -> Result<()> {
        let payload = serde_json::to_vec(&msg)?;
        self.transport
            .send(
                &self.peer_addr,
                TransportMessage::Control {
                    operation: PIPELINE_ACTIVATION_OPERATION.to_string(),
                    payload: Bytes::from(payload),
                },
            )
            .await
    }

    pub async fn recv_activation(&self) -> Result<(String, ActivationMessage)> {
        loop {
            let (from, msg) = self.transport.recv().await?;
            match msg {
                TransportMessage::Control { operation, payload }
                    if operation == PIPELINE_ACTIVATION_OPERATION =>
                {
                    let activation = serde_json::from_slice::<ActivationMessage>(&payload)?;
                    return Ok((from, activation));
                }
                TransportMessage::Control { .. } | TransportMessage::TensorData { .. } => continue,
            }
        }
    }

    pub async fn probe_health(&self) -> Result<StageLifecycleSnapshot> {
        match self
            .rpc_lifecycle(StageLifecycleRequest::ProbeHealth)
            .await?
        {
            StageLifecycleResponse::State(state) => Ok(state),
            other => bail!("unexpected probe_health response: {other:?}"),
        }
    }

    pub async fn begin_drain(&self) -> Result<StageLifecycleSnapshot> {
        match self
            .rpc_lifecycle(StageLifecycleRequest::BeginDrain)
            .await?
        {
            StageLifecycleResponse::State(state) => Ok(state),
            other => bail!("unexpected begin_drain response: {other:?}"),
        }
    }

    pub async fn barrier(&self, barrier_id: u64) -> Result<StageLifecycleSnapshot> {
        match self
            .rpc_lifecycle(StageLifecycleRequest::Barrier { barrier_id })
            .await?
        {
            StageLifecycleResponse::BarrierAck {
                barrier_id: ack_id,
                state,
            } => {
                if ack_id != barrier_id {
                    bail!("barrier ack mismatch: expected {barrier_id}, got {ack_id}");
                }
                Ok(state)
            }
            other => bail!("unexpected barrier response: {other:?}"),
        }
    }

    pub async fn cancel_request(&self, request_id: &RequestId) -> Result<StageLifecycleSnapshot> {
        match self
            .rpc_lifecycle(StageLifecycleRequest::CancelRequest {
                request_id: request_id.as_str().to_string(),
            })
            .await?
        {
            StageLifecycleResponse::CancelAck { state, .. } => Ok(state),
            other => bail!("unexpected cancel_request response: {other:?}"),
        }
    }

    pub async fn shutdown_peer(&self) -> Result<StageLifecycleSnapshot> {
        match self.rpc_lifecycle(StageLifecycleRequest::Shutdown).await? {
            StageLifecycleResponse::State(state) => Ok(state),
            other => bail!("unexpected shutdown response: {other:?}"),
        }
    }

    pub fn peer_addr(&self) -> &str {
        &self.peer_addr
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    async fn rpc_lifecycle(
        &self,
        request: StageLifecycleRequest,
    ) -> Result<StageLifecycleResponse> {
        let request_bytes = serde_json::to_vec(&request)?;
        let response_bytes = self
            .transport
            .rpc_call(&self.peer_addr, &request_bytes)
            .await?;
        serde_json::from_slice(&response_bytes).map_err(Into::into)
    }
}

impl fmt::Debug for TransportStageEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransportStageEndpoint")
            .field("peer_addr", &self.peer_addr)
            .field("label", &self.label)
            .finish()
    }
}

/// Transport-backed stage link using the shared distributed transport layer.
#[derive(Clone, Debug)]
pub struct TransportStageLink {
    pub upstream_stage: u32,
    pub downstream_stage: u32,
    pub upstream: TransportStageEndpoint,
    pub downstream: TransportStageEndpoint,
}

impl TransportStageLink {
    pub fn new(
        upstream_stage: u32,
        downstream_stage: u32,
        upstream_transport: Arc<dyn Transport>,
        downstream_transport: Arc<dyn Transport>,
    ) -> Result<Self> {
        let upstream_addr = upstream_transport.local_addr()?;
        let downstream_addr = downstream_transport.local_addr()?;
        Ok(Self {
            upstream_stage,
            downstream_stage,
            upstream: TransportStageEndpoint::new(
                upstream_transport,
                downstream_addr.clone(),
                format!("transport-stage-{upstream_stage}->{downstream_stage}"),
            ),
            downstream: TransportStageEndpoint::new(
                downstream_transport,
                upstream_addr,
                format!("transport-stage-{downstream_stage}->{upstream_stage}"),
            ),
        })
    }
}

/// Sending half of an activation channel with optional timeout.
#[derive(Clone)]
pub struct ActivationSender {
    inner: mpsc::Sender<ActivationMessage>,
    config: Arc<ChannelConfig>,
    /// Label for logging/debugging (e.g., "stage-0->stage-1").
    label: String,
}

impl ActivationSender {
    /// Send an activation message, blocking if the channel is full (back-pressure).
    ///
    /// Returns an error if the receiver has been dropped or the send times out.
    pub async fn send(&self, msg: ActivationMessage) -> Result<()> {
        match self.config.send_timeout {
            Some(timeout) => {
                let permit = tokio::time::timeout(timeout, self.inner.reserve())
                    .await
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "activation send timed out after {timeout:?} on channel '{}'",
                            self.label
                        )
                    })?
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "activation channel '{}' closed (receiver dropped)",
                            self.label
                        )
                    })?;
                permit.send(msg);
                Ok(())
            }
            None => self.inner.send(msg).await.map_err(|_| {
                anyhow::anyhow!(
                    "activation channel '{}' closed (receiver dropped)",
                    self.label
                )
            }),
        }
    }

    /// Try to send without blocking. Returns `Err` if the channel is full or closed.
    pub fn try_send(&self, msg: ActivationMessage) -> Result<()> {
        self.inner.try_send(msg).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                anyhow::anyhow!("activation channel '{}' full (back-pressure)", self.label)
            }
            mpsc::error::TrySendError::Closed(_) => {
                anyhow::anyhow!(
                    "activation channel '{}' closed (receiver dropped)",
                    self.label
                )
            }
        })
    }

    /// Number of messages currently buffered in the channel.
    pub fn queued(&self) -> usize {
        self.config.capacity - self.inner.capacity()
    }

    /// True if the receiver has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// The label for this channel (e.g., "stage-0->stage-1").
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Debug for ActivationSender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActivationSender")
            .field("label", &self.label)
            .field("queued", &self.queued())
            .finish()
    }
}

/// Receiving half of an activation channel.
pub struct ActivationReceiver {
    inner: mpsc::Receiver<ActivationMessage>,
    config: Arc<ChannelConfig>,
    /// Label for logging/debugging (e.g., "stage-0->stage-1").
    label: String,
}

impl ActivationReceiver {
    /// Receive the next activation message, waiting if the channel is empty.
    ///
    /// Returns `None` if the sender has been dropped (channel closed).
    pub async fn recv(&mut self) -> Result<Option<ActivationMessage>> {
        match self.config.recv_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, self.inner.recv()).await {
                Ok(msg) => Ok(msg),
                Err(_) => bail!(
                    "activation recv timed out after {timeout:?} on channel '{}'",
                    self.label
                ),
            },
            None => Ok(self.inner.recv().await),
        }
    }

    /// Try to receive without blocking. Returns `None` if no message is available.
    pub fn try_recv(&mut self) -> Option<ActivationMessage> {
        self.inner.try_recv().ok()
    }

    /// The label for this channel.
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Debug for ActivationReceiver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActivationReceiver")
            .field("label", &self.label)
            .finish()
    }
}

/// Create a paired (sender, receiver) activation channel.
pub fn activation_channel(
    label: impl Into<String>,
    config: ChannelConfig,
) -> (ActivationSender, ActivationReceiver) {
    let capacity = config.capacity.max(1);
    let (tx, rx) = mpsc::channel(capacity);
    let config = Arc::new(config);
    let label = label.into();
    (
        ActivationSender {
            inner: tx,
            config: Arc::clone(&config),
            label: label.clone(),
        },
        ActivationReceiver {
            inner: rx,
            config,
            label,
        },
    )
}

/// Bidirectional activation channel between two adjacent stages.
///
/// Forward path (stage N -> N+1) and reverse path (N+1 -> N).
#[derive(Debug)]
pub struct PipelineChannel {
    pub forward_tx: ActivationSender,
    pub forward_rx: ActivationReceiver,
    pub reverse_tx: ActivationSender,
    pub reverse_rx: ActivationReceiver,
}

impl PipelineChannel {
    /// Create a new bidirectional pipeline channel between `from_stage` and
    /// `to_stage` with the given configuration.
    pub fn new(from_stage: u32, to_stage: u32, config: &ChannelConfig) -> Self {
        let forward_label = format!("stage-{from_stage}->stage-{to_stage}");
        let reverse_label = format!("stage-{to_stage}->stage-{from_stage}");

        let (forward_tx, forward_rx) = activation_channel(forward_label, config.clone());
        let (reverse_tx, reverse_rx) = activation_channel(reverse_label, config.clone());

        Self {
            forward_tx,
            forward_rx,
            reverse_tx,
            reverse_rx,
        }
    }

    /// Split this channel into the two halves needed by adjacent stages.
    ///
    /// Returns `(StageEndpoint for stage N, StageEndpoint for stage N+1)`.
    pub fn split(self) -> (StageEndpoint, StageEndpoint) {
        let left = StageEndpoint {
            send_forward: self.forward_tx,
            recv_reverse: self.reverse_rx,
        };
        let right = StageEndpoint {
            send_forward: self.reverse_tx,
            recv_reverse: self.forward_rx,
        };
        (left, right)
    }
}

/// One side of a [`PipelineChannel`], held by a single stage.
#[derive(Debug)]
pub struct StageEndpoint {
    pub send_forward: ActivationSender,
    pub recv_reverse: ActivationReceiver,
}

/// A link connecting two adjacent pipeline stages with bidirectional channels.
/// For an N-stage pipeline there are N-1 links.
pub struct StageLink {
    pub upstream_stage: u32,
    pub downstream_stage: u32,
    pub forward_tx: ActivationSender,
    pub forward_rx: ActivationReceiver,
    pub reverse_tx: ActivationSender,
    pub reverse_rx: ActivationReceiver,
}

impl StageLink {
    /// Create a link between two adjacent stages.
    pub fn new(upstream: u32, downstream: u32, config: &ChannelConfig) -> Self {
        let fwd_label = format!("fwd-{upstream}->{downstream}");
        let rev_label = format!("rev-{downstream}->{upstream}");

        let (forward_tx, forward_rx) = activation_channel(fwd_label, config.clone());
        let (reverse_tx, reverse_rx) = activation_channel(rev_label, config.clone());

        Self {
            upstream_stage: upstream,
            downstream_stage: downstream,
            forward_tx,
            forward_rx,
            reverse_tx,
            reverse_rx,
        }
    }
}

impl fmt::Debug for StageLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageLink")
            .field("upstream", &self.upstream_stage)
            .field("downstream", &self.downstream_stage)
            .finish()
    }
}

/// Build a chain of [`StageLink`]s for an N-stage pipeline.
///
/// Returns N-1 links connecting stages 0..N-1 in sequence.
///
/// # Errors
///
/// Returns an error if `num_stages` is less than 2.
pub fn build_pipeline_links(num_stages: u32, config: &ChannelConfig) -> Result<Vec<StageLink>> {
    if num_stages < 2 {
        bail!("pipeline requires at least 2 stages, got {num_stages}");
    }
    let links = (0..num_stages - 1)
        .map(|i| StageLink::new(i, i + 1, config))
        .collect();
    Ok(links)
}

/// Current monotonic timestamp in nanoseconds (for latency measurement).
fn timestamp_nanos() -> u64 {
    use std::time::Instant;
    // Use a lazy-static-like anchor to keep times relative.
    // For simplicity, use the process start time via Instant::now() delta.
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

fn hash_error(message: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    message.hash(&mut hasher);
    hasher.finish()
}

/// Compute one-way latency from the message timestamp to now.
pub fn activation_latency(msg: &ActivationMessage) -> Duration {
    let now = timestamp_nanos();
    let delta_ns = now.saturating_sub(msg.timestamp_ns);
    Duration::from_nanos(delta_ns)
}

/// Validate that an activation message is well-formed.
pub fn validate_activation(msg: &ActivationMessage) -> Result<()> {
    if msg.num_stages < 2 {
        bail!("num_stages must be >= 2, got {}", msg.num_stages);
    }
    if msg.stage_index >= msg.num_stages {
        bail!(
            "stage_index {} out of range for {}-stage pipeline",
            msg.stage_index,
            msg.num_stages
        );
    }
    if msg.tensor_data.is_empty() {
        bail!("tensor_data is empty");
    }
    if !msg.is_reverse_path && msg.stage_index == msg.num_stages - 1 {
        bail!(
            "forward message from last stage {} has nowhere to go",
            msg.stage_index
        );
    }
    if msg.is_reverse_path && msg.stage_index == 0 {
        bail!("reverse message from stage 0 has nowhere to go");
    }
    Ok(())
}

#[cfg(test)]
#[path = "activation_transfer_tests.rs"]
mod tests;
