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

//! Pipeline parallel serving integration.
//!
//! Connects the pipeline parallelism infrastructure to the serving stack
//! so that API requests flow transparently through the pipeline:
//!
//! - **First stage** receives tokenized requests from the batch scheduler
//!   and injects them into the pipeline.
//! - **Middle stages** receive activations, run their layer subset, and
//!   forward outputs downstream.
//! - **Last stage** runs the final layers + lm_head and produces tokens
//!   that are returned to the first stage for API response assembly.
//!
//! Key types:
//!
//! - [`StageRole`] — first, middle, last, or single-stage (no PP).
//! - [`PipelineServingConfig`] — configuration for the serving coordinator.
//! - [`PipelineRequest`] — wraps an API request for pipeline processing.
//! - [`PipelineResponse`] — wraps generated output from the last stage.
//! - [`StageHealth`] — health status of a pipeline stage.
//! - [`PipelineCoordinator`] — multi-stage coordination and request routing.
//! - [`ChunkedPrefillPipeline`] — chunked prefill across pipeline stages.
//!
//! Used by: server startup, batch scheduler, model worker

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::distributed::request_tracker::RequestId;

use super::cache_manager::SequenceId;
use super::metrics::PipelineMetrics;
use super::schedule::PipelineConfig;

// ---------------------------------------------------------------------------
// StageRole
// ---------------------------------------------------------------------------

/// Role of a pipeline stage in the serving topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StageRole {
    /// First stage: receives requests from the API server / batch scheduler,
    /// runs embedding + first layer subset, sends activations downstream,
    /// and collects generated tokens from the reverse path.
    First,
    /// Middle stage: receives activations from the upstream stage, runs its
    /// layer subset, and forwards activations downstream.
    Middle,
    /// Last stage: receives activations from the upstream stage, runs the
    /// final layer subset + lm_head, produces tokens, and sends them back
    /// on the reverse path.
    Last,
    /// Single stage: pipeline parallelism is not active; all layers run
    /// on one device. The serving path is identical to non-PP mode.
    SingleStage,
}

impl StageRole {
    /// Determine the role from stage index and total stage count.
    pub fn from_index(stage_index: u32, num_stages: u32) -> Self {
        if num_stages <= 1 {
            return Self::SingleStage;
        }
        match stage_index {
            0 => Self::First,
            i if i == num_stages - 1 => Self::Last,
            _ => Self::Middle,
        }
    }

    /// Whether this role is the entry point for API requests.
    pub fn is_entry_point(self) -> bool {
        matches!(self, Self::First | Self::SingleStage)
    }

    /// Whether this role produces final tokens (logits -> sampling).
    pub fn produces_tokens(self) -> bool {
        matches!(self, Self::Last | Self::SingleStage)
    }
}

impl fmt::Display for StageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::First => write!(f, "First"),
            Self::Middle => write!(f, "Middle"),
            Self::Last => write!(f, "Last"),
            Self::SingleStage => write!(f, "SingleStage"),
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineServingConfig
// ---------------------------------------------------------------------------

/// Configuration for the pipeline serving coordinator.
#[derive(Debug, Clone)]
pub struct PipelineServingConfig {
    /// Total number of pipeline stages.
    pub num_stages: u32,
    /// Index of this stage (0-based).
    pub stage_index: u32,
    /// Timeout for inter-stage communication. If an activation or reverse
    /// message is not received within this duration, the request fails.
    pub stage_timeout: Duration,
    /// Maximum number of sequences that can be in-flight in the pipeline.
    pub max_in_flight: usize,
    /// Micro-batch size for pipeline scheduling.
    pub micro_batch_size: usize,
    /// Prefill chunk size in tokens (0 = disabled, full prefill per step).
    pub prefill_chunk_size: usize,
}

impl PipelineServingConfig {
    /// Create a new serving config.
    pub fn new(num_stages: u32, stage_index: u32) -> Result<Self> {
        ensure!(
            stage_index < num_stages,
            "stage_index {stage_index} out of range for {num_stages}-stage pipeline"
        );
        Ok(Self {
            num_stages,
            stage_index,
            stage_timeout: Duration::from_secs(30),
            micro_batch_size: 1,
            max_in_flight: 64,
            prefill_chunk_size: 0,
        })
    }

    /// Set the inter-stage timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.stage_timeout = timeout;
        self
    }

    /// Set the maximum in-flight sequences.
    #[must_use]
    pub fn with_max_in_flight(mut self, max: usize) -> Self {
        self.max_in_flight = max;
        self
    }

    /// Set the micro-batch size.
    #[must_use]
    pub fn with_micro_batch_size(mut self, size: usize) -> Self {
        self.micro_batch_size = size.max(1);
        self
    }

    /// Set the prefill chunk size.
    #[must_use]
    pub fn with_prefill_chunk_size(mut self, size: usize) -> Self {
        self.prefill_chunk_size = size;
        self
    }

    /// Derive the [`StageRole`] for this config.
    pub fn role(&self) -> StageRole {
        StageRole::from_index(self.stage_index, self.num_stages)
    }

    /// Whether pipeline parallelism is active (more than one stage).
    pub fn is_pipeline_active(&self) -> bool {
        self.num_stages > 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_stages > 0, "num_stages must be > 0");
        ensure!(
            self.stage_index < self.num_stages,
            "stage_index {} out of range for {}-stage pipeline",
            self.stage_index,
            self.num_stages
        );
        ensure!(self.max_in_flight > 0, "max_in_flight must be > 0");
        ensure!(self.micro_batch_size > 0, "micro_batch_size must be > 0");
        ensure!(!self.stage_timeout.is_zero(), "stage_timeout must be > 0");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PipelineRequest / PipelineResponse
// ---------------------------------------------------------------------------

/// A request packaged for pipeline processing.
///
/// Wraps the tokenized input and metadata needed to route through the
/// pipeline stages. Created by the first stage from an API request.
#[derive(Debug, Clone)]
pub struct PipelineRequest {
    /// Unique identifier for this request.
    pub request_id: RequestId,
    /// Sequence ID for KV cache tracking.
    pub sequence_id: SequenceId,
    /// Tokenized input IDs (only meaningful on the first stage).
    pub token_ids: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// When this request was submitted to the pipeline.
    pub submitted_at: Instant,
    /// Number of prefill tokens already processed (for chunked prefill).
    pub prefill_offset: usize,
    /// Whether this request is in the decode phase (prefill complete).
    pub is_decoding: bool,
}

impl PipelineRequest {
    /// Create a new pipeline request.
    pub fn new(
        request_id: RequestId,
        sequence_id: SequenceId,
        token_ids: Vec<u32>,
        max_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            token_ids,
            max_tokens,
            submitted_at: Instant::now(),
            prefill_offset: 0,
            is_decoding: false,
        }
    }

    /// Remaining prompt tokens to process (for chunked prefill).
    pub fn remaining_prefill_tokens(&self) -> usize {
        self.token_ids.len().saturating_sub(self.prefill_offset)
    }

    /// Whether the full prompt has been prefilled.
    pub fn is_prefill_complete(&self) -> bool {
        self.prefill_offset >= self.token_ids.len()
    }

    /// Time elapsed since the request was submitted.
    pub fn elapsed(&self) -> Duration {
        self.submitted_at.elapsed()
    }
}

impl fmt::Display for PipelineRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineReq[id={} seq={} tokens={} prefill_off={} decoding={}]",
            self.request_id,
            self.sequence_id,
            self.token_ids.len(),
            self.prefill_offset,
            self.is_decoding,
        )
    }
}

/// Response produced by the pipeline (from the last stage).
#[derive(Debug, Clone)]
pub struct PipelineResponse {
    /// Request ID this response corresponds to.
    pub request_id: RequestId,
    /// Sequence ID for cache tracking.
    pub sequence_id: SequenceId,
    /// Generated token IDs.
    pub generated_tokens: Vec<u32>,
    /// Whether the sequence is complete (EOS or max tokens).
    pub is_finished: bool,
    /// Time spent in the pipeline for this response.
    pub pipeline_latency: Duration,
    /// Error message if the request failed in the pipeline.
    pub error: Option<String>,
}

impl PipelineResponse {
    /// Create a successful response with generated tokens.
    pub fn success(
        request_id: RequestId,
        sequence_id: SequenceId,
        generated_tokens: Vec<u32>,
        is_finished: bool,
        pipeline_latency: Duration,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            generated_tokens,
            is_finished,
            pipeline_latency,
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(request_id: RequestId, sequence_id: SequenceId, error: String) -> Self {
        Self {
            request_id,
            sequence_id,
            generated_tokens: Vec::new(),
            is_finished: true,
            pipeline_latency: Duration::ZERO,
            error: Some(error),
        }
    }

    /// Whether this response indicates an error.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

impl fmt::Display for PipelineResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref err) = self.error {
            write!(f, "PipelineResp[id={} ERROR: {err}]", self.request_id,)
        } else {
            write!(
                f,
                "PipelineResp[id={} tokens={} finished={} latency={:.2}ms]",
                self.request_id,
                self.generated_tokens.len(),
                self.is_finished,
                self.pipeline_latency.as_secs_f64() * 1000.0,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// StageHealth
// ---------------------------------------------------------------------------

/// Health status of a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StageHealth {
    /// Stage is operating normally.
    Healthy,
    /// Stage is responding but with elevated latency or errors.
    Degraded,
    /// Stage has failed (unresponsive or crashed).
    Failed,
    /// Stage health is unknown (not yet probed).
    Unknown,
}

impl StageHealth {
    /// Whether the stage is usable for processing.
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

impl fmt::Display for StageHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Failed => write!(f, "failed"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// FailedRequest
// ---------------------------------------------------------------------------

/// A request that failed due to a stage failure.
#[derive(Debug, Clone)]
pub struct FailedRequest {
    /// The request that failed.
    pub request_id: RequestId,
    /// Sequence ID.
    pub sequence_id: SequenceId,
    /// Stage where the failure was detected.
    pub failed_stage: u32,
    /// Description of the failure.
    pub reason: String,
}

impl fmt::Display for FailedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FailedReq[id={} stage={} reason={}]",
            self.request_id, self.failed_stage, self.reason,
        )
    }
}

// ---------------------------------------------------------------------------
// PipelineCoordinator
// ---------------------------------------------------------------------------

/// Coordinates multi-stage pipeline serving.
///
/// The coordinator runs on the first stage and manages:
/// - Request intake from the API server / batch scheduler
/// - Activation flow tracking through pipeline stages
/// - Result collection from the last stage via reverse-path messages
/// - Stage health monitoring and failure handling
/// - Timeout enforcement for in-flight requests
///
/// Used by: server startup, batch scheduler, model worker
pub struct PipelineCoordinator {
    /// Serving configuration.
    config: PipelineServingConfig,
    /// In-flight requests keyed by request ID.
    in_flight: HashMap<String, InFlightEntry>,
    /// Stage health status for each stage.
    stage_health: Vec<StageHealth>,
    /// Pipeline metrics for the current step.
    metrics: PipelineMetrics,
    /// Monotonically increasing sequence ID counter.
    next_sequence_id: SequenceId,
    /// Whether the pipeline is draining and refusing new requests.
    draining: bool,
}

/// Internal tracking entry for an in-flight request.
struct InFlightEntry {
    request: PipelineRequest,
    /// Channel to send the response back to the caller.
    response_tx: Option<oneshot::Sender<PipelineResponse>>,
    /// Current pipeline stage the request is at.
    current_stage: u32,
    /// Tokens generated so far (accumulated from reverse-path messages).
    generated_tokens: Vec<u32>,
}

impl fmt::Debug for InFlightEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InFlightEntry")
            .field("request_id", &self.request.request_id)
            .field("current_stage", &self.current_stage)
            .field("generated_count", &self.generated_tokens.len())
            .finish()
    }
}

impl PipelineCoordinator {
    /// Create a new pipeline coordinator.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid.
    pub fn new(config: PipelineServingConfig) -> Result<Self> {
        config.validate()?;
        let num_stages = config.num_stages as usize;
        let metrics = PipelineMetrics::new(config.num_stages, 0);
        Ok(Self {
            config,
            in_flight: HashMap::new(),
            stage_health: vec![StageHealth::Unknown; num_stages],
            metrics,
            next_sequence_id: 0,
            draining: false,
        })
    }

    /// Submit a request to the pipeline.
    ///
    /// Returns a oneshot receiver that will deliver the pipeline response
    /// once the request completes (or fails).
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline is at capacity or the first stage
    /// is not healthy.
    pub fn submit_request(
        &mut self,
        mut request: PipelineRequest,
    ) -> Result<oneshot::Receiver<PipelineResponse>> {
        if self.draining {
            bail!("pipeline is draining; cannot accept new requests");
        }
        // Check capacity.
        if self.in_flight.len() >= self.config.max_in_flight {
            bail!(
                "pipeline at capacity: {} in-flight requests (max {})",
                self.in_flight.len(),
                self.config.max_in_flight
            );
        }

        // Check first stage health.
        if self.config.is_pipeline_active() {
            let first_health = self
                .stage_health
                .first()
                .copied()
                .unwrap_or(StageHealth::Unknown);
            if !first_health.is_usable() && first_health != StageHealth::Unknown {
                bail!("first pipeline stage is {first_health}; cannot accept requests");
            }
        }

        // Assign sequence ID if not set.
        if request.sequence_id == 0 {
            self.next_sequence_id += 1;
            request.sequence_id = self.next_sequence_id;
        }

        let (tx, rx) = oneshot::channel();
        let key = request.request_id.as_str().to_string();

        // Reject duplicate request IDs to prevent silently orphaning the
        // previous oneshot sender, which would leave the old caller hanging.
        if self.in_flight.contains_key(&key) {
            bail!("duplicate request ID: {key}");
        }

        self.in_flight.insert(
            key,
            InFlightEntry {
                request,
                response_tx: Some(tx),
                current_stage: 0,
                generated_tokens: Vec::new(),
            },
        );

        Ok(rx)
    }

    /// Process the output of a stage: advance the request to the next stage
    /// or collect results if this was the last stage.
    ///
    /// Called when an activation or token output arrives from a stage.
    /// Tokens are accumulated in the in-flight entry. The response is only
    /// delivered when `is_finished` is true (EOS or max tokens reached),
    /// so that multi-token generation works correctly across decode steps.
    pub fn process_stage_output(
        &mut self,
        request_id: &RequestId,
        stage_index: u32,
        generated_token: Option<u32>,
        is_finished: bool,
    ) -> Result<()> {
        let key = request_id.as_str();
        let entry = self
            .in_flight
            .get_mut(key)
            .ok_or_else(|| anyhow::anyhow!("request {key} not found in pipeline"))?;

        entry.current_stage = stage_index;

        if let Some(token) = generated_token {
            entry.generated_tokens.push(token);
        }

        // Only deliver when the sequence is explicitly finished (EOS or max
        // tokens). Delivering on the first last-stage token would remove the
        // entry from in_flight, causing subsequent decode steps to fail with
        // "request not found".
        if is_finished {
            self.deliver_response(request_id, true);
        }

        Ok(())
    }

    /// Collect the result for a request.
    ///
    /// If the request has finished (all tokens generated or EOS), the
    /// response is delivered through the oneshot channel. Otherwise this
    /// is a no-op.
    pub fn collect_result(&self, request_id: &RequestId) -> Option<&PipelineRequest> {
        self.in_flight.get(request_id.as_str()).map(|e| &e.request)
    }

    /// Handle a stage failure: fail all in-flight requests on the affected
    /// stage and return them as [`FailedRequest`]s.
    pub fn handle_stage_failure(&mut self, stage_index: u32) -> Vec<FailedRequest> {
        // Mark the stage as failed.
        if let Some(health) = self.stage_health.get_mut(stage_index as usize) {
            *health = StageHealth::Failed;
        }

        let reason = format!("pipeline stage {stage_index} failed");
        let mut failed = Vec::new();

        // Collect keys of affected requests.
        let affected_keys: Vec<String> = self
            .in_flight
            .iter()
            .filter(|(_, entry)| entry.current_stage == stage_index)
            .map(|(key, _)| key.clone())
            .collect();

        for key in affected_keys {
            if let Some(mut entry) = self.in_flight.remove(&key) {
                failed.push(FailedRequest {
                    request_id: entry.request.request_id.clone(),
                    sequence_id: entry.request.sequence_id,
                    failed_stage: stage_index,
                    reason: reason.clone(),
                });

                // Send error response through the channel.
                if let Some(tx) = entry.response_tx.take() {
                    let _ = tx.send(PipelineResponse::error(
                        entry.request.request_id,
                        entry.request.sequence_id,
                        reason.clone(),
                    ));
                }
            }
        }

        failed
    }

    /// Enforce timeouts on all in-flight requests.
    ///
    /// Returns the list of requests that timed out and were failed.
    pub fn enforce_timeouts(&mut self) -> Vec<FailedRequest> {
        let timeout = self.config.stage_timeout;
        let mut timed_out_keys = Vec::new();

        for (key, entry) in &self.in_flight {
            if entry.request.elapsed() > timeout {
                timed_out_keys.push(key.clone());
            }
        }

        let mut failed = Vec::new();
        for key in timed_out_keys {
            if let Some(mut entry) = self.in_flight.remove(&key) {
                let reason = format!(
                    "request timed out after {:.1}s (stage {})",
                    entry.request.elapsed().as_secs_f64(),
                    entry.current_stage,
                );
                failed.push(FailedRequest {
                    request_id: entry.request.request_id.clone(),
                    sequence_id: entry.request.sequence_id,
                    failed_stage: entry.current_stage,
                    reason: reason.clone(),
                });
                if let Some(tx) = entry.response_tx.take() {
                    let _ = tx.send(PipelineResponse::error(
                        entry.request.request_id,
                        entry.request.sequence_id,
                        reason,
                    ));
                }
            }
        }

        failed
    }

    /// Update the health status of a stage.
    pub fn update_stage_health(&mut self, stage_index: u32, health: StageHealth) {
        if let Some(slot) = self.stage_health.get_mut(stage_index as usize) {
            *slot = health;
        }
    }

    /// Begin a graceful drain. Existing requests may finish, but new requests
    /// are rejected until the coordinator is shut down or recreated.
    pub fn begin_drain(&mut self) {
        self.draining = true;
    }

    /// Whether the pipeline is draining.
    pub fn is_draining(&self) -> bool {
        self.draining
    }

    /// Whether the drain has completed and there are no in-flight requests.
    pub fn drain_complete(&self) -> bool {
        self.draining && self.in_flight.is_empty()
    }

    /// Force shutdown of the pipeline coordinator and fail all remaining
    /// in-flight requests.
    pub fn shutdown(&mut self, reason: impl Into<String>) -> Vec<FailedRequest> {
        self.draining = true;
        let reason = reason.into();
        let mut failed = Vec::with_capacity(self.in_flight.len());
        for (_, mut entry) in self.in_flight.drain() {
            let failed_request = FailedRequest {
                request_id: entry.request.request_id.clone(),
                sequence_id: entry.request.sequence_id,
                failed_stage: entry.current_stage,
                reason: reason.clone(),
            };
            if let Some(tx) = entry.response_tx.take() {
                let _ = tx.send(PipelineResponse::error(
                    entry.request.request_id.clone(),
                    entry.request.sequence_id,
                    reason.clone(),
                ));
            }
            failed.push(failed_request);
        }
        failed
    }

    /// Get the health status of a stage.
    pub fn stage_health(&self, stage_index: u32) -> StageHealth {
        self.stage_health
            .get(stage_index as usize)
            .copied()
            .unwrap_or(StageHealth::Unknown)
    }

    /// Whether all stages are healthy or unknown (usable).
    pub fn all_stages_usable(&self) -> bool {
        self.stage_health
            .iter()
            .all(|h| h.is_usable() || *h == StageHealth::Unknown)
    }

    /// Number of in-flight requests.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Whether the pipeline can accept more requests.
    pub fn can_accept(&self) -> bool {
        !self.draining
            && self.in_flight.len() < self.config.max_in_flight
            && self.all_stages_usable()
    }

    /// Reference to the serving config.
    pub fn config(&self) -> &PipelineServingConfig {
        &self.config
    }

    /// Reference to the pipeline metrics.
    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    /// Allocate the next sequence ID.
    pub fn allocate_sequence_id(&mut self) -> SequenceId {
        self.next_sequence_id += 1;
        self.next_sequence_id
    }

    /// Deliver a response for the given request through its oneshot channel.
    fn deliver_response(&mut self, request_id: &RequestId, is_finished: bool) {
        let key = request_id.as_str();
        if let Some(mut entry) = self.in_flight.remove(key)
            && let Some(tx) = entry.response_tx.take()
        {
            let latency = entry.request.elapsed();
            let _ = tx.send(PipelineResponse::success(
                entry.request.request_id,
                entry.request.sequence_id,
                entry.generated_tokens,
                is_finished,
                latency,
            ));
        }
    }
}

impl fmt::Debug for PipelineCoordinator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCoordinator")
            .field("config", &self.config)
            .field("in_flight", &self.in_flight.len())
            .field("stage_health", &self.stage_health)
            .finish()
    }
}

impl fmt::Display for PipelineCoordinator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineCoordinator[stages={} in_flight={}/{} health={:?}]",
            self.config.num_stages,
            self.in_flight.len(),
            self.config.max_in_flight,
            self.stage_health,
        )
    }
}

// ---------------------------------------------------------------------------
// ChunkedPrefillPipeline
// ---------------------------------------------------------------------------

/// Manages chunked prefill across pipeline stages.
///
/// When a long prompt arrives, instead of processing all tokens in one pass
/// (which would stall the pipeline), the prompt is split into chunks. Each
/// chunk flows through all stages before the next chunk is sent. This
/// allows other requests to interleave between chunks, reducing latency
/// for concurrent requests.
///
/// Used by: pipeline coordinator, batch scheduler
#[derive(Debug)]
pub struct ChunkedPrefillPipeline {
    /// Chunk size in tokens.
    chunk_size: usize,
    /// Active prefill sessions keyed by request ID string.
    sessions: HashMap<String, PrefillSession>,
}

/// Tracks the prefill progress for a single request.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PrefillSession {
    /// The request being prefilled (retained for debugging and logging).
    request_id: RequestId,
    /// Sequence ID (retained for cache coordination).
    sequence_id: SequenceId,
    /// Total prompt length.
    prompt_len: usize,
    /// Number of tokens processed so far.
    offset: usize,
    /// When this session started (retained for latency tracking).
    started_at: Instant,
}

impl ChunkedPrefillPipeline {
    /// Create a new chunked prefill pipeline.
    ///
    /// If `chunk_size` is 0, chunking is disabled and prompts are processed
    /// in a single pass.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size: if chunk_size == 0 {
                usize::MAX
            } else {
                chunk_size
            },
            sessions: HashMap::new(),
        }
    }

    /// Start a prefill session for a request.
    ///
    /// Returns the first chunk range `(start, end)` to process.
    pub fn begin_prefill(&mut self, request: &PipelineRequest) -> (usize, usize) {
        let prompt_len = request.token_ids.len();
        let end = self.chunk_size.min(prompt_len);

        self.sessions.insert(
            request.request_id.as_str().to_string(),
            PrefillSession {
                request_id: request.request_id.clone(),
                sequence_id: request.sequence_id,
                prompt_len,
                offset: 0,
                started_at: Instant::now(),
            },
        );

        (0, end)
    }

    /// Advance a prefill session after a chunk has been processed.
    ///
    /// Returns the next chunk range `(start, end)` or `None` if the
    /// prefill is complete.
    pub fn advance_prefill(
        &mut self,
        request_id: &RequestId,
        tokens_processed: usize,
    ) -> Option<(usize, usize)> {
        let key = request_id.as_str();
        let session = self.sessions.get_mut(key)?;

        session.offset += tokens_processed;
        if session.offset >= session.prompt_len {
            // Prefill complete; remove session.
            self.sessions.remove(key);
            return None;
        }

        let start = session.offset;
        let end = (start + self.chunk_size).min(session.prompt_len);
        Some((start, end))
    }

    /// Check if a request is currently being prefilled.
    pub fn is_prefilling(&self, request_id: &RequestId) -> bool {
        self.sessions.contains_key(request_id.as_str())
    }

    /// Cancel a prefill session.
    pub fn cancel_prefill(&mut self, request_id: &RequestId) {
        self.sessions.remove(request_id.as_str());
    }

    /// Number of active prefill sessions.
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Get the progress (offset / total) for a session.
    pub fn progress(&self, request_id: &RequestId) -> Option<(usize, usize)> {
        self.sessions
            .get(request_id.as_str())
            .map(|s| (s.offset, s.prompt_len))
    }
}

impl fmt::Display for ChunkedPrefillPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChunkedPrefill[chunk_size={} active={}]",
            if self.chunk_size == usize::MAX {
                "unlimited".to_string()
            } else {
                self.chunk_size.to_string()
            },
            self.sessions.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// API compatibility helpers
// ---------------------------------------------------------------------------

/// Check whether pipeline parallelism should be enabled based on the
/// server startup config.
///
/// Returns `None` if PP is not configured (single-stage), or
/// `Some(PipelineServingConfig)` if PP is active.
pub fn detect_pipeline_config(
    num_stages: u32,
    stage_index: u32,
    stage_timeout_secs: u64,
    prefill_chunk_size: usize,
) -> Option<PipelineServingConfig> {
    if num_stages <= 1 {
        return None;
    }
    // Use the validated constructor to reject invalid stage_index values.
    let config = PipelineServingConfig::new(num_stages, stage_index)
        .ok()?
        .with_timeout(Duration::from_secs(stage_timeout_secs))
        .with_prefill_chunk_size(prefill_chunk_size);
    Some(config)
}

/// Determine whether a request should follow the pipeline path or the
/// standard (non-PP) path.
///
/// This is the API compatibility layer: when PP is not active, all
/// requests follow the standard path. When PP is active, the first
/// stage routes requests into the pipeline.
pub fn should_use_pipeline(config: Option<&PipelineServingConfig>) -> bool {
    config.is_some_and(|c| c.is_pipeline_active())
}

/// Convert a pipeline config to the schedule config used by the
/// pipeline execution loop.
pub fn to_pipeline_schedule_config(config: &PipelineServingConfig) -> Result<PipelineConfig> {
    PipelineConfig::new(config.num_stages, config.micro_batch_size)
}

#[cfg(test)]
#[path = "serving_tests.rs"]
mod tests;
