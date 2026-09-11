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

//! Prefill node scheduler for disaggregated inference.
//!
//! In disaggregated serving, the prefill node is responsible only for prompt
//! processing and KV cache generation -- it does not perform token decoding.
//! After prefill completes and the first token is sampled, the scheduler
//! serializes the KV cache state and hands it off to a decode node.
//!
//! # Key Design Decisions
//!
//! - **No decode loop**: The scheduler explicitly skips `execute_batched_decode`
//!   when the node role is `Prefill`.
//! - **Shorter-first scheduling**: Shorter prompts are prioritized for better
//!   throughput, as they free resources faster.
//! - **Memory pressure monitoring**: Concurrent prefill count is bounded by
//!   configurable memory thresholds to prevent OOM.
//! - **Chunked prefill**: Long prompts can be prefilled in chunks, with
//!   progressive cache transfer starting before full prefill completes.
//!
//! Used by: disaggregated serving pipeline, Scheduler

use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::distributed::kv_cache_serde::types::{
    SerializableCacheState, SerializableSamplingState,
};
use crate::distributed::kv_cache_transfer::TransferConfig;
use crate::distributed::request_tracker::RequestId;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the prefill node scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillSchedulerConfig {
    /// Maximum number of concurrent prefill operations.
    /// Bounded by GPU memory and compute capacity.
    pub max_concurrent_prefills: usize,

    /// Timeout for the entire handoff process (serialize + transfer + ack).
    pub transfer_timeout: Duration,

    /// Memory utilization threshold (0.0 -- 1.0) above which new prefill
    /// requests are rejected. Prevents OOM under concurrent load.
    pub memory_threshold: f64,

    /// Enable chunked prefill for long sequences. When enabled, prompts
    /// exceeding `chunk_size_tokens` are processed in chunks and cache
    /// layers are transferred progressively.
    pub chunked_prefill_enabled: bool,

    /// Number of tokens per chunk when chunked prefill is enabled.
    pub chunk_size_tokens: usize,

    /// Maximum number of retry attempts for failed handoffs before
    /// marking the request as failed.
    pub max_handoff_retries: u32,

    /// Transfer configuration for KV cache handoff.
    pub transfer_config: TransferConfig,
}

impl Default for PrefillSchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_prefills: 4,
            transfer_timeout: Duration::from_secs(30),
            memory_threshold: 0.85,
            chunked_prefill_enabled: true,
            chunk_size_tokens: 2048,
            max_handoff_retries: 2,
            transfer_config: TransferConfig::default(),
        }
    }
}

// ── Request and Result Types ─────────────────────────────────────────

/// A prefill request submitted to the scheduler.
///
/// Contains all information needed to run prompt prefill and prepare
/// the handoff to a decode node.
#[derive(Debug, Clone)]
pub struct PrefillRequest {
    /// Unique request identifier (shared with the distributed tracker).
    pub request_id: RequestId,

    /// Tokenized prompt (token IDs).
    pub prompt_tokens: Vec<i32>,

    /// Sampling parameters for decode continuation.
    pub sampling_params: SerializableSamplingState,

    /// Whether this is a VLM request requiring vision embedding prefill.
    pub is_vlm: bool,

    /// Opaque image data for VLM requests (already preprocessed).
    /// Empty for text-only requests.
    pub image_data: Vec<u8>,

    /// Priority hint: lower values indicate higher priority.
    /// Default is the prompt length (shorter prompts first).
    pub priority: usize,

    /// Timestamp when the request was submitted.
    pub submitted_at: Instant,
}

impl PrefillRequest {
    /// Create a new text-only prefill request.
    ///
    /// Priority defaults to prompt length (shorter prompts processed first).
    pub fn new(
        request_id: RequestId,
        prompt_tokens: Vec<i32>,
        sampling_params: SerializableSamplingState,
    ) -> Self {
        let priority = prompt_tokens.len();
        Self {
            request_id,
            prompt_tokens,
            sampling_params,
            is_vlm: false,
            image_data: Vec::new(),
            priority,
            submitted_at: Instant::now(),
        }
    }

    /// Create a new VLM prefill request with image data.
    pub fn new_vlm(
        request_id: RequestId,
        prompt_tokens: Vec<i32>,
        sampling_params: SerializableSamplingState,
        image_data: Vec<u8>,
    ) -> Self {
        let priority = prompt_tokens.len();
        Self {
            request_id,
            prompt_tokens,
            sampling_params,
            is_vlm: true,
            image_data,
            priority,
            submitted_at: Instant::now(),
        }
    }

    /// Return the prompt length in tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

/// Wrapper for priority queue ordering (min-heap by priority).
struct PrioritizedRequest(PrefillRequest);

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.0.priority == other.0.priority
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (lower priority value = higher priority).
        other.0.priority.cmp(&self.0.priority)
    }
}

/// Result of a completed prefill operation.
///
/// Contains the first sampled token and the serialized KV cache state
/// ready for transfer to a decode node.
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Request ID this result corresponds to.
    pub request_id: RequestId,

    /// The first token sampled from the prefill forward pass output.
    pub first_token: i32,

    /// Serialized KV cache state (ready for transfer).
    pub cache_state: SerializableCacheState,

    /// Time spent on the prefill forward pass.
    pub prefill_duration: Duration,

    /// Prompt length in tokens.
    pub prompt_len: usize,

    /// Whether this was a VLM prefill.
    pub is_vlm: bool,
}

impl PrefillResult {
    /// Prefill throughput in tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        if self.prefill_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.prompt_len as f64 / self.prefill_duration.as_secs_f64()
    }
}

// ── Handoff Protocol ─────────────────────────────────────────────────

/// Status of a KV cache handoff from prefill to decode node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HandoffStatus {
    /// Handoff is queued but not yet started.
    Pending,
    /// Cache serialization is in progress.
    Serializing,
    /// Cache data is being transferred to the decode node.
    Transferring,
    /// Decode node has acknowledged receipt of the cache.
    Acknowledged,
    /// Handoff failed (with reason).
    Failed { reason: String },
}

impl fmt::Display for HandoffStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Serializing => write!(f, "serializing"),
            Self::Transferring => write!(f, "transferring"),
            Self::Acknowledged => write!(f, "acknowledged"),
            Self::Failed { reason } => write!(f, "failed: {reason}"),
        }
    }
}

/// A handoff record tracking the transfer of a prefill result to a decode node.
#[derive(Debug, Clone)]
pub struct PrefillHandoff {
    /// Request ID being handed off.
    pub request_id: RequestId,

    /// Target decode node ID.
    pub decode_node_id: String,

    /// Current handoff status.
    pub status: HandoffStatus,

    /// Number of retry attempts so far.
    pub retry_count: u32,

    /// Timestamp when the handoff was initiated.
    pub initiated_at: Instant,

    /// The prefill result to be transferred.
    pub result: PrefillResult,
}

impl PrefillHandoff {
    /// Create a new handoff targeting the specified decode node.
    pub fn new(result: PrefillResult, decode_node_id: String) -> Self {
        Self {
            request_id: result.request_id.clone(),
            decode_node_id,
            status: HandoffStatus::Pending,
            retry_count: 0,
            initiated_at: Instant::now(),
            result,
        }
    }

    /// Check if the handoff has exceeded the retry limit.
    pub fn exceeded_retries(&self, max_retries: u32) -> bool {
        self.retry_count >= max_retries
    }

    /// Check if the handoff has exceeded the timeout.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.initiated_at.elapsed() > timeout
    }

    /// Whether no measurable time has passed since this handoff was created.
    ///
    /// Exists so a test can wait for the clock to tick instead of assuming it
    /// already has; `is_timed_out` is a strict comparison, so a zero timeout is
    /// not satisfied until `elapsed` is non-zero.
    #[cfg(test)]
    pub(crate) fn elapsed_is_zero(&self) -> bool {
        self.initiated_at.elapsed() == Duration::ZERO
    }
}

/// Protocol interface for the prefill-to-decode handoff.
///
/// Implementations handle the actual serialization, transfer, and
/// acknowledgment of KV cache data. This trait allows testing with
/// mock transports.
pub trait HandoffProtocol: Send + Sync {
    /// Execute the handoff: serialize cache, transfer to decode node,
    /// and wait for acknowledgment.
    ///
    /// Returns `Ok(())` on successful acknowledged handoff, or an error
    /// describing the failure.
    fn execute_handoff(
        &self,
        handoff: &mut PrefillHandoff,
        transfer_config: &TransferConfig,
        timeout: Duration,
    ) -> Result<()>;

    /// Select the best available decode node for a handoff.
    ///
    /// Implementations may consider node load, network proximity, or
    /// affinity hints.
    fn select_decode_node(&self) -> Result<String>;
}

// ── Chunked Prefill Coordination ─────────────────────────────────────

/// Coordinates progressive cache transfer during chunked prefill.
///
/// When a long prompt is prefilled in chunks, completed cache layers
/// can be transferred before the entire prefill finishes. This reduces
/// TTFT by overlapping transfer with compute.
///
/// # Flow
///
/// ```text
/// Chunk 1 (tokens 0..2048):    [prefill] -> [transfer layers 0..N]
/// Chunk 2 (tokens 2048..4096): [prefill] -> [transfer layers 0..N]
///                                            (incremental update)
/// Final chunk:                 [prefill] -> [transfer + first token]
/// ```
#[derive(Debug)]
pub struct ChunkedPrefillCoordinator {
    /// Total prompt length in tokens.
    total_tokens: usize,

    /// Number of tokens per chunk.
    chunk_size: usize,

    /// Number of chunks completed so far.
    chunks_completed: AtomicUsize,

    /// Number of layers that have been transferred.
    layers_transferred: AtomicUsize,

    /// Total number of model layers.
    total_layers: usize,

    /// Whether the final chunk (with first token) has been processed.
    is_finalized: std::sync::atomic::AtomicBool,

    /// Bytes transferred so far.
    bytes_transferred: AtomicU64,
}

impl ChunkedPrefillCoordinator {
    /// Create a new coordinator for a prompt of the given length.
    pub fn new(total_tokens: usize, chunk_size: usize, total_layers: usize) -> Self {
        Self {
            total_tokens,
            chunk_size,
            chunks_completed: AtomicUsize::new(0),
            layers_transferred: AtomicUsize::new(0),
            total_layers,
            is_finalized: std::sync::atomic::AtomicBool::new(false),
            bytes_transferred: AtomicU64::new(0),
        }
    }

    /// Return the total number of model layers.
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Return the total number of chunks for this prompt.
    pub fn total_chunks(&self) -> usize {
        if self.total_tokens == 0 {
            return 0;
        }
        self.total_tokens.div_ceil(self.chunk_size)
    }

    /// Return the token range for the given chunk index.
    ///
    /// Returns `None` if the chunk index is out of range.
    pub fn chunk_range(&self, chunk_index: usize) -> Option<(usize, usize)> {
        if chunk_index >= self.total_chunks() {
            return None;
        }
        let start = chunk_index * self.chunk_size;
        let end = (start + self.chunk_size).min(self.total_tokens);
        Some((start, end))
    }

    /// Check whether the given chunk index is the final chunk.
    pub fn is_final_chunk(&self, chunk_index: usize) -> bool {
        chunk_index + 1 >= self.total_chunks()
    }

    /// Record that a chunk has been completed.
    ///
    /// Returns the new count of completed chunks.
    pub fn mark_chunk_completed(&self) -> usize {
        self.chunks_completed.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Return how many chunks have been completed.
    pub fn completed_chunks(&self) -> usize {
        self.chunks_completed.load(Ordering::SeqCst)
    }

    /// Record that layers have been transferred for the current chunk.
    pub fn mark_layers_transferred(&self, count: usize) {
        self.layers_transferred.fetch_add(count, Ordering::SeqCst);
    }

    /// Return how many layers have been transferred in total.
    pub fn transferred_layers(&self) -> usize {
        self.layers_transferred.load(Ordering::SeqCst)
    }

    /// Record bytes transferred.
    pub fn add_bytes_transferred(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::SeqCst);
    }

    /// Return total bytes transferred so far.
    pub fn total_bytes_transferred(&self) -> u64 {
        self.bytes_transferred.load(Ordering::SeqCst)
    }

    /// Mark the coordinator as finalized (all chunks processed, first token sampled).
    pub fn finalize(&self) {
        self.is_finalized
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if all chunks have been processed and finalized.
    pub fn is_complete(&self) -> bool {
        self.is_finalized.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Return progress as a fraction (0.0 -- 1.0).
    pub fn progress(&self) -> f64 {
        let total = self.total_chunks();
        if total == 0 {
            return 1.0;
        }
        self.completed_chunks() as f64 / total as f64
    }
}

// ── Prefill Scheduler ────────────────────────────────────────────────

/// The prefill node scheduler for disaggregated inference.
///
/// Manages a priority queue of prefill requests, enforces concurrency
/// and memory limits, and coordinates handoff to decode nodes.
///
/// # Architecture
///
/// ```text
/// Incoming       ┌──────────────────────┐     Handoff
/// Requests  ---> │   PrefillScheduler   │ ---> Decode
///                │                      │      Nodes
///                │  - Priority Queue    │
///                │  - Concurrency Ctrl  │
///                │  - Memory Monitor    │
///                │  - Chunked Prefill   │
///                └──────────────────────┘
/// ```
///
/// Used by: disaggregated serving pipeline, server model worker
pub struct PrefillScheduler {
    /// Scheduler configuration.
    config: PrefillSchedulerConfig,

    /// Priority queue of pending prefill requests (shorter prompts first).
    queue: Mutex<BinaryHeap<PrioritizedRequest>>,

    /// Number of currently active (in-flight) prefill operations.
    active_prefills: AtomicUsize,

    /// Current memory utilization (0.0 -- 1.0), updated externally.
    memory_utilization: std::sync::atomic::AtomicU64,

    /// Total requests enqueued since creation.
    total_enqueued: AtomicU64,

    /// Total requests completed since creation.
    total_completed: AtomicU64,

    /// Total handoffs initiated since creation.
    total_handoffs: AtomicU64,

    /// Total handoff failures since creation.
    total_handoff_failures: AtomicU64,

    /// Active handoffs awaiting acknowledgment.
    active_handoffs: Mutex<Vec<PrefillHandoff>>,
}

impl fmt::Debug for PrefillScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillScheduler")
            .field("max_concurrent", &self.config.max_concurrent_prefills)
            .field(
                "active_prefills",
                &self.active_prefills.load(Ordering::SeqCst),
            )
            .field("queued", &self.queue_len())
            .field("memory_threshold", &self.config.memory_threshold)
            .finish()
    }
}

impl PrefillScheduler {
    /// Create a new prefill scheduler with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `max_concurrent_prefills` is 0, `chunk_size_tokens` is 0,
    /// or `memory_threshold` is not in `(0.0, 1.0]`.
    pub fn new(config: PrefillSchedulerConfig) -> Self {
        assert!(
            config.max_concurrent_prefills > 0,
            "max_concurrent_prefills must be > 0"
        );
        assert!(
            config.chunk_size_tokens > 0,
            "chunk_size_tokens must be > 0"
        );
        assert!(
            config.memory_threshold > 0.0 && config.memory_threshold <= 1.0,
            "memory_threshold must be in (0.0, 1.0], got {}",
            config.memory_threshold
        );
        Self {
            config,
            queue: Mutex::new(BinaryHeap::new()),
            active_prefills: AtomicUsize::new(0),
            memory_utilization: std::sync::atomic::AtomicU64::new(0),
            total_enqueued: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
            total_handoffs: AtomicU64::new(0),
            total_handoff_failures: AtomicU64::new(0),
            active_handoffs: Mutex::new(Vec::new()),
        }
    }

    /// Return the scheduler configuration.
    pub fn config(&self) -> &PrefillSchedulerConfig {
        &self.config
    }

    // ── Request Management ───────────────────────────────────────────

    /// Enqueue a prefill request.
    ///
    /// Returns an error if memory pressure is above the configured threshold.
    pub fn enqueue(&self, request: PrefillRequest) -> Result<()> {
        // Check memory pressure before accepting.
        let mem = self.current_memory_utilization();
        if mem > self.config.memory_threshold {
            anyhow::bail!(
                "memory pressure too high ({:.1}% > {:.1}% threshold); \
                 rejecting prefill request {}",
                mem * 100.0,
                self.config.memory_threshold * 100.0,
                request.request_id
            );
        }

        let mut queue = self
            .queue
            .lock()
            .map_err(|e| anyhow::anyhow!("queue lock poisoned: {e}"))?;
        queue.push(PrioritizedRequest(request));
        self.total_enqueued.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Dequeue the next prefill request if concurrency allows.
    ///
    /// Returns `None` if the queue is empty or the concurrency limit
    /// has been reached.
    ///
    /// This method atomically increments the active prefill count before
    /// returning the request. Callers must NOT also call `mark_prefill_started`
    /// for requests obtained via `try_dequeue` -- use `mark_prefill_started`
    /// only for externally-initiated prefills not sourced from the queue.
    pub fn try_dequeue(&self) -> Option<PrefillRequest> {
        // Acquire the queue lock first to serialize concurrency checks
        // with active count increments, preventing TOCTOU races.
        let mut queue = self.queue.lock().ok()?;

        // Check concurrency limit under the lock.
        let active = self.active_prefills.load(Ordering::SeqCst);
        if active >= self.config.max_concurrent_prefills {
            return None;
        }

        // Check memory pressure.
        if self.current_memory_utilization() > self.config.memory_threshold {
            return None;
        }

        if let Some(prioritized) = queue.pop() {
            self.active_prefills.fetch_add(1, Ordering::SeqCst);
            Some(prioritized.0)
        } else {
            None
        }
    }

    /// Mark a prefill operation as started (increment active count).
    ///
    /// Called when a prefill forward pass begins for an externally-initiated
    /// prefill (not dequeued via `try_dequeue`, which already increments the
    /// count). Returns the new active count.
    pub fn mark_prefill_started(&self) -> usize {
        self.active_prefills.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Mark a prefill operation as completed (decrement active count).
    ///
    /// Called after the prefill forward pass finishes and the first token
    /// is sampled. Returns the new active count. Safe against underflow:
    /// if the count is already zero, it remains zero.
    pub fn mark_prefill_completed(&self) -> usize {
        loop {
            let current = self.active_prefills.load(Ordering::SeqCst);
            if current == 0 {
                // Already at zero; do not underflow.
                self.total_completed.fetch_add(1, Ordering::SeqCst);
                return 0;
            }
            match self.active_prefills.compare_exchange(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    self.total_completed.fetch_add(1, Ordering::SeqCst);
                    return current - 1;
                }
                Err(_) => continue, // Retry on contention.
            }
        }
    }

    /// Return the number of requests waiting in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Return the number of currently active prefill operations.
    pub fn active_count(&self) -> usize {
        self.active_prefills.load(Ordering::SeqCst)
    }

    /// Check whether the scheduler can accept another prefill.
    pub fn can_accept(&self) -> bool {
        let active = self.active_prefills.load(Ordering::SeqCst);
        active < self.config.max_concurrent_prefills
            && self.current_memory_utilization() <= self.config.memory_threshold
    }

    /// Check if this scheduler should skip the decode loop.
    ///
    /// Always returns `true` -- prefill nodes never enter the decode loop.
    /// This is the key behavioral difference from a standard scheduler.
    pub fn should_skip_decode(&self) -> bool {
        true
    }

    // ── Memory Management ────────────────────────────────────────────

    /// Update the current memory utilization (0.0 -- 1.0).
    ///
    /// Called by the runtime to report GPU/system memory usage.
    pub fn update_memory_utilization(&self, utilization: f64) {
        let clamped = utilization.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.memory_utilization.store(bits, Ordering::SeqCst);
    }

    /// Return the current memory utilization (0.0 -- 1.0).
    pub fn current_memory_utilization(&self) -> f64 {
        let bits = self.memory_utilization.load(Ordering::SeqCst);
        f64::from_bits(bits)
    }

    /// Check whether memory pressure exceeds the threshold.
    pub fn is_memory_pressure_high(&self) -> bool {
        self.current_memory_utilization() > self.config.memory_threshold
    }

    // ── Handoff Management ───────────────────────────────────────────

    /// Initiate a handoff of a prefill result to a decode node.
    ///
    /// Creates a `PrefillHandoff` record and adds it to the active
    /// handoffs list. The actual transfer is performed by the caller
    /// using a [`HandoffProtocol`] implementation.
    pub fn initiate_handoff(
        &self,
        result: PrefillResult,
        decode_node_id: String,
    ) -> Result<PrefillHandoff> {
        let handoff = PrefillHandoff::new(result, decode_node_id);
        self.total_handoffs.fetch_add(1, Ordering::SeqCst);

        let mut handoffs = self
            .active_handoffs
            .lock()
            .map_err(|e| anyhow::anyhow!("handoff lock poisoned: {e}"))?;
        handoffs.push(handoff.clone());

        Ok(handoff)
    }

    /// Mark a handoff as acknowledged (successfully received by decode node).
    ///
    /// Removes the handoff from the active list. After acknowledgment,
    /// the caller should free the KV cache memory.
    pub fn acknowledge_handoff(&self, request_id: &RequestId) -> Result<()> {
        let mut handoffs = self
            .active_handoffs
            .lock()
            .map_err(|e| anyhow::anyhow!("handoff lock poisoned: {e}"))?;

        if let Some(pos) = handoffs.iter().position(|h| h.request_id == *request_id) {
            handoffs.remove(pos);
            Ok(())
        } else {
            anyhow::bail!("handoff not found for request {request_id}")
        }
    }

    /// Mark a handoff as failed.
    ///
    /// Updates the status and increments the retry counter. Returns
    /// `true` if the handoff can be retried, `false` if retries are
    /// exhausted.
    pub fn fail_handoff(&self, request_id: &RequestId, reason: &str) -> Result<bool> {
        self.total_handoff_failures.fetch_add(1, Ordering::SeqCst);

        let mut handoffs = self
            .active_handoffs
            .lock()
            .map_err(|e| anyhow::anyhow!("handoff lock poisoned: {e}"))?;

        let pos = handoffs.iter().position(|h| h.request_id == *request_id);

        match pos {
            Some(idx) => {
                handoffs[idx].retry_count += 1;
                handoffs[idx].status = HandoffStatus::Failed {
                    reason: reason.to_string(),
                };

                if handoffs[idx].exceeded_retries(self.config.max_handoff_retries) {
                    // Remove from active list -- caller should handle the failure.
                    handoffs.remove(idx);
                    Ok(false)
                } else {
                    // Reset status to Pending for retry.
                    handoffs[idx].status = HandoffStatus::Pending;
                    Ok(true)
                }
            }
            None => {
                anyhow::bail!("handoff not found for request {request_id}")
            }
        }
    }

    /// Return the number of active (in-flight) handoffs.
    pub fn active_handoff_count(&self) -> usize {
        self.active_handoffs.lock().map(|h| h.len()).unwrap_or(0)
    }

    /// Collect timed-out handoffs and return their request IDs.
    ///
    /// Timed-out handoffs are removed from the active list.
    pub fn collect_timed_out_handoffs(&self) -> Vec<RequestId> {
        let mut handoffs = match self.active_handoffs.lock() {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let timeout = self.config.transfer_timeout;
        let mut timed_out = Vec::new();

        handoffs.retain(|h| {
            if h.is_timed_out(timeout) {
                timed_out.push(h.request_id.clone());
                false
            } else {
                true
            }
        });

        if !timed_out.is_empty() {
            self.total_handoff_failures
                .fetch_add(timed_out.len() as u64, Ordering::SeqCst);
        }

        timed_out
    }

    // ── Chunked Prefill ──────────────────────────────────────────────

    /// Create a chunked prefill coordinator for a request.
    ///
    /// Returns `None` if chunked prefill is disabled or the prompt is
    /// shorter than one chunk.
    pub fn create_chunked_coordinator(
        &self,
        prompt_len: usize,
        total_layers: usize,
    ) -> Option<ChunkedPrefillCoordinator> {
        if !self.config.chunked_prefill_enabled {
            return None;
        }
        if prompt_len <= self.config.chunk_size_tokens {
            return None;
        }

        Some(ChunkedPrefillCoordinator::new(
            prompt_len,
            self.config.chunk_size_tokens,
            total_layers,
        ))
    }

    // ── Statistics ───────────────────────────────────────────────────

    /// Return the total number of requests enqueued since creation.
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued.load(Ordering::SeqCst)
    }

    /// Return the total number of prefills completed since creation.
    pub fn total_completed(&self) -> u64 {
        self.total_completed.load(Ordering::SeqCst)
    }

    /// Return the total number of handoffs initiated since creation.
    pub fn total_handoffs(&self) -> u64 {
        self.total_handoffs.load(Ordering::SeqCst)
    }

    /// Return the total number of handoff failures since creation.
    pub fn total_handoff_failures(&self) -> u64 {
        self.total_handoff_failures.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[path = "prefill_scheduler_tests.rs"]
mod tests;
