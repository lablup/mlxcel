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

//! Decode node scheduler for disaggregated inference.
//!
//! In disaggregated serving, the decode node is responsible only for token
//! generation -- it receives pre-computed KV caches from prefill nodes and
//! generates tokens without re-running prefill.
//!
//! # Key Design Decisions
//!
//! - **No prefill phase**: The scheduler always skips prefill; sequences enter
//!   the active batch directly from the ingestion queue.
//! - **Async cache ingestion**: Incoming KV caches are queued asynchronously
//!   and do NOT block ongoing decode steps.
//! - **Memory-aware admission**: New sequences are admitted only when
//!   sufficient memory exists, preventing OOM under concurrent load.
//! - **Dynamic batch sizing**: The active batch grows and shrinks based on
//!   available memory and active sequence count.
//! - **Completion notification**: When sequences finish (EOS, max_tokens, or
//!   error), the scheduler emits completion events for the request router.
//!
//! Used by: disaggregated serving pipeline, Scheduler

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::distributed::kv_cache_serde::types::{
    SerializableCacheState, SerializableSamplingState,
};
use crate::distributed::request_tracker::RequestId;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the decode node scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeSchedulerConfig {
    /// Maximum number of sequences that can be decoded simultaneously.
    /// Bounded by GPU memory and compute capacity.
    pub max_batch_size: usize,

    /// Maximum total sequences managed (active + queued).
    /// Prevents unbounded queue growth under sustained load.
    pub max_sequences: usize,

    /// Memory utilization threshold (0.0 -- 1.0) above which new
    /// sequences are rejected. Prevents OOM under concurrent load.
    pub memory_threshold: f64,

    /// Maximum number of pending ingestion requests before the queue
    /// rejects new arrivals. Provides backpressure to prefill nodes.
    pub ingestion_queue_size: usize,
}

impl Default for DecodeSchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequences: 128,
            memory_threshold: 0.85,
            ingestion_queue_size: 64,
        }
    }
}

// ── Request and Sequence Types ───────────────────────────────────────

/// A decode request received from a prefill node.
///
/// Contains the first sampled token, the serialized KV cache state,
/// and sampling parameters needed to continue token generation.
#[derive(Debug, Clone)]
pub struct DecodeRequest {
    /// Unique request identifier (shared with the distributed tracker).
    pub request_id: RequestId,

    /// The first token sampled by the prefill node.
    pub first_token: i32,

    /// Serialized KV cache state from the prefill node.
    pub cache_state: SerializableCacheState,

    /// Sampling parameters for decode continuation.
    pub sampling_params: SerializableSamplingState,

    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Timestamp when the request was received.
    pub received_at: Instant,
}

impl DecodeRequest {
    /// Create a new decode request.
    pub fn new(
        request_id: RequestId,
        first_token: i32,
        cache_state: SerializableCacheState,
        sampling_params: SerializableSamplingState,
        max_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            first_token,
            cache_state,
            sampling_params,
            max_tokens,
            received_at: Instant::now(),
        }
    }

    /// Return the prompt length from the cache metadata.
    pub fn prompt_len(&self) -> usize {
        self.cache_state.metadata.prompt_len
    }
}

/// Status of a sequence in the decode scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SequenceStatus {
    /// Waiting in the ingestion queue for admission.
    Queued,
    /// Actively being decoded in the current batch.
    Decoding,
    /// Completed successfully (EOS or max_tokens reached).
    Completed,
    /// Failed due to an error.
    Failed { reason: String },
}

impl fmt::Display for SequenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Decoding => write!(f, "decoding"),
            Self::Completed => write!(f, "completed"),
            Self::Failed { reason } => write!(f, "failed: {reason}"),
        }
    }
}

/// Reason a sequence completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CompletionReason {
    /// End-of-sequence token was generated.
    Eos,
    /// Maximum token count reached.
    MaxTokens,
    /// Stopped by a user-specified stop token.
    StopToken { token_id: i32 },
    /// An error occurred during decoding.
    Error { reason: String },
}

impl fmt::Display for CompletionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eos => write!(f, "eos"),
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::StopToken { token_id } => write!(f, "stop_token({token_id})"),
            Self::Error { reason } => write!(f, "error: {reason}"),
        }
    }
}

/// A sequence actively being decoded or waiting in the queue.
#[derive(Debug, Clone)]
pub struct DecodeSequence {
    /// Unique request identifier.
    pub request_id: RequestId,

    /// Number of tokens generated so far (excluding the first token from prefill).
    pub tokens_generated: usize,

    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Whether this sequence has completed.
    pub is_complete: bool,

    /// Assigned cache slot identifier (set upon admission to active batch).
    pub cache_slot_id: Option<u64>,

    /// Current status.
    pub status: SequenceStatus,

    /// Timestamp when the sequence was admitted to the active batch.
    pub admitted_at: Option<Instant>,
}

impl DecodeSequence {
    /// Create a new decode sequence from a decode request.
    fn from_request(request: &DecodeRequest) -> Self {
        Self {
            request_id: request.request_id.clone(),
            tokens_generated: 0,
            max_tokens: request.max_tokens,
            is_complete: false,
            cache_slot_id: None,
            status: SequenceStatus::Queued,
            admitted_at: None,
        }
    }

    /// Check if the sequence has reached its token limit.
    pub fn has_reached_limit(&self) -> bool {
        self.tokens_generated >= self.max_tokens
    }

    /// Record a generated token. Returns `true` if the sequence has
    /// now reached its token limit.
    pub fn record_token(&mut self) -> bool {
        self.tokens_generated += 1;
        self.has_reached_limit()
    }

    /// Return the decode throughput if the sequence has been admitted.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let admitted = self.admitted_at?;
        let elapsed = admitted.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            return None;
        }
        Some(self.tokens_generated as f64 / elapsed)
    }
}

// ── Completion Notification ──────────────────────────────────────────

/// A completion event emitted when a sequence finishes.
///
/// The request router uses these events to send responses back to clients
/// and free upstream resources.
#[derive(Debug, Clone)]
pub struct CompletionEvent {
    /// Request ID of the completed sequence.
    pub request_id: RequestId,

    /// Why the sequence completed.
    pub reason: CompletionReason,

    /// Total tokens generated (excluding the prefill first token).
    pub tokens_generated: usize,

    /// Cache slot that was freed (if any).
    pub freed_cache_slot: Option<u64>,
}

/// Trait for receiving sequence completion notifications.
///
/// Implementations send completion events to the request router,
/// metrics collector, or other interested consumers.
pub trait CompletionNotifier: Send + Sync {
    /// Called when a sequence completes. Implementations should not block.
    fn notify_completion(&self, event: CompletionEvent);
}

// ── Ingestion Statistics ─────────────────────────────────────────────

/// Statistics for cache ingestion and decode operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestionStats {
    /// Total KV caches ingested (accepted into the queue).
    pub total_ingested: u64,
    /// Total KV caches rejected (queue full or memory pressure).
    pub total_rejected: u64,
    /// Total sequences completed (any reason).
    pub total_completed: u64,
    /// Total sequences failed.
    pub total_failed: u64,
    /// Total tokens generated across all sequences.
    pub total_tokens_generated: u64,
}

// ── Decode Scheduler ─────────────────────────────────────────────────

/// The decode node scheduler for disaggregated inference.
///
/// Manages an ingestion queue for incoming KV caches from prefill nodes,
/// an active batch of currently-decoding sequences, and memory-aware
/// admission control.
///
/// # Architecture
///
/// ```text
/// Prefill      ┌──────────────────────┐     Completion
/// Nodes  ----> │   DecodeScheduler    │ ----> Router
///              │                      │
///              │  - Ingestion Queue   │
///              │  - Active Batch      │
///              │  - Memory Monitor    │
///              │  - Completion Notify │
///              └──────────────────────┘
/// ```
///
/// Used by: disaggregated serving pipeline, server model worker
pub struct DecodeScheduler {
    /// Scheduler configuration.
    config: DecodeSchedulerConfig,

    /// Bounded ingestion queue for incoming KV caches from prefill nodes.
    /// New requests are appended to the back; admission pops from the front.
    ingestion_queue: Mutex<VecDeque<DecodeRequest>>,

    /// Active batch: sequences currently being decoded.
    /// Keyed by request ID for O(1) lookup.
    active_batch: Mutex<HashMap<RequestId, DecodeSequence>>,

    /// Next cache slot ID to assign.
    next_cache_slot: AtomicU64,

    /// Current memory utilization (0.0 -- 1.0), updated externally.
    memory_utilization: AtomicU64,

    /// Total ingested count.
    total_ingested: AtomicU64,

    /// Total rejected count.
    total_rejected: AtomicU64,

    /// Total completed count.
    total_completed: AtomicU64,

    /// Total failed count.
    total_failed: AtomicU64,

    /// Total tokens generated across all sequences.
    total_tokens_generated: AtomicU64,

    /// Completed events waiting to be drained by the router.
    completion_events: Mutex<Vec<CompletionEvent>>,
}

impl fmt::Debug for DecodeScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodeScheduler")
            .field("max_batch_size", &self.config.max_batch_size)
            .field("active_batch_size", &self.active_batch_size())
            .field("queued", &self.ingestion_queue_len())
            .field("memory_threshold", &self.config.memory_threshold)
            .finish()
    }
}

impl DecodeScheduler {
    /// Create a new decode scheduler with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `max_batch_size` is 0, `max_sequences` is 0,
    /// `ingestion_queue_size` is 0, or `memory_threshold` is not
    /// in `(0.0, 1.0]`.
    pub fn new(config: DecodeSchedulerConfig) -> Self {
        assert!(config.max_batch_size > 0, "max_batch_size must be > 0");
        assert!(config.max_sequences > 0, "max_sequences must be > 0");
        assert!(
            config.ingestion_queue_size > 0,
            "ingestion_queue_size must be > 0"
        );
        assert!(
            config.memory_threshold > 0.0 && config.memory_threshold <= 1.0,
            "memory_threshold must be in (0.0, 1.0], got {}",
            config.memory_threshold
        );
        Self {
            config,
            ingestion_queue: Mutex::new(VecDeque::new()),
            active_batch: Mutex::new(HashMap::new()),
            next_cache_slot: AtomicU64::new(1),
            memory_utilization: AtomicU64::new(0),
            total_ingested: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            completion_events: Mutex::new(Vec::new()),
        }
    }

    /// Return the scheduler configuration.
    pub fn config(&self) -> &DecodeSchedulerConfig {
        &self.config
    }

    // ── Cache Ingestion ─────────────────────────────────────────────

    /// Ingest a KV cache from a prefill node.
    ///
    /// The request is placed in the bounded ingestion queue. This method
    /// does NOT block ongoing decode steps -- it only acquires the queue
    /// lock briefly to push the request.
    ///
    /// Returns an error if:
    /// - The ingestion queue is full (backpressure).
    /// - Memory pressure exceeds the configured threshold.
    /// - The total sequence count (queued + active) would exceed `max_sequences`.
    #[must_use = "caller should check whether the ingestion was rejected"]
    pub fn ingest_cache(&self, request: DecodeRequest) -> Result<()> {
        // Check memory pressure first (best-effort; not under lock).
        let mem = self.current_memory_utilization();
        if mem > self.config.memory_threshold {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "memory pressure too high ({:.1}% > {:.1}% threshold); \
                 rejecting decode request {}",
                mem * 100.0,
                self.config.memory_threshold * 100.0,
                request.request_id
            );
        }

        // Acquire both locks in consistent order (queue, then batch) to
        // get an atomic view of total sequence count.
        let mut queue = self
            .ingestion_queue
            .lock()
            .map_err(|e| anyhow::anyhow!("ingestion queue lock poisoned: {e}"))?;

        // Check queue capacity.
        if queue.len() >= self.config.ingestion_queue_size {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "ingestion queue full ({}/{} slots); rejecting decode request {}",
                queue.len(),
                self.config.ingestion_queue_size,
                request.request_id
            );
        }

        // Check total sequence count (queued + active) atomically by
        // holding both locks, preventing TOCTOU races on the count.
        let batch = self
            .active_batch
            .lock()
            .map_err(|e| anyhow::anyhow!("active batch lock poisoned: {e}"))?;
        let total = queue.len() + batch.len();
        drop(batch);

        if total >= self.config.max_sequences {
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!(
                "total sequence limit reached ({}/{} sequences); \
                 rejecting decode request {}",
                total,
                self.config.max_sequences,
                request.request_id
            );
        }

        queue.push_back(request);
        self.total_ingested.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // ── Batch Admission ─────────────────────────────────────────────

    /// Admit sequences from the ingestion queue into the active batch.
    ///
    /// Moves sequences from the front of the queue into the active batch,
    /// up to the available capacity (considering `max_batch_size` and
    /// memory pressure).
    ///
    /// Returns the list of newly admitted sequences (with their assigned
    /// cache slot IDs).
    pub fn admit_sequences(&self) -> Vec<DecodeSequence> {
        let capacity = self.available_capacity();
        if capacity == 0 {
            return Vec::new();
        }

        let mut queue = match self.ingestion_queue.lock() {
            Ok(q) => q,
            Err(e) => {
                eprintln!(
                    "[decode_scheduler] ingestion queue lock poisoned in admit_sequences: {e}"
                );
                return Vec::new();
            }
        };

        let mut batch = match self.active_batch.lock() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[decode_scheduler] active batch lock poisoned in admit_sequences: {e}");
                return Vec::new();
            }
        };

        let to_admit = capacity.min(queue.len());
        let mut admitted = Vec::with_capacity(to_admit);

        for _ in 0..to_admit {
            if let Some(request) = queue.pop_front() {
                let mut seq = DecodeSequence::from_request(&request);
                let slot_id = self.next_cache_slot.fetch_add(1, Ordering::Relaxed);
                seq.cache_slot_id = Some(slot_id);
                seq.status = SequenceStatus::Decoding;
                seq.admitted_at = Some(Instant::now());

                batch.insert(seq.request_id.clone(), seq.clone());
                admitted.push(seq);
            }
        }

        admitted
    }

    // ── Sequence Completion ─────────────────────────────────────────

    /// Mark a sequence as complete and free its cache slot.
    ///
    /// Removes the sequence from the active batch, records a completion
    /// event, and updates statistics.
    ///
    /// Returns an error if the request ID is not found in the active batch.
    #[must_use = "caller should check for completion errors"]
    pub fn mark_sequence_complete(
        &self,
        request_id: &RequestId,
        reason: CompletionReason,
    ) -> Result<()> {
        let mut batch = self
            .active_batch
            .lock()
            .map_err(|e| anyhow::anyhow!("active batch lock poisoned: {e}"))?;

        let seq = batch
            .remove(request_id)
            .ok_or_else(|| anyhow::anyhow!("sequence not found in active batch: {request_id}"))?;

        let is_failure = matches!(reason, CompletionReason::Error { .. });

        let event = CompletionEvent {
            request_id: request_id.clone(),
            reason,
            tokens_generated: seq.tokens_generated,
            freed_cache_slot: seq.cache_slot_id,
        };

        // Drop the batch lock before acquiring the events lock.
        drop(batch);

        if is_failure {
            self.total_failed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.total_completed.fetch_add(1, Ordering::Relaxed);
        }
        self.total_tokens_generated
            .fetch_add(seq.tokens_generated as u64, Ordering::Relaxed);

        let mut events = self
            .completion_events
            .lock()
            .map_err(|e| anyhow::anyhow!("completion events lock poisoned: {e}"))?;
        events.push(event);

        Ok(())
    }

    /// Record a token generated for a specific sequence.
    ///
    /// Returns `true` if the sequence has now reached its token limit
    /// (caller should mark it complete with `CompletionReason::MaxTokens`).
    ///
    /// Returns an error if the request ID is not found.
    #[must_use = "caller must check if the token limit was reached"]
    pub fn record_token(&self, request_id: &RequestId) -> Result<bool> {
        let mut batch = self
            .active_batch
            .lock()
            .map_err(|e| anyhow::anyhow!("active batch lock poisoned: {e}"))?;

        let seq = batch
            .get_mut(request_id)
            .ok_or_else(|| anyhow::anyhow!("sequence not found in active batch: {request_id}"))?;

        Ok(seq.record_token())
    }

    /// Drain all pending completion events.
    ///
    /// Returns the events accumulated since the last drain. The caller
    /// (typically the request router) uses these to send responses.
    pub fn drain_completion_events(&self) -> Vec<CompletionEvent> {
        match self.completion_events.lock() {
            Ok(mut events) => std::mem::take(&mut *events),
            Err(e) => {
                eprintln!("[decode_scheduler] completion events lock poisoned: {e}");
                Vec::new()
            }
        }
    }

    // ── Capacity and State ──────────────────────────────────────────

    /// Check if this scheduler should skip the prefill phase.
    ///
    /// Always returns `true` -- decode nodes never run prefill.
    /// This is the key behavioral difference from a standard scheduler.
    pub fn should_skip_prefill(&self) -> bool {
        true
    }

    /// Return the number of additional sequences that can be admitted.
    ///
    /// Considers `max_batch_size` and memory pressure.
    pub fn available_capacity(&self) -> usize {
        if self.is_memory_pressure_high() {
            return 0;
        }
        let active = self.active_batch_size();
        self.config.max_batch_size.saturating_sub(active)
    }

    /// Return the number of sequences in the active batch.
    pub fn active_batch_size(&self) -> usize {
        match self.active_batch.lock() {
            Ok(b) => b.len(),
            Err(e) => {
                eprintln!(
                    "[decode_scheduler] active batch lock poisoned in active_batch_size: {e}"
                );
                0
            }
        }
    }

    /// Return the number of requests waiting in the ingestion queue.
    pub fn ingestion_queue_len(&self) -> usize {
        match self.ingestion_queue.lock() {
            Ok(q) => q.len(),
            Err(e) => {
                eprintln!(
                    "[decode_scheduler] ingestion queue lock poisoned in ingestion_queue_len: {e}"
                );
                0
            }
        }
    }

    /// Return a snapshot of the active batch sequence IDs and their statuses.
    pub fn active_sequence_ids(&self) -> Vec<(RequestId, SequenceStatus)> {
        match self.active_batch.lock() {
            Ok(batch) => batch
                .iter()
                .map(|(id, seq)| (id.clone(), seq.status.clone()))
                .collect(),
            Err(e) => {
                eprintln!(
                    "[decode_scheduler] active batch lock poisoned in active_sequence_ids: {e}"
                );
                Vec::new()
            }
        }
    }

    /// Check whether the scheduler can accept another ingestion.
    pub fn can_accept(&self) -> bool {
        if self.is_memory_pressure_high() {
            return false;
        }
        let queue_len = self.ingestion_queue_len();
        let active = self.active_batch_size();
        queue_len < self.config.ingestion_queue_size
            && (queue_len + active) < self.config.max_sequences
    }

    // ── Memory Management ───────────────────────────────────────────

    /// Update the current memory utilization (0.0 -- 1.0).
    ///
    /// Called by the runtime to report GPU/system memory usage.
    pub fn update_memory_utilization(&self, utilization: f64) {
        let clamped = utilization.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.memory_utilization.store(bits, Ordering::Release);
    }

    /// Return the current memory utilization (0.0 -- 1.0).
    pub fn current_memory_utilization(&self) -> f64 {
        let bits = self.memory_utilization.load(Ordering::Acquire);
        f64::from_bits(bits)
    }

    /// Check whether memory pressure exceeds the threshold.
    pub fn is_memory_pressure_high(&self) -> bool {
        self.current_memory_utilization() > self.config.memory_threshold
    }

    // ── Statistics ──────────────────────────────────────────────────

    /// Return a snapshot of the ingestion and decode statistics.
    pub fn stats(&self) -> IngestionStats {
        IngestionStats {
            total_ingested: self.total_ingested.load(Ordering::Relaxed),
            total_rejected: self.total_rejected.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_tokens_generated: self.total_tokens_generated.load(Ordering::Relaxed),
        }
    }

    /// Return the total number of KV caches ingested.
    pub fn total_ingested(&self) -> u64 {
        self.total_ingested.load(Ordering::Relaxed)
    }

    /// Return the total number of KV caches rejected.
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected.load(Ordering::Relaxed)
    }

    /// Return the total number of sequences completed.
    pub fn total_completed(&self) -> u64 {
        self.total_completed.load(Ordering::Relaxed)
    }

    /// Return the total number of sequences that failed.
    pub fn total_failed(&self) -> u64 {
        self.total_failed.load(Ordering::Relaxed)
    }

    /// Return the total number of tokens generated.
    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "decode_scheduler_tests.rs"]
mod tests;
