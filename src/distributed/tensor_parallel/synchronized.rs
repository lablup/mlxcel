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

//! Core types for tensor-parallel synchronized execution.
//!
//! All TP ranks must execute the **exact same operations in lockstep**. This
//! module defines the protocol types that enable coordinated execution:
//!
//! - [`TPExecutionConfig`] — per-rank execution parameters
//! - [`StepDecision`] — scheduling decision broadcast from rank 0
//! - [`SamplingMode`] — how token sampling is parallelized
//! - [`TPBarrier`] — synchronization barrier with timeout and deadlock prevention
//! - [`RankStatus`] — individual rank health state
//! - [`TPGroupHealth`] — aggregate health of the TP group
//!
//! Used by: tp_scheduler (rank 0), tp_executor (all ranks)

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Unique identifier for a sequence within the TP group.
pub type SequenceId = u64;

/// Unique identifier for a TP execution step.
pub type StepId = u64;

// ---------------------------------------------------------------------------
// Execution configuration
// ---------------------------------------------------------------------------

/// Per-rank configuration for tensor-parallel synchronized execution.
#[derive(Debug, Clone, PartialEq)]
pub struct TPExecutionConfig {
    /// This rank's index (0-based).
    pub tp_rank: usize,
    /// Total number of TP ranks.
    pub tp_size: usize,
    /// Whether this rank is the coordinator (rank 0).
    pub is_coordinator: bool,
    /// Maximum time to wait at a barrier before declaring a deadlock.
    pub barrier_timeout: Duration,
    /// Maximum time to wait for a single step to complete.
    pub step_timeout: Duration,
    /// Maximum number of sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum sequence length before forced eviction.
    pub max_seq_len: usize,
}

impl TPExecutionConfig {
    /// Create a config for the given rank and total size.
    pub fn new(tp_rank: usize, tp_size: usize) -> Self {
        Self {
            tp_rank,
            tp_size,
            is_coordinator: tp_rank == 0,
            barrier_timeout: Duration::from_secs(30),
            step_timeout: Duration::from_secs(60),
            max_batch_size: 32,
            max_seq_len: 8192,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.tp_size >= 1,
            "tp_size must be >= 1, got {}",
            self.tp_size
        );
        ensure!(
            self.tp_rank < self.tp_size,
            "tp_rank {} out of range for tp_size {}",
            self.tp_rank,
            self.tp_size
        );
        ensure!(
            self.is_coordinator == (self.tp_rank == 0),
            "is_coordinator must match tp_rank == 0"
        );
        ensure!(self.max_batch_size >= 1, "max_batch_size must be >= 1");
        ensure!(self.max_seq_len >= 1, "max_seq_len must be >= 1");
        Ok(())
    }
}

impl Default for TPExecutionConfig {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

// ---------------------------------------------------------------------------
// Step decisions (broadcast from rank 0 to all ranks)
// ---------------------------------------------------------------------------

/// A scheduling decision made by rank 0 and broadcast to all ranks.
/// Every rank receives the same `StepDecision` and must execute it identically,
/// ensuring lockstep execution across the TP group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepDecision {
    /// Process prefill (prompt encoding) for the given sequences in order.
    Prefill {
        step_id: StepId,
        seq_ids: Vec<SequenceId>,
        token_counts: Vec<usize>,
    },
    /// Process decode (autoregressive generation) for the given sequences.
    Decode {
        step_id: StepId,
        seq_ids: Vec<SequenceId>,
    },
    /// Admit a new sequence — all ranks allocate KV cache.
    AdmitSequence {
        step_id: StepId,
        seq_id: SequenceId,
        prompt_len: usize,
    },
    /// Evict a sequence — all ranks free KV cache.
    EvictSequence {
        step_id: StepId,
        seq_id: SequenceId,
        reason: EvictionReason,
    },
    /// Synchronization barrier — all ranks must acknowledge before proceeding.
    Barrier { step_id: StepId },
    /// Graceful shutdown — all ranks terminate after receiving this.
    Shutdown { step_id: StepId },
}

impl StepDecision {
    /// Extract the step ID from any decision variant.
    pub fn step_id(&self) -> StepId {
        match self {
            Self::Prefill { step_id, .. }
            | Self::Decode { step_id, .. }
            | Self::AdmitSequence { step_id, .. }
            | Self::EvictSequence { step_id, .. }
            | Self::Barrier { step_id }
            | Self::Shutdown { step_id } => *step_id,
        }
    }

    /// Human-readable name for the decision kind.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Prefill { .. } => "prefill",
            Self::Decode { .. } => "decode",
            Self::AdmitSequence { .. } => "admit_sequence",
            Self::EvictSequence { .. } => "evict_sequence",
            Self::Barrier { .. } => "barrier",
            Self::Shutdown { .. } => "shutdown",
        }
    }
}

impl fmt::Display for StepDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prefill {
                step_id,
                seq_ids,
                token_counts,
            } => {
                write!(
                    f,
                    "Prefill(step={step_id}, seqs={seq_ids:?}, tokens={token_counts:?})"
                )
            }
            Self::Decode { step_id, seq_ids } => {
                write!(f, "Decode(step={step_id}, seqs={seq_ids:?})")
            }
            Self::AdmitSequence {
                step_id,
                seq_id,
                prompt_len,
            } => {
                write!(f, "Admit(step={step_id}, seq={seq_id}, len={prompt_len})")
            }
            Self::EvictSequence {
                step_id,
                seq_id,
                reason,
            } => {
                write!(f, "Evict(step={step_id}, seq={seq_id}, reason={reason})")
            }
            Self::Barrier { step_id } => write!(f, "Barrier(step={step_id})"),
            Self::Shutdown { step_id } => write!(f, "Shutdown(step={step_id})"),
        }
    }
}

/// Reason for evicting a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EvictionReason {
    /// Sequence reached the end-of-sequence token.
    EndOfSequence,
    /// Sequence reached the maximum length.
    MaxLengthReached,
    /// Evicted due to memory pressure.
    MemoryPressure,
    /// Client cancelled the request.
    Cancelled,
    /// Preempted for a higher-priority sequence.
    Preempted,
    /// Error during processing.
    Error,
}

impl fmt::Display for EvictionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndOfSequence => write!(f, "eos"),
            Self::MaxLengthReached => write!(f, "max_length"),
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Preempted => write!(f, "preempted"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling mode
// ---------------------------------------------------------------------------

/// How token sampling is parallelized across TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SamplingMode {
    /// Each rank holds a slice of the vocabulary logits. Rank 0 gathers all
    /// slices via all-gather, samples from the full logit vector, and broadcasts
    /// the chosen token to all ranks.
    VocabParallel,

    /// The LM head is replicated on every rank, so each rank has full logits.
    /// Rank 0 samples and broadcasts the token to all ranks. This avoids the
    /// all-gather but uses more memory for the LM head weights.
    ReplicatedLmHead,
}

impl fmt::Display for SamplingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VocabParallel => write!(f, "vocab_parallel"),
            Self::ReplicatedLmHead => write!(f, "replicated_lm_head"),
        }
    }
}

// ---------------------------------------------------------------------------
// Barrier synchronization
// ---------------------------------------------------------------------------

/// Synchronization barrier with timeout for TP ranks. Tracks which ranks
/// have arrived and detects stragglers that exceed the timeout.
#[derive(Debug, Clone)]
pub struct TPBarrier {
    step_id: StepId,
    tp_size: usize,
    arrived: HashMap<usize, Instant>,
    created_at: Instant,
    timeout: Duration,
}

impl TPBarrier {
    /// Create a new barrier for the given step.
    pub fn new(step_id: StepId, tp_size: usize, timeout: Duration) -> Self {
        Self {
            step_id,
            tp_size,
            arrived: HashMap::with_capacity(tp_size),
            created_at: Instant::now(),
            timeout,
        }
    }

    /// Record a rank arriving at the barrier.
    ///
    /// Returns `Ok(true)` if all ranks have arrived, `Ok(false)` if still waiting.
    /// A rank arriving twice is a protocol violation and returns an error.
    pub fn arrive(&mut self, rank: usize) -> Result<bool> {
        ensure!(
            rank < self.tp_size,
            "rank {rank} out of range for tp_size {}",
            self.tp_size
        );
        ensure!(
            !self.arrived.contains_key(&rank),
            "rank {rank} already arrived at barrier for step {}",
            self.step_id
        );
        self.arrived.insert(rank, Instant::now());
        Ok(self.arrived.len() == self.tp_size)
    }

    /// Check whether the barrier has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.created_at.elapsed() > self.timeout
    }

    /// How long since the barrier was created.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// The step ID this barrier is for.
    pub fn step_id(&self) -> StepId {
        self.step_id
    }

    /// Number of ranks that have arrived.
    pub fn arrived_count(&self) -> usize {
        self.arrived.len()
    }

    /// Whether all ranks have arrived.
    pub fn is_complete(&self) -> bool {
        self.arrived.len() == self.tp_size
    }

    /// Return the ranks that have NOT yet arrived (stragglers).
    pub fn missing_ranks(&self) -> Vec<usize> {
        (0..self.tp_size)
            .filter(|r| !self.arrived.contains_key(r))
            .collect()
    }

    /// Return the ranks that have arrived.
    pub fn arrived_ranks(&self) -> Vec<usize> {
        let mut ranks: Vec<usize> = self.arrived.keys().copied().collect();
        ranks.sort_unstable();
        ranks
    }
}

// ---------------------------------------------------------------------------
// Rank and group health
// ---------------------------------------------------------------------------

/// Health status of a single TP rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RankStatus {
    /// Rank is ready and waiting for the next step decision.
    Ready,
    /// Rank is currently executing a step.
    Executing,
    /// Rank has completed its current step and is waiting at the barrier.
    WaitingAtBarrier,
    /// Rank has failed and cannot continue.
    Failed,
    /// Rank has been shut down gracefully.
    ShutDown,
}

impl fmt::Display for RankStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "ready"),
            Self::Executing => write!(f, "executing"),
            Self::WaitingAtBarrier => write!(f, "waiting_at_barrier"),
            Self::Failed => write!(f, "failed"),
            Self::ShutDown => write!(f, "shut_down"),
        }
    }
}

/// Aggregate health tracking for the entire TP group. TP cannot tolerate
/// rank failures: if any rank fails, the entire group must be stopped.
#[derive(Debug, Clone)]
pub struct TPGroupHealth {
    statuses: HashMap<usize, RankStatus>,
    tp_size: usize,
    last_heartbeat: HashMap<usize, Instant>,
    heartbeat_timeout: Duration,
}

impl TPGroupHealth {
    /// Create a new health tracker with all ranks initially `Ready`.
    pub fn new(tp_size: usize, heartbeat_timeout: Duration) -> Self {
        let now = Instant::now();
        let statuses = (0..tp_size).map(|r| (r, RankStatus::Ready)).collect();
        let last_heartbeat = (0..tp_size).map(|r| (r, now)).collect();
        Self {
            statuses,
            tp_size,
            last_heartbeat,
            heartbeat_timeout,
        }
    }

    /// Update a rank's status and refresh its heartbeat timestamp.
    pub fn update_status(&mut self, rank: usize, status: RankStatus) {
        if rank < self.tp_size {
            self.statuses.insert(rank, status);
            self.last_heartbeat.insert(rank, Instant::now());
        }
    }

    /// Record a heartbeat from a rank (without changing its status).
    pub fn heartbeat(&mut self, rank: usize) {
        if rank < self.tp_size {
            self.last_heartbeat.insert(rank, Instant::now());
        }
    }

    /// Get the status of a specific rank.
    pub fn rank_status(&self, rank: usize) -> Option<RankStatus> {
        self.statuses.get(&rank).copied()
    }

    /// Check if the TP group is healthy (no failed ranks, no heartbeat timeouts).
    pub fn is_healthy(&self) -> bool {
        let now = Instant::now();
        for rank in 0..self.tp_size {
            if let Some(&status) = self.statuses.get(&rank)
                && status == RankStatus::Failed
            {
                return false;
            }
            if let Some(&last) = self.last_heartbeat.get(&rank)
                && now.duration_since(last) > self.heartbeat_timeout
            {
                return false;
            }
        }
        true
    }

    /// Return the list of failed or timed-out ranks.
    pub fn unhealthy_ranks(&self) -> Vec<usize> {
        let now = Instant::now();
        let mut result = Vec::new();
        for rank in 0..self.tp_size {
            let failed = self
                .statuses
                .get(&rank)
                .map(|s| *s == RankStatus::Failed)
                .unwrap_or(true);
            let timed_out = self
                .last_heartbeat
                .get(&rank)
                .map(|t| now.duration_since(*t) > self.heartbeat_timeout)
                .unwrap_or(true);
            if failed || timed_out {
                result.push(rank);
            }
        }
        result
    }

    /// Check if all ranks are in the `Ready` state.
    pub fn all_ready(&self) -> bool {
        (0..self.tp_size).all(|r| {
            self.statuses
                .get(&r)
                .map(|s| *s == RankStatus::Ready)
                .unwrap_or(false)
        })
    }

    /// Check if all ranks are shut down.
    pub fn all_shut_down(&self) -> bool {
        (0..self.tp_size).all(|r| {
            self.statuses
                .get(&r)
                .map(|s| *s == RankStatus::ShutDown)
                .unwrap_or(false)
        })
    }

    /// Total number of ranks in the group.
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }
}

// ---------------------------------------------------------------------------
// Token broadcast result
// ---------------------------------------------------------------------------

/// Result of a synchronized sampling step, broadcast from rank 0 to all ranks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampledTokens {
    /// Step ID this sampling result is for.
    pub step_id: StepId,
    /// Sampled token IDs, one per active sequence (same order as the decode batch).
    pub tokens: Vec<u32>,
    /// Sequences that completed (hit EOS or max length) in this step.
    pub completed_seq_ids: Vec<SequenceId>,
}

#[cfg(test)]
#[path = "synchronized_tests.rs"]
mod tests;
