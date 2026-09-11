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

//! TP Executor — per-rank execution engine for tensor-parallel inference.
//!
//! Every TP rank (including rank 0) runs a `TPExecutor` that receives
//! [`StepDecision`] values from the scheduler broadcast and executes them
//! in lockstep. The executor:
//!
//! - Maintains identical sequence state on every rank
//! - Participates in barrier synchronization
//! - Handles token sampling coordination (rank 0 samples, others receive)
//! - Reports rank health status
//! - Detects and propagates errors to stop the entire TP group
//!
//! Used by: tensor_parallel main inference loop (all ranks)

use std::collections::{HashMap, HashSet};
use std::fmt;

use anyhow::{Result, bail, ensure};

use super::synchronized::{
    EvictionReason, RankStatus, SampledTokens, SamplingMode, SequenceId, StepDecision, StepId,
    TPBarrier, TPExecutionConfig, TPGroupHealth,
};

// ---------------------------------------------------------------------------
// Per-rank sequence tracking
// ---------------------------------------------------------------------------

/// Lightweight per-rank view of a sequence's state.
///
/// Unlike the scheduler's `SequenceInfo`, this only tracks what the executor
/// needs: whether the sequence has cache allocated and its current position.
#[derive(Debug, Clone)]
pub struct ExecutorSequenceState {
    /// Sequence ID.
    pub seq_id: SequenceId,
    /// Whether KV cache has been allocated for this sequence on this rank.
    pub cache_allocated: bool,
    /// Whether this sequence has been prefilled.
    pub prefilled: bool,
    /// Current position (prompt_len + generated tokens).
    pub position: usize,
}

// ---------------------------------------------------------------------------
// Step execution result
// ---------------------------------------------------------------------------

/// Outcome of executing a single step decision on this rank.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepOutcome {
    /// Step executed successfully. Contains partial logits for sampling
    /// (only meaningful for decode steps on rank 0).
    Executed { step_id: StepId },

    /// Barrier reached — waiting for other ranks.
    BarrierReached { step_id: StepId },

    /// Sequence admitted — KV cache allocated.
    SequenceAdmitted { step_id: StepId, seq_id: SequenceId },

    /// Sequence evicted — KV cache freed.
    SequenceEvicted { step_id: StepId, seq_id: SequenceId },

    /// Shutdown acknowledged — rank is terminating.
    ShutdownAcknowledged { step_id: StepId },

    /// Error during execution.
    Error { step_id: StepId, message: String },
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Per-rank execution engine for tensor-parallel inference.
///
/// Each rank creates one `TPExecutor` that processes [`StepDecision`] values
/// received from rank 0's broadcast. All executors must process the same
/// decisions in the same order for lockstep execution.
pub struct TPExecutor {
    /// Execution configuration.
    config: TPExecutionConfig,
    /// How token sampling is parallelized.
    sampling_mode: SamplingMode,
    /// Current rank status.
    status: RankStatus,
    /// Last successfully executed step ID.
    last_executed_step: Option<StepId>,
    /// Per-sequence state on this rank.
    sequences: HashMap<SequenceId, ExecutorSequenceState>,
    /// Active sequence IDs (mirrors the scheduler's active set).
    active_set: HashSet<SequenceId>,
    /// Active barrier (if any).
    current_barrier: Option<TPBarrier>,
    /// TP group health tracker (all ranks track this for consistency checks).
    group_health: TPGroupHealth,
    /// Total steps executed since creation.
    steps_executed: u64,
    /// Total errors encountered.
    error_count: u64,
}

impl TPExecutor {
    /// Create a new executor for the given rank.
    pub fn new(config: TPExecutionConfig, sampling_mode: SamplingMode) -> Result<Self> {
        config.validate()?;
        let tp_size = config.tp_size;
        let heartbeat_timeout = config.barrier_timeout * 2;
        Ok(Self {
            config,
            sampling_mode,
            status: RankStatus::Ready,
            last_executed_step: None,
            sequences: HashMap::new(),
            active_set: HashSet::new(),
            current_barrier: None,
            group_health: TPGroupHealth::new(tp_size, heartbeat_timeout),
            steps_executed: 0,
            error_count: 0,
        })
    }

    /// Get the current rank status.
    pub fn status(&self) -> RankStatus {
        self.status
    }

    /// Get this rank's index.
    pub fn rank(&self) -> usize {
        self.config.tp_rank
    }

    /// Whether this rank is the coordinator.
    pub fn is_coordinator(&self) -> bool {
        self.config.is_coordinator
    }

    /// Get the sampling mode.
    pub fn sampling_mode(&self) -> SamplingMode {
        self.sampling_mode
    }

    /// Number of active sequences on this rank.
    pub fn active_count(&self) -> usize {
        self.active_set.len()
    }

    /// Total steps executed.
    pub fn steps_executed(&self) -> u64 {
        self.steps_executed
    }

    /// Total errors encountered.
    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    /// Get the group health tracker.
    pub fn group_health(&self) -> &TPGroupHealth {
        &self.group_health
    }

    /// Get a mutable reference to the group health tracker.
    pub fn group_health_mut(&mut self) -> &mut TPGroupHealth {
        &mut self.group_health
    }

    // -----------------------------------------------------------------------
    // Step execution
    // -----------------------------------------------------------------------

    /// Execute a step decision received from rank 0. Validates step ordering
    /// and dispatches to the appropriate handler. The actual model forward
    /// pass is NOT performed here — the caller runs it based on `StepOutcome`.
    pub fn execute_step(&mut self, decision: &StepDecision) -> Result<StepOutcome> {
        // Validate step ordering.
        let step_id = decision.step_id();
        if let Some(last) = self.last_executed_step {
            ensure!(
                step_id > last,
                "step ID {step_id} is not greater than last executed step {last}"
            );
        }

        // Check group health before executing.
        if !self.group_health.is_healthy() {
            let unhealthy = self.group_health.unhealthy_ranks();
            self.status = RankStatus::Failed;
            bail!(
                "TP group unhealthy: ranks {unhealthy:?} are failed/timed out; \
                 cannot execute step {step_id}"
            );
        }

        self.status = RankStatus::Executing;
        self.group_health
            .update_status(self.config.tp_rank, RankStatus::Executing);

        let outcome = match decision {
            StepDecision::Prefill {
                step_id,
                seq_ids,
                token_counts,
            } => self.handle_prefill(*step_id, seq_ids, token_counts),
            StepDecision::Decode { step_id, seq_ids } => self.handle_decode(*step_id, seq_ids),
            StepDecision::AdmitSequence {
                step_id,
                seq_id,
                prompt_len,
            } => self.handle_admit(*step_id, *seq_id, *prompt_len),
            StepDecision::EvictSequence {
                step_id,
                seq_id,
                reason,
            } => self.handle_evict(*step_id, *seq_id, *reason),
            StepDecision::Barrier { step_id } => self.handle_barrier(*step_id),
            StepDecision::Shutdown { step_id } => self.handle_shutdown(*step_id),
        };

        match &outcome {
            Ok(step_outcome) => {
                self.last_executed_step = Some(step_id);
                self.steps_executed += 1;
                // Preserve status set by handlers for barrier/shutdown;
                // otherwise transition back to Ready.
                match step_outcome {
                    StepOutcome::BarrierReached { .. }
                    | StepOutcome::ShutdownAcknowledged { .. } => {
                        // Status already set by the handler.
                    }
                    _ => {
                        self.status = RankStatus::Ready;
                        self.group_health
                            .update_status(self.config.tp_rank, RankStatus::Ready);
                    }
                }
            }
            Err(_) => {
                self.error_count += 1;
                self.status = RankStatus::Failed;
                self.group_health
                    .update_status(self.config.tp_rank, RankStatus::Failed);
            }
        }

        outcome
    }

    /// Apply sampled tokens from rank 0, updating local sequence positions.
    ///
    /// Called after rank 0 broadcasts the sampling result. All ranks must
    /// call this with the same `SampledTokens` to stay synchronized.
    pub fn apply_sampled_tokens(&mut self, sampled: &SampledTokens) -> Result<()> {
        // Get the active decoding sequence IDs in order.
        let mut decode_ids: Vec<SequenceId> = self
            .active_set
            .iter()
            .filter(|id| self.sequences.get(id).map(|s| s.prefilled).unwrap_or(false))
            .copied()
            .collect();
        decode_ids.sort_unstable();

        ensure!(
            sampled.tokens.len() == decode_ids.len(),
            "sampled token count ({}) does not match active decode count ({})",
            sampled.tokens.len(),
            decode_ids.len()
        );

        // Advance positions.
        for &seq_id in &decode_ids {
            if let Some(state) = self.sequences.get_mut(&seq_id) {
                state.position += 1;
            }
        }

        // Remove completed sequences.
        for &seq_id in &sampled.completed_seq_ids {
            self.active_set.remove(&seq_id);
            self.sequences.remove(&seq_id);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Decision handlers
    // -----------------------------------------------------------------------

    fn handle_prefill(
        &mut self,
        step_id: StepId,
        seq_ids: &[SequenceId],
        token_counts: &[usize],
    ) -> Result<StepOutcome> {
        ensure!(
            seq_ids.len() == token_counts.len(),
            "seq_ids and token_counts must have the same length"
        );

        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let state = self.sequences.get_mut(&seq_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "sequence {seq_id} not found on rank {}",
                    self.config.tp_rank
                )
            })?;

            ensure!(
                state.cache_allocated,
                "sequence {seq_id} has no cache allocated"
            );

            state.prefilled = true;
            state.position = token_counts[i];
        }

        Ok(StepOutcome::Executed { step_id })
    }

    fn handle_decode(&self, step_id: StepId, seq_ids: &[SequenceId]) -> Result<StepOutcome> {
        for &seq_id in seq_ids {
            let state = self.sequences.get(&seq_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "sequence {seq_id} not found on rank {}",
                    self.config.tp_rank
                )
            })?;

            ensure!(state.prefilled, "sequence {seq_id} has not been prefilled");
        }

        Ok(StepOutcome::Executed { step_id })
    }

    fn handle_admit(
        &mut self,
        step_id: StepId,
        seq_id: SequenceId,
        _prompt_len: usize,
    ) -> Result<StepOutcome> {
        ensure!(
            !self.sequences.contains_key(&seq_id),
            "sequence {seq_id} already exists on rank {}",
            self.config.tp_rank
        );

        self.sequences.insert(
            seq_id,
            ExecutorSequenceState {
                seq_id,
                cache_allocated: true,
                prefilled: false,
                position: 0,
            },
        );
        self.active_set.insert(seq_id);

        Ok(StepOutcome::SequenceAdmitted { step_id, seq_id })
    }

    fn handle_evict(
        &mut self,
        step_id: StepId,
        seq_id: SequenceId,
        _reason: EvictionReason,
    ) -> Result<StepOutcome> {
        self.sequences.remove(&seq_id);
        self.active_set.remove(&seq_id);

        Ok(StepOutcome::SequenceEvicted { step_id, seq_id })
    }

    fn handle_barrier(&mut self, step_id: StepId) -> Result<StepOutcome> {
        self.status = RankStatus::WaitingAtBarrier;
        self.group_health
            .update_status(self.config.tp_rank, RankStatus::WaitingAtBarrier);

        // Create a barrier and immediately register this rank's arrival.
        let mut barrier = TPBarrier::new(step_id, self.config.tp_size, self.config.barrier_timeout);
        barrier.arrive(self.config.tp_rank)?;
        self.current_barrier = Some(barrier);

        Ok(StepOutcome::BarrierReached { step_id })
    }

    fn handle_shutdown(&mut self, step_id: StepId) -> Result<StepOutcome> {
        self.status = RankStatus::ShutDown;
        self.group_health
            .update_status(self.config.tp_rank, RankStatus::ShutDown);
        Ok(StepOutcome::ShutdownAcknowledged { step_id })
    }

    // -----------------------------------------------------------------------
    // Barrier management
    // -----------------------------------------------------------------------

    /// Register another rank's arrival at the current barrier.
    ///
    /// Returns `Ok(true)` if all ranks have arrived and the barrier is complete.
    pub fn barrier_arrive(&mut self, rank: usize) -> Result<bool> {
        let barrier = self
            .current_barrier
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no active barrier"))?;

        let complete = barrier.arrive(rank)?;
        if complete {
            self.current_barrier = None;
            self.status = RankStatus::Ready;
            self.group_health
                .update_status(self.config.tp_rank, RankStatus::Ready);
        }
        Ok(complete)
    }

    /// Check if the current barrier has timed out.
    pub fn barrier_timed_out(&self) -> bool {
        self.current_barrier
            .as_ref()
            .map(|b| b.is_timed_out())
            .unwrap_or(false)
    }

    /// Get the missing ranks at the current barrier (if any).
    pub fn barrier_missing_ranks(&self) -> Vec<usize> {
        self.current_barrier
            .as_ref()
            .map(|b| b.missing_ranks())
            .unwrap_or_default()
    }

    /// Report this rank as failed, propagating to the group health tracker.
    pub fn report_failure(&mut self, message: &str) {
        self.status = RankStatus::Failed;
        self.group_health
            .update_status(self.config.tp_rank, RankStatus::Failed);
        self.error_count += 1;
        // In a real implementation, this would also send the failure to other
        // ranks via the transport layer. For now, we update local state.
        let _ = message; // used by the transport layer in production
    }
}

impl fmt::Debug for TPExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TPExecutor")
            .field("rank", &self.config.tp_rank)
            .field("tp_size", &self.config.tp_size)
            .field("status", &self.status)
            .field("active_count", &self.active_set.len())
            .field("steps_executed", &self.steps_executed)
            .field("error_count", &self.error_count)
            .finish()
    }
}

#[cfg(test)]
#[path = "tp_executor_tests.rs"]
mod tests;
