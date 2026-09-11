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

//! TP Scheduler — rank 0 coordinator for tensor-parallel execution.
//!
//! The scheduler runs **only on rank 0** and is responsible for:
//!
//! - Making all batch composition decisions (which sequences to prefill/decode)
//! - Admitting new sequences and evicting completed/preempted ones
//! - Broadcasting [`StepDecision`] to all ranks before each inference step
//! - Coordinating token sampling and result broadcasting
//! - Continuous batching: interleaving prefill and decode steps
//!
//! Other ranks never make independent scheduling decisions. They receive and
//! execute decisions from this scheduler via the broadcast channel.
//!
//! Used by: tensor_parallel main inference loop (rank 0 only)

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Instant;

use anyhow::{Result, bail, ensure};

use super::synchronized::{
    EvictionReason, SampledTokens, SamplingMode, SequenceId, StepDecision, StepId,
    TPExecutionConfig,
};

// ---------------------------------------------------------------------------
// Sequence state tracking
// ---------------------------------------------------------------------------

/// State of a sequence within the TP scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SequenceState {
    /// Waiting to be admitted into the active batch.
    Waiting,
    /// Currently in the active batch, needs prefill.
    PendingPrefill,
    /// Currently in the active batch, generating tokens.
    Decoding,
    /// Completed (EOS, max length, or cancelled).
    Completed,
    /// Evicted from the batch (can be re-admitted if needed).
    Evicted,
}

impl fmt::Display for SequenceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Waiting => write!(f, "waiting"),
            Self::PendingPrefill => write!(f, "pending_prefill"),
            Self::Decoding => write!(f, "decoding"),
            Self::Completed => write!(f, "completed"),
            Self::Evicted => write!(f, "evicted"),
        }
    }
}

/// Metadata tracked per sequence by the scheduler.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Sequence ID.
    pub seq_id: SequenceId,
    /// Current state.
    pub state: SequenceState,
    /// Initial prompt length in tokens.
    pub prompt_len: usize,
    /// Number of tokens generated so far.
    pub generated_tokens: usize,
    /// Maximum tokens to generate for this sequence.
    pub max_tokens: usize,
    /// Priority (lower = higher priority). Default: 0.
    pub priority: u32,
    /// When the sequence was submitted.
    pub submitted_at: Instant,
    /// When the sequence was admitted to the active batch (if ever).
    pub admitted_at: Option<Instant>,
}

impl SequenceInfo {
    /// Total tokens processed so far (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_len + self.generated_tokens
    }

    /// Whether the sequence has reached its generation limit.
    pub fn at_max_tokens(&self) -> bool {
        self.generated_tokens >= self.max_tokens
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Continuous batching scheduler for tensor-parallel inference.
///
/// Runs on rank 0 only. Produces [`StepDecision`] values that are broadcast
/// to all ranks for lockstep execution.
pub struct TPScheduler {
    /// Execution configuration.
    config: TPExecutionConfig,
    /// How token sampling is parallelized.
    sampling_mode: SamplingMode,
    /// Monotonically increasing step counter.
    next_step_id: StepId,
    /// All tracked sequences by ID.
    sequences: HashMap<SequenceId, SequenceInfo>,
    /// Queue of sequences waiting to be admitted (FIFO with priority).
    waiting_queue: VecDeque<SequenceId>,
    /// Set of currently active sequence IDs (in the batch).
    active_set: Vec<SequenceId>,
    /// Sequences that need prefill before they can decode.
    pending_prefill: Vec<SequenceId>,
}

impl TPScheduler {
    /// Create a new scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid or if this is not rank 0.
    pub fn new(config: TPExecutionConfig, sampling_mode: SamplingMode) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.is_coordinator,
            "TPScheduler must run on the coordinator (rank 0), but tp_rank={}",
            config.tp_rank
        );
        Ok(Self {
            config,
            sampling_mode,
            next_step_id: 0,
            sequences: HashMap::new(),
            waiting_queue: VecDeque::new(),
            active_set: Vec::new(),
            pending_prefill: Vec::new(),
        })
    }

    /// Get the sampling mode.
    pub fn sampling_mode(&self) -> SamplingMode {
        self.sampling_mode
    }

    /// Get the current step ID (the next one that will be issued).
    pub fn next_step_id(&self) -> StepId {
        self.next_step_id
    }

    /// Allocate the next step ID.
    fn alloc_step_id(&mut self) -> StepId {
        let id = self.next_step_id;
        self.next_step_id += 1;
        id
    }

    // -----------------------------------------------------------------------
    // Sequence management
    // -----------------------------------------------------------------------

    /// Submit a new sequence to the scheduler.
    ///
    /// The sequence enters the waiting queue and will be admitted when there
    /// is space in the active batch.
    pub fn submit_sequence(
        &mut self,
        seq_id: SequenceId,
        prompt_len: usize,
        max_tokens: usize,
        priority: u32,
    ) -> Result<()> {
        ensure!(
            !self.sequences.contains_key(&seq_id),
            "sequence {seq_id} already exists"
        );
        ensure!(prompt_len >= 1, "prompt_len must be >= 1");
        ensure!(max_tokens >= 1, "max_tokens must be >= 1");

        let info = SequenceInfo {
            seq_id,
            state: SequenceState::Waiting,
            prompt_len,
            generated_tokens: 0,
            max_tokens,
            priority,
            submitted_at: Instant::now(),
            admitted_at: None,
        };
        self.sequences.insert(seq_id, info);
        self.waiting_queue.push_back(seq_id);
        Ok(())
    }

    /// Number of sequences currently in the active batch.
    pub fn active_count(&self) -> usize {
        self.active_set.len()
    }

    /// Number of sequences waiting to be admitted.
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Total number of tracked sequences (all states).
    pub fn total_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get the state of a sequence.
    pub fn sequence_state(&self, seq_id: SequenceId) -> Option<SequenceState> {
        self.sequences.get(&seq_id).map(|s| s.state)
    }

    /// Get the info for a sequence.
    pub fn sequence_info(&self, seq_id: SequenceId) -> Option<&SequenceInfo> {
        self.sequences.get(&seq_id)
    }

    // -----------------------------------------------------------------------
    // Step decision generation
    // -----------------------------------------------------------------------

    /// Generate the next batch of decisions for one inference cycle.
    ///
    /// This implements continuous batching: admit new sequences if there is
    /// space, schedule prefill for newly admitted sequences, then schedule
    /// decode for all active sequences that have been prefilled.
    ///
    /// Returns a list of decisions that must be broadcast to all ranks in order.
    pub fn schedule_step(&mut self) -> Result<Vec<StepDecision>> {
        let mut decisions = Vec::new();

        // 1. Admit waiting sequences up to max_batch_size.
        while self.active_set.len() < self.config.max_batch_size {
            if let Some(seq_id) = self.waiting_queue.pop_front() {
                // Check state and get prompt_len without holding a mutable borrow.
                let prompt_len = match self.sequences.get(&seq_id) {
                    Some(info) if info.state == SequenceState::Waiting => info.prompt_len,
                    _ => continue, // skip cancelled/evicted sequences
                };

                let step_id = self.alloc_step_id();

                if let Some(info) = self.sequences.get_mut(&seq_id) {
                    info.state = SequenceState::PendingPrefill;
                    info.admitted_at = Some(Instant::now());
                }
                self.active_set.push(seq_id);
                self.pending_prefill.push(seq_id);
                decisions.push(StepDecision::AdmitSequence {
                    step_id,
                    seq_id,
                    prompt_len,
                });
            } else {
                break;
            }
        }

        // 2. Schedule prefill for pending sequences.
        if !self.pending_prefill.is_empty() {
            let prefill_ids: Vec<SequenceId> = self.pending_prefill.drain(..).collect();
            let token_counts: Vec<usize> = prefill_ids
                .iter()
                .filter_map(|id| self.sequences.get(id).map(|s| s.prompt_len))
                .collect();

            let step_id = self.alloc_step_id();

            // Transition to Decoding state.
            for &seq_id in &prefill_ids {
                if let Some(info) = self.sequences.get_mut(&seq_id) {
                    info.state = SequenceState::Decoding;
                }
            }

            decisions.push(StepDecision::Prefill {
                step_id,
                seq_ids: prefill_ids,
                token_counts,
            });
        }

        // 3. Schedule decode for all active decoding sequences.
        // Sort by sequence ID for a canonical ordering that matches the executor.
        let mut decode_ids: Vec<SequenceId> = self
            .active_set
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id)
                    .map(|s| s.state == SequenceState::Decoding)
                    .unwrap_or(false)
            })
            .copied()
            .collect();
        decode_ids.sort_unstable();

        if !decode_ids.is_empty() {
            let step_id = self.alloc_step_id();
            decisions.push(StepDecision::Decode {
                step_id,
                seq_ids: decode_ids,
            });
        }

        Ok(decisions)
    }

    /// Record sampled tokens from rank 0 and update sequence states.
    ///
    /// Returns a list of eviction decisions for completed sequences.
    pub fn record_sampled_tokens(&mut self, sampled: &SampledTokens) -> Result<Vec<StepDecision>> {
        let mut evictions = Vec::new();

        // Find active decoding sequences in canonical sorted order (must match executor).
        let mut decode_ids: Vec<SequenceId> = self
            .active_set
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id)
                    .map(|s| s.state == SequenceState::Decoding)
                    .unwrap_or(false)
            })
            .copied()
            .collect();
        decode_ids.sort_unstable();

        ensure!(
            sampled.tokens.len() == decode_ids.len(),
            "sampled token count ({}) does not match decode batch size ({})",
            sampled.tokens.len(),
            decode_ids.len()
        );

        // Update generated token counts.
        for &seq_id in &decode_ids {
            if let Some(info) = self.sequences.get_mut(&seq_id) {
                info.generated_tokens += 1;
            }
        }

        // Check for completed sequences.
        for &seq_id in &sampled.completed_seq_ids {
            let reason = if self
                .sequences
                .get(&seq_id)
                .map(|s| s.at_max_tokens())
                .unwrap_or(false)
            {
                EvictionReason::MaxLengthReached
            } else {
                EvictionReason::EndOfSequence
            };

            let step_id = self.alloc_step_id();
            evictions.push(StepDecision::EvictSequence {
                step_id,
                seq_id,
                reason,
            });

            if let Some(info) = self.sequences.get_mut(&seq_id) {
                info.state = SequenceState::Completed;
            }
            self.active_set.retain(|id| *id != seq_id);
        }

        Ok(evictions)
    }

    /// Manually evict a sequence (e.g., due to memory pressure or cancellation).
    pub fn evict_sequence(
        &mut self,
        seq_id: SequenceId,
        reason: EvictionReason,
    ) -> Result<StepDecision> {
        let info = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| anyhow::anyhow!("sequence {seq_id} not found"))?;

        match info.state {
            SequenceState::Completed | SequenceState::Evicted => {
                bail!("sequence {seq_id} is already {}", info.state);
            }
            _ => {}
        }

        info.state = SequenceState::Evicted;
        self.active_set.retain(|id| *id != seq_id);
        self.waiting_queue.retain(|id| *id != seq_id);
        self.pending_prefill.retain(|id| *id != seq_id);

        let step_id = self.alloc_step_id();
        Ok(StepDecision::EvictSequence {
            step_id,
            seq_id,
            reason,
        })
    }

    /// Generate a barrier decision.
    pub fn barrier(&mut self) -> StepDecision {
        let step_id = self.alloc_step_id();
        StepDecision::Barrier { step_id }
    }

    /// Generate a shutdown decision.
    pub fn shutdown(&mut self) -> StepDecision {
        let step_id = self.alloc_step_id();
        StepDecision::Shutdown { step_id }
    }

    /// Check if there is any work to do (active or waiting sequences).
    pub fn has_work(&self) -> bool {
        !self.active_set.is_empty() || !self.waiting_queue.is_empty()
    }
}

impl fmt::Debug for TPScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TPScheduler")
            .field("tp_rank", &self.config.tp_rank)
            .field("sampling_mode", &self.sampling_mode)
            .field("next_step_id", &self.next_step_id)
            .field("active_count", &self.active_set.len())
            .field("waiting_count", &self.waiting_queue.len())
            .field("pending_prefill", &self.pending_prefill.len())
            .finish()
    }
}

#[cfg(test)]
#[path = "tp_scheduler_tests.rs"]
mod tests;
