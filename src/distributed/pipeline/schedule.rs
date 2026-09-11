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

//! Pipeline schedule implementations for pipeline parallelism.
//!
//! Defines the [`PipelineSchedule`] trait and concrete schedule
//! implementations:
//!
//! - [`GPipeSchedule`] — forward all micro-batches through the pipeline,
//!   then collect results. Simple, predictable, good baseline.
//!
//! The schedule drives the pipeline execution loop by emitting
//! [`ScheduleAction`]s that tell each stage what to do next.
//!
//! Used by: pipeline execution loop, distributed scheduler

use std::collections::{HashSet, VecDeque};
use std::fmt;

use anyhow::{Result, ensure};

/// Configuration for a pipeline schedule.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of stages in the pipeline.
    pub num_stages: u32,
    /// Size of each micro-batch (number of sequences).
    pub micro_batch_size: usize,
    /// Maximum number of micro-batches in flight simultaneously.
    /// 0 means unlimited (bounded only by the total number of micro-batches).
    pub max_in_flight: usize,
}

impl PipelineConfig {
    /// Create a new pipeline config.
    pub fn new(num_stages: u32, micro_batch_size: usize) -> Result<Self> {
        ensure!(num_stages >= 2, "pipeline requires at least 2 stages");
        ensure!(micro_batch_size > 0, "micro_batch_size must be > 0");
        Ok(Self {
            num_stages,
            micro_batch_size,
            max_in_flight: 0,
        })
    }

    /// Set the maximum number of micro-batches in flight.
    #[must_use]
    pub fn with_max_in_flight(mut self, max: usize) -> Self {
        self.max_in_flight = max;
        self
    }
}

/// Actions that the schedule can emit to drive the pipeline execution loop.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScheduleAction {
    /// Forward micro-batch `micro_batch_id` through stage `stage_index`.
    Forward {
        stage_index: u32,
        micro_batch_id: u32,
    },
    /// Stage `stage_index` should receive an activation from the previous stage.
    Receive {
        stage_index: u32,
        micro_batch_id: u32,
    },
    /// Flush completed micro-batch `micro_batch_id` from the pipeline
    /// (collect its results at the last stage).
    Flush { micro_batch_id: u32 },
    /// No work available right now; the executor should yield or wait.
    Idle,
    /// All micro-batches have been processed; the pipeline step is complete.
    Done,
}

impl fmt::Display for ScheduleAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward {
                stage_index,
                micro_batch_id,
            } => write!(f, "Forward(stage={stage_index}, mb={micro_batch_id})"),
            Self::Receive {
                stage_index,
                micro_batch_id,
            } => write!(f, "Receive(stage={stage_index}, mb={micro_batch_id})"),
            Self::Flush { micro_batch_id } => write!(f, "Flush(mb={micro_batch_id})"),
            Self::Idle => write!(f, "Idle"),
            Self::Done => write!(f, "Done"),
        }
    }
}

/// Trait for pipeline schedule implementations.
///
/// A schedule produces a sequence of [`ScheduleAction`]s that the pipeline
/// execution loop consumes. The executor calls [`next_action`] repeatedly
/// until it returns [`ScheduleAction::Done`].
pub trait PipelineSchedule: fmt::Debug + Send {
    /// Get the next action for the pipeline to execute.
    fn next_action(&mut self) -> ScheduleAction;

    /// Notify the schedule that a micro-batch has completed its forward
    /// pass through a stage (so the schedule can advance its state).
    fn notify_forward_complete(&mut self, stage_index: u32, micro_batch_id: u32);

    /// Notify the schedule that a micro-batch has been flushed
    /// (results collected).
    fn notify_flush_complete(&mut self, micro_batch_id: u32);

    /// Mark a micro-batch as finished (EOS or max tokens reached).
    /// The schedule should arrange for its remaining stages to be flushed.
    fn mark_sequence_done(&mut self, micro_batch_id: u32);

    /// Total number of micro-batches in this schedule.
    fn num_micro_batches(&self) -> u32;

    /// Number of pipeline stages.
    fn num_stages(&self) -> u32;

    /// Whether all micro-batches have been fully processed.
    fn is_complete(&self) -> bool;
}

/// GPipe-style pipeline schedule.
///
/// All micro-batches are forwarded through the pipeline in order.
/// For each micro-batch, the schedule emits Forward actions for
/// stages 0, 1, ..., N-1, then a Flush action. Micro-batches at
/// the same stage are processed sequentially, but different stages
/// can work on different micro-batches concurrently (the steady-state
/// pattern).
///
/// Execution pattern for 3 stages, 4 micro-batches:
/// ```text
/// Time ->
/// Stage 0: [mb0] [mb1] [mb2] [mb3]
/// Stage 1:       [mb0] [mb1] [mb2] [mb3]
/// Stage 2:             [mb0] [mb1] [mb2] [mb3]
///                                  ^flush mb0 ...
/// ```
#[derive(Debug)]
pub struct GPipeSchedule {
    config: PipelineConfig,
    num_micro_batches: u32,

    /// For each micro-batch: which stage it has completed through.
    /// `None` means it hasn't started yet; `Some(s)` means it has
    /// completed stage `s` and is ready for stage `s+1`.
    mb_completed_stage: Vec<Option<u32>>,

    /// Micro-batches that have been fully forwarded through all stages
    /// and are ready for flush.
    ready_to_flush: VecDeque<u32>,

    /// Micro-batches that have been flushed (results collected).
    flushed: HashSet<u32>,

    /// Micro-batches whose sequences are done (EOS/max tokens).
    /// These should be flushed as soon as they exit the last stage.
    done_sequences: HashSet<u32>,

    /// Queue of pending actions to emit.
    action_queue: VecDeque<ScheduleAction>,

    /// Whether we have enqueued all initial forward actions.
    initial_actions_generated: bool,
}

impl GPipeSchedule {
    /// Create a new GPipe schedule.
    ///
    /// # Arguments
    ///
    /// * `config` - Pipeline configuration.
    /// * `num_micro_batches` - Total number of micro-batches to process.
    pub fn new(config: PipelineConfig, num_micro_batches: u32) -> Result<Self> {
        ensure!(num_micro_batches > 0, "must have at least 1 micro-batch");

        Ok(Self {
            config,
            num_micro_batches,
            mb_completed_stage: vec![None; num_micro_batches as usize],
            ready_to_flush: VecDeque::new(),
            flushed: HashSet::new(),
            done_sequences: HashSet::new(),
            action_queue: VecDeque::new(),
            initial_actions_generated: false,
        })
    }

    /// Generate the initial batch of forward actions.
    ///
    /// In GPipe, we start by sending all micro-batches into stage 0
    /// sequentially (they only need a Receive for stages > 0).
    fn generate_initial_actions(&mut self) {
        for mb in 0..self.num_micro_batches {
            self.action_queue.push_back(ScheduleAction::Forward {
                stage_index: 0,
                micro_batch_id: mb,
            });
        }
        self.initial_actions_generated = true;
    }

    /// Schedule the next forward action for a micro-batch that just
    /// completed a stage.
    fn schedule_next_forward(&mut self, stage_index: u32, micro_batch_id: u32) {
        let next_stage = stage_index + 1;
        if next_stage < self.config.num_stages {
            // Emit Receive (downstream stage receives from upstream) then Forward.
            self.action_queue.push_back(ScheduleAction::Receive {
                stage_index: next_stage,
                micro_batch_id,
            });
            self.action_queue.push_back(ScheduleAction::Forward {
                stage_index: next_stage,
                micro_batch_id,
            });
        } else {
            // Micro-batch has exited the last stage; queue for flush.
            self.ready_to_flush.push_back(micro_batch_id);
            self.action_queue
                .push_back(ScheduleAction::Flush { micro_batch_id });
        }
    }

    /// Number of micro-batches currently in the pipeline (not yet flushed).
    pub fn in_flight(&self) -> usize {
        let started = self
            .mb_completed_stage
            .iter()
            .filter(|s| s.is_some())
            .count();
        started.saturating_sub(self.flushed.len())
    }
}

impl PipelineSchedule for GPipeSchedule {
    fn next_action(&mut self) -> ScheduleAction {
        if !self.initial_actions_generated {
            self.generate_initial_actions();
        }

        // Drain the action queue.
        if let Some(action) = self.action_queue.pop_front() {
            return action;
        }

        // Check if we are done.
        if self.is_complete() {
            return ScheduleAction::Done;
        }

        // Nothing to do right now; executor should wait for a
        // notify_forward_complete callback.
        ScheduleAction::Idle
    }

    fn notify_forward_complete(&mut self, stage_index: u32, micro_batch_id: u32) {
        // Guard: ignore out-of-range stage or micro-batch IDs.
        if stage_index >= self.config.num_stages || micro_batch_id >= self.num_micro_batches {
            return;
        }
        if let Some(slot) = self.mb_completed_stage.get_mut(micro_batch_id as usize) {
            *slot = Some(stage_index);
        }
        self.schedule_next_forward(stage_index, micro_batch_id);
    }

    fn notify_flush_complete(&mut self, micro_batch_id: u32) {
        // Guard: only accept valid micro-batch IDs that are actually
        // in the ready_to_flush queue to prevent premature completion.
        if micro_batch_id >= self.num_micro_batches {
            return;
        }
        self.ready_to_flush.retain(|&id| id != micro_batch_id);
        self.flushed.insert(micro_batch_id);
    }

    fn mark_sequence_done(&mut self, micro_batch_id: u32) {
        self.done_sequences.insert(micro_batch_id);
    }

    fn num_micro_batches(&self) -> u32 {
        self.num_micro_batches
    }

    fn num_stages(&self) -> u32 {
        self.config.num_stages
    }

    fn is_complete(&self) -> bool {
        self.flushed.len() == self.num_micro_batches as usize
    }
}

impl fmt::Display for GPipeSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPipeSchedule[stages={} mbs={} flushed={} in_flight={}]",
            self.config.num_stages,
            self.num_micro_batches,
            self.flushed.len(),
            self.in_flight(),
        )
    }
}

/// Create a GPipe schedule from a pipeline config and batch/micro-batch sizes.
///
/// Convenience function that computes the number of micro-batches from
/// the batch size and micro-batch size in the config.
pub fn create_gpipe_schedule(config: PipelineConfig, batch_size: usize) -> Result<GPipeSchedule> {
    ensure!(batch_size > 0, "batch_size must be > 0");
    let num_micro_batches = batch_size.div_ceil(config.micro_batch_size) as u32;
    GPipeSchedule::new(config, num_micro_batches)
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
