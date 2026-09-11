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

//! Micro-batch splitting and management for pipeline parallelism.
//!
//! Splits incoming batches into smaller micro-batches that can flow through
//! pipeline stages concurrently, reducing bubble time. Each micro-batch
//! carries a contiguous slice of the original batch's sequences.
//!
//! Used by: pipeline schedule, pipeline execution loop

use std::fmt;

use anyhow::{Result, ensure};

use crate::distributed::request_tracker::RequestId;

/// Specification for a single micro-batch: which slice of the original
/// batch it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBatchSpec {
    /// Zero-based micro-batch index within the pipeline step.
    pub id: u32,
    /// Start index (inclusive) into the original batch.
    pub start_index: usize,
    /// End index (exclusive) into the original batch.
    pub end_index: usize,
    /// Number of sequences in this micro-batch.
    pub size: usize,
}

impl fmt::Display for MicroBatchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MicroBatch[id={} range={}..{} size={}]",
            self.id, self.start_index, self.end_index, self.size
        )
    }
}

/// A micro-batch with associated request IDs and metadata, used during
/// pipeline execution.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// Micro-batch specification (index range).
    pub spec: MicroBatchSpec,
    /// Request IDs for sequences in this micro-batch.
    pub request_ids: Vec<RequestId>,
    /// Current pipeline stage this micro-batch is at (0-based).
    pub current_stage: u32,
    /// Whether this micro-batch has been marked as complete (EOS or max tokens).
    pub completed: bool,
}

impl MicroBatch {
    /// Create a new micro-batch at stage 0.
    pub fn new(spec: MicroBatchSpec, request_ids: Vec<RequestId>) -> Self {
        Self {
            spec,
            request_ids,
            current_stage: 0,
            completed: false,
        }
    }

    /// Advance this micro-batch to the next stage.
    pub fn advance_stage(&mut self) {
        self.current_stage += 1;
    }

    /// Mark this micro-batch as completed (all sequences finished).
    pub fn mark_completed(&mut self) {
        self.completed = true;
    }

    /// Number of sequences in this micro-batch.
    pub fn size(&self) -> usize {
        self.spec.size
    }
}

impl fmt::Display for MicroBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MicroBatch[id={} stage={} size={} completed={}]",
            self.spec.id, self.current_stage, self.spec.size, self.completed
        )
    }
}

/// Split a batch of `batch_size` sequences into micro-batches of at most
/// `micro_batch_size` sequences each.
///
/// Returns a vector of [`MicroBatchSpec`]s covering the full batch.
/// The last micro-batch may be smaller than `micro_batch_size` if
/// `batch_size` is not evenly divisible.
///
/// # Errors
///
/// Returns an error if `batch_size` or `micro_batch_size` is zero.
pub fn split_into_micro_batches(
    batch_size: usize,
    micro_batch_size: usize,
) -> Result<Vec<MicroBatchSpec>> {
    ensure!(batch_size > 0, "batch_size must be > 0");
    ensure!(micro_batch_size > 0, "micro_batch_size must be > 0");

    let num_micro_batches = batch_size.div_ceil(micro_batch_size);
    let mut specs = Vec::with_capacity(num_micro_batches);

    let mut start = 0;
    for id in 0..num_micro_batches {
        let end = (start + micro_batch_size).min(batch_size);
        specs.push(MicroBatchSpec {
            id: id as u32,
            start_index: start,
            end_index: end,
            size: end - start,
        });
        start = end;
    }

    Ok(specs)
}

/// Compute the optimal micro-batch size given the batch size and number of
/// pipeline stages.
///
/// The heuristic targets at least `num_stages * 2` micro-batches for good
/// pipeline utilization (reducing bubble fraction to ~1/(2*num_micro_batches)).
/// Clamps to `[1, batch_size]`.
pub fn suggested_micro_batch_size(batch_size: usize, num_stages: usize) -> usize {
    if batch_size == 0 || num_stages == 0 {
        return 1;
    }
    // Target 2x as many micro-batches as stages for reasonable utilization.
    let target_count = num_stages.saturating_mul(2).max(1);
    let suggested = batch_size.div_ceil(target_count).max(1);
    suggested.min(batch_size)
}

#[cfg(test)]
#[path = "micro_batch_tests.rs"]
mod tests;
