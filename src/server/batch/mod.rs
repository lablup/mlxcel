// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Request lifecycle types and batch scheduling primitives.
//!
//! This module defines the control-plane layer between HTTP handlers and the
//! batch scheduler. It provides:
//!
//! - [`SequenceState`] / [`FinishReason`] -- state machine governing each
//!   sequence from arrival through completion.
//! - [`SequenceInfo`] -- per-request context carrying prompt, sampling config,
//!   VLM embeddings, generated tokens, and response channel.
//! - [`PrefillQueue`] -- FIFO queue for requests waiting to be prefilled.
//! - [`ActiveBatch`] -- O(1)-lookup map of sequences currently decoding.
//! - [`BatchSchedulerAction`] -- scheduler decision output.
//! - [`BatchScheduler`] -- core iteration-level scheduler.

mod active;
pub mod observability;
mod queue;
pub(crate) mod scheduler;
#[cfg(test)]
mod scheduler_prompt_cache_tests;
mod sequence;
/// speculative-decoding burst driver. Folded into the
/// scheduler's `execute_prefill` dispatch when the worker's
/// [`crate::server::SpeculativeDispatch`] is kind-specific and the
/// per-request preconditions hold.
pub(crate) mod speculative_burst;
#[cfg(test)]
mod speculative_burst_tests;

pub use active::ActiveBatch;
pub use observability::{BatchObservability, ObservabilitySnapshot};
pub use queue::PrefillQueue;
pub use scheduler::BatchScheduler;
pub use sequence::{
    BatchSchedulerAction, FinishReason, RequestPriority, SequenceInfo, SequenceState,
};
