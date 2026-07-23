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
/// Adaptive MTP enable/decline policy (issue #333): profiles the first few
/// B=1 MTP burst requests of a (target, drafter, hardware) pairing and settles
/// to a data-driven verdict, overriding the static per-hardware gate while
/// keeping `MLXCEL_ENABLE_MTP_B1` as a manual force.
pub(crate) mod mtp_policy;
pub mod observability;
mod prefill_cohort;
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
/// Tick-cooperative B=1 MTP speculative slices (issue #734): serves a
/// speculative request one round per scheduler tick instead of running the
/// whole burst inside one tick, so concurrent classic-decode rows advance
/// between rounds (removes the burst head-of-line block measured by #638).
pub(crate) mod speculative_slice;
#[cfg(test)]
mod speculative_slice_tests;
/// Streaming-safe stop-string matcher (issue #449 M3 Stage 2d). Pure logic with
/// no device state, so it is always compiled and unit-tested in ordinary
/// `cargo test`; only the `xla-iree` serve worker consumes it today, so its
/// items read as dead code when that feature is off.
#[cfg_attr(not(feature = "xla-iree"), allow(dead_code))]
mod stop_matcher;
/// OpenXLA / IREE serve worker (issue #449 M3 Stage 2c): adapts the
/// `mlxcel-xla` continuous-batching engine to the [`BatchEngine`] contract.
/// Behind `xla-iree` (real IREE execution); the MLX serving path is unaffected.
#[cfg(feature = "xla-iree")]
mod xla_preprocess;
#[cfg(feature = "xla-iree")]
pub(crate) mod xla_worker;

pub use active::ActiveBatch;
pub use observability::{BatchObservability, ObservabilitySnapshot, PromptCacheLastRejectSnapshot};
pub use queue::PrefillQueue;
pub use scheduler::BatchScheduler;
pub use sequence::{
    BatchSchedulerAction, FinishReason, RequestPriority, SequenceInfo, SequenceState,
};
#[cfg(feature = "xla-iree")]
pub(crate) use xla_worker::XlaServeWorker;

/// Backend-neutral batching contract for a model worker (issue #449 M3 Stage 2c,
/// the cross-backend batching seam ADR 0004 deferred).
///
/// A worker owns its model plus KV/scheduling, consumes [`ModelRequest`]s off the
/// `request_rx` it was constructed with, and streams [`GenerateEvent`]s per request
/// until it receives [`ModelRequest::Shutdown`]. The MLX [`BatchScheduler`] and the
/// OpenXLA [`XlaServeWorker`] both satisfy it, so `ModelProvider` drives either
/// backend through one contract.
///
/// Each worker is constructed and run entirely on its own thread (the worker
/// functions in `model_worker.rs` build it inside `thread::spawn`), so it is never
/// moved across threads and needs no `Send` bound. That matters for the OpenXLA
/// worker, whose IREE context is thread-affine (`!Send`) by design.
///
/// [`ModelRequest`]: crate::server::model_provider::ModelRequest
/// [`ModelRequest::Shutdown`]: crate::server::model_provider::ModelRequest::Shutdown
/// [`GenerateEvent`]: crate::server::model_provider::GenerateEvent
pub(crate) trait BatchEngine {
    /// Run the worker's serve loop to completion (until shutdown). Called once,
    /// on the worker thread.
    fn serve(&mut self);
}

impl BatchEngine for BatchScheduler {
    fn serve(&mut self) {
        // Forward to the scheduler's existing inherent run loop. Method-call
        // syntax resolves to the inherent `BatchScheduler::run` (inherent methods
        // shadow trait methods), so the MLX serving behavior is unchanged.
        self.run();
    }
}
