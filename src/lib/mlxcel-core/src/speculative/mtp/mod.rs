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

//! Multi-Token Prediction (MTP) round-loop generator. Originally built for
//! the Gemma 4 assistant drafter family; since issue #1165 also drives the
//! Qwen 3.5 MTP drafter family through the same generic [`MtpGenerator`],
//! parameterized over any target implementing [`target::MtpTarget`].
//!
//! Each drafter here has a different cache relationship with its target. The
//! Gemma 4 drafter owns no KV cache and attends the target's shared K/V; the
//! Qwen 3.5 MTP drafter owns accumulated KV history, destroyed on `reset`, and
//! consumes the target's post-norm hidden state. Both verify the block in one
//! batched forward and roll back with `rollback_speculative_cache` (per row).
//!
//! ## Round-loop reference
//!
//! Mirrors the upstream Python `_mtp_rounds` and `_speculative_walk` in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/mtp.py.
//!
//! ## Submodules
//!
//! - [`walk`] — `_speculative_walk` accept logic (standalone, well-tested).
//! - [`generator`] — [`MtpGenerator`] round-loop driver.
//! - [`target`]: [`MtpTarget`] trait that a target model implements to
//!   expose its speculative hooks (`forward_with_speculative_sinks`,
//!   `rollback_speculative_cache`) to the round-loop driver without forcing
//!   this crate to depend on the outer `mlxcel` crate. Implemented by the
//!   Gemma 4 wrapper and, since issue #1165, the Qwen 3.5 (dense and MoE,
//!   text and VLM) targets.

pub(crate) mod adaptive;
pub mod generator;
pub mod round_loop_batched;
pub mod target;
pub mod tree;
pub mod walk;

#[cfg(test)]
mod tests;

pub use generator::{MtpAcceptanceSummary, MtpGenerator, MtpSessionState, MtpStepOutput};
pub use round_loop_batched::{MtpBatchedGenerator, MtpBatchedRunOutput};
pub use target::{
    MtpBatchedVerifyForwardOutput, MtpBatchedVerifyOutput, MtpTarget, MtpVerifyOutput,
};
pub use walk::{speculative_walk, speculative_walk_batched};
