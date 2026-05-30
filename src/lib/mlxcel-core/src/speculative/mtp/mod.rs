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

//! Multi-Token Prediction (MTP) round-loop generator for the Gemma 4
//! assistant drafter family.
//!
//! This module is a **peer** to [`crate::speculative::SpeculativeGenerator`],
//! not an extension. The classic speculative path and the MTP path differ in
//! ways that would make a shared base struct more confusing than helpful:
//!
//! | Aspect                    | Classic speculative           | MTP                                       |
//! |---------------------------|-------------------------------|-------------------------------------------|
//! | Drafter KV cache          | Owns its own KV cache         | Has no own KV cache                       |
//! | Drafter input             | Last accepted token           | Bonus token + target's last hidden        |
//! | Verify                    | Single batched forward        | Single batched forward (same)             |
//! | Rollback                  | `trim_caches`                 | `rollback_speculative_cache` (per-row)    |
//! | Cross-attention           | Each model attends own cache  | Drafter attends target's shared K/V       |
//!
//! Both paths satisfy the same external contract — the
//! [`crate::generate::LanguageModel`] target trait + a [`crate::drafter::Drafter`]
//! drafter — so the caller-facing entry-point on each generator (`generate(...)`)
//! takes the same shape (`prompt_tokens`, `max_tokens`, `sampling`).
//!
//! ## Round-loop reference
//!
//! Mirrors the upstream Python `_mtp_rounds` and `_speculative_walk` in
//! `references/mlx-vlm/mlx_vlm/generate.py` (lines 456-619).
//!
//! ## Submodules
//!
//! - [`walk`] — `_speculative_walk` accept logic (standalone, well-tested).
//! - [`generator`] — [`MtpGenerator`] round-loop driver.
//! - [`target`] — [`MtpTarget`] trait that the Gemma 4 wrapper implements
//!   to expose its speculative hooks (`forward_with_speculative_sinks`,
//!   `rollback_speculative_cache`) to the round-loop driver without
//!   forcing this crate to depend on the outer `mlxcel` crate.

pub(crate) mod adaptive;
pub mod generator;
pub mod round_loop_batched;
pub mod target;
pub mod walk;

#[cfg(test)]
mod tests;

pub use generator::MtpGenerator;
pub use round_loop_batched::{MtpBatchedGenerator, MtpBatchedRunOutput};
pub use target::{
    MtpBatchedVerifyForwardOutput, MtpBatchedVerifyOutput, MtpTarget, MtpVerifyOutput,
};
pub use walk::{speculative_walk, speculative_walk_batched};
