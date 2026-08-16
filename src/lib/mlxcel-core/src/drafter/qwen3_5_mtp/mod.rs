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

//! Qwen 3.5 / 3.6 / 3.8 Multi-Token Prediction (MTP) drafter — the split-out
//! MTP head (`fc` fusion + one full-attention decoder layer) that pairs with
//! `qwen3_5`-family targets for MTP speculative decoding.
//!
//! Top-level overview:
//!
//! - [`config`] — `Qwen35MtpConfig` (top-level `block_size`) and the
//!   target-mirroring `Qwen35MtpTextConfig` subset.
//! - [`layer`] — `Qwen35MtpDecoderLayer` (gated-Q full attention + SwiGLU
//!   MLP, drafter-owned KV cache).
//! - [`model`] — `Qwen35MtpDraftModel` implementing
//!   [`crate::drafter::Drafter`], including the stateful
//!   prompt-prefill / accept-verified lifecycle hooks.
//!
//! Upstream reference:
//! https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/speculative/drafters/qwen3_5_mtp.

pub mod config;
pub mod layer;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{Qwen35MtpConfig, Qwen35MtpTextConfig};
pub use model::Qwen35MtpDraftModel;
