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

//! Gemma 4 Multi-Token Prediction (MTP) "assistant" drafter — 4-layer
//! transformer with pre/post projections and frozen RoPE cross-attention
//! into the target's last full-attention / sliding-attention K/V slabs.
//!
//! Top-level overview:
//!
//! - [`config`] — `Gemma4AssistantConfig`, `DrafterTextConfig`, RoPE params.
//! - [`layer`] — `DraftDecoderLayer` (KV-shared-only) + `DrafterAttention` +
//!   `DrafterMlp`.
//! - [`model`] — `Gemma4AssistantDraftModel` implementing
//!   [`crate::drafter::Drafter`].
//!
//! Upstream reference:
//! https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/speculative/drafters/gemma4_assistant.

pub mod config;
pub mod layer;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{
    DrafterQuantizationArgs, DrafterRopeParameters, DrafterTextConfig, Gemma4AssistantConfig,
};
pub use model::Gemma4AssistantDraftModel;
