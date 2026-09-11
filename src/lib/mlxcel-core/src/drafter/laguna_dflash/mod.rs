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

//! Laguna DFlash drafter (`poolside/Laguna-*-DFlash`).
//!
//! Poolside publishes a DFlash speculator for every Laguna release. Its
//! shape differs from the Qwen 3.5 DFlash drafter in [`super::dflash`] on
//! every axis the attention touches, so it is a sibling module rather than
//! a configuration of that one:
//!
//! - fused `self_attn.qkv_proj` (q, k, v rows), per-head `q_norm` /
//!   `k_norm`, and a per-head softplus output gate `g_proj`;
//! - sliding-window (512) causal attention from the proposal block over the
//!   captured context, with a temporal context window per layer
//!   ([`cache::LagunaDFlashContextCache`]) instead of a growing `KVCache`;
//! - one `aux_hidden_norms[i]` RMSNorm per captured target layer ahead of
//!   `fc`, and the context passing through each layer's `input_layernorm`
//!   before its K/V projection.
//!
//! Reference: vLLM `vllm/model_executor/models/laguna_dflash.py`
//! (vllm-project/vllm#46853), which is what the published checkpoint is
//! served with. Where the issue text and that implementation disagree (the
//! context normalization and the per-query window), this port follows the
//! implementation.
//!
//! The boxed [`Drafter`](crate::drafter::Drafter) is built by the `Dflash`
//! arm of [`load_drafter`](crate::drafter::load_drafter) when the drafter's
//! `config.json` declares `model_type: "laguna"`.

pub mod attention;
pub mod cache;
pub mod config;
pub mod drafter;
pub mod layer;
pub mod model;
pub mod sanitize;

pub use attention::LagunaDFlashAttention;
pub use cache::LagunaDFlashContextCache;
pub use config::{LAGUNA_DFLASH_ARCHITECTURE, LAGUNA_MODEL_TYPE, LagunaDFlashConfig};
pub use drafter::LagunaDFlashDrafter;
pub use layer::LagunaDFlashDecoderLayer;
pub use model::LagunaDFlashDraftModel;
pub use sanitize::{expected_weight_keys, sanitize_weights};

#[cfg(test)]
mod tests;
