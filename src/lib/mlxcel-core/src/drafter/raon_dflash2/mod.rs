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

//! Raon-1 mini DFlash2 drafter (`KRAFTON/Raon-1-mini-dflash2`, issue #3436).
//!
//! KRAFTON's speculator for Raon-1 mini declares
//! `architectures: ["DFlash2DraftModel"]` and `model_type: "qwen3"`, and
//! differs from the Qwen 3.5 ([`super::dflash`]) and Laguna
//! ([`super::laguna_dflash`]) drafters on every axis, so it is a sibling
//! module:
//!
//! - **features**: all four mHC streams of five target layers, `[n, 5 * 4 *
//!   2560]` per position, through `fc` and `hidden_norm`;
//! - **context**: per layer, `k = rope(k_norm(k_proj(hn)))` and `v =
//!   v_proj(hn)` over the newest `swa_window_size` positions, no
//!   `input_layernorm` on the context ([`cache`]);
//! - **layer**: a grouped two-tap convolution ([`conv`]) wraps both the
//!   attention and the MLP, attention is non-causal ([`attention`]);
//! - **output**: the target's `lm_head` scores the block, and a low-rank
//!   candidate selector ([`selector`]) re-ranks the top 16 candidates per
//!   step against their predecessor before the greedy chain picks the draft.
//!
//! Reference: `DFlash2Draft` in the vendor MLX runtime
//! (`KRAFTON/Raon-1-mini-mlx/raon_mlx_model.py`, section 4), which follows
//! the speculator's vLLM implementation. The bf16 repository is loaded; the
//! NVFP4 copy bundled under `Raon-1-mini-mlx/dflash2/` uses a vendor key
//! format (`w.*`, `q.*` with `fmt.*` metadata) and is refused by name.
//!
//! The boxed [`Drafter`](crate::drafter::Drafter) is built by the `Dflash`
//! arm of [`load_drafter`](crate::drafter::load_drafter) when the drafter's
//! `config.json` declares the `DFlash2DraftModel` architecture.

pub mod attention;
pub mod cache;
pub mod config;
pub mod conv;
pub mod drafter;
pub mod layer;
pub mod model;
pub mod sanitize;
pub mod selector;

pub use config::{RAON_DFLASH2_ARCHITECTURE, RaonDFlash2Config};
pub use drafter::RaonDFlash2Drafter;
pub use model::RaonDFlash2DraftModel;

#[cfg(test)]
mod tests;
