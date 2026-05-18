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

//! Qwen 3.5 DFlash drafter (issue #635).
//!
//! ## What this module ports
//!
//! Rust port of upstream
//! `references/mlx-vlm/mlx_vlm/speculative/drafters/qwen3_dflash/`:
//!
//! - [`config::DFlashConfig`] — the dataclass + JSON loader (config.py).
//! - [`cache::DFlashKVCache`] — type alias for [`crate::layers::KVCache`]
//!   (the upstream `DFlashKVCache = KVCache` line at the bottom of
//!   dflash.py).
//! - [`attention::DFlashAttention`] — split-projection attention.
//! - [`layer::DFlashDecoderLayer`] — one drafter transformer block.
//! - [`mlp::DFlashMlp`] — SwiGLU MLP used by every decoder layer.
//! - [`model::DFlashDraftModel`] — assembled 5-layer drafter.
//! - [`drafter::DFlashDrafter`] — adapter implementing
//!   [`crate::drafter::Drafter`] for the boxed factory return.
//!
//! ## What is novel vs. the rest of mlxcel
//!
//! - **Split-projection attention.** `k_proj` and `v_proj` are applied
//!   separately to two inputs: the proposal sequence `x` and the context
//!   buffer `x_ctx`. The cache receives only the context K/V; the
//!   proposal K/V is concatenated onto the fetched tensors post-hoc and
//!   is NOT cached. This is what lets the drafter run a single masked
//!   forward over `block_size` proposal positions while still attending
//!   to the full context history.
//! - **Multi-layer hidden state input.** The drafter's per-step input is
//!   the concatenation of the target's hidden states at
//!   `target_layer_ids = [1, 8, 15, 22, 29]`, projected through `fc`
//!   from `5 * hidden_size` down to `hidden_size`. This pipeline is fed
//!   by the Qwen 3.5 target hooks that landed in #654/#634.
//! - **Mask-token block.** `block = [bonus, mask_id, mask_id, ...,
//!   mask_id]` of shape `[B, block_size]`. The drafter runs ONE forward
//!   and samples per position from `logits[:, 1 - block_size:]` —
//!   contrast with the MTP family which runs `K` small autoregressive
//!   forwards.
//!
//! ## Drafter trait wiring
//!
//! The boxed [`Drafter`](crate::drafter::Drafter) implementation is
//! constructed by the `Dflash` arm of
//! [`load_drafter`](crate::drafter::load_drafter). See the module
//! docstring on `crate::drafter` for the trait surface and method matrix.

pub mod attention;
pub mod cache;
pub mod config;
pub mod drafter;
pub mod layer;
pub mod mlp;
pub mod model;
/// DFlash speculative-decoding round-loop driver (issue #636 / epic #633
/// sub-12). B=1 only; batched DFlash lives in [`round_loop_batched`].
pub mod round_loop;
/// DFlash speculative-decoding round-loop driver, B > 1 with continuous
/// batching and per-row GDN-aware rollback (issue #637 / epic #633 sub-13).
pub mod round_loop_batched;

pub use attention::DFlashAttention;
pub use cache::DFlashKVCache;
pub use config::DFlashConfig;
pub use drafter::DFlashDrafter;
pub use layer::DFlashDecoderLayer;
pub use mlp::DFlashMlp;
pub use model::DFlashDraftModel;
pub use round_loop::{
    DFlashGenerator, DFlashRunOutput, SpeculativeTarget, DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN_ID,
};
pub use round_loop_batched::{DFlashBatchedGenerator, DFlashBatchedRunOutput};

use crate::ffi::{self, MlxArray};

/// Materialize an integer argmax tensor into host token ids with one
/// contiguous copy.
///
/// Used by: `DFlashDrafter::draft_block`, `DFlashDrafter::draft_block_batched`,
/// `DFlashDraftModel::draft_block`, `DFlashGenerator::run`.
pub(crate) fn materialize_argmax_i32_vec(argmax: &MlxArray, expected_len: usize) -> Vec<i32> {
    let itemsize = ffi::array_itemsize(argmax);
    let bytes = ffi::array_to_raw_bytes(argmax);
    match itemsize {
        4 => bytes
            .chunks_exact(4)
            .take(expected_len)
            .map(|chunk| i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        8 => bytes
            .chunks_exact(8)
            .take(expected_len)
            .map(|chunk| {
                i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as i32
            })
            .collect(),
        _ => {
            let flat = ffi::reshape(argmax, &[expected_len as i32]);
            let mut out = Vec::with_capacity(expected_len);
            for i in 0..expected_len {
                let cell = ffi::slice(&flat, &[i as i32], &[(i + 1) as i32]);
                let scalar = ffi::reshape(&cell, &[]);
                out.push(ffi::item_i32(&scalar));
            }
            out
        }
    }
}
