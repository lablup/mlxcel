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

//! Inkling 2's bundled DSpark drafter (issue #3374).
//!
//! Both Inkling 2 releases ship a BF16 drafter in `mtp.safetensors`, described
//! by the `mtp_config` of the checkpoint's own `config.json`, so the target
//! directory doubles as the `--draft-model`. It is a DFlash-kind drafter: one
//! bidirectional forward over `[anchor, mask_embed x (block_size - 1)]`
//! attends over a persistent cache of fused target context, then an RNN
//! Markov head picks the proposals one by one on top of the target LM head.
//!
//! Ported from mlx-vlm `mlx_vlm/speculative/drafters/inkling2_dspark/`
//! (466069f5; `config.py`, `dspark.py`, `split.py`), itself adapted from
//! vLLM's Inkling 2 DSpark (Apache-2.0):
//!
//! - [`config`]: `mtp_config` parsing and the upstream validation.
//! - [`model`]: `DraftAttention`, `DraftMLP`, `DraftLayer` and `Model.mix`.
//! - [`markov`]: `MarkovHead`, including the one-time f64 fold of
//!   `hidden_out` into the target LM head.
//! - [`drafter`]: `Model.bind` / `Model.draft_block` behind
//!   [`crate::drafter::Drafter`]. Greedy only, B = 1.
//!
//! The target half (the verify forward with pre-layer captures, rollback and
//! the exactness probe) lives in the binary crate's
//! `models::inkling2::speculative`.

pub mod config;
pub mod drafter;
pub mod markov;
pub mod model;

use std::path::Path;

pub use config::{
    INKLING2_MODEL_TYPES, Inkling2DsparkConfig, Inkling2TargetShape, MTP_WEIGHTS_FILE,
    is_inkling2_dspark_dir,
};
pub use drafter::Inkling2DsparkDraftModel;

use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};

/// What an Inkling 2 target lends its bundled drafter at bind time.
///
/// Used by: [`crate::generate::LanguageModel::inkling2_dspark_binding`],
/// [`Inkling2DsparkDraftModel`].
pub struct Inkling2DsparkBinding {
    /// The token table; the drafter embeds its anchor with
    /// `embed_norm(embed_tokens(anchor))`, as the target embeds its input.
    pub embed_tokens: UnifiedEmbedding,
    pub embed_norm: RMSNorm,
    /// The LM head as upstream's drafter applies it: dense in the checkpoint
    /// dtype (BF16, not the f32 copy the target's own product uses), or the
    /// target's quantized head.
    pub lm_head: UnifiedLinear,
    pub shape: Inkling2TargetShape,
}

/// Default verify width of the drafter bundled at `dir` (its trained
/// `block_size`), or `None` when `dir` bundles none or its config is invalid.
///
/// Used by: `resolve_draft_block_size_detailed` in the binary crate.
pub fn peek_configured_block_size(dir: &Path) -> Option<usize> {
    if !is_inkling2_dspark_dir(dir) {
        return None;
    }
    Inkling2DsparkConfig::from_dir(dir)
        .ok()
        .map(|config| config.default_verify_width())
}

#[cfg(test)]
mod tests;
