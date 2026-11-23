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

//! The RoPE shared by the DeepSeek V3.2 MLA attention and its DSA indexer.
//!
//! Upstream builds both with `initialize_rope(..., scaling_config=
//! args.rope_scaling)`, which for the DeepSeek-V3.2 checkpoints is `YarnRoPE`
//! (`factor` 40 over `original_max_position_embeddings` 4096). Before #3178
//! both used plain RoPE, which matches YaRN only on the high-frequency pairs
//! the correction ramp leaves alone; past 4096 positions the low-frequency
//! pairs rotate 40x too fast and long-context output degrades.
//!
//! The rope itself is the shared [`MlaRope`] (#3295); this module keeps the
//! V3.2-specific construction: the head-width checks and the `None` training
//! context fallback. An absent block, or one that resolves to the default
//! scheme, keeps the exact plain `fast_rope(.., base, 1.0, offset)` call the
//! model made before.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py (YarnRoPE)

use super::ModelArgs;
use crate::models::mla_rope::MlaRope;

/// The V3.2 name for the shared MLA rope.
pub(super) type DsaRope = MlaRope;

impl MlaRope {
    /// Resolve the checkpoint's `rope_scaling` block for a V3.2 model.
    ///
    /// No training-context fallback is passed: upstream's `YarnRoPE` defaults a
    /// missing `original_max_position_embeddings` to 4096, not to
    /// `max_position_embeddings`, and the shared resolver does the same when
    /// given `None`.
    ///
    /// Refuses a `qk_rope_head_dim` that is zero, odd, or wider than the
    /// indexer head (`index_head_dim`, the narrower of the two tensors this
    /// rotates) before the YaRN table is built, so a malformed config fails
    /// at load instead of panicking or reading out of range in the kernel.
    pub(super) fn from_args(args: &ModelArgs) -> Result<Self, String> {
        let dims = args.qk_rope_head_dim;
        if dims == 0 || !dims.is_multiple_of(2) {
            return Err(format!(
                "qk_rope_head_dim must be a positive even number, got {dims}"
            ));
        }
        let widest = args.index_head_dim.min(args.qk_nope_head_dim + dims);
        if dims > widest {
            return Err(format!(
                "qk_rope_head_dim ({dims}) exceeds the head dim it rotates ({widest}); check qk_nope_head_dim and index_head_dim"
            ));
        }
        Ok(MlaRope::resolve(
            args.rope_scaling.as_ref(),
            dims,
            args.rope_theta,
            None,
            &args.model_type,
        ))
    }
}
