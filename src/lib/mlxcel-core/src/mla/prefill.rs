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

//! Multi-token MLA attention with the rope half folded into the SDPA head
//! dimension (issue #3071).
//!
//! An MLA score is `scale * (q_nope . k_nope + q_pe . k_pe)`. The DeepSeek V3
//! family and the absorbed latent path used to compute the rope term as its
//! own matmul, `(q_pe * scale) @ k_peᵀ`, and hand the `[B, H, L, S]` result to
//! SDPA as an additive mask. That tensor is dense over every head, query and
//! key, is built in f32 once the causal mask is added, and exists before SDPA
//! runs, so the query-axis chunking in [`crate::layers::attention`] cannot
//! bound it. At a 28K-token prefill on a 16-head checkpoint it is several GB
//! per layer.
//!
//! Concatenating `[q_nope | q_pe]` and `[k_nope | k_pe]` along the head
//! dimension gives SDPA the same dot product in one operand, the way
//! <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v2.py>
//! does it. The mask then stays the caller's `[.., L, S]` causal or padding
//! mask, or MLX's causal mode when there is none, and the only score buffer
//! left is the one SDPA's own fallback materializes, which the chunk gate in
//! [`crate::layers::attention`] and [`crate::causal_attention`] bounds.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::layers::attention;
use crate::utils::repeat_kv;

/// Which mask a folded MLA attention call applies.
#[derive(Clone, Copy)]
pub enum FoldedMask<'a> {
    /// No mask: every query sees every key.
    None,
    /// Bottom-right aligned causal mask over `S >= L` keys, through MLX's
    /// causal mode (and the chunked causal fallback), with no mask array.
    Causal,
    /// A caller-supplied additive mask broadcastable to `[B, H, L, S]`.
    Array(&'a MlxArray),
}

/// MLA attention over `[nope | rope]` concatenated queries and keys.
///
/// * `q_nope` `[B, H, L, Dn]`: the non-rope query, or the absorbed query in
///   latent space (`Dn = kv_lora_rank`).
/// * `q_pe`   `[B, H, L, P]`, already rotated.
/// * `k_nope` `[B, Hk, S, Dn]`: per-head keys (`Hk = H`) or the latent itself
///   (`Hk = 1`).
/// * `k_pe`   `[B, 1, S, P]`, already rotated; repeated to `Hk` heads.
/// * `v`      `[B, Hk, S, Dv]`.
///
/// Returns `[B, H, L, Dv]`. The value head dim may differ from `Dn + P`; MLX
/// does not fuse that shape, and the fallback it takes is query-chunked by the
/// shared attention entry points when its score buffer would exceed the
/// budget.
///
/// Used by: DeepSeek V3 (and Kimi-VL, Kimi-K2, Youtu through it) and DeepSeek
/// V3.2 multi-token attention in both the materializing and absorbed forms;
/// [`crate::mla::decode::absorbed_attention`] for a multi-token absorbed step
/// (DeepSeek V2 through `MlaLatentCache::attend`); the multi-token MLA steps of
/// GLM4MoeLite (and its pipeline stage and MTP drafter), KimiLinear, KimiK3
/// and LongcatFlashNgram (#3099).
#[must_use]
pub fn rope_folded_attention(
    q_nope: &MlxArray,
    q_pe: &MlxArray,
    k_nope: &MlxArray,
    k_pe: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: FoldedMask<'_>,
) -> UniquePtr<MlxArray> {
    let kv_heads = ffi::array_shape(k_nope)[1];
    let k_pe_heads = ffi::array_shape(k_pe)[1];
    let k_pe = if kv_heads != k_pe_heads {
        repeat_kv(k_pe, kv_heads / k_pe_heads)
    } else {
        ffi::reshape(k_pe, &ffi::array_shape(k_pe))
    };
    let queries = crate::concatenate(q_nope, q_pe, -1);
    let keys = crate::concatenate(k_nope, &k_pe, -1);
    match mask {
        FoldedMask::Array(m) => attention(&queries, &keys, v, scale, Some(m), 0.0, 0),
        FoldedMask::Causal => crate::causal_attention(&queries, &keys, v, scale, 0.0, 0),
        FoldedMask::None => attention(&queries, &keys, v, scale, None, 0.0, 0),
    }
}
