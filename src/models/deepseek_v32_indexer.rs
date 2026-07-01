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

//! DeepSeek Sparse Attention (DSA) "lightning indexer".
//!
//! Shared between `deepseek_v32` and `glm_moe_dsa`. Per decoder layer, the
//! indexer scores every cached key against the current query and selects the
//! top-`index_topk` positions the main MLA attention should attend to. When
//! the running `kv_len` is at or below `index_topk`, selection is skipped and
//! the caller falls back to dense attention (the trained model is numerically
//! identical to dense at short context).
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py (Indexer)
//! Per-model `indexer_rope_interleave` default (traditional=false for
//! deepseek_v32, true for glm_moe_dsa) landed in ml-explore/mlx-lm#1431.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ModelArgs;

/// nn.LayerNorm default epsilon (mlx `nn.LayerNorm(dims, eps=1e-5)`).
const LAYER_NORM_EPS: f32 = 1e-5;

/// Environment variable that forces the dense full-attention fallback,
/// disabling the indexer entirely for A/B comparison against the pre-#509
/// behavior. When set (to any value) the indexer is not loaded.
pub(super) const DENSE_FALLBACK_ENV: &str = "MLXCEL_DSA_DENSE";

/// The DSA lightning indexer for one attention block.
pub(super) struct Indexer {
    wq_b: UnifiedLinear,
    wk: UnifiedLinear,
    k_norm: LayerNorm,
    weights_proj: UnifiedLinear,
    n_heads: i32,
    head_dim: i32,
    rope_head_dim: i32,
    index_topk: i32,
    softmax_scale: f32,
    rope_base: f32,
    /// `traditional` flag threaded into the indexer RoPE. This is the
    /// per-model `indexer_rope_interleave` value: `false` (non-interleaved)
    /// for deepseek_v32, `true` (interleaved) for glm_moe_dsa. Getting it
    /// wrong silently corrupts key selection with no crash.
    pub(super) rope_traditional: bool,
}

impl Indexer {
    /// Load the indexer for `{attn_prefix}.indexer.*`. Returns `Ok(None)` when
    /// the checkpoint carries no indexer weights (so the dense fallback is
    /// preserved) or when [`DENSE_FALLBACK_ENV`] is set.
    pub(super) fn load(
        weights: &WeightMap,
        args: &ModelArgs,
        attn_prefix: &str,
    ) -> Result<Option<Self>, String> {
        let prefix = format!("{}.indexer", attn_prefix);
        let wk_key = format!("{}.wk.weight", prefix);
        if !weights.contains_key(&wk_key) {
            return Ok(None);
        }
        if std::env::var_os(DENSE_FALLBACK_ENV).is_some() {
            return Ok(None);
        }

        let group_size = args.group_size();
        let bits = args.bits();

        let wq_b =
            UnifiedLinear::from_weights(weights, &format!("{}.wq_b", prefix), group_size, bits)?;
        let wk = UnifiedLinear::from_weights(weights, &format!("{}.wk", prefix), group_size, bits)?;
        let weights_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.weights_proj", prefix),
            group_size,
            bits,
        )?;

        let k_norm_weight = super::get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
        let k_norm_bias = weights
            .get(&format!("{}.k_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let k_norm = LayerNorm::new(k_norm_weight, k_norm_bias, LAYER_NORM_EPS);

        let head_dim = args.index_head_dim as i32;
        Ok(Some(Self {
            wq_b,
            wk,
            k_norm,
            weights_proj,
            n_heads: args.index_n_heads as i32,
            head_dim,
            rope_head_dim: args.qk_rope_head_dim as i32,
            index_topk: args.index_topk as i32,
            softmax_scale: (head_dim as f32).powf(-0.5),
            rope_base: args.rope_theta,
            rope_traditional: args.indexer_rope_interleave,
        }))
    }

    /// Indexer key for the new tokens: `rope(reshape(k_norm(wk(x))))`.
    /// Shape `[b, 1, s, index_head_dim]`. RoPE is applied over the first
    /// `qk_rope_head_dim` dims with the running cache `offset`. This is what
    /// gets cached (concatenated onto `kv_latent`).
    pub(super) fn keys(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let s = shape[1];
        let k = self.wk.forward(x);
        let k = self.k_norm.forward(&k);
        let k = mlxcel_core::reshape(&k, &[b, 1, s, self.head_dim]);
        mlxcel_core::fast_rope(
            &k,
            self.rope_head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        )
    }

    /// Indexer query: `rope(reshape(wq_b(qr)))`, where `qr` is the LoRA-reduced
    /// query hidden `q_a_layernorm(q_a_proj(x))`. Shape
    /// `[b, index_n_heads, s, index_head_dim]`.
    pub(super) fn queries(&self, qr: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(qr);
        let b = shape[0];
        let s = shape[1];
        let q = self.wq_b.forward(qr);
        let q = mlxcel_core::reshape(&q, &[b, s, self.n_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        mlxcel_core::fast_rope(
            &q,
            self.rope_head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        )
    }

    /// Raw per-head combination weights `weights_proj(x)`, shape `[b, s, n_heads]`.
    /// The `n_heads**-0.5 * head_dim**-0.5` scaling is applied inside
    /// [`indexer_top_indices`].
    pub(super) fn weights(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.weights_proj.forward(x)
    }

    /// Select the top-`index_topk` key positions for each query. Returns `None`
    /// when `kv_len <= index_topk` (dense fallback). `q`/`k`/`weights` come from
    /// [`Self::queries`], the cache-fetched indexer keys, and [`Self::weights`].
    pub(super) fn top_indices(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        weights: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        indexer_top_indices(
            q,
            k,
            weights,
            mask,
            self.n_heads,
            self.softmax_scale,
            self.index_topk,
        )
    }
}

/// Pure top-k index selection, factored out for unit testing.
///
/// Mirrors upstream `Indexer.__call__` scoring exactly:
/// `scores = relu(q @ k^T)`, per-head `weights * (n_heads**-0.5 * softmax_scale)`,
/// combine over heads, apply the (additive) causal `mask`, then
/// `argpartition(..., kth=kv_len - index_topk)[..., -index_topk:]`.
///
/// - `q`: `[b, n_heads, s, head_dim]`
/// - `k`: `[b, 1, kv_len, head_dim]`
/// - `weights`: `[b, s, n_heads]` (raw `weights_proj` output, unscaled)
/// - `mask`: optional additive mask broadcastable to `[b, 1, s, kv_len]`
///   (0 to keep, `-inf` to drop)
///
/// Returns `None` when `kv_len <= index_topk`; otherwise `[b, 1, s, index_topk]`.
pub(super) fn indexer_top_indices(
    q: &MlxArray,
    k: &MlxArray,
    weights: &MlxArray,
    mask: Option<&MlxArray>,
    n_heads: i32,
    softmax_scale: f32,
    index_topk: i32,
) -> Option<UniquePtr<MlxArray>> {
    let kv_len = mlxcel_core::array_shape(k)[2];
    if kv_len <= index_topk {
        return None;
    }

    // scores = relu(q @ k^T) -> [b, n_heads, s, kv_len] (heads broadcast over k).
    let k_t = mlxcel_core::transpose_axes(k, &[0, 1, 3, 2]);
    let scores = mlxcel_core::matmul(q, &k_t);
    let scores = mlxcel_core::relu(&scores);

    // w = weights * (n_heads**-0.5 * softmax_scale), reshaped to [b, n_heads, s, 1].
    let w_scale = (n_heads as f32).powf(-0.5) * softmax_scale;
    let w_scale = mlxcel_core::full_f32(&[1], w_scale, mlxcel_core::array_dtype(weights));
    let w = mlxcel_core::multiply(weights, &w_scale);
    let w = mlxcel_core::transpose_axes(&w, &[0, 2, 1]);
    let w = mlxcel_core::expand_dims(&w, -1);

    // Combine over heads: [b, 1, s, kv_len].
    let scores = mlxcel_core::multiply(&scores, &w);
    let scores = mlxcel_core::sum_axis(&scores, 1, true);

    // Apply the additive causal mask (adds -inf to disallowed positions).
    let scores = match mask {
        Some(m) => mlxcel_core::add(&scores, m),
        None => scores,
    };

    // Top-`index_topk` largest along the key axis.
    let kth = kv_len - index_topk;
    let part = mlxcel_core::argpartition(&scores, kth, -1);
    Some(slice_axis(&part, -1, kth, kv_len))
}
