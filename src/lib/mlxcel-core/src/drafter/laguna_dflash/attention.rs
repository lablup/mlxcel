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

//! [`LagunaDFlashAttention`]: fused-QKV, QK-normed, per-head gated
//! sliding-window attention from the proposal block over the cached context.
//!
//! Mirrors the vLLM `DFlashLagunaModel` (vllm-project/vllm#46853): the
//! context rows and the proposal rows go through the same `qkv_proj`,
//! `q_norm` / `k_norm` apply per head before RoPE, RoPE rotates all
//! `head_dim` dims at the absolute positions, and the attention is ordinary
//! causal sliding-window attention over `[context | block]`, so every
//! proposal position sees the `W - 1` positions before it and itself. The
//! gate is `softplus(g_proj(x))` per head, computed in f32.

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::ops::concatenate;
use crate::utils::create_causal_mask_with_window_full;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::LagunaDFlashContextCache;
use super::config::LagunaDFlashConfig;

/// One drafter attention block.
pub struct LagunaDFlashAttention {
    pub qkv_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub g_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub window: i32,
}

impl LagunaDFlashAttention {
    /// Forward over the proposal block.
    ///
    /// - `x`: normed proposal rows `[B, L, H]`.
    /// - `x_ctx`: normed context rows `[B, T, H]` for absolute positions
    ///   `cache.offset() .. cache.offset() + T`. Rows older than the window
    ///   (`T > W - 1`) are dropped before projection.
    /// - `cache`: this layer's context window; receives only the context K/V.
    ///
    /// Returns `o_proj(gated attention)`, `[B, L, H]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: &MlxArray,
        cache: &mut LagunaDFlashContextCache,
    ) -> UniquePtr<MlxArray> {
        let x_shape = ffi::array_shape(x);
        let ctx_shape = ffi::array_shape(x_ctx);
        debug_assert_eq!(x_shape.len(), 3, "x must be [B, L, H]");
        debug_assert_eq!(ctx_shape.len(), 3, "x_ctx must be [B, T, H]");
        let b = x_shape[0];
        let l = x_shape[1];
        let t_full = ctx_shape[1];

        // Context positions older than `W - 1` before the block cannot be
        // attended by any proposal row: skip them before projection and
        // advance the absolute offset past them.
        let keep = cache.capacity();
        let (x_ctx, t) = if t_full > keep {
            let dropped = t_full - keep;
            cache.advance(dropped);
            (
                ffi::slice(
                    x_ctx,
                    &[0, dropped, 0],
                    &[ctx_shape[0], t_full, ctx_shape[2]],
                ),
                keep,
            )
        } else {
            (ffi::copy(x_ctx), t_full)
        };

        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        let qkv = self.qkv_proj.forward(x);
        let ctx_qkv = self.qkv_proj.forward(&x_ctx);
        let total = q_size + 2 * kv_size;
        let queries = ffi::slice(&qkv, &[0, 0, 0], &[b, l, q_size]);
        let prop_keys = ffi::slice(&qkv, &[0, 0, q_size], &[b, l, q_size + kv_size]);
        let prop_values = ffi::slice(&qkv, &[0, 0, q_size + kv_size], &[b, l, total]);
        let ctx_keys = ffi::slice(&ctx_qkv, &[0, 0, q_size], &[b, t, q_size + kv_size]);
        let ctx_values = ffi::slice(&ctx_qkv, &[0, 0, q_size + kv_size], &[b, t, total]);

        let queries = ffi::reshape(&queries, &[b, l, self.n_heads, self.head_dim]);
        let prop_keys = ffi::reshape(&prop_keys, &[b, l, self.n_kv_heads, self.head_dim]);
        let prop_values = ffi::reshape(&prop_values, &[b, l, self.n_kv_heads, self.head_dim]);
        let ctx_keys = ffi::reshape(&ctx_keys, &[b, t, self.n_kv_heads, self.head_dim]);
        let ctx_values = ffi::reshape(&ctx_values, &[b, t, self.n_kv_heads, self.head_dim]);

        // Per-head RMSNorm before transpose and RoPE, on both sides.
        let queries = self.q_norm.forward(&queries);
        let prop_keys = self.k_norm.forward(&prop_keys);
        let ctx_keys = self.k_norm.forward(&ctx_keys);

        let queries = ffi::transpose_axes(&queries, &[0, 2, 1, 3]);
        let prop_keys = ffi::transpose_axes(&prop_keys, &[0, 2, 1, 3]);
        let prop_values = ffi::transpose_axes(&prop_values, &[0, 2, 1, 3]);
        let ctx_keys = ffi::transpose_axes(&ctx_keys, &[0, 2, 1, 3]);
        let ctx_values = ffi::transpose_axes(&ctx_values, &[0, 2, 1, 3]);

        // Absolute positions: context rows continue the cache, the block
        // follows the last context row.
        let ctx_offset = cache.offset();
        let block_offset = ctx_offset + t;
        let queries = ffi::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            block_offset,
        );
        let prop_keys = ffi::fast_rope(
            &prop_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            block_offset,
        );
        let ctx_keys = ffi::fast_rope(
            &ctx_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            ctx_offset,
        );

        // Only the context K/V enters the cache; the proposal K/V is
        // concatenated for this forward and never stored.
        let (keys, values) = cache.update_and_fetch(ctx_keys, ctx_values);
        let prior = ffi::array_shape(&keys)[2];
        let keys = concatenate(&keys, &prop_keys, 2);
        let values = concatenate(&values, &prop_values, 2);

        // Causal sliding window over `[context | block]`: row `j` (absolute
        // position `block_offset + j`) attends every column whose position
        // lies in `[block_offset + j - W + 1, block_offset + j]`.
        let mask = create_causal_mask_with_window_full(l, prior, Some(self.window));
        let attn = crate::layers::attention(
            &queries,
            &keys,
            &values,
            self.scale,
            Some(&mask),
            0.0,
            self.window,
        );

        // [B, n_heads, L, D] -> [B, L, n_heads * D], then the per-head gate.
        let attn = ffi::transpose_axes(&attn, &[0, 2, 1, 3]);
        let gate = self.g_proj.forward(x);
        let gate = crate::utils::softplus(&ffi::astype(&gate, crate::dtype::FLOAT32));
        let gate = ffi::astype(&gate, ffi::array_dtype(&attn));
        let gate = ffi::expand_dims(&gate, -1);
        let gated = ffi::multiply(&attn, &gate);
        let gated = ffi::reshape(&gated, &[b, l, q_size]);
        self.o_proj.forward(&gated)
    }

    /// Load one layer's attention weights (`{prefix}.qkv_proj`, `.o_proj`,
    /// `.g_proj`, `.q_norm`, `.k_norm`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &LagunaDFlashConfig,
        window: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), group_size, bits)
        };
        let qkv_proj = linear("qkv_proj")?;
        let o_proj = linear("o_proj")?;
        let g_key = format!("{prefix}.g_proj.weight");
        let g_rows = weights
            .get(&g_key)
            .map(|w| ffi::array_shape(w)[0])
            .ok_or_else(|| format!("Weight not found: {g_key}"))?;
        let n_heads = config.num_attention_heads as i32;
        if g_rows != n_heads {
            return Err(format!(
                "{g_key} emits {g_rows} gate logits but the per-head gate needs one per query \
                 head ({n_heads})"
            ));
        }
        let g_proj = linear("g_proj")?;
        let norm = |leaf: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(RMSNorm::new(w, config.rms_norm_eps))
        };
        Ok(Self {
            qkv_proj,
            o_proj,
            g_proj,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            n_heads,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5),
            rope_base: config.rope_theta,
            window: window as i32,
        })
    }
}
