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

//! `DFlashAttention` — split-projection attention with cache-write
//! restricted to the context K/V only.
//!
//! Apple Silicon precision rules apply (see `docs/apple-silicon-precision.md`):
//! the inputs and all intermediate tensors stay in the dtype the caller
//! passes in (bf16 or f16 for non-quantized weights; dequantized output
//! for quantized weights). No `f32` promotion.

use crate::cache::KVCache;
use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::config::DFlashConfig;

/// DFlash attention block.
///
/// This is the only attention layer in mlxcel with **split projections**:
/// `k_proj` and `v_proj` are each applied twice per forward — once to the
/// proposal sequence `x`, once to the context buffer `x_ctx`. Only the
/// context K/V is written into [`KVCache`] via `update_and_fetch`; the
/// proposal K/V is concatenated onto the fetched tensors post-hoc and
/// never enters the cache. This is what makes the single masked draft
/// forward possible.
///
/// Per upstream `DFlashAttention.__call__`:
///
/// - `q_proj`, `k_proj`, `v_proj`, `o_proj` are bias-free (DFlashConfig's
///   `attention_bias = False`).
/// - `q_norm`, `k_norm` are RMSNorm over `head_dim`; applied **after**
///   reshape and **before** transpose, on both ctx and proposal sides.
/// - RoPE offsets:
///     - `queries` at `cache.offset + S` (the proposal positions follow
///       the past context plus the freshly-projected ctx).
///     - `ctx_keys` at `cache.offset` (ctx tokens replay the past offset).
///     - `prop_keys` at `cache.offset + S` (proposal positions live
///       past the ctx tail).
/// - Scale = `head_dim ** -0.5`.
pub struct DFlashAttention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl DFlashAttention {
    /// Forward with split projections.
    ///
    /// - `x` — proposal sequence, shape `[B, L, hidden_size]`.
    /// - `x_ctx` — context buffer, shape `[B, S, hidden_size]`.
    /// - `cache` — the drafter's own per-layer K/V cache. Updated in
    ///   place with the context-side K/V only.
    ///
    /// Returns `[B, L, hidden_size]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: &MlxArray,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let x_shape = ffi::array_shape(x);
        let ctx_shape = ffi::array_shape(x_ctx);
        debug_assert_eq!(
            x_shape.len(),
            3,
            "DFlashAttention: x must be [B, L, H], got shape {x_shape:?}"
        );
        debug_assert_eq!(
            ctx_shape.len(),
            3,
            "DFlashAttention: x_ctx must be [B, S, H], got shape {ctx_shape:?}"
        );
        let b = x_shape[0];
        let l = x_shape[1];
        let s = ctx_shape[1];

        // Project both inputs.
        let queries = self.q_proj.forward(x);
        let ctx_keys = self.k_proj.forward(x_ctx);
        let ctx_values = self.v_proj.forward(x_ctx);
        let prop_keys = self.k_proj.forward(x);
        let prop_values = self.v_proj.forward(x);

        // Reshape to [B, seq, n_*, head_dim] for norm + transpose.
        let queries = ffi::reshape(&queries, &[b, l, self.n_heads, self.head_dim]);
        let ctx_keys = ffi::reshape(&ctx_keys, &[b, s, self.n_kv_heads, self.head_dim]);
        let ctx_values = ffi::reshape(&ctx_values, &[b, s, self.n_kv_heads, self.head_dim]);
        let prop_keys = ffi::reshape(&prop_keys, &[b, l, self.n_kv_heads, self.head_dim]);
        let prop_values = ffi::reshape(&prop_values, &[b, l, self.n_kv_heads, self.head_dim]);

        // RMSNorm over head_dim (last axis).
        let queries = self.q_norm.forward(&queries);
        let ctx_keys = self.k_norm.forward(&ctx_keys);
        let prop_keys = self.k_norm.forward(&prop_keys);

        // Transpose to [B, n_heads_or_kv, seq, head_dim].
        let queries = ffi::transpose_axes(&queries, &[0, 2, 1, 3]);
        let ctx_keys = ffi::transpose_axes(&ctx_keys, &[0, 2, 1, 3]);
        let ctx_values = ffi::transpose_axes(&ctx_values, &[0, 2, 1, 3]);
        let prop_keys = ffi::transpose_axes(&prop_keys, &[0, 2, 1, 3]);
        let prop_values = ffi::transpose_axes(&prop_values, &[0, 2, 1, 3]);

        // RoPE offsets per upstream:
        //   queries: cache.offset + S  (past + ctx)
        //   ctx_keys: cache.offset     (past)
        //   prop_keys: cache.offset + S  (proposal sits right after ctx)
        let past_offset = cache.offset;
        let after_ctx_offset = past_offset + s;

        // rope_dims = head_dim (full RoPE, traditional=False)
        let queries = ffi::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            after_ctx_offset,
        );
        let ctx_keys = ffi::fast_rope(
            &ctx_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            past_offset,
        );
        let prop_keys = ffi::fast_rope(
            &prop_keys,
            self.head_dim,
            false,
            self.rope_base,
            1.0,
            after_ctx_offset,
        );

        // Cache write: ONLY context K/V. The proposal K/V is concatenated
        // post-hoc and never enters the cache. This is the load-bearing
        // invariant of the DFlash drafter forward — pin it in the comments
        // so a future edit to this method cannot silently regress to
        // writing proposal K/V into the cache (which would corrupt the
        // next round's offset and double-count proposal positions in
        // every subsequent attention forward).
        let (keys, values) = cache.update_and_fetch(ctx_keys, ctx_values);

        // Concatenate proposal K/V along the seq axis (axis=2 for
        // [B, H, S+L, D] vs [B, H, S, D]).
        let keys_combined = crate::ops::concatenate(&keys, &prop_keys, 2);
        let values_combined = crate::ops::concatenate(&values, &prop_values, 2);

        // Fast scaled-dot-product attention with no mask (upstream passes
        // none — the drafter's masked forward bakes the mask into the
        // input tokens, not into the attention).
        let attn = unsafe {
            ffi::fast_scaled_dot_product_attention(
                &queries,
                &keys_combined,
                &values_combined,
                self.scale,
                std::ptr::null(),
            )
        };

        // [B, n_heads, L, head_dim] -> [B, L, n_heads, head_dim] -> [B, L, hidden]
        let attn = ffi::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = ffi::reshape(&attn, &[b, l, self.n_heads * self.head_dim]);

        self.o_proj.forward(&attn)
    }

    /// Load DFlash attention weights for one layer.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DFlashConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_key_value_heads as i32;
        let head_dim = config.head_dim as i32;

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), group_size, bits)?;

        // RMSNorm weights are stored as plain f32/f16/bf16 tensors and are
        // never quantized — load them as regular tensors through ffi::copy.
        let q_norm_w = weights
            .get(&format!("{prefix}.q_norm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.q_norm.weight"))?;
        let k_norm_w = weights
            .get(&format!("{prefix}.k_norm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.k_norm.weight"))?;

        let q_norm = RMSNorm::new(q_norm_w, config.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_w, config.rms_norm_eps);

        let scale = (config.head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            rope_base: config.rope_theta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    /// Build a synthetic DFlashAttention with deterministic small weights.
    ///
    /// Dimensions:
    /// - hidden_size = 8
    /// - n_heads = 2, n_kv_heads = 1, head_dim = 4
    ///   (so q out = 8, k/v out = 4, o in = 8, o out = 8)
    ///
    /// Weight stencils:
    /// - All projection weights are populated with identity-ish patterns to
    ///   keep the test reproducible across runs.
    /// - Norm weights are 1.0 so RMSNorm passes through (no further scaling).
    fn build_synthetic_attention() -> DFlashAttention {
        let mut weights: WeightMap = std::collections::HashMap::new();
        // q_proj: [8 out, 8 in] populated with identity (so q = x).
        let mut q_data = vec![0.0_f32; 64];
        for i in 0..8 {
            q_data[i * 8 + i] = 1.0;
        }
        weights.insert(
            "self_attn.q_proj.weight".to_string(),
            ffi::from_slice_f32(&q_data, &[8, 8]),
        );
        // k_proj / v_proj: [4 out, 8 in] populated with a simple pattern.
        let mut kv_data = vec![0.0_f32; 32];
        for i in 0..4 {
            kv_data[i * 8 + i] = 1.0;
        }
        weights.insert(
            "self_attn.k_proj.weight".to_string(),
            ffi::from_slice_f32(&kv_data, &[4, 8]),
        );
        weights.insert(
            "self_attn.v_proj.weight".to_string(),
            ffi::from_slice_f32(&kv_data, &[4, 8]),
        );
        // o_proj: [8 out, 8 in] identity.
        let mut o_data = vec![0.0_f32; 64];
        for i in 0..8 {
            o_data[i * 8 + i] = 1.0;
        }
        weights.insert(
            "self_attn.o_proj.weight".to_string(),
            ffi::from_slice_f32(&o_data, &[8, 8]),
        );
        // q_norm / k_norm weights: ones over head_dim = 4 → identity RMSNorm.
        weights.insert(
            "self_attn.q_norm.weight".to_string(),
            ffi::ones(&[4], dtype::FLOAT32),
        );
        weights.insert(
            "self_attn.k_norm.weight".to_string(),
            ffi::ones(&[4], dtype::FLOAT32),
        );

        let cfg = DFlashConfig {
            hidden_size: 8,
            intermediate_size: 16, // unused in this test
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            attention_bias: false,
            tie_word_embeddings: true,
            block_size: 4,
            mask_token_id: 31,
            target_layer_ids: vec![0],
            num_target_layers: 1,
        };
        DFlashAttention::from_weights(&weights, "self_attn", &cfg, 64, 4).unwrap()
    }

    /// Acceptance pin (issue #635):
    ///
    /// > `DFlashAttention`: with a fixed ctx and prop input and a synthetic
    /// > cache, the cache after `update_and_fetch` contains ONLY context K/V
    /// > (not proposal K/V).
    ///
    /// We assert the cache offset advances by exactly `S` (context length),
    /// not `S + L`. This is the split-projection invariant — if a future
    /// edit accidentally writes proposal K/V into the cache, the offset
    /// will jump to `S + L` and the round-loop driver in sub-12 will
    /// silently corrupt the next round.
    #[test]
    fn cache_offset_advances_by_ctx_only_not_by_prop() {
        let attn = build_synthetic_attention();

        // B = 1, S = 3 (ctx length), L = 2 (prop length).
        // Use rank-1 patterns so the forward returns a well-defined output;
        // contents do not matter for this test, only the cache offset does.
        let x_ctx = ffi::from_slice_f32(
            &[
                // [1, 3, 8] context
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, //
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, //
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, //
            ],
            &[1, 3, 8],
        );
        let x = ffi::from_slice_f32(
            &[
                // [1, 2, 8] proposal
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, //
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, //
            ],
            &[1, 2, 8],
        );
        let mut cache = KVCache::new();

        let initial_offset = cache.offset;
        assert_eq!(initial_offset, 0, "fresh cache must have offset 0");

        let _out = attn.forward(&x, &x_ctx, &mut cache);

        // After one forward, the cache must contain ONLY the context K/V.
        // Offset must equal `initial_offset + S = 3`, not `initial_offset + S + L = 5`.
        assert_eq!(
            cache.offset, 3,
            "cache offset must advance by S=3 (ctx only), not by S+L=5; \
             got offset={} (regression: proposal K/V leaked into cache)",
            cache.offset
        );
    }

    /// The attention output preserves the proposal shape `[B, L, hidden]`.
    /// This is the second half of the split-projection contract: the
    /// forward returns one row per proposal position, not one row per
    /// (proposal + context) position.
    #[test]
    fn forward_output_shape_matches_proposal_shape() {
        let attn = build_synthetic_attention();
        let x_ctx = ffi::from_slice_f32(
            &[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, //
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, //
            ],
            &[1, 2, 8],
        );
        let x = ffi::from_slice_f32(
            &[
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, //
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, //
                0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, //
            ],
            &[1, 3, 8],
        );
        let mut cache = KVCache::new();

        let out = attn.forward(&x, &x_ctx, &mut cache);
        ffi::eval(&out);

        let out_shape = ffi::array_shape(&out);
        // Out must be [B=1, L=3, hidden=8] — matches proposal length, not
        // context length nor concatenation length.
        assert_eq!(
            out_shape,
            vec![1, 3, 8],
            "DFlashAttention output shape must be [B, L_prop, hidden]; got {out_shape:?}"
        );
    }

    /// Two consecutive forwards (e.g. two draft rounds with fresh proposals
    /// but the same context buffer length) must each advance the cache by
    /// only `S`, not by `S + L`. This pins the invariant across rounds:
    /// proposal K/V is concatenated post-hoc per forward and is NEVER
    /// retained between rounds.
    #[test]
    fn cache_offset_advances_only_by_ctx_across_multiple_rounds() {
        let attn = build_synthetic_attention();
        let x_ctx = ffi::from_slice_f32(
            &[0.1_f32; 24], // [1, 3, 8] all 0.1
            &[1, 3, 8],
        );
        let x = ffi::from_slice_f32(
            &[1.0_f32; 16], // [1, 2, 8] all 1.0
            &[1, 2, 8],
        );
        let mut cache = KVCache::new();

        let _ = attn.forward(&x, &x_ctx, &mut cache);
        assert_eq!(cache.offset, 3, "after round 1: offset = 3 (S only)");

        let _ = attn.forward(&x, &x_ctx, &mut cache);
        assert_eq!(
            cache.offset, 6,
            "after round 2: offset = 6 (2*S only). If proposal K/V leaked, \
             offset would be 10 (2*(S+L)). Got {}",
            cache.offset
        );
    }
}
