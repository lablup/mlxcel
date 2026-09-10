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

//! Sliding-window attention of the Muse Glimmer assistant drafter (issue
//! #1343).
//!
//! Split-projection attention like the Qwen 3.5 DFlash layer, with three
//! differences the published checkpoint fixes:
//!
//! - per-head `q_norm` / `k_norm` (`RMSNorm(head_dim)`) applied before RoPE;
//! - the context rows and the proposal rows go through `k_proj` / `v_proj`
//!   as ONE concatenated input, so a single RoPE at `cache.offset` places the
//!   context at `offset..offset + S` and the proposals right after it;
//! - a bidirectional sliding-window mask over ABSOLUTE positions,
//!   `|query_pos - key_pos| <= sliding_window`, instead of the maskless SDPA
//!   the Qwen drafter runs. Only the context rows enter the cache; the
//!   proposal K/V is concatenated onto the fetched window and never stored.

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::ops::concatenate;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::MuseAssistantContextCache;
use super::config::MuseAssistantConfig;

/// Boolean `[q_len, k_len]` attend matrix of the bidirectional sliding
/// window: `true` where `|(query_start + i) - (key_start + j)| <= window`.
///
/// Not causal on purpose: a draft block is predicted in one non-causal
/// forward, every proposal slot sees every other slot and the whole visible
/// context, and the window is the only restriction.
///
/// Used by: [`MuseAssistantAttention::forward`], the muse drafter tests.
pub fn bidirectional_sliding_mask_bool(
    query_start: i32,
    q_len: i32,
    key_start: i32,
    k_len: i32,
    window: i32,
) -> UniquePtr<MlxArray> {
    let q_idx = ffi::reshape(
        &ffi::arange_i32(query_start, query_start + q_len, 1),
        &[q_len, 1],
    );
    let k_idx = ffi::reshape(
        &ffi::arange_i32(key_start, key_start + k_len, 1),
        &[1, k_len],
    );
    let dist = ffi::abs(&ffi::subtract(&q_idx, &k_idx));
    let window_arr = ffi::from_slice_i32(&[window], &[1]);
    ffi::less_equal(&dist, &window_arr)
}

/// The additive `[1, 1, q_len, k_len]` bias of
/// [`bidirectional_sliding_mask_bool`]: `0` where the pair attends, `-inf`
/// elsewhere, in `dtype`. Every query row keeps at least itself, so no row
/// is all `-inf`. The drafter forward hands the fused SDPA the boolean form
/// instead, which is dtype-free; this is the form a caller that composes
/// masks additively wants.
pub fn bidirectional_sliding_mask(
    query_start: i32,
    q_len: i32,
    key_start: i32,
    k_len: i32,
    window: i32,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let attend = bidirectional_sliding_mask_bool(query_start, q_len, key_start, k_len, window);
    let zero = ffi::full_f32(&[q_len, k_len], 0.0, dtype);
    let neg_inf = ffi::full_f32(&[q_len, k_len], f32::NEG_INFINITY, dtype);
    let bias = ffi::where_cond(&attend, &zero, &neg_inf);
    ffi::reshape(&bias, &[1, 1, q_len, k_len])
}

/// One drafter attention block.
pub struct MuseAssistantAttention {
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
    pub rope_theta: f32,
    pub sliding_window: i32,
}

impl MuseAssistantAttention {
    /// Load `{prefix}.{q,k,v,o}_proj` and `{prefix}.{q,k}_norm`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MuseAssistantConfig,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let load_norm = |name: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{name}.weight");
            let weight = weights
                .get(&key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(RMSNorm::new(weight, config.rms_norm_eps))
        };
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.k_proj"),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.v_proj"),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.o_proj"),
                group_size,
                bits,
            )?,
            q_norm: load_norm("q_norm")?,
            k_norm: load_norm("k_norm")?,
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5),
            rope_theta: config.rope_theta(),
            sliding_window: config.sliding_window as i32,
        })
    }

    /// Attend the `[B, L, hidden]` proposal rows `x` over the `[B, S,
    /// hidden]` context rows plus the cached window, then over themselves.
    ///
    /// `S` may be zero on a round whose context was consumed by an earlier
    /// layer call; the cache is then left as it is.
    pub fn forward(
        &self,
        x: &MlxArray,
        context: &MlxArray,
        cache: &mut MuseAssistantContextCache,
    ) -> UniquePtr<MlxArray> {
        let x_shape = ffi::array_shape(x);
        let ctx_shape = ffi::array_shape(context);
        debug_assert_eq!(x_shape.len(), 3, "x must be [B, L, H], got {x_shape:?}");
        debug_assert_eq!(
            ctx_shape.len(),
            3,
            "context must be [B, S, H], got {ctx_shape:?}"
        );
        let b = x_shape[0];
        let l = x_shape[1];
        let s = ctx_shape[1];
        let past = cache.offset;

        // Queries: proposal rows at absolute positions `past + S ..`.
        let queries = self.q_proj.forward(x);
        let queries = ffi::reshape(&queries, &[b, l, self.n_heads, self.head_dim]);
        let queries = self.q_norm.forward(&queries);
        let queries = ffi::transpose_axes(&queries, &[0, 2, 1, 3]);
        let queries = ffi::fast_rope(
            &queries,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            past + s,
        );

        // Keys / values over `[context ; x]` in one projection. One RoPE at
        // `past` puts the context at `past..past + S` and the proposals at
        // `past + S..`, the same positions the queries were given.
        let kv_in = if s > 0 {
            concatenate(context, x, 1)
        } else {
            ffi::copy(x)
        };
        let keys = self.k_proj.forward(&kv_in);
        let keys = ffi::reshape(&keys, &[b, s + l, self.n_kv_heads, self.head_dim]);
        let keys = self.k_norm.forward(&keys);
        let keys = ffi::transpose_axes(&keys, &[0, 2, 1, 3]);
        let keys = ffi::fast_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, past);
        let values = self.v_proj.forward(&kv_in);
        let values = ffi::reshape(&values, &[b, s + l, self.n_kv_heads, self.head_dim]);
        let values = ffi::transpose_axes(&values, &[0, 2, 1, 3]);

        // Only the context rows enter the cache. The proposal rows are
        // concatenated after the fetched window and never stored, which is
        // what keeps `cache.offset` equal to the committed row count.
        let (keys_all, values_all) = if s > 0 {
            let ctx_keys = ffi::slice(
                &keys,
                &[0, 0, 0, 0],
                &[b, self.n_kv_heads, s, self.head_dim],
            );
            let ctx_values = ffi::slice(
                &values,
                &[0, 0, 0, 0],
                &[b, self.n_kv_heads, s, self.head_dim],
            );
            let prop_keys = ffi::slice(
                &keys,
                &[0, 0, s, 0],
                &[b, self.n_kv_heads, s + l, self.head_dim],
            );
            let prop_values = ffi::slice(
                &values,
                &[0, 0, s, 0],
                &[b, self.n_kv_heads, s + l, self.head_dim],
            );
            let (cached_keys, cached_values) = cache.append(ctx_keys, ctx_values);
            (
                concatenate(&cached_keys, &prop_keys, 2),
                concatenate(&cached_values, &prop_values, 2),
            )
        } else {
            match cache.fetch() {
                Some((cached_keys, cached_values)) => (
                    concatenate(&cached_keys, &keys, 2),
                    concatenate(&cached_values, &values, 2),
                ),
                None => (keys, values),
            }
        };

        // A boolean mask (`true` attends), which the fused SDPA takes for any
        // activation dtype. An additive bias would have to match the dtype
        // the kernel promotes q/k/v to, and that is not the query dtype when
        // the target's residual streams and the drafter's dense weights
        // differ; a mismatch is an MLX throw across the cxx bridge.
        let k_len = cache.len() + l;
        let mask = bidirectional_sliding_mask_bool(
            cache.offset,
            l,
            cache.key_start(),
            k_len,
            self.sliding_window,
        );
        let mask_ptr: *const MlxArray = mask.as_ref().map_or(std::ptr::null(), |m| m);
        // SAFETY: every array is alive for the duration of the call and the
        // mask pointer is borrowed from `mask`, which outlives the call.
        let attn = unsafe {
            ffi::fast_scaled_dot_product_attention(
                &queries,
                &keys_all,
                &values_all,
                self.scale,
                mask_ptr,
            )
        };
        let attn = ffi::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = ffi::reshape(&attn, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }
}
