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

//! Talkie attention: RoPE with the negated table, then the weightless q/k
//! norm, then the per-head query gain.

use mlxcel_core::batched_decode::{SplitAttention, SplitAttentionShape};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{ModelArgs, apply_gain, gain, norm};
use crate::models::nanochat::mirrored_rope_freqs;

pub struct Attention {
    pub attn_query: UnifiedLinear,
    pub attn_key: UnifiedLinear,
    pub attn_value: UnifiedLinear,
    pub attn_resid: UnifiedLinear,
    /// Per-head query gain, reshaped to `[1, H, 1, 1]`.
    pub head_gain: UniquePtr<MlxArray>,
    pub freqs: UniquePtr<MlxArray>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub eps: f32,
}

impl Attention {
    /// The q/k pipeline after the projections: RoPE, then the weightless
    /// norm, then the per-head gain on q. The gain must follow the norm or
    /// the norm erases it.
    pub(crate) fn finish_qk(
        &self,
        q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let q = norm(&q, self.eps);
        let k = norm(&k, self.eps);
        (apply_gain(&q, &self.head_gain), k)
    }

    pub(crate) fn rotate_norm_gain(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let q =
            mlxcel_core::fast_rope_with_freqs(q, self.head_dim, false, 1.0, offset, &self.freqs);
        let k =
            mlxcel_core::fast_rope_with_freqs(k, self.head_dim, false, 1.0, offset, &self.freqs);
        self.finish_qk(q, k)
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let split = |t: UniquePtr<MlxArray>| {
            let t = mlxcel_core::reshape(&t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = split(self.attn_query.forward(x));
        let k = split(self.attn_key.forward(x));
        let v = split(self.attn_value.forward(x));

        let (q, k) = self.rotate_norm_gain(&q, &k, cache.offset);
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // A prefill with no caller-supplied mask takes the causal path: every
        // generation path passes `mask == None` and expects the model to build
        // its own causal mask.
        let attn = if let Some(m) = mask {
            let mask_ptr = m as *const _;
            // SAFETY: `mask_ptr` derives from a live `&MlxArray` that outlives
            // this call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, self.num_heads * self.head_dim]);
        self.attn_resid.forward(&attn)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let (gs, bits) = (args.group_size(), args.bits());
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), gs, bits)
        };
        let num_heads = args.num_attention_heads as i32;
        let head_gain = gain(weights, &format!("{prefix}.head_gain.head_g"))?;
        let gain_len: i64 = mlxcel_core::array_shape(&head_gain)
            .iter()
            .map(|&d| i64::from(d))
            .product();
        if gain_len != i64::from(num_heads) {
            return Err(format!(
                "talkie: {prefix}.head_gain.head_g has {gain_len} entries, expected {num_heads}"
            ));
        }
        Ok(Self {
            attn_query: lin("attn_query")?,
            attn_key: lin("attn_key")?,
            attn_value: lin("attn_value")?,
            attn_resid: lin("attn_resid")?,
            head_gain: mlxcel_core::reshape(&head_gain, &[1, num_heads, 1, 1]),
            freqs: mirrored_rope_freqs(args.head_dim, args.rope_theta),
            num_heads,
            head_dim: args.head_dim as i32,
            scale: (args.head_dim as f32).powf(-0.5),
            eps: args.rms_norm_eps,
        })
    }
}

impl SplitAttention for Attention {
    fn split_shape(&self) -> SplitAttentionShape {
        SplitAttentionShape {
            num_heads: self.num_heads,
            num_kv_heads: self.num_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }
    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let rope = |t: &MlxArray| {
            mlxcel_core::fast_rope_batched_with_freqs(
                t,
                self.head_dim,
                false,
                1.0,
                offsets,
                &self.freqs,
            )
        };
        self.finish_qk(rope(q), rope(k))
    }
}
