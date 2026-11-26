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

//! [`DFlash2Attention`]: Qwen3-style attention from the proposal block over
//! `[context window | block]`, non-causal.
//!
//! q, k and v come from the block rows with per-head `q_norm` / `k_norm`
//! and RoPE at the block's absolute positions; the context keys were built
//! the same way from `hn` when the context was appended
//! ([`DFlash2Attention::context_kv`]). The scale is `head_dim ** -0.5` and
//! there is no mask: every block row sees the whole window and the whole
//! block.

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::ops::concatenate;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::ContextWindow;
use super::config::RaonDFlash2Config;

pub struct DFlash2Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl DFlash2Attention {
    /// # Errors
    /// Returns a message naming a missing tensor.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &RaonDFlash2Config,
    ) -> Result<Self, String> {
        let linear =
            |leaf: &str| UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), 64, 4);
        let norm = |leaf: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))?;
            Ok(RMSNorm::new(ffi::copy(w), config.rms_norm_eps))
        };
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            n_heads: config.num_attention_heads as i32,
            n_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5),
            rope_theta: config.rope_theta,
        })
    }

    /// `[n, heads * D]` rows to `[1, heads, n, D]`, normed per head when a
    /// norm is given.
    fn heads(&self, x: &MlxArray, heads: i32, norm: Option<&RMSNorm>) -> UniquePtr<MlxArray> {
        let n = ffi::array_shape(x)[0];
        let x = ffi::reshape(x, &[1, n, heads, self.head_dim]);
        let x = match norm {
            Some(norm) => norm.forward(&x),
            None => x,
        };
        ffi::transpose_axes(&x, &[0, 2, 1, 3])
    }

    fn rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        ffi::fast_rope(x, self.head_dim, false, self.rope_theta, 1.0, offset)
    }

    /// Context K/V for `hn` (`[T, hidden]`, the normed `fc` projection of the
    /// target features at positions `offset..offset + T`).
    pub fn context_kv(
        &self,
        hn: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let k = self.heads(
            &self.k_proj.forward(hn),
            self.n_kv_heads,
            Some(&self.k_norm),
        );
        let v = self.heads(&self.v_proj.forward(hn), self.n_kv_heads, None);
        (self.rope(&k, offset), v)
    }

    /// Attention of block rows `x` (`[n, hidden]`, positions `pos..pos + n`)
    /// over `[window | block]`; returns `o_proj` of it, `[n, hidden]`.
    pub fn forward(&self, x: &MlxArray, pos: i32, window: &ContextWindow) -> UniquePtr<MlxArray> {
        let n = ffi::array_shape(x)[0];
        let q = self.heads(&self.q_proj.forward(x), self.n_heads, Some(&self.q_norm));
        let k = self.heads(&self.k_proj.forward(x), self.n_kv_heads, Some(&self.k_norm));
        let v = self.heads(&self.v_proj.forward(x), self.n_kv_heads, None);
        let q = self.rope(&q, pos);
        let k = self.rope(&k, pos);
        let (k, v) = match window.recent(n) {
            Some((ck, cv)) => (concatenate(&ck, &k, 2), concatenate(&cv, &v, 2)),
            None => (k, v),
        };
        // SAFETY: a null mask pointer selects unmasked attention, and the
        // three operands outlive the call. The fast kernel expands the
        // grouped KV heads (the plain bridge SDPA broadcasts them, which only
        // works for one KV head).
        let o = unsafe {
            ffi::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let o = ffi::transpose_axes(&o, &[0, 2, 1, 3]);
        let o = ffi::reshape(&o, &[n, self.n_heads * self.head_dim]);
        self.o_proj.forward(&o)
    }
}
