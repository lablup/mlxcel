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
// Portions of this file are derived from mlx-vlm
// (https://github.com/Blaizzy/mlx-vlm), Copyright 2025 Prince Canuma,
// licensed under the MIT License. See the top-level NOTICE file for the
// attribution carried forward under the MIT License.

//! Shared layers for the Qwen3-Omni speech stack (stage 2): the GQA
//! attention used by both the talker decoder and the code predictor, the
//! thinker-to-talker resize MLPs, and the sampling helper.
//!
//! The reference talker applies MRoPE with all three position axes equal to
//! the sequential text position, which is exactly standard half-split RoPE,
//! so `fast_rope` (non-traditional) is used here.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/talker.py>.
//!
//! Used by: Qwen3-Omni MoE speech pipeline (talker.rs).

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(super) fn load_rms_norm(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<RMSNorm, String> {
    let key = format!("{prefix}.weight");
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Qwen3-Omni talker weight missing: {key}"))?;
    Ok(RMSNorm::new(weight, eps))
}

/// Sample one token from `[1, vocab]` logits with temperature + nucleus
/// filtering (the reference `top_p_sampling`). `top_p >= 1.0` degrades to
/// pure temperature sampling; `temperature == 0.0` is greedy.
pub(super) fn sample_logits(logits: &MlxArray, temperature: f32, top_p: f32) -> i32 {
    let token = mlxcel_core::fused_sample(logits, temperature, 0, top_p, 0.0);
    mlxcel_core::eval(&token);
    mlxcel_core::item_i32(&token)
}

/// GQA attention with QK-RMSNorm and standard (half-split) RoPE, shared by
/// the talker decoder layers and the code predictor layers.
pub(super) struct SpeechAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_base: f32,
    scale: f32,
}

impl SpeechAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f32,
        rope_base: f32,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?,
            k_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), gs, bits)?,
            o_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?,
            q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), rms_norm_eps)?,
            k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), rms_norm_eps)?,
            num_heads: num_heads as i32,
            num_kv_heads: num_kv_heads as i32,
            head_dim: head_dim as i32,
            rope_base,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub(super) fn forward(&self, x: &MlxArray, cache: &mut KVCache) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // QK-RMSNorm before RoPE, per the reference.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        let (k, v) = cache.update_and_fetch(k, v);

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&k, n_rep)
        } else {
            mlxcel_core::copy(&k)
        };
        let v = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&v, n_rep)
        } else {
            mlxcel_core::copy(&v)
        };

        // The talker and code predictor always run on fresh caches, so a
        // multi-token step is a pure prefill (causal) and a single-token step
        // attends over the whole cache (no mask).
        let output = if l > 1 {
            mlxcel_core::causal_attention(&q, &k, &v, self.scale, 0.0, 0)
        } else {
            // SAFETY: null mask means full attention over cached keys.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

/// SwiGLU-style resize MLP (`linear_fc2(silu(linear_fc1(x)))`) mapping the
/// thinker hidden size (2048) into the talker hidden size (1024). Both
/// linears carry biases.
pub struct ResizeMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl ResizeMlp {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_fc2"), gs, bits)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::silu(&h);
        self.fc2.forward(&h)
    }
}
