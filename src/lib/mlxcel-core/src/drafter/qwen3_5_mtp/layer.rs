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

//! Decoder layer for the Qwen 3.5 MTP drafter.
//!
//! Upstream builds each MTP layer as `Qwen3_5DecoderLayer(replace(text_config,
//! num_hidden_layers=mtp_layers, full_attention_interval=1))` — the
//! `full_attention_interval=1` forces every MTP layer onto the full-attention
//! path, never gated-delta. The published Qwen3.8-27B MTP tensors confirm it
//! (`self_attn.*` only, no `linear_attn.*`).
//!
//! This is an in-crate port of the target-side full-attention layer
//! (`Qwen3NextAttention` + dense `MLP` in `src/models/qwen3_next.rs` of the
//! `mlxcel` binary crate, which sits above `mlxcel-core` and is therefore not
//! reachable from here — the same layering constraint that gave DFlash its own
//! in-crate attention/MLP). Differences from the target-side layer:
//!
//! - No MRoPE branch: the drafter never sees multimodal position ids
//!   (upstream's drafter passes plain integer `position_ids`, and the
//!   target-side DFlash speculative forward takes the same standard-RoPE path).
//! - The RoPE offset is an explicit `rope_offset` argument instead of
//!   `cache.offset`: the drafter tracks its own logical `_next_position`,
//!   which can run ahead of its cache length when the prompt-side drafter
//!   prefill was skipped (adopted prompt prefix, or a reset between slice
//!   grants). Cache storage geometry still comes from the cache itself.
//! - No `target_verify` per-position path: drafter outputs never need
//!   bit-parity with single-token decode — greedy parity is enforced on the
//!   TARGET's verify pass, and drafter numerics only influence acceptance.
//!
//! The Q projection carries the Qwen 3.5 `attn_output_gate` layout: `q_proj`
//! emits queries and a sigmoid gate concatenated per head
//! (`[B, L, num_heads, 2 * head_dim]`), and the attention output is
//! `o_proj(attn * sigmoid(gate))`.

use crate::cache::KVCache;
use crate::drafter::dflash::mlp::DFlashMlp;
use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::config::Qwen35MtpTextConfig;

/// Full-attention block of one MTP decoder layer (gated-Q Qwen 3.5 layout).
pub struct Qwen35MtpAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_dims: i32,
    rope_base: f32,
}

impl Qwen35MtpAttention {
    /// Attention forward over the drafter's own KV cache.
    ///
    /// - `x`: `[B, L, hidden_size]` input (post `input_layernorm`).
    /// - `mask`: `Some(causal mask)` for multi-token forwards (drafter
    ///   prompt prefill and accepted-token extension), `None` for the
    ///   single-token draft steps.
    /// - `cache`: this layer's own KV cache; keys/values are appended.
    /// - `rope_offset`: absolute position of `x`'s first token in the
    ///   target sequence (the drafter's `next_position`).
    pub fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        rope_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Gated Q projection: [B, L, 2 * num_heads * head_dim], reshaped to
        // [B, L, num_heads, 2 * head_dim] and split into queries + gate.
        let q_proj_output = self.q_proj.forward(x);
        let q_proj_reshaped = ffi::reshape(&q_proj_output, &[b, l, self.num_heads, -1]);
        let queries = ffi::slice(
            &q_proj_reshaped,
            &[0, 0, 0, 0],
            &[b, l, self.num_heads, self.head_dim],
        );
        let q_last_dim = ffi::array_shape(&q_proj_reshaped)[3];
        let gate = ffi::slice(
            &q_proj_reshaped,
            &[0, 0, 0, self.head_dim],
            &[b, l, self.num_heads, q_last_dim],
        );
        let gate = ffi::reshape(&gate, &[b, l, -1]);

        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        let queries = ffi::reshape(&queries, &[b, l, self.num_heads, self.head_dim]);
        let keys = ffi::reshape(&keys, &[b, l, self.num_kv_heads, self.head_dim]);
        let values = ffi::reshape(&values, &[b, l, self.num_kv_heads, self.head_dim]);

        // Per-head RMSNorm over head_dim, applied before transpose — same
        // order as the target-side layer.
        let queries = self.q_norm.forward(&queries);
        let keys = self.k_norm.forward(&keys);

        let queries = ffi::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = ffi::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = ffi::transpose_axes(&values, &[0, 2, 1, 3]);

        // Partial rotary embedding at the drafter's logical position.
        let queries = ffi::fast_rope(
            &queries,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            rope_offset,
        );
        let keys = ffi::fast_rope(
            &keys,
            self.rope_dims,
            false,
            self.rope_base,
            1.0,
            rope_offset,
        );

        let (cache_k, cache_v) = cache.update_and_fetch(keys, values);

        let attn_out =
            crate::layers::attention(&queries, &cache_k, &cache_v, self.scale, mask, 0.0, 0);

        let output = ffi::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let output = ffi::reshape(&output, &[b, l, -1]);

        // Sigmoid output gate, then o_proj — the `attn_output_gate` path.
        let gate_sigmoid = ffi::sigmoid(&gate);
        let gated = ffi::multiply(&output, &gate_sigmoid);
        self.o_proj.forward(&gated)
    }

    /// Load one layer's attention weights from `{prefix}.{q,k,v,o}_proj` and
    /// `{prefix}.{q,k}_norm`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Qwen35MtpTextConfig,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), group_size, bits)?;

        let q_norm_w = weights
            .get(&format!("{prefix}.q_norm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.q_norm.weight"))?;
        let k_norm_w = weights
            .get(&format!("{prefix}.k_norm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.k_norm.weight"))?;

        let head_dim = config.head_dim_resolved() as i32;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_w, config.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_w, config.rms_norm_eps),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: config.rope_dims(),
            rope_base: config.rope_theta(),
        })
    }
}

/// One MTP decoder layer: pre-norm attention + pre-norm SwiGLU MLP with
/// residual connections, mirroring the target-side `Qwen35DecoderLayer`
/// full-attention arm.
pub struct Qwen35MtpDecoderLayer {
    attention: Qwen35MtpAttention,
    mlp: DFlashMlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen35MtpDecoderLayer {
    /// Layer forward. Argument semantics match
    /// [`Qwen35MtpAttention::forward`].
    pub fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
        rope_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let r = self.attention.forward(&normed, mask, cache, rope_offset);
        let h = ffi::add(x, &r);
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        ffi::add(&h, &mlp_out)
    }

    /// Load one layer from `{prefix}.self_attn.*`, `{prefix}.mlp.*`,
    /// `{prefix}.input_layernorm.weight`,
    /// `{prefix}.post_attention_layernorm.weight`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Qwen35MtpTextConfig,
    ) -> Result<Self, String> {
        let attention =
            Qwen35MtpAttention::from_weights(weights, &format!("{prefix}.self_attn"), config)?;
        let mlp = DFlashMlp::from_weights(
            weights,
            &format!("{prefix}.mlp"),
            config.group_size(),
            config.bits(),
        )?;

        let input_norm_w = weights
            .get(&format!("{prefix}.input_layernorm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.input_layernorm.weight"))?;
        let post_norm_w = weights
            .get(&format!("{prefix}.post_attention_layernorm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.post_attention_layernorm.weight"))?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_w, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_w, config.rms_norm_eps),
        })
    }
}
