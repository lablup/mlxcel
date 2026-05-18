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

//! KV-shared-only decoder layer used by the Gemma 4 MTP assistant drafter.
//!
//! This is a deliberate, narrowly-scoped re-implementation of the gemma4
//! [`DecoderLayer`](crate::models::gemma4::DecoderLayer) (in the `mlxcel`
//! crate) for the **strict subset** of features the drafter exercises:
//!
//! - **No own KV cache.** Self-attention always reads K/V from a caller-
//!   supplied `shared_kv` tuple (the target's last full / SWA layer K/V).
//! - **No K/V projection.** With `kv_shared_only=True` on every layer, the
//!   layer never computes its own K or V, so `k_proj`, `v_proj`, `k_norm`,
//!   and `v_norm` are all dropped.
//! - **No MoE.** The drafter is always dense (`enable_moe_block=False`).
//! - **No per-layer input gating.** Drafter `text_config` always has
//!   `hidden_size_per_layer_input = 0`, so the per-layer-input gate /
//!   projection / norm and `layer_scalar` are all absent here.
//! - **Frozen RoPE position.** Cross-attention queries are rotated at the
//!   caller-supplied bonus-token offset and held constant across the K
//!   autoregressive steps within a single draft block.
//!
//! Placing this layer inside `mlxcel-core` avoids a circular dependency on the
//! `mlxcel` crate (where the full gemma4 `DecoderLayer` lives). The drafter
//! checkpoint follows the same HF weight-key conventions as the target
//! (`self_attn.q_proj`, `self_attn.q_norm`, `self_attn.o_proj`,
//! `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `input_layernorm`,
//! `post_attention_layernorm`, `pre_feedforward_layernorm`,
//! `post_feedforward_layernorm`), so weight loading is straightforward.

use crate::drafter::gemma4_assistant::config::DrafterTextConfig;
use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::rope_proportional::{
    apply_proportional_rope, apply_proportional_rope_batched, compute_proportional_rope_freqs,
};
use crate::weights::WeightMap;
use cxx::UniquePtr;

/// Frozen RoPE anchor for a drafter block.
///
/// B=1 uses a scalar anchor. Batched MTP uses per-row anchors after rows
/// accept different numbers of speculative tokens; the attention layer
/// applies the same anchor to every autoregressive step within the block.
#[derive(Clone, Copy)]
pub(crate) enum RopeOffset<'a> {
    Scalar(i32),
    PerRow(&'a [i32]),
}

/// MLP block (gate_proj + up_proj + down_proj) for the drafter.
///
/// The activation is GELU-approx applied to `gate_proj(x)` then multiplied
/// element-wise with `up_proj(x)` (the standard `GeGLU` pattern Gemma uses).
/// Uses the same `compiled_gelu_mlp_fp16` fast path the gemma4 target uses
/// when both projections are in fp16; falls back to the op-at-a-time chain
/// otherwise. No MoE.
pub(crate) struct DrafterMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DrafterMlp {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &DrafterTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                config.group_size(),
                config.bits(),
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                config.group_size(),
                config.bits(),
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                config.group_size(),
                config.bits(),
            )?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(out) = crate::layers::compiled_gelu_mlp_fp16(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return out;
        }
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let hidden = ffi::compiled_geglu_approx_activation(&gate, &up);
        self.down_proj.forward(&hidden)
    }
}

/// Self-attention block for a KV-shared drafter layer. Q-only projection,
/// reads K/V from the caller-supplied target slabs.
pub(crate) struct DrafterAttention {
    q_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    /// `Some(freqs)` iff this layer uses proportional RoPE (full-attention).
    proportional_rope_freqs: Option<UniquePtr<MlxArray>>,
    proportional_partial_rotary_factor: f32,
    rope_dims: i32,
    rope_theta: f32,
    /// Sliding-window size for sliding-attention layers; 0 for full-attention.
    /// Forwarded into `mlxcel_core::layers::attention` so the SDPA kernel can
    /// short-circuit window-trim when `kv_len <= window_size`.
    window_size: i32,
    /// `scale` passed to SDPA. Matches the gemma4 attention default of `1.0`
    /// (RMSNorm-normalised Q already produces unit-norm vectors).
    scale: f32,
}

impl DrafterAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &DrafterTextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim_for_layer(layer_idx);
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_kv_heads_for_layer(layer_idx);
        let rope_params = config.rope_params_for_layer(layer_idx);
        let rope_dims = (head_dim as f32 * rope_params.partial_rotary_factor) as i32;
        let proportional_rope_freqs = if rope_params.rope_type == "proportional" {
            compute_proportional_rope_freqs(
                head_dim,
                rope_params.partial_rotary_factor,
                rope_params.rope_theta,
                1.0,
            )
        } else {
            None
        };

        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                config.group_size(),
                config.bits(),
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.o_proj"),
                config.group_size(),
                config.bits(),
            )?,
            q_norm: RMSNorm::new(
                weight_copy(weights, &format!("{prefix}.q_norm.weight"))?,
                config.rms_norm_eps,
            ),
            n_heads,
            n_kv_heads,
            head_dim,
            proportional_rope_freqs,
            proportional_partial_rotary_factor: rope_params.partial_rotary_factor,
            rope_dims,
            rope_theta: rope_params.rope_theta,
            window_size: if config.is_sliding_layer(layer_idx) {
                config.sliding_window as i32
            } else {
                0
            },
            scale: 1.0,
        })
    }

    /// KV-shared forward: Q is projected and rotated; K/V come from the
    /// caller's shared tuple. Mirrors the upstream Python `forward(..., cache=None,
    /// shared_kv=kv, offset=offset)` shape for `kv_shared_only=True` layers.
    ///
    /// - `x`: drafter hidden, shape `[B, L=1, hidden_size]` per step.
    /// - `mask`: optional additive mask compatible with `attention(...)`.
    /// - `shared_keys` / `shared_values`: target's last-layer K/V slabs for
    ///   this layer's type. Both `[B, num_kv_heads, kv_len, head_dim]`.
    /// - `offset`: bonus-token absolute position used to rotate Q. The same
    ///   value MUST be passed for every step in a draft block so cross-
    ///   attention queries are aligned to the frozen anchor.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        shared_keys: &MlxArray,
        shared_values: &MlxArray,
        offset: RopeOffset<'_>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q_proj_out = self.q_proj.forward(x);
        let queries = ffi::reshape(&q_proj_out, &[b, l, self.n_heads, self.head_dim]);
        let queries = self.q_norm.forward(&queries);
        let queries = ffi::transpose_axes(&queries, &[0, 2, 1, 3]);
        let queries = match (self.proportional_rope_freqs.as_deref(), offset) {
            (Some(freqs), RopeOffset::Scalar(offset)) => apply_proportional_rope(
                &queries,
                self.head_dim,
                self.proportional_partial_rotary_factor,
                offset,
                Some(freqs),
            ),
            (Some(freqs), RopeOffset::PerRow(offsets)) => apply_proportional_rope_batched(
                &queries,
                self.head_dim,
                self.proportional_partial_rotary_factor,
                offsets,
                Some(freqs),
            ),
            (None, RopeOffset::Scalar(offset)) => ffi::fast_rope(
                &queries,
                self.rope_dims,
                false,
                self.rope_theta,
                1.0,
                offset,
            ),
            (None, RopeOffset::PerRow(offsets)) => crate::fast_rope_batched(
                &queries,
                self.rope_dims,
                false,
                self.rope_theta,
                1.0,
                offsets,
            ),
        };

        let attn_out = crate::layers::attention(
            &queries,
            shared_keys,
            shared_values,
            self.scale,
            mask,
            0.0,
            self.window_size,
        );

        let attn_out = ffi::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = ffi::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    /// Layer's effective `n_kv_heads` — exposed for diagnostics / tests.
    #[allow(dead_code)]
    pub(crate) fn n_kv_heads(&self) -> i32 {
        self.n_kv_heads
    }

    /// Layer's window size (0 for full-attention layers).
    #[allow(dead_code)]
    pub(crate) fn window_size(&self) -> i32 {
        self.window_size
    }
}

/// Decoder layer for the drafter. KV-shared-only (no own cache), dense MLP,
/// no per-layer input.
pub(crate) struct DraftDecoderLayer {
    self_attn: DrafterAttention,
    mlp: DrafterMlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
    /// Learned per-layer output scalar from Gemma 4 assistant checkpoints.
    ///
    /// Used by: Gemma 4 MTP assistant drafter. Upstream reuses the target
    /// `DecoderLayer` implementation, whose final step multiplies every layer
    /// output by `layer_scalar`. The 31B assistant checkpoint ships non-trivial
    /// values (well below 1.0), so omitting this multiply makes the drafter
    /// distribution diverge and collapses MTP acceptance.
    layer_scalar: Option<UniquePtr<MlxArray>>,
    layer_type: String,
}

impl DraftDecoderLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &DrafterTextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: DrafterAttention::from_weights(
                weights,
                config,
                layer_idx,
                &format!("{prefix}.self_attn"),
            )?,
            mlp: DrafterMlp::from_weights(weights, config, &format!("{prefix}.mlp"))?,
            input_layernorm: RMSNorm::new(
                weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            ),
            post_attention_layernorm: RMSNorm::new(
                weight_copy(
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                config.rms_norm_eps,
            ),
            pre_feedforward_layernorm: RMSNorm::new(
                weight_copy(
                    weights,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                config.rms_norm_eps,
            ),
            post_feedforward_layernorm: RMSNorm::new(
                weight_copy(
                    weights,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                config.rms_norm_eps,
            ),
            layer_scalar: weights
                .get(&format!("{prefix}.layer_scalar"))
                .map(|w| ffi::copy(w)),
            layer_type: config.layer_type(layer_idx).to_string(),
        })
    }

    /// Forward through the layer. Mirrors the gemma4 layer flow:
    ///
    /// `x → input_layernorm → self_attn(shared_kv, offset)
    ///   → post_attention_layernorm → +residual
    ///   → pre_feedforward_layernorm → mlp
    ///   → post_feedforward_layernorm → +residual`
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        shared_keys: &MlxArray,
        shared_values: &MlxArray,
        offset: RopeOffset<'_>,
    ) -> UniquePtr<MlxArray> {
        let h_attn = self.input_layernorm.forward(x);
        let h_attn = self
            .self_attn
            .forward(&h_attn, mask, shared_keys, shared_values, offset);
        let h_attn = self.post_attention_layernorm.forward(&h_attn);
        let after_attn = ffi::add(x, &h_attn);

        let ffn_in = self.pre_feedforward_layernorm.forward(&after_attn);
        let ffn_out = self.mlp.forward(&ffn_in);
        let ffn_out = self.post_feedforward_layernorm.forward(&ffn_out);
        let h = ffi::add(&after_attn, &ffn_out);
        if let Some(layer_scalar) = &self.layer_scalar {
            ffi::multiply(&h, layer_scalar)
        } else {
            h
        }
    }

    pub(crate) fn layer_type(&self) -> &str {
        &self.layer_type
    }
}

/// Copy a tensor by name out of the weight map (or return a descriptive error
/// if missing). Local helper to keep the per-field error messages identical to
/// the upstream gemma4 loader.
fn weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| ffi::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}
