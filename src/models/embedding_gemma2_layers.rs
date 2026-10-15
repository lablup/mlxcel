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

//! Transformer block components for EmbeddingGemma 2: bidirectional local/global
//! self-attention (`q_norm`, `k_norm`, scale-free `v_norm`, SDPA `scale = 1.0`),
//! GeGLU MLP, per-layer PLE gating block (`ple_block`), and `layer_scalar`.

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use super::{EmbeddingGemma2LayerConfig, quant_params_for_prefix};

pub(super) fn rms_norm_from_weights(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<RMSNorm, String> {
    let key = format!("{prefix}.weight");
    let w = weights
        .get(&key)
        .map(|arr| mlxcel_core::copy(arr))
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(w, eps))
}

/// Bidirectional self-attention layer with Q/K `RMSNorm`, scale-free V `RMSNormNoScale`,
/// per-layer `head_dim`/`num_key_value_heads`/`rope_theta`, and SDPA `scale = 1.0`.
pub struct EmbeddingGemma2Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub is_sliding: bool,
}

impl EmbeddingGemma2Attention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        layer_cfg: &EmbeddingGemma2LayerConfig,
        rms_norm_eps: f32,
        raw_config: &Value,
    ) -> Result<Self, String> {
        let (gs, bits) = quant_params_for_prefix(raw_config, prefix);
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?,
            k_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), gs, bits)?,
            o_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?,
            q_norm: rms_norm_from_weights(weights, &format!("{prefix}.q_norm"), rms_norm_eps)?,
            k_norm: rms_norm_from_weights(weights, &format!("{prefix}.k_norm"), rms_norm_eps)?,
            n_heads: layer_cfg.num_attention_heads,
            n_kv_heads: layer_cfg.num_key_value_heads,
            head_dim: layer_cfg.head_dim,
            rope_theta: layer_cfg.rope_theta,
            rms_norm_eps,
            is_sliding: layer_cfg.is_sliding,
        })
    }

    pub fn forward(&self, x: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_theta, 1.0, 0);

        let k = self.k_proj.forward(x);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let k = self.k_norm.forward(&k);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_theta, 1.0, 0);

        let v = self.v_proj.forward(x);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::fast_rms_norm_no_weight(&v, self.rms_norm_eps);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // EmbeddingGemma 2 / Gemma 4 applies Q and K RMSNorm before SDPA and sets scale = 1.0.
        // The bidirectional window or padding mask is supplied via `mask`, so `window_size = 0`.
        let attn = mlxcel_core::layers::attention(&q, &k, &v, 1.0, Some(mask), 0.0, 0);
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }
}

/// GeGLU feed-forward block (`gelu_approx(gate_proj(x)) * up_proj(x)` -> `down_proj`).
pub struct EmbeddingGemma2Mlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl EmbeddingGemma2Mlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        raw_config: &Value,
    ) -> Result<Self, String> {
        let (gs, bits) = quant_params_for_prefix(raw_config, prefix);
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), gs, bits)?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                gs,
                bits,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let act = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        self.down_proj.forward(&act)
    }
}

/// Per-layer PLE block (`layers.{i}.ple_block`).
pub struct EmbeddingGemma2PleBlock {
    pub per_layer_input_gate: UnifiedLinear,
    pub per_layer_projection: UnifiedLinear,
    pub post_per_layer_input_norm: RMSNorm,
}

impl EmbeddingGemma2PleBlock {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        rms_norm_eps: f32,
        raw_config: &Value,
    ) -> Result<Option<Self>, String> {
        let gate_key = format!("{prefix}.per_layer_input_gate.weight");
        if !weights.contains_key(&gate_key) {
            return Ok(None);
        }
        let (gs, bits) = quant_params_for_prefix(raw_config, prefix);
        Ok(Some(Self {
            per_layer_input_gate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.per_layer_input_gate"),
                gs,
                bits,
            )?,
            per_layer_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.per_layer_projection"),
                gs,
                bits,
            )?,
            post_per_layer_input_norm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.post_per_layer_input_norm"),
                rms_norm_eps,
            )?,
        }))
    }

    pub fn forward(&self, x: &MlxArray, ple_i: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.per_layer_input_gate.forward(x);
        let gated = mlxcel_core::compiled_geglu_approx_activation(&gate, ple_i);
        let proj = self.per_layer_projection.forward(&gated);
        let normed = self.post_per_layer_input_norm.forward(&proj);
        mlxcel_core::add(x, &normed)
    }
}

/// Single EmbeddingGemma 2 encoder layer.
pub struct EmbeddingGemma2Layer {
    pub input_layernorm: RMSNorm,
    pub self_attn: EmbeddingGemma2Attention,
    pub post_attention_layernorm: RMSNorm,
    pub pre_feedforward_layernorm: RMSNorm,
    pub mlp: EmbeddingGemma2Mlp,
    pub post_feedforward_layernorm: RMSNorm,
    pub ple_block: Option<EmbeddingGemma2PleBlock>,
    pub layer_scalar: Option<UniquePtr<MlxArray>>,
}

impl EmbeddingGemma2Layer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        layer_cfg: &EmbeddingGemma2LayerConfig,
        rms_norm_eps: f32,
        raw_config: &Value,
    ) -> Result<Self, String> {
        let layer_scalar = weights
            .get(&format!("{prefix}.layer_scalar"))
            .map(|w| mlxcel_core::copy(w));
        Ok(Self {
            input_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.input_layernorm"),
                rms_norm_eps,
            )?,
            self_attn: EmbeddingGemma2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                layer_cfg,
                rms_norm_eps,
                raw_config,
            )?,
            post_attention_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                rms_norm_eps,
            )?,
            pre_feedforward_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.pre_feedforward_layernorm"),
                rms_norm_eps,
            )?,
            mlp: EmbeddingGemma2Mlp::from_weights(weights, &format!("{prefix}.mlp"), raw_config)?,
            post_feedforward_layernorm: rms_norm_from_weights(
                weights,
                &format!("{prefix}.post_feedforward_layernorm"),
                rms_norm_eps,
            )?,
            ple_block: EmbeddingGemma2PleBlock::from_weights(
                weights,
                &format!("{prefix}.ple_block"),
                rms_norm_eps,
                raw_config,
            )?,
            layer_scalar,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        mask: &MlxArray,
        ple_i: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn = self.self_attn.forward(&normed, mask);
        let attn_normed = self.post_attention_layernorm.forward(&attn);
        let after_attn = mlxcel_core::add(x, &attn_normed);

        let ffn_in = self.pre_feedforward_layernorm.forward(&after_attn);
        let ffn_out = self.mlp.forward(&ffn_in);
        let ffn_normed = self.post_feedforward_layernorm.forward(&ffn_out);
        let mut h = mlxcel_core::add(&after_attn, &ffn_normed);

        if let (Some(ple_block), Some(ple_slice)) = (&self.ple_block, ple_i) {
            h = ple_block.forward(&h, ple_slice);
        }
        if let Some(scalar) = &self.layer_scalar {
            let scalar_cast = mlxcel_core::astype(scalar, mlxcel_core::array_dtype(&h));
            h = mlxcel_core::multiply(&h, &scalar_cast);
        }
        h
    }
}
