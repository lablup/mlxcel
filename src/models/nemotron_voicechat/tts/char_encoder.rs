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

//! Bidirectional T5Gemma character encoder used inside the char-aware
//! subword embedding.
//!
//! Ports `T5GemmaSelfAttention`, `T5GemmaEncoderLayer` and `T5GemmaEncoder`
//! from `mlx_vlm/models/nemotron_voicechat/tts.py`. Attention is computed
//! explicitly (not through SDPA) because the reference applies a `tanh`
//! logit soft-cap and a key-padding mask before a float32 softmax; every
//! scalar is applied in the activation dtype the way a Python float literal
//! is.

use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::CharEncoderConfig;
use super::norm_mlp::{OffsetRmsNorm, load_mlp, mlp_args, scalar};
use crate::models::gemma3::MLP;

struct SelfAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f64,
    softcap: f64,
    rope_base: f32,
}

impl SelfAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &CharEncoderConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        if cfg.num_key_value_heads == 0 || cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
            return Err(format!(
                "{prefix}: num_attention_heads {} is not a multiple of num_key_value_heads {}",
                cfg.num_attention_heads, cfg.num_key_value_heads
            ));
        }
        Ok(Self {
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            o_proj: lin("o_proj")?,
            n_heads: cfg.num_attention_heads as i32,
            n_kv_heads: cfg.num_key_value_heads as i32,
            head_dim: cfg.head_dim as i32,
            scale: cfg.query_pre_attn_scalar.powf(-0.5),
            softcap: cfg.attn_logit_softcapping,
            rope_base: cfg.rope_base,
        })
    }

    /// `x`: `[N, L, H]`; `key_mask`: bool `[N, 1, 1, L]`.
    fn forward(&self, x: &MlxArray, key_mask: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let heads = |t: UniquePtr<MlxArray>, n: i32| {
            let t = mlxcel_core::reshape(&t, &[b, l, n, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = heads(self.q_proj.forward(x), self.n_heads);
        let k = heads(self.k_proj.forward(x), self.n_kv_heads);
        let mut v = heads(self.v_proj.forward(x), self.n_kv_heads);
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, 0);
        let mut k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, 0);
        let repeats = self.n_heads / self.n_kv_heads;
        if repeats > 1 {
            k = mlxcel_core::repeat(&k, repeats, 1);
            v = mlxcel_core::repeat(&v, repeats, 1);
        }

        let act = mlxcel_core::array_dtype(&q);
        let kt = mlxcel_core::transpose_axes(&k, &[0, 1, 3, 2]);
        let scores = mlxcel_core::multiply(&mlxcel_core::matmul(&q, &kt), &scalar(self.scale, act));
        let capped = mlxcel_core::tanh(&mlxcel_core::divide(&scores, &scalar(self.softcap, act)));
        let scores = mlxcel_core::multiply(&capped, &scalar(self.softcap, act));
        let scores = mlxcel_core::where_cond(key_mask, &scores, &scalar(-1e30, act));
        let probs = mlxcel_core::softmax(&mlxcel_core::astype(&scores, dtype::FLOAT32), -1);
        let probs = mlxcel_core::astype(&probs, act);
        let out = mlxcel_core::matmul(&probs, &v);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

struct EncoderLayer {
    self_attn: SelfAttention,
    pre_self_attn_layernorm: OffsetRmsNorm,
    post_self_attn_layernorm: OffsetRmsNorm,
    mlp: MLP,
    pre_feedforward_layernorm: OffsetRmsNorm,
    post_feedforward_layernorm: OffsetRmsNorm,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &CharEncoderConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm = |name: &str| {
            OffsetRmsNorm::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )
        };
        Ok(Self {
            self_attn: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                cfg,
                group_size,
                bits,
            )?,
            pre_self_attn_layernorm: norm("pre_self_attn_layernorm")?,
            post_self_attn_layernorm: norm("post_self_attn_layernorm")?,
            mlp: load_mlp(
                weights,
                &format!("{prefix}.mlp"),
                &mlp_args(group_size, bits),
            )?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
        })
    }

    fn forward(&self, x: &MlxArray, key_mask: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward(&self.pre_self_attn_layernorm.forward(x), key_mask);
        let x = mlxcel_core::add(x, &self.post_self_attn_layernorm.forward(&attn));
        let ff = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&x));
        mlxcel_core::add(&x, &self.post_feedforward_layernorm.forward(&ff))
    }
}

/// `T5GemmaEncoder`: `sqrt(hidden)`-scaled input, encoder layers, final norm.
pub struct CharEncoder {
    layers: Vec<EncoderLayer>,
    norm: OffsetRmsNorm,
    scale: f64,
}

impl CharEncoder {
    /// Load `{prefix}.layers.{i}.*` and `{prefix}.norm.weight` (the reference
    /// key tree is `embed_subword.backbone.encoder`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &CharEncoderConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| {
                EncoderLayer::from_weights(
                    weights,
                    &format!("{prefix}.layers.{idx}"),
                    cfg,
                    group_size,
                    bits,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            norm: OffsetRmsNorm::from_weights(
                weights,
                &format!("{prefix}.norm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            scale: (cfg.hidden_size as f64).sqrt(),
        })
    }

    /// `inputs_embeds`: `[N, L, H]`; `mask`: bool `[N, L]` (true = real char).
    pub fn forward(&self, inputs_embeds: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let act = mlxcel_core::array_dtype(inputs_embeds);
        let mut x = mlxcel_core::multiply(inputs_embeds, &scalar(self.scale, act));
        let mshape = mlxcel_core::array_shape(mask);
        let key_mask = mlxcel_core::reshape(mask, &[mshape[0], 1, 1, mshape[1]]);
        for layer in &self.layers {
            x = layer.forward(&x, &key_mask);
        }
        self.norm.forward(&x)
    }
}
