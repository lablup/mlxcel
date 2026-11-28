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

//! Talkie (`talkie`, `TalkieForCausalLM`) text model implementation.
//!
//! Reference: `mlx_lm/models/talkie.py` (ml-explore/mlx-lm#1492) and the
//! transformers port `lewtun/talkie-1930-13b-it-hf`.
//!
//! A Llama-shaped multi-head decoder with a SwiGLU MLP and no biases, plus
//! choices that each change the logits:
//!
//! - Every RMSNorm is weightless (`eps` = the fp32 machine epsilon): the
//!   embedding norm, the pre-attention and pre-MLP norms, the q/k norms and
//!   the final norm.
//! - RoPE rotates with the opposite sign to NeoX over the full `head_dim`:
//!   standard non-traditional RoPE with a negated frequency table.
//! - Attention: RoPE on q and k, then the weightless q/k norm, then a learned
//!   per-head gain on q only, then SDPA with scale `head_dim ** -0.5`.
//! - Learned scalar gains on the attention and MLP residual branches, and an
//!   embedding skip connection: every block adds `embed_skip * rms(embed(x))`.
//! - `logits = lm_head_gain * (rms(h) @ lm_head.T)`.
//!
//! Gains stay at checkpoint dtype and are cast to the activation dtype at use,
//! as upstream does with `astype(x.dtype)`.

use mlxcel_core::batched_decode::forward_split_attention;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode};
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear, compiled_swiglu_mlp};
use mlxcel_core::utils::{pipeline_hint, silu};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

#[path = "talkie_attention.rs"]
mod attention;
#[path = "talkie_config.rs"]
mod config;
pub use attention::Attention;
pub use config::{EosTokenId, ModelArgs, Quantization, TALKIE_EOS_TOKEN_IDS};

/// Weightless RMSNorm: `x * rsqrt(mean(x^2, -1) + eps)`.
#[inline]
fn norm(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::fast_rms_norm_no_weight(x, eps)
}

/// `x * gain`, with the gain cast to `x`'s dtype at use.
#[inline]
fn apply_gain(x: &MlxArray, gain: &MlxArray) -> UniquePtr<MlxArray> {
    let gain = mlxcel_core::astype(gain, mlxcel_core::array_dtype(x));
    mlxcel_core::multiply(x, &gain)
}

/// A required gain tensor, copied at checkpoint dtype.
fn gain(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("talkie: missing gain tensor {key}"))
}

// SwiGLU MLP: `mlp_resid(silu(mlp_gate(x)) * mlp_linear(x))`.

pub struct MLP {
    pub mlp_gate: UnifiedLinear,
    pub mlp_linear: UnifiedLinear,
    pub mlp_resid: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if let Some(out) = compiled_swiglu_mlp(x, &self.mlp_gate, &self.mlp_linear, &self.mlp_resid)
        {
            return out;
        }
        let gated = mlxcel_core::multiply(
            &silu(&self.mlp_gate.forward(x)),
            &self.mlp_linear.forward(x),
        );
        self.mlp_resid.forward(&gated)
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
        Ok(Self {
            mlp_gate: lin("mlp_gate")?,
            mlp_linear: lin("mlp_linear")?,
            mlp_resid: lin("mlp_resid")?,
        })
    }
}

// Transformer block.

pub struct TransformerBlock {
    pub attn: Attention,
    pub mlp: MLP,
    pub attn_gain: UniquePtr<MlxArray>,
    pub mlp_gain: UniquePtr<MlxArray>,
    pub embed_skip: UniquePtr<MlxArray>,
    pub eps: f32,
}

impl TransformerBlock {
    /// The MLP half and the embedding skip, shared by both forwards.
    fn finish(&self, h: &MlxArray, e_x: &MlxArray) -> UniquePtr<MlxArray> {
        let m = self.mlp.forward(&norm(h, self.eps));
        let h = mlxcel_core::add(h, &apply_gain(&m, &self.mlp_gain));
        mlxcel_core::add(&h, &apply_gain(e_x, &self.embed_skip))
    }

    /// `e_x` is the normalized token embedding, shared by every block.
    pub fn forward(
        &self,
        x: &MlxArray,
        e_x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let a = self.attn.forward(&norm(x, self.eps), cache, mask);
        let h = mlxcel_core::add(x, &apply_gain(&a, &self.attn_gain));
        self.finish(&h, e_x)
    }

    pub fn forward_batched(
        &self,
        x: &MlxArray,
        e_x: &MlxArray,
        caches: &mut [&mut KVCache],
        layer_idx: usize,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = norm(x, self.eps);
        let q = self.attn.attn_query.forward(&normed);
        let k = self.attn.attn_key.forward(&normed);
        let v = self.attn.attn_value.forward(&normed);
        let seq_len = mlxcel_core::array_shape(&q)[1];
        let metadata = crate::models::batched_decode_cache::uniform_layer_metadata(
            caches, seq_len, 0, "Talkie", layer_idx,
        );
        let a = self.attn.attn_resid.forward(&forward_split_attention(
            &self.attn, &q, &k, &v, caches, &metadata, mask,
        ));
        let h = mlxcel_core::add(x, &apply_gain(&a, &self.attn_gain));
        self.finish(&h, e_x)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.blocks.{layer_idx}");
        Ok(Self {
            attn: Attention::from_weights(weights, args, &format!("{prefix}.attn"))?,
            mlp: MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            attn_gain: gain(weights, &format!("{prefix}.attn_gain.a_g"))?,
            mlp_gain: gain(weights, &format!("{prefix}.mlp_gain.a_g"))?,
            embed_skip: gain(weights, &format!("{prefix}.embed_skip.a_g"))?,
            eps: args.rms_norm_eps,
        })
    }
}

// Model.

pub struct TalkieModel {
    pub embed: UnifiedEmbedding,
    pub blocks: Vec<TransformerBlock>,
    pub lm_head: UnifiedLinear,
    /// `lm_head_gain.w_g`; `None` when the checkpoint omits it (gain 1.0).
    pub lm_head_gain: Option<UniquePtr<MlxArray>>,
    pub eps: f32,
    eos_token_ids: Vec<i32>,
}

impl TalkieModel {
    /// Run the blocks and the final norm. `row` slices one position before
    /// the final norm so a prefill projects a single row through the head.
    fn hidden_states(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        // The block stream starts from the normalized embedding, and every
        // block's skip adds that same normalized tensor.
        let e_x = norm(&self.embed.forward(input_ids), self.eps);
        let mut h = mlxcel_core::copy(&e_x);
        let n = self.blocks.len();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &e_x, &mut caches[i], mask);
            pipeline_hint(&h, i, n);
        }
        let h = mlxcel_core::generate::select_row(h, row);
        norm(&h, self.eps)
    }

    pub(crate) fn logits_from_hidden(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let logits = self.lm_head.forward(h);
        match &self.lm_head_gain {
            Some(g) => apply_gain(&logits, g),
            None => logits,
        }
    }

    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, caches, mask, None);
        self.logits_from_hidden(&h)
    }

    pub fn forward_batched_impl(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let e_x = norm(&self.embed.forward(input_ids), self.eps);
        let mut h = mlxcel_core::copy(&e_x);
        let n = self.blocks.len();
        for layer_idx in 0..n {
            let mut caches = crate::models::batched_decode_cache::collect_layer_caches(
                batch_caches,
                "Talkie",
                layer_idx,
            );
            h = self.blocks[layer_idx].forward_batched(&h, &e_x, &mut caches, layer_idx, mask);
            pipeline_hint(&h, layer_idx, n);
        }
        self.logits_from_hidden(&norm(&h, self.eps))
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.blocks.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.validate()?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        let sanitized;
        let weights = if needs_sanitize(weights) {
            sanitized = sanitize_weights(weights);
            &sanitized
        } else {
            weights
        };

        let (gs, bits) = (args.group_size(), args.bits());
        let embed = UnifiedEmbedding::from_weights(weights, "model.embed", gs, bits)?;
        let blocks = (0..args.num_hidden_layers)
            .map(|i| TransformerBlock::from_weights(weights, args, i))
            .collect::<Result<Vec<_>, _>>()?;
        // Both head layouts reach this point as `lm_head.weight` (plus
        // `scales` / `biases` when quantized), so one builder serves both.
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?;
        let lm_head_gain = weights.get(LM_HEAD_GAIN).map(|g| mlxcel_core::copy(g));
        if lm_head_gain.is_none() {
            tracing::debug!(
                "talkie: checkpoint has no {LM_HEAD_GAIN}; using an output gain of 1.0"
            );
        }

        Ok(Self {
            embed,
            blocks,
            lm_head,
            lm_head_gain,
            eps: args.rms_norm_eps,
            eos_token_ids: args.eos_token_ids(),
        })
    }
}

// Weight sanitation.

const LM_HEAD_GAIN: &str = "lm_head_gain.w_g";

/// Upstream drops precomputed rotary buffers.
fn is_rotary_buffer(key: &str) -> bool {
    key.contains("rotary") || key.ends_with("cos") || key.ends_with("sin")
}

fn needs_sanitize(weights: &WeightMap) -> bool {
    weights
        .keys()
        .any(|k| k == "lm_head" || is_rotary_buffer(k))
}

/// Drop rotary buffers and rename the upstream bare `lm_head` tensor to
/// `lm_head.weight`. Idempotent: an already sanitized map passes through.
pub fn sanitize_weights(weights: &WeightMap) -> WeightMap {
    weights
        .iter()
        .filter(|(k, _)| !is_rotary_buffer(k))
        .map(|(k, v)| {
            let key = if k == "lm_head" {
                "lm_head.weight".to_string()
            } else {
                k.clone()
            };
            (key, mlxcel_core::copy(v))
        })
        .collect()
}

// LanguageModel trait implementation.

impl LanguageModel for TalkieModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        TalkieModel::forward(self, input_ids, caches, mask)
    }

    /// The final norm, the head and its gain act per position, so slicing the
    /// hidden row first equals slicing the full logits.
    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // No embeddings path: decline so the default keeps its behavior.
        if input_embeddings.is_some() {
            return None;
        }
        let h = self.hidden_states(input_ids, caches, mask, Some(last_pos));
        Some(self.logits_from_hidden(&h))
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }
    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode(self, input_ids, batch_caches, |caches| {
            self.forward_batched_impl(input_ids, caches, mask)
        })
    }

    fn make_caches(&self) -> Vec<KVCache> {
        TalkieModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "talkie_tests.rs"]
mod tests;
