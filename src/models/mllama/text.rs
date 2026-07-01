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

//! Llama 3.2 Vision text backbone with interleaved gated cross-attention.
//!
//! Faithful port of
//! `references/mlx-vlm/mlx_vlm/models/mllama/language.py`.
//!
//! The backbone is a standard Llama-3 decoder in which the layers listed in
//! `cross_attention_layers` are replaced by [`MllamaCrossAttentionDecoderLayer`]
//! adapters that attend to the vision tower's features. Self-attention layers
//! are the ordinary Llama-3 block, so they reuse
//! [`crate::models::llama3::TransformerBlock`] verbatim (fused QKV, plain RoPE
//! with `base = rope_theta`). The cross-attention adapters add per-head
//! `q_norm`/`k_norm` (RMSNorm over `head_dim`) and two learned `tanh` gates on
//! the attention and MLP residual branches.

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear, attention_from_ptr};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MllamaTextConfig;
use crate::models::llama3::{MLP, ModelArgs, TransformerBlock};

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    Ok(RMSNorm::new(
        get_weight_copy(weights, &format!("{prefix}.weight"))?,
        eps,
    ))
}

/// Gated cross-attention: queries come from the text stream, keys/values from
/// the projected vision features (`cross_attention_states`). Mirrors
/// `MllamaTextCrossAttention` in the reference.
pub struct MllamaTextCrossAttention {
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
}

impl MllamaTextCrossAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaTextConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
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
            q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), config.rms_norm_eps)?,
            k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), config.rms_norm_eps)?,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `hidden_states`: `[B, q_len, hidden]`.
    /// `cross_states`: `[B, kv_len, hidden]` projected vision features.
    /// `mask`: optional `[B, 1, q_len, kv_len]` additive cross-attention mask.
    fn forward(
        &self,
        hidden_states: &MlxArray,
        cross_states: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let b = shape[0];
        let q_len = shape[1];

        // Query from the text stream, per-head RMSNorm over head_dim.
        let query = self.q_proj.forward(hidden_states);
        let query = mlxcel_core::reshape(&query, &[b, q_len, self.num_heads, self.head_dim]);
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let query = self.q_norm.forward(&query);

        // Key/Value from the vision features.
        let kv_len = mlxcel_core::array_shape(cross_states)[1];
        let key = self.k_proj.forward(cross_states);
        let key = mlxcel_core::reshape(&key, &[b, kv_len, self.num_kv_heads, self.head_dim]);
        let key = mlxcel_core::transpose_axes(&key, &[0, 2, 1, 3]);
        let key = self.k_norm.forward(&key);

        let value = self.v_proj.forward(cross_states);
        let value = mlxcel_core::reshape(&value, &[b, kv_len, self.num_kv_heads, self.head_dim]);
        let value = mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3]);

        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        // GQA is handled inside the shared attention kernel (num_heads vs
        // num_kv_heads). Cross-attention is never causal.
        let attn =
            unsafe { attention_from_ptr(&query, &key, &value, self.scale, mask_ptr, 0.0, 0) };

        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, q_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn)
    }
}

/// A gated cross-attention decoder layer (`MllamaCrossAttentionDecoderLayer`).
pub struct MllamaCrossAttentionDecoderLayer {
    input_layernorm: RMSNorm,
    cross_attn: MllamaTextCrossAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
    attn_gate: UniquePtr<MlxArray>,
    mlp_gate: UniquePtr<MlxArray>,
}

impl MllamaCrossAttentionDecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &MllamaTextConfig,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let group_size = args.group_size();
        let bits = args.bits();
        Ok(Self {
            input_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.rms_norm_eps,
            )?,
            cross_attn: MllamaTextCrossAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.cross_attn"),
                group_size,
                bits,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                config.rms_norm_eps,
            )?,
            mlp: MLP::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            attn_gate: get_weight_copy(weights, &format!("{prefix}.cross_attn_attn_gate"))?,
            mlp_gate: get_weight_copy(weights, &format!("{prefix}.cross_attn_mlp_gate"))?,
        })
    }

    /// Forward with vision cross-attention state.
    ///
    /// When `cross_states` is `None` (a text-only request with no image), the
    /// layer is a pass-through: with no image features to attend to there is
    /// nothing for cross-attention to contribute. This matches HuggingFace
    /// `MllamaForConditionalGeneration`, which skips the cross-attention block
    /// when `cross_attention_states is None` and the cache is empty.
    fn forward(
        &self,
        hidden_states: &MlxArray,
        cross_states: Option<&MlxArray>,
        cross_mask: Option<&MlxArray>,
        full_text_row_masked_out_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let Some(cross_states) = cross_states else {
            return mlxcel_core::copy(hidden_states);
        };

        // Gated cross-attention branch: h = h + tanh(attn_gate) * attn(norm(h)).
        let normed = self.input_layernorm.forward(hidden_states);
        let attn = self.cross_attn.forward(&normed, cross_states, cross_mask);
        let attn_gate = mlxcel_core::tanh(&self.attn_gate);
        let gated_attn = mlxcel_core::multiply(&attn_gate, &attn);
        let hidden_states = mlxcel_core::add(hidden_states, &gated_attn);

        // Gated MLP branch, optionally zeroing rows with no visible image.
        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let mut mlp_out = self.mlp.forward(&normed);
        if let Some(row_mask) = full_text_row_masked_out_mask {
            // row_mask: [B, 1, q_len, 1] -> [B, q_len, 1], broadcasts over hidden.
            let row_mask = mlxcel_core::squeeze_axis(row_mask, 1);
            mlp_out = mlxcel_core::multiply(&row_mask, &mlp_out);
        }
        let mlp_gate = mlxcel_core::tanh(&self.mlp_gate);
        let gated_mlp = mlxcel_core::multiply(&mlp_gate, &mlp_out);
        mlxcel_core::add(&hidden_states, &gated_mlp)
    }
}

/// One decoder layer: either a standard Llama-3 self-attention block or a gated
/// cross-attention adapter.
enum TextLayer {
    SelfAttn(Box<TransformerBlock>),
    Cross(Box<MllamaCrossAttentionDecoderLayer>),
}

/// The interleaved self/cross-attention text model (`MllamaTextModel` +
/// `LanguageModel` head from the reference, fused into one struct).
pub struct MllamaTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TextLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    num_layers: usize,
}

impl MllamaTextModel {
    pub fn from_weights(weights: &WeightMap, config: &MllamaTextConfig) -> Result<Self, String> {
        let args = config.to_llama3_args();
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            if config.is_cross_attention_layer(idx) {
                layers.push(TextLayer::Cross(Box::new(
                    MllamaCrossAttentionDecoderLayer::from_weights(weights, config, &args, idx)?,
                )));
            } else {
                layers.push(TextLayer::SelfAttn(Box::new(
                    TransformerBlock::from_weights(weights, &args, idx)?,
                )));
            }
        }

        let norm = load_rms_norm(weights, "model.norm", config.rms_norm_eps)?;
        let lm_head = if config.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            num_layers: config.num_hidden_layers,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    pub fn embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Full forward with optional vision cross-attention state.
    ///
    /// - `input_embeds` overrides `input_ids` when present (VLM inject path).
    /// - Self-attention layers consume `caches[i]` and `mask`.
    /// - Cross-attention layers consume `cross_states` / `cross_mask` /
    ///   `full_text_row_masked_out_mask`; their `caches[i]` slot is left unused.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: Option<&MlxArray>,
        input_embeds: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        cross_states: Option<&MlxArray>,
        cross_mask: Option<&MlxArray>,
        full_text_row_masked_out_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match (input_embeds, input_ids) {
            (Some(embeds), _) => mlxcel_core::copy(embeds),
            (None, Some(ids)) => self.embed_tokens.forward(ids),
            (None, None) => panic!("MllamaTextModel::forward requires input_ids or input_embeds"),
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = match layer {
                TextLayer::SelfAttn(block) => block.forward(&h, &mut caches[i], mask),
                TextLayer::Cross(layer) => {
                    layer.forward(&h, cross_states, cross_mask, full_text_row_masked_out_mask)
                }
            };
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }
}
