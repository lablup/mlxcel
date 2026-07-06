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

//! Qwen3-Omni talker: autoregressive MoE codec-token decoder (stage 2).
//!
//! The talker is a 20-layer Qwen3-MoE-style decoder (128 experts, top-6,
//! shared expert with sigmoid gate, QK-RMSNorm GQA attention) over a 3072-way
//! codec vocabulary. It is conditioned on the thinker through two SwiGLU
//! resize MLPs (`text_projection` for token embeddings, `hidden_projection`
//! for multimodal hidden states). Each generated frame's first codebook comes
//! from `codec_head`; a 5-layer dense code predictor then emits the remaining
//! `num_code_groups - 1` residual codebooks autoregressively (fresh KV caches
//! per frame), and the sum of all `num_code_groups` code embeddings (plus the
//! next trailing projected text embedding, or the projected `tts_pad` once
//! text is exhausted) forms the next talker input embedding.
//!
//! Attention, RoPE handling, resize MLPs, and the sampling helper live in
//! the sibling [`super::speech_layers`] module. Sampling mirrors the
//! reference: temperature + nucleus (top-p) only; the first codebook uses the
//! configured talker temperature/top-p, the code predictor uses the same
//! temperature with top-p 0.8 fixed.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/talker.py>.
//!
//! Used by: Qwen3-Omni MoE speech pipeline (speech.rs).

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::qwen3_next::{MLP, Quantization, Qwen3NextConfig, SparseMoeBlock};

use super::speech_config::{CodePredictorConfig, TalkerConfig, TalkerTextConfig};
use super::speech_layers::{ResizeMlp, SpeechAttention, load_rms_norm, sample_logits};

/// Bridge the talker text config into the `Qwen3NextConfig` shape so the
/// shared `SparseMoeBlock` / `MLP` loaders (also reused by qwen3_5) apply.
/// Only the MoE and quantization fields are consumed by those loaders; the
/// linear-attention fields are inert placeholders.
fn moe_bridge_config(cfg: &TalkerTextConfig, gs: i32, bits: i32) -> Qwen3NextConfig {
    Qwen3NextConfig {
        model_type: "qwen3_omni_moe_talker".to_string(),
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        linear_num_value_heads: 0,
        linear_num_key_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        num_experts: cfg.num_experts,
        num_experts_per_tok: cfg.num_experts_per_tok,
        decoder_sparse_step: 1,
        moe_intermediate_size: cfg.moe_intermediate_size,
        shared_expert_intermediate_size: cfg.shared_expert_intermediate_size,
        mlp_only_layers: Vec::new(),
        full_attention_interval: 4,
        rms_norm_eps: cfg.rms_norm_eps,
        vocab_size: cfg.vocab_size,
        rope_theta: cfg.rope_theta,
        partial_rotary_factor: 1.0,
        max_position_embeddings: None,
        norm_topk_prob: cfg.norm_topk_prob,
        tie_word_embeddings: false,
        attention_bias: false,
        quantization: Some(Quantization {
            group_size: gs,
            bits,
        }),
    }
}

struct TalkerDecoderLayer {
    attn: SpeechAttention,
    mlp: SparseMoeBlock,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl TalkerDecoderLayer {
    fn forward(&self, x: &MlxArray, cache: &mut KVCache) -> UniquePtr<MlxArray> {
        let r = self.attn.forward(&self.input_layernorm.forward(x), cache);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

struct CodePredictorLayer {
    attn: SpeechAttention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl CodePredictorLayer {
    fn forward(&self, x: &MlxArray, cache: &mut KVCache) -> UniquePtr<MlxArray> {
        let r = self.attn.forward(&self.input_layernorm.forward(x), cache);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

/// Residual-codebook predictor: per generated frame, emits codebooks
/// `1..num_code_groups` given the talker hidden state and the first codebook.
struct CodePredictor {
    layers: Vec<CodePredictorLayer>,
    norm: RMSNorm,
    /// `num_code_groups - 1` embeddings, one per residual codebook.
    codec_embeddings: Vec<UnifiedEmbedding>,
    /// `num_code_groups - 1` heads; head `i` predicts codebook `i + 1`.
    lm_heads: Vec<UnifiedLinear>,
}

impl CodePredictor {
    fn from_weights(
        weights: &WeightMap,
        cfg: &CodePredictorConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
        moe_cfg: &Qwen3NextConfig,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}.model.layers.{i}");
            layers.push(CodePredictorLayer {
                attn: SpeechAttention::from_weights(
                    weights,
                    &format!("{lp}.self_attn"),
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    cfg.rope_theta,
                    gs,
                    bits,
                )?,
                mlp: MLP::from_weights(weights, moe_cfg, &format!("{lp}.mlp"))?,
                input_layernorm: load_rms_norm(
                    weights,
                    &format!("{lp}.input_layernorm"),
                    cfg.rms_norm_eps,
                )?,
                post_attention_layernorm: load_rms_norm(
                    weights,
                    &format!("{lp}.post_attention_layernorm"),
                    cfg.rms_norm_eps,
                )?,
            });
        }

        let n_residual = cfg.num_code_groups - 1;
        let mut codec_embeddings = Vec::with_capacity(n_residual);
        let mut lm_heads = Vec::with_capacity(n_residual);
        for i in 0..n_residual {
            codec_embeddings.push(UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.model.codec_embedding.{i}"),
                gs,
                bits,
            )?);
            lm_heads.push(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.lm_head.{i}"),
                gs,
                bits,
            )?);
        }

        Ok(Self {
            layers,
            norm: load_rms_norm(weights, &format!("{prefix}.model.norm"), cfg.rms_norm_eps)?,
            codec_embeddings,
            lm_heads,
        })
    }

    /// One decoder pass; returns `[1, vocab]` logits for `head_idx` at the
    /// last position.
    fn forward_embeds(
        &self,
        embeds: &MlxArray,
        caches: &mut [KVCache],
        head_idx: usize,
    ) -> UniquePtr<MlxArray> {
        let mut h = mlxcel_core::copy(embeds);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i]);
        }
        let h = self.norm.forward(&h);
        let shape = mlxcel_core::array_shape(&h);
        let last = mlxcel_core::slice(&h, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
        let logits = self.lm_heads[head_idx].forward(&last);
        mlxcel_core::reshape(&logits, &[1, -1])
    }
}

/// One generated codec frame: `num_code_groups` codes (first codebook from
/// the talker, the rest from the code predictor).
pub type CodecFrame = Vec<i32>;

pub struct Talker {
    layers: Vec<TalkerDecoderLayer>,
    norm: RMSNorm,
    codec_embedding: UnifiedEmbedding,
    pub text_projection: ResizeMlp,
    pub hidden_projection: ResizeMlp,
    code_predictor: CodePredictor,
    codec_head: UnifiedLinear,
    pub cfg: TalkerConfig,
}

impl Talker {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &TalkerConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let text_cfg = &cfg.text_config;
        let moe_cfg = moe_bridge_config(text_cfg, gs, bits);

        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for i in 0..text_cfg.num_hidden_layers {
            let lp = format!("{prefix}.model.layers.{i}");
            layers.push(TalkerDecoderLayer {
                attn: SpeechAttention::from_weights(
                    weights,
                    &format!("{lp}.self_attn"),
                    text_cfg.num_attention_heads,
                    text_cfg.num_key_value_heads,
                    text_cfg.head_dim,
                    text_cfg.rms_norm_eps,
                    text_cfg.rope_theta,
                    gs,
                    bits,
                )?,
                mlp: SparseMoeBlock::from_weights(weights, &moe_cfg, &format!("{lp}.mlp"))?,
                input_layernorm: load_rms_norm(
                    weights,
                    &format!("{lp}.input_layernorm"),
                    text_cfg.rms_norm_eps,
                )?,
                post_attention_layernorm: load_rms_norm(
                    weights,
                    &format!("{lp}.post_attention_layernorm"),
                    text_cfg.rms_norm_eps,
                )?,
            });
        }

        Ok(Self {
            layers,
            norm: load_rms_norm(
                weights,
                &format!("{prefix}.model.norm"),
                text_cfg.rms_norm_eps,
            )?,
            codec_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.model.codec_embedding"),
                gs,
                bits,
            )?,
            text_projection: ResizeMlp::from_weights(
                weights,
                &format!("{prefix}.text_projection"),
                gs,
                bits,
            )?,
            hidden_projection: ResizeMlp::from_weights(
                weights,
                &format!("{prefix}.hidden_projection"),
                gs,
                bits,
            )?,
            code_predictor: CodePredictor::from_weights(
                weights,
                &cfg.code_predictor_config,
                &format!("{prefix}.code_predictor"),
                gs,
                bits,
                &moe_cfg,
            )?,
            codec_head: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.codec_head"),
                gs,
                bits,
            )?,
            cfg: cfg.clone(),
        })
    }

    /// Embed talker codec token ids (`[1, n]` int) with the talker's own
    /// codec embedding table.
    pub fn embed_codec_ids(&self, ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.codec_embedding.forward(ids)
    }

    /// One talker decoder pass over `[1, L, hidden]` embeddings. Returns
    /// `(last_hidden [1, 1, hidden] post-norm, logits [1, vocab])`.
    fn forward_embeds(
        &self,
        embeds: &MlxArray,
        caches: &mut [KVCache],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut h = mlxcel_core::copy(embeds);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i]);
        }
        let h = self.norm.forward(&h);
        let shape = mlxcel_core::array_shape(&h);
        let last = mlxcel_core::slice(&h, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
        let logits = self.codec_head.forward(&last);
        let logits = mlxcel_core::reshape(&logits, &[1, -1]);
        (last, logits)
    }

    /// Predict the residual codebooks for one frame and build the next talker
    /// input embedding (sum of all `num_code_groups` code embeddings).
    /// Mirrors the reference `prepare_inputs_for_generation`; the code
    /// predictor samples with the talker temperature and a fixed top-p 0.8.
    fn predict_frame(
        &self,
        past_hidden: &MlxArray,
        token: i32,
        temperature: f32,
    ) -> (CodecFrame, UniquePtr<MlxArray>) {
        const CODE_PREDICTOR_TOP_P: f32 = 0.8;
        let n_groups = self.cfg.num_code_groups;

        let token_arr = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let last_id_hidden = self.codec_embedding.forward(&token_arr);

        // Fresh predictor caches every frame, per the reference.
        let mut cp_caches: Vec<KVCache> = (0..self.code_predictor.layers.len())
            .map(|_| KVCache::new())
            .collect();

        let cp_input = mlxcel_core::concatenate(past_hidden, &last_id_hidden, 1);
        let logits = self
            .code_predictor
            .forward_embeds(&cp_input, &mut cp_caches, 0);
        let mut cp_token = sample_logits(&logits, temperature, CODE_PREDICTOR_TOP_P);

        let mut codes: CodecFrame = vec![token, cp_token];
        let mut sum_embed = mlxcel_core::copy(&last_id_hidden);

        for step in 1..(n_groups - 1) {
            let id_arr = mlxcel_core::from_slice_i32(&[cp_token], &[1, 1]);
            let embed = self.code_predictor.codec_embeddings[step - 1].forward(&id_arr);
            sum_embed = mlxcel_core::add(&sum_embed, &embed);

            let logits = self
                .code_predictor
                .forward_embeds(&embed, &mut cp_caches, step);
            cp_token = sample_logits(&logits, temperature, CODE_PREDICTOR_TOP_P);
            codes.push(cp_token);
        }

        let id_arr = mlxcel_core::from_slice_i32(&[cp_token], &[1, 1]);
        let last_residual = self.code_predictor.codec_embeddings[n_groups - 2].forward(&id_arr);
        sum_embed = mlxcel_core::add(&sum_embed, &last_residual);

        (codes, sum_embed)
    }

    /// Autoregressive codec-frame generation. `inputs_embeds` is the
    /// conditioning prefix `[1, S, hidden]`; `trailing_text_hidden`
    /// (`[1, T, hidden]`) supplies the projected answer-token embedding added
    /// to each step's input until exhausted, after which `tts_pad_embed`
    /// (`[1, 1, hidden]`) takes over. Stops at `codec_eos_token_id` or
    /// `max_frames`.
    pub fn generate_codes(
        &self,
        inputs_embeds: &MlxArray,
        trailing_text_hidden: &MlxArray,
        tts_pad_embed: &MlxArray,
        max_frames: usize,
        temperature: f32,
        top_p: f32,
    ) -> Vec<CodecFrame> {
        let mut caches: Vec<KVCache> = (0..self.layers.len()).map(|_| KVCache::new()).collect();
        let trailing_len = mlxcel_core::array_shape(trailing_text_hidden)[1];

        let (mut hidden, logits) = self.forward_embeds(inputs_embeds, &mut caches);
        let mut token = sample_logits(&logits, temperature, top_p);

        let mut frames: Vec<CodecFrame> = Vec::new();
        for step in 0..max_frames {
            if token == self.cfg.codec_eos_token_id {
                break;
            }

            let (codes, sum_embed) = self.predict_frame(&hidden, token, temperature);
            frames.push(codes);

            let trailing = if (step as i32) < trailing_len {
                let hs = mlxcel_core::array_shape(trailing_text_hidden)[2];
                mlxcel_core::slice(
                    trailing_text_hidden,
                    &[0, step as i32, 0],
                    &[1, step as i32 + 1, hs],
                )
            } else {
                mlxcel_core::copy(tts_pad_embed)
            };
            let next_embed = mlxcel_core::add(&sum_embed, &trailing);

            let (h, logits) = self.forward_embeds(&next_embed, &mut caches);
            hidden = h;
            mlxcel_core::eval(&hidden);
            token = sample_logits(&logits, temperature, top_p);
        }

        frames
    }
}
