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

//! Qwen3-Omni code2wav codec vocoder (stage 2).
//!
//! Converts talker codec frames (16 codebooks x 2048 codes, 12.5 Hz) into a
//! 24 kHz waveform: a shared code embedding (summed-codebook mean) feeds an
//! 8-layer causal pre-transformer, two ConvNeXt upsample stages (2x each),
//! then a BigVGAN-style decoder (initial conv, four transposed-conv blocks
//! with SnakeBeta-activated residual units at rates 8/5/4/3, SnakeBeta, and a
//! final conv to one channel), for a total upsampling of 1920 samples per
//! frame. Decoding is chunked (300 frames per chunk with 25 frames of left
//! context) exactly like the reference `chunked_decode`. The convolutional
//! blocks live in [`super::code2wav_blocks`].
//!
//! All tensors here use the MLX channels-last conv layout `[B, L, C]`; the
//! mlx-community checkpoint ships conv weights already in the MLX
//! `(out, kernel, in)` layout, so no transposes happen at load.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/code2wav.py>.
//!
//! Used by: Qwen3-Omni MoE speech pipeline (speech.rs).

use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::code2wav_blocks::{
    CausalConv1d, CausalTransConv1d, ConvNeXtBlock, DecoderBlock, SnakeBeta, get_weight, to_vec_f32,
};
use super::speech_config::Code2WavConfig;
use super::talker::CodecFrame;

/// Pre-transformer attention: 16-head MHA (no GQA, no QK norm), standard
/// RoPE (theta 10000), causal over the chunk (no KV cache; chunks are short).
struct Code2WavAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    rope_base: f32,
    scale: f32,
}

impl Code2WavAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &Code2WavConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?,
            k_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), gs, bits)?,
            o_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?,
            num_heads: cfg.num_attention_heads as i32,
            head_dim: head_dim as i32,
            rope_base: cfg.rope_theta,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let to_bhsd = |t: &MlxArray| {
            let t = mlxcel_core::reshape(t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = to_bhsd(&self.q_proj.forward(x));
        let k = to_bhsd(&self.k_proj.forward(x));
        let v = to_bhsd(&self.v_proj.forward(x));
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, 0);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, 0);
        let attn = mlxcel_core::causal_attention(&q, &k, &v, self.scale, 0.0, 0);
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, -1]);
        self.o_proj.forward(&attn)
    }
}

struct Code2WavTransformerLayer {
    self_attn: Code2WavAttention,
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    attn_scale: UniquePtr<MlxArray>,
    mlp_scale: UniquePtr<MlxArray>,
}

impl Code2WavTransformerLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &Code2WavConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let rms = |key: &str| -> Result<RMSNorm, String> {
            Ok(RMSNorm::new(
                get_weight(weights, &format!("{prefix}.{key}.weight"))?,
                cfg.rms_norm_eps,
            ))
        };
        Ok(Self {
            self_attn: Code2WavAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                cfg,
                gs,
                bits,
            )?,
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.up_proj"),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.down_proj"),
                gs,
                bits,
            )?,
            input_layernorm: rms("input_layernorm")?,
            post_attention_layernorm: rms("post_attention_layernorm")?,
            attn_scale: get_weight(weights, &format!("{prefix}.self_attn_layer_scale.scale"))?,
            mlp_scale: get_weight(weights, &format!("{prefix}.mlp_layer_scale.scale"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let r = self.self_attn.forward(&self.input_layernorm.forward(x));
        let r = mlxcel_core::multiply(&self.attn_scale, &r);
        let h = mlxcel_core::add(x, &r);

        let normed = self.post_attention_layernorm.forward(&h);
        let gate = self.gate_proj.forward(&normed);
        let up = self.up_proj.forward(&normed);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        let r = self.down_proj.forward(&activated);
        let r = mlxcel_core::multiply(&self.mlp_scale, &r);
        mlxcel_core::add(&h, &r)
    }
}

pub struct Code2WavModel {
    code_embedding: UnifiedEmbedding,
    pre_layers: Vec<Code2WavTransformerLayer>,
    pre_norm: RMSNorm,
    /// (transposed conv, ConvNeXt) per upsampling ratio.
    upsample: Vec<(CausalTransConv1d, ConvNeXtBlock)>,
    decoder_in: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_snake: SnakeBeta,
    decoder_out: CausalConv1d,
    num_quantizers: usize,
    codebook_size: usize,
    samples_per_frame: usize,
}

impl Code2WavModel {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &Code2WavConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut pre_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            pre_layers.push(Code2WavTransformerLayer::from_weights(
                weights,
                &format!("{prefix}.pre_transformer.layers.{i}"),
                cfg,
                gs,
                bits,
            )?);
        }

        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        for (i, &ratio) in cfg.upsampling_ratios.iter().enumerate() {
            upsample.push((
                CausalTransConv1d::from_weights(
                    weights,
                    &format!("{prefix}.upsample.{i}.0.conv"),
                    ratio as i32,
                    ratio as i32,
                )?,
                ConvNeXtBlock::from_weights(
                    weights,
                    &format!("{prefix}.upsample.{i}.1"),
                    cfg.hidden_size,
                    gs,
                    bits,
                )?,
            ));
        }

        let mut decoder_blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
            decoder_blocks.push(DecoderBlock::from_weights(
                weights,
                &format!("{prefix}.decoder.{}", i + 1),
                rate as i32,
            )?);
        }
        let n_blocks = cfg.upsample_rates.len();

        Ok(Self {
            code_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{prefix}.code_embedding"),
                gs,
                bits,
            )?,
            pre_norm: RMSNorm::new(
                get_weight(weights, &format!("{prefix}.pre_transformer.norm.weight"))?,
                cfg.rms_norm_eps,
            ),
            pre_layers,
            upsample,
            decoder_in: CausalConv1d::from_weights(
                weights,
                &format!("{prefix}.decoder.0.conv"),
                7,
                1,
                1,
                1,
            )?,
            decoder_blocks,
            decoder_snake: SnakeBeta::from_weights(
                weights,
                &format!("{prefix}.decoder.{}", n_blocks + 1),
            )?,
            decoder_out: CausalConv1d::from_weights(
                weights,
                &format!("{prefix}.decoder.{}.conv", n_blocks + 2),
                7,
                1,
                1,
                1,
            )?,
            num_quantizers: cfg.num_quantizers,
            codebook_size: cfg.codebook_size,
            samples_per_frame: cfg.samples_per_frame(),
        })
    }

    /// Decode one chunk of codes `[1, Q, L]` into a waveform `[1, L * factor, 1]`
    /// clipped to `[-1, 1]`.
    fn forward_codes(&self, codes: &MlxArray) -> UniquePtr<MlxArray> {
        // Offset codebook q into its own slice of the shared embedding table,
        // embed, then average over the quantizer axis.
        let offsets: Vec<i32> = (0..self.num_quantizers)
            .map(|q| (q * self.codebook_size) as i32)
            .collect();
        let offsets = mlxcel_core::from_slice_i32(&offsets, &[1, self.num_quantizers as i32, 1]);
        let ids = mlxcel_core::add(codes, &offsets);
        let embedded = self.code_embedding.forward(&ids); // [1, Q, L, H]
        let mut h = mlxcel_core::mean_axis(&embedded, 1, false); // [1, L, H]

        for layer in &self.pre_layers {
            h = layer.forward(&h);
        }
        h = self.pre_norm.forward(&h);

        for (trans_conv, convnext) in &self.upsample {
            h = trans_conv.forward(&h);
            h = convnext.forward(&h);
        }

        let mut wav = self.decoder_in.forward(&h);
        for block in &self.decoder_blocks {
            wav = block.forward(&wav);
        }
        wav = self.decoder_snake.forward(&wav);
        wav = self.decoder_out.forward(&wav); // [1, samples, 1]

        let dtype = mlxcel_core::array_dtype(&wav);
        let lo = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[-1.0], &[1]), dtype);
        let hi = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[1.0], &[1]), dtype);
        mlxcel_core::clip(&wav, &lo, &hi)
    }

    /// Decode all frames chunk-by-chunk (chunk 300, left context 25, matching
    /// the reference `chunked_decode`) and return host samples at
    /// [`super::speech_config::SPEECH_SAMPLE_RATE`].
    pub fn decode_chunked(&self, frames: &[CodecFrame]) -> Result<Vec<f32>, String> {
        const CHUNK_SIZE: usize = 300;
        const LEFT_CONTEXT: usize = 25;

        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.num_quantizers;
        let total = frames.len();
        for (i, frame) in frames.iter().enumerate() {
            if frame.len() != q {
                return Err(format!(
                    "code2wav frame {i} carries {} codes, expected {q}",
                    frame.len()
                ));
            }
            for (qi, &code) in frame.iter().enumerate() {
                if code < 0 || code as usize >= self.codebook_size {
                    return Err(format!(
                        "code2wav frame {i} quantizer {qi} has codec id {code}, expected 0..{}",
                        self.codebook_size.saturating_sub(1)
                    ));
                }
            }
        }

        // [1, Q, L] layout, frame-major input.
        let mut data = vec![0i32; q * total];
        for (l, frame) in frames.iter().enumerate() {
            for (qi, &code) in frame.iter().enumerate() {
                data[qi * total + l] = code;
            }
        }

        let factor = self.samples_per_frame;
        let mut samples = Vec::with_capacity(total * factor);
        let mut start = 0usize;
        while start < total {
            let end = (start + CHUNK_SIZE).min(total);
            let context_start = start.saturating_sub(LEFT_CONTEXT);
            let chunk_len = end - context_start;

            let mut chunk = vec![0i32; q * chunk_len];
            for qi in 0..q {
                chunk[qi * chunk_len..(qi + 1) * chunk_len]
                    .copy_from_slice(&data[qi * total + context_start..qi * total + end]);
            }
            let codes = mlxcel_core::from_slice_i32(&chunk, &[1, q as i32, chunk_len as i32]);
            let wav = self.forward_codes(&codes);

            let valid_start = ((start - context_start) * factor) as i32;
            let valid_len = ((end - start) * factor) as i32;
            let shape = mlxcel_core::array_shape(&wav);
            let valid = mlxcel_core::slice(
                &wav,
                &[0, valid_start, 0],
                &[shape[0], valid_start + valid_len, shape[2]],
            );
            samples.extend(to_vec_f32(&valid)?);
            start = end;
        }

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::super::speech_config::Code2WavConfig;

    #[test]
    fn samples_per_frame_is_product_of_all_upsample_stages() {
        let cfg: Code2WavConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        // 2 * 2 * 8 * 5 * 4 * 3 = 1920 samples per 12.5 Hz frame = 80 ms at 24 kHz.
        assert_eq!(cfg.samples_per_frame(), 1920);
        assert_eq!(
            cfg.samples_per_frame() as u32 * 25 / 2,
            super::super::speech_config::SPEECH_SAMPLE_RATE
        );
    }
}
