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

//! Qwen3-Omni audio modality (thinker `audio_tower.*`).
//!
//! Consumes a 128-bin log-mel spectrogram at 100 frames/second (no fixed
//! 30-second padding; the true frame count `L` drives everything):
//!
//! 1. The time axis is split into `2 * n_window` = 100-frame chunks (the last
//!    chunk keeps the remainder), padded to the longest chunk.
//! 2. Three 3x3 stride-2 Conv2d + GELU stages (`1 -> 480 -> 480 -> 480`)
//!    halve both the mel and time axes with `out = ceil(in / 2)` each.
//! 3. The output is flattened `(chunks, t_out, 480 * 16)` and projected by a
//!    bias-free `conv_out` linear to `d_model` = 1280.
//! 4. A sinusoidal position table (computed at construction, never loaded)
//!    restarts at 0 for every chunk.
//! 5. Valid frames are gathered per chunk (the per-chunk conv output length),
//!    totalling `out_len(L) = 13 * (L / 100) + convdown3(L % 100)` frames.
//! 6. 32 pre-LN encoder layers with block-diagonal attention over windows of
//!    `13 * (n_window_infer / (2 * n_window))` = 104 post-conv frames.
//! 7. `ln_post` -> `proj1` -> GELU -> `proj2` to `output_dim` = 2048 rows that
//!    scatter into the token stream at `audio_token_id` positions.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/audio.py>.
//!
//! The speech-output stack (stage 2) lives in the sibling modules: the
//! [`talker`] MoE codec decoder, the [`code2wav`] vocoder, and the
//! [`speech`] pipeline that ties them to a loaded thinker.
//!
//! Used by: Qwen3-Omni MoE VLM (thinker).

pub mod code2wav;
mod code2wav_blocks;
pub mod speech;
pub mod speech_config;
#[cfg(test)]
mod speech_config_tests;
pub mod speech_layers;
pub mod talker;

pub use speech::{Qwen3OmniSpeech, SpeechOutput};
pub use speech_config::{Qwen3OmniSpeechConfig, SPEECH_SAMPLE_RATE};

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::encoders::gemma3n::Conv2dLayer;

fn default_d_model() -> usize {
    1280
}
fn default_layers() -> usize {
    32
}
fn default_heads() -> usize {
    20
}
fn default_ffn() -> usize {
    5120
}
fn default_mels() -> usize {
    128
}
fn default_output_dim() -> usize {
    2048
}
fn default_downsample_hidden() -> usize {
    480
}
fn default_conv_chunksize() -> usize {
    500
}
fn default_n_window() -> usize {
    50
}
fn default_n_window_infer() -> usize {
    800
}
fn default_max_source_positions() -> usize {
    1500
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3OmniAudioConfig {
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_ffn")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_mels")]
    pub num_mel_bins: usize,
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
    #[serde(default = "default_downsample_hidden")]
    pub downsample_hidden_size: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

/// One stride-2 conv stage: `ceil(x / 2)`.
fn conv_down(x: usize) -> usize {
    x.div_ceil(2)
}

/// Post-conv length of one chunk of `len` mel frames (three stride-2 stages).
pub fn chunk_out_len(len: usize) -> usize {
    conv_down(conv_down(conv_down(len)))
}

/// Encoder output frames for a clip of `l` mel frames: 13 per full second
/// (100-frame chunk) plus the conv-downsampled remainder.
pub fn audio_out_len(l: usize) -> usize {
    let full = l / 100;
    let rem = l % 100;
    full * chunk_out_len(100) + if rem > 0 { chunk_out_len(rem) } else { 0 }
}

fn load_layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Qwen3-Omni audio weight missing: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

struct EncoderLayer {
    self_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl EncoderLayer {
    /// `x`: `(1, frames, d_model)`; attention is block-diagonal over
    /// `window_lens` segments along the frame axis.
    fn forward(&self, x: &MlxArray, window_lens: &[i32]) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let normed = self.self_attn_layer_norm.forward(x);
        let q = self.q_proj.forward(&normed);
        let k = self.k_proj.forward(&normed);
        let v = self.v_proj.forward(&normed);
        let to_bhsd = |t: &MlxArray| {
            let t = mlxcel_core::reshape(t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = to_bhsd(&q);
        let k = to_bhsd(&k);
        let v = to_bhsd(&v);

        let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(window_lens.len());
        let mut offset = 0i32;
        for &len in window_lens {
            let seg = |t: &MlxArray| {
                mlxcel_core::slice(
                    t,
                    &[0, 0, offset, 0],
                    &[b, self.num_heads, offset + len, self.head_dim],
                )
            };
            let (qs, ks, vs) = (seg(&q), seg(&k), seg(&v));
            // SAFETY: segment slices are valid; null mask (full attention).
            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &qs,
                    &ks,
                    &vs,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
            outputs.push(attn);
            offset += len;
        }
        let attn = if outputs.len() == 1 {
            outputs.into_iter().next().unwrap()
        } else {
            crate::vision::encoders::qwen2_vl::concat_many(&outputs, 2)
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, -1]);
        let attn = self.out_proj.forward(&attn);
        let h = mlxcel_core::add(x, &attn);

        let normed = self.final_layer_norm.forward(&h);
        let m = self.fc1.forward(&normed);
        let m = mlxcel_core::gelu(&m);
        let m = self.fc2.forward(&m);
        mlxcel_core::add(&h, &m)
    }
}

pub struct Qwen3OmniAudioEncoder {
    conv2d1: Conv2dLayer,
    conv2d2: Conv2dLayer,
    conv2d3: Conv2dLayer,
    conv_out: UnifiedLinear, // (d_model, 480 * mel_down), no bias
    pos_table: Vec<f32>,     // (max_source_positions, d_model) sinusoids
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: UnifiedLinear,
    proj2: UnifiedLinear,
    num_mel_bins: usize,
    d_model: usize,
    n_window: usize,
    n_window_infer: usize,
}

impl Qwen3OmniAudioEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Qwen3OmniAudioConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let conv = |name: &str| -> Result<Conv2dLayer, String> {
            let w = weights
                .get(&format!("{prefix}.{name}.weight"))
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| {
                    format!("Qwen3-Omni audio weight missing: {prefix}.{name}.weight")
                })?;
            let b = weights
                .get(&format!("{prefix}.{name}.bias"))
                .map(|b| mlxcel_core::copy(b));
            Ok(Conv2dLayer {
                weight: w,
                bias: b,
                stride_h: 2,
                stride_w: 2,
                padding_h: 1,
                padding_w: 1,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
            })
        };

        // Sinusoidal position table: first half sine, second half cosine,
        // max timescale 10000. Computed here, never loaded.
        let half = config.d_model / 2;
        let log_inc = (10_000f32).ln() / (half as f32 - 1.0);
        let mut pos_table = Vec::with_capacity(config.max_source_positions * config.d_model);
        for p in 0..config.max_source_positions {
            for j in 0..half {
                pos_table.push((p as f32 * (-(log_inc * j as f32)).exp()).sin());
            }
            for j in 0..half {
                pos_table.push((p as f32 * (-(log_inc * j as f32)).exp()).cos());
            }
        }

        let head_dim = config.d_model / config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let lp = format!("{prefix}.layers.{i}");
            layers.push(EncoderLayer {
                self_attn_layer_norm: load_layer_norm(
                    weights,
                    &format!("{lp}.self_attn_layer_norm"),
                )?,
                final_layer_norm: load_layer_norm(weights, &format!("{lp}.final_layer_norm"))?,
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{lp}.self_attn.q_proj"),
                    gs,
                    bits,
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{lp}.self_attn.k_proj"),
                    gs,
                    bits,
                )?,
                v_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{lp}.self_attn.v_proj"),
                    gs,
                    bits,
                )?,
                out_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{lp}.self_attn.out_proj"),
                    gs,
                    bits,
                )?,
                fc1: UnifiedLinear::from_weights(weights, &format!("{lp}.fc1"), gs, bits)?,
                fc2: UnifiedLinear::from_weights(weights, &format!("{lp}.fc2"), gs, bits)?,
                num_heads: config.encoder_attention_heads as i32,
                head_dim: head_dim as i32,
                scale: (head_dim as f32).powf(-0.5),
            });
        }

        Ok(Self {
            conv2d1: conv("conv2d1")?,
            conv2d2: conv("conv2d2")?,
            conv2d3: conv("conv2d3")?,
            conv_out: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.conv_out"),
                gs,
                bits,
            )?,
            pos_table,
            layers,
            ln_post: load_layer_norm(weights, &format!("{prefix}.ln_post"))?,
            proj1: UnifiedLinear::from_weights(weights, &format!("{prefix}.proj1"), gs, bits)?,
            proj2: UnifiedLinear::from_weights(weights, &format!("{prefix}.proj2"), gs, bits)?,
            num_mel_bins: config.num_mel_bins,
            d_model: config.d_model,
            n_window: config.n_window,
            n_window_infer: config.n_window_infer,
        })
    }

    /// Encode one clip's log-mel features. `mel`: row-major
    /// `[num_frames][num_mel_bins]` (the [`crate::audio::whisper_mel`] layout).
    /// Returns `(audio_out_len(L), output_dim)` feature rows.
    pub fn forward(&self, mel: &[f32], num_frames: usize) -> UniquePtr<MlxArray> {
        let mels = self.num_mel_bins;
        let chunk_step = 2 * self.n_window; // 100 mel frames per chunk

        // Chunk boundaries along the time axis.
        let mut chunk_lens: Vec<usize> = Vec::new();
        let mut remaining = num_frames;
        while remaining > 0 {
            let len = remaining.min(chunk_step);
            chunk_lens.push(len);
            remaining -= len;
        }
        let num_chunks = chunk_lens.len();
        let max_len = *chunk_lens.iter().max().unwrap_or(&0);

        // Build (chunks, mels, max_len, 1) zero-padded, transposing the
        // frame-major input to mel-major per chunk.
        let mut data = vec![0f32; num_chunks * mels * max_len];
        let mut start = 0usize;
        for (ci, &clen) in chunk_lens.iter().enumerate() {
            for m in 0..mels {
                for t in 0..clen {
                    data[ci * mels * max_len + m * max_len + t] = mel[(start + t) * mels + m];
                }
            }
            start += clen;
        }
        let x = mlxcel_core::from_slice_f32(
            &data,
            &[num_chunks as i32, mels as i32, max_len as i32, 1],
        );

        // Conv front-end (channels-last NHWC: height = mel, width = time).
        let x = mlxcel_core::gelu(&self.conv2d1.forward(&x));
        let x = mlxcel_core::gelu(&self.conv2d2.forward(&x));
        let x = mlxcel_core::gelu(&self.conv2d3.forward(&x)); // (chunks, mel_dn, t_out, 480)

        // Flatten (channel-major over mel): (chunks, t_out, 480 * mel_dn).
        let s = mlxcel_core::array_shape(&x);
        let (chunks, mel_dn, t_out, c) = (s[0], s[1], s[2], s[3]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 3, 1]); // (chunks, t_out, c, mel_dn)
        let x = mlxcel_core::reshape(&x, &[chunks, t_out, c * mel_dn]);
        let x = self.conv_out.forward(&x); // (chunks, t_out, d_model)

        // Positions restart at 0 for every chunk.
        let pos_len = (t_out as usize).min(self.pos_table.len() / self.d_model);
        let pos = mlxcel_core::from_slice_f32(
            &self.pos_table[..pos_len * self.d_model],
            &[1, pos_len as i32, self.d_model as i32],
        );
        let pos = mlxcel_core::astype(&pos, mlxcel_core::array_dtype(&x));
        let x = mlxcel_core::add(&x, &pos);

        // Gather the valid frames of each chunk in order.
        let mut valid: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(num_chunks);
        for (ci, &clen) in chunk_lens.iter().enumerate() {
            let out_len = chunk_out_len(clen) as i32;
            let seg = mlxcel_core::slice(
                &x,
                &[ci as i32, 0, 0],
                &[ci as i32 + 1, out_len, self.d_model as i32],
            );
            valid.push(seg);
        }
        let mut h = if valid.len() == 1 {
            valid.into_iter().next().unwrap()
        } else {
            crate::vision::encoders::qwen2_vl::concat_many(&valid, 1)
        }; // (1, total_frames, d_model)

        // Block-diagonal attention windows: full chunks contribute
        // chunk_out_len(100) frames each; a window spans
        // n_window_infer / (2 * n_window) chunks worth of frames.
        let total_frames: usize = chunk_lens.iter().map(|&l| chunk_out_len(l)).sum();
        let window = chunk_out_len(chunk_step) * (self.n_window_infer / chunk_step);
        let mut window_lens: Vec<i32> = Vec::new();
        let mut left = total_frames;
        while left > 0 {
            let take = left.min(window);
            window_lens.push(take as i32);
            left -= take;
        }

        for layer in &self.layers {
            h = layer.forward(&h, &window_lens);
        }

        let h = self.ln_post.forward(&h);
        let h = self.proj1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        let h = self.proj2.forward(&h); // (1, total_frames, output_dim)
        let out_shape = mlxcel_core::array_shape(&h);
        mlxcel_core::reshape(&h, &[out_shape[1], out_shape[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_out_len_matches_reference_formula() {
        // 13 output frames per full 100-frame second.
        assert_eq!(audio_out_len(100), 13);
        assert_eq!(audio_out_len(500), 65);
        // Remainder path: 163 -> 13 + ceil(ceil(ceil(63/2)/2)/2) = 13 + 8 = 21.
        assert_eq!(audio_out_len(163), 21);
        assert_eq!(audio_out_len(1), 1);
        assert_eq!(audio_out_len(0), 0);
    }

    #[test]
    fn chunk_out_len_is_triple_ceil_div() {
        assert_eq!(chunk_out_len(100), 13);
        assert_eq!(chunk_out_len(8), 1);
        assert_eq!(chunk_out_len(63), 8);
    }
}
