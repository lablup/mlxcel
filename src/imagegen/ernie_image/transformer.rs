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

//! `ErnieImageTransformer`: a single-stream DiT over image and text tokens.
//!
//! A port of `ErnieImagePatchEmbed`, `ErnieImageTimestepEmbedding`,
//! `ErnieImageFinalNorm` and `ErnieImageTransformer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/transformer.py>.
//! Image tokens come first, then the projected text tokens. Image tokens sit at
//! rotary position `(text_length, y, x)` and text token `i` at `(i, 0, 0)`;
//! padded text tokens (classifier-free guidance batches prompts of different
//! lengths) are masked out of every key.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::Linear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::blocks::{
    BlockInputs, ErnieBlock, affine, linear, per_token, rope_tables, split_last, timestep_embedding,
};
use super::config::ErnieTransformerConfig;

/// `nn.LayerNorm(affine=False)` epsilon of the final norm is `config.eps`.
pub struct ErnieTransformer {
    config: ErnieTransformerConfig,
    /// Patch-embed conv kernel in MLX layout `[O, p, p, C]`.
    x_embedder_weight: UniquePtr<MlxArray>,
    x_embedder_bias: Option<UniquePtr<MlxArray>>,
    text_proj: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    adaln_modulation: Linear,
    layers: Vec<ErnieBlock>,
    final_norm_linear: Linear,
    final_linear: Linear,
}

impl ErnieTransformer {
    /// Build from sanitized weights (see [`super::weights::sanitize_transformer_weights`]).
    pub fn from_weights(config: ErnieTransformerConfig, weights: &WeightMap) -> Result<Self> {
        config.validate()?;
        if config.patch_size != 1 {
            // Upstream reshapes the output to the unpatched latent grid, so it
            // only runs patch size 1 too (both published checkpoints).
            bail!(
                "ERNIE-Image patch_size {} is not supported (only 1)",
                config.patch_size
            );
        }
        let x_embedder_weight = weights
            .get("x_embedder.proj.weight")
            .map(|w| mlxcel_core::copy(w))
            .context("ERNIE-Image transformer weight not found: x_embedder.proj.weight")?;
        let x_embedder_bias = weights
            .get("x_embedder.proj.bias")
            .map(|b| mlxcel_core::copy(b));
        let head_dim = config.head_dim();
        let layers = (0..config.num_layers)
            .map(|i| {
                ErnieBlock::load(
                    weights,
                    &format!("layers.{i}"),
                    config.num_attention_heads,
                    head_dim,
                    config.qk_layernorm,
                    config.eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x_embedder_weight,
            x_embedder_bias,
            text_proj: linear(weights, "text_proj")?,
            time_linear_1: linear(weights, "time_embedding.linear_1")?,
            time_linear_2: linear(weights, "time_embedding.linear_2")?,
            adaln_modulation: linear(weights, "adaln_modulation")?,
            layers,
            final_norm_linear: linear(weights, "final_norm.linear")?,
            final_linear: linear(weights, "final_linear")?,
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &ErnieTransformerConfig {
        &self.config
    }

    /// `[B, C, H, W]` latents -> `[B, H*W, hidden]` tokens through the
    /// patch-embed conv.
    fn embed_patches(&self, latents: &MlxArray) -> UniquePtr<MlxArray> {
        let nhwc = mlxcel_core::transpose_axes(latents, &[0, 2, 3, 1]);
        let p = self.config.patch_size;
        let mut x = mlxcel_core::conv2d(&nhwc, &self.x_embedder_weight, p, p, 0, 0, 1, 1, 1);
        if let Some(bias) = &self.x_embedder_bias {
            x = mlxcel_core::add(&x, bias);
        }
        let s = mlxcel_core::array_shape(&x);
        mlxcel_core::reshape(&x, &[s[0], s[1] * s[2], s[3]])
    }

    /// Conditioning vector `[B, hidden]` of the timesteps: the sinusoidal
    /// embedding cast to the timestep dtype, then the two-layer SiLU MLP.
    fn conditioning(&self, timestep: &MlxArray) -> UniquePtr<MlxArray> {
        let emb = timestep_embedding(timestep, self.config.hidden_size);
        let emb = mlxcel_core::astype(&emb, mlxcel_core::array_dtype(timestep));
        let h = mlxcel_core::silu(&affine(&self.time_linear_1, &emb));
        affine(&self.time_linear_2, &h)
    }

    /// Velocity prediction `[B, out_channels, H, W]` for `latents`
    /// (`[B, in_channels, H, W]`) at `timestep` (`[B]`, in the latent dtype),
    /// conditioned on `text` (`[B, T, text_in_dim]`, zero-padded) whose row `b`
    /// holds `text_lengths[b]` valid tokens.
    pub fn forward(
        &self,
        latents: &MlxArray,
        timestep: &MlxArray,
        text: &MlxArray,
        text_lengths: &[i32],
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(latents);
        if shape.len() != 4 || shape[1] != self.config.in_channels {
            bail!(
                "expected [B, {}, H, W] latents, got {shape:?}",
                self.config.in_channels
            );
        }
        let (b, h, w) = (shape[0], shape[2], shape[3]);
        let text_shape = mlxcel_core::array_shape(text);
        if text_shape.len() != 3 || text_shape[0] != b || text_lengths.len() != b as usize {
            bail!(
                "text {text_shape:?} and {} lengths do not match batch {b}",
                text_lengths.len()
            );
        }
        let text_tokens = text_shape[1];
        if text_lengths.iter().any(|&n| n < 0 || n > text_tokens) {
            bail!("text lengths {text_lengths:?} exceed the {text_tokens} text tokens");
        }

        let image = self.embed_patches(latents);
        let stream_dtype = mlxcel_core::array_dtype(&image);
        let image_tokens = h * w;
        let text = mlxcel_core::astype(text, stream_dtype);
        let text = self.text_proj.forward(&text);
        let mut sequence = mlxcel_core::concatenate(&image, &text, 1);
        let seq_len = image_tokens + text_tokens;

        let positions = positions(b, h, w, text_tokens, text_lengths);
        let (cos, sin) = rope_tables(
            &positions,
            &self.config.rope_axes_dim,
            self.config.rope_theta,
        )?;
        let cos = mlxcel_core::astype(&cos, stream_dtype);
        let sin = mlxcel_core::astype(&sin, stream_dtype);
        let mask = key_mask(b, image_tokens, text_tokens, text_lengths, stream_dtype);

        let conditioning = self.conditioning(timestep);
        let modulation = mlxcel_core::astype(
            &affine(&self.adaln_modulation, &mlxcel_core::silu(&conditioning)),
            dtype::FLOAT32,
        );
        let modulation = split_last(&modulation, 6);
        let inputs = BlockInputs {
            modulation: &modulation,
            cos: &cos,
            sin: &sin,
            mask: &mask,
        };
        for layer in &self.layers {
            sequence = layer.forward(&sequence, &inputs);
        }

        // `ErnieImageFinalNorm`: affine-free LayerNorm, then the conditioning's
        // scale and shift.
        let scale_shift = split_last(&affine(&self.final_norm_linear, &conditioning), 2);
        // SAFETY: null weight and bias select the affine-free form; `sequence`
        // is alive for the call.
        let normed = unsafe {
            mlxcel_core::fast_layer_norm(
                &sequence,
                std::ptr::null(),
                std::ptr::null(),
                self.config.eps,
            )
        };
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&scale_shift[0]));
        let scaled = mlxcel_core::multiply(
            &normed,
            &mlxcel_core::add(&one, &per_token(&scale_shift[0])),
        );
        let out = mlxcel_core::add(&scaled, &per_token(&scale_shift[1]));
        let out = mlxcel_core::astype(&out, mlxcel_core::array_dtype(&normed));
        let width = mlxcel_core::array_shape(&out)[2];
        let image_out = mlxcel_core::slice(&out, &[0, 0, 0], &[b, image_tokens, width]);
        let image_out = affine(&self.final_linear, &image_out);
        debug_assert_eq!(seq_len, mlxcel_core::array_shape(&out)[1]);
        let grid = mlxcel_core::reshape(&image_out, &[b, h, w, self.config.out_channels]);
        Ok(mlxcel_core::transpose_axes(&grid, &[0, 3, 1, 2]))
    }
}

/// `[B, H*W + T, 3]` f32 rotary positions: image cell `(y, x)` of row `b` at
/// `(text_lengths[b], y, x)`, text token `i` at `(i, 0, 0)`.
fn positions(
    b: i32,
    h: i32,
    w: i32,
    text_tokens: i32,
    text_lengths: &[i32],
) -> UniquePtr<MlxArray> {
    let per_row = ((h * w + text_tokens) * 3) as usize;
    let mut data = Vec::with_capacity(per_row * b as usize);
    for &length in text_lengths {
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[length as f32, y as f32, x as f32]);
            }
        }
        for i in 0..text_tokens {
            data.extend_from_slice(&[i as f32, 0.0, 0.0]);
        }
    }
    mlxcel_core::from_slice_f32(&data, &[b, h * w + text_tokens, 3])
}

/// `[B, 1, 1, L]` additive key mask: 0 for image tokens and valid text tokens,
/// `-inf` for padding, built in f32 and cast to `dt` as upstream does.
fn key_mask(
    b: i32,
    image_tokens: i32,
    text_tokens: i32,
    text_lengths: &[i32],
    dt: i32,
) -> UniquePtr<MlxArray> {
    let len = (image_tokens + text_tokens) as usize;
    let mut data = Vec::with_capacity(len * b as usize);
    for &length in text_lengths {
        data.extend(std::iter::repeat_n(0.0_f32, image_tokens as usize));
        data.extend((0..text_tokens).map(|i| if i < length { 0.0 } else { f32::NEG_INFINITY }));
    }
    let mask = mlxcel_core::from_slice_f32(&data, &[b, 1, 1, len as i32]);
    mlxcel_core::astype(&mask, dt)
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
