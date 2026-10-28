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

//! The LLaDA-Image denoiser.
//!
//! A port of `LLaDAImageTransformer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/transformer.py>,
//! assembled from the shared [`crate::imagegen::dit_blocks`] with
//! parameter-free RMS norms ([`BlockNorms::Weightless`]) and q/k rounded back
//! to the activation dtype after the f32 rotation
//! ([`RopePrecision::RoundToInput`]). Patch size is 1: every latent cell of
//! the `[B, 128, H/16, W/16]` FLUX.2 latents is one token.
//!
//! Generation runs the image tokens through the noise refiner and the
//! caption (plus, in VQ mode, the SigVQ semantic features) through the
//! context refiner, then the joint layers over `[image, caption]`. Editing
//! uses `[caption, caption, source, target, sigvq]` with per-token
//! modulation: noisy tokens read the timestep embedding, clean tokens the
//! embedding of `t = 0`. Every stream is padded to a multiple of 32 with its
//! learned pad token; positions follow upstream's `sequence_positions` and
//! `image_positions`.

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::conditioning::{Dense, bool_array, plain_rms_norm, shaped};
use super::config::TransformerConfig;
use crate::imagegen::dit_blocks::{
    ADALN_EMBED_DIM, BiasAdd, BlockNorms, FinalLayer, MRoPE, RopePrecision, TimestepEmbedder,
    ZImageTransformerBlock,
};

/// Every token stream is padded to a multiple of this length.
pub const SEQUENCE_MULTIPLE: i32 = 32;

fn padding(length: i32) -> i32 {
    (-length).rem_euclid(SEQUENCE_MULTIPLE)
}

/// `sequence_positions(length, start)`: `(start + i, 0, 0)` rows, then zero
/// rows up to the padded length. Row-major `[L_padded * 3]`.
#[must_use]
pub fn sequence_positions(length: i32, start: i32) -> Vec<f32> {
    let mut out: Vec<f32> = (0..length)
        .flat_map(|i| [(start + i) as f32, 0.0, 0.0])
        .collect();
    out.resize(3 * (length + padding(length)) as usize, 0.0);
    out
}

/// `image_positions(height, width, start)`: `(start, row, column)` rows, then
/// zero rows up to the padded length.
#[must_use]
pub fn image_positions(height: i32, width: i32, start: i32) -> Vec<f32> {
    let length = height * width;
    let mut out = Vec::with_capacity(3 * (length + padding(length)) as usize);
    for r in 0..height {
        for c in 0..width {
            out.extend([start as f32, r as f32, c as f32]);
        }
    }
    out.resize(3 * (length + padding(length)) as usize, 0.0);
    out
}

fn positions_array(values: &[f32]) -> UniquePtr<MlxArray> {
    let len = (values.len() / 3) as i32;
    mlxcel_core::from_slice_f32(values, &[1, len, 3])
}

/// `pad_tokens`: append `(-L) % 32` copies of `token` (`[1, D]`).
fn pad_tokens(x: &MlxArray, token: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let pad = padding(shape[1]);
    if pad == 0 {
        return mlxcel_core::copy(x);
    }
    let token = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(token, &[1, 1, shape[2]]),
        &[shape[0], pad, shape[2]],
    );
    // Upstream's concatenate promotes; the CUDA build of MLX would not.
    let token = mlxcel_core::astype(&token, upstream_dtype(x, &token));
    let x = mlxcel_core::astype(x, upstream_dtype(x, &token));
    mlxcel_core::concatenate(&x, &token, 1)
}

/// Upstream MLX's promotion of two floating dtypes (see `dit_blocks`).
fn upstream_dtype(a: &MlxArray, b: &MlxArray) -> i32 {
    let (da, db) = (mlxcel_core::array_dtype(a), mlxcel_core::array_dtype(b));
    if da == db { da } else { dtype::FLOAT32 }
}

fn concat(parts: &[&MlxArray]) -> UniquePtr<MlxArray> {
    let dt = parts
        .iter()
        .skip(1)
        .fold(mlxcel_core::array_dtype(parts[0]), |acc, p| {
            if acc == mlxcel_core::array_dtype(p) {
                acc
            } else {
                dtype::FLOAT32
            }
        });
    let cast: Vec<UniquePtr<MlxArray>> = parts.iter().map(|p| mlxcel_core::astype(p, dt)).collect();
    let refs: Vec<&MlxArray> = cast.iter().map(|c| &**c).collect();
    mlxcel_core::concatenate_many(&refs, 1)
}

/// `[RMSNorm, Linear]` embedders (`cap_embedder`, `semantic_embedder`,
/// `sigvq_embedder`).
struct NormDense {
    dense: Dense,
    eps: f32,
}

impl NormDense {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.dense.forward(&plain_rms_norm(x, self.eps))
    }
}

/// The denoiser.
pub struct LladaImageTransformer {
    config: TransformerConfig,
    x_embedder: Dense,
    t_embedder: TimestepEmbedder,
    cap_embedder: NormDense,
    semantic_embedder: NormDense,
    sigvq_embedder: NormDense,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    sigvq_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    final_layer: FinalLayer,
    x_pad_token: UniquePtr<MlxArray>,
    cap_pad_token: UniquePtr<MlxArray>,
    sigvq_pad_token: UniquePtr<MlxArray>,
    rope: MRoPE,
}

impl LladaImageTransformer {
    /// Build from weights already passed through
    /// [`super::weights::sanitize_transformer_weights`].
    pub fn from_weights(config: TransformerConfig, weights: &WeightMap) -> Result<Self> {
        Self::from_weights_with_rope(config, weights, RopePrecision::RoundToInput)
    }

    /// [`Self::from_weights`] with a chosen RoPE precision. Upstream rounds q
    /// and k back to bf16 after rotation; the parity tests build the
    /// `KeepPromoted` variant as a wrong-port control.
    pub(crate) fn from_weights_with_rope(
        config: TransformerConfig,
        weights: &WeightMap,
        rope: RopePrecision,
    ) -> Result<Self> {
        config.validate()?;
        let (dim, heads, head_dim) = (config.dim, config.n_heads, config.head_dim());
        let blocks = |name: &str, count: i32, modulation: bool| {
            (0..count)
                .map(|i| {
                    ZImageTransformerBlock::load_with_norms(
                        weights,
                        &format!("{name}.{i}"),
                        heads,
                        head_dim,
                        modulation,
                        rope,
                        BlockNorms::Weightless,
                    )
                    .map(|b| b.with_bias_add(BiasAdd::Fused))
                })
                .collect::<Result<Vec<_>>>()
        };
        let embed = |name: &str, in_dim: i32| -> Result<NormDense> {
            Ok(NormDense {
                dense: Dense::load(weights, &format!("{name}.1"), in_dim, dim, true)?,
                eps: config.norm_eps,
            })
        };
        let n_ref = config.n_refiner_layers;
        Ok(Self {
            x_embedder: Dense::load(weights, "x_embedder", config.in_channels, dim, true)?,
            t_embedder: TimestepEmbedder::load(weights, "t_embedder", ADALN_EMBED_DIM)?
                .with_bias_add(BiasAdd::Fused),
            cap_embedder: embed("cap_embedder", config.cap_feat_dim)?,
            semantic_embedder: embed("semantic_embedder", config.semantic_feat_dim)?,
            sigvq_embedder: embed("sigvq_embedder", config.semantic_feat_dim)?,
            noise_refiner: blocks("noise_refiner", n_ref, true)?,
            context_refiner: blocks("context_refiner", n_ref, false)?,
            sigvq_refiner: blocks("sigvq_refiner", n_ref, false)?,
            layers: blocks("layers", config.n_layers, true)?,
            final_layer: FinalLayer::load(weights, "final_layer")?.with_bias_add(BiasAdd::Fused),
            x_pad_token: shaped(weights, "x_pad_token", &[1, dim])?,
            cap_pad_token: shaped(weights, "cap_pad_token", &[1, dim])?,
            sigvq_pad_token: shaped(weights, "sigvq_pad_token", &[1, dim])?,
            rope: MRoPE {
                sections: config.axes_dims.clone(),
                theta: config.rope_theta,
            },
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// `x_embedder` dtype: the transformer's compute dtype.
    #[must_use]
    pub fn dtype(&self) -> i32 {
        mlxcel_core::array_dtype(&self.x_pad_token)
    }

    /// `[B, C, H, W] -> [B, H * W, C]`, embedded and padded.
    fn embed_image(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let patches = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(x, &[0, 2, 3, 1]),
            &[s[0], s[2] * s[3], s[1]],
        );
        pad_tokens(&self.x_embedder.forward(&patches), &self.x_pad_token)
    }

    fn timestep(&self, t: &MlxArray) -> UniquePtr<MlxArray> {
        let scale = mlxcel_core::full_f32(&[], self.config.t_scale, mlxcel_core::array_dtype(t));
        self.t_embedder.forward(&mlxcel_core::multiply(t, &scale))
    }

    fn check_inputs(&self, x: &MlxArray, caption: &MlxArray) -> Result<(i32, i32, i32, i32)> {
        let s = mlxcel_core::array_shape(x);
        if s.len() != 4 || s[1] != self.config.in_channels {
            bail!(
                "LLaDA-Image latents must be [B, {}, H, W], got {s:?}",
                self.config.in_channels
            );
        }
        let c = mlxcel_core::array_shape(caption);
        if c.len() != 3 || c[2] != self.config.cap_feat_dim || c[1] < 1 {
            bail!(
                "LLaDA-Image caption features must be [B, L, {}], got {c:?}",
                self.config.cap_feat_dim
            );
        }
        Ok((s[0], s[1], s[2], s[3]))
    }

    fn final_output(
        &self,
        tokens: &MlxArray,
        start: i32,
        time: &MlxArray,
        (b, c, h, w): (i32, i32, i32, i32),
    ) -> UniquePtr<MlxArray> {
        let image =
            mlxcel_core::slice(tokens, &[0, start, 0], &[b, start + h * w, self.config.dim]);
        let out = self.final_layer.forward(&image, time);
        let out = mlxcel_core::reshape(&out, &[b, h, w, c]);
        mlxcel_core::transpose_axes(&out, &[0, 3, 1, 2])
    }

    /// The flow prediction for latents `x` (`[B, 128, H, W]`) at time `t`
    /// (`[1]` or `[B]`, the sigma) given caption features `caption`
    /// (`[B, L, cap_feat_dim]`). `semantic` (`[B, N, semantic_feat_dim]`, `N`
    /// may be 0) adds the SigVQ features of VQ generation; `source_latents`
    /// selects editing, where `semantic` is the source image's features.
    pub fn forward(
        &self,
        x: &MlxArray,
        t: &MlxArray,
        caption: &MlxArray,
        semantic: Option<&MlxArray>,
        source_latents: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let dims = self.check_inputs(x, caption)?;
        if let Some(source) = source_latents {
            let Some(semantic) = semantic else {
                bail!("Editing requires SigVQ features");
            };
            return self.edit(x, t, caption, semantic, source, dims);
        }
        let (_, _, h, w) = dims;
        let mut image_tokens = self.embed_image(x);
        let mut caption_tokens =
            pad_tokens(&self.cap_embedder.forward(caption), &self.cap_pad_token);
        let cap_len = mlxcel_core::array_shape(caption)[1];
        let mut cap_pos = sequence_positions(cap_len, 1);
        let mut image_start = mlxcel_core::array_shape(&caption_tokens)[1] + 1;
        if let Some(semantic) = semantic.filter(|s| mlxcel_core::array_shape(s)[1] > 0) {
            let features = self.semantic_embedder.forward(semantic);
            let features = pad_tokens(&features, &self.cap_pad_token);
            let n = mlxcel_core::array_shape(semantic)[1];
            cap_pos.extend(sequence_positions(n, image_start));
            caption_tokens = concat(&[&caption_tokens, &features]);
            image_start = mlxcel_core::array_shape(&caption_tokens)[1] + 1;
        }
        let image_pos = image_positions(h, w, image_start);
        let (cap_cos, cap_sin) = self.rope.compute_freqs(&positions_array(&cap_pos))?;
        let (image_cos, image_sin) = self.rope.compute_freqs(&positions_array(&image_pos))?;
        let time = self.timestep(t);
        for block in &self.noise_refiner {
            image_tokens =
                block.forward(&image_tokens, &image_cos, &image_sin, Some(&time), None)?;
        }
        for block in &self.context_refiner {
            caption_tokens = block.forward(&caption_tokens, &cap_cos, &cap_sin, None, None)?;
        }
        let mut tokens = concat(&[&image_tokens, &caption_tokens]);
        let cos = concat(&[&image_cos, &cap_cos]);
        let sin = concat(&[&image_sin, &cap_sin]);
        for block in &self.layers {
            tokens = block.forward(&tokens, &cos, &sin, Some(&time), None)?;
        }
        Ok(self.final_output(&tokens, 0, &time, dims))
    }

    fn edit(
        &self,
        x: &MlxArray,
        t: &MlxArray,
        caption: &MlxArray,
        semantic: &MlxArray,
        source: &MlxArray,
        dims: (i32, i32, i32, i32),
    ) -> Result<UniquePtr<MlxArray>> {
        let (b, _, h, w) = dims;
        if mlxcel_core::array_shape(source) != mlxcel_core::array_shape(x) {
            bail!("Source and target latent shapes must match");
        }
        let time = self.timestep(&mlxcel_core::abs(t));
        let clean_time = self.timestep(&mlxcel_core::zeros_like(t));
        let captions = pad_tokens(&self.cap_embedder.forward(caption), &self.cap_pad_token);
        let cap_len = mlxcel_core::array_shape(caption)[1];
        let padded_cap = mlxcel_core::array_shape(&captions)[1];
        let mut cap_pos = sequence_positions(cap_len, 1);
        cap_pos.extend(sequence_positions(cap_len, cap_len + 3));
        let captions = concat(&[&captions, &captions]);

        let source_tokens = self.embed_image(source);
        let target_tokens = self.embed_image(x);
        let image_len = mlxcel_core::array_shape(&source_tokens)[1];
        let mut images = concat(&[&source_tokens, &target_tokens]);
        let mut image_pos = image_positions(h, w, cap_len + 1);
        image_pos.extend(image_positions(h, w, 2 * cap_len + 3));
        let image_noise = {
            let mut v = Vec::with_capacity((b * 2 * image_len) as usize);
            for _ in 0..b {
                v.extend(std::iter::repeat_n(false, image_len as usize));
                v.extend(std::iter::repeat_n(true, image_len as usize));
            }
            bool_array(&v, &[b, 2 * image_len])
        };
        let (image_cos, image_sin) = self.rope.compute_freqs(&positions_array(&image_pos))?;
        for block in &self.noise_refiner {
            images = block.forward_token_modulated(
                &images,
                &image_cos,
                &image_sin,
                &time,
                &clean_time,
                &image_noise,
                None,
            )?;
        }
        let (cap_cos, cap_sin) = self.rope.compute_freqs(&positions_array(&cap_pos))?;
        let mut captions = captions;
        for block in &self.context_refiner {
            captions = block.forward(&captions, &cap_cos, &cap_sin, None, None)?;
        }
        let cap_noise = {
            let mut v = Vec::with_capacity((b * 2 * padded_cap) as usize);
            for _ in 0..b {
                v.extend(std::iter::repeat_n(false, padded_cap as usize));
                v.extend(std::iter::repeat_n(true, padded_cap as usize));
            }
            v
        };
        let total_caption = 2 * padded_cap;
        let mut token_parts = vec![captions, images];
        let mut positions = cap_pos;
        positions.extend(image_pos);
        // Per-row noise flags: [captions, images, features].
        let mut noise_rows: Vec<Vec<bool>> = (0..b as usize)
            .map(|row| {
                let mut v = cap_noise
                    [row * total_caption as usize..(row + 1) * total_caption as usize]
                    .to_vec();
                v.extend(std::iter::repeat_n(false, image_len as usize));
                v.extend(std::iter::repeat_n(true, image_len as usize));
                v
            })
            .collect();
        let n = mlxcel_core::array_shape(semantic)[1];
        // CFG drops the semantic sequence altogether on the unconditional branch.
        if n > 0 {
            let mut features = pad_tokens(
                &self.sigvq_embedder.forward(semantic),
                &self.sigvq_pad_token,
            );
            let feature_pos = sequence_positions(n, total_caption + 2 * image_len + 1);
            let (fc, fs) = self.rope.compute_freqs(&positions_array(&feature_pos))?;
            for block in &self.sigvq_refiner {
                features = block.forward(&features, &fc, &fs, None, None)?;
            }
            let feature_len = mlxcel_core::array_shape(&features)[1];
            for row in &mut noise_rows {
                row.extend(std::iter::repeat_n(false, feature_len as usize));
            }
            positions.extend(feature_pos);
            token_parts.push(features);
        }
        let refs: Vec<&MlxArray> = token_parts.iter().map(|p| &**p).collect();
        let mut tokens = concat(&refs);
        let total = mlxcel_core::array_shape(&tokens)[1];
        let noise = bool_array(&noise_rows.concat(), &[b, total]);
        let (cos, sin) = self.rope.compute_freqs(&positions_array(&positions))?;
        for block in &self.layers {
            tokens = block.forward_token_modulated(
                &tokens,
                &cos,
                &sin,
                &time,
                &clean_time,
                &noise,
                None,
            )?;
        }
        // Only the noisy target image is decoded, so its final modulation is uniform.
        let start = total_caption + image_len;
        Ok(self.final_output(&tokens, start, &time, dims))
    }
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
