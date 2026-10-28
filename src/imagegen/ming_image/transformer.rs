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

//! The Ming-Image NextDiT, a single-stream Lumina-2.0-style transformer.
//!
//! A port of `MingImageTransformer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ming_image/transformer.py>
//! (mlx-vlm `73d366ec`), assembled from the shared
//! [`crate::imagegen::dit_blocks`] like [`crate::imagegen::z_image`]'s, with
//! three differences from Z-Image:
//!
//! * `cap_feats_2` (already at the model width) is appended to the embedded
//!   `cap_feats` before the context refiner;
//! * both streams are padded to a multiple of [`SEQ_MULTIPLE`] with zeros
//!   (`alignment_padding_mode: "zero_masked"`) instead of learned pad
//!   tokens, and every attention masks the padded keys: the noise refiner
//!   over the image stream, the context refiner over the caption stream, the
//!   joint layers over both;
//! * the pipeline feeds bf16 latents and a bf16 timestep, so the timestep
//!   embedding, every modulation and the first attention inputs are bf16; the
//!   stream turns f32 at the first attention output (f32 RoPE tables).
//!
//! Positions: caption tokens `t = 1..=padded_cap_len`, image tokens
//! `(f + padded_cap_len + 1, h, w)`, padded image tokens at `(0, 0, 0)`.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use crate::imagegen::dit_blocks::{
    FinalLayer, MRoPE, RopePrecision, TimestepEmbedder, ZImageTransformerBlock,
};
use crate::imagegen::z_image::config::ZImageTransformerConfig;
use crate::imagegen::z_image::transformer::build_position_ids;

/// Both token streams are padded to a multiple of this length.
pub const SEQ_MULTIPLE: i32 = 32;
/// Width of the sinusoidal timestep features.
const TIMESTEP_FREQUENCY_SIZE: i32 = 256;

fn tensor(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("Ming-Image transformer weight not found: {key}"))
}

fn round_up(value: i32) -> i32 {
    (value + SEQ_MULTIPLE - 1) / SEQ_MULTIPLE * SEQ_MULTIPLE
}

/// `[B, L, D] -> [B, target, D]`, zero-padded (upstream `_pad_sequence`).
fn pad_zeros(x: &MlxArray, target: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    if s[1] == target {
        return mlxcel_core::copy(x);
    }
    let zeros = mlxcel_core::zeros(&[s[0], target - s[1], s[2]], mlxcel_core::array_dtype(x));
    mlxcel_core::concatenate(x, &zeros, 1)
}

/// Boolean `[1, 1, 1, total]` keep-mask over keys: the first `valid` of each
/// `(valid, total)` segment, concatenated; `None` when nothing is padded.
#[must_use]
pub fn key_mask(segments: &[(i32, i32)]) -> Option<UniquePtr<MlxArray>> {
    if segments.iter().all(|(v, t)| v == t) {
        return None;
    }
    let keep: Vec<i32> = segments
        .iter()
        .flat_map(|&(valid, total)| (0..total).map(move |i| i32::from(i < valid)))
        .collect();
    let len = keep.len() as i32;
    let mask = mlxcel_core::from_slice_i32(&keep, &[1, 1, 1, len]);
    Some(mlxcel_core::astype(&mask, dtype::BOOL))
}

/// The top-level transformer.
pub struct MingImageTransformer {
    config: ZImageTransformerConfig,
    t_embedder: TimestepEmbedder,
    cap_norm: RMSNorm,
    cap_linear: Linear,
    x_embedder: Linear,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    final_layer: FinalLayer,
    rope: MRoPE,
}

impl MingImageTransformer {
    /// Build from weights in the block layout (see [`super::weights::load_transformer`]).
    pub fn from_weights(config: ZImageTransformerConfig, weights: &WeightMap) -> Result<Self> {
        config.validate()?;
        let heads = config.num_attention_heads;
        let head_dim = config.head_dim();
        let blocks = |name: &str, count: i32, modulation: bool| {
            (0..count)
                .map(|i| {
                    ZImageTransformerBlock::load(
                        weights,
                        &format!("{name}.{i}"),
                        heads,
                        head_dim,
                        modulation,
                        RopePrecision::KeepPromoted,
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        let linear = |prefix: &str| {
            Linear::from_weights(weights, prefix)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("Ming-Image transformer weight {prefix}"))
        };
        Ok(Self {
            t_embedder: TimestepEmbedder::load(weights, "t_embedder", TIMESTEP_FREQUENCY_SIZE)?,
            cap_norm: RMSNorm::new(tensor(weights, "cap_embedder.0.weight")?, config.norm_eps),
            cap_linear: linear("cap_embedder.1")?,
            x_embedder: linear("x_embedder")?,
            noise_refiner: blocks("noise_refiner", config.n_refiner_layers, true)?,
            context_refiner: blocks("context_refiner", config.n_context_refiner_layers, false)?,
            layers: blocks("layers", config.num_hidden_layers, true)?,
            final_layer: FinalLayer::load(weights, "final_layer")?,
            rope: MRoPE {
                sections: config.rope_sections.clone(),
                theta: config.rope_theta,
            },
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &ZImageTransformerConfig {
        &self.config
    }

    /// `[B, C, F, H, W] -> ([B, Ft*Ht*Wt, pF*pH*pW*C], (Ft, Ht, Wt))`.
    fn patchify(&self, x: &MlxArray) -> Result<(UniquePtr<MlxArray>, (i32, i32, i32))> {
        let shape = mlxcel_core::array_shape(x);
        if shape.len() != 5 {
            bail!("Ming-Image latents must be [B, C, F, H, W], got {shape:?}");
        }
        let (b, c, f, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (p, pf) = (self.config.patch_size, self.config.f_patch_size);
        if c != self.config.in_channels || f % pf != 0 || h % p != 0 || w % p != 0 {
            bail!("Ming-Image latents {shape:?} do not match the patch grid");
        }
        let (ft, ht, wt) = (f / pf, h / p, w / p);
        let x = mlxcel_core::reshape(x, &[b, c, ft, pf, ht, p, wt, p]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 6, 3, 5, 7, 1]);
        Ok((
            mlxcel_core::reshape(&x, &[b, ft * ht * wt, pf * p * p * c]),
            (ft, ht, wt),
        ))
    }

    fn unpatchify(&self, x: &MlxArray, (ft, ht, wt): (i32, i32, i32)) -> UniquePtr<MlxArray> {
        let b = mlxcel_core::array_shape(x)[0];
        let (p, pf, c) = (
            self.config.patch_size,
            self.config.f_patch_size,
            self.config.in_channels,
        );
        let x = mlxcel_core::reshape(x, &[b, ft, ht, wt, pf, p, p, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 7, 1, 4, 2, 5, 3, 6]);
        mlxcel_core::reshape(&x, &[b, c, ft * pf, ht * p, wt * p])
    }

    /// One denoising forward: latents `x` (`[B, C, F, H, W]`), time `t`
    /// (`[1]`, `1 - sigma`), caption features `cap_feats` (`[1, L, cap_feat_dim]`)
    /// and optional `cap_feats_2` (`[1, L2, dim]`), shared across the batch.
    pub fn forward(
        &self,
        x: &MlxArray,
        t: &MlxArray,
        cap_feats: &MlxArray,
        cap_feats_2: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let dim = self.config.hidden_size;
        let (patches, grid) = self.patchify(x)?;
        let (ft, ht, wt) = grid;
        let batch = mlxcel_core::array_shape(&patches)[0];
        let image_len = ft * ht * wt;
        let padded_image_len = round_up(image_len);

        let cap_shape = mlxcel_core::array_shape(cap_feats);
        if cap_shape.len() != 3 || cap_shape[2] != self.config.text_embed_dim || cap_shape[1] < 1 {
            bail!(
                "Ming-Image caption features must be [B, L, {}], got {cap_shape:?}",
                self.config.text_embed_dim
            );
        }
        let mut cap = self.cap_linear.forward(&self.cap_norm.forward(cap_feats));
        if let Some(extra) = cap_feats_2 {
            let s = mlxcel_core::array_shape(extra);
            if s.len() != 3 || s[0] != cap_shape[0] || s[2] != dim {
                bail!(
                    "Ming-Image cap_feats_2 must be [{}, L, {dim}], got {s:?}",
                    cap_shape[0]
                );
            }
            let extra = mlxcel_core::astype(extra, mlxcel_core::array_dtype(&cap));
            cap = mlxcel_core::concatenate(&cap, &extra, 1);
        }
        let cap_len = mlxcel_core::array_shape(&cap)[1];
        let padded_cap_len = round_up(cap_len);
        let mut cap = pad_zeros(&cap, padded_cap_len);

        let mut img = pad_zeros(&self.x_embedder.forward(&patches), padded_image_len);
        let t_scale = mlxcel_core::full_f32(
            &[1],
            self.config.timestep_scale,
            mlxcel_core::array_dtype(t),
        );
        let t_emb = self.t_embedder.forward(&mlxcel_core::multiply(t, &t_scale));
        let t_emb = mlxcel_core::astype(&t_emb, mlxcel_core::array_dtype(&img));

        let (cap_pos, mut img_pos) = build_position_ids(padded_cap_len, ft, ht, wt);
        img_pos.resize(3 * padded_image_len as usize, 0.0);
        let cap_pos = mlxcel_core::from_slice_f32(&cap_pos, &[1, padded_cap_len, 3]);
        let img_pos = mlxcel_core::from_slice_f32(&img_pos, &[1, padded_image_len, 3]);
        let (img_cos, img_sin) = self.rope.compute_freqs(&img_pos)?;
        let (cap_cos, cap_sin) = self.rope.compute_freqs(&cap_pos)?;

        let img_mask = key_mask(&[(image_len, padded_image_len)]);
        for block in &self.noise_refiner {
            img = block.forward(&img, &img_cos, &img_sin, Some(&t_emb), img_mask.as_deref())?;
        }
        let cap_mask = key_mask(&[(cap_len, padded_cap_len)]);
        for block in &self.context_refiner {
            cap = block.forward(&cap, &cap_cos, &cap_sin, None, cap_mask.as_deref())?;
        }

        let cap_dims = mlxcel_core::array_shape(&cap);
        if cap_dims[0] != batch {
            cap = mlxcel_core::broadcast_to(&cap, &[batch, cap_dims[1], cap_dims[2]]);
        }
        let mut unified = mlxcel_core::concatenate(&img, &cap, 1);
        let cos = mlxcel_core::concatenate(&img_cos, &cap_cos, 1);
        let sin = mlxcel_core::concatenate(&img_sin, &cap_sin, 1);
        let mask = key_mask(&[(image_len, padded_image_len), (cap_len, padded_cap_len)]);
        for block in &self.layers {
            unified = block.forward(&unified, &cos, &sin, Some(&t_emb), mask.as_deref())?;
        }

        let image_tokens = mlxcel_core::slice(&unified, &[0, 0, 0], &[batch, image_len, dim]);
        let out = self.final_layer.forward(&image_tokens, &t_emb);
        Ok(self.unpatchify(&out, grid))
    }
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
