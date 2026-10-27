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

//! The Z-Image single-stream DiT.
//!
//! A port of `ZImageTransformer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/transformer.py>
//! (`#L193-L372`), assembled from the shared [`crate::imagegen::dit_blocks`]:
//! patchify, padding of both streams to a multiple of 32 tokens, 3-axis
//! positions, a noise refiner over the image tokens, a context refiner over
//! the caption tokens, the joint `layers`, and the final layer over the image
//! tokens only.
//!
//! Dtypes follow upstream: f32 latents and an f32 timestep keep the image
//! stream and the joint stream in f32 against half-precision weights.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::ZImageTransformerConfig;
use crate::imagegen::dit_blocks::{
    FinalLayer, MRoPE, RopePrecision, TimestepEmbedder, ZImageTransformerBlock,
};

/// Both token streams are padded to a multiple of this length.
pub const SEQ_MULTI_OF: i32 = 32;
/// Width of the sinusoidal timestep features.
const TIMESTEP_FREQUENCY_SIZE: i32 = 256;

fn tensor(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("Z-Image transformer weight not found: {key}"))
}

fn linear(weights: &WeightMap, prefix: &str) -> Result<Linear> {
    Linear::from_weights(weights, prefix)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Z-Image transformer weight {prefix}"))
}

/// `[B, L, D] -> [B, L + pad, D]`, repeating the last token.
fn pad_repeat_last(x: &MlxArray, pad: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (b, l, d) = (shape[0], shape[1], shape[2]);
    let last = mlxcel_core::slice(x, &[0, l - 1, 0], &[b, l, d]);
    let repeated = mlxcel_core::repeat(&last, pad, 1);
    mlxcel_core::concatenate(x, &repeated, 1)
}

/// Keep the first `len` tokens of `x` and append `pad` copies of `token` (`[1, D]`).
fn replace_tail(x: &MlxArray, len: i32, pad: i32, token: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (b, d) = (shape[0], shape[2]);
    let head = mlxcel_core::slice(x, &[0, 0, 0], &[b, len, d]);
    let tail = mlxcel_core::broadcast_to(&mlxcel_core::reshape(token, &[1, 1, d]), &[b, pad, d]);
    mlxcel_core::concatenate(&head, &tail, 1)
}

/// `(cap_pos, img_pos)`, each `[L, 3]` flattened: caption `t = 1..=cap_len`
/// on axis 0, image `(f + cap_len + 1, h, w)` in `f, h, w` order.
#[must_use]
pub fn build_position_ids(cap_len: i32, ft: i32, ht: i32, wt: i32) -> (Vec<f32>, Vec<f32>) {
    let cap = (1..=cap_len).flat_map(|t| [t as f32, 0.0, 0.0]).collect();
    let start = cap_len + 1;
    let mut img = Vec::with_capacity((ft * ht * wt * 3) as usize);
    for f in 0..ft {
        for h in 0..ht {
            for w in 0..wt {
                img.extend([(f + start) as f32, h as f32, w as f32]);
            }
        }
    }
    (cap, img)
}

/// The top-level transformer.
pub struct ZImageTransformer {
    config: ZImageTransformerConfig,
    t_embedder: TimestepEmbedder,
    cap_norm: RMSNorm,
    /// `cap_embedder.1`: weight transposed to `[in, out]`, and its bias.
    cap_weight_t: UniquePtr<MlxArray>,
    cap_bias: UniquePtr<MlxArray>,
    x_embedder: Linear,
    x_pad_token: UniquePtr<MlxArray>,
    cap_pad_token: UniquePtr<MlxArray>,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    final_layer: FinalLayer,
    rope: MRoPE,
}

impl ZImageTransformer {
    /// Build from weights already passed through `sanitize_transformer_weights`.
    pub fn from_weights(config: ZImageTransformerConfig, weights: &WeightMap) -> Result<Self> {
        config.validate()?;
        let heads = config.num_attention_heads;
        let head_dim = config.head_dim();
        let rope = RopePrecision::KeepPromoted;
        let blocks = |name: &str, count: i32, modulation: bool| {
            (0..count)
                .map(|i| {
                    ZImageTransformerBlock::load(
                        weights,
                        &format!("{name}.{i}"),
                        heads,
                        head_dim,
                        modulation,
                        rope,
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            t_embedder: TimestepEmbedder::load(weights, "t_embedder", TIMESTEP_FREQUENCY_SIZE)?,
            cap_norm: RMSNorm::new(tensor(weights, "cap_embedder.0.weight")?, config.norm_eps),
            cap_weight_t: {
                let weight = tensor(weights, "cap_embedder.1.weight")?;
                mlxcel_core::transpose(&weight)
            },
            cap_bias: tensor(weights, "cap_embedder.1.bias")?,
            x_embedder: linear(weights, "x_embedder")?,
            x_pad_token: tensor(weights, "x_pad_token")?,
            cap_pad_token: tensor(weights, "cap_pad_token")?,
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
            bail!("Z-Image latents must be [B, C, F, H, W], got {shape:?}");
        }
        let (b, c, f, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (p, pf) = (self.config.patch_size, self.config.f_patch_size);
        if c != self.config.in_channels || f % pf != 0 || h % p != 0 || w % p != 0 {
            bail!("Z-Image latents {shape:?} do not match the patch grid");
        }
        let (ft, ht, wt) = (f / pf, h / p, w / p);
        let x = mlxcel_core::reshape(x, &[b, c, ft, pf, ht, p, wt, p]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 6, 3, 5, 7, 1]);
        let x = mlxcel_core::reshape(&x, &[b, ft * ht * wt, pf * p * p * c]);
        Ok((x, (ft, ht, wt)))
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

    /// `cap_embedder`: RMSNorm, then `addmm(bias, x, w.T)`, both in the
    /// checkpoint's dtype like upstream's `nn.Linear` (a separate matmul and
    /// bias add rounds twice and measurably drifts from it).
    fn embed_caption(&self, cap_feats: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.cap_norm.forward(cap_feats);
        mlxcel_core::addmm(&self.cap_bias, &normed, &self.cap_weight_t, 1.0, 1.0)
    }

    /// Predict the flow for latents `x` (`[B, C, F, H, W]`) at time `t` (`[B]`
    /// or `[1]`, `1 - sigma`) given caption features `cap_feats` (`[1 or B, L,
    /// text_embed_dim]`).
    pub fn forward(
        &self,
        x: &MlxArray,
        t: &MlxArray,
        cap_feats: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let dim = self.config.hidden_size;
        let batch = mlxcel_core::array_shape(x)[0];
        let (mut patches, grid) = self.patchify(x)?;
        let (ft, ht, wt) = grid;
        let image_len = ft * ht * wt;
        let image_padding = (-image_len).rem_euclid(SEQ_MULTI_OF);
        if image_padding > 0 {
            patches = pad_repeat_last(&patches, image_padding);
        }
        let cap_shape = mlxcel_core::array_shape(cap_feats);
        if cap_shape.len() != 3 || cap_shape[2] != self.config.text_embed_dim || cap_shape[1] < 1 {
            bail!(
                "Z-Image caption features must be [B, L, {}], got {cap_shape:?}",
                self.config.text_embed_dim
            );
        }
        let cap_len = cap_shape[1];
        let cap_padding = (-cap_len).rem_euclid(SEQ_MULTI_OF);
        let mut cap_feats = mlxcel_core::copy(cap_feats);
        if cap_padding > 0 {
            cap_feats = pad_repeat_last(&cap_feats, cap_padding);
        }
        let padded_cap_len = cap_len + cap_padding;

        let t_scale = mlxcel_core::full_f32(
            &[1],
            self.config.timestep_scale,
            mlxcel_core::array_dtype(t),
        );
        let t_emb = self.t_embedder.forward(&mlxcel_core::multiply(t, &t_scale));

        let mut img = self.x_embedder.forward(&patches);
        if image_padding > 0 {
            img = replace_tail(&img, image_len, image_padding, &self.x_pad_token);
        }
        let mut cap = self.embed_caption(&cap_feats);
        if cap_padding > 0 {
            let token = mlxcel_core::astype(&self.cap_pad_token, mlxcel_core::array_dtype(&cap));
            cap = replace_tail(&cap, cap_len, cap_padding, &token);
        }

        let (cap_pos, mut img_pos) = build_position_ids(padded_cap_len, ft, ht, wt);
        img_pos.resize(img_pos.len() + 3 * image_padding as usize, 0.0);
        let cap_pos = mlxcel_core::from_slice_f32(&cap_pos, &[1, padded_cap_len, 3]);
        let img_pos = mlxcel_core::from_slice_f32(&img_pos, &[1, image_len + image_padding, 3]);
        let (img_cos, img_sin) = self.rope.compute_freqs(&img_pos)?;
        let (cap_cos, cap_sin) = self.rope.compute_freqs(&cap_pos)?;

        for block in &self.noise_refiner {
            img = block.forward(&img, &img_cos, &img_sin, Some(&t_emb), None)?;
        }
        for block in &self.context_refiner {
            cap = block.forward(&cap, &cap_cos, &cap_sin, None, None)?;
        }

        let cap_dims = mlxcel_core::array_shape(&cap);
        if cap_dims[0] != batch {
            cap = mlxcel_core::broadcast_to(&cap, &[batch, cap_dims[1], cap_dims[2]]);
        }
        let mut unified = mlxcel_core::concatenate(&img, &cap, 1);
        let cos = mlxcel_core::concatenate(&img_cos, &cap_cos, 1);
        let sin = mlxcel_core::concatenate(&img_sin, &cap_sin, 1);
        for block in &self.layers {
            unified = block.forward(&unified, &cos, &sin, Some(&t_emb), None)?;
        }

        let image_tokens = mlxcel_core::slice(&unified, &[0, 0, 0], &[batch, image_len, dim]);
        let out = self.final_layer.forward(&image_tokens, &t_emb);
        Ok(self.unpatchify(&out, grid))
    }
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
