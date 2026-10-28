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

//! The Mage-VAE decoder: `CoDDecoder` (latent to a 384-channel condition at
//! latent resolution) and `DConvDenoiser` (condition to pixels, one 16x16
//! patch per latent cell, decoded by a per-pixel MLP), from `vae.py`
//! (mlx-vlm `73d366ec`).
//!
//! The decoder denoises from zero noise at `t = 0`, so the noise image and the
//! `s_embedder.proj1` conv of it (no bias) are exact zeros; this port builds
//! the zeros directly instead of running the conv.

use anyhow::Result;
use mlxcel_core::layers::RMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::blocks::{DiCoBlock, EPS, TimestepEmbedder};
use crate::imagegen::mage_flow::layers::{Dense, LayerNorm, modulate, rms_norm, silu, split_last};
use crate::imagegen::vae::{Conv2d, GroupNorm};

/// Hidden width of the decoder.
pub(super) const CHANNELS: i32 = 384;
/// Pixels per latent cell along each axis.
pub(super) const PATCH: i32 = 16;
/// Per-pixel feature width of the decoder MLP.
const PIXEL_DIM: i32 = 32;
/// Side of a local-attention window, in latent cells.
const ATTENTION_PATCH: i32 = 32;
const GROUPS: i32 = 32;
const BLOCKS: i32 = 21;

/// `ResnetBlock`: two GroupNorm-SiLU-conv stages and a residual.
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
}

impl ResnetBlock {
    fn load(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::load(w, &format!("{p}.norm1"), GROUPS, EPS)?,
            conv1: Conv2d::load(w, &format!("{p}.conv1"), 1, 1)?,
            norm2: GroupNorm::load(w, &format!("{p}.norm2"), GROUPS, EPS)?,
            conv2: Conv2d::load(w, &format!("{p}.conv2"), 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.conv1.forward(&silu(&self.norm1.forward_native(x)));
        let h = self.conv2.forward(&silu(&self.norm2.forward_native(&h)));
        mlxcel_core::add(x, &h)
    }
}

/// Pad `x` (NHWC) on the bottom and right by repeating its last row and column
/// (`mx.pad(mode="edge")`).
pub(super) fn edge_pad(x: &MlxArray, pad_h: i32, pad_w: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let mut x = mlxcel_core::copy(x);
    if pad_h > 0 {
        let last = mlxcel_core::slice(&x, &[0, s[1] - 1, 0, 0], &[s[0], s[1], s[2], s[3]]);
        let rep = mlxcel_core::broadcast_to(&last, &[s[0], pad_h, s[2], s[3]]);
        x = mlxcel_core::concatenate(&x, &rep, 1);
    }
    if pad_w > 0 {
        let h = mlxcel_core::array_shape(&x)[1];
        let last = mlxcel_core::slice(&x, &[0, 0, s[2] - 1, 0], &[s[0], h, s[2], s[3]]);
        let rep = mlxcel_core::broadcast_to(&last, &[s[0], h, pad_w, s[3]]);
        x = mlxcel_core::concatenate(&x, &rep, 2);
    }
    x
}

/// `LocalAttentionBlock`: single-head attention over all channels inside
/// each 32x32 window of the latent grid, scores and softmax in f32.
///
/// A window holds at most 1024 tokens, so the f32 score tensor is
/// `windows * 4 MB` (64 MB at 2048x2048); no query chunking is needed.
pub(super) struct LocalAttentionBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

/// Geometry of [`LocalAttentionBlock::patchify`].
pub(super) struct Windows {
    batch: i32,
    height: i32,
    width: i32,
    rows: i32,
    cols: i32,
}

impl LocalAttentionBlock {
    fn load(w: &WeightMap, p: &str) -> Result<Self> {
        let conv = |name: &str| Conv2d::load(w, &format!("{p}.{name}"), 1, 0);
        Ok(Self {
            norm: GroupNorm::load(w, &format!("{p}.norm"), GROUPS, EPS)?,
            q: conv("q")?,
            k: conv("k")?,
            v: conv("v")?,
            proj_out: conv("proj_out")?,
        })
    }

    pub(super) fn patchify(x: &MlxArray) -> (UniquePtr<MlxArray>, Windows) {
        let s = mlxcel_core::array_shape(x);
        let (batch, height, width, channels) = (s[0], s[1], s[2], s[3]);
        let p = ATTENTION_PATCH;
        let ceil = |v: i32| (v + p - 1) / p * p;
        let (padded_h, padded_w) = (ceil(height), ceil(width));
        let x = edge_pad(x, padded_h - height, padded_w - width);
        let (rows, cols) = (padded_h / p, padded_w / p);
        let x = mlxcel_core::reshape(&x, &[batch, rows, p, cols, p, channels]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
        let x = mlxcel_core::reshape(&x, &[batch * rows * cols, p * p, channels]);
        (
            x,
            Windows {
                batch,
                height,
                width,
                rows,
                cols,
            },
        )
    }

    pub(super) fn unpatchify(x: &MlxArray, g: &Windows) -> UniquePtr<MlxArray> {
        let p = ATTENTION_PATCH;
        let channels = *mlxcel_core::array_shape(x).last().expect("rank 3");
        let x = mlxcel_core::reshape(x, &[g.batch, g.rows, g.cols, p, p, channels]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
        let x = mlxcel_core::reshape(&x, &[g.batch, g.rows * p, g.cols * p, channels]);
        mlxcel_core::slice(&x, &[0, 0, 0, 0], &[g.batch, g.height, g.width, channels])
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.norm.forward_native(x);
        let (query, geometry) = Self::patchify(&self.q.forward(&hidden));
        let (key, _) = Self::patchify(&self.k.forward(&hidden));
        let (value, _) = Self::patchify(&self.v.forward(&hidden));
        let f32 = |a: &MlxArray| mlxcel_core::astype(a, dtype::FLOAT32);
        let scores = mlxcel_core::matmul(
            &f32(&query),
            &mlxcel_core::transpose_axes(&f32(&key), &[0, 2, 1]),
        );
        // `channels**-0.5` is a Python double rounded once to f32.
        let scale =
            mlxcel_core::full_f32(&[1], f64::from(CHANNELS).powf(-0.5) as f32, dtype::FLOAT32);
        let scores = mlxcel_core::softmax(&mlxcel_core::multiply(&scores, &scale), -1);
        let attended = mlxcel_core::astype(
            &mlxcel_core::matmul(&scores, &f32(&value)),
            mlxcel_core::array_dtype(&value),
        );
        let hidden = Self::unpatchify(&attended, &geometry);
        mlxcel_core::add(x, &self.proj_out.forward(&hidden))
    }
}

/// One decoder block: a resnet or a local-attention block.
enum DecoderBlock {
    Resnet(ResnetBlock),
    Attention(LocalAttentionBlock),
}

/// `CoDDecoder`: latent `[B, h, w, 128]` to the condition `[B, h, w, 384]`.
pub(super) struct CoDDecoder {
    conv_in: Conv2d,
    blocks: Vec<DecoderBlock>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl CoDDecoder {
    pub(super) fn load(w: &WeightMap, p: &str) -> Result<Self> {
        let blocks = (0..5)
            .map(|i| {
                let prefix = format!("{p}.block.{i}");
                Ok(if i % 2 == 0 {
                    DecoderBlock::Resnet(ResnetBlock::load(w, &prefix)?)
                } else {
                    DecoderBlock::Attention(LocalAttentionBlock::load(w, &prefix)?)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: Conv2d::load(w, &format!("{p}.conv_in"), 1, 1)?,
            blocks,
            norm_out: GroupNorm::load(w, &format!("{p}.norm_out"), GROUPS, EPS)?,
            conv_out: Conv2d::load(w, &format!("{p}.conv_out"), 1, 1)?,
        })
    }

    pub(super) fn forward(&self, latent: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.conv_in.forward(latent);
        for block in &self.blocks {
            h = match block {
                DecoderBlock::Resnet(b) => b.forward(&h),
                DecoderBlock::Attention(b) => b.forward(&h),
            };
        }
        self.conv_out
            .forward(&silu(&self.norm_out.forward_native(&h)))
    }
}

/// The NeRF-style DCT basis of one `PATCH x PATCH` patch, `[1, 256, 64]` f32.
fn dct_basis() -> UniquePtr<MlxArray> {
    const FREQS: i32 = 8;
    let f32_scalar = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
    let positions = mlxcel_core::linspace(0.0, 1.0, PATCH, dtype::FLOAT32);
    // `meshgrid(p, p, indexing="ij")`: pos_y varies by row, pos_x by column.
    let pos_y = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&positions, &[PATCH, 1]),
        &[PATCH, PATCH],
    );
    let pos_x = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&positions, &[1, PATCH]),
        &[PATCH, PATCH],
    );
    let pos_x = mlxcel_core::reshape(&pos_x, &[-1, 1, 1]);
    let pos_y = mlxcel_core::reshape(&pos_y, &[-1, 1, 1]);
    let freqs = mlxcel_core::linspace(0.0, FREQS as f32, FREQS, dtype::FLOAT32);
    let fx = mlxcel_core::reshape(&freqs, &[1, FREQS, 1]);
    let fy = mlxcel_core::reshape(&freqs, &[1, 1, FREQS]);
    let one = f32_scalar(1.0);
    let coefficients = mlxcel_core::divide(
        &one,
        &mlxcel_core::add(&one, &mlxcel_core::multiply(&fx, &fy)),
    );
    let pi = f32_scalar(std::f32::consts::PI);
    let cos_x = mlxcel_core::cos(&mlxcel_core::multiply(
        &mlxcel_core::multiply(&pi, &pos_x),
        &fx,
    ));
    let cos_y = mlxcel_core::cos(&mlxcel_core::multiply(
        &mlxcel_core::multiply(&pi, &pos_y),
        &fy,
    ));
    let dct = mlxcel_core::multiply(&mlxcel_core::multiply(&cos_x, &cos_y), &coefficients);
    mlxcel_core::reshape(&dct, &[1, PATCH * PATCH, FREQS * FREQS])
}

/// `MLPResBlock`: AdaLN-modulated two-layer SiLU MLP per pixel.
struct MlpResBlock {
    in_ln: LayerNorm,
    linear_1: Dense,
    linear_2: Dense,
    ada_ln: Dense,
}

impl MlpResBlock {
    fn load(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            in_ln: LayerNorm::affine(w, &format!("{p}.in_ln"), EPS)?,
            linear_1: Dense::load(w, &format!("{p}.linear_1"), None)?,
            linear_2: Dense::load(w, &format!("{p}.linear_2"), None)?,
            ada_ln: Dense::load(w, &format!("{p}.adaLN_modulation.linear"), None)?,
        })
    }

    fn forward(&self, x: &MlxArray, condition: &MlxArray) -> UniquePtr<MlxArray> {
        let m = split_last(&self.ada_ln.forward(&mlxcel_core::silu(condition)), 3);
        let hidden = modulate(&self.in_ln.forward(x), &m[0], &m[1]);
        let hidden = self
            .linear_2
            .forward(&mlxcel_core::silu(&self.linear_1.forward(&hidden)));
        mlxcel_core::add(x, &mlxcel_core::multiply(&m[2], &hidden))
    }
}

/// `DConvDenoiser`: the condition `[B, h, w, 384]` to pixels `[B, 16h, 16w, 3]`.
pub(super) struct DConvDenoiser {
    t_embedder: TimestepEmbedder,
    y_embedder_x: Conv2d,
    x_embedder: Dense,
    s_embedder_proj2: Conv2d,
    blocks: Vec<DiCoBlock>,
    cond_embed: Dense,
    input_proj: Dense,
    res_blocks: Vec<MlpResBlock>,
    final_norm: RMSNorm,
    final_linear: Dense,
}

impl DConvDenoiser {
    pub(super) fn load(w: &WeightMap, p: &str) -> Result<Self> {
        let blocks = (0..BLOCKS)
            .map(|i| DiCoBlock::load(w, &format!("{p}.blocks.{i}"), CHANNELS))
            .collect::<Result<Vec<_>>>()?;
        let res_blocks = (0..3)
            .map(|i| MlpResBlock::load(w, &format!("{p}.dec_net.res_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            t_embedder: TimestepEmbedder::load(w, &format!("{p}.t_embedder"))?,
            y_embedder_x: Conv2d::load(w, &format!("{p}.y_embedder_x"), 1, 0)?,
            x_embedder: Dense::load(w, &format!("{p}.x_embedder.linear"), None)?,
            s_embedder_proj2: Conv2d::load(w, &format!("{p}.s_embedder.proj2"), 1, 0)?,
            blocks,
            cond_embed: Dense::load(w, &format!("{p}.dec_net.cond_embed"), None)?,
            input_proj: Dense::load(w, &format!("{p}.dec_net.input_proj"), None)?,
            res_blocks,
            final_norm: rms_norm(w, &format!("{p}.final_layer.norm.weight"), EPS)?,
            final_linear: Dense::load(w, &format!("{p}.final_layer.linear"), None)?,
        })
    }

    pub(super) fn forward(&self, condition: &MlxArray) -> UniquePtr<MlxArray> {
        let mut spatial = self.spatial_input(condition);
        let temb = self.temb(condition);
        for block in &self.blocks {
            spatial = block.forward(&spatial, &temb);
        }
        self.pixel_head(condition, &spatial)
    }

    /// The `t = 0` embedding the DiCo blocks read.
    pub(super) fn temb(&self, condition: &MlxArray) -> UniquePtr<MlxArray> {
        let batch = mlxcel_core::array_shape(condition)[0];
        self.t_embedder
            .zero(batch, mlxcel_core::array_dtype(condition))
    }

    /// The input of the first DiCo block: `s_embedder` of the zero noise
    /// image (`proj1` has no bias, so its output is exactly zero) and the
    /// condition.
    pub(super) fn spatial_input(&self, condition: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(condition);
        let noise_embed = mlxcel_core::zeros(
            &[s[0], s[1], s[2], 128],
            mlxcel_core::array_dtype(condition),
        );
        self.s_embedder_proj2
            .forward(&mlxcel_core::concatenate(&noise_embed, condition, -1))
    }

    /// DiCo block `index` (tests feed it upstream's own input).
    #[cfg(test)]
    pub(super) fn block(&self, index: usize) -> &DiCoBlock {
        &self.blocks[index]
    }

    /// Everything after the DiCo blocks: the per-pixel decoder of each
    /// 16x16 patch, from the condition and the blocks' output `spatial`.
    pub(super) fn pixel_head(
        &self,
        condition: &MlxArray,
        spatial: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(condition);
        let (batch, gh, gw) = (s[0], s[1], s[2]);
        let cells = gh * gw;
        let dt = mlxcel_core::array_dtype(condition);
        let spatial = mlxcel_core::reshape(spatial, &[batch * cells, CHANNELS]);

        let pixels = PATCH * PATCH;
        // The zero noise image, patchified: still zero.
        let noise_patches = mlxcel_core::zeros(&[batch, cells, pixels, 3], dt);
        let cond_patches = mlxcel_core::reshape(
            &self.y_embedder_x.forward(condition),
            &[batch, cells, PIXEL_DIM, pixels],
        );
        let cond_patches = mlxcel_core::transpose_axes(&cond_patches, &[0, 1, 3, 2]);
        let x = mlxcel_core::concatenate(&noise_patches, &cond_patches, -1);
        let x = mlxcel_core::reshape(&x, &[batch * cells, pixels, PIXEL_DIM + 3]);
        // `NerfEmbedder`: append the DCT basis (cast to the activation dtype).
        let dct = mlxcel_core::astype(&dct_basis(), dt);
        let dct = mlxcel_core::broadcast_to(&dct, &[batch * cells, pixels, 64]);
        let x = self
            .x_embedder
            .forward(&mlxcel_core::concatenate(&x, &dct, -1));
        // `DecoderMLP`.
        let mut x = self.input_proj.forward(&x);
        let cond = mlxcel_core::reshape(
            &self.cond_embed.forward(&spatial),
            &[batch * cells, pixels, PIXEL_DIM],
        );
        for block in &self.res_blocks {
            x = block.forward(&x, &cond);
        }
        let x = self.final_linear.forward(&self.final_norm.forward(&x));
        let x = mlxcel_core::reshape(&x, &[batch, gh, gw, PATCH, PATCH, 3]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
        mlxcel_core::reshape(&x, &[batch, gh * PATCH, gw * PATCH, 3])
    }
}
