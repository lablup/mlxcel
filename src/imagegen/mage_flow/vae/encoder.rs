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

//! The Mage-VAE encoder `DConvEncoder` (`vae.py`, mlx-vlm `73d366ec`): an
//! image `[B, H, W, 3]` in `[-1, 1]` to the posterior mean and clipped
//! log-variance, each `[B, H/16, W/16, 128]`. Only the edit variants load it.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::blocks::{DiCoBlock, EPS, EncoderDiCoBlock, TimestepEmbedder};
use super::decoder::{CHANNELS, PATCH};
use crate::imagegen::mage_flow::layers::{LayerNorm, scalar_like, split_last};
use crate::imagegen::vae::Conv2d;

const LATENT_CHANNELS: i32 = 128;
const HEAD_CHANNELS: i32 = 768;
const BLOCKS: i32 = 21;

pub(super) struct DConvEncoder {
    patch_cond_embed: Conv2d,
    head_blocks: Vec<EncoderDiCoBlock>,
    proj_down: Conv2d,
    z_proj: Conv2d,
    fuse_proj: Conv2d,
    t_embedder: TimestepEmbedder,
    blocks: Vec<DiCoBlock>,
    norm_out: LayerNorm,
    proj_out: Conv2d,
}

impl DConvEncoder {
    pub(super) fn load(w: &WeightMap, p: &str) -> Result<Self> {
        let conv = |name: &str| Conv2d::load(w, &format!("{p}.{name}"), 1, 0);
        let head_blocks = (0..2)
            .map(|i| EncoderDiCoBlock::load(w, &format!("{p}.head_blocks.{i}"), HEAD_CHANNELS))
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..BLOCKS)
            .map(|i| DiCoBlock::load(w, &format!("{p}.blocks.{i}"), CHANNELS))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_cond_embed: Conv2d::load(w, &format!("{p}.patch_cond_embed"), PATCH, 0)?,
            head_blocks,
            proj_down: conv("proj_down")?,
            z_proj: conv("z_proj")?,
            fuse_proj: conv("fuse_proj")?,
            t_embedder: TimestepEmbedder::load(w, &format!("{p}.t_embedder"))?,
            blocks,
            norm_out: LayerNorm::affine(w, &format!("{p}.norm_out"), EPS)?,
            proj_out: conv("proj_out")?,
        })
    }

    /// `(mean, clip(logvar, -20, 10))`.
    pub(super) fn forward(&self, image: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let s = mlxcel_core::array_shape(image);
        let dt = mlxcel_core::array_dtype(image);
        let batch = s[0];
        let latent = mlxcel_core::zeros(&[batch, s[1] / PATCH, s[2] / PATCH, LATENT_CHANNELS], dt);
        let mut condition = self.patch_cond_embed.forward(image);
        for block in &self.head_blocks {
            condition = block.forward(&condition);
        }
        let condition = self.proj_down.forward(&condition);
        let mut x = self.fuse_proj.forward(&mlxcel_core::concatenate(
            &condition,
            &self.z_proj.forward(&latent),
            -1,
        ));
        let temb = self.t_embedder.zero(batch, dt);
        for block in &self.blocks {
            x = block.forward(&x, &temb);
        }
        self.moments(&x)
    }

    /// The `t = 0` embedding the DiCo blocks read.
    #[cfg(test)]
    pub(super) fn temb(&self, batch: i32, dt: i32) -> UniquePtr<MlxArray> {
        self.t_embedder.zero(batch, dt)
    }

    /// DiCo block `index` (tests feed it upstream's own input).
    #[cfg(test)]
    pub(super) fn block(&self, index: usize) -> &DiCoBlock {
        &self.blocks[index]
    }

    /// The posterior from the last DiCo block's output.
    pub(super) fn moments(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut halves = split_last(&self.proj_out.forward(&self.norm_out.forward(x)), 2);
        let logvar = halves.pop().expect("two halves");
        let mean = halves.pop().expect("two halves");
        let logvar = mlxcel_core::clip(
            &logvar,
            &scalar_like(-20.0, &logvar),
            &scalar_like(10.0, &logvar),
        );
        (mean, logvar)
    }
}
