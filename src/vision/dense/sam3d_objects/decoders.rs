// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Occupancy, Gaussian, and mesh-feature inference decoders.
use super::{
    config::Config,
    layers::{Array, FeedForward, Linear, layer_norm, position_embedding},
    sparse::{Grid, window_attention},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
#[path = "decoders_geometry.rs"]
mod geometry;
#[path = "decoders_structure.rs"]
mod structure;
pub use geometry::{GaussianDecoder, MeshDecoder};
pub use structure::StructureDecoder;

struct WindowBlock {
    qkv: Linear,
    out: Linear,
    mlp: FeedForward,
    heads: i32,
    window: i32,
    shift: i32,
}
impl WindowBlock {
    fn from_weights(w: &WeightMap, p: &str, c: &Config, index: usize) -> Result<Self> {
        Ok(Self {
            qkv: Linear::from_weights(w, &format!("{p}.attn.to_qkv"))?,
            out: Linear::from_weights(w, &format!("{p}.attn.to_out"))?,
            mlp: FeedForward::from_weights(w, &format!("{p}.mlp"))?,
            heads: c.decoder_heads as i32,
            window: c.window_size as i32,
            shift: (c.window_size / 2 * (index % 2)) as i32,
        })
    }
    fn forward(&self, x: &MlxArray, grid: &Grid) -> Result<Array> {
        let n = mx::array_shape(x)[0];
        let h = mx::reshape(
            &self.qkv.forward(&layer_norm(x, 1e-6)),
            &[n, 3, self.heads, -1],
        );
        let h = window_attention(&h, grid, self.window, self.shift)?;
        let x = mx::add(x, &self.out.forward(&h));
        Ok(mx::add(&x, &self.mlp.forward(&layer_norm(&x, 1e-6))))
    }
}
struct DecoderBase {
    input: Linear,
    blocks: Vec<WindowBlock>,
}
impl DecoderBase {
    fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        ensure!(
            c.decoder_channels >= 6
                && c.decoder_heads > 0
                && c.decoder_channels.is_multiple_of(c.decoder_heads),
            "invalid decoder channel/head configuration"
        );
        Ok(Self {
            input: Linear::from_weights(w, &format!("{p}.input_layer"))?,
            blocks: (0..c.decoder_blocks)
                .map(|i| WindowBlock::from_weights(w, &format!("{p}.blocks.{i}"), c, i))
                .collect::<Result<_>>()?,
        })
    }
    fn encode(&self, x: &MlxArray, grid: &Grid) -> Result<Array> {
        let mut x = self
            .input
            .forward(&mx::astype(x, mx::array_dtype(&self.input.weight)));
        let channels = mx::array_shape(&x)[1];
        let pos = position_embedding(&mx::slice_last_dim(&grid.coords, 1, 4), channels as usize);
        x = mx::add(&x, &mx::astype(&pos, mx::array_dtype(&x)));
        for block in &self.blocks {
            x = block.forward(&x, grid)?;
        }
        Ok(x)
    }
}
