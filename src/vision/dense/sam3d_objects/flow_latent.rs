// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Sparse residual stem and latent transformer, with shared guidance encoding.
use super::super::{
    config::Config,
    layers::{
        Array, LayerNorm, Linear, ModulatedBlock, PreparedContext, TimestepEmbedder, layer_norm,
        modulate, position_embedding,
    },
    sparse::{Grid, SparseConv, pool},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
pub type LatentCondition = Vec<PreparedContext>;
struct SparseResBlock {
    norm: LayerNorm,
    conv1: SparseConv,
    conv2: SparseConv,
    emb: Linear,
    skip: Option<Linear>,
}
impl SparseResBlock {
    fn from_weights(w: &WeightMap, p: &str, input: usize, channels: usize) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::from_weights(w, &format!("{p}.norm1"), 1e-6)?,
            conv1: SparseConv::from_weights(w, &format!("{p}.conv1"), input, channels, 3)?,
            conv2: SparseConv::from_weights(w, &format!("{p}.conv2"), channels, channels, 3)?,
            emb: Linear::from_weights(w, &format!("{p}.emb_layers.1"))?,
            skip: (input != channels)
                .then(|| Linear::from_weights(w, &format!("{p}.skip_connection")))
                .transpose()?,
        })
    }
    fn forward(
        &self,
        x: &MlxArray,
        grid: &Grid,
        time: &MlxArray,
        parent: Option<&MlxArray>,
    ) -> Result<Array> {
        let emb = self.emb.forward(&mx::compiled_silu(time));
        let c = mx::array_shape(&emb)[1] / 2;
        let scale = mx::slice_last_dim(&emb, 0, c);
        let shift = mx::slice_last_dim(&emb, c, c * 2);
        let h = mx::compiled_silu(&self.norm.forward(x));
        let h = match parent {
            Some(p) => self.conv1.from_parents(&h, p, grid)?,
            None => self.conv1.forward(&h, grid)?,
        };
        let h = modulate(&layer_norm(&h, 1e-6), &scale, &shift)?;
        let skip = match &self.skip {
            Some(s) => s.forward(x),
            None => mx::copy(x),
        };
        let skip = match parent {
            Some(p) => mx::take(&skip, p, 0),
            None => skip,
        };
        Ok(mx::add(
            &self.conv2.forward(&mx::compiled_silu(&h), grid)?,
            &skip,
        ))
    }
}
struct Stem {
    input: Array,
    low: Array,
    tokens: Array,
    parent: Array,
    time: Array,
}
pub struct LatentFlow {
    time: TimestepEmbedder,
    input: Linear,
    input_blocks: [SparseResBlock; 2],
    blocks: Vec<ModulatedBlock>,
    output_blocks: [SparseResBlock; 2],
    output: Linear,
}
impl LatentFlow {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        let (hidden, io) = (c.hidden_size, c.io_channels);
        Ok(Self {
            time: TimestepEmbedder::from_weights(w, &format!("{p}.t_embedder"))?,
            input: Linear::from_weights(w, &format!("{p}.input_layer"))?,
            input_blocks: [
                SparseResBlock::from_weights(w, &format!("{p}.input_blocks.0"), io, io)?,
                SparseResBlock::from_weights(w, &format!("{p}.input_blocks.1"), io, hidden)?,
            ],
            blocks: (0..c.num_blocks)
                .map(|i| ModulatedBlock::from_weights(w, &format!("{p}.blocks.{i}"), c.num_heads))
                .collect::<Result<_>>()?,
            output_blocks: [
                SparseResBlock::from_weights(w, &format!("{p}.out_blocks.0"), 2 * hidden, io)?,
                SparseResBlock::from_weights(w, &format!("{p}.out_blocks.1"), 2 * io, io)?,
            ],
            output: Linear::from_weights(w, &format!("{p}.out_layer"))?,
        })
    }
    pub fn prepare_condition(&self, tokens: Option<&MlxArray>) -> Result<LatentCondition> {
        self.blocks
            .iter()
            .map(|b| b.cross_attn.prepare(tokens))
            .collect()
    }
    fn encode(&self, x: &MlxArray, grid: &Grid, time: &MlxArray) -> Result<Stem> {
        let h = self
            .input
            .forward(&mx::astype(x, mx::array_dtype(&self.input.weight)));
        let time = self.time.forward(time);
        let input = self.input_blocks[0].forward(&h, grid, &time, None)?;
        let (h, child, parent) = pool(&input, grid)?;
        let low = self.input_blocks[1].forward(&h, &child, &time, None)?;
        let s = mx::array_shape(&low);
        let pos = position_embedding(&mx::slice_last_dim(&child.coords, 1, 4), s[1] as usize);
        let tokens = mx::reshape(
            &mx::add(&low, &mx::astype(&pos, mx::array_dtype(&low))),
            &[1, s[0], s[1]],
        );
        Ok(Stem {
            input,
            low,
            tokens,
            parent,
            time,
        })
    }
    fn decode(&self, stem: &Stem, grid: &Grid, condition: &LatentCondition) -> Result<Array> {
        ensure!(
            condition.len() == self.blocks.len(),
            "latent context depth mismatch"
        );
        let mut h = mx::copy(&stem.tokens);
        for (block, context) in self.blocks.iter().zip(condition) {
            h = block.forward(&h, &stem.time, context)?;
        }
        let s = mx::array_shape(&stem.low);
        let h = mx::concatenate(&mx::reshape(&h, &s), &stem.low, -1);
        let h = self.output_blocks[0].forward(&h, grid, &stem.time, Some(&stem.parent))?;
        let h = self.output_blocks[1].forward(
            &mx::concatenate(&h, &stem.input, -1),
            grid,
            &stem.time,
            None,
        )?;
        Ok(self.output.forward(&layer_norm(&h, 1e-5)))
    }
    pub fn forward(
        &self,
        x: &MlxArray,
        grid: &Grid,
        time: &MlxArray,
        condition: &LatentCondition,
    ) -> Result<Array> {
        self.decode(&self.encode(x, grid, time)?, grid, condition)
    }
    pub fn guided(
        &self,
        x: &MlxArray,
        grid: &Grid,
        time: &MlxArray,
        condition: &LatentCondition,
        uncondition: &LatentCondition,
    ) -> Result<(Array, Array)> {
        let stem = self.encode(x, grid, time)?;
        Ok((
            self.decode(&stem, grid, condition)?,
            self.decode(&stem, grid, uncondition)?,
        ))
    }
}
