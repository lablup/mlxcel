// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Dense 3D occupancy decoder and exact depth/height/width pixel shuffle.
use super::super::{
    bundle::array,
    config::Config,
    layers::{Array, LayerNorm},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
struct Conv3d {
    weight: Array,
    bias: Array,
}
impl Conv3d {
    fn from_weights(w: &WeightMap, p: &str, input: usize, output: usize) -> Result<Self> {
        let weight = array(w, &format!("{p}.weight"))?;
        let bias = array(w, &format!("{p}.bias"))?;
        ensure!(
            mx::array_shape(&weight) == [output as i32, 3, 3, 3, input as i32],
            "invalid 3D convolution {p}"
        );
        ensure!(
            mx::array_shape(&bias) == [output as i32],
            "invalid 3D convolution bias {p}"
        );
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: &MlxArray) -> Array {
        mx::add(
            &mx::conv3d(x, &self.weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            &self.bias,
        )
    }
}
struct DenseResBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    conv1: Conv3d,
    conv2: Conv3d,
}
impl DenseResBlock {
    fn from_weights(w: &WeightMap, p: &str, c: usize) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::from_weights(w, &format!("{p}.norm1"), 1e-5)?,
            norm2: LayerNorm::from_weights(w, &format!("{p}.norm2"), 1e-5)?,
            conv1: Conv3d::from_weights(w, &format!("{p}.conv1"), c, c)?,
            conv2: Conv3d::from_weights(w, &format!("{p}.conv2"), c, c)?,
        })
    }
    fn forward(&self, x: &MlxArray) -> Array {
        let h = self
            .conv1
            .forward(&mx::compiled_silu(&self.norm1.forward(x)));
        mx::add(
            x,
            &self
                .conv2
                .forward(&mx::compiled_silu(&self.norm2.forward(&h))),
        )
    }
}
enum DenseBlock {
    Residual(DenseResBlock),
    Upsample(Conv3d),
}
impl DenseBlock {
    fn forward(&self, x: &MlxArray) -> Array {
        match self {
            Self::Residual(block) => block.forward(x),
            Self::Upsample(conv) => {
                let x = conv.forward(x);
                let s = mx::array_shape(&x);
                let x = mx::reshape(&x, &[s[0], s[1], s[2], s[3], s[4] / 8, 2, 2, 2]);
                mx::reshape(
                    &mx::transpose_axes(&x, &[0, 1, 5, 2, 6, 3, 7, 4]),
                    &[s[0], s[1] * 2, s[2] * 2, s[3] * 2, s[4] / 8],
                )
            }
        }
    }
}
pub struct StructureDecoder {
    input: Conv3d,
    middle: Vec<DenseResBlock>,
    blocks: Vec<DenseBlock>,
    norm: LayerNorm,
    output: Conv3d,
}
impl StructureDecoder {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        let channels = &c.structure_channels;
        ensure!(
            !channels.is_empty(),
            "structure decoder needs channel levels"
        );
        let mut blocks = Vec::new();
        for (i, &ch) in channels.iter().enumerate() {
            for _ in 0..c.structure_res_blocks {
                blocks.push(DenseBlock::Residual(DenseResBlock::from_weights(
                    w,
                    &format!("{p}.blocks.{}", blocks.len()),
                    ch,
                )?));
            }
            if i + 1 < channels.len() {
                blocks.push(DenseBlock::Upsample(Conv3d::from_weights(
                    w,
                    &format!("{p}.blocks.{}.conv", blocks.len()),
                    ch,
                    channels[i + 1] * 8,
                )?));
            }
        }
        Ok(Self {
            input: Conv3d::from_weights(
                w,
                &format!("{p}.input_layer"),
                c.latent_channels,
                channels[0],
            )?,
            middle: (0..c.structure_res_blocks)
                .map(|i| {
                    DenseResBlock::from_weights(w, &format!("{p}.middle_block.{i}"), channels[0])
                })
                .collect::<Result<_>>()?,
            blocks,
            norm: LayerNorm::from_weights(w, &format!("{p}.out_layer.0"), 1e-5)?,
            output: Conv3d::from_weights(
                w,
                &format!("{p}.out_layer.2"),
                channels[channels.len() - 1],
                1,
            )?,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Result<Array> {
        let s = mx::array_shape(x);
        ensure!(s.len() == 3, "structure decoder expects BNC latents");
        let side = (s[1] as f64).cbrt().round() as i32;
        ensure!(
            side > 0 && i64::from(side).pow(3) == i64::from(s[1]),
            "structure latent length must be cubic"
        );
        let x = mx::reshape(x, &[s[0], side, side, side, s[2]]);
        let mut x = self
            .input
            .forward(&mx::astype(&x, mx::array_dtype(&self.input.weight)));
        for block in &self.middle {
            x = block.forward(&x);
        }
        for block in &self.blocks {
            x = block.forward(&x);
        }
        Ok(self
            .output
            .forward(&mx::compiled_silu(&self.norm.forward(&x))))
    }
}
