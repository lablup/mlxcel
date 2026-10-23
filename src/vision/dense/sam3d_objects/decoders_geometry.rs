// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Sparse Gaussian parameters and mesh-feature subdivision heads.
use super::super::{
    bundle::array,
    config::Config,
    layers::{Array, Linear, add_scalar, layer_norm},
    sparse::{Grid, SparseConv, SparseGroupNorm},
};
use super::DecoderBase;
use crate::vision::dense::geometry_io::Gaussians;
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
pub struct GaussianDecoder {
    base: DecoderBase,
    output: Linear,
    offset: Array,
    count: i32,
}
impl GaussianDecoder {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        let output = Linear::from_weights(w, &format!("{p}.out_layer"))?;
        let width = mx::array_shape(&output.weight)[0];
        ensure!(
            width > 0 && width % 14 == 0,
            "Gaussian output width must be 14 times splat count"
        );
        let count = width / 14;
        let offset = array(w, &format!("{p}.offset_perturbation"))?;
        ensure!(
            mx::array_shape(&offset) == [count, 3],
            "Gaussian offset shape mismatch"
        );
        Ok(Self {
            base: DecoderBase::from_weights(w, p, c)?,
            output,
            offset,
            count,
        })
    }
    pub fn forward(&self, x: &MlxArray, grid: &Grid) -> Result<Gaussians> {
        let h = mx::astype(
            &self
                .output
                .forward(&layer_norm(&*self.base.encode(x, grid)?, 1e-5)),
            mx::dtype::FLOAT32,
        );
        let n = self.count;
        let xyz = mx::reshape(&mx::slice_last_dim(&h, 0, n * 3), &[-1, n, 3]);
        let color = mx::reshape(&mx::slice_last_dim(&h, n * 3, n * 6), &[-1, 3]);
        let scaling = mx::reshape(&mx::slice_last_dim(&h, n * 6, n * 9), &[-1, 3]);
        let rotation = mx::reshape(&mx::slice_last_dim(&h, n * 9, n * 13), &[-1, 4]);
        let opacity = mx::reshape(&mx::slice_last_dim(&h, n * 13, n * 14), &[-1, 1]);
        let coords = mx::reshape(
            &mx::astype(&mx::slice_last_dim(&grid.coords, 1, 4), mx::dtype::FLOAT32),
            &[-1, 1, 3],
        );
        let base = mx::divide_scalar(&add_scalar(&coords, 0.5), grid.resolution as f32);
        let displacement = mx::multiply_scalar(
            &mx::tanh(&mx::add(&xyz, &self.offset)),
            0.75 / grid.resolution as f32,
        );
        let positions = add_scalar(&mx::reshape(&mx::add(&base, &displacement), &[-1, 3]), -0.5);
        let scale = mx::softplus(&add_scalar(&scaling, 0.004_f64.exp_m1().ln() as f32));
        let scales = mx::sqrt(&add_scalar(
            &mx::multiply(&scale, &scale),
            0.0009_f64.powi(2) as f32,
        ));
        let rotation = mx::add(
            &mx::multiply_scalar(&rotation, 0.1),
            &mx::from_slice_f32(&[1.0, 0.0, 0.0, 0.0], &[4]),
        );
        let squared = mx::sum_axis(&mx::multiply(&rotation, &rotation), -1, true);
        let rotations = mx::multiply(
            &rotation,
            &mx::rsqrt(&mx::maximum(&squared, &mx::from_slice_f32(&[1e-24], &[]))),
        );
        let opacities = mx::sigmoid(&add_scalar(&opacity, (0.1_f64 / 0.9).ln() as f32));
        Ok(Gaussians {
            positions: rows3(&positions),
            sh_dc: rows3(&color),
            scales: rows3(&scales),
            rotations: mx::utils::array_to_vec_f32(&rotations)
                .chunks_exact(4)
                .map(|v| [v[0], v[1], v[2], v[3]])
                .collect(),
            opacities: mx::utils::array_to_vec_f32(&opacities),
        })
    }
}
fn rows3(x: &MlxArray) -> Vec<[f32; 3]> {
    mx::utils::array_to_vec_f32(x)
        .chunks_exact(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect()
}
struct SubdivideBlock {
    act: SparseGroupNorm,
    first: SparseConv,
    norm: SparseGroupNorm,
    last: SparseConv,
    skip: SparseConv,
}
impl SubdivideBlock {
    fn from_weights(w: &WeightMap, p: &str, input: usize, channels: usize) -> Result<Self> {
        Ok(Self {
            act: SparseGroupNorm::from_weights(w, &format!("{p}.act_layers.0"), input, 32)?,
            first: SparseConv::from_weights(w, &format!("{p}.out_layers.0"), input, channels, 3)?,
            norm: SparseGroupNorm::from_weights(w, &format!("{p}.out_layers.1"), channels, 32)?,
            last: SparseConv::from_weights(w, &format!("{p}.out_layers.3"), channels, channels, 3)?,
            skip: SparseConv::from_weights(w, &format!("{p}.skip_connection"), input, channels, 1)?,
        })
    }
    fn forward(&self, x: &MlxArray, grid: &Grid) -> Result<(Array, Grid)> {
        let grid = grid.subdivide()?;
        let parent = mx::repeat(&mx::arange_i32(0, mx::array_shape(x)[0], 1), 8, 0);
        let h =
            self.first
                .from_parents(&mx::compiled_silu(&*self.act.forward(x)?), &parent, &grid)?;
        let h = self
            .last
            .forward(&mx::compiled_silu(&*self.norm.forward(&h)?), &grid)?;
        Ok((
            mx::add(&h, &*self.skip.from_parents(x, &parent, &grid)?),
            grid,
        ))
    }
}
pub struct MeshDecoder {
    base: DecoderBase,
    upsample: [SubdivideBlock; 2],
    output: Linear,
}
impl MeshDecoder {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        let ch = c.decoder_channels;
        Ok(Self {
            base: DecoderBase::from_weights(w, p, c)?,
            upsample: [
                SubdivideBlock::from_weights(w, &format!("{p}.upsample.0"), ch, ch / 4)?,
                SubdivideBlock::from_weights(w, &format!("{p}.upsample.1"), ch / 4, ch / 8)?,
            ],
            output: Linear::from_weights(w, &format!("{p}.out_layer"))?,
        })
    }
    pub fn forward(&self, x: &MlxArray, grid: &Grid) -> Result<(Array, Grid)> {
        let x = self.base.encode(x, grid)?;
        let (x, grid) = self.upsample[0].forward(&x, grid)?;
        let (x, grid) = self.upsample[1].forward(&x, &grid)?;
        Ok((self.output.forward(&x), grid))
    }
}
