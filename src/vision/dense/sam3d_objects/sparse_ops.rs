// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::{Array, Grid, concat, host_i32, validate, weight};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
#[path = "sparse_attention.rs"]
mod attention;
pub use attention::window_attention;

pub struct SparseConv {
    weight: Array,
    bias: Array,
    input: i32,
    output: i32,
    kernel: i32,
}
impl SparseConv {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(kernel, 1 | 3) && in_channels > 0 && out_channels > 0,
            "sparse convolution requires positive channels and kernel 1 or 3"
        );
        let w = weight(weights, &format!("{prefix}.conv.weight"))?;
        let b = weight(weights, &format!("{prefix}.conv.bias"))?;
        ensure!(
            mx::array_shape(&w)
                == [
                    out_channels as i32,
                    kernel as i32,
                    kernel as i32,
                    kernel as i32,
                    in_channels as i32
                ],
            "sparse convolution {prefix} weight shape mismatch"
        );
        ensure!(
            mx::array_shape(&b) == [out_channels as i32],
            "sparse convolution {prefix} bias shape mismatch"
        );
        Ok(Self {
            weight: w,
            bias: b,
            input: in_channels as i32,
            output: out_channels as i32,
            kernel: kernel as i32,
        })
    }
    pub fn forward(&self, x: &MlxArray, grid: &Grid) -> Result<Array> {
        ensure!(
            validate(x, grid.len())? == self.input,
            "sparse convolution input channels mismatch"
        );
        if grid.len() == 0 {
            return Ok(mx::zeros(&[0, self.output], mx::array_dtype(&self.weight)));
        }
        let taps = self.kernel.pow(3);
        let w = mx::transpose(&mx::reshape(
            &self.weight,
            &[self.output, taps * self.input],
        ));
        if self.kernel == 1 {
            return Ok(mx::addmm(&self.bias, x, &w, 1.0, 1.0));
        }
        let padded = mx::concatenate(x, &mx::zeros(&[1, self.input], mx::array_dtype(x)), 0);
        let neighbors = grid.neighbors()?;
        let rows = ((16 << 20) / (27 * self.input as usize * mx::array_itemsize(x))).max(1);
        let mut outputs = Vec::new();
        for start in (0..grid.len()).step_by(rows) {
            let end = (start + rows).min(grid.len());
            let indices = mx::slice(neighbors, &[start as i32, 0], &[end as i32, 27]);
            let values = mx::reshape(
                &mx::take(&padded, &indices, 0),
                &[(end - start) as i32, 27 * self.input],
            );
            let out = mx::addmm(&self.bias, &values, &w, 1.0, 1.0);
            mx::eval(&out);
            outputs.push(out);
        }
        concat(&outputs, 0)
    }
    pub fn from_parents(&self, x: &MlxArray, parent: &MlxArray, grid: &Grid) -> Result<Array> {
        let xs = mx::array_shape(x);
        ensure!(
            xs.len() == 2 && xs[0] > 0 && validate(x, xs[0] as usize)? == self.input,
            "sparse parent feature shape mismatch"
        );
        ensure!(
            mx::array_shape(parent) == [grid.len() as i32]
                && mx::array_dtype(parent) == mx::dtype::INT32,
            "sparse parent map must be int32 [N]"
        );
        ensure!(
            host_i32(parent).iter().all(|&i| i >= 0 && i < xs[0]),
            "sparse parent index out of bounds"
        );
        if grid.len() == 0 {
            return Ok(mx::zeros(&[0, self.output], mx::array_dtype(&self.weight)));
        }
        if self.kernel == 1 {
            let w = mx::transpose(&mx::reshape(&self.weight, &[self.output, self.input]));
            return Ok(mx::take(&mx::addmm(&self.bias, x, &w, 1.0, 1.0), parent, 0));
        }
        let w = mx::astype(
            &mx::transpose_axes(
                &mx::reshape(&self.weight, &[self.output, 27, self.input]),
                &[2, 1, 0],
            ),
            mx::dtype::FLOAT32,
        );
        let x = mx::astype(x, mx::dtype::FLOAT32);
        let rows = mx::take(
            &mx::concatenate(parent, &mx::from_slice_i32(&[xs[0]], &[1]), 0),
            grid.neighbors()?,
            0,
        );
        let channels =
            (self.output as usize).min(((256 << 20) / ((xs[0] as usize + 1) * 27 * 4)).max(1));
        let voxels = ((64 << 20) / (27 * channels * 4)).max(1);
        let mut columns = Vec::new();
        for c in (0..self.output as usize).step_by(channels) {
            let width = channels.min(self.output as usize - c) as i32;
            let wc = mx::reshape(
                &mx::slice(&w, &[0, 0, c as i32], &[self.input, 27, c as i32 + width]),
                &[self.input, 27 * width],
            );
            let projected = mx::concatenate(
                &mx::reshape(&mx::matmul(&x, &wc), &[xs[0] * 27, width]),
                &mx::zeros(&[27, width], mx::dtype::FLOAT32),
                0,
            );
            mx::eval(&projected);
            let mut chunks = Vec::new();
            for start in (0..grid.len()).step_by(voxels) {
                let end = (start + voxels).min(grid.len());
                let idx = mx::slice(&rows, &[start as i32, 0], &[end as i32, 27]);
                let idx = mx::add(
                    &mx::multiply(&idx, &mx::from_slice_i32(&[27], &[1])),
                    &mx::arange_i32(0, 27, 1),
                );
                let out = mx::sum_axis(&mx::take(&projected, &idx, 0), 1, false);
                mx::eval(&out);
                chunks.push(out);
            }
            let out = concat(&chunks, 0)?;
            mx::eval(&out);
            columns.push(out);
        }
        Ok(mx::astype(
            &mx::add(&*concat(&columns, 1)?, &self.bias),
            mx::array_dtype(&self.weight),
        ))
    }
}

pub fn pool(x: &MlxArray, grid: &Grid) -> Result<(Array, Grid, Array)> {
    let channels = validate(x, grid.len())?;
    let (child, parent) = grid.downsample()?;
    let n = grid.len() as i32;
    let m = child.len() as i32;
    if n == 0 {
        return Ok((mx::zeros(&[0, channels], mx::array_dtype(x)), child, parent));
    }
    let sums = mx::scatter_add(
        &mx::zeros(&[m, channels], mx::dtype::FLOAT32),
        &parent,
        &mx::reshape(&mx::astype(x, mx::dtype::FLOAT32), &[n, 1, channels]),
        0,
    );
    // The torch reference includes the initial zero in scatter_reduce(mean).
    let counts = mx::scatter_add(
        &mx::ones(&[m, 1], mx::dtype::FLOAT32),
        &parent,
        &mx::ones(&[n, 1, 1], mx::dtype::FLOAT32),
        0,
    );
    Ok((
        mx::astype(&mx::divide(&sums, &counts), mx::array_dtype(x)),
        child,
        parent,
    ))
}

pub struct SparseGroupNorm {
    weight: Array,
    bias: Array,
    groups: i32,
    channels: i32,
}
impl SparseGroupNorm {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        groups: usize,
    ) -> Result<Self> {
        let groups = groups.min(channels);
        ensure!(
            groups > 0 && channels.is_multiple_of(groups),
            "invalid sparse normalization groups"
        );
        let w = weight(weights, &format!("{prefix}.weight"))?;
        let b = weight(weights, &format!("{prefix}.bias"))?;
        ensure!(
            mx::array_shape(&w) == [channels as i32] && mx::array_shape(&b) == [channels as i32],
            "sparse normalization weight shape mismatch"
        );
        Ok(Self {
            weight: w,
            bias: b,
            groups: groups as i32,
            channels: channels as i32,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Result<Array> {
        let s = mx::array_shape(x);
        ensure!(
            s.len() == 2 && validate(x, s[0] as usize)? == self.channels,
            "sparse normalization channel mismatch"
        );
        if s[0] == 0 {
            return Ok(mx::copy(x));
        }
        let h = mx::reshape(
            &mx::astype(x, mx::dtype::FLOAT32),
            &[s[0], self.groups, self.channels / self.groups],
        );
        let mean = mx::mean_axis(&mx::mean_axis(&h, 2, true), 0, true);
        let centered = mx::subtract(&h, &mean);
        let var = mx::mean_axis(
            &mx::mean_axis(&mx::multiply(&centered, &centered), 2, true),
            0,
            true,
        );
        let h = mx::reshape(
            &mx::multiply(
                &centered,
                &mx::rsqrt(&mx::add(&var, &mx::from_slice_f32(&[1e-5], &[1]))),
            ),
            &s,
        );
        Ok(mx::astype(
            &mx::add(&mx::multiply(&h, &self.weight), &self.bias),
            mx::array_dtype(x),
        ))
    }
}
