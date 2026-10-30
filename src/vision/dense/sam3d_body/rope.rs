// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! DINOv3 axial learned-period RoPE, with split-half rotation and prefix bypass.
use super::layers::{Array, as_dtype, floating_array as array, shape, upstream_dtype};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};

pub struct Rope {
    periods: Array,
    head_dim: usize,
}
impl Rope {
    pub fn from_weights(w: &WeightMap, prefix: &str, head_dim: usize) -> Result<Self> {
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(4) && head_dim <= i32::MAX as usize,
            "Body RoPE head_dim must be a positive multiple of four"
        );
        let periods = array(w, &format!("{prefix}.periods"))?;
        shape(&periods, &[(head_dim / 4) as i32], prefix)?;
        Ok(Self { periods, head_dim })
    }
    pub fn forward(&self, height: usize, width: usize) -> Result<(Array, Array)> {
        let count = height
            .checked_mul(width)
            .filter(|&v| v > 0 && v <= i32::MAX as usize)
            .ok_or_else(|| anyhow::anyhow!("Body RoPE grid exceeds MLX dimensions"))?;
        ensure!(
            count
                .checked_mul(self.head_dim)
                .is_some_and(|n| n <= i32::MAX as usize),
            "Body RoPE table exceeds MLX dimensions"
        );
        let scalar = |v: f32| mx::from_slice_f32(&[v], &[]);
        let coords = |size: usize| {
            let centers = mx::arange_f32(0.5, size as f32, 1.);
            mx::subtract(
                &mx::multiply(&mx::divide(&centers, &scalar(size as f32)), &scalar(2.)),
                &scalar(1.),
            )
        };
        let h = mx::broadcast_to(
            &mx::reshape(&coords(height), &[height as i32, 1]),
            &[height as i32, width as i32],
        );
        let w = mx::broadcast_to(
            &mx::reshape(&coords(width), &[1, width as i32]),
            &[height as i32, width as i32],
        );
        let coords = mx::concatenate(
            &mx::reshape(&h, &[count as i32, 1, 1]),
            &mx::reshape(&w, &[count as i32, 1, 1]),
            1,
        );
        let periods = mx::reshape(
            &mx::astype(&self.periods, mx::dtype::FLOAT32),
            &[1, 1, (self.head_dim / 4) as i32],
        );
        // Python evaluates 2*pi as a double, then converts the weak tensor scalar.
        let angles = mx::divide(
            &mx::multiply(&scalar((2. * std::f64::consts::PI) as f32), &coords),
            &periods,
        );
        let angles = mx::reshape(&angles, &[count as i32, (self.head_dim / 2) as i32]);
        let angles = mx::concatenate(&angles, &angles, -1);
        Ok((mx::sin(&angles), mx::cos(&angles)))
    }
}

fn rotate_half(x: &MlxArray) -> Array {
    let dims = mx::array_shape(x);
    let half = dims[3] / 2;
    let first = mx::slice(x, &[0, 0, 0, 0], &[dims[0], dims[1], dims[2], half]);
    let last = mx::slice(x, &[0, 0, 0, half], &dims);
    mx::concatenate(&mx::negative(&last), &first, -1)
}

pub fn apply_rope(
    q: &MlxArray,
    k: &MlxArray,
    sin: &MlxArray,
    cos: &MlxArray,
    prefix: usize,
) -> Result<(Array, Array)> {
    let dims = mx::array_shape(q);
    ensure!(
        dims.len() == 4 && dims.iter().all(|&n| n > 0) && dims[3] % 2 == 0,
        "Body RoPE expects positive BHND with even D"
    );
    shape(k, &dims, "RoPE key")?;
    ensure!(
        prefix < dims[2] as usize,
        "Body RoPE prefix consumes all tokens"
    );
    let table_shape = [dims[2] - prefix as i32, dims[3]];
    shape(sin, &table_shape, "RoPE sine")?;
    shape(cos, &table_shape, "RoPE cosine")?;
    ensure!(
        mx::array_dtype(sin) == mx::dtype::FLOAT32 && mx::array_dtype(cos) == mx::dtype::FLOAT32,
        "Body RoPE tables must be F32"
    );
    let sin = mx::reshape(sin, &[1, 1, table_shape[0], table_shape[1]]);
    let cos = mx::reshape(cos, &[1, 1, table_shape[0], table_shape[1]]);
    // Upstream promotes BF16 queries against the F32 tables, prefix included.
    let dtype = upstream_dtype(&[q, k, &sin]);
    let apply = |x: &MlxArray| {
        let x = as_dtype(x, dtype);
        let x = &*x;
        let leading = mx::slice(
            x,
            &[0, 0, 0, 0],
            &[dims[0], dims[1], prefix as i32, dims[3]],
        );
        let patches = mx::slice(x, &[0, 0, prefix as i32, 0], &dims);
        let rotated = mx::add(
            &mx::multiply(&patches, &cos),
            &mx::multiply(&rotate_half(&patches), &sin),
        );
        mx::concatenate(&leading, &rotated, 2)
    };
    Ok((apply(q), apply(k)))
}

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;
