// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Gaussian position embeddings and the body's invalid-point prompt.
use super::{
    layers::{Array, array},
    ops::{cat, scalar_add},
};
use anyhow::Result;
use mlxcel_core::{self as mx, weights::WeightMap};

use crate::vision::weights::load_dense_param;

pub struct PromptEncoder {
    matrix: Array,
    invalid: Array,
    dim: i32,
}
impl PromptEncoder {
    pub fn from_weights(w: &WeightMap, dim: usize) -> Result<Self> {
        let invalid_key = "prompt_encoder.invalid_point_embed.weight";
        if !w.contains_key(invalid_key) {
            anyhow::bail!("missing Body weight {invalid_key}");
        }
        Ok(Self {
            matrix: array(
                w,
                "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            )?,
            invalid: load_dense_param(w, invalid_key, &[1, dim as i32], None)
                .map_err(anyhow::Error::msg)?,
            dim: dim as i32,
        })
    }
    pub fn invalid(&self) -> Array {
        mx::reshape(&self.invalid, &[1, 1, self.dim])
    }
    pub fn dense_pe(&self, h: i32, w: i32) -> Array {
        let y = mx::divide_scalar(
            &scalar_add(&mx::arange_f32(0.0, h as f32, 1.0), 0.5),
            h as f32,
        );
        let x = mx::divide_scalar(
            &scalar_add(&mx::arange_f32(0.0, w as f32, 1.0), 0.5),
            w as f32,
        );
        let gy = mx::broadcast_to(&mx::reshape(&y, &[h, 1, 1]), &[h, w, 1]);
        let gx = mx::broadcast_to(&mx::reshape(&x, &[1, w, 1]), &[h, w, 1]);
        let coords = mx::reshape(&cat(&[&gx, &gy], -1), &[1, h * w, 2]);
        let coords = scalar_add(&mx::multiply_scalar(&coords, 2.0), -1.0);
        let matrix = mx::multiply_scalar(
            &mx::multiply_scalar(&mx::multiply_scalar(&self.matrix, 1.0), 2.0),
            std::f32::consts::PI,
        );
        let phase = mx::matmul(&coords, &matrix);
        mx::reshape(
            &cat(&[&mx::sin(&phase), &mx::cos(&phase)], -1),
            &[1, h, w, self.dim],
        )
    }
}
