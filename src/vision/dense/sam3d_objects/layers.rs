// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Dense inference primitives matching the pinned SAM 3D Objects implementation.
use super::bundle::array;
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, UniquePtr, weights::WeightMap};

#[path = "layers_attention.rs"]
mod attention;
#[path = "layers_blocks.rs"]
mod blocks;
pub use attention::{Attention, PreparedContext, attend};
pub use blocks::{ModulatedBlock, MultiBlock};
pub type Array = UniquePtr<MlxArray>;

pub struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}
impl Linear {
    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self> {
        let weight = array(weights, &format!("{prefix}.weight"))?;
        let shape = mx::array_shape(&weight);
        ensure!(
            shape.len() == 2 && shape.iter().all(|&d| d > 0),
            "invalid linear weight {prefix}: {shape:?}"
        );
        let bias = weights.get(&format!("{prefix}.bias")).map(|v| mx::copy(v));
        if let Some(bias) = &bias {
            ensure!(
                mx::array_shape(bias) == [shape[0]],
                "invalid linear bias {prefix}"
            );
        }
        Ok(Self { weight, bias })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        let wt = mx::transpose(&self.weight);
        match &self.bias {
            Some(b) => mx::addmm(b, x, &wt, 1.0, 1.0),
            None => mx::matmul(x, &wt),
        }
    }
}

pub struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}
impl LayerNorm {
    pub fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self> {
        let weight = array(weights, &format!("{prefix}.weight"))?;
        let bias = array(weights, &format!("{prefix}.bias"))?;
        ensure!(
            mx::array_shape(&weight).len() == 1
                && mx::array_shape(&weight) == mx::array_shape(&bias),
            "invalid norm {prefix}"
        );
        Ok(Self { weight, bias, eps })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        let dtype = mx::array_dtype(x);
        let xf = mx::astype(x, mx::dtype::FLOAT32);
        // SAFETY: affine pointers refer to owned arrays alive for the call.
        let y = unsafe { mx::fast_layer_norm(&xf, &*self.weight, &*self.bias, self.eps) };
        mx::astype(&y, dtype)
    }
}

pub fn layer_norm(x: &MlxArray, eps: f32) -> Array {
    // SAFETY: null pointers select the documented non-affine layer norm.
    unsafe { mx::fast_layer_norm(x, std::ptr::null(), std::ptr::null(), eps) }
}
pub fn add_scalar(x: &MlxArray, value: f32) -> Array {
    mx::add(x, &mx::full_f32(&[], value, mx::array_dtype(x)))
}
pub fn modulate(x: &MlxArray, scale: &MlxArray, shift: &MlxArray) -> Result<Array> {
    Ok(mx::compiled_objects_modulate(x, scale, shift)?)
}
pub fn gated_residual(x: &MlxArray, h: &MlxArray, gate: &MlxArray) -> Result<Array> {
    Ok(mx::compiled_objects_gated_residual(x, h, gate)?)
}
pub fn position_embedding(coords: &MlxArray, channels: usize) -> Array {
    let dim = channels / 6;
    let freq = mx::exp(&mx::divide_scalar(
        &mx::multiply_scalar(
            &mx::arange_f32(0.0, dim as f32, 1.0),
            -(10000_f64.ln() as f32),
        ),
        dim as f32,
    ));
    let s = mx::array_shape(coords);
    let phase = mx::multiply(
        &mx::reshape(&mx::astype(coords, mx::dtype::FLOAT32), &[s[0], 3, 1]),
        &freq,
    );
    let out = mx::reshape(
        &mx::concatenate(&mx::sin(&phase), &mx::cos(&phase), -1),
        &[s[0], (dim * 6) as i32],
    );
    mx::pad(&out, &[0, 0, 0, (channels - dim * 6) as i32], 0.0)
}

pub struct TimestepEmbedder {
    first: Linear,
    last: Linear,
}
impl TimestepEmbedder {
    pub fn from_weights(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            first: Linear::from_weights(w, &format!("{p}.mlp.0"))?,
            last: Linear::from_weights(w, &format!("{p}.mlp.2"))?,
        })
    }
    pub fn forward(&self, t: &MlxArray) -> Array {
        let freq = mx::exp(&mx::divide_scalar(
            &mx::multiply_scalar(&mx::arange_f32(0.0, 128.0, 1.0), -(10000_f64.ln() as f32)),
            128.0,
        ));
        let phase = mx::multiply(
            &mx::astype(&mx::reshape(t, &[-1, 1]), mx::dtype::FLOAT32),
            &freq,
        );
        let x = mx::astype(
            &mx::concatenate(&mx::cos(&phase), &mx::sin(&phase), -1),
            mx::array_dtype(&self.first.weight),
        );
        self.last
            .forward(&mx::compiled_silu(&self.first.forward(&x)))
    }
}
pub struct FeedForward {
    first: Linear,
    last: Linear,
}
impl FeedForward {
    pub fn from_weights(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            first: Linear::from_weights(w, &format!("{p}.mlp.0"))?,
            last: Linear::from_weights(w, &format!("{p}.mlp.2"))?,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Result<Array> {
        let h = mx::compiled_objects_gelu_tanh(&self.first.forward(x))?;
        Ok(self.last.forward(&h))
    }
}
pub struct HeadNorm {
    gamma: Array,
}
impl HeadNorm {
    pub fn from_weights(w: &WeightMap, p: &str, heads: usize, dim: usize) -> Result<Self> {
        let heads = i32::try_from(heads)?;
        let dim = i32::try_from(dim)?;
        ensure!(
            heads > 0 && dim > 0,
            "invalid head normalization dimensions {p}"
        );
        let gamma = array(w, &format!("{p}.gamma"))?;
        ensure!(
            mx::array_shape(&gamma) == [heads, dim],
            "invalid head normalization {p}.gamma: expected [{heads}, {dim}], got {:?}",
            mx::array_shape(&gamma)
        );
        Ok(Self { gamma })
    }
    pub fn forward(&self, x: &MlxArray) -> Result<Array> {
        Ok(mx::compiled_objects_head_norm(x, &self.gamma)?)
    }
}

#[cfg(test)]
#[path = "layers_tests.rs"]
mod tests;
