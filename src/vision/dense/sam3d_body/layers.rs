// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Body-local primitives matching pinned MLX Linear, LayerNorm and SwiGLU.
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, MlxArray, UniquePtr, weights::WeightMap};

pub type Array = UniquePtr<MlxArray>;

pub fn array(weights: &WeightMap, key: &str) -> Result<Array> {
    let value = weights
        .get(key)
        .with_context(|| format!("missing Body weight {key}"))?;
    Ok(mx::copy(value))
}

pub fn floating_array(weights: &WeightMap, key: &str) -> Result<Array> {
    let value = array(weights, key)?;
    ensure!(
        matches!(
            mx::array_dtype(&value),
            mx::dtype::FLOAT32 | mx::dtype::BFLOAT16 | mx::dtype::FLOAT16
        ),
        "Body weight {key} must be floating point"
    );
    Ok(value)
}

pub fn shape(value: &MlxArray, expected: &[i32], name: &str) -> Result<()> {
    let actual = mx::array_shape(value);
    ensure!(
        actual == expected,
        "Body {name}: expected {expected:?}, got {actual:?}"
    );
    Ok(())
}

pub struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}
impl Linear {
    pub fn from_weights(w: &WeightMap, prefix: &str) -> Result<Self> {
        let weight = floating_array(w, &format!("{prefix}.weight"))?;
        let dims = mx::array_shape(&weight);
        ensure!(
            dims.len() == 2 && dims.iter().all(|&n| n > 0),
            "invalid Body linear {prefix}: {dims:?}"
        );
        let bias_key = format!("{prefix}.bias");
        let bias = w
            .contains_key(&bias_key)
            .then(|| floating_array(w, &bias_key))
            .transpose()?;
        if let Some(bias) = &bias {
            shape(bias, &[dims[0]], &bias_key)?;
        }
        Ok(Self { weight, bias })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        let weight = mx::transpose(&self.weight);
        match &self.bias {
            Some(bias) => mx::addmm(bias, x, &weight, 1., 1.),
            None => mx::matmul(x, &weight),
        }
    }
}

pub struct Norm {
    pub weight: Array,
    pub bias: Array,
    eps: f32,
    upcast: bool,
}
impl Norm {
    pub fn from_weights(w: &WeightMap, prefix: &str, eps: f32, upcast: bool) -> Result<Self> {
        ensure!(
            eps.is_finite() && eps > 0.,
            "invalid Body norm epsilon {eps}"
        );
        let weight = floating_array(w, &format!("{prefix}.weight"))?;
        let bias = floating_array(w, &format!("{prefix}.bias"))?;
        let dims = mx::array_shape(&weight);
        ensure!(
            dims.len() == 1 && dims[0] > 0,
            "invalid Body norm {prefix}: {dims:?}"
        );
        shape(&bias, &dims, &format!("{prefix}.bias"))?;
        Ok(Self {
            weight,
            bias,
            eps,
            upcast,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        let input = if self.upcast {
            mx::astype(x, mx::dtype::FLOAT32)
        } else {
            mx::copy(x)
        };
        // SAFETY: both affine pointers reference owned rank-one arrays alive
        // for this call; constructors validate their equal channel counts.
        let output = unsafe { mx::fast_layer_norm(&input, &*self.weight, &*self.bias, self.eps) };
        if self.upcast {
            mx::astype(&output, mx::array_dtype(x))
        } else {
            output
        }
    }
}

pub struct LayerScale {
    gamma: Array,
}
impl LayerScale {
    pub fn from_weights(w: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        ensure!(
            channels > 0 && channels <= i32::MAX as usize,
            "invalid Body LayerScale channels"
        );
        let gamma = floating_array(w, &format!("{prefix}.gamma"))?;
        shape(&gamma, &[channels as i32], prefix)?;
        Ok(Self { gamma })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        mx::multiply(x, &self.gamma)
    }
}

pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl SwiGlu {
    pub fn from_weights(
        w: &WeightMap,
        prefix: &str,
        channels: usize,
        hidden: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: linear(w, &format!("{prefix}.w1"), channels, hidden, true)?,
            up: linear(w, &format!("{prefix}.w2"), channels, hidden, true)?,
            down: linear(w, &format!("{prefix}.w3"), hidden, channels, true)?,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        let gate = mx::compiled_silu(&self.gate.forward(x));
        self.down.forward(&mx::multiply(&gate, &self.up.forward(x)))
    }
}

pub(super) fn linear(
    w: &WeightMap,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    ensure!(
        input > 0 && output > 0 && input <= i32::MAX as usize && output <= i32::MAX as usize,
        "invalid Body linear dimensions"
    );
    let layer = Linear::from_weights(w, prefix)?;
    shape(
        &layer.weight,
        &[output as i32, input as i32],
        &format!("{prefix}.weight"),
    )?;
    ensure!(
        layer.bias.is_some() == bias,
        "Body {prefix}: expected bias={bias}"
    );
    Ok(layer)
}

#[cfg(test)]
#[path = "layers_tests.rs"]
mod tests;
