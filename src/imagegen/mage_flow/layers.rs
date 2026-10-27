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

//! Small MLX `nn` layers the Mage-Flow transformer and VAE share.
//!
//! Each reproduces MLX's own arithmetic in the activation dtype: `nn.Linear`
//! is `addmm(bias, x, W.T)` (one rounding of a bf16 output, through
//! [`crate::vision::dense::sam3::linear::UnifiedLinear`], which also runs a
//! quantized projection), and `nn.LayerNorm` / `nn.RMSNorm` are the fused
//! kernels.

use anyhow::{Context, Result};
use mlxcel_core::layers::{RMSNorm, UnifiedLinear as CoreUnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::dense::sam3::linear::UnifiedLinear;

/// Quantization of the projections a loader builds; dense when `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearQuant {
    pub group_size: i32,
    pub bits: i32,
    /// `affine`, `mxfp4`, `mxfp8` or `nvfp4`.
    pub mode: &'static str,
}

impl LinearQuant {
    /// From a spec [`crate::imagegen::quant::QuantSpec::validated`] accepted.
    pub fn from_spec(spec: &crate::imagegen::quant::QuantSpec) -> Result<Self> {
        let mode = match spec.mode.as_str() {
            "affine" => "affine",
            "mxfp4" => "mxfp4",
            "mxfp8" => "mxfp8",
            "nvfp4" => "nvfp4",
            other => anyhow::bail!("Unsupported Mage-Flow quantization mode: {other}"),
        };
        Ok(Self {
            group_size: spec.group_size,
            bits: spec.bits,
            mode,
        })
    }
}

/// `nn.Linear`, or `nn.QuantizedLinear` when `<prefix>.scales` is present.
pub struct Dense(Projection);

enum Projection {
    /// `addmm(bias, x, W.T)`, upstream's single rounding.
    Plain(UnifiedLinear),
    Quantized(CoreUnifiedLinear),
}

impl Dense {
    pub fn load(weights: &WeightMap, prefix: &str, quant: Option<LinearQuant>) -> Result<Self> {
        if !weights.contains_key(&format!("{prefix}.weight")) {
            anyhow::bail!("Mage-Flow weight not found: {prefix}.weight");
        }
        let context = || format!("Mage-Flow weight {prefix}");
        if weights.contains_key(&format!("{prefix}.scales")) {
            let Some(q) = quant else {
                anyhow::bail!(
                    "Mage-Flow weight {prefix} is quantized but the checkpoint has no quantization metadata"
                );
            };
            return CoreUnifiedLinear::from_weights_with_mode(
                weights,
                prefix,
                q.group_size,
                q.bits,
                q.mode,
            )
            .map(|l| Self(Projection::Quantized(l)))
            .map_err(anyhow::Error::msg)
            .with_context(context);
        }
        // A dense projection ignores the group size and bits.
        UnifiedLinear::from_weights(weights, prefix, 64, 4)
            .map(|l| Self(Projection::Plain(l)))
            .map_err(anyhow::Error::msg)
            .with_context(context)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.0 {
            Projection::Plain(linear) => linear.forward(x),
            Projection::Quantized(linear) => linear.forward(x),
        }
    }

    /// Whether this projection runs a quantized matmul.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(self.0, Projection::Quantized(_))
    }
}

/// Copy a tensor by key, naming it when absent.
pub fn tensor(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("Mage-Flow weight not found: {key}"))
}

/// `nn.RMSNorm(eps)` loaded from `<key>`.
pub fn rms_norm(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm> {
    Ok(RMSNorm::new(tensor(weights, key)?, eps))
}

/// `nn.LayerNorm(dims, eps, affine)`: affine reads `<prefix>.weight` and
/// `<prefix>.bias`.
pub struct LayerNorm {
    weight: Option<UniquePtr<MlxArray>>,
    bias: Option<UniquePtr<MlxArray>>,
    eps: f32,
}

impl LayerNorm {
    /// `affine=False`.
    #[must_use]
    pub fn plain(eps: f32) -> Self {
        Self {
            weight: None,
            bias: None,
            eps,
        }
    }

    pub fn affine(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: Some(tensor(weights, &format!("{prefix}.weight"))?),
            bias: Some(tensor(weights, &format!("{prefix}.bias"))?),
            eps,
        })
    }

    /// `nn.LayerNorm`, computed as `mx.fast.layer_norm` does (f32 statistics
    /// and affine, one cast back) but from separate ops.
    ///
    /// The fused CUDA kernel of MLX (`backend/cuda/layer_norm.cu`, also in
    /// mlx 0.32.4) reuses one shared-memory buffer for its mean and variance
    /// block reductions with no barrier between them, so a fast warp can
    /// overwrite a partial sum another warp has not read yet. On 384-wide bf16
    /// rows (the VAE encoder's `norm_out`) a few rows per call come out wrong
    /// and the edit path's reference latents changed by about 1e-1 between
    /// identical calls, in upstream as here. The composed form is
    /// deterministic and matches the fused kernel's good rows to one bf16 ulp.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dt = mlxcel_core::array_dtype(x);
        let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let centered = mlxcel_core::subtract(&xf, &mlxcel_core::mean_axis(&xf, -1, true));
        let var = mlxcel_core::mean_axis(&mlxcel_core::square(&centered), -1, true);
        let eps = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::dtype::FLOAT32);
        let mut y = mlxcel_core::multiply(
            &centered,
            &mlxcel_core::rsqrt(&mlxcel_core::add(&var, &eps)),
        );
        if let Some(w) = &self.weight {
            y = mlxcel_core::multiply(&y, &mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32));
        }
        if let Some(b) = &self.bias {
            y = mlxcel_core::add(&y, &mlxcel_core::astype(b, mlxcel_core::dtype::FLOAT32));
        }
        mlxcel_core::astype(&y, dt)
    }
}

/// A scalar in `like`'s dtype, so an op with it keeps that dtype (a Python
/// float literal in upstream).
#[must_use]
pub fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// `mx.split(x, n, axis=-1)`.
#[must_use]
pub fn split_last(x: &MlxArray, n: i32) -> Vec<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape.len() - 1;
    let width = shape[last] / n;
    (0..n)
        .map(|i| {
            let mut starts = vec![0; shape.len()];
            let mut stops = shape.clone();
            starts[last] = i * width;
            stops[last] = (i + 1) * width;
            mlxcel_core::slice(x, &starts, &stops)
        })
        .collect()
}

/// `x * (1 + scale) + shift` in `x`'s dtype; `scale` and `shift` broadcast.
#[must_use]
pub fn modulate(x: &MlxArray, shift: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let one = scalar_like(1.0, scale);
    let scaled = mlxcel_core::multiply(x, &mlxcel_core::add(&one, scale));
    mlxcel_core::add(&scaled, shift)
}

/// `x[:, None, :]` on `[B, D]`.
#[must_use]
pub fn seq_axis(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(x, 1)
}

/// `x[:, None, None]` on `[B, C]`: broadcast over NHWC spatial axes.
#[must_use]
pub fn spatial_axes(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(&mlxcel_core::expand_dims(x, 1), 1)
}

/// MLX `nn.gelu` (exact erf form) in the activation dtype:
/// `x * (1 + erf(x / sqrt(2))) / 2`, its Python scalars weak-typed to `x`.
///
/// The shared `mlxcel_core::gelu` evaluates the constants in f32 and casts
/// back, which rounds a bf16 activation differently from upstream.
#[must_use]
pub fn gelu_erf(x: &MlxArray) -> UniquePtr<MlxArray> {
    let erf = mlxcel_core::erf(&mlxcel_core::divide(
        x,
        &scalar_like(std::f32::consts::SQRT_2, x),
    ));
    let scaled = mlxcel_core::multiply(x, &mlxcel_core::add(&scalar_like(1.0, x), &erf));
    mlxcel_core::divide(&scaled, &scalar_like(2.0, x))
}

/// `x * sigmoid(x)`, upstream's `_nonlinearity` and MLX `nn.silu`.
#[must_use]
pub fn silu(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::multiply(x, &mlxcel_core::sigmoid(x))
}
