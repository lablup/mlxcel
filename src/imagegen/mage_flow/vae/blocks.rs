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

//! Mage-VAE blocks shared by its encoder and decoder: the timestep embedder,
//! `DiCoBlock` and `EncoderDiCoBlock` (`vae.py`, mlx-vlm `73d366ec`).
//!
//! Activations are NHWC. Every op runs in the activation dtype as upstream's
//! MLX layers do; the channel-attention mean is MLX's native-dtype mean.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use crate::imagegen::mage_flow::layers::{
    Dense, LayerNorm, gelu_erf, modulate, scalar_like, spatial_axes, split_last,
};
use crate::imagegen::vae::Conv2d;

/// Epsilon of every VAE norm.
pub(super) const EPS: f32 = 1e-6;

/// `TimestepEmbedder` (`frequency_size` 256): plain f32 frequencies,
/// `[cos, sin]`, cast to `dt`, then `linear_2(silu(linear_1(.)))`.
pub(super) struct TimestepEmbedder {
    linear_1: Dense,
    linear_2: Dense,
}

impl TimestepEmbedder {
    pub(super) fn load(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            linear_1: Dense::load(w, &format!("{p}.linear_1"), None)?,
            linear_2: Dense::load(w, &format!("{p}.linear_2"), None)?,
        })
    }

    /// Embedding of `t = 0` for a batch of `batch`, in `dt` (the only time
    /// the VAE ever sees).
    pub(super) fn zero(&self, batch: i32, dt: i32) -> UniquePtr<MlxArray> {
        let half = 128;
        let f32_scalar = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
        let freqs = mlxcel_core::exp(&mlxcel_core::divide(
            &mlxcel_core::multiply(
                &f32_scalar(-(10000.0_f64.ln()) as f32),
                &mlxcel_core::arange_f32(0.0, half as f32, 1.0),
            ),
            &f32_scalar(half as f32),
        ));
        let t = mlxcel_core::astype(&mlxcel_core::zeros(&[batch, 1], dt), dtype::FLOAT32);
        let args = mlxcel_core::multiply(&t, &mlxcel_core::expand_dims(&freqs, 0));
        let emb = mlxcel_core::concatenate(&mlxcel_core::cos(&args), &mlxcel_core::sin(&args), -1);
        let emb = mlxcel_core::astype(&emb, dt);
        self.linear_2
            .forward(&mlxcel_core::silu(&self.linear_1.forward(&emb)))
    }
}

/// `mx.mean(x, axis=(1, 2), keepdims=True)` on NHWC in `x`'s dtype: the
/// f32-accumulated sum rounded once to that dtype, times `dtype(1 / n)`.
///
/// This is MLX's `sum(x) * (1 / n)` with an f32 accumulator. The CUDA build of
/// the upstream oracle (mlx 0.32.4) accumulates a bf16 sum in bf16, which moves
/// some channel means by one bf16 ulp; with an f32-accumulated mean the
/// oracle's encoder output is reproduced exactly (measured during the port).
fn spatial_mean(x: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let flat = mlxcel_core::reshape(x, &[s[0], s[1] * s[2], s[3]]);
    let sum = mlxcel_core::sum_axis(&flat, 1, true);
    let inverse = (1.0 / f64::from(s[1] * s[2])) as f32;
    let mean = mlxcel_core::multiply(&sum, &scalar_like(inverse, x));
    mlxcel_core::reshape(&mean, &[s[0], 1, 1, s[3]])
}

/// The convolution trunk both DiCo blocks share.
struct DiCoConvs {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    ca_conv: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
}

impl DiCoConvs {
    fn load(w: &WeightMap, p: &str, channels: i32) -> Result<Self> {
        let conv = |name: &str| Conv2d::load(w, &format!("{p}.{name}"), 1, 0);
        Ok(Self {
            conv1: conv("conv1")?,
            conv2: Conv2d::load_grouped(w, &format!("{p}.conv2"), 1, 1, channels)?,
            conv3: conv("conv3")?,
            ca_conv: conv("ca_conv")?,
            conv4: conv("conv4")?,
            conv5: conv("conv5")?,
        })
    }

    /// `gelu(conv2(conv1(x)))` with sigmoid channel attention applied.
    fn attend(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = gelu_erf(&self.conv2.forward(&self.conv1.forward(x)));
        let attention = mlxcel_core::sigmoid(&self.ca_conv.forward(&spatial_mean(&x)));
        mlxcel_core::multiply(&x, &attention)
    }

    fn mlp(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.conv5.forward(&gelu_erf(&self.conv4.forward(x)))
    }
}

/// `DiCoBlock`: AdaLN-modulated DiCo block reading a `[B, C]` conditioning.
pub(super) struct DiCoBlock {
    convs: DiCoConvs,
    norm: LayerNorm,
    ada_ln: Dense,
}

impl DiCoBlock {
    pub(super) fn load(w: &WeightMap, p: &str, channels: i32) -> Result<Self> {
        Ok(Self {
            convs: DiCoConvs::load(w, p, channels)?,
            norm: LayerNorm::plain(EPS),
            ada_ln: Dense::load(w, &format!("{p}.adaLN_modulation.linear"), None)?,
        })
    }

    pub(super) fn forward(
        &self,
        inputs: &MlxArray,
        conditioning: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mods = split_last(&self.ada_ln.forward(&mlxcel_core::silu(conditioning)), 6);
        let m: Vec<UniquePtr<MlxArray>> = mods.iter().map(|t| spatial_axes(t)).collect();
        let (shift_attn, scale_attn, gate_attn) = (&m[0], &m[1], &m[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&m[3], &m[4], &m[5]);
        let x = modulate(&self.norm.forward(inputs), shift_attn, scale_attn);
        let x = self.convs.conv3.forward(&self.convs.attend(&x));
        let x = mlxcel_core::add(inputs, &mlxcel_core::multiply(gate_attn, &x));
        let residual = modulate(&self.norm.forward(&x), shift_mlp, scale_mlp);
        let residual = self.convs.mlp(&residual);
        mlxcel_core::add(&x, &mlxcel_core::multiply(gate_mlp, &residual))
    }
}

/// `EncoderDiCoBlock`: the unconditioned DiCo block with affine norms.
pub(super) struct EncoderDiCoBlock {
    convs: DiCoConvs,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderDiCoBlock {
    pub(super) fn load(w: &WeightMap, p: &str, channels: i32) -> Result<Self> {
        Ok(Self {
            convs: DiCoConvs::load(w, p, channels)?,
            norm1: LayerNorm::affine(w, &format!("{p}.norm1"), EPS)?,
            norm2: LayerNorm::affine(w, &format!("{p}.norm2"), EPS)?,
        })
    }

    pub(super) fn forward(&self, inputs: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.convs.attend(&self.norm1.forward(inputs));
        let x = mlxcel_core::add(inputs, &self.convs.conv3.forward(&x));
        mlxcel_core::add(&x, &self.convs.mlp(&self.norm2.forward(&x)))
    }
}
