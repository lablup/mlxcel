// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Style-adaptive residual blocks shared by the prosody predictor and decoder.
//!
//! Two distinct block types from StyleTTS2 / iSTFTNet:
//! - [`AdaIn1d`]: instance norm whose affine `(gamma, beta)` is produced from
//!   the 128-d style vector by a learned linear map (`fc: 128 -> 2C`).
//! - [`AdainResBlk1d`]: a residual block with `LeakyReLU(0.2)`, two 3-tap convs,
//!   two AdaIN norms, an optional learned `1x1` shortcut, and an optional `2x`
//!   transposed-conv upsampling pool (with a left pad mimicking PyTorch
//!   `output_padding=1`). Used by the F0/N stacks and the decoder encode/decode.
//!
//! The Snake-activated `AdaINResBlock1` used by the iSTFTNet generator MRF lives
//! in [`super::decoder`] because it is generator-local.

use mlxcel_core::{MlxArray, UniquePtr};

use super::ops;
use super::weights::{ConvWn, Weights};

const ADAIN_EPS: f32 = 1e-5;
const LEAKY: f32 = 0.2;

/// Style-conditioned instance norm: `(1 + gamma) * instnorm(x) + beta` where
/// `[gamma, beta] = fc(style)`.
pub(crate) struct AdaIn1d {
    fc_w: UniquePtr<MlxArray>,
    fc_b: UniquePtr<MlxArray>,
    channels: i32,
}

impl AdaIn1d {
    pub(crate) fn load(w: &Weights, prefix: &str, channels: i32) -> Result<Self, String> {
        let (fc_w, fc_b) = w.linear(&format!("{prefix}.fc"))?;
        let fc_b = fc_b.ok_or_else(|| format!("kokoro: {prefix}.fc.bias missing"))?;
        Ok(Self {
            fc_w,
            fc_b,
            channels,
        })
    }

    /// Apply to `(C, T)` activation with the `(1, 128)` style row.
    pub(crate) fn forward(
        &self,
        x: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = ops::linear(style, &self.fc_w, Some(&self.fc_b)); // (1, 2C)
        let h = ops::reshape(&h, &[2 * self.channels, 1]); // (2C, 1)
        let gamma = ops::slice(&h, &[0, 0], &[self.channels, 1]);
        let beta = ops::slice(&h, &[self.channels, 0], &[2 * self.channels, 1]);
        let normed = ops::instance_norm_ct(x, ADAIN_EPS);
        let scaled = ops::mul(&ops::add_scalar(&gamma, 1.0), &normed);
        ops::add(&scaled, &beta)
    }
}

/// Residual block with AdaIN norms, two 3-tap convs, optional `1x1` shortcut,
/// and optional `2x` upsampling.
pub(crate) struct AdainResBlk1d {
    conv1: ConvWn,
    conv2: ConvWn,
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    conv1x1: Option<ConvWn>,
    pool: Option<ConvWn>,
    dim_in: i32,
    upsample: bool,
}

impl AdainResBlk1d {
    /// Load from `prefix`. `learned_sc` forces the `1x1` shortcut even when
    /// `dim_in == dim_out` (the decoder's decode blocks set this).
    pub(crate) fn load(
        w: &Weights,
        prefix: &str,
        dim_in: i32,
        dim_out: i32,
        upsample: bool,
        learned_sc: bool,
    ) -> Result<Self, String> {
        let use_sc = learned_sc || dim_in != dim_out;
        let conv1 = ConvWn::load(w, &format!("{prefix}.conv1"), 1, 1, 1, 1, true)?;
        let conv2 = ConvWn::load(w, &format!("{prefix}.conv2"), 1, 1, 1, 1, true)?;
        let norm1 = AdaIn1d::load(w, &format!("{prefix}.norm1"), dim_in)?;
        let norm2 = AdaIn1d::load(w, &format!("{prefix}.norm2"), dim_out)?;
        let conv1x1 = if use_sc {
            Some(ConvWn::load(
                w,
                &format!("{prefix}.conv1x1"),
                1,
                0,
                1,
                1,
                false,
            )?)
        } else {
            None
        };
        let pool = if upsample {
            // ConvTranspose1d(dim_in, dim_in, k=3, s=2, groups=dim_in, pad=1, out_pad=1)
            Some(ConvWn::load_transposed(
                w,
                &format!("{prefix}.pool"),
                2,
                1,
                1,
                dim_in,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            pool,
            dim_in,
            upsample,
        })
    }

    /// Forward over `(C, T)` with the style row, returning `(C_out, T')`.
    pub(crate) fn forward(
        &self,
        x: &UniquePtr<MlxArray>,
        style: &UniquePtr<MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Residual branch.
        let mut xr = self.norm1.forward(x, style);
        xr = ops::leaky_relu(&xr, LEAKY);
        if let Some(pool) = &self.pool {
            xr = pool.forward(&xr);
        }
        xr = self.conv1.forward(&xr);
        xr = self.norm2.forward(&xr, style);
        xr = ops::leaky_relu(&xr, LEAKY);
        xr = self.conv2.forward(&xr);

        // Shortcut branch.
        let mut xs = if self.upsample {
            ops::upsample_nearest_ct(x, 2)
        } else {
            mlxcel_core::copy(x.as_ref().expect("kokoro: null residual input"))
        };
        if let Some(c) = &self.conv1x1 {
            xs = c.forward(&xs);
        }
        let _ = self.dim_in;

        // Align lengths defensively (upsample paths can differ by one frame).
        let (xr, xs) = align_time(&xr, &xs);
        ops::mul_scalar(&ops::add(&xr, &xs), std::f32::consts::FRAC_1_SQRT_2)
    }
}

/// Trim two `(C, T)` activations to their shared minimum time length.
pub(crate) fn align_time(
    a: &UniquePtr<MlxArray>,
    b: &UniquePtr<MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let sa = ops::shape(a);
    let sb = ops::shape(b);
    let ta = *sa.last().unwrap_or(&0);
    let tb = *sb.last().unwrap_or(&0);
    let t = ta.min(tb);
    let ca = sa[0];
    let cb = sb[0];
    let at = if ta > t {
        ops::slice(a, &[0, 0], &[ca, t])
    } else {
        mlxcel_core::copy(a.as_ref().expect("kokoro: null a"))
    };
    let bt = if tb > t {
        ops::slice(b, &[0, 0], &[cb, t])
    } else {
        mlxcel_core::copy(b.as_ref().expect("kokoro: null b"))
    };
    (at, bt)
}
