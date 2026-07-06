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
// Portions of this file are derived from mlx-vlm
// (https://github.com/Blaizzy/mlx-vlm), Copyright 2025 Prince Canuma,
// licensed under the MIT License. See the top-level NOTICE file for the
// attribution carried forward under the MIT License.

//! Convolutional building blocks of the Qwen3-Omni code2wav vocoder:
//! SnakeBeta activation, causal (transposed) 1D convolutions, ConvNeXt
//! upsample blocks, and the BigVGAN-style decoder residual units. All blocks
//! operate on the MLX channels-last conv layout `[B, L, C]`.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/code2wav.py>.
//!
//! Used by: Qwen3-Omni MoE code2wav vocoder (code2wav.rs).

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Read an MLX array back into a host `Vec<f32>` (cast, materialize through
/// the fallible FFI readback, parse little-endian).
pub(super) fn to_vec_f32(a: &MlxArray) -> Result<Vec<f32>, String> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    let bytes = mlxcel_core::try_array_to_raw_bytes(&f)
        .map_err(|e| format!("code2wav eval failed: {e}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect())
}

pub(super) fn get_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Qwen3-Omni code2wav weight missing: {key}"))
}

/// SnakeBeta activation: `x + sin(x * exp(alpha))^2 / (exp(beta) + 1e-9)`,
/// with per-channel log-scale `alpha` / `beta` broadcast over `[B, L, C]`.
/// `exp_alpha` and the denominator are precomputed at load (they are
/// constants; MLX materializes them once at the first eval).
pub(super) struct SnakeBeta {
    exp_alpha: UniquePtr<MlxArray>,
    denom: UniquePtr<MlxArray>,
}

impl SnakeBeta {
    pub(super) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let alpha = get_weight(weights, &format!("{prefix}.alpha"))?;
        let beta = get_weight(weights, &format!("{prefix}.beta"))?;
        let exp_alpha = mlxcel_core::exp(&alpha);
        let exp_beta = mlxcel_core::exp(&beta);
        // Keep the epsilon in the parameter dtype (weak-scalar semantics of
        // the reference: adding a Python float does not promote bf16).
        let eps = mlxcel_core::from_slice_f32(&[1e-9], &[1]);
        let eps = mlxcel_core::astype(&eps, mlxcel_core::array_dtype(&exp_beta));
        let denom = mlxcel_core::add(&exp_beta, &eps);
        Ok(Self { exp_alpha, denom })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let ax = mlxcel_core::multiply(x, &self.exp_alpha);
        let s = mlxcel_core::sin(&ax);
        let s2 = mlxcel_core::multiply(&s, &s);
        let frac = mlxcel_core::divide(&s2, &self.denom);
        mlxcel_core::add(x, &frac)
    }
}

/// Left-padded ("causal") 1D convolution over `[B, L, C]`, mirroring the
/// reference `CausalConvNet` (left pad `k_eff - stride`, plus right padding
/// to complete the last frame; zero for the stride-1 convs used here).
pub(super) struct CausalConv1d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    dilation: i32,
    groups: i32,
    kernel_eff: i32,
    padding: i32,
}

impl CausalConv1d {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kernel: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Self, String> {
        let kernel_eff = (kernel - 1) * dilation + 1;
        Ok(Self {
            weight: get_weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|b| mlxcel_core::copy(b)),
            stride,
            dilation,
            groups,
            kernel_eff,
            padding: kernel_eff - stride,
        })
    }

    /// Right padding needed so the final (partial) frame is completed:
    /// `ideal_length - length` with
    /// `ideal = (ceil(n_frames) - 1) * stride + (k_eff - padding)`.
    fn extra_padding(&self, length: i32) -> i32 {
        let numer = length - self.kernel_eff + self.padding;
        let n_frames_ceil =
            numer.div_euclid(self.stride) + i32::from(numer.rem_euclid(self.stride) != 0) + 1;
        let ideal = (n_frames_ceil - 1) * self.stride + (self.kernel_eff - self.padding);
        (ideal - length).max(0)
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let length = mlxcel_core::array_shape(x)[1];
        let extra = self.extra_padding(length);
        let padded = mlxcel_core::pad(x, &[0, 0, self.padding, extra, 0, 0], 0.0);
        let mut y = mlxcel_core::conv1d(
            &padded,
            &self.weight,
            self.stride,
            0,
            self.dilation,
            self.groups,
        );
        if let Some(bias) = &self.bias {
            y = mlxcel_core::add(&y, bias);
        }
        y
    }
}

/// Causal transposed 1D convolution over `[B, L, C]`: full transposed conv
/// then trim `kernel - stride` positions from the end, so `L_out = L * stride`.
pub(super) struct CausalTransConv1d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    right_trim: i32,
}

impl CausalTransConv1d {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kernel: i32,
        stride: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: get_weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|b| mlxcel_core::copy(b)),
            stride,
            right_trim: kernel - stride,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut y = mlxcel_core::conv_transpose1d(x, &self.weight, self.stride, 0, 1, 0, 1);
        if let Some(bias) = &self.bias {
            y = mlxcel_core::add(&y, bias);
        }
        let shape = mlxcel_core::array_shape(&y);
        mlxcel_core::slice(
            &y,
            &[0, 0, 0],
            &[shape[0], shape[1] - self.right_trim, shape[2]],
        )
    }
}

/// ConvNeXt block: depthwise causal conv (k=7), LayerNorm (eps 1e-6),
/// pointwise 4x expansion with exact GELU, gamma scale, residual.
pub(super) struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: UnifiedLinear,
    pwconv2: UnifiedLinear,
    gamma: UniquePtr<MlxArray>,
}

impl ConvNeXtBlock {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            dwconv: CausalConv1d::from_weights(
                weights,
                &format!("{prefix}.dwconv.conv"),
                7,
                1,
                1,
                dim as i32,
            )?,
            norm: LayerNorm::new(
                get_weight(weights, &format!("{prefix}.norm.weight"))?,
                weights
                    .get(&format!("{prefix}.norm.bias"))
                    .map(|b| mlxcel_core::copy(b)),
                1e-6,
            ),
            pwconv1: UnifiedLinear::from_weights(weights, &format!("{prefix}.pwconv1"), gs, bits)?,
            pwconv2: UnifiedLinear::from_weights(weights, &format!("{prefix}.pwconv2"), gs, bits)?,
            gamma: get_weight(weights, &format!("{prefix}.gamma"))?,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.dwconv.forward(x);
        let h = self.norm.forward(&h);
        let h = self.pwconv1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        let h = self.pwconv2.forward(&h);
        let h = mlxcel_core::multiply(&self.gamma, &h);
        mlxcel_core::add(x, &h)
    }
}

/// Decoder residual unit: SnakeBeta, dilated causal conv (k=7), SnakeBeta,
/// pointwise causal conv (k=1), residual.
struct ResUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResUnit {
    fn from_weights(weights: &WeightMap, prefix: &str, dilation: i32) -> Result<Self, String> {
        Ok(Self {
            act1: SnakeBeta::from_weights(weights, &format!("{prefix}.act1"))?,
            conv1: CausalConv1d::from_weights(
                weights,
                &format!("{prefix}.conv1.conv"),
                7,
                1,
                dilation,
                1,
            )?,
            act2: SnakeBeta::from_weights(weights, &format!("{prefix}.act2"))?,
            conv2: CausalConv1d::from_weights(
                weights,
                &format!("{prefix}.conv2.conv"),
                1,
                1,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.act1.forward(x);
        let h = self.conv1.forward(&h);
        let h = self.act2.forward(&h);
        let h = self.conv2.forward(&h);
        mlxcel_core::add(&h, x)
    }
}

/// One decoder upsample block: SnakeBeta, transposed conv (k = 2 * rate,
/// stride = rate, halving the channel count), three residual units with
/// dilations 1 / 3 / 9.
pub(super) struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalTransConv1d,
    res_units: Vec<ResUnit>,
}

impl DecoderBlock {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        rate: i32,
    ) -> Result<Self, String> {
        let mut res_units = Vec::with_capacity(3);
        for (i, dilation) in [1, 3, 9].into_iter().enumerate() {
            res_units.push(ResUnit::from_weights(
                weights,
                &format!("{prefix}.block.{}", i + 2),
                dilation,
            )?);
        }
        Ok(Self {
            snake: SnakeBeta::from_weights(weights, &format!("{prefix}.block.0"))?,
            upsample: CausalTransConv1d::from_weights(
                weights,
                &format!("{prefix}.block.1.conv"),
                2 * rate,
                rate,
            )?,
            res_units,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.snake.forward(x);
        h = self.upsample.forward(&h);
        for unit in &self.res_units {
            h = unit.forward(&h);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn causal_conv_padding_arithmetic() {
        // Stride-1 conv: left pad k_eff - 1, no extra right padding, so the
        // output length equals the input length.
        // k=7 d=1: k_eff 7, padding 6. k=7 d=9: k_eff 55, padding 54.
        for (kernel, dilation) in [(7, 1), (7, 3), (7, 9), (1, 1), (2, 1)] {
            let kernel_eff = (kernel - 1) * dilation + 1;
            let padding = kernel_eff - 1; // stride 1
            for length in [1i32, 13, 300, 325] {
                let numer = length - kernel_eff + padding;
                let n_frames_ceil = numer.div_euclid(1) + i32::from(numer.rem_euclid(1) != 0) + 1;
                let ideal = (n_frames_ceil - 1) + (kernel_eff - padding);
                assert_eq!(ideal - length, 0, "k={kernel} d={dilation} len={length}");
            }
        }
    }

    #[test]
    fn snake_beta_with_zero_params_is_x_plus_sin_squared() {
        // alpha = beta = 0 => exp = 1 => y = x + sin(x)^2 / (1 + 1e-9).
        let mut weights = mlxcel_core::weights::WeightMap::new();
        weights.insert(
            "act.alpha".to_string(),
            mlxcel_core::zeros(&[4], mlxcel_core::dtype::FLOAT32),
        );
        weights.insert(
            "act.beta".to_string(),
            mlxcel_core::zeros(&[4], mlxcel_core::dtype::FLOAT32),
        );
        let snake = super::SnakeBeta::from_weights(&weights, "act").unwrap();

        let xs = [-2.0f32, -0.5, 0.0, 1.5];
        let x = mlxcel_core::from_slice_f32(&xs, &[1, 1, 4]);
        let y = snake.forward(&x);
        let y = super::to_vec_f32(&y).unwrap();
        for (i, &xi) in xs.iter().enumerate() {
            let expected = xi + xi.sin().powi(2) / (1.0 + 1e-9);
            assert!(
                (y[i] - expected).abs() < 1e-5,
                "snake({xi}) = {} expected {expected}",
                y[i]
            );
        }
    }
}
