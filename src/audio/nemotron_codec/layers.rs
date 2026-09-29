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

//! Codec building blocks: convolutions, channel LayerNorm, ConvNeXt block.
//!
//! Ports `ChannelLayerNorm` and `ConvNeXtBlock1d` from
//! `mlx_audio/codec/models/nemotron_voicechat/codec.py`, plus the weight
//! layout handling of its `sanitize`. The released checkpoint stores codec
//! convolutions in PyTorch layout (`Conv1d` `[out, in/g, K]`,
//! `ConvTranspose1d` `[in, out, K]`); the loaders here accept that layout or
//! the MLX layout (`[out, K, in/g]`) by shape, so an already converted map
//! loads unchanged.
//!
//! Scalars follow MLX's weak typing: a Python float in the reference adopts
//! the dtype of the array it meets, so every scalar here is materialized in
//! the operand dtype ([`scalar_like`]). That keeps the bf16 decoder numerics
//! identical to the reference instead of silently promoting to f32.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::cache::{CacheKey, CausalConv1dCache};

/// Scalar with the dtype of `like` (MLX weak-scalar semantics).
pub(crate) fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// Sum over the last axis with MLX's native reduction in the input dtype.
///
/// `mlxcel_core::sum_axis` / `mean_axis` widen half-precision inputs to f32
/// and round once, which differs from `mx.sum` on bf16 in about 20% of the
/// codebook norms and flips residual-VQ codes. `einsum` lowers a pure
/// reduction to the native `sum`, matching the reference bit for bit.
pub(crate) fn native_sum_last(x: &MlxArray, keepdims: bool) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let letters = &"abcdefgh"[..shape.len().min(8)];
    let spec = format!("{letters}->{}", &letters[..letters.len().saturating_sub(1)]);
    let operands = [x as *const MlxArray];
    // SAFETY: the operand pointer borrows `x`, which outlives the call.
    let summed = unsafe { mlxcel_core::einsum(&spec, &operands) };
    if keepdims {
        let mut kept = shape;
        if let Some(last) = kept.last_mut() {
            *last = 1;
        }
        mlxcel_core::reshape(&summed, &kept)
    } else {
        summed
    }
}

/// Mean over the last axis matching `mx.mean` for every dtype: native
/// `sum * (1 / N)` for half precision (see [`native_sum_last`]).
pub(crate) fn native_mean_last(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(x);
    if dtype != mlxcel_core::dtype::BFLOAT16 && dtype != mlxcel_core::dtype::FLOAT16 {
        return mlxcel_core::mean_axis(x, -1, true);
    }
    let n = mlxcel_core::array_shape(x)
        .last()
        .copied()
        .unwrap_or(1)
        .max(1);
    let sum = native_sum_last(x, true);
    mlxcel_core::multiply(&sum, &scalar_like(1.0 / n as f32, x))
}

pub(crate) fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("codec weight not found: {key}"))
}

fn to_i32(value: usize, what: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("codec: {what} {value} exceeds i32 range"))
}

/// Load a `Conv1d` kernel as MLX `[out, K, in/g]`, transposing a PyTorch
/// `[out, in/g, K]` kernel. A shape that fits both (K == in/g) is treated as
/// the PyTorch layout, as the reference `sanitize` does unconditionally.
fn conv1d_kernel(
    weights: &WeightMap,
    key: &str,
    out: usize,
    in_per_group: usize,
    kernel: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let w = copy_weight(weights, key)?;
    let shape = mlxcel_core::array_shape(&w);
    let (o, k, i) = (
        to_i32(out, "channels")?,
        to_i32(kernel, "kernel")?,
        to_i32(in_per_group, "channels")?,
    );
    if shape == [o, i, k] {
        Ok(mlxcel_core::transpose_axes(&w, &[0, 2, 1]))
    } else if shape == [o, k, i] {
        Ok(w)
    } else {
        Err(format!(
            "codec weight {key} has shape {shape:?}, expected MLX [{o}, {k}, {i}] or \
             PyTorch [{o}, {i}, {k}]"
        ))
    }
}

/// Load a `ConvTranspose1d` kernel as MLX `[out, K, in]`, transposing a
/// PyTorch `[in, out, K]` kernel.
fn conv_transpose1d_kernel(
    weights: &WeightMap,
    key: &str,
    input: usize,
    out: usize,
    kernel: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let w = copy_weight(weights, key)?;
    let shape = mlxcel_core::array_shape(&w);
    let (i, o, k) = (
        to_i32(input, "channels")?,
        to_i32(out, "channels")?,
        to_i32(kernel, "kernel")?,
    );
    if shape == [i, o, k] {
        Ok(mlxcel_core::transpose_axes(&w, &[1, 2, 0]))
    } else if shape == [o, k, i] {
        Ok(w)
    } else {
        Err(format!(
            "codec weight {key} has shape {shape:?}, expected MLX [{o}, {k}, {i}] or \
             PyTorch [{i}, {o}, {k}]"
        ))
    }
}

/// `nn.Conv1d` with zero padding (the codec never pads inside the conv).
pub(crate) struct Conv1d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    groups: i32,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self, String> {
        if groups == 0 || !in_channels.is_multiple_of(groups) {
            return Err(format!(
                "codec conv {prefix}: {in_channels} channels not divisible by {groups} groups"
            ));
        }
        let weight = conv1d_kernel(
            weights,
            &format!("{prefix}.weight"),
            out_channels,
            in_channels / groups,
            kernel,
        )?;
        let bias = if bias {
            Some(copy_weight(weights, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride: to_i32(stride, "stride")?,
            groups: to_i32(groups, "groups")?,
        })
    }

    /// `x`: `[B, T, C_in]` -> `[B, T_out, C_out]`.
    pub(crate) fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let y = mlxcel_core::try_conv1d(x, &self.weight, self.stride, 0, 1, self.groups)
            .map_err(|e| format!("codec conv1d failed: {e}"))?;
        Ok(match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        })
    }
}

/// `nn.ConvTranspose1d` without bias or padding (decoder upsampling).
pub(crate) struct ConvTranspose1d {
    weight: UniquePtr<MlxArray>,
    stride: i32,
}

impl ConvTranspose1d {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: conv_transpose1d_kernel(
                weights,
                &format!("{prefix}.weight"),
                in_channels,
                out_channels,
                kernel,
            )?,
            stride: to_i32(stride, "stride")?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::conv_transpose1d(x, &self.weight, self.stride, 0, 1, 0, 1)
    }
}

/// `nn.gelu` (exact erf form) with weak scalars: `x * (1 + erf(x / sqrt2)) / 2`.
pub(crate) fn gelu_exact(x: &MlxArray) -> UniquePtr<MlxArray> {
    let scaled = mlxcel_core::divide(x, &scalar_like(std::f32::consts::SQRT_2, x));
    let erf = mlxcel_core::erf(&scaled);
    let one_plus = mlxcel_core::add(&erf, &scalar_like(1.0, x));
    let prod = mlxcel_core::multiply(x, &one_plus);
    mlxcel_core::divide(&prod, &scalar_like(2.0, x))
}

/// LayerNorm over the channel (last) axis of `[B, T, C]`, eps 1e-6.
pub(crate) struct ChannelLayerNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    eps: f32,
}

impl ChannelLayerNorm {
    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: copy_weight(weights, &format!("{prefix}.bias"))?,
            eps: 1e-6,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mean = native_mean_last(x);
        let var = mlxcel_core::var_axis(x, -1, true, 0);
        let centered = mlxcel_core::subtract(x, &mean);
        let inv = mlxcel_core::rsqrt(&mlxcel_core::add(&var, &scalar_like(self.eps, &var)));
        let normalized = mlxcel_core::multiply(&centered, &inv);
        let scaled = mlxcel_core::multiply(&normalized, &self.weight);
        mlxcel_core::add(&scaled, &self.bias)
    }
}

/// Causal ConvNeXt block: depthwise conv (left-padded K-1), channel LN,
/// pointwise expand x4, GELU, pointwise project, residual.
pub(crate) struct ConvNeXtBlock1d {
    kernel_size: usize,
    layer_id: usize,
    dwconv: Conv1d,
    norm: ChannelLayerNorm,
    pwconv1: Conv1d,
    pwconv2: Conv1d,
}

impl ConvNeXtBlock1d {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        kernel_size: usize,
        layer_id: usize,
    ) -> Result<Self, String> {
        let hidden = 4 * channels;
        Ok(Self {
            kernel_size,
            layer_id,
            dwconv: Conv1d::from_weights(
                weights,
                &format!("{prefix}.dwconv"),
                channels,
                channels,
                kernel_size,
                1,
                channels,
                true,
            )?,
            norm: ChannelLayerNorm::from_weights(weights, &format!("{prefix}.norm"))?,
            pwconv1: Conv1d::from_weights(
                weights,
                &format!("{prefix}.pwconv1"),
                channels,
                hidden,
                1,
                1,
                1,
                true,
            )?,
            pwconv2: Conv1d::from_weights(
                weights,
                &format!("{prefix}.pwconv2"),
                hidden,
                channels,
                1,
                1,
                1,
                true,
            )?,
        })
    }

    /// `x`: `[B, T, C]`. With a cache, the left context comes from the
    /// previous call instead of zero padding.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: Option<&mut CausalConv1dCache>,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let left = self.kernel_size - 1;
        let hidden = match cache {
            None => {
                let pad = to_i32(left, "padding")?;
                mlxcel_core::pad(x, &[0, 0, pad, 0, 0, 0], 0.0)
            }
            Some(cache) => cache.update(x, CacheKey::Block(self.layer_id), left, flush)?,
        };
        let hidden = self.dwconv.forward(&hidden)?;
        let hidden = self.norm.forward(&hidden);
        let hidden = self.pwconv1.forward(&hidden)?;
        let hidden = gelu_exact(&hidden);
        let hidden = self.pwconv2.forward(&hidden)?;
        Ok(mlxcel_core::add(x, &hidden))
    }
}
