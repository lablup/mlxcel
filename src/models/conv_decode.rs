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

//! Shared single-step (decode) depthwise short-conv fast path.
//!
//! MLX 0.32.1's CUDA backend dispatches a single-output-position (L=1) bf16
//! depthwise `conv1d` to cuDNN's generic per-channel engine
//! (`convolve_common_engine_float_NHWC`), which launches one kernel per channel
//! and dominates decode. Issue #748 / PR #751 first hit this on LFM2's
//! `ShortConv` and replaced the L=1 step with a broadcast multiply plus an axis
//! sum against a decode weight precomputed at load. The identical rolling-window
//! pad + depthwise `conv1d` call pattern executed every decode step exists in the
//! SSM / hybrid families (mamba2, falcon-h1, granite-4.0-h, mamba, jamba, plamo2,
//! nemotron-h, kimi linear, qwen3.5 linear-attention, …), so issue #752 lifts the
//! two #751 helpers here for reuse.
//!
//! Both helpers are pure, depend only on `mlxcel-core`, and are checkpoint-free,
//! so the parity tests in `conv_decode_tests.rs` (and the per-family test files)
//! pin the elementwise step against `conv1d` without loading a model.

use mlxcel_core::{MlxArray, UniquePtr};

/// Materialize the time-major weight used by the decode fast path.
///
/// `conv_weight` is `[channels, kernel, 1]` (MLX depthwise layout); transposing
/// to `[1, kernel, channels]` lets a single broadcast multiply-and-sum over the
/// `kernel` axis replace `conv1d`. Materialized once at load so decode never
/// reshapes the weight.
pub(crate) fn build_conv_decode_weight(conv_weight: &MlxArray) -> UniquePtr<MlxArray> {
    // [channels, kernel, 1] -> [1, kernel, channels].
    let w = mlxcel_core::transpose_axes(conv_weight, &[2, 1, 0]);
    let w = mlxcel_core::contiguous(&w, false);
    mlxcel_core::eval(&w);
    w
}

/// Single decode step of the depthwise causal short conv, computed as a
/// broadcast weighted sum instead of `conv1d` (issue #748 / #752).
///
/// For `padded` of shape `[batch, kernel, channels]` and `decode_weight` of
/// shape `[1, kernel, channels]` this returns `[batch, 1, channels]` where
/// `out[b, 0, c] = sum_k padded[b, k, c] * decode_weight[0, k, c]`, which is
/// exactly what a stride-1, no-pad, dilation-1, `groups == channels` `conv1d`
/// produces for a length-1 output. The two-kernel elementwise form avoids the
/// `conv1d` CUDA dispatch (MLX 0.32.1) that sends this tiny bf16 depthwise conv
/// to cuDNN's generic `convolve_common_engine`, which launches one kernel per
/// channel and dominates decode.
///
/// `in_dtype` is the (possibly bf16/f16) activation dtype; the decode weight is
/// cast to it only when it differs, so quantized checkpoints whose non-quantized
/// conv weight is stored at a wider precision still multiply in the activation
/// dtype (MLX widens the reduce accumulator for half dtypes).
pub(crate) fn short_conv_decode_step(
    padded: &MlxArray,
    decode_weight: &MlxArray,
    in_dtype: i32,
) -> UniquePtr<MlxArray> {
    let prod = if mlxcel_core::array_dtype(decode_weight) == in_dtype {
        mlxcel_core::multiply(padded, decode_weight)
    } else {
        let w = mlxcel_core::astype(decode_weight, in_dtype);
        mlxcel_core::multiply(padded, &w)
    };
    mlxcel_core::sum_axis(&prod, 1, true) // [batch, 1, channels]
}
