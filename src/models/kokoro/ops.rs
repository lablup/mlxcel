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

//! Shared MLX building blocks for the Kokoro StyleTTS2 + iSTFTNet port.
//!
//! These are thin wrappers over the `mlxcel_core` FFI that encode the calling
//! conventions Kokoro needs: weight-norm reconstruction, channels-last
//! convolution from PyTorch-layout weights, instance/layer normalization, a
//! hand-rolled bidirectional LSTM (MLX exposes no recurrent primitive), and a
//! few small helpers. Everything runs on the [`AudioWorker`] thread, so these
//! functions only build and return lazy MLX graphs; the caller evaluates once
//! at the end.
//!
//! Convolution layout: `mlx::core::conv1d`/`conv_transpose1d` expect inputs as
//! `(N, L, C_in)` and weights as `(C_out, K, C_in)`. PyTorch/safetensors stores
//! conv weights as `(C_out, C_in, K)` (and transposed convs as `(C_in, C_out,
//! K)`), so callers reconstruct the weight-norm tensor in PyTorch layout and the
//! conv wrappers here transpose the kernel/channel axes before the FFI call.

use mlxcel_core::{MlxArray, UniquePtr};

use mlxcel_core::dtype::FLOAT32;

/// Wrap a borrowed array reference for an op that takes `&MlxArray`.
#[inline]
pub(crate) fn r(a: &UniquePtr<MlxArray>) -> &MlxArray {
    a.as_ref().expect("kokoro: null MlxArray")
}

/// Build a scalar `f32` array (shape `[1]`).
#[inline]
pub(crate) fn scalar(v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[v], &[1])
}

/// `a + b` with broadcasting.
#[inline]
pub(crate) fn add(a: &UniquePtr<MlxArray>, b: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::add(r(a), r(b))
}

/// `a - b` with broadcasting.
#[inline]
pub(crate) fn sub(a: &UniquePtr<MlxArray>, b: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::subtract(r(a), r(b))
}

/// `a * b` with broadcasting.
#[inline]
pub(crate) fn mul(a: &UniquePtr<MlxArray>, b: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::multiply(r(a), r(b))
}

/// `a / b` with broadcasting.
#[inline]
pub(crate) fn div(a: &UniquePtr<MlxArray>, b: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::divide(r(a), r(b))
}

/// `a + scalar`.
#[inline]
pub(crate) fn add_scalar(a: &UniquePtr<MlxArray>, v: f32) -> UniquePtr<MlxArray> {
    let s = scalar(v);
    mlxcel_core::add(r(a), r(&s))
}

/// `a * scalar`.
#[inline]
pub(crate) fn mul_scalar(a: &UniquePtr<MlxArray>, v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::multiply_scalar(r(a), v)
}

/// `a / scalar`.
#[inline]
pub(crate) fn div_scalar(a: &UniquePtr<MlxArray>, v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::divide_scalar(r(a), v)
}

/// Matrix product `a @ b`.
#[inline]
pub(crate) fn matmul(a: &UniquePtr<MlxArray>, b: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::matmul(r(a), r(b))
}

/// `exp(a)`.
#[inline]
pub(crate) fn exp(a: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::exp(r(a))
}

/// `sin(a)`.
#[inline]
pub(crate) fn sin(a: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::sin(r(a))
}

/// `tanh(a)`.
#[inline]
pub(crate) fn tanh(a: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::tanh(r(a))
}

/// `sigmoid(a)`.
#[inline]
pub(crate) fn sigmoid(a: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    mlxcel_core::sigmoid(r(a))
}

/// `reshape(a, shape)`.
#[inline]
pub(crate) fn reshape(a: &UniquePtr<MlxArray>, shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::reshape(r(a), shape)
}

/// Permute axes.
#[inline]
pub(crate) fn transpose(a: &UniquePtr<MlxArray>, axes: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::transpose_axes(r(a), axes)
}

/// Swap two axes.
#[inline]
pub(crate) fn swap_axes(a: &UniquePtr<MlxArray>, x: i32, y: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::swap_axes(r(a), x, y)
}

/// Insert a size-1 axis at `axis`.
#[inline]
pub(crate) fn expand_dims(a: &UniquePtr<MlxArray>, axis: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(r(a), axis)
}

/// Concatenate two arrays along `axis`.
#[inline]
pub(crate) fn concat2(
    a: &UniquePtr<MlxArray>,
    b: &UniquePtr<MlxArray>,
    axis: i32,
) -> UniquePtr<MlxArray> {
    mlxcel_core::concatenate(r(a), r(b), axis)
}

/// Concatenate a list of arrays along `axis` by pairwise folding (the
/// `mlxcel_core` root only re-exports the safe two-array `concatenate`).
pub(crate) fn concat(parts: &[&UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    assert!(!parts.is_empty(), "kokoro: concat of empty list");
    let mut acc = mlxcel_core::copy(r(parts[0]));
    for p in &parts[1..] {
        acc = mlxcel_core::concatenate(r(&acc), r(p), axis);
    }
    acc
}

/// Slice `a[starts..stops]` (per-axis bounds).
#[inline]
pub(crate) fn slice(a: &UniquePtr<MlxArray>, starts: &[i32], stops: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(r(a), starts, stops)
}

/// Mean over `axis`, keeping the reduced dim.
#[inline]
pub(crate) fn mean_axis(a: &UniquePtr<MlxArray>, axis: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::mean_axis(r(a), axis, true)
}

/// Population variance over `axis` (ddof = 0), keeping the reduced dim.
#[inline]
pub(crate) fn var_axis(a: &UniquePtr<MlxArray>, axis: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::var_axis(r(a), axis, true, 0)
}

/// Sum over `axis`.
#[inline]
pub(crate) fn sum_axis(a: &UniquePtr<MlxArray>, axis: i32, keepdims: bool) -> UniquePtr<MlxArray> {
    mlxcel_core::sum_axis(r(a), axis, keepdims)
}

/// Leaky ReLU with the given negative slope.
#[inline]
pub(crate) fn leaky_relu(a: &UniquePtr<MlxArray>, slope: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::leaky_relu(r(a), slope)
}

/// Gather rows of an embedding table by integer ids: `table[ids]`.
///
/// `ids` are `i32`; `table` is `(vocab, dim)`. Returns `(len(ids), dim)`.
pub(crate) fn embed(table: &UniquePtr<MlxArray>, ids: &[i32]) -> UniquePtr<MlxArray> {
    let idx = mlxcel_core::from_slice_i32(ids, &[ids.len() as i32]);
    mlxcel_core::take(r(table), r(&idx), 0)
}

/// NumPy-style tanh-approximation GELU (`gelu_new`), matching the activation
/// the PLBert weights were trained with. MLX's `gelu` is the exact erf form,
/// which differs numerically, so this is built from `tanh`.
pub(crate) fn gelu_new(x: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let x3 = mul(&mul(x, x), x);
    let inner = add(x, &mul_scalar(&x3, 0.044715));
    let t = tanh(&mul_scalar(&inner, SQRT_2_OVER_PI));
    let g = add_scalar(&t, 1.0);
    mul(&mul_scalar(x, 0.5), &g)
}

/// Linear layer `y = x @ w^T + b`.
///
/// `w` is the PyTorch `(out, in)` weight; `x` is `(..., in)`. `bias` is
/// optional `(out,)`.
pub(crate) fn linear(
    x: &UniquePtr<MlxArray>,
    w: &UniquePtr<MlxArray>,
    bias: Option<&UniquePtr<MlxArray>>,
) -> UniquePtr<MlxArray> {
    let wt = swap_axes(w, 0, 1); // (in, out)
    let y = matmul(x, &wt);
    match bias {
        Some(b) => add(&y, b),
        None => y,
    }
}

/// Layer normalization over the last axis with explicit affine `weight`/`bias`.
pub(crate) fn layer_norm(
    x: &UniquePtr<MlxArray>,
    weight: &UniquePtr<MlxArray>,
    bias: &UniquePtr<MlxArray>,
    eps: f32,
) -> UniquePtr<MlxArray> {
    let mu = mean_axis(x, -1);
    let xc = sub(x, &mu);
    let var = var_axis(x, -1);
    let denom = mlxcel_core::rsqrt(r(&add_scalar(&var, eps)));
    let norm = mul(&xc, &denom);
    add(&mul(&norm, weight), bias)
}

/// Instance normalization over the time axis for a `(C, T)` activation: each
/// channel is normalized over `T` with its own mean/variance, no affine.
pub(crate) fn instance_norm_ct(x: &UniquePtr<MlxArray>, eps: f32) -> UniquePtr<MlxArray> {
    let mu = mean_axis(x, 1); // mean over T -> (C,1)
    let xc = sub(x, &mu);
    let var = var_axis(x, 1);
    let denom = mlxcel_core::rsqrt(r(&add_scalar(&var, eps)));
    mul(&xc, &denom)
}

/// Reconstruct a weight-norm tensor `w = g * v / ||v||` in PyTorch conv layout.
///
/// `v` is `(out, in, k)` (or `(in, out, k)` for transposed convs); `g` is
/// `(out, 1, 1)`. The L2 norm is taken over every axis except axis 0, matching
/// `torch.nn.utils.weight_norm(dim=0)`.
pub(crate) fn weight_norm(g: &UniquePtr<MlxArray>, v: &UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    // ||v|| over axes (1,2): sum of squares then sqrt, keepdims.
    let sq = mul(v, v);
    let s1 = sum_axis(&sq, 1, true);
    let s2 = sum_axis(&s1, 2, true);
    let norm = mlxcel_core::sqrt(r(&s2)); // (out,1,1)
    let scaled = div(v, &norm);
    mul(&scaled, g)
}

/// 1-D convolution from a PyTorch-layout weight `(C_out, C_in/groups, K)`.
///
/// `x` is `(C, L)` (single example, channels-first). Internally transposes to
/// MLX channels-last (`(1, L, C)` input, `(C_out, K, C_in)` weight), runs the
/// FFI conv, and returns `(C_out, L_out)`. `bias` is optional `(C_out,)`.
pub(crate) fn conv1d(
    x: &UniquePtr<MlxArray>,
    w_pt: &UniquePtr<MlxArray>,
    bias: Option<&UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> UniquePtr<MlxArray> {
    let xin = expand_dims(&swap_axes(x, 0, 1), 0); // (1, L, C_in)
    let w = transpose(w_pt, &[0, 2, 1]); // (C_out, K, C_in/groups)
    let y = mlxcel_core::conv1d(r(&xin), r(&w), stride, padding, dilation, groups); // (1, L_out, C_out)
    let y = mlxcel_core::squeeze_axis(r(&y), 0); // (L_out, C_out)
    let y = swap_axes(&y, 0, 1); // (C_out, L_out)
    match bias {
        Some(b) => add(&y, &expand_dims(b, 1)),
        None => y,
    }
}

/// 1-D transposed convolution from a PyTorch-layout weight `(C_in, C_out/groups, K)`.
///
/// `x` is `(C, L)`. Returns `(C_out, L_out)`. MLX's `conv_transpose1d` expects
/// the weight as `(C_out, K, C_in/groups)`. The PyTorch -> MLX transpose differs
/// by grouping:
/// - `groups == 1`: PyTorch `(C_in, C_out, K)` -> MLX `(C_out, K, C_in)` (axes
///   `[1, 2, 0]`).
/// - depthwise (`groups == C_in == C_out`): PyTorch `(C_in, 1, K)` -> MLX
///   `(C_out, K, 1)` (axes `[0, 2, 1]`), since `C_in == C_out` and the second
///   PyTorch axis is the per-group output (`= 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose1d(
    x: &UniquePtr<MlxArray>,
    w_pt: &UniquePtr<MlxArray>,
    bias: Option<&UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    dilation: i32,
    output_padding: i32,
    groups: i32,
) -> UniquePtr<MlxArray> {
    let xin = expand_dims(&swap_axes(x, 0, 1), 0); // (1, L, C_in)
    let w = if groups > 1 {
        transpose(w_pt, &[0, 2, 1]) // depthwise: (C_out, K, 1)
    } else {
        transpose(w_pt, &[1, 2, 0]) // (C_out, K, C_in)
    };
    let y = mlxcel_core::conv_transpose1d(
        r(&xin),
        r(&w),
        stride,
        padding,
        dilation,
        output_padding,
        groups,
    );
    let y = mlxcel_core::squeeze_axis(r(&y), 0); // (L_out, C_out)
    let y = swap_axes(&y, 0, 1); // (C_out, L_out)
    match bias {
        Some(b) => add(&y, &expand_dims(b, 1)),
        None => y,
    }
}

/// Nearest-neighbour upsample of a `(C, T)` activation along T by an integer
/// factor. Implemented as `repeat` along the time axis.
pub(crate) fn upsample_nearest_ct(x: &UniquePtr<MlxArray>, scale: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::repeat(r(x), scale, 1)
}

/// One direction of an LSTM over a `(T, in)` sequence, PyTorch gate order
/// (`i, f, g, o`). Weights are the PyTorch `weight_ih`/`weight_hh` `(4H, *)`
/// and biases `(4H,)`. Returns the hidden-state sequence `(T, H)`.
///
/// MLX exposes no recurrent kernel, so this scans in Rust: each step builds a
/// small MLX graph for the gates. The sequences here are short (one phoneme or
/// frame sequence), so the per-step overhead is acceptable.
pub(crate) fn lstm_dir(
    x: &UniquePtr<MlxArray>,
    t: usize,
    hidden: usize,
    w_ih: &UniquePtr<MlxArray>,
    w_hh: &UniquePtr<MlxArray>,
    b_ih: &UniquePtr<MlxArray>,
    b_hh: &UniquePtr<MlxArray>,
    reverse: bool,
) -> Vec<UniquePtr<MlxArray>> {
    let h_i = hidden as i32;
    let wih_t = swap_axes(w_ih, 0, 1); // (in, 4H)
    let whh_t = swap_axes(w_hh, 0, 1); // (H, 4H)
    let bias = add(b_ih, b_hh); // (4H,)

    let mut h = mlxcel_core::zeros(&[1, h_i], FLOAT32);
    let mut c = mlxcel_core::zeros(&[1, h_i], FLOAT32);
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(t);

    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        let xt = slice(x, &[ti as i32, 0], &[ti as i32 + 1, i32::MAX]); // (1, in)
        let gates = add(&add(&matmul(&xt, &wih_t), &matmul(&h, &whh_t)), &bias); // (1, 4H)
        let i_g = sigmoid(&slice(&gates, &[0, 0], &[1, h_i]));
        let f_g = sigmoid(&slice(&gates, &[0, h_i], &[1, 2 * h_i]));
        let g_g = tanh(&slice(&gates, &[0, 2 * h_i], &[1, 3 * h_i]));
        let o_g = sigmoid(&slice(&gates, &[0, 3 * h_i], &[1, 4 * h_i]));
        c = add(&mul(&f_g, &c), &mul(&i_g, &g_g));
        h = mul(&o_g, &tanh(&c));
        outputs.push(reshape(&h, &[1, h_i]));
    }
    if reverse {
        outputs.reverse();
    }
    outputs
}

/// Read an MLX array back to a host `Vec<f32>`. Casts to `f32`, makes the
/// buffer contiguous, evaluates, then parses 4-byte little-endian chunks.
pub(crate) fn to_vec_f32(a: &UniquePtr<MlxArray>) -> Result<Vec<f32>, String> {
    let f = mlxcel_core::astype(r(a), FLOAT32);
    let c = mlxcel_core::contiguous(r(&f), false);
    mlxcel_core::try_eval(r(&c)).map_err(|e| format!("kokoro eval failed: {e}"))?;
    let bytes = mlxcel_core::array_to_raw_bytes(r(&c));
    Ok(bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect())
}

/// Shape of an MLX array as a `Vec<i32>`.
#[inline]
pub(crate) fn shape(a: &UniquePtr<MlxArray>) -> Vec<i32> {
    mlxcel_core::array_shape(r(a))
}
