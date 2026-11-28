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

//! Raon_mHC: the multi-stream residual of the Raon-1 family.
//!
//! The residual is carried as `n` streams (`[B, L, n, D]`). Each sublayer site
//! (attention, MLP) reads a per-token selector from the flattened streams and
//! returns three float32 coefficient sets:
//!
//! ```text
//! inv   = rsqrt(mean(flat.f32^2) + selector_eps)              flat = streams.reshape(n*D)
//! proj  = (flat @ [phi_pre; phi_post; phi_z]^T).f32 * inv      GEMM in the model dtype
//! h_pre  = sigmoid(alpha_pre  * proj[:n]   + b_pre)
//! h_post = post_factor * sigmoid(alpha_post * proj[n:2n] + b_post)
//! Z      = alpha_z * proj[2n:] + b_z                          [n-1, n-1]
//! E      = Q Z Q^T                                            Q: Helmert basis [n, n-1]
//! B      = (I + 11^T / n) / 2
//! rho    = max(0, max(-E / B));  Phi = B + E / (1 + rho)
//! h_res  = (1 - gamma) I + gamma Phi                          gamma = sigmoid(eta)
//! ```
//!
//! `Q^T 1 = 0`, so `E` has zero row and column sums and every `Phi` (and every
//! `h_res`) is doubly stochastic; `rho` scales `E` back until `Phi` is
//! non-negative. The branch input is `sum_s h_pre[s] * streams[s]` (float32,
//! rounded once) and the write-back is `h_res @ streams + h_post * branch`
//! (float32, rounded once). This is not DeepSeek-V4's Sinkhorn mHC; only the
//! stream shape is shared.
//!
//! Reference: `RaonMHC` in
//! https://huggingface.co/KRAFTON/Raon-1-mini/blob/6691a076/modeling_raon1_mini.py
//!
//! Used by: Raon1Mini (both sites of every decoder layer); kept free of
//! Raon-1-mini types so Raon-1 nano (#3438) can load it unchanged.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

#[path = "raon_mhc_decode.rs"]
mod decode;
use decode::CompiledConsts;
pub use decode::DecodeFusion;

/// Orthonormal Helmert-style basis `Q` (`[n, n-1]`, row-major) with
/// `Q^T Q = I` and `Q^T 1 = 0`:
/// `Q[row, col] = (1 if row <= col, -(col + 1) if row == col + 1, else 0) / sqrt((col + 1)(col + 2))`.
pub fn helmert_basis(n: usize) -> Vec<f32> {
    let cols = n.saturating_sub(1);
    let mut q = vec![0.0f32; n * cols];
    for col in 0..cols {
        let k = (col + 1) as f64;
        let norm = (k * (k + 1.0)).sqrt();
        for row in 0..n {
            let value = if row < col + 1 {
                1.0
            } else if row == col + 1 {
                -k
            } else {
                0.0
            };
            q[row * cols + col] = (value / norm) as f32;
        }
    }
    q
}

/// The three coefficient sets one site produces for every token, float32.
pub struct MhcRoute {
    /// `[B, L, n]`.
    pub h_pre: UniquePtr<MlxArray>,
    /// `[B, L, n]`, already times `post_factor`.
    pub h_post: UniquePtr<MlxArray>,
    /// `[B, L, n, n]`, doubly stochastic.
    pub h_res: UniquePtr<MlxArray>,
}

/// One Raon_mHC site (`self_attn_raon_mhc` or `mlp_raon_mhc`).
pub struct RaonMhc {
    /// `[n*D, 2n + (n-1)^2]` in the model dtype: `[phi_pre; phi_post; phi_z]^T`.
    selector_t: UniquePtr<MlxArray>,
    /// The same plane row-major, `[2n + (n-1)^2, n*D]`, for one-row products.
    selector: UniquePtr<MlxArray>,
    /// Whether a one-row selector product takes the row-major form
    /// ([`DecodeFusion::selector`]).
    row_major_single: bool,
    alpha_pre: f32,
    alpha_post: f32,
    alpha_z: f32,
    /// `[n]`, `[n]`, `[(n-1)^2]` float32.
    b_pre: UniquePtr<MlxArray>,
    b_post: UniquePtr<MlxArray>,
    b_z: UniquePtr<MlxArray>,
    /// `sigmoid(res.eta)`.
    gamma: f32,
    /// `[n, n-1]` float32 Helmert basis, its transpose, `I` and `B`.
    basis: UniquePtr<MlxArray>,
    basis_t: UniquePtr<MlxArray>,
    eye: UniquePtr<MlxArray>,
    chart0: UniquePtr<MlxArray>,
    streams: i32,
    selector_eps: f32,
    post_factor: f32,
    /// The same coefficients as arrays, for the compiled decode graphs.
    compiled: CompiledConsts,
}

fn scalar_f32(weights: &WeightMap, key: &str) -> Result<f32, String> {
    let w = weights
        .get(key)
        .ok_or_else(|| format!("Missing weight: {key}"))?;
    let numel: i32 = mlxcel_core::array_shape(w).iter().product();
    if numel != 1 {
        return Err(format!(
            "{key}: expected a scalar, got shape {:?}",
            mlxcel_core::array_shape(w)
        ));
    }
    let v = mlxcel_core::astype(w, dtype::FLOAT32);
    mlxcel_core::eval(&v);
    Ok(mlxcel_core::item_f32(&mlxcel_core::reshape(&v, &[])))
}

fn vector_f32(weights: &WeightMap, key: &str, len: usize) -> Result<UniquePtr<MlxArray>, String> {
    let w = weights
        .get(key)
        .ok_or_else(|| format!("Missing weight: {key}"))?;
    let numel: i32 = mlxcel_core::array_shape(w).iter().product();
    if numel as usize != len {
        return Err(format!(
            "{key}: {numel} entries, expected {len} (shape {:?})",
            mlxcel_core::array_shape(w)
        ));
    }
    Ok(mlxcel_core::reshape(
        &mlxcel_core::astype(w, dtype::FLOAT32),
        &[len as i32],
    ))
}

fn selector_plane(
    weights: &WeightMap,
    key: &str,
    rows: usize,
    width: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let w = weights
        .get(key)
        .ok_or_else(|| format!("Missing weight: {key}"))?;
    let shape = mlxcel_core::array_shape(w);
    if shape != [rows as i32, width as i32] {
        return Err(format!(
            "{key}: expected shape [{rows}, {width}] (streams * hidden_size), got {shape:?}"
        ));
    }
    Ok(mlxcel_core::copy(w))
}

impl RaonMhc {
    /// Load the site under `prefix` (`model.layers.N.self_attn_raon_mhc`).
    /// `act_dtype` is the model dtype the selector GEMM runs in.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        streams: usize,
        hidden_size: usize,
        selector_eps: f32,
        post_factor: f32,
        act_dtype: i32,
    ) -> Result<Self, String> {
        if streams < 2 {
            return Err(format!(
                "{prefix}: raon_mhc needs at least 2 streams, got {streams}"
            ));
        }
        let n = streams;
        let m = (n - 1) * (n - 1);
        let width = n * hidden_size;
        let phi_pre = selector_plane(weights, &format!("{prefix}.phi_pre"), n, width)?;
        let phi_post = selector_plane(weights, &format!("{prefix}.phi_post"), n, width)?;
        let phi_z = selector_plane(weights, &format!("{prefix}.res.phi_z"), m, width)?;
        let selector = mlxcel_core::concatenate_many(&[&*phi_pre, &*phi_post, &*phi_z], 0);
        let selector = mlxcel_core::astype(&selector, act_dtype);
        let selector_t = mlxcel_core::contiguous(&mlxcel_core::swap_axes(&selector, 0, 1), false);

        let eta = scalar_f32(weights, &format!("{prefix}.res.eta"))?;
        let gamma = (1.0 / (1.0 + (-(eta as f64)).exp())) as f32;

        let basis = helmert_basis(n);
        let basis = mlxcel_core::from_slice_f32(&basis, &[n as i32, (n - 1) as i32]);
        let basis_t = mlxcel_core::contiguous(&mlxcel_core::swap_axes(&basis, 0, 1), false);
        let eye = mlxcel_core::eye(n as i32, n as i32, 0, dtype::FLOAT32);
        let chart0: Vec<f32> = (0..n * n)
            .map(|i| {
                let diag = if i / n == i % n { 1.0 } else { 0.0 };
                0.5 * (diag + 1.0 / n as f32)
            })
            .collect();
        let chart0 = mlxcel_core::from_slice_f32(&chart0, &[n as i32, n as i32]);

        let alpha_pre = scalar_f32(weights, &format!("{prefix}.alpha_pre"))?;
        let alpha_post = scalar_f32(weights, &format!("{prefix}.alpha_post"))?;
        let alpha_z = scalar_f32(weights, &format!("{prefix}.res.alpha_z"))?;
        let b_pre = vector_f32(weights, &format!("{prefix}.b_pre"), n)?;
        let b_post = vector_f32(weights, &format!("{prefix}.b_post"), n)?;
        let b_z = vector_f32(weights, &format!("{prefix}.res.b_z"), m)?;
        let compiled = CompiledConsts::new(
            [alpha_pre, alpha_post, alpha_z],
            [&*b_pre, &*b_post, &*b_z],
            (n, m),
            selector_eps,
            post_factor,
            gamma,
            &eye,
        );

        Ok(Self {
            selector_t,
            selector,
            row_major_single: DecodeFusion::from_env().selector,
            alpha_pre,
            alpha_post,
            alpha_z,
            b_pre,
            b_post,
            b_z,
            gamma,
            basis,
            basis_t,
            eye,
            chart0,
            streams: n as i32,
            selector_eps,
            post_factor,
            compiled,
        })
    }

    /// The selector GEMM `flat @ [phi_pre; phi_post; phi_z]^T` in the model
    /// dtype, `flat` being `[.., n*D]`.
    ///
    /// A product of exactly one row (a decode step, or one row of a
    /// speculative verify block, which is split per row) is formed as
    /// `selector @ flat^T` over the row-major plane. With the transposed
    /// plane on the right MLX runs its transposed gemv kernel there, which
    /// took about 255 us for this `[n*D, 17]` shape on an M3 Ultra against
    /// about 35 us row-major, at 50 sites per token (issue #3458). Decode
    /// steps and verify rows both take this form, so they stay bit-equal;
    /// products of several rows (prefill, batched decode) keep the plain
    /// matmul and its bytes.
    fn selector_product(&self, flat: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(flat);
        let (lead, width) = shape.split_at(shape.len() - 1);
        if !self.row_major_single || lead.iter().product::<i32>() != 1 {
            return mlxcel_core::matmul(flat, &self.selector_t);
        }
        let column = mlxcel_core::reshape(flat, &[width[0], 1]);
        let out = mlxcel_core::matmul(&self.selector, &column);
        let mut out_shape = lead.to_vec();
        out_shape.push(mlxcel_core::array_shape(&self.selector)[0]);
        mlxcel_core::reshape(&out, &out_shape)
    }

    /// The per-token coefficients for `streams` (`[B, L, n, D]`).
    pub fn route(&self, streams: &MlxArray) -> MhcRoute {
        let shape = mlxcel_core::array_shape(streams);
        let (b, l, n, d) = (shape[0], shape[1], shape[2], shape[3]);
        debug_assert_eq!(n, self.streams, "raon_mhc: stream count");
        let nm = n - 1;

        let flat = mlxcel_core::reshape(streams, &[b, l, n * d]);
        let flat_f32 = mlxcel_core::astype(&flat, dtype::FLOAT32);
        let ms = mlxcel_core::mean_axis(&mlxcel_core::square(&flat_f32), -1, true);
        let eps = mlxcel_core::full_f32(&[1], self.selector_eps, dtype::FLOAT32);
        let inv = mlxcel_core::rsqrt(&mlxcel_core::add(&ms, &eps));
        // One row at a time inside a speculative verify block, so each row
        // equals its decode step (issue #3436); a plain matmul elsewhere.
        let proj = mlxcel_core::row_invariant::per_row(&flat, |x| self.selector_product(x));
        let proj = mlxcel_core::multiply(&mlxcel_core::astype(&proj, dtype::FLOAT32), &inv);

        let part = |start: i32, end: i32| mlxcel_core::slice(&proj, &[0, 0, start], &[b, l, end]);
        let t_pre = mlxcel_core::add(
            &mlxcel_core::multiply_scalar(&part(0, n), self.alpha_pre),
            &self.b_pre,
        );
        let t_post = mlxcel_core::add(
            &mlxcel_core::multiply_scalar(&part(n, 2 * n), self.alpha_post),
            &self.b_post,
        );
        let t_z = mlxcel_core::add(
            &mlxcel_core::multiply_scalar(&part(2 * n, 2 * n + nm * nm), self.alpha_z),
            &self.b_z,
        );
        let h_pre = mlxcel_core::sigmoid(&t_pre);
        let h_post = mlxcel_core::multiply_scalar(&mlxcel_core::sigmoid(&t_post), self.post_factor);

        let z = mlxcel_core::reshape(&t_z, &[b, l, nm, nm]);
        // A batched matmul over the positions: inside a verify block, one
        // position at a time, since CUDA picks its kernel (and so the
        // summation order) by the block length.
        let error = mlxcel_core::row_invariant::per_position(&z, |z| {
            mlxcel_core::matmul(&mlxcel_core::matmul(&self.basis, z), &self.basis_t)
        });
        let ratio = mlxcel_core::divide(&mlxcel_core::negative(&error), &self.chart0);
        let ratio = mlxcel_core::reshape(&ratio, &[b, l, n * n]);
        let rho = mlxcel_core::max_axis(&ratio, -1, true);
        let rho = mlxcel_core::maximum(&rho, &mlxcel_core::zeros(&[1], dtype::FLOAT32));
        let one = mlxcel_core::full_f32(&[1], 1.0, dtype::FLOAT32);
        let denom = mlxcel_core::expand_dims(&mlxcel_core::add(&rho, &one), -1);
        let chart = mlxcel_core::add(&self.chart0, &mlxcel_core::divide(&error, &denom));
        let h_res = mlxcel_core::add(
            &mlxcel_core::multiply_scalar(&self.eye, 1.0 - self.gamma),
            &mlxcel_core::multiply_scalar(&chart, self.gamma),
        );
        MhcRoute {
            h_pre,
            h_post,
            h_res,
        }
    }
}

/// `a @ b` for `[1, L, .., ..]` operands batched over the positions. Inside a
/// speculative verify block each position is multiplied alone, so row `t`
/// of a block equals its decode step: CUDA picks the batched kernel, and so
/// the summation order, by the block length. Elsewhere one `matmul`.
fn matmul_per_position(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(a);
    let length = shape[1];
    if !mlxcel_core::row_invariant::in_verify_block()
        || shape[0] != 1
        || !(2..=mlxcel_core::row_invariant::MAX_ROWS).contains(&length)
    {
        return mlxcel_core::matmul(a, b);
    }
    let at = |x: &MlxArray, t: i32| mlxcel_core::utils::slice_axis(x, 1, t, t + 1);
    let parts: Vec<_> = (0..length)
        .map(|t| mlxcel_core::matmul(&at(a, t), &at(b, t)))
        .collect();
    mlxcel_core::concatenate_many(&parts.iter().map(|p| &**p).collect::<Vec<_>>(), 1)
}

/// Branch input `sum_s h_pre[s] * streams[s]` in float32, rounded to `dtype`.
/// `streams_f32` is `[B, L, n, D]` float32, `h_pre` `[B, L, n]`.
pub fn aggregate(streams_f32: &MlxArray, h_pre: &MlxArray, out_dtype: i32) -> UniquePtr<MlxArray> {
    let weights = mlxcel_core::expand_dims(h_pre, -2); // [B, L, 1, n]
    let mixed = matmul_per_position(&weights, streams_f32); // [B, L, 1, D]
    mlxcel_core::astype(&mlxcel_core::squeeze_axis(&mixed, -2), out_dtype)
}

/// Write-back `h_res @ streams + h_post * branch` in float32, rounded once.
/// `branch` is `[B, L, D]` in any dtype; the result is `[B, L, n, D]` in `out_dtype`.
#[cfg(test)]
pub fn mix(
    streams_f32: &MlxArray,
    branch: &MlxArray,
    route: &MhcRoute,
    out_dtype: i32,
) -> UniquePtr<MlxArray> {
    mix_parts(streams_f32, branch, &route.h_post, &route.h_res, out_dtype)
}

fn mix_parts(
    streams_f32: &MlxArray,
    branch: &MlxArray,
    h_post: &MlxArray,
    h_res: &MlxArray,
    out_dtype: i32,
) -> UniquePtr<MlxArray> {
    let mixed = matmul_per_position(h_res, streams_f32);
    let branch = mlxcel_core::expand_dims(&mlxcel_core::astype(branch, dtype::FLOAT32), -2);
    let post = mlxcel_core::expand_dims(h_post, -1);
    let written = mlxcel_core::add(&mixed, &mlxcel_core::multiply(&post, &branch));
    mlxcel_core::astype(&written, out_dtype)
}

/// `[B, L, D]` to `[B, L, n, D]`: every stream starts as the embedding.
pub fn expand_streams(h: &MlxArray, streams: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(h);
    let h = mlxcel_core::expand_dims(h, 2);
    let expanded = mlxcel_core::broadcast_to(&h, &[shape[0], shape[1], streams as i32, shape[2]]);
    mlxcel_core::contiguous(&expanded, false)
}

/// `mean(streams.f32, axis=stream).astype(dtype)`, the reduction before the final norm.
pub fn reduce_streams(streams: &MlxArray, out_dtype: i32) -> UniquePtr<MlxArray> {
    let f = mlxcel_core::astype(streams, dtype::FLOAT32);
    mlxcel_core::astype(&mlxcel_core::mean_axis(&f, 2, false), out_dtype)
}

#[cfg(test)]
#[path = "raon_mhc_tests.rs"]
mod tests;
