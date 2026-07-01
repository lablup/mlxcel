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

//! Shared Gated Delta Rule implementation for linear attention layers.
//!
//! This module provides the core gated delta net primitives used by models
//! that employ hybrid transformer + linear attention architectures.
//!
//! Used by: Qwen3Next, Qwen3.5, KimiLinear
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py

use mlxcel_core::utils::{silu, softplus, stack_arrays};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Chunk length for the chunked parallel prefill scan (`gated_delta_chunked`).
///
/// The scan trades the per-token loop's `O(T)` sequential dependency for an
/// `O(T/C + C)` one: `C`-length intra-chunk work done in parallel across all
/// chunks plus a `T/C`-step inter-chunk state scan. `64` follows the standard
/// delta-rule chunk size and keeps the intra-chunk `C x C` matrices small.
const GATED_DELTA_CHUNK_SIZE: usize = 64;

// Cache.
/// Cache for GatedDeltaNet (linear attention) layers.
/// Stores conv1d state and recurrent SSM state.
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub struct GatedDeltaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>, // [batch, kernel-1, conv_dim]
    pub state_cache: Option<UniquePtr<MlxArray>>, // [batch, num_v_heads, head_v_dim, head_k_dim]
    pub offset: i32,
}

impl GatedDeltaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            state_cache: None,
            offset: 0,
        }
    }

    pub fn advance(&mut self, step: i32) {
        self.offset += step;
    }

    pub fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        super::recurrent_snapshot::push_optional(
            snapshot,
            format!("{prefix}.conv_state"),
            &self.conv_state,
        );
        super::recurrent_snapshot::push_optional(
            snapshot,
            format!("{prefix}.state_cache"),
            &self.state_cache,
        );
    }

    pub fn restore_from(
        &mut self,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        self.conv_state =
            super::recurrent_snapshot::restore_optional(snapshot, format!("{prefix}.conv_state"));
        self.state_cache =
            super::recurrent_snapshot::restore_optional(snapshot, format!("{prefix}.state_cache"));
        self.offset = snapshot.token_len() as i32;
    }
}

impl Default for GatedDeltaCache {
    fn default() -> Self {
        Self::new()
    }
}

// Core Functions.
/// Compute gating values from A_log, a, and dt_bias.
/// g = exp(-exp(A_log.astype(float32)) * softplus(a + dt_bias))
///
/// Upstream Python casts the result back to a.dtype; here we keep float32
/// so the ops-path state accumulation stays in higher precision (the Python
/// Metal kernel path uses float32 state internally for the same reason).
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub fn compute_g(a_log: &MlxArray, a: &MlxArray, dt_bias: &MlxArray) -> UniquePtr<MlxArray> {
    let a_plus_dt = mlxcel_core::add(a, dt_bias);
    let sp = softplus(&a_plus_dt);
    // Cast a_log to float32 before exp() for numerical precision
    let a_log_f32 = mlxcel_core::astype(a_log, dtype::FLOAT32);
    let exp_a_log = mlxcel_core::exp(&a_log_f32);
    let neg_product = mlxcel_core::negative(&mlxcel_core::multiply(&exp_a_log, &sp));
    // Keep result in float32 (no cast back to input dtype)
    mlxcel_core::exp(&neg_product)
}

/// Single recurrent step of the gated delta rule.
///
/// Shapes:
///   - q, k: [B, H, Dk]
///   - v: [B, H, Dv]
///   - g: [B, H] or [B, H, Dk] (float32 from compute_g)
///   - beta: [B, H]
///   - state: [B, H, Dv, Dk] (float32)
///
/// Returns: (y: [B, H, Dv] in q dtype, new_state: [B, H, Dv, Dk] in float32)
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub fn gated_delta_step(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Fast path: no mask (common during decode) — use fused C++ kernel
    // Replaces ~26 FFI round-trips with a single call.
    if mask.is_none() {
        let q_dtype = mlxcel_core::array_dtype(q);
        let mut output = mlxcel_core::UniquePtr::null();
        let mut new_state = mlxcel_core::UniquePtr::null();
        unsafe {
            mlxcel_core::fused_gated_delta_decode_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                q_dtype,
                &mut output,
                &mut new_state,
            );
        }
        return (output, new_state);
    }

    // Slow path: mask present (rare — only during batch dim mismatch recovery)
    let old_state = mlxcel_core::copy(state);

    // Decay: state = state * g
    let g_ndim = mlxcel_core::array_ndim(g);
    let decay = if g_ndim == 2 {
        let g1 = mlxcel_core::expand_dims(g, -1);
        mlxcel_core::expand_dims(&g1, -1)
    } else if g_ndim == 3 {
        mlxcel_core::expand_dims(g, -2)
    } else {
        panic!("Unsupported gating shape");
    };

    let mut new_state = mlxcel_core::multiply(state, &decay);

    let k_exp = mlxcel_core::expand_dims(k, -2);
    let kv_mem = mlxcel_core::sum_axis(&mlxcel_core::multiply(&new_state, &k_exp), -1, false);

    let beta_exp = mlxcel_core::expand_dims(beta, -1);
    let delta = mlxcel_core::multiply(&mlxcel_core::subtract(v, &kv_mem), &beta_exp);

    let delta_exp = mlxcel_core::expand_dims(&delta, -1);
    new_state = mlxcel_core::add(&new_state, &mlxcel_core::multiply(&k_exp, &delta_exp));

    let q_exp = mlxcel_core::expand_dims(q, -2);
    let y = mlxcel_core::sum_axis(&mlxcel_core::multiply(&new_state, &q_exp), -1, false);

    // Apply mask
    let m = mask.unwrap();
    let m1 = mlxcel_core::expand_dims(m, 1);
    let m2 = mlxcel_core::expand_dims(&m1, 2);
    let m3 = mlxcel_core::expand_dims(&m2, 3);
    let new_state = mlxcel_core::where_cond(&m3, &new_state, &old_state);

    let q_dtype = mlxcel_core::array_dtype(q);
    let y = if mlxcel_core::array_dtype(&y) != q_dtype {
        mlxcel_core::astype(&y, q_dtype)
    } else {
        y
    };

    (y, new_state)
}

/// Ops-based implementation for prompt prefill (sequential loop).
///
/// State is initialized as float32 when not provided, and remains float32
/// across timesteps to prevent underflow/overflow with bfloat16 inputs.
///
/// Shapes:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized), float32
///   - beta: [B, T, Hv]
///   - state: [B, Hv, Dv, Dk] (float32)
///
/// Returns: (y: [B, T, Hv, Dv] in q dtype, state: [B, Hv, Dv, Dk] in float32)
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub fn gated_delta_ops(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: Option<&MlxArray>,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(q);
    let b = shape[0];
    let t = shape[1] as usize;
    let hk = shape[2];
    let dk = shape[3];
    let v_shape = mlxcel_core::array_shape(v);
    let hv = v_shape[2];
    let dv = v_shape[3];

    // Initialize state if not provided; use float32 for numerical precision
    // (prevents underflow/overflow in long sequences when input is bfloat16)
    let current_state = if let Some(s) = state {
        mlxcel_core::copy(s)
    } else {
        mlxcel_core::zeros(&[b, hv, dv, dk], dtype::FLOAT32)
    };

    // Metal kernel path: handles both T=1 and T>1 in a single GPU dispatch.
    // The kernel handles GQA internally via hk_idx = hv_idx / (Hv / Hk),
    // so q and k are passed with their original Hk heads (not repeated).
    if mlxcel_core::gated_delta_kernel_available()
        && supports_metal_gated_delta_kernel(hk, hv, dk, dv)
    {
        let mut output = mlxcel_core::UniquePtr::null();
        let mut new_state = mlxcel_core::UniquePtr::null();
        let mask_ptr: *const mlxcel_core::MlxArray = match mask {
            Some(m) => m as *const mlxcel_core::MlxArray,
            None => std::ptr::null(),
        };
        // SAFETY: Metal kernel reads from valid input arrays, writes to output/new_state.
        // mask_ptr is null when mask is None, which the C++ side handles correctly.
        unsafe {
            mlxcel_core::metal_gated_delta_forward(
                q,
                k,
                v,
                g,
                beta,
                &current_state,
                mask_ptr,
                &mut output,
                &mut new_state,
            );
        }
        return (output, new_state);
    }

    // Fallback: ops-based path for non-Metal devices and shapes outside the
    // custom kernel contract.
    let mut current_state = current_state;

    // Compute repeat factor for GQA
    let repeat_factor = hv / hk;

    // Repeat q and k if needed (skip copy for repeat_factor=1)
    let q_rep;
    let q_ref = if repeat_factor > 1 {
        q_rep = mlxcel_core::repeat(q, repeat_factor as i32, -2);
        q_rep.as_ref().unwrap()
    } else {
        q
    };
    let k_rep;
    let k_ref = if repeat_factor > 1 {
        k_rep = mlxcel_core::repeat(k, repeat_factor as i32, -2);
        k_rep.as_ref().unwrap()
    } else {
        k
    };

    // Fast path: single-token decode (T=1) -- skip slice+squeeze loop entirely.
    // Directly squeeze the time dimension and call gated_delta_step once.
    if t == 1 {
        let q_t = mlxcel_core::squeeze_axis(q_ref, 1);
        let k_t = mlxcel_core::squeeze_axis(k_ref, 1);
        let v_t = mlxcel_core::squeeze_axis(v, 1);

        let g_t = mlxcel_core::squeeze_axis(g, 1);

        let beta_t = mlxcel_core::squeeze_axis(beta, 1);

        let mask_t = mask.map(|m| mlxcel_core::squeeze_axis(m, 1));

        let (y, new_state) = gated_delta_step(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &current_state,
            mask_t.as_deref(),
        );

        // Add back time dimension: [B, Hv, Dv] -> [B, 1, Hv, Dv]
        let y = mlxcel_core::expand_dims(&y, 1);
        return (y, new_state);
    }

    // Multi-token path (prefill). On the non-Metal fallback, scalar gating with
    // no mask routes through the chunked parallel scan below; vectorized gating
    // ([B, T, Hv, Dk]) or an explicit batch-recovery mask falls through to the
    // sequential reference loop that follows.
    if mask.is_none() && mlxcel_core::array_ndim(g) == 3 {
        return gated_delta_chunked(
            q_ref,
            k_ref,
            v,
            g,
            beta,
            &current_state,
            GATED_DELTA_CHUNK_SIZE,
            b,
            t,
            hv,
            dk,
            dv,
        );
    }

    // Sequential fallback (vectorized gating or mask): one step per timestep.
    let mut ys = Vec::with_capacity(t);
    for t_idx in 0..t {
        let q_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                q_ref,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dk],
            ),
            1,
        );
        let k_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                k_ref,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dk],
            ),
            1,
        );
        let v_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(
                v,
                &[0, t_idx as i32, 0, 0],
                &[b, (t_idx + 1) as i32, hv, dv],
            ),
            1,
        );

        let g_ndim = mlxcel_core::array_ndim(g);
        let g_t = if g_ndim == 4 {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(
                    g,
                    &[0, t_idx as i32, 0, 0],
                    &[b, (t_idx + 1) as i32, hv, dk],
                ),
                1,
            )
        } else {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(g, &[0, t_idx as i32, 0], &[b, (t_idx + 1) as i32, hv]),
                1,
            )
        };

        let beta_t = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(beta, &[0, t_idx as i32, 0], &[b, (t_idx + 1) as i32, hv]),
            1,
        );

        let mask_t = mask.map(|m| {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(m, &[0, t_idx as i32], &[b, (t_idx + 1) as i32]),
                1,
            )
        });

        let (y, new_state) = gated_delta_step(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &current_state,
            mask_t.as_deref(),
        );

        ys.push(y);
        current_state = new_state;
    }

    // Stack outputs: [B, T, Hv, Dv]
    let y = stack_arrays(&ys, 1);

    (y, current_state)
}

/// Chunked parallel prefill scan for the gated delta rule (scalar gating).
///
/// Computes the exact same recurrence as the sequential per-token loop, but
/// with `O(T/C + C)` sequential depth instead of `O(T)`. Used for prefill on
/// backends without the fused Metal kernel (CUDA/CPU). `q_ref`/`k_ref` are
/// already GQA-repeated to `Hv` heads; `g` is scalar gating `[B, T, Hv]`.
///
/// Standard delta-rule chunking with gating. Split the sequence into chunks of
/// length `C`. Within a chunk (local positions `r`, incoming state `S`), the
/// reference recurrence is `S_r = g_r S_{r-1} + w_r k_r^T` with
/// `w_r = beta_r (v_r - g_r S_{r-1} k_r)` and output `y_r = S_r q_r`. Writing
/// the cumulative gate `gamma_r = prod_{i<=r} g_i` and the pseudo-values
/// `W = (I + T)^{-1} RHS`, where `T[r,j] = beta_r (gamma_r/gamma_j)(k_r . k_j)`
/// is strictly lower triangular, the chunk becomes:
///   - `Wv = Ainv (beta ⊙ V)`, `R = Ainv (beta·gamma ⊙ K)` (state-independent),
///   - state scan `S_out = gamma_last ⊙ S - S (R^T Kd) + (Wv^T Kd)`,
///   - output `y = (Pmat Wv) + Qeff S^T` with `Pmat[r,j]=(gamma_r/gamma_j)(k_j.q_r)`
///     lower-triangular and `Qeff = gamma ⊙ Q - Pmat R`.
///
/// `Ainv = (I + T)^{-1}` is formed by the finite Neumann series (T is nilpotent).
/// All decay ratios are kept in the log domain and clamped so no entry overflows.
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
#[allow(clippy::too_many_arguments)]
fn gated_delta_chunked(
    q_ref: &MlxArray,
    k_ref: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    initial_state: &MlxArray,
    chunk: usize,
    b: i32,
    t: usize,
    hv: i32,
    dk: i32,
    dv: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let q_dtype = mlxcel_core::array_dtype(q_ref);

    let cc = chunk as i32;
    let nn = t.div_ceil(chunk) as i32; // number of chunks
    let t_pad = nn * cc;
    let pad_after = t_pad - t as i32;

    // Scan math runs in float32 (the sequential reference accumulates state in
    // float32); cast the possibly-bf16 inputs up front.
    let qf = mlxcel_core::astype(q_ref, dtype::FLOAT32);
    let kf = mlxcel_core::astype(k_ref, dtype::FLOAT32);
    let vf = mlxcel_core::astype(v, dtype::FLOAT32);
    let betaf = mlxcel_core::astype(beta, dtype::FLOAT32);
    let gf = mlxcel_core::astype(g, dtype::FLOAT32);

    // Pad the time axis to a whole number of chunks. Query/key/value/beta pad
    // with zero (inert tokens); the gate pads with one so padded positions
    // apply no decay. With beta = 0 the padded tokens never update the state,
    // so the post-chunk state equals the state after the last real token and
    // the discarded padded outputs cannot corrupt earlier (causal) positions.
    let (qf, kf, vf, betaf, gf) = if pad_after > 0 {
        let pad4 = [0, 0, 0, pad_after, 0, 0, 0, 0];
        let pad3 = [0, 0, 0, pad_after, 0, 0];
        (
            mlxcel_core::pad(&qf, &pad4, 0.0),
            mlxcel_core::pad(&kf, &pad4, 0.0),
            mlxcel_core::pad(&vf, &pad4, 0.0),
            mlxcel_core::pad(&betaf, &pad3, 0.0),
            mlxcel_core::pad(&gf, &pad3, 1.0),
        )
    } else {
        (qf, kf, vf, betaf, gf)
    };

    // Reshape [B, T_pad, Hv, D] -> [B, Hv, N, C, D] (and [B, T_pad, Hv] gate/beta
    // -> [B, Hv, N, C]) so intra-chunk matmuls batch over (B, Hv, N) and
    // contract the last two axes.
    let to_chunks_4d = |x: &MlxArray| -> UniquePtr<MlxArray> {
        let u = mlxcel_core::unflatten(x, 1, &[nn, cc]); // [B, N, C, Hv, D]
        mlxcel_core::transpose_axes(&u, &[0, 3, 1, 2, 4]) // [B, Hv, N, C, D]
    };
    let to_chunks_3d = |x: &MlxArray| -> UniquePtr<MlxArray> {
        let u = mlxcel_core::unflatten(x, 1, &[nn, cc]); // [B, N, C, Hv]
        mlxcel_core::transpose_axes(&u, &[0, 3, 1, 2]) // [B, Hv, N, C]
    };
    let qc = to_chunks_4d(&qf);
    let kc = to_chunks_4d(&kf);
    let vc = to_chunks_4d(&vf);
    let betac = to_chunks_3d(&betaf);
    let gc = to_chunks_3d(&gf);

    // Log-domain cumulative gate: L[.., r] = sum_{i<=r} log g_i = log gamma_r.
    // Floor the log so an underflowed gate cannot inject -inf.
    let neg_cap = mlxcel_core::multiply_scalar(&mlxcel_core::ones_like(&gc), -60.0);
    let lg = mlxcel_core::maximum(&mlxcel_core::log(&gc), &neg_cap);
    let l = mlxcel_core::cumsum(&lg, -1, false, true); // inclusive cumsum over C
    let gamma = mlxcel_core::exp(&l); // [B, Hv, N, C]

    // Decay-ratio matrix ratio[r, j] = gamma_r/gamma_j = exp(L_r - L_j). Clamp
    // the exponent to <= 0 so no entry overflows; the upper triangle is masked
    // out where used.
    let li = mlxcel_core::expand_dims(&l, -1); // [.., C, 1]
    let lj = mlxcel_core::expand_dims(&l, -2); // [.., 1, C]
    let diff = mlxcel_core::subtract(&li, &lj);
    let diff = mlxcel_core::minimum(&diff, &mlxcel_core::zeros_like(&diff));
    let ratio = mlxcel_core::exp(&diff); // [.., C, C]

    // Triangular masks / identity acting on the last two axes.
    let ones_cc = mlxcel_core::ones(&[cc, cc], dtype::FLOAT32);
    let mask_strict = mlxcel_core::tril(&ones_cc, -1); // r > j
    let mask_incl = mlxcel_core::tril(&ones_cc, 0); // r >= j
    let eye_cc = mlxcel_core::identity(cc, dtype::FLOAT32);

    // kk[r, j] = k_r . k_j ; qk[r, j] = q_r . k_j
    let kt = mlxcel_core::swap_axes(&kc, -1, -2); // [.., Dk, C]
    let kk = mlxcel_core::matmul(&kc, &kt);
    let qk = mlxcel_core::matmul(&qc, &kt);

    let beta_e = mlxcel_core::expand_dims(&betac, -1); // [.., C, 1]
    let gamma_e = mlxcel_core::expand_dims(&gamma, -1); // [.., C, 1]

    // Strictly-lower intra-chunk matrix T[r, j] = beta_r (gamma_r/gamma_j)(k_r . k_j).
    let t_mat = mlxcel_core::multiply(
        &mlxcel_core::multiply(&mlxcel_core::multiply(&beta_e, &ratio), &kk),
        &mask_strict,
    );

    // Ainv = (I + T)^{-1} via the finite Neumann series (T strictly lower, so
    // T^C = 0): sum_{p=0}^{C-1} (-T)^p, computed with C-1 Horner steps.
    let t_shape = mlxcel_core::array_shape(&t_mat);
    let mut ainv = mlxcel_core::broadcast_to(&eye_cc, &t_shape); // [.., C, C]
    for _ in 1..chunk {
        ainv = mlxcel_core::subtract(&eye_cc, &mlxcel_core::matmul(&t_mat, &ainv));
    }

    // Wv = Ainv (beta ⊙ V) ; R = Ainv (beta·gamma ⊙ K).
    let bg_e = mlxcel_core::expand_dims(&mlxcel_core::multiply(&betac, &gamma), -1); // [.., C, 1]
    let wv = mlxcel_core::matmul(&ainv, &mlxcel_core::multiply(&beta_e, &vc)); // [.., C, Dv]
    let r_mat = mlxcel_core::matmul(&ainv, &mlxcel_core::multiply(&bg_e, &kc)); // [.., C, Dk]

    // d_j = gamma_last/gamma_j (<= 1) ; Kd = d ⊙ K.
    let l_last = mlxcel_core::slice(&l, &[0, 0, 0, cc - 1], &[b, hv, nn, cc]); // [.., 1]
    let d_e = mlxcel_core::expand_dims(&mlxcel_core::exp(&mlxcel_core::subtract(&l_last, &l)), -1);
    let kd = mlxcel_core::multiply(&d_e, &kc); // [.., C, Dk]

    // State-scan operators (independent of the incoming state).
    let bmat = mlxcel_core::matmul(&mlxcel_core::swap_axes(&r_mat, -1, -2), &kd); // [.., Dk, Dk]
    let cmat = mlxcel_core::matmul(&mlxcel_core::swap_axes(&wv, -1, -2), &kd); // [.., Dv, Dk]
    let gamma_last = mlxcel_core::exp(&l_last); // [B, Hv, N, 1]

    // Output operators: Pmat[r, j] = (gamma_r/gamma_j)(k_j . q_r) for j <= r.
    let pmat = mlxcel_core::multiply(&mlxcel_core::multiply(&ratio, &qk), &mask_incl);
    let pmat_wv = mlxcel_core::matmul(&pmat, &wv); // [.., C, Dv]
    let qeff = mlxcel_core::subtract(
        &mlxcel_core::multiply(&gamma_e, &qc),
        &mlxcel_core::matmul(&pmat, &r_mat),
    ); // [.., C, Dk]

    // Sequential inter-chunk state scan (N steps). S: [B, Hv, Dv, Dk].
    let mut state = mlxcel_core::astype(initial_state, dtype::FLOAT32);
    let mut ys: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(nn as usize);
    for n in 0..nn {
        let qeff_n = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(&qeff, &[0, 0, n, 0, 0], &[b, hv, n + 1, cc, dk]),
            2,
        ); // [B, Hv, C, Dk]
        let pmwv_n = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(&pmat_wv, &[0, 0, n, 0, 0], &[b, hv, n + 1, cc, dv]),
            2,
        ); // [B, Hv, C, Dv]
        let bmat_n = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(&bmat, &[0, 0, n, 0, 0], &[b, hv, n + 1, dk, dk]),
            2,
        ); // [B, Hv, Dk, Dk]
        let cmat_n = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(&cmat, &[0, 0, n, 0, 0], &[b, hv, n + 1, dv, dk]),
            2,
        ); // [B, Hv, Dv, Dk]
        let gl_n = mlxcel_core::expand_dims(
            &mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(&gamma_last, &[0, 0, n, 0], &[b, hv, n + 1, 1]),
                2,
            ),
            -1,
        ); // [B, Hv, 1, 1]

        // y_n = Pmat Wv + Qeff S^T
        let y_inter = mlxcel_core::matmul(&qeff_n, &mlxcel_core::swap_axes(&state, -1, -2));
        ys.push(mlxcel_core::add(&pmwv_n, &y_inter));

        // S_out = gamma_last ⊙ S - S B + Cst
        let decayed = mlxcel_core::multiply(&gl_n, &state);
        let corrected = mlxcel_core::matmul(&state, &bmat_n);
        state = mlxcel_core::add(&mlxcel_core::subtract(&decayed, &corrected), &cmat_n);
    }

    // Reassemble outputs [B, Hv, N, C, Dv] -> [B, T, Hv, Dv], dropping padding.
    let y = stack_arrays(&ys, 2); // [B, Hv, N, C, Dv]
    let y = mlxcel_core::reshape(&y, &[b, hv, t_pad, dv]); // [B, Hv, T_pad, Dv]
    let y = mlxcel_core::transpose_axes(&y, &[0, 2, 1, 3]); // [B, T_pad, Hv, Dv]
    let y = if pad_after > 0 {
        mlxcel_core::slice(&y, &[0, 0, 0, 0], &[b, t as i32, hv, dv])
    } else {
        y
    };
    let y = if mlxcel_core::array_dtype(&y) != q_dtype {
        mlxcel_core::astype(&y, q_dtype)
    } else {
        y
    };

    (y, state)
}

/// Return whether the custom Metal gated-delta kernel supports this shape.
///
/// The kernel stores `Dk / 32` values in a compile-time stack array per SIMD
/// lane, so `Dk` must cover at least one full SIMD group and be exactly
/// divisible by 32. Its GQA mapping also requires an integral `Hv / Hk`.
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
fn supports_metal_gated_delta_kernel(hk: i32, hv: i32, dk: i32, dv: i32) -> bool {
    hk > 0 && hv > 0 && dv > 0 && dk >= 32 && dk % 32 == 0 && hv >= hk && hv % hk == 0
}

/// Main gated delta update function.
///
/// Computes beta = sigmoid(b) and g = exp(-exp(A_log.float32) * softplus(a + dt_bias)),
/// then runs the ops-based implementation with float32 state accumulation.
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub fn gated_delta_update(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a: &MlxArray,
    b: &MlxArray,
    a_log: &MlxArray,
    dt_bias: &MlxArray,
    state: Option<&MlxArray>,
    mask: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Compute beta = sigmoid(b)
    let beta = mlxcel_core::sigmoid(b);

    // Compute gating g = exp(-exp(A_log.float32) * softplus(a + dt_bias)); stays float32
    let g = compute_g(a_log, a, dt_bias);

    // Run the ops-based implementation
    gated_delta_ops(q, k, v, &g, &beta, state, mask)
}

/// Fast RMS normalization without a learned scale, followed by scalar scaling.
///
/// Mirrors mlx-lm's `mx.fast.rms_norm(x, None, eps) * scale` path for linear
/// attention q/k normalization, avoiding the expanded square/mean/sqrt/divide
/// graph on every decode step.
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub fn scaled_fast_rms_norm_no_weight(x: &MlxArray, scale: f32, eps: f32) -> UniquePtr<MlxArray> {
    let normed = mlxcel_core::fast_rms_norm_no_weight(x, eps);
    mlxcel_core::multiply_scalar(&normed, scale)
}

// RMSNorm with optional gating.
/// RMSNorm with optional SwiGLU gating (for GatedDeltaNet output).
/// Gating: silu(gate) * rms_norm(x)
///
/// Used by: Qwen3Next, Qwen3.5, KimiLinear
pub struct RMSNormGated {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl RMSNormGated {
    pub fn new(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &MlxArray, gate: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let target_dtype = mlxcel_core::array_dtype(x);

        // Reference mlx-lm uses mx.fast.rms_norm here. Keeping this as the
        // fast kernel avoids expanding the gated delta decode graph with
        // square/mean/sqrt/divide/multiply per linear-attention layer.
        let scaled = mlxcel_core::fast_rms_norm(x, &self.weight, self.eps);

        // Apply SwiGLU gating: silu(gate) * x
        // Python mlx-lm promotes the gated path to float32 before restoring the
        // hidden-state dtype so Qwen3Next/Qwen3.5 keep the expected precision.
        if let Some(g) = gate {
            precise_swiglu_gate(&scaled, g, target_dtype)
        } else {
            restore_dtype(scaled, target_dtype)
        }
    }
}

fn restore_dtype(value: UniquePtr<MlxArray>, target_dtype: i32) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(&value) == target_dtype {
        value
    } else {
        mlxcel_core::astype(&value, target_dtype)
    }
}

fn precise_swiglu_gate(x: &MlxArray, gate: &MlxArray, target_dtype: i32) -> UniquePtr<MlxArray> {
    let gate_f32 = mlxcel_core::astype(gate, dtype::FLOAT32);
    let gate_silu = silu(&gate_f32);
    let x_f32 = mlxcel_core::astype(x, dtype::FLOAT32);
    let product = mlxcel_core::multiply(&gate_silu, &x_f32);
    restore_dtype(product, target_dtype)
}

#[cfg(test)]
#[path = "gated_delta_tests.rs"]
mod tests;
