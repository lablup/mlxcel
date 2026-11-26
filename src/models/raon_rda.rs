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

//! Raon Delta Attention (RDA): the gated delta-rule linear attention of the
//! Raon-1 family.
//!
//! ```text
//! x      = x * token_mask                                padded rows contribute nothing
//! qkv    = [q_proj(x) | k_proj(x) | v_proj(x)]           [B, T, 3P], P = H * D
//! qkv    = silu(conv1d(qkv.f32; qkv_conv1d)).astype(dt)   depthwise causal, kernel K, one rounding
//! q      = (q.f32 / max(|q|, 1e-6)).astype(dt) * D^-0.5   rounded before the scale
//! k      = (k.f32 / max(|k|, 1e-6)).astype(dt)
//! ratio  = sigmoid(b + w) / sigmoid(b)                   b = b_proj(x).f32, w = rda_w_proj(x).f32
//! v      = (v.f32 * ratio).astype(dt)                     write gate as a value scale
//! beta   = sigmoid(b).astype(dt)
//! g      = exp(lb * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias[h, :])))
//! out, S = gated_delta(q, k, v, g, beta, S)              S: [B, H, D, D] float32
//! y      = o_proj((rms_norm(out.f32; o_norm) * sigmoid(g_proj(x).f32)).astype(dt))
//! ```
//!
//! The three checkpoint convs (`q_conv1d`, `k_conv1d`, `v_conv1d`) are
//! concatenated in q, k, v order into one `qkv_conv1d` by sanitize, which is
//! exact for a depthwise conv. The conv state is the last `K - 1` raw
//! (pre-conv) projection rows.
//!
//! Reference: `Raon1MiniRDA` in
//! https://huggingface.co/KRAFTON/Raon-1-mini/blob/6691a076/modeling_raon1_mini.py
//!
//! Used by: Raon1Mini; kept free of Raon-1-mini config types so Raon-1 nano
//! (#3438) can load it unchanged.

use crate::models::gated_delta::{compute_g_lower_bounded, gated_delta_ops_precise};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// L2-normalization floor of the reference (`F.normalize(eps=1e-6)`).
pub const RDA_L2NORM_EPS: f32 = 1e-6;

/// Recurrent state of one RDA layer: the raw conv window (`[B, K-1, 3P]` in
/// the activation dtype) and the float32 delta-rule state (`[B, H, D, D]`).
#[derive(Default)]
pub struct RaonDeltaCache {
    pub conv: Option<UniquePtr<MlxArray>>,
    pub ssm: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
}

impl RaonDeltaCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Geometry of one RDA layer.
#[derive(Debug, Clone, Copy)]
pub struct RdaSpec {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kernel_size: usize,
    pub gate_lower_bound: Option<f32>,
    pub norm_eps: f32,
    pub group_size: i32,
    pub bits: i32,
}

pub struct RaonDeltaAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    g_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    f_a_proj: UnifiedLinear,
    f_b_proj: UnifiedLinear,
    b_proj: UnifiedLinear,
    w_proj: UnifiedLinear,
    /// `[3P, K, 1]` float32 (exact upcast of the checkpoint conv).
    conv_weight: UniquePtr<MlxArray>,
    /// `[1, K, 3P]` float32, the single-token form.
    conv_decode_weight: UniquePtr<MlxArray>,
    /// `[H, 1]` float32.
    a_log: UniquePtr<MlxArray>,
    /// `[H, D]` float32.
    dt_bias: UniquePtr<MlxArray>,
    /// `[D]` float32.
    o_norm: UniquePtr<MlxArray>,
    spec: RdaSpec,
}

fn take(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Missing weight: {key}"))
}

fn check_rows(weights: &WeightMap, key: &str, rows: usize, what: &str) -> Result<(), String> {
    if let Some(w) = weights.get(key) {
        let shape = mlxcel_core::array_shape(w);
        if shape.first().copied() != Some(rows as i32) {
            return Err(format!(
                "{key}: shape {shape:?}, expected {rows} output rows ({what})"
            ));
        }
    }
    Ok(())
}

fn flat_f32(weights: &WeightMap, key: &str, len: usize) -> Result<UniquePtr<MlxArray>, String> {
    let w = take(weights, key)?;
    let numel: i32 = mlxcel_core::array_shape(&w).iter().product();
    if numel as usize != len {
        return Err(format!("{key}: {numel} entries, expected {len}"));
    }
    Ok(mlxcel_core::reshape(
        &mlxcel_core::astype(&w, dtype::FLOAT32),
        &[len as i32],
    ))
}

impl RaonDeltaAttention {
    pub fn from_weights(weights: &WeightMap, prefix: &str, spec: RdaSpec) -> Result<Self, String> {
        let p = spec.num_heads * spec.head_dim;
        let (gs, bits) = (spec.group_size, spec.bits);
        let linear = |leaf: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), gs, bits)
        };
        for leaf in ["q_proj", "k_proj", "v_proj", "g_proj", "f_b_proj"] {
            check_rows(
                weights,
                &format!("{prefix}.{leaf}.weight"),
                p,
                "rda num_heads * head_dim",
            )?;
        }
        for leaf in ["b_proj", "rda_w_proj"] {
            check_rows(
                weights,
                &format!("{prefix}.{leaf}.weight"),
                spec.num_heads,
                "rda num_heads",
            )?;
        }
        check_rows(
            weights,
            &format!("{prefix}.o_proj.weight"),
            spec.hidden_size,
            "hidden_size",
        )?;

        let conv_key = format!("{prefix}.qkv_conv1d.weight");
        let conv = take(weights, &conv_key)?;
        let expected = [3 * p as i32, spec.kernel_size as i32, 1];
        if mlxcel_core::array_shape(&conv) != expected {
            return Err(format!(
                "{conv_key}: expected shape {expected:?} (q|k|v channels, kernel, 1), got {:?}",
                mlxcel_core::array_shape(&conv)
            ));
        }
        let conv_weight = mlxcel_core::astype(&conv, dtype::FLOAT32);
        let conv_decode_weight = crate::models::conv_decode::build_conv_decode_weight(&conv_weight);

        let a_log = flat_f32(weights, &format!("{prefix}.A_log"), spec.num_heads)?;
        let a_log = mlxcel_core::reshape(&a_log, &[spec.num_heads as i32, 1]);
        let dt_bias = flat_f32(weights, &format!("{prefix}.dt_bias"), p)?;
        let dt_bias =
            mlxcel_core::reshape(&dt_bias, &[spec.num_heads as i32, spec.head_dim as i32]);
        let o_norm = flat_f32(weights, &format!("{prefix}.o_norm.weight"), spec.head_dim)?;

        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            g_proj: linear("g_proj")?,
            o_proj: linear("o_proj")?,
            f_a_proj: linear("f_a_proj")?,
            f_b_proj: linear("f_b_proj")?,
            b_proj: linear("b_proj")?,
            w_proj: linear("rda_w_proj")?,
            conv_weight,
            conv_decode_weight,
            a_log,
            dt_bias,
            o_norm,
            spec,
        })
    }

    fn projection_dim(&self) -> i32 {
        (self.spec.num_heads * self.spec.head_dim) as i32
    }

    /// Depthwise causal conv over `[state | qkv]` in float32, SiLU, one
    /// rounding to the input dtype. Returns `(out, new_raw_window)`.
    pub fn short_conv(
        &self,
        qkv: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(qkv);
        let (b, t) = (shape[0], shape[1]);
        let in_dtype = mlxcel_core::array_dtype(qkv);
        let channels = 3 * self.projection_dim();
        let keep = self.spec.kernel_size as i32 - 1;
        let state = match state {
            Some(s) => mlxcel_core::copy(s),
            None => mlxcel_core::zeros(&[b, keep, channels], in_dtype),
        };
        let padded = mlxcel_core::concatenate(&state, qkv, 1);
        let padded_f32 = mlxcel_core::astype(&padded, dtype::FLOAT32);
        let out = if t == 1 {
            let prod = mlxcel_core::multiply(&padded_f32, &self.conv_decode_weight);
            mlxcel_core::sum_axis(&prod, 1, true)
        } else {
            mlxcel_core::conv1d(&padded_f32, &self.conv_weight, 1, 0, 1, channels)
        };
        let out = mlxcel_core::astype(&silu(&out), in_dtype);
        let new_state = if keep > 0 {
            mlxcel_core::utils::materialized_state_tail(&padded, keep)
        } else {
            mlxcel_core::zeros(&[b, 0, channels], in_dtype)
        };
        (out, new_state)
    }

    /// One RDA layer over `x` (`[B, T, hidden]`). `token_mask` (`[B, T]`,
    /// 1 for real tokens, 0 for padding) zeroes padded rows before every
    /// projection, so they enter neither the conv window nor the state.
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut RaonDeltaCache,
        token_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, t) = (shape[0], shape[1]);
        let dt = mlxcel_core::array_dtype(x);
        let (h, d, p) = (
            self.spec.num_heads as i32,
            self.spec.head_dim as i32,
            self.projection_dim(),
        );
        let x = match token_mask {
            Some(m) => {
                let m = mlxcel_core::astype(&mlxcel_core::expand_dims(m, -1), dt);
                mlxcel_core::multiply(x, &m)
            }
            None => mlxcel_core::copy(x),
        };

        let q = self.q_proj.forward(&x);
        let k = self.k_proj.forward(&x);
        let v = self.v_proj.forward(&x);
        let qkv = mlxcel_core::concatenate_many(&[&*q, &*k, &*v], -1);
        let (qkv, conv_state) = self.short_conv(&qkv, cache.conv.as_deref());
        cache.conv = Some(conv_state);

        let per_head = [b, t, h, d];
        let part = |i: i32| {
            let s = mlxcel_core::slice(&qkv, &[0, 0, i * p], &[b, t, (i + 1) * p]);
            mlxcel_core::reshape(&s, &per_head)
        };
        let q = l2_normalize_round(&part(0), dt);
        let scale = (self.spec.head_dim as f32).powf(-0.5);
        let q = mlxcel_core::astype(
            &mlxcel_core::multiply_scalar(&mlxcel_core::astype(&q, dtype::FLOAT32), scale),
            dt,
        );
        let k = l2_normalize_round(&part(1), dt);

        let beta_raw = mlxcel_core::astype(&self.b_proj.forward(&x), dtype::FLOAT32);
        let delta = mlxcel_core::astype(&self.w_proj.forward(&x), dtype::FLOAT32);
        let ratio = write_ratio(&beta_raw, &delta);
        let v = mlxcel_core::multiply(
            &mlxcel_core::astype(&part(2), dtype::FLOAT32),
            &mlxcel_core::expand_dims(&ratio, -1),
        );
        let v = mlxcel_core::astype(&v, dt);
        let beta = mlxcel_core::astype(&mlxcel_core::sigmoid(&beta_raw), dt);

        let decay = self.f_b_proj.forward(&self.f_a_proj.forward(&x));
        let decay = mlxcel_core::reshape(&decay, &per_head);
        let g = rda_decay(
            &self.a_log,
            &decay,
            &self.dt_bias,
            self.spec.gate_lower_bound,
        );

        let (out, ssm) = rda_core(&q, &k, &v, &g, &beta, cache.ssm.as_deref());
        cache.ssm = Some(ssm);
        cache.offset += t;

        let out_f32 = mlxcel_core::astype(&out, dtype::FLOAT32);
        let normed = mlxcel_core::fast_rms_norm(&out_f32, &self.o_norm, self.spec.norm_eps);
        let gate = mlxcel_core::reshape(&self.g_proj.forward(&x), &per_head);
        let gate = mlxcel_core::sigmoid(&mlxcel_core::astype(&gate, dtype::FLOAT32));
        let gated = mlxcel_core::astype(&mlxcel_core::multiply(&normed, &gate), dt);
        self.o_proj
            .forward(&mlxcel_core::reshape(&gated, &[b, t, p]))
    }
}

/// The delta-rule recurrence of one RDA layer: `q`, `k`, `v`, `beta` in the
/// activation dtype, the decay `g` (`[B, T, H, D]`) and the state in float32.
/// Returns the output in the activation dtype and the float32 state.
///
/// The reference keeps the log decay and the state in fp32, so this takes
/// the precise kernel: the standard one rounds `g` to bf16, where a decay of
/// 0.998 becomes 0.996, and that error compounds once per token.
///
/// Used by: [`RaonDeltaAttention::forward`]
pub fn rda_core(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
    state: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    gated_delta_ops_precise(q, k, v, g, beta, state, None)
}

/// RDA decay `g = exp(log_alpha)`, float32, with
/// `s = exp(A_log[h]) * (a.f32 + dt_bias[h, :])` and
/// `log_alpha = lb * sigmoid(s)`, or `-softplus(s)` when `gate_lower_bound`
/// is null.
///
/// The bounded form is [`compute_g_lower_bounded`]. The unbounded one is not
/// [`crate::models::gated_delta::compute_g`]: that is
/// `-exp(A_log) * softplus(a + dt_bias)`, with `exp(A_log)` outside the
/// softplus, where the reference scales inside it.
///
/// Used by: [`RaonDeltaAttention::forward`]
pub fn rda_decay(
    a_log: &MlxArray,
    a: &MlxArray,
    dt_bias: &MlxArray,
    gate_lower_bound: Option<f32>,
) -> UniquePtr<MlxArray> {
    if gate_lower_bound.is_some() {
        return compute_g_lower_bounded(a_log, a, dt_bias, gate_lower_bound);
    }
    let a_plus_dt = mlxcel_core::add(&mlxcel_core::astype(a, dtype::FLOAT32), dt_bias);
    let scale = mlxcel_core::exp(&mlxcel_core::astype(a_log, dtype::FLOAT32));
    let scaled = mlxcel_core::multiply(&scale, &a_plus_dt);
    let softplus = mlxcel_core::utils::softplus(&scaled);
    mlxcel_core::exp(&mlxcel_core::negative(&softplus))
}

/// `(x.f32 / max(||x||, 1e-6)).astype(dt)` over the last axis.
pub fn l2_normalize_round(x: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
    let xf = mlxcel_core::astype(x, dtype::FLOAT32);
    let norm = mlxcel_core::sqrt(&mlxcel_core::sum_axis(&mlxcel_core::square(&xf), -1, true));
    let floor = mlxcel_core::full_f32(&[1], RDA_L2NORM_EPS, dtype::FLOAT32);
    let denom = mlxcel_core::maximum(&norm, &floor);
    mlxcel_core::astype(&mlxcel_core::divide(&xf, &denom), dt)
}

/// RDA write ratio `sigmoid(b + w) / sigmoid(b)`, evaluated as the reference
/// does, `exp(logsigmoid(b + w) - logsigmoid(b))`, with
/// `logsigmoid(z) = -softplus(-z)`. Float32 in, float32 out.
pub fn write_ratio(beta_raw: &MlxArray, delta: &MlxArray) -> UniquePtr<MlxArray> {
    use mlxcel_core::utils::softplus;
    let sum = mlxcel_core::add(beta_raw, delta);
    let log_write = mlxcel_core::negative(&softplus(&mlxcel_core::negative(&sum)));
    let log_erase = mlxcel_core::negative(&softplus(&mlxcel_core::negative(beta_raw)));
    mlxcel_core::exp(&mlxcel_core::subtract(&log_write, &log_erase))
}
