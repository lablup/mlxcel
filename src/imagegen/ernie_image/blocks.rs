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

//! ERNIE-Image DiT blocks: rotary tables, timestep embedding, attention, MLP
//! and the AdaLN transformer block.
//!
//! A port of `timestep_embedding`, `rope_frequencies`, `ErnieImageAttention`,
//! `ErnieImageMLP`, `_modulate` and `ErnieImageTransformerBlock` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/transformer.py>.
//!
//! Two details differ from the Z-Image blocks in
//! [`crate::imagegen::dit_blocks`], which is why these are not reused:
//! the rotary tables repeat each angle on adjacent entries but rotate with the
//! half-split `rotate_half`, and the modulation and gates run as explicit f32
//! islands cast back to the stream dtype.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

pub(super) fn linear(weights: &WeightMap, prefix: &str) -> Result<Linear> {
    Linear::from_weights(weights, prefix)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("ERNIE-Image transformer weight {prefix}"))
}

fn rms_norm(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm> {
    let w = weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("ERNIE-Image transformer weight not found: {key}"))?;
    Ok(RMSNorm::new(w, eps))
}

fn scalar(value: f32, dt: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dt)
}

/// `[B, D]` -> `[B, 1, D]`.
/// `nn.Linear`: `addmm(bias, x, W.T)` when there is a bias (one rounding of
/// the biased product, as MLX's `nn.Linear` does), else `x @ W.T`.
pub(super) fn affine(layer: &Linear, x: &MlxArray) -> UniquePtr<MlxArray> {
    let wt = mlxcel_core::transpose(&layer.weight);
    match &layer.bias {
        Some(bias) => mlxcel_core::addmm(bias, x, &wt, 1.0, 1.0),
        None => mlxcel_core::matmul(x, &wt),
    }
}

/// MLX `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2` with the constants in the
/// input dtype (Python scalars are weakly typed), so a bf16 activation stays
/// bf16 at every step instead of being computed in f32 and rounded once.
pub(super) fn gelu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let erf = mlxcel_core::erf(&mlxcel_core::divide(
        x,
        &scalar(std::f32::consts::SQRT_2, dt),
    ));
    let y = mlxcel_core::multiply(x, &mlxcel_core::add(&scalar(1.0, dt), &erf));
    mlxcel_core::divide(&y, &scalar(2.0, dt))
}

pub(super) fn per_token(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(x, 1)
}

/// Split the last axis into `n` equal chunks.
pub(super) fn split_last(x: &MlxArray, n: i32) -> Vec<UniquePtr<MlxArray>> {
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

/// Upstream `timestep_embedding(t, dim)`: `[sin, cos]` of `t * exp(-ln(10000)
/// * i / half)` in f32, `[B, dim]`. `dim` must be even (the published
/// checkpoints use 4096).
pub fn timestep_embedding(timesteps: &MlxArray, dim: i32) -> UniquePtr<MlxArray> {
    let t = mlxcel_core::astype(timesteps, dtype::FLOAT32);
    let t = mlxcel_core::reshape(&t, &[-1, 1]);
    let half = dim / 2;
    let ramp = mlxcel_core::arange_f32(0.0, half as f32, 1.0);
    let exponent = mlxcel_core::divide(
        &mlxcel_core::multiply(&scalar(-(10_000.0_f64.ln()) as f32, dtype::FLOAT32), &ramp),
        &scalar(half as f32, dtype::FLOAT32),
    );
    let freqs = mlxcel_core::expand_dims(&mlxcel_core::exp(&exponent), 0);
    let phases = mlxcel_core::multiply(&t, &freqs);
    mlxcel_core::concatenate(&mlxcel_core::sin(&phases), &mlxcel_core::cos(&phases), -1)
}

/// Upstream `rope_frequencies`: `(cos, sin)`, each `[B, 1, L, head_dim]` f32,
/// from `positions` (`[B, L, 3]` f32) and the per-axis widths. Each axis
/// contributes `positions[..., axis] / theta^(2j / dim)`, every angle repeated
/// on two adjacent entries.
pub fn rope_tables(
    positions: &MlxArray,
    axes_dim: &[i32],
    theta: f32,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let shape = mlxcel_core::array_shape(positions);
    if shape.len() != 3 || shape[2] != axes_dim.len() as i32 {
        bail!(
            "expected [B, L, {}] positions, got {shape:?}",
            axes_dim.len()
        );
    }
    let (b, l) = (shape[0], shape[1]);
    let mut parts = Vec::with_capacity(axes_dim.len());
    for (axis, &dim) in axes_dim.iter().enumerate() {
        let axis = axis as i32;
        let ramp = mlxcel_core::divide(
            &mlxcel_core::arange_f32(0.0, dim as f32, 2.0),
            &scalar(dim as f32, dtype::FLOAT32),
        );
        let omega = mlxcel_core::divide(
            &scalar(1.0, dtype::FLOAT32),
            &mlxcel_core::power(&scalar(theta, dtype::FLOAT32), &ramp),
        );
        let pos = mlxcel_core::slice(positions, &[0, 0, axis], &[b, l, axis + 1]);
        let angles = mlxcel_core::multiply(&pos, &omega);
        let pairs = mlxcel_core::stack_owned(&[mlxcel_core::copy(&angles), angles], -1);
        parts.push(mlxcel_core::reshape(&pairs, &[b, l, dim]));
    }
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    let angles = mlxcel_core::expand_dims(&mlxcel_core::concatenate_many(&refs, -1), 1);
    Ok((mlxcel_core::cos(&angles), mlxcel_core::sin(&angles)))
}

/// `rotate_half`: `concat(-x[..., D/2:], x[..., :D/2])`.
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let halves = split_last(x, 2);
    mlxcel_core::concatenate(&mlxcel_core::negative(&halves[1]), &halves[0], -1)
}

/// `x * cos + rotate_half(x) * sin` on `[B, H, L, D]`.
fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(
        &mlxcel_core::multiply(x, cos),
        &mlxcel_core::multiply(&rotate_half(x), sin),
    )
}

/// `_modulate`: `(x.f32 * (1 + scale) + shift).astype(x.dtype)`, with f32
/// `[B, D]` shift and scale.
fn modulate(x: &MlxArray, shift: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let x32 = mlxcel_core::astype(x, dtype::FLOAT32);
    let one_plus = mlxcel_core::add(&scalar(1.0, dtype::FLOAT32), &per_token(scale));
    let y = mlxcel_core::add(&mlxcel_core::multiply(&x32, &one_plus), &per_token(shift));
    mlxcel_core::astype(&y, dt)
}

/// `h + (gate.f32 * branch.f32).astype(h.dtype)`.
fn gated_residual(h: &MlxArray, gate: &MlxArray, branch: &MlxArray) -> UniquePtr<MlxArray> {
    let branch = mlxcel_core::astype(branch, dtype::FLOAT32);
    let gated = mlxcel_core::multiply(&per_token(gate), &branch);
    mlxcel_core::add(h, &mlxcel_core::astype(&gated, mlxcel_core::array_dtype(h)))
}

/// Per-call inputs every block shares.
pub struct BlockInputs<'a> {
    /// Six f32 `[B, D]` tensors: shift, scale, gate for attention, then MLP.
    pub modulation: &'a [UniquePtr<MlxArray>],
    /// `[B, 1, L, head_dim]` tables in the stream dtype.
    pub cos: &'a MlxArray,
    pub sin: &'a MlxArray,
    /// `[B, 1, 1, L]` additive mask in the stream dtype.
    pub mask: &'a MlxArray,
}

/// `ErnieImageAttention`: q/k RMS norms, rotary, SDPA over the joint sequence.
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Option<RMSNorm>,
    norm_k: Option<RMSNorm>,
    num_heads: i32,
    head_dim: i32,
}

impl Attention {
    fn forward(&self, x: &MlxArray, inputs: &BlockInputs<'_>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let heads = [b, l, self.num_heads, self.head_dim];
        let mut q = mlxcel_core::reshape(&self.to_q.forward(x), &heads);
        let mut k = mlxcel_core::reshape(&self.to_k.forward(x), &heads);
        let v = mlxcel_core::reshape(&self.to_v.forward(x), &heads);
        if let (Some(nq), Some(nk)) = (&self.norm_q, &self.norm_k) {
            q = nq.forward(&q);
            k = nk.forward(&k);
        }
        let to_heads = |t: &MlxArray| mlxcel_core::transpose_axes(t, &[0, 2, 1, 3]);
        let q = apply_rope(&to_heads(&q), inputs.cos, inputs.sin);
        let k = apply_rope(&to_heads(&k), inputs.cos, inputs.sin);
        // The shared dispatch query-chunks the score matrix MLX's fallback
        // would materialize (4096 image tokens at 1024x1024, 16384 at 2048).
        let out = mlxcel_core::layers::attention(
            &q,
            &k,
            &to_heads(&v),
            (self.head_dim as f32).powf(-0.5),
            Some(inputs.mask),
            0.0,
            0,
        );
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[b, l, self.num_heads * self.head_dim],
        );
        self.to_out.forward(&out)
    }
}

/// `ErnieImageMLP`: `linear_fc2(up(x) * gelu(gate(x)))`, exact (erf) GELU.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    linear_fc2: Linear,
}

impl Mlp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = gelu(&self.gate_proj.forward(x));
        self.linear_fc2
            .forward(&mlxcel_core::multiply(&self.up_proj.forward(x), &gate))
    }
}

/// `ErnieImageTransformerBlock`: AdaLN-modulated attention and MLP.
pub struct ErnieBlock {
    adaln_sa_ln: RMSNorm,
    attention: Attention,
    adaln_mlp_ln: RMSNorm,
    mlp: Mlp,
}

impl ErnieBlock {
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        qk_norm: bool,
        eps: f32,
    ) -> Result<Self> {
        let attn = format!("{prefix}.self_attention");
        let (norm_q, norm_k) = if qk_norm {
            (
                Some(rms_norm(weights, &format!("{attn}.norm_q.weight"), eps)?),
                Some(rms_norm(weights, &format!("{attn}.norm_k.weight"), eps)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            adaln_sa_ln: rms_norm(weights, &format!("{prefix}.adaLN_sa_ln.weight"), eps)?,
            attention: Attention {
                to_q: linear(weights, &format!("{attn}.to_q"))?,
                to_k: linear(weights, &format!("{attn}.to_k"))?,
                to_v: linear(weights, &format!("{attn}.to_v"))?,
                to_out: linear(weights, &format!("{attn}.to_out"))?,
                norm_q,
                norm_k,
                num_heads,
                head_dim,
            },
            adaln_mlp_ln: rms_norm(weights, &format!("{prefix}.adaLN_mlp_ln.weight"), eps)?,
            mlp: Mlp {
                gate_proj: linear(weights, &format!("{prefix}.mlp.gate_proj"))?,
                up_proj: linear(weights, &format!("{prefix}.mlp.up_proj"))?,
                linear_fc2: linear(weights, &format!("{prefix}.mlp.linear_fc2"))?,
            },
        })
    }

    pub fn forward(&self, h: &MlxArray, inputs: &BlockInputs<'_>) -> UniquePtr<MlxArray> {
        let m = inputs.modulation;
        let normed = modulate(&self.adaln_sa_ln.forward(h), &m[0], &m[1]);
        let attended = self.attention.forward(&normed, inputs);
        let h = gated_residual(h, &m[2], &attended);
        let normed = modulate(&self.adaln_mlp_ln.forward(&h), &m[3], &m[4]);
        let mlp = self.mlp.forward(&normed);
        gated_residual(&h, &m[5], &mlp)
    }
}

#[cfg(test)]
#[path = "blocks_tests.rs"]
mod tests;
