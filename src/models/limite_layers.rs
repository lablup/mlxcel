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

//! Numerics of the Paradigma Limite decoder (issue #2360): the weightless
//! dtype-eps RMSNorm, the bf16 scalar fold, the partial interleaved RoPE, the
//! MUDD mixer, and the attention block with value embeddings, XSA and the
//! per-head gate. Every function mirrors one line of
//! `mlx_vlm/models/limite/language.py`, including its dtype casts, because
//! each one is a parity trap (see the issue's numerics list).

use mlxcel_core::cache::{KvAttention, attend_batched_rows};
use mlxcel_core::layers::{KVCache, Linear, RotatingKVCache};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// `torch.finfo(dtype).eps`, the epsilon `F.rms_norm(eps=None)` resolves to.
pub(crate) fn finfo_eps(dtype_id: i32) -> f32 {
    match dtype_id {
        dtype::BFLOAT16 => 0.0078125,
        dtype::FLOAT16 => 0.000_976_562_5,
        _ => 1.192_092_9e-7,
    }
}

/// Weightless RMSNorm over the last axis with the input dtype's epsilon.
pub(crate) fn rms_norm(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::fast_rms_norm_no_weight(x, finfo_eps(mlxcel_core::array_dtype(x)))
}

/// `fold(scale, W) = scale.astype(W.dtype) * W`: the f32 scalar is rounded to
/// the weight dtype before the multiply.
pub(crate) fn fold(scale: &MlxArray, weight: &MlxArray) -> UniquePtr<MlxArray> {
    let scale = mlxcel_core::astype(scale, mlxcel_core::array_dtype(weight));
    mlxcel_core::multiply(&scale, weight)
}

/// A one-element array of `value` in `dtype_id`, the weak Python scalar of
/// the reference (it takes the other operand's dtype).
pub(crate) fn scalar(value: f32, dtype_id: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dtype_id)
}

/// `a.astype(x.dtype) * x` for a learned 0-dim scalar `a`.
fn scale_by(a: &MlxArray, x: &MlxArray) -> UniquePtr<MlxArray> {
    let a = mlxcel_core::astype(a, mlxcel_core::array_dtype(x));
    mlxcel_core::multiply(&a, x)
}

/// MLX `nn.gelu` in the input dtype: `x * (1 + erf(x / sqrt(2))) / 2`.
/// `mlxcel_core::gelu` computes in f32 and casts back, which rounds
/// differently from the reference's bf16 arithmetic.
pub(crate) fn gelu_erf(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let t = mlxcel_core::divide(x, &scalar(std::f32::consts::SQRT_2, dt));
    let u = mlxcel_core::add(&scalar(1.0, dt), &mlxcel_core::erf(&t));
    mlxcel_core::divide(&mlxcel_core::multiply(x, &u), &scalar(2.0, dt))
}

/// `x @ w.T` on a dense `[out, in]` weight.
pub(crate) fn matmul_t(x: &MlxArray, w: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::matmul(x, &mlxcel_core::transpose(w))
}

/// Partial interleaved RoPE: `n_pairs` of `head_dim / 2` pairs rotate, the
/// rest pass through (zero frequency). Pairs are adjacent lanes and rotate
/// `(a, b) -> (a cos + b sin, b cos - a sin)`; cos/sin are computed in f32
/// and rounded to bf16, as the reference pins (`rope_cos_sin_dtype`).
pub(crate) struct PartialRope {
    /// `[head_dim]` f32: each pair's frequency twice, then zeros.
    freqs: UniquePtr<MlxArray>,
    /// `[head_dim]` +1 on even lanes, -1 on odd lanes, bf16.
    lane_sign: UniquePtr<MlxArray>,
    head_dim: i32,
}

impl PartialRope {
    pub(crate) fn new(n_pairs: i32, head_dim: i32, base: f32) -> Self {
        let base = scalar(1.0 / base, dtype::FLOAT32);
        let exps = mlxcel_core::linspace(0.0, 1.0, n_pairs, dtype::FLOAT32);
        let freq = mlxcel_core::repeat(&mlxcel_core::power(&base, &exps), 2, 0);
        let pad = mlxcel_core::zeros(&[head_dim - 2 * n_pairs], dtype::FLOAT32);
        let freqs = mlxcel_core::concatenate(&freq, &pad, 0);
        let signs: Vec<f32> = (0..head_dim)
            .map(|i| if i % 2 == 1 { -1.0 } else { 1.0 })
            .collect();
        let lane_sign = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&signs, &[head_dim]),
            dtype::BFLOAT16,
        );
        Self {
            freqs,
            lane_sign,
            head_dim,
        }
    }

    /// Rotate `x` (`[B, L, H, head_dim]`) at `positions` (`[P, L]` f32 with
    /// `P` 1 or `B`).
    pub(crate) fn apply(&self, x: &MlxArray, positions: &MlxArray) -> UniquePtr<MlxArray> {
        let pos_shape = mlxcel_core::array_shape(positions);
        let (p, l) = (pos_shape[0], pos_shape[1]);
        let pos = mlxcel_core::reshape(positions, &[p, l, 1, 1]);
        let freqs = mlxcel_core::reshape(&self.freqs, &[1, 1, 1, self.head_dim]);
        let theta = mlxcel_core::multiply(&pos, &freqs);
        let cos = mlxcel_core::astype(&mlxcel_core::cos(&theta), dtype::BFLOAT16);
        let sin = mlxcel_core::astype(&mlxcel_core::sin(&theta), dtype::BFLOAT16);
        let sin = mlxcel_core::multiply(&sin, &self.lane_sign);

        let dt = mlxcel_core::array_dtype(x);
        let shape = mlxcel_core::array_shape(x);
        let mut pairs = shape.clone();
        pairs.pop();
        pairs.extend([self.head_dim / 2, 2]);
        let x2 = mlxcel_core::reshape(x, &pairs);
        let last = pairs.len() as i32 - 1;
        let a = slice_axis(&x2, last, 0, 1);
        let b = slice_axis(&x2, last, 1, 2);
        let flipped = mlxcel_core::reshape(&mlxcel_core::concatenate(&b, &a, last), &shape);
        let lhs = mlxcel_core::multiply(&mlxcel_core::astype(&cos, dt), x);
        let rhs = mlxcel_core::multiply(&mlxcel_core::astype(&sin, dt), &flipped);
        mlxcel_core::add(&lhs, &rhs)
    }
}

/// `[1, len]` f32 positions starting at `offset`.
pub(crate) fn positions_from_offset(offset: i32, len: i32) -> UniquePtr<MlxArray> {
    let pos = mlxcel_core::arange_f32(offset as f32, (offset + len) as f32, 1.0);
    mlxcel_core::reshape(&pos, &[1, len])
}

/// `[B, 1]` f32 positions, one per decode row.
pub(crate) fn positions_from_offsets(offsets: &[i32]) -> UniquePtr<MlxArray> {
    let pos: Vec<f32> = offsets.iter().map(|&o| o as f32).collect();
    mlxcel_core::from_slice_f32(&pos, &[offsets.len() as i32, 1])
}

/// One MUDD way (attention input or residual) of one mixing layer.
pub(crate) struct MuddWay {
    /// `dense2[layer, :n]` in the activation dtype, `[n, inter]`.
    pub dense2: UniquePtr<MlxArray>,
    /// `bias[layer, :n]` in the activation dtype, `[n]`.
    pub bias: UniquePtr<MlxArray>,
}

/// `combine(values, x_cur)`: per-position tap weights from
/// `gelu(rms_norm(x_cur) @ dense1.T)`, accumulated strictly left to right in
/// the activation dtype (a stacked contraction rounds differently).
pub(crate) fn mudd_combine(
    values: &[&MlxArray],
    x_cur: &MlxArray,
    dense1: &MlxArray,
    way: &MuddWay,
) -> UniquePtr<MlxArray> {
    let inner = gelu_erf(&matmul_t(&rms_norm(x_cur), dense1));
    let w = mlxcel_core::add(&matmul_t(&inner, &way.dense2), &way.bias);
    let last = mlxcel_core::array_ndim(&w) as i32 - 1;
    let mut out: Option<UniquePtr<MlxArray>> = None;
    for (i, value) in values.iter().enumerate() {
        let wi = slice_axis(&w, last, i as i32, i as i32 + 1);
        let wi = mlxcel_core::astype(&wi, mlxcel_core::array_dtype(value));
        let term = mlxcel_core::multiply(&wi, value);
        out = Some(match out {
            None => term,
            Some(acc) => mlxcel_core::add(&acc, &term),
        });
    }
    out.expect("a MUDD layer has at least one tap")
}

/// Sliding layers hold a `RotatingKVCache`, global layers a growing `KVCache`.
pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    pub(crate) fn offset(&self) -> i32 {
        match self {
            Cache::Standard(c) => c.offset,
            Cache::Rotating(c) => c.offset,
        }
    }

    #[cfg(test)]
    pub(crate) fn mode(&self) -> mlxcel_core::cache::KVCacheMode {
        match self {
            Cache::Standard(c) => c.mode,
            Cache::Rotating(c) => c.mode,
        }
    }

    /// Keys the cache currently retains.
    pub(crate) fn visible_len(&self) -> usize {
        match self {
            Cache::Standard(c) => c.live_len().max(0) as usize,
            Cache::Rotating(c) => c.visible_len().max(0) as usize,
        }
    }

    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Cache::Standard(c) => c.update_and_fetch(k, v),
            Cache::Rotating(c) => c.update_and_fetch(k, v),
        }
    }
}

impl KvAttention for Cache {
    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Standard(c) => c.attend(q, k, v, scale, mask),
            Self::Rotating(c) => c.attend(q, k, v, scale, mask),
        }
    }
}

/// Value-embedding gate of a `ve_layers` attention block.
pub(crate) struct ValueEmbed {
    /// `ve_gate[:n_kv]` in the activation dtype, `[heads, channels]`.
    pub gate: UniquePtr<MlxArray>,
    pub gate_scale: f32,
    pub stored_heads: i32,
    pub ve_dim: i32,
}

/// XSA value-direction removal of an `xsa_layers` attention block.
pub(crate) struct Xsa {
    /// `tanh(xsa_alpha)` in f32, `[1, 1, H, 1]`.
    pub alpha: UniquePtr<MlxArray>,
    pub eps: f32,
}

/// Per-head output gate.
pub(crate) struct AttnGate {
    /// `[H, channels]` in the activation dtype.
    pub weight: UniquePtr<MlxArray>,
    pub scale: f32,
}

pub(crate) struct LimiteAttention {
    /// Projections with `qkv_scale` / `o_scale` already folded in.
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub uses_rope: bool,
    pub is_global: bool,
    /// Keys a sliding layer sees (`sliding_window + 1`); 0 on global layers.
    pub window: i32,
    pub value_embed: Option<ValueEmbed>,
    pub xsa: Option<Xsa>,
    pub gate: Option<AttnGate>,
}

/// Q, K, V in `[B, H, L, D]` plus the XSA unit values (`[B, L, H, D]` f32).
struct Projected {
    q: UniquePtr<MlxArray>,
    k: UniquePtr<MlxArray>,
    v: UniquePtr<MlxArray>,
    v_hat: Option<UniquePtr<MlxArray>>,
}

impl LimiteAttention {
    fn project(
        &self,
        attn_in: &MlxArray,
        ve_rows: Option<&MlxArray>,
        positions: &MlxArray,
        rope: &PartialRope,
    ) -> Projected {
        let shape = mlxcel_core::array_shape(attn_in);
        let (b, l) = (shape[0], shape[1]);
        let q = self.q_proj.forward(attn_in);
        let k = self.k_proj.forward(attn_in);
        let v = self.v_proj.forward(attn_in);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let mut v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        if let (Some(ve), Some(rows)) = (&self.value_embed, ve_rows) {
            // The cache and attention both consume the adjusted values.
            v = self.add_value_embeddings(attn_in, v, rows, ve);
        }

        let mut q = rms_norm(&q);
        let mut k = rms_norm(&k);
        if self.uses_rope {
            q = rope.apply(&q, positions);
            k = rope.apply(&k, positions);
        }

        let v_hat = self.xsa.as_ref().map(|xsa| {
            let vk = if self.num_kv_heads < self.num_heads {
                mlxcel_core::repeat(&v, self.num_heads / self.num_kv_heads, 2)
            } else {
                mlxcel_core::copy(&v)
            };
            let vf = mlxcel_core::astype(&vk, dtype::FLOAT32);
            let norm = mlxcel_core::linalg_norm(&vf, -1, true);
            let norm = mlxcel_core::maximum(&norm, &scalar(xsa.eps, dtype::FLOAT32));
            mlxcel_core::divide(&vf, &norm)
        });

        Projected {
            q: mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]),
            k: mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]),
            v: mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]),
            v_hat,
        }
    }

    fn add_value_embeddings(
        &self,
        attn_in: &MlxArray,
        v: UniquePtr<MlxArray>,
        rows: &MlxArray,
        ve: &ValueEmbed,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(attn_in);
        let (b, l) = (shape[0], shape[1]);
        let rows = mlxcel_core::astype(rows, mlxcel_core::array_dtype(&v));
        let mut emb = mlxcel_core::reshape(&rows, &[b, l, ve.stored_heads, ve.ve_dim]);
        if ve.ve_dim < self.head_dim {
            let pad = mlxcel_core::zeros(
                &[b, l, ve.stored_heads, self.head_dim - ve.ve_dim],
                mlxcel_core::array_dtype(&emb),
            );
            emb = mlxcel_core::concatenate(&emb, &pad, 3);
        }
        if ve.stored_heads > self.num_kv_heads {
            emb = slice_axis(&emb, 2, 0, self.num_kv_heads);
        }
        let channels = mlxcel_core::array_shape(&ve.gate)[1];
        let gate_in = slice_axis(attn_in, 2, 0, channels);
        let gate = mlxcel_core::sigmoid(&matmul_t(&gate_in, &ve.gate));
        let gate = mlxcel_core::multiply(
            &scalar(ve.gate_scale, mlxcel_core::array_dtype(&gate)),
            &gate,
        );
        let gated = mlxcel_core::multiply(&mlxcel_core::expand_dims(&gate, -1), &emb);
        mlxcel_core::add(&v, &gated)
    }

    /// XSA, the per-head gate and `o_proj` on the SDPA output `[B, H, L, D]`.
    fn finish(
        &self,
        y: UniquePtr<MlxArray>,
        attn_in: &MlxArray,
        v_hat: Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(attn_in);
        let (b, l) = (shape[0], shape[1]);
        let mut y = mlxcel_core::transpose_axes(&y, &[0, 2, 1, 3]);
        if let (Some(xsa), Some(vn)) = (&self.xsa, v_hat) {
            let dt = mlxcel_core::array_dtype(&y);
            let yf = mlxcel_core::astype(&y, dtype::FLOAT32);
            let proj = mlxcel_core::sum_axis(&mlxcel_core::multiply(&yf, &vn), -1, true);
            let removed = mlxcel_core::multiply(&mlxcel_core::multiply(&xsa.alpha, &proj), &vn);
            y = mlxcel_core::subtract(&y, &mlxcel_core::astype(&removed, dt));
        }
        if let Some(gate) = &self.gate {
            let channels = mlxcel_core::array_shape(&gate.weight)[1];
            let gate_in = slice_axis(attn_in, 2, 0, channels);
            let g = mlxcel_core::sigmoid(&matmul_t(&gate_in, &gate.weight));
            let g = mlxcel_core::multiply(&scalar(gate.scale, mlxcel_core::array_dtype(&g)), &g);
            let g = mlxcel_core::astype(
                &mlxcel_core::expand_dims(&g, -1),
                mlxcel_core::array_dtype(&y),
            );
            y = mlxcel_core::multiply(&y, &g);
        }
        let y = mlxcel_core::reshape(&y, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&y)
    }

    pub(crate) fn forward(
        &self,
        attn_in: &MlxArray,
        ve_rows: Option<&MlxArray>,
        positions: &MlxArray,
        rope: &PartialRope,
        cache: &mut Cache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let l = mlxcel_core::array_shape(attn_in)[1];
        let p = self.project(attn_in, ve_rows, positions, rope);
        let (k, v) = cache.update_and_fetch(p.k, p.v);
        // A mask-less multi-token forward is a single-row prefill that left
        // causality to MLX's causal mode or native window route (#3072).
        let y = if l > 1 && mask.is_none() {
            mlxcel_core::maskless_prefill_attention(&p.q, &k, &v, self.scale, self.window)
        } else if let Some(mask) = mask.filter(|_| l > 1) {
            // SAFETY: `mask` is a live reference for the duration of the call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &p.q,
                    &k,
                    &v,
                    self.scale,
                    mask as *const MlxArray,
                    0.0,
                    self.window,
                )
            }
        } else {
            mlxcel_core::causal_attention(&p.q, &k, &v, self.scale, 0.0, self.window)
        };
        self.finish(y, attn_in, p.v_hat)
    }

    pub(crate) fn forward_batched_decode(
        &self,
        attn_in: &MlxArray,
        ve_rows: Option<&MlxArray>,
        positions: &MlxArray,
        rope: &PartialRope,
        caches: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let p = self.project(attn_in, ve_rows, positions, rope);
        let y = attend_batched_rows(&p.q, &p.k, &p.v, caches, self.scale, None);
        self.finish(y, attn_in, p.v_hat)
    }
}

/// SwiGLU MLP: `down(silu(gate(h)) * up(h))`.
pub(crate) struct LimiteMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl LimiteMlp {
    pub(crate) fn forward(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let up = self.up_proj.forward(h);
        let act = mlxcel_core::multiply(&mlxcel_core::silu(&self.gate_proj.forward(h)), &up);
        self.down_proj.forward(&act)
    }
}

/// The four learned residual lambdas of a decoder layer, 0-dim f32.
pub(crate) struct Lambdas {
    pub resid_attn: UniquePtr<MlxArray>,
    pub post_attn: UniquePtr<MlxArray>,
    pub resid_mlp: UniquePtr<MlxArray>,
    pub post_mlp: UniquePtr<MlxArray>,
}

pub(crate) struct LimiteLayer {
    pub attn: LimiteAttention,
    pub mlp: LimiteMlp,
    pub lambdas: Lambdas,
}

impl LimiteLayer {
    /// `x = ra * base + pa * attn_out; out = rm * x + pm * mlp(rms_norm(x))`.
    pub(crate) fn residual(
        &self,
        residual_base: &MlxArray,
        attn_out: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let lam = &self.lambdas;
        let x = mlxcel_core::add(
            &scale_by(&lam.resid_attn, residual_base),
            &scale_by(&lam.post_attn, attn_out),
        );
        let mlp_out = self.mlp.forward(&rms_norm(&x));
        mlxcel_core::add(
            &scale_by(&lam.resid_mlp, &x),
            &scale_by(&lam.post_mlp, &mlp_out),
        )
    }
}
