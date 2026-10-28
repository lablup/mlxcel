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

//! Layers of the Qwen-Image-2.1 single-stream DiT, a port of the module
//! classes in `transformer.py` (mlx-vlm `73d366ec`).
//!
//! Everything runs in the checkpoint's bf16 except upstream's explicit f32
//! islands, which are written out here because the CUDA build of mlxcel's
//! MLX promotes bf16 with f32 to bf16: the timestep sinusoid, the
//! zero-centered text RMSNorm, the RoPE angles and the rotation itself
//! (rounded back to the activation dtype).
//!
//! `nn.LayerNorm(affine=False)` uses Mage-Flow's composed
//! [`LayerNorm::plain`] instead of the fused kernel upstream calls: the fused
//! MLX CUDA kernel has a shared-memory race between its two block reductions
//! (see that type), and the composed form matches its correct rows to one
//! bf16 ulp while staying deterministic, which the KV-cache on/off
//! equivalence depends on.

use anyhow::Result;
use mlxcel_core::layers::RMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::kv_cache::{KvLayer, KvMode};
use crate::imagegen::mage_flow::layers::{
    Dense, LayerNorm, LinearQuant, rms_norm, scalar_like, silu, split_last, tensor,
};

/// Width of the sinusoidal timestep embedding.
const TIME_FREQ_DIM: i32 = 256;
/// RoPE base.
pub const ROPE_THETA: f32 = 10000.0;

fn f32_scalar(v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], v, dtype::FLOAT32)
}

/// Upstream `_sinusoidal_timesteps(t)` cast to `dt`: f32 frequencies
/// `exp(-ln(10000) * arange(128) / 128)`, arguments `(1000 * t) * f`,
/// `[cos, sin]`.
#[must_use]
pub fn sinusoidal_timesteps(t: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
    let half = TIME_FREQ_DIM / 2;
    let exponent = mlxcel_core::divide(
        &mlxcel_core::multiply(
            &f32_scalar(-(10000.0_f64.ln()) as f32),
            &mlxcel_core::arange_f32(0.0, half as f32, 1.0),
        ),
        &f32_scalar(half as f32),
    );
    let freqs = mlxcel_core::exp(&exponent);
    let t = mlxcel_core::astype(t, dtype::FLOAT32);
    let args = mlxcel_core::multiply(
        &mlxcel_core::reshape(&mlxcel_core::multiply(&f32_scalar(1000.0), &t), &[-1, 1]),
        &mlxcel_core::reshape(&freqs, &[1, -1]),
    );
    let emb = mlxcel_core::concatenate(&mlxcel_core::cos(&args), &mlxcel_core::sin(&args), -1);
    mlxcel_core::astype(&emb, dt)
}

/// `QwenImageTimestepEmbed`.
pub struct TimestepEmbed {
    linear_1: Dense,
    linear_2: Dense,
}

impl TimestepEmbed {
    pub fn load(w: &WeightMap, p: &str, q: Option<LinearQuant>) -> Result<Self> {
        Ok(Self {
            linear_1: Dense::load(w, &format!("{p}.linear_1"), q)?,
            linear_2: Dense::load(w, &format!("{p}.linear_2"), q)?,
        })
    }

    pub fn forward(&self, t: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
        let emb = sinusoidal_timesteps(t, dt);
        self.linear_2.forward(&silu(&self.linear_1.forward(&emb)))
    }
}

/// `QwenImageZeroCenterRMSNorm`: `x * rsqrt(mean(x^2) + eps) * (w + 1)` in
/// f32, cast back to `x`'s dtype.
pub struct ZeroCenterRmsNorm {
    weight_plus_one: UniquePtr<MlxArray>,
    eps: f32,
}

impl ZeroCenterRmsNorm {
    pub fn load(w: &WeightMap, key: &str, eps: f32) -> Result<Self> {
        let weight = tensor(w, key)?;
        let weight = mlxcel_core::astype(&weight, dtype::FLOAT32);
        Ok(Self {
            weight_plus_one: mlxcel_core::add(&weight, &f32_scalar(1.0)),
            eps,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let xf = mlxcel_core::astype(x, dtype::FLOAT32);
        let ms = mlxcel_core::mean_axis(&mlxcel_core::square(&xf), -1, true);
        let rrms = mlxcel_core::rsqrt(&mlxcel_core::add(&ms, &f32_scalar(self.eps)));
        let y = mlxcel_core::multiply(&mlxcel_core::multiply(&xf, &rrms), &self.weight_plus_one);
        mlxcel_core::astype(&y, mlxcel_core::array_dtype(x))
    }
}

/// `QwenImageTextProjection`: `out(gelu_approx(in(text_norm(x))))`.
pub struct TextProjection {
    text_norm: ZeroCenterRmsNorm,
    in_layer: Dense,
    out_layer: Dense,
}

impl TextProjection {
    pub fn load(w: &WeightMap, p: &str, eps: f32, q: Option<LinearQuant>) -> Result<Self> {
        Ok(Self {
            text_norm: ZeroCenterRmsNorm::load(w, &format!("{p}.text_norm.weight"), eps)?,
            in_layer: Dense::load(w, &format!("{p}.in_layer"), q)?,
            out_layer: Dense::load(w, &format!("{p}.out_layer"), q)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.text_norm.forward(x);
        self.out_layer
            .forward(&mlxcel_core::gelu_tanh(&self.in_layer.forward(&x)))
    }
}

/// `QwenImageSwiGLU`: `out(silu(gate(x)) * proj(x))`.
pub struct SwiGlu {
    proj: Dense,
    out: Dense,
    gate: Dense,
}

impl SwiGlu {
    pub fn load(w: &WeightMap, p: &str, q: Option<LinearQuant>) -> Result<Self> {
        Ok(Self {
            proj: Dense::load(w, &format!("{p}.proj"), q)?,
            out: Dense::load(w, &format!("{p}.out"), q)?,
            gate: Dense::load(w, &format!("{p}.gate_layer"), q)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gated = mlxcel_core::multiply(&silu(&self.gate.forward(x)), &self.proj.forward(x));
        self.out.forward(&gated)
    }
}

/// Upstream `_select_modulation_rows`: `params` is `[2, D]` (the real
/// timestep row, then the zero-timestep row) with `causal_condition`, or
/// `[B, D]` without; `target` (`[S]` bool) picks the real row per token.
#[must_use]
pub fn select_modulation_rows(params: &MlxArray, target: Option<&MlxArray>) -> UniquePtr<MlxArray> {
    let Some(mask) = target else {
        return mlxcel_core::expand_dims(params, 1);
    };
    let s = mlxcel_core::array_shape(params);
    let real = mlxcel_core::expand_dims(&mlxcel_core::slice(params, &[0, 0], &[s[0] - 1, s[1]]), 1);
    let zero = mlxcel_core::expand_dims(
        &mlxcel_core::slice(params, &[s[0] - 1, 0], &[s[0], s[1]]),
        0,
    );
    let mask = mlxcel_core::reshape(mask, &[1, -1, 1]);
    mlxcel_core::where_cond(&mask, &real, &zero)
}

/// `nn.LayerNorm(affine=False)`.
#[must_use]
pub fn layer_norm(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    LayerNorm::plain(eps).forward(x)
}

/// `QwenImageAdaLayerNormContinuous`: `norm(x) * (1 + scale)`, no shift.
pub struct AdaLayerNormContinuous {
    linear: Dense,
    eps: f32,
}

impl AdaLayerNormContinuous {
    pub fn load(w: &WeightMap, p: &str, eps: f32, q: Option<LinearQuant>) -> Result<Self> {
        Ok(Self {
            linear: Dense::load(w, &format!("{p}.linear"), q)?,
            eps,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        temb: &MlxArray,
        target: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let cond = mlxcel_core::astype(&silu(temb), mlxcel_core::array_dtype(x));
        let scale = select_modulation_rows(&self.linear.forward(&cond), target);
        let one = scalar_like(1.0, &scale);
        mlxcel_core::multiply(&layer_norm(x, self.eps), &mlxcel_core::add(&one, &scale))
    }
}

/// Upstream `_apply_rope` on `[B, H, S, D]`: adjacent pairs rotated by the
/// `[S, D / 2]` f32 tables, computed in f32 and cast back to `x`'s dtype.
#[must_use]
pub fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let (b, h, l, d) = (s[0], s[1], s[2], s[3]);
    let pairs = mlxcel_core::astype(
        &mlxcel_core::reshape(x, &[b, h, l, d / 2, 2]),
        dtype::FLOAT32,
    );
    let x0 = mlxcel_core::reshape(
        &mlxcel_core::slice(&pairs, &[0, 0, 0, 0, 0], &[b, h, l, d / 2, 1]),
        &[b, h, l, d / 2],
    );
    let x1 = mlxcel_core::reshape(
        &mlxcel_core::slice(&pairs, &[0, 0, 0, 0, 1], &[b, h, l, d / 2, 2]),
        &[b, h, l, d / 2],
    );
    let out0 = mlxcel_core::subtract(
        &mlxcel_core::multiply(&x0, cos),
        &mlxcel_core::multiply(&x1, sin),
    );
    let out1 = mlxcel_core::add(
        &mlxcel_core::multiply(&x0, sin),
        &mlxcel_core::multiply(&x1, cos),
    );
    let stacked = mlxcel_core::concatenate(
        &mlxcel_core::expand_dims(&out0, -1),
        &mlxcel_core::expand_dims(&out1, -1),
        -1,
    );
    mlxcel_core::astype(
        &mlxcel_core::reshape(&stacked, &[b, h, l, d]),
        mlxcel_core::array_dtype(x),
    )
}

/// `QwenImageRope.for_layout`: per-token `(frame, height, width)` positions
/// for interleaved text and image blocks `(start, height, width)`.
#[must_use]
pub fn rope_positions(seq_len: i32, blocks: &[(i32, i32, i32)]) -> [Vec<i32>; 3] {
    let (mut frame, mut height, mut width) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cursor, mut position) = (0, 0);
    let text =
        |frame: &mut Vec<i32>, height: &mut Vec<i32>, width: &mut Vec<i32>, from: i32, n: i32| {
            for p in from..from + n {
                frame.push(p);
                height.push(p);
                width.push(p);
            }
        };
    for &(start, h, w) in blocks {
        text(
            &mut frame,
            &mut height,
            &mut width,
            position,
            start - cursor,
        );
        position += start - cursor;
        for y in -(h - h / 2)..h / 2 {
            for x in -(w - w / 2)..w / 2 {
                frame.push(position);
                height.push(y);
                width.push(x);
            }
        }
        cursor = start + h * w;
        position += h.max(w);
    }
    text(
        &mut frame,
        &mut height,
        &mut width,
        position,
        seq_len - cursor,
    );
    [frame, height, width]
}

/// `(cos, sin)` `[S, head_dim / 2]` f32 for `positions`: per axis
/// `index * 1 / theta ** (arange(0, dim, 2) / dim)`, concatenated.
pub fn rope_tables(
    positions: &[Vec<i32>; 3],
    axes_dim: [i32; 3],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut parts = Vec::with_capacity(3);
    for (axis, &dim) in positions.iter().zip(axes_dim.iter()) {
        let index: Vec<f32> = axis.iter().map(|&p| p as f32).collect();
        let index = mlxcel_core::from_slice_f32(&index, &[index.len() as i32, 1]);
        let exponents = mlxcel_core::divide(
            &mlxcel_core::arange_f32(0.0, dim as f32, 2.0),
            &f32_scalar(dim as f32),
        );
        let inv = mlxcel_core::divide(
            &f32_scalar(1.0),
            &mlxcel_core::power(&f32_scalar(ROPE_THETA), &exponents),
        );
        parts.push(mlxcel_core::multiply(
            &index,
            &mlxcel_core::reshape(&inv, &[1, -1]),
        ));
    }
    let angles = mlxcel_core::concatenate_many(&parts.iter().map(|a| &**a).collect::<Vec<_>>(), -1);
    (mlxcel_core::cos(&angles), mlxcel_core::sin(&angles))
}

/// `QwenImageAttention`, with the reference KV cache.
pub struct Attention {
    to_q: Dense,
    to_k: Dense,
    to_v: Dense,
    to_out: Dense,
    norm_q: RMSNorm,
    norm_k: RMSNorm,
    heads: i32,
    head_dim: i32,
}

/// What one attention call does with its layer's cache.
pub struct CacheUse<'a> {
    pub layer: &'a mut KvLayer,
    pub mode: KvMode,
    pub prefix_len: i32,
}

impl Attention {
    pub fn load(
        w: &WeightMap,
        p: &str,
        heads: i32,
        head_dim: i32,
        eps: f32,
        q: Option<LinearQuant>,
    ) -> Result<Self> {
        let dense = |name: &str| Dense::load(w, &format!("{p}.{name}"), q);
        let norm = |name: &str| rms_norm(w, &format!("{p}.{name}.weight"), eps);
        Ok(Self {
            to_q: dense("to_q")?,
            to_k: dense("to_k")?,
            to_v: dense("to_v")?,
            to_out: dense("to_out.0")?,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            heads,
            head_dim,
        })
    }

    fn heads(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        mlxcel_core::reshape(x, &[s[0], s[1], self.heads, self.head_dim])
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
        cache: Option<CacheUse<'_>>,
    ) -> Result<UniquePtr<MlxArray>> {
        let s = mlxcel_core::array_shape(x);
        let (b, l) = (s[0], s[1]);
        let t = |a: &MlxArray| mlxcel_core::transpose_axes(a, &[0, 2, 1, 3]);
        let q = t(&self.norm_q.forward(&self.heads(&self.to_q.forward(x))));
        let k = t(&self.norm_k.forward(&self.heads(&self.to_k.forward(x))));
        let v = t(&self.heads(&self.to_v.forward(x)));
        let q = apply_rope(&q, cos, sin);
        let mut k = apply_rope(&k, cos, sin);
        let mut v = v;
        if let Some(cache) = cache {
            match cache.mode {
                KvMode::Extract => cache.layer.store(&k, &v, cache.prefix_len),
                KvMode::Cached => {
                    let (pk, pv) = cache.layer.get()?;
                    k = mlxcel_core::concatenate(pk, &k, 2);
                    v = mlxcel_core::concatenate(pv, &v, 2);
                }
            }
        }
        // Through the shared dispatch, which query-chunks the score matrix of
        // a long joint sequence (an edit at 1024x1024 is about 8k tokens).
        let out = mlxcel_core::layers::attention(
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
            mask,
            0.0,
            0,
        );
        let out = mlxcel_core::reshape(&t(&out), &[b, l, self.heads * self.head_dim]);
        Ok(self.to_out.forward(&out))
    }
}

/// `QwenImageTransformerBlock`.
pub struct Block {
    attn: Attention,
    mlp: SwiGlu,
    eps: f32,
}

/// `x * (1 + scale)` and `gate` of one `[scale, gate]` modulation half.
fn modulate(
    x: &MlxArray,
    params: &MlxArray,
    target: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let [scale, gate]: [UniquePtr<MlxArray>; 2] = split_last(params, 2)
        .try_into()
        .unwrap_or_else(|_| unreachable!("split into 2"));
    let scale = select_modulation_rows(&scale, target);
    let gate = select_modulation_rows(&gate, target);
    let one = scalar_like(1.0, &scale);
    (
        mlxcel_core::multiply(x, &mlxcel_core::add(&one, &scale)),
        gate,
    )
}

impl Block {
    pub fn load(
        w: &WeightMap,
        p: &str,
        heads: i32,
        head_dim: i32,
        eps: f32,
        q: Option<LinearQuant>,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(w, &format!("{p}.attn"), heads, head_dim, eps, q)?,
            mlp: SwiGlu::load(w, &format!("{p}.img_mlp"), q)?,
            eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &MlxArray,
        modulation: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
        target: Option<&MlxArray>,
        cache: Option<CacheUse<'_>>,
    ) -> Result<UniquePtr<MlxArray>> {
        let [mod1, mod2]: [UniquePtr<MlxArray>; 2] = split_last(modulation, 2)
            .try_into()
            .unwrap_or_else(|_| unreachable!("split into 2"));
        let (modulated, gate1) = modulate(&layer_norm(x, self.eps), &mod1, target);
        let attn = self.attn.forward(&modulated, cos, sin, mask, cache)?;
        let x = mlxcel_core::add(x, &mlxcel_core::multiply(&mlxcel_core::tanh(&gate1), &attn));
        let (modulated, gate2) = modulate(&layer_norm(&x, self.eps), &mod2, target);
        let mlp = self.mlp.forward(&modulated);
        Ok(mlxcel_core::add(
            &x,
            &mlxcel_core::multiply(&mlxcel_core::tanh(&gate2), &mlp),
        ))
    }
}
