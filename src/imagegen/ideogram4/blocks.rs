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

//! Ideogram 4 DiT building blocks: linear, norms, interleaved M-RoPE,
//! attention, MLP and the AdaLN transformer block.
//!
//! Ports of the classes in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/transformer.py>
//! that reproduce upstream's arithmetic op for op in bf16: the norms take
//! their mean in f32 and round it back to the activation dtype before the
//! `rsqrt` (so they are not the fused `RMSNorm` kernel), the biased linears
//! are one `addmm`, and the rotary tables are rounded to bf16 before the
//! rotation. The shared Z-Image blocks rotate adjacent pairs over contiguous
//! sections; this M-RoPE interleaves the three position axes over the
//! frequencies and rotates halves, so it is not shared.

use anyhow::{Context, Result};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::Ideogram4TransformerConfig;

fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// A floating tensor of exactly `shape`. A mismatch would otherwise surface
/// as an MLX C++ exception when the graph is built, which aborts the process.
pub(super) fn tensor(weights: &WeightMap, key: &str, shape: &[i32]) -> Result<UniquePtr<MlxArray>> {
    let w = weights
        .get(key)
        .with_context(|| format!("Ideogram 4 transformer weight not found: {key}"))?;
    let found = mlxcel_core::array_shape(w);
    if found != shape {
        anyhow::bail!(
            "Ideogram 4 transformer weight {key} has shape {found:?}, expected {shape:?}"
        );
    }
    let dt = mlxcel_core::array_dtype(w);
    if !matches!(dt, dtype::FLOAT16 | dtype::BFLOAT16 | dtype::FLOAT32) {
        anyhow::bail!(
            "Ideogram 4 transformer weight {key} has MLX dtype {dt}, expected a floating dtype (an FP8 weight needs its weight_scale)"
        );
    }
    Ok(mlxcel_core::copy(w))
}

/// `nn.Linear`: `addmm(bias, x, W.T)` with a bias (MLX's single rounding),
/// else `x @ W.T`.
pub struct Dense {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl Dense {
    /// Load `<prefix>.weight` as `[out, inp]` and, when present,
    /// `<prefix>.bias` as `[out]`.
    pub fn load(weights: &WeightMap, prefix: &str, out: i32, inp: i32) -> Result<Self> {
        let bias_key = format!("{prefix}.bias");
        Ok(Self {
            weight: tensor(weights, &format!("{prefix}.weight"), &[out, inp])?,
            bias: weights
                .contains_key(&bias_key)
                .then(|| tensor(weights, &bias_key, &[out]))
                .transpose()?,
        })
    }

    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = mlxcel_core::transpose(&self.weight);
        match &self.bias {
            Some(bias) => mlxcel_core::addmm(bias, x, &wt, 1.0, 1.0),
            None => mlxcel_core::matmul(x, &wt),
        }
    }

    /// Dtype of the weight (upstream's `param_dtype`).
    #[must_use]
    pub fn dtype(&self) -> i32 {
        mlxcel_core::array_dtype(&self.weight)
    }
}

/// `Ideogram4RMSNorm`: `x * rsqrt(mean(x.f32^2).astype(x.dtype) + eps) * w`.
pub struct Norm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl Norm {
    /// Load `key` as a `[len]` weight.
    pub fn load(weights: &WeightMap, key: &str, eps: f32, len: i32) -> Result<Self> {
        Ok(Self {
            weight: tensor(weights, key, &[len])?,
            eps,
        })
    }

    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dt = mlxcel_core::array_dtype(x);
        let x32 = mlxcel_core::astype(x, dtype::FLOAT32);
        let variance = mlxcel_core::mean_axis(&mlxcel_core::square(&x32), -1, true);
        let variance = mlxcel_core::astype(&variance, dt);
        let inv = mlxcel_core::rsqrt(&mlxcel_core::add(&variance, &scalar_like(self.eps, x)));
        mlxcel_core::multiply(&mlxcel_core::multiply(x, &inv), &self.weight)
    }
}

/// Upstream `_layer_norm_no_affine` (eps 1e-6), mean and variance in f32,
/// each rounded back to the activation dtype.
#[must_use]
pub fn layer_norm_no_affine(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let mean = mlxcel_core::mean_axis(&mlxcel_core::astype(x, dtype::FLOAT32), -1, true);
    let centered = mlxcel_core::subtract(x, &mlxcel_core::astype(&mean, dt));
    let var = mlxcel_core::mean_axis(
        &mlxcel_core::square(&mlxcel_core::astype(&centered, dtype::FLOAT32)),
        -1,
        true,
    );
    let var = mlxcel_core::astype(&var, dt);
    mlxcel_core::multiply(
        &centered,
        &mlxcel_core::rsqrt(&mlxcel_core::add(&var, &scalar_like(eps, x))),
    )
}

/// `Ideogram4MRoPE`: frequency `i` of the half head rotates with position
/// axis `selector[i]`. Every index starts on axis 0 (time); indices
/// `1, 4, 7, ...` below `3 * section[1]` take axis 1 (height) and
/// `2, 5, 8, ...` below `3 * section[2]` take axis 2 (width).
pub struct MRope {
    inv_freq: UniquePtr<MlxArray>,
    selector: UniquePtr<MlxArray>,
}

/// The axis each of the `head_dim / 2` frequencies reads.
#[must_use]
pub fn axis_selector(head_dim: i32, section: [i32; 3]) -> Vec<i32> {
    let half = (head_dim / 2) as usize;
    let mut selector = vec![0; half];
    for (axis, offset) in [(1usize, 1usize), (2, 2)] {
        let end = (i64::from(section[axis]) * 3).max(0) as usize;
        for index in (offset..end).step_by(3) {
            if index < half {
                selector[index] = axis as i32;
            }
        }
    }
    selector
}

impl MRope {
    #[must_use]
    pub fn new(head_dim: i32, theta: f32, section: [i32; 3]) -> Self {
        // `1 / base ** (arange(0, hd, 2, f32) / hd)` as upstream computes it
        // in f32, with the power taken on the host in f64 and rounded once.
        // The rope arguments reach past 65536, so one ulp of `inv_freq`
        // moves them by about 1e-2. The M1 Ultra GPU's `pow` put the table
        // up to 1.4e-3 (cos) from upstream's CUDA one; computed here it is
        // within the 1e-4 the parity test allows.
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| {
                let exponent = i as f32 / head_dim as f32;
                let power = f64::from(theta).powf(f64::from(exponent)) as f32;
                1.0 / power
            })
            .collect();
        let inv_freq = mlxcel_core::from_slice_f32(&inv_freq, &[inv_freq.len() as i32]);
        let selector = axis_selector(head_dim, section);
        let n = selector.len() as i32;
        Self {
            inv_freq: mlxcel_core::reshape(&inv_freq, &[1, 1, n]),
            selector: mlxcel_core::from_slice_i32(&selector, &[n]),
        }
    }

    /// `(cos, sin)` for `[B, L, 3]` integer positions, each `[B, L, head_dim]`
    /// in f32 (`emb = concat([freqs, freqs])`).
    #[must_use]
    pub fn tables(&self, position_ids: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let selected = mlxcel_core::take(position_ids, &self.selector, 2);
        let freqs = mlxcel_core::multiply(
            &mlxcel_core::astype(&selected, dtype::FLOAT32),
            &self.inv_freq,
        );
        let emb = mlxcel_core::concatenate(&freqs, &freqs, -1);
        (mlxcel_core::cos(&emb), mlxcel_core::sin(&emb))
    }
}

/// Upstream `rotate_half`: `concat([-x2, x1])` over the last axis.
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape.len() - 1;
    let half = shape[last] / 2;
    let mut start = vec![0; shape.len()];
    let mut stop = shape.clone();
    stop[last] = half;
    let x1 = mlxcel_core::slice(x, &start, &stop);
    start[last] = half;
    stop[last] = shape[last];
    let x2 = mlxcel_core::slice(x, &start, &stop);
    mlxcel_core::concatenate(&mlxcel_core::negative(&x2), &x1, -1)
}

/// `x * cos + rotate_half(x) * sin`; `x` is `[B, H, L, D]`, the tables
/// `[B, 1, L, D]` in `x`'s dtype.
fn rotate(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(
        &mlxcel_core::multiply(x, cos),
        &mlxcel_core::multiply(&rotate_half(x), sin),
    )
}

/// `Ideogram4Attention`: fused QKV, per-head RMS norms (eps 1e-5), M-RoPE,
/// and SDPA in f32.
pub struct Attention {
    qkv: Dense,
    norm_q: Norm,
    norm_k: Norm,
    o: Dense,
    num_heads: i32,
    head_dim: i32,
}

impl Attention {
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        config: &Ideogram4TransformerConfig,
    ) -> Result<Self> {
        let (e, d) = (config.emb_dim, config.head_dim());
        Ok(Self {
            qkv: Dense::load(weights, &format!("{prefix}.qkv"), 3 * e, e)?,
            norm_q: Norm::load(weights, &format!("{prefix}.norm_q.weight"), 1e-5, d)?,
            norm_k: Norm::load(weights, &format!("{prefix}.norm_k.weight"), 1e-5, d)?,
            o: Dense::load(weights, &format!("{prefix}.o"), e, e)?,
            num_heads: config.num_heads,
            head_dim: d,
        })
    }

    /// `cos`/`sin` are `[B, 1, L, D]` in the activation dtype. `mask` is the
    /// additive segment mask, `None` when every token shares one segment.
    pub fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l, width) = (shape[0], shape[1], shape[2]);
        let (h, d) = (self.num_heads, self.head_dim);
        let qkv = mlxcel_core::reshape(&self.qkv.forward(x), &[b, l, 3, h, d]);
        let part = |i: i32| {
            let s = mlxcel_core::slice(&qkv, &[0, 0, i, 0, 0], &[b, l, i + 1, h, d]);
            mlxcel_core::reshape(&s, &[b, l, h, d])
        };
        let heads = |t: &MlxArray| mlxcel_core::transpose_axes(t, &[0, 2, 1, 3]);
        let q = heads(&self.norm_q.forward(&part(0)));
        let k = heads(&self.norm_k.forward(&part(1)));
        let v = heads(&part(2));
        let q = rotate(&q, cos, sin);
        let k = rotate(&k, cos, sin);
        let f32 = |t: &MlxArray| mlxcel_core::astype(t, dtype::FLOAT32);
        // Through the shared dispatch: f32 q/k/v never reach CUDA's fused SDPA,
        // so MLX's fallback would materialize the whole `[B, heads, L, L]` f32
        // score matrix; `attention` query-chunks it to
        // `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` with row-identical math.
        let out = mlxcel_core::layers::attention(
            &f32(&q),
            &f32(&k),
            &f32(&v),
            (d as f32).powf(-0.5),
            mask,
            0.0,
            0,
        );
        let out = mlxcel_core::astype(&out, mlxcel_core::array_dtype(x));
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[b, l, width],
        );
        self.o.forward(&out)
    }
}

/// `Ideogram4MLP`: `w2(silu(w1(x)) * w3(x))`.
pub struct Mlp {
    w1: Dense,
    w2: Dense,
    w3: Dense,
}

impl Mlp {
    pub fn load(weights: &WeightMap, prefix: &str, emb: i32, inter: i32) -> Result<Self> {
        Ok(Self {
            w1: Dense::load(weights, &format!("{prefix}.w1"), inter, emb)?,
            w2: Dense::load(weights, &format!("{prefix}.w2"), emb, inter)?,
            w3: Dense::load(weights, &format!("{prefix}.w3"), inter, emb)?,
        })
    }

    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = mlxcel_core::silu(&self.w1.forward(x));
        self.w2
            .forward(&mlxcel_core::multiply(&gate, &self.w3.forward(x)))
    }
}

/// Split the last axis into `n` equal parts.
pub fn split_last(x: &MlxArray, n: i32) -> Vec<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape.len() - 1;
    let width = shape[last] / n;
    (0..n)
        .map(|i| {
            let mut start = vec![0; shape.len()];
            let mut stop = shape.clone();
            start[last] = i * width;
            stop[last] = (i + 1) * width;
            mlxcel_core::slice(x, &start, &stop)
        })
        .collect()
}

/// `Ideogram4TransformerBlock`: tanh-gated attention and MLP branches with
/// pre- and post-norms, modulated by the shared AdaLN input.
pub struct Block {
    attention: Attention,
    feed_forward: Mlp,
    attention_norm1: Norm,
    ffn_norm1: Norm,
    attention_norm2: Norm,
    ffn_norm2: Norm,
    adaln_modulation: Dense,
}

impl Block {
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        config: &Ideogram4TransformerConfig,
    ) -> Result<Self> {
        let e = config.emb_dim;
        let norm = |name: &str| {
            Norm::load(
                weights,
                &format!("{prefix}.{name}.weight"),
                config.norm_eps,
                e,
            )
        };
        Ok(Self {
            attention: Attention::load(weights, &format!("{prefix}.attention"), config)?,
            feed_forward: Mlp::load(
                weights,
                &format!("{prefix}.feed_forward"),
                e,
                config.intermediate_size,
            )?,
            attention_norm1: norm("attention_norm1")?,
            ffn_norm1: norm("ffn_norm1")?,
            attention_norm2: norm("attention_norm2")?,
            ffn_norm2: norm("ffn_norm2")?,
            adaln_modulation: Dense::load(
                weights,
                &format!("{prefix}.adaln_modulation"),
                4 * e,
                config.adaln_dim,
            )?,
        })
    }

    /// `adaln_input` is `[B, 1, adaln_dim]` (already `silu`'d).
    pub fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
        adaln_input: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let m = split_last(&self.adaln_modulation.forward(adaln_input), 4);
        let one = scalar_like(1.0, &m[0]);
        let scale_msa = mlxcel_core::add(&one, &m[0]);
        let gate_msa = mlxcel_core::tanh(&m[1]);
        let scale_mlp = mlxcel_core::add(&one, &m[2]);
        let gate_mlp = mlxcel_core::tanh(&m[3]);
        let attn = self.attention.forward(
            &mlxcel_core::multiply(&self.attention_norm1.forward(x), &scale_msa),
            cos,
            sin,
            mask,
        );
        let x = mlxcel_core::add(
            x,
            &mlxcel_core::multiply(&gate_msa, &self.attention_norm2.forward(&attn)),
        );
        let mlp = self.feed_forward.forward(&mlxcel_core::multiply(
            &self.ffn_norm1.forward(&x),
            &scale_mlp,
        ));
        mlxcel_core::add(
            &x,
            &mlxcel_core::multiply(&gate_mlp, &self.ffn_norm2.forward(&mlp)),
        )
    }
}

#[cfg(test)]
#[path = "blocks_tests.rs"]
mod tests;
