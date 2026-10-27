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

//! FLUX.2 transformer embeddings and small layers.
//!
//! Ports of `timestep_guidance_embeddings.py`, `modulation.py`,
//! `pos_embed.py`, `ada_layer_norm_continuous.py` and `feed_forward.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/flux2/transformer>,
//! plus the linear-layer seam every block loads through.
//!
//! Every scalar constant is created in the dtype of the array it meets, so the
//! MLX promotion here matches upstream's weakly typed Python scalars.
//!
//! Used by: Flux2, Bonsai.

use anyhow::{Context, Result, anyhow};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::QuantParams;

/// A bias-free projection over the last axis.
pub trait Flux2Linear {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray>;
}

impl Flux2Linear for UnifiedLinear {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        UnifiedLinear::forward(self, x)
    }
}

/// Where the transformer's linear layers come from.
///
/// [`AffineLinears`] loads dense or MLX affine-quantized layers; Bonsai's
/// ternary transformer (#2428) supplies its own source and reuses every block.
pub trait LinearSource {
    fn load(&self, weights: &WeightMap, prefix: &str) -> Result<Box<dyn Flux2Linear>>;
}

/// Dense layers, or MLX affine-quantized ones where `<prefix>.scales` exists.
#[derive(Debug, Clone, Copy, Default)]
pub struct AffineLinears {
    /// Declared parameters; per-layer shapes are reconciled by `UnifiedLinear`.
    pub quant: Option<QuantParams>,
}

impl LinearSource for AffineLinears {
    fn load(&self, weights: &WeightMap, prefix: &str) -> Result<Box<dyn Flux2Linear>> {
        let q = self.quant.unwrap_or(QuantParams {
            bits: 4,
            group_size: 64,
        });
        if weights.contains_key(&format!("{prefix}.scales"))
            && !weights.contains_key(&format!("{prefix}.biases"))
        {
            return Err(anyhow!(
                "Quantized module {prefix} is missing affine bias tensors"
            ));
        }
        let layer =
            UnifiedLinear::from_weights_with_mode(weights, prefix, q.group_size, q.bits, "affine")
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("FLUX.2 transformer weight {prefix}"))?;
        Ok(Box::new(layer))
    }
}

/// A scalar in `like`'s dtype.
pub(crate) fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// `nn.LayerNorm(dim, eps=1e-6, affine=False)`.
pub(crate) fn layer_norm(x: &MlxArray) -> UniquePtr<MlxArray> {
    // SAFETY: null weight and bias are the documented "no affine" arguments.
    unsafe { mlxcel_core::fast_layer_norm(x, std::ptr::null(), std::ptr::null(), 1e-6) }
}

/// `(1 + scale) * x + shift`.
pub(crate) fn modulate(x: &MlxArray, shift: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let one = scalar_like(1.0, scale);
    mlxcel_core::add(
        &mlxcel_core::multiply(&mlxcel_core::add(&one, scale), x),
        shift,
    )
}

/// Split the last axis at the given widths.
pub(crate) fn split_last(x: &MlxArray, widths: &[i32]) -> Vec<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape.len() - 1;
    let mut start = 0;
    widths
        .iter()
        .map(|w| {
            let mut starts = vec![0; shape.len()];
            let mut stops = shape.clone();
            starts[last] = start;
            stops[last] = start + w;
            start += w;
            mlxcel_core::slice(x, &starts, &stops)
        })
        .collect()
}

/// `Flux2TimestepGuidanceEmbeddings`: sinusoidal embedding (in f32) then a
/// two-layer SiLU MLP, plus the same for guidance when the model embeds it.
pub struct TimestepGuidanceEmbeddings {
    in_channels: i32,
    linear_1: Box<dyn Flux2Linear>,
    linear_2: Box<dyn Flux2Linear>,
    guidance: Option<(Box<dyn Flux2Linear>, Box<dyn Flux2Linear>)>,
}

impl TimestepGuidanceEmbeddings {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        in_channels: i32,
        guidance_embeds: bool,
    ) -> Result<Self> {
        let guidance = if guidance_embeds {
            Some((
                source.load(weights, "time_guidance_embed.guidance_linear_1")?,
                source.load(weights, "time_guidance_embed.guidance_linear_2")?,
            ))
        } else {
            None
        };
        Ok(Self {
            in_channels,
            linear_1: source.load(weights, "time_guidance_embed.linear_1")?,
            linear_2: source.load(weights, "time_guidance_embed.linear_2")?,
            guidance,
        })
    }

    /// `timestep` and `guidance` are `[B]` in any dtype; the MLP runs on f32
    /// embeddings, as upstream.
    pub fn forward(&self, timestep: &MlxArray, guidance: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let emb = timestep_embedding(timestep, self.in_channels);
        let t = self
            .linear_2
            .forward(&mlxcel_core::silu(&self.linear_1.forward(&emb)));
        match (guidance, &self.guidance) {
            (Some(g), Some((l1, l2))) => {
                let emb = timestep_embedding(g, self.in_channels);
                mlxcel_core::add(&t, &l2.forward(&mlxcel_core::silu(&l1.forward(&emb))))
            }
            _ => t,
        }
    }
}

/// `_timestep_embedding(t.astype(f32), dim)` with `flip_sin_to_cos=True`:
/// `[cos(t * f), sin(t * f)]` for `f = exp(-ln(10000) * i / half)`.
pub fn timestep_embedding(timesteps: &MlxArray, dim: i32) -> UniquePtr<MlxArray> {
    let half = dim / 2;
    let t = mlxcel_core::astype(timesteps, dtype::FLOAT32);
    let idx = mlxcel_core::astype(&mlxcel_core::arange_i32(0, half, 1), dtype::FLOAT32);
    let f32s = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
    let exponent = mlxcel_core::divide(
        &mlxcel_core::multiply(&f32s(-(10000.0_f64.ln() as f32)), &idx),
        &f32s(half as f32),
    );
    let freqs = mlxcel_core::exp(&exponent);
    let n = mlxcel_core::array_shape(&t)[0];
    let args = mlxcel_core::multiply(
        &mlxcel_core::reshape(&t, &[n, 1]),
        &mlxcel_core::reshape(&freqs, &[1, half]),
    );
    let emb = mlxcel_core::concatenate(&mlxcel_core::cos(&args), &mlxcel_core::sin(&args), -1);
    if dim % 2 == 1 {
        let pad = mlxcel_core::zeros(&[n, 1], dtype::FLOAT32);
        return mlxcel_core::concatenate(&emb, &pad, -1);
    }
    emb
}

/// `Flux2Modulation`: `linear(silu(temb))` split into `3 * sets` chunks of
/// `[B, 1, D]`, grouped as `sets` triples `(shift, scale, gate)`.
pub struct Modulation {
    linear: Box<dyn Flux2Linear>,
    sets: usize,
}

/// `(shift, scale, gate)`.
pub type ModParams = [UniquePtr<MlxArray>; 3];

impl Modulation {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        sets: usize,
    ) -> Result<Self> {
        Ok(Self {
            linear: source.load(weights, &format!("{prefix}.linear"))?,
            sets,
        })
    }

    pub fn forward(&self, temb: &MlxArray) -> Vec<ModParams> {
        let mut m = self.linear.forward(&mlxcel_core::silu(temb));
        if mlxcel_core::array_shape(&m).len() == 2 {
            m = mlxcel_core::expand_dims(&m, 1);
        }
        let shape = mlxcel_core::array_shape(&m);
        let chunk = shape[shape.len() - 1] / (3 * self.sets as i32);
        let mut parts = split_last(&m, &vec![chunk; 3 * self.sets]).into_iter();
        (0..self.sets)
            .map(|_| {
                let shift = parts.next().expect("chunk count");
                let scale = parts.next().expect("chunk count");
                let gate = parts.next().expect("chunk count");
                [shift, scale, gate]
            })
            .collect()
    }
}

/// `Flux2PosEmbed`: per-axis RoPE angles for `[S, 4]` ids, concatenated.
/// Returns `(cos, sin)`, each `[S, head_dim / 2]` f32.
pub fn pos_embed(
    ids: &MlxArray,
    theta: f32,
    axes_dim: &[i32],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let pos = mlxcel_core::astype(ids, dtype::FLOAT32);
    let s = mlxcel_core::array_shape(&pos)[0];
    let mut cos_parts = Vec::with_capacity(axes_dim.len());
    let mut sin_parts = Vec::with_capacity(axes_dim.len());
    for (i, &dim) in axes_dim.iter().enumerate() {
        let col = mlxcel_core::slice(&pos, &[0, i as i32], &[s, i as i32 + 1]);
        let idx = mlxcel_core::astype(&mlxcel_core::arange_i32(0, dim, 2), dtype::FLOAT32);
        let scale = mlxcel_core::divide(
            &idx,
            &mlxcel_core::full_f32(&[1], dim as f32, dtype::FLOAT32),
        );
        let omega = mlxcel_core::divide(
            &mlxcel_core::full_f32(&[1], 1.0, dtype::FLOAT32),
            &mlxcel_core::power(&mlxcel_core::full_f32(&[1], theta, dtype::FLOAT32), &scale),
        );
        let out = mlxcel_core::multiply(&col, &mlxcel_core::reshape(&omega, &[1, dim / 2]));
        cos_parts.push(mlxcel_core::cos(&out));
        sin_parts.push(mlxcel_core::sin(&out));
    }
    let cat = |parts: &[UniquePtr<MlxArray>]| {
        mlxcel_core::concatenate_many(&parts.iter().map(|a| &**a).collect::<Vec<_>>(), -1)
    };
    (cat(&cos_parts), cat(&sin_parts))
}

/// `AdaLayerNormContinuous` (affine-free LayerNorm, eps 1e-6): `linear(silu(c))`
/// split as `(scale, shift)`, note the order, then `norm(x) * (1 + scale) + shift`.
pub struct AdaLayerNormContinuous {
    linear: Box<dyn Flux2Linear>,
    dim: i32,
}

impl AdaLayerNormContinuous {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            linear: source.load(weights, &format!("{prefix}.linear"))?,
            dim,
        })
    }

    /// `x` is `[B, S, D]`; `cond` is `[B, D]`. `compute_dtype` is upstream's
    /// `ModelConfig.precision`, applied to `silu(cond)` before the linear.
    pub fn forward(
        &self,
        x: &MlxArray,
        cond: &MlxArray,
        compute_dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let c = mlxcel_core::astype(&mlxcel_core::silu(cond), compute_dtype);
        let e = self.linear.forward(&c);
        let parts = split_last(&e, &[self.dim, self.dim]);
        let scale = mlxcel_core::expand_dims(&parts[0], 1);
        let shift = mlxcel_core::expand_dims(&parts[1], 1);
        let one = scalar_like(1.0, &scale);
        mlxcel_core::add(
            &mlxcel_core::multiply(&layer_norm(x), &mlxcel_core::add(&one, &scale)),
            &shift,
        )
    }
}

/// `silu(x1) * x2` over the two halves of the last axis.
pub fn swiglu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let half = shape[shape.len() - 1] / 2;
    let parts = split_last(x, &[half, half]);
    mlxcel_core::multiply(&mlxcel_core::silu(&parts[0]), &parts[1])
}

/// `Flux2FeedForward`: `linear_out(swiglu(linear_in(x)))`.
pub struct FeedForward {
    linear_in: Box<dyn Flux2Linear>,
    linear_out: Box<dyn Flux2Linear>,
}

impl FeedForward {
    pub fn load(weights: &WeightMap, source: &dyn LinearSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_in: source.load(weights, &format!("{prefix}.linear_in"))?,
            linear_out: source.load(weights, &format!("{prefix}.linear_out"))?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear_out.forward(&swiglu(&self.linear_in.forward(x)))
    }
}
