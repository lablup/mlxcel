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

//! Z-Image DiT building blocks.
//!
//! A port of `TimestepEmbedder`, `FeedForward`, `MRoPE`, `apply_rotary`,
//! `Attention`, `ZImageTransformerBlock`, `FinalLayer` and
//! `sanitize_transformer_weights` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/transformer.py>
//! (`#L14-L177`). They are shared by the Z-Image, LLaDA-Image and Ming-Image
//! transformers; each of those assembles its own stack from these blocks.
//!
//! Weights load by key prefix from a [`WeightMap`] whose names follow the
//! post-[`sanitize_transformer_weights`] layout, for example
//! `layers.0.attention.to_q.weight`.
//!
//! The blocks are dense (unquantized). They return the dtype the arithmetic
//! settles on, exactly as upstream: with f32 RoPE tables and a half-precision
//! activation the rotated q and k are f32 unless [`RopePrecision::RoundToInput`]
//! is chosen, and the residual stream follows them. Where upstream mixes
//! dtypes (rotation, SDPA, residual adds) the blocks cast explicitly to
//! upstream MLX's promotion, because the CUDA build of mlxcel's MLX promotes
//! bf16 with f32 to bf16 instead.
//!
//! [`Attention`] runs its SDPA through [`mlxcel_core::layers::attention`], so
//! the f32 score matrix MLX's CUDA fallback materializes is query-chunked to
//! the `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` budget at large image sizes.
//!
//! Used by: ZImage, LLaDAImage, MingImage.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Width of the timestep conditioning vector the AdaLN heads read.
pub const ADALN_EMBED_DIM: i32 = 256;

/// `nn.RMSNorm` default epsilon, which the Z-Image blocks rely on.
const RMS_EPS: f32 = 1e-5;
/// `nn.LayerNorm(eps=1e-6)` of the final layer.
const FINAL_NORM_EPS: f32 = 1e-6;

fn linear(weights: &WeightMap, prefix: &str) -> Result<Linear> {
    Linear::from_weights(weights, prefix)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Z-Image block weight {prefix}"))
}

fn rms_norm(weights: &WeightMap, key: &str, eps: f32) -> Result<RMSNorm> {
    let w = weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("Z-Image block weight not found: {key}"))?;
    Ok(RMSNorm::new(w, eps))
}

/// Split the last axis into `n` equal chunks.
fn split_last(x: &MlxArray, n: i32) -> Vec<UniquePtr<MlxArray>> {
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

/// `x[..., None, :]` on a `[B, D]` tensor: `[B, 1, D]`.
fn add_seq_axis(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(x, 1)
}

fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, mlxcel_core::array_dtype(like))
}

/// Upstream MLX's promotion of two floating dtypes: equal stays, anything
/// with f32 is f32, and a bf16 / f16 mix is f32.
///
/// The CUDA build of mlxcel's MLX patches bf16 with f32 to give bf16, so a
/// mixed op there silently drops to bf16. The blocks cast to this dtype
/// explicitly wherever upstream mixes dtypes, so they compute the same thing
/// on every backend.
fn upstream_promotion(a: i32, b: i32) -> i32 {
    if a == b {
        a
    } else if a == dtype::FLOAT64 || b == dtype::FLOAT64 {
        dtype::FLOAT64
    } else {
        dtype::FLOAT32
    }
}

fn to_dtype(x: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(x) == dt {
        mlxcel_core::copy(x)
    } else {
        mlxcel_core::astype(x, dt)
    }
}

/// `a + b` with [`upstream_promotion`]: the residual adds of the blocks.
fn add_promoted(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
    let (da, db) = (mlxcel_core::array_dtype(a), mlxcel_core::array_dtype(b));
    if da == db {
        return mlxcel_core::add(a, b);
    }
    let dt = upstream_promotion(da, db);
    mlxcel_core::add(&to_dtype(a, dt), &to_dtype(b, dt))
}

/// `a * b` with [`upstream_promotion`]: the modulation products of the
/// blocks. Identical to a plain multiply when the dtypes agree (Z-Image's
/// all-f32 stream); Ming-Image's bf16 modulation meets the f32 stream here.
fn multiply_promoted(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
    let (da, db) = (mlxcel_core::array_dtype(a), mlxcel_core::array_dtype(b));
    if da == db {
        return mlxcel_core::multiply(a, b);
    }
    let dt = upstream_promotion(da, db);
    mlxcel_core::multiply(&to_dtype(a, dt), &to_dtype(b, dt))
}

/// Sinusoidal timestep embedding followed by a two-layer SiLU MLP.
pub struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_size: i32,
}

impl TimestepEmbedder {
    /// Load `<prefix>.linear1` and `<prefix>.linear2`.
    pub fn load(weights: &WeightMap, prefix: &str, frequency_size: i32) -> Result<Self> {
        Ok(Self {
            linear1: linear(weights, &format!("{prefix}.linear1"))?,
            linear2: linear(weights, &format!("{prefix}.linear2"))?,
            frequency_size,
        })
    }

    /// `t` is any shape that flattens to `[N]`; the result is `[N, out_size]`.
    pub fn forward(&self, t: &MlxArray) -> UniquePtr<MlxArray> {
        let half = self.frequency_size / 2;
        let exponent = (0..half)
            .map(|i| -(10000.0_f32.ln()) * i as f32 / half as f32)
            .collect::<Vec<_>>();
        let freqs = mlxcel_core::exp(&mlxcel_core::from_slice_f32(&exponent, &[half]));
        let t_f32 = mlxcel_core::astype(&mlxcel_core::reshape(t, &[-1, 1]), dtype::FLOAT32);
        let args = mlxcel_core::multiply(&t_f32, &mlxcel_core::expand_dims(&freqs, 0));
        let emb = mlxcel_core::concatenate(&mlxcel_core::cos(&args), &mlxcel_core::sin(&args), -1);
        let emb = mlxcel_core::astype(&emb, mlxcel_core::array_dtype(t));
        self.linear2
            .forward(&mlxcel_core::silu(&self.linear1.forward(&emb)))
    }
}

/// SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`, no biases.
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn load(weights: &WeightMap, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: linear(weights, &format!("{prefix}.w1"))?,
            w2: linear(weights, &format!("{prefix}.w2"))?,
            w3: linear(weights, &format!("{prefix}.w3"))?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = mlxcel_core::silu(&self.w1.forward(x));
        self.w2
            .forward(&mlxcel_core::multiply(&gate, &self.w3.forward(x)))
    }
}

/// Multi-axis rotary embedding: each position axis owns a slice of the head
/// dimension (`sections`), rotated with its own frequencies.
#[derive(Debug, Clone, PartialEq)]
pub struct MRoPE {
    pub sections: Vec<i32>,
    pub theta: f32,
}

impl Default for MRoPE {
    fn default() -> Self {
        Self {
            sections: vec![32, 48, 48],
            theta: 256.0,
        }
    }
}

impl MRoPE {
    /// `position_ids` is `[B, L, sections.len()]`; returns `(cos, sin)`, each
    /// `[B, L, sum(sections)]` in f32 with every frequency repeated twice.
    pub fn compute_freqs(
        &self,
        position_ids: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let shape = mlxcel_core::array_shape(position_ids);
        if shape.len() != 3 || shape[2] != self.sections.len() as i32 {
            bail!(
                "position_ids must be [B, L, {}], got {shape:?}",
                self.sections.len()
            );
        }
        let (b, l) = (shape[0], shape[1]);
        let mut cos_parts = Vec::with_capacity(self.sections.len());
        let mut sin_parts = Vec::with_capacity(self.sections.len());
        for (i, &s) in self.sections.iter().enumerate() {
            if s <= 0 || s % 2 != 0 {
                bail!("RoPE section sizes must be positive and even, got {s}");
            }
            let ids = mlxcel_core::slice(position_ids, &[0, 0, i as i32], &[b, l, i as i32 + 1]);
            let ids = mlxcel_core::astype(&ids, dtype::FLOAT32);
            let inv_freq: Vec<f32> = (0..s / 2)
                .map(|k| 1.0 / self.theta.powf((2 * k) as f32 / s as f32))
                .collect();
            let inv_freq = mlxcel_core::from_slice_f32(&inv_freq, &[1, 1, s / 2]);
            let angle = mlxcel_core::multiply(&ids, &inv_freq);
            cos_parts.push(mlxcel_core::repeat(&mlxcel_core::cos(&angle), 2, 2));
            sin_parts.push(mlxcel_core::repeat(&mlxcel_core::sin(&angle), 2, 2));
        }
        Ok((concat_last(&cos_parts), concat_last(&sin_parts)))
    }
}

fn concat_last(parts: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    mlxcel_core::concatenate_many(&refs, -1)
}

/// Take element 0 or 1 of the trailing pair axis of `[..., D/2, 2]`.
fn pair_element(x: &MlxArray, index: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let rank = shape.len();
    let mut starts = vec![0; rank];
    let mut stops = shape.clone();
    starts[rank - 1] = index;
    stops[rank - 1] = index + 1;
    let picked = mlxcel_core::slice(x, &starts, &stops);
    mlxcel_core::reshape(&picked, &shape[..rank - 1])
}

/// Rotate adjacent pairs of `x` (`[B, L, H, D]`) by the angles in `cos` and
/// `sin` (`[B, L, D]`, each frequency repeated twice).
pub fn apply_rotary(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    if shape.len() != 4 || shape[3] % 2 != 0 {
        bail!("apply_rotary expects [B, L, H, D] with even D, got {shape:?}");
    }
    let (b, l, h, d) = (shape[0], shape[1], shape[2], shape[3]);
    // The tables keep their own batch size (1 when shared across the batch)
    // and broadcast, as upstream's `cos.reshape(*cos.shape[:-1], D // 2, 2)`.
    let table = |t: &MlxArray| {
        let t = mlxcel_core::expand_dims(t, 2);
        let tb = mlxcel_core::array_shape(&t)[0];
        let t = mlxcel_core::reshape(&t, &[tb, l, 1, d / 2, 2]);
        pair_element(&t, 0)
    };
    let (cos_r, sin_r) = (table(cos), table(sin));
    let (dx, dc) = (mlxcel_core::array_dtype(x), mlxcel_core::array_dtype(cos));
    let x_pairs = if dx == dc {
        mlxcel_core::reshape(x, &[b, l, h, d / 2, 2])
    } else {
        mlxcel_core::reshape(
            &to_dtype(x, upstream_promotion(dx, dc)),
            &[b, l, h, d / 2, 2],
        )
    };
    let (x_real, x_imag) = (pair_element(&x_pairs, 0), pair_element(&x_pairs, 1));
    let out_real = mlxcel_core::subtract(
        &mlxcel_core::multiply(&x_real, &cos_r),
        &mlxcel_core::multiply(&x_imag, &sin_r),
    );
    let out_imag = mlxcel_core::add(
        &mlxcel_core::multiply(&x_real, &sin_r),
        &mlxcel_core::multiply(&x_imag, &cos_r),
    );
    let stacked = mlxcel_core::stack_owned(&[out_real, out_imag], -1);
    Ok(mlxcel_core::reshape(&stacked, &shape))
}

/// What happens to q and k after rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopePrecision {
    /// Keep the promoted dtype (f32 tables with half-precision activations give
    /// f32 q and k). This is Z-Image's `Attention`.
    #[default]
    KeepPromoted,
    /// Round q and k back to the activation dtype before attention, as
    /// LLaDA-Image's `Attention` does.
    RoundToInput,
}

/// Multi-head self-attention with q/k RMS norms and rotary embedding.
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RMSNorm,
    norm_k: RMSNorm,
    num_heads: i32,
    head_dim: i32,
    rope: RopePrecision,
}

impl Attention {
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        rope: RopePrecision,
    ) -> Result<Self> {
        Ok(Self {
            to_q: linear(weights, &format!("{prefix}.to_q"))?,
            to_k: linear(weights, &format!("{prefix}.to_k"))?,
            to_v: linear(weights, &format!("{prefix}.to_v"))?,
            to_out: linear(weights, &format!("{prefix}.to_out.0"))?,
            norm_q: rms_norm(weights, &format!("{prefix}.norm_q.weight"), RMS_EPS)?,
            norm_k: rms_norm(weights, &format!("{prefix}.norm_k.weight"), RMS_EPS)?,
            num_heads,
            head_dim,
            rope,
        })
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let heads = [b, l, self.num_heads, self.head_dim];
        let q = self
            .norm_q
            .forward(&mlxcel_core::reshape(&self.to_q.forward(x), &heads));
        let k = self
            .norm_k
            .forward(&mlxcel_core::reshape(&self.to_k.forward(x), &heads));
        let v = mlxcel_core::reshape(&self.to_v.forward(x), &heads);
        let mut q = apply_rotary(&q, cos, sin)?;
        let mut k = apply_rotary(&k, cos, sin)?;
        if self.rope == RopePrecision::RoundToInput {
            let dt = mlxcel_core::array_dtype(x);
            q = mlxcel_core::astype(&q, dt);
            k = mlxcel_core::astype(&k, dt);
        }
        // SDPA runs in `result_type(q, k, v)`; promote v the way upstream does
        // rather than letting a bf16 v pull f32 q and k down.
        let v = to_dtype(
            &v,
            upstream_promotion(mlxcel_core::array_dtype(&q), mlxcel_core::array_dtype(&v)),
        );
        let to_heads = |t: &MlxArray| mlxcel_core::transpose_axes(t, &[0, 2, 1, 3]);
        // Through the shared dispatch, not the raw FFI: f32 q/k never reach
        // CUDA's fused SDPA, so MLX's fallback would materialize the whole
        // `[B, heads, L, L]` f32 score matrix (~500 GB per layer at 4096x4096).
        // `attention` query-chunks it to `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`
        // with row-identical math, and is the same single SDPA call below it.
        let out = mlxcel_core::layers::attention(
            &to_heads(&q),
            &to_heads(&k),
            &to_heads(&v),
            (self.head_dim as f32).powf(-0.5),
            mask,
            0.0,
            0,
        );
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[b, l, -1],
        );
        Ok(self.to_out.forward(&out))
    }
}

/// Transformer block with optional AdaLN modulation from the timestep embedding.
///
/// With `modulation` the block reads `adaln_input` (`[B, adaln_dim]`) and gates
/// both residual branches; without it (the context refiner) it is a plain
/// pre-norm block with post-norms.
pub struct ZImageTransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm1: RMSNorm,
    attention_norm2: RMSNorm,
    ffn_norm1: RMSNorm,
    ffn_norm2: RMSNorm,
    ada_ln_modulation: Option<Linear>,
}

impl ZImageTransformerBlock {
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        modulation: bool,
        rope: RopePrecision,
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(
                weights,
                &format!("{prefix}.attention"),
                num_heads,
                head_dim,
                rope,
            )?,
            feed_forward: FeedForward::load(weights, &format!("{prefix}.feed_forward"))?,
            attention_norm1: rms_norm(
                weights,
                &format!("{prefix}.attention_norm1.weight"),
                RMS_EPS,
            )?,
            attention_norm2: rms_norm(
                weights,
                &format!("{prefix}.attention_norm2.weight"),
                RMS_EPS,
            )?,
            ffn_norm1: rms_norm(weights, &format!("{prefix}.ffn_norm1.weight"), RMS_EPS)?,
            ffn_norm2: rms_norm(weights, &format!("{prefix}.ffn_norm2.weight"), RMS_EPS)?,
            ada_ln_modulation: if modulation {
                Some(linear(weights, &format!("{prefix}.adaLN_modulation.0"))?)
            } else {
                None
            },
        })
    }

    /// Whether the block expects an `adaln_input`.
    #[must_use]
    pub fn has_modulation(&self) -> bool {
        self.ada_ln_modulation.is_some()
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
        adaln_input: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        match (&self.ada_ln_modulation, adaln_input) {
            (Some(modulation), Some(adaln)) => {
                let chunks = split_last(&modulation.forward(adaln), 4);
                let one = |c: &MlxArray| scalar_like(1.0, c);
                let scale_msa = add_seq_axis(&mlxcel_core::add(&one(&chunks[0]), &chunks[0]));
                let gate_msa = add_seq_axis(&mlxcel_core::tanh(&chunks[1]));
                let scale_mlp = add_seq_axis(&mlxcel_core::add(&one(&chunks[2]), &chunks[2]));
                let gate_mlp = add_seq_axis(&mlxcel_core::tanh(&chunks[3]));

                let attn_in = multiply_promoted(&self.attention_norm1.forward(x), &scale_msa);
                let attn_out = self.attention.forward(&attn_in, cos, sin, mask)?;
                let x = add_promoted(
                    x,
                    &multiply_promoted(&gate_msa, &self.attention_norm2.forward(&attn_out)),
                );
                let ffn_in = multiply_promoted(&self.ffn_norm1.forward(&x), &scale_mlp);
                let ffn_out = self.feed_forward.forward(&ffn_in);
                Ok(add_promoted(
                    &x,
                    &multiply_promoted(&gate_mlp, &self.ffn_norm2.forward(&ffn_out)),
                ))
            }
            (Some(_), None) => bail!("this Z-Image block is modulated and needs an adaln_input"),
            (None, _) => {
                let attn_out =
                    self.attention
                        .forward(&self.attention_norm1.forward(x), cos, sin, mask)?;
                let x = add_promoted(x, &self.attention_norm2.forward(&attn_out));
                let ffn_out = self.feed_forward.forward(&self.ffn_norm1.forward(&x));
                Ok(add_promoted(&x, &self.ffn_norm2.forward(&ffn_out)))
            }
        }
    }
}

/// Final projection: affine-free LayerNorm, AdaLN scale from the timestep
/// embedding, then a linear map to patch space.
pub struct FinalLayer {
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl FinalLayer {
    pub fn load(weights: &WeightMap, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: linear(weights, &format!("{prefix}.linear"))?,
            ada_ln_modulation: linear(weights, &format!("{prefix}.adaLN_modulation.0"))?,
        })
    }

    pub fn forward(&self, x: &MlxArray, c: &MlxArray) -> UniquePtr<MlxArray> {
        let modulation = self.ada_ln_modulation.forward(&mlxcel_core::silu(c));
        let scale = add_seq_axis(&mlxcel_core::add(
            &scalar_like(1.0, &modulation),
            &modulation,
        ));
        // SAFETY: null weight and bias select the affine-free normalization.
        let normed = unsafe {
            mlxcel_core::fast_layer_norm(x, std::ptr::null(), std::ptr::null(), FINAL_NORM_EPS)
        };
        self.linear.forward(&multiply_promoted(&normed, &scale))
    }
}

/// Map the source checkpoint's key names onto the block layout above:
///
/// - `all_final_layer.2-1.*` -> `final_layer.*`, with `adaLN_modulation.1.`
///   renamed `adaLN_modulation.0.`
/// - `all_x_embedder.2-1.*` -> `x_embedder.*`
/// - `t_embedder.mlp.0.` / `t_embedder.mlp.2.` -> `t_embedder.linear1.` / `linear2.`
///
/// Every other key passes through unchanged.
#[must_use]
pub fn sanitize_transformer_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .map(|(key, value)| (sanitize_key(&key), value))
        .collect()
}

fn sanitize_key(key: &str) -> String {
    let mut key = if let Some(rest) = key.strip_prefix("all_final_layer.2-1.") {
        format!("final_layer.{rest}").replace("adaLN_modulation.1.", "adaLN_modulation.0.")
    } else if let Some(rest) = key.strip_prefix("all_x_embedder.2-1.") {
        format!("x_embedder.{rest}")
    } else {
        key.to_string()
    };
    key = key.replace("t_embedder.mlp.0.", "t_embedder.linear1.");
    key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
}

#[cfg(test)]
#[path = "dit_blocks_tests.rs"]
mod tests;
