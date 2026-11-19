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

//! `MOSSImageTransformer`, the 42-block single-stream DiT of MOSS-Image, a
//! port of `dit.py` in `felix-ab/moss-image-mlx` (which mirrors OpenMOSS's
//! `pipeline_moss_image.py` operation by operation).
//!
//! The sequence is the image's latent tokens followed by the prompt's
//! conditioner tokens. Every block shares one adaLN modulation computed from
//! the timestep ([`MossTransformer::modulation`]); attention has no mask
//! (image and text see each other), carries a 4-axis RoPE (row, time,
//! height, width) and per-head RMS-normalised queries and keys; the
//! feed-forward uses the SiTU gate. The final head modulation applies to the
//! image tokens only.
//!
//! **Deviation, recorded:** the reference's optional `precise` mode keeps the
//! residual stream in fp32. This port runs in the checkpoint's dtype
//! throughout (the same choice the other quantized families make), which is
//! the mode the published MLX 8-bit tier was validated in.
//!
//! The negative branch of classifier-free guidance skips one block
//! (`uncond_skip_layers`); [`MossTransformer::forward`] takes the skip set per
//! call so one loaded model serves both branches.

use anyhow::{Context, Result, ensure};
use mlxcel_core::layers::RMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::TransformerConfig;
use crate::imagegen::mage_flow::layers::{
    Dense, LinearQuant, modulate, rms_norm, scalar_like, split_last,
};
use crate::imagegen::quant::QuantSpec;

/// The reference's norm epsilon, on every RMSNorm in the transformer.
pub const NORM_EPS: f32 = 1e-6;

/// `x.astype(f32) * sigmoid(x.astype(f32))`, rounded back: torch's fused silu.
fn silu_like_torch(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let x32 = mlxcel_core::astype(x, dtype::FLOAT32);
    mlxcel_core::astype(
        &mlxcel_core::multiply(&x32, &mlxcel_core::sigmoid(&x32)),
        dt,
    )
}

/// SiTU: `(4 tanh(gate/4) sigmoid(gate)) * (25 tanh(hidden/25))`.
fn situ(hidden: &MlxArray, gate: &MlxArray) -> UniquePtr<MlxArray> {
    let gate_quarter = mlxcel_core::multiply(
        &mlxcel_core::tanh(&mlxcel_core::multiply(gate, &scalar_like(0.25, gate))),
        &scalar_like(4.0, gate),
    );
    let hidden_25 = mlxcel_core::multiply(
        &mlxcel_core::tanh(&mlxcel_core::multiply(
            hidden,
            &scalar_like(1.0 / 25.0, hidden),
        )),
        &scalar_like(25.0, hidden),
    );
    mlxcel_core::multiply(
        &mlxcel_core::multiply(&gate_quarter, &mlxcel_core::sigmoid(gate)),
        &hidden_25,
    )
}

/// The logical `[out, in]` of the projection at `prefix`.
///
/// A quantized projection (`<prefix>.scales` present) packs `in` into
/// `in * bits / 32` u32 words; the row count is `out` in both layouts.
fn linear_dims(w: &WeightMap, prefix: &str, quant: Option<LinearQuant>) -> Result<(i32, i32)> {
    let key = format!("{prefix}.weight");
    let weight = w
        .get(&key)
        .with_context(|| format!("MOSS-Image weight not found: {key}"))?;
    let shape = mlxcel_core::array_shape(weight);
    ensure!(
        shape.len() == 2,
        "MOSS-Image weight {key} must be 2-D, got {shape:?}"
    );
    let packed = w.contains_key(&format!("{prefix}.scales"));
    let inn = match (packed, quant) {
        (false, _) => shape[1],
        (true, Some(q)) if q.bits > 0 => shape[1] * 32 / q.bits,
        (true, _) => {
            anyhow::bail!(
                "MOSS-Image weight {prefix} is quantized but the checkpoint has no quantization metadata"
            )
        }
    };
    Ok((shape[0], inn))
}

/// Refuse a projection whose logical shape is not `[out, inn]`.
///
/// Every width checked here is one the forward pass relies on (a reshape, a
/// residual add, a concatenation), so a checkpoint the check rejects could not
/// have run: MLX would abort the process on the first mismatched op instead of
/// returning an error.
fn expect_linear(
    w: &WeightMap,
    prefix: &str,
    quant: Option<LinearQuant>,
    out: i32,
    inn: i32,
) -> Result<()> {
    let actual = linear_dims(w, prefix, quant)?;
    ensure!(
        actual == (out, inn),
        "MOSS-Image weight {prefix} has logical shape [{}, {}], expected [{out}, {inn}]",
        actual.0,
        actual.1
    );
    Ok(())
}

/// Refuse an RMSNorm weight that is not `[width]`.
fn expect_norm(w: &WeightMap, key: &str, width: i32) -> Result<()> {
    let weight = w
        .get(key)
        .with_context(|| format!("MOSS-Image weight not found: {key}"))?;
    let shape = mlxcel_core::array_shape(weight);
    ensure!(
        shape == [width],
        "MOSS-Image weight {key} has shape {shape:?}, expected [{width}]"
    );
    Ok(())
}

/// Check the norms and projections of one block (`TextBlock` or `Block`)
/// against the residual width `dim` and the per-head width `head_dim`.
fn expect_block_norms(w: &WeightMap, prefix: &str, dim: i32, head_dim: i32) -> Result<()> {
    expect_norm(w, &format!("{prefix}.norm1.weight"), dim)?;
    expect_norm(w, &format!("{prefix}.norm2.weight"), dim)?;
    expect_norm(w, &format!("{prefix}.q_norm.weight"), head_dim)?;
    expect_norm(w, &format!("{prefix}.k_norm.weight"), head_dim)
}

/// `(sin, cos)` for the 4-axis RoPE, each `[1, n, 1, head_dim]`.
///
/// Coordinates `(r, t, h, w)`: image tokens have `r = 0`, `t = text_length`
/// and the centred pixel position; text tokens have `r = 0`, `t = index` and
/// `h = w = 0`. Computed on the CPU in f32 and cast to `dt`, as the reference
/// does.
#[must_use]
pub fn rope_table(
    height: i32,
    width: i32,
    text_length: i32,
    head_dim: i32,
    base: f32,
    dt: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let bands_n = head_dim / 8;
    let bands: Vec<f32> = (0..bands_n)
        .map(|i| 1.0 / base.powf(i as f32 / bands_n as f32))
        .collect();
    let n = (height * width + text_length) as usize;
    // Half a head: the four axes each contribute `head_dim / 8` bands, so one
    // token's rotation is `4 * head_dim / 8 = head_dim / 2` values, which the
    // concatenation below doubles back into a full head. Striding by the full
    // `head_dim` here makes the doubling write one head past the end of the
    // last row (the panic: len 528000, index 528000 at n = 4125, head_dim = 128).
    let half = (head_dim / 2) as usize;
    let mut sin = vec![0f32; n * half];
    let mut cos = vec![0f32; n * half];
    let centre_h = (height - 1) as f32 / 2.0;
    let centre_w = (width - 1) as f32 / 2.0;
    let mut phase = [0f32; 4];
    let push = |token: usize, phase: [f32; 4], sin: &mut Vec<f32>, cos: &mut Vec<f32>| {
        for (axis, value) in phase.iter().enumerate() {
            for (band, &scale) in bands.iter().enumerate() {
                let index = token * half + axis * bands_n as usize + band;
                sin[index] = (value * scale).sin();
                cos[index] = (value * scale).cos();
            }
        }
    };
    let mut token = 0usize;
    for h in 0..height {
        for w in 0..width {
            phase[0] = 0.0;
            phase[1] = text_length as f32;
            phase[2] = h as f32 - centre_h;
            phase[3] = w as f32 - centre_w;
            push(token, phase, &mut sin, &mut cos);
            token += 1;
        }
    }
    for t in 0..text_length {
        phase = [0.0, t as f32, 0.0, 0.0];
        push(token, phase, &mut sin, &mut cos);
        token += 1;
    }
    // The reference concatenates the rotation onto itself: the same angle
    // drives both halves of each head.
    let mut doubled_sin = vec![0f32; n * head_dim as usize];
    let mut doubled_cos = vec![0f32; n * head_dim as usize];
    for row in 0..n {
        for i in 0..half {
            doubled_sin[row * head_dim as usize + i] = sin[row * half + i];
            doubled_cos[row * head_dim as usize + i] = cos[row * half + i];
            doubled_sin[row * head_dim as usize + half + i] = sin[row * half + i];
            doubled_cos[row * head_dim as usize + half + i] = cos[row * half + i];
        }
    }
    let shape = [1, n as i32, 1, head_dim];
    (
        mlxcel_core::astype(&mlxcel_core::from_slice_f32(&doubled_sin, &shape), dt),
        mlxcel_core::astype(&mlxcel_core::from_slice_f32(&doubled_cos, &shape), dt),
    )
}

/// The reference's float64 timestep sinusoid (`cos` first), returned as f32.
#[must_use]
pub fn timestep_embedding(timestep: f32, dim: i32) -> Vec<f32> {
    let half = (dim / 2) as usize;
    let t = f64::from(timestep);
    let mut out = Vec::with_capacity(2 * half);
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for i in 0..half {
        let freq = 10000f64.powf(-(i as f64) / half as f64);
        let phase = t * freq;
        cos.push(phase.cos() as f32);
        sin.push(phase.sin() as f32);
    }
    out.extend_from_slice(&cos);
    out.extend_from_slice(&sin);
    out
}

/// Rotate `x` `[b, n, heads, hd]` by `(sin, cos)` `[1, n, 1, hd]`.
fn rotate(x: &MlxArray, sin: &MlxArray, cos: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let hd = shape[3];
    let half = hd / 2;
    let first = mlxcel_core::slice(x, &[0, 0, 0, 0], &[shape[0], shape[1], shape[2], half]);
    let second = mlxcel_core::slice(x, &[0, 0, 0, half], &[shape[0], shape[1], shape[2], hd]);
    let rotated = mlxcel_core::concatenate(&mlxcel_core::negative(&second), &first, 3);
    mlxcel_core::add(
        &mlxcel_core::multiply(x, cos),
        &mlxcel_core::multiply(&rotated, sin),
    )
}

/// Per-head RMSNorm: reshape to `[b, n, heads, hd]`, normalise, reshape back.
fn head_norm(norm: &RMSNorm, x: &MlxArray, heads: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (b, n, dim) = (shape[0], shape[1], shape[2]);
    let hd = dim / heads;
    let reshaped = mlxcel_core::reshape(x, &[b, n, heads, hd]);
    mlxcel_core::reshape(&norm.forward(&reshaped), &[b, n, dim])
}

/// Unmasked scaled dot-product attention over `[b, n, dim]` inputs.
fn attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    sin: &MlxArray,
    cos: &MlxArray,
    heads: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(q);
    let (b, n, dim) = (shape[0], shape[1], shape[2]);
    let hd = dim / heads;
    let to_heads = |x: &MlxArray| mlxcel_core::reshape(x, &[b, n, heads, hd]);
    let q = mlxcel_core::transpose_axes(&rotate(&to_heads(q), sin, cos), &[0, 2, 1, 3]);
    let k = mlxcel_core::transpose_axes(&rotate(&to_heads(k), sin, cos), &[0, 2, 1, 3]);
    let v = mlxcel_core::transpose_axes(&to_heads(v), &[0, 2, 1, 3]);
    let out = mlxcel_core::layers::attention(&q, &k, &v, 1.0 / (hd as f32).sqrt(), None, 0.0, 0);
    mlxcel_core::reshape(
        &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
        &[b, n, dim],
    )
}

struct TextBlock {
    norm1: RMSNorm,
    norm2: RMSNorm,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    qkv: Dense,
    proj: Dense,
    ffn_gate: Dense,
    ffn_up: Dense,
    ffn_down: Dense,
}

impl TextBlock {
    fn load(
        w: &WeightMap,
        prefix: &str,
        quant: Option<LinearQuant>,
        dim: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let dense = |name: &str| Dense::load(w, &format!("{prefix}.{name}"), quant);
        let norm = |name: &str| rms_norm(w, &format!("{prefix}.{name}.weight"), NORM_EPS);
        expect_block_norms(w, prefix, dim, head_dim)?;
        expect_linear(w, &format!("{prefix}.qkv"), quant, 3 * dim, dim)?;
        expect_linear(w, &format!("{prefix}.proj"), quant, dim, dim)?;
        let (ffn, _) = linear_dims(w, &format!("{prefix}.ffn_up"), quant)?;
        expect_linear(w, &format!("{prefix}.ffn_up"), quant, ffn, dim)?;
        expect_linear(w, &format!("{prefix}.ffn_gate"), quant, ffn, dim)?;
        expect_linear(w, &format!("{prefix}.ffn_down"), quant, dim, ffn)?;
        Ok(Self {
            norm1: norm("norm1")?,
            norm2: norm("norm2")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            qkv: dense("qkv")?,
            proj: dense("proj")?,
            ffn_gate: dense("ffn_gate")?,
            ffn_up: dense("ffn_up")?,
            ffn_down: dense("ffn_down")?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        sin: &MlxArray,
        cos: &MlxArray,
        heads: i32,
    ) -> Result<UniquePtr<MlxArray>> {
        let qkv = self.qkv.forward(&self.norm1.forward(x));
        let parts = split_last(&qkv, 3);
        ensure!(parts.len() == 3, "text qkv split");
        let q = head_norm(&self.q_norm, &parts[0], heads);
        let k = head_norm(&self.k_norm, &parts[1], heads);
        let attn = attention(&q, &k, &parts[2], sin, cos, heads);
        let x = mlxcel_core::add(x, &self.proj.forward(&attn));
        let y = self.norm2.forward(&x);
        let hidden = self.ffn_up.forward(&y);
        let gate = self.ffn_gate.forward(&y);
        Ok(mlxcel_core::add(
            &x,
            &self.ffn_down.forward(&situ(&hidden, &gate)),
        ))
    }
}

struct TextProjection {
    proj: Dense,
    blocks: Vec<TextBlock>,
}

impl TextProjection {
    fn load(
        w: &WeightMap,
        prefix: &str,
        quant: Option<LinearQuant>,
        config: &TransformerConfig,
    ) -> Result<Self> {
        let (dim, head_dim) = (config.dim, config.head_dim());
        expect_linear(w, &format!("{prefix}.proj"), quant, dim, config.text_dim)?;
        Ok(Self {
            proj: Dense::load(w, &format!("{prefix}.proj"), quant)?,
            blocks: (0..2)
                .map(|i| TextBlock::load(w, &format!("{prefix}.blocks.{i}"), quant, dim, head_dim))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        sin: &MlxArray,
        cos: &MlxArray,
        heads: i32,
    ) -> Result<UniquePtr<MlxArray>> {
        let mut x = self.proj.forward(x);
        for block in &self.blocks {
            x = block.forward(&x, sin, cos, heads)?;
        }
        Ok(x)
    }
}

struct Block {
    norm1: RMSNorm,
    norm2: RMSNorm,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    ffn_up_gate: Dense,
    ffn_down: Dense,
}

impl Block {
    fn load(
        w: &WeightMap,
        prefix: &str,
        quant: Option<LinearQuant>,
        dim: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let dense = |name: &str| Dense::load(w, &format!("{prefix}.{name}"), quant);
        let norm = |name: &str| rms_norm(w, &format!("{prefix}.{name}.weight"), NORM_EPS);
        expect_block_norms(w, prefix, dim, head_dim)?;
        for name in ["q", "k", "v", "o"] {
            expect_linear(w, &format!("{prefix}.proj.{name}"), quant, dim, dim)?;
        }
        let (up_gate, _) = linear_dims(w, &format!("{prefix}.proj.ffn_up_gate"), quant)?;
        ensure!(
            up_gate % 2 == 0,
            "MOSS-Image weight {prefix}.proj.ffn_up_gate has an odd output width {up_gate}"
        );
        expect_linear(
            w,
            &format!("{prefix}.proj.ffn_up_gate"),
            quant,
            up_gate,
            dim,
        )?;
        expect_linear(
            w,
            &format!("{prefix}.proj.ffn_down"),
            quant,
            dim,
            up_gate / 2,
        )?;
        Ok(Self {
            norm1: norm("norm1")?,
            norm2: norm("norm2")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            q: dense("proj.q")?,
            k: dense("proj.k")?,
            v: dense("proj.v")?,
            o: dense("proj.o")?,
            ffn_up_gate: dense("proj.ffn_up_gate")?,
            ffn_down: dense("proj.ffn_down")?,
        })
    }

    /// One block of the joint sequence. `mod` is `[1, 6, dim]`.
    fn forward(
        &self,
        x: &MlxArray,
        mods: &[UniquePtr<MlxArray>; 6],
        sin: &MlxArray,
        cos: &MlxArray,
        heads: i32,
    ) -> UniquePtr<MlxArray> {
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = mods;
        let y = modulate(&self.norm1.forward(x), shift_a, scale_a);
        let q = head_norm(&self.q_norm, &self.q.forward(&y), heads);
        let k = head_norm(&self.k_norm, &self.k.forward(&y), heads);
        let v = self.v.forward(&y);
        let attn = attention(&q, &k, &v, sin, cos, heads);
        let x = mlxcel_core::add(x, &mlxcel_core::multiply(&self.o.forward(&attn), gate_a));
        let y = modulate(&self.norm2.forward(&x), shift_f, scale_f);
        let parts = split_last(&self.ffn_up_gate.forward(&y), 2);
        mlxcel_core::add(
            &x,
            &mlxcel_core::multiply(&self.ffn_down.forward(&situ(&parts[0], &parts[1])), gate_f),
        )
    }
}

/// The MOSS-Image diffusion transformer.
pub struct MossTransformer {
    config: TransformerConfig,
    /// Activation dtype: the dtype of the loaded projections.
    dt: i32,
    video_embedder: Dense,
    text_embedder: TextProjection,
    time_embedding: [Dense; 2],
    shared_block_modulation: Dense,
    layers: Vec<Block>,
    final_norm_video: RMSNorm,
    final_linear_video: Dense,
    video_head_modulation: Dense,
}

impl MossTransformer {
    /// Load from a `transformer/` weight map.
    ///
    /// `quant` comes from the checkpoint's `moss_image_mlx.quantization`
    /// metadata; an exported MLX checkpoint is quantized, the original
    /// Diffusers tree is dense.
    pub fn load(
        w: &WeightMap,
        config: TransformerConfig,
        quant: Option<LinearQuant>,
    ) -> Result<Self> {
        config.validate()?;
        let dt = w
            .get("video_embedder.weight")
            .map(|t| mlxcel_core::array_dtype(t))
            .context("MOSS-Image weight not found: video_embedder.weight")?;
        let (dim, head_dim) = (config.dim, config.head_dim());
        expect_linear(w, "video_embedder", quant, dim, config.latent_channels)?;
        let (time_hidden, _) = linear_dims(w, "time_embedding.0", quant)?;
        expect_linear(w, "time_embedding.0", quant, time_hidden, config.freq_dim)?;
        let (time_out, _) = linear_dims(w, "time_embedding.1", quant)?;
        expect_linear(w, "time_embedding.1", quant, time_out, time_hidden)?;
        expect_linear(w, "shared_block_modulation", quant, 6 * dim, time_out)?;
        expect_linear(w, "video_head_modulation", quant, 2 * dim, time_out)?;
        expect_norm(w, "final_norm_video.weight", dim)?;
        expect_linear(w, "final_linear_video", quant, config.latent_channels, dim)?;
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            layers.push(Block::load(
                w,
                &format!("layers.{i}"),
                quant,
                dim,
                head_dim,
            )?);
        }
        let time_dense = |name: &str| Dense::load(w, name, quant);
        Ok(Self {
            video_embedder: time_dense("video_embedder")?,
            text_embedder: TextProjection::load(w, "text_embedder", quant, &config)?,
            time_embedding: [
                time_dense("time_embedding.0")?,
                time_dense("time_embedding.1")?,
            ],
            shared_block_modulation: time_dense("shared_block_modulation")?,
            layers,
            final_norm_video: rms_norm(w, "final_norm_video.weight", NORM_EPS)?,
            final_linear_video: time_dense("final_linear_video")?,
            video_head_modulation: time_dense("video_head_modulation")?,
            config,
            dt,
        })
    }

    /// The transformer's shape.
    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// The activation dtype.
    #[must_use]
    pub fn dt(&self) -> i32 {
        self.dt
    }

    /// Shared block modulation `[1, 6, dim]` and head modulation `[1, 2, dim]`.
    fn modulation(&self, timestep: f32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let emb = timestep_embedding(timestep, self.config.freq_dim);
        let emb = mlxcel_core::from_slice_f32(&emb, &[1, self.config.freq_dim]);
        let emb = mlxcel_core::astype(&emb, self.dt);
        let time =
            self.time_embedding[1].forward(&silu_like_torch(&self.time_embedding[0].forward(&emb)));
        let dim = self.config.dim;
        let block =
            mlxcel_core::reshape(&self.shared_block_modulation.forward(&time), &[1, 6, dim]);
        let head = mlxcel_core::reshape(&self.video_head_modulation.forward(&time), &[1, 2, dim]);
        (block, head)
    }

    /// Velocity prediction.
    ///
    /// `latents` is `[1, latent_channels, height, width]`; `context` is
    /// `[1, text_length, text_dim]` (unpadded). `skip_layers` names blocks to
    /// omit — the negative branch of CFG skips one.
    pub fn forward(
        &self,
        latents: &MlxArray,
        timestep: f32,
        context: &MlxArray,
        skip_layers: &[i32],
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(latents);
        ensure!(
            shape[0] == 1,
            "MOSS-Image evaluates one sample at a time (batch 1), got {}",
            shape[0]
        );
        let (height, width) = (shape[2], shape[3]);
        let c = self.config.latent_channels;
        let n_img = height * width;
        ensure!(
            shape[1] == c,
            "MOSS-Image latents must carry {c} channels, got {}",
            shape[1]
        );
        let context_shape = mlxcel_core::array_shape(context);
        ensure!(
            context_shape[0] == 1 && context_shape[2] == self.config.text_dim,
            "MOSS-Image context must be [1, L, {}], got {:?}",
            self.config.text_dim,
            context_shape
        );
        let text_length = context_shape[1];

        let video =
            mlxcel_core::transpose_axes(&mlxcel_core::reshape(latents, &[1, c, n_img]), &[0, 2, 1]);
        let video = self
            .video_embedder
            .forward(&mlxcel_core::astype(&video, self.dt));
        let (sin, cos) = rope_table(
            height,
            width,
            text_length,
            self.config.head_dim(),
            self.config.rope_base,
            self.dt,
        );
        let text = self.text_embedder.forward(
            context,
            &mlxcel_core::slice(
                &sin,
                &[0, n_img, 0, 0],
                &[1, n_img + text_length, 1, self.config.head_dim()],
            ),
            &mlxcel_core::slice(
                &cos,
                &[0, n_img, 0, 0],
                &[1, n_img + text_length, 1, self.config.head_dim()],
            ),
            self.config.num_heads,
        )?;
        let mut x = mlxcel_core::concatenate(&video, &text, 1);
        let (block_mod, head_mod) = self.modulation(timestep);
        let mods = split_blocks(&block_mod, self.config.dim);
        for (index, layer) in self.layers.iter().enumerate() {
            if skip_layers.contains(&(index as i32)) {
                continue;
            }
            x = layer.forward(&x, &mods, &sin, &cos, self.config.num_heads);
        }
        let image = mlxcel_core::slice(&x, &[0, 0, 0], &[1, n_img, self.config.dim]);
        let normed = self.final_norm_video.forward(&image);
        // The head modulation carries two ROWS (`[1, 2, dim]`), not two halves of
        // one row: splitting the last axis here yields `[1, 2, dim / 2]` and the
        // broadcast against `[1, n_img, dim]` aborts with
        // `Shapes (1,1024,3072) and (1,2,1536) cannot be broadcast`.
        let dim = self.config.dim;
        let [shift, scale]: [UniquePtr<MlxArray>; 2] = std::array::from_fn(|i| {
            let i = i as i32;
            mlxcel_core::slice(&head_mod, &[0, i, 0], &[1, i + 1, dim])
        });
        let out = modulate(&normed, &shift, &scale);
        let out = self.final_linear_video.forward(&out);
        Ok(mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1]),
            &[1, c, height, width],
        ))
    }
}

/// The six `[1, 1, dim]` modulation rows of one timestep.
fn split_blocks(mods: &MlxArray, dim: i32) -> [UniquePtr<MlxArray>; 6] {
    std::array::from_fn(|i| {
        let i = i as i32;
        mlxcel_core::slice(mods, &[0, i, 0], &[1, i + 1, dim])
    })
}

/// Map the original Diffusers state dict onto the module names and drop the
/// derived constants. An exported MLX checkpoint needs no remapping.
#[must_use]
pub fn sanitize(weights: &WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in weights.iter() {
        let key = if key == "rope.bands" {
            continue;
        } else {
            key.replace("time_embedding.2.", "time_embedding.1.")
        };
        out.insert(key, mlxcel_core::copy(value));
    }
    out
}

/// Build the transformer from a loaded weight map, applying the checkpoint's
/// quantization metadata when present.
pub fn load_transformer(
    weights: &WeightMap,
    config: TransformerConfig,
    quantization: Option<&serde_json::Value>,
) -> Result<MossTransformer> {
    let quant = match quantization {
        None | Some(serde_json::Value::Null) => None,
        Some(spec) => {
            let bits = spec.get("bits").and_then(|v| v.as_i64()).unwrap_or(0);
            let group_size = spec.get("group_size").and_then(|v| v.as_i64()).unwrap_or(0);
            let bits = i32::try_from(bits).unwrap_or(0);
            let group_size = i32::try_from(group_size).unwrap_or(0);
            let mode = spec
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("affine");
            let validated = QuantSpec::validated("MOSS-Image", mode, bits, group_size)?;
            Some(LinearQuant::from_spec(&validated)?)
        }
    };
    MossTransformer::load(weights, config, quant)
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
mod tests;
