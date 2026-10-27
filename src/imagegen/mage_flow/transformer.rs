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

//! The Mage-Flow NR-MMDiT transformer.
//!
//! A port of `transformer.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mage_flow>
//! (mlx-vlm `73d366ec`): double-stream blocks with joint text-image attention,
//! multi-scale 2-D RoPE (`msrope`) over the image tokens only, and a
//! `qwen_proj` timestep embedding.
//!
//! Precision follows upstream exactly: everything runs in the checkpoint's
//! bf16 except the explicit f32 islands upstream writes (the timestep
//! frequencies and arguments, and the rotation, both cast back), so no dtype
//! promotion question arises on the CUDA build.
//!
//! The RoPE tables come from the shared [`MRoPE`] and are applied with the
//! shared [`apply_rotary`] (adjacent pairs, the same layout upstream uses).
//! Attention runs through [`mlxcel_core::layers::attention`], which
//! query-chunks the score matrix above `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`.

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::RMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::TransformerConfig;
use super::layers::{Dense, LayerNorm, LinearQuant, modulate, rms_norm, seq_axis, split_last};
use crate::imagegen::dit_blocks::{MRoPE, apply_rotary};

/// Epsilon of every norm in the transformer.
const EPS: f32 = 1e-6;
/// Width of the sinusoidal timestep embedding.
const TIME_FREQ_DIM: i32 = 256;

/// `(frames, height, width)` of one latent grid in the packed image stream.
pub type ImageShape = (i32, i32, i32);

/// Per-token `(frame, height, width)` positions of `image_shapes`, as
/// upstream's `image_rope_frequencies` lays them out: grid `i` sits at frame
/// `i`, and each spatial axis is centered (`y - (H - H // 2)`).
#[must_use]
pub fn rope_positions(image_shapes: &[ImageShape]) -> Vec<[f32; 3]> {
    let mut rows = Vec::new();
    for (index, &(frames, height, width)) in image_shapes.iter().enumerate() {
        let index = index as i32;
        for f in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    rows.push([
                        (index + f) as f32,
                        (y - (height - height / 2)) as f32,
                        (x - (width - width / 2)) as f32,
                    ]);
                }
            }
        }
    }
    rows
}

/// `(cos, sin)` of the image tokens, each `[1, L, head_dim]` f32 with every
/// frequency repeated twice (the shared [`apply_rotary`] layout).
pub fn rope_tables(
    image_shapes: &[ImageShape],
    axes_dim: [i32; 3],
    theta: f32,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let rows = rope_positions(image_shapes);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let ids = mlxcel_core::from_slice_f32(&flat, &[1, rows.len() as i32, 3]);
    MRoPE {
        sections: axes_dim.to_vec(),
        theta,
    }
    .compute_freqs(&ids)
}

/// Upstream `timestep_embedding(t, 256, scale=1000)`: bf16-rounded
/// frequencies (`exp(...).astype(dtype).astype(f32)`), f32 arguments
/// `1000 * t * f`, `[cos, sin]`, cast to `dtype`.
#[must_use]
pub fn timestep_embedding(t: &MlxArray, dt: i32) -> UniquePtr<MlxArray> {
    let half = TIME_FREQ_DIM / 2;
    let f32_scalar = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
    let exponent = mlxcel_core::divide(
        &mlxcel_core::multiply(
            &f32_scalar(-(10000.0_f64.ln()) as f32),
            &mlxcel_core::arange_f32(0.0, half as f32, 1.0),
        ),
        &f32_scalar(half as f32),
    );
    let freqs = mlxcel_core::astype(
        &mlxcel_core::astype(&mlxcel_core::exp(&exponent), dt),
        dtype::FLOAT32,
    );
    let t = mlxcel_core::astype(&mlxcel_core::reshape(t, &[-1, 1]), dtype::FLOAT32);
    let args = mlxcel_core::multiply(&t, &mlxcel_core::reshape(&freqs, &[1, -1]));
    let args = mlxcel_core::multiply(&f32_scalar(1000.0), &args);
    let emb = mlxcel_core::concatenate(&mlxcel_core::cos(&args), &mlxcel_core::sin(&args), -1);
    mlxcel_core::astype(&emb, dt)
}

/// `JointAttention`: per-stream q/k/v with head RMS norms, RoPE on the image
/// stream, one SDPA over `[text, image]`.
struct JointAttention {
    to_q: Dense,
    to_k: Dense,
    to_v: Dense,
    add_q: Dense,
    add_k: Dense,
    add_v: Dense,
    norm_q: RMSNorm,
    norm_k: RMSNorm,
    norm_added_q: RMSNorm,
    norm_added_k: RMSNorm,
    to_out: Dense,
    to_add_out: Dense,
    heads: i32,
    head_dim: i32,
}

impl JointAttention {
    fn load(
        w: &WeightMap,
        p: &str,
        heads: i32,
        head_dim: i32,
        q: Option<LinearQuant>,
    ) -> Result<Self> {
        let dense = |name: &str| Dense::load(w, &format!("{p}.{name}"), q);
        let norm = |name: &str| rms_norm(w, &format!("{p}.{name}.weight"), EPS);
        Ok(Self {
            to_q: dense("to_q")?,
            to_k: dense("to_k")?,
            to_v: dense("to_v")?,
            add_q: dense("add_q_proj")?,
            add_k: dense("add_k_proj")?,
            add_v: dense("add_v_proj")?,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            norm_added_q: norm("norm_added_q")?,
            norm_added_k: norm("norm_added_k")?,
            to_out: dense("to_out")?,
            to_add_out: dense("to_add_out")?,
            heads,
            head_dim,
        })
    }

    fn heads(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        mlxcel_core::reshape(x, &[s[0], s[1], self.heads, self.head_dim])
    }

    fn forward(
        &self,
        image: &MlxArray,
        text: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let dt = mlxcel_core::array_dtype(image);
        let img_q = self.norm_q.forward(&self.heads(&self.to_q.forward(image)));
        let img_k = self.norm_k.forward(&self.heads(&self.to_k.forward(image)));
        let img_v = self.heads(&self.to_v.forward(image));
        let txt_q = self
            .norm_added_q
            .forward(&self.heads(&self.add_q.forward(text)));
        let txt_k = self
            .norm_added_k
            .forward(&self.heads(&self.add_k.forward(text)));
        let txt_v = self.heads(&self.add_v.forward(text));
        // Upstream rotates in f32 and casts back to the activation dtype.
        let img_q = mlxcel_core::astype(&*apply_rotary(&img_q, cos, sin)?, dt);
        let img_k = mlxcel_core::astype(&*apply_rotary(&img_k, cos, sin)?, dt);

        let to_heads = |a: &MlxArray, b: &MlxArray| {
            mlxcel_core::transpose_axes(&mlxcel_core::concatenate(a, b, 1), &[0, 2, 1, 3])
        };
        let (q, k, v) = (
            to_heads(&txt_q, &img_q),
            to_heads(&txt_k, &img_k),
            to_heads(&txt_v, &img_v),
        );
        let out = mlxcel_core::layers::attention(
            &q,
            &k,
            &v,
            (self.head_dim as f32).powf(-0.5),
            None,
            0.0,
            0,
        );
        let (b, t, l) = (
            mlxcel_core::array_shape(image)[0],
            mlxcel_core::array_shape(text)[1],
            mlxcel_core::array_shape(image)[1],
        );
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[b, t + l, -1],
        );
        let width = mlxcel_core::array_shape(&out)[2];
        let txt = mlxcel_core::slice(&out, &[0, 0, 0], &[b, t, width]);
        let img = mlxcel_core::slice(&out, &[0, t, 0], &[b, t + l, width]);
        Ok((self.to_out.forward(&img), self.to_add_out.forward(&txt)))
    }
}

/// `FeedForward`: `linear_out(gelu_approx(linear_in(x)))` (tanh GELU).
struct FeedForward {
    linear_in: Dense,
    linear_out: Dense,
}

impl FeedForward {
    fn load(w: &WeightMap, p: &str, q: Option<LinearQuant>) -> Result<Self> {
        Ok(Self {
            linear_in: Dense::load(w, &format!("{p}.linear_in"), q)?,
            linear_out: Dense::load(w, &format!("{p}.linear_out"), q)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear_out
            .forward(&mlxcel_core::gelu_tanh(&self.linear_in.forward(x)))
    }
}

/// `x * (1 + scale) + shift` and `gate` of one `[shift, scale, gate]` chunk.
fn modulated(x: &MlxArray, params: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let [shift, scale, gate]: [UniquePtr<MlxArray>; 3] = split_last(params, 3)
        .try_into()
        .unwrap_or_else(|_| unreachable!("split into 3"));
    (
        modulate(x, &seq_axis(&shift), &seq_axis(&scale)),
        seq_axis(&gate),
    )
}

/// `x + gate * y`.
fn gated(x: &MlxArray, gate: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(x, &mlxcel_core::multiply(gate, y))
}

/// `MageFlowTransformerBlock`.
struct Block {
    img_mod: Dense,
    txt_mod: Dense,
    attn: JointAttention,
    img_mlp: FeedForward,
    txt_mlp: FeedForward,
    norm: LayerNorm,
}

impl Block {
    fn load(
        w: &WeightMap,
        p: &str,
        cfg: &TransformerConfig,
        q: Option<LinearQuant>,
    ) -> Result<Self> {
        Ok(Self {
            img_mod: Dense::load(w, &format!("{p}.img_mod.linear"), q)?,
            txt_mod: Dense::load(w, &format!("{p}.txt_mod.linear"), q)?,
            attn: JointAttention::load(w, &format!("{p}.attn"), cfg.num_heads, cfg.head_dim(), q)?,
            img_mlp: FeedForward::load(w, &format!("{p}.img_mlp"), q)?,
            txt_mlp: FeedForward::load(w, &format!("{p}.txt_mlp"), q)?,
            norm: LayerNorm::plain(EPS),
        })
    }

    fn forward(
        &self,
        image: &MlxArray,
        text: &MlxArray,
        temb: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let [img_mod1, img_mod2]: [UniquePtr<MlxArray>; 2] =
            split_last(&self.img_mod.forward(&mlxcel_core::silu(temb)), 2)
                .try_into()
                .unwrap_or_else(|_| unreachable!("split into 2"));
        let [txt_mod1, txt_mod2]: [UniquePtr<MlxArray>; 2] =
            split_last(&self.txt_mod.forward(&mlxcel_core::silu(temb)), 2)
                .try_into()
                .unwrap_or_else(|_| unreachable!("split into 2"));
        let (img_norm, img_gate1) = modulated(&self.norm.forward(image), &img_mod1);
        let (txt_norm, txt_gate1) = modulated(&self.norm.forward(text), &txt_mod1);
        let (img_attn, txt_attn) = self.attn.forward(&img_norm, &txt_norm, cos, sin)?;
        let image = gated(image, &img_gate1, &img_attn);
        let text = gated(text, &txt_gate1, &txt_attn);
        let (img_norm, img_gate2) = modulated(&self.norm.forward(&image), &img_mod2);
        let (txt_norm, txt_gate2) = modulated(&self.norm.forward(&text), &txt_mod2);
        let image = gated(&image, &img_gate2, &self.img_mlp.forward(&img_norm));
        let text = gated(&text, &txt_gate2, &self.txt_mlp.forward(&txt_norm));
        Ok((image, text))
    }
}

/// `MageFlowTransformer`, loaded from upstream's sanitized key names
/// (see [`super::weights::sanitize_transformer_weights`]).
pub struct MageFlowTransformer {
    config: TransformerConfig,
    img_in: Dense,
    txt_norm: RMSNorm,
    txt_in: Dense,
    time_linear_1: Dense,
    time_linear_2: Dense,
    blocks: Vec<Block>,
    norm_out_linear: Dense,
    norm_out: LayerNorm,
    proj_out: Dense,
}

impl MageFlowTransformer {
    /// Build from sanitized weights. `quant` applies to every projection that
    /// carries `.scales`, as upstream's `nn.quantize` predicate does; a
    /// `.scales` without `quant` is an error.
    pub fn from_weights(
        config: TransformerConfig,
        w: &WeightMap,
        quant: Option<LinearQuant>,
    ) -> Result<Self> {
        let dense = |name: &str| Dense::load(w, name, quant);
        let blocks = (0..config.depth)
            .map(|i| Block::load(w, &format!("transformer_blocks.{i}"), &config, quant))
            .collect::<Result<Vec<_>>>()?;
        if w.contains_key(&format!(
            "transformer_blocks.{}.img_mod.linear.weight",
            config.depth
        )) {
            bail!(
                "the Mage-Flow transformer checkpoint has more blocks than depth {}",
                config.depth
            );
        }
        Ok(Self {
            img_in: dense("img_in")?,
            txt_norm: rms_norm(w, "txt_norm.weight", EPS)?,
            txt_in: dense("txt_in")?,
            time_linear_1: dense("time_text_embed.timestep_embedder.linear_1")?,
            time_linear_2: dense("time_text_embed.timestep_embedder.linear_2")?,
            blocks,
            norm_out_linear: dense("norm_out.linear")?,
            norm_out: LayerNorm::plain(EPS),
            proj_out: dense("proj_out")?,
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Velocity of the packed image tokens `img` (`[B, L, in_channels]`) given
    /// the conditioner states `txt` (`[B, T, context_in_dim]`) at `sigma`.
    /// `image_shapes` lays out `img`: the target grid first, then each
    /// reference grid of an edit.
    pub fn forward(
        &self,
        img: &MlxArray,
        txt: &MlxArray,
        sigma: f32,
        image_shapes: &[ImageShape],
    ) -> Result<UniquePtr<MlxArray>> {
        let (img_shape, txt_shape) = (mlxcel_core::array_shape(img), mlxcel_core::array_shape(txt));
        if img_shape.len() != 3 || txt_shape.len() != 3 {
            bail!("img and txt must both have shape [B, L, D]");
        }
        let tokens: i32 = image_shapes.iter().map(|(f, h, w)| f * h * w).sum();
        if tokens != img_shape[1] {
            bail!(
                "image_shapes cover {tokens} tokens but img has {}",
                img_shape[1]
            );
        }
        let (cos, sin) = rope_tables(image_shapes, self.config.axes_dim, self.config.theta)
            .context("cannot build the Mage-Flow RoPE tables")?;
        let mut image = self.img_in.forward(img);
        let mut text = self.txt_in.forward(&self.txt_norm.forward(txt));
        let dt = mlxcel_core::array_dtype(&image);
        // `mx.full((B,), sigma, dtype=latents.dtype)`, then `.astype(image.dtype)`.
        let t = mlxcel_core::full_f32(&[img_shape[0]], sigma, mlxcel_core::array_dtype(img));
        let t = mlxcel_core::astype(&t, dt);
        let temb = self.time_linear_2.forward(&mlxcel_core::silu(
            &self.time_linear_1.forward(&timestep_embedding(&t, dt)),
        ));
        for block in &self.blocks {
            (image, text) = block.forward(&image, &text, &temb, &cos, &sin)?;
        }
        // `AdaptiveLayerNorm` names its chunks scale, then shift.
        let [scale, shift]: [UniquePtr<MlxArray>; 2] =
            split_last(&self.norm_out_linear.forward(&mlxcel_core::silu(&temb)), 2)
                .try_into()
                .unwrap_or_else(|_| unreachable!("split into 2"));
        let out = modulate(
            &self.norm_out.forward(&image),
            &seq_axis(&shift),
            &seq_axis(&scale),
        );
        Ok(self.proj_out.forward(&out))
    }
}

#[cfg(test)]
#[path = "transformer_tests.rs"]
pub(crate) mod tests;
