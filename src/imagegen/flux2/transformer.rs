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

//! The FLUX.2 transformer.
//!
//! A port of `Flux2Transformer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/transformer/transformer.py>,
//! including the per-token modulation blend (`_blend_*_mod_params`) the
//! `9b-kv` reference-cache extraction uses, and the key renames of upstream
//! `weights.py::load_transformer`. Weight names are the Diffusers ones.
//!
//! Used by: Flux2, Bonsai.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::blocks::{DoubleStreamBlock, SingleStreamBlock};
use super::config::Flux2TransformerConfig;
use super::embeddings::{
    AdaLayerNormContinuous, Flux2Linear, LinearSource, ModParams, Modulation,
    TimestepGuidanceEmbeddings, pos_embed,
};
use super::kv_cache::{Flux2KvCache, KvCacheMode};

/// Upstream `load_transformer` key renames: the key
/// `prepare_transformer_weights` stores a checkpoint tensor under.
///
/// Used by: `prepare_transformer_weights`, Bonsai's packed-module list.
pub(crate) fn rename_key(key: &str) -> String {
    key.replace(
        "time_guidance_embed.timestep_embedder.",
        "time_guidance_embed.",
    )
    .replace(".to_out.0.", ".to_out.")
}

/// Rename keys and cast floating tensors to `compute_dtype` (upstream
/// `_cast_float`); packed quantized planes are left alone.
#[must_use]
pub fn prepare_transformer_weights(raw: WeightMap, compute_dtype: i32) -> WeightMap {
    raw.into_iter()
        .map(|(key, value)| {
            let value = match mlxcel_core::array_dtype(&value) {
                dtype::FLOAT16 | dtype::BFLOAT16 | dtype::FLOAT32 => {
                    mlxcel_core::astype(&value, compute_dtype)
                }
                _ => value,
            };
            (rename_key(&key), value)
        })
        .collect()
}

/// One transformer call's inputs.
pub struct TransformerInput<'a> {
    /// `[B, S_img, in_channels]` packed latents (plus reference tokens).
    pub hidden: &'a MlxArray,
    /// `[B, S_txt, joint_attention_dim]` prompt embeddings.
    pub encoder: &'a MlxArray,
    /// The scheduler timestep (`sigma * 1000`).
    pub timestep: f32,
    /// `[B, S_img, 4]` or `[S_img, 4]` image ids; batch 0 is used.
    pub img_ids: &'a MlxArray,
    /// `[B, S_txt, 4]` or `[S_txt, 4]` text ids; batch 0 is used.
    pub txt_ids: &'a MlxArray,
    pub guidance: Option<f32>,
}

/// `Flux2Transformer`.
pub struct Flux2Transformer {
    config: Flux2TransformerConfig,
    compute_dtype: i32,
    time_guidance_embed: TimestepGuidanceEmbeddings,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    x_embedder: Box<dyn Flux2Linear>,
    context_embedder: Box<dyn Flux2Linear>,
    blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Box<dyn Flux2Linear>,
}

/// `_expand_mod_param`: `[B, 1, D]` broadcast to `[B, len, D]`.
fn expand_param(param: &MlxArray, len: i32) -> UniquePtr<MlxArray> {
    let param = if mlxcel_core::array_shape(param).len() == 2 {
        mlxcel_core::expand_dims(param, 1)
    } else {
        mlxcel_core::copy(param)
    };
    let shape = mlxcel_core::array_shape(&param);
    if shape[1] == len {
        return param;
    }
    mlxcel_core::broadcast_to(&param, &[shape[0], len, shape[2]])
}

fn seq_slice(x: &MlxArray, start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    mlxcel_core::slice(x, &[0, start, 0], &[s[0], stop, s[2]])
}

/// `_blend_mod_param_set`: reference tokens take `reference`'s modulation,
/// the rest keep `image`'s. `image` covers `[ref, img]` of length `len`.
fn blend_set(image: &ModParams, reference: &ModParams, num_ref: i32, len: i32) -> ModParams {
    let blend = |i: usize| {
        let img = expand_param(&image[i], len);
        let r = expand_param(&reference[i], num_ref);
        mlxcel_core::concatenate(&r, &seq_slice(&img, num_ref, len), 1)
    };
    [blend(0), blend(1), blend(2)]
}

/// `_blend_single_mod_params`: `[txt, ref, img]` order.
fn blend_single(
    image: &ModParams,
    reference: &ModParams,
    num_txt: i32,
    num_ref: i32,
    len: i32,
) -> ModParams {
    let blend = |i: usize| {
        let img = expand_param(&image[i], len);
        let r = expand_param(&reference[i], num_ref);
        mlxcel_core::concatenate_many(
            &[
                &seq_slice(&img, 0, num_txt),
                &r,
                &seq_slice(&img, num_txt + num_ref, len),
            ],
            1,
        )
    };
    [blend(0), blend(1), blend(2)]
}

/// `ids[0]` for a batched `[B, S, 4]` id tensor.
fn first_batch(ids: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(ids);
    if s.len() == 3 {
        let x = mlxcel_core::slice(ids, &[0, 0, 0], &[1, s[1], s[2]]);
        mlxcel_core::reshape(&x, &[s[1], s[2]])
    } else {
        mlxcel_core::copy(ids)
    }
}

impl Flux2Transformer {
    /// Build from prepared weights (see [`prepare_transformer_weights`]).
    pub fn from_weights(
        weights: &WeightMap,
        config: Flux2TransformerConfig,
        source: &dyn LinearSource,
        compute_dtype: i32,
    ) -> Result<Self> {
        config.validate()?;
        let heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let blocks = (0..config.num_layers)
            .map(|i| {
                DoubleStreamBlock::load(
                    weights,
                    source,
                    &format!("transformer_blocks.{i}"),
                    heads,
                    head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..config.num_single_layers)
            .map(|i| {
                SingleStreamBlock::load(
                    weights,
                    source,
                    &format!("single_transformer_blocks.{i}"),
                    heads,
                    head_dim,
                    config.mlp_hidden_dim(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_guidance_embed: TimestepGuidanceEmbeddings::load(
                weights,
                source,
                config.timestep_guidance_channels,
                config.guidance_embeds,
            )?,
            mod_img: Modulation::load(weights, source, "double_stream_modulation_img", 2)?,
            mod_txt: Modulation::load(weights, source, "double_stream_modulation_txt", 2)?,
            mod_single: Modulation::load(weights, source, "single_stream_modulation", 1)?,
            x_embedder: source.load(weights, "x_embedder")?,
            context_embedder: source.load(weights, "context_embedder")?,
            blocks,
            single_blocks,
            norm_out: AdaLayerNormContinuous::load(
                weights,
                source,
                "norm_out",
                config.inner_dim(),
            )?,
            proj_out: source.load(weights, "proj_out")?,
            config,
            compute_dtype,
        })
    }

    /// Load `<transformer_dir>` (single file or sharded) with `config`.
    pub fn load(
        transformer_dir: &Path,
        config: Flux2TransformerConfig,
        source: &dyn LinearSource,
        compute_dtype: i32,
    ) -> Result<Self> {
        let raw = mlxcel_core::weights::load_weights_from_dir(transformer_dir)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "cannot load FLUX.2 transformer weights from {}",
                    transformer_dir.display()
                )
            })?;
        source.check_weights(&raw)?;
        let weights = prepare_transformer_weights(raw, compute_dtype);
        Self::from_weights(&weights, config, source, compute_dtype)
    }

    #[must_use]
    pub fn config(&self) -> &Flux2TransformerConfig {
        &self.config
    }

    /// `[B]` timestep in the hidden dtype, scaled by 1000 when it is a sigma.
    ///
    /// Upstream tests `max(t) <= 1` after the cast to the hidden dtype. Every
    /// element holds `value`, and rounding is monotone with 1.0 representable,
    /// so the host answers that without a device read except just above 1.0,
    /// where a bf16 or f16 cast can round down to 1.0.
    fn timestep_array(value: f32, batch: i32, dt: i32) -> UniquePtr<MlxArray> {
        let t = mlxcel_core::astype(&mlxcel_core::full_f32(&[batch], value, dtype::FLOAT32), dt);
        let is_sigma = if value <= 1.0 {
            true
        } else if value > 1.01 || value.is_nan() {
            false
        } else {
            mlxcel_core::utils::array_to_vec_f32(&t)
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max)
                <= 1.0
        };
        let scale = if is_sigma { 1000.0 } else { 1.0 };
        mlxcel_core::multiply(&t, &mlxcel_core::full_f32(&[1], scale, dt))
    }

    /// One denoising prediction, `[B, S_out, out_channels]`.
    ///
    /// With `kv = Some((cache, Extract { num_ref_tokens }))` the first
    /// `num_ref_tokens` image tokens are reference tokens: their K/V are stored
    /// and they are dropped from the output. With `Cached`, the stored K/V join
    /// every attention.
    pub fn forward(
        &self,
        input: &TransformerInput<'_>,
        mut kv: Option<(&mut Flux2KvCache, KvCacheMode)>,
    ) -> Result<UniquePtr<MlxArray>> {
        let hidden_dtype = mlxcel_core::array_dtype(input.hidden);
        let hs = mlxcel_core::array_shape(input.hidden);
        let (batch, num_img) = (hs[0], hs[1]);
        let num_txt = mlxcel_core::array_shape(input.encoder)[1];

        let timestep = Self::timestep_array(input.timestep, batch, hidden_dtype);
        let guidance = input
            .guidance
            .map(|g| Self::timestep_array(g, batch, hidden_dtype));
        let temb = mlxcel_core::astype(
            &self
                .time_guidance_embed
                .forward(&timestep, guidance.as_deref()),
            self.compute_dtype,
        );

        let mode = kv.as_ref().map(|(_, m)| *m);
        let num_ref = match mode {
            Some(KvCacheMode::Extract { num_ref_tokens }) => {
                if num_ref_tokens < 1 || num_ref_tokens >= num_img {
                    bail!("num_ref_tokens {num_ref_tokens} must be in 1..{num_img}");
                }
                if let Some((cache, _)) = kv.as_mut() {
                    cache.num_ref_tokens = num_ref_tokens;
                }
                num_ref_tokens
            }
            _ => 0,
        };
        let ref_temb = (num_ref > 0).then(|| {
            // ref_fixed_timestep 0.0, times 1000, in the timestep dtype.
            let zero = mlxcel_core::zeros(&[batch], hidden_dtype);
            mlxcel_core::astype(
                &self.time_guidance_embed.forward(&zero, guidance.as_deref()),
                self.compute_dtype,
            )
        });

        let mut hidden = self.x_embedder.forward(input.hidden);
        let mut encoder = self.context_embedder.forward(input.encoder);

        let theta = self.config.rope_theta;
        let axes = &self.config.axes_dims_rope;
        let (img_cos, img_sin) = pos_embed(&first_batch(input.img_ids), theta, axes);
        let (txt_cos, txt_sin) = pos_embed(&first_batch(input.txt_ids), theta, axes);
        let cos = mlxcel_core::concatenate(&txt_cos, &img_cos, 0);
        let sin = mlxcel_core::concatenate(&txt_sin, &img_sin, 0);
        let rope = (&*cos, &*sin);

        let mut mod_img = self.mod_img.forward(&temb);
        let mod_txt = self.mod_txt.forward(&temb);
        if let Some(ref_temb) = &ref_temb {
            let ref_img = self.mod_img.forward(ref_temb);
            mod_img = mod_img
                .iter()
                .zip(&ref_img)
                .map(|(i, r)| blend_set(i, r, num_ref, num_img))
                .collect();
        }

        for (index, block) in self.blocks.iter().enumerate() {
            let cache = kv.as_mut().map(|(c, _)| &mut c.double[index]);
            let (e, h) = block.forward(&hidden, &encoder, &mod_img, &mod_txt, rope, mode, cache)?;
            encoder = e;
            hidden = h;
        }

        let mut hidden = mlxcel_core::concatenate(&encoder, &hidden, 1);
        let total = mlxcel_core::array_shape(&hidden)[1];
        let mut mod_single = self
            .mod_single
            .forward(&temb)
            .into_iter()
            .next()
            .context("single-stream modulation produced no parameter set")?;
        if let Some(ref_temb) = &ref_temb {
            let reference = self
                .mod_single
                .forward(ref_temb)
                .into_iter()
                .next()
                .context("single-stream modulation produced no parameter set")?;
            mod_single = blend_single(&mod_single, &reference, num_txt, num_ref, total);
        }
        for (index, block) in self.single_blocks.iter().enumerate() {
            let cache = kv.as_mut().map(|(c, _)| &mut c.single[index]);
            hidden = block.forward(
                &hidden,
                &mod_single,
                rope,
                num_txt,
                self.compute_dtype,
                mode,
                cache,
            )?;
        }

        let start = num_txt + num_ref;
        let hidden = seq_slice(&hidden, start, total);
        let hidden = self.norm_out.forward(&hidden, &temb, self.compute_dtype);
        Ok(self.proj_out.forward(&hidden))
    }
}
