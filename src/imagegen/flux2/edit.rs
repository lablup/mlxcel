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

//! FLUX.2 reference-image editing.
//!
//! A port of `Flux2ImageEdit.edit_array` and its helpers in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/pipeline.py>
//! (`#L313-L545`). Each reference is resized to the output size, encoded by the
//! VAE, cropped to an even grid, patchified, batch-norm normalized and packed,
//! with grid ids at `t = 10 + 10 * index`. Without the K/V cache the reference
//! tokens follow the image tokens in every call; `FLUX.2-klein-9b-kv` instead
//! puts them first on step 0, stores their K/V, and leaves them out afterwards.
//!
//! The reference image is fed to the VAE in f32 (upstream builds it with
//! NumPy), so encoded reference latents are f32 and MLX promotion carries the
//! edit forward pass in f32, as upstream; the stored latents keep the compute
//! dtype because the scheduler step casts back.

use anyhow::{Context, Result, bail};
use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

use super::kv_cache::{Flux2KvCache, KvCacheMode};
use super::pipeline::{Flux2Pipeline, LATENT_CHANNELS, RunOptions, VAE_SCALE_FACTOR};
use super::prompt::PromptEmbeds;
use super::scheduler::Flux2Scheduler;
use super::transformer::{Flux2Transformer, TransformerInput};
use crate::imagegen::Flux2Vae;
use crate::imagegen::latent::{
    pack_latents, patchify_latents, prepare_grid_ids, prepare_packed_latents, unpack_latents,
};

/// Upstream `_round_edit_dimension`: down to a multiple of 16 in `256..=2048`.
#[must_use]
pub fn round_edit_dimension(value: f64) -> u32 {
    let v = (value.max(0.0) as u64 / 16 * 16).clamp(256, 2048);
    v as u32
}

/// Upstream `_resolve_edit_dimensions` without an explicit size: scale the
/// first reference down to at most 1024x1024 pixels of area and round.
#[must_use]
pub fn resolve_edit_dimensions(image_width: u32, image_height: u32) -> (u32, u32) {
    let area = f64::from(image_width) * f64::from(image_height);
    let scale = (1024.0 * 1024.0 / area).sqrt().min(1.0);
    (
        round_edit_dimension(f64::from(image_width) * scale),
        round_edit_dimension(f64::from(image_height) * scale),
    )
}

/// `_load_reference_image` + `_reference_image_array`: RGB, resized with a
/// Lanczos filter when the size differs, `x / 127.5 - 1` in f32, `[1, 3, H, W]`.
pub fn reference_image_array(image: &DynamicImage, width: u32, height: u32) -> UniquePtr<MlxArray> {
    let mut rgb = image.to_rgb8();
    if rgb.width() != width || rgb.height() != height {
        rgb = image::imageops::resize(&rgb, width, height, FilterType::Lanczos3);
    }
    let (w, h) = (width as usize, height as usize);
    let mut chw = vec![0.0_f32; 3 * w * h];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            chw[c * w * h + i] = f32::from(px[c]) / 127.5 - 1.0;
        }
    }
    mlxcel_core::from_slice_f32(&chw, &[1, 3, height as i32, width as i32])
}

/// `_crop_to_even_spatial` on `[B, C, H, W]`.
fn crop_to_even(latents: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(latents);
    mlxcel_core::slice(
        latents,
        &[0, 0, 0, 0],
        &[s[0], s[1], s[2] - s[2] % 2, s[3] - s[3] % 2],
    )
}

/// `_prepare_reference_latents`: packed reference tokens and their ids.
pub fn prepare_reference_latents(
    vae: &Flux2Vae,
    images: &[DynamicImage],
    width: u32,
    height: u32,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let mut packed = Vec::with_capacity(images.len());
    let mut ids = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        let pixels = reference_image_array(image, width, height);
        let latents = crop_to_even(&*vae.encode(&pixels)?);
        let latents = vae.bn_normalize(&*patchify_latents(&latents)?)?;
        packed.push(pack_latents(&latents)?);
        ids.push(prepare_grid_ids(&latents, 10 + 10 * i32::try_from(index)?)?);
    }
    let refs = mlxcel_core::concatenate_many(&packed.iter().map(|a| &**a).collect::<Vec<_>>(), 1);
    let ids = mlxcel_core::concatenate_many(&ids.iter().map(|a| &**a).collect::<Vec<_>>(), 1);
    mlxcel_core::eval(&refs);
    mlxcel_core::eval(&ids);
    Ok((refs, ids))
}

/// Inputs shared by every step of one edit.
struct EditStep<'a> {
    transformer: &'a Flux2Transformer,
    latent_ids: &'a MlxArray,
    reference: &'a MlxArray,
    reference_ids: &'a MlxArray,
}

impl EditStep<'_> {
    /// `_predict_edit`: `[latents, refs]` in, the latent tokens out.
    fn predict(
        &self,
        latents: &MlxArray,
        embeds: &PromptEmbeds,
        t: f32,
    ) -> Result<UniquePtr<MlxArray>> {
        let hidden = mlxcel_core::concatenate(latents, self.reference, 1);
        let ids = mlxcel_core::concatenate(self.latent_ids, self.reference_ids, 1);
        let out = self.transformer.forward(
            &TransformerInput {
                hidden: &hidden,
                encoder: &embeds.embeds,
                timestep: t,
                img_ids: &ids,
                txt_ids: &embeds.text_ids,
                guidance: None,
            },
            None,
        )?;
        let s = mlxcel_core::array_shape(&out);
        let n = mlxcel_core::array_shape(latents)[1];
        Ok(mlxcel_core::slice(&out, &[0, 0, 0], &[s[0], n, s[2]]))
    }

    /// `_predict_edit_with_kv_cache`: extract on the first call (`[refs,
    /// latents]`), then attend to the cached reference K/V.
    fn predict_cached(
        &self,
        latents: &MlxArray,
        embeds: &PromptEmbeds,
        t: f32,
        cache: &mut Flux2KvCache,
    ) -> Result<UniquePtr<MlxArray>> {
        if cache.is_complete() {
            return self.transformer.forward(
                &TransformerInput {
                    hidden: latents,
                    encoder: &embeds.embeds,
                    timestep: t,
                    img_ids: self.latent_ids,
                    txt_ids: &embeds.text_ids,
                    guidance: None,
                },
                Some((cache, KvCacheMode::Cached)),
            );
        }
        let hidden = mlxcel_core::concatenate(self.reference, latents, 1);
        let ids = mlxcel_core::concatenate(self.reference_ids, self.latent_ids, 1);
        let num_ref_tokens = mlxcel_core::array_shape(self.reference)[1];
        self.transformer.forward(
            &TransformerInput {
                hidden: &hidden,
                encoder: &embeds.embeds,
                timestep: t,
                img_ids: &ids,
                txt_ids: &embeds.text_ids,
                guidance: None,
            },
            Some((cache, KvCacheMode::Extract { num_ref_tokens })),
        )
    }
}

impl Flux2Pipeline {
    /// Upstream `edit_array` for one output image: `[1, H, W, 3]` u8.
    #[allow(clippy::too_many_arguments)]
    pub fn edit(
        &self,
        prompt: &str,
        negative: Option<&str>,
        images: &[DynamicImage],
        seed: u64,
        steps: u32,
        width: u32,
        height: u32,
        guidance: f32,
        options: &RunOptions,
    ) -> Result<UniquePtr<MlxArray>> {
        let packed = self.edit_latents(
            prompt, negative, images, seed, steps, width, height, guidance, options,
        )?;
        self.decode(&packed, height, width, options.tiled_vae, false)
    }

    /// The denoising loop of [`Self::edit`]: final patchified latents.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_latents(
        &self,
        prompt: &str,
        negative: Option<&str>,
        images: &[DynamicImage],
        seed: u64,
        steps: u32,
        width: u32,
        height: u32,
        guidance: f32,
        options: &RunOptions,
    ) -> Result<UniquePtr<MlxArray>> {
        if images.is_empty() {
            bail!("At least one reference image is required");
        }
        let (positive, negative) = self.encode_pair(prompt, negative, guidance, options, false)?;
        self.ensure_transformer_and_vae()?;
        let (reference, reference_ids) = {
            let vae = self.vae.borrow();
            let vae = vae.as_ref().context("VAE is not loaded")?;
            prepare_reference_latents(vae, images, width, height)?
        };
        let noise = prepare_packed_latents(
            seed,
            height,
            width,
            1,
            LATENT_CHANNELS,
            VAE_SCALE_FACTOR,
            self.compute_dtype,
        )?;
        let scheduler = Flux2Scheduler::new(((height / 16) * (width / 16)) as i32, steps)?;
        let transformer = self.transformer.borrow();
        let transformer = transformer.as_ref().context("transformer is not loaded")?;
        let step = EditStep {
            transformer,
            latent_ids: &noise.latent_ids,
            reference: &reference,
            reference_ids: &reference_ids,
        };
        let layers = (
            transformer.config().num_layers,
            transformer.config().num_single_layers,
        );
        let use_cache = self.variant.uses_reference_kv_cache();
        let mut cache = Flux2KvCache::new(layers.0, layers.1);
        let mut negative_cache = Flux2KvCache::new(layers.0, layers.1);

        let mut latents = noise.latents;
        for (i, &t) in scheduler.timesteps().iter().enumerate() {
            let mut pred = if use_cache {
                step.predict_cached(&latents, &positive, t, &mut cache)?
            } else {
                step.predict(&latents, &positive, t)?
            };
            if let Some(negative) = &negative {
                let neg = if use_cache {
                    step.predict_cached(&latents, negative, t, &mut negative_cache)?
                } else {
                    step.predict(&latents, negative, t)?
                };
                pred = Self::cfg(&pred, &neg, guidance);
            }
            if i == 0 {
                for a in cache.arrays().into_iter().chain(negative_cache.arrays()) {
                    mlxcel_core::eval(a);
                }
            }
            latents = scheduler.step(&pred, i, &latents)?;
            mlxcel_core::eval(&latents);
        }
        cache.clear();
        negative_cache.clear();
        unpack_latents(&latents, noise.latent_height, noise.latent_width)
    }
}
