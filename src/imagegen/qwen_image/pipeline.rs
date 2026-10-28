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

//! The Qwen-Image denoising loop, a port of `QwenImagePipeline._sample` and
//! the reference handling of `edit_array` in mlx-vlm `73d366ec`
//! `pipeline.py`.
//!
//! The noise is `mx.random.seed(seed); mx.random.normal((1, 1, 64, h, w))`
//! in f32, cast to bf16 and packed to `[1, h*w, 64]`; the transformer gets
//! `bf16(timestep) / 1000` in bf16 and the dynamic-shift Euler step
//! ([`FlowMatchEulerDiscrete::dynamic`]) accumulates in f32. An edit puts
//! the reference latents in front of the target, and with the KV cache only
//! the first step runs them through the transformer.

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, RgbaImage};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::image_dimensions;
use super::kv_cache::{ImageShape, KvCache, KvMode};
use super::reference::resize_lanczos_rgba;
use super::transformer::{EditLayout, QwenImageTransformer};
use super::vae::QwenImageVae;
use crate::imagegen::mage_flow::layers::scalar_like;
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;

fn eval(x: &MlxArray) -> Result<()> {
    mlxcel_core::try_eval(x).map_err(|e| anyhow::anyhow!("Qwen-Image: {e}"))
}

/// The initial target latents `[1, grid_h * grid_w, z_dim]`, bf16.
#[must_use]
pub fn initial_noise(grid_h: i32, grid_w: i32, z_dim: i32, seed: u64) -> UniquePtr<MlxArray> {
    mlxcel_core::random_seed(seed);
    // SAFETY: a null key draws from the global generator just seeded.
    let noise = unsafe {
        mlxcel_core::random_normal(
            &[1, 1, z_dim, grid_h, grid_w],
            dtype::FLOAT32,
            std::ptr::null(),
        )
    };
    let noise = mlxcel_core::astype(&noise, dtype::BFLOAT16);
    let noise = mlxcel_core::reshape(&noise, &[1, z_dim, grid_h * grid_w]);
    mlxcel_core::transpose_axes(&noise, &[0, 2, 1])
}

/// One edit reference, resized as upstream resizes it.
pub struct Reference {
    /// The `RGBA` image after `resize(_image_dimensions(...), LANCZOS)`.
    pub image: RgbaImage,
}

/// Upstream's reference preparation: `convert("RGBA")`, then Pillow
/// Lanczos to `_image_dimensions(output_resolution, width / height)`.
pub fn resize_references(
    images: &[DynamicImage],
    output_resolution: u32,
) -> Result<Vec<Reference>> {
    ensure!(
        !images.is_empty(),
        "At least one reference image is required"
    );
    images
        .iter()
        .map(|image| {
            let rgba = image.to_rgba8();
            let (w, h) = rgba.dimensions();
            let (tw, th) = image_dimensions(output_resolution, f64::from(w) / f64::from(h));
            Ok(Reference {
                image: resize_lanczos_rgba(&rgba, tw, th)?,
            })
        })
        .collect()
}

/// The reference latents `[1, sum(h*w), z_dim]` bf16 and their `(1, h, w)`
/// grids: `vae.encode` of `pixels / 127.5 - 1` (bf16), normalized.
pub fn reference_latents(
    vae: &QwenImageVae,
    references: &[Reference],
) -> Result<(UniquePtr<MlxArray>, Vec<ImageShape>)> {
    let z = vae.config().z_dim;
    let mut latents = Vec::with_capacity(references.len());
    let mut shapes = Vec::with_capacity(references.len());
    for reference in references {
        let (w, h) = reference.image.dimensions();
        let values: Vec<f32> = reference
            .image
            .as_raw()
            .iter()
            .map(|&p| f32::from(p) / 127.5 - 1.0)
            .collect();
        let pixels = mlxcel_core::from_slice_f32(&values, &[1, h as i32, w as i32, 4]);
        let pixels = mlxcel_core::astype(&pixels, dtype::BFLOAT16);
        let normalized = vae.encode_normalized(&pixels)?;
        eval(&normalized)?;
        let s = mlxcel_core::array_shape(&normalized);
        shapes.push((1, s[2], s[3]));
        let packed = mlxcel_core::reshape(&normalized, &[1, z, s[2] * s[3]]);
        latents.push(mlxcel_core::transpose_axes(&packed, &[0, 2, 1]));
    }
    let refs: Vec<&MlxArray> = latents.iter().map(|a| &**a).collect();
    let joined = mlxcel_core::astype(&mlxcel_core::concatenate_many(&refs, 1), dtype::BFLOAT16);
    eval(&joined)?;
    Ok((joined, shapes))
}

/// One conditioning branch: the prompt states and, for an edit, which of
/// its tokens are `<|image_pad|>`.
pub struct Branch<'a> {
    pub hidden: &'a MlxArray,
    pub image_pad_mask: Option<&'a [bool]>,
}

/// The edit-only inputs of the loop.
pub struct EditInputs<'a> {
    pub latents: &'a MlxArray,
    pub shapes: &'a [ImageShape],
    pub use_kv_cache: bool,
}

/// Settings of one image.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    pub grid_h: i32,
    pub grid_w: i32,
}

fn edit_layout<'a>(edit: Option<&EditInputs<'a>>, branch: &Branch<'a>) -> Option<EditLayout<'a>> {
    edit.map(|e| EditLayout {
        reference_shapes: e.shapes,
        image_pad_mask: branch.image_pad_mask.unwrap_or(&[]),
    })
}

/// Upstream `_sample` up to the final latents `[1, h*w, z_dim]` bf16.
/// `negative` runs CFG (`neg + g * (pred - neg)`).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &QwenImageTransformer,
    schedule_params: crate::imagegen::scheduler::DynamicShift,
    z_dim: i32,
    positive: &Branch<'_>,
    negative: Option<&Branch<'_>>,
    edit: Option<&EditInputs<'_>>,
    sampling: Sampling,
) -> Result<UniquePtr<MlxArray>> {
    // Upstream's bf16 noise, conditioning and reference latents, in the
    // transformer's dtype (a no-op for the bf16 default; bf16 to f16 is
    // exact for the noise).
    let dt = transformer.dtype();
    let cast = |a: &MlxArray| mlxcel_core::astype(a, dt);
    let (pos_hidden, neg_hidden) = (cast(positive.hidden), negative.map(|n| cast(n.hidden)));
    let ref_latents = edit.map(|e| cast(e.latents));
    let tokens = sampling.grid_h * sampling.grid_w;
    let schedule = FlowMatchEulerDiscrete::dynamic(
        u32::try_from(tokens).context("latent grid too large")?,
        sampling.steps,
        schedule_params,
    )?;
    let img_shape = (1, sampling.grid_h, sampling.grid_w);
    let mut latents = cast(&initial_noise(
        sampling.grid_h,
        sampling.grid_w,
        z_dim,
        sampling.seed,
    ));
    let cached = edit.is_some_and(|e| e.use_kv_cache)
        && sampling.steps > 1
        && transformer.config().causal_condition;
    let layers = transformer.num_layers();
    let mut cache = cached.then(|| KvCache::new(layers));
    let mut negative_cache = (cached && negative.is_some()).then(|| KvCache::new(layers));
    let (pos_layout, neg_layout) = (
        edit_layout(edit, positive),
        negative.and_then(|n| edit_layout(edit, n)),
    );
    for index in 0..schedule.num_steps() {
        let t = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&[schedule.timesteps()[index]], &[1]),
            mlxcel_core::array_dtype(&latents),
        );
        let t = mlxcel_core::divide(&t, &scalar_like(1000.0, &t));
        let mode = if index == 0 {
            KvMode::Extract
        } else {
            KvMode::Cached
        };
        let model_input = match &ref_latents {
            Some(r) if cache.is_none() || index == 0 => mlxcel_core::concatenate(r, &latents, 1),
            _ => mlxcel_core::copy(&latents),
        };
        let mut pred = transformer.forward(
            &model_input,
            &pos_hidden,
            &t,
            img_shape,
            pos_layout.as_ref(),
            cache.as_mut().map(|c| (c, mode)),
        )?;
        if let Some(neg_hidden) = &neg_hidden {
            let neg_pred = transformer.forward(
                &model_input,
                neg_hidden,
                &t,
                img_shape,
                neg_layout.as_ref(),
                negative_cache.as_mut().map(|c| (c, mode)),
            )?;
            let delta = mlxcel_core::subtract(&pred, &neg_pred);
            pred = mlxcel_core::add(
                &neg_pred,
                &mlxcel_core::multiply(&scalar_like(sampling.guidance, &delta), &delta),
            );
        }
        latents = schedule.step(&pred, index, &latents)?;
        eval(&latents)?;
        if index == 0 {
            for c in cache.iter().chain(negative_cache.iter()) {
                for a in c.arrays() {
                    eval(a)?;
                }
            }
        }
    }
    // The decoder does not need the per-layer prefix buffers.
    drop(cache);
    drop(negative_cache);
    Ok(latents)
}

/// Upstream's decode: `latents.T.reshape(1, z, 1, h, w).f32 * std + mean`,
/// `vae.decode`, then `((clip(x) + 1) / 2 * 255).astype(uint8)` (a
/// truncating cast). Returns `H * W * channels` bytes, `channels` 3 keeping
/// RGB (generation) or 4 keeping RGBA (an edit).
pub fn decode_to_pixels(
    vae: &QwenImageVae,
    latents: &MlxArray,
    grid_h: i32,
    grid_w: i32,
    channels: i32,
) -> Result<Vec<u8>> {
    let z = vae.config().z_dim;
    let packed = mlxcel_core::transpose_axes(latents, &[0, 2, 1]);
    let packed = mlxcel_core::reshape(&packed, &[1, z, grid_h, grid_w]);
    let decoded = vae.decode_normalized(&packed)?;
    eval(&decoded)?;
    to_pixels(&decoded, channels)
}

/// The byte conversion of [`decode_to_pixels`] on a decoder output
/// `[1, H, W, C]` f32.
pub fn to_pixels(decoded: &MlxArray, channels: i32) -> Result<Vec<u8>> {
    let s = mlxcel_core::array_shape(decoded);
    ensure!(
        s.len() == 4 && s[0] == 1 && s[3] >= channels,
        "unexpected Qwen-Image decoder output {s:?}"
    );
    let x = mlxcel_core::slice(decoded, &[0, 0, 0, 0], &[1, s[1], s[2], channels]);
    let x = mlxcel_core::clip(&x, &scalar_like(-1.0, &x), &scalar_like(1.0, &x));
    let x = mlxcel_core::add(&x, &scalar_like(1.0, &x));
    let x = mlxcel_core::multiply(
        &mlxcel_core::divide(&x, &scalar_like(2.0, &x)),
        &scalar_like(255.0, &x),
    );
    let x = mlxcel_core::astype(
        &mlxcel_core::reshape(&x, &[s[1], s[2], channels]),
        dtype::UINT8,
    );
    eval(&x)?;
    let bytes =
        mlxcel_core::try_array_to_raw_bytes(&x).map_err(|e| anyhow::anyhow!("Qwen-Image: {e}"))?;
    let expected = (s[1] * s[2] * channels) as usize;
    ensure!(
        bytes.len() == expected,
        "decoder output has {} bytes, expected {expected}",
        bytes.len()
    );
    Ok(bytes)
}
