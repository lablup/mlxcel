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

//! The Mage-Flow denoising loops, a port of `MageFlowPipeline.generate_array`
//! and `edit_array` in mlx-vlm `73d366ec` `pipeline.py`.
//!
//! Everything runs in bf16 as upstream does: the initial noise is
//! `mx.random.normal((1, h*w, 128), key=mx.random.key(seed), dtype=bfloat16)`,
//! the transformer receives each sigma cast to the latent dtype, and the
//! static-shift Euler step ([`FlowMatchEulerDiscrete::static_shift`]) casts
//! its step size to the latent dtype. An edit appends the VAE latents of the
//! references (resized to the output size) after the target tokens at every
//! step and keeps only the target part of the velocity.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::layers::scalar_like;
use super::transformer::{ImageShape, MageFlowTransformer};
use super::vae::{LATENT_CHANNELS, MageVae};
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;
use crate::vision::dense::preprocess::{PillowResizeFilter, resize_pillow_rgb8};

/// Classifier-free guidance of one request.
pub struct Guidance<'a> {
    pub positive: &'a MlxArray,
    /// `None` runs the conditional branch alone.
    pub negative: Option<&'a MlxArray>,
    pub scale: f32,
    pub renormalization: bool,
}

/// Sampling settings shared by both loops.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub steps: u32,
    pub static_shift: f32,
    pub seed: u64,
    /// Output size in latent cells.
    pub grid_h: i32,
    pub grid_w: i32,
}

fn eval(x: &MlxArray) -> Result<()> {
    mlxcel_core::try_eval(x).map_err(|e| anyhow::anyhow!("Mage-Flow: {e}"))
}

/// The initial target latents `[1, grid_h * grid_w, 128]`, bf16.
#[must_use]
pub fn initial_noise(grid_h: i32, grid_w: i32, seed: u64) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is a live array for the call.
    unsafe {
        mlxcel_core::random_normal(
            &[1, grid_h * grid_w, LATENT_CHANNELS],
            dtype::BFLOAT16,
            &*key,
        )
    }
}

/// Upstream `_guided_velocity`.
fn guided_velocity(
    transformer: &MageFlowTransformer,
    latents: &MlxArray,
    guidance: &Guidance<'_>,
    sigma: f32,
    shapes: &[ImageShape],
) -> Result<UniquePtr<MlxArray>> {
    let conditional = transformer.forward(latents, guidance.positive, sigma, shapes)?;
    let Some(negative) = guidance.negative else {
        return Ok(conditional);
    };
    let unconditional = transformer.forward(latents, negative, sigma, shapes)?;
    let delta = mlxcel_core::subtract(&conditional, &unconditional);
    let combined = mlxcel_core::add(
        &unconditional,
        &mlxcel_core::multiply(&scalar_like(guidance.scale, &delta), &delta),
    );
    if !guidance.renormalization {
        return Ok(combined);
    }
    let cond_norm = mlxcel_core::linalg_norm(&conditional, -1, true);
    let combined_norm = mlxcel_core::linalg_norm(&combined, -1, true);
    let scaled = mlxcel_core::multiply(&combined, &cond_norm);
    Ok(mlxcel_core::divide(
        &scaled,
        &mlxcel_core::add(&combined_norm, &scalar_like(1e-6, &combined_norm)),
    ))
}

/// The text-to-image loop; returns the final latents `[1, h*w, 128]`.
pub fn denoise_generate(
    transformer: &MageFlowTransformer,
    guidance: &Guidance<'_>,
    sampling: Sampling,
) -> Result<UniquePtr<MlxArray>> {
    let schedule = FlowMatchEulerDiscrete::static_shift(sampling.steps, sampling.static_shift)?;
    let shapes = [(1, sampling.grid_h, sampling.grid_w)];
    let mut latents = initial_noise(sampling.grid_h, sampling.grid_w, sampling.seed);
    for index in 0..schedule.num_steps() {
        let sigma = schedule.sigmas()[index];
        let velocity = guided_velocity(transformer, &latents, guidance, sigma, &shapes)?;
        latents = schedule.step(&velocity, index, &latents)?;
        eval(&latents)?;
    }
    Ok(latents)
}

/// Upstream `_resize_reference` for every reference, stacked:
/// `[N, height, width, 3]` bf16 in `[-1, 1]` (PIL bicubic, then
/// `x / 127.5 - 1` in f32).
pub fn reference_pixels(
    images: &[DynamicImage],
    width: u32,
    height: u32,
) -> Result<UniquePtr<MlxArray>> {
    ensure!(
        !images.is_empty(),
        "At least one reference image is required"
    );
    let mut values = Vec::with_capacity(images.len() * (width * height * 3) as usize);
    for image in images {
        let resized =
            resize_pillow_rgb8(&image.to_rgb8(), width, height, PillowResizeFilter::Bicubic)?;
        values.extend(resized.as_raw().iter().map(|&p| f32::from(p) / 127.5 - 1.0));
    }
    let n = i32::try_from(images.len()).context("too many reference images")?;
    let pixels = mlxcel_core::from_slice_f32(&values, &[n, height as i32, width as i32, 3]);
    Ok(mlxcel_core::astype(&pixels, dtype::BFLOAT16))
}

/// The reference latents of an edit, packed `[1, N * h * w, 128]`:
/// `MageVAE.encode` of [`reference_pixels`] with `key(seed)`.
pub fn reference_tokens(
    vae: &MageVae,
    references: &MlxArray,
    sample_posterior: bool,
    seed: u64,
) -> Result<UniquePtr<MlxArray>> {
    let key = mlxcel_core::random_key(seed);
    let latents = vae.encode(references, sample_posterior, &key)?;
    eval(&latents)?;
    let s = mlxcel_core::array_shape(&latents);
    Ok(mlxcel_core::reshape(
        &latents,
        &[1, s[0] * s[1] * s[2], LATENT_CHANNELS],
    ))
}

/// The edit loop over `references` (`[1, N * h * w, 128]` from
/// [`reference_tokens`]); returns the final target latents `[1, h*w, 128]`.
pub fn denoise_edit(
    transformer: &MageFlowTransformer,
    guidance: &Guidance<'_>,
    sampling: Sampling,
    references: &MlxArray,
) -> Result<UniquePtr<MlxArray>> {
    let schedule = FlowMatchEulerDiscrete::static_shift(sampling.steps, sampling.static_shift)?;
    let target_len = sampling.grid_h * sampling.grid_w;
    let ref_len = mlxcel_core::array_shape(references)[1];
    ensure!(
        ref_len > 0 && ref_len % target_len == 0,
        "reference latents do not tile the output grid"
    );
    let shapes: Vec<ImageShape> =
        vec![(1, sampling.grid_h, sampling.grid_w); 1 + (ref_len / target_len) as usize];
    let mut target = initial_noise(sampling.grid_h, sampling.grid_w, sampling.seed);
    for index in 0..schedule.num_steps() {
        let sigma = schedule.sigmas()[index];
        let combined = mlxcel_core::concatenate(&target, references, 1);
        let velocity = guided_velocity(transformer, &combined, guidance, sigma, &shapes)?;
        let width = *mlxcel_core::array_shape(&velocity).last().context("rank")?;
        let velocity = mlxcel_core::slice(&velocity, &[0, 0, 0], &[1, target_len, width]);
        target = schedule.step(&velocity, index, &target)?;
        eval(&target)?;
    }
    Ok(target)
}

/// Decode final latents `[1, h*w, 128]` into the VAE output `[1, H, W, 3]`.
pub fn decode(
    vae: &MageVae,
    latents: &MlxArray,
    grid_h: i32,
    grid_w: i32,
) -> Result<UniquePtr<MlxArray>> {
    let latents = mlxcel_core::reshape(latents, &[1, grid_h, grid_w, LATENT_CHANNELS]);
    let decoded = vae.decode(&latents);
    eval(&decoded)?;
    Ok(decoded)
}

/// Upstream `_image_array`: `round(127.5 * (clip(x, -1, 1) + 1))` as uint8,
/// computed in the decoder dtype; returns the `H * W * 3` bytes.
pub fn to_pixels(decoded: &MlxArray) -> Result<Vec<u8>> {
    let s = mlxcel_core::array_shape(decoded);
    ensure!(
        s.len() == 4 && s[0] == 1 && s[3] == 3,
        "unexpected decoder output {s:?}"
    );
    let x = mlxcel_core::reshape(decoded, &[s[1], s[2], 3]);
    let x = mlxcel_core::clip(&x, &scalar_like(-1.0, &x), &scalar_like(1.0, &x));
    let x = mlxcel_core::multiply(
        &scalar_like(127.5, &x),
        &mlxcel_core::add(&x, &scalar_like(1.0, &x)),
    );
    let x = mlxcel_core::astype(&mlxcel_core::round(&x), dtype::UINT8);
    eval(&x)?;
    let bytes =
        mlxcel_core::try_array_to_raw_bytes(&x).map_err(|e| anyhow::anyhow!("Mage-Flow: {e}"))?;
    ensure!(
        bytes.len() == (s[1] * s[2] * 3) as usize,
        "decoder output has {} bytes, expected {}",
        bytes.len(),
        s[1] * s[2] * 3
    );
    Ok(bytes)
}
