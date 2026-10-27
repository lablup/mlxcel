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

//! Z-Image sampling: text-to-image and img2img.
//!
//! A port of `ZImagePipeline._denoise`, `_decode_latents`, `generate_array`
//! and `edit_array` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/pipeline.py>
//! (`#L112-L292`). Noise is `normal([1, 16, 1, H/8, W/8], f32)` drawn from
//! `random_key(seed)`, the transformer input time is `1 - sigma`, and the
//! Euler step runs on `-prediction` in the latent dtype. Classifier-free
//! guidance applies only when the guidance exceeds 1 and only while
//! `t <= cfg_truncation`, with the conditional prediction as the base.

use anyhow::{Result, bail};
use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::transformer::ZImageTransformer;
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;
use crate::imagegen::vae::Flux2Vae;

/// Pixels per latent cell.
pub const VAE_SCALE: u32 = 8;

/// Conditioning shared by every image of one request.
pub struct Conditioning<'a> {
    pub cap_feats: &'a MlxArray,
    /// Present only when CFG is on (guidance above 1).
    pub negative: Option<&'a MlxArray>,
    pub guidance: f32,
    pub cfg_truncation: f64,
}

fn scalar(value: f32, dt: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dt)
}

/// `[n, C, 1, H / 8, W / 8]` standard-normal f32 noise from `random_key(seed)`.
#[must_use]
pub fn initial_noise(seed: u64, channels: i32, width: u32, height: u32) -> UniquePtr<MlxArray> {
    // `2 * (height // (vae_scale * 2))`, as upstream rounds to the patch grid.
    let latent_h = 2 * (height / (VAE_SCALE * 2)) as i32;
    let latent_w = 2 * (width / (VAE_SCALE * 2)) as i32;
    noise_like(seed, &[1, channels, 1, latent_h, latent_w])
}

fn noise_like(seed: u64, shape: &[i32]) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is alive for the duration of the call.
    unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*key) }
}

/// Run `schedule` from `start_index` to the end.
pub fn denoise(
    transformer: &ZImageTransformer,
    schedule: &FlowMatchEulerDiscrete,
    start_index: usize,
    mut latents: UniquePtr<MlxArray>,
    cond: &Conditioning<'_>,
) -> Result<UniquePtr<MlxArray>> {
    let steps = schedule.sigmas().iter().enumerate();
    for (i, &sigma) in steps.take(schedule.num_steps()).skip(start_index) {
        let t_value = 1.0_f32 - sigma;
        let t = mlxcel_core::from_slice_f32(&[t_value], &[1]);
        let mut prediction = transformer.forward(&latents, &t, cond.cap_feats)?;
        if let Some(negative) = cond.negative
            && f64::from(t_value) <= cond.cfg_truncation
        {
            let negative_prediction = transformer.forward(&latents, &t, negative)?;
            let g = scalar(cond.guidance, mlxcel_core::array_dtype(&prediction));
            let delta = mlxcel_core::subtract(&prediction, &negative_prediction);
            prediction = mlxcel_core::add(&prediction, &mlxcel_core::multiply(&g, &delta));
        }
        latents = schedule.step(&mlxcel_core::negative(&prediction), i, &latents)?;
        mlxcel_core::eval(&latents);
    }
    Ok(latents)
}

/// Decode `[1, C, 1, h, w]` latents to tightly packed HWC `u8` RGB pixels.
pub fn decode_to_pixels(vae: &Flux2Vae, latents: &MlxArray) -> Result<Vec<u8>> {
    let decoded = vae.decode(latents)?;
    let nhwc = mlxcel_core::transpose_axes(&decoded, &[0, 2, 3, 1]);
    let dt = mlxcel_core::array_dtype(&nhwc);
    let unit = mlxcel_core::add(
        &mlxcel_core::divide(&nhwc, &scalar(2.0, dt)),
        &scalar(0.5, dt),
    );
    let clipped = mlxcel_core::clip(&unit, &scalar(0.0, dt), &scalar(1.0, dt));
    let rounded = mlxcel_core::round(&mlxcel_core::multiply(&clipped, &scalar(255.0, dt)));
    let bytes = mlxcel_core::astype(&rounded, dtype::UINT8);
    mlxcel_core::eval(&bytes);
    Ok(mlxcel_core::array_to_raw_bytes(&bytes).to_vec())
}

/// `[1, 3, H, W]` f32 pixels in `[-1, 1]` from `image`, resized to `width x
/// height` with Lanczos when it differs (upstream: PIL `LANCZOS`, which
/// copies an image already at the target size).
#[must_use]
pub fn reference_pixels(image: &DynamicImage, width: u32, height: u32) -> UniquePtr<MlxArray> {
    let rgb = image.to_rgb8();
    let rgb = if rgb.width() == width && rgb.height() == height {
        rgb
    } else {
        image::imageops::resize(&rgb, width, height, FilterType::Lanczos3)
    };
    let values: Vec<f32> = rgb
        .as_raw()
        .iter()
        .map(|&v| f32::from(v) / 127.5 - 1.0)
        .collect();
    let nhwc = mlxcel_core::from_slice_f32(&values, &[1, height as i32, width as i32, 3]);
    mlxcel_core::transpose_axes(&nhwc, &[0, 3, 1, 2])
}

/// Encode reference pixels to `[1, C, 1, h, w]` latents (`(mean - shift) * scale`).
pub fn encode_reference(vae: &Flux2Vae, pixels: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let encoded = vae.encode(pixels)?;
    let latents = mlxcel_core::expand_dims(&encoded, 2);
    mlxcel_core::eval(&latents);
    Ok(latents)
}

/// The img2img starting latents: `(1 - sigma) * image + sigma * noise`, with
/// `sigma = sigmas[start_index]` cast to the image-latent dtype.
pub fn noised_reference(
    image_latents: &MlxArray,
    schedule: &FlowMatchEulerDiscrete,
    start_index: usize,
    seed: u64,
) -> Result<UniquePtr<MlxArray>> {
    if start_index >= schedule.num_steps() {
        bail!("img2img start index {start_index} is past the schedule");
    }
    let noise = noise_like(seed, &mlxcel_core::array_shape(image_latents));
    let dt = mlxcel_core::array_dtype(image_latents);
    let sigma = scalar(schedule.sigmas()[start_index], dt);
    let keep = mlxcel_core::subtract(&scalar(1.0, dt), &sigma);
    Ok(mlxcel_core::add(
        &mlxcel_core::multiply(&keep, image_latents),
        &mlxcel_core::multiply(&sigma, &noise),
    ))
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
