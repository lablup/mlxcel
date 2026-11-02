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

//! The Ming-Image denoising loop and decode, a port of
//! `MingImagePipeline.generate_array` and `_decode` (mlx-vlm `73d366ec`).
//!
//! The noise is `normal([1, z, 1, H / 8, W / 8], f32)` from
//! `random_key(seed)`. Each step feeds the transformer `bf16(latents)` and
//! `bf16(1 - sigma)` and takes the f32 velocity; the Euler step of
//! [`FlowMatchEulerDiscrete::dynamic`] (Ming's [`scheduler_shift`]) moves
//! the f32 latents by `-velocity`. The decode is
//! `vae.decode(latents / scaling_factor + shift_factor)`, clipped, mapped to
//! `[0, 255]` and truncated to `uint8`, all four channels (RGBA).
//!
//! Upstream draws one noise tensor for a batch of `num_images`; this port
//! runs image `i` alone with seed `seed + i`, the runtime's convention, so
//! `-n 2 --seed 0` returns exactly the images of seeds 0 and 1.

use anyhow::{Context, Result};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::{DOWNSAMPLE, scheduler_shift};
use super::text_encoder::Encoded;
use super::transformer::MingImageTransformer;
use crate::imagegen::qwen_image::vae::QwenImageVae;
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;

fn eval(x: &MlxArray) -> Result<()> {
    mlxcel_core::try_eval(x).map_err(|e| anyhow::anyhow!("Ming-Image: {e}"))
}

/// `[1, z_dim, 1, height / 8, width / 8]` standard-normal f32 noise from
/// `random_key(seed)`.
#[must_use]
pub fn initial_noise(seed: u64, z_dim: i32, width: u32, height: u32) -> UniquePtr<MlxArray> {
    let shape = [
        1,
        z_dim,
        1,
        (height / DOWNSAMPLE) as i32,
        (width / DOWNSAMPLE) as i32,
    ];
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is alive for the duration of the call.
    unsafe { mlxcel_core::random_normal(&shape, dtype::FLOAT32, &*key) }
}

/// The schedule of a `width x height` image.
pub fn schedule(
    transformer: &MingImageTransformer,
    width: u32,
    height: u32,
    steps: u32,
) -> Result<FlowMatchEulerDiscrete> {
    let p = transformer.config().patch_size as u32;
    let seq = (height / DOWNSAMPLE / p) * (width / DOWNSAMPLE / p);
    FlowMatchEulerDiscrete::dynamic(seq, steps, scheduler_shift(seq))
}

/// One denoising step's transformer call: `bf16(latents)` at `bf16(1 - sigma)`,
/// returned as f32.
pub fn velocity(
    transformer: &MingImageTransformer,
    latents: &MlxArray,
    sigma: f32,
    cond: &Encoded,
) -> Result<UniquePtr<MlxArray>> {
    let t = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0 - sigma], &[1]),
        dtype::BFLOAT16,
    );
    let x = mlxcel_core::astype(latents, dtype::BFLOAT16);
    let pred = transformer.forward(&x, &t, &cond.cap_feats, Some(&cond.cap_feats_2))?;
    Ok(mlxcel_core::astype(&pred, dtype::FLOAT32))
}

/// Run every step from `noise`; returns the final f32 latents.
pub fn denoise(
    transformer: &MingImageTransformer,
    schedule: &FlowMatchEulerDiscrete,
    noise: UniquePtr<MlxArray>,
    cond: &Encoded,
) -> Result<UniquePtr<MlxArray>> {
    let mut latents = noise;
    for (i, &sigma) in schedule
        .sigmas()
        .iter()
        .take(schedule.num_steps())
        .enumerate()
    {
        let pred = velocity(transformer, &latents, sigma, cond)?;
        latents = schedule.step(&mlxcel_core::negative(&pred), i, &latents)?;
        eval(&latents)?;
    }
    Ok(latents)
}

/// Decode `[1, z, 1, h, w]` latents into `H * W * 4` RGBA bytes.
pub fn decode_to_pixels(vae: &QwenImageVae, latents: &MlxArray) -> Result<Vec<u8>> {
    let cfg = vae.config();
    let s = mlxcel_core::array_shape(latents);
    anyhow::ensure!(
        s.len() == 5 && s[0] == 1 && s[2] == 1,
        "unexpected Ming-Image latents {s:?}"
    );
    let scale = cfg
        .scaling_factor
        .context("the Ming-Image VAE has no scaling_factor")?;
    let shift = cfg.shift_factor.unwrap_or(0.0);
    let x = mlxcel_core::astype(latents, dtype::FLOAT32);
    let x = mlxcel_core::reshape(&x, &[1, s[1], s[3], s[4]]);
    let x = mlxcel_core::divide(&x, &mlxcel_core::full_f32(&[1], scale, dtype::FLOAT32));
    let x = mlxcel_core::add(&x, &mlxcel_core::full_f32(&[1], shift, dtype::FLOAT32));
    let decoded = vae.decode_normalized(&x)?;
    eval(&decoded)?;
    crate::imagegen::qwen_image::pipeline::to_pixels(&decoded, 4)
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
