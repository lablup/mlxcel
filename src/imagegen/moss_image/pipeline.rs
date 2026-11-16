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

//! MOSS-Image sampling: the flow-match Euler schedule snapped to the
//! training grid (a port of `scheduler.py`), classifier-free guidance with the
//! negative branch's block skip, and the FLUX.2 VAE decode. A port of
//! `pipeline.py` in `felix-ab/moss-image-mlx`.
//!
//! **Deviation, recorded:** the reference draws its initial noise from a
//! torch CPU generator so the official seeds reproduce bit for bit. This port
//! draws from MLX's generator with the same seed; the distributions and all
//! downstream math agree, but a given seed does not reproduce the reference's
//! exact image.

use anyhow::{Context, Result, anyhow};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::{DEFAULT_SIDE, TRAINING_STEPS};
use super::transformer::MossTransformer;
use crate::imagegen::mage_flow::layers::scalar_like;
use crate::imagegen::vae::Flux2Vae;

/// `torch.linspace(start, end, steps)` in f32: the first half counts up from
/// `start`, the second down from `end`, each value one rounded product.
#[must_use]
pub fn torch_linspace(start: f32, end: f32, steps: u32) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let start64 = f64::from(start);
    let end64 = f64::from(end);
    let step = f64::from((end - start) / (steps - 1) as f32);
    let halfway = (steps / 2) as usize;
    let mut out = Vec::with_capacity(steps as usize);
    for i in 0..halfway {
        out.push((start64 + step * i as f64) as f32);
    }
    for i in halfway..steps as usize {
        out.push((end64 - step * (steps as usize - i - 1) as f64) as f32);
    }
    out
}

fn grid(count: u32, shift: f32, total: u32) -> Vec<f32> {
    let lower = shift / (total as f32 + shift - 1.0);
    torch_linspace(1.0, lower, count)
        .into_iter()
        .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
        .collect()
}

/// Sigmas snapped to the training grid, followed by a terminal 0.
#[must_use]
pub fn sigmas_for(steps: u32, shift: f32) -> Vec<f32> {
    let mut training = grid(TRAINING_STEPS, 1.0, TRAINING_STEPS);
    training.reverse();
    let raw = grid(steps, shift, TRAINING_STEPS);
    let mut snapped: Vec<f32> = Vec::with_capacity(raw.len());
    for sigma in raw {
        // `searchsorted(training, sigma, side="left")` clamped to `0..total-1`.
        let upper = training
            .iter()
            .position(|&t| t >= sigma)
            .unwrap_or(training.len() - 1)
            .min(training.len() - 1);
        let lower = upper.saturating_sub(1);
        let nearest = if (sigma - training[lower]).abs() <= (training[upper] - sigma).abs() {
            lower
        } else {
            upper
        };
        let value = training[nearest];
        if snapped.last() != Some(&value) {
            snapped.push(value);
        }
    }
    snapped.push(0.0);
    snapped
}

/// The timesteps of a sigma schedule (`sigma * total`).
#[must_use]
pub fn timesteps_for(sigmas: &[f32]) -> Vec<f32> {
    sigmas[..sigmas.len() - 1]
        .iter()
        .map(|s| s * TRAINING_STEPS as f32)
        .collect()
}

/// The initial latents `[1, latent_channels, height / 16, width / 16]` in f32.
#[must_use]
pub fn initial_noise(
    height: u32,
    width: u32,
    latent_channels: i32,
    seed: u64,
) -> UniquePtr<MlxArray> {
    mlxcel_core::random_seed(seed);
    // SAFETY: a null key draws from the global generator just seeded.
    unsafe {
        mlxcel_core::random_normal(
            &[
                1,
                latent_channels,
                (height / 16) as i32,
                (width / 16) as i32,
            ],
            dtype::FLOAT32,
            std::ptr::null(),
        )
    }
}

/// Denoise from `latents`, returning the final f32 latents.
///
/// `positive` and `negative` are the conditioner's `[1, L, text_dim]` hidden
/// states. `skip_layers` is the negative branch's block skip.
pub fn sample(
    model: &MossTransformer,
    positive: &MlxArray,
    negative: Option<&MlxArray>,
    latents: UniquePtr<MlxArray>,
    steps: u32,
    guidance: f32,
    shift: f32,
    skip_layers: &[i32],
) -> Result<UniquePtr<MlxArray>> {
    let sigmas = sigmas_for(steps, shift);
    let timesteps = timesteps_for(&sigmas);
    let mut latents = latents;
    for (index, &t) in timesteps.iter().enumerate() {
        let cond = model.forward(&latents, t, positive, &[])?;
        let prediction = match negative {
            Some(negative) if guidance != 1.0 => {
                let uncond = model.forward(&latents, t, negative, skip_layers)?;
                // uncond + guidance * (cond - uncond)
                mlxcel_core::add(
                    &uncond,
                    &mlxcel_core::multiply(
                        &mlxcel_core::subtract(&cond, &uncond),
                        &scalar_like(guidance, &cond),
                    ),
                )
            }
            _ => cond,
        };
        let dt = sigmas[index + 1] - sigmas[index];
        latents = mlxcel_core::add(
            &latents,
            &mlxcel_core::multiply(&prediction, &scalar_like(dt, &prediction)),
        );
        mlxcel_core::try_eval(&latents).map_err(|e| anyhow!("MOSS-Image step {index}: {e}"))?;
    }
    Ok(latents)
}

/// Decode packed latents to interleaved RGB `u8` (`round(clip(x/2+0.5, 0, 1) * 255)`).
pub fn decode_to_pixels(vae: &Flux2Vae, latents: &MlxArray) -> Result<Vec<u8>> {
    // The published MOSS checkpoint ships the VAE **decoder alone**: 142 tensors,
    // no `bn.*` running statistics. Its latents are therefore not batch-norm
    // normalized, and `decode_packed_latents` — which exists to undo that
    // normalization — refuses to run. Unpatchify and hand the result straight to
    // the decoder, which still applies the config's `scaling_factor`/`shift_factor`.
    let unpacked = crate::imagegen::latent::unpatchify_latents(&mlxcel_core::astype(
        latents,
        vae.dtype(),
    ))?;
    let decoded = vae
        .decode(&unpacked)
        .context("MOSS-Image VAE decode")?;
    let dt = mlxcel_core::array_dtype(&decoded);
    let unit = mlxcel_core::add(
        &mlxcel_core::divide(&decoded, &scalar_like(2.0, &decoded)),
        &scalar_like(0.5, &decoded),
    );
    let clipped = mlxcel_core::clip(&unit, &scalar_like(0.0, &unit), &scalar_like(1.0, &unit));
    let nhwc = mlxcel_core::transpose_axes(&clipped, &[0, 2, 3, 1]);
    let nhwc = mlxcel_core::astype(&nhwc, dtype::FLOAT32);
    let scaled = mlxcel_core::multiply(&nhwc, &scalar_like(255.0, &nhwc));
    let bytes = mlxcel_core::astype(&mlxcel_core::round(&scaled), dtype::UINT8);
    let _ = dt;
    mlxcel_core::try_array_to_raw_bytes(&bytes).context("MOSS-Image VAE decode")
}

/// The default size.
#[must_use]
pub fn default_size() -> (u32, u32) {
    (DEFAULT_SIDE, DEFAULT_SIDE)
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
