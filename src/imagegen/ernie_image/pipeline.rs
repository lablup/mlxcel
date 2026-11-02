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

//! ERNIE-Image sampling: text-to-image and SDEdit img2img.
//!
//! A port of `_pad_text`, `_to_image_array`, `generate_array`, `edit_array`,
//! `_denoise` and `_decode` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/pipeline.py>.
//! Noise is `normal([1, 128, H/16, W/16], bf16)` from `random_key(seed)`; the
//! transformer reads the timestep in the latent dtype; classifier-free guidance
//! runs the negative and positive prompts as one batch of two when the
//! guidance exceeds 1, with the unconditional prediction as the base. Latents
//! are patchified FLUX.2 latents, decoded by the shared FLUX.2 VAE.

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::scheduler::ErnieScheduler;
use super::transformer::ErnieTransformer;
use crate::imagegen::latent::patchify_latents;
use crate::imagegen::vae::Flux2Vae;

/// Pixels per latent cell (8x VAE, then 2x2 patches).
pub const LATENT_SCALE: u32 = 16;

fn scalar(value: f32, dt: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dt)
}

/// Text conditioning for one image: the prompts stacked as one batch.
pub struct TextBatch {
    /// `[B, T, text_dim]`, each row zero-padded to the longest prompt.
    pub hidden: UniquePtr<MlxArray>,
    /// Valid tokens of each row.
    pub lengths: Vec<i32>,
    /// Classifier-free guidance scale; CFG is on when `> 1` (then `B == 2`:
    /// negative, positive).
    pub guidance: f32,
}

impl TextBatch {
    /// Upstream `_pad_text`: right-pad every `[1, T_i, D]` embedding with
    /// zeros to the longest and concatenate.
    pub fn new(embeddings: &[&MlxArray], guidance: f32) -> Result<Self> {
        if embeddings.is_empty() {
            bail!("At least one text embedding is required");
        }
        let lengths: Vec<i32> = embeddings
            .iter()
            .map(|e| mlxcel_core::array_shape(e)[1])
            .collect();
        let max = lengths.iter().copied().max().unwrap_or(0);
        let padded: Vec<UniquePtr<MlxArray>> = embeddings
            .iter()
            .zip(&lengths)
            .map(|(e, &len)| {
                if len == max {
                    mlxcel_core::copy(e)
                } else {
                    let shape = mlxcel_core::array_shape(e);
                    let pad = mlxcel_core::zeros(
                        &[shape[0], max - len, shape[2]],
                        mlxcel_core::array_dtype(e),
                    );
                    mlxcel_core::concatenate(e, &pad, 1)
                }
            })
            .collect();
        let refs: Vec<&MlxArray> = padded.iter().map(|p| &**p).collect();
        Ok(Self {
            hidden: mlxcel_core::concatenate_many(&refs, 0),
            lengths,
            guidance,
        })
    }

    #[must_use]
    pub fn cfg(&self) -> bool {
        self.guidance > 1.0
    }
}

/// `normal([1, channels, H / 16, W / 16], bf16)` from `random_key(seed)`,
/// as upstream draws it.
#[must_use]
pub fn initial_noise(seed: u64, channels: i32, width: u32, height: u32) -> UniquePtr<MlxArray> {
    let shape = [
        1,
        channels,
        (height / LATENT_SCALE) as i32,
        (width / LATENT_SCALE) as i32,
    ];
    noise_like(seed, &shape, dtype::BFLOAT16)
}

fn noise_like(seed: u64, shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is alive for the duration of the call.
    unsafe { mlxcel_core::random_normal(shape, dt, &*key) }
}

/// One transformer call on `latents` at step `index`, with CFG combined.
pub fn predict(
    transformer: &ErnieTransformer,
    scheduler: &ErnieScheduler,
    index: usize,
    latents: &MlxArray,
    text: &TextBatch,
) -> Result<UniquePtr<MlxArray>> {
    let dt = mlxcel_core::array_dtype(latents);
    let input = if text.cfg() {
        mlxcel_core::concatenate(latents, latents, 0)
    } else {
        mlxcel_core::copy(latents)
    };
    let batch = mlxcel_core::array_shape(&input)[0];
    let t = mlxcel_core::full_f32(&[batch], scheduler.timesteps()[index], dt);
    let prediction = transformer.forward(&input, &t, &text.hidden, &text.lengths)?;
    if !text.cfg() {
        return Ok(prediction);
    }
    let s = mlxcel_core::array_shape(&prediction);
    let half = |start: i32| {
        mlxcel_core::slice(
            &prediction,
            &[start, 0, 0, 0],
            &[start + 1, s[1], s[2], s[3]],
        )
    };
    let (uncond, cond) = (half(0), half(1));
    let pdt = mlxcel_core::array_dtype(&prediction);
    let delta = mlxcel_core::subtract(&cond, &uncond);
    Ok(mlxcel_core::add(
        &uncond,
        &mlxcel_core::multiply(&scalar(text.guidance, pdt), &delta),
    ))
}

/// Upstream `_denoise`: run `scheduler` from `start_index` to the end.
pub fn denoise(
    transformer: &ErnieTransformer,
    scheduler: &ErnieScheduler,
    start_index: usize,
    mut latents: UniquePtr<MlxArray>,
    text: &TextBatch,
) -> Result<UniquePtr<MlxArray>> {
    for index in start_index..scheduler.num_steps() {
        let prediction = predict(transformer, scheduler, index, &latents, text)?;
        latents = scheduler.step(&prediction, index, &latents)?;
        // `try_eval`: an MLX failure (allocation, kernel) must surface as an
        // error, not a C++ exception that aborts the process.
        mlxcel_core::try_eval(&latents)
            .with_context(|| format!("ERNIE-Image denoising step {index} failed"))?;
    }
    Ok(latents)
}

/// SDEdit starting latents from a reference: VAE-encode `pixels`
/// (`[1, 3, H, W]` f32 in `[-1, 1]`), patchify, batch-norm normalize, then
/// `(1 - sigma) * source + sigma * noise` at `sigmas[start_index]`, all in
/// `dt` (upstream's encoded latents are bf16).
pub fn noised_reference(
    vae: &Flux2Vae,
    pixels: &MlxArray,
    scheduler: &ErnieScheduler,
    start_index: usize,
    seed: u64,
    dt: i32,
) -> Result<UniquePtr<MlxArray>> {
    if start_index >= scheduler.num_steps() {
        bail!("img2img start index {start_index} is past the schedule");
    }
    let encoded = vae.encode(pixels)?;
    let encoded = mlxcel_core::astype(&encoded, dt);
    let packed = patchify_latents(&encoded)?;
    let source = vae.bn_normalize(&packed)?;
    let noise = noise_like(seed, &mlxcel_core::array_shape(&source), dt);
    let sigma = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[scheduler.sigmas()[start_index]], &[]),
        dt,
    );
    let keep = mlxcel_core::subtract(&scalar(1.0, dt), &sigma);
    let latents = mlxcel_core::add(
        &mlxcel_core::multiply(&keep, &source),
        &mlxcel_core::multiply(&sigma, &noise),
    );
    mlxcel_core::try_eval(&latents).context("ERNIE-Image reference image encode failed")?;
    Ok(latents)
}

/// Decode patchified latents in the VAE's own dtype.
///
/// The transformer stays bf16, but under `MLXCEL_CUDA_F16_NORMALIZE` the VAE
/// weights are f16, and bf16 x f16 promotes the whole decode to f32 (OOM at
/// 1024x1024 with TF32 off). Cast here, not inside `Flux2Vae`, because callers
/// own the latent dtype (Z-Image feeds f32 on purpose).
pub fn decode_latents(vae: &Flux2Vae, latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    vae.decode_packed_latents(&mlxcel_core::astype(latents, vae.dtype()))
}

/// Upstream `_decode` + `_to_image_array`: decode patchified latents, then
/// `round(clip(x / 2 + 0.5, 0, 1).f32 * 255)` as HWC `u8`.
pub fn decode_to_pixels(vae: &Flux2Vae, latents: &MlxArray) -> Result<Vec<u8>> {
    let decoded = decode_latents(vae, latents)?;
    let dt = mlxcel_core::array_dtype(&decoded);
    let unit = mlxcel_core::add(
        &mlxcel_core::divide(&decoded, &scalar(2.0, dt)),
        &scalar(0.5, dt),
    );
    let clipped = mlxcel_core::clip(&unit, &scalar(0.0, dt), &scalar(1.0, dt));
    let nhwc = mlxcel_core::transpose_axes(&clipped, &[0, 2, 3, 1]);
    let nhwc = mlxcel_core::astype(&nhwc, dtype::FLOAT32);
    let scaled = mlxcel_core::multiply(&nhwc, &scalar(255.0, dtype::FLOAT32));
    let bytes = mlxcel_core::astype(&mlxcel_core::round(&scaled), dtype::UINT8);
    mlxcel_core::try_array_to_raw_bytes(&bytes).context("ERNIE-Image VAE decode failed")
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
