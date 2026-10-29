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

//! LLaDA-Image sampling.
//!
//! A port of the numeric half of `LLaDAImagePipeline` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/pipeline.py>:
//! prompt encoding, the noise draws, the denoising loop, the source-image
//! encoding of an edit and the decode. Component lifetimes live in
//! [`super::LladaImageModel`].
//!
//! Randomness follows MLX's global key sequence exactly: after
//! `mx.random.seed(seed)` every draw splits the state, keeps the first half
//! as the new state and draws from the second. The initial noise is draw 1
//! (`normal([1, 128, H/16, W/16])` in f32, rounded through bf16); with
//! stochastic sampling, step `i` re-noises from draw `i + 2`.

use anyhow::{Context, Result};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::conditioning::{QueryFormer, TextProjection, bool_array, query_attention_mask};
use super::config::LATENT_SCALE;
use super::scheduler::LladaScheduler;
use super::transformer::LladaImageTransformer;
use crate::imagegen::vae::Flux2Vae;
use crate::models::Llada2MoeModel;

/// MLX's global random state after `mx.random.seed(seed)`.
pub struct KeySequence {
    state: UniquePtr<MlxArray>,
}

impl KeySequence {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: mlxcel_core::random_key(seed),
        }
    }

    /// `KeySequence.next()`: split, keep `[0]`, return `[1]`.
    pub fn next_key(&mut self) -> UniquePtr<MlxArray> {
        let keys = mlxcel_core::random_split_key(&self.state, 2);
        let take = |row: i32| {
            mlxcel_core::reshape(&mlxcel_core::slice(&keys, &[row, 0], &[row + 1, 2]), &[2])
        };
        self.state = take(0);
        take(1)
    }

    /// `mx.random.normal(shape, float32)` from the next key.
    pub fn normal_f32(&mut self, shape: &[i32]) -> UniquePtr<MlxArray> {
        let key = self.next_key();
        // SAFETY: `key` is alive for the duration of the call.
        unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*key) }
    }
}

/// Latent shape `[1, channels, H / 16, W / 16]`.
#[must_use]
pub fn latent_shape(channels: i32, width: u32, height: u32) -> [i32; 4] {
    [
        1,
        channels,
        (height / LATENT_SCALE) as i32,
        (width / LATENT_SCALE) as i32,
    ]
}

/// The first draw after seeding, rounded through `dt` and back to f32.
pub fn initial_latents(rng: &mut KeySequence, shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let noise = rng.normal_f32(shape);
    mlxcel_core::astype(&mlxcel_core::astype(&noise, dt), dtype::FLOAT32)
}

/// Upstream `_encode_prompt` without its cache: the projected caption
/// features `[1, T + Q, projection_dim]` of `ids`.
pub fn encode_prompt(
    text_encoder: &Llada2MoeModel,
    queryformer: &QueryFormer,
    projection: &TextProjection,
    ids: &[i32],
) -> Result<UniquePtr<MlxArray>> {
    let t = ids.len();
    let ids_arr = mlxcel_core::from_slice_i32(ids, &[1, t as i32]);
    let embeddings = text_encoder.embed(&ids_arr);
    let queries = queryformer.forward(&embeddings, &bool_array(&vec![true; t], &[1, t as i32]));
    let combined = mlxcel_core::concatenate(&embeddings, &queries, 1);
    let q = queryformer.num_queries() as usize;
    let positions: Vec<i32> = (0..(t + q) as i32).collect();
    let positions = mlxcel_core::from_slice_i32(&positions, &[1, (t + q) as i32]);
    let hidden = text_encoder
        .forward_embeds(&combined, Some(&query_attention_mask(t, q)), &positions)
        .map_err(anyhow::Error::msg)?;
    let output = projection.forward(&hidden);
    mlxcel_core::try_eval(&output).context("LLaDA-Image prompt encoding failed")?;
    Ok(output)
}

/// What the transformer is conditioned on besides the caption.
pub struct Conditioning<'a> {
    pub caption: &'a MlxArray,
    /// Negative caption; CFG runs when present.
    pub negative: Option<&'a MlxArray>,
    /// SigVQ features (VQ generation or the edit source).
    pub semantic: Option<&'a MlxArray>,
    /// BN-normalized source latents of an edit.
    pub source: Option<&'a MlxArray>,
    pub guidance: f32,
}

/// The negated flow prediction in f32, with CFG combined.
pub(crate) fn predict(
    transformer: &LladaImageTransformer,
    input: &MlxArray,
    time: &MlxArray,
    cond: &Conditioning<'_>,
) -> Result<UniquePtr<MlxArray>> {
    let negate =
        |x: UniquePtr<MlxArray>| mlxcel_core::negative(&mlxcel_core::astype(&x, dtype::FLOAT32));
    let prediction =
        negate(transformer.forward(input, time, cond.caption, cond.semantic, cond.source)?);
    let Some(negative) = cond.negative else {
        return Ok(prediction);
    };
    // `semantic[:, :0]`: the unconditional branch drops the SigVQ features.
    let empty = cond.semantic.map(|s| {
        let shape = mlxcel_core::array_shape(s);
        mlxcel_core::slice(s, &[0, 0, 0], &[shape[0], 0, shape[2]])
    });
    let unconditional =
        negate(transformer.forward(input, time, negative, empty.as_deref(), cond.source)?);
    let guidance = mlxcel_core::full_f32(&[], cond.guidance, dtype::FLOAT32);
    Ok(mlxcel_core::add(
        &unconditional,
        &mlxcel_core::multiply(
            &guidance,
            &mlxcel_core::subtract(&prediction, &unconditional),
        ),
    ))
}

/// The denoising loop of `generate_array`, from the seeded `rng` (whose
/// first draw is already the initial `latents`). Returns the final f32
/// latents.
pub fn denoise(
    transformer: &LladaImageTransformer,
    scheduler: &LladaScheduler,
    stochastic: bool,
    rng: &mut KeySequence,
    mut latents: UniquePtr<MlxArray>,
    cond: &Conditioning<'_>,
) -> Result<UniquePtr<MlxArray>> {
    let dt = transformer.dtype();
    for index in 0..scheduler.steps() {
        let time = scheduler.timestep(index, dt);
        let input = mlxcel_core::astype(&latents, dt);
        let prediction = predict(transformer, &input, &time, cond)?;
        let noise = stochastic.then(|| rng.normal_f32(&mlxcel_core::array_shape(&latents)));
        latents = scheduler.step(&prediction, &latents, index, stochastic, noise.as_deref())?;
        // `try_eval`: an MLX failure (allocation, kernel) must surface as an
        // error, not a C++ exception that aborts the process.
        mlxcel_core::try_eval(&latents)
            .with_context(|| format!("LLaDA-Image denoising step {index} failed"))?;
    }
    Ok(latents)
}

/// Decode f32 latents: `decode_packed_latents(latents.bf16)`, then
/// `round(clip(x.f32 / 2 + 0.5, 0, 1) * 255)` as HWC `u8`.
pub fn decode_to_pixels(vae: &Flux2Vae, latents: &MlxArray) -> Result<Vec<u8>> {
    let decoded =
        vae.decode_packed_latents(&mlxcel_core::astype(latents, Flux2Vae::default_dtype()))?;
    let f = |v: f32| mlxcel_core::full_f32(&[], v, dtype::FLOAT32);
    let x = mlxcel_core::astype(&decoded, dtype::FLOAT32);
    let unit = mlxcel_core::add(&mlxcel_core::divide(&x, &f(2.0)), &f(0.5));
    let clipped = mlxcel_core::clip(&unit, &f(0.0), &f(1.0));
    let hwc = mlxcel_core::transpose_axes(&clipped, &[0, 2, 3, 1]);
    let bytes = mlxcel_core::astype(
        &mlxcel_core::round(&mlxcel_core::multiply(&hwc, &f(255.0))),
        dtype::UINT8,
    );
    mlxcel_core::try_array_to_raw_bytes(&bytes).context("LLaDA-Image VAE decode failed")
}

/// `[1, H, W, 3]` f32 pixels in `[-1, 1]` from HWC `u8` RGB, computed on the
/// device as upstream's `mx.array(pixels)[None] / 127.5 - 1`.
#[must_use]
pub fn normalized_pixels(rgb: &[u8], width: u32, height: u32) -> UniquePtr<MlxArray> {
    let values: Vec<f32> = rgb.iter().map(|&v| f32::from(v)).collect();
    let x = mlxcel_core::from_slice_f32(&values, &[1, height as i32, width as i32, 3]);
    let f = |v: f32| mlxcel_core::full_f32(&[], v, dtype::FLOAT32);
    mlxcel_core::subtract(&mlxcel_core::divide(&x, &f(127.5)), &f(1.0))
}

/// The 2x2 mean of `[1, H, W, 3]` pixels (upstream's half-size bilinear
/// interpolation with `align_corners=False`).
#[must_use]
pub fn half_resolution(pixels: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(pixels);
    let (h, w) = (s[1] / 2, s[2] / 2);
    let x = mlxcel_core::reshape(pixels, &[1, h, 2, w, 2, 3]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    let x = mlxcel_core::reshape(&x, &[1, h, w, 4, 3]);
    mlxcel_core::mean_axis_native_dtype(&x, 3, false)
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
