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

//! FLUX.2 Klein text-to-image and image editing (family `flux2`).
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/flux2>:
//! the Klein 4B and 9B checkpoints, distilled (4 steps, guidance 1.0) and base
//! (50 steps, guidance 4.0), the `9b-kv` reference K/V cache, dense bf16
//! Diffusers trees and MLX affine-quantized exports. See
//! `docs/image-generation.md`.
//!
//! Batched generation follows Blaizzy/mlx-vlm#2355: `n` images share the
//! conditioning and come from one noise tensor of batch `n` drawn from `seed`,
//! so every image reports `seed`, and image `i` is not what `seed + i` gives
//! alone. Edits run once per image with seed `seed + i`.

mod attention;
mod blocks;
pub mod config;
mod edit;
pub mod embeddings;
pub mod kv_cache;
pub mod pipeline;
pub mod prompt;
pub mod scheduler;
pub mod tiling;
pub mod transformer;
mod weights;

#[cfg(test)]
mod scheduler_fixtures;

use std::path::Path;

use anyhow::{Result, bail};
use mlxcel_core::dtype;
use serde_json::{Map, Value};

use self::config::{Flux2Layout, Flux2TransformerConfig, Flux2Variant, read_quant_params};
use self::pipeline::{Flux2Pipeline, RunOptions};
use self::prompt::DEFAULT_MAX_SEQUENCE_LENGTH;
use super::Flux2Vae;
use super::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};

pub use self::config::validate_dimensions;
pub use self::edit::{resolve_edit_dimensions, round_edit_dimension};

/// Probe family name.
pub const FAMILY: &str = "flux2";

/// Read `max_sequence_length` and `tiled_vae` from `--gen-kwargs`.
pub fn run_options(extra: &Map<String, Value>) -> Result<RunOptions> {
    let max_sequence_length = match extra.get("max_sequence_length") {
        None | Some(Value::Null) => DEFAULT_MAX_SEQUENCE_LENGTH,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=DEFAULT_MAX_SEQUENCE_LENGTH as u64).contains(&n) => n as usize,
            _ => bail!(
                "max_sequence_length must be an integer between 1 and \
                 {DEFAULT_MAX_SEQUENCE_LENGTH}, got {v}"
            ),
        },
    };
    let tiled_vae = match extra.get("tiled_vae") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(v) => bail!("tiled_vae must be true or false, got {v}"),
    };
    Ok(RunOptions {
        max_sequence_length,
        tiled_vae,
    })
}

/// The transformer runs in bf16 on every backend, as upstream does
/// (`ModelConfig.precision`). Under f16 its activations overflow: the 4B
/// generate parity case decodes NaN latents with f16 weights, while the text
/// encoder and the VAE hold f16 (see docs/image-generation.md).
const TRANSFORMER_DTYPE: i32 = dtype::BFLOAT16;

/// A loaded FLUX.2 checkpoint.
pub struct Flux2Model {
    pipeline: Flux2Pipeline,
}

impl Flux2Model {
    /// Check the layout and configs, load the tokenizer and text encoder.
    pub fn load(path: &Path) -> Result<Self> {
        let layout = Flux2Layout::from_dir(path)?;
        let variant = Flux2Variant::from_dir(path)?;
        let transformer_config = Flux2TransformerConfig::resolve(&layout.transformer, variant)?;
        let transformer_quant = read_quant_params(&layout.transformer)?;
        let pipeline = Flux2Pipeline::new(
            layout,
            variant,
            transformer_config,
            transformer_quant,
            TRANSFORMER_DTYPE,
            Flux2Vae::default_dtype(),
        )?;
        Ok(Self { pipeline })
    }

    #[must_use]
    pub fn variant(&self) -> Flux2Variant {
        self.pipeline.variant
    }

    /// The pipeline, for callers that need latents rather than pixels.
    #[must_use]
    pub fn pipeline(&self) -> &Flux2Pipeline {
        &self.pipeline
    }

    fn pixels(array: &mlxcel_core::MlxArray, index: i32) -> Result<(Vec<u8>, u32, u32)> {
        let s = mlxcel_core::array_shape(array);
        if s.len() != 4 || s[3] != 3 {
            bail!("decoded image has shape {s:?}, expected [B, H, W, 3]");
        }
        let one = mlxcel_core::slice(array, &[index, 0, 0, 0], &[index + 1, s[1], s[2], s[3]]);
        let one = mlxcel_core::copy(&one);
        mlxcel_core::eval(&one);
        let bytes = mlxcel_core::array_to_raw_bytes(&one).to_vec();
        Ok((bytes, s[2] as u32, s[1] as u32))
    }
}

/// [`super::registry::FAMILIES`] constructor.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(Flux2Model::load(path)?))
}

impl ImageGenerationModel for Flux2Model {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, _task: ImageTask) -> bool {
        true
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        let v = self.pipeline.variant;
        ImageSamplingDefaults {
            steps: v.default_steps(),
            guidance: v.default_guidance(),
        }
    }

    /// Batched generation shares one noise tensor and reports `req.seed` for
    /// every image; edits run per image with `req.seed + i`.
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        validate_dimensions(req.width, req.height)?;
        let defaults = self.defaults(req.task);
        let steps = req.resolve_steps(defaults);
        let guidance = req.resolve_guidance(defaults);
        let options = run_options(&req.extra)?;
        let negative = req.negative_prompt.as_deref();
        let mut out = Vec::with_capacity(req.n as usize);
        match req.task {
            ImageTask::Generate => {
                let batch = i32::try_from(req.n)?;
                let images = self.pipeline.generate(
                    &req.prompt,
                    negative,
                    req.seed,
                    steps,
                    req.width,
                    req.height,
                    guidance,
                    batch,
                    &options,
                )?;
                for i in 0..batch {
                    let (pixels, width, height) = Self::pixels(&images, i)?;
                    out.push(GeneratedImage {
                        pixels,
                        width,
                        height,
                        channels: 3,
                        seed: req.seed,
                        revised_prompt: None,
                    });
                }
            }
            ImageTask::Edit => {
                for i in 0..req.n {
                    let seed = req.seed_for(i);
                    let image = self.pipeline.edit(
                        &req.prompt,
                        negative,
                        &req.reference_images,
                        seed,
                        steps,
                        req.width,
                        req.height,
                        guidance,
                        &options,
                    )?;
                    let (pixels, width, height) = Self::pixels(&image, 0)?;
                    out.push(GeneratedImage {
                        pixels,
                        width,
                        height,
                        channels: 3,
                        seed,
                        revised_prompt: None,
                    });
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;
