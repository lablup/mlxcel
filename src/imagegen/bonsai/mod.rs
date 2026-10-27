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

//! Prism Bonsai, ternary 2-bit FLUX.2 Klein 4B text-to-image (family `bonsai`).
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/bonsai>
//! for `prism-ml/bonsai-image-ternary-4B-mlx-2bit`. The pipeline is FLUX.2's
//! generate loop unchanged (upstream `BonsaiImage.generate_array`), so this
//! module only injects what differs:
//!
//! - the transformer from `transformer-packed-mflux/`, whose
//!   `quantized_modules` load as 2-bit group-128 affine linears
//!   ([`loader::PackedLinears`]); `klein_fast`'s megakernel and fused
//!   norm-RoPE Metal kernels are speed work over the same arithmetic and are
//!   not ported;
//! - the MLX 4-bit Qwen3 conditioner from `text_encoder-mlx-4bit/`;
//! - prompts padded to `max_sequence_length` (upstream
//!   `bucketed_seq_len=False`), where FLUX.2 buckets them.
//!
//! The VAE is the checkpoint's `vae/`. Upstream instead downloads
//! `black-forest-labs/FLUX.2-small-decoder` at load time; see
//! docs/image-generation.md for why this port does not.
//!
//! Generation only. As upstream's server does, `n` images run one at a time
//! with seed `seed + i`.

pub mod config;
pub mod loader;

use std::path::Path;

use anyhow::{Result, bail};
use mlxcel_core::dtype;

use self::config::PackedQuantConfig;
use self::loader::PackedLinears;
use super::Flux2Vae;
use super::flux2::config::{Flux2TransformerConfig, Flux2Variant, validate_dimensions};
use super::flux2::pipeline::Flux2Pipeline;
use super::flux2::{Flux2Model, run_options};
use super::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};

/// Probe family name.
pub const FAMILY: &str = "bonsai";

/// Upstream `BonsaiImageGenerationModel.default_sampling` (Blaizzy/mlx-vlm#2373).
pub const DEFAULTS: ImageSamplingDefaults = ImageSamplingDefaults {
    steps: 4,
    guidance: 1.0,
};

/// Upstream builds every dense linear and the quantized matmul inputs in
/// bf16 (`DenseLinearKernel`, `QuantizedLinearKernel`).
const TRANSFORMER_DTYPE: i32 = dtype::BFLOAT16;

/// A loaded Bonsai checkpoint.
pub struct BonsaiModel {
    pipeline: Flux2Pipeline,
}

impl BonsaiModel {
    /// Check the layout and `quantization_config.json`, load the tokenizer
    /// and the text encoder. The transformer and VAE load on first use.
    pub fn load(path: &Path) -> Result<Self> {
        let layout = config::layout(path)?;
        let quant = PackedQuantConfig::from_dir(&layout.transformer)?;
        // Bonsai is the 4B Klein; the packed directory's config.json gives the
        // shape and the 4B variant only fills in a missing one.
        let variant = Flux2Variant::Klein4b;
        let transformer_config = Flux2TransformerConfig::resolve(&layout.transformer, variant)?;
        let pipeline = Flux2Pipeline::with_linear_source(
            layout,
            variant,
            transformer_config,
            Box::new(PackedLinears::new(quant)),
            TRANSFORMER_DTYPE,
            Flux2Vae::default_dtype(),
            false,
        )?;
        Ok(Self { pipeline })
    }

    /// The pipeline, for callers that need latents rather than pixels.
    #[must_use]
    pub fn pipeline(&self) -> &Flux2Pipeline {
        &self.pipeline
    }
}

/// [`super::registry::FAMILIES`] constructor.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(BonsaiModel::load(path)?))
}

impl ImageGenerationModel for BonsaiModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        task == ImageTask::Generate
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        DEFAULTS
    }

    /// Upstream `validate_dimensions` and the FLUX.2 `--gen-kwargs`.
    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        if req.task != ImageTask::Generate {
            bail!("Bonsai supports text-to-image generation only");
        }
        validate_dimensions(req.width, req.height)?;
        run_options(&req.extra)?;
        Ok(())
    }

    /// One image per seed, `req.seed + i`, as upstream's server loops.
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        self.check_request(req)?;
        let steps = req.resolve_steps(DEFAULTS);
        let guidance = req.resolve_guidance(DEFAULTS);
        let options = run_options(&req.extra)?;
        let mut out = Vec::with_capacity(req.n as usize);
        for i in 0..req.n {
            let seed = req.seed_for(i);
            let image = self.pipeline.generate(
                &req.prompt,
                req.negative_prompt.as_deref(),
                seed,
                steps,
                req.width,
                req.height,
                guidance,
                1,
                &options,
            )?;
            let (pixels, width, height) = Flux2Model::pixels(&image, 0)?;
            out.push(GeneratedImage {
                pixels,
                width,
                height,
                channels: 3,
                seed,
                revised_prompt: None,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
