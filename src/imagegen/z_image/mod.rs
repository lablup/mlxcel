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

//! Z-Image (Tongyi-MAI Z-Image-Turbo and Z-Image): text-to-image and img2img.
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/z_image>
//! on top of the shared runtime: the [`crate::imagegen::dit_blocks`]
//! transformer blocks, the FLUX.2 VAE built from the FLUX.1 `AutoencoderKL`
//! config, the shared Qwen3 conditioner, and the static-shift flow-match
//! scheduler. The raw Diffusers tree loads directly; no conversion step.
//!
//! The variant follows `scheduler/scheduler_config.json`: `shift <= 3` is
//! Turbo (8 steps, no CFG), otherwise base (50 steps, guidance 4).
//! Model-specific request options (`--gen-kwargs` or extra JSON fields):
//! `cfg_truncation` (default 1.0), `strength` (img2img, default 0.6) and
//! `negative_prompt` (when the request's own field is absent).
//!
//! All three components stay resident (about 20 GB for the bf16 checkpoints).
//! Upstream drops the text encoder before loading the transformer; this port
//! keeps it so a server request does not reload 8 GB.

pub mod config;
pub mod pipeline;
pub mod prompt;
pub mod transformer;
pub mod weights;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use self::config::{ZImageConfig, ZImageOptions, img2img_start_index};
use self::pipeline::Conditioning;
use self::prompt::{PromptFormatter, encode_prompt};
use self::transformer::ZImageTransformer;
use crate::imagegen::dit_blocks::sanitize_transformer_weights;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::qwen3_text_encoder::Qwen3TextEncoder;
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};

/// Probe and registry family name.
pub const FAMILY: &str = "z_image";

/// A loaded Z-Image checkpoint.
pub struct ZImageModel {
    root: PathBuf,
    config: ZImageConfig,
    formatter: PromptFormatter,
    encoder: Qwen3TextEncoder,
    transformer: ZImageTransformer,
    vae: Flux2Vae,
}

/// Registry constructor: `ImageFamily { name: "z_image", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(ZImageModel::load(path)?))
}

/// Load `<root>/transformer/`: sanitize the keys, dequantize a converted
/// quantized directory, then store the weights as [`weights::WEIGHT_DTYPE`]
/// (f32 for a tensor the f16 cast would change, see [`weights::fits_f16`]).
pub fn load_transformer(root: &Path, config: &ZImageConfig) -> Result<ZImageTransformer> {
    let (weights, metadata) = weights::load_component(&root.join("transformer"))?;
    let mut weights =
        weights::dequantize_weights(sanitize_transformer_weights(weights), &metadata)?;
    let widened = weights::cast_floats(&mut weights, weights::reads_caption_stream);
    if !widened.is_empty() {
        tracing::warn!(
            "Z-Image transformer: {} tensor(s) do not fit f16 and stay f32: {}",
            widened.len(),
            widened.join(", ")
        );
    }
    ZImageTransformer::from_weights(config.transformer.clone(), &weights)
        .context("cannot build the Z-Image transformer")
}

/// Load `<root>/vae/` with its encoder (img2img needs it).
pub fn load_vae(root: &Path) -> Result<Flux2Vae> {
    let dir = root.join("vae");
    let vae_config = Flux2VaeConfig::from_dir(&dir)?;
    let (raw, metadata) = weights::load_component(&dir)?;
    let raw = weights::dequantize_weights(weights::sanitize_vae_weights(raw), &metadata)?;
    let (dtype, offending) = weights::vae_weight_dtype(&raw);
    if !offending.is_empty() {
        tracing::warn!(
            "Z-Image VAE: {} tensor(s) do not fit f16, so the VAE runs in f32: {}",
            offending.len(),
            offending.join(", ")
        );
    }
    Flux2Vae::from_raw_weights(vae_config, raw, true, dtype).context("cannot build the Z-Image VAE")
}

/// Check everything about `req` that does not need the GPU: the options,
/// the schedule for its step count, the guidance range and, for an edit,
/// the reference image count and the strength's start index. Every
/// failure is an [`InvalidImageRequest`].
pub fn validate_request(config: &ZImageConfig, req: &ImageRequest) -> Result<ZImageOptions> {
    let check = || -> Result<ZImageOptions> {
        let options = ZImageOptions::from_extra(&req.extra)?;
        let steps = req.resolve_steps(config.defaults());
        match req.task {
            ImageTask::Generate => drop(config.generate_schedule(steps)?),
            ImageTask::Edit => {
                drop(config.edit_schedule(steps)?);
                if req.reference_images.len() != 1 {
                    bail!("Z-Image img2img requires exactly one source image");
                }
                let start = img2img_start_index(steps, options.strength);
                if (steps as usize).saturating_sub(start) < 1 {
                    bail!(
                        "Z-Image strength {} produces no denoising steps for {steps} steps",
                        options.strength
                    );
                }
            }
        }
        config.check_guidance(req.resolve_guidance(config.defaults()))?;
        Ok(options)
    };
    check().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
}

impl ZImageModel {
    /// Load every component of the checkpoint at `root`.
    pub fn load(root: &Path) -> Result<Self> {
        let config = ZImageConfig::from_model_dir(root)?;
        let formatter = PromptFormatter::load(&root.join("tokenizer"))?;
        let encoder = Qwen3TextEncoder::load(&root.join("text_encoder"))
            .context("cannot load the Z-Image text encoder")?;
        if encoder.hidden_size() != config.transformer.text_embed_dim {
            bail!(
                "text encoder width {} does not match the transformer cap_feat_dim {}",
                encoder.hidden_size(),
                config.transformer.text_embed_dim
            );
        }
        let transformer = load_transformer(root, &config)?;
        let vae = load_vae(root)?;
        if vae.config().latent_channels != config.transformer.in_channels {
            bail!(
                "VAE latent_channels {} does not match the transformer in_channels {}",
                vae.config().latent_channels,
                config.transformer.in_channels
            );
        }
        Ok(Self {
            root: root.to_path_buf(),
            config,
            formatter,
            encoder,
            transformer,
            vae,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn config(&self) -> &ZImageConfig {
        &self.config
    }

    /// Caption features of `prompt` (`[1, valid_len, hidden]`).
    pub fn encode(&self, prompt: &str) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
        encode_prompt(&self.formatter, &self.encoder, prompt)
    }

    /// Final latents of image `index` of `req`, before decoding. Public for the
    /// parity tests, which compare them against upstream.
    pub fn sample_latents(
        &self,
        req: &ImageRequest,
        index: u32,
        cond: &Conditioning<'_>,
    ) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
        let steps = req.resolve_steps(self.config.defaults());
        let seed = req.seed_for(index);
        match req.task {
            ImageTask::Generate => {
                let schedule = self.config.generate_schedule(steps)?;
                let noise = pipeline::initial_noise(
                    seed,
                    self.config.transformer.in_channels,
                    req.width,
                    req.height,
                );
                pipeline::denoise(&self.transformer, &schedule, 0, noise, cond)
            }
            ImageTask::Edit => {
                let [reference] = req.reference_images.as_slice() else {
                    bail!("Z-Image img2img requires exactly one source image");
                };
                let options = ZImageOptions::from_extra(&req.extra)?;
                let schedule = self.config.edit_schedule(steps)?;
                let start = img2img_start_index(steps, options.strength);
                if (steps as usize).saturating_sub(start) < 1 {
                    bail!(
                        "Z-Image strength {} produces no denoising steps for {steps} steps",
                        options.strength
                    );
                }
                let pixels = pipeline::reference_pixels(reference, req.width, req.height);
                let image_latents = pipeline::encode_reference(&self.vae, &pixels)?;
                let latents = pipeline::noised_reference(&image_latents, &schedule, start, seed)?;
                pipeline::denoise(&self.transformer, &schedule, start, latents, cond)
            }
        }
    }

    /// Validate `req` and encode the conditioning, then run `f` with it.
    pub fn with_conditioning<T>(
        &self,
        req: &ImageRequest,
        f: impl FnOnce(&Conditioning<'_>) -> Result<T>,
    ) -> Result<T> {
        let options = validate_request(&self.config, req)?;
        let guidance = req.resolve_guidance(self.config.defaults());
        let cap_feats = self.encode(&req.prompt)?;
        let negative = if guidance > 1.0 {
            let text = req
                .negative_prompt
                .clone()
                .or(options.negative_prompt)
                .unwrap_or_default();
            Some(self.encode(&text)?)
        } else {
            None
        };
        f(&Conditioning {
            cap_feats: &cap_feats,
            negative: negative.as_deref(),
            guidance,
            cfg_truncation: options.cfg_truncation,
        })
    }
}

impl ImageGenerationModel for ZImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        matches!(task, ImageTask::Generate | ImageTask::Edit)
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        self.config.defaults()
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        self.with_conditioning(req, |cond| {
            (0..req.n)
                .map(|index| {
                    let latents = self.sample_latents(req, index, cond)?;
                    let pixels = pipeline::decode_to_pixels(&self.vae, &latents)?;
                    Ok(GeneratedImage {
                        pixels,
                        width: req.width,
                        height: req.height,
                        channels: 3,
                        seed: req.seed_for(index),
                        revised_prompt: None,
                    })
                })
                .collect()
        })
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
