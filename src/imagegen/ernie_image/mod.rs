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

//! ERNIE-Image (Baidu ERNIE-Image and ERNIE-Image-Turbo): text-to-image and
//! SDEdit img2img.
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/ernie_image>
//! on the shared runtime: the single-stream DiT in [`transformer`], the
//! in-tree Ministral 3 decoder as text conditioner and prompt enhancer
//! ([`text_encoder`], [`prompt_enhancer`]), the static-shift scheduler in
//! [`scheduler`], and the shared FLUX.2 VAE and latent patchify. The raw
//! Diffusers tree loads directly.
//!
//! Turbo runs 8 steps at guidance 1 (no CFG; edits default to guidance 3),
//! base 50 steps at guidance 4. Request options (`--gen-kwargs` or extra JSON
//! fields): `image_strength` (alias `strength`, img2img, default 0.6),
//! `negative_prompt`, `use_prompt_enhancer` and `prompt_enhancer_max_tokens`.
//!
//! The text encoder, transformer and VAE stay resident (about 24 GB in bf16);
//! the prompt enhancer (about 7 GB) loads on first use and stays. Upstream
//! drops the encoder and enhancer after each prompt; keeping them lets a
//! server answer the next request without reloading.

pub mod blocks;
pub mod config;
pub mod pipeline;
pub mod prompt_enhancer;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod weights;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use self::config::{ErnieOptions, ErnieVariant, default_edit_size, edit_start_index};
use self::pipeline::TextBatch;
use self::prompt_enhancer::{ErniePromptEnhancer, enhancer_available, should_enhance};
use self::scheduler::ErnieScheduler;
use self::text_encoder::ErnieTextEncoder;
use self::transformer::ErnieTransformer;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::vae::Flux2Vae;
use crate::imagegen::z_image::pipeline::reference_pixels;

/// Probe and registry family name.
pub const FAMILY: &str = "ernie_image";

/// A loaded ERNIE-Image checkpoint.
pub struct ErnieImageModel {
    root: PathBuf,
    variant: ErnieVariant,
    text_encoder: ErnieTextEncoder,
    transformer: ErnieTransformer,
    vae: Flux2Vae,
    has_enhancer: bool,
    enhancer: RefCell<Option<ErniePromptEnhancer>>,
}

/// Registry constructor: `ImageFamily { name: "ernie_image", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(ErnieImageModel::load(path)?))
}

/// Everything about `req` that needs no weights: the size grid, the options,
/// the reference count of an edit and the enhancer request. The error is the
/// caller's, an [`InvalidImageRequest`].
pub fn check_request(root_has_enhancer: bool, req: &ImageRequest) -> Result<ErnieOptions> {
    let check = || -> Result<ErnieOptions> {
        config::validate_dimensions(req.width, req.height)?;
        let options = ErnieOptions::from_extra(&req.extra)?;
        if req.task == ImageTask::Edit && req.reference_images.len() != 1 {
            bail!("ERNIE-Image img2img accepts exactly one source image");
        }
        should_enhance(
            options.use_prompt_enhancer,
            root_has_enhancer,
            req.task == ImageTask::Edit,
        )?;
        Ok(options)
    };
    check().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
}

impl ErnieImageModel {
    /// Load the text encoder, transformer and VAE of the checkpoint at `root`.
    pub fn load(root: &Path) -> Result<Self> {
        let variant = ErnieVariant::from_dir(root)?;
        let text_encoder = ErnieTextEncoder::load(root)?;
        let transformer = weights::load_transformer(root)?;
        let tc = transformer.config();
        if text_encoder.hidden_size() != tc.text_in_dim {
            bail!(
                "text encoder width {} does not match the transformer text_in_dim {}",
                text_encoder.hidden_size(),
                tc.text_in_dim
            );
        }
        let vae = weights::load_vae(root, true)?;
        let packed = vae.config().latent_channels * 4;
        if packed != tc.in_channels || packed != tc.out_channels {
            bail!(
                "VAE latent_channels {} (x4 packed) does not match the transformer channels {}/{}",
                vae.config().latent_channels,
                tc.in_channels,
                tc.out_channels
            );
        }
        Ok(Self {
            root: root.to_path_buf(),
            variant,
            text_encoder,
            transformer,
            vae,
            has_enhancer: enhancer_available(root),
            enhancer: RefCell::new(None),
        })
    }

    #[must_use]
    pub fn variant(&self) -> ErnieVariant {
        self.variant
    }

    #[must_use]
    pub fn transformer(&self) -> &ErnieTransformer {
        &self.transformer
    }

    #[must_use]
    pub fn text_encoder(&self) -> &ErnieTextEncoder {
        &self.text_encoder
    }

    #[must_use]
    pub fn vae(&self) -> &Flux2Vae {
        &self.vae
    }

    /// The enhanced prompt, loading `pe/` on first use.
    fn enhance(
        &self,
        prompt: &str,
        req: &ImageRequest,
        seed: u64,
        max: Option<usize>,
    ) -> Result<String> {
        let mut slot = self.enhancer.borrow_mut();
        if slot.is_none() {
            *slot = Some(ErniePromptEnhancer::load(&self.root)?);
        }
        let enhancer = slot.as_ref().context("prompt enhancer failed to load")?;
        enhancer.enhance(prompt, req.width, req.height, seed, max)
    }

    /// The text batch for `prompt` under `guidance` (negative first with CFG).
    pub fn text_batch(&self, prompt: &str, negative: &str, guidance: f32) -> Result<TextBatch> {
        let positive = self.text_encoder.encode(prompt)?;
        if guidance > 1.0 {
            let negative = self.text_encoder.encode(negative)?;
            TextBatch::new(&[&negative, &positive], guidance)
        } else {
            TextBatch::new(&[&positive], guidance)
        }
    }

    /// Final latents of one image (`[1, 128, H/16, W/16]`) before decoding.
    /// Public for the parity tests.
    pub fn sample_latents(
        &self,
        req: &ImageRequest,
        seed: u64,
        steps: u32,
        text: &TextBatch,
        options: &ErnieOptions,
    ) -> Result<UniquePtr<MlxArray>> {
        let scheduler = ErnieScheduler::new(steps)?;
        let channels = self.transformer.config().in_channels;
        match req.task {
            ImageTask::Generate => {
                let noise = pipeline::initial_noise(seed, channels, req.width, req.height);
                pipeline::denoise(&self.transformer, &scheduler, 0, noise, text)
            }
            ImageTask::Edit => {
                let [reference] = req.reference_images.as_slice() else {
                    bail!("ERNIE-Image img2img accepts exactly one source image");
                };
                let start = edit_start_index(steps, options.image_strength);
                let pixels = reference_pixels(reference, req.width, req.height);
                let latents = pipeline::noised_reference(
                    &self.vae,
                    &pixels,
                    &scheduler,
                    start,
                    seed,
                    weights::TRANSFORMER_DTYPE,
                )?;
                pipeline::denoise(&self.transformer, &scheduler, start, latents, text)
            }
        }
    }
}

impl ImageGenerationModel for ErnieImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        matches!(task, ImageTask::Generate | ImageTask::Edit)
    }

    fn defaults(&self, task: ImageTask) -> ImageSamplingDefaults {
        match task {
            ImageTask::Generate => self.variant.generate_defaults(),
            ImageTask::Edit => self.variant.edit_defaults(),
        }
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        check_request(self.has_enhancer, req).map(drop)
    }

    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        default_edit_size
    }

    /// Image `i` runs upstream's whole `generate_array` / `edit_array` with
    /// seed `seed + i`, prompt enhancement included.
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let options = check_request(self.has_enhancer, req)?;
        let defaults = self.defaults(req.task);
        let steps = req.resolve_steps(defaults);
        let guidance = req.resolve_guidance(defaults);
        let enhance = should_enhance(
            options.use_prompt_enhancer,
            self.has_enhancer,
            req.task == ImageTask::Edit,
        )?;
        // Guidance 1 disables CFG, so the negative prompt is never encoded.
        let negative = req
            .negative_prompt
            .clone()
            .or_else(|| options.negative_prompt.clone())
            .unwrap_or_default();
        let plain = if enhance {
            None
        } else {
            Some(self.text_batch(&req.prompt, &negative, guidance)?)
        };
        (0..req.n)
            .map(|index| {
                let seed = req.seed_for(index);
                let (revised, enhanced_text);
                let text = match &plain {
                    Some(text) => {
                        revised = None;
                        text
                    }
                    None => {
                        let prompt = self.enhance(
                            &req.prompt,
                            req,
                            seed,
                            options.prompt_enhancer_max_tokens,
                        )?;
                        enhanced_text = self.text_batch(&prompt, &negative, guidance)?;
                        revised = Some(prompt);
                        &enhanced_text
                    }
                };
                let latents = self.sample_latents(req, seed, steps, text, &options)?;
                let pixels = pipeline::decode_to_pixels(&self.vae, &latents)?;
                Ok(GeneratedImage {
                    pixels,
                    width: req.width,
                    height: req.height,
                    channels: 3,
                    seed,
                    revised_prompt: revised,
                })
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "parity_tests.rs"]
mod parity_tests;
