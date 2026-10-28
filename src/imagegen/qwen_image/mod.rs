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

//! Qwen-Image-2.1 (`qwen_image`): text-to-image and reference-image editing,
//! a port of `mlx_vlm/models/qwen_image` at mlx-vlm `73d366ec`. The
//! original Diffusers checkpoint (`Qwen/Qwen-Image-2.1`) loads directly.
//!
//! Components: a Qwen3-VL conditioner (`text_encoder/` through mlxcel's
//! Qwen3-VL loader, the tokenizer and image processor of `processor/`), the
//! 32-block single-stream [`transformer::QwenImageTransformer`] and the
//! [`vae::QwenImageVae`] (shared with Ming-Image). The conditioner runs and
//! is dropped (with the MLX buffer cache cleared) before the transformer and
//! VAE load, which then stay resident; later requests reload the conditioner.
//!
//! An edit takes one or more references, each opened as RGBA and resized
//! with Pillow's Lanczos to an area of `output_resolution**2` (default 1024)
//! at its own aspect; the conditioner sees it composited on white and the
//! VAE encodes all four channels, so the output is RGBA and keeps the
//! reference's transparency. With no size of its own an edit is the size of
//! its last resized reference. Generation output is RGB.
//!
//! Request options (`--gen-kwargs` or extra JSON fields): `negative_prompt`
//! (used when the request has none; default `" "`, encoded only when
//! guidance > 1), and for edits `output_resolution` (256 to 2048) and
//! `use_kv_cache` (default true; see [`kv_cache`]).

pub mod blocks;
pub mod config;
pub mod kv_cache;
pub mod pipeline;
pub mod reference;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use self::config::{QwenImageOptions, QwenImageVaeConfig, TransformerConfig};
use self::kv_cache::ImageShape;
use self::pipeline::{Branch, EditInputs, Reference, Sampling};
use self::text_encoder::{Encoded, TextEncoder};
use self::transformer::QwenImageTransformer;
use self::vae::QwenImageVae;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::scheduler::DynamicShift;

/// Probe and registry family name.
pub const FAMILY: &str = "qwen_image";

/// Registry constructor: `ImageFamily { name: "qwen_image", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(QwenImageModel::load(path)?))
}

/// The transformer and VAE, loaded once and kept.
pub struct Denoiser {
    pub transformer: QwenImageTransformer,
    pub vae: QwenImageVae,
}

/// A request with every default resolved.
#[derive(Debug, Clone)]
pub struct ResolvedRequest {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    /// The negative prompt CFG encodes; `None` runs without CFG.
    pub negative: Option<String>,
    pub options: QwenImageOptions,
}

/// The conditioner outputs of one request.
pub struct Conditioning {
    pub positive: Encoded,
    pub negative: Option<Encoded>,
}

/// A Qwen-Image checkpoint. The components load lazily (see the module
/// docs) behind `RefCell`s, because [`ImageGenerationModel::generate`] takes
/// `&self`; the model is owned by one thread (the CLI or the image worker).
pub struct QwenImageModel {
    root: PathBuf,
    transformer_config: TransformerConfig,
    vae_config: QwenImageVaeConfig,
    scheduler: DynamicShift,
    text_encoder: RefCell<Option<TextEncoder>>,
    denoiser: RefCell<Option<Denoiser>>,
    /// Whether the resident VAE has its encoder (an edit needs it).
    vae_has_encoder: RefCell<bool>,
}

fn branch(e: &Encoded) -> Branch<'_> {
    Branch {
        hidden: &e.hidden,
        image_pad_mask: e.image_pad_mask.as_deref(),
    }
}

/// Check that `root` has the component directories this port reads.
fn check_layout(root: &Path) -> Result<()> {
    for path in [
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "processor/tokenizer.json",
    ] {
        if !root.join(path).exists() {
            bail!("Qwen-Image checkpoint {} is missing {path}", root.display());
        }
    }
    Ok(())
}

impl QwenImageModel {
    /// Read the configs; weights load on the first request.
    pub fn load(root: &Path) -> Result<Self> {
        check_layout(root)?;
        let vae_config = QwenImageVaeConfig::from_dir(&root.join("vae"))?;
        vae_config.validate()?;
        Ok(Self {
            root: root.to_path_buf(),
            transformer_config: TransformerConfig::from_dir(&root.join("transformer"))?,
            vae_config,
            scheduler: config::scheduler_params(root)?,
            text_encoder: RefCell::new(None),
            denoiser: RefCell::new(None),
            vae_has_encoder: RefCell::new(false),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn ensure_text_encoder(&self) -> Result<()> {
        let mut slot = self.text_encoder.borrow_mut();
        if slot.is_none() {
            *slot = Some(TextEncoder::load(&self.root)?);
        }
        Ok(())
    }

    /// Drop the conditioner and return its buffers to the device.
    fn evict_text_encoder(&self) {
        if self.text_encoder.borrow_mut().take().is_some() {
            mlxcel_core::clear_memory_cache();
        }
    }

    fn ensure_denoiser(&self, need_encoder: bool) -> Result<()> {
        let mut slot = self.denoiser.borrow_mut();
        if slot.is_none() {
            let transformer = QwenImageTransformer::load(&self.root)
                .context("cannot build the Qwen-Image transformer")?;
            *slot = Some(Denoiser {
                transformer,
                vae: self.load_vae(need_encoder)?,
            });
            *self.vae_has_encoder.borrow_mut() = need_encoder;
        } else if need_encoder && !*self.vae_has_encoder.borrow() {
            let denoiser = slot.as_mut().expect("checked above");
            denoiser.vae = self.load_vae(true)?;
            *self.vae_has_encoder.borrow_mut() = true;
        }
        Ok(())
    }

    fn load_vae(&self, include_encoder: bool) -> Result<QwenImageVae> {
        QwenImageVae::load(
            &self.root.join("vae"),
            self.vae_config.clone(),
            include_encoder,
        )
        .context("cannot build the Qwen-Image VAE")
    }

    /// Resolve and check everything about `req` that needs no weights. Every
    /// failure is an [`InvalidImageRequest`].
    pub fn resolve(&self, req: &ImageRequest) -> Result<ResolvedRequest> {
        let check = || -> Result<ResolvedRequest> {
            let options = QwenImageOptions::from_extra(&req.extra, req.task)?;
            let (width, height) = match req.task {
                ImageTask::Generate => (req.width, req.height),
                ImageTask::Edit => {
                    let (Some(first), Some(last)) =
                        (req.reference_images.first(), req.reference_images.last())
                    else {
                        bail!("At least one reference image is required");
                    };
                    // The caller fills a missing size with `default_edit_size`
                    // of the first reference; that size means "none given",
                    // and upstream then takes the last resized reference.
                    if (req.width, req.height)
                        == config::default_edit_size(first.width(), first.height())
                    {
                        config::image_dimensions(
                            options.output_resolution,
                            f64::from(last.width()) / f64::from(last.height().max(1)),
                        )
                    } else {
                        for (label, v) in [("width", req.width), ("height", req.height)] {
                            if !v.is_multiple_of(config::EDIT_MULTIPLE) {
                                bail!(
                                    "Qwen-Image edit {label} must be a multiple of {}, got {v}",
                                    config::EDIT_MULTIPLE
                                );
                            }
                        }
                        (req.width, req.height)
                    }
                }
            };
            config::validate_dimensions(width, height)?;
            if req.task == ImageTask::Edit {
                let sizes: Vec<(u32, u32)> = req
                    .reference_images
                    .iter()
                    .map(|r| {
                        config::image_dimensions(
                            options.output_resolution,
                            f64::from(r.width()) / f64::from(r.height().max(1)),
                        )
                    })
                    .collect();
                let tokens = config::edit_image_tokens(width, height, &sizes);
                if tokens > config::MAX_EDIT_IMAGE_TOKENS {
                    bail!(
                        "Qwen-Image edit needs {tokens} image tokens (output plus resized references), at most {}",
                        config::MAX_EDIT_IMAGE_TOKENS
                    );
                }
            }
            let defaults = config::defaults(req.task);
            let steps = req.resolve_steps(defaults);
            if steps < 1 {
                bail!("steps must be >= 1, got {steps}");
            }
            let guidance = req.resolve_guidance(defaults);
            let negative = (guidance > 1.0).then(|| {
                req.negative_prompt
                    .clone()
                    .or_else(|| options.negative_prompt.clone())
                    .unwrap_or_else(|| " ".to_string())
            });
            Ok(ResolvedRequest {
                width,
                height,
                steps,
                guidance,
                negative,
                options,
            })
        };
        check().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
    }

    /// Encode the prompt (and the CFG negative) of `req`, then evict the
    /// conditioner. `references` are the resized edit references.
    pub fn encode(
        &self,
        req: &ImageRequest,
        resolved: &ResolvedRequest,
        references: &[Reference],
    ) -> Result<Conditioning> {
        self.ensure_text_encoder()?;
        let result = (|| -> Result<Conditioning> {
            let slot = self.text_encoder.borrow();
            let encoder = slot.as_ref().expect("loaded above");
            let images: Vec<_> = references.iter().map(|r| r.image.clone()).collect();
            let encode = |text: &str| match req.task {
                ImageTask::Generate => encoder.encode(text),
                ImageTask::Edit => encoder.encode_edit(text, &images),
            };
            let positive = encode(&req.prompt)?;
            let negative = resolved.negative.as_deref().map(encode).transpose()?;
            Ok(Conditioning { positive, negative })
        })();
        self.evict_text_encoder();
        let conditioning = result?;
        let width = *mlxcel_core::array_shape(&conditioning.positive.hidden)
            .last()
            .context("rank")?;
        if width != self.transformer_config.context_in_dim {
            bail!(
                "text encoder width {width} does not match the transformer context_in_dim {}",
                self.transformer_config.context_in_dim
            );
        }
        Ok(conditioning)
    }

    /// Resized references of an edit request (none for generation).
    pub fn references(
        &self,
        req: &ImageRequest,
        resolved: &ResolvedRequest,
    ) -> Result<Vec<Reference>> {
        match req.task {
            ImageTask::Generate => Ok(Vec::new()),
            ImageTask::Edit => pipeline::resize_references(
                &req.reference_images,
                resolved.options.output_resolution,
            ),
        }
    }

    /// Reference latents and grids of an edit, from the resident VAE.
    pub fn reference_latents(
        &self,
        references: &[Reference],
    ) -> Result<(UniquePtr<MlxArray>, Vec<ImageShape>)> {
        self.ensure_denoiser(true)?;
        let slot = self.denoiser.borrow();
        pipeline::reference_latents(&slot.as_ref().expect("loaded above").vae, references)
    }

    /// Final latents `[1, h*w, 64]` of one image. Public for the parity tests.
    pub fn sample_latents(
        &self,
        resolved: &ResolvedRequest,
        cond: &Conditioning,
        edit: Option<(&MlxArray, &[ImageShape])>,
        seed: u64,
    ) -> Result<UniquePtr<MlxArray>> {
        self.ensure_denoiser(edit.is_some())?;
        let slot = self.denoiser.borrow();
        let denoiser = slot.as_ref().expect("loaded above");
        let positive = branch(&cond.positive);
        let negative = cond.negative.as_ref().map(branch);
        let edit_inputs = edit.map(|(latents, shapes)| EditInputs {
            latents,
            shapes,
            use_kv_cache: resolved.options.use_kv_cache,
        });
        pipeline::denoise(
            &denoiser.transformer,
            self.scheduler,
            self.vae_config.z_dim,
            &positive,
            negative.as_ref(),
            edit_inputs.as_ref(),
            Sampling {
                steps: resolved.steps,
                guidance: resolved.guidance,
                seed,
                grid_h: (resolved.height / config::DOWNSAMPLE) as i32,
                grid_w: (resolved.width / config::DOWNSAMPLE) as i32,
            },
        )
    }

    /// Decode final latents into `H * W * channels` bytes.
    pub fn decode(
        &self,
        latents: &MlxArray,
        resolved: &ResolvedRequest,
        channels: i32,
    ) -> Result<Vec<u8>> {
        let slot = self.denoiser.borrow();
        let denoiser = slot
            .as_ref()
            .context("the Qwen-Image denoiser is not loaded")?;
        pipeline::decode_to_pixels(
            &denoiser.vae,
            latents,
            (resolved.height / config::DOWNSAMPLE) as i32,
            (resolved.width / config::DOWNSAMPLE) as i32,
            channels,
        )
    }
}

impl ImageGenerationModel for QwenImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, _task: ImageTask) -> bool {
        true
    }

    fn variant(&self) -> Option<&'static str> {
        Some("qwen-image-2.1")
    }

    fn defaults(&self, task: ImageTask) -> ImageSamplingDefaults {
        config::defaults(task)
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        self.resolve(req).map(drop)
    }

    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        config::default_edit_size
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let resolved = self.resolve(req)?;
        let references = self.references(req, &resolved)?;
        let cond = self.encode(req, &resolved, &references)?;
        let edit = if references.is_empty() {
            None
        } else {
            Some(self.reference_latents(&references)?)
        };
        let channels = if req.task == ImageTask::Edit { 4 } else { 3 };
        (0..req.n)
            .map(|index| {
                let seed = req.seed_for(index);
                let latents = self.sample_latents(
                    &resolved,
                    &cond,
                    edit.as_ref().map(|(l, s)| (&**l, s.as_slice())),
                    seed,
                )?;
                let pixels = self.decode(&latents, &resolved, channels)?;
                Ok(GeneratedImage {
                    pixels,
                    width: resolved.width,
                    height: resolved.height,
                    channels: channels as u8,
                    seed,
                    revised_prompt: None,
                })
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod parity_tests;
