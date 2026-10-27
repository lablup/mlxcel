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

//! Mage-Flow (mage-flow-community): text-to-image and multi-reference image
//! editing, a port of `mlx_vlm/models/mage_flow` at mlx-vlm `73d366ec`.
//!
//! Six checkpoints share one layout: Mage-Flow, Mage-Flow-Base and
//! Mage-Flow-Turbo generate; Mage-Flow-Edit, Mage-Flow-Edit-Base and
//! Mage-Flow-Edit-Turbo edit. The variant comes from the directory name (see
//! [`config::variant_from_local_path`]) and fixes the task and the default
//! steps and guidance; a request for the other task is an
//! [`InvalidImageRequest`] naming the variant.
//!
//! Components: a Qwen3-VL conditioner (the checkpoint's `text_encoder/`,
//! loaded through mlxcel's Qwen3-VL loader), the 12-block dual-stream
//! [`transformer::MageFlowTransformer`] and the [`vae::MageVae`]. Like
//! upstream's default runtime, the conditioner is loaded for each request and
//! dropped (with the MLX buffer cache cleared) before the transformer and VAE
//! load, which then stay resident. From the second request on, the
//! conditioner is reloaded from disk while the transformer and VAE are
//! resident, so the steady-state peak is all three (about 16.5 GB of bf16
//! weights: 8.3 GB conditioner, 7.7 GB transformer, 0.3 GB VAE); only the
//! first request runs the conditioner alone.
//!
//! Request options (`--gen-kwargs` or extra JSON fields): `negative_prompt`
//! (when the request's own field is absent; default `" "`), `static_shift`
//! (default 6.0), `renormalization` (default false), and for edits
//! `max_size` (output long edge taken from the first reference),
//! `vl_cond_long_edge` (default 384, at most 2048; `null` or `<= 0` keeps
//! the reference size, capped at 2048x2048 pixels) and `sample_posterior` (default true, as upstream's pipeline).
//!
//! `max_size` sizes the output only when the request's size is the default
//! (the first reference's [`config::default_edit_size`]); an explicit size
//! wins, as upstream's explicit `width` and `height` do. A reference past the
//! shared 16-megapixel check is still rejected before `max_size` can shrink the
//! output, because that check runs on the defaulted size first. An edit is also
//! bounded to [`config::MAX_DIT_IMAGE_TOKENS`] joint DiT image tokens.

pub mod conditioner;
pub mod config;
pub mod layers;
pub mod pipeline;
pub mod transformer;
pub mod vae;
pub mod weights;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use self::conditioner::Conditioner;
use self::config::{MageFlowOptions, MageFlowVariant, TransformerConfig};
use self::pipeline::{Guidance, Sampling};
use self::transformer::MageFlowTransformer;
use self::vae::MageVae;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};

/// Probe and registry family name.
pub const FAMILY: &str = "mage_flow";

/// Registry constructor: `ImageFamily { name: "mage_flow", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(MageFlowModel::load(path)?))
}

/// The transformer and VAE, loaded once and kept.
pub struct Denoiser {
    pub transformer: MageFlowTransformer,
    pub vae: MageVae,
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
    pub options: MageFlowOptions,
}

/// The conditioner outputs of one request.
pub struct Conditioning {
    pub positive: UniquePtr<MlxArray>,
    pub negative: Option<UniquePtr<MlxArray>>,
}

/// A Mage-Flow checkpoint.
///
/// The components load lazily in upstream's order (see the module docs);
/// [`ImageGenerationModel::generate`] takes `&self`, so they live in
/// `RefCell`s. The model is owned by one thread (the CLI or the server's
/// image worker).
pub struct MageFlowModel {
    root: PathBuf,
    variant: MageFlowVariant,
    transformer_config: TransformerConfig,
    conditioner: RefCell<Option<Conditioner>>,
    denoiser: RefCell<Option<Denoiser>>,
}

/// Check that `root` has the component directories this port reads.
fn check_layout(root: &Path) -> Result<()> {
    for path in [
        "transformer/config.json",
        "vae",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
    ] {
        if !root.join(path).exists() {
            bail!("Mage-Flow checkpoint {} is missing {path}", root.display());
        }
    }
    Ok(())
}

impl MageFlowModel {
    /// Resolve the variant and read the configs; the conditioner loads now
    /// (upstream's pipeline constructor does the same), the transformer and
    /// VAE on the first request.
    pub fn load(root: &Path) -> Result<Self> {
        let model = Self::open(root)?;
        model.ensure_conditioner()?;
        Ok(model)
    }

    /// [`Self::load`] without loading any weights.
    pub fn open(root: &Path) -> Result<Self> {
        check_layout(root)?;
        let variant = config::variant_from_local_path(root)?;
        let config_path = root.join("transformer/config.json");
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("cannot read {}", config_path.display()))?,
        )
        .with_context(|| format!("cannot parse {}", config_path.display()))?;
        let transformer_config = TransformerConfig::from_json(&config)?;
        Ok(Self {
            root: root.to_path_buf(),
            variant,
            transformer_config,
            conditioner: RefCell::new(None),
            denoiser: RefCell::new(None),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn variant_spec(&self) -> MageFlowVariant {
        self.variant
    }

    fn ensure_conditioner(&self) -> Result<()> {
        let mut slot = self.conditioner.borrow_mut();
        if slot.is_none() {
            *slot = Some(Conditioner::load(&self.root)?);
        }
        Ok(())
    }

    /// Drop the conditioner and return its buffers to the device.
    fn evict_conditioner(&self) {
        if self.conditioner.borrow_mut().take().is_some() {
            mlxcel_core::clear_memory_cache();
        }
    }

    fn ensure_denoiser(&self) -> Result<()> {
        let mut slot = self.denoiser.borrow_mut();
        if slot.is_none() {
            let transformer = weights::load_transformer(&self.root)
                .context("cannot build the Mage-Flow transformer")?;
            let vae = weights::load_vae(&self.root, self.variant.task == ImageTask::Edit)
                .context("cannot build the Mage-VAE")?;
            *slot = Some(Denoiser { transformer, vae });
        }
        Ok(())
    }

    /// Resolve and check everything about `req` that needs no weights. Every
    /// failure is an [`InvalidImageRequest`].
    pub fn resolve(&self, req: &ImageRequest) -> Result<ResolvedRequest> {
        let check = || -> Result<ResolvedRequest> {
            self.variant.require(req.task)?;
            let options = MageFlowOptions::from_extra(&req.extra, req.task)?;
            // Upstream applies `max_size` only when no size is given. The
            // request always carries a size here (the caller fills in
            // `default_edit_size` of the first reference when it has none), so
            // a size that differs from that default was asked for and wins.
            let (width, height) = match (req.task, options.max_size, req.reference_images.first()) {
                (ImageTask::Edit, Some(max_size), Some(first))
                    if (req.width, req.height)
                        == config::default_edit_size(first.width(), first.height()) =>
                {
                    config::edit_dimensions(first.width(), first.height(), Some(max_size))
                }
                _ => (req.width, req.height),
            };
            config::validate_dimensions(width, height)?;
            if req.task == ImageTask::Edit && req.reference_images.is_empty() {
                bail!("At least one reference image is required");
            }
            if req.task == ImageTask::Edit {
                config::validate_edit_tokens(width, height, req.reference_images.len())?;
            }
            let defaults = self.variant.defaults();
            let steps = req.resolve_steps(defaults);
            if steps < 1 {
                bail!("steps must be >= 1, got {steps}");
            }
            let guidance = req.resolve_guidance(defaults);
            let text = req
                .negative_prompt
                .clone()
                .or_else(|| options.negative_prompt.clone())
                .unwrap_or_else(|| " ".to_string());
            let negative = match req.task {
                // `negative_prompt if guidance > 1.0 and negative_prompt else None`
                ImageTask::Generate => (guidance > 1.0 && !text.is_empty()).then_some(text),
                // `(negative_prompt or " ") if guidance > 1.0 else None`
                ImageTask::Edit => (guidance > 1.0).then(|| {
                    if text.is_empty() {
                        " ".to_string()
                    } else {
                        text
                    }
                }),
            };
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
    /// conditioner.
    pub fn encode(&self, req: &ImageRequest, resolved: &ResolvedRequest) -> Result<Conditioning> {
        self.ensure_conditioner()?;
        let result = (|| -> Result<Conditioning> {
            let slot = self.conditioner.borrow();
            let conditioner = slot.as_ref().expect("loaded above");
            let encode = |text: &str| match req.task {
                ImageTask::Generate => conditioner.encode(text),
                ImageTask::Edit => conditioner.encode_edit(
                    text,
                    &req.reference_images,
                    resolved.options.vl_cond_long_edge,
                ),
            };
            let positive = encode(&req.prompt)?;
            let negative = resolved.negative.as_deref().map(encode).transpose()?;
            Ok(Conditioning { positive, negative })
        })();
        self.evict_conditioner();
        let conditioning = result?;
        let width = *mlxcel_core::array_shape(&conditioning.positive)
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

    /// Final latents `[1, h*w, 128]` of image `index` of `req`. Public for the
    /// parity tests, which compare them against upstream.
    pub fn sample_latents(
        &self,
        req: &ImageRequest,
        resolved: &ResolvedRequest,
        cond: &Conditioning,
        index: u32,
    ) -> Result<UniquePtr<MlxArray>> {
        self.ensure_denoiser()?;
        let slot = self.denoiser.borrow();
        let denoiser = slot.as_ref().expect("loaded above");
        let guidance = Guidance {
            positive: &cond.positive,
            negative: cond.negative.as_deref(),
            scale: resolved.guidance,
            renormalization: resolved.options.renormalization,
        };
        let sampling = Sampling {
            steps: resolved.steps,
            static_shift: resolved.options.static_shift,
            seed: req.seed_for(index),
            grid_h: (resolved.height / config::DOWNSAMPLE) as i32,
            grid_w: (resolved.width / config::DOWNSAMPLE) as i32,
        };
        match req.task {
            ImageTask::Generate => {
                pipeline::denoise_generate(&denoiser.transformer, &guidance, sampling)
            }
            ImageTask::Edit => {
                let pixels = pipeline::reference_pixels(
                    &req.reference_images,
                    resolved.width,
                    resolved.height,
                )?;
                let references = pipeline::reference_tokens(
                    &denoiser.vae,
                    &pixels,
                    resolved.options.sample_posterior,
                    sampling.seed,
                )?;
                pipeline::denoise_edit(&denoiser.transformer, &guidance, sampling, &references)
            }
        }
    }

    /// Decode final latents into `H * W * 3` bytes.
    pub fn decode(&self, latents: &MlxArray, resolved: &ResolvedRequest) -> Result<Vec<u8>> {
        self.ensure_denoiser()?;
        let slot = self.denoiser.borrow();
        let denoiser = slot.as_ref().expect("loaded above");
        let decoded = pipeline::decode(
            &denoiser.vae,
            latents,
            (resolved.height / config::DOWNSAMPLE) as i32,
            (resolved.width / config::DOWNSAMPLE) as i32,
        )?;
        pipeline::to_pixels(&decoded)
    }
}

impl ImageGenerationModel for MageFlowModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        self.variant.task == task
    }

    fn variant(&self) -> Option<&'static str> {
        Some(self.variant.name)
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        self.variant.defaults()
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        self.resolve(req).map(drop)
    }

    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        config::default_edit_size
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let resolved = self.resolve(req)?;
        let cond = self.encode(req, &resolved)?;
        (0..req.n)
            .map(|index| {
                let latents = self.sample_latents(req, &resolved, &cond, index)?;
                let pixels = self.decode(&latents, &resolved)?;
                Ok(GeneratedImage {
                    pixels,
                    width: resolved.width,
                    height: resolved.height,
                    channels: 3,
                    seed: req.seed_for(index),
                    revised_prompt: None,
                })
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
