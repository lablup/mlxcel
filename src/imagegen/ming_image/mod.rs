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

//! Ming-Image-0.1-Design (inclusionAI): RGBA text-to-image.
//!
//! A port of
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/ming_image>
//! (`73d366ec`) on top of the shared runtime. The conditioner is the
//! BailingMoeV2 backbone of Ming-flash-omni run over the templated prompt
//! plus a 256-token query block, with the per-token `image_gate` router on
//! the query tokens ([`crate::models::bailing_moe::image_router`]); the
//! query hidden states go through a 28-layer bidirectional Qwen2 connector
//! into `cap_feats`, and three selected backbone layers of the text tokens
//! become `cap_feats_2`. The denoiser is the Z-Image NextDiT with the extra
//! caption stream and masked padding ([`transformer`]); the VAE is the
//! Qwen-Image VAE decoding four channels, so every image is RGBA.
//!
//! Only generation is implemented (an edit request is a 400 naming the
//! task); the model is trained for guidance 1.0, and any other guidance is
//! rejected. Image `i` of a request uses seed `seed + i`.
//!
//! The text encoder loads for each request and is dropped before the
//! transformer and VAE load (they stay resident), as Qwen-Image does: the
//! bf16 backbone alone is about 33 GB. The prompt is templated and checked
//! (block tokens, [`config::MAX_PROMPT_TOKENS`]) with a tokenizer the model
//! keeps, before that load, so a rejected prompt never pays for it.

pub mod config;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;
pub mod weights;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use self::config::MingImageConfig;
use self::text_encoder::{Encoded, PromptFormatter, TextEncoder};
use self::transformer::MingImageTransformer;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::qwen_image::vae::QwenImageVae;

/// Probe and registry family name.
pub const FAMILY: &str = "ming_image";

/// Registry constructor: `ImageFamily { name: "ming_image", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(MingImageModel::load(path)?))
}

/// The transformer and VAE, loaded once and kept.
pub struct Denoiser {
    pub transformer: MingImageTransformer,
    pub vae: QwenImageVae,
}

/// A request with every default resolved.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedRequest {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
}

/// A Ming-Image checkpoint. The components load lazily behind `RefCell`s,
/// because [`ImageGenerationModel::generate`] takes `&self`; the model is
/// owned by one thread (the CLI or the image worker).
pub struct MingImageModel {
    root: PathBuf,
    config: MingImageConfig,
    formatter: RefCell<Option<PromptFormatter>>,
    text_encoder: RefCell<Option<TextEncoder>>,
    denoiser: RefCell<Option<Denoiser>>,
}

/// Check that `root` has the component directories this port reads.
fn check_layout(root: &Path) -> Result<()> {
    for path in [
        "mllm/config.json",
        "mllm/tokenizer.json",
        "connector/config.json",
        "mlp/config.json",
        "transformer/config.json",
        "vae/config.json",
    ] {
        if !root.join(path).exists() {
            bail!("Ming-Image checkpoint {} is missing {path}", root.display());
        }
    }
    Ok(())
}

/// Load `<root>/transformer/` in the dtypes [`weights::load_transformer`]
/// chooses.
pub fn load_transformer(root: &Path, config: &MingImageConfig) -> Result<MingImageTransformer> {
    let (weights, widened) = weights::load_transformer(root)?;
    if !widened.is_empty() {
        tracing::warn!(
            "Ming-Image transformer: {} tensor(s) do not fit f16 and stay f32: {}",
            widened.len(),
            widened.join(", ")
        );
    }
    MingImageTransformer::from_weights(config.dit.clone(), &weights)
        .context("cannot build the Ming-Image transformer")
}

/// Load `<root>/vae/` (decoder only).
pub fn load_vae(root: &Path, config: &MingImageConfig) -> Result<QwenImageVae> {
    QwenImageVae::new(config.vae.clone(), weights::load_vae_weights(root)?, false)
        .context("cannot build the Ming-Image VAE")
}

/// Load the conditioner of `root`.
pub fn load_text_encoder(root: &Path, config: &MingImageConfig) -> Result<TextEncoder> {
    let weights = weights::load_text_encoder(root, config)?;
    TextEncoder::new(root, config, weights).context("cannot build the Ming-Image text encoder")
}

/// Check everything about `req` that needs no weights. Every failure is an
/// [`InvalidImageRequest`].
pub fn resolve_request(req: &ImageRequest) -> Result<ResolvedRequest> {
    let check = || -> Result<ResolvedRequest> {
        if req.task != ImageTask::Generate {
            bail!(
                "Ming-Image-0.1-Design implements text-to-image generation only, not image edits"
            );
        }
        if req.prompt.len() > config::MAX_PROMPT_BYTES {
            bail!(
                "the Ming-Image prompt is {} bytes, the limit is {}",
                req.prompt.len(),
                config::MAX_PROMPT_BYTES
            );
        }
        crate::imagegen::qwen_image::config::validate_dimensions(req.width, req.height)?;
        let defaults = config::defaults();
        let steps = req.resolve_steps(defaults);
        if steps < 1 {
            bail!("steps must be >= 1, got {steps}");
        }
        let guidance = req.resolve_guidance(defaults);
        if guidance != config::GUIDANCE {
            bail!("Ming-Image-0.1-Design is trained for guidance 1.0 (CFG off), got {guidance}");
        }
        Ok(ResolvedRequest {
            width: req.width,
            height: req.height,
            steps,
        })
    };
    check().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
}

impl MingImageModel {
    /// Read the configs; weights load on the first request.
    pub fn load(root: &Path) -> Result<Self> {
        check_layout(root)?;
        Ok(Self {
            root: root.to_path_buf(),
            config: MingImageConfig::from_model_dir(root)?,
            formatter: RefCell::new(None),
            text_encoder: RefCell::new(None),
            denoiser: RefCell::new(None),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn config(&self) -> &MingImageConfig {
        &self.config
    }

    /// Run `f` with the conditioner, loading it first if needed.
    pub fn with_text_encoder<T>(&self, f: impl FnOnce(&TextEncoder) -> Result<T>) -> Result<T> {
        {
            let mut slot = self.text_encoder.borrow_mut();
            if slot.is_none() {
                *slot = Some(load_text_encoder(&self.root, &self.config)?);
            }
        }
        let slot = self.text_encoder.borrow();
        f(slot.as_ref().expect("loaded above"))
    }

    /// Drop the conditioner and return its buffers to the device.
    pub fn evict_text_encoder(&self) {
        if self.text_encoder.borrow_mut().take().is_some() {
            mlxcel_core::clear_memory_cache();
        }
    }

    /// Upstream `_tokenize(prompt)` with the model's own tokenizer, loaded
    /// on first use and kept (it is small). A prompt with a block token or
    /// over the token limit is an [`InvalidImageRequest`].
    pub fn input_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut slot = self.formatter.borrow_mut();
        if slot.is_none() {
            *slot = Some(PromptFormatter::load(
                &self.root.join("mllm"),
                &self.config,
            )?);
        }
        slot.as_ref().expect("loaded above").input_ids(prompt)
    }

    /// Encode `prompt`, then drop the conditioner. The prompt is checked
    /// before the conditioner loads.
    pub fn encode(&self, prompt: &str) -> Result<Encoded> {
        let ids = self.input_ids(prompt)?;
        let result = self.with_text_encoder(|encoder| Ok(encoder.encode_ids(&ids)?.encoded));
        self.evict_text_encoder();
        result
    }

    /// Run `f` with the transformer and VAE, loading them first if needed.
    pub fn with_denoiser<T>(&self, f: impl FnOnce(&Denoiser) -> Result<T>) -> Result<T> {
        {
            let mut slot = self.denoiser.borrow_mut();
            if slot.is_none() {
                *slot = Some(Denoiser {
                    transformer: load_transformer(&self.root, &self.config)?,
                    vae: load_vae(&self.root, &self.config)?,
                });
            }
        }
        let slot = self.denoiser.borrow();
        f(slot.as_ref().expect("loaded above"))
    }

    /// Final f32 latents `[1, z, 1, h, w]` of one image. Public for the
    /// parity tests.
    pub fn sample_latents(
        &self,
        resolved: &ResolvedRequest,
        cond: &Encoded,
        seed: u64,
    ) -> Result<UniquePtr<MlxArray>> {
        self.with_denoiser(|d| {
            let schedule = pipeline::schedule(
                &d.transformer,
                resolved.width,
                resolved.height,
                resolved.steps,
            )?;
            let noise = pipeline::initial_noise(
                seed,
                self.config.dit.in_channels,
                resolved.width,
                resolved.height,
            );
            pipeline::denoise(&d.transformer, &schedule, noise, cond)
        })
    }

    /// RGBA bytes of decoded latents.
    pub fn decode(&self, latents: &MlxArray) -> Result<Vec<u8>> {
        self.with_denoiser(|d| pipeline::decode_to_pixels(&d.vae, latents))
    }
}

impl ImageGenerationModel for MingImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        task == ImageTask::Generate
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        config::defaults()
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        resolve_request(req).map(drop)
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let resolved = resolve_request(req)?;
        let cond = self.encode(&req.prompt)?;
        (0..req.n)
            .map(|index| {
                let seed = req.seed_for(index);
                let latents = self.sample_latents(&resolved, &cond, seed)?;
                Ok(GeneratedImage {
                    pixels: self.decode(&latents)?,
                    width: resolved.width,
                    height: resolved.height,
                    channels: 4,
                    seed,
                    revised_prompt: None,
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
