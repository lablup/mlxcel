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

//! MOSS-Image (`moss_image`): OpenMOSS's 5B bilingual text-rendering DiT
//! with JSON layout control, a port of `felix-ab/moss-image-mlx`.
//!
//! Components: the Qwen3.5-9B prompt conditioner (`text_encoder/`, through
//! mlxcel's Qwen3.5 loader), the 42-block single-stream
//! [`transformer::MossTransformer`] and the FLUX.2 VAE decoder (`vae/`,
//! reused from [`crate::imagegen::vae`]). Only text-to-image is implemented;
//! the checkpoint has no editing task.
//!
//! The three stages are loaded in turn and dropped between them — the
//! conditioner and the transformer cannot be resident at the same time on the
//! 16 GiB hosts this family targets — so one request peaks at the largest
//! stage rather than their sum. A request decodes every image after sampling,
//! which keeps that staging intact for `n > 1`.

pub mod config;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use mlxcel_core::weights::load_weights_from_dir;

use self::config::TransformerConfig;
use self::text_encoder::TextEncoder;
use self::transformer::{load_transformer, sanitize};
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};

/// Probe family name.
pub const FAMILY: &str = "moss_image";

/// The family's entry point for [`super::registry::FAMILIES`].
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(MossImageModel::load(path)?))
}

/// One MOSS-Image checkpoint directory.
pub struct MossImageModel {
    root: PathBuf,
    config: TransformerConfig,
    /// `moss_image_mlx.quantization` from the transformer config, when the
    /// checkpoint is an exported MLX tier.
    quantization: Option<serde_json::Value>,
    /// Whether the transformer weights are already in the exported MLX layout.
    exported: bool,
}

impl MossImageModel {
    /// Read the configs and validate the directory layout. No weights load
    /// here; [`ImageGenerationModel::generate`] stages them.
    pub fn load(root: &Path) -> Result<Self> {
        let transformer_dir = root.join("transformer");
        let config = TransformerConfig::from_dir(&transformer_dir)?;
        let vae = Flux2VaeConfig::from_dir(&root.join("vae"))?;
        vae.validate().context("MOSS-Image VAE config")?;
        if vae.latent_channels * 4 != config.latent_channels {
            bail!(
                "MOSS-Image VAE latent_channels {} (x4 packed) does not match the transformer's {}",
                vae.latent_channels,
                config.latent_channels
            );
        }
        ensure!(
            root.join("text_encoder/config.json").exists(),
            "MOSS-Image checkpoint has no text_encoder/config.json"
        );
        ensure!(
            root.join("tokenizer/tokenizer.json").exists(),
            "MOSS-Image checkpoint has no tokenizer/tokenizer.json"
        );
        let transformer_config: serde_json::Value =
            std::fs::read_to_string(transformer_dir.join("config.json"))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or(serde_json::Value::Null);
        let exported = transformer_config.get("moss_image_mlx");
        let quantization = exported
            .and_then(|m| m.get("quantization"))
            .filter(|q| !q.is_null())
            .cloned();
        Ok(Self {
            root: root.to_path_buf(),
            config,
            quantization,
            exported: exported.is_some(),
        })
    }

    /// The transformer shape.
    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Encode the prompt and its negative, then drop the conditioner.
    fn condition(
        &self,
        req: &ImageRequest,
        guidance: f32,
    ) -> Result<(
        mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
        Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    )> {
        let encoder = TextEncoder::load(
            &self.root.join("text_encoder"),
            &self.root.join("tokenizer"),
        )?;
        let positive = encoder.encode(&req.prompt)?;
        let negative = if guidance != 1.0 {
            let text = req.negative_prompt.clone().unwrap_or_default();
            Some(encoder.encode(&text)?)
        } else {
            None
        };
        Ok((positive, negative))
    }
}

impl ImageGenerationModel for MossImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        matches!(task, ImageTask::Generate)
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        config::defaults()
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        if !matches!(req.task, ImageTask::Generate) {
            bail!("MOSS-Image has no image-editing task");
        }
        config::validate_size(req.width, req.height)
    }

    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        self.check_request(req)?;
        let steps = req.steps.unwrap_or_else(|| config::defaults().steps);
        let guidance = req.guidance.unwrap_or_else(|| config::defaults().guidance);

        // Stage 1 — the conditioner.
        let (positive, negative) = self.condition(req, guidance)?;
        mlxcel_core::memory::clear_cache();

        // Stage 2 — the transformer.
        let weights = load_weights_from_dir(&self.root.join("transformer"))
            .map_err(|e| anyhow!("MOSS-Image transformer weights: {e}"))?;
        let weights = if self.exported {
            weights
        } else {
            sanitize(&weights)
        };
        let model = load_transformer(&weights, self.config.clone(), self.quantization.as_ref())?;
        drop(weights);
        let mut latents = Vec::with_capacity(req.n as usize);
        for index in 0..req.n {
            let noise = pipeline::initial_noise(
                req.height,
                req.width,
                self.config.latent_channels,
                req.seed_for(index),
            );
            latents.push(pipeline::sample(
                &model,
                &positive,
                negative.as_deref(),
                noise,
                steps,
                guidance,
                config::SHIFT,
                config::UNCOND_SKIP_LAYERS,
            )?);
        }
        drop(model);
        mlxcel_core::memory::clear_cache();

        // Stage 3 — the FLUX.2 VAE decoder.
        // `Flux2Vae::load` takes the MODEL root and appends `vae/` itself; passing
        // the vae directory here doubled it (`vae/vae`) and the decode failed
        // only after the conditioner and the transformer had already run.
        let vae = Flux2Vae::load(&self.root, false)?;
        let mut images = Vec::with_capacity(latents.len());
        for (index, latents) in latents.iter().enumerate() {
            let pixels = pipeline::decode_to_pixels(&vae, latents)?;
            images.push(GeneratedImage {
                pixels,
                width: req.width,
                height: req.height,
                channels: 3,
                seed: req.seed_for(index as u32),
                revised_prompt: None,
            });
        }
        Ok(images)
    }
}
