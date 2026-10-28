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

//! Ideogram 4 text-to-image (family `ideogram4`).
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/ideogram4>
//! for the `ideogram-ai/ideogram-4-fp8` checkpoint (license-gated, Ideogram 4
//! Non-Commercial): a 34-layer single-stream DiT conditioned on 13 stacked
//! hidden states of the Qwen3-VL 8B text tower (through the shared
//! [`crate::imagegen::Qwen3TextEncoder`]), a second unconditional
//! transformer for classifier-free guidance, the FLUX.2 VAE, and the
//! logit-normal sampler presets. Weights ship as per-row FP8 and are
//! dequantized to bf16 at load ([`weights::dequantize_fp8_rowwise`]).
//! Editing is not supported (upstream has none).
//!
//! Request options (`--gen-kwargs` or extra JSON fields): `sampler_preset`
//! (`V4_DEFAULT_20` default, `V4_QUALITY_48`, `V4_TURBO_12`),
//! `auto_json_caption` (default true: plain text is wrapped into a minimal
//! JSON caption and returned as `revised_prompt`), `guidance_schedule`, `mu`
//! and `std`. An explicit `--steps` or `--guidance` overrides the preset as
//! upstream does. `prompt_expansion_model` and a non-empty negative prompt
//! are rejected.
//!
//! Memory: as upstream, the text encoder (about 16 GB in bf16) is loaded for
//! each request and dropped after encoding, before the two transformers
//! (about 18.6 GB each in bf16) and the VAE load; those stay resident.

pub mod blocks;
pub mod config;
pub mod latent_norm;
pub mod pipeline;
pub mod prompting;
pub mod scheduler;
pub mod transformer;
pub mod weights;

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use self::config::{Ideogram4Options, Ideogram4TransformerConfig, SamplingPlan};
use self::pipeline::{Denoiser, PromptTokenizer, SequenceLayout};
use self::prompting::{NormalizedPrompt, normalize_prompt};
use self::scheduler::get_preset;
use self::transformer::Ideogram4Transformer;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::vae::{Flux2Vae, Flux2VaeConfig};

/// Probe and registry family name.
pub const FAMILY: &str = "ideogram4";

/// Files a usable snapshot must contain (upstream `validate_model_layout`).
const REQUIRED_FILES: &[&str] = &[
    "model_index.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "unconditional_transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
];

/// A loaded Ideogram 4 checkpoint. Weights load on first use.
pub struct Ideogram4Model {
    root: PathBuf,
    tokenizer: PromptTokenizer,
    conditional: RefCell<Option<Ideogram4Transformer>>,
    unconditional: RefCell<Option<Ideogram4Transformer>>,
    vae: RefCell<Option<Flux2Vae>>,
}

/// Registry constructor: `ImageFamily { name: "ideogram4", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(Ideogram4Model::load(path)?))
}

/// Everything about one request that needs no weights.
#[derive(Debug, Clone)]
pub struct PreparedRequest {
    pub prompt: NormalizedPrompt,
    pub token_ids: Vec<i32>,
    pub plan: SamplingPlan,
}

/// Check that `root` is an Ideogram 4 snapshot (upstream
/// `validate_model_layout` plus `variant_from_local_path`).
fn validate_layout(root: &Path) -> Result<()> {
    let missing: Vec<&str> = REQUIRED_FILES
        .iter()
        .copied()
        .filter(|f| !root.join(f).is_file())
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} is missing required Ideogram 4 files: {}",
            root.display(),
            missing.join(", ")
        );
    }
    let path = root.join("model_index.json");
    let index: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .with_context(|| format!("cannot parse {}", path.display()))?;
    if index.get("_class_name").and_then(Value::as_str) != Some("Ideogram4Pipeline") {
        bail!("{} is not an Ideogram4Pipeline snapshot", root.display());
    }
    Ok(())
}

impl Ideogram4Model {
    /// Validate the snapshot, read every component config and load the
    /// tokenizer. No weights are read here.
    pub fn load(root: &Path) -> Result<Self> {
        validate_layout(root)?;
        let cond = Ideogram4TransformerConfig::from_dir(&root.join("transformer"))?;
        let uncond = Ideogram4TransformerConfig::from_dir(&root.join("unconditional_transformer"))?;
        if cond.in_channels != uncond.in_channels {
            bail!("the two Ideogram 4 transformers disagree on in_channels");
        }
        let vae = Flux2VaeConfig::from_dir(&root.join("vae"))?;
        if (vae.latent_channels * 4) as usize != latent_norm::LATENT_DIM
            || cond.in_channels as usize != latent_norm::LATENT_DIM
        {
            bail!(
                "Ideogram 4 expects 128 latent channels (VAE latent_channels {} x 4, transformer in_channels {})",
                vae.latent_channels,
                cond.in_channels
            );
        }
        Ok(Self {
            root: root.to_path_buf(),
            tokenizer: PromptTokenizer::load(&root.join("tokenizer"))?,
            conditional: RefCell::new(None),
            unconditional: RefCell::new(None),
            vae: RefCell::new(None),
        })
    }

    /// Validate `req` and resolve its prompt, tokens and sampling plan. Every
    /// error is the caller's ([`InvalidImageRequest`]).
    pub fn prepare(&self, req: &ImageRequest) -> Result<PreparedRequest> {
        let prepare = || -> Result<PreparedRequest> {
            if req.task != ImageTask::Generate {
                bail!("Ideogram 4 supports text-to-image generation only");
            }
            config::validate_dimensions(req.width, req.height)?;
            if req.prompt.trim().is_empty() {
                bail!("prompt must not be empty");
            }
            // Upstream's unconditional transformer takes no text, so a
            // negative prompt would be silently dropped; refuse it instead.
            if req
                .negative_prompt
                .as_deref()
                .is_some_and(|n| !n.trim().is_empty())
            {
                bail!("Ideogram 4 has no negative prompt: its unconditional branch takes no text");
            }
            let options = Ideogram4Options::from_extra(&req.extra)?;
            let plan = options.plan(req.steps, req.guidance)?;
            let prompt = normalize_prompt(&req.prompt, options.auto_json_caption)?;
            let token_ids = self.tokenizer.token_ids(&prompt.text)?;
            Ok(PreparedRequest {
                prompt,
                token_ids,
                plan,
            })
        };
        prepare().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
    }

    /// Load the two transformers and the VAE if they are not resident, one
    /// at a time.
    fn ensure_transformers_and_vae(&self) -> Result<()> {
        if self.conditional.borrow().is_none() {
            let t = weights::load_transformer(&self.root, "transformer")?;
            *self.conditional.borrow_mut() = Some(t);
        }
        if self.unconditional.borrow().is_none() {
            let t = weights::load_transformer(&self.root, "unconditional_transformer")?;
            *self.unconditional.borrow_mut() = Some(t);
        }
        if self.vae.borrow().is_none() {
            *self.vae.borrow_mut() = Some(weights::load_vae(&self.root)?);
        }
        Ok(())
    }

    /// Upstream `_encode_text` with its encoder lifecycle: load the text
    /// encoder, encode, drop it and clear the MLX cache.
    pub fn encode_text(
        &self,
        token_ids: &[i32],
        num_image_tokens: i32,
    ) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
        let encoder = weights::load_text_encoder(&self.root)?;
        let features = pipeline::text_features(&encoder, token_ids, num_image_tokens);
        drop(encoder);
        mlxcel_core::clear_memory_cache();
        features
    }

    /// Final latents (`[1, N, 128]` f32) of one image, and the layout they
    /// decode with. Public for the parity tests.
    pub fn sample(
        &self,
        prepared: &PreparedRequest,
        width: u32,
        height: u32,
        seeds: &[u64],
    ) -> Result<(
        SequenceLayout,
        Vec<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    )> {
        let layout = SequenceLayout::new(prepared.token_ids.len(), width, height);
        let features = self.encode_text(&prepared.token_ids, layout.num_image_tokens)?;
        self.ensure_transformers_and_vae()?;
        let conditional = self.conditional.borrow();
        let unconditional = self.unconditional.borrow();
        let (Some(conditional), Some(unconditional)) =
            (conditional.as_ref(), unconditional.as_ref())
        else {
            bail!("Ideogram 4 transformers failed to load");
        };
        let llm = conditional.project_llm_features(&features, &layout.indicator)?;
        drop(features);
        mlxcel_core::try_eval(llm.array()).context("Ideogram 4 text projection failed")?;
        let denoiser = Denoiser {
            conditional,
            unconditional,
            llm: &llm,
            layout: &layout,
        };
        let latents = seeds
            .iter()
            .map(|&seed| {
                let noise = pipeline::initial_noise(seed, layout.num_image_tokens);
                denoiser.denoise(noise, &prepared.plan, width, height)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((layout, latents))
    }

    /// Decode final latents to HWC RGB bytes.
    pub fn decode(
        &self,
        latents: &mlxcel_core::MlxArray,
        layout: &SequenceLayout,
    ) -> Result<Vec<u8>> {
        let vae = self.vae.borrow();
        let vae = vae.as_ref().context("the Ideogram 4 VAE is not loaded")?;
        pipeline::decode_to_pixels(vae, latents, layout.grid_h, layout.grid_w)
    }
}

impl ImageGenerationModel for Ideogram4Model {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        task == ImageTask::Generate
    }

    /// The default preset's step count and final guidance (upstream
    /// `_preset_defaults`): 20 steps, guidance 7.
    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        let preset = get_preset(None).expect("the default preset exists");
        let guidance = preset.guidance_schedule.last().copied().unwrap_or(7.0) as f32;
        ImageSamplingDefaults::new(preset.num_steps, guidance)
            .expect("the default preset is a valid sampling default")
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        self.prepare(req).map(drop)
    }

    /// Image `i` is upstream's `generate_array` with seed `seed + i`; the
    /// prompt is encoded once.
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let prepared = self.prepare(req)?;
        let seeds: Vec<u64> = (0..req.n).map(|i| req.seed_for(i)).collect();
        let (layout, latents) = self.sample(&prepared, req.width, req.height, &seeds)?;
        let revised = prepared.prompt.revised_prompt();
        seeds
            .iter()
            .zip(&latents)
            .map(|(&seed, z)| {
                Ok(GeneratedImage {
                    pixels: self.decode(z, &layout)?,
                    width: req.width,
                    height: req.height,
                    channels: 3,
                    seed,
                    revised_prompt: revised.clone(),
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
