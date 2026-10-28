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

//! LLaDA-Image (inclusionAI LLaDA-Image and LLaDA-Image-Turbo): text-to-image,
//! VQ-conditioned generation and single-image editing.
//!
//! A port of <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/llada_image>
//! on the shared runtime. The prompt is encoded by the in-tree LLaDA-2 MoE
//! decoder ([`crate::models::Llada2MoeModel`] with
//! [`Llada2Numerics::LladaImage`](crate::models::llada2_moe::Llada2Numerics::LladaImage))
//! together with a QueryFormer and a text projection ([`conditioning`]); the
//! denoiser ([`transformer`]) is built from the shared Z-Image blocks; the
//! latents are FLUX.2 latents decoded by the shared VAE.
//!
//! Modes (`--gen-kwargs generation_mode=...`): `text` (default), `vq`, where
//! the LLaDA-2 prior first generates discrete image tokens that SigVQ embeds
//! ([`vq_sampling`], [`sigvq`]), and editing (`--task edit` or
//! `/v1/images/edits`, exactly one source image). Turbo runs 4 stochastic
//! steps at guidance 1, the base checkpoint 50 Euler steps at guidance 5.
//! Other options: `negative_prompt` (CFG only) and `stochastic_sampling`.
//!
//! Components load and unload the way upstream's pipeline does, so peak
//! memory stays near the largest component: the text encoder (about 31 GB in
//! bf16) is dropped after the prompt is encoded and before the transformer
//! (about 13 GB) loads; the transformer is dropped before decoding; loading
//! the text encoder for a new prompt drops the transformer and the VAE. Up to
//! two projected prompts are cached across requests.

pub mod conditioning;
pub mod config;
pub mod pipeline;
pub mod scheduler;
pub mod sigvq;
pub mod transformer;
pub mod vq_sampling;
pub mod weights;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

use self::conditioning::{QueryFormer, TextProjection, format_prompt};
use self::config::{
    GenerationMode, LladaOptions, MAX_TEXT_TOKENS, SchedulerConfig, SigVqConfig, TransformerConfig,
    default_edit_size, validate_dimensions, validate_model_layout,
};
use self::pipeline::{Conditioning, KeySequence};
use self::scheduler::LladaScheduler;
use self::transformer::LladaImageTransformer;
use crate::imagegen::pipeline::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
    InvalidImageRequest,
};
use crate::imagegen::vae::Flux2Vae;
use crate::imagegen::z_image::pipeline::reference_rgb;
use crate::models::Llada2MoeModel;
use crate::tokenizer::MlxcelTokenizer;

/// Probe and registry family name.
pub const FAMILY: &str = "llada_image";

/// Projected prompts kept across requests (`prompt_cache_size`).
pub const PROMPT_CACHE_SIZE: usize = 2;

/// Registry constructor: `ImageFamily { name: "llada_image", load }`.
pub fn load(path: &Path) -> Result<Box<dyn ImageGenerationModel>> {
    Ok(Box::new(LladaImageModel::load(path)?))
}

/// The prompt-side components, resident together.
struct TextStack {
    encoder: Llada2MoeModel,
    /// `None` when only the VQ prior needed the encoder.
    queryformer: Option<QueryFormer>,
    projection: Option<TextProjection>,
}

/// Everything about a request that needs no weights.
#[derive(Debug, Clone)]
pub struct ResolvedRequest {
    pub options: LladaOptions,
    pub steps: u32,
    pub guidance: f32,
    /// The negative prompt, `None` for upstream's unconditional prompt.
    pub negative_prompt: Option<String>,
}

/// A loaded LLaDA-Image checkpoint. Only configs and the tokenizer are read
/// at load; the components load on first use.
pub struct LladaImageModel {
    root: PathBuf,
    scheduler: SchedulerConfig,
    transformer_config: TransformerConfig,
    sigvq_config: SigVqConfig,
    tokenizer: MlxcelTokenizer,
    text: RefCell<Option<TextStack>>,
    transformer: RefCell<Option<LladaImageTransformer>>,
    vae: RefCell<Option<Flux2Vae>>,
    prompt_cache: RefCell<VecDeque<(Option<String>, UniquePtr<MlxArray>)>>,
}

fn release() {
    mlxcel_core::clear_memory_cache();
}

impl LladaImageModel {
    pub fn load(root: &Path) -> Result<Self> {
        let root = validate_model_layout(root)?;
        let scheduler = SchedulerConfig::from_root(&root)?;
        let transformer_config = TransformerConfig::from_dir(&root.join("transformer"))?;
        let sigvq_config = SigVqConfig::from_dir(&root.join("sigvq"))?;
        if sigvq_config.semantic_embed_dim != transformer_config.semantic_feat_dim {
            bail!(
                "SigVQ semantic_embed_dim {} does not match the transformer semantic_feat_dim {}",
                sigvq_config.semantic_embed_dim,
                transformer_config.semantic_feat_dim
            );
        }
        let tokenizer = crate::tokenizer::load_tokenizer(&root.join("tokenizer"))
            .context("cannot load the LLaDA-Image tokenizer")?;
        Ok(Self {
            root,
            scheduler,
            transformer_config,
            sigvq_config,
            tokenizer,
            text: RefCell::new(None),
            transformer: RefCell::new(None),
            vae: RefCell::new(None),
            prompt_cache: RefCell::new(VecDeque::new()),
        })
    }

    #[must_use]
    pub fn scheduler_config(&self) -> &SchedulerConfig {
        &self.scheduler
    }

    #[must_use]
    pub fn tokenizer(&self) -> &MlxcelTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Validate `req` and resolve its options and sampling plan. Every error
    /// is the caller's ([`InvalidImageRequest`]).
    pub fn resolve(&self, req: &ImageRequest) -> Result<ResolvedRequest> {
        let resolve = || -> Result<ResolvedRequest> {
            let edit = req.task == ImageTask::Edit;
            let options = LladaOptions::from_extra(&req.extra, edit)?;
            validate_dimensions(req.width, req.height, options.mode)?;
            if edit && req.reference_images.len() != 1 {
                bail!("LLaDA-Image editing requires exactly one source image");
            }
            if !edit && !req.reference_images.is_empty() {
                bail!("An input image is required only for editing mode");
            }
            if options.mode == GenerationMode::Vq {
                vq_sampling::vq_grid(req.height, req.width)?;
            }
            let defaults = self.scheduler.defaults();
            let steps = req.resolve_steps(defaults);
            let guidance = req.resolve_guidance(defaults);
            if steps < 1 || !guidance.is_finite() || guidance < 0.0 {
                bail!("steps must be positive and guidance finite and nonnegative");
            }
            let negative_prompt = req
                .negative_prompt
                .clone()
                .or_else(|| options.negative_prompt.clone());
            Ok(ResolvedRequest {
                options,
                steps,
                guidance,
                negative_prompt,
            })
        };
        resolve().map_err(|e| InvalidImageRequest(format!("{e:#}")).into())
    }

    /// Upstream `tokenize`: the formatted prompt, truncated to 2048 tokens.
    pub fn tokenize(&self, prompt: Option<&str>) -> Result<Vec<i32>> {
        let mut ids: Vec<i32> = self
            .tokenizer
            .encode(&format_prompt(prompt), true)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        ids.truncate(MAX_TEXT_TOKENS);
        Ok(ids)
    }

    /// Load the text encoder (and, when `with_conditioning`, the QueryFormer
    /// and text projection) if absent. A first load drops the transformer and
    /// the VAE, as upstream's pipeline does before reloading its encoder.
    fn ensure_text(&self, with_conditioning: bool) -> Result<()> {
        let mut slot = self.text.borrow_mut();
        if slot.is_none() {
            *self.transformer.borrow_mut() = None;
            *self.vae.borrow_mut() = None;
            release();
            *slot = Some(TextStack {
                encoder: weights::load_text_encoder(&self.root)?,
                queryformer: None,
                projection: None,
            });
        }
        let stack = slot.as_mut().expect("loaded above");
        if with_conditioning && stack.queryformer.is_none() {
            stack.queryformer = Some(weights::load_queryformer(&self.root)?);
            stack.projection = Some(weights::load_text_projection(&self.root)?);
        }
        Ok(())
    }

    fn evict_text(&self) {
        if self.text.borrow_mut().take().is_some() {
            release();
        }
    }

    /// Upstream `_encode_prompt`, cached: the projected caption of `prompt`.
    pub fn encode_prompt(&self, prompt: Option<&str>) -> Result<UniquePtr<MlxArray>> {
        let key = prompt.map(str::to_string);
        {
            let mut cache = self.prompt_cache.borrow_mut();
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(pos).expect("position is valid");
                let value = mlxcel_core::copy(&entry.1);
                cache.push_back(entry);
                return Ok(value);
            }
        }
        self.ensure_text(true)?;
        let ids = self.tokenize(prompt)?;
        let output = {
            let slot = self.text.borrow();
            let stack = slot.as_ref().expect("loaded above");
            pipeline::encode_prompt(
                &stack.encoder,
                stack.queryformer.as_ref().expect("loaded above"),
                stack.projection.as_ref().expect("loaded above"),
                &ids,
            )?
        };
        let mut cache = self.prompt_cache.borrow_mut();
        cache.push_back((key, mlxcel_core::copy(&output)));
        while cache.len() > PROMPT_CACHE_SIZE {
            cache.pop_front();
        }
        Ok(output)
    }

    /// The prior's raw VQ window for `prompt` at `width x height`.
    pub fn vq_tokens(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
    ) -> Result<vq_sampling::VqTokens> {
        self.ensure_text(false)?;
        let (ids, uncond, count) =
            vq_sampling::vq_prompt_ids(&self.tokenizer, prompt, height, width)?;
        let head = weights::load_lm_head(&self.root)?;
        let slot = self.text.borrow();
        let stack = slot.as_ref().expect("loaded above");
        let tokens = vq_sampling::generate_vq_tokens(&stack.encoder, &head, &ids, &uncond, count);
        drop(head);
        release();
        tokens
    }

    /// Upstream `_generate_semantic_features`: VQ tokens, then SigVQ
    /// features (`[1, N, semantic_feat_dim]`). The text stack is dropped
    /// after the prior runs.
    pub fn vq_features(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
    ) -> Result<UniquePtr<MlxArray>> {
        let tokens = self.vq_tokens(prompt, width, height)?;
        self.evict_text();
        let ids = tokens.codebook_ids(self.sigvq_config.codebook_size)?;
        let sigvq = weights::load_sigvq(&self.root, false)?;
        let ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
        let features = sigvq.embed_tokens(&ids);
        mlxcel_core::try_eval(&features).context("LLaDA-Image SigVQ embedding failed")?;
        Ok(features)
    }

    fn ensure_vae(&self, include_encoder: bool) -> Result<()> {
        let mut slot = self.vae.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|v| include_encoder && !v.has_encoder())
        {
            *slot = None;
            *slot = Some(weights::load_vae(&self.root, include_encoder)?);
        }
        Ok(())
    }

    /// Upstream `_encode_source_image` from RGB `u8` pixels already at
    /// `width x height`: `(source latents, semantic features)`.
    pub fn encode_source_rgb(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let pixels = pipeline::normalized_pixels(rgb, width, height);
        let sigvq = weights::load_sigvq(&self.root, true)?;
        let semantic = sigvq.features_from_pixels(&mlxcel_core::astype(
            &pipeline::half_resolution(&pixels),
            mlxcel_core::dtype::BFLOAT16,
        ))?;
        mlxcel_core::try_eval(&semantic).context("LLaDA-Image SigVQ encode failed")?;
        drop(sigvq);
        release();
        self.ensure_vae(true)?;
        let slot = self.vae.borrow();
        let vae = slot.as_ref().expect("loaded above");
        let nchw = mlxcel_core::transpose_axes(&pixels, &[0, 3, 1, 2]);
        let encoded = vae.encode(&mlxcel_core::astype(&nchw, Flux2Vae::default_dtype()))?;
        let packed = crate::imagegen::latent::patchify_latents(&encoded)?;
        let source = vae.bn_normalize(&packed)?;
        mlxcel_core::try_eval(&source).context("LLaDA-Image source image encode failed")?;
        Ok((source, semantic))
    }

    fn ensure_transformer(&self) -> Result<()> {
        let mut slot = self.transformer.borrow_mut();
        if slot.is_none() {
            *slot = Some(weights::load_transformer(&self.root)?);
        }
        Ok(())
    }

    /// Final f32 latents of one image from prepared conditioning. Public for
    /// the parity tests.
    pub fn sample_latents(
        &self,
        resolved: &ResolvedRequest,
        width: u32,
        height: u32,
        seed: u64,
        cond: &Conditioning<'_>,
    ) -> Result<UniquePtr<MlxArray>> {
        self.ensure_transformer()?;
        let slot = self.transformer.borrow();
        let transformer = slot.as_ref().expect("loaded above");
        let dt = transformer.dtype();
        let scheduler = LladaScheduler::new(self.scheduler, resolved.steps)?;
        let stochastic = resolved
            .options
            .stochastic_sampling
            .unwrap_or(self.scheduler.stochastic_sampling);
        let mut rng = KeySequence::new(seed);
        let shape = pipeline::latent_shape(self.transformer_config.in_channels, width, height);
        let latents = pipeline::initial_latents(&mut rng, &shape, dt);
        pipeline::denoise(transformer, &scheduler, stochastic, &mut rng, latents, cond)
    }

    /// Drop the transformer and decode `latents` into `H * W * 3` bytes.
    pub fn decode(&self, latents: &MlxArray) -> Result<Vec<u8>> {
        if self.transformer.borrow_mut().take().is_some() {
            release();
        }
        self.ensure_vae(false)?;
        let slot = self.vae.borrow();
        pipeline::decode_to_pixels(slot.as_ref().expect("loaded above"), latents)
    }

    fn transformer_dtype(&self) -> Result<i32> {
        self.ensure_transformer()?;
        Ok(self
            .transformer
            .borrow()
            .as_ref()
            .map(LladaImageTransformer::dtype)
            .expect("loaded above"))
    }
}

impl ImageGenerationModel for LladaImageModel {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn supports(&self, task: ImageTask) -> bool {
        matches!(task, ImageTask::Generate | ImageTask::Edit)
    }

    fn variant(&self) -> Option<&'static str> {
        Some(self.scheduler.variant())
    }

    fn defaults(&self, _task: ImageTask) -> ImageSamplingDefaults {
        self.scheduler.defaults()
    }

    fn check_request(&self, req: &ImageRequest) -> Result<()> {
        self.resolve(req).map(drop)
    }

    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        default_edit_size
    }

    /// Image `i` runs upstream's whole `generate_array` with seed `seed + i`;
    /// the prompt features, VQ features and edit source are computed once.
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>> {
        let resolved = self.resolve(req)?;
        let caption = self.encode_prompt(Some(&req.prompt))?;
        let negative = if resolved.guidance > 1.0 {
            Some(self.encode_prompt(resolved.negative_prompt.as_deref())?)
        } else {
            None
        };
        let mut semantic = None;
        if resolved.options.mode == GenerationMode::Vq {
            semantic = Some(self.vq_features(&req.prompt, req.width, req.height)?);
        }
        self.evict_text();
        let mut source = None;
        if resolved.options.mode == GenerationMode::Editing {
            let [reference] = req.reference_images.as_slice() else {
                bail!("LLaDA-Image editing requires exactly one source image");
            };
            let rgb = reference_rgb(reference, req.width, req.height);
            let (latents, features) =
                self.encode_source_rgb(rgb.as_raw(), req.width, req.height)?;
            source = Some(latents);
            semantic = Some(features);
        }
        let dt = self.transformer_dtype()?;
        let cast = |x: &MlxArray| mlxcel_core::astype(x, dt);
        let caption = cast(&caption);
        let negative = negative.as_deref().map(cast);
        let semantic = semantic.as_deref().map(cast);
        let source = source.as_deref().map(cast);
        let cond = Conditioning {
            caption: &caption,
            negative: negative.as_deref(),
            semantic: semantic.as_deref(),
            source: source.as_deref(),
            guidance: resolved.guidance,
        };
        (0..req.n)
            .map(|index| {
                let seed = req.seed_for(index);
                let latents = self.sample_latents(&resolved, req.width, req.height, seed, &cond)?;
                let pixels = self.decode(&latents)?;
                Ok(GeneratedImage {
                    pixels,
                    width: req.width,
                    height: req.height,
                    channels: 3,
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
