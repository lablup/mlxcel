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

//! The FLUX.2 Klein pipeline: components, prompt cache and generation.
//!
//! A port of `Flux2Image` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/pipeline.py>
//! (`#L124-L283`). As upstream: the text encoder is loaded at construction,
//! dropped after each prompt encode (`evict_text_encoder=True`) and reloaded
//! for the next unseen prompt; the transformer and VAE are loaded on the first
//! request and stay resident; encoded prompts are cached. CFG runs only when
//! guidance exceeds 1, against the negative prompt `" "`.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Context, Result};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::{Flux2Layout, Flux2TransformerConfig, Flux2Variant, QuantParams};
use super::embeddings::{AffineLinears, scalar_like};
use super::prompt::{Flux2Tokenizer, NEGATIVE_PROMPT, PromptEmbeds, encode_prompt};
use super::scheduler::Flux2Scheduler;
use super::tiling::{TilingConfig, decode_image_tiled};
use super::transformer::{Flux2Transformer, TransformerInput};
use super::weights::{load_text_encoder, load_vae};
use crate::imagegen::Flux2Vae;
use crate::imagegen::latent::prepare_packed_latents;

/// Latent channels and VAE downscale of every FLUX.2 checkpoint.
pub const LATENT_CHANNELS: i32 = 32;
pub const VAE_SCALE_FACTOR: u32 = 8;
/// Encoded prompts kept per pipeline (upstream's cache is unbounded).
const PROMPT_CACHE_ENTRIES: usize = 32;

/// Per-request knobs from `--gen-kwargs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOptions {
    pub max_sequence_length: usize,
    /// `None` is upstream's mode default (`auto` for generation, `off` for edits).
    pub tiled_vae: Option<bool>,
}

type PromptKey = (String, usize, bool);

/// Components of one FLUX.2 checkpoint, loaded and evicted as upstream does.
pub struct Flux2Pipeline {
    pub(super) layout: Flux2Layout,
    pub(super) variant: Flux2Variant,
    pub(super) transformer_config: Flux2TransformerConfig,
    pub(super) transformer_quant: Option<QuantParams>,
    pub(super) compute_dtype: i32,
    /// The VAE's dtype: the project's load policy (`Flux2Vae::default_dtype`).
    pub(super) vae_dtype: i32,
    pub(super) tokenizer: Flux2Tokenizer,
    text_encoder: RefCell<Option<crate::imagegen::Qwen3TextEncoder>>,
    pub(super) transformer: RefCell<Option<Flux2Transformer>>,
    pub(super) vae: RefCell<Option<Flux2Vae>>,
    prompt_cache: RefCell<HashMap<PromptKey, PromptEmbeds>>,
}

/// A copy of cached prompt embeddings (cheap: MLX arrays share buffers).
fn clone_embeds(p: &PromptEmbeds) -> PromptEmbeds {
    PromptEmbeds {
        embeds: mlxcel_core::copy(&p.embeds),
        text_ids: mlxcel_core::copy(&p.text_ids),
    }
}

impl Flux2Pipeline {
    /// Check the layout, read the configs and the tokenizer, and load the text
    /// encoder. The transformer and VAE load on the first request.
    pub fn new(
        layout: Flux2Layout,
        variant: Flux2Variant,
        transformer_config: Flux2TransformerConfig,
        transformer_quant: Option<QuantParams>,
        compute_dtype: i32,
        vae_dtype: i32,
    ) -> Result<Self> {
        let tokenizer = Flux2Tokenizer::load(&layout.tokenizer)?;
        let text_encoder = load_text_encoder(&layout.text_encoder, variant)?;
        Ok(Self {
            layout,
            variant,
            transformer_config,
            transformer_quant,
            compute_dtype,
            vae_dtype,
            tokenizer,
            text_encoder: RefCell::new(Some(text_encoder)),
            transformer: RefCell::new(None),
            vae: RefCell::new(None),
            prompt_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Upstream `_encode_prompt`: cached, then the text encoder is evicted.
    pub(super) fn encode(
        &self,
        prompt: &str,
        max_sequence_length: usize,
        bucketed: bool,
    ) -> Result<PromptEmbeds> {
        let key = (prompt.to_string(), max_sequence_length, bucketed);
        if let Some(hit) = self.prompt_cache.borrow().get(&key) {
            return Ok(clone_embeds(hit));
        }
        let mut slot = self.text_encoder.borrow_mut();
        if slot.is_none() {
            *slot = Some(load_text_encoder(&self.layout.text_encoder, self.variant)?);
        }
        let encoder = slot.as_ref().context("text encoder slot is empty")?;
        let embeds = encode_prompt(
            &self.tokenizer,
            encoder,
            prompt,
            max_sequence_length,
            bucketed,
        )?;
        mlxcel_core::eval(&embeds.text_ids);
        // evict_text_encoder=True
        *slot = None;
        drop(slot);
        mlxcel_core::clear_memory_cache();
        let mut cache = self.prompt_cache.borrow_mut();
        if cache.len() >= PROMPT_CACHE_ENTRIES {
            cache.clear();
        }
        cache.insert(key, clone_embeds(&embeds));
        Ok(embeds)
    }

    /// The prompt and, when guidance is above 1, the negative prompt.
    pub(super) fn encode_pair(
        &self,
        prompt: &str,
        negative: Option<&str>,
        guidance: f32,
        options: &RunOptions,
        bucketed: bool,
    ) -> Result<(PromptEmbeds, Option<PromptEmbeds>)> {
        let positive = self.encode(prompt, options.max_sequence_length, bucketed)?;
        let negative = if guidance > 1.0 {
            Some(self.encode(
                negative.unwrap_or(NEGATIVE_PROMPT),
                options.max_sequence_length,
                bucketed,
            )?)
        } else {
            None
        };
        Ok((positive, negative))
    }

    /// Load the transformer and VAE (with its encoder) if they are not resident.
    pub(super) fn ensure_transformer_and_vae(&self) -> Result<()> {
        if self.transformer.borrow().is_none() {
            let source = AffineLinears {
                quant: self.transformer_quant,
            };
            let transformer = Flux2Transformer::load(
                &self.layout.transformer,
                self.transformer_config.clone(),
                &source,
                self.compute_dtype,
            )?;
            *self.transformer.borrow_mut() = Some(transformer);
        }
        if self.vae.borrow().is_none() {
            *self.vae.borrow_mut() = Some(load_vae(&self.layout.vae, self.vae_dtype)?);
        }
        Ok(())
    }

    /// `pred = neg + g * (pos - neg)`.
    pub(super) fn cfg(
        positive: &MlxArray,
        negative: &MlxArray,
        guidance: f32,
    ) -> UniquePtr<MlxArray> {
        let diff = mlxcel_core::subtract(positive, negative);
        mlxcel_core::add(
            negative,
            &mlxcel_core::multiply(&scalar_like(guidance, &diff), &diff),
        )
    }

    /// Upstream `generate_array` for `batch` images from one noise tensor:
    /// `[batch, H, W, 3]` u8.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: Option<&str>,
        seed: u64,
        steps: u32,
        width: u32,
        height: u32,
        guidance: f32,
        batch: i32,
        options: &RunOptions,
    ) -> Result<UniquePtr<MlxArray>> {
        let packed = self.generate_latents(
            prompt, negative, seed, steps, width, height, guidance, batch, options,
        )?;
        self.decode(&packed, height, width, options.tiled_vae, true)
    }

    /// The denoising loop of [`Self::generate`]: the final patchified latents
    /// `[batch, 128, H/16, W/16]`, as upstream hands them to the VAE.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_latents(
        &self,
        prompt: &str,
        negative: Option<&str>,
        seed: u64,
        steps: u32,
        width: u32,
        height: u32,
        guidance: f32,
        batch: i32,
        options: &RunOptions,
    ) -> Result<UniquePtr<MlxArray>> {
        let (positive, negative) = self.encode_pair(prompt, negative, guidance, options, true)?;
        self.ensure_transformer_and_vae()?;
        let broadcast = |p: PromptEmbeds| {
            if batch == 1 {
                return p;
            }
            let es = mlxcel_core::array_shape(&p.embeds);
            let is = mlxcel_core::array_shape(&p.text_ids);
            PromptEmbeds {
                embeds: mlxcel_core::broadcast_to(&p.embeds, &[batch, es[1], es[2]]),
                text_ids: mlxcel_core::broadcast_to(&p.text_ids, &[batch, is[1], is[2]]),
            }
        };
        let positive = broadcast(positive);
        let negative = negative.map(broadcast);

        let noise = prepare_packed_latents(
            seed,
            height,
            width,
            batch,
            LATENT_CHANNELS,
            VAE_SCALE_FACTOR,
            self.compute_dtype,
        )?;
        let scheduler = Flux2Scheduler::new(((height / 16) * (width / 16)) as i32, steps)?;
        let transformer = self.transformer.borrow();
        let transformer = transformer.as_ref().context("transformer is not loaded")?;
        let mut latents = noise.latents;
        for (i, &timestep) in scheduler.timesteps().iter().enumerate() {
            let predict = |embeds: &PromptEmbeds| {
                transformer.forward(
                    &TransformerInput {
                        hidden: &latents,
                        encoder: &embeds.embeds,
                        timestep,
                        img_ids: &noise.latent_ids,
                        txt_ids: &embeds.text_ids,
                        guidance: None,
                    },
                    None,
                )
            };
            let mut pred = predict(&positive)?;
            if let Some(negative) = &negative {
                pred = Self::cfg(&pred, &*predict(negative)?, guidance);
            }
            latents = scheduler.step(&pred, i, &latents)?;
            mlxcel_core::eval(&latents);
        }
        let s = mlxcel_core::array_shape(&latents);
        let grid = mlxcel_core::reshape(
            &latents,
            &[s[0], noise.latent_height, noise.latent_width, s[2]],
        );
        Ok(mlxcel_core::transpose_axes(&grid, &[0, 3, 1, 2]))
    }

    /// `vae.decode_packed_latents(packed, tiling_config=_resolve_tiling(...))`
    /// followed by `_to_image_array`'s float part; returns `[B, H, W, 3]` u8.
    pub fn decode(
        &self,
        packed: &MlxArray,
        height: u32,
        width: u32,
        tiled_override: Option<bool>,
        auto_mode: bool,
    ) -> Result<UniquePtr<MlxArray>> {
        let vae = self.vae.borrow();
        let vae = vae.as_ref().context("VAE is not loaded")?;
        let tiling = match (tiled_override, auto_mode) {
            (Some(t), _) => TilingConfig::resolve(height, width, Some(t)),
            (None, true) => TilingConfig::resolve(height, width, None),
            (None, false) => None,
        };
        let packed = mlxcel_core::astype(packed, self.vae_dtype);
        let latents = vae.denormalize_packed_latents(&packed)?;
        let decoded = match tiling.filter(TilingConfig::enabled) {
            Some(config) => {
                let scale = vae.config().spatial_scale();
                let overlap = config.overlap * scale;
                decode_image_tiled(
                    &latents,
                    &|tile| vae.decode(tile),
                    (config.tile_size, config.tile_size),
                    (overlap, overlap),
                    scale,
                )?
            }
            None => vae.decode(&latents)?,
        };
        Ok(to_image_array(&decoded))
    }

    /// The transformer's dtype.
    #[must_use]
    pub fn compute_dtype(&self) -> i32 {
        self.compute_dtype
    }

    /// The VAE's dtype.
    #[must_use]
    pub fn vae_dtype(&self) -> i32 {
        self.vae_dtype
    }
}

/// Upstream `_to_image_array`: `clip(x / 2 + 0.5, 0, 1)` in `x`'s dtype, NHWC,
/// then `round(x * 255)` in f32 to u8. Returns `[B, H, W, C]` u8.
#[must_use]
pub fn to_image_array(decoded: &MlxArray) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::add(
        &mlxcel_core::divide(decoded, &scalar_like(2.0, decoded)),
        &scalar_like(0.5, decoded),
    );
    let x = mlxcel_core::clip(&x, &scalar_like(0.0, &x), &scalar_like(1.0, &x));
    let x = mlxcel_core::astype(
        &mlxcel_core::transpose_axes(&x, &[0, 2, 3, 1]),
        dtype::FLOAT32,
    );
    let x = mlxcel_core::round(&mlxcel_core::multiply(
        &x,
        &mlxcel_core::full_f32(&[1], 255.0, dtype::FLOAT32),
    ));
    let out = mlxcel_core::astype(&x, dtype::UINT8);
    mlxcel_core::eval(&out);
    out
}
