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

//! FLUX.2 VAE configuration, read from `vae/config.json`.
//!
//! Constructing the VAE from these fields rather than constants lets a second
//! family reuse it with a different layout, for example the FLUX.1
//! `AutoencoderKL` that Z-Image builds (16 latent channels, no
//! `post_quant_conv`, no `bn.*`): a missing `post_quant_conv` or `bn` weight
//! simply leaves that stage out.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::layers::check_block_channels;

/// Architecture of a FLUX.2-style VAE.
///
/// Field names follow Diffusers `AutoencoderKLFlux2`. Defaults are the
/// published FLUX.2 VAE.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Flux2VaeConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub latent_channels: i32,
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: i32,
    pub norm_num_groups: i32,
    pub mid_block_add_attention: bool,
    /// `eps` of the batch-norm statistics applied to packed latents.
    pub batch_norm_eps: f32,
    /// Decoder inputs are `latents / scaling_factor + shift_factor`.
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl Default for Flux2VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            mid_block_add_attention: true,
            batch_norm_eps: 1e-4,
            scaling_factor: 1.0,
            shift_factor: 0.0,
        }
    }
}

impl Flux2VaeConfig {
    /// Group-norm epsilon used throughout the encoder and decoder.
    pub const NORM_EPS: f32 = 1e-6;

    /// Parse `config.json` text.
    pub fn from_json(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text).context("invalid FLUX.2 VAE config.json")?;
        config.validate()?;
        Ok(config)
    }

    /// Read `<vae_dir>/config.json`; a missing file yields the defaults.
    pub fn from_dir(vae_dir: &Path) -> Result<Self> {
        let path = vae_dir.join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::from_json(&text)
    }

    /// Check the fields the loader depends on.
    pub fn validate(&self) -> Result<()> {
        check_block_channels(&self.block_out_channels, self.norm_num_groups)?;
        if self.latent_channels < 1 || self.in_channels < 1 || self.out_channels < 1 {
            bail!("FLUX.2 VAE channel counts must be positive");
        }
        if self.layers_per_block < 1 {
            bail!("FLUX.2 VAE layers_per_block must be positive");
        }
        if self.scaling_factor == 0.0 || !self.scaling_factor.is_finite() {
            bail!("FLUX.2 VAE scaling_factor must be finite and non-zero");
        }
        Ok(())
    }

    /// Spatial downscale between pixels and the unpatchified latent grid.
    #[must_use]
    pub fn spatial_scale(&self) -> i32 {
        1 << (self.block_out_channels.len() as u32 - 1)
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
