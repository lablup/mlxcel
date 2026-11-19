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

//! MOSS-Image configuration: the transformer shape, the sampling defaults and
//! the checkpoint layout, read from `transformer/config.json`.
//!
//! The transformer is a 42-block single-stream DiT with a Qwen3.5-9B prompt
//! conditioner and the FLUX.2 VAE decoder. Everything here is metadata, so a
//! request is validated before any weight is loaded.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use crate::imagegen::pipeline::ImageSamplingDefaults;

/// Pixels per latent cell: the FLUX.2 VAE's 8x plus the 2x2 patch packing.
pub const DOWNSAMPLE: u32 = 16;
/// The official size; a multiple of [`DOWNSAMPLE`].
pub const DEFAULT_SIDE: u32 = 1024;
/// Upstream `validate_dimensions` bounds (mirrors the other families).
pub const MIN_SIDE: u32 = 256;
pub const MAX_SIDE: u32 = 2048;
/// Prompt tokens the conditioner keeps (upstream `max_sequence_length`).
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// Upstream defaults: 50 steps, CFG 3.
pub const GENERATE_DEFAULTS: (u32, f32) = (50, 3.0);
/// Upstream `shift` of the flow-match schedule.
pub const SHIFT: f32 = 5.0;
/// The training grid the sigmas are snapped to.
pub const TRAINING_STEPS: u32 = 1000;
/// The block the negative branch skips (upstream `uncond_skip_layers`).
pub const UNCOND_SKIP_LAYERS: &[i32] = &[22];

/// Steps and guidance this family defaults to.
#[must_use]
pub fn defaults() -> ImageSamplingDefaults {
    let (steps, guidance) = GENERATE_DEFAULTS;
    ImageSamplingDefaults { steps, guidance }
}

fn read_json(path: &Path) -> Result<Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

fn int(c: &Value, key: &str, default: i64) -> Result<i32> {
    match c.get(key) {
        None | Some(Value::Null) => Ok(default as i32),
        Some(v) => v
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .filter(|n| *n > 0)
            .with_context(|| {
                format!("MOSS-Image config {key} must be a positive integer, got {v}")
            }),
    }
}

fn float(c: &Value, key: &str, default: f32) -> Result<f32> {
    match c.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite() && *n > 0.0)
            .with_context(|| format!("MOSS-Image config {key} must be a positive number, got {v}")),
    }
}

/// `transformer/config.json` (`MOSSImageTransformer`).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    /// Residual width (3072 in the SFT checkpoint).
    pub dim: i32,
    /// Feed-forward width (8192).
    pub ffn_dim: i32,
    /// Attention heads (24).
    pub num_heads: i32,
    /// Blocks (42).
    pub num_layers: i32,
    /// Prompt-conditioner width (4096, Qwen3.5-9B).
    pub text_dim: i32,
    /// Timestep-sinusoid width (256).
    pub freq_dim: i32,
    /// Latent channels (128).
    pub latent_channels: i32,
    /// RoPE base (10000).
    pub rope_base: f32,
}

impl TransformerConfig {
    /// `transformer/config.json` under `dir`, defaults where a key is absent.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let c = read_json(&dir.join("config.json"))?;
        let config = Self {
            dim: int(&c, "dim", 3072)?,
            ffn_dim: int(&c, "ffn_dim", 8192)?,
            num_heads: int(&c, "num_heads", 24)?,
            num_layers: int(&c, "num_layers", 42)?,
            text_dim: int(&c, "text_dim", 4096)?,
            freq_dim: int(&c, "freq_dim", 256)?,
            latent_channels: int(&c, "latent_channels", 128)?,
            rope_base: float(&c, "rope_base", 10000.0)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Reject a shape the implementation cannot run.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.dim > 0, "MOSS-Image dim must be positive");
        ensure!(
            self.dim % self.num_heads == 0,
            "MOSS-Image dim {} is not divisible by {} heads",
            self.dim,
            self.num_heads
        );
        let head_dim = self.dim / self.num_heads;
        ensure!(
            head_dim % 8 == 0,
            "MOSS-Image head dim {head_dim} must be a multiple of 8 (the 4-axis RoPE needs four bands)"
        );
        ensure!(self.ffn_dim > 0, "MOSS-Image ffn_dim must be positive");
        ensure!(
            self.num_layers > 0,
            "MOSS-Image num_layers must be positive"
        );
        ensure!(self.text_dim > 0, "MOSS-Image text_dim must be positive");
        ensure!(
            self.freq_dim % 2 == 0,
            "MOSS-Image freq_dim {} must be even (cos then sin)",
            self.freq_dim
        );
        ensure!(
            self.latent_channels > 0,
            "MOSS-Image latent_channels must be positive"
        );
        if let Some(&layer) = UNCOND_SKIP_LAYERS.first() {
            ensure!(
                layer < self.num_layers,
                "MOSS-Image skip layer {layer} is outside 0..{}",
                self.num_layers
            );
        }
        Ok(())
    }

    /// The per-head attention width.
    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.dim / self.num_heads
    }
}

/// Validate the request's `width`/`height` against this family.
pub fn validate_size(width: u32, height: u32) -> Result<()> {
    if width < MIN_SIDE || height < MIN_SIDE || width > MAX_SIDE || height > MAX_SIDE {
        bail!("MOSS-Image size {width}x{height} is outside {MIN_SIDE}..={MAX_SIDE}");
    }
    if !width.is_multiple_of(DOWNSAMPLE) || !height.is_multiple_of(DOWNSAMPLE) {
        bail!("MOSS-Image width and height must be multiples of {DOWNSAMPLE}");
    }
    Ok(())
}
