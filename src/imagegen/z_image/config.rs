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

//! Z-Image configuration, read from the Diffusers JSON files only.
//!
//! A port of `ZImageConfig` and `ZImageTransformerConfig` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/config.py>,
//! plus the request options and step arithmetic of `pipeline.py`. Nothing here
//! reads a weight, so defaults and request checks run before any load.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::imagegen::pipeline::ImageSamplingDefaults;
use crate::imagegen::scheduler::FlowMatchEulerDiscrete;

/// Scheduler `shift` at or below which a checkpoint is the Turbo variant.
const TURBO_MAX_SHIFT: f32 = 3.0;
/// Upstream's `ZImageConfig.scheduler_shift` default.
const DEFAULT_SHIFT: f32 = 3.0;
/// Default img2img `strength`.
pub const DEFAULT_STRENGTH: f64 = 0.6;

/// Turbo (distilled, no CFG) or base (CFG) checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Turbo,
    Base,
}

impl Variant {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Turbo => "turbo",
            Variant::Base => "base",
        }
    }
}

/// `transformer/config.json` (Diffusers `ZImageTransformer2DModel` names).
#[derive(Debug, Clone, PartialEq)]
pub struct ZImageTransformerConfig {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub in_channels: i32,
    pub text_embed_dim: i32,
    pub num_hidden_layers: i32,
    pub n_refiner_layers: i32,
    pub n_context_refiner_layers: i32,
    pub patch_size: i32,
    pub f_patch_size: i32,
    pub adaln_embed_dim: i32,
    pub rope_sections: Vec<i32>,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub timestep_scale: f32,
}

impl Default for ZImageTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3840,
            num_attention_heads: 30,
            num_key_value_heads: 30,
            in_channels: 16,
            text_embed_dim: 2560,
            num_hidden_layers: 30,
            n_refiner_layers: 2,
            n_context_refiner_layers: 2,
            patch_size: 2,
            f_patch_size: 1,
            adaln_embed_dim: 256,
            rope_sections: vec![32, 48, 48],
            rope_theta: 256.0,
            norm_eps: 1e-5,
            timestep_scale: 1000.0,
        }
    }
}

fn int(config: &Map<String, Value>, key: &str, default: i32) -> Result<i32> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .with_context(|| format!("Z-Image config field {key} must be an integer, got {v}")),
    }
}

fn float(config: &Map<String, Value>, key: &str, default: f32) -> Result<f32> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .map(|n| n as f32)
            .with_context(|| format!("Z-Image config field {key} must be a number, got {v}")),
    }
}

/// First element of a list field such as `all_patch_size: [2]`.
fn first_int(config: &Map<String, Value>, key: &str, default: i32) -> Result<i32> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Array(items)) => items
            .first()
            .and_then(Value::as_i64)
            .and_then(|n| i32::try_from(n).ok())
            .with_context(|| format!("Z-Image config field {key} must list an integer")),
        Some(v) => bail!("Z-Image config field {key} must be a list, got {v}"),
    }
}

impl ZImageTransformerConfig {
    /// Parse a `transformer/config.json` object; absent fields take upstream's defaults.
    pub fn from_value(config: &Value) -> Result<Self> {
        let config = config
            .as_object()
            .context("transformer/config.json must be a JSON object")?;
        let d = Self::default();
        let n_refiner_layers = int(config, "n_refiner_layers", d.n_refiner_layers)?;
        let rope_sections = match config.get("axes_dims") {
            None | Some(Value::Null) => d.rope_sections.clone(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| v.as_i64().and_then(|n| i32::try_from(n).ok()))
                .collect::<Option<Vec<_>>>()
                .context("Z-Image axes_dims must be a list of integers")?,
            Some(v) => bail!("Z-Image axes_dims must be a list, got {v}"),
        };
        let parsed = Self {
            hidden_size: int(config, "dim", d.hidden_size)?,
            num_attention_heads: int(config, "n_heads", d.num_attention_heads)?,
            num_key_value_heads: int(config, "n_kv_heads", d.num_key_value_heads)?,
            in_channels: int(config, "in_channels", d.in_channels)?,
            text_embed_dim: int(config, "cap_feat_dim", d.text_embed_dim)?,
            num_hidden_layers: int(config, "n_layers", d.num_hidden_layers)?,
            n_refiner_layers,
            n_context_refiner_layers: int(config, "n_context_refiner_layers", n_refiner_layers)?,
            patch_size: first_int(config, "all_patch_size", d.patch_size)?,
            f_patch_size: first_int(config, "all_f_patch_size", d.f_patch_size)?,
            adaln_embed_dim: int(config, "adaln_embed_dim", d.adaln_embed_dim)?,
            rope_sections,
            rope_theta: float(config, "rope_theta", d.rope_theta)?,
            norm_eps: float(config, "norm_eps", d.norm_eps)?,
            timestep_scale: float(config, "t_scale", d.timestep_scale)?,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    /// The checks upstream's `ZImageTransformer.__init__` makes, plus positivity.
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("dim", self.hidden_size),
            ("n_heads", self.num_attention_heads),
            ("in_channels", self.in_channels),
            ("cap_feat_dim", self.text_embed_dim),
            ("patch_size", self.patch_size),
            ("f_patch_size", self.f_patch_size),
            ("adaln_embed_dim", self.adaln_embed_dim),
        ];
        if let Some((name, value)) = positive.iter().find(|(_, v)| *v < 1) {
            bail!("Z-Image {name} must be positive, got {value}");
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            bail!(
                "Z-Image dim {} is not divisible by n_heads {}",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if self.num_key_value_heads != self.num_attention_heads {
            bail!("Z-Image grouped-query attention is not supported");
        }
        if self.rope_sections.iter().sum::<i32>() != self.head_dim() {
            bail!("Z-Image RoPE sections must sum to the attention head dimension");
        }
        Ok(())
    }

    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// `min(dim, adaln_embed_dim)`, the width of the timestep embedding.
    #[must_use]
    pub fn adaln_dim(&self) -> i32 {
        self.hidden_size.min(self.adaln_embed_dim)
    }
}

/// Everything about a checkpoint that does not need a weight.
#[derive(Debug, Clone, PartialEq)]
pub struct ZImageConfig {
    pub transformer: ZImageTransformerConfig,
    pub scheduler_shift: f32,
    pub variant: Variant,
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Missing Z-Image config: {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

impl ZImageConfig {
    /// Read `transformer/config.json` and `scheduler/scheduler_config.json`.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let transformer = ZImageTransformerConfig::from_value(&read_json(
            &root.join("transformer/config.json"),
        )?)?;
        let scheduler = read_json(&root.join("scheduler/scheduler_config.json"))?;
        let shift = match scheduler.get("shift") {
            None | Some(Value::Null) => DEFAULT_SHIFT,
            Some(v) => v
                .as_f64()
                .with_context(|| format!("scheduler shift must be a number, got {v}"))?
                as f32,
        };
        Self::new(transformer, shift)
    }

    pub fn new(transformer: ZImageTransformerConfig, scheduler_shift: f32) -> Result<Self> {
        if !scheduler_shift.is_finite() || scheduler_shift <= 0.0 {
            bail!("Z-Image scheduler shift must be finite and positive, got {scheduler_shift}");
        }
        let variant = if scheduler_shift <= TURBO_MAX_SHIFT {
            Variant::Turbo
        } else {
            Variant::Base
        };
        Ok(Self {
            transformer,
            scheduler_shift,
            variant,
        })
    }

    /// Turbo 8 steps, guidance 0; base 50 steps, guidance 4.
    ///
    /// Upstream's edit model uses 8 for Turbo and its generation model reads
    /// `config.default_steps` (9); this port uses 8 for both, as the issue
    /// specifies.
    #[must_use]
    pub fn defaults(&self) -> ImageSamplingDefaults {
        let (steps, guidance) = match self.variant {
            Variant::Turbo => (8, 0.0),
            Variant::Base => (50, 4.0),
        };
        ImageSamplingDefaults { steps, guidance }
    }

    /// Text-to-image schedule: `steps - 1` static-shift steps (upstream skips
    /// the terminal zero timestep), so `steps` must be at least 2.
    pub fn generate_schedule(&self, steps: u32) -> Result<FlowMatchEulerDiscrete> {
        if steps < 2 {
            bail!("Z-Image steps must be at least 2, got {steps}");
        }
        FlowMatchEulerDiscrete::static_shift(steps - 1, self.scheduler_shift)
    }

    /// img2img schedule: all `steps` static-shift steps.
    pub fn edit_schedule(&self, steps: u32) -> Result<FlowMatchEulerDiscrete> {
        if steps < 1 {
            bail!("Z-Image steps must be at least 1, got {steps}");
        }
        FlowMatchEulerDiscrete::static_shift(steps, self.scheduler_shift)
    }

    /// Reject a guidance Turbo cannot honor (upstream's message).
    pub fn check_guidance(&self, guidance: f32) -> Result<()> {
        if self.variant == Variant::Turbo && !(0.0..=1.0).contains(&guidance) {
            bail!(
                "Z-Image Turbo does not support classifier-free guidance; \
                 use a guidance value between 0 and 1 to keep it disabled"
            );
        }
        Ok(())
    }
}

/// `_img2img_start_index`: `int(max(steps - min(steps * strength, steps), 0))`,
/// in f64 like the Python float arithmetic it ports.
#[must_use]
pub fn img2img_start_index(steps: u32, strength: f64) -> usize {
    let steps = f64::from(steps);
    (steps - (steps * strength).min(steps)).max(0.0) as usize
}

/// The model-specific request knobs (`--gen-kwargs` / extra JSON fields).
#[derive(Debug, Clone, PartialEq)]
pub struct ZImageOptions {
    pub cfg_truncation: f64,
    pub strength: f64,
    /// `negative_prompt` from `extra`, used when the request field is absent.
    pub negative_prompt: Option<String>,
}

fn extra_number(extra: &Map<String, Value>, key: &str, default: f64) -> Result<f64> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .with_context(|| format!("Z-Image {key} must be a number, got {v}")),
    }
}

impl ZImageOptions {
    /// Parse and range-check `extra`.
    pub fn from_extra(extra: &Map<String, Value>) -> Result<Self> {
        let cfg_truncation = extra_number(extra, "cfg_truncation", 1.0)?;
        if !(0.0..=1.0).contains(&cfg_truncation) {
            bail!("Z-Image cfg_truncation must be in [0, 1], got {cfg_truncation}");
        }
        let strength = extra_number(extra, "strength", DEFAULT_STRENGTH)?;
        if !(strength > 0.0 && strength <= 1.0) {
            bail!("Z-Image strength must be in (0, 1], got {strength}");
        }
        let negative_prompt = match extra.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("Z-Image negative_prompt must be a string, got {v}"),
        };
        if let Some(n) = extra.get("num_images") {
            bail!("Z-Image num_images is not accepted; use n (got {n})");
        }
        Ok(Self {
            cfg_truncation,
            strength,
            negative_prompt,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
