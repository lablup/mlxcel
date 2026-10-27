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

//! ERNIE-Image configuration, variants and request options.
//!
//! A port of `config.py` and the request arithmetic of `pipeline.py` and
//! `model.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/ernie_image>.
//! Nothing here reads a weight, so defaults and request checks run before any
//! load.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::imagegen::pipeline::ImageSamplingDefaults;

/// Default img2img `image_strength` (`model.py#L212`).
pub const DEFAULT_IMAGE_STRENGTH: f64 = 0.6;

/// Longest prompt the conditioner reads, in tokens (`max_sequence_length`).
pub const MAX_TEXT_TOKENS: usize = 2048;

/// `ErnieImageTransformerConfig` (`transformer/config.json`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ErnieTransformerConfig {
    pub hidden_size: i32,
    pub ffn_hidden_size: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: i32,
    pub num_attention_heads: i32,
    pub patch_size: i32,
    pub qk_layernorm: bool,
    pub rope_axes_dim: Vec<i32>,
    pub rope_theta: f32,
    pub text_in_dim: i32,
    pub eps: f32,
}

impl Default for ErnieTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            ffn_hidden_size: 12288,
            in_channels: 128,
            out_channels: 128,
            num_layers: 36,
            num_attention_heads: 32,
            patch_size: 1,
            qk_layernorm: true,
            rope_axes_dim: vec![32, 48, 48],
            rope_theta: 256.0,
            text_in_dim: 3072,
            eps: 1e-6,
        }
    }
}

impl ErnieTransformerConfig {
    /// Parse `config.json` text and check it the way upstream's `from_dict` does.
    pub fn from_json(text: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(text).context("invalid ERNIE-Image transformer config.json")?;
        config.validate()?;
        Ok(config)
    }

    /// Read `<transformer_dir>/config.json`; a missing file yields the defaults.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::from_json(&text)
    }

    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size < 1 || self.num_attention_heads < 1 || self.num_layers < 1 {
            bail!("ERNIE-Image hidden_size, num_attention_heads and num_layers must be positive");
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            bail!("hidden_size must be divisible by num_attention_heads");
        }
        if self.patch_size < 1 {
            bail!("ERNIE-Image patch_size must be positive");
        }
        if self.rope_axes_dim.len() != 3 || self.rope_axes_dim.iter().any(|d| *d < 2 || d % 2 != 0)
        {
            bail!(
                "rope_axes_dim must be three positive even widths, got {:?}",
                self.rope_axes_dim
            );
        }
        if self.rope_axes_dim.iter().sum::<i32>() != self.head_dim() {
            bail!(
                "rope_axes_dim must sum to the attention head dimension: {:?} vs {}",
                self.rope_axes_dim,
                self.head_dim()
            );
        }
        Ok(())
    }
}

/// Base (`baidu/ERNIE-Image`) or Turbo (`baidu/ERNIE-Image-Turbo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErnieVariant {
    Base,
    Turbo,
}

impl ErnieVariant {
    /// Upstream's variant name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "ernie-image",
            Self::Turbo => "ernie-image-turbo",
        }
    }

    /// Generation defaults: 50 steps at guidance 4, or 8 steps at guidance 1.
    #[must_use]
    pub fn generate_defaults(self) -> ImageSamplingDefaults {
        match self {
            Self::Base => ImageSamplingDefaults {
                steps: 50,
                guidance: 4.0,
            },
            Self::Turbo => ImageSamplingDefaults {
                steps: 8,
                guidance: 1.0,
            },
        }
    }

    /// Edit defaults: the generation steps with `edit_default_guidance`
    /// (Turbo raises guidance to 3 because SDEdit barely edits without CFG).
    #[must_use]
    pub fn edit_defaults(self) -> ImageSamplingDefaults {
        let generate = self.generate_defaults();
        match self {
            Self::Base => generate,
            Self::Turbo => ImageSamplingDefaults {
                steps: generate.steps,
                guidance: 3.0,
            },
        }
    }

    /// Upstream `get_variant` for a name or repo id (case-insensitive).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let key = name.trim().trim_end_matches('/').to_lowercase();
        match key.as_str() {
            "ernie-image" | "baidu/ernie-image" | "ernie-image-base" => Some(Self::Base),
            "ernie-image-turbo" | "baidu/ernie-image-turbo" | "ernie-turbo" => Some(Self::Turbo),
            _ => None,
        }
    }

    /// Upstream `variant_from_local_path`: `mlx_ernie_image.json`'s `source`
    /// or `variant` when present, else the directory name (Turbo when it
    /// contains `turbo`). The two published checkpoints share every config
    /// file that would otherwise tell them apart (the scheduler `shift` and the
    /// transformer shape are identical), so the name is all upstream has too.
    pub fn from_dir(root: &Path) -> Result<Self> {
        let metadata = root.join("mlx_ernie_image.json");
        if metadata.exists() {
            let text = std::fs::read_to_string(&metadata)
                .with_context(|| format!("cannot read {}", metadata.display()))?;
            let value: Value = serde_json::from_str(&text)
                .with_context(|| format!("invalid JSON in {}", metadata.display()))?;
            for key in ["source", "variant"] {
                if let Some(variant) = value
                    .get(key)
                    .and_then(Value::as_str)
                    .and_then(Self::from_name)
                {
                    return Ok(variant);
                }
            }
        }
        let name = hub_name(root);
        Ok(if name.contains("turbo") {
            Self::Turbo
        } else {
            Self::Base
        })
    }
}

/// The directory name, or the repo id of a Hugging Face cache snapshot
/// (`models--baidu--ERNIE-Image-Turbo/snapshots/<rev>`), lowercased.
fn hub_name(root: &Path) -> String {
    let leaf = root
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let parent = root.parent();
    if parent
        .and_then(Path::file_name)
        .is_some_and(|n| n == "snapshots")
        && let Some(repo) = parent
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .map(|n| n.to_string_lossy().to_lowercase())
        && let Some(stripped) = repo.strip_prefix("models--")
    {
        return stripped.replace("--", "/");
    }
    leaf
}

/// Upstream `validate_dimensions`: each side at least 16 and a multiple of 16.
pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    for (label, value) in [("width", width), ("height", height)] {
        if value < 16 {
            bail!("{label} must be at least 16, got {value}");
        }
        if !value.is_multiple_of(16) {
            bail!("{label} must be a multiple of 16, got {value}");
        }
    }
    Ok(())
}

/// Upstream `_round_edit_dimension`: down to a multiple of 16 in `16..=2048`.
#[must_use]
pub fn round_edit_dimension(value: f64) -> u32 {
    ((value.max(0.0) as u64 / 16 * 16).clamp(16, 2048)) as u32
}

/// Upstream `_load_edit_image` without an explicit size: scale the source
/// toward 1024x1024 pixels of area (never up), at most 2048 on a side, at
/// least 256 on the short side when that fits, then round to 16.
#[must_use]
pub fn default_edit_size(width: u32, height: u32) -> (u32, u32) {
    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let max_scale = (2048.0 / w).min(2048.0 / h);
    let mut scale = 1.0_f64
        .min((1024.0 * 1024.0 / (w * h)).sqrt())
        .min(max_scale);
    let min_scale = 256.0 / w.min(h);
    if min_scale <= max_scale {
        scale = scale.max(min_scale);
    }
    (
        round_edit_dimension(w * scale),
        round_edit_dimension(h * scale),
    )
}

/// First denoising step of an SDEdit run: `steps - max(1, min(steps,
/// round(steps * strength)))`, with Python's ties-to-even `round`.
#[must_use]
pub fn edit_start_index(steps: u32, strength: f64) -> usize {
    let denoise = (f64::from(steps) * strength).round_ties_even();
    let denoise = denoise.clamp(1.0, f64::from(steps.max(1))) as u32;
    steps.saturating_sub(denoise) as usize
}

/// The model-specific request knobs (`--gen-kwargs` or extra JSON fields).
#[derive(Debug, Clone, PartialEq)]
pub struct ErnieOptions {
    /// `image_strength` (alias `strength`), in `(0, 1]`.
    pub image_strength: f64,
    /// `negative_prompt` from `extra`, used when the request field is absent.
    pub negative_prompt: Option<String>,
    /// `use_prompt_enhancer`: `None` applies upstream's automatic rule.
    pub use_prompt_enhancer: Option<bool>,
    /// `prompt_enhancer_max_tokens`: cap on the enhancer's new tokens.
    pub prompt_enhancer_max_tokens: Option<usize>,
}

fn number(extra: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_f64()
            .map(Some)
            .with_context(|| format!("ERNIE-Image {key} is not a finite number")),
        Some(v) => bail!("ERNIE-Image {key} must be a number, got {v}"),
    }
}

impl ErnieOptions {
    /// Parse and range-check the options.
    pub fn from_extra(extra: &Map<String, Value>) -> Result<Self> {
        // `image_strength` is canonical; `strength` is the alias upstream
        // accepts so the edit command lines up with Z-Image.
        let image_strength = match number(extra, "image_strength")? {
            Some(v) => v,
            None => number(extra, "strength")?.unwrap_or(DEFAULT_IMAGE_STRENGTH),
        };
        if !(image_strength > 0.0 && image_strength <= 1.0) {
            bail!("image_strength must be in (0, 1], got {image_strength}");
        }
        let negative_prompt = match extra.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("ERNIE-Image negative_prompt must be a string, got {v}"),
        };
        let use_prompt_enhancer = match extra.get("use_prompt_enhancer") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(v) => bail!("ERNIE-Image use_prompt_enhancer must be a boolean, got {v}"),
        };
        let prompt_enhancer_max_tokens = match extra.get("prompt_enhancer_max_tokens") {
            None | Some(Value::Null) => None,
            Some(v) => match v.as_u64() {
                Some(n) if (1..=MAX_TEXT_TOKENS as u64).contains(&n) => Some(n as usize),
                _ => bail!(
                    "ERNIE-Image prompt_enhancer_max_tokens must be an integer in 1..={MAX_TEXT_TOKENS}, got {v}"
                ),
            },
        };
        Ok(Self {
            image_strength,
            negative_prompt,
            use_prompt_enhancer,
            prompt_enhancer_max_tokens,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
