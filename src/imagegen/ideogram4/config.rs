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

//! Ideogram 4 configuration, size rules and request options.
//!
//! A port of `Ideogram4TransformerConfig` and `validate_dimensions` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/config.py>
//! and of the option handling at the top of `generate_array` in `pipeline.py`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use super::scheduler::{SamplerPreset, get_preset};
use crate::imagegen::pipeline::MAX_IMAGE_STEPS;

/// Most transformer layers a config may declare (the checkpoint has 34).
pub const MAX_LAYERS: usize = 4096;

/// Upstream `Ideogram4TransformerConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ideogram4TransformerConfig {
    pub emb_dim: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub adaln_dim: i32,
    pub in_channels: i32,
    pub llm_features_dim: i32,
    pub rope_theta: f32,
    pub mrope_section: [i32; 3],
    pub norm_eps: f32,
}

impl Default for Ideogram4TransformerConfig {
    fn default() -> Self {
        Self {
            emb_dim: 4608,
            num_layers: 34,
            num_heads: 18,
            intermediate_size: 12288,
            adaln_dim: 512,
            in_channels: 128,
            llm_features_dim: 4096 * 13,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            norm_eps: 1e-5,
        }
    }
}

fn int(data: &Value, key: &str) -> Result<Option<i64>> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .with_context(|| format!("transformer config {key} must be an integer, got {v}")),
    }
}

impl Ideogram4TransformerConfig {
    /// Upstream `from_dict`: `emb_dim` from `num_attention_heads *
    /// attention_head_dim` when absent, `adaln_dim` spelled as on disk.
    pub fn from_json(data: &Value) -> Result<Self> {
        let d = Self::default();
        let heads = match int(data, "num_heads")? {
            Some(h) => h,
            None => int(data, "num_attention_heads")?.unwrap_or(i64::from(d.num_heads)),
        };
        let emb_dim = match (int(data, "emb_dim")?, int(data, "attention_head_dim")?) {
            (Some(e), _) => e,
            (None, Some(hd)) => heads.checked_mul(hd).with_context(|| {
                format!("transformer config num_heads {heads} * attention_head_dim {hd} overflows")
            })?,
            (None, None) => i64::from(d.emb_dim),
        };
        let mrope_section = match data.get("mrope_section") {
            None | Some(Value::Null) => d.mrope_section,
            Some(v) => {
                let s: Vec<i32> = serde_json::from_value(v.clone())
                    .context("transformer config mrope_section must be integers")?;
                s.try_into().map_err(|s: Vec<i32>| {
                    anyhow::anyhow!("mrope_section must have 3 entries, got {}", s.len())
                })?
            }
        };
        let float = |key: &str, default: f32| -> Result<f32> {
            match data.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(v) => v
                    .as_f64()
                    .map(|x| x as f32)
                    .with_context(|| format!("transformer config {key} must be a number")),
            }
        };
        let to_i32 = |v: i64, key: &str| -> Result<i32> {
            i32::try_from(v)
                .with_context(|| format!("transformer config {key} {v} is out of range"))
        };
        let config = Self {
            emb_dim: to_i32(emb_dim, "emb_dim")?,
            num_layers: int(data, "num_layers")?.map_or(d.num_layers, |v| v.max(0) as usize),
            num_heads: to_i32(heads, "num_heads")?,
            intermediate_size: to_i32(
                int(data, "intermediate_size")?.unwrap_or(i64::from(d.intermediate_size)),
                "intermediate_size",
            )?,
            adaln_dim: to_i32(
                int(data, "adaln_dim")?.unwrap_or(i64::from(d.adaln_dim)),
                "adaln_dim",
            )?,
            in_channels: to_i32(
                int(data, "in_channels")?.unwrap_or(i64::from(d.in_channels)),
                "in_channels",
            )?,
            llm_features_dim: to_i32(
                int(data, "llm_features_dim")?.unwrap_or(i64::from(d.llm_features_dim)),
                "llm_features_dim",
            )?,
            // Upstream `int(rope_theta)`.
            rope_theta: float("rope_theta", d.rope_theta)?.trunc(),
            mrope_section,
            norm_eps: float("norm_eps", d.norm_eps)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Read `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", path.display()))?;
        Self::from_json(&value).with_context(|| format!("invalid {}", path.display()))
    }

    /// Per-head width.
    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.emb_dim / self.num_heads
    }

    fn validate(&self) -> Result<()> {
        if self.num_heads <= 0 || self.emb_dim <= 0 || self.emb_dim % self.num_heads != 0 {
            bail!(
                "hidden_size {} must be divisible by num_heads {}",
                self.emb_dim,
                self.num_heads
            );
        }
        let half = self.head_dim() / 2;
        if self.head_dim() % 2 != 0 || self.mrope_section.iter().any(|&s| s < 0) {
            bail!("head_dim {} must be even", self.head_dim());
        }
        // Bound each entry before the arithmetic below so it cannot overflow.
        if self.mrope_section.iter().any(|&s| s > half) {
            bail!(
                "mrope_section {:?} does not fit head_dim {}",
                self.mrope_section,
                self.head_dim()
            );
        }
        // The axis selector writes indices `axis + 3k < 3 * section[axis]`,
        // which must stay below `head_dim / 2`.
        let reach = self.mrope_section[1..]
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if s > 0 {
                    3 * i64::from(s) - 2 + i as i64
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        if reach >= i64::from(half) {
            bail!(
                "mrope_section {:?} does not fit head_dim {}",
                self.mrope_section,
                self.head_dim()
            );
        }
        for (name, v) in [
            ("intermediate_size", self.intermediate_size),
            ("adaln_dim", self.adaln_dim),
            ("in_channels", self.in_channels),
            ("llm_features_dim", self.llm_features_dim),
        ] {
            if v <= 0 {
                bail!("{name} must be positive, got {v}");
            }
        }
        if self.num_layers == 0 || self.num_layers > MAX_LAYERS {
            bail!(
                "num_layers must be in 1..={MAX_LAYERS}, got {}",
                self.num_layers
            );
        }
        Ok(())
    }
}

/// Smallest and largest output side.
pub const MIN_SIDE: u32 = 256;
/// See [`MIN_SIDE`].
pub const MAX_SIDE: u32 = 2048;
/// Sides must be multiples of this (2x2 patches of 8x VAE latents).
pub const SIZE_MULTIPLE: u32 = 16;

/// Upstream `validate_dimensions`.
pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    for (name, value) in [("width", width), ("height", height)] {
        if !(MIN_SIDE..=MAX_SIDE).contains(&value) {
            bail!("{name} must be in [{MIN_SIDE}, {MAX_SIDE}], got {value}");
        }
        if value % SIZE_MULTIPLE != 0 {
            bail!("{name} must be divisible by {SIZE_MULTIPLE}, got {value}");
        }
    }
    let (w, h) = (f64::from(width), f64::from(height));
    if (w / h).max(h / w) > 6.0 {
        bail!("aspect ratio must be at most 6:1, got {width}x{height}");
    }
    Ok(())
}

/// The `--gen-kwargs` (or extra JSON fields) Ideogram 4 reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Ideogram4Options {
    pub preset: &'static SamplerPreset,
    pub auto_json_caption: bool,
    pub guidance_schedule: Option<Vec<f64>>,
    pub mu: Option<f64>,
    pub std: Option<f64>,
}

fn number(extra: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() => Ok(Some(x)),
            _ => bail!("Ideogram 4 {key} must be a finite number, got {v}"),
        },
    }
}

impl Ideogram4Options {
    /// Parse `extra`. `prompt_expansion_model` (local prompt expansion through
    /// a second model, mlx-vlm#1276) is not supported and is rejected rather
    /// than ignored.
    pub fn from_extra(extra: &Map<String, Value>) -> Result<Self> {
        if let Some(v) = extra.get("prompt_expansion_model")
            && !v.is_null()
        {
            bail!(
                "Ideogram 4 prompt_expansion_model is not supported by mlxcel; \
                 send a JSON caption or a plain-text prompt (wrapped automatically)"
            );
        }
        let preset = match extra.get("sampler_preset") {
            None | Some(Value::Null) => get_preset(None)?,
            Some(Value::String(s)) => get_preset(Some(s))?,
            Some(v) => bail!("Ideogram 4 sampler_preset must be a string, got {v}"),
        };
        let auto_json_caption = match extra.get("auto_json_caption") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!("Ideogram 4 auto_json_caption must be a boolean, got {v}"),
        };
        let guidance_schedule = match extra.get("guidance_schedule") {
            None | Some(Value::Null) => None,
            Some(Value::Array(items)) if items.len() > MAX_IMAGE_STEPS as usize => bail!(
                "Ideogram 4 guidance_schedule has {} entries, over the {MAX_IMAGE_STEPS} step limit",
                items.len()
            ),
            Some(Value::Array(items)) => Some(
                items
                    .iter()
                    .map(|v| match v.as_f64() {
                        Some(x) if x.is_finite() => Ok(x),
                        _ => bail!(
                            "Ideogram 4 guidance_schedule entries must be finite numbers, got {v}"
                        ),
                    })
                    .collect::<Result<Vec<f64>>>()?,
            ),
            Some(v) => bail!("Ideogram 4 guidance_schedule must be a list of numbers, got {v}"),
        };
        let std = number(extra, "std")?;
        if std.is_some_and(|s| s <= 0.0) {
            bail!("Ideogram 4 std must be positive");
        }
        Ok(Self {
            preset,
            auto_json_caption,
            guidance_schedule,
            mu: number(extra, "mu")?,
            std,
        })
    }

    /// Upstream's step and guidance resolution: `steps` and `guidance` are the
    /// caller's explicit choices. The preset's per-step schedule applies only
    /// when neither a schedule nor a guidance is given and the step count is
    /// the preset's.
    pub fn plan(&self, steps: Option<u32>, guidance: Option<f32>) -> Result<SamplingPlan> {
        let num_steps = steps.unwrap_or(self.preset.num_steps);
        if num_steps < 1 {
            bail!("steps must be >= 1, got {num_steps}");
        }
        let schedule = match &self.guidance_schedule {
            Some(s) => Some(s.clone()),
            None if guidance.is_none() && num_steps == self.preset.num_steps => {
                Some(self.preset.guidance_schedule.to_vec())
            }
            None => None,
        };
        if let Some(s) = &schedule
            && s.len() != num_steps as usize
        {
            bail!(
                "guidance_schedule must have {num_steps} items, got {}",
                s.len()
            );
        }
        let last = *self
            .preset
            .guidance_schedule
            .last()
            .context("preset has an empty guidance schedule")?;
        Ok(SamplingPlan {
            steps: num_steps,
            guidance: guidance.map_or(last, f64::from),
            guidance_schedule: schedule,
            mu: self.mu.unwrap_or(self.preset.mu),
            std: self.std.unwrap_or(self.preset.std),
        })
    }
}

/// What one generation runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingPlan {
    pub steps: u32,
    /// Constant guidance when `guidance_schedule` is `None`.
    pub guidance: f64,
    pub guidance_schedule: Option<Vec<f64>>,
    pub mu: f64,
    pub std: f64,
}

impl SamplingPlan {
    /// Guidance at loop index `i` (the loop runs `steps - 1` down to 0).
    #[must_use]
    pub fn guidance_at(&self, i: usize) -> f64 {
        self.guidance_schedule
            .as_ref()
            .map_or(self.guidance, |s| s[i])
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
