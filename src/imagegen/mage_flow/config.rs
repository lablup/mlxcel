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

//! Mage-Flow variants, transformer config and request options.
//!
//! A port of `config.py` (`VARIANTS`, `get_variant`, `variant_from_local_path`,
//! `validate_dimensions`) and of `_edit_dimensions` in `pipeline.py` from
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mage_flow>
//! (mlx-vlm `73d366ec`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::imagegen::pipeline::{ImageSamplingDefaults, ImageTask};

/// Smallest and largest output side upstream accepts.
pub const MIN_SIDE: u32 = 512;
pub const MAX_SIDE: u32 = 2048;
/// Pixels per latent cell (`MageVAE.downsample_factor`).
pub const DOWNSAMPLE: u32 = 16;
/// Default `vl_cond_long_edge` of an edit (upstream `MageFlowImageEditModel.edit`).
pub const DEFAULT_VL_COND_LONG_EDGE: i64 = 384;
/// Largest long edge the conditioner sees an edit reference at, in pixels
/// (mlxcel bound; upstream has none). A larger `vl_cond_long_edge` is a `400`,
/// and a reference kept at its own size (`null` or `<= 0`) is capped to
/// [`MAX_VL_COND_PIXELS`]. Without it one request could send 16 references at
/// the checkpoint's 4096x4096 processor ceiling, about 262k Qwen3-VL tokens,
/// through a vision tower whose evaluation aborts the process on failure.
pub const MAX_VL_COND_LONG_EDGE: u32 = MAX_SIDE;
/// Pixel ceiling of one edit reference in the conditioner
/// (`MAX_VL_COND_LONG_EDGE` squared): at most 4096 visual tokens per reference.
pub const MAX_VL_COND_PIXELS: u64 = MAX_VL_COND_LONG_EDGE as u64 * MAX_VL_COND_LONG_EDGE as u64;
/// Default static shift of the flow-match schedule.
pub const DEFAULT_STATIC_SHIFT: f32 = 6.0;

/// One of the six published checkpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MageFlowVariant {
    /// Upstream variant name (`mage-flow-turbo`, ...).
    pub name: &'static str,
    pub repo_id: &'static str,
    pub task: ImageTask,
    pub default_steps: u32,
    pub default_guidance: f32,
    aliases: &'static [&'static str],
}

impl MageFlowVariant {
    /// Default steps and guidance (upstream `ImageSamplingDefaults.from_config`).
    #[must_use]
    pub fn defaults(&self) -> ImageSamplingDefaults {
        ImageSamplingDefaults {
            steps: self.default_steps,
            guidance: self.default_guidance,
        }
    }

    /// Error unless this checkpoint implements `task`, naming the variant the
    /// way upstream's `generate_array` / `edit_array` do.
    pub fn require(&self, task: ImageTask) -> Result<()> {
        if self.task == task {
            return Ok(());
        }
        match self.task {
            ImageTask::Edit => bail!(
                "{} is an image-edit checkpoint; it does not support task generate",
                self.repo_id
            ),
            ImageTask::Generate => bail!(
                "{} is a text-to-image checkpoint; it does not support task edit",
                self.repo_id
            ),
        }
    }
}

const fn variant(
    name: &'static str,
    repo_id: &'static str,
    task: ImageTask,
    default_steps: u32,
    default_guidance: f32,
    aliases: &'static [&'static str],
) -> MageFlowVariant {
    MageFlowVariant {
        name,
        repo_id,
        task,
        default_steps,
        default_guidance,
        aliases,
    }
}

/// Upstream `VARIANTS`, in its order.
pub const VARIANTS: [MageFlowVariant; 6] = [
    variant(
        "mage-flow-base",
        "mage-flow-community/Mage-Flow-Base",
        ImageTask::Generate,
        30,
        5.0,
        &["microsoft/Mage-Flow-Base", "mage-flow-4b-base"],
    ),
    variant(
        "mage-flow",
        "mage-flow-community/Mage-Flow",
        ImageTask::Generate,
        20,
        5.0,
        &["microsoft/Mage-Flow", "mage-flow-4b"],
    ),
    variant(
        "mage-flow-turbo",
        "mage-flow-community/Mage-Flow-Turbo",
        ImageTask::Generate,
        4,
        1.0,
        &["microsoft/Mage-Flow-Turbo", "mage-flow-4b-turbo"],
    ),
    variant(
        "mage-flow-edit-base",
        "mage-flow-community/Mage-Flow-Edit-Base",
        ImageTask::Edit,
        30,
        5.0,
        &["microsoft/Mage-Flow-Edit-Base", "mage-flow-edit-4b-base"],
    ),
    variant(
        "mage-flow-edit",
        "mage-flow-community/Mage-Flow-Edit",
        ImageTask::Edit,
        30,
        5.0,
        &["microsoft/Mage-Flow-Edit", "mage-flow-edit-4b"],
    ),
    variant(
        "mage-flow-edit-turbo",
        "mage-flow-community/Mage-Flow-Edit-Turbo",
        ImageTask::Edit,
        4,
        1.0,
        &["microsoft/Mage-Flow-Edit-Turbo", "mage-flow-edit-4b-turbo"],
    ),
];

fn by_name(name: &str) -> MageFlowVariant {
    *VARIANTS
        .iter()
        .find(|v| v.name == name)
        .expect("name is a VARIANTS entry")
}

/// Upstream `get_variant`: the name, repo id, repo basename or an alias,
/// case-insensitive.
pub fn get_variant(name: &str) -> Result<MageFlowVariant> {
    let key = name.trim().trim_end_matches('/').to_lowercase();
    VARIANTS
        .iter()
        .find(|v| {
            let basename = v.repo_id.rsplit('/').next().unwrap_or(v.repo_id);
            [v.name, v.repo_id, basename]
                .iter()
                .chain(v.aliases)
                .any(|alias| alias.to_lowercase() == key)
        })
        .copied()
        .with_context(|| {
            let mut names: Vec<&str> = VARIANTS.iter().map(|v| v.name).collect();
            names.sort_unstable();
            format!(
                "Unknown Mage-Flow variant {name:?}. Supported: {}",
                names.join(", ")
            )
        })
}

/// The variant a directory-name fragment names, by upstream's keyword rules:
/// `turbo`, then `base`, each split by `edit`; `edit` alone is Mage-Flow-Edit.
fn variant_from_name(name: &str) -> Option<MageFlowVariant> {
    let name = name.to_lowercase().replace('_', "-");
    let edit = name.contains("edit");
    if name.contains("turbo") {
        return Some(by_name(if edit {
            "mage-flow-edit-turbo"
        } else {
            "mage-flow-turbo"
        }));
    }
    if name.contains("base") {
        return Some(by_name(if edit {
            "mage-flow-edit-base"
        } else {
            "mage-flow-base"
        }));
    }
    edit.then(|| by_name("mage-flow-edit"))
}

/// The path component the keyword rules read: the nearest one (from the leaf
/// up) that mentions `mage`, else the leaf itself.
///
/// Upstream matches the keywords anywhere in the whole path string, so a
/// parent directory such as `/data/database/` would turn every checkpoint
/// under it into a `-base` variant. Reading one component keeps upstream's
/// result for a plain checkpoint directory and for the Hugging Face cache
/// layout (`models--mage-flow-community--Mage-Flow-Turbo/snapshots/<sha>`)
/// without that hazard.
fn naming_component(root: &Path) -> String {
    let components: Vec<String> = root
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    components
        .iter()
        .rev()
        .find(|c| c.to_lowercase().contains("mage"))
        .or_else(|| components.last())
        .cloned()
        .unwrap_or_default()
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text)
        .map(Some)
        .with_context(|| format!("cannot parse {}", path.display()))
}

/// Upstream `variant_from_local_path`: the `variant` of `mlx_mage_flow.json`
/// (written by upstream's converter), else the keyword rules on the
/// checkpoint's directory name (see [`naming_component`]), else Mage-Flow when
/// `model_index.json` names `MageFlowPipeline`.
pub fn variant_from_local_path(root: &Path) -> Result<MageFlowVariant> {
    if let Some(metadata) = read_json(&root.join("mlx_mage_flow.json"))?
        && let Some(name) = metadata.get("variant").and_then(Value::as_str)
        && !name.is_empty()
    {
        return get_variant(name);
    }
    if let Some(found) = variant_from_name(&naming_component(root)) {
        return Ok(found);
    }
    if let Some(index) = read_json(&root.join("model_index.json"))?
        && index.get("_class_name").and_then(Value::as_str) == Some("MageFlowPipeline")
    {
        return Ok(by_name("mage-flow"));
    }
    bail!(
        "Could not infer a Mage-Flow variant from local model path: {}. Use a directory name \
         containing Base, Turbo, and/or Edit.",
        root.display()
    )
}

/// Upstream `validate_dimensions`: each side in `512..=2048` and a multiple of
/// 16. Together these bound the aspect ratio at 4:1.
pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    for (label, value) in [("width", width), ("height", height)] {
        if !(MIN_SIDE..=MAX_SIDE).contains(&value) {
            bail!("{label} must be in [{MIN_SIDE}, {MAX_SIDE}], got {value}");
        }
        if !value.is_multiple_of(DOWNSAMPLE) {
            bail!("{label} must be a multiple of {DOWNSAMPLE}, got {value}");
        }
    }
    Ok(())
}

fn divisible_by_16(value: u32) -> u32 {
    (DOWNSAMPLE * (value / DOWNSAMPLE)).max(DOWNSAMPLE)
}

/// Upstream `_edit_dimensions` without an explicit size: scale the reference
/// so its long edge is `target` (default its own long edge), then round each
/// side down to a multiple of 16.
#[must_use]
pub fn edit_dimensions(ref_width: u32, ref_height: u32, max_size: Option<u32>) -> (u32, u32) {
    let target = max_size.unwrap_or_else(|| ref_width.max(ref_height));
    // Python's `round` is round-half-even on the float quotient.
    let scaled = |side: u32, long: u32| {
        (f64::from(side) * f64::from(target) / f64::from(long)).round_ties_even() as u32
    };
    let (width, height) = if ref_height >= ref_width {
        (scaled(ref_width, ref_height), target)
    } else {
        (target, scaled(ref_height, ref_width))
    };
    (divisible_by_16(width), divisible_by_16(height))
}

/// The edit output size of a reference with no requested size.
///
/// Used by: [`crate::imagegen::ImageGenerationModel::default_edit_size`] of
/// Mage-Flow.
#[must_use]
pub fn default_edit_size(width: u32, height: u32) -> (u32, u32) {
    edit_dimensions(width, height, None)
}

/// `transformer/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub context_in_dim: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub depth: i32,
    pub axes_dim: [i32; 3],
    pub theta: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            context_in_dim: 2560,
            hidden_size: 3072,
            num_heads: 24,
            depth: 12,
            axes_dim: [16, 56, 56],
            theta: 10000.0,
        }
    }
}

fn int_field(config: &Value, key: &str, default: i32) -> Result<i32> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .filter(|n| *n > 0)
            .with_context(|| format!("Mage-Flow transformer {key} must be a positive integer")),
    }
}

impl TransformerConfig {
    /// Read the fields upstream's `load_transformer` reads, with its defaults.
    pub fn from_json(config: &Value) -> Result<Self> {
        let defaults = Self::default();
        let axes_dim = match config.get("axes_dim") {
            None | Some(Value::Null) => defaults.axes_dim,
            Some(v) => {
                let axes: Vec<i32> = serde_json::from_value(v.clone())
                    .context("Mage-Flow transformer axes_dim must be three integers")?;
                <[i32; 3]>::try_from(axes.as_slice())
                    .ok()
                    .filter(|a| a.iter().all(|d| *d > 0 && d % 2 == 0))
                    .context(
                        "Mage-Flow transformer axes_dim must be three positive even integers",
                    )?
            }
        };
        let theta = match config.get("theta") {
            None | Some(Value::Null) => defaults.theta,
            Some(v) => v
                .as_f64()
                .filter(|t| t.is_finite() && *t > 0.0)
                .context("Mage-Flow transformer theta must be a positive number")?
                as f32,
        };
        let parsed = Self {
            in_channels: int_field(config, "in_channels", defaults.in_channels)?,
            out_channels: int_field(config, "out_channels", defaults.out_channels)?,
            context_in_dim: int_field(config, "context_in_dim", defaults.context_in_dim)?,
            hidden_size: int_field(config, "hidden_size", defaults.hidden_size)?,
            num_heads: int_field(config, "num_heads", defaults.num_heads)?,
            depth: int_field(config, "depth", defaults.depth)?,
            axes_dim,
            theta,
        };
        if parsed.hidden_size % parsed.num_heads != 0 {
            bail!("Mage-Flow hidden_size must be a multiple of num_heads");
        }
        let head_dim = parsed.hidden_size / parsed.num_heads;
        if parsed.axes_dim.iter().sum::<i32>() != head_dim {
            bail!(
                "axes_dim must sum to head_dim ({head_dim}), got {:?}",
                parsed.axes_dim
            );
        }
        Ok(parsed)
    }

    /// Per-head width.
    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }
}

/// Model-specific request options (`--gen-kwargs` or extra JSON fields).
#[derive(Debug, Clone, PartialEq)]
pub struct MageFlowOptions {
    /// Used when the request's own `negative_prompt` is absent; default `" "`.
    pub negative_prompt: Option<String>,
    pub static_shift: f32,
    pub renormalization: bool,
    /// Edit only: long edge of the output, sized from the first reference.
    pub max_size: Option<u32>,
    /// Edit only: long edge the conditioner sees each reference at; `None`
    /// (or a value `<= 0`) keeps the reference's own size.
    pub vl_cond_long_edge: Option<u32>,
    /// Edit only: sample the VAE posterior for the reference latents.
    pub sample_posterior: bool,
}

fn extra_bool(extra: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => bail!("Mage-Flow {key} must be a boolean, got {v}"),
    }
}

fn extra_side(extra: &Map<String, Value>, key: &str) -> Result<Option<i64>> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .with_context(|| format!("Mage-Flow {key} must be an integer, got {v}")),
    }
}

impl MageFlowOptions {
    /// Parse and range-check `extra` for `task`; edit-only keys are rejected
    /// on a generate request.
    pub fn from_extra(extra: &Map<String, Value>, task: ImageTask) -> Result<Self> {
        if task == ImageTask::Generate {
            for key in ["max_size", "vl_cond_long_edge", "sample_posterior"] {
                if extra.contains_key(key) {
                    bail!("Mage-Flow {key} applies to image edits only");
                }
            }
        }
        let static_shift = match extra.get("static_shift") {
            None | Some(Value::Null) => DEFAULT_STATIC_SHIFT,
            Some(v) => v
                .as_f64()
                .filter(|s| s.is_finite() && *s > 0.0)
                .with_context(|| {
                    format!("Mage-Flow static_shift must be a positive number, got {v}")
                })? as f32,
        };
        let negative_prompt = match extra.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("Mage-Flow negative_prompt must be a string, got {v}"),
        };
        let max_size = match extra_side(extra, "max_size")? {
            None => None,
            Some(v) if (1..=i64::from(MAX_SIDE) * 4).contains(&v) => Some(v as u32),
            Some(v) => bail!("Mage-Flow max_size must be a positive integer, got {v}"),
        };
        let vl_cond_long_edge = match extra.get("vl_cond_long_edge") {
            // An explicit null keeps the references at their own size, as upstream.
            Some(Value::Null) => None,
            _ => match extra_side(extra, "vl_cond_long_edge")?.unwrap_or(DEFAULT_VL_COND_LONG_EDGE)
            {
                v if v <= 0 => None,
                v if v <= i64::from(MAX_VL_COND_LONG_EDGE) => Some(v as u32),
                v => bail!(
                    "Mage-Flow vl_cond_long_edge must be at most {MAX_VL_COND_LONG_EDGE}, got {v}"
                ),
            },
        };
        Ok(Self {
            negative_prompt,
            static_shift,
            renormalization: extra_bool(extra, "renormalization", false)?,
            max_size,
            vl_cond_long_edge,
            sample_posterior: extra_bool(extra, "sample_posterior", true)?,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
