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

//! LLaDA-Image configuration and request options.
//!
//! A port of `config.py`, `LLaDAImageScheduler.__init__` and the request
//! arithmetic of `pipeline.py` and `model.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/llada_image>.
//! Nothing here reads a weight, so defaults and request checks run before
//! any load.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::imagegen::pipeline::ImageSamplingDefaults;

/// `model_index.json` `_class_name` of a LLaDA-Image checkpoint.
pub const PIPELINE_CLASS_NAME: &str = "LLaDAImagePipeline";

/// Longest prompt the conditioner reads, in tokens (`max_sequence_length`).
pub const MAX_TEXT_TOKENS: usize = 2048;

/// Pixels per latent cell (8x FLUX.2 VAE, then 2x2 patches).
pub const LATENT_SCALE: u32 = 16;

/// Components every checkpoint carries (`validate_model_layout`).
pub const REQUIRED_COMPONENTS: [&str; 5] = [
    "text_encoder",
    "queryformer",
    "text_projection",
    "transformer",
    "vae",
];

fn read_json(path: &Path) -> Result<Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

fn read_config<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let value = read_json(path)?;
    serde_json::from_value(value).with_context(|| format!("invalid {}", path.display()))
}

/// Upstream `validate_model_layout`: the `_class_name`, then a `config.json`
/// and at least one safetensors file for every [`REQUIRED_COMPONENTS`] entry.
pub fn validate_model_layout(root: &Path) -> Result<PathBuf> {
    let index = read_json(&root.join("model_index.json"))?;
    if index.get("_class_name").and_then(Value::as_str) != Some(PIPELINE_CLASS_NAME) {
        bail!("Not a LLaDA-Image checkpoint: {}", root.display());
    }
    for component in REQUIRED_COMPONENTS {
        let dir = root.join(component);
        if !dir.join("config.json").is_file() {
            bail!("missing {}", dir.join("config.json").display());
        }
        let has_weights = std::fs::read_dir(&dir)
            .with_context(|| format!("cannot list {}", dir.display()))?
            .flatten()
            .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"));
        if !has_weights {
            bail!("No {component} weights in {}", root.display());
        }
    }
    Ok(root.to_path_buf())
}

/// `LLaDAImageTransformerConfig` (`transformer/config.json`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TransformerConfig {
    pub dim: i32,
    pub n_heads: i32,
    pub n_layers: i32,
    pub n_refiner_layers: i32,
    pub in_channels: i32,
    pub cap_feat_dim: i32,
    pub semantic_feat_dim: i32,
    pub norm_eps: f32,
    pub qk_norm: bool,
    pub rope_theta: f32,
    pub axes_dims: Vec<i32>,
    pub t_scale: f32,
    pub all_patch_size: Vec<i32>,
    pub all_f_patch_size: Vec<i32>,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            n_heads: 30,
            n_layers: 30,
            n_refiner_layers: 2,
            in_channels: 128,
            cap_feat_dim: 2560,
            semantic_feat_dim: 4096,
            norm_eps: 1e-5,
            qk_norm: true,
            rope_theta: 256.0,
            axes_dims: vec![32, 48, 48],
            t_scale: 1000.0,
            all_patch_size: vec![1],
            all_f_patch_size: vec![1],
        }
    }
}

impl TransformerConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(text).context("invalid LLaDA-Image transformer config.json")?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::from_json(&text)
    }

    #[must_use]
    pub fn head_dim(&self) -> i32 {
        self.dim / self.n_heads.max(1)
    }

    /// Upstream `__post_init__` and `from_dict`, plus the assumptions the
    /// shared blocks make (the timestep width is `min(dim, 256)` and the
    /// blocks' RMS epsilon is 1e-5).
    pub fn validate(&self) -> Result<()> {
        if self.all_patch_size != [1] || self.all_f_patch_size != [1] {
            bail!("LLaDA-Image currently supports only patch size 1");
        }
        if self.dim < 1 || self.n_heads < 1 || self.n_layers < 1 || self.n_refiner_layers < 0 {
            bail!("LLaDA-Image dim, n_heads and n_layers must be positive");
        }
        if self.dim % self.n_heads != 0 || self.axes_dims.iter().sum::<i32>() != self.head_dim() {
            bail!("LLaDA-Image head dimension must match axes_dims");
        }
        if self.axes_dims.len() != 3 {
            bail!(
                "LLaDA-Image needs three RoPE axes, got {:?}",
                self.axes_dims
            );
        }
        if self.dim < crate::imagegen::dit_blocks::ADALN_EMBED_DIM {
            bail!("LLaDA-Image dim must be at least 256");
        }
        if (self.norm_eps - 1e-5).abs() > f32::EPSILON {
            bail!("LLaDA-Image norm_eps must be 1e-5, got {}", self.norm_eps);
        }
        if !self.qk_norm {
            bail!("LLaDA-Image transformers without qk_norm are not supported");
        }
        Ok(())
    }
}

/// `queryformer/config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QueryFormerConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_queries: i32,
    #[serde(default = "default_aux_eps")]
    pub norm_eps: f32,
}

/// `text_projection/config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextProjectionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub projection_dim: i32,
    #[serde(default = "default_aux_eps")]
    pub norm_eps: f32,
}

/// `sigvq/config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SigVqConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub patch_size: i32,
    pub image_size: i32,
    pub codebook_size: i32,
    pub codebook_embed_dim: i32,
    pub semantic_embed_dim: i32,
    #[serde(default = "default_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_aux_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
}

fn default_aux_eps() -> f32 {
    1e-6
}

fn default_in_channels() -> i32 {
    3
}

fn default_true() -> bool {
    true
}

fn positive(label: &str, values: &[i32]) -> Result<()> {
    if values.iter().any(|v| *v < 1) {
        bail!("LLaDA-Image {label} sizes must be positive, got {values:?}");
    }
    Ok(())
}

impl QueryFormerConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config: Self = read_config(&dir.join("config.json"))?;
        positive(
            "queryformer",
            &[
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_queries,
            ],
        )?;
        if config.hidden_size % config.num_attention_heads != 0 {
            bail!("queryformer hidden_size must be divisible by num_attention_heads");
        }
        Ok(config)
    }
}

impl TextProjectionConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config: Self = read_config(&dir.join("config.json"))?;
        positive(
            "text_projection",
            &[
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.projection_dim,
            ],
        )?;
        if config.hidden_size % config.num_attention_heads != 0 {
            bail!("text_projection hidden_size must be divisible by num_attention_heads");
        }
        Ok(config)
    }
}

impl SigVqConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config: Self = read_config(&dir.join("config.json"))?;
        positive(
            "sigvq",
            &[
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.patch_size,
                config.image_size,
                config.codebook_size,
                config.codebook_embed_dim,
                config.semantic_embed_dim,
            ],
        )?;
        if config.hidden_size % config.num_attention_heads != 0 {
            bail!("sigvq hidden_size must be divisible by num_attention_heads");
        }
        if config.image_size % config.patch_size != 0 {
            bail!("sigvq image_size must be a multiple of patch_size");
        }
        Ok(config)
    }

    /// Side of the square position-embedding grid.
    #[must_use]
    pub fn position_grid(&self) -> i32 {
        self.image_size / self.patch_size
    }
}

/// `LLaDAImageScheduler`'s configuration (`scheduler/scheduler_config.json`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerConfig {
    /// Turbo: `linspace(1, 0, steps + 1)`; base: the f64 nonlinear schedule.
    pub use_uniform_sigmas: bool,
    pub shift: f32,
    /// Re-noise each step (Turbo default) instead of the Euler update.
    pub stochastic_sampling: bool,
}

impl SchedulerConfig {
    /// Upstream `LLaDAImageScheduler.__init__`: dynamic shifting and inverted
    /// sigmas are rejected; `shift` defaults to 3 (uniform) or 1, and
    /// `stochastic_sampling` to `use_uniform_sigmas`.
    pub fn from_value(config: &Value) -> Result<Self> {
        let flag = |key: &str| -> Result<Option<bool>> {
            match config.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Bool(b)) => Ok(Some(*b)),
                Some(v) => bail!("LLaDA-Image scheduler {key} must be a boolean, got {v}"),
            }
        };
        let use_uniform_sigmas = flag("use_uniform_sigmas")?.unwrap_or(false);
        if flag("use_dynamic_shifting")?.unwrap_or(false) || flag("invert_sigmas")?.unwrap_or(false)
        {
            bail!("LLaDA-Image requires a static, non-inverted flow schedule");
        }
        let shift = match config.get("shift") {
            None | Some(Value::Null) => {
                if use_uniform_sigmas {
                    3.0
                } else {
                    1.0
                }
            }
            Some(v) => v
                .as_f64()
                .with_context(|| format!("LLaDA-Image scheduler shift must be a number, got {v}"))?
                as f32,
        };
        if !shift.is_finite() || shift <= 0.0 {
            bail!("LLaDA-Image scheduler shift must be positive, got {shift}");
        }
        let stochastic_sampling = flag("stochastic_sampling")?.unwrap_or(use_uniform_sigmas);
        Ok(Self {
            use_uniform_sigmas,
            shift,
            stochastic_sampling,
        })
    }

    /// Read `<root>/scheduler/scheduler_config.json`.
    pub fn from_root(root: &Path) -> Result<Self> {
        Self::from_value(&read_json(
            &root.join("scheduler").join("scheduler_config.json"),
        )?)
    }

    /// The checkpoint variant name upstream reports.
    #[must_use]
    pub fn variant(&self) -> &'static str {
        if self.use_uniform_sigmas {
            "llada-image-turbo"
        } else {
            "llada-image"
        }
    }

    /// 4 steps at guidance 1 (Turbo) or 50 at guidance 5 (base), for both
    /// generation and editing.
    #[must_use]
    pub fn defaults(&self) -> ImageSamplingDefaults {
        if self.use_uniform_sigmas {
            ImageSamplingDefaults {
                steps: 4,
                guidance: 1.0,
            }
        } else {
            ImageSamplingDefaults {
                steps: 50,
                guidance: 5.0,
            }
        }
    }
}

/// How the transformer is conditioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    /// Prompt features only.
    Text,
    /// Prompt features plus SigVQ features of discrete image tokens the
    /// LLaDA-2 prior generates from the prompt.
    Vq,
    /// Single-image editing (`--task edit`, `/v1/images/edits`).
    Editing,
}

impl GenerationMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vq => "vq",
            Self::Editing => "editing",
        }
    }
}

/// The model-specific request knobs (`--gen-kwargs` or extra JSON fields).
#[derive(Debug, Clone, PartialEq)]
pub struct LladaOptions {
    pub mode: GenerationMode,
    /// `stochastic_sampling`: overrides the scheduler's setting when present.
    pub stochastic_sampling: Option<bool>,
    /// `negative_prompt` from `extra`, used when the request field is absent.
    pub negative_prompt: Option<String>,
}

impl LladaOptions {
    /// Upstream `generate` / `edit`: `generation_mode` is `text` or `vq` for
    /// generation (default `text`) and must be `editing` (the default) for
    /// an edit.
    pub fn from_extra(extra: &Map<String, Value>, edit: bool) -> Result<Self> {
        let mode = match extra.get("generation_mode") {
            None | Some(Value::Null) => {
                if edit {
                    GenerationMode::Editing
                } else {
                    GenerationMode::Text
                }
            }
            Some(Value::String(s)) => match (s.as_str(), edit) {
                ("text", false) => GenerationMode::Text,
                ("vq", false) => GenerationMode::Vq,
                ("editing", true) => GenerationMode::Editing,
                (_, false) => bail!(
                    "LLaDA-Image generation supports generation_mode text or vq, got {s:?}; use --task edit for editing"
                ),
                (_, true) => {
                    bail!(
                        "LLaDA-Image image editing requires generation_mode \"editing\", got {s:?}"
                    )
                }
            },
            Some(v) => bail!("LLaDA-Image generation_mode must be a string, got {v}"),
        };
        let stochastic_sampling = match extra.get("stochastic_sampling") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(v) => bail!("LLaDA-Image stochastic_sampling must be a boolean, got {v}"),
        };
        let negative_prompt = match extra.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("LLaDA-Image negative_prompt must be a string, got {v}"),
        };
        Ok(Self {
            mode,
            stochastic_sampling,
            negative_prompt,
        })
    }
}

/// Upstream `validate_dimensions` (both modes), plus editing's multiple of 32.
pub fn validate_dimensions(width: u32, height: u32, mode: GenerationMode) -> Result<()> {
    if width < 16 || height < 16 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        bail!(
            "LLaDA-Image width and height must be positive multiples of 16, got {width}x{height}"
        );
    }
    if mode == GenerationMode::Editing && (!width.is_multiple_of(32) || !height.is_multiple_of(32))
    {
        bail!("LLaDA-Image editing width and height must be multiples of 32, got {width}x{height}");
    }
    Ok(())
}

/// Upstream `edit_array`'s output size for a `width x height` source when the
/// request gives at most one side: no side scales the source toward 1024x1024
/// pixels of area (never up) and floors each side to 32 (at least 32); one
/// side keeps the aspect ratio and rounds the other to 32.
#[must_use]
pub fn edit_size(
    width: u32,
    height: u32,
    requested_width: Option<u32>,
    requested_height: Option<u32>,
) -> (u32, u32) {
    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    // Python's `round` is ties-to-even.
    let round32 = |v: f64| ((v / 32.0).round_ties_even() as u32 * 32).max(32);
    match (requested_width, requested_height) {
        (Some(rw), Some(rh)) => (rw, rh),
        (None, None) => {
            let scale = 1.0_f64.min((1024.0 * 1024.0 / (w * h)).sqrt());
            let side = |v: f64| (((v * scale) as u32) / 32 * 32).max(32);
            (side(w), side(h))
        }
        (None, Some(rh)) => (round32(w * f64::from(rh) / h), rh),
        (Some(rw), None) => (rw, round32(h * f64::from(rw) / w)),
    }
}

/// [`crate::imagegen::pipeline::ImageGenerationModel::default_edit_size`]:
/// [`edit_size`] with neither side requested.
#[must_use]
pub fn default_edit_size(width: u32, height: u32) -> (u32, u32) {
    edit_size(width, height, None, None)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
