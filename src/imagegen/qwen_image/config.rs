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

//! Qwen-Image-2.1 configuration: the transformer and VAE configs, the
//! scheduler parameters, the sampling defaults and the request options, a
//! port of `config.py` and the sizing rules of `pipeline.py` in mlx-vlm
//! `73d366ec` (`mlx_vlm/models/qwen_image`).

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use crate::imagegen::pipeline::{ImageSamplingDefaults, ImageTask};
use crate::imagegen::scheduler::DynamicShift;

/// Pixels per latent cell (`scale_factor_spatial`).
pub const DOWNSAMPLE: u32 = 16;
/// Edit sizes are multiples of this (upstream `// 32 * 32`): every reference
/// image slot of the conditioner covers 2x2 VAE latent cells.
pub const EDIT_MULTIPLE: u32 = 32;
/// Upstream `validate_dimensions` bounds.
pub const MIN_SIDE: u32 = 256;
pub const MAX_SIDE: u32 = 2048;
/// Upstream `edit_array` default `output_resolution`.
pub const DEFAULT_OUTPUT_RESOLUTION: u32 = 1024;
/// Largest joint image-token count of one edit forward (mlxcel bound;
/// upstream has none): the target plus every reference latent grid. 65536 is
/// four 2048x2048 grids, so the default 1024 edit takes up to 15 references.
pub const MAX_EDIT_IMAGE_TOKENS: u64 =
    4 * (MAX_SIDE / DOWNSAMPLE) as u64 * (MAX_SIDE / DOWNSAMPLE) as u64;
/// Prompt tokens the conditioner keeps after the template (upstream
/// `max_length`); generation truncates, an edit longer than this is refused.
pub const MAX_PROMPT_TOKENS: usize = 1024;

/// Upstream `QwenImageGenerationModel.default_sampling`.
pub const GENERATE_DEFAULTS: (u32, f32) = (30, 1.0);
/// Upstream `QwenImageEditModel.default_sampling`.
pub const EDIT_DEFAULTS: (u32, f32) = (40, 1.0);

/// Steps and guidance for `task`.
#[must_use]
pub fn defaults(task: ImageTask) -> ImageSamplingDefaults {
    let (steps, guidance) = match task {
        ImageTask::Generate => GENERATE_DEFAULTS,
        ImageTask::Edit => EDIT_DEFAULTS,
    };
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
                format!("Qwen-Image config {key} must be a positive integer, got {v}")
            }),
    }
}

/// `transformer/config.json` (`QwenImage21Transformer2DModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub num_layers: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub context_in_dim: i32,
    pub mlp_ratio: i32,
    pub axes_dims_rope: [i32; 3],
    pub eps: f32,
    pub causal_condition: bool,
}

impl Default for TransformerConfig {
    /// Upstream `QWEN_IMAGE_2_1_TRANSFORMER` over the class defaults.
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            in_channels: 64,
            out_channels: 64,
            context_in_dim: 4096,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
            causal_condition: true,
        }
    }
}

impl TransformerConfig {
    pub fn from_json(c: &Value) -> Result<Self> {
        let d = Self::default();
        let axes_dims_rope = match c.get("axes_dims_rope") {
            None | Some(Value::Null) => d.axes_dims_rope,
            Some(v) => {
                let axes: Vec<i32> = v
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_i64().map(|n| n as i32))
                            .collect()
                    })
                    .unwrap_or_default();
                axes.try_into().map_err(|_| {
                    anyhow::anyhow!("Qwen-Image axes_dims_rope must be three integers, got {v}")
                })?
            }
        };
        if let Some(p) = c.get("patch_size").and_then(Value::as_i64)
            && p != 1
        {
            bail!("Qwen-Image transformer patch_size must be 1, got {p}");
        }
        let config = Self {
            num_layers: int(c, "num_layers", d.num_layers.into())?,
            num_heads: int(c, "num_attention_heads", d.num_heads.into())?,
            head_dim: int(c, "attention_head_dim", d.head_dim.into())?,
            in_channels: int(c, "in_channels", d.in_channels.into())?,
            out_channels: int(c, "out_channels", d.out_channels.into())?,
            context_in_dim: int(c, "context_in_dim", d.context_in_dim.into())?,
            mlp_ratio: int(c, "mlp_ratio", d.mlp_ratio.into())?,
            axes_dims_rope,
            eps: c
                .get("eps")
                .and_then(Value::as_f64)
                .map_or(d.eps, |e| e as f32),
            causal_condition: c
                .get("causal_condition")
                .and_then(Value::as_bool)
                .unwrap_or(d.causal_condition),
        };
        ensure!(
            config.axes_dims_rope.iter().all(|a| *a > 0 && a % 2 == 0)
                && config.axes_dims_rope.iter().sum::<i32>() == config.head_dim,
            "Qwen-Image axes_dims_rope {:?} must be even and sum to attention_head_dim {}",
            config.axes_dims_rope,
            config.head_dim
        );
        Ok(config)
    }

    pub fn from_dir(transformer_dir: &Path) -> Result<Self> {
        Self::from_json(&read_json(&transformer_dir.join("config.json"))?)
    }

    /// `num_heads * head_dim`.
    #[must_use]
    pub fn inner_dim(&self) -> i32 {
        self.num_heads * self.head_dim
    }
}

/// The Qwen-Image VAE (`AutoencoderKLQwenImage21`) configuration.
///
/// Used by: QwenImage, MingImage (its 4-channel, non-residual config).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageVaeConfig {
    pub base_dim: i32,
    /// Decoder width; `base_dim` when the checkpoint has none.
    pub decoder_base_dim: i32,
    pub z_dim: i32,
    pub dim_mult: Vec<i32>,
    pub num_res_blocks: i32,
    pub temperal_downsample: Vec<bool>,
    pub in_channels: i32,
    pub out_channels: i32,
    pub is_residual: bool,
    /// Ming-Image's latent scale (`scaling_factor`); Qwen-Image uses the
    /// per-channel `latents_mean` / `latents_std` instead.
    pub scaling_factor: Option<f32>,
    pub shift_factor: Option<f32>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl Default for QwenImageVaeConfig {
    /// Upstream `QwenImageVAE` defaults (the Qwen-Image-2.1 VAE).
    fn default() -> Self {
        Self {
            base_dim: 96,
            decoder_base_dim: 144,
            z_dim: 64,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true, true],
            in_channels: 4,
            out_channels: 4,
            is_residual: true,
            scaling_factor: None,
            shift_factor: None,
            latents_mean: Vec::new(),
            latents_std: Vec::new(),
        }
    }
}

fn floats(c: &Value, key: &str) -> Result<Vec<f32>> {
    match c.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|x| x.is_finite())
                    .map(|x| x as f32)
                    .with_context(|| format!("Qwen-Image VAE {key} holds a non-number {v}"))
            })
            .collect(),
        Some(v) => bail!("Qwen-Image VAE {key} must be a list, got {v}"),
    }
}

impl QwenImageVaeConfig {
    /// `vae/config.json`, as upstream `load_vae` reads it.
    pub fn from_json(c: &Value) -> Result<Self> {
        let d = Self::default();
        let base_dim = int(c, "base_dim", d.base_dim.into())?;
        let dim_mult: Vec<i32> = match c.get("dim_mult") {
            None | Some(Value::Null) => d.dim_mult,
            Some(v) => v
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_i64().map(|n| n as i32))
                        .collect()
                })
                .filter(|m: &Vec<i32>| !m.is_empty() && m.iter().all(|x| *x > 0))
                .with_context(|| {
                    format!("Qwen-Image VAE dim_mult must be positive integers, got {v}")
                })?,
        };
        let temperal_downsample: Vec<bool> = match c.get("temperal_downsample") {
            None | Some(Value::Null) => d.temperal_downsample,
            Some(v) => v
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_bool).collect())
                .with_context(|| {
                    format!("Qwen-Image VAE temperal_downsample must be booleans, got {v}")
                })?,
        };
        let config = Self {
            base_dim,
            decoder_base_dim: match c.get("decoder_base_dim") {
                None | Some(Value::Null) => base_dim,
                _ => int(c, "decoder_base_dim", base_dim.into())?,
            },
            z_dim: int(c, "z_dim", d.z_dim.into())?,
            num_res_blocks: int(c, "num_res_blocks", d.num_res_blocks.into())?,
            in_channels: int(c, "in_channels", d.in_channels.into())?,
            out_channels: int(c, "out_channels", d.out_channels.into())?,
            is_residual: c
                .get("is_residual")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            scaling_factor: c
                .get("scaling_factor")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            shift_factor: c
                .get("shift_factor")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            latents_mean: floats(c, "latents_mean")?,
            latents_std: floats(c, "latents_std")?,
            dim_mult,
            temperal_downsample,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn from_dir(vae_dir: &Path) -> Result<Self> {
        Self::from_json(&read_json(&vae_dir.join("config.json"))?)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperal_downsample.len() + 1 == self.dim_mult.len(),
            "Qwen-Image VAE needs one temperal_downsample flag per downsampling stage ({} for dim_mult {:?}), got {}",
            self.dim_mult.len().saturating_sub(1),
            self.dim_mult,
            self.temperal_downsample.len()
        );
        for (name, stats) in [
            ("latents_mean", &self.latents_mean),
            ("latents_std", &self.latents_std),
        ] {
            ensure!(
                stats.is_empty() || stats.len() == self.z_dim as usize,
                "Qwen-Image VAE {name} has {} entries for z_dim {}",
                stats.len(),
                self.z_dim
            );
        }
        ensure!(
            self.latents_std.iter().all(|s| *s != 0.0),
            "Qwen-Image VAE latents_std must not contain zero"
        );
        Ok(())
    }

    /// Spatial factor between pixels and latents (`2 ** downsampling stages`).
    #[must_use]
    pub fn spatial_scale(&self) -> i32 {
        1 << self.temperal_downsample.len()
    }
}

/// The scheduler parameters upstream reads from
/// `scheduler/scheduler_config.json`, over the shared defaults.
pub fn scheduler_params(root: &Path) -> Result<DynamicShift> {
    let path = root.join("scheduler/scheduler_config.json");
    let mut params = DynamicShift::default();
    if !path.exists() {
        return Ok(params);
    }
    let c = read_json(&path)?;
    let float = |key: &str| c.get(key).and_then(Value::as_f64);
    let seq = |key: &str| {
        c.get(key)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
    };
    if let Some(v) = float("base_shift") {
        params.base_shift = v;
    }
    if let Some(v) = float("max_shift") {
        params.max_shift = v;
    }
    if let Some(v) = seq("base_image_seq_len") {
        params.base_image_seq_len = v;
    }
    if let Some(v) = seq("max_image_seq_len") {
        params.max_image_seq_len = v;
    }
    if let Some(v) = c.get("shift_terminal") {
        params.shift_terminal = v.as_f64().map(|t| t as f32);
    }
    if let Some(n) = c.get("num_train_timesteps").and_then(Value::as_f64)
        && n != 1000.0
    {
        bail!("Qwen-Image scheduler num_train_timesteps must be 1000, got {n}");
    }
    ensure!(
        params.max_image_seq_len > params.base_image_seq_len,
        "Qwen-Image scheduler max_image_seq_len must exceed base_image_seq_len"
    );
    Ok(params)
}

/// Upstream `validate_dimensions`: each side in `[256, 2048]`, a multiple of 16.
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

/// Python's `round` (half to even) of a finite f64 to an integer.
fn round_half_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Upstream `_image_dimensions(resolution, ratio)`: the `(width, height)`
/// with area about `resolution**2` at aspect `ratio`, each a multiple of 32.
#[must_use]
pub fn image_dimensions(resolution: u32, ratio: f64) -> (u32, u32) {
    let width = (f64::from(resolution) * f64::from(resolution) * ratio).sqrt();
    let w = (round_half_even(width / 32.0) * 32).max(32);
    let h = (round_half_even(width / ratio / 32.0) * 32).max(32);
    (w as u32, h as u32)
}

/// The output size of an edit with no size of its own: the reference resized
/// as upstream resizes it at the default `output_resolution`.
#[must_use]
pub fn default_edit_size(width: u32, height: u32) -> (u32, u32) {
    image_dimensions(
        DEFAULT_OUTPUT_RESOLUTION,
        f64::from(width) / f64::from(height.max(1)),
    )
}

/// `(1 + references) * tokens` of an edit, counting each reference at its
/// resized size.
#[must_use]
pub fn edit_image_tokens(width: u32, height: u32, references: &[(u32, u32)]) -> u64 {
    let grid = |w: u32, h: u32| u64::from(w / DOWNSAMPLE) * u64::from(h / DOWNSAMPLE);
    grid(width, height) + references.iter().map(|&(w, h)| grid(w, h)).sum::<u64>()
}

/// Request options (`--gen-kwargs` or extra JSON fields).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageOptions {
    /// Used when the request carries no negative prompt (default `" "`).
    pub negative_prompt: Option<String>,
    /// Edit only: the target area of each resized reference (default 1024).
    pub output_resolution: u32,
    /// Edit only: cache the prompt and reference keys and values after the
    /// first step (default true).
    pub use_kv_cache: bool,
}

fn bool_option(extra: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match extra.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => bail!("Qwen-Image {key} must be a boolean, got {v}"),
    }
}

impl QwenImageOptions {
    pub fn from_extra(extra: &Map<String, Value>, task: ImageTask) -> Result<Self> {
        if task == ImageTask::Generate {
            for key in ["output_resolution", "use_kv_cache"] {
                if extra.contains_key(key) {
                    bail!("Qwen-Image {key} applies to image edits only");
                }
            }
        }
        let negative_prompt = match extra.get("negative_prompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(v) => bail!("Qwen-Image negative_prompt must be a string, got {v}"),
        };
        let output_resolution = match extra.get("output_resolution") {
            None | Some(Value::Null) => DEFAULT_OUTPUT_RESOLUTION,
            Some(v) => match v.as_u64() {
                Some(r) if (u64::from(MIN_SIDE)..=u64::from(MAX_SIDE)).contains(&r) => r as u32,
                _ => bail!(
                    "Qwen-Image output_resolution must be an integer in [{MIN_SIDE}, {MAX_SIDE}], got {v}"
                ),
            },
        };
        Ok(Self {
            negative_prompt,
            output_resolution,
            use_kv_cache: bool_option(extra, "use_kv_cache", true)?,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
