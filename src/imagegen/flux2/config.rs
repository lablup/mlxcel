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

//! FLUX.2 Klein variants, transformer shape and checkpoint layout.
//!
//! A port of `Flux2Variant`, `VARIANTS`, `variant_from_local_path` and
//! `validate_dimensions` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/config.py>.
//! Everything here reads small JSON files only, so sampling defaults are known
//! before any weight is loaded.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

/// Smallest and largest output side FLUX.2 accepts.
pub const MIN_SIDE: u32 = 256;
pub const MAX_SIDE: u32 = 2048;
/// Output sides must be multiples of this (VAE scale 8 times the 2x2 patch).
pub const SIDE_MULTIPLE: u32 = 16;

/// One of the published FLUX.2 Klein checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Flux2Variant {
    Klein4b,
    Klein9b,
    KleinBase4b,
    KleinBase9b,
    Klein9bKv,
}

/// Shape of the Qwen3 conditioner a variant ships with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEncoderShape {
    pub hidden_size: i32,
    pub intermediate_size: i32,
}

impl Flux2Variant {
    /// Upstream variant name (`flux2-klein-4b`, ...).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Klein4b => "flux2-klein-4b",
            Self::Klein9b => "flux2-klein-9b",
            Self::KleinBase4b => "flux2-klein-base-4b",
            Self::KleinBase9b => "flux2-klein-base-9b",
            Self::Klein9bKv => "flux2-klein-9b-kv",
        }
    }

    /// Default step count: 4 for the distilled models, 50 for base.
    #[must_use]
    pub fn default_steps(self) -> u32 {
        match self {
            Self::KleinBase4b | Self::KleinBase9b => 50,
            _ => 4,
        }
    }

    /// Default guidance: 1.0 (no CFG) for the distilled models, 4.0 for base.
    #[must_use]
    pub fn default_guidance(self) -> f32 {
        match self {
            Self::KleinBase4b | Self::KleinBase9b => 4.0,
            _ => 1.0,
        }
    }

    /// Whether edits extract reference K/V on step 0 and reuse it.
    #[must_use]
    pub fn uses_reference_kv_cache(self) -> bool {
        self == Self::Klein9bKv
    }

    fn is_9b(self) -> bool {
        matches!(self, Self::Klein9b | Self::KleinBase9b | Self::Klein9bKv)
    }

    /// The transformer shape upstream builds for this variant.
    #[must_use]
    pub fn transformer_config(self) -> Flux2TransformerConfig {
        let mut config = Flux2TransformerConfig::default();
        if self.is_9b() {
            config.num_layers = 8;
            config.num_single_layers = 24;
            config.num_attention_heads = 32;
            config.joint_attention_dim = 12288;
        }
        config
    }

    /// The text-encoder shape upstream builds for this variant.
    #[must_use]
    pub fn text_encoder_shape(self) -> TextEncoderShape {
        if self.is_9b() {
            TextEncoderShape {
                hidden_size: 4096,
                intermediate_size: 12288,
            }
        } else {
            TextEncoderShape {
                hidden_size: 2560,
                intermediate_size: 9728,
            }
        }
    }

    fn with_size(distilled: bool, nine_b: bool) -> Self {
        match (distilled, nine_b) {
            (true, false) => Self::Klein4b,
            (true, true) => Self::Klein9b,
            (false, false) => Self::KleinBase4b,
            (false, true) => Self::KleinBase9b,
        }
    }

    /// Resolve the variant of a local checkpoint directory, as upstream's
    /// `variant_from_local_path` does: the directory name first (a Hugging Face
    /// cache snapshot uses its repository name), then `model_index.json`
    /// `is_distilled` with the transformer or text-encoder shape.
    pub fn from_dir(root: &Path) -> Result<Self> {
        let mut name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if let Some(parent) = root.parent()
            && parent.file_name().is_some_and(|n| n == "snapshots")
            && let Some(repo) = parent.parent().and_then(Path::file_name)
            && let Some(repo) = repo.to_string_lossy().strip_prefix("models--")
        {
            name = repo.replace("--", "/").to_lowercase();
        }
        if name.contains("kv") || root.join("flux-2-klein-9b-kv.safetensors").exists() {
            return Ok(Self::Klein9bKv);
        }
        let base = name.contains("base");
        if name.contains("9b") {
            return Ok(Self::with_size(!base, true));
        }
        if name.contains("4b") {
            return Ok(Self::with_size(!base, false));
        }

        let distilled = read_json(&root.join("model_index.json"))
            .and_then(|index| index.get("is_distilled").and_then(Value::as_bool))
            .unwrap_or(true);
        if let Some(config) = read_json(&root.join("transformer").join("config.json")) {
            let layers = config.get("num_layers").and_then(Value::as_i64);
            let heads = config.get("num_attention_heads").and_then(Value::as_i64);
            if layers == Some(8) || heads == Some(32) {
                return Ok(Self::with_size(distilled, true));
            }
            if layers == Some(5) || heads == Some(24) {
                return Ok(Self::with_size(distilled, false));
            }
        }
        if let Some(config) = read_json(&root.join("text_encoder").join("config.json")) {
            match config.get("hidden_size").and_then(Value::as_i64) {
                Some(4096) => return Ok(Self::with_size(distilled, true)),
                Some(2560) => return Ok(Self::with_size(distilled, false)),
                _ => {}
            }
        }
        bail!(
            "Could not infer the FLUX.2 variant from {}: use a directory whose name \
             contains 4b or 9b, or one with transformer/config.json",
            root.display()
        )
    }
}

/// `Flux2Transformer` constructor arguments (Diffusers `config.json` names).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Flux2TransformerConfig {
    pub patch_size: i32,
    pub in_channels: i32,
    pub out_channels: Option<i32>,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub attention_head_dim: i32,
    pub num_attention_heads: i32,
    pub joint_attention_dim: i32,
    pub timestep_guidance_channels: i32,
    pub mlp_ratio: f32,
    pub axes_dims_rope: Vec<i32>,
    pub rope_theta: f32,
    pub guidance_embeds: bool,
}

impl Default for Flux2TransformerConfig {
    fn default() -> Self {
        Self {
            patch_size: 1,
            in_channels: 128,
            out_channels: None,
            num_layers: 5,
            num_single_layers: 20,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 7680,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: false,
        }
    }
}

impl Flux2TransformerConfig {
    /// `num_attention_heads * attention_head_dim`.
    #[must_use]
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Width of each half of the SwiGLU input: `int(dim * mlp_ratio)`.
    #[must_use]
    pub fn mlp_hidden_dim(&self) -> i32 {
        (self.inner_dim() as f32 * self.mlp_ratio) as i32
    }

    /// Output channels; `None` in `config.json` means `in_channels`.
    #[must_use]
    pub fn out_channels(&self) -> i32 {
        self.out_channels.unwrap_or(self.in_channels)
    }

    /// Read `<transformer_dir>/config.json`, or fall back to the variant's
    /// shape when it is absent (MLX-quantized exports ship none).
    pub fn resolve(transformer_dir: &Path, variant: Flux2Variant) -> Result<Self> {
        let path = transformer_dir.join("config.json");
        if !path.exists() {
            return Ok(variant.transformer_config());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let config: Self = serde_json::from_str(&text)
            .with_context(|| format!("invalid FLUX.2 transformer config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Check the fields the transformer depends on.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size != 1 {
            bail!(
                "FLUX.2 transformer patch_size {} is not supported (expected 1)",
                self.patch_size
            );
        }
        let rope: i32 = self.axes_dims_rope.iter().sum();
        if self.axes_dims_rope.len() != 4 || rope != self.attention_head_dim {
            bail!(
                "FLUX.2 axes_dims_rope {:?} must have 4 axes summing to attention_head_dim {}",
                self.axes_dims_rope,
                self.attention_head_dim
            );
        }
        if self.axes_dims_rope.iter().any(|d| *d <= 0 || d % 2 != 0) {
            bail!("FLUX.2 axes_dims_rope entries must be positive and even");
        }
        if self.num_attention_heads < 1 || self.attention_head_dim < 1 || self.in_channels < 1 {
            bail!("FLUX.2 transformer dimensions must be positive");
        }
        Ok(())
    }
}

/// The four component directories a FLUX.2 checkpoint must have.
#[derive(Debug, Clone)]
pub struct Flux2Layout {
    pub root: PathBuf,
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    pub tokenizer: PathBuf,
}

impl Flux2Layout {
    /// Check that `transformer/`, `text_encoder/`, `vae/` and `tokenizer/`
    /// exist, naming the first one that does not.
    pub fn from_dir(root: &Path) -> Result<Self> {
        if !root.is_dir() {
            bail!("FLUX.2 model directory does not exist: {}", root.display());
        }
        let layout = Self {
            root: root.to_path_buf(),
            transformer: root.join("transformer"),
            text_encoder: root.join("text_encoder"),
            vae: root.join("vae"),
            tokenizer: root.join("tokenizer"),
        };
        for dir in [
            &layout.transformer,
            &layout.text_encoder,
            &layout.vae,
            &layout.tokenizer,
        ] {
            if !dir.is_dir() {
                bail!("FLUX.2 checkpoint is missing {}", dir.display());
            }
        }
        let tokenizer_json = layout.tokenizer.join("tokenizer.json");
        if !tokenizer_json.is_file() {
            bail!("FLUX.2 checkpoint is missing {}", tokenizer_json.display());
        }
        Ok(layout)
    }
}

/// Upstream `validate_dimensions`: each side in `256..=2048` and a multiple of 16.
pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    for (label, value) in [("width", width), ("height", height)] {
        if !(MIN_SIDE..=MAX_SIDE).contains(&value) {
            bail!("{label} must be in [{MIN_SIDE}, {MAX_SIDE}], got {value}");
        }
        if !value.is_multiple_of(SIDE_MULTIPLE) {
            bail!("{label} must be a multiple of {SIDE_MULTIPLE}, got {value}");
        }
    }
    Ok(())
}

/// MLX affine quantization parameters of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
    pub bits: i32,
    pub group_size: i32,
}

/// Quantization metadata of a component directory, read the way upstream's
/// `_quantization_config` reads it: `bits` / `num_bits` / `quantization_level`
/// and `group_size` / `quantization_group_size`, at the top level of the
/// safetensors index `metadata` or under its `quantization` object. `None`
/// when the directory names no quantization; shapes still decide per layer.
pub fn read_quant_params(dir: &Path) -> Result<Option<QuantParams>> {
    let Some(index) = read_json(&dir.join("model.safetensors.index.json")) else {
        return Ok(None);
    };
    let Some(metadata) = index.get("metadata").filter(|m| m.is_object()) else {
        return Ok(None);
    };
    let bits = metadata_int(metadata, &["bits", "num_bits", "quantization_level"])?;
    let group_size = metadata_int(metadata, &["group_size", "quantization_group_size"])?;
    if let Some(mode) = metadata_value(metadata, &["mode"]).and_then(Value::as_str)
        && mode != "affine"
    {
        bail!("Unsupported FLUX.2 quantization mode: {mode}");
    }
    Ok(match (bits, group_size) {
        (Some(bits), Some(group_size)) => Some(QuantParams { bits, group_size }),
        (Some(bits), None) => Some(QuantParams {
            bits,
            group_size: 64,
        }),
        (None, Some(group_size)) => Some(QuantParams {
            bits: 4,
            group_size,
        }),
        (None, None) => None,
    })
}

fn metadata_value<'a>(metadata: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let mut candidates = vec![metadata];
    for container in ["quantization", "quantization_config"] {
        if let Some(inner) = metadata.get(container).filter(|v| v.is_object()) {
            candidates.push(inner);
        }
    }
    candidates
        .into_iter()
        .find_map(|c| keys.iter().find_map(|k| c.get(*k)))
}

fn metadata_int(metadata: &Value, keys: &[&str]) -> Result<Option<i32>> {
    let Some(value) = metadata_value(metadata, keys) else {
        return Ok(None);
    };
    let parsed = match value {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    parsed
        .and_then(|v| i32::try_from(v).ok())
        .map(Some)
        .with_context(|| format!("Invalid quantization metadata value: {value}"))
}

/// The Qwen3 `config.json` the text encoder is built from.
///
/// `text_encoder/config.json` when the checkpoint ships one; otherwise (MLX
/// quantized exports) upstream's `Qwen3TextEncoder` defaults with the
/// variant's hidden and intermediate sizes, plus the index's quantization.
pub fn text_encoder_config(
    text_encoder_dir: &Path,
    variant: Flux2Variant,
    quant: Option<QuantParams>,
) -> Result<Value> {
    let path = text_encoder_dir.join("config.json");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        return serde_json::from_str(&crate::models::sanitize_config_json(&text))
            .with_context(|| format!("cannot parse {}", path.display()));
    }
    let shape = variant.text_encoder_shape();
    let mut config = json!({
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": shape.hidden_size,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": shape.intermediate_size,
        "max_position_embeddings": 40960,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "head_dim": 128,
        "attention_bias": false,
        "tie_word_embeddings": true,
    });
    if let Some(q) = quant {
        config["quantization"] = json!({ "bits": q.bits, "group_size": q.group_size });
    }
    Ok(config)
}

fn read_json(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
