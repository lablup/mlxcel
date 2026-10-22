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

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

pub const MAX_DINOV2_HIDDEN_SIZE: usize = 4_096;
pub const MAX_DINOV2_LAYERS: usize = 128;
pub const MAX_DINOV2_INTERMEDIATE_SIZE: usize = 65_536;
pub const MAX_DINOV2_IMAGE_DIMENSION: usize = 16_384;
pub const MAX_DINOV2_PATCH_SIZE: usize = 128;
pub const MAX_DINOV2_PATCH_TOKENS: usize = 16_384;
pub const MAX_DINOV2_BATCH_PATCH_TOKENS: usize = 65_536;
pub const MAX_DINOV2_BATCH_SEQUENCE_TOKENS: usize = 65_536;
pub const MAX_DINOV2_REGISTER_TOKENS: usize = 256;
pub const MAX_DINOV2_RUNTIME_PIXELS: usize = 32 * 1024 * 1024;

fn default_model_type() -> String {
    "dinov2".into()
}

fn default_hidden_size() -> usize {
    768
}

fn default_depth() -> usize {
    12
}

fn default_heads() -> usize {
    12
}

fn default_mlp_ratio() -> f32 {
    4.0
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

fn default_image_size() -> usize {
    518
}

fn default_patch_size() -> usize {
    14
}

fn default_num_channels() -> usize {
    3
}

fn default_true() -> bool {
    true
}

fn default_layerscale() -> f32 {
    1.0
}

fn default_hidden_act() -> String {
    "gelu".into()
}

fn default_intermediate_layers() -> Vec<usize> {
    vec![4, 11]
}

fn deserialize_size<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Size {
        Scalar(usize),
        Pair(Vec<usize>),
    }

    match Size::deserialize(deserializer)? {
        Size::Scalar(value) => Ok(value),
        Size::Pair(values) => match values.as_slice() {
            [height, width] if height == width => Ok(*height),
            [_, _] => Err(serde::de::Error::custom(
                "DINOv2 supports only square image and patch sizes",
            )),
            _ => Err(serde::de::Error::custom(
                "DINOv2 size lists must contain exactly two values",
            )),
        },
    }
}

// Used by: Dinov2, VideoDepthAnything, MoGe3, Sam3dObjects
/// Hugging Face-compatible DINOv2 architecture configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Dinov2Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_depth")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_image_size", deserialize_with = "deserialize_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size", deserialize_with = "deserialize_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_true")]
    pub qkv_bias: bool,
    #[serde(default = "default_layerscale")]
    pub layerscale_value: f32,
    pub use_swiglu_ffn: bool,
    pub num_register_tokens: usize,
    pub interpolate_offset: f32,
    pub interpolate_antialias: bool,
    #[serde(default = "default_intermediate_layers")]
    pub intermediate_layers: Vec<usize>,
}

impl Default for Dinov2Config {
    fn default() -> Self {
        Self::vitb14()
    }
}

impl Dinov2Config {
    fn preset(hidden_size: usize, depth: usize, heads: usize, swiglu: bool) -> Self {
        Self {
            model_type: default_model_type(),
            hidden_size,
            num_hidden_layers: depth,
            num_attention_heads: heads,
            mlp_ratio: default_mlp_ratio(),
            hidden_act: default_hidden_act(),
            layer_norm_eps: default_layer_norm_eps(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            num_channels: default_num_channels(),
            qkv_bias: true,
            layerscale_value: default_layerscale(),
            use_swiglu_ffn: swiglu,
            num_register_tokens: 0,
            interpolate_offset: 0.0,
            interpolate_antialias: false,
            intermediate_layers: default_intermediate_layers(),
        }
    }

    pub fn vits14() -> Self {
        Self::preset(384, 12, 6, false)
    }

    pub fn vitb14() -> Self {
        Self::preset(768, 12, 12, false)
    }

    pub fn vitl14() -> Self {
        let mut config = Self::preset(1024, 24, 16, false);
        config.intermediate_layers = vec![4, 11, 17, 23];
        config
    }

    pub fn vitg14() -> Self {
        let mut config = Self::preset(1536, 40, 24, true);
        config.intermediate_layers = vec![9, 19, 29, 39];
        config
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let config: Self = serde_json::from_value(value)
            .with_context(|| format!("decode {} as DINOv2", config_path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.model_type.as_str(), "dinov2" | "dinov2_with_registers"),
            "unsupported DINOv2 model_type `{}`",
            self.model_type
        );
        ensure!(
            self.hidden_size > 0 && self.hidden_size <= MAX_DINOV2_HIDDEN_SIZE,
            "DINOv2 hidden_size must be between 1 and {MAX_DINOV2_HIDDEN_SIZE}"
        );
        ensure!(
            self.hidden_size <= i32::MAX as usize / 3,
            "DINOv2 hidden_size or fused QKV size exceeds the MLX shape range"
        );
        ensure!(
            self.num_hidden_layers > 0 && self.num_hidden_layers <= MAX_DINOV2_LAYERS,
            "DINOv2 num_hidden_layers must be between 1 and {MAX_DINOV2_LAYERS}"
        );
        ensure!(
            self.num_attention_heads > 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads),
            "DINOv2 hidden_size must be divisible by num_attention_heads"
        );
        ensure!(self.patch_size > 0, "DINOv2 patch_size must be positive");
        ensure!(
            self.image_size > 0 && self.image_size.is_multiple_of(self.patch_size),
            "DINOv2 image_size must be a positive patch multiple"
        );
        ensure!(
            self.image_size <= MAX_DINOV2_IMAGE_DIMENSION
                && self.patch_size <= MAX_DINOV2_PATCH_SIZE,
            "DINOv2 image or patch size exceeds the supported limits ({MAX_DINOV2_IMAGE_DIMENSION}, {MAX_DINOV2_PATCH_SIZE})"
        );
        let base_grid = self.image_size / self.patch_size;
        let base_tokens = base_grid
            .checked_mul(base_grid)
            .and_then(|tokens| tokens.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("DINOv2 base token count overflowed"))?;
        ensure!(
            base_tokens <= MAX_DINOV2_PATCH_TOKENS + 1,
            "DINOv2 base patch count exceeds the safety limit {MAX_DINOV2_PATCH_TOKENS}"
        );
        ensure!(
            self.num_channels == 3,
            "DINOv2 supports RGB checkpoints only"
        );
        ensure!(
            self.mlp_ratio.is_finite() && self.mlp_ratio > 0.0,
            "DINOv2 mlp_ratio must be finite and positive"
        );
        let dense_intermediate = self.hidden_size as f64 * self.mlp_ratio as f64;
        ensure!(
            dense_intermediate <= MAX_DINOV2_INTERMEDIATE_SIZE as f64,
            "DINOv2 MLP intermediate size exceeds the safety limit {MAX_DINOV2_INTERMEDIATE_SIZE}"
        );
        if self.use_swiglu_ffn {
            let reduced = ((dense_intermediate as usize * 2 / 3) + 7) / 8 * 8;
            ensure!(
                reduced <= i32::MAX as usize / 2,
                "DINOv2 SwiGLU projection size exceeds the MLX shape range"
            );
        }
        ensure!(
            self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.0,
            "DINOv2 layer_norm_eps must be finite and positive"
        );
        ensure!(
            self.layerscale_value.is_finite(),
            "DINOv2 layerscale_value must be finite"
        );
        ensure!(
            self.interpolate_offset.is_finite() && self.interpolate_offset >= 0.0,
            "DINOv2 interpolate_offset must be finite and non-negative"
        );
        ensure!(
            self.num_register_tokens <= MAX_DINOV2_REGISTER_TOKENS,
            "DINOv2 register token count exceeds the safety limit {MAX_DINOV2_REGISTER_TOKENS}"
        );
        ensure!(
            self.intermediate_layers.len() <= self.num_hidden_layers,
            "DINOv2 intermediate layer list cannot exceed num_hidden_layers"
        );
        ensure!(
            self.intermediate_layers
                .iter()
                .all(|index| *index < self.num_hidden_layers),
            "DINOv2 intermediate layer indices must be smaller than num_hidden_layers"
        );
        let mut unique_layers = self.intermediate_layers.clone();
        unique_layers.sort_unstable();
        unique_layers.dedup();
        ensure!(
            unique_layers.len() == self.intermediate_layers.len(),
            "DINOv2 intermediate layer indices must be unique"
        );
        ensure!(
            self.hidden_act == "gelu",
            "DINOv2 supports hidden_act=gelu only"
        );
        Ok(())
    }
}
