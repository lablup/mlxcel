// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Video Depth Anything checkpoint configuration.

use crate::vision::encoders::dinov2::Dinov2Config;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

fn default_model_type() -> String {
    "video_depth_anything".into()
}
fn default_encoder() -> String {
    "vitl".into()
}
fn default_frames() -> usize {
    32
}
fn default_pe() -> String {
    "ape".into()
}
fn default_attention_heads() -> usize {
    8
}
fn default_transformer_blocks() -> usize {
    1
}
fn default_attention_blocks() -> usize {
    2
}
fn default_groups() -> usize {
    32
}
fn default_image_size() -> usize {
    518
}
fn default_patch_size() -> usize {
    14
}
fn default_mlp_ratio() -> f32 {
    4.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_offset() -> f32 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VdaConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_encoder")]
    pub encoder: String,
    pub metric: bool,
    pub features: Option<usize>,
    pub out_channels: Option<Vec<usize>>,
    pub use_bn: bool,
    pub use_clstoken: bool,
    #[serde(default = "default_frames")]
    pub num_frames: usize,
    #[serde(default = "default_pe")]
    pub pe: String,
    #[serde(default = "default_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_transformer_blocks")]
    pub num_transformer_block: usize,
    #[serde(default = "default_attention_blocks")]
    pub num_attention_blocks: usize,
    #[serde(default = "default_groups")]
    pub norm_num_groups: usize,
    pub embed_dim: Option<usize>,
    pub depth: Option<usize>,
    pub num_heads: Option<usize>,
    pub ffn: Option<String>,
    pub intermediate_layer_idx: Option<Vec<usize>>,
    #[serde(default = "default_image_size")]
    pub img_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_offset")]
    pub interpolate_offset: f32,
    pub interpolate_antialias: bool,
}

impl Default for VdaConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            encoder: default_encoder(),
            metric: false,
            features: None,
            out_channels: None,
            use_bn: false,
            use_clstoken: false,
            num_frames: default_frames(),
            pe: default_pe(),
            num_attention_heads: default_attention_heads(),
            num_transformer_block: default_transformer_blocks(),
            num_attention_blocks: default_attention_blocks(),
            norm_num_groups: default_groups(),
            embed_dim: None,
            depth: None,
            num_heads: None,
            ffn: None,
            intermediate_layer_idx: None,
            img_size: default_image_size(),
            patch_size: default_patch_size(),
            mlp_ratio: default_mlp_ratio(),
            layer_norm_eps: default_eps(),
            interpolate_offset: default_offset(),
            interpolate_antialias: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedVdaConfig {
    pub metric: bool,
    pub features: usize,
    pub out_channels: [usize; 4],
    pub num_frames: usize,
    pub num_attention_heads: usize,
    pub num_transformer_block: usize,
    pub num_attention_blocks: usize,
    pub norm_num_groups: usize,
    pub intermediate_layers: [usize; 4],
    pub dinov2: Dinov2Config,
}

impl VdaConfig {
    pub fn from_path(path: &Path) -> Result<ResolvedVdaConfig> {
        let config_path = path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        serde_json::from_str::<Self>(&raw)
            .with_context(|| format!("parse {}", config_path.display()))?
            .resolve()
    }

    pub fn resolve(self) -> Result<ResolvedVdaConfig> {
        ensure!(
            self.model_type == "video_depth_anything",
            "unsupported model_type `{}`",
            self.model_type
        );
        ensure!(
            !self.use_bn && !self.use_clstoken,
            "VDA batch norm and class-token fusion are unsupported"
        );
        ensure!(
            self.num_frames == 32,
            "VDA temporal window must be 32 frames"
        );
        ensure!(
            self.pe == "ape",
            "VDA supports only sinusoidal `ape` temporal positions"
        );
        ensure!(
            self.norm_num_groups > 0,
            "VDA norm_num_groups must be positive"
        );
        ensure!(
            self.ffn.as_deref().unwrap_or("mlp") == "mlp",
            "published VDA encoders require the DINOv2 MLP feed-forward layout"
        );
        let (hidden, depth, heads, features, channels, layers) = match self.encoder.as_str() {
            "vits" => (384, 12, 6, 64, [48, 96, 192, 384], [2, 5, 8, 11]),
            "vitb" => (768, 12, 12, 128, [96, 192, 384, 768], [2, 5, 8, 11]),
            "vitl" => (1024, 24, 16, 256, [256, 512, 1024, 1024], [4, 11, 17, 23]),
            other => anyhow::bail!("unknown VDA encoder `{other}`; expected vits, vitb, or vitl"),
        };
        let features = self.features.unwrap_or(features);
        let out_channels: [usize; 4] = self
            .out_channels
            .unwrap_or_else(|| channels.into())
            .try_into()
            .map_err(|_| anyhow::anyhow!("VDA out_channels must contain exactly four values"))?;
        let intermediate_layers: [usize; 4] = self
            .intermediate_layer_idx
            .unwrap_or_else(|| layers.into())
            .try_into()
            .map_err(|_| {
                anyhow::anyhow!("VDA intermediate_layer_idx must contain exactly four values")
            })?;
        ensure!(
            self.embed_dim.unwrap_or(hidden) == hidden
                && self.depth.unwrap_or(depth) == depth
                && self.num_heads.unwrap_or(heads) == heads,
            "VDA encoder dimensions must match the published `{}` preset",
            self.encoder
        );
        ensure!(
            features > 0 && features <= 512 && features.is_multiple_of(self.norm_num_groups),
            "VDA feature width must be positive, at most 512, and divisible by norm_num_groups"
        );
        ensure!(
            self.num_attention_heads == 8
                && self.num_transformer_block == 1
                && self.num_attention_blocks == 2,
            "VDA published temporal modules require 8 heads, one transformer block, and two attention blocks"
        );
        ensure!(
            out_channels.iter().all(|c| *c > 0 && *c <= 2048),
            "VDA output channels must be positive and at most 2048"
        );
        ensure!(
            [out_channels[2], out_channels[3], features, features]
                .iter()
                .all(|c| c.is_multiple_of(self.num_attention_heads)
                    && c.is_multiple_of(self.norm_num_groups)),
            "VDA temporal motion widths must be divisible by attention heads and norm groups"
        );
        let mut dinov2 = match self.encoder.as_str() {
            "vits" => Dinov2Config::vits14(),
            "vitb" => Dinov2Config::vitb14(),
            "vitl" => Dinov2Config::vitl14(),
            _ => unreachable!(),
        };
        dinov2.image_size = self.img_size;
        dinov2.patch_size = self.patch_size;
        dinov2.mlp_ratio = self.mlp_ratio;
        dinov2.layer_norm_eps = self.layer_norm_eps;
        dinov2.interpolate_offset = self.interpolate_offset;
        dinov2.interpolate_antialias = self.interpolate_antialias;
        dinov2.intermediate_layers = intermediate_layers.to_vec();
        dinov2.validate()?;
        Ok(ResolvedVdaConfig {
            metric: self.metric,
            features,
            out_channels,
            num_frames: self.num_frames,
            num_attention_heads: self.num_attention_heads,
            num_transformer_block: self.num_transformer_block,
            num_attention_blocks: self.num_attention_blocks,
            norm_num_groups: self.norm_num_groups,
            intermediate_layers,
            dinov2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolves_published_presets() {
        for (name, hidden, layers) in [
            ("vits", 384, [2, 5, 8, 11]),
            ("vitb", 768, [2, 5, 8, 11]),
            ("vitl", 1024, [4, 11, 17, 23]),
        ] {
            let cfg = VdaConfig {
                encoder: name.into(),
                ..Default::default()
            }
            .resolve()
            .unwrap();
            assert_eq!(cfg.dinov2.hidden_size, hidden);
            assert_eq!(cfg.intermediate_layers, layers);
        }
    }
}
