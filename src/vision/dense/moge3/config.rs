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

//! Validated MoGe3 checkpoint and inference configuration.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::vision::encoders::dinov2::Dinov2Config;

pub use super::config_components::{ConvStackConfig, RefinerConfig, ScaleHeadConfig};
use super::config_components::{MAX_CHANNELS, default_head, default_neck};
use super::sparse::neighbor_construction_peak_bytes;

pub const MAX_REFINE_STEPS: usize = 16;
pub const MAX_RUNTIME_PIXELS: usize = 32 * 1024 * 1024;
pub const MAX_DECODER_PIXELS: usize = 2_000_000;
pub const MAX_SPARSE_WORKING_BYTES: usize = 512 << 20;

fn default_model_type() -> String {
    "moge3".into()
}
fn default_backbone() -> String {
    "dinov2_vitl14".into()
}
fn default_layers() -> Vec<usize> {
    vec![5, 11, 17, 23]
}
fn default_dim_out() -> usize {
    1024
}
fn default_img_size() -> usize {
    518
}
fn default_patch_size() -> usize {
    14
}
fn default_mlp_ratio() -> f32 {
    4.0
}
fn default_interpolate_offset() -> f32 {
    0.1
}
fn default_eps() -> f32 {
    1e-6
}
fn default_token_range() -> [usize; 2] {
    [1200, 3600]
}
fn default_refiner_depth_resolution() -> usize {
    256
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EncoderConfig {
    #[serde(default = "default_backbone")]
    pub backbone: String,
    #[serde(default = "default_layers")]
    pub intermediate_layers: Vec<usize>,
    #[serde(default = "default_dim_out")]
    pub dim_out: usize,
    #[serde(default = "default_img_size")]
    pub img_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_interpolate_offset")]
    pub interpolate_offset: f32,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            backbone: default_backbone(),
            intermediate_layers: default_layers(),
            dim_out: default_dim_out(),
            img_size: default_img_size(),
            patch_size: default_patch_size(),
            mlp_ratio: default_mlp_ratio(),
            interpolate_offset: default_interpolate_offset(),
            layer_norm_eps: default_eps(),
        }
    }
}

impl EncoderConfig {
    pub fn dinov2(&self) -> Result<Dinov2Config> {
        let mut config = match self.backbone.as_str() {
            "dinov2_vitl14" => Dinov2Config::vitl14(),
            "dinov2_vitg14" => Dinov2Config::vitg14(),
            other => anyhow::bail!(
                "unsupported MoGe3 backbone `{other}`; expected dinov2_vitl14 or dinov2_vitg14"
            ),
        };
        config.intermediate_layers = self.intermediate_layers.clone();
        config.image_size = self.img_size;
        config.patch_size = self.patch_size;
        config.mlp_ratio = self.mlp_ratio;
        config.interpolate_offset = self.interpolate_offset;
        config.layer_norm_eps = self.layer_norm_eps;
        config.validate()?;
        ensure!(
            config.hidden_size == self.dim_out,
            "MoGe3 encoder dim_out must equal DINOv2 hidden size"
        );
        ensure!(
            self.intermediate_layers.len() == 4,
            "MoGe3 requires exactly four intermediate encoder layers"
        );
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Moge3Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub encoder: EncoderConfig,
    pub neck: ConvStackConfig,
    pub points_head: Option<ConvStackConfig>,
    pub mask_head: Option<ConvStackConfig>,
    pub normal_head: Option<ConvStackConfig>,
    pub scale_head: Option<ScaleHeadConfig>,
    #[serde(default = "default_token_range")]
    pub num_tokens_range: [usize; 2],
    pub refiner: Option<RefinerConfig>,
    #[serde(default = "default_refiner_depth_resolution")]
    pub refiner_depth_resolution: usize,
}
impl Default for Moge3Config {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            encoder: EncoderConfig::default(),
            neck: default_neck(1024),
            points_head: Some(default_head(3)),
            mask_head: Some(default_head(1)),
            normal_head: Some(default_head(3)),
            scale_head: Some(ScaleHeadConfig::default()),
            num_tokens_range: default_token_range(),
            refiner: Some(RefinerConfig::default()),
            refiner_depth_resolution: 256,
        }
    }
}
impl Moge3Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        let p = path.join("config.json");
        let raw = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let config: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "moge3",
            "unsupported MoGe3 model_type `{}`",
            self.model_type
        );
        let dino = self.encoder.dinov2()?;
        self.neck.validate("neck")?;
        ensure!(
            self.neck.dim_in.first().copied().flatten() == Some(self.encoder.dim_out + 2),
            "MoGe3 neck first input must be encoder dim + UV"
        );
        for (name, head) in [
            ("points_head", &self.points_head),
            ("mask_head", &self.mask_head),
            ("normal_head", &self.normal_head),
        ] {
            let v = head
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MoGe3 {name} is required"))?;
            v.validate(name)?;
            ensure!(
                v.dim_in.len() == self.neck.dim_res_blocks.len()
                    && v.dim_in
                        .iter()
                        .zip(&self.neck.dim_res_blocks)
                        .all(|(input, residual)| *input == Some(*residual)),
                "MoGe3 {name} input pyramid mismatch"
            );
        }
        for (name, head, expected) in [
            ("points_head", self.points_head.as_ref(), 3),
            ("mask_head", self.mask_head.as_ref(), 1),
            ("normal_head", self.normal_head.as_ref(), 3),
        ] {
            let outputs = head.expect("required head").output_dims();
            ensure!(
                outputs.last().copied().flatten() == Some(expected),
                "MoGe3 {name} final output must have {expected} channels"
            );
        }
        let scale = self
            .scale_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MoGe3 scale_head is required"))?;
        ensure!(
            scale.dims.len() >= 2
                && scale.dims.len() <= 8
                && scale.dims[0] == dino.hidden_size
                && scale.dims.last() == Some(&1)
                && scale.dims.iter().all(|&v| v > 0 && v <= MAX_CHANNELS),
            "MoGe3 scale head dimensions invalid"
        );
        ensure!(
            self.num_tokens_range[0] > 0
                && self.num_tokens_range[0] <= self.num_tokens_range[1]
                && self.num_tokens_range[1] <= 16_384,
            "MoGe3 token range invalid"
        );
        let finest_factor = 1usize
            .checked_shl((2 * (self.neck.dim_res_blocks.len() - 1)) as u32)
            .ok_or_else(|| anyhow::anyhow!("MoGe3 decoder scale overflow"))?;
        let max_finest = self.num_tokens_range[1]
            .checked_mul(finest_factor)
            .ok_or_else(|| anyhow::anyhow!("MoGe3 decoder pixel budget overflow"))?;
        ensure!(
            max_finest <= MAX_DECODER_PIXELS,
            "MoGe3 configured decoder can exceed {MAX_DECODER_PIXELS} active pixels"
        );
        ensure!(
            self.refiner_depth_resolution > 0 && self.refiner_depth_resolution <= 16_384,
            "MoGe3 refiner depth resolution invalid"
        );
        if let Some(v) = &self.refiner {
            v.validate()?;
            ensure!(
                v.in_channels == 3 && v.out_channels == 1,
                "MoGe3 refiner must map 3 input channels to 1 output channel"
            );
            ensure!(
                v.encoder_channels == self.encoder.dim_out + 2,
                "MoGe3 refiner encoder channels mismatch"
            );
            let finest_features = max_finest
                .checked_mul(v.model_channels[0])
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse working budget overflow"))?;
            ensure!(
                finest_features <= MAX_SPARSE_WORKING_BYTES,
                "MoGe3 configured sparse features exceed {MAX_SPARSE_WORKING_BYTES} bytes"
            );
            let retained_map = super::sparse::neighbor_map_bytes(max_finest)?;
            let construction_peak =
                neighbor_construction_peak_bytes(max_finest, v.model_channels[0], retained_map)?;
            ensure!(
                construction_peak <= MAX_SPARSE_WORKING_BYTES,
                "MoGe3 configured sparse construction requires {construction_peak} bytes"
            );
        }
        Ok(())
    }
}
