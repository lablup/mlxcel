// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Validated SAM 3D Objects bundle configuration.
use anyhow::{Result, ensure};
use serde::Deserialize;
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model_type: String,
    pub dtype: String,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub cond_channels: usize,
    pub latent_channels: usize,
    pub latent_resolution: usize,
    pub resolution: usize,
    pub io_channels: usize,
    pub decoder_channels: usize,
    pub decoder_heads: usize,
    pub decoder_blocks: usize,
    pub window_size: usize,
    pub structure_channels: Vec<usize>,
    pub structure_res_blocks: usize,
    pub dino_hidden_size: usize,
    pub dino_heads: usize,
    pub dino_layers: usize,
    pub dino_image_size: usize,
    pub image_size: usize,
    pub point_size: usize,
    pub point_patch_size: usize,
    pub point_channels: usize,
    pub point_heads: usize,
    pub ss_steps: usize,
    pub slat_steps: usize,
    pub ss_guidance: f32,
    pub slat_guidance: f32,
    pub ss_rescale_t: f64,
    pub slat_rescale_t: f64,
    pub downsample_ss_dist: usize,
    pub slat_mean: Vec<f32>,
    pub slat_std: Vec<f32>,
    pub depth_model: Option<super::super::moge3::Moge3Config>,
    pub depth_num_tokens: Option<usize>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "sam3d_objects".into(),
            dtype: "bfloat16".into(),
            hidden_size: 1024,
            num_heads: 16,
            num_blocks: 24,
            cond_channels: 1024,
            latent_channels: 8,
            latent_resolution: 16,
            resolution: 64,
            io_channels: 128,
            decoder_channels: 768,
            decoder_heads: 12,
            decoder_blocks: 12,
            window_size: 8,
            structure_channels: vec![512, 128, 32],
            structure_res_blocks: 2,
            dino_hidden_size: 1024,
            dino_heads: 16,
            dino_layers: 24,
            dino_image_size: 518,
            image_size: 518,
            point_size: 256,
            point_patch_size: 8,
            point_channels: 512,
            point_heads: 16,
            ss_steps: 25,
            slat_steps: 25,
            ss_guidance: 7.0,
            slat_guidance: 1.0,
            ss_rescale_t: 3.0,
            slat_rescale_t: 1.0,
            downsample_ss_dist: 1,
            slat_mean: vec![
                0.12211431,
                0.37204156,
                -1.26521907,
                -2.05276058,
                -3.10432536,
                -0.11294304,
                -0.85146744,
                0.45506954,
            ],
            slat_std: vec![
                2.37326008, 2.13174402, 2.2413953, 2.30589401, 2.1191894, 1.8969511, 2.41684989,
                2.08374642,
            ],
            depth_model: None,
            depth_num_tokens: None,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "sam3d_objects",
            "expected sam3d_objects model_type"
        );
        ensure!(
            self.dtype == "bfloat16",
            "Objects bundle must preserve BF16 weights"
        );
        for (channels, heads) in [
            (self.hidden_size, self.num_heads),
            (self.decoder_channels, self.decoder_heads),
            (self.dino_hidden_size, self.dino_heads),
            (self.point_channels, self.point_heads),
        ] {
            ensure!(
                channels > 0 && channels <= 4096 && heads > 0 && channels.is_multiple_of(heads),
                "invalid Objects attention dimensions"
            );
        }
        ensure!(
            self.num_blocks > 0
                && self.num_blocks <= 64
                && self.decoder_blocks > 0
                && self.decoder_blocks <= 64
                && self.dino_layers > 0
                && self.dino_layers <= 64,
            "invalid Objects block count"
        );
        ensure!(
            self.latent_channels == 8
                && self.latent_resolution == 16
                && self.resolution == 64
                && self.window_size == 8,
            "unsupported Objects grid configuration"
        );
        ensure!(
            self.image_size > 0
                && self.image_size <= 518
                && self.image_size.is_multiple_of(14)
                && self.dino_image_size == 518,
            "unsupported Objects image configuration"
        );
        ensure!(
            self.point_size == 256
                && self.point_patch_size == 8
                && self.io_channels > 0
                && self.io_channels <= 1024
                && self.cond_channels > 0
                && self.cond_channels <= 4096,
            "unsupported Objects point configuration"
        );
        ensure!(
            self.structure_channels.len() == 3
                && self.structure_channels.iter().all(|v| *v > 0 && *v <= 1024)
                && self.structure_res_blocks > 0
                && self.structure_res_blocks <= 8,
            "invalid Objects structure channels"
        );
        ensure!(
            self.downsample_ss_dist <= 1,
            "only one-voxel occupancy pruning is supported"
        );
        ensure!(
            self.ss_steps > 0 && self.slat_steps > 0,
            "sampling steps must be positive"
        );
        ensure!(
            [self.ss_guidance, self.slat_guidance]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
                && [self.ss_rescale_t, self.slat_rescale_t]
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0),
            "invalid sampling parameters"
        );
        ensure!(
            self.slat_mean.len() == 8
                && self.slat_std.len() == 8
                && self.slat_mean.iter().all(|v| v.is_finite())
                && self.slat_std.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid latent statistics"
        );
        if let Some(depth) = &self.depth_model {
            depth.validate()?;
        }
        Ok(())
    }
}
