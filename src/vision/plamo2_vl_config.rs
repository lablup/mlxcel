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

//! Config structs for PLaMo 2.1 VL (`model_type: "plamo2vl"`, issue #1380).
//!
//! The checkpoint flattens its SigLIP tower config into `vision_config` with a
//! `vision_encoder_` prefix and an empty `model_type`, next to the unprefixed
//! adapter keys (`image_token_id`, `image_feature_size`,
//! `image_proj_hidden_size`). The tiler reads `preprocessor_config.json`.
//!
//! Used by: `loading::load_plamo2_vl`, `vision::processors::plamo2_vl`.

use std::path::Path;

use serde::Deserialize;

use crate::vision::config::{VisionConfig, VisionHiddenActivation};

/// `vision_config` of a PLaMo 2.1 VL checkpoint. Defaults are the values both
/// shipped sizes (2B, 8B) carry.
#[derive(Debug, Clone, Deserialize)]
pub struct Plamo2VlVisionConfig {
    #[serde(rename = "vision_encoder_hidden_size", default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(
        rename = "vision_encoder_num_hidden_layers",
        default = "default_layers"
    )]
    pub num_hidden_layers: usize,
    #[serde(
        rename = "vision_encoder_intermediate_size",
        default = "default_intermediate_size"
    )]
    pub intermediate_size: usize,
    #[serde(
        rename = "vision_encoder_num_attention_heads",
        default = "default_heads"
    )]
    pub num_attention_heads: usize,
    #[serde(rename = "vision_encoder_patch_size", default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(rename = "vision_encoder_image_size", default = "default_image_size")]
    pub image_size: usize,
    #[serde(rename = "vision_encoder_num_channels", default = "default_channels")]
    pub num_channels: usize,
    #[serde(rename = "vision_encoder_layer_norm_eps", default = "default_eps")]
    pub layer_norm_eps: f32,
    #[serde(rename = "vision_encoder_hidden_act", default = "default_hidden_act")]
    pub hidden_act: String,
    /// `<plamo:image>`, the placeholder the adapter output is scattered onto.
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    /// Tower output width the adapter's `norm0` / `bias0` act on.
    #[serde(default = "default_hidden_size")]
    pub image_feature_size: usize,
    /// Adapter inner width (`linear1` output): 2048 (2B) / 4096 (8B).
    #[serde(default)]
    pub image_proj_hidden_size: Option<usize>,
}

fn default_hidden_size() -> usize {
    1152
}
fn default_layers() -> usize {
    27
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_image_size() -> usize {
    384
}
fn default_channels() -> usize {
    3
}
fn default_eps() -> f32 {
    1e-6
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_image_token_id() -> i32 {
    100_002
}

impl Plamo2VlVisionConfig {
    /// The shared SigLIP tower config (`siglip_vision_model`, no CLS token,
    /// learned absolute positions over `(image_size / patch_size)^2` patches).
    pub fn to_siglip_config(&self) -> Result<VisionConfig, String> {
        Ok(VisionConfig {
            model_type: "siglip_vision_model".to_string(),
            num_hidden_layers: self.num_hidden_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            patch_size: self.patch_size,
            image_size: self.image_size,
            num_channels: self.num_channels,
            layer_norm_eps: self.layer_norm_eps,
            hidden_act: VisionHiddenActivation::try_from(self.hidden_act.clone())?,
        })
    }
}

/// `size` in `preprocessor_config.json`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Plamo2VlSize {
    pub width: usize,
    #[serde(default)]
    pub height: Option<usize>,
}

/// The tiler's inputs from `preprocessor_config.json` (a `SiglipImageProcessor`
/// plus the PLaMo tiling keys).
#[derive(Debug, Clone, Deserialize)]
pub struct Plamo2VlProcessorConfig {
    #[serde(default = "default_size")]
    pub size: Plamo2VlSize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_downsample_ratio")]
    pub downsample_ratio: f32,
    #[serde(default = "default_tile_context_length")]
    pub tile_context_length: usize,
    #[serde(default = "default_half")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_half")]
    pub image_std: [f32; 3],
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
}

fn default_size() -> Plamo2VlSize {
    Plamo2VlSize {
        width: 384,
        height: Some(384),
    }
}
fn default_downsample_ratio() -> f32 {
    1.0
}
fn default_tile_context_length() -> usize {
    3072
}
fn default_half() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

impl Default for Plamo2VlProcessorConfig {
    fn default() -> Self {
        Self {
            size: default_size(),
            patch_size: default_patch_size(),
            downsample_ratio: default_downsample_ratio(),
            tile_context_length: default_tile_context_length(),
            image_mean: default_half(),
            image_std: default_half(),
            rescale_factor: default_rescale_factor(),
        }
    }
}

impl Plamo2VlProcessorConfig {
    /// Read `preprocessor_config.json` from `model_dir`, falling back to the
    /// shipped defaults when the file is absent. A file that is present but
    /// malformed is an error rather than a silent default.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("preprocessor_config.json");
        match std::fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str(&raw)
                .map_err(|e| format!("Failed to parse {}: {e}", path.display())),
            Err(_) => Ok(Self::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_config_reads_the_prefixed_keys() {
        let cfg: Plamo2VlVisionConfig = serde_json::from_value(serde_json::json!({
            "model_type": "",
            "image_feature_size": 1152,
            "image_proj_hidden_size": 2048,
            "image_token_id": 100002,
            "vision_encoder_hidden_act": "gelu_pytorch_tanh",
            "vision_encoder_hidden_size": 1152,
            "vision_encoder_image_size": 384,
            "vision_encoder_intermediate_size": 4304,
            "vision_encoder_layer_norm_eps": 1e-6,
            "vision_encoder_num_attention_heads": 16,
            "vision_encoder_num_channels": 3,
            "vision_encoder_num_hidden_layers": 27,
            "vision_encoder_patch_size": 14
        }))
        .unwrap();
        assert_eq!(cfg.image_proj_hidden_size, Some(2048));
        let siglip = cfg.to_siglip_config().unwrap();
        assert_eq!(siglip.num_hidden_layers, 27);
        assert_eq!(siglip.image_size / siglip.patch_size, 27);
        assert_eq!(siglip.hidden_act, VisionHiddenActivation::GeluPytorchTanh);
    }

    #[test]
    fn processor_config_defaults_match_the_shipped_file() {
        let parsed: Plamo2VlProcessorConfig = serde_json::from_value(serde_json::json!({
            "patch_size": 14, "tile_context_length": 3072, "downsample_ratio": 1.0,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 0.00392156862745098, "size": {"height": 384, "width": 384}
        }))
        .unwrap();
        let default = Plamo2VlProcessorConfig::default();
        assert_eq!(parsed.size.width, default.size.width);
        assert_eq!(parsed.tile_context_length, default.tile_context_length);
        assert!((parsed.rescale_factor - default.rescale_factor).abs() < 1e-9);
    }
}
