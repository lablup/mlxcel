// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use serde::Deserialize;
use serde_json::Value;

pub const DEFAULT_IMAGE_TOKEN_ID: i32 = 200_092;
pub const DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID: i32 = 200_090;
pub const DEFAULT_IMAGE_START_TOKEN_ID: i32 = 200_080;
pub const DEFAULT_IMAGE_END_TOKEN_ID: i32 = 200_081;
pub const DEFAULT_VIDEO_TOKEN_ID: i32 = 200_091;
pub const DEFAULT_PAD_TOKEN_ID: i32 = 200_018;

#[derive(Debug, Clone, Deserialize)]
pub struct MuseGlimmerConfig {
    pub text_config: MuseGlimmerTextConfig,
    #[serde(default)]
    pub vision_config: MuseGlimmerVisionConfig,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub video_token_id: Option<i32>,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MuseGlimmerTextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_post_norm_eps")]
    pub post_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_qk_scale_factor")]
    pub qk_scale_factor: f32,
    #[serde(default = "default_output_multiplier")]
    pub output_multiplier: f32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub layer_rope_theta: Vec<Option<f32>>,
    #[serde(default)]
    pub rope_parameters: Option<MuseRopeParameters>,
    #[serde(default)]
    pub quantization: Option<MuseQuantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MuseRopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MuseQuantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MuseGlimmerVisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_patch_temporal")]
    pub patch_temporal: usize,
    #[serde(default = "default_vision_merge_size")]
    pub merge_size: usize,
    #[serde(default = "default_vision_pos_emb_height")]
    pub pos_emb_height: usize,
    #[serde(default = "default_vision_pos_emb_width")]
    pub pos_emb_width: usize,
    #[serde(default = "default_vision_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<MuseVisionRopeParameters>,
}

impl Default for MuseGlimmerVisionConfig {
    fn default() -> Self {
        Self {
            model_type: default_vision_model_type(),
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_hidden_layers: default_vision_num_hidden_layers(),
            num_attention_heads: default_vision_num_attention_heads(),
            patch_size: default_vision_patch_size(),
            patch_temporal: default_vision_patch_temporal(),
            merge_size: default_vision_merge_size(),
            pos_emb_height: default_vision_pos_emb_height(),
            pos_emb_width: default_vision_pos_emb_width(),
            max_position_embeddings: default_vision_max_position_embeddings(),
            layer_norm_eps: default_vision_layer_norm_eps(),
            hidden_act: default_vision_hidden_act(),
            layer_types: Vec::new(),
            rope_parameters: Some(MuseVisionRopeParameters {
                rope_theta: default_vision_rope_theta(),
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MuseVisionRopeParameters {
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f32,
}

fn default_post_norm_eps() -> f32 {
    1e-8
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_sliding_window() -> usize {
    2048
}

fn default_qk_scale_factor() -> f32 {
    3.87
}

fn default_output_multiplier() -> f32 {
    0.196_116_13
}

fn default_rope_theta() -> f32 {
    500_000.0
}

fn default_out_hidden_size() -> usize {
    6144
}

fn default_projector_hidden_size() -> usize {
    4096
}

fn default_projector_hidden_act() -> String {
    "gelu".to_string()
}

fn default_vision_model_type() -> String {
    "muse_glimmer_vision".to_string()
}

fn default_vision_hidden_size() -> usize {
    1536
}

fn default_vision_intermediate_size() -> usize {
    8960
}

fn default_vision_num_hidden_layers() -> usize {
    50
}

fn default_vision_num_attention_heads() -> usize {
    16
}

fn default_vision_patch_size() -> usize {
    14
}

fn default_vision_patch_temporal() -> usize {
    2
}

fn default_vision_merge_size() -> usize {
    2
}

fn default_vision_pos_emb_height() -> usize {
    32
}

fn default_vision_pos_emb_width() -> usize {
    32
}

fn default_vision_max_position_embeddings() -> usize {
    1024
}

fn default_vision_layer_norm_eps() -> f32 {
    1e-5
}

fn default_vision_hidden_act() -> String {
    "gelu".to_string()
}

fn default_vision_rope_theta() -> f32 {
    10_000.0
}

impl MuseGlimmerTextConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(quantization) = &self.quantization {
            mlxcel_core::layers::validate_quantization_params(
                quantization.group_size,
                quantization.bits,
            )
            .map_err(|err| format!("Muse Glimmer {err}"))?;
        }
        if self.num_attention_heads == 0 {
            return Err("Muse Glimmer num_attention_heads must be non-zero".to_string());
        }
        if self.num_key_value_heads == 0 {
            return Err("Muse Glimmer num_key_value_heads must be non-zero".to_string());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "Muse Glimmer attention heads ({}) must be divisible by KV heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "Muse Glimmer layer_types length ({}) must equal num_hidden_layers ({})",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (idx, layer_type) in self.layer_types.iter().enumerate() {
            if layer_type != "sliding_attention" && layer_type != "full_attention" {
                return Err(format!(
                    "Muse Glimmer layer {idx} has unsupported layer_type {layer_type:?}"
                ));
            }
        }
        Ok(())
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|kind| kind == "sliding_attention")
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> Option<f32> {
        if !self.is_sliding_layer(layer_idx) {
            return None;
        }
        self.layer_rope_theta
            .get(layer_idx)
            .and_then(|v| *v)
            .or_else(|| self.rope_parameters.as_ref().map(|p| p.rope_theta))
            .or(Some(default_rope_theta()))
    }
}

/// Carry the MLX checkpoint's root-level quantization contract into the text
/// sub-config consumed by the decoder.
///
/// `mlx-community/Muse-Glimmer-30B-4bit` follows mlx-vlm's common layout: the
/// affine `{group_size, bits, mode}` block lives at the config root while the
/// packed tensors live below `language_model.*`. The Rust decoder receives
/// only `text_config`, so leaving the block at the root would make every
/// `UnifiedLinear` interpret the packed weights with fallback parameters.
pub(crate) fn inherit_muse_text_quantization(config: &mut Value) -> Result<(), String> {
    let root_quantization = config.get("quantization").filter(|value| !value.is_null());
    let compatibility_quantization = config
        .get("quantization_config")
        .filter(|value| !value.is_null());

    if let (Some(root), Some(compatibility)) = (root_quantization, compatibility_quantization)
        && root != compatibility
    {
        return Err("Muse Glimmer root quantization and quantization_config disagree".to_string());
    }

    let Some(quantization) = root_quantization.or(compatibility_quantization).cloned() else {
        return Ok(());
    };
    let text_config = config
        .get_mut("text_config")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "Muse Glimmer config.json has no text_config object".to_string())?;

    if let Some(nested) = text_config
        .get("quantization")
        .filter(|value| !value.is_null())
    {
        if nested != &quantization {
            return Err(
                "Muse Glimmer root and text_config quantization contracts disagree".to_string(),
            );
        }
    } else {
        text_config.insert("quantization".to_string(), quantization);
    }

    if let Some(mode) = text_config
        .get("quantization")
        .and_then(|quantization| quantization.get("mode"))
    {
        let mode = mode.as_str().ok_or_else(|| {
            "Muse Glimmer quantization.mode must be a string when present".to_string()
        })?;
        if mode != "affine" {
            return Err(format!(
                "Muse Glimmer quantization.mode must be \"affine\" for the pinned mlx-community checkpoint layout, got {mode:?}"
            ));
        }
    }
    Ok(())
}

impl MuseGlimmerConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.text_config.validate()?;
        self.vision_config.validate()?;
        if self.out_hidden_size == 0 || self.projector_hidden_size == 0 {
            return Err("Muse Glimmer projector sizes must be non-zero".to_string());
        }
        if self.projector_hidden_act != "gelu" {
            return Err(format!(
                "Muse Glimmer projector_hidden_act must be \"gelu\", got {:?}",
                self.projector_hidden_act
            ));
        }
        Ok(())
    }
}

impl MuseGlimmerVisionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads == 0 {
            return Err("Muse Glimmer vision num_attention_heads must be non-zero".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "Muse Glimmer vision hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.patch_size == 0 || self.patch_temporal == 0 || self.merge_size == 0 {
            return Err("Muse Glimmer vision patch and merge sizes must be non-zero".to_string());
        }
        if self.pos_emb_height == 0 || self.pos_emb_width == 0 {
            return Err("Muse Glimmer vision position grid must be non-zero".to_string());
        }
        let layer_types = self.layer_types();
        if layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "Muse Glimmer vision layer_types length ({}) must equal num_hidden_layers ({})",
                layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (idx, layer_type) in layer_types.iter().enumerate() {
            let expected = Self::expected_layer_type(idx, self.num_hidden_layers);
            if layer_type != expected {
                return Err(format!(
                    "Muse Glimmer vision layer {idx} must be {expected:?}, got {layer_type:?}"
                ));
            }
        }
        Ok(())
    }

    pub fn layer_types(&self) -> Vec<String> {
        if !self.layer_types.is_empty() {
            return self.layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|idx| Self::expected_layer_type(idx, self.num_hidden_layers).to_string())
            .collect()
    }

    pub fn expected_layer_type(layer_idx: usize, num_layers: usize) -> &'static str {
        if (layer_idx + 1).is_multiple_of(4) || layer_idx + 1 == num_layers {
            "full_attention"
        } else {
            "window_attention"
        }
    }

    pub fn is_window_layer(&self, layer_idx: usize) -> bool {
        self.layer_types()
            .get(layer_idx)
            .is_some_and(|kind| kind == "window_attention")
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.rope_theta)
            .unwrap_or_else(default_vision_rope_theta)
    }
}
