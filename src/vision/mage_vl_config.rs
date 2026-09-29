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

//! Mage-VL (`mage_vl`) configuration: the Mage-ViT `vision_config`, the
//! `text_config` RoPE lift, and the checkpoint's special-token ids.
//!
//! Used by: `loading::load_mage_vl`, `vision::encoders::mage_vl`.

use serde::Deserialize;
use serde_json::Value;

/// `<|image_pad|>`: one merged Mage-ViT feature row lands on each.
pub const MAGE_VL_IMAGE_TOKEN_ID: i32 = 151655;
/// `<|video_pad|>`: the codec-native video path is out of scope, but features
/// would land on these positions too, so the scatter accepts both.
pub const MAGE_VL_VIDEO_TOKEN_ID: i32 = 151656;
/// `<|vision_start|>`.
pub const MAGE_VL_VISION_START_TOKEN_ID: i32 = 151652;
/// `<|vision_end|>`.
pub const MAGE_VL_VISION_END_TOKEN_ID: i32 = 151653;
/// `<|im_end|>` and `<|endoftext|>`.
pub const MAGE_VL_DEFAULT_EOS_TOKEN_IDS: [i32; 2] = [151645, 151643];

/// Qwen2-VL `smart_resize` bounds shipped in `preprocessor_config.json`.
pub const MAGE_VL_DEFAULT_MIN_PIXELS: usize = 3136;
pub const MAGE_VL_DEFAULT_MAX_PIXELS: usize = 4_000_000;
/// CLIP normalization shipped in `preprocessor_config.json`.
pub const MAGE_VL_IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const MAGE_VL_IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Mage-ViT `vision_config`. Defaults are the released `microsoft/Mage-VL`
/// values, so a converted checkpoint that drops a key still loads the same
/// tower.
#[derive(Debug, Clone, Deserialize)]
pub struct MageVlVisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_layer_norm_type")]
    pub layer_norm_type: String,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_frame_windows_size")]
    pub frame_windows_size: usize,
    #[serde(default)]
    pub use_head: bool,
    #[serde(default)]
    pub use_patch_position_encoding: bool,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_patch_size() -> usize {
    16
}
fn default_num_channels() -> usize {
    3
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_layer_norm_type() -> String {
    "layer_norm".to_string()
}
fn default_hidden_act() -> String {
    "gelu".to_string()
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    1
}
fn default_out_hidden_size() -> usize {
    2560
}
fn default_frame_windows_size() -> usize {
    4
}

impl Default for MageVlVisionConfig {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({}))
            .expect("every MageVlVisionConfig field has a serde default")
    }
}

impl MageVlVisionConfig {
    /// Per-head width (64 for the released tower).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    /// Refuse the switches this port does not implement, naming each one,
    /// instead of silently running a different graph.
    ///
    /// * `use_patch_position_encoding: true` adds absolute `pos_emb_h/w` tables
    ///   in the merger.
    /// * `use_head: true` adds a post-layernorm and an attention-pooling head.
    /// * `layer_norm_type` other than `layer_norm` swaps every norm for RMSNorm.
    /// * `hidden_act` other than exact `gelu`.
    /// * A head width that does not split 4:6:6 over `head_dim / 2`.
    pub fn validate(&self) -> Result<(), String> {
        if self.use_patch_position_encoding {
            return Err(
                "Mage-VL vision_config.use_patch_position_encoding = true is not supported \
                 (absolute merger position tables are not implemented)"
                    .to_string(),
            );
        }
        if self.use_head {
            return Err(
                "Mage-VL vision_config.use_head = true is not supported (the released tower \
                 has no post-layernorm or pooling head)"
                    .to_string(),
            );
        }
        if self.layer_norm_type != "layer_norm" {
            return Err(format!(
                "Mage-VL vision_config.layer_norm_type = '{}' is not supported (expected \
                 'layer_norm')",
                self.layer_norm_type
            ));
        }
        if self.hidden_act != "gelu" {
            return Err(format!(
                "Mage-VL vision_config.hidden_act = '{}' is not supported (expected exact 'gelu')",
                self.hidden_act
            ));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !(self.head_dim() / 2).is_multiple_of(16)
            || !self.head_dim().is_multiple_of(2)
        {
            return Err(format!(
                "Mage-VL vision tower needs head_dim/2 divisible by 16 for the 4:6:6 rotary \
                 split (hidden_size={}, num_attention_heads={})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.temporal_patch_size != 1 {
            return Err(format!(
                "Mage-VL vision_config.temporal_patch_size = {} is not supported (expected 1)",
                self.temporal_patch_size
            ));
        }
        if self.spatial_merge_size == 0 {
            return Err("Mage-VL vision_config.spatial_merge_size must be positive".to_string());
        }
        Ok(())
    }
}

/// Copy a nested `rope_parameters.rope_theta` into the flat `rope_theta` key
/// that `models::qwen3::ModelArgs` reads, when the flat key is absent.
///
/// The released config stores the decoder base only as
/// `text_config.rope_parameters: {"rope_theta": 5000000, "rope_type":
/// "default"}`. Parsed as-is, `ModelArgs.rope_theta` would fall back to its
/// serde default of 10000 and the decoder would run fluently at short lengths
/// and drift at long ones. A non-default `rope_type` is carried into
/// `rope_scaling` (with the `type` key the Qwen3 resolver reads) unless the
/// config already has a non-null `rope_scaling`.
pub fn lift_rope_parameters(text_config: &mut Value) {
    let Some(obj) = text_config.as_object_mut() else {
        return;
    };
    let Some(params) = obj
        .get("rope_parameters")
        .and_then(Value::as_object)
        .cloned()
    else {
        return;
    };
    if obj.get("rope_theta").is_none_or(Value::is_null)
        && let Some(theta) = params.get("rope_theta").filter(|v| v.is_number())
    {
        obj.insert("rope_theta".to_string(), theta.clone());
    }
    let rope_type = params
        .get("rope_type")
        .or_else(|| params.get("type"))
        .and_then(Value::as_str);
    let has_scaling = obj.get("rope_scaling").is_some_and(|v| !v.is_null());
    if let Some(kind) = rope_type
        && kind != "default"
        && !has_scaling
    {
        let mut scaling = params.clone();
        scaling.remove("rope_theta");
        scaling.insert("type".to_string(), Value::String(kind.to_string()));
        obj.insert("rope_scaling".to_string(), Value::Object(scaling));
    }
}

/// Special-token ids read from the top of `config.json`, with the released
/// values as fallbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MageVlTokenIds {
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
}

impl MageVlTokenIds {
    pub fn from_config(config: &Value) -> Result<Self, String> {
        let read = |key: &str, default: i32| -> Result<i32, String> {
            match config.get(key) {
                None | Some(Value::Null) => Ok(default),
                Some(value) => value
                    .as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .filter(|v| *v >= 0)
                    .ok_or_else(|| {
                        format!("Mage-VL config.json `{key}` must be a non-negative 32-bit integer")
                    }),
            }
        };
        Ok(Self {
            image_token_id: read("image_token_id", MAGE_VL_IMAGE_TOKEN_ID)?,
            video_token_id: read("video_token_id", MAGE_VL_VIDEO_TOKEN_ID)?,
            vision_start_token_id: read("vision_start_token_id", MAGE_VL_VISION_START_TOKEN_ID)?,
            vision_end_token_id: read("vision_end_token_id", MAGE_VL_VISION_END_TOKEN_ID)?,
        })
    }
}

/// Stop ids: `generation_config.json` then `config.json` `eos_token_id`
/// (number or list), plus `<|im_end|>` and `<|endoftext|>`.
#[must_use]
pub fn resolve_eos_token_ids(generation_config: Option<&Value>, config: &Value) -> Vec<i32> {
    let mut ids: Vec<i32> = Vec::new();
    let mut push = |id: i64| {
        if let Ok(id) = i32::try_from(id)
            && id >= 0
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    };
    for source in [generation_config, Some(config)].into_iter().flatten() {
        match source.get("eos_token_id") {
            Some(Value::Number(n)) => {
                if let Some(id) = n.as_i64() {
                    push(id);
                }
            }
            Some(Value::Array(items)) => items.iter().filter_map(Value::as_i64).for_each(&mut push),
            _ => {}
        }
    }
    for id in MAGE_VL_DEFAULT_EOS_TOKEN_IDS {
        push(i64::from(id));
    }
    ids
}

/// `(min_pixels, max_pixels)` from `preprocessor_config.json`, falling back
/// to the released 3136 / 4000000. Accepts both the flat keys the checkpoint
/// ships and the `size.{shortest_edge,longest_edge}` spelling newer
/// transformers processors write.
#[must_use]
pub fn resolve_pixel_bounds(preprocessor_config: Option<&Value>) -> (usize, usize) {
    let get = |flat: &str, nested: &str| -> Option<usize> {
        let config = preprocessor_config?;
        config
            .get(flat)
            .and_then(Value::as_u64)
            .or_else(|| config.get("size")?.get(nested)?.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
    };
    let min = get("min_pixels", "shortest_edge").unwrap_or(MAGE_VL_DEFAULT_MIN_PIXELS);
    let max = get("max_pixels", "longest_edge").unwrap_or(MAGE_VL_DEFAULT_MAX_PIXELS);
    if min > max {
        (MAGE_VL_DEFAULT_MIN_PIXELS, MAGE_VL_DEFAULT_MAX_PIXELS)
    } else {
        (min, max)
    }
}

/// `(image_mean, image_std)` from `preprocessor_config.json`, falling back to
/// the CLIP statistics the checkpoint ships.
#[must_use]
pub fn resolve_normalization(preprocessor_config: Option<&Value>) -> ([f32; 3], [f32; 3]) {
    let read = |key: &str, default: [f32; 3]| -> [f32; 3] {
        preprocessor_config
            .and_then(|c| c.get(key))
            .and_then(Value::as_array)
            .and_then(|values| {
                let parsed: Vec<f32> = values
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|v| v as f32)
                    .collect();
                <[f32; 3]>::try_from(parsed.as_slice()).ok()
            })
            .unwrap_or(default)
    };
    (
        read("image_mean", MAGE_VL_IMAGE_MEAN),
        read("image_std", MAGE_VL_IMAGE_STD),
    )
}

#[cfg(test)]
#[path = "mage_vl_config_tests.rs"]
mod tests;
