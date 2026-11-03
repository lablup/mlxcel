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

//! Xiaomi MiMo-V2 / V2.5 / V2.6 (`mimo_v2`) loading boundary.
//!
//! The text hyperparameters sit at the top level of `config.json` rather than
//! under `text_config` (upstream `ModelConfig.from_dict` copies the fields
//! `TextConfig` declares into a synthetic `text_config`). This loader does the
//! same: a present `text_config` object wins, otherwise the flat root is the
//! text config.
//!
//! Audio is out of scope for issue #2409. The released checkpoints ship a
//! 20-channel codec tokenizer (`audio_tokenizer/`, imported from mlx-audio)
//! and a small `AudioEncoder`, and mlxcel has a port of neither. The loader
//! drops `audio_encoder.*` / `speech_embeddings.*` and refuses audio input with
//! a clear message instead of loading a half-model.

use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::LoadedModel;
use crate::models::MiMoV2FlashModel;
use crate::models::mimo_v2 as text_weights;
use crate::models::mimo_v2_flash::ModelArgs;
use crate::vision::MimoV2VlModel;
use crate::vision::encoders::mimovl::{MimovlVisionConfig, MimovlVisionEncoder};

use super::{
    QwenVisionConfigExt, load_vlm_weights_common_filtered, qwen_vl_processor,
    read_sanitized_vlm_config,
};

/// Token ids used when `config.json` omits a field. They are the published
/// MiMo-V2.6 values and also the Qwen2.5-VL processor's placeholders.
pub const DEFAULT_IMAGE_TOKEN_ID: i32 = 151_655;
pub const DEFAULT_VIDEO_TOKEN_ID: i32 = 151_656;
pub const DEFAULT_AUDIO_TOKEN_ID: i32 = 151_669;
pub const DEFAULT_VISION_START_TOKEN_ID: i32 = 151_652;
pub const DEFAULT_VISION_END_TOKEN_ID: i32 = 151_653;
pub const DEFAULT_AUDIO_START_TOKEN_ID: i32 = 151_673;
pub const DEFAULT_AUDIO_END_TOKEN_ID: i32 = 151_674;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoV2TokenIds {
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub audio_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
}

fn d_speech_vocab_size() -> usize {
    1280
}
fn d_speech_zeroemb_idx() -> usize {
    1024
}

fn int_or_string<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Value {
        Int(u64),
        Text(String),
    }
    let value = Value::deserialize(deserializer)?;
    let parsed = match value {
        Value::Int(value) => usize::try_from(value)
            .map_err(|_| serde::de::Error::custom("integer does not fit in usize"))?,
        Value::Text(text) => text
            .trim()
            .parse::<usize>()
            .map_err(|_| serde::de::Error::custom(format!("{text:?} is not an integer")))?,
    };
    Ok(parsed)
}

/// The parts of `audio_config` this issue touches.
///
/// The released V2.6 config ships `speech_vocab_size` and `speech_zeroemb_idx`
/// as JSON strings; upstream casts them in `AudioConfig.__post_init__`. They
/// are parsed here so a future audio port does not inherit a stringly typed
/// config, even though audio is out of scope today.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct MimoV2AudioConfig {
    #[serde(default = "d_speech_vocab_size", deserialize_with = "int_or_string")]
    pub speech_vocab_size: usize,
    #[serde(default = "d_speech_zeroemb_idx", deserialize_with = "int_or_string")]
    pub speech_zeroemb_idx: usize,
}

impl Default for MimoV2AudioConfig {
    fn default() -> Self {
        Self {
            speech_vocab_size: d_speech_vocab_size(),
            speech_zeroemb_idx: d_speech_zeroemb_idx(),
        }
    }
}

/// Parsed MiMo-V2 config.
#[derive(Debug, Clone)]
pub struct MimoV2Config {
    pub text: ModelArgs,
    pub vision: Option<MimovlVisionConfig>,
    /// Parsed for the flat-config contract; audio is not wired to a tower yet.
    #[cfg_attr(not(test), allow(dead_code))]
    pub audio: Option<MimoV2AudioConfig>,
    pub tokens: MimoV2TokenIds,
    pub skip_vision: bool,
}

fn token_id(
    root: &serde_json::Value,
    field: &str,
    default: i32,
    vocab_size: usize,
) -> Result<i32, String> {
    let raw = match root.get(field) {
        None | Some(serde_json::Value::Null) => return Ok(default),
        Some(value) => value
            .as_i64()
            .ok_or_else(|| format!("MiMo-V2 config.json `{field}` is not an integer"))?,
    };
    let id = i32::try_from(raw).map_err(|_| {
        format!("MiMo-V2 config.json `{field}={raw}` does not fit in a 32-bit token id")
    })?;
    if id < 0 || (id as usize) >= vocab_size {
        return Err(format!(
            "MiMo-V2 config.json `{field}={id}` is outside the checkpoint's vocabulary \
             (vocab_size={vocab_size}); an out-of-range vision token id silently misplaces every \
             image feature"
        ));
    }
    Ok(id)
}

impl MimoV2Config {
    /// Build the config from an already-parsed `config.json` object.
    pub fn from_value(root: &serde_json::Value) -> Result<Self, String> {
        let text_value = match root.get("text_config") {
            Some(serde_json::Value::Object(_)) => root
                .get("text_config")
                .cloned()
                .expect("text_config object just matched"),
            _ => root.clone(),
        };
        let text: ModelArgs = serde_json::from_value(text_value)
            .map_err(|error| format!("Failed to parse MiMo-V2 text config: {error}"))?;
        text.validate()?;

        let vision = match root.get("vision_config") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(
                serde_json::from_value::<MimovlVisionConfig>(value.clone())
                    .map_err(|error| format!("Failed to parse MiMo-V2 vision_config: {error}"))?,
            ),
        };
        if let Some(vision) = &vision {
            vision.validate()?;
        }

        let audio = match root.get("audio_config") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(
                serde_json::from_value::<MimoV2AudioConfig>(value.clone())
                    .map_err(|error| format!("Failed to parse MiMo-V2 audio_config: {error}"))?,
            ),
        };

        let vocab_size = text.vocab_size;
        let tokens = MimoV2TokenIds {
            image_token_id: token_id(root, "image_token_id", DEFAULT_IMAGE_TOKEN_ID, vocab_size)?,
            video_token_id: token_id(root, "video_token_id", DEFAULT_VIDEO_TOKEN_ID, vocab_size)?,
            audio_token_id: token_id(root, "audio_token_id", DEFAULT_AUDIO_TOKEN_ID, vocab_size)?,
            vision_start_token_id: token_id(
                root,
                "vision_start_token_id",
                DEFAULT_VISION_START_TOKEN_ID,
                vocab_size,
            )?,
            vision_end_token_id: token_id(
                root,
                "vision_end_token_id",
                DEFAULT_VISION_END_TOKEN_ID,
                vocab_size,
            )?,
            audio_start_token_id: token_id(
                root,
                "audio_start_token_id",
                DEFAULT_AUDIO_START_TOKEN_ID,
                vocab_size,
            )?,
            audio_end_token_id: token_id(
                root,
                "audio_end_token_id",
                DEFAULT_AUDIO_END_TOKEN_ID,
                vocab_size,
            )?,
        };

        Ok(Self {
            text,
            vision,
            audio,
            tokens,
            skip_vision: root
                .get("skip_vision")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
    }

    /// Parse a raw `config.json` string.
    pub fn from_json_str(raw: &str) -> Result<Self, String> {
        let raw = crate::models::sanitize_config_json(raw);
        let root: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| format!("Failed to parse config.json: {error}"))?;
        Self::from_value(&root)
    }
}

impl QwenVisionConfigExt for MimovlVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

pub(crate) fn load_mimo_v2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let config = MimoV2Config::from_json_str(&config_str).map_err(anyhow::Error::msg)?;

    // The released checkpoint is 173.8 GB and audio is out of scope, so the
    // audio stack is never materialized. A text-only load skips the tower too
    // (`skip_vision`), rather than loading and dropping it afterwards.
    let mut weights = load_vlm_weights_common_filtered(model_path, None, |key| {
        !key.starts_with("audio_encoder.")
            && !key.starts_with("speech_embeddings.")
            && (!config.skip_vision
                || (!key.starts_with("vision_tower.") && !key.starts_with("visual.")))
    })?;
    let vision_weights = text_weights::take_vision_weights(&mut weights);
    let text_weights =
        text_weights::sanitize_text_weights(weights, &config.text).map_err(anyhow::Error::msg)?;
    let text_model = MiMoV2FlashModel::from_weights(&text_weights, &config.text)
        .map_err(|error| anyhow::anyhow!("Failed to load MiMo-V2 text model: {error}"))?;

    let has_vision = !config.skip_vision
        && config.vision.is_some()
        && !vision_weights.is_empty()
        && crate::models::vlm_has_vision(&full_config, model_path);
    let (vision_encoder, processor, spatial_merge_size) = if has_vision {
        let mut vision_config = config
            .vision
            .clone()
            .ok_or_else(|| anyhow::anyhow!("MiMo-V2 config.json has no vision_config"))?;
        vision_config.quant_group_size = config.text.group_size();
        vision_config.quant_bits = config.text.bits();
        let encoder =
            MimovlVisionEncoder::from_weights(&vision_weights, &vision_config, "vision_tower")
                .map_err(|error| anyhow::anyhow!("Failed to load MiMo-V2 vision tower: {error}"))?;
        let processor = qwen_vl_processor(model_path, &vision_config)?;
        (Some(encoder), processor, vision_config.spatial_merge_size)
    } else {
        // A text-only load still needs a processor for the QwenVlRuntime
        // prompt info; the merge size comes from the config when present.
        let patch_size = config
            .vision
            .as_ref()
            .map_or(16, |vision| vision.patch_size);
        let temporal = config
            .vision
            .as_ref()
            .map_or(2, |vision| vision.temporal_patch_size);
        let merge = config
            .vision
            .as_ref()
            .map_or(2, |vision| vision.spatial_merge_size);
        let processor = processor_defaults(model_path, patch_size, temporal, merge)?;
        (None, processor, merge)
    };

    let tokens = config.tokens;
    Ok(LoadedModel::MiMoV2VLM(MimoV2VlModel {
        text_model,
        vision_encoder,
        text_only_path: (!has_vision).then(|| model_path.display().to_string()),
        processor,
        image_token_id: tokens.image_token_id,
        video_token_id: tokens.video_token_id,
        vision_start_token_id: tokens.vision_start_token_id,
        vision_end_token_id: tokens.vision_end_token_id,
        audio_token_id: tokens.audio_token_id,
        spatial_merge_size,
    }))
}

fn processor_defaults(
    model_path: &Path,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
) -> Result<crate::vision::processors::qwen2_vl::Qwen2VLProcessor> {
    let video_config = super::qwen_video_processor_config(model_path)?;
    Ok(crate::vision::processors::qwen2_vl::Qwen2VLProcessor::new(
        patch_size,
        temporal_patch_size,
        spatial_merge_size,
    )
    .with_video_config(video_config))
}

#[cfg(test)]
#[path = "vlm_mimo_v2_forward_tests.rs"]
mod forward_tests;
#[cfg(test)]
#[path = "vlm_mimo_v2_tests.rs"]
mod tests;
