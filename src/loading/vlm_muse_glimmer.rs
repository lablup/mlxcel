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

//! Muse Glimmer VLM loading boundary.
//!
//! Builds the concrete Muse model from the canonical dense checkpoint roots.
//! Request-time image expansion and ordered feature scatter are provided by
//! the dedicated Muse multimodal runtime.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models::muse_glimmer::{
    MuseGlimmerConfig, MuseGlimmerTextModel, MuseGlimmerTextWrapper,
};
use crate::vision::encoders::muse_glimmer::{
    MUSE_GLIMMER_VISION_TOWER_ROOT, MuseGlimmerVisionTower,
};
use crate::vision::encoders::muse_glimmer_fusion::{
    MUSE_GLIMMER_VISION_ADAPTER_ROOT, MUSE_GLIMMER_VISION_PROJECTION_ROOT, MuseGlimmerVisionFusion,
};
use crate::vision::muse_glimmer_vlm::MuseGlimmerVlmModel;
use crate::vision::processors::muse_glimmer::MuseGlimmerImageProcessor;

use super::{load_vlm_weights_common_filtered, read_sanitized_vlm_config};

pub(crate) const MUSE_GLIMMER_LANGUAGE_ROOT: &str = "model.language_model";
pub(crate) const MUSE_GLIMMER_LM_HEAD_ROOT: &str = "lm_head";
#[allow(dead_code)]
pub(crate) const MUSE_GLIMMER_DEFAULT_EOS: &[i32] = &[200_001, 200_008];
#[allow(dead_code)]
pub(crate) const MUSE_GLIMMER_IMAGE_TOKEN_ID: i32 = 200_092;
#[allow(dead_code)]
pub(crate) const MUSE_GLIMMER_VIDEO_TOKEN_ID: i32 = 200_091;
#[allow(dead_code)]
pub(crate) const MUSE_GLIMMER_PAD_TOKEN_ID: i32 = 200_018;

pub(crate) fn load_muse_glimmer_vlm(model_path: &Path) -> Result<LoadedModel> {
    Ok(LoadedModel::MuseGlimmerVLM(load_muse_glimmer_vlm_model(
        model_path,
    )?))
}

#[allow(dead_code)]
pub(crate) fn load_muse_glimmer_vlm_model(model_path: &Path) -> Result<MuseGlimmerVlmModel> {
    let config = parse_muse_glimmer_config(model_path)?;
    ensure_supported_muse_vlm_config(&config)?;
    if model_path.join("model.safetensors.index.json").exists() {
        let _ = read_muse_weight_inventory_from_index(model_path)?;
    }
    let processor = MuseGlimmerImageProcessor::from_model_dir(model_path, &config.vision_config)
        .map_err(anyhow::Error::msg)?;
    let weights = load_vlm_weights_common_filtered(model_path, None, keep_muse_glimmer_weight)?;
    ensure_dense_muse_weight_map(&weights)?;
    build_muse_glimmer_vlm_from_weights(
        &weights,
        &config,
        processor,
        crate::loading::read_eos_token_ids(model_path),
    )
}

pub(crate) fn parse_muse_glimmer_config(model_path: &Path) -> Result<MuseGlimmerConfig> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    serde_json::from_value(full_config)
        .map_err(|e| anyhow::anyhow!("Failed to parse Muse Glimmer config.json: {}", e))
}

pub(crate) fn build_muse_glimmer_vlm_from_weights(
    weights: &WeightMap,
    config: &MuseGlimmerConfig,
    processor: MuseGlimmerImageProcessor,
    eos_token_ids: Vec<i32>,
) -> Result<MuseGlimmerVlmModel> {
    ensure_supported_muse_vlm_config(config)?;
    ensure_dense_muse_weight_map(weights)?;
    let text_model = build_muse_glimmer_text_from_weights(weights, config, eos_token_ids)?;
    let text = MuseGlimmerTextWrapper::new(text_model);
    let vision_tower = MuseGlimmerVisionTower::from_weights(weights, &config.vision_config)
        .map_err(anyhow::Error::msg)?;
    let vision_fusion =
        MuseGlimmerVisionFusion::from_weights(weights, config).map_err(anyhow::Error::msg)?;
    MuseGlimmerVlmModel::new(text, vision_tower, vision_fusion, processor, config)
        .map_err(anyhow::Error::msg)
}

pub(crate) fn build_muse_glimmer_text_from_weights(
    weights: &WeightMap,
    config: &MuseGlimmerConfig,
    eos_token_ids: Vec<i32>,
) -> Result<MuseGlimmerTextModel> {
    let eos = if eos_token_ids.is_empty() {
        MUSE_GLIMMER_DEFAULT_EOS.to_vec()
    } else {
        eos_token_ids
    };
    let suppressed = vec![
        config.image_token_id.unwrap_or(MUSE_GLIMMER_IMAGE_TOKEN_ID),
        config.video_token_id.unwrap_or(MUSE_GLIMMER_VIDEO_TOKEN_ID),
        MUSE_GLIMMER_PAD_TOKEN_ID,
    ];
    MuseGlimmerTextModel::from_weights(
        weights,
        &config.text_config,
        MUSE_GLIMMER_LANGUAGE_ROOT,
        MUSE_GLIMMER_LM_HEAD_ROOT,
        eos,
        suppressed,
    )
    .map_err(anyhow::Error::msg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MuseWeightRoot {
    LanguageModel,
    VisionTower,
    VisionAdapter,
    VisionProjection,
    LmHead,
}

impl MuseWeightRoot {
    fn prefix(self) -> &'static str {
        match self {
            Self::LanguageModel => MUSE_GLIMMER_LANGUAGE_ROOT,
            Self::VisionTower => MUSE_GLIMMER_VISION_TOWER_ROOT,
            Self::VisionAdapter => MUSE_GLIMMER_VISION_ADAPTER_ROOT,
            Self::VisionProjection => MUSE_GLIMMER_VISION_PROJECTION_ROOT,
            Self::LmHead => MUSE_GLIMMER_LM_HEAD_ROOT,
        }
    }

    fn all() -> [Self; 5] {
        [
            Self::LanguageModel,
            Self::VisionTower,
            Self::VisionAdapter,
            Self::VisionProjection,
            Self::LmHead,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MuseWeightInventory {
    pub language_model: usize,
    pub vision_tower: usize,
    pub vision_adapter: usize,
    pub vision_projection: usize,
    pub lm_head: usize,
    pub total: usize,
}

impl MuseWeightInventory {
    fn increment(&mut self, root: MuseWeightRoot) {
        match root {
            MuseWeightRoot::LanguageModel => self.language_model += 1,
            MuseWeightRoot::VisionTower => self.vision_tower += 1,
            MuseWeightRoot::VisionAdapter => self.vision_adapter += 1,
            MuseWeightRoot::VisionProjection => self.vision_projection += 1,
            MuseWeightRoot::LmHead => self.lm_head += 1,
        }
        self.total += 1;
    }

    fn ensure_complete(self) -> Result<()> {
        for (count, label) in [
            (self.language_model, MUSE_GLIMMER_LANGUAGE_ROOT),
            (self.vision_tower, MUSE_GLIMMER_VISION_TOWER_ROOT),
            (self.vision_adapter, MUSE_GLIMMER_VISION_ADAPTER_ROOT),
            (self.vision_projection, MUSE_GLIMMER_VISION_PROJECTION_ROOT),
            (self.lm_head, MUSE_GLIMMER_LM_HEAD_ROOT),
        ] {
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "Muse Glimmer checkpoint is missing {label} weights"
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn classify_muse_weight_key(key: &str) -> Option<MuseWeightRoot> {
    MuseWeightRoot::all()
        .into_iter()
        .find(|root| key.starts_with(&format!("{}.", root.prefix())))
}

pub(crate) fn keep_muse_glimmer_weight(key: &str) -> bool {
    classify_muse_weight_key(key).is_some()
}

pub(crate) fn read_muse_weight_inventory_from_index(
    model_path: &Path,
) -> Result<MuseWeightInventory> {
    let index_path = model_path.join("model.safetensors.index.json");
    let raw = std::fs::read_to_string(&index_path)
        .map_err(|err| anyhow::anyhow!("Failed to read {}: {err}", index_path.display()))?;
    let index: Value = serde_json::from_str(&raw)
        .map_err(|err| anyhow::anyhow!("Failed to parse {}: {err}", index_path.display()))?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("Muse Glimmer safetensors index has no weight_map"))?;

    let mut inventory = MuseWeightInventory {
        language_model: 0,
        vision_tower: 0,
        vision_adapter: 0,
        vision_projection: 0,
        lm_head: 0,
        total: 0,
    };
    for key in weight_map.keys() {
        ensure_dense_muse_key(key)?;
        let root = classify_muse_weight_key(key)
            .ok_or_else(|| anyhow::anyhow!("Unknown Muse Glimmer checkpoint weight root: {key}"))?;
        inventory.increment(root);
    }
    inventory.ensure_complete()?;
    Ok(inventory)
}

fn ensure_supported_muse_vlm_config(config: &MuseGlimmerConfig) -> Result<()> {
    config.validate().map_err(anyhow::Error::msg)?;
    if config.text_config.quantization.is_some() {
        return Err(anyhow::anyhow!(
            "Muse Glimmer VLM supports only the canonical dense BF16 checkpoint; quantized text_config is not supported"
        ));
    }
    if config.vision_config.patch_temporal != 2 {
        return Err(anyhow::anyhow!(
            "Muse Glimmer VLM supports static image duplication only; video temporal layouts are not supported"
        ));
    }
    if config.vision_config.merge_size != 2 {
        return Err(anyhow::anyhow!(
            "Muse Glimmer VLM supports only the published 2x2 visual token merge"
        ));
    }
    Ok(())
}

fn ensure_dense_muse_weight_map(weights: &WeightMap) -> Result<()> {
    for key in weights.keys() {
        ensure_dense_muse_key(key)?;
    }
    Ok(())
}

fn ensure_dense_muse_key(key: &str) -> Result<()> {
    if [".scales", ".biases", ".global_scale"]
        .iter()
        .any(|suffix| key.ends_with(suffix))
    {
        return Err(anyhow::anyhow!(
            "Muse Glimmer VLM supports only canonical dense BF16 weights; found quantization sidecar {key}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "vlm_muse_glimmer_tests.rs"]
mod tests;
