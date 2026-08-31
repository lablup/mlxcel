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

//! Inkling HMLP VLM loading boundary.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use crate::LoadedModel;
use crate::models::InklingModel;
use crate::models::inkling::InklingConfig;
use crate::vision::InklingVlModel;
use crate::vision::encoders::inkling_hmlp::{InklingHmlpEncoder, InklingVisionConfig};
use crate::vision::processors::inkling::{InklingImageProcessor, InklingImageProcessorConfig};

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

pub(crate) fn load_inkling_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_raw, full_config) = read_sanitized_vlm_config(model_path)?;
    let config = InklingConfig::from_json_with_sidecar(model_path, &config_raw)
        .map_err(anyhow::Error::msg)?;
    let vision_config: InklingVisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in Inkling config.json"))?,
    )
    .map_err(|error| anyhow::anyhow!("Failed to parse Inkling vision_config: {error}"))?;
    if vision_config.text_hidden_size != config.text_config.hidden_size {
        anyhow::bail!(
            "Inkling vision text width {} does not match decoder hidden_size {}",
            vision_config.text_hidden_size,
            config.text_config.hidden_size
        );
    }
    let processor_config = load_processor_config(model_path)?;
    let processor = InklingImageProcessor::new(processor_config).map_err(anyhow::Error::msg)?;
    let weights = normalize_inkling_weights(load_vlm_weights_common(model_path, None)?)?;
    let (group_size, bits, _) = config.quantization();
    let vision = InklingHmlpEncoder::from_weights(&weights, &vision_config, group_size, bits)
        .map_err(anyhow::Error::msg)?;
    let image_token_id = config.image_token_id;
    let text = InklingModel::from_weights(config, weights).map_err(anyhow::Error::msg)?;
    Ok(LoadedModel::InklingVLM(InklingVlModel::new(
        text,
        vision,
        processor,
        image_token_id,
    )))
}

fn load_processor_config(model_path: &Path) -> Result<InklingImageProcessorConfig> {
    let path = model_path.join("processor_config.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(InklingImageProcessorConfig::default());
        }
        Err(error) => {
            return Err(anyhow::anyhow!(
                "Failed to read {}: {error}",
                path.display()
            ));
        }
    };
    let root: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("Failed to parse {}: {error}", path.display()))?;
    let config = root.get("image_processor").unwrap_or(&root).clone();
    serde_json::from_value(config)
        .map_err(|error| anyhow::anyhow!("Failed to parse Inkling image_processor: {error}"))
}

fn normalize_inkling_weight_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.visual.layers.linear_")
        && let Some((index, suffix)) = rest.split_once('.')
    {
        return format!("vision_tower.encoder_layers.{index}.projection.{suffix}");
    }
    if let Some(rest) = key.strip_prefix("model.visual.layers.norm_")
        && let Some((index, suffix)) = rest.split_once('.')
    {
        return format!("vision_tower.encoder_layers.{index}.layer_norm.{suffix}");
    }
    if let Some(suffix) = key.strip_prefix("model.visual.final_norm.") {
        return format!("vision_tower.final_norm.{suffix}");
    }
    key.to_string()
}

fn normalize_inkling_weights(raw: WeightMap) -> Result<WeightMap> {
    let mut normalized = WeightMap::new();
    for (raw_key, value) in raw {
        let key = normalize_inkling_weight_key(&raw_key);
        if normalized.insert(key.clone(), value).is_some() {
            anyhow::bail!("Inkling checkpoint key {raw_key:?} collides at normalized key {key:?}");
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::normalize_inkling_weight_key;

    #[test]
    fn normalizes_all_hmlp_parameter_kinds() {
        assert_eq!(
            normalize_inkling_weight_key("model.visual.layers.linear_3.weight"),
            "vision_tower.encoder_layers.3.projection.weight"
        );
        assert_eq!(
            normalize_inkling_weight_key("model.visual.layers.linear_3.scales"),
            "vision_tower.encoder_layers.3.projection.scales"
        );
        assert_eq!(
            normalize_inkling_weight_key("model.visual.layers.linear_3.biases"),
            "vision_tower.encoder_layers.3.projection.biases"
        );
        assert_eq!(
            normalize_inkling_weight_key("model.visual.layers.norm_2.weight"),
            "vision_tower.encoder_layers.2.layer_norm.weight"
        );
        assert_eq!(
            normalize_inkling_weight_key("model.visual.final_norm.weight"),
            "vision_tower.final_norm.weight"
        );
    }
}
