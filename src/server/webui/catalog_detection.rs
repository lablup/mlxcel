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

use serde_json::Value;

use crate::models::{
    ModelDetectionProbes, ModelType, config_has_embedding_architecture,
    embedding_variant_for_model_type, is_encoder_only_model_type,
    is_sequence_classification_architecture, modules_json_value_has_pooling,
};

use super::catalog_fs::read_json_bounded;
use super::catalog_types::{MAX_CONFIG_BYTES, MAX_INDEX_BYTES};

pub(super) struct BoundedCatalogDetectionProbes;

impl ModelDetectionProbes for BoundedCatalogDetectionProbes {
    fn is_kokoro_checkpoint(&self, model_path: &Path, config: &Value) -> anyhow::Result<bool> {
        Ok(config.get("istftnet").is_some()
            || regular_file_exists(&model_path.join(crate::models::kokoro::KOKORO_WEIGHT_FILE)))
    }

    fn is_embedding_checkpoint(
        &self,
        model_path: &Path,
        config: &Value,
    ) -> anyhow::Result<Option<ModelType>> {
        bounded_embedding_checkpoint(model_path, config)
    }

    fn gemma4_has_vision_weights(
        &self,
        model_path: &Path,
        _config: &Value,
    ) -> anyhow::Result<bool> {
        let prefixes = [
            "vision_tower.",
            "embed_vision.",
            "model.vision_tower.",
            "model.embed_vision.",
        ];
        if regular_file_exists(&model_path.join("processor_config.json")) {
            return Ok(true);
        }
        let Some(keys) = bounded_weight_keys(model_path)? else {
            return Err(anyhow::anyhow!(
                "Gemma 4 text-vs-vision classification requires a bounded SafeTensors index or processor_config.json; catalog metadata does not read SafeTensors headers"
            ));
        };
        Ok(keys
            .iter()
            .any(|key| prefixes.iter().any(|p| key.starts_with(p))))
    }

    fn inkling_dir_is_mtp_only(&self, model_path: &Path, _config: &Value) -> anyhow::Result<bool> {
        let Some(keys) = bounded_weight_keys(model_path)? else {
            return Err(anyhow::anyhow!(
                "Inkling MTP-vs-target classification requires a bounded SafeTensors index; catalog metadata does not read SafeTensors headers"
            ));
        };
        let has_mtp = keys
            .iter()
            .any(|key| key.starts_with("model.mtp.layers.") || key.starts_with("mtp.layers."));
        let has_target = keys.iter().any(|key| {
            key == "model.llm.embed.weight"
                || key == "model.embed_tokens.weight"
                || key.starts_with("model.llm.layers.")
                || key.starts_with("model.layers.")
        });
        Ok(has_mtp && !has_target)
    }

    fn inkling_has_vision_weights(
        &self,
        model_path: &Path,
        _config: &Value,
    ) -> anyhow::Result<bool> {
        bounded_weight_prefix(model_path, "model.visual.")
    }

    fn kimi_k3_has_vision_weights(
        &self,
        model_path: &Path,
        _config: &Value,
    ) -> anyhow::Result<bool> {
        bounded_weight_prefix(model_path, "vision_tower.")
    }
}

fn bounded_weight_prefix(model_path: &Path, prefix: &str) -> anyhow::Result<bool> {
    let Some(keys) = bounded_weight_keys(model_path)? else {
        return Err(anyhow::anyhow!(
            "vision capability classification requires a bounded SafeTensors index; catalog metadata does not read SafeTensors headers"
        ));
    };
    Ok(keys.iter().any(|key| key.starts_with(prefix)))
}

fn bounded_weight_keys(model_path: &Path) -> anyhow::Result<Option<Vec<String>>> {
    let index_path = model_path.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }
    let (json, err) = read_json_bounded(&index_path, MAX_INDEX_BYTES);
    let value = json.ok_or_else(|| {
        anyhow::anyhow!(err.unwrap_or_else(|| "safetensors index is unreadable".to_string()))
    })?;
    let weight_map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("safetensors index has no weight_map"))?;
    Ok(Some(weight_map.keys().cloned().collect()))
}

fn regular_file_exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file())
        .unwrap_or(false)
}

fn rooted_regular_file_exists(root: &Path, components: &[&str]) -> bool {
    if components.is_empty() {
        return false;
    }
    let Ok(meta) = std::fs::symlink_metadata(root) else {
        return false;
    };
    if !meta.file_type().is_dir() {
        return false;
    }
    let mut path = root.to_path_buf();
    for component in &components[..components.len() - 1] {
        path.push(component);
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            return false;
        };
        if !meta.file_type().is_dir() {
            return false;
        }
    }
    path.push(components[components.len() - 1]);
    regular_file_exists(&path)
}

fn bounded_embedding_checkpoint(
    model_path: &Path,
    config: &Value,
) -> anyhow::Result<Option<ModelType>> {
    if is_sequence_classification_architecture(config) {
        return Ok(None);
    }
    let Some(model_type_raw) = config.get("model_type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let model_type = model_type_raw.to_ascii_lowercase();
    let encoder_only = is_encoder_only_model_type(&model_type);
    let layout_says_embedding = encoder_only
        || config_has_embedding_architecture(config)
        || bounded_modules_json_has_pooling(model_path)?
        || rooted_regular_file_exists(model_path, &["1_Pooling", "config.json"]);
    if !layout_says_embedding {
        return Ok(None);
    }
    embedding_variant_for_model_type(&model_type).map(Some).ok_or_else(|| {
        anyhow::anyhow!(
            "{} is an embedding checkpoint, but model_type `{model_type_raw}` has no embedding family in mlxcel",
            model_path.display()
        )
    })
}

fn bounded_modules_json_has_pooling(model_path: &Path) -> anyhow::Result<bool> {
    let modules_path = model_path.join("modules.json");
    if !modules_path.exists() {
        return Ok(false);
    }
    let (json, err) = read_json_bounded(&modules_path, MAX_CONFIG_BYTES);
    let value = json.ok_or_else(|| {
        anyhow::anyhow!(err.unwrap_or_else(|| "modules.json is unreadable".to_string()))
    })?;
    Ok(modules_json_value_has_pooling(&value))
}
