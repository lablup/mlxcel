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

//! Dense-prediction family registry.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::predictor::DensePredictor;

pub type FamilyLoader = fn(&Path) -> Result<Box<dyn DensePredictor>>;

/// One checkpoint `model_type` to predictor-constructor mapping.
#[derive(Debug, Clone, Copy)]
pub struct DenseFamily {
    pub model_type: &'static str,
    /// Whether this family accepts text and box prompts through `mlxcel predict`.
    pub supports_prompts: bool,
    pub load: FamilyLoader,
}

/// Families this build can execute. Model implementation issues add rows here.
pub static FAMILIES: &[DenseFamily] = &[
    DenseFamily {
        model_type: "dinov2",
        supports_prompts: false,
        load: super::dinov2::Dinov2Predictor::load,
    },
    DenseFamily {
        model_type: "dinov2_with_registers",
        supports_prompts: false,
        load: super::dinov2::Dinov2Predictor::load,
    },
    DenseFamily {
        model_type: "sapiens2",
        supports_prompts: false,
        load: super::sapiens2::load,
    },
    DenseFamily {
        model_type: "sam3_video",
        supports_prompts: true,
        load: super::sam3::load,
    },
    DenseFamily {
        model_type: "sam3",
        supports_prompts: true,
        load: super::sam3::load,
    },
];

/// Dense checkpoint types recognized even before their model ports land.
pub const KNOWN_MODEL_TYPES: &[&str] = &[
    "dinov2",
    "dinov2_with_registers",
    "moge3",
    "sam3",
    "sam3_video",
    "sam3d_objects",
    "sapiens2",
    "video_depth_anything",
];

pub fn is_dense_model_type(model_type: &str) -> bool {
    KNOWN_MODEL_TYPES.contains(&model_type.to_ascii_lowercase().as_str())
}

fn checkpoint_model_type(path: &Path) -> Result<String> {
    let config_path = path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let config: Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", config_path.display()))?;
    config
        .get("model_type")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| anyhow::anyhow!("model_type not found in {}", config_path.display()))
}

pub fn resolve_family<'a>(path: &Path, families: &'a [DenseFamily]) -> Result<&'a DenseFamily> {
    let model_type = checkpoint_model_type(path)?;
    if !is_dense_model_type(&model_type) {
        bail!("checkpoint model_type `{model_type}` is not a recognized dense-prediction family");
    }
    families
        .iter()
        .find(|family| family.model_type == model_type)
        .ok_or_else(|| {
            anyhow::anyhow!("dense-prediction family {model_type} is not supported by this build")
        })
}

pub fn load_dense_model(path: &Path) -> Result<Box<dyn DensePredictor>> {
    load_dense_model_from(path, FAMILIES)
}

pub fn load_dense_model_from(
    path: &Path,
    families: &[DenseFamily],
) -> Result<Box<dyn DensePredictor>> {
    (resolve_family(path, families)?.load)(path)
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
