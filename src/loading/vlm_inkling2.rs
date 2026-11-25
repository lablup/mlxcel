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

//! Inkling 2 (`inkling2`, `inkling2_loop`) loading boundary.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::MlxArray;
use mlxcel_core::weight_materialize::{MaterializationPolicy, is_multimodal_tower_tensor};
use mlxcel_core::weights::WeightMap;

use crate::LoadedModel;
use crate::models::Inkling2Model;
use crate::models::inkling2::config::Inkling2Config;
use crate::models::inkling2::sanitize::TEXT_PREFIX;
use crate::models::inkling2::{dense_quantization, quant, sanitize};
use crate::vision::Inkling2VlModel;
use crate::vision::encoders::inkling2_vit::Inkling2VisionTower;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

pub(crate) fn load_inkling2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_raw, _) = read_sanitized_vlm_config(model_path)?;
    let config = Inkling2Config::from_json(&config_raw).map_err(anyhow::Error::msg)?;
    // The native release's per-layer NVFP4 / MXFP8 table; absent from MLX
    // conversions, whose expert planes are classified from their tensors.
    let table = quant::read_native_quant_table(model_path).map_err(anyhow::Error::msg)?;
    let weights = load_vlm_weights_common(model_path, None)?;
    let weights = sanitize::sanitize_weights(weights, config.text_config.hidden_size)
        .map_err(anyhow::Error::msg)?;
    if MaterializationPolicy::from_env() == MaterializationPolicy::Eager {
        materialize_by_layer(&weights, config.text_config.num_hidden_layers);
    }
    let quant = dense_quantization(&config);
    let vision = match &config.vision_config {
        Some(vision_config) if weights.keys().any(|key| key.starts_with("vision_tower.")) => Some(
            Inkling2VisionTower::from_weights(&weights, vision_config, quant)
                .map_err(anyhow::Error::msg)?,
        ),
        _ => None,
    };
    let image_token_id = config.image_token_id;
    let text =
        Inkling2Model::from_weights(config, weights, table.as_ref()).map_err(anyhow::Error::msg)?;
    Ok(LoadedModel::Inkling2VLM(Inkling2VlModel::new(
        text,
        vision,
        image_token_id,
    )))
}

/// The sanitized text tensors in evaluation order: one group per decoder
/// layer, then everything else. Modality-tower tensors are left out, since
/// the materialization policy keeps them lazy until a request needs them.
fn layer_groups(weights: &WeightMap, layers: usize) -> Vec<Vec<&str>> {
    let mut groups: Vec<Vec<&str>> = vec![Vec::new(); layers + 1];
    for name in weights.keys() {
        if is_multimodal_tower_tensor(name) {
            continue;
        }
        let layer = name
            .strip_prefix(TEXT_PREFIX)
            .and_then(|rest| rest.strip_prefix("layers."))
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(index, _)| index.parse::<usize>().ok())
            .filter(|&index| index < layers);
        groups[layer.unwrap_or(layers)].push(name.as_str());
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups
}

/// Restores the MLX buffer-cache limit when dropped.
struct CacheLimitGuard(u64);

impl Drop for CacheLimitGuard {
    fn drop(&mut self) {
        mlxcel_core::memory::set_cache_limit(self.0);
    }
}

/// Evaluate the sanitized weights one decoder layer at a time with the MLX
/// buffer cache off, the bounded load mlx-vlm's Inkling 2 README gives for
/// Small (#3373).
///
/// Sanitizing splits every interleaved expert `w13` into contiguous gate and
/// up planes, so about two thirds of the expert bytes are a lazy copy of a
/// checkpoint tensor. Left to the first token, all 48 layers' copies are
/// produced in one graph while their sources and the freed source buffers in
/// MLX's cache are still held: on Small (363 GiB of weights) the process
/// footprint peaked at about 459 GiB, 10% free memory on a 512 GB host.
/// Here each layer's copies are produced and its `w13` sources released
/// before the next layer is read, so the peak stays near one copy of the
/// weights. The values are the same ones the first token would compute.
fn materialize_by_layer(weights: &WeightMap, layers: usize) {
    let _cache = CacheLimitGuard(mlxcel_core::memory::set_cache_limit(0));
    for group in layer_groups(weights, layers) {
        let arrays: Vec<*const MlxArray> = group
            .iter()
            .filter_map(|name| weights.get(*name))
            .map(|array| &**array as *const MlxArray)
            .collect();
        if arrays.is_empty() {
            continue;
        }
        // SAFETY: every pointer refers to an array owned by `weights`, which
        // outlives both calls and is not mutated while they run.
        unsafe {
            mlxcel_core::eval_all(&arrays);
            mlxcel_core::detach_all(&arrays);
        }
    }
}

#[cfg(test)]
#[path = "vlm_inkling2_tests.rs"]
mod tests;
