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

use crate::LoadedModel;
use crate::models::Inkling2Model;
use crate::models::inkling2::config::Inkling2Config;
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
