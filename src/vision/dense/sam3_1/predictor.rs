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

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use super::super::sam3::options::PredictionOptions;
use super::super::sam3::processing::Sam3Processor;
use super::super::{DenseInput, DenseOutput, DensePredictor};
use super::config::Sam31Config;
use super::model::Sam31Model;

const TASKS: &[&str] = &["detect", "segment", "track"];

pub(crate) struct Sam31Predictor {
    pub(super) model: Sam31Model,
    pub(super) processor: Sam3Processor,
    #[cfg(test)]
    pub(super) parameter_dtypes: Vec<(String, i32)>,
}

impl Sam31Predictor {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let config: Sam31Config = serde_json::from_value(json.clone())
            .with_context(|| format!("decode {} as SAM 3.1", config_path.display()))?;
        let config = config.normalize();
        config.validate().map_err(anyhow::Error::msg)?;

        let mut weights = mlxcel_core::weights::load_weights_from_dir(path)
            .map_err(anyhow::Error::msg)
            .context("load SAM 3.1 weights")?;
        weights =
            super::super::sam3::sanitize::sanitize(weights).context("sanitize SAM 3.1 weights")?;
        if crate::models::bf16_to_f16_at_load(
            crate::models::config_has_quantization_metadata(&json),
            Some(&json),
        ) && crate::models::convert_bf16_weights(&mut weights)
        {
            crate::models::warn_bf16_precision();
        }
        #[cfg(test)]
        let parameter_dtypes = [
            "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight",
            "detector_model.dot_product_scoring.query_proj.weight",
            "tracker_model.memory_attention.layer_norm.weight",
            "tracker_model.output_valid_embed",
        ]
        .into_iter()
        .map(|name| {
            let dtype = weights
                .get(name)
                .map(|value| mlxcel_core::array_dtype(value))
                .ok_or_else(|| anyhow::anyhow!("SAM 3.1 dtype probe weight missing: {name}"))?;
            Ok((name.to_owned(), dtype))
        })
        .collect::<Result<Vec<_>>>()?;
        let processor = Sam3Processor::load(
            path,
            config
                .detector_config
                .vision_config
                .backbone_config
                .image_size,
            config.detector_config.text_config.max_position_embeddings,
            config.detector_config.text_config.eos_token_id as u32,
        )?;
        let model = Sam31Model::from_weights(weights, config).map_err(anyhow::Error::msg)?;
        Ok(Self {
            model,
            processor,
            #[cfg(test)]
            parameter_dtypes,
        })
    }
}

impl DensePredictor for Sam31Predictor {
    fn family(&self) -> &'static str {
        "sam3_1"
    }

    fn tasks(&self) -> &[&'static str] {
        TASKS
    }

    fn default_task(&self, input: &DenseInput) -> Option<&'static str> {
        matches!(input, DenseInput::Image(_)).then_some("segment")
    }

    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        let options = PredictionOptions::parse(input, opts)?;
        match input {
            DenseInput::Image(image) => super::super::sam3::image::predict_image(
                &self.model,
                &self.processor,
                image,
                &options,
            ),
            DenseInput::Video { frames, .. } => {
                super::video::track_video(&self.model, &self.processor, frames, &options)
            }
        }
    }
}

pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Sam31Predictor::load(path)?))
}
