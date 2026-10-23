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

//! MoGe3 monocular geometry prediction.

mod config;
mod config_components;
mod conv_stack;
mod encoder;
mod geometry;
mod model;
mod processor;
pub mod refiner;
pub mod sparse;
mod sparse_shape;
mod weights;

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{DenseInput, DenseOutput, DensePredictor};
pub use config::Moge3Config;
pub use model::{Moge3, Moge3Output};
pub use processor::InferOptions;

const TASKS: &[&str] = &["geometry"];
pub struct Moge3Predictor {
    model: Moge3,
}
pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Moge3Predictor {
        model: Moge3::load(path)?,
    }))
}

impl DensePredictor for Moge3Predictor {
    fn family(&self) -> &'static str {
        "moge3"
    }
    fn tasks(&self) -> &[&'static str] {
        TASKS
    }
    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        let DenseInput::Image(image) = input else {
            bail!("MoGe3 supports still images only")
        };
        let options = InferOptions::parse(self.model.config(), opts)?;
        let array = processor::image_to_array(image).context("preprocess")?;
        let array = mlxcel_core::squeeze_axis(&array, 0);
        let output = self.model.estimate_with_options(&array, options)?;
        let mut dense = DenseOutput::default();
        dense.tensors.insert("points".into(), output.points);
        dense.tensors.insert("depth".into(), output.depth);
        dense.tensors.insert("mask".into(), output.mask);
        dense.tensors.insert("normal".into(), output.normal);
        dense.tensors.insert("intrinsics".into(), output.intrinsics);
        if options.return_per_step {
            for (i, value) in output.points_per_step.into_iter().enumerate() {
                dense.tensors.insert(format!("points_step{i}"), value);
            }
            for (i, value) in output.depth_per_step.into_iter().enumerate() {
                dense.tensors.insert(format!("depth_step{i}"), value);
            }
            for (i, value) in output.intrinsics_per_step.into_iter().enumerate() {
                dense.tensors.insert(format!("intrinsics_step{i}"), value);
            }
        }
        dense
            .meta
            .insert("num_tokens".into(), json!(options.num_tokens));
        dense
            .meta
            .insert("refine_steps".into(), json!(options.refine_steps));
        dense
            .meta
            .insert("force_projection".into(), json!(options.force_projection));
        dense
            .meta
            .insert("apply_mask".into(), json!(options.apply_mask));
        if let Some(fov) = options.fov_x {
            dense.meta.insert("fov_x".into(), json!(fov));
        }
        Ok(dense)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
