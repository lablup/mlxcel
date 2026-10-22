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

//! Standalone DINOv2 feature extraction through `mlxcel predict`.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Map, Value};

use crate::vision::encoders::dinov2::Dinov2;

use super::predictor::{DenseInput, DenseOutput, DensePredictor};
use super::preprocess::{image_to_nhwc_f32, normalize_imagenet, resize_to_patch_multiple};

const TASKS: &[&str] = &["features"];

fn preprocess(image: &DynamicImage, patch_size: usize) -> Result<UniquePtr<MlxArray>> {
    let image = image_to_nhwc_f32(image);
    let image = resize_to_patch_multiple(&image, patch_size)?;
    normalize_imagenet(&image)
}

pub struct Dinov2Predictor {
    model: Dinov2,
}

impl Dinov2Predictor {
    pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
        Ok(Box::new(Self {
            model: Dinov2::load(path)?,
        }))
    }
}

impl DensePredictor for Dinov2Predictor {
    fn family(&self) -> &'static str {
        "dinov2"
    }

    fn tasks(&self) -> &[&'static str] {
        TASKS
    }

    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        ensure!(
            opts.get("task").and_then(Value::as_str) == Some("features"),
            "DINOv2 supports only the `features` task"
        );
        let DenseInput::Image(image) = input else {
            bail!("DINOv2 feature extraction requires an image input");
        };
        let image = preprocess(image, self.model.config.patch_size).context("preprocess")?;
        let result = self.model.forward(&image, None)?;

        let mut output = DenseOutput::default();
        output
            .tensors
            .insert("last_hidden_state".into(), result.last_hidden_state);
        output
            .tensors
            .insert("pooler_output".into(), result.pooler_output);
        output
            .tensors
            .insert("hidden_patch_tokens".into(), result.hidden_patch_tokens);
        output.meta.insert(
            "model_type".into(),
            Value::String(self.model.config.model_type.clone()),
        );
        output.meta.insert(
            "patch_size".into(),
            Value::from(self.model.config.patch_size as u64),
        );
        Ok(output)
    }
}
