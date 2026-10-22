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

use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value};

use super::super::DenseInput;

pub(crate) struct PredictionOptions {
    pub task: String,
    pub prompts: Vec<String>,
    pub boxes: Vec<[f32; 4]>,
    pub threshold: f32,
    pub nms: f32,
}

impl PredictionOptions {
    pub fn parse(input: &DenseInput, opts: &Map<String, Value>) -> Result<Self> {
        let task = opts
            .get("task")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("SAM 3 task is missing"))?
            .to_owned();
        let prompts = match opts.get("prompts") {
            None => Vec::new(),
            Some(Value::Array(values)) => values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow::anyhow!("SAM 3 prompts must be strings"))
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => bail!("SAM 3 prompts must be an array"),
        };
        ensure!(
            prompts.iter().all(|prompt| !prompt.trim().is_empty()),
            "SAM 3 prompts must not be empty"
        );
        let boxes = opts
            .get("boxes")
            .map(|value| serde_json::from_value::<Vec<[f32; 4]>>(value.clone()))
            .transpose()
            .map_err(|error| anyhow::anyhow!("parse SAM 3 boxes: {error}"))?
            .unwrap_or_default();
        ensure!(
            boxes
                .iter()
                .flatten()
                .all(|coordinate| coordinate.is_finite()),
            "SAM 3 boxes must be finite"
        );
        let default_threshold = if input.is_video() { 0.15 } else { 0.3 };
        let threshold = number(opts, "threshold", default_threshold)?;
        let nms = number(opts, "nms", 0.5)?;
        ensure!(
            (0.0..=1.0).contains(&threshold),
            "SAM 3 threshold must be between 0 and 1"
        );
        ensure!(
            (0.0..=1.0).contains(&nms),
            "SAM 3 NMS threshold must be between 0 and 1"
        );

        match (task.as_str(), input) {
            ("detect" | "segment", DenseInput::Image(_)) => ensure!(
                !prompts.is_empty(),
                "SAM 3 {task} requires at least one --prompt"
            ),
            ("track", DenseInput::Video { .. }) => ensure!(
                !prompts.is_empty() || !boxes.is_empty(),
                "SAM 3 track requires at least one --prompt or --boxes"
            ),
            ("detect" | "segment", DenseInput::Video { .. }) => {
                bail!("SAM 3 {task} requires --image")
            }
            ("track", DenseInput::Image(_)) => bail!("SAM 3 track requires --video"),
            _ => bail!("unsupported SAM 3 task `{task}`"),
        }
        Ok(Self {
            task,
            prompts,
            boxes,
            threshold,
            nms,
        })
    }
}

fn number(opts: &Map<String, Value>, name: &str, default: f32) -> Result<f32> {
    let value = match opts.get(name) {
        Some(value) => value
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("SAM 3 `{name}` must be a number"))?
            as f32,
        None => default,
    };
    ensure!(value.is_finite(), "SAM 3 `{name}` must be finite");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;

    use super::*;

    #[test]
    fn image_defaults_and_constraints_are_checked() {
        let input = DenseInput::Image(DynamicImage::new_rgb8(4, 4));
        let mut opts = Map::from_iter([
            ("task".into(), Value::String("segment".into())),
            (
                "prompts".into(),
                Value::Array(vec![Value::String("cat".into())]),
            ),
        ]);
        let parsed = PredictionOptions::parse(&input, &opts).unwrap();
        assert_eq!(parsed.threshold, 0.3);
        assert_eq!(parsed.nms, 0.5);
        opts.remove("prompts");
        assert!(PredictionOptions::parse(&input, &opts).is_err());
    }
}
