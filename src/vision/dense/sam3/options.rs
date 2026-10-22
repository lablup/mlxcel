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

const MAX_PROMPT_OBJECTS: usize = 64;
const MAX_MASK_WORKING_SET_BYTES: u64 = 512 * 1024 * 1024;
// Selected masks are resized as f32, converted to retained uint8 rows, then
// concatenated into a second uint8 output. Native masks and graph retention
// add headroom beyond that 6-byte floor.
const MASK_WORKING_SET_BYTES_PER_PIXEL: u64 = 8;
const LOW_RES_MASK_SIZE: u64 = 288;
const DETECTOR_QUERY_COUNT: usize = 200;

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
        ensure!(
            boxes
                .iter()
                .all(|bbox| bbox[2] > bbox[0] && bbox[3] > bbox[1]),
            "SAM 3 box right and bottom coordinates must exceed left and top"
        );
        ensure!(
            prompts.len().saturating_add(boxes.len()) <= MAX_PROMPT_OBJECTS,
            "SAM 3 accepts at most {MAX_PROMPT_OBJECTS} prompt objects per request"
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
        validate_mask_output_budget(input, &task, prompts.len(), boxes.len())?;
        Ok(Self {
            task,
            prompts,
            boxes,
            threshold,
            nms,
        })
    }
}

fn validate_mask_output_budget(
    input: &DenseInput,
    task: &str,
    prompt_count: usize,
    box_count: usize,
) -> Result<()> {
    let rows = match (task, input) {
        ("segment", DenseInput::Image(_)) => prompt_count.checked_mul(DETECTOR_QUERY_COUNT),
        ("track", DenseInput::Video { frames, .. }) => prompt_count
            .checked_add(box_count)
            .and_then(|objects| objects.checked_mul(frames.len())),
        _ => return Ok(()),
    }
    .ok_or_else(|| anyhow::anyhow!("SAM 3 mask row count overflowed"))?;
    let (width, height) = input
        .dimensions()
        .ok_or_else(|| anyhow::anyhow!("SAM 3 input has no dimensions"))?;
    let rows_per_resize = if task == "segment" {
        DETECTOR_QUERY_COUNT
    } else {
        1
    };
    validate_mask_bytes(width, height, rows, rows_per_resize)
}

fn validate_mask_bytes(width: u32, height: u32, rows: usize, rows_per_resize: usize) -> Result<()> {
    let output_bytes = (rows as u64)
        .checked_mul(u64::from(width))
        .and_then(|value| value.checked_mul(u64::from(height)))
        .ok_or_else(|| anyhow::anyhow!("SAM 3 mask output size overflowed"))?;
    let proportional_working_set = output_bytes
        .checked_mul(MASK_WORKING_SET_BYTES_PER_PIXEL)
        .ok_or_else(|| anyhow::anyhow!("SAM 3 mask working-set size overflowed"))?;
    let y_taps = if height < LOW_RES_MASK_SIZE as u32 {
        (2 * LOW_RES_MASK_SIZE).div_ceil(u64::from(height))
    } else {
        2
    };
    let height_pass_bytes = (rows_per_resize as u64)
        .checked_mul(u64::from(height))
        .and_then(|value| value.checked_mul(LOW_RES_MASK_SIZE))
        .and_then(|value| value.checked_mul(y_taps + 1))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| anyhow::anyhow!("SAM 3 mask interpolation workspace overflowed"))?;
    let retained_and_concatenated = output_bytes
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("SAM 3 retained mask size overflowed"))?;
    let interpolation_working_set = retained_and_concatenated
        .checked_add(height_pass_bytes)
        .ok_or_else(|| anyhow::anyhow!("SAM 3 mask working-set size overflowed"))?;
    let working_set_bytes = proportional_working_set.max(interpolation_working_set);
    ensure!(
        working_set_bytes <= MAX_MASK_WORKING_SET_BYTES,
        "SAM 3 worst-case mask working set requires {working_set_bytes} bytes for a {output_bytes}-byte output, exceeding the {MAX_MASK_WORKING_SET_BYTES}-byte limit"
    );
    Ok(())
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

    #[test]
    fn mask_output_budget_rejects_amplifying_requests() {
        assert!(
            validate_mask_bytes(2048, 2048, DETECTOR_QUERY_COUNT, DETECTOR_QUERY_COUNT).is_err()
        );
        assert!(validate_mask_bytes(1920, 1080, 32 * 9, 1).is_err());
        assert!(
            validate_mask_bytes(1, 160_000, DETECTOR_QUERY_COUNT, DETECTOR_QUERY_COUNT).is_err()
        );
        assert!(validate_mask_bytes(640, 480, DETECTOR_QUERY_COUNT, DETECTOR_QUERY_COUNT).is_ok());
    }
}
