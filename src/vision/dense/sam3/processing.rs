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

//! SAM 3 image and CLIP-text preprocessing.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::super::preprocess::resize_rgb8_bilinear_nhwc_f32;

pub(crate) struct TokenizedPrompt {
    pub input_ids: UniquePtr<MlxArray>,
    pub attention_mask: UniquePtr<MlxArray>,
}

pub(crate) struct Sam3Processor {
    tokenizer: MlxcelTokenizer,
    image_size: usize,
    max_text_length: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    pad_token_id: u32,
    eos_token_id: u32,
}

impl Sam3Processor {
    pub(crate) fn load(
        model_path: &Path,
        image_size: usize,
        max_text_length: usize,
        configured_eos_token_id: u32,
    ) -> Result<Self> {
        let model_image_size = image_size;
        let model_max_text_length = max_text_length;
        let processor_config = read_optional_json(&model_path.join("processor_config.json"))?;
        let image_processor = processor_config
            .as_ref()
            .and_then(|config| config.get("image_processor"));
        require_enabled(image_processor, "do_resize")?;
        require_enabled(image_processor, "do_rescale")?;
        require_enabled(image_processor, "do_normalize")?;
        if let Some(value) = image_processor.and_then(|config| config.get("rescale_factor")) {
            let factor = value
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("SAM 3 processor rescale_factor must be numeric"))?;
            ensure!(
                (factor - 1.0 / 255.0).abs() <= 1e-12,
                "SAM 3 processor rescale_factor must be 1/255"
            );
        }
        let image_size = image_processor
            .and_then(|config| config.get("size"))
            .and_then(|size| size.get("height"))
            .and_then(Value::as_u64)
            .map(|size| size as usize)
            .unwrap_or(image_size);
        let image_width = image_processor
            .and_then(|config| config.get("size"))
            .and_then(|size| size.get("width"))
            .and_then(Value::as_u64)
            .map(|size| size as usize)
            .unwrap_or(image_size);
        ensure!(image_size > 0, "SAM 3 image size must be positive");
        ensure!(
            image_size == image_width,
            "SAM 3 processor image size must be square"
        );
        ensure!(
            image_size == model_image_size,
            "SAM 3 processor image size {image_size} differs from model image size {model_image_size}"
        );
        let max_text_length = processor_config
            .as_ref()
            .and_then(|config| config.get("max_text_length"))
            .and_then(Value::as_u64)
            .map(|length| length as usize)
            .unwrap_or(max_text_length);
        ensure!(
            max_text_length >= 2,
            "SAM 3 text length must be at least two"
        );
        ensure!(
            max_text_length <= model_max_text_length,
            "SAM 3 processor text length {max_text_length} exceeds model position limit {model_max_text_length}"
        );
        let image_mean = image_channels(image_processor, "image_mean", [0.5; 3])?;
        let image_std = image_channels(image_processor, "image_std", [0.5; 3])?;
        ensure!(
            image_std.iter().all(|value| *value != 0.0),
            "SAM 3 image std values must be non-zero"
        );
        let tokenizer = load_tokenizer(model_path)
            .with_context(|| format!("load SAM 3 tokenizer from {}", model_path.display()))?;
        let tokenizer_config = read_optional_json(&model_path.join("tokenizer_config.json"))?;
        let token_id = |field: &str| {
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get(field))
                .and_then(Value::as_str)
                .and_then(|token| tokenizer.token_to_id(token))
        };
        let eos_token_id = token_id("eos_token").unwrap_or(configured_eos_token_id);
        let pad_token_id = token_id("pad_token").unwrap_or(eos_token_id);

        Ok(Self {
            tokenizer,
            image_size,
            max_text_length,
            image_mean,
            image_std,
            pad_token_id,
            eos_token_id,
        })
    }

    pub(crate) fn preprocess_image(&self, image: &DynamicImage) -> Result<UniquePtr<MlxArray>> {
        let pixels =
            resize_rgb8_bilinear_nhwc_f32(image, self.image_size as u32, self.image_size as u32)?;
        let mean = mlxcel_core::from_slice_f32(&self.image_mean, &[1, 1, 1, 3]);
        let std = mlxcel_core::from_slice_f32(&self.image_std, &[1, 1, 1, 3]);
        Ok(mlxcel_core::divide(
            &mlxcel_core::subtract(&pixels, &mean),
            &std,
        ))
    }

    pub(crate) fn tokenize(&self, prompt: &str) -> Result<TokenizedPrompt> {
        let ids = self
            .tokenizer
            .encode(prompt, true)
            .with_context(|| format!("tokenize SAM 3 prompt {prompt:?}"))?;
        let (ids, mask) = truncate_and_pad(
            ids,
            self.max_text_length,
            self.pad_token_id,
            self.eos_token_id,
        );
        let ids: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
        let mask: Vec<i32> = mask.into_iter().map(i32::from).collect();
        let length = self.max_text_length as i32;
        Ok(TokenizedPrompt {
            input_ids: mlxcel_core::from_slice_i32(&ids, &[1, length]),
            attention_mask: mlxcel_core::from_slice_i32(&mask, &[1, length]),
        })
    }
}

fn require_enabled(image_processor: Option<&Value>, name: &str) -> Result<()> {
    if let Some(value) = image_processor.and_then(|config| config.get(name)) {
        ensure!(
            value.as_bool() == Some(true),
            "SAM 3 processor `{name}` must be true"
        );
    }
    Ok(())
}

fn image_channels(
    image_processor: Option<&Value>,
    name: &str,
    default: [f32; 3],
) -> Result<[f32; 3]> {
    let Some(values) = image_processor
        .and_then(|config| config.get(name))
        .and_then(Value::as_array)
    else {
        return Ok(default);
    };
    ensure!(values.len() == 3, "SAM 3 `{name}` must have three channels");
    let mut output = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        output[index] = value
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("SAM 3 `{name}` values must be numbers"))?
            as f32;
    }
    ensure!(
        output.iter().all(|value| value.is_finite()),
        "SAM 3 `{name}` values must be finite"
    );
    Ok(output)
}

fn read_optional_json(path: &Path) -> Result<Option<Value>> {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))
            .map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}

fn truncate_and_pad(
    mut ids: Vec<u32>,
    max_length: usize,
    pad_token_id: u32,
    eos_token_id: u32,
) -> (Vec<u32>, Vec<u8>) {
    if ids.len() > max_length {
        ids.truncate(max_length);
        ids[max_length - 1] = eos_token_id;
    }
    let content_length = ids.len();
    ids.resize(max_length, pad_token_id);
    let mask = (0..max_length)
        .map(|index| u8::from(index < content_length))
        .collect();
    (ids, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_preserves_eos_and_padding_builds_attention_mask() {
        let (short, mask) = truncate_and_pad(vec![10, 11, 12], 5, 99, 12);
        assert_eq!(short, vec![10, 11, 12, 99, 99]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0]);

        let (long, mask) = truncate_and_pad(vec![10, 11, 12, 13, 14, 15], 5, 99, 77);
        assert_eq!(long, vec![10, 11, 12, 13, 77]);
        assert_eq!(mask, vec![1, 1, 1, 1, 1]);
    }
}
