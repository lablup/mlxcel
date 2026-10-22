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
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::vision::encoders::dinov2::Dinov2;

use super::dinov2_resize::resize_pil_bicubic_rgb;
use super::predictor::{DenseInput, DenseOutput, DensePredictor};
const TASKS: &[&str] = &["features"];
const DEFAULT_SHORTEST_EDGE: usize = 256;
const DEFAULT_CROP_SIZE: usize = 224;
const DEFAULT_RESCALE_FACTOR: f32 = 1.0 / 255.0;
const DEFAULT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DEFAULT_STD: [f32; 3] = [0.229, 0.224, 0.225];
const MAX_DINOV2_PROCESSOR_PIXELS: u64 = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct RawSize {
    #[serde(default)]
    shortest_edge: Option<usize>,
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    width: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawProcessorConfig {
    #[serde(default)]
    do_resize: Option<bool>,
    #[serde(default)]
    size: Option<RawSize>,
    #[serde(default)]
    resample: Option<u8>,
    #[serde(default)]
    do_center_crop: Option<bool>,
    #[serde(default)]
    crop_size: Option<RawSize>,
    #[serde(default)]
    do_rescale: Option<bool>,
    #[serde(default)]
    rescale_factor: Option<f32>,
    #[serde(default)]
    do_normalize: Option<bool>,
    #[serde(default)]
    image_mean: Option<Vec<f32>>,
    #[serde(default)]
    image_std: Option<Vec<f32>>,
    #[serde(default)]
    do_convert_rgb: Option<bool>,
}

/// Hugging Face `BitImageProcessor` settings used by public DINOv2 checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct Dinov2ImageProcessor {
    do_resize: bool,
    shortest_edge: usize,
    do_center_crop: bool,
    crop_height: usize,
    crop_width: usize,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl Default for Dinov2ImageProcessor {
    fn default() -> Self {
        Self {
            do_resize: true,
            shortest_edge: DEFAULT_SHORTEST_EDGE,
            do_center_crop: true,
            crop_height: DEFAULT_CROP_SIZE,
            crop_width: DEFAULT_CROP_SIZE,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
            do_normalize: true,
            image_mean: DEFAULT_MEAN,
            image_std: DEFAULT_STD,
        }
    }
}

fn rgb_triplet(values: Option<Vec<f32>>, default: [f32; 3], name: &str) -> Result<[f32; 3]> {
    let Some(values) = values else {
        return Ok(default);
    };
    ensure!(values.len() == 3, "DINOv2 {name} must contain three values");
    let result = [values[0], values[1], values[2]];
    ensure!(
        result.iter().all(|value| value.is_finite()),
        "DINOv2 {name} values must be finite"
    );
    Ok(result)
}

fn crop_dimensions(size: Option<RawSize>) -> Result<(usize, usize)> {
    let Some(size) = size else {
        return Ok((DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE));
    };
    match (size.height, size.width, size.shortest_edge) {
        (Some(height), Some(width), None) => Ok((height, width)),
        (None, None, Some(edge)) => Ok((edge, edge)),
        _ => bail!("DINOv2 crop_size must contain height and width"),
    }
}

impl Dinov2ImageProcessor {
    pub fn from_pretrained(path: &Path) -> Result<Self> {
        let config_path = path.join("preprocessor_config.json");
        if !config_path.is_file() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let raw: RawProcessorConfig = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", config_path.display()))?;
        ensure!(
            raw.do_convert_rgb.unwrap_or(true),
            "DINOv2 requires preprocessor RGB conversion"
        );
        ensure!(
            raw.resample.unwrap_or(3) == 3,
            "DINOv2 supports the Hugging Face PIL bicubic resampler (3)"
        );
        let shortest_edge = match raw.size {
            Some(RawSize {
                shortest_edge: Some(edge),
                height: None,
                width: None,
            }) => edge,
            None => DEFAULT_SHORTEST_EDGE,
            _ => bail!("DINOv2 size must contain only shortest_edge"),
        };
        let (crop_height, crop_width) = crop_dimensions(raw.crop_size)?;
        let image_mean = rgb_triplet(raw.image_mean, DEFAULT_MEAN, "image_mean")?;
        let image_std = rgb_triplet(raw.image_std, DEFAULT_STD, "image_std")?;
        let rescale_factor = raw.rescale_factor.unwrap_or(DEFAULT_RESCALE_FACTOR);
        ensure!(
            shortest_edge > 0 && crop_height > 0 && crop_width > 0,
            "DINOv2 processor sizes must be positive"
        );
        ensure!(
            shortest_edge <= u32::MAX as usize
                && crop_height <= u32::MAX as usize
                && crop_width <= u32::MAX as usize,
            "DINOv2 processor sizes exceed image limits"
        );
        let crop_pixels = (crop_height as u64)
            .checked_mul(crop_width as u64)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 crop pixel count overflowed"))?;
        ensure!(
            crop_pixels <= MAX_DINOV2_PROCESSOR_PIXELS,
            "DINOv2 crop exceeds the safety limit of {MAX_DINOV2_PROCESSOR_PIXELS} pixels"
        );
        ensure!(
            rescale_factor.is_finite() && rescale_factor > 0.0,
            "DINOv2 rescale_factor must be finite and positive"
        );
        ensure!(
            image_std.iter().all(|value| *value > 0.0),
            "DINOv2 image_std values must be positive"
        );
        Ok(Self {
            do_resize: raw.do_resize.unwrap_or(true),
            shortest_edge,
            do_center_crop: raw.do_center_crop.unwrap_or(true),
            crop_height,
            crop_width,
            do_rescale: raw.do_rescale.unwrap_or(true),
            rescale_factor,
            do_normalize: raw.do_normalize.unwrap_or(true),
            image_mean,
            image_std,
        })
    }

    fn resize_shortest_edge(&self, image: &RgbImage) -> Result<RgbImage> {
        if !self.do_resize {
            return Ok(image.clone());
        }
        let (width, height) = image.dimensions();
        ensure!(
            width > 0 && height > 0,
            "DINOv2 input image must not be empty"
        );
        ensure!(
            u64::from(width) * u64::from(height) <= MAX_DINOV2_PROCESSOR_PIXELS,
            "DINOv2 input exceeds the safety limit of {MAX_DINOV2_PROCESSOR_PIXELS} pixels"
        );
        let short = u64::from(width.min(height));
        let long = u64::from(width.max(height));
        let target_short = self.shortest_edge as u64;
        let target_long = target_short
            .checked_mul(long)
            .ok_or_else(|| anyhow::anyhow!("DINOv2 resized image dimension overflowed"))?
            / short;
        ensure!(
            target_long <= u64::from(u32::MAX),
            "DINOv2 resized image dimension exceeds image limits"
        );
        let (target_width, target_height) = if width <= height {
            (target_short as u32, target_long as u32)
        } else {
            (target_long as u32, target_short as u32)
        };
        ensure!(
            u64::from(target_width) * u64::from(target_height) <= MAX_DINOV2_PROCESSOR_PIXELS,
            "DINOv2 resize exceeds the safety limit of {MAX_DINOV2_PROCESSOR_PIXELS} pixels"
        );
        resize_pil_bicubic_rgb(image, target_width, target_height)
    }

    fn center_crop(&self, image: &RgbImage) -> Result<RgbImage> {
        if !self.do_center_crop {
            return Ok(image.clone());
        }
        let (width, height) = image.dimensions();
        let crop_width = self.crop_width as u32;
        let crop_height = self.crop_height as u32;
        if width >= crop_width && height >= crop_height {
            let left = (width - crop_width) / 2;
            let top = (height - crop_height) / 2;
            return Ok(
                image::imageops::crop_imm(image, left, top, crop_width, crop_height).to_image(),
            );
        }

        let padded_width = width.max(crop_width);
        let padded_height = height.max(crop_height);
        ensure!(
            u64::from(padded_width) * u64::from(padded_height) <= MAX_DINOV2_PROCESSOR_PIXELS,
            "DINOv2 center-crop padding exceeds the safety limit of {MAX_DINOV2_PROCESSOR_PIXELS} pixels"
        );
        let mut padded = ImageBuffer::from_pixel(padded_width, padded_height, Rgb([0, 0, 0]));
        let left_pad = (padded_width - width).div_ceil(2);
        let top_pad = (padded_height - height).div_ceil(2);
        image::imageops::replace(&mut padded, image, i64::from(left_pad), i64::from(top_pad));
        let left = (padded_width - crop_width) / 2;
        let top = (padded_height - crop_height) / 2;
        Ok(image::imageops::crop_imm(&padded, left, top, crop_width, crop_height).to_image())
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<UniquePtr<MlxArray>> {
        let (source_width, source_height) = image.dimensions();
        ensure!(
            source_width > 0 && source_height > 0,
            "DINOv2 input image must not be empty"
        );
        ensure!(
            u64::from(source_width) * u64::from(source_height) <= MAX_DINOV2_PROCESSOR_PIXELS,
            "DINOv2 input exceeds the safety limit of {MAX_DINOV2_PROCESSOR_PIXELS} pixels"
        );
        let rgb = image.to_rgb8();
        let rgb = self.resize_shortest_edge(&rgb)?;
        let rgb = self.center_crop(&rgb)?;
        let (width, height) = rgb.dimensions();
        ensure!(
            width > 0 && height > 0,
            "DINOv2 input image must not be empty"
        );
        let mut pixels = Vec::with_capacity(rgb.as_raw().len());
        for (index, value) in rgb.as_raw().iter().enumerate() {
            let channel = index % 3;
            let mut value = f32::from(*value);
            if self.do_rescale {
                value *= self.rescale_factor;
            }
            if self.do_normalize {
                value = (value - self.image_mean[channel]) / self.image_std[channel];
            }
            pixels.push(value);
        }
        Ok(mlxcel_core::from_slice_f32(
            &pixels,
            &[1, height as i32, width as i32, 3],
        ))
    }
}

pub struct Dinov2Predictor {
    model: Dinov2,
    processor: Dinov2ImageProcessor,
}

impl Dinov2Predictor {
    pub fn load(path: &Path) -> Result<Box<dyn DensePredictor>> {
        Ok(Box::new(Self {
            model: Dinov2::load(path)?,
            processor: Dinov2ImageProcessor::from_pretrained(path)?,
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
        let image = self.processor.preprocess(image).context("preprocess")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tensor(name: &str) -> (Vec<usize>, Vec<f32>) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/dinov2/small_reference.safetensors");
        let bytes = std::fs::read(path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let tensor = tensors.tensor(name).unwrap();
        assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
        let values = tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        (tensor.shape().to_vec(), values)
    }

    fn relative_rms(actual: &[f32], expected: &[f32]) -> f64 {
        assert_eq!(actual.len(), expected.len(), "relative RMS element count");
        let (error, reference) = actual.iter().zip(expected).fold(
            (0.0_f64, 0.0_f64),
            |(error, reference), (actual, expected)| {
                let delta = f64::from(*actual) - f64::from(*expected);
                (
                    error + delta * delta,
                    reference + f64::from(*expected) * f64::from(*expected),
                )
            },
        );
        (error / reference.max(f64::MIN_POSITIVE)).sqrt()
    }

    #[test]
    fn processor_reads_hf_resize_crop_and_normalization() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            r#"{
                "do_resize": true,
                "size": {"shortest_edge": 4},
                "resample": 3,
                "do_center_crop": true,
                "crop_size": {"height": 2, "width": 2},
                "do_rescale": true,
                "rescale_factor": 0.00392156862745098,
                "do_normalize": true,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "do_convert_rgb": true
            }"#,
        )
        .unwrap();
        let processor = Dinov2ImageProcessor::from_pretrained(dir.path()).unwrap();
        let image =
            DynamicImage::ImageRgb8(RgbImage::from_raw(2, 4, vec![255; 2 * 4 * 3]).unwrap());
        let pixels = processor.preprocess(&image).unwrap();
        assert_eq!(mlxcel_core::array_shape(&pixels), [1, 2, 2, 3]);
        assert!(
            mlxcel_core::utils::array_to_vec_f32(&pixels)
                .iter()
                .all(|value| (*value - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn processor_rejects_checkpoint_driven_allocation_amplification() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            r#"{
                "size": {"shortest_edge": 256},
                "crop_size": {"height": 5000, "width": 5000},
                "resample": 3
            }"#,
        )
        .unwrap();
        let error = Dinov2ImageProcessor::from_pretrained(dir.path()).unwrap_err();
        assert!(error.to_string().contains("crop exceeds the safety limit"));
    }

    #[test]
    fn real_hf_processor_and_predictor_match_reference() {
        let Ok(checkpoint) = std::env::var("MLXCEL_TEST_DINOV2_SMALL_DIR") else {
            eprintln!("skipping DINOv2 predictor parity: MLXCEL_TEST_DINOV2_SMALL_DIR is unset");
            return;
        };
        let _device_lock = mlxcel_core::streams::lock_default_device();
        let _cpu_device = mlxcel_core::streams::DefaultDeviceGuard::cpu();
        let source = (0..224 * 224 * 3)
            .map(|index| (index % 256) as u8)
            .collect::<Vec<_>>();
        let image = DynamicImage::ImageRgb8(RgbImage::from_raw(224, 224, source).unwrap());
        let processor = Dinov2ImageProcessor::from_pretrained(Path::new(&checkpoint)).unwrap();
        let processed = processor.preprocess(&image).unwrap();
        let (expected_shape, expected_pixels) = fixture_tensor("pixel_values_nhwc");
        assert_eq!(
            mlxcel_core::array_shape(&processed),
            expected_shape
                .iter()
                .map(|dimension| *dimension as i32)
                .collect::<Vec<_>>()
        );
        let actual_pixels = mlxcel_core::utils::array_to_vec_f32(&processed);
        let pixel_error = relative_rms(&actual_pixels, &expected_pixels);
        eprintln!("DINOv2 HF processor relative RMS: {pixel_error:.9e}");
        assert!(pixel_error < 1e-3, "processor relative RMS {pixel_error}");

        let predictor = Dinov2Predictor::load(Path::new(&checkpoint)).unwrap();
        let opts = Map::from_iter([("task".into(), Value::String("features".into()))]);
        let output = predictor.predict(&DenseInput::Image(image), &opts).unwrap();
        for name in ["last_hidden_state", "pooler_output", "hidden_patch_tokens"] {
            let actual_tensor = &output.tensors[name];
            let (expected_shape, expected) = fixture_tensor(name);
            assert_eq!(
                mlxcel_core::array_shape(actual_tensor),
                expected_shape
                    .iter()
                    .map(|dimension| *dimension as i32)
                    .collect::<Vec<_>>()
            );
            let actual = mlxcel_core::utils::array_to_vec_f32(actual_tensor);
            let error = relative_rms(&actual, &expected);
            eprintln!("DINOv2 predictor {name} relative RMS: {error:.9e}");
            assert!(error < 1e-3, "{name} relative RMS {error}");
        }
    }
}
