// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Pillow-compatible RF-DETR image preprocessing.

use std::io::Read;
use std::path::Path;

use image::{DynamicImage, GenericImageView};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::dense::preprocess::resize_rgb8_bilinear_nhwc_f32;

use super::config::RfDetrConfig;

const MAX_SOURCE_DIMENSION: u32 = 16_384;
const MAX_SOURCE_PIXELS: u64 = 64 * 1024 * 1024;
const MAX_SELECT: usize = 1024;

pub struct ProcessedImage {
    pub pixel_values: UniquePtr<MlxArray>,
    pub original_size: (u32, u32),
}

#[derive(Debug, Deserialize)]
struct ProcessorFile {
    #[serde(default)]
    config: ImageConfig,
    #[serde(default)]
    post_process_config: PostProcessConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ImageConfig {
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PostProcessConfig {
    num_select: usize,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self { num_select: 300 }
    }
}

pub struct RfDetrProcessor {
    resolution: usize,
    mean: [f32; 3],
    std: [f32; 3],
    num_select: usize,
}

impl RfDetrProcessor {
    pub fn from_pretrained(directory: &Path, config: &RfDetrConfig) -> Result<Self, String> {
        let path = directory.join("preprocessor_config.json");
        let file = if path.exists() {
            let text = crate::vision::detection::read_bounded_metadata(&path)?;
            serde_json::from_str::<ProcessorFile>(&text)
                .map_err(|error| format!("failed to parse {}: {error}", path.display()))?
        } else {
            ProcessorFile {
                config: ImageConfig::default(),
                post_process_config: PostProcessConfig::default(),
            }
        };
        if file
            .config
            .image_mean
            .iter()
            .chain(file.config.image_std.iter())
            .any(|value| !value.is_finite())
            || file.config.image_std.iter().any(|value| *value <= 0.0)
        {
            return Err("RF-DETR processor mean/std must be finite with positive std".into());
        }
        if file.post_process_config.num_select == 0
            || file.post_process_config.num_select > MAX_SELECT
        {
            return Err(format!(
                "RF-DETR num_select must be in 1..={MAX_SELECT}, got {}",
                file.post_process_config.num_select
            ));
        }
        Ok(Self {
            resolution: config.resolution,
            mean: file.config.image_mean,
            std: file.config.image_std,
            num_select: file.post_process_config.num_select,
        })
    }

    pub fn num_select(&self) -> usize {
        self.num_select
    }

    pub fn preprocess_path(&self, path: &Path) -> Result<ProcessedImage, String> {
        let limits = crate::current_image_input_limits();
        let file = std::fs::File::open(path)
            .map_err(|error| format!("failed to open RF-DETR input {}: {error}", path.display()))?;
        let mut bytes = Vec::new();
        file.take(limits.max_payload_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("failed to read RF-DETR input {}: {error}", path.display()))?;
        if bytes.len() > limits.max_payload_bytes {
            return Err(format!(
                "RF-DETR input {} exceeds the {} byte payload limit",
                path.display(),
                limits.max_payload_bytes
            ));
        }
        let mut images =
            crate::decode_image_payloads_with_limits(&[bytes], limits).map_err(|error| {
                format!("failed to decode RF-DETR input {}: {error}", path.display())
            })?;
        let image = images
            .pop()
            .ok_or_else(|| format!("image decoder returned no image for {}", path.display()))?;
        self.preprocess(&image)
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<ProcessedImage, String> {
        let (width, height) = image.dimensions();
        if width == 0
            || height == 0
            || width > MAX_SOURCE_DIMENSION
            || height > MAX_SOURCE_DIMENSION
            || u64::from(width) * u64::from(height) > MAX_SOURCE_PIXELS
        {
            return Err(format!(
                "RF-DETR source image {width}x{height} exceeds safe limits"
            ));
        }
        let resolution = u32::try_from(self.resolution)
            .map_err(|_| "RF-DETR processor resolution exceeds u32")?;
        let pixels = resize_rgb8_bilinear_nhwc_f32(image, resolution, resolution)
            .map_err(|error| format!("RF-DETR Pillow bilinear resize failed: {error}"))?;
        let mean = mlxcel_core::from_slice_f32(&self.mean, &[1, 1, 1, 3]);
        let std = mlxcel_core::from_slice_f32(&self.std, &[1, 1, 1, 3]);
        let pixel_values = mlxcel_core::divide(&mlxcel_core::subtract(&pixels, &mean), &std);
        Ok(ProcessedImage {
            pixel_values,
            original_size: (height, width),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg_config() -> RfDetrConfig {
        RfDetrConfig::from_json_str(
            r#"{"model_type":"rf-detr","resolution":384,"dec_layers":4,
                "num_queries":100,"patch_size":12,"num_windows":2,
                "out_feature_indexes":[3,6,9,12],"segmentation":true,
                "positional_encoding_size":32}"#,
        )
        .unwrap()
    }

    #[test]
    fn released_seg_num_select_may_exceed_query_count() {
        let root = std::env::temp_dir().join(format!(
            "mlxcel-rfdetr-seg-processor-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("preprocessor_config.json"),
            r#"{"post_process_config":{"num_select":300}}"#,
        )
        .unwrap();
        let processor = RfDetrProcessor::from_pretrained(&root, &seg_config()).unwrap();
        assert_eq!(processor.num_select(), 300);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn path_preprocessing_rejects_oversized_header_before_pixel_allocation() {
        let root = std::env::temp_dir().join(format!(
            "mlxcel-rfdetr-image-admission-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("oversized.bmp");
        let mut bmp = vec![0u8; 54];
        bmp[0..2].copy_from_slice(b"BM");
        bmp[2..6].copy_from_slice(&54u32.to_le_bytes());
        bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&20_000i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&1i32.to_le_bytes());
        bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
        bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
        std::fs::write(&path, bmp).unwrap();
        let processor = RfDetrProcessor {
            resolution: 384,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            num_select: 300,
        };
        let error = processor
            .preprocess_path(&path)
            .err()
            .expect("oversized BMP must be rejected");
        assert!(
            error.contains("decode") || error.contains("limit"),
            "{error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
