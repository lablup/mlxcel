// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! RT-DETRv2 image preprocessing.
//!
//! Port of `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/processing_rt_detr_v2.py`:
//! resize to `image_size` (bilinear), rescale by `rescale_factor` (1/255), and
//! optionally normalize with `image_mean` / `image_std`. The default
//! RT-DETRv2 preprocessor does NOT normalize (`do_normalize: false`); silently
//! adding mean/std subtraction is the classic way to get subtly-wrong outputs,
//! so the flag is read from `preprocessor_config.json` and defaulted to false.

use std::path::Path;

use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

const DEFAULT_IMAGE_SIZE: usize = 640;
const DEFAULT_RESCALE: f32 = 1.0 / 255.0;

/// Preprocessing configuration parsed from `preprocessor_config.json`.
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub image_size: usize,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            image_size: DEFAULT_IMAGE_SIZE,
            rescale_factor: DEFAULT_RESCALE,
            do_normalize: false,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawSize {
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    shortest_edge: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessor {
    #[serde(default)]
    size: Option<RawSize>,
    #[serde(default)]
    rescale_factor: Option<f32>,
    #[serde(default)]
    do_normalize: Option<bool>,
    #[serde(default)]
    image_mean: Option<Vec<f32>>,
    #[serde(default)]
    image_std: Option<Vec<f32>>,
}

impl ProcessorConfig {
    /// Build a processor config from a model directory. Reads
    /// `preprocessor_config.json` when present; otherwise falls back to
    /// `config.json`'s `image_size` for the resize target.
    pub fn from_pretrained<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();
        let mut cfg = ProcessorConfig::default();

        let preproc_path = dir.join("preprocessor_config.json");
        if preproc_path.is_file() {
            let raw_str = std::fs::read_to_string(&preproc_path)
                .map_err(|e| format!("failed to read {}: {e}", preproc_path.display()))?;
            let pp: RawPreprocessor = serde_json::from_str(&raw_str)
                .map_err(|e| format!("preprocessor_config parse error: {e}"))?;
            if let Some(size) = pp.size
                && let Some(h) = size.height.or(size.shortest_edge).or(size.width)
            {
                cfg.image_size = h;
            }
            if let Some(r) = pp.rescale_factor {
                cfg.rescale_factor = r;
            }
            if let Some(n) = pp.do_normalize {
                cfg.do_normalize = n;
            }
            if let Some(m) = pp.image_mean
                && m.len() == 3
            {
                cfg.image_mean = [m[0], m[1], m[2]];
            }
            if let Some(s) = pp.image_std
                && s.len() == 3
            {
                cfg.image_std = [s[0], s[1], s[2]];
            }
        } else {
            let config_path = dir.join("config.json");
            if config_path.is_file()
                && let Ok(s) = std::fs::read_to_string(&config_path)
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(&s)
                && let Some(sz) = v.get("image_size").and_then(|x| x.as_u64())
            {
                cfg.image_size = sz as usize;
            }
        }
        Ok(cfg)
    }
}

/// Output of preprocessing one image.
pub struct ProcessedImage {
    /// (1, image_size, image_size, 3) NHWC in [0, 1] (or normalized).
    pub pixel_values: UniquePtr<MlxArray>,
    /// Original `(width, height)` for rescaling boxes back to pixel coords.
    pub original_size: (u32, u32),
}

/// RT-DETRv2 image preprocessor.
pub struct RtDetrV2Processor {
    config: ProcessorConfig,
}

impl RtDetrV2Processor {
    pub fn new(config: ProcessorConfig) -> Self {
        Self { config }
    }

    pub fn from_pretrained<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        Ok(Self::new(ProcessorConfig::from_pretrained(dir)?))
    }

    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    /// Load and preprocess an image file.
    pub fn process_path<P: AsRef<Path>>(&self, path: P) -> Result<ProcessedImage, String> {
        let img = image::open(path.as_ref())
            .map_err(|e| format!("failed to open image {}: {e}", path.as_ref().display()))?;
        Ok(self.process_image(&img))
    }

    /// Preprocess a decoded image.
    pub fn process_image(&self, image: &image::DynamicImage) -> ProcessedImage {
        let rgb = image.to_rgb8();
        let original_size = (rgb.width(), rgb.height());

        let size = self.config.image_size as u32;
        // PIL `Image.Resampling.BILINEAR` == triangle (linear) kernel.
        let resized =
            image::imageops::resize(&rgb, size, size, image::imageops::FilterType::Triangle);

        let n = (size * size) as usize;
        let mut data = vec![0f32; n * 3];
        let rescale = self.config.rescale_factor;
        let do_norm = self.config.do_normalize;
        let mean = self.config.image_mean;
        let std = self.config.image_std;

        for (i, px) in resized.pixels().enumerate() {
            let base = i * 3;
            for c in 0..3 {
                let mut v = px[c] as f32 * rescale;
                if do_norm {
                    v = (v - mean[c]) / std[c];
                }
                data[base + c] = v;
            }
        }

        let pixel_values = mlxcel_core::from_slice_f32(&data, &[1, size as i32, size as i32, 3]);
        ProcessedImage {
            pixel_values,
            original_size,
        }
    }
}
