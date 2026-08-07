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

//! Florence-2 image preprocessing.
//!
//! Florence-2 ships a stock `CLIPImageProcessor` configured for a fixed
//! 768x768 square, so the pipeline is: resize straight to 768x768, rescale to
//! [0, 1], normalize, emit NCHW.
//!
//! Two details are easy to get wrong and are the reason this is its own
//! processor rather than a reuse of an existing one:
//!
//! - The mean and standard deviation are **ImageNet** values, not the CLIP
//!   ones the class name suggests and not the SigLIP 0.5s. Reusing
//!   `PIXTRAL_IMAGE_MEAN` / `PIXTRAL_IMAGE_STD` here would shift every channel.
//! - `size` carries `height` and `width` rather than `shortest_edge`, and
//!   `do_center_crop` is false, so the resize does **not** preserve aspect
//!   ratio. A non-square image is squashed, not letterboxed or cropped, and
//!   the original extent is carried separately because Florence-2's location
//!   bins are relative to it.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py

use std::path::Path;

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::ImageProcessor;

/// PIL's `Image.BICUBIC`, the only `resample` value Florence-2 ships.
const PIL_BICUBIC: u32 = 3;

/// Upper bound accepted for either edge of the `size` target. Real Florence-2
/// exports ship 768 on both axes.
///
/// The cap exists because `size` is the only field in this file that becomes an
/// allocation: [`Florence2ImageProcessor::preprocess_with_sizes`] sizes its
/// host buffer as `batch * 3 * height * width` f32 before it looks at a single
/// pixel. An unbounded value out of a hostile `preprocessor_config.json` is a
/// multi-gigabyte allocation and an out-of-memory abort rather than an error
/// return, and a value past `usize::MAX / 12` wraps that product in a release
/// build, leaving a short buffer for a write loop that still runs to the
/// configured extent. Same reasoning, and the same shape, as `MAX_LAYERS` and
/// `MAX_POSITION_EMBEDDINGS` in `src/models/florence2/checkpoint.rs`.
const MAX_SIZE_EDGE: usize = 8192;

/// Preprocessing settings read from `preprocessor_config.json`.
///
/// Every field is read from the file rather than hard-coded: the checkpoint is
/// the authority, and a variant that retrains at a different resolution would
/// otherwise be silently misprocessed.
#[derive(Debug, Clone, PartialEq)]
pub struct Florence2ImageProcessor {
    pub height: usize,
    pub width: usize,
    pub do_resize: bool,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

/// Preprocessed pixels plus the sizes needed to interpret the answer.
pub struct Florence2PixelValues {
    /// `[batch, 3, height, width]` f32, NCHW.
    pub pixel_values: UniquePtr<MlxArray>,
    /// `(width, height)` of each input image *before* the resize, in the same
    /// order as the batch. Florence-2's `<loc_*>` bins are relative to this,
    /// so post-processing needs it to produce pixel coordinates.
    pub original_sizes: Vec<(u32, u32)>,
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
    do_resize: Option<bool>,
    #[serde(default)]
    do_rescale: Option<bool>,
    #[serde(default)]
    rescale_factor: Option<f32>,
    #[serde(default)]
    do_normalize: Option<bool>,
    #[serde(default)]
    do_center_crop: Option<bool>,
    #[serde(default)]
    image_mean: Option<Vec<f32>>,
    #[serde(default)]
    image_std: Option<Vec<f32>>,
    #[serde(default)]
    resample: Option<u32>,
}

impl Florence2ImageProcessor {
    /// Read `preprocessor_config.json` from a checkpoint directory.
    ///
    /// The file is required. Falling back to defaults would turn a missing or
    /// truncated checkpoint into subtly wrong pixels instead of an error, and
    /// there is no second source for the mean and standard deviation.
    pub fn from_pretrained(model_path: &Path) -> Result<Self, String> {
        let path = model_path.join("preprocessor_config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("Florence-2: failed to read {}: {e}", path.display()))?;
        let parsed: RawPreprocessor = serde_json::from_str(&raw)
            .map_err(|e| format!("Florence-2: failed to parse {}: {e}", path.display()))?;
        Self::from_raw(parsed)
    }

    fn from_raw(parsed: RawPreprocessor) -> Result<Self, String> {
        // Center cropping would change which part of the image the location
        // bins refer to, so an unexpected `true` has to stop the load rather
        // than be ignored.
        if parsed.do_center_crop == Some(true) {
            return Err(
                "Florence-2 preprocessor_config.json sets do_center_crop: true, which this processor does not implement".to_string(),
            );
        }
        if let Some(resample) = parsed.resample
            && resample != PIL_BICUBIC
        {
            return Err(format!(
                "Florence-2 preprocessor_config.json sets resample: {resample}, expected {PIL_BICUBIC} (bicubic)"
            ));
        }

        let size = parsed
            .size
            .ok_or_else(|| "Florence-2 preprocessor_config.json has no size".to_string())?;
        // `shortest_edge` is the aspect-preserving CLIP form. Florence-2 does
        // not use it, and treating it as a square target would be wrong, so it
        // is only accepted as a square shorthand when no explicit pair exists.
        let (height, width) = match (size.height, size.width) {
            (Some(h), Some(w)) => (h, w),
            _ => {
                let edge = size.shortest_edge.or(size.height).or(size.width).ok_or_else(|| {
                    "Florence-2 preprocessor_config.json size has neither height/width nor shortest_edge".to_string()
                })?;
                (edge, edge)
            }
        };
        if height == 0 || width == 0 {
            return Err(format!(
                "Florence-2 preprocessor_config.json size {width}x{height} must be positive"
            ));
        }
        if height > MAX_SIZE_EDGE || width > MAX_SIZE_EDGE {
            return Err(format!(
                "Florence-2 preprocessor_config.json size {width}x{height} exceeds the {MAX_SIZE_EDGE} per-edge limit"
            ));
        }

        let image_mean = triple(parsed.image_mean, "image_mean")?.unwrap_or([0.485, 0.456, 0.406]);
        let image_std = triple(parsed.image_std, "image_std")?.unwrap_or([0.229, 0.224, 0.225]);
        if image_std.contains(&0.0) {
            return Err(format!(
                "Florence-2 preprocessor_config.json image_std {image_std:?} must be non-zero"
            ));
        }

        Ok(Self {
            height,
            width,
            do_resize: parsed.do_resize.unwrap_or(true),
            do_rescale: parsed.do_rescale.unwrap_or(true),
            rescale_factor: parsed.rescale_factor.unwrap_or(1.0 / 255.0),
            do_normalize: parsed.do_normalize.unwrap_or(true),
            image_mean,
            image_std,
        })
    }

    /// Preprocess a batch, keeping each image's original `(width, height)`.
    ///
    /// This is the entry point the Florence-2 processor uses;
    /// [`ImageProcessor::preprocess`] discards the sizes and so cannot drive
    /// the spatial tasks on its own.
    pub fn preprocess_with_sizes(&self, images: &[DynamicImage]) -> Florence2PixelValues {
        let channels = 3usize;
        let (height, width) = (self.height, self.width);
        let mut data = vec![0.0f32; images.len() * channels * height * width];
        let mut original_sizes = Vec::with_capacity(images.len());

        for (index, image) in images.iter().enumerate() {
            original_sizes.push((image.width(), image.height()));
            // `resize_exact` ignores aspect ratio, which is what
            // CLIPImageProcessor does for an explicit height/width `size`.
            // `CatmullRom` is the `image` crate's cubic filter and the closest
            // analog of PIL's `BICUBIC` (both are the a = -0.5 cubic with
            // support scaled by the resampling ratio).
            let resized = if self.do_resize {
                image.resize_exact(width as u32, height as u32, FilterType::CatmullRom)
            } else {
                image.clone()
            };
            // Divergence from upstream: `resized` is resized in its source color
            // type (RGBA stays RGBA through `resize_exact`), and `to_rgb8` here
            // then drops the alpha channel unconditionally. Florence-2's
            // `preprocessor_config.json` ships `"do_convert_rgb": null`, so
            // upstream's `convert_to_rgb` step is skipped entirely and a
            // non-opaque RGBA source is resized and fed to the model with alpha
            // still attached instead of being flattened
            // (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py).
            // We flatten unconditionally instead because this pipeline only
            // consumes RGB pixel data downstream; that is the safer, deterministic
            // choice, at the cost of not bit-matching upstream on RGBA inputs with
            // partial transparency.
            let rgb = resized.to_rgb8();
            // A `do_resize: false` config can hand back a differently sized
            // buffer; clamp rather than index out of bounds.
            let copy_h = height.min(rgb.height() as usize);
            let copy_w = width.min(rgb.width() as usize);

            let plane = height * width;
            let base = index * channels * plane;
            for y in 0..copy_h {
                for x in 0..copy_w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for (c, channel) in pixel.0.iter().take(channels).enumerate() {
                        let mut value = f32::from(*channel);
                        if self.do_rescale {
                            value *= self.rescale_factor;
                        }
                        if self.do_normalize {
                            value = (value - self.image_mean[c]) / self.image_std[c];
                        }
                        data[base + c * plane + y * width + x] = value;
                    }
                }
            }
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &data,
            &[
                images.len() as i32,
                channels as i32,
                height as i32,
                width as i32,
            ],
        );
        Florence2PixelValues {
            pixel_values,
            original_sizes,
        }
    }
}

fn triple(values: Option<Vec<f32>>, field: &str) -> Result<Option<[f32; 3]>, String> {
    match values {
        None => Ok(None),
        Some(v) if v.len() == 3 => Ok(Some([v[0], v[1], v[2]])),
        Some(v) => Err(format!(
            "Florence-2 preprocessor_config.json {field} must have 3 entries, got {}",
            v.len()
        )),
    }
}

impl ImageProcessor for Florence2ImageProcessor {
    /// Emits f32 NCHW. The fused model casts to the vision tower's own dtype
    /// in `Florence2Model::encode_image`, so no cast is applied here.
    fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        self.preprocess_with_sizes(images).pixel_values
    }
}

#[cfg(test)]
#[path = "florence2_tests.rs"]
mod florence2_tests;
