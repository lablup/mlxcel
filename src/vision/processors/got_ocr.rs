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

//! GOT-OCR 2.0 image processor.
//!
//! Port of `GOTImageEvalProcessor` in `modeling_GOT.py`, which is three torch
//! transforms and nothing else:
//!
//! ```python
//! transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC)
//! transforms.ToTensor()
//! transforms.Normalize(mean, std)   # CLIP mean/std
//! ```
//!
//! `Resize` with a `(h, w)` pair resizes to exactly that shape, so the aspect
//! ratio is not preserved and nothing is padded or tiled. A whole page is
//! squashed into one 1024x1024 canvas and becomes exactly 256 feature rows,
//! which is why the prompt's `<imgpad>` run is a constant rather than something
//! derived from the image.
//!
//! Emits channels-last `(n, 1024, 1024, 3)` because
//! [`crate::vision::encoders::deepseekocr_sam::SamEncoder::forward`] takes NHWC
//! directly; the reference's NCHW tensor is transposed by its first conv
//! instead.

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// OpenAI CLIP normalization, the `mean` default in `GOTImageEvalProcessor`.
///
/// Kept at the reference's published decimals rather than the nearest f32
/// spelling, so the constant can be diffed against `modeling_GOT.py` by eye.
/// The same allow sits on the Pixtral and Step-3 copies of these numbers.
#[allow(clippy::excessive_precision)]
pub const GOT_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// OpenAI CLIP normalization, the `std` default in `GOTImageEvalProcessor`.
#[allow(clippy::excessive_precision)]
pub const GOT_IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// The tower's fixed input side; `_build_GOT_vision` hardcodes `image_size = 1024`.
pub const GOT_IMAGE_SIZE: u32 = 1024;

#[derive(Clone, Debug)]
pub struct GotOcrImageProcessor {
    pub image_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for GotOcrImageProcessor {
    fn default() -> Self {
        Self {
            image_size: GOT_IMAGE_SIZE,
            mean: GOT_IMAGE_MEAN,
            std: GOT_IMAGE_STD,
        }
    }
}

impl GotOcrImageProcessor {
    /// Preprocess one batch into channels-last `(n, size, size, 3)` f32.
    ///
    /// `CatmullRom` is the repository's standing stand-in for PIL/torch
    /// `BICUBIC` (see the InternVL, Falcon-OCR and Gemma 4 processors); both
    /// are the Keys cubic filter.
    pub fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        let side = self.image_size as usize;
        let mut data = Vec::with_capacity(images.len() * side * side * 3);
        for image in images {
            let resized = image
                .resize_exact(self.image_size, self.image_size, FilterType::CatmullRom)
                .to_rgb8();
            for pixel in resized.pixels() {
                for channel in 0..3 {
                    let value = pixel[channel] as f32 / 255.0;
                    data.push((value - self.mean[channel]) / self.std[channel]);
                }
            }
        }
        mlxcel_core::from_slice_f32(&data, &[images.len() as i32, side as i32, side as i32, 3])
    }
}

#[cfg(test)]
#[path = "got_ocr_tests.rs"]
mod tests;
