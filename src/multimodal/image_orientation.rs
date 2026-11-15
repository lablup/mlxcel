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

//! Upright pictures: the EXIF orientation tag, applied when a picture is
//! decoded.
//!
//! A camera stores the sensor's rows and records how the device was held in
//! the EXIF `Orientation` tag, and a viewer turns the picture by it. mlx-vlm's
//! and transformers' `load_image` both do that (`PIL.ImageOps.exif_transpose`),
//! so the models here were measured on upright pictures. The `image` crate
//! decodes the stored pixels and leaves the tag to the caller, so without this
//! step a portrait photo from a phone reaches the model lying on its side.
//!
//! JPEG, PNG (`eXIf`) and WebP carry the tag. A format without EXIF, a missing
//! tag and metadata that cannot be read all mean "leave the picture as it is":
//! an unreadable tag is no reason to refuse a picture that decodes.

use std::io::Cursor;
use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, ImageResult};

/// The transform that turns the picture in `bytes` upright, from its EXIF
/// orientation tag. Only the header and the metadata are parsed.
///
/// Used by: [`turn_upright`], [`upright_dimensions`].
#[must_use]
pub fn exif_orientation(bytes: &[u8]) -> Orientation {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|reader| reader.into_decoder().ok())
        .and_then(|mut decoder| decoder.orientation().ok())
        .unwrap_or(Orientation::NoTransforms)
}

/// Whether `orientation` turns the picture by a quarter, so that width and
/// height trade places.
#[must_use]
pub fn swaps_dimensions(orientation: Orientation) -> bool {
    matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    )
}

/// Turn `image`, which was decoded from `bytes`, upright.
///
/// Used by: the request decode in `server::model_provider::model_worker`,
/// which every server family and `mlxcel generate` share.
pub fn turn_upright(image: &mut DynamicImage, bytes: &[u8]) {
    image.apply_orientation(exif_orientation(bytes));
}

/// `(width, height)` of the upright picture, from the stored dimensions in
/// its header.
///
/// Used by: the Kimi K3 prompt builder, which sizes an image prompt from the
/// header before the worker decodes the pixels.
#[must_use]
pub fn upright_dimensions(bytes: &[u8], (width, height): (u32, u32)) -> (u32, u32) {
    if swaps_dimensions(exif_orientation(bytes)) {
        (height, width)
    } else {
        (width, height)
    }
}

/// Decode a picture held in memory and turn it upright. The format is guessed
/// from the content, as `image::load_from_memory` does.
///
/// Used by: `mlxcel embed` and `mlxcel rerank`.
pub fn load_upright_from_memory(bytes: &[u8]) -> ImageResult<DynamicImage> {
    let mut image = image::load_from_memory(bytes)?;
    turn_upright(&mut image, bytes);
    Ok(image)
}

/// Open a picture file and turn it upright. The format comes from the file
/// extension, as with `image::open`.
///
/// Used by: the reference pictures of `mlxcel generate-image`, the diffusion
/// path of `mlxcel generate`, and the YOLO11 and RT-DETRv2 predictors.
pub fn open_upright(path: &Path) -> ImageResult<DynamicImage> {
    let mut image = image::open(path)?;
    let orientation = ImageReader::open(path)
        .ok()
        .and_then(|reader| reader.into_decoder().ok())
        .and_then(|mut decoder| decoder.orientation().ok())
        .unwrap_or(Orientation::NoTransforms);
    image.apply_orientation(orientation);
    Ok(image)
}

#[cfg(test)]
#[path = "image_orientation_test_pictures.rs"]
pub(crate) mod test_pictures;

#[cfg(test)]
#[path = "image_orientation_tests.rs"]
mod tests;
