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

//! Inkling 2 image processor: no resize, 40x40 row-major tiles.
//!
//! Derived from mlx-vlm `Inkling2ImageProcessor`
//! (`mlx_vlm/models/inkling2/processing_inkling2.py`, MIT). Transparent images
//! are composited onto an 8-px checkerboard of 204 / 153 with
//! `(rgb * a + bg * (255 - a) + 127) // 255`; the canvas is padded with -1
//! before the `x / 255 - 0.5) / 0.5` normalization, as upstream.

use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ImageProcessor;

pub const TILE: usize = 40;
const CHANNELS: usize = 3;
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;
/// Longest edge accepted, so a hostile image cannot request an unbounded
/// tile canvas (upstream has no bound; 16384 px is 168k tiles at most).
const MAX_EDGE: usize = 16_384;

pub struct Inkling2ProcessedImages {
    /// `[tiles, 40, 40, 3]` f32.
    pub pixel_values: UniquePtr<MlxArray>,
    /// `(rows, cols)` per image; each contributes `rows * cols` tokens.
    pub grids: Vec<(usize, usize)>,
}

impl Inkling2ProcessedImages {
    pub fn tiles_per_image(&self) -> Vec<usize> {
        self.grids.iter().map(|(rows, cols)| rows * cols).collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Inkling2ImageProcessor;

/// RGB bytes of `image`, compositing alpha onto upstream's checkerboard.
pub(crate) fn rgb_pixels(image: &DynamicImage) -> (usize, usize, Vec<u8>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    if !image.color().has_alpha() {
        return (width, height, image.to_rgb8().into_raw());
    }
    let rgba = image.to_rgba8();
    let mut out = Vec::with_capacity(width * height * CHANNELS);
    for (index, pixel) in rgba.pixels().enumerate() {
        let (y, x) = (index / width, index % width);
        let background: u32 = if (y / 8 + x / 8).is_multiple_of(2) {
            204
        } else {
            153
        };
        let alpha = u32::from(pixel[3]);
        for channel in 0..CHANNELS {
            let value =
                (u32::from(pixel[channel]) * alpha + background * (255 - alpha) + 127) / 255;
            out.push(value as u8);
        }
    }
    (width, height, out)
}

impl Inkling2ImageProcessor {
    pub fn preprocess_refs(
        &self,
        images: &[&DynamicImage],
    ) -> Result<Inkling2ProcessedImages, String> {
        if images.is_empty() {
            return Err("Inkling 2 image preprocessing requires at least one image".into());
        }
        let rescale = 1.0_f32 / 255.0;
        let pad = (-rescale - MEAN) / STD;
        let mut grids = Vec::with_capacity(images.len());
        let mut values = Vec::new();
        for &image in images {
            let (width, height, pixels) = rgb_pixels(image);
            if width == 0 || height == 0 {
                return Err("Inkling 2 images must have positive width and height".into());
            }
            if width > MAX_EDGE || height > MAX_EDGE {
                return Err(format!(
                    "Inkling 2 image {width}x{height} exceeds the {MAX_EDGE}-px edge limit"
                ));
            }
            let (rows, cols) = (height.div_ceil(TILE), width.div_ceil(TILE));
            grids.push((rows, cols));
            let count = rows * cols * TILE * TILE * CHANNELS;
            values
                .try_reserve(count)
                .map_err(|error| format!("Failed to reserve the Inkling 2 tile buffer: {error}"))?;
            for row in 0..rows {
                for col in 0..cols {
                    for ty in 0..TILE {
                        let y = row * TILE + ty;
                        for tx in 0..TILE {
                            let x = col * TILE + tx;
                            if y < height && x < width {
                                let base = (y * width + x) * CHANNELS;
                                for channel in 0..CHANNELS {
                                    let value = f32::from(pixels[base + channel]);
                                    values.push((value * rescale - MEAN) / STD);
                                }
                            } else {
                                values.extend_from_slice(&[pad; CHANNELS]);
                            }
                        }
                    }
                }
            }
        }
        let tiles: usize = grids.iter().map(|(rows, cols)| rows * cols).sum();
        let tiles = i32::try_from(tiles)
            .map_err(|_| "Inkling 2 tile count exceeds the MLX i32 shape limit".to_string())?;
        Ok(Inkling2ProcessedImages {
            pixel_values: mlxcel_core::from_slice_f32(
                &values,
                &[tiles, TILE as i32, TILE as i32, 3],
            ),
            grids,
        })
    }
}

impl ImageProcessor for Inkling2ImageProcessor {
    fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        let refs: Vec<&DynamicImage> = images.iter().collect();
        self.preprocess_refs(&refs)
            .expect("Inkling 2 image preprocessing failed")
            .pixel_values
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage, Rgba, RgbaImage};
    use mlxcel_core::utils::array_to_vec_f32;

    use super::*;

    #[test]
    fn grid_is_ceil_division_with_negative_one_padding() {
        let mut image = RgbImage::new(41, 39);
        image.put_pixel(40, 0, image::Rgb([255, 0, 51]));
        let output = Inkling2ImageProcessor
            .preprocess_refs(&[&DynamicImage::ImageRgb8(image)])
            .unwrap();
        assert_eq!(output.grids, vec![(1, 2)]);
        assert_eq!(
            mlxcel_core::array_shape(&output.pixel_values),
            [2, 40, 40, 3]
        );
        let values = array_to_vec_f32(&output.pixel_values);
        let second = &values[40 * 40 * 3..];
        // Pixel (x=40, y=0) is the first pixel of the second tile.
        let expected = [1.0, -1.0, (51.0 / 255.0 - 0.5) / 0.5];
        for (got, want) in second[..3].iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // Row 39 of the image does not exist: padded with -1 before normalizing.
        let padded = second[39 * 40 * 3];
        assert!(
            (padded - (-1.0 / 255.0 - 0.5) / 0.5).abs() < 1e-7,
            "{padded}"
        );
    }

    #[test]
    fn alpha_composites_onto_the_checkerboard() {
        let mut image = RgbaImage::from_pixel(16, 1, Rgba([0, 0, 0, 0]));
        image.put_pixel(1, 0, Rgba([255, 255, 255, 128]));
        let (_, _, rgb) = rgb_pixels(&DynamicImage::ImageRgba8(image));
        // x = 0..8 is a 204 square, x = 8..16 a 153 square.
        assert_eq!(&rgb[..3], &[204, 204, 204]);
        assert_eq!(rgb[8 * 3], 153);
        // (255 * 128 + 204 * 127 + 127) // 255 = 230.
        assert_eq!(rgb[3], 230);
    }

    #[test]
    fn grayscale_is_repeated_to_rgb() {
        let image = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(1, 1, image::Luma([51])));
        let (_, _, rgb) = rgb_pixels(&image);
        assert_eq!(rgb, [51, 51, 51]);
    }
}
