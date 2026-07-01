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

//! Mllama (Llama 3.2 Vision) image processor.
//!
//! Port of HuggingFace `MllamaImageProcessor` (the processor the mlx-vlm
//! reference delegates to via `AutoImageProcessor`). For each image it:
//!
//! 1. Picks the optimal tile arrangement `(tiles_h, tiles_w)` from the
//!    supported aspect ratios so at most `max_image_tiles` tiles are used and
//!    the image is upscaled as little as possible.
//! 2. Resizes the image to fit that canvas (preserving aspect ratio) and pads
//!    to `tiles_h * tile_size` by `tiles_w * tile_size`.
//! 3. Splits the canvas into `tiles_h * tiles_w` non-overlapping tiles, then
//!    rescales to `[0, 1]` and normalizes with the CLIP mean/std.
//! 4. Pads the tile axis up to `max_image_tiles` with zero tiles.
//!
//! Outputs, packed for the vision tower's `[B, num_media, num_tiles, C, H, W]`
//! contract:
//! - `pixel_values`: `[1, num_media, max_image_tiles, 3, tile, tile]`
//! - `aspect_ratio_ids`: int32 `[1, num_media]` (1-based index into the
//!   supported aspect ratios)
//! - `aspect_ratio_mask`: int32 `[1, num_media, max_image_tiles]`
//! - `num_tiles`: per-image real tile counts

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// CLIP normalization constants (Llama-3.2-Vision `preprocessor_config.json`).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Preprocessed multi-image inputs for the mllama vision tower.
pub struct MllamaImageInputs {
    pub pixel_values: UniquePtr<MlxArray>,
    pub aspect_ratio_ids: UniquePtr<MlxArray>,
    pub aspect_ratio_mask: UniquePtr<MlxArray>,
    pub num_tiles: Vec<usize>,
}

pub struct MllamaImageProcessor {
    pub tile_size: usize,
    pub max_image_tiles: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl MllamaImageProcessor {
    pub fn new(tile_size: usize, max_image_tiles: usize) -> Self {
        Self {
            tile_size,
            max_image_tiles,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// All supported `(tiles_h, tiles_w)` arrangements with
    /// `tiles_h * tiles_w <= max_image_tiles`, in HuggingFace enumeration order
    /// (height outer). The 1-based position is the `aspect_ratio_id`.
    pub fn supported_aspect_ratios(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for h in 1..=self.max_image_tiles {
            for w in 1..=self.max_image_tiles {
                if h * w <= self.max_image_tiles {
                    out.push((h, w));
                }
            }
        }
        out
    }

    /// Port of `get_optimal_tiled_canvas`. Returns `(tiles_h, tiles_w)`.
    ///
    /// Chooses the arrangement whose canvas best fits the image: prefer the
    /// smallest upscale when the image fits inside some canvas, otherwise the
    /// largest downscale; ties break toward fewer tiles.
    pub fn optimal_canvas(&self, image_h: usize, image_w: usize) -> (usize, usize) {
        let arrangements = self.supported_aspect_ratios();
        let tile = self.tile_size as f64;
        let (ih, iw) = (image_h.max(1) as f64, image_w.max(1) as f64);

        let scales: Vec<f64> = arrangements
            .iter()
            .map(|&(h, w)| {
                let scale_h = (h as f64 * tile) / ih;
                let scale_w = (w as f64 * tile) / iw;
                scale_h.min(scale_w)
            })
            .collect();

        let upscaling: Vec<f64> = scales.iter().copied().filter(|&s| s >= 1.0).collect();
        let selected_scale = if !upscaling.is_empty() {
            upscaling.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            scales.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };

        arrangements
            .iter()
            .zip(scales.iter())
            .filter(|(_, s)| (**s - selected_scale).abs() < f64::EPSILON)
            .map(|(&arr, _)| arr)
            .min_by_key(|&(h, w)| h * w)
            .unwrap_or((1, 1))
    }

    /// 1-based aspect-ratio id for the arrangement (its position in the
    /// supported list), or `0` when absent.
    pub fn aspect_ratio_id(&self, tiles_h: usize, tiles_w: usize) -> i32 {
        self.supported_aspect_ratios()
            .iter()
            .position(|&(h, w)| h == tiles_h && w == tiles_w)
            .map(|idx| idx as i32 + 1)
            .unwrap_or(0)
    }

    /// Resize (aspect-preserving) into the canvas and pad to
    /// `(tiles_h * tile, tiles_w * tile)`; then split row-major into tiles.
    fn split_into_tiles(
        &self,
        image: &image::DynamicImage,
        tiles_h: usize,
        tiles_w: usize,
    ) -> Vec<image::RgbImage> {
        let rgb = image.to_rgb8();
        let (iw, ih) = (rgb.width().max(1), rgb.height().max(1));
        let tile = self.tile_size as u32;
        let canvas_h = tiles_h as u32 * tile;
        let canvas_w = tiles_w as u32 * tile;

        // Aspect-preserving resize to fit the canvas.
        let scale = (canvas_w as f64 / iw as f64).min(canvas_h as f64 / ih as f64);
        let new_w = ((iw as f64 * scale).round() as u32).clamp(1, canvas_w);
        let new_h = ((ih as f64 * scale).round() as u32).clamp(1, canvas_h);
        let resized = image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(new_w, new_h, FilterType::CatmullRom)
            .to_rgb8();

        // Pad (bottom/right) to the full canvas with zeros.
        let mut canvas = image::RgbImage::new(canvas_w, canvas_h);
        for y in 0..new_h {
            for x in 0..new_w {
                canvas.put_pixel(x, y, *resized.get_pixel(x, y));
            }
        }

        // Row-major split: tile (row h, col w).
        let mut tiles = Vec::with_capacity(tiles_h * tiles_w);
        for th in 0..tiles_h as u32 {
            for tw in 0..tiles_w as u32 {
                let view = image::imageops::crop_imm(&canvas, tw * tile, th * tile, tile, tile);
                tiles.push(view.to_image());
            }
        }
        tiles
    }

    /// Append a tile as normalized channels-first `[C, H, W]` f32 values.
    fn append_normalized_chw(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        for c in 0..3 {
            for y in 0..self.tile_size {
                for x in 0..self.tile_size {
                    let value = tile.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                    out.push((value - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    /// Append zero-filled channels-first `[C, H, W]` values for a padding tile.
    fn append_zero_tile(&self, out: &mut Vec<f32>) {
        out.extend(std::iter::repeat_n(
            0.0f32,
            3 * self.tile_size * self.tile_size,
        ));
    }

    /// Full preprocessing for a batch of images (one prompt, `num_media`
    /// images).
    pub fn process(&self, images: &[image::DynamicImage]) -> MllamaImageInputs {
        let tile = self.tile_size as i32;
        let max_tiles = self.max_image_tiles;
        let num_media = images.len();

        let mut pixels: Vec<f32> = Vec::new();
        let mut ratio_ids: Vec<i32> = Vec::with_capacity(num_media);
        let mut ratio_mask: Vec<i32> = Vec::with_capacity(num_media * max_tiles);
        let mut num_tiles: Vec<usize> = Vec::with_capacity(num_media);

        for image in images {
            let (iw, ih) = {
                let rgb = image.to_rgb8();
                (rgb.width() as usize, rgb.height() as usize)
            };
            let (tiles_h, tiles_w) = self.optimal_canvas(ih, iw);
            let tiles = self.split_into_tiles(image, tiles_h, tiles_w);
            let used = tiles.len().min(max_tiles);

            for tile_img in tiles.iter().take(max_tiles) {
                self.append_normalized_chw(tile_img, &mut pixels);
            }
            for _ in used..max_tiles {
                self.append_zero_tile(&mut pixels);
            }

            ratio_ids.push(self.aspect_ratio_id(tiles_h, tiles_w));
            for i in 0..max_tiles {
                ratio_mask.push(if i < used { 1 } else { 0 });
            }
            num_tiles.push(used);
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &pixels,
            &[1, num_media as i32, max_tiles as i32, 3, tile, tile],
        );
        let aspect_ratio_ids = mlxcel_core::from_slice_i32(&ratio_ids, &[1, num_media as i32]);
        let aspect_ratio_mask =
            mlxcel_core::from_slice_i32(&ratio_mask, &[1, num_media as i32, max_tiles as i32]);

        MllamaImageInputs {
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            num_tiles,
        }
    }
}

impl ImageProcessor for MllamaImageProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        self.process(images).pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> MllamaImageProcessor {
        MllamaImageProcessor::new(560, 4)
    }

    #[test]
    fn supported_aspect_ratios_match_reference_order() {
        // Reference `supported_aspect_ratios` (config.py) in [h, w] order.
        let expected = vec![
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 2),
            (3, 1),
            (4, 1),
        ];
        assert_eq!(processor().supported_aspect_ratios(), expected);
    }

    #[test]
    fn aspect_ratio_id_is_one_based() {
        let p = processor();
        assert_eq!(p.aspect_ratio_id(1, 1), 1);
        assert_eq!(p.aspect_ratio_id(2, 2), 6);
        assert_eq!(p.aspect_ratio_id(4, 1), 8);
        assert_eq!(p.aspect_ratio_id(9, 9), 0);
    }

    #[test]
    fn optimal_canvas_prefers_square_for_square_image() {
        let p = processor();
        // A 560x560 image fits (1,1) exactly (scale 1.0), the minimal upscale.
        assert_eq!(p.optimal_canvas(560, 560), (1, 1));
        // A very wide image should pick a 1x4 (or 1xN) arrangement.
        let (h, w) = p.optimal_canvas(300, 2000);
        assert_eq!(h, 1);
        assert!(w >= 2);
    }

    #[test]
    fn process_packs_padded_tiles_and_masks() {
        let p = processor();
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(560, 560));
        let out = p.process(std::slice::from_ref(&img));
        mlxcel_core::eval(&out.pixel_values);

        assert_eq!(
            mlxcel_core::array_shape(&out.pixel_values),
            vec![1, 1, 4, 3, 560, 560]
        );
        assert_eq!(mlxcel_core::array_shape(&out.aspect_ratio_ids), vec![1, 1]);
        assert_eq!(
            mlxcel_core::array_shape(&out.aspect_ratio_mask),
            vec![1, 1, 4]
        );
        // A square image uses exactly one tile.
        assert_eq!(out.num_tiles, vec![1]);
    }
}
