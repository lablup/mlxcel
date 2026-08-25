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

//! LFM2-VL image processor (smart resize, splitting, and patch packing).
//!
//! Port of the LFM2-VL resize path, including the high-resolution
//! `do_image_splitting` mode. Small images are smart-resized so their
//! post-downsample token count lands in `[min_image_tokens, max_image_tokens]`.
//! Large images are resized onto a row-major grid of `tile_size` square tiles
//! chosen by aspect ratio; each tile is packed as its own view, with an
//! optional smart-resized thumbnail appended last. Each packed view keeps its
//! own `(h, w)` patch grid, and the output concatenates every view's patches
//! in prompt order (the KimiVL native-resolution pattern).
//!
//! Used by: LFM2-VL (`lfm2_vl` / `lfm2-vl`) VLM.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};

const SIGLIP_MEAN: f32 = 0.5;
const SIGLIP_STD: f32 = 0.5;

/// LFM2-VL high-resolution tiling policy from `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lfm2VlTilingPolicy {
    pub do_image_splitting: bool,
    pub tile_size: usize,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub max_pixels_tolerance: f32,
    pub use_thumbnail: bool,
}

impl Default for Lfm2VlTilingPolicy {
    fn default() -> Self {
        Self {
            do_image_splitting: true,
            tile_size: 512,
            min_tiles: 2,
            max_tiles: 10,
            max_pixels_tolerance: 2.0,
            use_thumbnail: false,
        }
    }
}

/// Prompt/vision layout for one logical input image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lfm2VlImageLayout {
    /// Per-view patch grids in row-major tile order; thumbnail, when present, is last.
    pub views: Vec<(i32, i32)>,
    /// Number of tile rows in the split grid; `1` for the single-view path.
    pub rows: usize,
    /// Number of tile columns in the split grid; `1` for the single-view path.
    pub cols: usize,
}

impl Lfm2VlImageLayout {
    pub fn is_tiled(&self) -> bool {
        self.rows > 1 || self.cols > 1
    }

    pub fn tile_count(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    pub fn has_thumbnail(&self) -> bool {
        self.is_tiled() && self.views.len() > self.tile_count()
    }
}

pub struct Lfm2VlProcessor {
    pub patch_size: usize,        // encoder_patch_size (16)
    pub downsample_factor: usize, // f (2)
    pub min_image_tokens: usize,  // 64
    pub max_image_tokens: usize,  // 256
    pub tiling: Lfm2VlTilingPolicy,
}

/// Return whether the reference processor takes the splitting path for an image.
pub fn is_image_too_large(
    height: u32,
    width: u32,
    patch_size: usize,
    downsample_factor: usize,
    max_image_tokens: usize,
    max_pixels_tolerance: f32,
) -> bool {
    let p = patch_size.max(1) as f64;
    let total = (patch_size.max(1) * downsample_factor.max(1)) as f64;
    let round_to = |v: f64| -> f64 { (v / total).round() * total };
    let h_bar = round_to(height as f64).max(p);
    let w_bar = round_to(width as f64).max(p);
    let limit = max_image_tokens.max(1) as f64
        * p
        * p
        * downsample_factor.max(1) as f64
        * downsample_factor.max(1) as f64
        * max_pixels_tolerance.max(0.0) as f64;
    h_bar * w_bar > limit
}

/// Candidate `(grid_w, grid_h)` tile ratios sorted by tile count.
pub fn target_tile_ratios(min_tiles: usize, max_tiles: usize) -> Vec<(usize, usize)> {
    let min_tiles = min_tiles.max(1);
    let max_tiles = max_tiles.max(min_tiles);
    let mut ratios = Vec::new();
    for grid_w in 1..=max_tiles {
        for grid_h in 1..=max_tiles {
            let tiles = grid_w * grid_h;
            if (min_tiles..=max_tiles).contains(&tiles) {
                ratios.push((grid_w, grid_h));
            }
        }
    }
    ratios.sort_by_key(|&(grid_w, grid_h)| (grid_w * grid_h, grid_w, grid_h));
    ratios
}

/// Pick the closest tile grid by aspect ratio, using the reference area tie-break.
pub fn find_closest_aspect_ratio(
    width: u32,
    height: u32,
    ratios: &[(usize, usize)],
    tile_size: usize,
) -> (usize, usize) {
    let aspect = width.max(1) as f64 / height.max(1) as f64;
    let image_area = width as f64 * height as f64;
    let tile_size = tile_size.max(1) as f64;
    let mut best = (1usize, 1usize);
    let mut best_diff = f64::INFINITY;
    for &(grid_w, grid_h) in ratios {
        let ratio = grid_w as f64 / grid_h.max(1) as f64;
        let diff = (aspect - ratio).abs();
        let area_tie_break = image_area > 0.5 * tile_size * tile_size * (grid_w * grid_h) as f64;
        if diff < best_diff || ((diff - best_diff).abs() <= f64::EPSILON && area_tie_break) {
            best = (grid_w, grid_h);
            best_diff = diff;
        }
    }
    best
}

/// Return `(grid_w, grid_h)` for an image under the configured splitting policy.
pub fn grid_layout(
    height: u32,
    width: u32,
    patch_size: usize,
    downsample_factor: usize,
    max_image_tokens: usize,
    tiling: Lfm2VlTilingPolicy,
) -> (usize, usize) {
    if !tiling.do_image_splitting
        || !is_image_too_large(
            height,
            width,
            patch_size,
            downsample_factor,
            max_image_tokens,
            tiling.max_pixels_tolerance,
        )
    {
        return (1, 1);
    }
    let ratios = target_tile_ratios(tiling.min_tiles, tiling.max_tiles);
    find_closest_aspect_ratio(width, height, &ratios, tiling.tile_size)
}

impl Lfm2VlProcessor {
    pub fn new(
        patch_size: usize,
        downsample_factor: usize,
        min_image_tokens: usize,
        max_image_tokens: usize,
        tiling: Lfm2VlTilingPolicy,
    ) -> Self {
        Self {
            patch_size: patch_size.max(1),
            downsample_factor: downsample_factor.max(1),
            min_image_tokens: min_image_tokens.max(1),
            max_image_tokens: max_image_tokens.max(1),
            tiling,
        }
    }

    /// Smart resize: return `(h1, w1)` both divisible by `total = P * f`, with the
    /// downsampled token count clamped to `[min_image_tokens, max_image_tokens]`.
    pub fn smart_resize(&self, h: u32, w: u32) -> (u32, u32) {
        let total = (self.patch_size * self.downsample_factor) as f64; // 32
        let p2f2 =
            (self.patch_size * self.patch_size * self.downsample_factor * self.downsample_factor)
                as f64;
        let min_pixels = self.min_image_tokens as f64 * p2f2;
        let max_pixels = self.max_image_tokens as f64 * p2f2;
        let (hf, wf) = (h as f64, w as f64);

        let round_to = |v: f64| -> f64 { (v / total).round() * total };
        let floor_to = |v: f64| -> f64 { (v / total).floor() * total };
        let ceil_to = |v: f64| -> f64 { (v / total).ceil() * total };

        let mut h1 = round_to(hf).max(total);
        let mut w1 = round_to(wf).max(total);
        if h1 * w1 > max_pixels {
            let beta = (hf * wf / max_pixels).sqrt();
            h1 = floor_to(hf / beta).max(total);
            w1 = floor_to(wf / beta).max(total);
        } else if h1 * w1 < min_pixels {
            let beta = (min_pixels / (hf * wf)).sqrt();
            h1 = ceil_to(hf * beta);
            w1 = ceil_to(wf * beta);
        }
        (h1 as u32, w1 as u32)
    }

    /// Produce the RGB views for one input image in the order consumed by the prompt.
    pub fn views_for_image(&self, image: &DynamicImage) -> (Vec<RgbImage>, usize, usize) {
        let (w, h) = (image.width(), image.height());
        let (grid_w, grid_h) = grid_layout(
            h,
            w,
            self.patch_size,
            self.downsample_factor,
            self.max_image_tokens,
            self.tiling,
        );
        if grid_w == 1 && grid_h == 1 {
            let (h1, w1) = self.smart_resize(h, w);
            return (
                vec![image.resize_exact(w1, h1, FilterType::Triangle).to_rgb8()],
                1,
                1,
            );
        }

        let tile_size = self.tiling.tile_size as u32;
        let resized = image
            .resize_exact(
                tile_size * grid_w as u32,
                tile_size * grid_h as u32,
                FilterType::Triangle,
            )
            .to_rgb8();
        let mut views = Vec::with_capacity(
            grid_w * grid_h + usize::from(self.tiling.use_thumbnail && grid_w * grid_h > 1),
        );
        for row in 0..grid_h as u32 {
            for col in 0..grid_w as u32 {
                let tile = image::imageops::crop_imm(
                    &resized,
                    col * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                )
                .to_image();
                views.push(tile);
            }
        }
        if self.tiling.use_thumbnail && grid_w * grid_h > 1 {
            let (h1, w1) = self.smart_resize(h, w);
            views.push(image.resize_exact(w1, h1, FilterType::Triangle).to_rgb8());
        }
        (views, grid_h, grid_w)
    }

    /// Normalize + pack one RGB view into `h_i*w_i` patch vectors of length
    /// `P*P*3`, appended to `out`. Returns the patch grid `(h_i, w_i)`.
    fn pack_rgb_image(&self, img: &RgbImage, out: &mut Vec<f32>) -> (i32, i32) {
        let p = self.patch_size as u32;
        let (grid_h, grid_w) = (img.height() / p, img.width() / p);
        for gr in 0..grid_h {
            for gc in 0..grid_w {
                for py in 0..p {
                    for px in 0..p {
                        let pixel = img.get_pixel(gc * p + px, gr * p + py);
                        for c in 0..3 {
                            let v = pixel[c] as f32 / 255.0;
                            out.push((v - SIGLIP_MEAN) / SIGLIP_STD);
                        }
                    }
                }
            }
        }
        (grid_h as i32, grid_w as i32)
    }

    /// Resize + normalize + pack one image into `h_i*w_i` patch vectors of length
    /// `P*P*3`, appended to `out`. Returns the patch grid `(h_i, w_i)`.
    #[cfg(test)]
    fn pack_image(&self, image: &DynamicImage, out: &mut Vec<f32>) -> (i32, i32) {
        let (views, _, _) = self.views_for_image(image);
        self.pack_rgb_image(&views[0], out)
    }

    /// Preprocess a batch of images. Returns the concatenated packed patches
    /// `[1, sum_i(h_i*w_i), P*P*3]` and one layout per logical input image.
    pub fn preprocess_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<Lfm2VlImageLayout>) {
        let patch_dim = (self.patch_size * self.patch_size * 3) as i32;
        let mut data: Vec<f32> = Vec::new();
        let mut layouts: Vec<Lfm2VlImageLayout> = Vec::with_capacity(images.len());
        for image in images {
            let (views, rows, cols) = self.views_for_image(image);
            let mut grids = Vec::with_capacity(views.len());
            for view in &views {
                grids.push(self.pack_rgb_image(view, &mut data));
            }
            layouts.push(Lfm2VlImageLayout {
                views: grids,
                rows,
                cols,
            });
        }
        let total_patches = layouts
            .iter()
            .flat_map(|layout| layout.views.iter())
            .map(|(h, w)| h * w)
            .sum::<i32>();
        let pixel_values =
            mlxcel_core::from_slice_f32(&data, &[1, total_patches.max(0), patch_dim]);
        (pixel_values, layouts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([255, 255, 255]),
        ))
    }

    fn processor() -> Lfm2VlProcessor {
        Lfm2VlProcessor::new(16, 2, 64, 256, Lfm2VlTilingPolicy::default())
    }

    #[test]
    fn smart_resize_divisible_by_total_and_clamped() {
        let p = processor();
        // Small image scaled up so token count >= min (64).
        let (h1, w1) = p.smart_resize(20, 20);
        assert_eq!(h1 % 32, 0);
        assert_eq!(w1 % 32, 0);
        let tokens = (h1 / 32) * (w1 / 32);
        assert!(tokens >= 64, "tokens {tokens} below min");
        // Large image scaled down so token count <= max (256).
        let (h2, w2) = p.smart_resize(4000, 4000);
        assert_eq!(h2 % 32, 0);
        assert_eq!(w2 % 32, 0);
        assert!((h2 / 32) * (w2 / 32) <= 256, "over max tokens");
    }

    #[test]
    fn target_tile_ratios_for_2_to_10() {
        let ratios = target_tile_ratios(2, 10);
        for expected in [(1, 2), (2, 1), (3, 3), (2, 5), (5, 2)] {
            assert!(ratios.contains(&expected), "missing ratio {expected:?}");
        }
        assert!(ratios.iter().all(|(w, h)| (2..=10).contains(&(w * h))));
        assert!(!ratios.contains(&(1, 1)));
    }

    #[test]
    fn closest_ratio_prefers_area_on_ties() {
        let ratios = target_tile_ratios(2, 10);
        assert_eq!(find_closest_aspect_ratio(1024, 1024, &ratios, 512), (2, 2));
        assert_eq!(find_closest_aspect_ratio(2048, 2048, &ratios, 512), (3, 3));
    }

    #[test]
    fn large_image_splits_into_row_major_tiles_plus_optional_thumbnail() {
        let mut policy = Lfm2VlTilingPolicy {
            use_thumbnail: true,
            ..Lfm2VlTilingPolicy::default()
        };
        let p = Lfm2VlProcessor::new(16, 2, 64, 256, policy);
        let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(2048, 1024)));
        assert_eq!(layouts.len(), 1);
        assert_eq!((layouts[0].cols, layouts[0].rows), (4, 2));
        assert_eq!(layouts[0].views.len(), 9);
        assert_eq!(&layouts[0].views[..8], &[(32, 32); 8]);
        assert_eq!(layouts[0].views[8], {
            let (h, w) = p.smart_resize(1024, 2048);
            ((h / 16) as i32, (w / 16) as i32)
        });
        assert!(layouts[0].has_thumbnail());

        policy.use_thumbnail = false;
        let p = Lfm2VlProcessor::new(16, 2, 64, 256, policy);
        let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(2048, 1024)));
        assert_eq!((layouts[0].cols, layouts[0].rows), (4, 2));
        assert_eq!(layouts[0].views.len(), 8);
        assert!(!layouts[0].has_thumbnail());
    }

    #[test]
    fn small_image_keeps_single_view() {
        let p = processor();
        let image = solid(640, 480);
        let mut legacy = Vec::new();
        let (h1, w1) = p.smart_resize(480, 640);
        let legacy_img = image.resize_exact(w1, h1, FilterType::Triangle).to_rgb8();
        let legacy_grid = p.pack_rgb_image(&legacy_img, &mut legacy);

        let mut current = Vec::new();
        let current_grid = p.pack_image(&image, &mut current);
        assert_eq!(current_grid, legacy_grid);
        assert_eq!(current, legacy);

        let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&image));
        assert_eq!(layouts[0].views, vec![legacy_grid]);
        assert_eq!((layouts[0].rows, layouts[0].cols), (1, 1));
    }

    #[test]
    fn packs_patch_dim_and_grid() {
        let p = processor();
        let (pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
        assert_eq!(layouts.len(), 1);
        let (gh, gw) = layouts[0].views[0];
        let shape = mlxcel_core::array_shape(&pixels);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], gh * gw);
        assert_eq!(shape[2], 16 * 16 * 3);
    }

    #[test]
    fn white_pixel_normalizes_to_one() {
        let p = processor();
        let (pixels, _) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&first);
        assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
    }
}
