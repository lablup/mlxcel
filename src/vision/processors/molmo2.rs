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

//! Molmo2 Image Processor
//!
//! Multi-scale overlapping crop preprocessing pipeline:
//! 1. High-res: select_tiling → resize → normalize → overlapping crops with margins
//! 2. Low-res: resize to base_size → normalize
//! 3. Combine: [low_res, hi_res_crops...] → flatten to patches
//! 4. Build pooling indices for 2D attention pooling
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo2/processing.py

/// Molmo2 image processor configuration and state
pub struct Molmo2Processor {
    pub base_image_size: (usize, usize), // (height, width), default (378, 378)
    pub max_crops: usize,
    pub overlap_margins: (usize, usize), // (left, right), default (4, 4)
    pub patch_size: usize,               // 14
    pub pooling_size: (usize, usize),    // (pool_h, pool_w), default (2, 2)
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

/// Output from image preprocessing
pub struct Molmo2ProcessorOutput {
    pub pixel_values: Vec<f32>, // [n_crops, n_patches, patch_dim] flattened
    pub pixel_values_shape: [i32; 3], // [n_crops, n_patches, patch_dim]
    pub image_token_pooling: Vec<i32>, // [total_pooled, pool_size] flattened
    pub image_token_pooling_shape: [i32; 2], // [total_pooled, pool_size]
    pub image_grid: [i32; 4],   // [lo_h, lo_w, hi_h, hi_w]
    pub image_num_crops: i32,   // total number of crops
}

impl Molmo2Processor {
    pub fn new(
        max_crops: usize,
        overlap_margins: Option<(usize, usize)>,
        patch_size: Option<usize>,
        pooling_size: Option<(usize, usize)>,
        base_image_size: Option<(usize, usize)>,
    ) -> Self {
        Self {
            base_image_size: base_image_size.unwrap_or((378, 378)),
            max_crops,
            overlap_margins: overlap_margins.unwrap_or((4, 4)),
            patch_size: patch_size.unwrap_or(14),
            pooling_size: pooling_size.unwrap_or((2, 2)),
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    /// Preprocess a single image into patches and pooling indices
    pub fn preprocess_image(&self, image: &image::DynamicImage) -> Molmo2ProcessorOutput {
        let img = image.to_rgb8();
        let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

        let base_h = self.base_image_size.0;
        let base_w = self.base_image_size.1;
        let crop_patches_h = base_h / self.patch_size;
        let crop_patches_w = base_w / self.patch_size;

        // Build overlapping crops (high-res)
        let (crop_arr, patch_idx_full, _tiling) =
            self.build_overlapping_crops(&img, orig_h, orig_w);

        // Build pooling indices for high-res
        let pooling_idx = self.arange_for_pooling(&patch_idx_full);
        let hi_h = pooling_idx.len();
        let hi_w = if hi_h > 0 { pooling_idx[0].len() } else { 0 };
        let pool_size = self.pooling_size.0 * self.pooling_size.1;

        // Flatten hi-res pooling indices
        let mut hi_pooling_flat: Vec<Vec<i32>> = Vec::new();
        for row in &pooling_idx {
            for col in row {
                hi_pooling_flat.push(col.clone());
            }
        }

        // Build resized image (low-res)
        let (resize_arr, resize_idx) = self.build_resized_image(&img, orig_h, orig_w);

        // Build pooling indices for low-res
        let resize_pooling_idx = self.arange_for_pooling(&resize_idx);
        let lo_h = resize_pooling_idx.len();
        let lo_w = if lo_h > 0 {
            resize_pooling_idx[0].len()
        } else {
            0
        };

        // Flatten lo-res pooling indices
        let mut lo_pooling_flat: Vec<Vec<i32>> = Vec::new();
        for row in &resize_pooling_idx {
            for col in row {
                lo_pooling_flat.push(col.clone());
            }
        }

        // Offset high-res pooling indices by one crop's patches (low-res comes first)
        let offset = (crop_patches_h * crop_patches_w) as i32;
        for pool in &mut hi_pooling_flat {
            for val in pool.iter_mut() {
                if *val >= 0 {
                    *val += offset;
                }
            }
        }

        // Concatenate: low-res first, then high-res
        let mut all_pooling = lo_pooling_flat;
        all_pooling.extend(hi_pooling_flat);

        // Combine crops: [low_res, hi_res_crops...]
        let mut all_crops = vec![resize_arr];
        all_crops.extend(crop_arr);
        let n_crops = all_crops.len();

        // Convert crops to patches
        let patches = self.batch_pixels_to_patches(&all_crops);

        let n_patches = crop_patches_h * crop_patches_w;
        let patch_dim = self.patch_size * self.patch_size * 3;

        // Flatten pixel values
        let pixel_values: Vec<f32> = patches.into_iter().flatten().flatten().collect();

        // Flatten pooling indices
        let total_pooled = all_pooling.len();
        let image_token_pooling: Vec<i32> = all_pooling.into_iter().flatten().collect();

        Molmo2ProcessorOutput {
            pixel_values,
            pixel_values_shape: [n_crops as i32, n_patches as i32, patch_dim as i32],
            image_token_pooling,
            image_token_pooling_shape: [total_pooled as i32, pool_size as i32],
            image_grid: [lo_h as i32, lo_w as i32, hi_h as i32, hi_w as i32],
            image_num_crops: n_crops as i32,
        }
    }

    /// Generate image token string from image grid
    pub fn get_image_tokens(&self, image_grid: &[i32; 4]) -> String {
        let [lo_h, lo_w, hi_h, hi_w] = *image_grid;

        // High-res tokens
        let mut hi_res = String::from("<im_start>");
        for _ in 0..hi_h {
            for _ in 0..hi_w {
                hi_res.push_str("<im_patch>");
            }
            hi_res.push_str("<im_col>");
        }
        hi_res.push_str("<im_end>");

        // Low-res tokens
        let mut lo_res = String::from("<low_res_im_start>");
        for _ in 0..lo_h {
            for _ in 0..lo_w {
                lo_res.push_str("<im_patch>");
            }
            lo_res.push_str("<im_col>");
        }
        lo_res.push_str("<im_end>");

        // Low-res first, then high-res
        format!("{}{}", lo_res, hi_res)
    }

    // Internal helpers.
    fn select_tiling(&self, h: usize, w: usize) -> (usize, usize) {
        let patch_size = self.base_image_size.0; // crop size (378)
        let max_crops = self.max_crops;

        let mut tilings: Vec<(usize, usize)> = Vec::new();
        for i in 1..=max_crops {
            for j in 1..=max_crops {
                if i * j <= max_crops {
                    tilings.push((i, j));
                }
            }
        }
        tilings.sort_by_key(|&(a, b)| (a * b, a));

        let original = [h as f32, w as f32];
        let mut best_idx = 0;
        let mut best_scale = 0.0f32;
        let mut all_less_than_one = true;

        for (idx, &(ti, tj)) in tilings.iter().enumerate() {
            let res_h = (ti * patch_size) as f32;
            let res_w = (tj * patch_size) as f32;
            let scale = (res_h / original[0]).min(res_w / original[1]);
            if scale >= 1.0 {
                all_less_than_one = false;
            }
            if all_less_than_one {
                if scale > best_scale {
                    best_scale = scale;
                    best_idx = idx;
                }
            } else if scale >= 1.0 && (scale < best_scale || best_scale < 1.0) {
                best_scale = scale;
                best_idx = idx;
            }
        }

        tilings[best_idx]
    }

    fn build_overlapping_crops(
        &self,
        img: &image::RgbImage,
        orig_h: usize,
        orig_w: usize,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<i32>>, (usize, usize)) {
        let crop_size = self.base_image_size.0;
        let (left_margin, right_margin) = self.overlap_margins;
        let total_margin_pixels = self.patch_size * (right_margin + left_margin);
        let crop_patches = crop_size / self.patch_size;
        let crop_window_patches = crop_patches - (right_margin + left_margin);
        let crop_window_size = crop_window_patches * self.patch_size;

        let tiling = self.select_tiling(
            orig_h.saturating_sub(total_margin_pixels),
            orig_w.saturating_sub(total_margin_pixels),
        );

        let target_h = tiling.0 * crop_window_size + total_margin_pixels;
        let target_w = tiling.1 * crop_window_size + total_margin_pixels;

        // Resize
        let resized = image::imageops::resize(
            img,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Triangle,
        );

        // Normalize
        let src = self.normalize_image(&resized);

        let n_crops = tiling.0 * tiling.1;
        let mut crop_arr: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_crops);
        let crop_patch_h = crop_size / self.patch_size;
        let crop_patch_w = crop_size / self.patch_size;

        // Build full patch index array
        let full_h = tiling.0 * crop_window_patches + left_margin + right_margin;
        let full_w = tiling.1 * crop_window_patches + left_margin + right_margin;
        let mut patch_idx_full = vec![vec![0i32; full_w]; full_h];

        let mut on_crop = 0;
        for i in 0..tiling.0 {
            let y0 = i * crop_window_size;
            for j in 0..tiling.1 {
                let x0 = j * crop_window_size;

                // Extract crop [y0..y0+crop_size, x0..x0+crop_size, 3]
                let mut crop = vec![vec![0.0f32; 3]; crop_size * crop_size];
                for cy in 0..crop_size {
                    for cx in 0..crop_size {
                        let sy = y0 + cy;
                        let sx = x0 + cx;
                        if sy < target_h && sx < target_w {
                            let idx = cy * crop_size + cx;
                            let src_idx = sy * target_w + sx;
                            crop[idx] = src[src_idx].clone();
                        }
                    }
                }
                crop_arr.push(crop);

                // Build patch index for this crop
                let mut patch_idx = vec![vec![0i32; crop_patch_w]; crop_patch_h];
                for (py, row) in patch_idx.iter_mut().enumerate() {
                    for (px, cell) in row.iter_mut().enumerate() {
                        *cell =
                            (on_crop * crop_patch_h * crop_patch_w + py * crop_patch_w + px) as i32;
                    }
                }

                // Mask margins
                if i != 0 {
                    for row in &mut patch_idx[..left_margin] {
                        for cell in row.iter_mut() {
                            *cell = -1;
                        }
                    }
                }
                if j != 0 {
                    for row in &mut patch_idx {
                        for cell in row[..left_margin].iter_mut() {
                            *cell = -1;
                        }
                    }
                }
                if i != tiling.0 - 1 {
                    for row in &mut patch_idx[(crop_patch_h - right_margin)..] {
                        for cell in row.iter_mut() {
                            *cell = -1;
                        }
                    }
                }
                if j != tiling.1 - 1 {
                    for row in &mut patch_idx {
                        for cell in row[(crop_patch_w - right_margin)..].iter_mut() {
                            *cell = -1;
                        }
                    }
                }

                // Write into full patch index array
                let y_start = i * crop_window_patches;
                let x_start = j * crop_window_patches;
                for (py, row) in patch_idx.iter().enumerate() {
                    for (px, &cell) in row.iter().enumerate() {
                        let fy = y_start + py;
                        let fx = x_start + px;
                        if cell >= 0 {
                            patch_idx_full[fy][fx] = cell;
                        }
                    }
                }

                on_crop += 1;
            }
        }

        (crop_arr, patch_idx_full, tiling)
    }

    fn build_resized_image(
        &self,
        img: &image::RgbImage,
        _orig_h: usize,
        _orig_w: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<i32>>) {
        let base_h = self.base_image_size.0;
        let base_w = self.base_image_size.1;

        let resized = image::imageops::resize(
            img,
            base_w as u32,
            base_h as u32,
            image::imageops::FilterType::Triangle,
        );

        let normalized = self.normalize_image(&resized);

        let crop_patch_h = base_h / self.patch_size;
        let crop_patch_w = base_w / self.patch_size;

        let mut resize_idx = vec![vec![0i32; crop_patch_w]; crop_patch_h];
        for (py, row) in resize_idx.iter_mut().enumerate() {
            for (px, cell) in row.iter_mut().enumerate() {
                *cell = (py * crop_patch_w + px) as i32;
            }
        }

        (normalized, resize_idx)
    }

    fn normalize_image(&self, img: &image::RgbImage) -> Vec<Vec<f32>> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut result = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let pixel = img.get_pixel(x as u32, y as u32);
                result.push(vec![
                    (pixel[0] as f32 / 255.0 - self.image_mean[0]) / self.image_std[0],
                    (pixel[1] as f32 / 255.0 - self.image_mean[1]) / self.image_std[1],
                    (pixel[2] as f32 / 255.0 - self.image_mean[2]) / self.image_std[2],
                ]);
            }
        }
        result
    }

    fn batch_pixels_to_patches(&self, crops: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
        let patch_size = self.patch_size;
        let crop_size = self.base_image_size.0;
        let n_patches_h = crop_size / patch_size;
        let n_patches_w = crop_size / patch_size;
        let n_patches = n_patches_h * n_patches_w;
        let patch_dim = patch_size * patch_size * 3;

        let mut result = Vec::with_capacity(crops.len());
        for crop in crops {
            // crop is [crop_size * crop_size, 3] flattened as [h*w][3]
            let mut patches = Vec::with_capacity(n_patches);
            for ph in 0..n_patches_h {
                for pw in 0..n_patches_w {
                    let mut patch = Vec::with_capacity(patch_dim);
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = ph * patch_size + dy;
                            let x = pw * patch_size + dx;
                            let idx = y * crop_size + x;
                            if idx < crop.len() {
                                patch.extend_from_slice(&crop[idx]);
                            } else {
                                patch.extend_from_slice(&[0.0, 0.0, 0.0]);
                            }
                        }
                    }
                    patches.push(patch);
                }
            }
            result.push(patches);
        }
        result
    }

    fn arange_for_pooling(&self, idx_arr: &[Vec<i32>]) -> Vec<Vec<Vec<i32>>> {
        let h = idx_arr.len();
        let w = if h > 0 {
            idx_arr[0].len()
        } else {
            return vec![];
        };
        let (pool_h, pool_w) = self.pooling_size;

        // Calculate centered padding
        let h_pad = pool_h * h.div_ceil(pool_h) - h;
        let w_pad = pool_w * w.div_ceil(pool_w) - w;

        let pad_top = h_pad / 2;
        let pad_bottom = h_pad.div_ceil(2);
        let pad_left = w_pad / 2;
        let pad_right = w_pad.div_ceil(2);

        let padded_h = h + pad_top + pad_bottom;
        let padded_w = w + pad_left + pad_right;

        // Create padded array
        let mut padded = vec![vec![-1i32; padded_w]; padded_h];
        for py in 0..h {
            for px in 0..w {
                padded[py + pad_top][px + pad_left] = idx_arr[py][px];
            }
        }

        // Rearrange into pooling windows
        let out_h = padded_h / pool_h;
        let out_w = padded_w / pool_w;

        let mut result = Vec::with_capacity(out_h);
        for oh in 0..out_h {
            let mut row = Vec::with_capacity(out_w);
            for ow in 0..out_w {
                let mut pool = Vec::with_capacity(pool_h * pool_w);
                for dh in 0..pool_h {
                    for dw in 0..pool_w {
                        pool.push(padded[oh * pool_h + dh][ow * pool_w + dw]);
                    }
                }
                row.push(pool);
            }
            result.push(row);
        }
        result
    }
}
