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

//! Phi4MM HD (high-definition) image processor.
//!
//! Phi4MM uses a dynamic HD transform that:
//! 1. Resizes the image preserving aspect ratio, padding to a crop grid
//! 2. Creates a global thumbnail (bicubic downscale to crop_size × crop_size)
//! 3. Splits the padded image into crop_size × crop_size sub-crops
//! 4. Each crop is independently processed through the vision encoder
//!
//! Used by: Phi4MM VLM

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-crop pixel data extracted from the original image.
pub struct Phi4MMCropInput {
    /// Flattened patches for this crop: [1, num_patches_per_crop, patch_dim]
    pub pixel_values: UniquePtr<MlxArray>,
}

/// Full HD-processed image data for one image.
pub struct Phi4MMImageInput {
    /// Pixel values for all crops: global crop first, then sub-crops.
    /// Each crop has [1, patches_per_crop, patch_dim] shape.
    pub crops: Vec<Phi4MMCropInput>,
    /// (h_crops, w_crops) — number of sub-crop rows and columns
    pub image_grid: (usize, usize),
    /// Number of patches per crop side after AvgPool2d (e.g., 16 for 32→16)
    pub pooled_grid_size: usize,
    /// Number of active rows in the downsampled attention mask
    pub active_rows: usize,
    /// Number of active columns in the downsampled attention mask
    pub active_cols: usize,
    /// Total number of image tokens this image will produce after HD transform
    pub num_img_tokens: usize,
}

pub struct Phi4MMProcessor {
    pub crop_size: usize,
    pub patch_size: usize,
    pub dynamic_hd: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Phi4MMProcessor {
    pub fn new(crop_size: usize, patch_size: usize, dynamic_hd: usize) -> Self {
        Self {
            crop_size,
            patch_size,
            dynamic_hd,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    pub fn preprocess(&self, images: &[DynamicImage]) -> Vec<Phi4MMImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image))
            .collect()
    }

    fn preprocess_single(&self, image: &DynamicImage) -> Phi4MMImageInput {
        let rgb = image.to_rgb8();
        let (orig_w, orig_h) = (rgb.width() as usize, rgb.height() as usize);
        let base = self.crop_size;

        // Determine crop grid
        let w_crop_num = orig_w.div_ceil(base); // ceil division
        let h_crop_num = orig_h.div_ceil(base);
        let (target_w, target_h) = if w_crop_num * h_crop_num > self.dynamic_hd {
            let aspect = orig_w as f32 / orig_h.max(1) as f32;
            let (tw, th) = find_closest_aspect_grid(aspect, self.dynamic_hd);
            (base * tw, base * th)
        } else {
            (base * w_crop_num, base * h_crop_num)
        };

        let w_crops = target_w / base;
        let h_crops = target_h / base;

        // Resize preserving aspect ratio, then pad
        let ratio_w = target_w as f32 / orig_w as f32;
        let ratio_h = target_h as f32 / orig_h as f32;
        let (new_w, new_h, pad_w, pad_h) = if ratio_w < ratio_h {
            let new_h = (orig_h as f32 * ratio_w) as usize;
            (target_w, new_h, 0, target_h - new_h)
        } else {
            let new_w = (orig_w as f32 * ratio_h) as usize;
            (new_w, target_h, target_w - new_w, 0)
        };

        // Compute attention mask (which patches are real vs padding)
        let mask_res = base / self.patch_size; // 448/14 = 32
        let mask_h = mask_res * h_crops;
        let mask_w = mask_res * w_crops;
        let mut attention_mask = vec![true; mask_h * mask_w];
        let pad_patches_w = pad_w / self.patch_size;
        let pad_patches_h = pad_h / self.patch_size;
        if pad_patches_w > 0 {
            for row in 0..mask_h {
                for col in (mask_w - pad_patches_w)..mask_w {
                    attention_mask[row * mask_w + col] = false;
                }
            }
        }
        if pad_patches_h > 0 {
            for row in (mask_h - pad_patches_h)..mask_h {
                for col in 0..mask_w {
                    attention_mask[row * mask_w + col] = false;
                }
            }
        }

        // Resize and pad image (white padding)
        let resized = image.resize_exact(new_w as u32, new_h as u32, FilterType::Triangle);
        let resized_rgb = resized.to_rgb8();
        let mut padded_pixels = vec![255u8; target_h * target_w * 3];
        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = resized_rgb.get_pixel(x as u32, y as u32);
                let offset = (y * target_w + x) * 3;
                padded_pixels[offset] = pixel[0];
                padded_pixels[offset + 1] = pixel[1];
                padded_pixels[offset + 2] = pixel[2];
            }
        }

        // Create global image (bicubic resize to crop_size × crop_size)
        // Use the padded image as source for consistency with Python
        let padded_img =
            image::RgbImage::from_raw(target_w as u32, target_h as u32, padded_pixels.clone())
                .expect("valid padded image");
        let padded_dyn = DynamicImage::ImageRgb8(padded_img);
        let global_img = padded_dyn.resize_exact(base as u32, base as u32, FilterType::CatmullRom);
        let global_rgb = global_img.to_rgb8();

        // Extract patches from global image
        let global_crop = self.extract_crop_patches(&global_rgb, base, base);

        // Split padded image into sub-crops
        let mut crops = vec![global_crop];
        for ch in 0..h_crops {
            for cw in 0..w_crops {
                let sub_crop = self.extract_sub_crop_patches(
                    &padded_pixels,
                    target_w,
                    ch * base,
                    cw * base,
                    base,
                    base,
                );
                crops.push(sub_crop);
            }
        }

        // Compute num_img_tokens from downsampled attention mask
        // Downsample: take every other element in both dimensions
        let half_mask_res = mask_res / 2 + mask_res % 2;
        // Per-crop masks, then downsample
        let mut ds_mask = vec![false; h_crops * half_mask_res * w_crops * half_mask_res];
        for ch in 0..h_crops {
            for cw in 0..w_crops {
                for mh in (0..mask_res).step_by(2) {
                    for mw in (0..mask_res).step_by(2) {
                        let src_row = ch * mask_res + mh;
                        let src_col = cw * mask_res + mw;
                        let val = attention_mask[src_row * mask_w + src_col];
                        let ds_row = ch * half_mask_res + mh / 2;
                        let ds_col = cw * half_mask_res + mw / 2;
                        ds_mask[ds_row * (w_crops * half_mask_res) + ds_col] = val;
                    }
                }
            }
        }

        let ds_total_w = w_crops * half_mask_res;
        let ds_total_h = h_crops * half_mask_res;
        let active_count: usize = ds_mask.iter().filter(|&&v| v).count();
        let active_rows = (0..ds_total_h)
            .filter(|&row| ds_mask[row * ds_total_w])
            .count();
        let active_cols = if ds_total_h > 0 {
            (0..ds_total_w).filter(|&col| ds_mask[col]).count()
        } else {
            0
        };

        let pooled_grid = mask_res / 2; // 32/2 = 16
        // Formula from Python: 256 + 1 + active_count + active_rows + pooled_grid
        let num_img_tokens =
            pooled_grid * pooled_grid + 1 + active_count + active_rows + pooled_grid;

        Phi4MMImageInput {
            crops,
            image_grid: (h_crops, w_crops),
            pooled_grid_size: pooled_grid,
            active_rows,
            active_cols,
            num_img_tokens,
        }
    }

    fn extract_crop_patches(
        &self,
        rgb: &image::RgbImage,
        width: usize,
        height: usize,
    ) -> Phi4MMCropInput {
        let ps = self.patch_size;
        let h_patches = height / ps;
        let w_patches = width / ps;
        let num_patches = h_patches * w_patches;
        let patch_dim = ps * ps * 3;

        let mut data = vec![0.0f32; num_patches * patch_dim];
        for ph in 0..h_patches {
            for pw in 0..w_patches {
                let patch_idx = ph * w_patches + pw;
                let patch_offset = patch_idx * patch_dim;
                for ih in 0..ps {
                    for iw in 0..ps {
                        let x = pw * ps + iw;
                        let y = ph * ps + ih;
                        let pixel = rgb.get_pixel(x as u32, y as u32);
                        let flat = (ih * ps + iw) * 3;
                        for c in 0..3 {
                            let val = pixel[c] as f32 / 255.0;
                            data[patch_offset + flat + c] =
                                (val - self.image_mean[c]) / self.image_std[c];
                        }
                    }
                }
            }
        }

        Phi4MMCropInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, num_patches as i32, patch_dim as i32],
            ),
        }
    }

    fn extract_sub_crop_patches(
        &self,
        padded_pixels: &[u8],
        padded_width: usize,
        start_y: usize,
        start_x: usize,
        crop_w: usize,
        crop_h: usize,
    ) -> Phi4MMCropInput {
        let ps = self.patch_size;
        let h_patches = crop_h / ps;
        let w_patches = crop_w / ps;
        let num_patches = h_patches * w_patches;
        let patch_dim = ps * ps * 3;

        let mut data = vec![0.0f32; num_patches * patch_dim];
        for ph in 0..h_patches {
            for pw in 0..w_patches {
                let patch_idx = ph * w_patches + pw;
                let patch_offset = patch_idx * patch_dim;
                for ih in 0..ps {
                    for iw in 0..ps {
                        let x = start_x + pw * ps + iw;
                        let y = start_y + ph * ps + ih;
                        let src_offset = (y * padded_width + x) * 3;
                        let flat = (ih * ps + iw) * 3;
                        for c in 0..3 {
                            let val = padded_pixels[src_offset + c] as f32 / 255.0;
                            data[patch_offset + flat + c] =
                                (val - self.image_mean[c]) / self.image_std[c];
                        }
                    }
                }
            }
        }

        Phi4MMCropInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, num_patches as i32, patch_dim as i32],
            ),
        }
    }
}

/// Find the closest aspect ratio grid (w_crops, h_crops) that fits within max_crops.
fn find_closest_aspect_grid(aspect_ratio: f32, max_crops: usize) -> (usize, usize) {
    let mut best = (1usize, 1usize);
    let mut best_diff = f32::MAX;

    for total in 1..=max_crops {
        for h in 1..=total {
            let w = total / h;
            if w * h > max_crops || w == 0 {
                continue;
            }
            let target_aspect = w as f32 / h as f32;
            let diff = (aspect_ratio - target_aspect).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (w, h);
            } else if (diff - best_diff).abs() < 1e-6 && w * h > best.0 * best.1 {
                best = (w, h);
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_closest_aspect_grid_square() {
        let (w, h) = find_closest_aspect_grid(1.0, 36);
        assert_eq!(w, h);
        assert!(w * h <= 36);
    }

    #[test]
    fn find_closest_aspect_grid_wide() {
        let (w, h) = find_closest_aspect_grid(2.0, 12);
        assert!(
            w > h,
            "wide image should have more columns: w={}, h={}",
            w,
            h
        );
    }

    #[test]
    fn find_closest_aspect_grid_tall() {
        let (w, h) = find_closest_aspect_grid(0.5, 12);
        assert!(h > w, "tall image should have more rows: w={}, h={}", w, h);
    }

    #[test]
    fn small_image_single_crop() {
        // 400×300 image with crop_size=448 → 1×1 crops
        let processor = Phi4MMProcessor::new(448, 14, 36);
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(400, 300));
        let result = processor.preprocess_single(&img);
        assert_eq!(result.image_grid, (1, 1));
        assert_eq!(result.crops.len(), 2); // 1 global + 1 sub
        assert!(result.num_img_tokens > 0);
    }
}
