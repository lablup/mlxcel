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

//! Kimi K3 (MoonViT3D) image processor: the navit resize rule, the chessboard
//! alpha fill, right/bottom padding to multiples of `merge * patch`, and the
//! `[gh * gw, 3, p, p]` patchify.
//!
//! Port of `KimiK3VisionProcessor` and `media_utils` from
//! https://huggingface.co/moonshotai/Kimi-K3/blob/main/kimi_k3_vision_processing.py
//! and https://huggingface.co/moonshotai/Kimi-K3/blob/main/media_utils.py,
//! configured by `preprocessor_config.json` `media_proc_cfg` (issue #1342).
//!
//! Per image of `w x h` pixels:
//!
//! ```text
//! s = min(1, sqrt(in_patch_limit / (max(1, w // p) * max(1, h // p))),
//!         side * p / w, side * p / h)
//! new_w = min(max(1, floor(w * s)), side * p);  new_h likewise
//! before_resize: flatten alpha onto the background now (RGB untouched)
//! resize to (new_w, new_h), bicubic; after_resize: flatten alpha onto the
//! chessboard (8 px squares, white 255 / gray 180, white top-left,
//! out = alpha * rgb + (1 - alpha) * bg)
//! pad right/bottom with black to multiples of merge * p
//! pixels = (rgb / 255 - mean) / std;  gh = H / p, gw = W / p
//! patches [gh * gw, 3, p, p] row-major over (gh, gw);  grid_thw = (1, gh, gw)
//! num_tokens = gh * gw / (merge * merge)
//! ```
//!
//! The image prompt uses the **original** `w x h`, before any resize.
//!
//! Known deviation: Pillow resizes palette (`P`) images with nearest-neighbour
//! sampling before the alpha fill; the `image` crate decodes palettes to RGB
//! or RGBA at load, so such files are resized bicubically here.
//!
//! Used by: `vision::kimi_k3_vl::KimiK3VLModel`, `multimodal::kimi_k3_prompt`,
//! `server::kimi_k3_chat` (token counts for the prompt).

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage, RgbaImage};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::encoders::moonvit3d::MoonViT3DGrid;

#[path = "kimi_k3_config.rs"]
mod config;

/// The published `media_proc_cfg` values.
pub const PUBLISHED_PATCH_SIZE: u32 = 14;
pub const PUBLISHED_MERGE_KERNEL_SIZE: u32 = 2;
pub const PUBLISHED_IN_PATCH_LIMIT: u32 = 65_536;
pub const PUBLISHED_PATCH_LIMIT_ON_ONE_SIDE: u32 = 512;

/// The navit resize parameters (the part of `media_proc_cfg` that decides
/// the token count of an image).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct KimiK3NavitConfig {
    #[serde(default = "default_patch_size")]
    pub patch_size: u32,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: u32,
    #[serde(default = "default_in_patch_limit")]
    pub in_patch_limit: u32,
    #[serde(default = "default_patch_limit_on_one_side")]
    pub patch_limit_on_one_side: u32,
}

fn default_patch_size() -> u32 {
    PUBLISHED_PATCH_SIZE
}
fn default_merge_kernel_size() -> u32 {
    PUBLISHED_MERGE_KERNEL_SIZE
}
fn default_in_patch_limit() -> u32 {
    PUBLISHED_IN_PATCH_LIMIT
}
fn default_patch_limit_on_one_side() -> u32 {
    PUBLISHED_PATCH_LIMIT_ON_ONE_SIDE
}

impl Default for KimiK3NavitConfig {
    fn default() -> Self {
        Self {
            patch_size: PUBLISHED_PATCH_SIZE,
            merge_kernel_size: PUBLISHED_MERGE_KERNEL_SIZE,
            in_patch_limit: PUBLISHED_IN_PATCH_LIMIT,
            patch_limit_on_one_side: PUBLISHED_PATCH_LIMIT_ON_ONE_SIDE,
        }
    }
}

/// The geometry the navit rule assigns to one `w x h` image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NavitResizePlan {
    pub new_w: u32,
    pub new_h: u32,
    pub pad_w: u32,
    pub pad_h: u32,
    /// Patch grid of the padded image.
    pub grid_h: u32,
    pub grid_w: u32,
    /// `<|media_pad|>` count: `grid_h * grid_w / merge^2`.
    pub num_tokens: u32,
}

impl NavitResizePlan {
    /// The padded size handed to the patchifier.
    pub fn padded(&self) -> (u32, u32) {
        (self.new_w + self.pad_w, self.new_h + self.pad_h)
    }
}

impl KimiK3NavitConfig {
    /// `navit_resize_image` from the reference `media_utils`, in f64 like the
    /// Python it mirrors.
    pub fn plan(&self, w: u32, h: u32) -> Result<NavitResizePlan, String> {
        if w == 0 || h == 0 {
            return Err(format!("Kimi K3 image has an empty dimension: {w}x{h}"));
        }
        if self.patch_size == 0 || self.merge_kernel_size == 0 {
            return Err(
                "Kimi K3 media_proc_cfg: patch_size and merge_kernel_size must be positive".into(),
            );
        }
        let p = f64::from(self.patch_size);
        let side_px = f64::from(self.patch_limit_on_one_side) * p;
        let wf = f64::from(w);
        let hf = f64::from(h);
        let s1 = (f64::from(self.in_patch_limit)
            / (f64::from(w / self.patch_size).max(1.0) * f64::from(h / self.patch_size).max(1.0)))
        .sqrt();
        let s2 = side_px / wf;
        let s3 = side_px / hf;
        let scale = 1.0f64.min(s1).min(s2).min(s3);
        // `int()` truncates toward zero; the operands are positive.
        let new_w = ((wf * scale).trunc() as u32).max(1).min(side_px as u32);
        let new_h = ((hf * scale).trunc() as u32).max(1).min(side_px as u32);

        let factor = self.merge_kernel_size * self.patch_size;
        let pad_h = (factor - new_h % factor) % factor;
        let pad_w = (factor - new_w % factor) % factor;
        let token_h = (new_h + pad_h) / factor;
        let token_w = (new_w + pad_w) / factor;
        for (name, tokens) in [("height", token_h), ("width", token_w)] {
            if tokens * self.merge_kernel_size > self.patch_limit_on_one_side {
                return Err(format!(
                    "Kimi K3 navit resize: token_{name} {tokens} * merge_kernel_size {} exceeds \
                     patch_limit_on_one_side {} for a {w}x{h} image",
                    self.merge_kernel_size, self.patch_limit_on_one_side
                ));
            }
        }
        Ok(NavitResizePlan {
            new_w,
            new_h,
            pad_w,
            pad_h,
            grid_h: token_h * self.merge_kernel_size,
            grid_w: token_w * self.merge_kernel_size,
            num_tokens: token_h * token_w,
        })
    }
}

/// The chessboard the alpha channel is flattened onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessboardConfig {
    pub square_size: u32,
    pub white_on_top_left: bool,
    pub white: u8,
    pub gray: u8,
}

impl Default for ChessboardConfig {
    /// The published `transparent_bg_config`.
    fn default() -> Self {
        Self {
            square_size: 8,
            white_on_top_left: true,
            white: 255,
            gray: 180,
        }
    }
}

impl ChessboardConfig {
    /// The background value under pixel `(x, y)`.
    pub fn value_at(&self, x: u32, y: u32) -> u8 {
        let parity = (y / self.square_size + x / self.square_size) % 2;
        let gray_parity = if self.white_on_top_left { 1 } else { 0 };
        if parity == gray_parity {
            self.gray
        } else {
            self.white
        }
    }
}

/// Which `transparent_bg_config.pattern` fills the transparent area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparentBackground {
    White,
    Black,
    Gray,
    Chessboard(ChessboardConfig),
}

impl TransparentBackground {
    fn value_at(&self, x: u32, y: u32) -> u8 {
        match self {
            Self::White => 255,
            Self::Black => 0,
            Self::Gray => 128,
            Self::Chessboard(cfg) => cfg.value_at(x, y),
        }
    }
}

/// When the alpha channel is flattened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillStage {
    /// `before_resize`: the reference default.
    #[default]
    BeforeResize,
    /// `after_resize`: what the published K3 config selects.
    AfterResize,
}

/// `alpha * rgb + (1 - alpha) * bg`, in f32 with truncation to `u8`, exactly
/// as `fill_transparent_bg_with` computes it.
pub fn composite_onto_background(rgba: &RgbaImage, background: TransparentBackground) -> RgbImage {
    let (w, h) = rgba.dimensions();
    let mut out = RgbImage::new(w, h);
    for (x, y, px) in rgba.enumerate_pixels() {
        let alpha = f32::from(px[3]) / 255.0;
        let bg = f32::from(background.value_at(x, y));
        let blend = |c: u8| (alpha * f32::from(c) + (1.0 - alpha) * bg) as u8;
        out.put_pixel(x, y, image::Rgb([blend(px[0]), blend(px[1]), blend(px[2])]));
    }
    out
}

/// One preprocessed image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KimiK3PreparedImage {
    pub grid: MoonViT3DGrid,
    /// `(w, h)` before any resize; the size the image prompt names.
    pub original_size: (u32, u32),
    pub plan: NavitResizePlan,
}

/// A preprocessed batch.
pub struct KimiK3PreparedImages {
    /// `[total_patches, 3, p, p]` f32, images concatenated in order.
    pub pixel_values: UniquePtr<MlxArray>,
    pub images: Vec<KimiK3PreparedImage>,
}

impl KimiK3PreparedImages {
    pub fn grids(&self) -> Vec<MoonViT3DGrid> {
        self.images.iter().map(|i| i.grid).collect()
    }
}

/// The Kimi K3 image processor.
#[derive(Debug, Clone, PartialEq)]
pub struct KimiK3ImageProcessor {
    pub navit: KimiK3NavitConfig,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// `None` mirrors `transparent_bg_config: null`: alpha is dropped by a
    /// plain RGB conversion instead of being composited.
    pub background: Option<TransparentBackground>,
    pub fill_stage: FillStage,
}

impl Default for KimiK3ImageProcessor {
    /// The published `preprocessor_config.json`.
    fn default() -> Self {
        Self {
            navit: KimiK3NavitConfig::default(),
            mean: [0.5; 3],
            std: [0.5; 3],
            background: Some(TransparentBackground::Chessboard(
                ChessboardConfig::default(),
            )),
            fill_stage: FillStage::AfterResize,
        }
    }
}

impl KimiK3ImageProcessor {
    /// The navit geometry of a `w x h` image.
    pub fn plan(&self, w: u32, h: u32) -> Result<NavitResizePlan, String> {
        self.navit.plan(w, h)
    }

    /// Flatten alpha onto the configured background, or drop it.
    fn flatten(&self, image: &DynamicImage) -> RgbImage {
        match self.background {
            Some(bg) if image.color().has_alpha() => {
                composite_onto_background(&image.to_rgba8(), bg)
            }
            _ => image.to_rgb8(),
        }
    }

    /// Resize (bicubic), fill and pad one image to its planned size.
    fn resized_rgb(&self, image: &DynamicImage, plan: &NavitResizePlan) -> RgbImage {
        let (w, h) = (image.width(), image.height());
        let same = w == plan.new_w && h == plan.new_h;
        let rgb = match self.fill_stage {
            FillStage::BeforeResize => {
                let rgb = self.flatten(image);
                if same {
                    rgb
                } else {
                    image::imageops::resize(&rgb, plan.new_w, plan.new_h, FilterType::CatmullRom)
                }
            }
            FillStage::AfterResize => {
                if self.background.is_some() && image.color().has_alpha() {
                    let rgba = image.to_rgba8();
                    let rgba = if same {
                        rgba
                    } else {
                        image::imageops::resize(
                            &rgba,
                            plan.new_w,
                            plan.new_h,
                            FilterType::CatmullRom,
                        )
                    };
                    self.flatten(&DynamicImage::ImageRgba8(rgba))
                } else {
                    let rgb = image.to_rgb8();
                    if same {
                        rgb
                    } else {
                        image::imageops::resize(
                            &rgb,
                            plan.new_w,
                            plan.new_h,
                            FilterType::CatmullRom,
                        )
                    }
                }
            }
        };
        let (pw, ph) = plan.padded();
        if plan.pad_w == 0 && plan.pad_h == 0 {
            return rgb;
        }
        let mut padded = RgbImage::from_pixel(pw, ph, image::Rgb([0, 0, 0]));
        image::imageops::replace(&mut padded, &rgb, 0, 0);
        padded
    }

    /// Normalize and patchify one image, appending `[gh * gw, 3, p, p]` f32
    /// values to `out` (row-major over `(gh, gw)`; `[C, p, p]` per patch).
    pub fn prepare(
        &self,
        image: &DynamicImage,
        out: &mut Vec<f32>,
    ) -> Result<KimiK3PreparedImage, String> {
        let original_size = (image.width(), image.height());
        let plan = self.plan(original_size.0, original_size.1)?;
        let rgb = self.resized_rgb(image, &plan);
        let p = self.navit.patch_size;
        let (gh, gw) = (plan.grid_h, plan.grid_w);
        debug_assert_eq!((rgb.width(), rgb.height()), (gw * p, gh * p));
        let channels: Vec<(usize, f32, f32)> = (0..3)
            .map(|c| (c, self.mean[c], 1.0 / self.std[c]))
            .collect();
        out.reserve((gh * gw * 3 * p * p) as usize);
        for row in 0..gh {
            for col in 0..gw {
                for &(c, mean, inv_std) in &channels {
                    for py in 0..p {
                        for px in 0..p {
                            let v = rgb.get_pixel(col * p + px, row * p + py)[c];
                            out.push((f32::from(v) / 255.0 - mean) * inv_std);
                        }
                    }
                }
            }
        }
        Ok(KimiK3PreparedImage {
            grid: MoonViT3DGrid::image(gh as i32, gw as i32),
            original_size,
            plan,
        })
    }

    /// Preprocess a batch of images into one `[total_patches, 3, p, p]`
    /// tensor plus the per-image grid and original size.
    pub fn preprocess_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> Result<KimiK3PreparedImages, String> {
        let p = self.navit.patch_size as i32;
        let mut all: Vec<f32> = Vec::new();
        let mut prepared = Vec::with_capacity(images.len());
        let mut total = 0i32;
        for image in images {
            let item = self.prepare(image, &mut all)?;
            total += item.grid.token_count();
            prepared.push(item);
        }
        let pixel_values = mlxcel_core::from_slice_f32(&all, &[total, 3, p, p]);
        Ok(KimiK3PreparedImages {
            pixel_values,
            images: prepared,
        })
    }
}

/// The reference `make_image_prompt` expanded to the token count:
/// `<|media_begin|>image {w}x{h}<|media_content|>` + `<|media_pad|>` x
/// `num_tokens` + `<|media_end|>`, with `w x h` the original size.
pub fn image_prompt_text(w: u32, h: u32, num_tokens: u32) -> String {
    let mut text = String::with_capacity(48 + 13 * num_tokens as usize);
    text.push_str(&format!("<|media_begin|>image {w}x{h}<|media_content|>"));
    for _ in 0..num_tokens {
        text.push_str("<|media_pad|>");
    }
    text.push_str("<|media_end|>");
    text
}

#[cfg(test)]
#[path = "kimi_k3_tests.rs"]
mod tests;
