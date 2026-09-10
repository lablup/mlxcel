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
//! Under the published `media_proc_cfg` the 65,536-patch budget is what binds:
//! `patch_limit_on_one_side` (512) is reached only through the side scale
//! `s2` / `s3`, which caps a side at `512 * 14 = 7168` pixels, itself a
//! multiple of the 28-pixel padding unit, so the per-side assertion the
//! reference makes after padding cannot fire on this config. The real ceiling
//! is the patch budget plus padding: just under 17,000 merged tokens, about
//! 67,000 patches, for one image of a shape near 1821x7069. The per-side
//! check is kept because a checkpoint may configure the two limits
//! differently.
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

/// Layout of the emitted patch tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchLayout {
    /// `[N, 3, p, p]`, what the reference processor emits and what the
    /// fixture oracle holds.
    ChannelsFirst,
    /// `[N, p, p, 3]`, what MLX's `conv2d` reads, so what the tower wants.
    ChannelsLast,
}

/// Default ceiling on the merged media tokens (`<|media_pad|>` placeholders)
/// one request may ask for, summed over its images.
///
/// The navit rule alone bounds a single image at just under 17,000 merged
/// tokens (about 67,000 patches, at a shape near 1821x7069), and the tower's
/// attention is quadratic in the patch count of one image, so a handful of
/// large images is already a long computation. Nothing upstream of this
/// bounds the total: the per-request image count and the per-image pixel
/// dimensions are capped, but 16 images at 4000x3000 still ask for 247,104
/// media tokens and 988,416 tower tokens. The budget is that missing bound.
/// It admits one worst-case image with room to spare and about 30 images of
/// ordinary size (a 1024x768 photo costs 1,036).
pub const DEFAULT_MAX_MEDIA_TOKENS_PER_REQUEST: u32 = 32_768;

/// The environment variable that overrides
/// [`DEFAULT_MAX_MEDIA_TOKENS_PER_REQUEST`].
pub const MAX_MEDIA_TOKENS_ENV: &str = "MLXCEL_KIMI_K3_MAX_MEDIA_TOKENS";

/// The media-token budget of one request: [`MAX_MEDIA_TOKENS_ENV`] when it
/// parses to a positive number, [`DEFAULT_MAX_MEDIA_TOKENS_PER_REQUEST`]
/// otherwise.
#[must_use]
pub fn max_media_tokens_per_request() -> u32 {
    std::env::var(MAX_MEDIA_TOKENS_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .filter(|budget| *budget > 0)
        .unwrap_or(DEFAULT_MAX_MEDIA_TOKENS_PER_REQUEST)
}

/// Refuse a request whose images together ask for more than the budget.
///
/// Checked where the image dimensions are first known (the prompt render on
/// the server, the plan on the worker), so an oversized request is answered
/// with an error before any pixel is decoded or any tower block runs.
pub fn check_media_token_budget<I: IntoIterator<Item = u32>>(per_image: I) -> Result<(), String> {
    let budget = max_media_tokens_per_request();
    let mut total: u64 = 0;
    let mut count = 0usize;
    for tokens in per_image {
        total += u64::from(tokens);
        count += 1;
    }
    if total > u64::from(budget) {
        return Err(format!(
            "Kimi K3: {count} image(s) ask for {total} media tokens, over the per-request budget \
             of {budget}; send fewer or smaller images, or raise {MAX_MEDIA_TOKENS_ENV}"
        ));
    }
    Ok(())
}

/// The published `media_proc_cfg` values.
pub const PUBLISHED_PATCH_SIZE: u32 = 14;
pub const PUBLISHED_MERGE_KERNEL_SIZE: u32 = 2;
pub const PUBLISHED_IN_PATCH_LIMIT: u32 = 65_536;
pub const PUBLISHED_PATCH_LIMIT_ON_ONE_SIDE: u32 = 512;

/// Backstop bound on `media_proc_cfg.patch_size`.
///
/// The same 128 that `processors::locateanything` uses, and for the same
/// reason: the patch size is read from `preprocessor_config.json`, which is as
/// untrusted as `config.json`, and it multiplies into both the pad unit and
/// the per-patch allocation. A `patch_size` of 100000 on a 100x100 image
/// rounds that image up to one 200000x200000 pad unit.
pub const MAX_PATCH_SIZE: u32 = 128;

/// Backstop bound on `media_proc_cfg.merge_kernel_size`. Real spatial merges
/// are 1, 2 or 4.
pub const MAX_MERGE_KERNEL: u32 = 16;

/// Bound on `media_proc_cfg.in_patch_limit`, the pre-padding patch budget of
/// one image.
///
/// The published 65536 costs about 154 MB of f32 patch data at
/// `patch_size: 14`. 262144 leaves a checkpoint room to raise it fourfold and
/// still holds one image's patch tensor inside about 616 MB.
pub const MAX_IN_PATCH_LIMIT: u32 = 262_144;

/// Bound on `media_proc_cfg.patch_limit_on_one_side` (published: 512).
pub const MAX_PATCH_LIMIT_ON_ONE_SIDE: u32 = 4_096;

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
    /// Refuse geometry outside the bounds above, naming the field.
    ///
    /// These values come from the checkpoint's `preprocessor_config.json` and
    /// nothing downstream re-derives them, so an out-of-range one turns
    /// straight into an allocation size.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value, max) in [
            ("patch_size", self.patch_size, MAX_PATCH_SIZE),
            (
                "merge_kernel_size",
                self.merge_kernel_size,
                MAX_MERGE_KERNEL,
            ),
            ("in_patch_limit", self.in_patch_limit, MAX_IN_PATCH_LIMIT),
            (
                "patch_limit_on_one_side",
                self.patch_limit_on_one_side,
                MAX_PATCH_LIMIT_ON_ONE_SIDE,
            ),
        ] {
            if value == 0 {
                return Err(format!("Kimi K3 media_proc_cfg: {name} must be positive"));
            }
            if value > max {
                return Err(format!(
                    "Kimi K3 media_proc_cfg: {name} is {value}, over the supported maximum {max}"
                ));
            }
        }
        Ok(())
    }

    /// `navit_resize_image` from the reference `media_utils`, in f64 like the
    /// Python it mirrors.
    pub fn plan(&self, w: u32, h: u32) -> Result<NavitResizePlan, String> {
        if w == 0 || h == 0 {
            return Err(format!("Kimi K3 image has an empty dimension: {w}x{h}"));
        }
        self.validate()?;
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

        // The padding and grid arithmetic runs in u64: `patch_size`,
        // `merge_kernel_size` and `patch_limit_on_one_side` come from the
        // checkpoint's `preprocessor_config.json`, and a config that names
        // large values must produce a named error rather than a wrapped
        // product and an allocation sized from it.
        let factor = u64::from(self.merge_kernel_size) * u64::from(self.patch_size);
        let pad_h = (factor - u64::from(new_h) % factor) % factor;
        let pad_w = (factor - u64::from(new_w) % factor) % factor;
        let token_h = (u64::from(new_h) + pad_h) / factor;
        let token_w = (u64::from(new_w) + pad_w) / factor;
        for (name, tokens) in [("height", token_h), ("width", token_w)] {
            if tokens * u64::from(self.merge_kernel_size) > u64::from(self.patch_limit_on_one_side)
            {
                return Err(format!(
                    "Kimi K3 navit resize: token_{name} {tokens} * merge_kernel_size {} exceeds \
                     patch_limit_on_one_side {} for a {w}x{h} image",
                    self.merge_kernel_size, self.patch_limit_on_one_side
                ));
            }
        }
        let grid_h = token_h * u64::from(self.merge_kernel_size);
        let grid_w = token_w * u64::from(self.merge_kernel_size);
        // The patch tensor is `[grid_h * grid_w, 3, p, p]` f32, and MLX shapes
        // are i32, so the element count has to fit one.
        let elements =
            grid_h * grid_w * 3 * u64::from(self.patch_size) * u64::from(self.patch_size);
        if elements > i32::MAX as u64 {
            return Err(format!(
                "Kimi K3 navit resize: a {w}x{h} image would cut a {grid_h}x{grid_w} patch grid, \
                 {elements} pixel values, past what one tensor can hold"
            ));
        }
        Ok(NavitResizePlan {
            new_w,
            new_h,
            pad_w: pad_w as u32,
            pad_h: pad_h as u32,
            grid_h: grid_h as u32,
            grid_w: grid_w as u32,
            num_tokens: (token_h * token_w) as u32,
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
    ///
    /// A zero `square_size` is refused when the config is parsed; the clamp
    /// here keeps a hand-built value from dividing by zero.
    pub fn value_at(&self, x: u32, y: u32) -> u8 {
        let size = self.square_size.max(1);
        let parity = (y / size + x / size) % 2;
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

    /// The navit geometry of one image, with no pixel work: what the prompt
    /// needs to size its `<|media_pad|>` run, and what the budget check reads.
    pub fn plan_image(&self, image: &DynamicImage) -> Result<KimiK3PreparedImage, String> {
        let original_size = (image.width(), image.height());
        let plan = self.plan(original_size.0, original_size.1)?;
        Ok(KimiK3PreparedImage {
            grid: MoonViT3DGrid::image(plan.grid_h as i32, plan.grid_w as i32),
            original_size,
            plan,
        })
    }

    /// [`Self::plan_image`] over a batch, refusing a request that asks for
    /// more media tokens than [`max_media_tokens_per_request`].
    pub fn plan_images(&self, images: &[DynamicImage]) -> Result<Vec<KimiK3PreparedImage>, String> {
        let planned: Vec<KimiK3PreparedImage> = images
            .iter()
            .map(|image| self.plan_image(image))
            .collect::<Result<_, _>>()?;
        check_media_token_budget(planned.iter().map(|item| item.plan.num_tokens))?;
        Ok(planned)
    }

    /// Normalize and patchify one image, appending `[gh * gw, 3, p, p]` f32
    /// values to `out` (row-major over `(gh, gw)`; `[C, p, p]` per patch).
    ///
    /// This is the reference processor's layout, which is what the fixture
    /// oracle holds. The tower reads the channels-last form
    /// ([`Self::prepare_image_array`]).
    pub fn prepare(
        &self,
        image: &DynamicImage,
        out: &mut Vec<f32>,
    ) -> Result<KimiK3PreparedImage, String> {
        self.prepare_into(image, PatchLayout::ChannelsFirst, out)
    }

    /// [`Self::prepare`] in either patch layout.
    pub fn prepare_into(
        &self,
        image: &DynamicImage,
        layout: PatchLayout,
        out: &mut Vec<f32>,
    ) -> Result<KimiK3PreparedImage, String> {
        let item = self.plan_image(image)?;
        let plan = item.plan;
        let rgb = self.resized_rgb(image, &plan);
        let p = self.navit.patch_size;
        let (gh, gw) = (plan.grid_h, plan.grid_w);
        if (rgb.width(), rgb.height()) != (gw * p, gh * p) {
            return Err(format!(
                "Kimi K3 preprocess: the padded image is {}x{} but the plan asks for {}x{}",
                rgb.width(),
                rgb.height(),
                gw * p,
                gh * p
            ));
        }
        let norm: [(f32, f32); 3] = [
            (self.mean[0], 1.0 / self.std[0]),
            (self.mean[1], 1.0 / self.std[1]),
            (self.mean[2], 1.0 / self.std[2]),
        ];
        out.reserve((u64::from(gh) * u64::from(gw) * 3 * u64::from(p) * u64::from(p)) as usize);
        for row in 0..gh {
            for col in 0..gw {
                match layout {
                    PatchLayout::ChannelsFirst => {
                        for (c, &(mean, inv_std)) in norm.iter().enumerate() {
                            for py in 0..p {
                                for px in 0..p {
                                    let v = rgb.get_pixel(col * p + px, row * p + py)[c];
                                    out.push((f32::from(v) / 255.0 - mean) * inv_std);
                                }
                            }
                        }
                    }
                    PatchLayout::ChannelsLast => {
                        for py in 0..p {
                            for px in 0..p {
                                let pixel = rgb.get_pixel(col * p + px, row * p + py);
                                for (c, &(mean, inv_std)) in norm.iter().enumerate() {
                                    out.push((f32::from(pixel[c]) / 255.0 - mean) * inv_std);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(item)
    }

    /// Preprocess one image straight into the `[gh * gw, p, p, 3]` array the
    /// tower reads, in `dtype`.
    ///
    /// The per-image form is what the request path uses: a batch tensor of
    /// the whole request would hold every image's f32 pixels at once, and the
    /// tower slices it back apart per image regardless, because attention is
    /// per image.
    pub fn prepare_image_array(
        &self,
        image: &DynamicImage,
        dtype: i32,
    ) -> Result<(UniquePtr<MlxArray>, KimiK3PreparedImage), String> {
        let mut values: Vec<f32> = Vec::new();
        let item = self.prepare_into(image, PatchLayout::ChannelsLast, &mut values)?;
        let p = self.navit.patch_size as i32;
        let array = mlxcel_core::from_slice_f32(&values, &[item.grid.token_count(), p, p, 3]);
        drop(values);
        let array = if dtype == mlxcel_core::dtype::FLOAT32 {
            array
        } else {
            mlxcel_core::astype(&array, dtype)
        };
        Ok((array, item))
    }

    /// Preprocess a batch of images into one `[total_patches, 3, p, p]`
    /// tensor plus the per-image grid and original size.
    ///
    /// Held for the fixture comparison and for callers that want the
    /// reference layout in one array. The request path streams one image at a
    /// time through [`Self::prepare_image_array`] instead.
    pub fn preprocess_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> Result<KimiK3PreparedImages, String> {
        let p = self.navit.patch_size as i32;
        let planned = self.plan_images(images)?;
        let mut all: Vec<f32> = Vec::new();
        let mut prepared = Vec::with_capacity(images.len());
        let mut total = 0i32;
        for image in images {
            let item = self.prepare(image, &mut all)?;
            total += item.grid.token_count();
            prepared.push(item);
        }
        debug_assert_eq!(planned.len(), prepared.len());
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
