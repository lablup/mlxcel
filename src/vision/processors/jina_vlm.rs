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

//! Jina VLM image processor (`cropping_method: overlap-and-resize`).
//!
//! Port of `JinaVLMImageProcessor.molmo_overlap_and_resize_cropping` from the
//! checkpoint's own `image_processing_jvlm.py`. The pipeline is:
//!
//! 1. `smart_resize` the source to a patch-aligned size inside the
//!    `[min_pixels, max_pixels]` budget, preserving aspect ratio.
//! 2. Pick a crop tiling, stretch the image to `tiling * crop_window + margins`
//!    (aspect ratio is *not* preserved here: `preserve_aspect_ratio: false`),
//!    and cut overlapping 378x378 crops.
//! 3. Emit a `<im_start> [<im_patch> * w <im_col>] * h <im_end>` token block for
//!    the crops, prefixed by the same block for a 378x378 thumbnail.
//! 4. Build `image_input_idx`, mapping each pooled patch to the position of its
//!    `<im_patch>` token.
//!
//! Two places diverge from the mlx-vlm port and follow the checkpoint's own
//! processor instead, because that is the code the model was trained with:
//!
//! - The token grid uses `_molmo_get_patches_from_tiling`, which rounds each
//!   crop window up to a multiple of the pooling size *individually*. The
//!   mlx-vlm-equivalent shortcut (`tiles * window + margins`, then round up)
//!   agrees only when the crop window is already even, which it is for Molmo
//!   (16 patches) and is not for Jina VLM (19 patches).
//! - The coverage mask is padded with a trailing `-1` row rather than a leading
//!   `1.0` row. That is an upstream off-by-one (the thumbnail is prepended to
//!   the crops but the mask row is appended), and it makes the connector treat
//!   the last crop as partially padded. It is reproduced deliberately; see
//!   [`JinaVlmProcessor::preprocess_image`].
//!
//! Normalization is `minmax`: `image_min + x * (image_max - image_min)`, i.e.
//! `[0, 1] -> [-1, 1]`, not the CLIP mean/std the config also carries (those
//! fields are inert for `normalization_method: minmax`).

/// Special token ids used to frame the image block. Defaults match the released
/// `jinaai/jina-vlm` tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JinaVlmImageTokens {
    pub image_start_id: i32,
    pub image_end_id: i32,
    pub image_patch_id: i32,
    pub image_col_id: i32,
}

impl Default for JinaVlmImageTokens {
    fn default() -> Self {
        Self {
            image_start_id: 151936,
            image_end_id: 151937,
            image_patch_id: 151938,
            image_col_id: 151939,
        }
    }
}

/// Resolved image-processor configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct JinaVlmProcessor {
    /// Crop side in pixels, `(height, width)`; square in every released config.
    pub base_input_size: (usize, usize),
    pub patch_size: usize,
    pub max_crops: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub overlap_margins: (usize, usize),
    pub pooling_h: usize,
    pub pooling_w: usize,
    pub token_length_h: usize,
    pub token_length_w: usize,
    pub use_column_tokens: bool,
    pub image_min: f32,
    pub image_max: f32,
    pub tokens: JinaVlmImageTokens,
}

impl Default for JinaVlmProcessor {
    fn default() -> Self {
        Self {
            base_input_size: (378, 378),
            patch_size: 14,
            max_crops: 12,
            min_pixels: 3136,
            max_pixels: 1_003_520,
            overlap_margins: (4, 4),
            pooling_h: 2,
            pooling_w: 2,
            token_length_h: 14,
            token_length_w: 14,
            use_column_tokens: true,
            image_min: -1.0,
            image_max: 1.0,
            tokens: JinaVlmImageTokens::default(),
        }
    }
}

/// Output of preprocessing a single image.
#[derive(Debug, Clone, PartialEq)]
pub struct JinaVlmProcessorOutput {
    /// `[n_crops, n_patches, patch_dim]` flattened row-major.
    pub pixel_values: Vec<f32>,
    pub pixel_values_shape: [i32; 3],
    /// The `<im_start> ... <im_end>` token block that replaces `<|image|>`.
    pub image_token_ids: Vec<i32>,
    /// For each pooled patch, the position of its `<im_patch>` token inside
    /// `image_token_ids`. Negative entries are padding and are skipped.
    pub image_input_idx: Vec<i32>,
    /// `[n_crops, n_patches]` per-patch coverage, flattened.
    pub image_masks: Vec<f32>,
    pub image_masks_shape: [i32; 2],
}

impl JinaVlmProcessor {
    pub fn tokens_per_image(&self) -> usize {
        self.token_length_h * self.token_length_w
    }

    fn patches_per_crop(&self) -> (usize, usize) {
        (
            self.base_input_size.0 / self.patch_size,
            self.base_input_size.1 / self.patch_size,
        )
    }

    /// Preprocess one image into crops, the image-token block, `image_input_idx`
    /// and the per-patch coverage mask.
    ///
    /// The one intentional deviation from upstream is that the intermediate
    /// `smart_resize` output stays in `f32` instead of being rounded back to
    /// `uint8`, which upstream does only because `resize_in_float32` is off.
    /// That is a sub-`1/255` difference per channel and does not change the
    /// crop tiling, the token block, or `image_input_idx`.
    pub fn preprocess_image(&self, image: &image::DynamicImage) -> JinaVlmProcessorOutput {
        let rgb = image.to_rgb8();
        let (src_w, src_h) = (rgb.width() as usize, rgb.height() as usize);

        // Stage 1: patch-aligned aspect-preserving resize inside the pixel budget.
        let (resized_h, resized_w) = smart_resize(
            src_h,
            src_w,
            self.patch_size,
            self.min_pixels,
            self.max_pixels,
        );
        // Sample the 8-bit source directly instead of widening it first. The
        // decoded image is sized by the request, so a full-resolution `f32` copy
        // would cost 12 bytes per source pixel before anything has bounded it,
        // while `smart_resize` has already capped the destination at
        // `max_pixels`. Only the bounded destination is `f32`.
        let base = resize_bilinear_u8(rgb.as_raw(), src_h, src_w, resized_h, resized_w);
        drop(rgb);

        self.crop_image(&base, resized_h, resized_w)
    }

    fn crop_image(&self, image: &[f32], img_h: usize, img_w: usize) -> JinaVlmProcessorOutput {
        let patch = self.patch_size;
        let (crop_patches_h, crop_patches_w) = self.patches_per_crop();
        let crop_size = self.base_input_size.0;
        let (left_margin, right_margin) = self.overlap_margins;
        let total_margin_pixels = patch * (right_margin + left_margin);
        let crop_window_patches = crop_patches_h.saturating_sub(right_margin + left_margin);
        let crop_window_size = crop_window_patches * patch;

        let tiling = select_tiling(
            img_h as i64 - total_margin_pixels as i64,
            img_w as i64 - total_margin_pixels as i64,
            crop_window_size,
            self.max_crops,
        );

        // Stage 2: stretch to the tiling target. Aspect ratio is not preserved,
        // and the coverage mask is therefore all ones.
        let target_h = tiling.0 * crop_window_size + total_margin_pixels;
        let target_w = tiling.1 * crop_window_size + total_margin_pixels;
        let src = resize_bilinear(image, img_h, img_w, target_h, target_w);
        let src = self.normalize(&src);

        let n_patches = crop_patches_h * crop_patches_w;
        let patch_dim = patch * patch * 3;

        let mut crop_pixels: Vec<f32> = Vec::new();
        let mut crop_masks: Vec<f32> = Vec::new();
        // Per-crop `[token_length_h][token_length_w]` ordering grid.
        let mut patch_ordering: Vec<Vec<i32>> = Vec::with_capacity(tiling.0 * tiling.1);

        let mut running = 0i32;
        for i in 0..tiling.0 {
            let y0 = i * crop_window_size;
            let crop_y0 = if i == 0 {
                0
            } else {
                left_margin / self.pooling_h
            };
            let mut crop_h = crop_patches_h - (right_margin + left_margin);
            if i == 0 {
                crop_h += left_margin;
            }
            if i == tiling.0 - 1 {
                crop_h += right_margin;
            }

            for j in 0..tiling.1 {
                let x0 = j * crop_window_size;
                let crop_x0 = if j == 0 {
                    0
                } else {
                    left_margin / self.pooling_w
                };
                let mut crop_w = crop_patches_w - (right_margin + left_margin);
                if j == 0 {
                    crop_w += left_margin;
                }
                if j == tiling.1 - 1 {
                    crop_w += right_margin;
                }

                let pooled_h = crop_h.div_ceil(self.pooling_h);
                let pooled_w = crop_w.div_ceil(self.pooling_w);

                let mut grid = vec![-1i32; self.token_length_h * self.token_length_w];
                let mut value = running;
                for py in 0..pooled_h {
                    for px in 0..pooled_w {
                        let ty = crop_y0 + py;
                        let tx = crop_x0 + px;
                        if ty < self.token_length_h && tx < self.token_length_w {
                            grid[ty * self.token_length_w + tx] = value;
                        }
                        value += 1;
                    }
                }
                patch_ordering.push(grid);
                running += (pooled_h * pooled_w) as i32;

                // Extract one crop plus its coverage mask, then patchify.
                let (pixels, mask) = extract_crop(&src, target_h, target_w, y0, x0, crop_size);
                let (patched, patched_mask) = patchify_crop(
                    &pixels,
                    &mask,
                    crop_size,
                    patch,
                    crop_patches_h,
                    crop_patches_w,
                );
                crop_pixels.extend(patched);
                crop_masks.extend(patched_mask);
            }
        }

        let mut flat_ordering = self.resort_patch_ordering(&patch_ordering, tiling);

        // Token grid for the high-resolution crops.
        let grid_h = get_patches_from_tiling(
            tiling.0,
            self.pooling_h,
            crop_patches_h,
            crop_window_patches,
            left_margin,
            right_margin,
        );
        let grid_w = get_patches_from_tiling(
            tiling.1,
            self.pooling_w,
            crop_patches_w,
            crop_window_patches,
            left_margin,
            right_margin,
        );
        let crop_tokens = self.build_token_block(grid_h / self.pooling_h, grid_w / self.pooling_w);

        // Thumbnail: the whole image squeezed into one crop, emitted first.
        let thumb = resize_bilinear(
            image,
            img_h,
            img_w,
            self.base_input_size.0,
            self.base_input_size.1,
        );
        let thumb = self.normalize(&thumb);
        let (thumb_patches, _) = patchify_crop(
            &thumb,
            &vec![1.0f32; self.base_input_size.0 * self.base_input_size.1],
            crop_size,
            patch,
            crop_patches_h,
            crop_patches_w,
        );

        let mut pixel_values = thumb_patches;
        pixel_values.extend(crop_pixels);

        // The thumbnail occupies pooled slots [0, tokens_per_image); the crop
        // slots shift up by that much.
        let tpi = self.tokens_per_image() as i32;
        for v in flat_ordering.iter_mut() {
            if *v >= 0 {
                *v += tpi;
            }
        }
        let mut full_ordering: Vec<i32> = (0..tpi).collect();
        full_ordering.extend(flat_ordering);

        let mut image_token_ids = self.build_token_block(self.token_length_h, self.token_length_w);
        image_token_ids.extend(crop_tokens);

        let image_input_idx = self.build_image_input_idx(&image_token_ids, &full_ordering);

        // Upstream appends the sentinel row *after* the crop masks even though
        // the thumbnail was prepended to the pixel values, so mask row k belongs
        // to crop k+1. Reproduced verbatim: it is what the model was trained
        // with, and it is what makes the last crop read as partially padded.
        let n_crops = tiling.0 * tiling.1 + 1;
        let mut image_masks = crop_masks;
        image_masks.extend(std::iter::repeat_n(-1.0f32, n_patches));

        JinaVlmProcessorOutput {
            pixel_values,
            pixel_values_shape: [n_crops as i32, n_patches as i32, patch_dim as i32],
            image_token_ids,
            image_input_idx,
            image_masks,
            image_masks_shape: [n_crops as i32, n_patches as i32],
        }
    }

    /// `image_min + x * (image_max - image_min)` over `[0, 1]` inputs.
    fn normalize(&self, x: &[f32]) -> Vec<f32> {
        let span = self.image_max - self.image_min;
        x.iter().map(|v| self.image_min + v * span).collect()
    }

    /// `<im_start> [<im_patch> * w <im_col>] * h <im_end>`.
    fn build_token_block(&self, h: usize, w: usize) -> Vec<i32> {
        let t = &self.tokens;
        let mut out = Vec::with_capacity(h * (w + 1) + 2);
        out.push(t.image_start_id);
        for _ in 0..h {
            out.extend(std::iter::repeat_n(t.image_patch_id, w));
            if self.use_column_tokens {
                out.push(t.image_col_id);
            }
        }
        out.push(t.image_end_id);
        out
    }

    /// Reorder the crop-by-crop patch numbering into left-to-right image order.
    ///
    /// Mirrors upstream: reshape to `[tiles_h, tiles_w, token_h, token_w]`,
    /// transpose to `[tiles_h, token_h, tiles_w, token_w]`, then project the
    /// non-negative entries back into the original sparse layout.
    fn resort_patch_ordering(
        &self,
        patch_ordering: &[Vec<i32>],
        tiling: (usize, usize),
    ) -> Vec<i32> {
        let th = self.token_length_h;
        let tw = self.token_length_w;

        let mut flat: Vec<i32> = Vec::with_capacity(patch_ordering.len() * th * tw);
        for grid in patch_ordering {
            flat.extend_from_slice(grid);
        }

        let mut transposed_valid: Vec<i32> = Vec::new();
        for i in 0..tiling.0 {
            for y in 0..th {
                for j in 0..tiling.1 {
                    for x in 0..tw {
                        let value = patch_ordering[i * tiling.1 + j][y * tw + x];
                        if value >= 0 {
                            transposed_valid.push(value);
                        }
                    }
                }
            }
        }

        let mut k = 0usize;
        for v in flat.iter_mut() {
            if *v >= 0 {
                *v = transposed_valid[k];
                k += 1;
            }
        }
        flat
    }

    /// Map each pooled patch to the token position that receives its feature.
    ///
    /// Upstream:
    /// ```text
    /// positions                = nonzero(tokens == <im_patch>)
    /// sorted[patch_order[valid]] = arange(n_valid)
    /// ex                       = -1; ex[valid] = sorted
    /// out                      = positions[ex * (ex >= 0)] * (ex >= 0) - 10000 * (ex < 0)
    /// ```
    fn build_image_input_idx(&self, image_token_ids: &[i32], patch_order: &[i32]) -> Vec<i32> {
        let patch_id = self.tokens.image_patch_id;
        let positions: Vec<i32> = image_token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == patch_id).then_some(i as i32))
            .collect();
        let n_tokens = positions.len();

        let mut sorted = vec![0i32; n_tokens];
        let mut next = 0i32;
        for &v in patch_order.iter() {
            if v >= 0 {
                if (v as usize) < n_tokens {
                    sorted[v as usize] = next;
                }
                next += 1;
            }
        }

        let mut result = Vec::with_capacity(patch_order.len());
        let mut seen_valid = 0usize;
        for &v in patch_order.iter() {
            if v >= 0 {
                let slot = sorted.get(seen_valid).copied().unwrap_or(0);
                result.push(positions.get(slot as usize).copied().unwrap_or(0));
                seen_valid += 1;
            } else {
                result.push(-10000);
            }
        }
        result
    }
}

/// Round a patch grid dimension up to a whole number of pooling windows,
/// per crop window, exactly as `_molmo_get_patches_from_tiling` does.
fn get_patches_from_tiling(
    num_tiles: usize,
    pooling: usize,
    crop_patches: usize,
    crop_window_patches: usize,
    left_margin: usize,
    right_margin: usize,
) -> usize {
    let round_up = |n: usize| n.div_ceil(pooling) * pooling;
    if num_tiles > 1 {
        let left = round_up(crop_window_patches + left_margin);
        let middle = round_up(crop_window_patches);
        let right = round_up(crop_window_patches + right_margin);
        left + (num_tiles - 2) * middle + right
    } else {
        round_up(crop_patches)
    }
}

/// Resize so both sides are multiples of `factor` and the pixel count sits in
/// `[min_pixels, max_pixels]`, keeping the aspect ratio as close as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = factor as f64;
    // Python's `round` is banker's rounding; the two rules disagree at every
    // `dim % factor == factor / 2` and would change the crop tiling there.
    let mut h_bar = (round_half_to_even(height as f64 / f) * f) as usize;
    let mut w_bar = (round_half_to_even(width as f64 / f) * f) as usize;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max(((height as f64 / beta / f).floor() * f) as usize);
        w_bar = factor.max(((width as f64 / beta / f).floor() * f) as usize);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / f).ceil() * f) as usize;
        w_bar = ((width as f64 * beta / f).ceil() * f) as usize;
    }

    (h_bar, w_bar)
}

fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    if (x - x.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - x.signum()
    } else {
        rounded
    }
}

/// Choose `(rows, cols)` of crop windows covering `(h, w)`.
///
/// `h` and `w` are signed because upstream subtracts the margin pixels first
/// and lets the result go to zero or negative for very small images; the
/// resulting infinite or negative scales are load-bearing tie-breakers.
pub fn select_tiling(h: i64, w: i64, window: usize, max_crops: usize) -> (usize, usize) {
    let mut tilings: Vec<(usize, usize)> = Vec::new();
    for i in 1..=max_crops {
        for j in 1..=max_crops {
            if i * j <= max_crops {
                tilings.push((i, j));
            }
        }
    }
    tilings.sort_by_key(|&(a, b)| (a * b, a));

    let scales: Vec<f32> = tilings
        .iter()
        .map(|&(i, j)| {
            let rh = (i * window) as f32 / h as f32;
            let rw = (j * window) as f32 / w as f32;
            rh.min(rw)
        })
        .collect();

    if scales.iter().all(|&s| s < 1.0) {
        let mut best = 0usize;
        let mut best_scale = f32::NEG_INFINITY;
        for (i, &s) in scales.iter().enumerate() {
            if s > best_scale {
                best_scale = s;
                best = i;
            }
        }
        tilings[best]
    } else {
        let mut best = 0usize;
        let mut best_scale = f32::INFINITY;
        for (i, &s) in scales.iter().enumerate() {
            let s = if s < 1.0 { 1e9 } else { s };
            if s < best_scale {
                best_scale = s;
                best = i;
            }
        }
        tilings[best]
    }
}

/// Bilinear resample with `align_corners = false` and no antialiasing, matching
/// `torchvision.transforms.Resize(..., InterpolationMode.BILINEAR,
/// antialias=False)` (`interpolation: bilinear`, `antialias: false` in the
/// checkpoint's `preprocessor_config.json`).
///
/// Input and output are row-major RGB triples.
fn resize_bilinear(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return src.to_vec();
    }
    resize_bilinear_sampled(src_h, src_w, dst_h, dst_w, |i| src[i])
}

/// [`resize_bilinear`] over an 8-bit source, scaling each sample into `[0, 1]`
/// as it is read.
///
/// This exists so the first resize never materialises a full-resolution `f32`
/// copy of the decoded image: that copy is 12 bytes per source pixel and is
/// sized by whoever supplied the image, whereas the destination is already
/// bounded by `smart_resize`.
///
/// The result is bit-identical to converting the whole source first: the same
/// `v as f32 / 255.0` values are combined by the same weights in the same
/// order, so every output channel is the same IEEE expression.
fn resize_bilinear_u8(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return src.iter().map(|&v| v as f32 / 255.0).collect();
    }
    resize_bilinear_sampled(src_h, src_w, dst_h, dst_w, |i| src[i] as f32 / 255.0)
}

/// Shared body of [`resize_bilinear`] and [`resize_bilinear_u8`]. `sample`
/// returns the source channel at a flat row-major RGB index; everything else
/// (weights, edge clamping, channel order, the final clip) is identical for
/// both entry points.
fn resize_bilinear_sampled(
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    sample: impl Fn(usize) -> f32,
) -> Vec<f32> {
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;
    let mut out = vec![0.0f32; dst_h * dst_w * 3];

    for dy in 0..dst_h {
        let sy = source_index(dy, scale_y);
        let y0 = sy.floor();
        let ly = sy - y0;
        let y0 = (y0 as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);

        for dx in 0..dst_w {
            let sx = source_index(dx, scale_x);
            let x0 = sx.floor();
            let lx = sx - x0;
            let x0 = (x0 as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);

            let base00 = (y0 * src_w + x0) * 3;
            let base01 = (y0 * src_w + x1) * 3;
            let base10 = (y1 * src_w + x0) * 3;
            let base11 = (y1 * src_w + x1) * 3;
            let dst = (dy * dst_w + dx) * 3;

            for c in 0..3 {
                let top = sample(base00 + c) * (1.0 - lx) + sample(base01 + c) * lx;
                let bottom = sample(base10 + c) * (1.0 - lx) + sample(base11 + c) * lx;
                out[dst + c] = top * (1.0 - ly) + bottom * ly;
            }
        }
    }

    // Upstream clips the resampled image back into its input range; bilinear
    // cannot overshoot, but the clamp keeps the contract explicit.
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}

/// `area_pixel_compute_source_index` for `align_corners = false`.
#[inline]
fn source_index(dst: usize, scale: f32) -> f32 {
    let idx = scale * (dst as f32 + 0.5) - 0.5;
    if idx < 0.0 { 0.0 } else { idx }
}

/// Cut a `size x size` crop at `(y0, x0)`, zero-padding (and marking uncovered)
/// anything past the source edges.
fn extract_crop(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    y0: usize,
    x0: usize,
    size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut pixels = vec![0.0f32; size * size * 3];
    let mut mask = vec![0.0f32; size * size];
    for cy in 0..size {
        let sy = y0 + cy;
        if sy >= src_h {
            continue;
        }
        for cx in 0..size {
            let sx = x0 + cx;
            if sx >= src_w {
                continue;
            }
            let s = (sy * src_w + sx) * 3;
            let d = (cy * size + cx) * 3;
            pixels[d..d + 3].copy_from_slice(&src[s..s + 3]);
            mask[cy * size + cx] = 1.0;
        }
    }
    (pixels, mask)
}

/// Rearrange one square crop into `[h * w, patch * patch * 3]` patches, and its
/// mask into `[h * w]` (mean coverage per patch).
fn patchify_crop(
    pixels: &[f32],
    mask: &[f32],
    size: usize,
    patch: usize,
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let patch_dim = patch * patch * 3;
    let mut out = Vec::with_capacity(h * w * patch_dim);
    let mut mask_out = Vec::with_capacity(h * w);

    for ph in 0..h {
        for pw in 0..w {
            let mut covered = 0.0f32;
            for dy in 0..patch {
                let y = ph * patch + dy;
                for dx in 0..patch {
                    let x = pw * patch + dx;
                    let idx = y * size + x;
                    out.extend_from_slice(&pixels[idx * 3..idx * 3 + 3]);
                    covered += mask[idx];
                }
            }
            mask_out.push(covered / (patch * patch) as f32);
        }
    }
    (out, mask_out)
}

#[cfg(test)]
#[path = "jina_vlm_tests.rs"]
mod tests;
