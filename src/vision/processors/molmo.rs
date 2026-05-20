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

//! Molmo v1 Image Processor.
//!
//! Mirrors `references/mlx-vlm/mlx_vlm/models/molmo/processing_molmo.py`:
//! a single global (resized) image plus overlapping high-res crops selected by
//! `select_tiling`, each rearranged into 14x14 patches. It also produces the
//! `image_input_idx` array (where each image patch's pooled feature lands in the
//! image-token region) and the per-patch `image_masks` consumed by the
//! `pad_and_partial_pad` embedding.
//!
//! Differs from the Molmo2 processor (`processors::molmo2`): Molmo v1 emits the
//! global image FIRST then the crops, computes `image_input_idx`/`image_masks`
//! instead of pooling indices, and uses the `<im_start>/<im_patch>/<im_col>/
//! <im_end>` token vocabulary directly (no `<low_res_im_start>`).

const OPENAI_CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const OPENAI_CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Special token IDs used to build the image-token region. Defaults match
/// Molmo-7B's tokenizer (`<im_start>=152064` … `<im_col>=152067`).
#[derive(Debug, Clone, Copy)]
pub struct MolmoImageTokens {
    pub image_patch_id: i32,
    pub image_col_id: i32,
    pub image_start_id: i32,
    pub image_end_id: i32,
}

impl Default for MolmoImageTokens {
    fn default() -> Self {
        Self {
            image_start_id: 152064,
            image_end_id: 152065,
            image_patch_id: 152066,
            image_col_id: 152067,
        }
    }
}

/// Molmo v1 image processor configuration.
pub struct MolmoProcessor {
    pub base_image_size: (usize, usize), // (height, width), default (336, 336)
    pub max_crops: usize,
    pub overlap_margins: (usize, usize), // (left, right), default (4, 4)
    pub patch_size: usize,               // 14
    pub image_token_length_h: usize,     // 12
    pub image_token_length_w: usize,     // 12
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub tokens: MolmoImageTokens,
}

/// Output from preprocessing a single image.
pub struct MolmoProcessorOutput {
    pub pixel_values: Vec<f32>, // [n_crops, n_patches, patch_dim] flattened
    pub pixel_values_shape: [i32; 3],
    /// Image-token IDs (the `<im_start>…<im_end>` region) to prepend to the prompt.
    pub image_token_ids: Vec<i32>,
    /// `image_input_idx` flattened: where each image patch maps inside the
    /// image-token region. Negative entries (-1) are skipped during the merge.
    pub image_input_idx: Vec<i32>,
    pub image_input_idx_len: i32,
    /// `image_masks` flattened: [n_crops, n_patches] per-patch coverage in [0,1].
    pub image_masks: Vec<f32>,
    pub image_masks_shape: [i32; 2],
}

impl MolmoProcessor {
    pub fn new(
        max_crops: usize,
        overlap_margins: Option<(usize, usize)>,
        patch_size: Option<usize>,
        base_image_size: Option<(usize, usize)>,
        image_token_length: Option<(usize, usize)>,
        image_mean: Option<[f32; 3]>,
        image_std: Option<[f32; 3]>,
        tokens: MolmoImageTokens,
    ) -> Self {
        let (h, w) = image_token_length.unwrap_or((12, 12));
        Self {
            base_image_size: base_image_size.unwrap_or((336, 336)),
            max_crops,
            overlap_margins: overlap_margins.unwrap_or((4, 4)),
            patch_size: patch_size.unwrap_or(14),
            image_token_length_h: h,
            image_token_length_w: w,
            image_mean: image_mean.unwrap_or(OPENAI_CLIP_MEAN),
            image_std: image_std.unwrap_or(OPENAI_CLIP_STD),
            tokens,
        }
    }

    fn tokens_per_image(&self) -> usize {
        self.image_token_length_h * self.image_token_length_w
    }

    /// Preprocess a single image into crops, image-token IDs, `image_input_idx`,
    /// and `image_masks`. Faithful port of
    /// `MolmoImageProcessor.image_to_patches_and_tokens` + `build_image_input_idx`.
    pub fn preprocess_image(&self, image: &image::DynamicImage) -> MolmoProcessorOutput {
        let img = image.to_rgb8();
        let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

        let base_h = self.base_image_size.0;
        let base_w = self.base_image_size.1;
        let patch_d = self.patch_size;
        let crop_size = base_h;
        let image_base_patch_h = base_h / patch_d;
        let image_base_patch_w = base_w / patch_d;

        let (left_margin, right_margin) = self.overlap_margins;
        let total_margin_pixels = patch_d * (right_margin + left_margin);
        let crop_patches = base_h / patch_d;
        let crop_window_patches = crop_patches - (right_margin + left_margin);
        let crop_window_size = crop_window_patches * patch_d;

        let tiling = select_tiling(
            orig_h.saturating_sub(total_margin_pixels),
            orig_w.saturating_sub(total_margin_pixels),
            crop_window_size,
            self.max_crops,
        );

        let target_h = tiling.0 * crop_window_size + total_margin_pixels;
        let target_w = tiling.1 * crop_window_size + total_margin_pixels;

        // Resize + normalize the high-res source and its coverage mask.
        let (src, src_mask) = self.resize_and_pad(&img, target_h, target_w, orig_h, orig_w);

        let mut patches_arr: Vec<Vec<Vec<f32>>> = Vec::new(); // [crop][h*w][3]
        let mut mask_arr: Vec<Vec<bool>> = Vec::new(); // [crop][h*w]
        let mut patch_ordering: Vec<Vec<Vec<i32>>> = Vec::new(); // [crop][tok_h][tok_w]

        let mut on = 0i32;
        for i in 0..tiling.0 {
            let y0 = i * crop_window_size;
            let crop_y0 = if i == 0 { 0 } else { left_margin / 2 };
            let mut crop_h = image_base_patch_h as i64 - (right_margin + left_margin) as i64;
            if i == 0 {
                crop_h += left_margin as i64;
            }
            if i == tiling.0 - 1 {
                crop_h += right_margin as i64;
            }

            for j in 0..tiling.1 {
                let x0 = j * crop_window_size;
                let crop_x0 = if j == 0 { 0 } else { left_margin / 2 };
                let mut crop_w = image_base_patch_w as i64 - (right_margin + left_margin) as i64;
                if j == 0 {
                    crop_w += left_margin as i64;
                }
                if j == tiling.1 - 1 {
                    crop_w += right_margin as i64;
                }

                let pooled_w = (crop_w + 1) / 2;
                let pooled_h = (crop_h + 1) / 2;

                // ordering[pooled_h, pooled_w] = arange(on, on+pooled_h*pooled_w)
                // then pad to (tok_h, tok_w) with -1 at (crop_y0, crop_x0).
                let mut ordering =
                    vec![vec![-1i32; self.image_token_length_w]; self.image_token_length_h];
                let mut v = on;
                for py in 0..pooled_h as usize {
                    for px in 0..pooled_w as usize {
                        let ty = crop_y0 + py;
                        let tx = crop_x0 + px;
                        if ty < self.image_token_length_h && tx < self.image_token_length_w {
                            ordering[ty][tx] = v;
                        }
                        v += 1;
                    }
                }
                patch_ordering.push(ordering);
                on += (pooled_h * pooled_w) as i32;

                // Extract crop pixels + mask [crop_size, crop_size].
                let mut crop_px = vec![[0.0f32; 3]; crop_size * crop_size];
                let mut crop_mask = vec![false; crop_size * crop_size];
                for cy in 0..crop_size {
                    for cx in 0..crop_size {
                        let sy = y0 + cy;
                        let sx = x0 + cx;
                        let dst = cy * crop_size + cx;
                        if sy < target_h && sx < target_w {
                            crop_px[dst] = src[sy * target_w + sx];
                            crop_mask[dst] = src_mask[sy * target_w + sx];
                        }
                    }
                }
                // Rearrange into patches: [n_patches][patch_dim], mask mean -> [n_patches]
                let (p, m) = rearrange_crop(
                    &crop_px,
                    &crop_mask,
                    crop_size,
                    patch_d,
                    image_base_patch_h,
                    image_base_patch_w,
                );
                patches_arr.push(p);
                mask_arr.push(m);
            }
        }

        // Re-sort patch_ordering by transposing token grid (matches reference).
        let mut flat_ordering = self.resort_patch_ordering(&patch_ordering, tiling);

        // Build the high-res image-token IDs.
        let h_patches = tiling.0 * crop_window_patches + (right_margin + left_margin);
        let w_patches = tiling.1 * crop_window_patches + (right_margin + left_margin);
        let hi_tokens = self.build_crop_tokens(h_patches.div_ceil(2), w_patches.div_ceil(2));

        // Global (resized) image: [n_patches][patch_dim] + ordering 0..tokens_per_image.
        let (global_px, _global_mask) = self.resize_and_pad(&img, base_h, base_w, orig_h, orig_w);
        let global_patches = rearrange_global(
            &global_px,
            base_h,
            base_w,
            patch_d,
            image_base_patch_h,
            image_base_patch_w,
        );
        // Prepend global image to crops.
        // Patches: global image FIRST, then the high-res crops (reference order).
        patches_arr.insert(0, global_patches);
        let n_patches = image_base_patch_h * image_base_patch_w;

        // patch_ordering offset: high-res indices shift by tokens_per_image, then
        // global occupies [0, tokens_per_image).
        let tpi = self.tokens_per_image() as i32;
        for v in flat_ordering.iter_mut() {
            if *v >= 0 {
                *v += tpi;
            }
        }
        let mut full_ordering: Vec<i32> = (0..tpi).collect();
        full_ordering.extend(flat_ordering);

        // Global image tokens come first, then high-res tokens.
        let global_tokens =
            self.build_crop_tokens(self.image_token_length_h, self.image_token_length_w);
        let mut image_token_ids = global_tokens;
        image_token_ids.extend(hi_tokens);

        // image_input_idx: positions of <im_patch> tokens, reordered by full_ordering.
        let image_input_idx = self.build_image_input_idx(&image_token_ids, &full_ordering);

        // Flatten pixel values.
        let n_crops = patches_arr.len();
        let patch_dim = patch_d * patch_d * 3;
        let mut pixel_values: Vec<f32> = Vec::with_capacity(n_crops * n_patches * patch_dim);
        for crop in &patches_arr {
            for patch in crop {
                pixel_values.extend_from_slice(patch);
            }
        }

        // image_masks: the high-res crop masks, then a trailing -1 sentinel row.
        // Reference: `img_mask = np.pad(img_mask, [[0, 1], [0, 0]], -1)` — the -1
        // row is appended AFTER the crop masks (the global slot is offset by one,
        // mirroring upstream so output matches the mlx-vlm Python reference).
        let mut image_masks: Vec<f32> = Vec::with_capacity(n_crops * n_patches);
        for crop_mask in &mask_arr {
            for &b in crop_mask {
                image_masks.push(if b { 1.0 } else { 0.0 });
            }
        }
        image_masks.extend(std::iter::repeat_n(-1.0, n_patches));

        let idx_len = image_input_idx.len() as i32;
        MolmoProcessorOutput {
            pixel_values,
            pixel_values_shape: [n_crops as i32, n_patches as i32, patch_dim as i32],
            image_token_ids,
            image_input_idx,
            image_input_idx_len: idx_len,
            image_masks,
            image_masks_shape: [n_crops as i32, n_patches as i32],
        }
    }

    /// Build the `<im_start> [<im_patch>*w <im_col>]*h <im_end>` token block.
    fn build_crop_tokens(&self, h: usize, w: usize) -> Vec<i32> {
        let t = &self.tokens;
        let mut out = vec![t.image_start_id];
        for _ in 0..h {
            for _ in 0..w {
                out.push(t.image_patch_id);
            }
            out.push(t.image_col_id);
        }
        out.push(t.image_end_id);
        out
    }

    /// Transpose-resort the per-crop token grid into a flat patch ordering,
    /// mirroring the reference `patch_ordering[valid] = patch_ordering_rh[...]`.
    // The (i, y, j, x) traversal interleaves crop-grid and token-grid axes, so
    // `crop = i*tiling.1 + j` cannot be expressed as a simple iterator.
    #[allow(clippy::needless_range_loop)]
    fn resort_patch_ordering(
        &self,
        patch_ordering: &[Vec<Vec<i32>>],
        tiling: (usize, usize),
    ) -> Vec<i32> {
        let th = self.image_token_length_h;
        let tw = self.image_token_length_w;

        // Flatten in crop-row-major order.
        let mut flat: Vec<i32> = Vec::new();
        for crop in patch_ordering {
            for row in crop {
                flat.extend_from_slice(row);
            }
        }

        // patch_ordering_rh: reshape [tiling0, tiling1, th, tw] -> transpose
        // [0,2,1,3] -> flatten, keep only >= 0.
        let mut rh_valid: Vec<i32> = Vec::new();
        for i in 0..tiling.0 {
            for y in 0..th {
                for j in 0..tiling.1 {
                    for x in 0..tw {
                        let crop = i * tiling.1 + j;
                        let val = patch_ordering[crop][y][x];
                        if val >= 0 {
                            rh_valid.push(val);
                        }
                    }
                }
            }
        }

        // Assign rh_valid into the valid slots of `flat`, in order.
        let mut k = 0usize;
        for v in flat.iter_mut() {
            if *v >= 0 {
                *v = rh_valid[k];
                k += 1;
            }
        }
        flat
    }

    /// Port of `build_image_input_idx`: for each entry of `patch_order`, map to
    /// the position of its `<im_patch>` token within `image_token_ids`, or -100
    /// where the patch is padding.
    ///
    /// Reference:
    /// ```text
    /// positions   = nonzero(tokens == <im_patch>)
    /// sorted[patch_order[valid]] = arange(n_valid)   # scatter into token slots
    /// ex          = -1; ex[valid] = sorted
    /// out         = positions[ex * (ex>=0)] * (ex>=0) - 100 * (ex<0)
    /// ```
    fn build_image_input_idx(&self, image_token_ids: &[i32], patch_order: &[i32]) -> Vec<i32> {
        let patch_id = self.tokens.image_patch_id;

        // Positions of <im_patch> tokens in the token stream.
        let positions: Vec<i32> = image_token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| if t == patch_id { Some(i as i32) } else { None })
            .collect();
        let n_tokens = positions.len();

        let valid: Vec<bool> = patch_order.iter().map(|&v| v >= 0).collect();

        // sorted[patch_order[k]] = running_valid_index, for each valid k.
        let mut sorted = vec![0i32; n_tokens];
        let mut next = 0i32;
        for (k, &v) in patch_order.iter().enumerate() {
            if valid[k] {
                let slot = v as usize;
                if slot < n_tokens {
                    sorted[slot] = next;
                }
                next += 1;
            }
        }

        // ex[k] = sorted at the running valid count; -1 where invalid.
        // out[k] = positions[ex[k]] if valid else -100.
        let mut result = Vec::with_capacity(patch_order.len());
        let mut seen_valid = 0usize;
        for &is_valid in &valid {
            if is_valid {
                let ex = sorted[seen_valid.min(n_tokens.saturating_sub(1))];
                let pos = positions.get(ex as usize).copied().unwrap_or(0);
                result.push(pos);
                seen_valid += 1;
            } else {
                result.push(-100);
            }
        }
        result
    }

    /// Resize-and-pad an image to (target_h, target_w) preserving aspect ratio
    /// (bilinear, centered pad). Returns row-major normalized RGB and a coverage
    /// mask (true where real image pixels live, false in the pad border).
    fn resize_and_pad(
        &self,
        img: &image::RgbImage,
        target_h: usize,
        target_w: usize,
        orig_h: usize,
        orig_w: usize,
    ) -> (Vec<[f32; 3]>, Vec<bool>) {
        let scale = (target_h as f32 / orig_h as f32).min(target_w as f32 / orig_w as f32);
        let scaled_h = (orig_h as f32 * scale) as usize;
        let scaled_w = (orig_w as f32 * scale) as usize;
        let scaled_h = scaled_h.clamp(1, target_h);
        let scaled_w = scaled_w.clamp(1, target_w);

        let resized = image::imageops::resize(
            img,
            scaled_w as u32,
            scaled_h as u32,
            image::imageops::FilterType::Triangle,
        );

        let top = (target_h - scaled_h) / 2;
        let left = (target_w - scaled_w) / 2;

        let mut out = vec![[0.0f32; 3]; target_h * target_w];
        let mut mask = vec![false; target_h * target_w];
        for y in 0..scaled_h {
            for x in 0..scaled_w {
                let px = resized.get_pixel(x as u32, y as u32);
                let dy = top + y;
                let dx = left + x;
                let dst = dy * target_w + dx;
                out[dst] = [
                    (px[0] as f32 / 255.0 - self.image_mean[0]) / self.image_std[0],
                    (px[1] as f32 / 255.0 - self.image_mean[1]) / self.image_std[1],
                    (px[2] as f32 / 255.0 - self.image_mean[2]) / self.image_std[2],
                ];
                mask[dst] = true;
            }
        }
        (out, mask)
    }
}

/// `select_tiling`: choose (rows, cols) of crops that best covers (h, w).
fn select_tiling(h: usize, w: usize, patch_size: usize, max_num_patches: usize) -> (usize, usize) {
    let mut tilings: Vec<(usize, usize)> = Vec::new();
    for i in 1..=max_num_patches {
        for j in 1..=max_num_patches {
            if i * j <= max_num_patches {
                tilings.push((i, j));
            }
        }
    }
    tilings.sort_by_key(|&(a, b)| (a * b, a));

    let orig = [h as f32, w as f32];
    let scales: Vec<f32> = tilings
        .iter()
        .map(|&(i, j)| {
            let rh = (i * patch_size) as f32;
            let rw = (j * patch_size) as f32;
            (rh / orig[0]).min(rw / orig[1])
        })
        .collect();

    let all_less_than_one = scales.iter().all(|&s| s < 1.0);
    if all_less_than_one {
        // argmax
        let mut bi = 0;
        let mut bs = f32::MIN;
        for (i, &s) in scales.iter().enumerate() {
            if s > bs {
                bs = s;
                bi = i;
            }
        }
        tilings[bi]
    } else {
        // argmin over scales >= 1 (others set to +inf)
        let mut bi = 0;
        let mut bs = f32::MAX;
        for (i, &s) in scales.iter().enumerate() {
            let s = if s < 1.0 { 1e9 } else { s };
            if s < bs {
                bs = s;
                bi = i;
            }
        }
        tilings[bi]
    }
}

/// Rearrange one crop's pixels into `[h*w][dh*dw*3]` patches and average its
/// mask into `[h*w]` (then thresholded later). `crop` is row-major [size*size].
fn rearrange_crop(
    crop: &[[f32; 3]],
    mask: &[bool],
    crop_size: usize,
    patch: usize,
    h: usize,
    w: usize,
) -> (Vec<Vec<f32>>, Vec<bool>) {
    let mut patches = Vec::with_capacity(h * w);
    let mut mask_out = Vec::with_capacity(h * w);
    for ph in 0..h {
        for pw in 0..w {
            let mut vals = Vec::with_capacity(patch * patch * 3);
            let mut covered = 0usize;
            for dy in 0..patch {
                for dx in 0..patch {
                    let y = ph * patch + dy;
                    let x = pw * patch + dx;
                    let idx = y * crop_size + x;
                    if idx < crop.len() {
                        vals.extend_from_slice(&crop[idx]);
                        if mask[idx] {
                            covered += 1;
                        }
                    } else {
                        vals.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }
            }
            patches.push(vals);
            // Mean coverage > 0.5 marks the patch as real (matches mean(mask)).
            mask_out.push(covered * 2 >= patch * patch);
        }
    }
    (patches, mask_out)
}

/// Rearrange the resized global image into `[h*w][dh*dw*3]` patches.
fn rearrange_global(
    img: &[[f32; 3]],
    img_h: usize,
    img_w: usize,
    patch: usize,
    h: usize,
    w: usize,
) -> Vec<Vec<f32>> {
    let _ = (img_h, img_w);
    let mut patches = Vec::with_capacity(h * w);
    for ph in 0..h {
        for pw in 0..w {
            let mut vals = Vec::with_capacity(patch * patch * 3);
            for dy in 0..patch {
                for dx in 0..patch {
                    let y = ph * patch + dy;
                    let x = pw * patch + dx;
                    let idx = y * (w * patch) + x;
                    if idx < img.len() {
                        vals.extend_from_slice(&img[idx]);
                    } else {
                        vals.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }
            }
            patches.push(vals);
        }
    }
    patches
}
