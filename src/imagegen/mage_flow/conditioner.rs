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

//! The Mage-Flow conditioner, a port of `text_encoder.py` (mlx-vlm
//! `73d366ec`): the final-norm hidden states of the checkpoint's own
//! Qwen3-VL (`<root>/text_encoder`), loaded through mlxcel's generative
//! Qwen3-VL loader, with the template prefix trimmed.
//!
//! Generation wraps the prompt in [`GENERATION_TEMPLATE`] and drops the first
//! 34 positions; editing puts one `Image N: <vision block>` per reference in
//! front of the instruction, wraps it in [`EDIT_TEMPLATE`], and drops 64.
//! Each edit reference is shrunk to `vl_cond_long_edge` on its long side (PIL
//! bicubic) and then sized by upstream's numpy Qwen3-VL processor rule
//! (`_smart_resize_image`, factor 32, PIL bicubic) before patching.
//!
//! Two request bounds upstream does not have keep the edit conditioner's
//! input bounded (the vision tower runs inside the shared Qwen3-VL embedding
//! path, whose evaluation is not fallible): the processor pixel ceiling is
//! capped at [`MAX_VL_COND_PIXELS`] per reference, and an edit instruction
//! longer than [`MAX_SEQUENCE_LENGTH`] tokens is a `400` (generation truncates
//! instead, as upstream does).

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MAX_VL_COND_PIXELS;
use crate::imagegen::pipeline::InvalidImageRequest;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::dense::preprocess::{PillowResizeFilter, resize_pillow_rgb8};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

pub const GENERATION_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
pub const EDIT_TEMPLATE: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Template positions dropped from a generation prompt.
pub const GENERATION_DROP: usize = 34;
/// Template positions dropped from an edit prompt.
pub const EDIT_DROP: usize = 64;
/// Upstream `MageFlowRuntimeConfig.max_sequence_length`.
pub const MAX_SEQUENCE_LENGTH: usize = 2048;
/// `patch_size * merge_size` of the Qwen3-VL vision tower.
const RESIZE_FACTOR: u32 = 32;

/// The generation prompt text.
#[must_use]
pub fn generation_text(prompt: &str) -> String {
    GENERATION_TEMPLATE.replacen("{}", prompt, 1)
}

/// The edit prompt text for `references` images.
#[must_use]
pub fn edit_text(prompt: &str, references: usize) -> String {
    let prefix: String = (1..=references)
        .map(|i| format!("Image {i}: {IMAGE_PLACEHOLDER}"))
        .collect();
    EDIT_TEMPLATE.replacen("{}", &format!("{prefix}{prompt}"), 1)
}

/// Upstream `resize_long_edge`: shrink so the long side is `maximum` (PIL
/// bicubic, Python's half-to-even `round`); smaller images and `None` keep
/// their size.
pub fn resize_long_edge(image: &RgbImage, maximum: Option<u32>) -> Result<RgbImage> {
    let (w, h) = image.dimensions();
    let long = w.max(h);
    let Some(maximum) = maximum.filter(|&m| m > 0 && long > m) else {
        return Ok(image.clone());
    };
    let scale = f64::from(maximum) / f64::from(long);
    let side = |v: u32| ((f64::from(v) * scale).round_ties_even() as u32).max(1);
    resize_pillow_rgb8(image, side(w), side(h), PillowResizeFilter::Bicubic)
}

/// Upstream `_smart_resize_image` (HF `smart_resize`): `(height, width)`
/// multiples of `factor` with an area in `[min_pixels, max_pixels]`.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32)> {
    ensure!(height > 0 && width > 0, "image size must be positive");
    let (h, w) = (f64::from(height), f64::from(width));
    let ratio = h.max(w) / h.min(w);
    if ratio > 200.0 {
        bail!("absolute aspect ratio must be smaller than 200, got {ratio}");
    }
    let f = f64::from(factor);
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// The processor inputs of one edit prompt.
pub struct EditInputs {
    /// Token ids with each `<|image_pad|>` expanded to its visual tokens.
    pub input_ids: Vec<i32>,
    /// `[sum(t*h*w), 3*2*16*16]` f32 patch rows.
    pub pixel_values: UniquePtr<MlxArray>,
    pub grid_thw: Vec<(i32, i32, i32)>,
}

fn tokenize(tokenizer: &MlxcelTokenizer, text: &str) -> Result<Vec<i32>> {
    tokenizer
        .encode(text, true)?
        .into_iter()
        .map(|id| i32::try_from(id).context("token id out of range"))
        .collect()
}

/// Token ids of a generation prompt, truncated as upstream's tokenizer call
/// does (`max_length = 2048 + 34`).
pub fn generation_ids(tokenizer: &MlxcelTokenizer, prompt: &str) -> Result<Vec<i32>> {
    let mut ids = tokenize(tokenizer, &generation_text(prompt))?;
    ids.truncate(MAX_SEQUENCE_LENGTH + GENERATION_DROP);
    Ok(ids)
}

/// Reject an edit instruction longer than [`MAX_SEQUENCE_LENGTH`] tokens with
/// an [`InvalidImageRequest`]. Upstream's edit path does not truncate, and
/// truncating after placeholder expansion would cut visual tokens, so an
/// over-long instruction is refused before any reference is processed.
pub fn check_edit_prompt_length(tokenizer: &MlxcelTokenizer, prompt: &str) -> Result<()> {
    let tokens = tokenizer.encode(prompt, false)?.len();
    if tokens > MAX_SEQUENCE_LENGTH {
        return Err(InvalidImageRequest(format!(
            "Mage-Flow edit prompt is {tokens} tokens; the maximum is {MAX_SEQUENCE_LENGTH}"
        ))
        .into());
    }
    Ok(())
}

/// The processor pixel ceiling of an edit reference: the checkpoint's own
/// `max_pixels`, capped at [`MAX_VL_COND_PIXELS`].
#[must_use]
pub fn reference_max_pixels(config_max_pixels: u64) -> u64 {
    config_max_pixels.min(MAX_VL_COND_PIXELS)
}

/// What the edit path needs besides the model: the Qwen3-VL processor and
/// tokenizer, the image token and the processor's pixel bounds.
pub struct EditPreprocessor<'a> {
    pub processor: &'a Qwen2VLProcessor,
    pub tokenizer: &'a MlxcelTokenizer,
    pub image_token_id: i32,
    /// `size.shortest_edge` / `size.longest_edge` of the preprocessor config.
    pub min_pixels: u64,
    pub max_pixels: u64,
}

impl EditPreprocessor<'_> {
    /// The reference images the vision tower sees: long edge first, then the
    /// processor's own size rule.
    pub fn reference_images(
        &self,
        images: &[DynamicImage],
        long_edge: Option<u32>,
    ) -> Result<Vec<RgbImage>> {
        images
            .iter()
            .map(|image| {
                let image = resize_long_edge(&image.to_rgb8(), long_edge)?;
                let (w, h) = image.dimensions();
                // An unusable reference shape is the client's fault.
                let (th, tw) = smart_resize(h, w, RESIZE_FACTOR, self.min_pixels, self.max_pixels)
                    .map_err(|e| InvalidImageRequest(format!("Mage-Flow reference: {e:#}")))?;
                if (tw, th) == (w, h) {
                    Ok(image)
                } else {
                    resize_pillow_rgb8(&image, tw, th, PillowResizeFilter::Bicubic)
                }
            })
            .collect()
    }

    /// The processor outputs of an edit prompt (`Qwen3VLProcessor.__call__`).
    pub fn inputs(
        &self,
        prompt: &str,
        images: &[DynamicImage],
        long_edge: Option<u32>,
    ) -> Result<EditInputs> {
        ensure!(
            !images.is_empty(),
            "At least one reference image is required"
        );
        check_edit_prompt_length(self.tokenizer, prompt)?;
        let refs: Vec<DynamicImage> = self
            .reference_images(images, long_edge)?
            .into_iter()
            .map(DynamicImage::ImageRgb8)
            .collect();
        let (pixel_values, grid_thw) = self.processor.preprocess_with_grid(&refs);
        let patch = self.processor.patch_size as u32;
        for (image, &(t, gh, gw)) in refs.iter().zip(&grid_thw) {
            // The processor must keep the size chosen above.
            ensure!(
                t == 1 && gh as u32 * patch == image.height() && gw as u32 * patch == image.width(),
                "the Qwen3-VL processor resized a Mage-Flow reference to grid {:?}",
                (t, gh, gw)
            );
        }
        let merge = self.processor.spatial_merge_size.max(1) as i32;
        let counts: Vec<usize> = grid_thw
            .iter()
            .map(|&(t, h, w)| (t * (h / merge) * (w / merge)) as usize)
            .collect();
        let ids = tokenize(self.tokenizer, &edit_text(prompt, images.len()))?;
        let mask = vec![1; ids.len()];
        let (input_ids, _) = crate::models::qwen3_vl_embedding::expand_image_placeholders(
            &ids,
            &mask,
            self.image_token_id,
            &counts,
        )?;
        Ok(EditInputs {
            input_ids,
            pixel_values,
            grid_thw,
        })
    }
}

/// The loaded Qwen3-VL conditioner.
pub struct Conditioner {
    vlm: crate::vision::Qwen3VLModel,
    tokenizer: MlxcelTokenizer,
    min_pixels: u64,
    max_pixels: u64,
}

/// `size.shortest_edge` / `size.longest_edge` of the preprocessor config
/// (upstream maps them to `min_pixels` / `max_pixels`), else upstream's
/// processor defaults.
pub fn pixel_bounds(dir: &Path) -> (u64, u64) {
    let config: Option<serde_json::Value> =
        std::fs::read_to_string(dir.join("preprocessor_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
    let read = |nested: &str, flat: &str| {
        config.as_ref().and_then(|c| {
            c.get("size")
                .and_then(|s| s.get(nested))
                .or_else(|| c.get(flat))
                .and_then(serde_json::Value::as_u64)
        })
    };
    (
        read("shortest_edge", "min_pixels").unwrap_or(56 * 56),
        read("longest_edge", "max_pixels").unwrap_or(14 * 14 * 4 * 1280),
    )
}

impl Conditioner {
    /// Load `<root>/text_encoder`.
    pub fn load(root: &Path) -> Result<Self> {
        let dir = root.join("text_encoder");
        let loaded = crate::loading::load_qwen3_vl(&dir).with_context(|| {
            format!(
                "failed to load the Mage-Flow text encoder {}",
                dir.display()
            )
        })?;
        let crate::LoadedModel::Qwen3VL(mut vlm) = loaded else {
            bail!("{} did not load as a Qwen3-VL model", dir.display());
        };
        let (min_pixels, max_pixels) = pixel_bounds(&dir);
        let max_pixels = reference_max_pixels(max_pixels);
        let min_pixels = min_pixels.min(max_pixels);
        // The images reach the processor at their final size already; these
        // bounds keep its own resize rule from moving them.
        vlm.processor.min_pixels = usize::try_from(min_pixels).unwrap_or(usize::MAX);
        vlm.processor.max_pixels = usize::try_from(max_pixels).unwrap_or(usize::MAX);
        let tokenizer = crate::tokenizer::load_tokenizer(&dir)
            .with_context(|| format!("failed to load the tokenizer in {}", dir.display()))?;
        Ok(Self {
            vlm,
            tokenizer,
            min_pixels,
            max_pixels,
        })
    }

    fn preprocessor(&self) -> EditPreprocessor<'_> {
        EditPreprocessor {
            processor: &self.vlm.processor,
            tokenizer: &self.tokenizer,
            image_token_id: self.vlm.image_token_id,
            min_pixels: self.min_pixels,
            max_pixels: self.max_pixels,
        }
    }

    /// Final hidden states of `ids` with the first `drop` positions removed.
    fn hidden_tail(
        &self,
        ids: &[i32],
        embeddings: Option<&MlxArray>,
        drop: usize,
        empty: &str,
    ) -> Result<UniquePtr<MlxArray>> {
        if ids.len() <= drop {
            bail!("{empty}");
        }
        let len = i32::try_from(ids.len()).context("prompt too long")?;
        let input_ids = mlxcel_core::from_slice_i32(ids, &[1, len]);
        let text_model = &self.vlm.text_model;
        let mut caches = text_model.make_caches();
        let hidden = text_model.forward_hidden(&input_ids, embeddings, &mut caches, None);
        let width = *mlxcel_core::array_shape(&hidden).last().context("rank")?;
        let tail = mlxcel_core::slice(&hidden, &[0, drop as i32, 0], &[1, len, width]);
        mlxcel_core::try_eval(&tail).map_err(|e| anyhow::anyhow!("Mage-Flow conditioner: {e}"))?;
        Ok(tail)
    }

    /// `MageFlowTextEncoder.encode`: `[1, L - 34, hidden]`.
    pub fn encode(&self, prompt: &str) -> Result<UniquePtr<MlxArray>> {
        let ids = generation_ids(&self.tokenizer, prompt)?;
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        self.hidden_tail(
            &ids,
            None,
            GENERATION_DROP,
            "Mage-Flow prompt was empty after template trimming",
        )
    }

    /// `MageFlowTextEncoder.encode_edit`: `[1, L - 64, hidden]`.
    pub fn encode_edit(
        &self,
        prompt: &str,
        images: &[DynamicImage],
        long_edge: Option<u32>,
    ) -> Result<UniquePtr<MlxArray>> {
        let inputs = self.preprocessor().inputs(prompt, images, long_edge)?;
        let len = i32::try_from(inputs.input_ids.len()).context("prompt too long")?;
        let input_ids = mlxcel_core::from_slice_i32(&inputs.input_ids, &[1, len]);
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        // Sets the M-RoPE positions and the DeepStack features the forward reads.
        let merged =
            self.vlm
                .get_input_embeddings(&input_ids, &inputs.pixel_values, &inputs.grid_thw);
        let result = self.hidden_tail(
            &inputs.input_ids,
            Some(&merged.inputs_embeds),
            EDIT_DROP,
            "Mage-Flow edit prompt was empty after template trimming",
        );
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        result
    }
}

#[cfg(test)]
#[path = "conditioner_tests.rs"]
pub(crate) mod tests;
