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

//! The Qwen-Image-2.1 prompt conditioner, a port of `text_encoder.py`
//! (mlx-vlm `73d366ec`): the checkpoint's Qwen3-VL (`<root>/text_encoder`,
//! loaded through mlxcel's Qwen3-VL loader) returns its last decoder layer's
//! hidden states *before* the final RMSNorm
//! ([`crate::models::qwen3_vl`]'s `forward_hidden_pre_norm`), and the leading
//! system-turn tokens are dropped.
//!
//! The tokenizer and the image processor config live in `<root>/processor`.
//! `drop_idx` is the token count of the system prefix, tokenized on its own
//! without special tokens, as upstream computes it at load. The raw template
//! string is tokenized directly (not through the chat template).
//!
//! An edit puts `<imageN><|vision_start|><|image_pad|><|vision_end|>` per
//! reference (joined by a space) in front of the instruction; each reference
//! is composited onto white through its alpha channel (PIL `paste` with the
//! alpha as mask) and handed to the Qwen3-VL processor at its own size, which
//! the pipeline has already made a multiple of 32. The `<|image_pad|>`
//! positions of the trimmed prompt are returned as the image-slot mask the
//! transformer replaces with VAE tokens.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, RgbaImage};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MAX_PROMPT_TOKENS;
use super::reference::composite_on_white;
use crate::imagegen::mage_flow::conditioner::{VISION_MARKERS, pixel_bounds, smart_resize};
use crate::imagegen::pipeline::InvalidImageRequest;
use crate::tokenizer::MlxcelTokenizer;

pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
/// `patch_size * merge_size` of the Qwen3-VL vision tower.
const RESIZE_FACTOR: u32 = 32;

/// The system turn the conditioner drops.
#[must_use]
pub fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// Upstream `PROMPT_TEMPLATE_T2I.format(prompt)`.
#[must_use]
pub fn generation_text(prompt: &str) -> String {
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// Upstream `PROMPT_TEMPLATE_TI2I.format(prefix, prompt)` for `references`.
#[must_use]
pub fn edit_text(prompt: &str, references: usize) -> String {
    let prefix = (1..=references)
        .map(|i| format!("<image{i}>{IMAGE_PLACEHOLDER}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{}<|im_start|>user\n{prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

fn tokenize(tokenizer: &MlxcelTokenizer, text: &str, special: bool) -> Result<Vec<i32>> {
    tokenizer
        .encode(text, special)?
        .into_iter()
        .map(|id| i32::try_from(id).context("token id out of range"))
        .collect()
}

/// Upstream `drop_idx`: the system prefix's token count without special
/// tokens.
pub fn drop_idx(tokenizer: &MlxcelTokenizer) -> Result<usize> {
    Ok(tokenize(tokenizer, &system_prefix(), false)?.len())
}

/// Token ids of a generation prompt, truncated as upstream's tokenizer call
/// (`max_length = 1024 + drop_idx`).
pub fn generation_ids(tokenizer: &MlxcelTokenizer, prompt: &str, drop: usize) -> Result<Vec<i32>> {
    let mut ids = tokenize(tokenizer, &generation_text(prompt), true)?;
    ids.truncate(MAX_PROMPT_TOKENS + drop);
    Ok(ids)
}

/// Reject an edit instruction that carries a Qwen3-VL vision marker (the
/// template adds one block per reference itself, and a stray marker would
/// desynchronize the image features from their positions) with an
/// [`InvalidImageRequest`]. Needs no tokenizer, so request validation runs it
/// before the conditioner loads.
pub fn check_edit_markers(prompt: &str) -> Result<()> {
    if let Some(marker) = VISION_MARKERS.iter().find(|m| prompt.contains(**m)) {
        return Err(InvalidImageRequest(format!(
            "Qwen-Image edit prompt must not contain the vision marker {marker}"
        ))
        .into());
    }
    Ok(())
}

/// Reject an edit instruction that carries a Qwen3-VL vision marker
/// ([`check_edit_markers`]) or is longer than [`MAX_PROMPT_TOKENS`]
/// (upstream's edit path does not truncate, and cutting after placeholder
/// expansion would drop visual tokens). Both are [`InvalidImageRequest`]s.
pub fn check_edit_prompt(tokenizer: &MlxcelTokenizer, prompt: &str) -> Result<()> {
    check_edit_markers(prompt)?;
    let tokens = tokenizer.encode(prompt, false)?.len();
    if tokens > MAX_PROMPT_TOKENS {
        return Err(InvalidImageRequest(format!(
            "Qwen-Image edit prompt is {tokens} tokens; the maximum is {MAX_PROMPT_TOKENS}"
        ))
        .into());
    }
    Ok(())
}

/// Reject a resized edit reference of `width x height` that the Qwen3-VL
/// processor would resize again (past its 200:1 aspect limit, or outside its
/// `[min_pixels, max_pixels]` bounds) with an [`InvalidImageRequest`]. Pure,
/// so request validation runs it on the predicted sizes before any load.
pub fn check_reference_size(
    width: u32,
    height: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<()> {
    let (th, tw) = smart_resize(height, width, RESIZE_FACTOR, min_pixels, max_pixels)
        .map_err(|e| InvalidImageRequest(format!("Qwen-Image reference: {e:#}")))?;
    if (tw, th) != (width, height) {
        return Err(InvalidImageRequest(format!(
            "Qwen-Image reference of {width}x{height} would be resized to {tw}x{th} by the Qwen3-VL processor; use another output_resolution"
        ))
        .into());
    }
    Ok(())
}

/// One encoded prompt: `[1, L - drop_idx, hidden]` bf16 hidden states and,
/// for an edit, the `<|image_pad|>` mask of the same positions.
pub struct Encoded {
    pub hidden: UniquePtr<MlxArray>,
    pub image_pad_mask: Option<Vec<bool>>,
}

/// The loaded Qwen3-VL conditioner.
pub struct TextEncoder {
    vlm: crate::vision::Qwen3VLModel,
    tokenizer: MlxcelTokenizer,
    drop_idx: usize,
    min_pixels: u64,
    max_pixels: u64,
}

impl TextEncoder {
    /// Load `<root>/text_encoder` with the tokenizer of `<root>/processor`.
    pub fn load(root: &Path) -> Result<Self> {
        let dir = root.join("text_encoder");
        let loaded = crate::loading::load_qwen3_vl(&dir).with_context(|| {
            format!(
                "failed to load the Qwen-Image text encoder {}",
                dir.display()
            )
        })?;
        let crate::LoadedModel::Qwen3VL(mut vlm) = loaded else {
            bail!("{} did not load as a Qwen3-VL model", dir.display());
        };
        let processor_dir = root.join("processor");
        let (min_pixels, max_pixels) = pixel_bounds(&processor_dir);
        // The references reach the processor at their final size; these keep
        // its own resize rule from moving them.
        vlm.processor.min_pixels = usize::try_from(min_pixels).unwrap_or(usize::MAX);
        vlm.processor.max_pixels = usize::try_from(max_pixels).unwrap_or(usize::MAX);
        let tokenizer = crate::tokenizer::load_tokenizer(&processor_dir).with_context(|| {
            format!(
                "failed to load the tokenizer in {}",
                processor_dir.display()
            )
        })?;
        let drop_idx = drop_idx(&tokenizer)?;
        Ok(Self {
            vlm,
            tokenizer,
            drop_idx,
            min_pixels,
            max_pixels,
        })
    }

    #[must_use]
    pub fn tokenizer(&self) -> &MlxcelTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn drop_idx(&self) -> usize {
        self.drop_idx
    }

    /// Pre-norm hidden states of `ids` with the first `drop_idx` removed.
    fn hidden_tail(
        &self,
        ids: &[i32],
        embeddings: Option<&MlxArray>,
        empty: &str,
    ) -> Result<UniquePtr<MlxArray>> {
        let drop = self.drop_idx;
        if ids.len() <= drop {
            return Err(InvalidImageRequest(empty.to_string()).into());
        }
        let len = i32::try_from(ids.len()).context("prompt too long")?;
        let input_ids = mlxcel_core::from_slice_i32(ids, &[1, len]);
        let text_model = &self.vlm.text_model;
        let mut caches = text_model.make_caches();
        let hidden = text_model.forward_hidden_pre_norm(&input_ids, embeddings, &mut caches, None);
        let width = *mlxcel_core::array_shape(&hidden).last().context("rank")?;
        let tail = mlxcel_core::slice(&hidden, &[0, drop as i32, 0], &[1, len, width]);
        let tail = mlxcel_core::astype(&tail, mlxcel_core::dtype::BFLOAT16);
        mlxcel_core::try_eval(&tail).map_err(|e| anyhow::anyhow!("Qwen-Image conditioner: {e}"))?;
        Ok(tail)
    }

    /// Token ids of a generation prompt.
    pub fn generation_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        generation_ids(&self.tokenizer, prompt, self.drop_idx)
    }

    /// `QwenImageTextEncoder.encode` (cast to bf16 as the pipeline does).
    pub fn encode(&self, prompt: &str) -> Result<Encoded> {
        let ids = self.generation_ids(prompt)?;
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        let hidden = self.hidden_tail(
            &ids,
            None,
            "Qwen-Image prompt was empty after template trimming",
        )?;
        Ok(Encoded {
            hidden,
            image_pad_mask: None,
        })
    }

    /// The expanded token ids, patch rows and grids of an edit prompt over
    /// `references` (already sized to multiples of 32).
    pub fn edit_inputs(
        &self,
        prompt: &str,
        references: &[RgbaImage],
    ) -> Result<(Vec<i32>, UniquePtr<MlxArray>, Vec<(i32, i32, i32)>)> {
        ensure!(
            !references.is_empty(),
            "At least one reference image is required"
        );
        check_edit_prompt(&self.tokenizer, prompt)?;
        let refs: Vec<DynamicImage> = references
            .iter()
            .map(|r| DynamicImage::ImageRgb8(composite_on_white(r)))
            .collect();
        for image in &refs {
            check_reference_size(
                image.width(),
                image.height(),
                self.min_pixels,
                self.max_pixels,
            )?;
        }
        let (pixel_values, grid_thw) = self.vlm.processor.preprocess_with_grid(&refs);
        let patch = self.vlm.processor.patch_size as u32;
        for (image, &(t, gh, gw)) in refs.iter().zip(&grid_thw) {
            ensure!(
                t == 1 && gh as u32 * patch == image.height() && gw as u32 * patch == image.width(),
                "the Qwen3-VL processor resized a Qwen-Image reference to grid {:?}",
                (t, gh, gw)
            );
        }
        let merge = self.vlm.processor.spatial_merge_size.max(1) as i32;
        let counts: Vec<usize> = grid_thw
            .iter()
            .map(|&(t, h, w)| (t * (h / merge) * (w / merge)) as usize)
            .collect();
        let ids = tokenize(&self.tokenizer, &edit_text(prompt, references.len()), true)?;
        let mask = vec![1; ids.len()];
        let (input_ids, _) = crate::models::qwen3_vl_embedding::expand_image_placeholders(
            &ids,
            &mask,
            self.vlm.image_token_id,
            &counts,
        )?;
        Ok((input_ids, pixel_values, grid_thw))
    }

    /// `QwenImageTextEncoder.encode_edit`: hidden states and image-slot mask
    /// of the trimmed prompt.
    pub fn encode_edit(&self, prompt: &str, references: &[RgbaImage]) -> Result<Encoded> {
        let (ids, pixel_values, grid_thw) = self.edit_inputs(prompt, references)?;
        let len = i32::try_from(ids.len()).context("prompt too long")?;
        let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, len]);
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        // Sets the M-RoPE positions and the DeepStack features the forward reads.
        let merged = self
            .vlm
            .get_input_embeddings(&input_ids, &pixel_values, &grid_thw);
        let result = self.hidden_tail(
            &ids,
            Some(&merged.inputs_embeds),
            "Qwen-Image edit prompt was empty after template trimming",
        );
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        let image_token = self.vlm.image_token_id;
        let image_pad_mask = ids
            .iter()
            .skip(self.drop_idx)
            .map(|&id| id == image_token)
            .collect();
        Ok(Encoded {
            hidden: result?,
            image_pad_mask: Some(image_pad_mask),
        })
    }
}

#[cfg(test)]
#[path = "text_encoder_tests.rs"]
pub(crate) mod tests;
