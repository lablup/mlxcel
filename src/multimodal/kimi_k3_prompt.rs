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

//! Kimi K3 image prompt placement at the token level (issue #1342).
//!
//! Each image contributes one block to the prompt:
//!
//! ```text
//! <|media_begin|>image {w}x{h}<|media_content|> <|media_pad|> x n <|media_end|>
//! ```
//!
//! with `w x h` the original pixel size and `n = grid_h * grid_w / 4` from the
//! navit rule (`processors::kimi_k3`). The four bracketing tokens are control
//! ids, so the block is assembled from ids rather than re-encoded from text.
//!
//! Two callers reach this module:
//! - The server renders the block inside the XTML prompt
//!   (`server::kimi_k3_chat`) before the worker runs, so the worker's prompt
//!   already carries every `<|media_pad|>` run. Here the runs are verified
//!   against the grids the processor actually produced.
//! - `mlxcel generate --image` hands a raw prompt with no placeholders; the
//!   blocks are spliced in front of it, in image order.
//!
//! Used by: `multimodal::vlm_runtime` (`VlmRuntimeRef::KimiK3`),
//! `loading::vlm_kimi_k3` (reading the media ids).

use std::path::Path;

use anyhow::{Result, anyhow};
use image::DynamicImage;

use crate::vision::kimi_k3_vl::{KimiK3MediaTokenIds, KimiK3VLModel};
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::kimi_k3::KimiK3PreparedImage;

/// The spellings of the four media control tokens.
pub const MEDIA_BEGIN: &str = "<|media_begin|>";
pub const MEDIA_CONTENT: &str = "<|media_content|>";
pub const MEDIA_PAD: &str = "<|media_pad|>";
pub const MEDIA_END: &str = "<|media_end|>";

/// Resolve the media control ids from `<model_dir>/tokenizer_config.json`
/// (`added_tokens_decoder`), the same file the tiktoken loader names the K3
/// control block from.
pub fn read_media_token_ids(model_dir: &Path) -> Result<KimiK3MediaTokenIds> {
    let path = model_dir.join("tokenizer_config.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| anyhow!("Kimi K3: failed to read {}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow!("Kimi K3: failed to parse {}: {e}", path.display()))?;
    media_token_ids_from_tokenizer_config(&value)
        .map_err(|e| anyhow!("Kimi K3: {}: {e}", path.display()))
}

/// [`read_media_token_ids`] over an already parsed `tokenizer_config.json`.
pub fn media_token_ids_from_tokenizer_config(
    config: &serde_json::Value,
) -> Result<KimiK3MediaTokenIds> {
    let decoder = config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow!("tokenizer_config.json has no added_tokens_decoder object"))?;
    let lookup = |name: &str| -> Result<i32> {
        decoder
            .iter()
            .find(|(_, entry)| {
                entry.get("content").and_then(serde_json::Value::as_str) == Some(name)
            })
            .and_then(|(id, _)| id.parse::<i32>().ok())
            .ok_or_else(|| anyhow!("added_tokens_decoder does not name {name}"))
    };
    Ok(KimiK3MediaTokenIds {
        begin: lookup(MEDIA_BEGIN)?,
        content: lookup(MEDIA_CONTENT)?,
        pad: lookup(MEDIA_PAD)?,
        end: lookup(MEDIA_END)?,
    })
}

/// The id form of one image block, with `label_ids` the BPE encoding of
/// `image {w}x{h}`.
pub fn image_block_ids(ids: KimiK3MediaTokenIds, label_ids: &[i32], num_tokens: usize) -> Vec<i32> {
    let mut block = Vec::with_capacity(label_ids.len() + num_tokens + 3);
    block.push(ids.begin);
    block.extend_from_slice(label_ids);
    block.push(ids.content);
    block.extend(std::iter::repeat_n(ids.pad, num_tokens));
    block.push(ids.end);
    block
}

/// The `image {w}x{h}` label the block carries between the control tokens.
pub fn image_label(w: u32, h: u32) -> String {
    format!("image {w}x{h}")
}

/// What the placeholder pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedKimiK3Tokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
    /// `true` when the runs were already in the prompt (server render),
    /// `false` when they were spliced in front of it (CLI).
    pub prerendered: bool,
}

/// Maximal runs of `pad` in `prompt_tokens`, in order.
fn pad_runs(prompt_tokens: &[i32], pad: i32) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut current = 0usize;
    for &token in prompt_tokens {
        if token == pad {
            current += 1;
        } else if current > 0 {
            runs.push(current);
            current = 0;
        }
    }
    if current > 0 {
        runs.push(current);
    }
    runs
}

/// Verify the pre-rendered `<|media_pad|>` runs against the grids, or splice
/// one block per image in front of a placeholder-free prompt.
///
/// `encode` turns the `image {w}x{h}` label into ids (no special tokens).
pub fn place_kimi_k3_image_tokens<E>(
    prompt_tokens: &mut Vec<i32>,
    images: &[KimiK3PreparedImage],
    ids: KimiK3MediaTokenIds,
    mut encode: E,
) -> Result<InsertedKimiK3Tokens>
where
    E: FnMut(&str) -> Vec<i32>,
{
    let expected: Vec<usize> = images
        .iter()
        .map(|img| img.plan.num_tokens as usize)
        .collect();
    let total: usize = expected.iter().sum();
    let runs = pad_runs(prompt_tokens, ids.pad);
    if !runs.is_empty() {
        if runs != expected {
            return Err(anyhow!(
                "Kimi K3: the prompt carries <|media_pad|> runs of {runs:?} but the {} image(s) \
                 need {expected:?} (grid_h * grid_w / 4 each); the rendered prompt and the \
                 images that reached the worker disagree, which also happens when one of the \
                 request's images failed to decode and was dropped",
                images.len()
            ));
        }
        return Ok(InsertedKimiK3Tokens {
            image_blocks: images.len(),
            total_image_tokens: total as i32,
            prerendered: true,
        });
    }

    let mut spliced = Vec::with_capacity(prompt_tokens.len() + total + 8 * images.len());
    for img in images {
        let (w, h) = img.original_size;
        let label = encode(&image_label(w, h));
        if label.contains(&ids.pad) {
            return Err(anyhow!(
                "Kimi K3: the image label {:?} encoded to a <|media_pad|> id",
                image_label(w, h)
            ));
        }
        spliced.extend(image_block_ids(ids, &label, img.plan.num_tokens as usize));
    }
    spliced.extend_from_slice(prompt_tokens);
    *prompt_tokens = spliced;
    Ok(InsertedKimiK3Tokens {
        image_blocks: images.len(),
        total_image_tokens: total as i32,
        prerendered: false,
    })
}

/// Preprocess `images`, place (or verify) the placeholder runs, and merge the
/// projected features into the text embeddings.
pub fn compute_kimi_k3_image_embeddings<E>(
    model: &KimiK3VLModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[DynamicImage],
    encode: E,
) -> Result<(InputEmbeddings, InsertedKimiK3Tokens)>
where
    E: FnMut(&str) -> Vec<i32>,
{
    if images.is_empty() {
        return Err(anyhow!("Kimi K3 image embedding requested with no images"));
    }
    // Geometry first: it is what sizes the placeholder runs, and it is
    // cheap. A prompt that disagrees with the images, or a request over the
    // media-token budget, is refused before any pixel is normalized and any
    // tower block runs.
    let planned = model
        .processor
        .plan_images(images)
        .map_err(|e| anyhow!("{e}"))?;
    let stats = place_kimi_k3_image_tokens(prompt_tokens, &planned, model.media_token_ids, encode)?;
    let features = model
        .project_image_stream(images, &planned)
        .map_err(|e| anyhow!("{e}"))?;
    let input_ids = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let inputs_embeds = model.text.embed_tokens.forward(&input_ids);
    let embeddings = crate::vision::kimi_k3_vl::merge_media_features(
        model.media_placeholder_token_id,
        &features,
        &inputs_embeds,
        &input_ids,
    )
    .map_err(|e| anyhow!("{e}"))?;
    Ok((embeddings, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::encoders::moonvit3d::MoonViT3DGrid;
    use crate::vision::processors::kimi_k3::{KimiK3NavitConfig, NavitResizePlan};

    const IDS: KimiK3MediaTokenIds = KimiK3MediaTokenIds {
        begin: 163_602,
        content: 163_603,
        pad: 163_605,
        end: 163_604,
    };

    fn prepared(w: u32, h: u32) -> KimiK3PreparedImage {
        let plan: NavitResizePlan = KimiK3NavitConfig::default().plan(w, h).unwrap();
        KimiK3PreparedImage {
            grid: MoonViT3DGrid::image(plan.grid_h as i32, plan.grid_w as i32),
            original_size: (w, h),
            plan,
        }
    }

    fn encode(text: &str) -> Vec<i32> {
        text.bytes().map(i32::from).collect()
    }

    #[test]
    fn media_ids_come_from_added_tokens_decoder() {
        let cfg = serde_json::json!({
            "added_tokens_decoder": {
                "163602": {"content": "<|media_begin|>", "special": true},
                "163603": {"content": "<|media_content|>", "special": true},
                "163604": {"content": "<|media_end|>", "special": true},
                "163605": {"content": "<|media_pad|>", "special": true},
                "163584": {"content": "[BOS]", "special": true}
            }
        });
        assert_eq!(media_token_ids_from_tokenizer_config(&cfg).unwrap(), IDS);
        let missing =
            serde_json::json!({"added_tokens_decoder": {"1": {"content": "<|media_pad|>"}}});
        let err = media_token_ids_from_tokenizer_config(&missing).unwrap_err();
        assert!(err.to_string().contains("<|media_begin|>"), "{err}");
    }

    #[test]
    fn splices_one_block_per_image_in_front_of_a_raw_prompt() {
        // 100x100 -> 16 tokens; 224x224 -> 64 tokens.
        let images = [prepared(100, 100), prepared(224, 224)];
        let mut prompt = vec![10, 20, 30];
        let stats = place_kimi_k3_image_tokens(&mut prompt, &images, IDS, encode).unwrap();
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 80);
        assert!(!stats.prerendered);

        let label = encode("image 100x100");
        let first_len = 1 + label.len() + 1 + 16 + 1;
        assert_eq!(prompt[0], IDS.begin);
        assert_eq!(&prompt[1..1 + label.len()], &label[..]);
        assert_eq!(prompt[1 + label.len()], IDS.content);
        assert!(
            prompt[2 + label.len()..2 + label.len() + 16]
                .iter()
                .all(|&t| t == IDS.pad)
        );
        assert_eq!(prompt[first_len - 1], IDS.end);
        assert_eq!(prompt[first_len], IDS.begin);
        assert_eq!(&prompt[prompt.len() - 3..], &[10, 20, 30]);
        assert_eq!(pad_runs(&prompt, IDS.pad), vec![16, 64]);
    }

    #[test]
    fn verifies_prerendered_runs_and_rejects_a_mismatch() {
        let images = [prepared(100, 100)];
        let mut prompt = vec![1, IDS.begin, 5, IDS.content];
        prompt.extend(std::iter::repeat_n(IDS.pad, 16));
        prompt.extend([IDS.end, 7]);
        let before = prompt.clone();
        let stats = place_kimi_k3_image_tokens(&mut prompt, &images, IDS, encode).unwrap();
        assert!(stats.prerendered);
        assert_eq!(stats.total_image_tokens, 16);
        assert_eq!(prompt, before);

        // 15 pads for a 16-token image: refused, not padded or truncated.
        let mut short = vec![1];
        short.extend(std::iter::repeat_n(IDS.pad, 15));
        let err = place_kimi_k3_image_tokens(&mut short, &images, IDS, encode).unwrap_err();
        assert!(err.to_string().contains("[15]"), "{err}");

        // Two images rendered as one merged run of the right total: refused,
        // because the runs must match per image.
        let two = [prepared(100, 100), prepared(100, 100)];
        let mut merged = vec![1];
        merged.extend(std::iter::repeat_n(IDS.pad, 32));
        let err = place_kimi_k3_image_tokens(&mut merged, &two, IDS, encode).unwrap_err();
        assert!(err.to_string().contains("[32]"), "{err}");
    }
}
