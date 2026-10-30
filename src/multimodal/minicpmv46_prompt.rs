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

//! MiniCPM-V 4.6 (`minicpmv4_6`) image placeholder expansion (issue #2547).
//!
//! transformers' `MiniCPMV4_6Processor` replaces each image marker in the
//! rendered text with
//! `<image_id>{i}</image_id><image>` + `<|image_pad|>` * N + `</image>`, where
//! `i` counts the images of the prompt from 0 (`use_image_id: true` in the
//! checkpoint's `preprocessor_config.json`) and N is the image's feature
//! count. The checkpoint's chat template renders an image content item as
//! `<|image_pad|>` and joins content items with `\n`, so a one-image turn reads
//! `<|im_start|>user\n<image_id>0</image_id><image>..</image>\nQUESTION<|im_end|>`.
//! mlx-vlm expands the same block, from its `(<image>./</image>)` marker.
//!
//! MiniCPM-o shares the bounds logic ([`crate::minicpmo_prompt`]) but keeps its
//! own `<unk>` filler and no image id; this module is the 4.6 layout:
//!
//! 1. `<|image_pad|>` (the template's marker) and `<image>./</image>` (the
//!    MiniCPM-V inline marker mlx-vlm and older clients write) are replaced in
//!    order, one image each.
//! 2. Without a marker, each block and a `\n` open the last
//!    `<|im_start|>user\n` turn, which is where the template puts an image item
//!    listed ahead of the question.
//! 3. With neither marker nor user turn (the local checkpoint copy ships no
//!    chat template, or `--no-chat-template`), the turn is wrapped in the
//!    template's ChatML user and assistant openers.
//!
//! Before #2547 every 4.6 prompt went through the MiniCPM-o rule, which left
//! out the `<image_id>0</image_id>` prefix and the newline after the block, and
//! left a template-rendered `<|image_pad|>` marker in the prompt as text.

use crate::minicpmo_prompt::{
    MINICPMO_IMAGE_END_TOKEN, MINICPMO_IMAGE_INLINE_MARKER, MINICPMO_IMAGE_START_TOKEN,
    MiniCPMOPromptTokens, compute_minicpmo_image_bounds,
};

/// The image marker the checkpoint's chat template renders, and the filler
/// token of the expanded block.
pub const MINICPMV46_IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
pub const MINICPMV46_IMAGE_ID_START_TOKEN: &str = "<image_id>";
pub const MINICPMV46_IMAGE_ID_END_TOKEN: &str = "</image_id>";
const USER_TURN: &str = "<|im_start|>user\n";

/// The expanded block for image `image_index` with `feature_size` features.
pub fn minicpmv46_image_placeholder(image_index: usize, feature_size: usize) -> String {
    format!(
        "{MINICPMV46_IMAGE_ID_START_TOKEN}{image_index}{MINICPMV46_IMAGE_ID_END_TOKEN}\
         {MINICPMO_IMAGE_START_TOKEN}{}{MINICPMO_IMAGE_END_TOKEN}",
        MINICPMV46_IMAGE_PAD_TOKEN.repeat(feature_size)
    )
}

/// Rewrite `prompt` so every image is an expanded block (see the module docs).
pub fn expand_minicpmv46_image_placeholders(
    prompt: &str,
    image_feature_sizes: &[usize],
) -> Result<String, String> {
    let num_images = image_feature_sizes.len();
    if num_images == 0 {
        return Ok(prompt.to_string());
    }
    let placeholders: Vec<String> = image_feature_sizes
        .iter()
        .enumerate()
        .map(|(index, &size)| minicpmv46_image_placeholder(index, size))
        .collect();

    let segments = split_on_image_markers(prompt);

    let markers = segments.len() - 1;
    if markers > 0 {
        if markers != num_images {
            return Err(format!(
                "MiniCPM-V 4.6 prompt contains {markers} image marker(s) but {num_images} image(s) were provided"
            ));
        }
        let mut expanded = String::from(segments[0]);
        for (placeholder, segment) in placeholders.iter().zip(&segments[1..]) {
            expanded.push_str(placeholder);
            expanded.push_str(segment);
        }
        if expanded.contains("<|im_start|>") {
            return Ok(expanded);
        }
        return Ok(format!(
            "{USER_TURN}{expanded}<|im_end|>\n<|im_start|>assistant\n"
        ));
    }

    let mut image_items = String::new();
    for placeholder in &placeholders {
        image_items.push_str(placeholder);
        image_items.push('\n');
    }
    if let Some(pos) = prompt.rfind(USER_TURN) {
        let insert_at = pos + USER_TURN.len();
        return Ok(format!(
            "{}{image_items}{}",
            &prompt[..insert_at],
            &prompt[insert_at..]
        ));
    }
    Ok(format!(
        "{USER_TURN}{image_items}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

/// Split `prompt` on both image markers in order of appearance, in one pass,
/// so the `<|image_pad|>` filler of an expanded block is never re-read as a
/// marker.
///
/// The prompt is user-controlled and this runs on the batch scheduler before
/// any length check, so the split must stay linear: each marker kind keeps the
/// offset of its next occurrence and is searched again only once the cursor
/// has passed it. Re-searching the remainder for every kind on every marker
/// was quadratic when one kind is absent (100k typed markers took 24 s).
fn split_on_image_markers(prompt: &str) -> Vec<&str> {
    const MARKERS: [&str; 2] = [MINICPMV46_IMAGE_PAD_TOKEN, MINICPMO_IMAGE_INLINE_MARKER];
    let find_from = |marker: &str, from: usize| prompt[from..].find(marker).map(|at| from + at);
    let mut next = MARKERS.map(|marker| find_from(marker, 0));
    let mut segments: Vec<&str> = Vec::new();
    let mut cursor = 0;
    while let Some((at, kind)) = next
        .iter()
        .enumerate()
        .filter_map(|(kind, at)| at.map(|at| (at, kind)))
        .min()
    {
        segments.push(&prompt[cursor..at]);
        cursor = at + MARKERS[kind].len();
        for (slot, marker) in next.iter_mut().zip(MARKERS) {
            if slot.is_some_and(|pos| pos < cursor) {
                *slot = find_from(marker, cursor);
            }
        }
    }
    segments.push(&prompt[cursor..]);
    segments
}

fn single_token_id<E>(text: &str, encode: &mut E) -> Result<i32, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    match encode(text, false).as_slice() {
        [id] => Ok(*id),
        other => Err(format!(
            "MiniCPM-V 4.6 expected `{text}` to encode to one token, got {}",
            other.len()
        )),
    }
}

/// Expand the image placeholders, encode the prompt, and locate each image's
/// feature span (the tokens between `<image>` and `</image>`).
pub fn prepare_minicpmv46_prompt_tokens<E>(
    prompt: &str,
    image_feature_sizes: &[usize],
    encode: &mut E,
) -> Result<MiniCPMOPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let num_images = image_feature_sizes.len();
    let text = expand_minicpmv46_image_placeholders(prompt, image_feature_sizes)?;
    let tokens = encode(&text, true);
    if num_images == 0 {
        return Ok(MiniCPMOPromptTokens {
            tokens,
            image_slots: 0,
            image_bounds: Vec::new(),
        });
    }

    let image_start_id = single_token_id(MINICPMO_IMAGE_START_TOKEN, encode)?;
    let image_end_id = single_token_id(MINICPMO_IMAGE_END_TOKEN, encode)?;
    let image_bounds = compute_minicpmo_image_bounds(&tokens, image_start_id, image_end_id)?;
    if image_bounds.len() != num_images {
        return Err(format!(
            "MiniCPM-V 4.6 encoded prompt produced {} image bounds but {num_images} image(s) were provided",
            image_bounds.len()
        ));
    }
    for (&(start, end), &size) in image_bounds.iter().zip(image_feature_sizes) {
        if end - start != size {
            return Err(format!(
                "MiniCPM-V 4.6 image span holds {} token(s) but the image has {size} feature(s)",
                end - start
            ));
        }
    }
    Ok(MiniCPMOPromptTokens {
        tokens,
        image_slots: image_bounds.len(),
        image_bounds,
    })
}

#[cfg(test)]
#[path = "minicpmv46_prompt_tests.rs"]
mod tests;
