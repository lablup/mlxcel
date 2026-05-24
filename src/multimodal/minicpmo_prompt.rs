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

//! MiniCPM-o prompt normalization helpers.
//!
//! The upstream processor expands each `<image>` marker into:
//! `<image>` + repeated `<unk>` placeholders + `</image>`.
//! Keeping the string rewrite and image-bound extraction here lets CLI and
//! server share the same prompt preparation logic.

pub const MINICPMO_IMAGE_START_TOKEN: &str = "<image>";
pub const MINICPMO_IMAGE_END_TOKEN: &str = "</image>";
pub const MINICPMO_IMAGE_INLINE_MARKER: &str = "<image>./</image>";
pub const MINICPMO_UNK_TOKEN: &str = "<unk>";
const MINICPMO_IMAGE_MARKER_SENTINEL: &str = "__MLXCEL_MINICPMO_IMAGE__";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniCPMOPromptTokens {
    pub tokens: Vec<i32>,
    pub image_slots: usize,
    pub image_bounds: Vec<(usize, usize)>,
}

pub fn minicpmo_image_placeholder(image_feature_size: usize) -> String {
    format!(
        "{}{}{}",
        MINICPMO_IMAGE_START_TOKEN,
        MINICPMO_UNK_TOKEN.repeat(image_feature_size),
        MINICPMO_IMAGE_END_TOKEN
    )
}

fn minicpmo_image_placeholders(image_feature_sizes: &[usize]) -> Vec<String> {
    image_feature_sizes
        .iter()
        .map(|&size| minicpmo_image_placeholder(size))
        .collect()
}

pub fn count_minicpmo_image_markers(prompt: &str) -> usize {
    prompt.matches(MINICPMO_IMAGE_INLINE_MARKER).count()
        + prompt.matches(MINICPMO_IMAGE_START_TOKEN).count()
        - prompt.matches(MINICPMO_IMAGE_END_TOKEN).count()
}

pub fn ensure_minicpmo_image_placeholders(
    prompt: &str,
    num_images: usize,
    image_feature_size: usize,
) -> Result<String, String> {
    ensure_minicpmo_image_placeholders_with_sizes(prompt, &vec![image_feature_size; num_images])
}

pub fn ensure_minicpmo_image_placeholders_with_sizes(
    prompt: &str,
    image_feature_sizes: &[usize],
) -> Result<String, String> {
    let num_images = image_feature_sizes.len();
    let placeholders = minicpmo_image_placeholders(image_feature_sizes);

    let normalized = prompt.replace(MINICPMO_IMAGE_INLINE_MARKER, MINICPMO_IMAGE_MARKER_SENTINEL);
    let marker_count = normalized.matches(MINICPMO_IMAGE_START_TOKEN).count();
    let normalized = normalized.replace(MINICPMO_IMAGE_START_TOKEN, MINICPMO_IMAGE_MARKER_SENTINEL);

    // Count inline markers that were replaced by sentinel
    let sentinel_count = normalized.matches(MINICPMO_IMAGE_MARKER_SENTINEL).count();

    if marker_count > 0 || sentinel_count > 0 {
        let total_markers = marker_count.max(sentinel_count);
        if total_markers != num_images {
            return Err(format!(
                "MiniCPM-o prompt contains {} <image> placeholders but {} image(s) were provided",
                total_markers, num_images
            ));
        }
        let mut expanded = String::with_capacity(
            normalized.len() + placeholders.iter().map(String::len).sum::<usize>(),
        );
        let mut parts = normalized.split(MINICPMO_IMAGE_MARKER_SENTINEL);
        if let Some(first) = parts.next() {
            expanded.push_str(first);
        }
        for (placeholder, part) in placeholders.iter().zip(parts) {
            expanded.push_str(placeholder);
            expanded.push_str(part);
        }
        // If prompt already has chat formatting, return as-is
        if expanded.contains("<|im_start|>") {
            return Ok(expanded);
        }
        // Otherwise wrap in chat template
        return Ok(format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            expanded
        ));
    }

    if num_images == 0 {
        return Ok(prompt.to_string());
    }

    let image_prefix = placeholders.concat();
    if let Some(pos) = prompt.rfind("<|im_start|>user\n") {
        let insert_pos = pos + "<|im_start|>user\n".len();
        Ok(format!(
            "{}{}{}",
            &prompt[..insert_pos],
            image_prefix,
            &prompt[insert_pos..]
        ))
    } else {
        // Wrap in Qwen3-style chat template (MiniCPM-o uses Qwen3 as backbone)
        Ok(format!(
            "<|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n",
            image_prefix, prompt
        ))
    }
}

pub fn compute_minicpmo_image_bounds(
    input_ids: &[i32],
    image_start_id: i32,
    image_end_id: i32,
) -> Result<Vec<(usize, usize)>, String> {
    let mut start_positions = Vec::new();
    let mut end_positions = Vec::new();

    for (idx, token_id) in input_ids.iter().copied().enumerate() {
        if token_id == image_start_id {
            start_positions.push(idx + 1);
        } else if token_id == image_end_id {
            end_positions.push(idx);
        }
    }

    if start_positions.len() != end_positions.len() {
        return Err(format!(
            "MiniCPM-o encoded prompt produced {} image starts but {} image ends",
            start_positions.len(),
            end_positions.len()
        ));
    }

    let mut bounds = Vec::with_capacity(start_positions.len());
    for (start, end) in start_positions.into_iter().zip(end_positions) {
        if end < start {
            return Err("MiniCPM-o image placeholder end appeared before start".to_string());
        }
        bounds.push((start, end));
    }

    Ok(bounds)
}

fn single_token_id<E>(text: &str, mut encode: E) -> Result<i32, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let tokens = encode(text, false);
    if tokens.len() != 1 {
        return Err(format!(
            "MiniCPM-o expected `{}` to encode to one token, got {}",
            text,
            tokens.len()
        ));
    }
    Ok(tokens[0])
}

pub fn prepare_minicpmo_prompt_tokens<E>(
    prompt: &str,
    num_images: usize,
    image_feature_size: usize,
    mut encode: E,
) -> Result<MiniCPMOPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    prepare_minicpmo_prompt_tokens_with_image_feature_sizes(
        prompt,
        &vec![image_feature_size; num_images],
        &mut encode,
    )
}

pub fn prepare_minicpmo_prompt_tokens_with_image_feature_sizes<E>(
    prompt: &str,
    image_feature_sizes: &[usize],
    mut encode: E,
) -> Result<MiniCPMOPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let num_images = image_feature_sizes.len();
    let text = ensure_minicpmo_image_placeholders_with_sizes(prompt, image_feature_sizes)?;
    let tokens = encode(&text, true);

    if num_images == 0 {
        return Ok(MiniCPMOPromptTokens {
            tokens,
            image_slots: 0,
            image_bounds: Vec::new(),
        });
    }

    let image_start_id = single_token_id(MINICPMO_IMAGE_START_TOKEN, &mut encode)?;
    let image_end_id = single_token_id(MINICPMO_IMAGE_END_TOKEN, &mut encode)?;
    let image_bounds = compute_minicpmo_image_bounds(&tokens, image_start_id, image_end_id)?;

    if image_bounds.len() != num_images {
        return Err(format!(
            "MiniCPM-o encoded prompt produced {} image bounds but {} image(s) were provided",
            image_bounds.len(),
            num_images
        ));
    }

    Ok(MiniCPMOPromptTokens {
        tokens,
        image_slots: image_bounds.len(),
        image_bounds,
    })
}

#[cfg(test)]
#[path = "minicpmo_prompt_tests.rs"]
mod tests;
