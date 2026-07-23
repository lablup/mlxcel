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

//! Token-exact Phi4MM image/audio placeholder normalization.

use crate::phi4_siglip_prompt::{PHI4_SIGLIP_IMAGE_TOKEN, PHI4_SIGLIP_IMAGE_TOKEN_INDEX};

pub const PHI4MM_AUDIO_TOKEN_ID: i32 = 200_011;
const INTERNAL_AUDIO_TOKEN: &str = "<|mlxcel_phi4mm_audio|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi4MMPromptTokens {
    pub tokens: Vec<i32>,
    pub image_slots: usize,
    pub audio_slots: usize,
}

fn normalize_numbered_image_tags(prompt: &str, num_images: usize) -> String {
    let mut text = prompt.to_string();
    for image_num in 1..=num_images {
        text = text.replace(&format!("<|image_{image_num}|>"), PHI4_SIGLIP_IMAGE_TOKEN);
    }
    text
}

fn ensure_phi4mm_image_tags(prompt: &str, num_images: usize) -> String {
    if num_images == 0 || prompt.contains(PHI4_SIGLIP_IMAGE_TOKEN) {
        return prompt.to_string();
    }
    let block = PHI4_SIGLIP_IMAGE_TOKEN.repeat(num_images);
    if let Some(position) = prompt.find("<|user|>\n") {
        let mut text = prompt.to_string();
        text.insert_str(position + "<|user|>\n".len(), &block);
        text
    } else if let Some(position) = prompt.find("<|user|>") {
        let mut text = prompt.to_string();
        text.insert_str(position + "<|user|>".len(), &block);
        text
    } else {
        format!("{block}{prompt}")
    }
}

fn numbered_audio_tags(prompt: &str) -> Result<Vec<usize>, String> {
    let mut tags = Vec::new();
    let mut rest = prompt;
    while let Some(start) = rest.find("<|audio_") {
        rest = &rest[start + "<|audio_".len()..];
        let end = rest
            .find("|>")
            .ok_or("malformed Phi4MM audio placeholder: missing |>")?;
        let number = rest[..end]
            .parse::<usize>()
            .map_err(|_| "malformed Phi4MM audio placeholder: expected a positive number")?;
        if number == 0 {
            return Err("Phi4MM audio placeholders are numbered from 1".into());
        }
        tags.push(number);
        rest = &rest[end + 2..];
    }
    if prompt.match_indices("<|audio").count() != tags.len() {
        return Err("malformed Phi4MM audio placeholder: expected <|audio_N|>".into());
    }
    Ok(tags)
}

fn ensure_audio_tags(prompt: &str, num_audios: usize) -> Result<String, String> {
    let tags = numbered_audio_tags(prompt)?;
    if !tags.is_empty() {
        let expected: Vec<usize> = (1..=num_audios).collect();
        if tags != expected {
            return Err(format!(
                "Phi4MM audio placeholders must appear exactly once in order {:?}; found {:?}",
                expected, tags
            ));
        }
        return Ok(prompt.to_string());
    }
    if num_audios == 0 {
        return Ok(prompt.to_string());
    }
    let block: String = (1..=num_audios)
        .map(|index| format!("<|audio_{index}|>"))
        .collect();
    if let Some(position) = prompt.find("<|user|>\n") {
        let mut text = prompt.to_string();
        text.insert_str(position + "<|user|>\n".len(), &block);
        Ok(text)
    } else if let Some(position) = prompt.find("<|user|>") {
        let mut text = prompt.to_string();
        text.insert_str(position + "<|user|>".len(), &block);
        Ok(text)
    } else {
        Ok(format!("{block}{prompt}"))
    }
}

pub fn prepare_phi4mm_prompt_tokens<E>(
    prompt: &str,
    num_images: usize,
    num_audios: usize,
    mut encode: E,
) -> Result<Phi4MMPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if prompt.contains(INTERNAL_AUDIO_TOKEN) || prompt.contains("<|endoftext11|>") {
        return Err("Phi4MM prompt contains a reserved audio token literal".into());
    }

    let text = normalize_numbered_image_tags(prompt, num_images);
    let text = ensure_phi4mm_image_tags(&text, num_images);
    let text = ensure_audio_tags(&text, num_audios)?;
    let image_slots = text.matches(PHI4_SIGLIP_IMAGE_TOKEN).count();
    if image_slots != num_images {
        return Err(format!(
            "Phi4MM prompt contains {image_slots} image placeholders but {num_images} image(s) were provided"
        ));
    }

    let audio_slots = numbered_audio_tags(&text)?.len();
    if audio_slots != num_audios {
        return Err(format!(
            "Phi4MM prompt contains {audio_slots} audio placeholders but {num_audios} audio clip(s) were provided"
        ));
    }
    let mut normalized = text;
    for audio_num in 1..=num_audios {
        normalized = normalized.replace(&format!("<|audio_{audio_num}|>"), INTERNAL_AUDIO_TOKEN);
    }

    let mut tokens = Vec::new();
    let mut remaining = normalized.as_str();
    let mut first_chunk = true;
    loop {
        let image_pos = remaining.find(PHI4_SIGLIP_IMAGE_TOKEN);
        let audio_pos = remaining.find(INTERNAL_AUDIO_TOKEN);
        let next = match (image_pos, audio_pos) {
            (None, None) => None,
            (Some(position), None) => Some((position, PHI4_SIGLIP_IMAGE_TOKEN)),
            (None, Some(position)) => Some((position, INTERNAL_AUDIO_TOKEN)),
            (Some(image), Some(audio)) if image <= audio => Some((image, PHI4_SIGLIP_IMAGE_TOKEN)),
            (Some(_), Some(audio)) => Some((audio, INTERNAL_AUDIO_TOKEN)),
        };
        let Some((position, marker)) = next else {
            tokens.extend(encode(remaining, first_chunk));
            break;
        };
        tokens.extend(encode(&remaining[..position], first_chunk));
        first_chunk = false;
        tokens.push(if marker == PHI4_SIGLIP_IMAGE_TOKEN {
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX
        } else {
            PHI4MM_AUDIO_TOKEN_ID
        });
        remaining = &remaining[position + marker.len()..];
    }

    Ok(Phi4MMPromptTokens {
        tokens,
        image_slots,
        audio_slots,
    })
}

/// Expand each normalized media sentinel into the exact encoder-row count.
/// The walk preserves the prompt's mixed image/audio order and rejects any
/// cardinality drift before text embedding lookup can observe a negative ID.
pub fn expand_phi4mm_placeholders(
    tokens: &[i32],
    image_sizes: &[usize],
    audio_sizes: &[usize],
) -> Result<Vec<i32>, String> {
    let mut images = image_sizes.iter();
    let mut audios = audio_sizes.iter();
    let mut image_count = 0usize;
    let mut audio_count = 0usize;
    let mut output = Vec::new();

    for &token in tokens {
        if token == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
            let size = *images
                .next()
                .ok_or("Phi4MM prompt has more image placeholders than image inputs")?;
            if size == 0 {
                return Err("Phi4MM image encoder produced zero tokens".into());
            }
            output.extend(std::iter::repeat_n(token, size));
            image_count += 1;
        } else if token == PHI4MM_AUDIO_TOKEN_ID {
            let size = *audios
                .next()
                .ok_or("Phi4MM prompt has more audio placeholders than audio inputs")?;
            if size == 0 {
                return Err("Phi4MM audio encoder produced zero tokens".into());
            }
            output.extend(std::iter::repeat_n(token, size));
            audio_count += 1;
        } else {
            output.push(token);
        }
    }

    if image_count != image_sizes.len() || audio_count != audio_sizes.len() {
        return Err(format!(
            "Phi4MM placeholder cardinality mismatch: prompt has {image_count} image/{audio_count} audio placeholders, inputs have {}/{}",
            image_sizes.len(),
            audio_sizes.len()
        ));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "phi4mm_prompt_tests.rs"]
mod tests;
