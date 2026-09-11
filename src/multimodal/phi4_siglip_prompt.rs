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

//! Phi4-SigLIP prompt normalization helpers.
//!
//! The upstream processor tokenizes `<image>` placeholders into the sentinel
//! token id `-200`. Keeping that mapping in one module lets CLI and server use
//! the same prompt-to-image-slot rules.

pub const PHI4_SIGLIP_IMAGE_TOKEN: &str = "<image>";
pub const PHI4_SIGLIP_IMAGE_TOKEN_INDEX: i32 = -200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi4SigLipPromptTokens {
    pub tokens: Vec<i32>,
    pub image_slots: usize,
}

pub fn count_phi4_siglip_image_tokens(prompt: &str) -> usize {
    prompt.matches(PHI4_SIGLIP_IMAGE_TOKEN).count()
}

pub fn ensure_phi4_siglip_image_tokens(prompt: &str, num_images: usize) -> String {
    if num_images == 0 || count_phi4_siglip_image_tokens(prompt) > 0 {
        return prompt.to_string();
    }

    let image_tokens = format!("{}\n", PHI4_SIGLIP_IMAGE_TOKEN).repeat(num_images);
    // Insert image tokens after <|user|> tag (with or without trailing newline)
    if let Some(pos) = prompt.find("<|user|>\n") {
        let mut text = prompt.to_string();
        text.insert_str(pos + "<|user|>\n".len(), &image_tokens);
        text
    } else if let Some(pos) = prompt.find("<|user|>") {
        let mut text = prompt.to_string();
        text.insert_str(pos + "<|user|>".len(), &image_tokens);
        text
    } else {
        format!("{}{}", image_tokens, prompt)
    }
}

pub fn prepare_phi4_siglip_prompt_tokens<E>(
    prompt: &str,
    num_images: usize,
    mut encode: E,
) -> Result<Phi4SigLipPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let text = ensure_phi4_siglip_image_tokens(prompt, num_images);
    let image_slots = count_phi4_siglip_image_tokens(&text);

    if image_slots != num_images {
        return Err(format!(
            "Phi4-SigLIP prompt contains {} <image> placeholders but {} image(s) were provided",
            image_slots, num_images
        ));
    }

    if image_slots == 0 {
        return Ok(Phi4SigLipPromptTokens {
            tokens: encode(&text, true),
            image_slots: 0,
        });
    }

    let chunks: Vec<&str> = text.split(PHI4_SIGLIP_IMAGE_TOKEN).collect();
    let mut tokens = Vec::new();
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if !chunk.is_empty() {
            tokens.extend(encode(chunk, chunk_idx == 0));
        } else if chunk_idx == 0 {
            tokens.extend(encode("", true));
        }

        if chunk_idx + 1 < chunks.len() {
            tokens.push(PHI4_SIGLIP_IMAGE_TOKEN_INDEX);
        }
    }

    Ok(Phi4SigLipPromptTokens {
        tokens,
        image_slots,
    })
}

#[cfg(test)]
#[path = "phi4_siglip_prompt_tests.rs"]
mod tests;
