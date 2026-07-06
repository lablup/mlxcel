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

//! FastVLM prompt normalization helpers.
//!
//! FastVLM tokenizes each `<image>` placeholder into the sentinel id `-200`
//! (the string is not a vocabulary token). One sentinel is spliced per image;
//! the runtime later expands each sentinel to `mm_tokens_per_image` (256) copies
//! and scatters the image embeddings over those positions (LLaVA merge).

pub const FASTVLM_IMAGE_TOKEN: &str = "<image>";
pub const FASTVLM_IMAGE_TOKEN_INDEX: i32 = -200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastvlmPromptTokens {
    pub tokens: Vec<i32>,
    pub image_slots: usize,
}

pub fn count_fastvlm_image_tokens(prompt: &str) -> usize {
    prompt.matches(FASTVLM_IMAGE_TOKEN).count()
}

/// Insert one `<image>` per image after the Qwen2 user tag when the prompt
/// carries no placeholder (leaves prompts that already contain `<image>` alone).
pub fn ensure_fastvlm_image_tokens(prompt: &str, num_images: usize) -> String {
    if num_images == 0 || count_fastvlm_image_tokens(prompt) > 0 {
        return prompt.to_string();
    }

    let image_tokens = format!("{FASTVLM_IMAGE_TOKEN}\n").repeat(num_images);
    for tag in ["<|im_start|>user\n", "<|im_start|>user"] {
        if let Some(pos) = prompt.find(tag) {
            let mut text = prompt.to_string();
            text.insert_str(pos + tag.len(), &image_tokens);
            return text;
        }
    }
    format!("{image_tokens}{prompt}")
}

/// Split the rendered prompt on `<image>`, tokenize each text segment without
/// special tokens, and splice the `-200` sentinel between segments.
pub fn prepare_fastvlm_prompt_tokens<E>(
    prompt: &str,
    num_images: usize,
    mut encode: E,
) -> Result<FastvlmPromptTokens, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let text = ensure_fastvlm_image_tokens(prompt, num_images);
    let image_slots = count_fastvlm_image_tokens(&text);

    if image_slots != num_images {
        return Err(format!(
            "FastVLM prompt contains {image_slots} <image> placeholders but {num_images} image(s) were provided"
        ));
    }

    if image_slots == 0 {
        return Ok(FastvlmPromptTokens {
            tokens: encode(&text, true),
            image_slots: 0,
        });
    }

    let chunks: Vec<&str> = text.split(FASTVLM_IMAGE_TOKEN).collect();
    let mut tokens = Vec::new();
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if !chunk.is_empty() {
            tokens.extend(encode(chunk, chunk_idx == 0));
        } else if chunk_idx == 0 {
            tokens.extend(encode("", true));
        }
        if chunk_idx + 1 < chunks.len() {
            tokens.push(FASTVLM_IMAGE_TOKEN_INDEX);
        }
    }

    Ok(FastvlmPromptTokens {
        tokens,
        image_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_encode(text: &str, _special: bool) -> Vec<i32> {
        // One token per non-space char, deterministic and nonzero.
        text.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c as i32)
            .collect()
    }

    #[test]
    fn splices_one_sentinel_per_image() {
        let out = prepare_fastvlm_prompt_tokens("a<image>b", 1, dummy_encode).unwrap();
        assert_eq!(out.image_slots, 1);
        assert_eq!(out.tokens.iter().filter(|&&t| t == -200).count(), 1);
        // 'a' before, sentinel, 'b' after.
        assert_eq!(out.tokens[0], 'a' as i32);
        assert_eq!(out.tokens[1], -200);
        assert_eq!(out.tokens[2], 'b' as i32);
    }

    #[test]
    fn inserts_placeholder_after_user_tag() {
        let out = prepare_fastvlm_prompt_tokens("<|im_start|>user\nhi<|im_end|>", 1, dummy_encode)
            .unwrap();
        assert_eq!(out.image_slots, 1);
        assert_eq!(out.tokens.iter().filter(|&&t| t == -200).count(), 1);
    }

    #[test]
    fn count_mismatch_errors() {
        assert!(prepare_fastvlm_prompt_tokens("<image><image>", 1, dummy_encode).is_err());
    }
}
