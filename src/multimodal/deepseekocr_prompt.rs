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

//! DeepSeek-OCR prompt image-token expansion.
//!
//! Each `<image>` (id 128815) placeholder expands to a flat run whose length is
//! the per-image placeholder count decided by the processor (global mosaic +
//! view separator, plus the tile mosaic when the image is cropped). The run
//! tokens are all identical; the vision features fill them in feature order.

pub struct InsertedDeepSeekOcrTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Expand each `<image>` placeholder in `prompt_tokens` to `counts[i]` copies
/// (in order). Returns `None` when there are no images or the placeholder count
/// does not match the image count.
pub fn insert_deepseekocr_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    counts: &[i32],
    image_token_id: i32,
) -> Option<InsertedDeepSeekOcrTokens> {
    if prompt_tokens.is_empty() || counts.is_empty() {
        return None;
    }
    let placeholders = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    // One placeholder per image: expand each in place.
    if placeholders == counts.len() {
        let total: i32 = counts.iter().sum();
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total as usize);
        let mut idx = 0usize;
        for &tok in prompt_tokens.iter() {
            if tok == image_token_id {
                for _ in 0..counts[idx] {
                    expanded.push(image_token_id);
                }
                idx += 1;
            } else {
                expanded.push(tok);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedDeepSeekOcrTokens {
            image_blocks: counts.len(),
            total_image_tokens: total,
        });
    }

    // No placeholder: splice one run per image after the first token (BOS).
    if placeholders == 0 {
        let total: i32 = counts.iter().sum();
        let bos = prompt_tokens[0];
        let rest = prompt_tokens[1..].to_vec();
        let mut out = Vec::with_capacity(prompt_tokens.len() + total as usize);
        out.push(bos);
        for &c in counts {
            for _ in 0..c {
                out.push(image_token_id);
            }
        }
        out.extend(rest);
        *prompt_tokens = out;
        return Some(InsertedDeepSeekOcrTokens {
            image_blocks: counts.len(),
            total_image_tokens: total,
        });
    }

    None
}
