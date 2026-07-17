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

/// Unlimited-OCR variant: a single literal `<image>` placeholder covers *all*
/// pages, so one placeholder expands to the sum of every page's placeholder
/// count (in page order). Falls back to [`insert_deepseekocr_image_tokens`] for
/// the one-placeholder-per-image and no-placeholder layouts.
pub fn insert_unlimited_ocr_image_tokens(
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

    // Single `<image>` covering every page: expand it to the summed run. This
    // also matches the common single-page case (one placeholder, one count).
    if placeholders == 1 {
        let total: i32 = counts.iter().sum();
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total as usize);
        for &tok in prompt_tokens.iter() {
            if tok == image_token_id {
                for _ in 0..total {
                    expanded.push(image_token_id);
                }
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

    insert_deepseekocr_image_tokens(prompt_tokens, counts, image_token_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: i32 = 128815;

    #[test]
    fn unlimited_single_image_expands_in_place() {
        let mut toks = vec![0, IMG, 5, 6];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[3], IMG).unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 3);
        assert_eq!(toks, vec![0, IMG, IMG, IMG, 5, 6]);
    }

    #[test]
    fn unlimited_single_placeholder_covers_all_pages() {
        // One literal <image> covers three pages: expand to the summed count.
        let mut toks = vec![0, IMG, 9];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[2, 3, 4], IMG).unwrap();
        assert_eq!(stats.image_blocks, 3);
        assert_eq!(stats.total_image_tokens, 9);
        assert_eq!(toks.iter().filter(|&&t| t == IMG).count(), 9);
        assert_eq!(toks.first().copied(), Some(0));
        assert_eq!(toks.last().copied(), Some(9));
    }

    #[test]
    fn unlimited_no_placeholder_falls_back_to_splice() {
        let mut toks = vec![0, 7, 8];
        let stats = insert_unlimited_ocr_image_tokens(&mut toks, &[2, 2], IMG).unwrap();
        assert_eq!(stats.total_image_tokens, 4);
        // Spliced after BOS.
        assert_eq!(toks, vec![0, IMG, IMG, IMG, IMG, 7, 8]);
    }

    #[test]
    fn unlimited_empty_inputs_return_none() {
        let mut empty: Vec<i32> = vec![];
        assert!(insert_unlimited_ocr_image_tokens(&mut empty, &[1], IMG).is_none());
        let mut toks = vec![0, IMG];
        assert!(insert_unlimited_ocr_image_tokens(&mut toks, &[], IMG).is_none());
    }
}
