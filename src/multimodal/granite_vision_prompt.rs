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

//! Granite Vision image-token expansion.
//!
//! The templated prompt carries one `<image>` (id 49155) per image. Each is
//! expanded into `num_image_tokens` copies so the placeholder count matches the
//! packed feature count from `granite_vision::GraniteVisionVLModel`. The count
//! comes from the shared AnyRes helper, so packing and expansion cannot drift.

use crate::vision::processors::anyres::{AnyResTileInfo, num_image_tokens};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedGraniteVisionTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Expand each `<image>` placeholder in `prompt_tokens` into `num_image_tokens`
/// copies (one per image, in order). If the prompt has no placeholder, splice
/// the runs after the first token. Returns `None` when nothing needs doing or
/// the placeholder count is neither `0` nor `infos.len()` (already expanded).
pub fn insert_granite_vision_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    infos: &[AnyResTileInfo],
    image_token_index: i32,
    feature_side: i32,
    base_tokens: i32,
) -> Option<InsertedGraniteVisionTokens> {
    if prompt_tokens.is_empty() || infos.is_empty() {
        return None;
    }
    let counts: Vec<i32> = infos
        .iter()
        .map(|info| num_image_tokens(info, feature_side, base_tokens))
        .collect();
    let total: i32 = counts.iter().sum();

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_index)
        .count();

    // Case 1: one placeholder per image — expand each in place.
    if placeholder_count == infos.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total as usize);
        let mut idx = 0usize;
        for &tok in prompt_tokens.iter() {
            if tok == image_token_index {
                for _ in 0..counts[idx] {
                    expanded.push(image_token_index);
                }
                idx += 1;
            } else {
                expanded.push(tok);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedGraniteVisionTokens {
            image_blocks: infos.len(),
            total_image_tokens: total,
        });
    }

    // A non-zero placeholder count that does not match the image count means the
    // prompt was already expanded (or is malformed): leave it alone.
    if placeholder_count != 0 {
        return None;
    }

    // Case 2: no placeholder — splice the runs after the first token.
    let bos = prompt_tokens[0];
    let rest = prompt_tokens[1..].to_vec();
    let mut image_tokens = Vec::with_capacity(total as usize);
    for &c in &counts {
        for _ in 0..c {
            image_tokens.push(image_token_index);
        }
    }
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);

    Some(InsertedGraniteVisionTokens {
        image_blocks: infos.len(),
        total_image_tokens: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::processors::anyres::AnyResProcessor;

    fn pins() -> Vec<(i32, i32)> {
        let mut p = Vec::new();
        for w in (384..=3840).step_by(384) {
            p.push((384, w));
        }
        for w in (384..=1920).step_by(384) {
            p.push((768, w));
        }
        for w in (384..=1152).step_by(384) {
            p.push((1152, w));
        }
        for h in [1536, 1920] {
            for w in [384, 768] {
                p.push((h, w));
            }
        }
        for h in [2304, 2688, 3072, 3456, 3840] {
            p.push((h, 384));
        }
        p
    }

    #[test]
    fn expands_single_placeholder_to_feature_count() {
        let proc = AnyResProcessor::new(pins(), 384);
        let info = proc.tile_info(500, 1000); // 4009 tokens
        let mut tokens = vec![1, 49155, 5, 6];
        let stats =
            insert_granite_vision_image_tokens(&mut tokens, &[info], 49155, 27, 729).unwrap();
        assert_eq!(stats.total_image_tokens, 4009);
        assert_eq!(tokens.iter().filter(|&&t| t == 49155).count(), 4009);
        assert_eq!(tokens.len(), 3 + 4009); // 1, 5, 6 + expanded run
    }

    #[test]
    fn already_expanded_is_left_alone() {
        let proc = AnyResProcessor::new(pins(), 384);
        let info = proc.tile_info(384, 384);
        let mut tokens = vec![1, 49155, 49155, 49155, 5];
        // 3 placeholders but 1 image -> untouched.
        assert!(insert_granite_vision_image_tokens(&mut tokens, &[info], 49155, 27, 729).is_none());
    }
}
