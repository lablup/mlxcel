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

//! DeepSeek-VL2 prompt image-token expansion.
//!
//! Each checkpoint-specific `<image>` placeholder expands to a flat run whose length is
//! the per-image placeholder count decided by the processor (global mosaic +
//! view separator + local mosaic). The run tokens are all identical; the vision
//! features fill them in feature order `[global, view_separator, local]`.

pub struct InsertedDeepSeekVl2Tokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Expand each `<image>` placeholder in `prompt_tokens` to `counts[i]` copies
/// (in order). When no placeholder is present, splice one run per image after
/// the first token (BOS). Returns `None` when there are no images or the
/// placeholder count does not match the image count.
pub fn insert_deepseek_vl2_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    counts: &[i32],
    image_token_id: i32,
) -> Option<InsertedDeepSeekVl2Tokens> {
    if prompt_tokens.is_empty() || counts.is_empty() {
        return None;
    }
    let placeholders = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    if placeholders != 0 && placeholders != counts.len() {
        return None;
    }
    let total = counts.iter().try_fold(
        0i32,
        |sum, &n| {
            if n <= 0 { None } else { sum.checked_add(n) }
        },
    )?;
    let length = (prompt_tokens.len() - placeholders).checked_add(total as usize)?;
    i32::try_from(length).ok()?; // MLX sequence dimensions are i32.
    let mut expanded = Vec::new();
    expanded.try_reserve_exact(length).ok()?;
    if placeholders == 0 {
        // Raw (--no-chat-template) prompt: keep the historical BOS fallback.
        expanded.push(prompt_tokens[0]);
        expanded.extend(std::iter::repeat_n(image_token_id, total as usize));
        expanded.extend_from_slice(&prompt_tokens[1..]);
    } else {
        let mut blocks = counts.iter();
        for &token in prompt_tokens.iter() {
            if token == image_token_id {
                expanded.extend(std::iter::repeat_n(
                    image_token_id,
                    *blocks.next()? as usize,
                ));
            } else {
                expanded.push(token);
            }
        }
    }
    *prompt_tokens = expanded;
    Some(InsertedDeepSeekVl2Tokens {
        image_blocks: counts.len(),
        total_image_tokens: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_single_placeholder_in_place() {
        let mut toks = vec![100000, 100003, 42];
        let stats = insert_deepseek_vl2_image_tokens(&mut toks, &[421], 100003).unwrap();
        assert_eq!(stats.total_image_tokens, 421);
        assert_eq!(toks.len(), 2 + 421);
        assert_eq!(toks[0], 100000);
        assert_eq!(*toks.last().unwrap(), 42);
    }

    #[test]
    fn splices_run_after_bos_when_no_placeholder() {
        let mut toks = vec![100000, 42, 43];
        let stats = insert_deepseek_vl2_image_tokens(&mut toks, &[421], 100003).unwrap();
        assert_eq!(stats.total_image_tokens, 421);
        assert_eq!(toks[0], 100000);
        assert_eq!(toks[1], 100003);
        assert_eq!(toks[1 + 421], 42);
    }

    #[test]
    fn mismatched_placeholder_count_returns_none() {
        let mut toks = vec![100000, 100003, 100003, 42];
        assert!(insert_deepseek_vl2_image_tokens(&mut toks, &[421], 100003).is_none());
    }
}

#[cfg(test)]
mod bounds {
    use super::*;

    #[test]
    fn rejects_invalid_counts_without_changing_ids() {
        for counts in [&[][..], &[0], &[-1], &[i32::MAX], &[i32::MAX, 1]] {
            for original in [vec![0], vec![0, 42]] {
                let mut ids = original.clone();
                assert!(insert_deepseek_vl2_image_tokens(&mut ids, counts, 42).is_none());
                assert_eq!(ids, original);
            }
        }
    }

    #[test]
    fn large_interleaved_image_list_preserves_every_boundary() {
        let mut ids = [42, 9].repeat(100_000);
        let counts = vec![2; 100_000];
        insert_deepseek_vl2_image_tokens(&mut ids, &counts, 42).unwrap();
        assert_eq!(ids, [42, 42, 9].repeat(100_000));
    }
}
