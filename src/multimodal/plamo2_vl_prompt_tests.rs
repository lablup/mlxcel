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

use super::*;

const IDS: Plamo2VlImageTokenIds = Plamo2VlImageTokenIds {
    bos: Some(1),
    img_start: 100_000,
    image: 100_002,
    img_end: 100_001,
};

/// The exact string upstream `_apply_ja01_template` builds for one image.
#[test]
fn format_matches_the_reference_template() {
    let expected = "以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。要求を適切に補完するよう応答を書いてください。\n\n### 指示:\n<plamo:image>この画像の色は何色ですか。\n\n### 応答:\n";
    assert_eq!(
        format_plamo2_vl_prompt("この画像の色は何色ですか。", 1),
        expected
    );
}

#[test]
fn bare_and_templated_text_converge() {
    let bare = build_plamo2_vl_prompt("Describe this image.", 1);
    let rendered = format_plamo2_vl_prompt("Describe this image.", 0);
    let server = build_plamo2_vl_prompt(&rendered, 1);
    assert_eq!(bare.0, server.0);
    assert!(!bare.1);
    assert!(server.1);
    // Markers the caller already placed are kept.
    let placed = format_plamo2_vl_prompt("Q", 2);
    assert_eq!(build_plamo2_vl_prompt(&placed, 2).0, placed);
}

/// Toy tokenizer: each `<plamo:image>` is id 100002, every other char its code.
fn toy_encode(text: &str, _add_special: bool) -> Vec<i32> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(stripped) = rest.strip_prefix(PLAMO2_VL_IMAGE_TAG) {
            out.push(100_002);
            rest = stripped;
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch as i32 % 1000 + 10);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
}

#[test]
fn siglip_framing_and_bos() {
    let (tokens, stats) =
        prepare_plamo2_vl_prompt_tokens("Q", &[1], 729, IDS, &mut toy_encode).unwrap();
    assert_eq!(tokens[0], 1, "BOS leads");
    assert_eq!(tokens.iter().filter(|&&t| t == 1).count(), 1);
    let start = tokens.iter().position(|&t| t == 100_000).unwrap();
    let end = tokens.iter().position(|&t| t == 100_001).unwrap();
    assert_eq!(end - start - 1, 729);
    assert!(tokens[start + 1..end].iter().all(|&t| t == 100_002));
    assert_eq!(
        stats,
        Plamo2VlPromptStats {
            image_blocks: 1,
            total_tiles: 1,
            total_image_tokens: 729,
            pre_templated: false,
        }
    );

    // Without `add_bos_token` (the mlx-vlm conversions) nothing is prepended.
    let no_bos = Plamo2VlImageTokenIds { bos: None, ..IDS };
    let (bare, _) =
        prepare_plamo2_vl_prompt_tokens("Q", &[1], 729, no_bos, &mut toy_encode).unwrap();
    assert_eq!(bare.len() + 1, tokens.len());
    assert!(!bare.contains(&1));

    // Two images with 4 and 1 tiles, in order.
    let (tokens, stats) =
        prepare_plamo2_vl_prompt_tokens("Q", &[4, 1], 729, IDS, &mut toy_encode).unwrap();
    assert_eq!(stats.total_image_tokens, 5 * 729);
    let starts: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == 100_000)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(tokens[starts[0] + 1 + 4 * 729], 100_001);
    assert_eq!(tokens[starts[1] + 1 + 729], 100_001);
}

#[test]
fn placeholder_count_mismatch_is_an_error() {
    let err = expand_plamo2_vl_image_tokens(&[1, 5, 6], &[1], 729, IDS).unwrap_err();
    assert!(err.contains("0 <plamo:image>"), "{err}");
}
