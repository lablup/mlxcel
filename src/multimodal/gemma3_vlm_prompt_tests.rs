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
use crate::vlm_prompt::apply_image_token_blocks;

// Gemma 3 vocabulary: `\n` is 107 and each longer run of newlines has the next
// id, up to 31 newlines (137).
const NEWLINE_RUNS: [i32; 31] = [
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
];
const BOI: i32 = 255_999;
const EOI: i32 = 256_000;
const SOFT: i32 = 262_144;

fn gemma3_info(mm_tokens_per_image: usize) -> ImageTokenBlockInfo {
    // `loading::vlm_gemma` builds this for every Gemma 3 VLM checkpoint.
    ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: SOFT,
        mm_tokens_per_image,
        boi_token_id: BOI,
        eoi_token_id: EOI,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: vec![108],
        block_suffix_tokens: vec![108],
    }
}

#[test]
fn newline_run_table_stops_at_first_multi_token_run() {
    let mut encode = |text: &str, _add_special: bool| -> Vec<i32> {
        let len = text.len();
        if len <= 3 {
            vec![106 + len as i32]
        } else {
            vec![109, 106 + (len - 3) as i32]
        }
    };
    assert_eq!(newline_run_token_ids(&mut encode), vec![107, 108, 109]);
}

/// Issue #2547 regression pin, with the real `gemma-3-4b-it` token ids.
///
/// The chat template renders the one-image turn as
/// `<bos><start_of_turn>user\n<start_of_image>Describe the image.<end_of_turn>\n<start_of_turn>model\n`
/// (`TEMPLATE_IDS`). transformers' `Gemma3Processor.apply_chat_template` then
/// yields 272 tokens, with `user` followed by the single `\n\n\n` token (109).
/// The token-level expansion alone produced 273, `\n` (107) then `\n\n` (108).
#[test]
fn gemma3_one_image_prompt_matches_transformers_ids() {
    const TEMPLATE_IDS: [i32; 14] = [
        2, 105, 2364, 107, BOI, 82858, 506, 2471, 236_761, 106, 107, 105, 4368, 107,
    ];
    let info = gemma3_info(256);
    let mut tokens = TEMPLATE_IDS.to_vec();
    apply_image_token_blocks(&mut tokens, info.clone(), 1).unwrap();
    assert_eq!(tokens.len(), 273, "token-level expansion before the merge");

    let merges = merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS);

    let mut expected = vec![2, 105, 2364, 109, BOI];
    expected.extend(std::iter::repeat_n(SOFT, 256));
    expected.extend([
        EOI, 108, 82858, 506, 2471, 236_761, 106, 107, 105, 4368, 107,
    ]);
    assert_eq!(merges, 1);
    assert_eq!(tokens.len(), 272);
    assert_eq!(tokens, expected);
}

#[test]
fn suffix_wrapper_merges_with_a_following_newline_run() {
    // `<start_of_image>\nQ` renders `<end_of_image>\n\n\nQ` in the reference.
    let info = gemma3_info(2);
    let mut tokens = vec![2, BOI, 107, 900];
    apply_image_token_blocks(&mut tokens, info.clone(), 1).unwrap();
    assert_eq!(tokens, vec![2, 108, BOI, SOFT, SOFT, EOI, 108, 107, 900]);

    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS),
        1
    );
    assert_eq!(tokens, vec![2, 108, BOI, SOFT, SOFT, EOI, 109, 900]);
}

#[test]
fn wrapper_next_to_text_or_bos_is_left_alone() {
    let info = gemma3_info(1);
    // Image first after BOS (no template newline) and text right after the block.
    let mut tokens = vec![
        2, 108, BOI, SOFT, EOI, 108, 900, 108, BOI, SOFT, EOI, 108, 106,
    ];
    let before = tokens.clone();
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS),
        0
    );
    assert_eq!(tokens, before);
}

#[test]
fn every_image_block_is_merged() {
    let info = gemma3_info(1);
    let mut tokens = vec![2, 107, BOI, 108, BOI, 900];
    apply_image_token_blocks(&mut tokens, info.clone(), 2).unwrap();
    assert_eq!(
        tokens,
        vec![
            2, 107, 108, BOI, SOFT, EOI, 108, 108, 108, BOI, SOFT, EOI, 108, 900
        ]
    );
    // The reference text is `\n` + `\n\n<boi>..<eoi>\n\n` + `\n\n` +
    // `\n\n<boi>..<eoi>\n\n` + text: three newlines, then six between the
    // blocks, then the last wrapper next to the text.
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS),
        2
    );
    assert_eq!(
        tokens,
        vec![2, 109, BOI, SOFT, EOI, 112, BOI, SOFT, EOI, 108, 900]
    );
}

#[test]
fn a_run_past_the_longest_token_splits_like_the_reference() {
    // 31 + 2 newlines: the reference encodes 33 newlines as [31-run, 2-run],
    // which is what the stream already holds, so nothing changes.
    let info = gemma3_info(1);
    let mut tokens = vec![2, 137, 108, BOI, SOFT, EOI, 108, 106];
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS),
        0
    );
    assert_eq!(tokens, vec![2, 137, 108, BOI, SOFT, EOI, 108, 106]);
    // 30 + 2 newlines are 32, one past the longest token: [31-run, 1-run].
    let mut tokens = vec![2, 136, 108, BOI, SOFT, EOI, 108, 106];
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &info, &NEWLINE_RUNS),
        1
    );
    assert_eq!(tokens, vec![2, 137, 107, BOI, SOFT, EOI, 108, 106]);
}

#[test]
fn non_gemma_layouts_are_untouched() {
    let mut paligemma = ImageTokenBlockInfo {
        use_boi_eoi: false,
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
        ..gemma3_info(1)
    };
    let mut tokens = vec![SOFT, 2, 107, 108];
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &paligemma, &NEWLINE_RUNS),
        0
    );
    paligemma.use_boi_eoi = true;
    assert_eq!(
        merge_image_block_newline_runs(&mut tokens, &paligemma, &NEWLINE_RUNS),
        0
    );
    assert_eq!(tokens, vec![SOFT, 2, 107, 108]);
}
