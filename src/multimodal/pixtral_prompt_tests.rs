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

use super::*;

// Match the checkpoint ids: [IMG]=10, [IMG_BREAK]=12, [IMG_END]=13.
const IMG: i32 = 10;
const BRK: i32 = 12;
const END: i32 = 13;

/// Build the expected row-structured block for a `rows x cols` grid.
fn expected_block(rows: usize, cols: usize) -> Vec<i32> {
    let mut v = Vec::new();
    for row in 0..rows {
        v.extend(std::iter::repeat_n(IMG, cols));
        if row + 1 < rows {
            v.push(BRK);
        }
    }
    v.push(END);
    v
}

#[test]
fn three_by_five_grid_layout_is_exact() {
    // 3 rows x 5 cols: 5 IMG, BREAK, 5 IMG, BREAK, 5 IMG, END.
    let block = expected_block(3, 5);
    let expected = vec![
        IMG, IMG, IMG, IMG, IMG, BRK, // row 0 + break
        IMG, IMG, IMG, IMG, IMG, BRK, // row 1 + break
        IMG, IMG, IMG, IMG, IMG, END, // row 2 + end (no break after last row)
    ];
    assert_eq!(block, expected);
    assert_eq!(block.iter().filter(|&&t| t == IMG).count(), 15);
    assert_eq!(block.iter().filter(|&&t| t == BRK).count(), 2);
    assert_eq!(block.iter().filter(|&&t| t == END).count(), 1);
}

#[test]
fn single_placeholder_expands_in_place() {
    let mut tokens = vec![1, IMG, 2, 3];
    let stats = insert_pixtral_image_tokens(&mut tokens, &[(3, 5)], IMG, BRK, END).unwrap();

    let mut expected = vec![1];
    expected.extend(expected_block(3, 5));
    expected.extend([2, 3]);
    assert_eq!(tokens, expected);
    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 15);
    // The [IMG] count in the final prompt equals the reported feature count.
    assert_eq!(
        tokens.iter().filter(|&&t| t == IMG).count() as i32,
        stats.total_image_tokens
    );
}

#[test]
fn no_placeholder_splices_after_bos() {
    // BOS then plain text; block goes right after BOS.
    let mut tokens = vec![1, 7, 8, 9];
    let stats = insert_pixtral_image_tokens(&mut tokens, &[(2, 2)], IMG, BRK, END).unwrap();

    let mut expected = vec![1];
    expected.extend(expected_block(2, 2));
    expected.extend([7, 8, 9]);
    assert_eq!(tokens, expected);
    assert_eq!(stats.total_image_tokens, 4);
}

#[test]
fn multiple_images_each_get_own_grid() {
    let mut tokens = vec![1, IMG, 5, IMG, 6];
    let grids = [(2usize, 3usize), (1usize, 4usize)];
    let stats = insert_pixtral_image_tokens(&mut tokens, &grids, IMG, BRK, END).unwrap();

    let mut expected = vec![1];
    expected.extend(expected_block(2, 3));
    expected.push(5);
    expected.extend(expected_block(1, 4));
    expected.push(6);
    assert_eq!(tokens, expected);
    assert_eq!(stats.image_blocks, 2);
    let expected_total: i32 = grids.iter().map(|&(h, w)| (h * w) as i32).sum();
    assert_eq!(stats.total_image_tokens, expected_total);
    assert_eq!(
        tokens.iter().filter(|&&t| t == IMG).count() as i32,
        stats.total_image_tokens
    );
}

#[test]
fn single_row_has_no_break() {
    // A 1-row image emits only IMG*cols followed by END, no break token.
    let block = expected_block(1, 4);
    assert_eq!(block, vec![IMG, IMG, IMG, IMG, END]);
    assert!(!block.contains(&BRK));
}

#[test]
fn placeholder_count_mismatch_returns_none() {
    // Two placeholders but only one grid -> ambiguous, refuse. This is the
    // path a caller reaches by planting a literal [IMG] marker in the prompt;
    // the runtime turns this None into a request error before any encoder or
    // merge runs, so the prompt must be left UNEXPANDED (not partially mutated).
    let mut tokens = vec![1, IMG, IMG, 2];
    let before = tokens.clone();
    assert!(insert_pixtral_image_tokens(&mut tokens, &[(2, 2)], IMG, BRK, END).is_none());
    assert_eq!(
        tokens, before,
        "a rejected expansion must not mutate the prompt"
    );
}

#[test]
fn empty_inputs_return_none() {
    let mut empty: Vec<i32> = Vec::new();
    assert!(insert_pixtral_image_tokens(&mut empty, &[(2, 2)], IMG, BRK, END).is_none());

    let mut tokens = vec![1, IMG, 2];
    assert!(insert_pixtral_image_tokens(&mut tokens, &[], IMG, BRK, END).is_none());
}

#[test]
fn feature_count_matches_across_aspect_ratios() {
    // The [IMG] count must equal tokens_h * tokens_w for a range of shapes.
    for &(rows, cols) in &[(1usize, 1usize), (64, 16), (16, 64), (55, 28), (36, 22)] {
        let mut tokens = vec![1, IMG, 2];
        let stats =
            insert_pixtral_image_tokens(&mut tokens, &[(rows, cols)], IMG, BRK, END).unwrap();
        assert_eq!(stats.total_image_tokens, (rows * cols) as i32);
        assert_eq!(
            tokens.iter().filter(|&&t| t == IMG).count(),
            rows * cols,
            "grid {rows}x{cols}"
        );
    }
}
