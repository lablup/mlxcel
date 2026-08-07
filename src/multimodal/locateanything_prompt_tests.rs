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

//! Unit tests for the LocateAnything image-token placeholder expansion.

use super::*;

/// Token ids from the real `mlx-community/LocateAnything-3B-4bit`
/// `added_tokens.json`.
const IMG_CONTEXT: i32 = 151_665; // <IMG_CONTEXT>
const IMG_START: i32 = 151_666; // <img>
const IMG_END: i32 = 151_667; // </img>
const IM_START: i32 = 151_644; // <|im_start|>

#[test]
fn merged_token_count_matches_the_processor_formula() {
    // (grid_h * grid_w) / (merge_h * merge_w).
    assert_eq!(merged_token_count((6, 6), [2, 2]), 9);
    assert_eq!(merged_token_count((4, 8), [2, 2]), 8);
    assert_eq!(merged_token_count((2, 2), [2, 2]), 1);
    // A degenerate merge kernel must not divide by zero.
    assert_eq!(merged_token_count((3, 3), [0, 0]), 9);
}

#[test]
fn rewrites_the_chat_template_image_markers() {
    // The shipped chat_template.jinja renders `<image-1>` for the first image
    // content part. It is plain text, not a vocabulary token, so it has to be
    // rewritten before tokenization.
    let prompt = "<|im_start|>user\n<image-1>Detect all objects.<|im_end|>\n";
    let (out, stats) = expand_locateanything_image_markers(prompt, &[3])
        .expect("expansion")
        .expect("marker present");

    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 3);
    assert_eq!(
        out,
        "<|im_start|>user\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></img>\
         Detect all objects.<|im_end|>\n"
    );
}

#[test]
fn rewrites_multiple_markers_in_render_order() {
    let prompt = "a<image-1>b<image-2>c";
    let (out, stats) = expand_locateanything_image_markers(prompt, &[1, 2])
        .expect("expansion")
        .expect("markers present");
    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 3);
    assert_eq!(
        out,
        "a<img><IMG_CONTEXT></img>b<img><IMG_CONTEXT><IMG_CONTEXT></img>c"
    );
}

#[test]
fn reports_no_marker_so_the_caller_can_splice_tokens() {
    // A `--no-chat-template` run has no marker; the caller falls back to the
    // token-level splice.
    let out = expand_locateanything_image_markers("Detect all objects.", &[4]).expect("expansion");
    assert!(out.is_none());
}

#[test]
fn a_marker_like_string_that_is_not_a_marker_is_left_alone() {
    // `<image->` has no digits and `<image-1` has no closing bracket, so
    // neither is a marker and neither consumes an image slot.
    let out =
        expand_locateanything_image_markers("<image-> and <image-1 tail", &[4]).expect("expansion");
    assert!(out.is_none());
}

#[test]
fn refuses_a_marker_count_that_disagrees_with_the_image_count() {
    let err = expand_locateanything_image_markers("<image-1><image-2>", &[4])
        .expect_err("two markers, one image");
    assert!(err.contains("more <image-N> markers"), "unexpected: {err}");

    let err = expand_locateanything_image_markers("<image-1>", &[4, 4])
        .expect_err("one marker, two images");
    assert!(err.contains("1 <image-N> marker"), "unexpected: {err}");
}

#[test]
fn splices_a_framed_block_after_the_first_token() {
    let mut prompt = vec![IM_START, 200, 300];
    let stats =
        insert_locateanything_image_tokens(&mut prompt, &[4], IMG_START, IMG_CONTEXT, IMG_END)
            .unwrap();

    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 4);
    // [<|im_start|>, <img>, ctx x4, </img>, body...]
    assert_eq!(prompt[0], IM_START);
    assert_eq!(prompt[1], IMG_START);
    assert_eq!(&prompt[2..6], &[IMG_CONTEXT; 4]);
    assert_eq!(prompt[6], IMG_END);
    assert_eq!(&prompt[7..], &[200, 300]);
}

#[test]
fn expands_an_existing_placeholder_in_place() {
    let mut prompt = vec![IM_START, IMG_CONTEXT, 300];
    let stats =
        insert_locateanything_image_tokens(&mut prompt, &[9], IMG_START, IMG_CONTEXT, IMG_END)
            .unwrap();

    assert_eq!(stats.total_image_tokens, 9);
    assert_eq!(prompt[0], IM_START);
    assert_eq!(prompt[1], IMG_START);
    assert_eq!(&prompt[2..11], &[IMG_CONTEXT; 9]);
    assert_eq!(prompt[11], IMG_END);
    assert_eq!(prompt[12], 300);
}

#[test]
fn per_image_counts_drive_independent_block_sizes() {
    // Native resolution: two images with different grids get different runs.
    let mut prompt = vec![IM_START, 200];
    let stats =
        insert_locateanything_image_tokens(&mut prompt, &[9, 4], IMG_START, IMG_CONTEXT, IMG_END)
            .unwrap();

    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 13);
    // block 0: <img> + 9 ctx + </img>, block 1: <img> + 4 ctx + </img>.
    assert_eq!(prompt[1], IMG_START);
    assert_eq!(&prompt[2..11], &[IMG_CONTEXT; 9]);
    assert_eq!(prompt[11], IMG_END);
    assert_eq!(prompt[12], IMG_START);
    assert_eq!(&prompt[13..17], &[IMG_CONTEXT; 4]);
    assert_eq!(prompt[17], IMG_END);
    assert_eq!(prompt[18], 200);
}

#[test]
fn context_token_run_length_equals_the_feature_row_count() {
    // The invariant the merge step depends on: the number of <IMG_CONTEXT> ids
    // in the final prompt must equal the sum of the per-image counts.
    let mut prompt = vec![IM_START, 7, 8, 9];
    let counts = [9usize, 4, 25];
    let stats =
        insert_locateanything_image_tokens(&mut prompt, &counts, IMG_START, IMG_CONTEXT, IMG_END)
            .unwrap();

    let observed = prompt.iter().filter(|&&t| t == IMG_CONTEXT).count();
    assert_eq!(observed, counts.iter().sum::<usize>());
    assert_eq!(observed, stats.total_image_tokens);
    // Framing tokens are emitted exactly once per image.
    assert_eq!(prompt.iter().filter(|&&t| t == IMG_START).count(), 3);
    assert_eq!(prompt.iter().filter(|&&t| t == IMG_END).count(), 3);
}

#[test]
fn returns_none_for_empty_inputs() {
    let mut empty: Vec<i32> = vec![];
    assert!(
        insert_locateanything_image_tokens(&mut empty, &[4], IMG_START, IMG_CONTEXT, IMG_END)
            .is_none()
    );
    let mut prompt = vec![IM_START, 1];
    assert!(
        insert_locateanything_image_tokens(&mut prompt, &[], IMG_START, IMG_CONTEXT, IMG_END)
            .is_none()
    );
}
