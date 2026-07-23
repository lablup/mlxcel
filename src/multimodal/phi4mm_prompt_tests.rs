// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// Licensed under the Apache License, Version 2.0

use super::{PHI4MM_AUDIO_TOKEN_ID, expand_phi4mm_placeholders, prepare_phi4mm_prompt_tokens};
use crate::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;

fn encode_bytes(text: &str, add_special: bool) -> Vec<i32> {
    let mut out = if add_special { vec![101] } else { Vec::new() };
    out.extend(text.bytes().map(i32::from));
    out
}

#[test]
fn normalizes_mixed_numbered_tags_in_prompt_order() {
    let prepared = prepare_phi4mm_prompt_tokens(
        "<|user|>\n<|image_1|>\n<|audio_1|>\nDescribe both.",
        1,
        1,
        encode_bytes,
    )
    .unwrap();
    assert_eq!(prepared.image_slots, 1);
    assert_eq!(prepared.audio_slots, 1);
    let image = prepared
        .tokens
        .iter()
        .position(|token| *token == PHI4_SIGLIP_IMAGE_TOKEN_INDEX)
        .unwrap();
    let audio = prepared
        .tokens
        .iter()
        .position(|token| *token == PHI4MM_AUDIO_TOKEN_ID)
        .unwrap();
    assert!(image < audio);
}

#[test]
fn synthesizes_one_ordered_tag_per_audio_clip() {
    let prepared =
        prepare_phi4mm_prompt_tokens("<|user|>\nTranscribe.", 0, 2, encode_bytes).unwrap();
    assert_eq!(prepared.audio_slots, 2);
    assert_eq!(
        prepared
            .tokens
            .iter()
            .filter(|token| **token == PHI4MM_AUDIO_TOKEN_ID)
            .count(),
        2
    );

    let exact = prepare_phi4mm_prompt_tokens("<|user|>Transcribe.", 0, 1, encode_bytes).unwrap();
    let audio = exact
        .tokens
        .iter()
        .position(|token| *token == PHI4MM_AUDIO_TOKEN_ID)
        .unwrap();
    assert_eq!(exact.tokens[audio + 1], i32::from(b'T'));
}

#[test]
fn synthesized_image_sentinel_is_directly_followed_by_prompt_text() {
    let prepared = prepare_phi4mm_prompt_tokens("<|user|>Describe it.", 1, 0, encode_bytes)
        .expect("missing image tag should be synthesized");
    let position = prepared
        .tokens
        .iter()
        .position(|token| *token == PHI4_SIGLIP_IMAGE_TOKEN_INDEX)
        .expect("image sentinel");
    assert_eq!(prepared.tokens[position + 1], b'D' as i32);
}

#[test]
fn rejects_duplicate_out_of_order_missing_and_extra_audio_tags() {
    for (prompt, count) in [
        ("<|audio_1|><|audio_1|>", 2),
        ("<|audio_2|><|audio_1|>", 2),
        ("<|audio_1|>", 2),
        ("<|audio_1|><|audio_2|>", 1),
    ] {
        assert!(prepare_phi4mm_prompt_tokens(prompt, 0, count, encode_bytes).is_err());
    }
}

#[test]
fn rejects_audio_tag_without_audio_payload() {
    let error = prepare_phi4mm_prompt_tokens("<|audio_1|>", 0, 0, encode_bytes).unwrap_err();
    assert!(error.contains("exactly once"));
}

#[test]
fn rejects_unnumbered_or_partially_malformed_audio_tags() {
    for prompt in ["<|audio|>", "<|audio_1|><|audio|>"] {
        let error = prepare_phi4mm_prompt_tokens(prompt, 0, 1, encode_bytes).unwrap_err();
        assert!(error.contains("expected <|audio_N|>"), "{error}");
    }
}

#[test]
fn expands_mixed_placeholders_in_prompt_order_and_rejects_drift() {
    let tokens = [
        PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
        PHI4MM_AUDIO_TOKEN_ID,
        PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
    ];
    assert_eq!(
        expand_phi4mm_placeholders(&tokens, &[2, 1], &[3]).unwrap(),
        vec![
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
            PHI4MM_AUDIO_TOKEN_ID,
            PHI4MM_AUDIO_TOKEN_ID,
            PHI4MM_AUDIO_TOKEN_ID,
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
        ]
    );
    assert!(expand_phi4mm_placeholders(&tokens, &[2], &[3]).is_err());
    assert!(expand_phi4mm_placeholders(&tokens, &[2, 1], &[]).is_err());
    assert!(expand_phi4mm_placeholders(&tokens, &[2, 1, 4], &[3]).is_err());
}

#[test]
fn adjacent_audio_tags_expand_as_distinct_clip_runs() {
    let prepared =
        prepare_phi4mm_prompt_tokens("<|audio_1|><|audio_2|>", 0, 2, encode_bytes).unwrap();
    assert_eq!(
        expand_phi4mm_placeholders(&prepared.tokens, &[], &[2, 3])
            .unwrap()
            .iter()
            .filter(|token| **token == PHI4MM_AUDIO_TOKEN_ID)
            .count(),
        5
    );
}
