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

use super::{
    MOONDREAM2_LEGACY_BOS_ID, MOONDREAM2_STARMIE_BOS_ID, MOONDREAM2_STARMIE_CAPTION_NORMAL,
    MOONDREAM2_STARMIE_QUERY_PREFIX, MOONDREAM2_STARMIE_QUERY_SUFFIX, Moondream2PromptMode,
    Moondream2PromptStyle, detect_moondream2_prompt_style, prepare_moondream2_prompt_tokens,
};

/// Byte-length encoder: deterministic and injective enough to assert structure.
fn encode(text: &str, _add_special: bool) -> Vec<i32> {
    text.bytes().map(|b| b as i32).collect()
}

// ----------------------------------------------------------------------------
// Template constants
// ----------------------------------------------------------------------------

#[test]
fn legacy_bos_id_is_the_gpt2_endoftext_token() {
    // The legacy moondream2 GPT-2/CodeGen tokenizer uses `<|endoftext|>`
    // (id 50256) as its begin-of-text token.
    assert_eq!(MOONDREAM2_LEGACY_BOS_ID, 50256);
}

#[test]
fn starmie_template_ids_match_the_shipped_reference_and_moondream3() {
    // The 2025-06-21 checkpoint's config.py declares bos_id = 0 and
    // templates.query = {prefix: [1, 15381, 2], suffix: [3]},
    // templates.caption.normal = [1, 32708, 2, 6382, 3] under the
    // moondream/starmie-v1 tokenizer, the same contract the working
    // Moondream3 port uses.
    assert_eq!(
        MOONDREAM2_STARMIE_BOS_ID,
        crate::moondream3_prompt::MOONDREAM3_BOS_ID
    );
    assert_eq!(
        MOONDREAM2_STARMIE_QUERY_PREFIX,
        crate::moondream3_prompt::MOONDREAM3_QUERY_PREFIX
    );
    assert_eq!(
        MOONDREAM2_STARMIE_QUERY_SUFFIX,
        crate::moondream3_prompt::MOONDREAM3_QUERY_SUFFIX
    );
    assert_eq!(
        MOONDREAM2_STARMIE_CAPTION_NORMAL,
        crate::moondream3_prompt::MOONDREAM3_CAPTION_NORMAL
    );
}

// ----------------------------------------------------------------------------
// Starmie-era prompt shaping (2025-06-21+ checkpoints)
// ----------------------------------------------------------------------------

#[test]
fn starmie_image_query_wraps_question_in_template_ids() {
    let prepared = prepare_moondream2_prompt_tokens(
        "Count the cats.",
        1,
        Moondream2PromptStyle::StarmieTemplates,
        encode,
    )
    .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);

    let mut expected = vec![1, 15381, 2];
    expected.extend(encode("Count the cats.", false));
    expected.push(3);
    assert_eq!(prepared.tokens, expected);
}

#[test]
fn starmie_empty_image_prompt_uses_caption_template() {
    let prepared =
        prepare_moondream2_prompt_tokens("   ", 1, Moondream2PromptStyle::StarmieTemplates, encode)
            .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Caption);
    assert_eq!(prepared.tokens, vec![1, 32708, 2, 6382, 3]);
}

#[test]
fn starmie_text_only_leads_with_bos_zero() {
    let prepared = prepare_moondream2_prompt_tokens(
        "What is this?",
        0,
        Moondream2PromptStyle::StarmieTemplates,
        encode,
    )
    .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);

    let mut expected = vec![0, 1, 15381, 2];
    expected.extend(encode("What is this?", false));
    expected.push(3);
    assert_eq!(prepared.tokens, expected);
}

// ----------------------------------------------------------------------------
// Legacy-era prompt shaping (2025-01-09 .. 2025-04-14 checkpoints)
// ----------------------------------------------------------------------------

#[test]
fn legacy_text_only_prompt_leads_with_bos_and_frames_question() {
    let prepared = prepare_moondream2_prompt_tokens(
        "What is this?",
        0,
        Moondream2PromptStyle::LegacyQuestionAnswer,
        encode,
    )
    .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);
    assert_eq!(prepared.tokens[0], MOONDREAM2_LEGACY_BOS_ID);
    let text: String = prepared.tokens[1..]
        .iter()
        .map(|&b| b as u8 as char)
        .collect();
    assert!(text.contains("Question: What is this?"));
    assert!(text.trim_end().ends_with("Answer:"));
}

#[test]
fn legacy_image_prompt_omits_bos_for_wrapper_prefix() {
    let prepared = prepare_moondream2_prompt_tokens(
        "Count the cats.",
        1,
        Moondream2PromptStyle::LegacyQuestionAnswer,
        encode,
    )
    .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);
    // The wrapper prepends BOS + image tokens, so the first token here is text.
    assert_ne!(prepared.tokens.first(), Some(&MOONDREAM2_LEGACY_BOS_ID));
    let text: String = prepared.tokens.iter().map(|&b| b as u8 as char).collect();
    assert!(text.contains("Question: Count the cats."));
}

#[test]
fn legacy_empty_image_prompt_falls_back_to_caption() {
    let prepared = prepare_moondream2_prompt_tokens(
        "   ",
        1,
        Moondream2PromptStyle::LegacyQuestionAnswer,
        encode,
    )
    .unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Caption);
    assert!(!prepared.tokens.is_empty());
}

// ----------------------------------------------------------------------------
// Shared validation
// ----------------------------------------------------------------------------

#[test]
fn text_only_empty_prompt_is_rejected_in_both_styles() {
    for style in [
        Moondream2PromptStyle::StarmieTemplates,
        Moondream2PromptStyle::LegacyQuestionAnswer,
    ] {
        assert!(prepare_moondream2_prompt_tokens("", 0, style, encode).is_err());
    }
}

#[test]
fn multiple_images_are_rejected_in_both_styles() {
    for style in [
        Moondream2PromptStyle::StarmieTemplates,
        Moondream2PromptStyle::LegacyQuestionAnswer,
    ] {
        assert!(prepare_moondream2_prompt_tokens("hi", 2, style, encode).is_err());
    }
}

// ----------------------------------------------------------------------------
// Era detection
// ----------------------------------------------------------------------------

#[test]
fn detects_starmie_era_from_shipped_moondream_py() {
    let dir = tempfile::tempdir().expect("tempdir");
    // The 2025-06-21 snapshot ships moondream.py naming the starmie repo AND
    // the stale GPT-2 tokenizer.json; moondream.py must win.
    std::fs::write(
        dir.path().join("moondream.py"),
        "self.tokenizer = Tokenizer.from_pretrained(\"moondream/starmie-v1\")\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), "{\"model\":{}}").unwrap();
    assert_eq!(
        detect_moondream2_prompt_style(dir.path()),
        Moondream2PromptStyle::StarmieTemplates
    );
}

#[test]
fn detects_legacy_era_from_moondream_py_without_starmie() {
    let dir = tempfile::tempdir().expect("tempdir");
    // 2025-01-09 .. 2025-04-14 snapshots pin the GPT-2 tokenizer repo.
    std::fs::write(
        dir.path().join("moondream.py"),
        "self.tokenizer = Tokenizer.from_pretrained(\n    \"vikhyatk/moondream2\", revision=\"2025-01-09\"\n)\n",
    )
    .unwrap();
    assert_eq!(
        detect_moondream2_prompt_style(dir.path()),
        Moondream2PromptStyle::LegacyQuestionAnswer
    );
}

#[test]
fn detects_starmie_era_from_tokenizer_json_when_moondream_py_is_absent() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A pruned conversion without reference code but with the starmie
    // tokenizer placed locally.
    std::fs::write(
        dir.path().join("tokenizer.json"),
        "{\"added_tokens\":[{\"id\":1,\"content\":\"<|md_reserved_0|>\"}]}",
    )
    .unwrap();
    assert_eq!(
        detect_moondream2_prompt_style(dir.path()),
        Moondream2PromptStyle::StarmieTemplates
    );
}

#[test]
fn detection_defaults_to_legacy_without_discriminating_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        detect_moondream2_prompt_style(dir.path()),
        Moondream2PromptStyle::LegacyQuestionAnswer
    );
}
