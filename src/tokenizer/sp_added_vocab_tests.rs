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

//! SentencePiece added-token matching tests (#1758).

use super::*;
use crate::tokenizer::{MlxcelTokenizer, SentencePieceTokenizer, parse_special_tokens};
use sentencepiece::SentencePieceProcessor;

fn text(s: &str) -> Piece {
    Piece::Text(s.to_string())
}

fn vocab(entries: &[(u32, AddedToken)]) -> SpAddedVocab {
    SpAddedVocab::new(entries)
}

fn plain(content: &str, special: bool) -> AddedToken {
    AddedToken::from(content.to_string(), special)
}

#[test]
fn special_and_non_special_tokens_both_split_out() {
    let v = vocab(&[
        (100, plain("<|im_start|>", true)),
        (101, plain("<tool_call>", false)),
        (102, plain("</tool_call>", false)),
    ]);
    assert_eq!(
        v.split("<|im_start|>call<tool_call>{}</tool_call>", true),
        vec![
            Piece::Token(100),
            text("call"),
            Piece::Token(101),
            text("{}"),
            Piece::Token(102),
        ]
    );
    assert_eq!(v.id_of("<tool_call>"), Some(101));
    assert_eq!(v.id_of("<|im_start|>"), Some(100));
    assert_eq!(v.id_of("tool_call"), None);
}

#[test]
fn text_without_any_added_token_is_one_piece() {
    let v = vocab(&[(101, plain("<tool_call>", false))]);
    assert_eq!(v.split("a < b > c", true), vec![text("a < b > c")]);
    assert_eq!(v.split("", true), Vec::<Piece>::new());
    assert_eq!(SpAddedVocab::new(&[]).split("x", true), vec![text("x")]);
}

/// Leftmost wins over longer, and at one position the longest wins.
#[test]
fn matching_is_leftmost_then_longest() {
    let v = vocab(&[
        (10, plain("<a>", false)),
        (11, plain("<a><b>", false)),
        (12, plain("a><b", false)),
    ]);
    assert_eq!(
        v.split("x<a><b>y<a>", true),
        vec![text("x"), Piece::Token(11), text("y"), Piece::Token(10)]
    );
}

/// `parse_special: false` keeps a control token written into the text as
/// text, while a non-special (user-defined) token is still recognized.
#[test]
fn parse_special_off_matches_only_non_special_tokens() {
    let v = vocab(&[
        (100, plain("<|im_start|>", true)),
        (101, plain("<tool_call>", false)),
    ]);
    assert_eq!(
        v.split("<|im_start|><tool_call>", false),
        vec![text("<|im_start|>"), Piece::Token(101)]
    );
    assert_eq!(v.lookup("<|im_start|>", false), None);
}

#[test]
fn rstrip_and_lstrip_eat_neighbouring_whitespace() {
    let v = vocab(&[
        (1, plain("<r>", false).rstrip(true)),
        (2, plain("<l>", false).lstrip(true)),
    ]);
    assert_eq!(
        v.split("a <r>  b", true),
        vec![text("a "), Piece::Token(1), text("b")]
    );
    assert_eq!(
        v.split("a  <l> b", true),
        vec![text("a"), Piece::Token(2), text(" b")]
    );
    // Fully stripped neighbours disappear rather than becoming empty pieces.
    assert_eq!(v.split("<r>   ", true), vec![Piece::Token(1)]);
}

/// A `single_word` token touching a non-space neighbour is folded into it
/// and so stops being a token, which is what keeps a common substring from
/// firing inside a longer word.
#[test]
fn single_word_tokens_only_match_as_whole_words() {
    let v = vocab(&[(7, plain("cat", false).single_word(true))]);
    assert_eq!(v.split("cat", true), vec![Piece::Token(7)]);
    assert_eq!(
        v.split("a cat sat", true),
        vec![text("a "), Piece::Token(7), text(" sat")]
    );
    assert_eq!(v.split("concat", true), vec![text("concat")]);
    assert_eq!(v.split("catalog", true), vec![text("catalog")]);
    assert_eq!(
        v.split("a catalog", true),
        vec![text("a "), text("catalog")]
    );
}

/// `rstrip` runs first, but the `single_word` fold appends the right
/// neighbour as it was before stripping, exactly as the Python loop does.
#[test]
fn a_single_word_right_fold_uses_the_unstripped_neighbour() {
    let v = vocab(&[(4, plain("<w>", false).rstrip(true).single_word(true))]);
    assert_eq!(v.split("<w>\tx", true), vec![text("<w>\tx")]);
    assert_eq!(v.split("<w> x", true), vec![Piece::Token(4), text("x")]);
}

/// A `single_word` token glued to another added token is folded into it,
/// and the merged string is no longer any token, so both become text.
#[test]
fn a_single_word_token_next_to_another_token_turns_both_into_text() {
    let v = vocab(&[
        (1, plain("<a>", false).single_word(true)),
        (2, plain("<b>", false)),
    ]);
    assert_eq!(v.split("<a><b>", true), vec![text("<a><b>")]);
    assert_eq!(
        v.split("<a> <b>", true),
        vec![Piece::Token(1), text(" "), Piece::Token(2)]
    );
}

/// A multibyte content whose first byte also starts a different character in
/// the text only matches on the character it really is.
#[test]
fn multibyte_contents_match_only_whole_characters() {
    let v = vocab(&[(9, plain("가나", false))]);
    assert_eq!(v.split("각가나", true), vec![text("각"), Piece::Token(9)]);
    assert_eq!(v.split("가각", true), vec![text("가각")]);
}

#[test]
fn a_content_declared_twice_resolves_to_the_higher_id() {
    let v = vocab(&[(5, plain("<x>", false)), (9, plain("<x>", true))]);
    assert_eq!(v.id_of("<x>"), Some(9));
    assert_eq!(v.split("<x>", true), vec![Piece::Token(9)]);
}

#[test]
fn empty_contents_are_ignored() {
    let v = vocab(&[(5, plain("", false))]);
    assert_eq!(v.split("abc", true), vec![text("abc")]);
    assert_eq!(v.id_of(""), None);
}

#[test]
fn multibyte_text_around_tokens_is_preserved() {
    let v = vocab(&[(3, plain("<t>", false))]);
    assert_eq!(
        v.split("가<t>😀", true),
        vec![text("가"), Piece::Token(3), text("😀")]
    );
}

#[test]
fn reading_keeps_flags_and_falls_back_to_added_tokens_json() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = serde_json::json!({
        "added_tokens_decoder": {
            "100": {"content": "<|im_start|>", "special": true},
            "101": {"content": "<tool_call>", "special": false, "rstrip": true},
            "bogus": {"content": "<skip>", "special": true},
            "102": {"special": false}
        }
    });
    let tokens = read_sp_added_tokens(Some(&config), dir.path());
    let summary: Vec<(u32, &str, bool, bool)> = tokens
        .iter()
        .map(|(id, t)| (*id, t.content.as_str(), t.special, t.rstrip))
        .collect();
    assert_eq!(
        summary,
        vec![
            (100, "<|im_start|>", true, false),
            (101, "<tool_call>", false, true)
        ]
    );

    // No decoder table: `added_tokens.json` entries are plain, non-special.
    std::fs::write(
        dir.path().join("added_tokens.json"),
        r#"{"<think>": 7, "</think>": 8}"#,
    )
    .unwrap();
    let tokens = read_sp_added_tokens(Some(&serde_json::json!({})), dir.path());
    let summary: Vec<(u32, &str, bool)> = tokens
        .iter()
        .map(|(id, t)| (*id, t.content.as_str(), t.special))
        .collect();
    assert_eq!(summary, vec![(7, "<think>", false), (8, "</think>", false)]);
    // The decoder wins when both are present.
    let tokens = read_sp_added_tokens(Some(&config), dir.path());
    assert_eq!(tokens.len(), 2);
}

// ---------------------------------------------------------------------------
// End to end through `SentencePieceTokenizer`, on a real (tiny) SentencePiece
// model built in memory so the test needs no checkpoint.
// ---------------------------------------------------------------------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn len_field(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    varint(u64::from(field << 3 | 2), &mut out);
    varint(payload.len() as u64, &mut out);
    out.extend_from_slice(payload);
    out
}

fn varint_field(field: u32, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    varint(u64::from(field << 3), &mut out);
    varint(value, &mut out);
    out
}

/// A character-model `ModelProto`: `<unk>`, `<s>`, `</s>`, then one normal
/// piece per character of `chars`, with an identity normalizer and no dummy
/// prefix so ids are easy to predict (piece `n` of `chars` is id `n + 3`).
fn char_model(chars: &str) -> SentencePieceProcessor {
    let piece = |text: &str, kind: u64| {
        let mut body = len_field(1, text.as_bytes());
        body.push(2 << 3 | 5); // score, fixed32
        body.extend_from_slice(&0f32.to_le_bytes());
        body.extend(varint_field(3, kind));
        len_field(1, &body)
    };
    let mut proto = piece("<unk>", 2);
    proto.extend(piece("<s>", 3));
    proto.extend(piece("</s>", 3));
    for c in chars.chars() {
        proto.extend(piece(&c.to_string(), 1));
    }
    // trainer_spec.model_type = CHAR
    proto.extend(len_field(2, &varint_field(3, 4)));
    let mut normalizer = len_field(1, b"identity");
    normalizer.extend(varint_field(3, 0)); // add_dummy_prefix
    normalizer.extend(varint_field(4, 0)); // remove_extra_whitespaces
    normalizer.extend(varint_field(5, 1)); // escape_whitespaces
    proto.extend(len_field(3, &normalizer));
    SentencePieceProcessor::from_serialized_proto(&proto).expect("char model loads")
}

const CHARS: &str = "<>/|_abceilmnorst{}\u{2581}";

/// Id of a character in [`CHARS`].
fn char_id(c: char) -> u32 {
    CHARS.chars().position(|x| x == c).expect("char in model") as u32 + 3
}

fn sp_tokenizer() -> MlxcelTokenizer {
    let added = vec![
        (100, plain("<|im_start|>", true)),
        (101, plain("<tool_call>", false)),
        (102, plain("</tool_call>", false)),
    ];
    MlxcelTokenizer::SentencePiece(SentencePieceTokenizer::new(
        char_model(CHARS),
        added,
        Some(1),
        false,
    ))
}

#[test]
fn non_special_added_tokens_resolve_and_encode_as_one_id() {
    let tok = sp_tokenizer();
    assert_eq!(tok.token_to_id("<|im_start|>"), Some(100));
    assert_eq!(tok.token_to_id("<tool_call>"), Some(101));
    assert_eq!(tok.token_to_id("</tool_call>"), Some(102));
    // Ordinary pieces still resolve through the model.
    assert_eq!(tok.token_to_id("a"), Some(char_id('a')));

    assert_eq!(tok.encode("<tool_call>", true).unwrap(), vec![101]);
    assert_eq!(
        tok.encode("<|im_start|>a<tool_call>{}</tool_call>", true)
            .unwrap(),
        vec![100, char_id('a'), 101, char_id('{'), char_id('}'), 102]
    );
}

#[test]
fn skip_special_tokens_still_drops_only_special_tokens() {
    let tok = sp_tokenizer();
    let ids = tok
        .encode("<|im_start|>a<tool_call>b</tool_call>", true)
        .unwrap();
    assert_eq!(
        tok.decode(&ids, false).unwrap(),
        "<|im_start|>a<tool_call>b</tool_call>"
    );
    assert_eq!(
        tok.decode(&ids, true).unwrap(),
        "a<tool_call>b</tool_call>",
        "the special token is skipped, the non-special ones are text"
    );
}

#[test]
fn parse_special_off_still_recognizes_non_special_tokens() {
    let tok = sp_tokenizer();
    let ids = tok
        .encode_with_special("<|im_start|><tool_call>", false, false)
        .unwrap();
    assert_eq!(ids.last(), Some(&101));
    assert!(
        !ids.contains(&100),
        "a control spelling stays text: {ids:?}"
    );
    assert_eq!(ids.len(), "<|im_start|>".chars().count() + 1);
}

/// The loader feeds `added_tokens_decoder` straight into the tokenizer:
/// non-special entries are no longer decode-only.
#[test]
fn the_loader_registers_non_special_entries_for_encoding() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::json!({
            "added_tokens_decoder": {
                "100": {"content": "<|im_start|>", "special": true},
                "101": {"content": "<tool_call>", "special": false}
            }
        })
        .to_string(),
    )
    .unwrap();
    let config = parse_special_tokens(dir.path());
    let tok = SentencePieceTokenizer::new(char_model(CHARS), config.added_tokens, None, false);
    assert_eq!(tok.token_to_id("<tool_call>"), Some(101));
    assert_eq!(tok.encode("<tool_call>", true).unwrap(), vec![101]);
    assert_eq!(tok.decode(&[100, 101], true).unwrap(), "<tool_call>");
}

/// A content declared twice, once special at a lower id and once not at a
/// higher one, resolves to the same (higher) id through `token_to_id` and
/// `encode`, as transformers' `_added_tokens_encoder` does.
#[test]
fn token_to_id_and_encode_agree_on_a_duplicated_content() {
    let tok = MlxcelTokenizer::SentencePiece(SentencePieceTokenizer::new(
        char_model(CHARS),
        vec![(100, plain("<x>", true)), (101, plain("<x>", false))],
        None,
        false,
    ));
    assert_eq!(tok.token_to_id("<x>"), Some(101));
    assert_eq!(tok.encode("<x>", true).unwrap(), vec![101]);
}
