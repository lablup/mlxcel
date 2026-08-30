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

use super::{decode_vocab_texts, derive_breaker_heads, resolve_breaker_strings, unescape_breaker};
use crate::tokenizer::MlxcelTokenizer;

/// A tokenizer whose vocabulary is chosen for this test: every entry is a
/// single character and the BPE model has no merges and no byte fallback, so a
/// string tokenizes to one id per character and a character outside the
/// vocabulary tokenizes to nothing.
///
/// That makes all three cases the resolver has to tell apart reachable without
/// a real checkpoint: exactly one token (`a`, and the newline at id 2), more
/// than one (`ab`), and none (`z`). The newline entry is what lets the escape
/// handling be tested end to end rather than only at the string level.
fn fixture_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {"a": 0, "b": 1, "\n": 2, "\t": 3, " ": 4},
            "merges": []
        }
    }"#;
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("fixture tokenizer builds"),
    )
}

/// A tokenizer shaped like the SentencePiece-derived checkpoints in the model
/// zoo: a `Prepend "▁"` normalizer followed by `Replace " " -> "▁"`, which is
/// what Mixtral, Phi-3, MiniCPM, LLaVA and eight other local checkpoints carry.
///
/// This fixture exists because the obvious implementation (encode the breaker
/// on its own) is wrong for exactly this shape, in two different ways, and the
/// plain fixture above cannot express either. With the vocabulary and merges
/// below:
///
/// - `encode("\n")` normalizes to `"▁\n"` and yields TWO tokens, so a newline
///   would have failed startup even though it is a single vocabulary entry.
/// - `encode(" ")` normalizes to `"▁▁"` and yields ONE token, the DOUBLE-space
///   entry, so a space would have silently resolved to the wrong breaker.
fn prepend_normalizer_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
            ]
        },
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {"▁": 0, "a": 1, "\n": 2, "▁a": 3, "▁▁": 4},
            "merges": ["▁ a", "▁ ▁"]
        }
    }"#;
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .expect("prepend-normalizer fixture builds"),
    )
}

fn breakers(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn fixture_tokenizer_behaves_as_the_tests_assume() {
    let tokenizer = fixture_tokenizer();
    assert_eq!(tokenizer.vocab_size(), 5);
    assert_eq!(tokenizer.token_piece_bytes(2), Some(b"\n".to_vec()));
    assert_eq!(
        tokenizer.encode_with_special("b", false, false).unwrap(),
        vec![1]
    );
}

#[test]
fn resolve_breaker_strings_defaults_to_the_b10621_set_when_absent() {
    assert_eq!(
        resolve_breaker_strings(&[]),
        vec![
            "\n".to_string(),
            ":".to_string(),
            "\"".to_string(),
            "*".to_string()
        ],
        "an absent flag inherits b10621's default breaker set"
    );
}

#[test]
fn resolve_breaker_strings_replaces_defaults_and_honors_the_none_sentinel() {
    assert_eq!(
        resolve_breaker_strings(&breakers(&["x"])),
        vec!["x".to_string()],
        "giving the flag replaces the default set outright"
    );
    assert_eq!(
        resolve_breaker_strings(&breakers(&["none"])),
        Vec::<String>::new(),
        "the none sentinel runs DRY with no breakers"
    );
    assert_eq!(
        resolve_breaker_strings(&breakers(&["x", "none", "y"])),
        vec!["y".to_string()],
        "none clears what accumulated so far, later values re-add (b10621's per-value handler)"
    );
}

#[test]
fn resolve_breaker_strings_interprets_shell_escapes_and_skips_empties() {
    assert_eq!(
        resolve_breaker_strings(&breakers(&["\\n", "", "\\t"])),
        vec!["\n".to_string(), "\t".to_string()]
    );
}

#[test]
fn derive_breaker_heads_marks_containing_tokens_as_empty_tail_heads() {
    let tokenizer = fixture_tokenizer();
    let texts = decode_vocab_texts(&tokenizer);
    let heads = derive_breaker_heads(&tokenizer, &texts, &breakers(&["\n"]));
    assert_eq!(
        heads.get(&2),
        Some(&vec![Vec::<i32>::new()]),
        "the newline token contains the breaker outright: an empty-tail head"
    );
    assert_eq!(
        heads.len(),
        1,
        "no other token carries or starts the breaker"
    );
}

#[test]
fn derive_breaker_heads_builds_tails_for_straddled_breakers() {
    let tokenizer = fixture_tokenizer();
    let texts = decode_vocab_texts(&tokenizer);
    // "ab" never fits inside one fixture token, but token "a" ends with its
    // first character, so the head is `a` with the tail tokenization of "b".
    let heads = derive_breaker_heads(&tokenizer, &texts, &breakers(&["ab"]));
    assert_eq!(
        heads.get(&0),
        Some(&vec![vec![1]]),
        "a token ending inside the breaker heads a tail sequence"
    );
    assert!(
        !heads.contains_key(&1),
        "a token that neither contains nor starts the breaker is no head"
    );
}

#[test]
fn derive_breaker_heads_caps_the_breaker_and_the_tail_at_upstream_limits() {
    let tokenizer = fixture_tokenizer();
    let texts = decode_vocab_texts(&tokenizer);
    // A 50-byte breaker truncates to upstream's 40-byte cap, and the derived
    // tail truncates to upstream's 20-token cap.
    let long = "a".repeat(50);
    let heads = derive_breaker_heads(&tokenizer, &texts, &[long]);
    let tails = heads.get(&0).expect("token a heads the truncated breaker");
    assert_eq!(tails.len(), 1);
    assert_eq!(
        tails[0].len(),
        20,
        "the 39-token remainder is capped at MAX_SEQ_LEN 20"
    );
    assert!(tails[0].iter().all(|&id| id == 0));
}

#[test]
fn derive_breaker_heads_over_a_normalizing_tokenizer_uses_decoded_text() {
    // The prepend-normalizer fixture is the shape that broke the old
    // encode-based resolution; the vocabulary SCAN never encodes the breaker
    // itself, so the newline entry resolves as a head directly.
    let tokenizer = prepend_normalizer_tokenizer();
    let texts = decode_vocab_texts(&tokenizer);
    let heads = derive_breaker_heads(&tokenizer, &texts, &breakers(&["\n"]));
    assert!(
        heads
            .get(&2)
            .is_some_and(|tails| tails.iter().any(Vec::is_empty)),
        "the newline vocabulary entry is an empty-tail head, no anchor trick needed"
    );
}

#[test]
fn unescape_breaker_interprets_the_four_supported_escapes() {
    assert_eq!(unescape_breaker("\\n"), "\n");
    assert_eq!(unescape_breaker("\\t"), "\t");
    assert_eq!(unescape_breaker("\\r"), "\r");
    assert_eq!(unescape_breaker("\\\\"), "\\");
}

#[test]
fn unescape_breaker_leaves_everything_else_exactly_as_typed() {
    assert_eq!(unescape_breaker("\\x41"), "\\x41");
    assert_eq!(unescape_breaker("a\\"), "a\\");
    assert_eq!(unescape_breaker("plain"), "plain");
}
