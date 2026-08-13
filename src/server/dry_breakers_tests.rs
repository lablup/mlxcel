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

use super::{resolve_dry_sequence_breakers, unescape_breaker};
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

fn breakers(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn fixture_tokenizer_behaves_as_the_tests_assume() {
    let tokenizer = fixture_tokenizer();

    assert_eq!(tokenizer.encode("a", false).unwrap(), vec![0]);
    assert_eq!(tokenizer.encode("\n", false).unwrap(), vec![2]);
    assert_eq!(tokenizer.encode("ab", false).unwrap(), vec![0, 1]);
    assert_eq!(tokenizer.encode("z", false).unwrap(), Vec::<u32>::new());
}

#[test]
fn unescape_breaker_interprets_the_four_supported_escapes() {
    assert_eq!(unescape_breaker("\\n"), "\n");
    assert_eq!(unescape_breaker("\\t"), "\t");
    assert_eq!(unescape_breaker("\\r"), "\r");
    assert_eq!(unescape_breaker("\\\\"), "\\");
    // `\\n` is a literal backslash followed by an `n`, not a newline: the
    // escape hatch has to actually work, or a breaker containing a backslash
    // could not be expressed at all.
    assert_eq!(unescape_breaker("\\\\n"), "\\n");
}

#[test]
fn unescape_breaker_leaves_everything_else_exactly_as_typed() {
    // A string with no backslash is untouched, including a single space,
    // which is a legitimate breaker in most vocabularies.
    assert_eq!(unescape_breaker(" "), " ");
    assert_eq!(unescape_breaker("Hello"), "Hello");
    // An unrecognised escape is preserved rather than guessed at, so a
    // Windows-style path fragment is never silently rewritten.
    assert_eq!(unescape_breaker("\\d"), "\\d");
    assert_eq!(unescape_breaker("a\\qb"), "a\\qb");
    // A trailing lone backslash keeps itself instead of being dropped.
    assert_eq!(unescape_breaker("a\\"), "a\\");
}

#[test]
fn resolve_dry_sequence_breakers_maps_single_token_strings_to_ids() {
    let tokenizer = fixture_tokenizer();

    let resolved =
        resolve_dry_sequence_breakers(&tokenizer, &breakers(&["a", "b"])).expect("must resolve");

    assert_eq!(resolved, vec![0, 1]);
}

/// The end-to-end shape of the documented usage: a shell passes
/// `--dry-sequence-breaker '\n'` through as the two characters `\` and `n`,
/// and that has to reach the sampler as the newline token, not as whatever
/// the literal two-character string happens to encode to.
#[test]
fn resolve_dry_sequence_breakers_accepts_the_escapes_the_help_text_advertises() {
    let tokenizer = fixture_tokenizer();

    let resolved = resolve_dry_sequence_breakers(&tokenizer, &breakers(&["\\n", "\\t"]))
        .expect("the documented `\\n` and `\\t` examples must resolve");

    assert_eq!(resolved, vec![2, 3]);
}

#[test]
fn resolve_dry_sequence_breakers_rejects_a_multi_token_string_by_name() {
    let tokenizer = fixture_tokenizer();

    let error = resolve_dry_sequence_breakers(&tokenizer, &breakers(&["a", "ab"]))
        .expect_err("a multi-token breaker must fail startup");
    let message = error.to_string();

    assert!(
        message.contains("\"ab\""),
        "the error must name the offending string so the operator can find it: {message}"
    );
    assert!(
        message.contains("2 tokens"),
        "the error must say how many tokens it encoded to: {message}"
    );
    assert!(
        message.contains("--dry-sequence-breaker"),
        "the error must name the flag: {message}"
    );
}

/// A breaker the vocabulary cannot represent at all is the same class of
/// failure as a multi-token one, and must not be mistaken for "no breaker".
#[test]
fn resolve_dry_sequence_breakers_rejects_a_string_that_encodes_to_nothing() {
    let tokenizer = fixture_tokenizer();

    let error = resolve_dry_sequence_breakers(&tokenizer, &breakers(&["z"]))
        .expect_err("a breaker outside the vocabulary must fail startup");
    let message = error.to_string();

    assert!(message.contains("\"z\""), "must name the string: {message}");
    assert!(
        message.contains("0 tokens"),
        "must report the count it actually got: {message}"
    );
}

#[test]
fn resolve_dry_sequence_breakers_skips_empty_entries_from_a_stray_delimiter() {
    let tokenizer = fixture_tokenizer();

    // `--dry-sequence-breaker 'a,'` splits to `["a", ""]` on the clap value
    // delimiter. The trailing empty is a typo, not a breaker.
    let resolved = resolve_dry_sequence_breakers(&tokenizer, &breakers(&["a", ""]))
        .expect("a stray delimiter must not fail startup");

    assert_eq!(resolved, vec![0]);
}

/// A single space is a real breaker in most vocabularies, so the skip rule has
/// to be "empty", not "blank after trimming".
#[test]
fn resolve_dry_sequence_breakers_keeps_a_whitespace_breaker() {
    let tokenizer = fixture_tokenizer();

    let resolved = resolve_dry_sequence_breakers(&tokenizer, &breakers(&[" "]))
        .expect("a space is a legitimate single-token breaker");

    assert_eq!(resolved, vec![4]);
}

#[test]
fn resolve_dry_sequence_breakers_rejects_a_flag_that_yielded_nothing() {
    let tokenizer = fixture_tokenizer();

    let error = resolve_dry_sequence_breakers(&tokenizer, &breakers(&["", ""]))
        .expect_err("a flag that produced no breaker at all is a configuration error");

    assert!(
        error.to_string().contains("no usable breaker"),
        "the error must say the flag was set but produced nothing: {error}"
    );
}

#[test]
fn resolve_dry_sequence_breakers_accepts_an_absent_flag() {
    let tokenizer = fixture_tokenizer();

    // An absent flag is an empty slice, which is the default and must not be
    // confused with "set but empty".
    assert_eq!(
        resolve_dry_sequence_breakers(&tokenizer, &[]).expect("an absent flag must be fine"),
        Vec::<i32>::new()
    );
}
