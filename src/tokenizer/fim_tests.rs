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

//! FIM vocabulary discovery tests (#1442).

use super::*;
use std::collections::HashMap;

fn vocab(entries: &[(&str, u32)]) -> impl Fn(&str) -> Option<u32> + use<> {
    let map: HashMap<String, u32> = entries
        .iter()
        .map(|(name, id)| ((*name).to_string(), *id))
        .collect();
    move |name: &str| map.get(name).copied()
}

#[test]
fn a_qwen_style_vocabulary_resolves_the_triple() {
    let tokens = FimTokens::discover(vocab(&[
        ("<|fim_prefix|>", 151659),
        ("<|fim_middle|>", 151660),
        ("<|fim_suffix|>", 151661),
        ("<|fim_pad|>", 151662),
        ("<|repo_name|>", 151663),
        ("<|file_sep|>", 151664),
    ]));
    let triple = tokens.require_triple().expect("triple resolves");
    assert_eq!(triple.pre.spelling, "<|fim_prefix|>");
    assert_eq!(triple.pre.id, 151659);
    assert_eq!(triple.suf.spelling, "<|fim_suffix|>");
    assert_eq!(triple.mid.spelling, "<|fim_middle|>");
    assert_eq!(
        tokens.rep.as_ref().map(|t| t.spelling),
        Some("<|repo_name|>")
    );
    assert_eq!(
        tokens.sep.as_ref().map(|t| t.spelling),
        Some("<|file_sep|>")
    );
}

#[test]
fn a_deepseek_coder_vocabulary_resolves_its_fullwidth_spellings() {
    // DeepSeek-Coder writes its markers with the FULLWIDTH vertical line
    // U+FF5C, and b10621 scans for exactly that; the ASCII-pipe form does not
    // occur in any published checkpoint and must not be what mlxcel matches on.
    let tokens = FimTokens::discover(vocab(&[
        ("<\u{FF5C}fim\u{2581}begin\u{FF5C}>", 32013),
        ("<\u{FF5C}fim\u{2581}hole\u{FF5C}>", 32014),
        ("<\u{FF5C}fim\u{2581}end\u{FF5C}>", 32015),
    ]));
    let triple = tokens.require_triple().expect("triple resolves");
    assert_eq!(triple.pre.id, 32013);
    assert_eq!(triple.suf.id, 32014);
    assert_eq!(triple.mid.id, 32015);
    assert_eq!(tokens.rep, None);
    assert_eq!(tokens.sep, None);

    let ascii_pipe = FimTokens::discover(vocab(&[
        ("<|fim\u{2581}begin|>", 32013),
        ("<|fim\u{2581}hole|>", 32014),
        ("<|fim\u{2581}end|>", 32015),
    ]));
    assert!(
        !ascii_pipe.supports_infill(),
        "the ASCII-pipe spelling is not in b10621's list and must not resolve"
    );
}

#[test]
fn the_spelling_lists_match_the_pinned_binary() {
    // Read out of libllama in the pinned b10621 archive rather than
    // transcribed. Each vocabulary here declares exactly one spelling, so a
    // dropped or mistyped entry fails rather than being masked by a sibling.
    for spelling in [
        "<|fim_prefix|>",
        "<fim-prefix>",
        "<fim_prefix>",
        "<\u{FF5C}fim\u{2581}begin\u{FF5C}>",
        "<PRE>",
        "\u{2581}<PRE>",
        "<|code_prefix|>",
        "<|prefix|>",
    ] {
        let tokens = FimTokens::discover(vocab(&[(spelling, 1)]));
        assert_eq!(
            tokens.pre.as_ref().map(|t| t.spelling),
            Some(spelling),
            "prefix spelling {spelling:?} is not recognized"
        );
    }
    for spelling in [
        "<|fim_suffix|>",
        "<fim-suffix>",
        "<fim_suffix>",
        "<\u{FF5C}fim\u{2581}hole\u{FF5C}>",
        "<SUF>",
        "\u{2581}<SUF>",
        "<|code_suffix|>",
        "<|suffix|>",
    ] {
        let tokens = FimTokens::discover(vocab(&[(spelling, 1)]));
        assert_eq!(tokens.suf.as_ref().map(|t| t.spelling), Some(spelling));
    }
    for spelling in [
        "<|fim_middle|>",
        "<fim-middle>",
        "<fim_middle>",
        "<\u{FF5C}fim\u{2581}end\u{FF5C}>",
        "<MID>",
        "\u{2581}<MID>",
        "<|code_middle|>",
        "<|middle|>",
    ] {
        let tokens = FimTokens::discover(vocab(&[(spelling, 1)]));
        assert_eq!(tokens.mid.as_ref().map(|t| t.spelling), Some(spelling));
    }
    for spelling in [
        "<|fim_repo|>",
        "<|repo_name|>",
        "<fim-repo>",
        "<REPO>",
        "<reponame>",
    ] {
        let tokens = FimTokens::discover(vocab(&[(spelling, 1)]));
        assert_eq!(tokens.rep.as_ref().map(|t| t.spelling), Some(spelling));
    }
    let tokens = FimTokens::discover(vocab(&[("<|file_sep|>", 1)]));
    assert_eq!(
        tokens.sep.as_ref().map(|t| t.spelling),
        Some("<|file_sep|>")
    );

    // A PAD spelling is resolved by b10621 but never written by format_infill,
    // so mlxcel does not discover it and must not mistake it for a marker.
    let pad = FimTokens::discover(vocab(&[("<|fim_pad|>", 1)]));
    assert_eq!(pad, FimTokens::default());
}

#[test]
fn a_codellama_sentencepiece_vocabulary_resolves_the_underscored_spellings() {
    let tokens = FimTokens::discover(vocab(&[
        ("\u{2581}<PRE>", 32007),
        ("\u{2581}<SUF>", 32008),
        ("\u{2581}<MID>", 32009),
    ]));
    let triple = tokens.require_triple().expect("triple resolves");
    assert_eq!(triple.pre.spelling, "\u{2581}<PRE>");
    assert_eq!(triple.suf.spelling, "\u{2581}<SUF>");
    assert_eq!(triple.mid.spelling, "\u{2581}<MID>");
}

#[test]
fn a_chat_only_vocabulary_names_every_missing_token() {
    let tokens = FimTokens::discover(vocab(&[("<|im_start|>", 1), ("<|im_end|>", 2)]));
    assert!(!tokens.supports_infill());
    let err = tokens.require_triple().expect_err("no FIM tokens");
    assert_eq!(
        err,
        "Infill is not supported by this model: prefix token is missing. suffix token is \
         missing. middle token is missing. "
    );
}

#[test]
fn a_partial_vocabulary_names_only_what_is_missing() {
    let tokens = FimTokens::discover(vocab(&[("<|fim_prefix|>", 10), ("<|fim_middle|>", 11)]));
    let err = tokens.require_triple().expect_err("suffix is missing");
    assert_eq!(
        err,
        "Infill is not supported by this model: suffix token is missing. "
    );
}

#[test]
fn the_first_matching_spelling_wins() {
    // A vocabulary carrying two spellings resolves to upstream's first, so two
    // servers scanning the same checkpoint cannot disagree.
    let tokens = FimTokens::discover(vocab(&[
        ("<fim_prefix>", 2),
        ("<|fim_prefix|>", 1),
        ("<|fim_suffix|>", 3),
        ("<|fim_middle|>", 4),
    ]));
    assert_eq!(tokens.pre.as_ref().map(|t| t.id), Some(1));
}
