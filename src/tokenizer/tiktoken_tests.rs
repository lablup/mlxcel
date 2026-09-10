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

//! Tests for the tiktoken BPE backend and its two families.

use super::*;
use std::path::PathBuf;

fn test_model_path() -> PathBuf {
    PathBuf::from("models/hunyuan-13b")
}

fn tiktoken_file() -> PathBuf {
    test_model_path().join("hy.tiktoken")
}

/// Write a synthetic `.tiktoken` vocabulary of `ranks` single-byte tokens
/// plus the `tokenizer_config.json` a family would ship, and load it.
fn synthetic_tokenizer(
    ranks: usize,
    tokenizer_config: &str,
) -> (tempfile::TempDir, TiktokenTokenizer) {
    use base64::Engine;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut vocab = String::new();
    for rank in 0..ranks {
        let token = base64::engine::general_purpose::STANDARD.encode([b'a' + rank as u8]);
        vocab.push_str(&format!("{token} {rank}\n"));
    }
    std::fs::write(dir.path().join("vocab.tiktoken"), vocab).expect("write vocab");
    std::fs::write(dir.path().join("tokenizer_config.json"), tokenizer_config)
        .expect("write config");
    let tokenizer =
        TiktokenTokenizer::from_file(&dir.path().join("vocab.tiktoken"), dir.path())
            .expect("load tiktoken");
    (dir, tokenizer)
}

/// `tokenizer_class == "QWenTokenizer"` selects the QWen special table, so
/// the ids follow `tokenization_qwen.py`: `<|endoftext|>` at
/// `len(mergeable_ranks)`, `<|im_start|>`/`<|im_end|>` next, 205 extras,
/// then the nine image/reference tags ending at `<imgpad>`.
///
/// With a 3-rank vocabulary that is `<|im_end|>` = 5 and `<imgpad>` = 219,
/// the same offsets that put them at 151645 and 151859 on GOT-OCR 2.0's
/// 151643-rank `qwen.tiktoken`.
#[test]
fn qwen_tokenizer_class_selects_qwen_specials() {
    let (_dir, t) = synthetic_tokenizer(
        3,
        r#"{"tokenizer_class": "QWenTokenizer", "added_tokens_decoder": {}}"#,
    );
    assert_eq!(t.token_to_id("<|endoftext|>"), Some(3));
    assert_eq!(t.token_to_id("<|im_start|>"), Some(4));
    assert_eq!(t.token_to_id("<|im_end|>"), Some(5));
    assert_eq!(t.token_to_id("<|extra_0|>"), Some(6));
    assert_eq!(t.token_to_id("<|extra_204|>"), Some(210));
    assert_eq!(t.token_to_id("<ref>"), Some(211));
    assert_eq!(t.token_to_id("<img>"), Some(217));
    assert_eq!(t.token_to_id("</img>"), Some(218));
    assert_eq!(t.token_to_id("<imgpad>"), Some(219));
    // The HunYuan-only spellings must not exist under the QWen table.
    assert_eq!(t.token_to_id("<|startoftext|>"), None);
    assert_eq!(t.token_to_id("<|bos|>"), None);
}

/// A checkpoint that declares the tokenizer only through `auto_map` (no
/// `tokenizer_class`) still reaches the QWen table.
#[test]
fn qwen_auto_map_selects_qwen_specials() {
    let (_dir, t) = synthetic_tokenizer(
        3,
        r#"{"auto_map": {"AutoTokenizer": ["tokenization_qwen.QWenTokenizer", null]}}"#,
    );
    assert_eq!(t.token_to_id("<|im_end|>"), Some(5));
    assert_eq!(t.token_to_id("<imgpad>"), Some(219));
}

/// Without the QWen marker the HunYuan table is unchanged: five named
/// specials then 205 extras, and none of the QWen-only spellings resolve.
#[test]
fn hunyuan_table_unchanged_without_qwen_class() {
    let (_dir, t) = synthetic_tokenizer(3, r#"{"added_tokens_decoder": {}}"#);
    assert_eq!(t.token_to_id("<|endoftext|>"), Some(3));
    assert_eq!(t.token_to_id("<|startoftext|>"), Some(4));
    assert_eq!(t.token_to_id("<|bos|>"), Some(5));
    assert_eq!(t.token_to_id("<|eos|>"), Some(6));
    assert_eq!(t.token_to_id("<|pad|>"), Some(7));
    assert_eq!(t.token_to_id("<|extra_0|>"), Some(8));
    assert_eq!(t.token_to_id("<|im_end|>"), None);
    assert_eq!(t.token_to_id("<imgpad>"), None);
}

/// The QWen framing tags round-trip through encode/decode as single ids,
/// which is what lets the fixed GOT conversation carry an image block as
/// plain text.
#[test]
fn qwen_image_tags_encode_as_single_specials() {
    let (_dir, t) = synthetic_tokenizer(3, r#"{"tokenizer_class": "QWenTokenizer"}"#);
    let ids = t.encode("<img><imgpad></img>", false).expect("encode");
    assert_eq!(ids, vec![217, 219, 218]);
    assert_eq!(
        t.decode(&ids, false).expect("decode"),
        "<img><imgpad></img>"
    );
    assert_eq!(t.decode(&ids, true).expect("decode"), "");
}

#[test]
#[ignore] // Requires model files
fn test_load_tiktoken() {
    let tokenizer = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path());
    assert!(
        tokenizer.is_ok(),
        "Failed to load tiktoken: {:?}",
        tokenizer.err()
    );
    let t = tokenizer.unwrap();
    assert!(!t.encoder.is_empty());
    assert!(!t.special_encoder.is_empty());
    assert!(t.special_encoder.contains_key("<|eos|>"));
}

#[test]
#[ignore] // Requires model files
fn test_encode_decode_roundtrip() {
    let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
    let text = "Hello, world!";
    let ids = t.encode(text, false).unwrap();
    assert!(!ids.is_empty());
    let decoded = t.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

#[test]
#[ignore] // Requires model files
fn test_encode_chinese() {
    let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
    let text = "你好，世界";
    let ids = t.encode(text, false).unwrap();
    assert!(!ids.is_empty());
    let decoded = t.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

#[test]
#[ignore] // Requires model files
fn test_special_tokens() {
    let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
    let text = "<|eos|>";
    let ids = t.encode(text, false).unwrap();
    assert_eq!(ids.len(), 1);
    let decoded = t.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// Kimi K3 family
// ---------------------------------------------------------------------------

/// The tokenizer-only Kimi K3 checkpoint the K3 tests read.
///
/// It ships `config.json`, `generation_config.json`, `tokenizer_config.json`
/// and `tiktoken.model` and no weights, so nothing here loads a model.
const K3_DIR: &str = "models/kimi-k3-tokenizer";

/// Load the K3 tokenizer, or report the skip and return `None`.
fn k3_tokenizer(test_name: &str) -> Option<TiktokenTokenizer> {
    let dir = std::path::Path::new(K3_DIR);
    if !dir.join("tiktoken.model").exists() {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            test_name,
            "models/kimi-k3-tokenizer is absent",
        );
        return None;
    }
    Some(
        TiktokenTokenizer::from_file(&dir.join("tiktoken.model"), dir)
            .expect("load the Kimi K3 tiktoken vocabulary"),
    )
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from("tests/fixtures/kimi_k3").join(name)
}

#[test]
fn kimi_k3_family_is_detected_from_config_model_type() {
    let dir = std::path::Path::new(K3_DIR);
    if !dir.join("config.json").exists() {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            "kimi_k3_family_is_detected_from_config_model_type",
            "models/kimi-k3-tokenizer is absent",
        );
        return;
    }
    assert_eq!(TiktokenFamily::detect(dir), TiktokenFamily::KimiK3);
}

#[test]
fn control_tokens_named_from_config_else_reserved() {
    let Some(tok) = k3_tokenizer("control_tokens_named_from_config_else_reserved") else {
        return;
    };
    assert_eq!(tok.family(), TiktokenFamily::KimiK3);
    assert_eq!(tok.vocab_size(), 163840);
    assert_eq!(tok.control_id("<|open|>"), Some(163587));
    assert_eq!(tok.control_id("<|close|>"), Some(163588));
    assert_eq!(tok.control_id("<|sep|>"), Some(163589));
    assert_eq!(tok.control_id("<|end_of_msg|>"), Some(163586));
    assert_eq!(tok.control_id("<|media_pad|>"), Some(163605));
    assert_eq!(tok.control_id("[BOS]"), Some(163584));
    assert_eq!(tok.control_id("[PAD]"), Some(163839));
    // 163592 has no `added_tokens_decoder` entry, so it keeps the reserved name.
    assert_eq!(tok.control_id("<|reserved_token_163592|>"), Some(163592));
    let ids = tok.kimi_k3_control_ids().expect("K3 control ids");
    assert_eq!(ids.open, 163587);
    assert_eq!(ids.close, 163588);
    assert_eq!(ids.sep, 163589);
    assert_eq!(ids.end_of_msg, 163586);
    assert_eq!(ids.bos, 163584);
    assert_eq!(ids.eos, 163585);
}

#[test]
fn every_control_token_matches_the_reference_name_table() {
    let Some(tok) = k3_tokenizer("every_control_token_matches_the_reference_name_table") else {
        return;
    };
    let raw = std::fs::read_to_string(fixture_path("control_tokens.json"))
        .expect("read control_tokens.json");
    let table: serde_json::Value = serde_json::from_str(&raw).expect("parse control_tokens.json");

    assert_eq!(
        tok.vocab_size() as u64,
        table["vocab_size"].as_u64().unwrap()
    );
    let base = table["base"].as_u64().unwrap() as u32;
    assert_eq!(tok.vocab_size() as u32, base + 256);

    // All 256 entries, not a hand-picked few: an off-by-one anywhere in the
    // block would shift every id above it and still pass a spot check.
    let names = table["names"].as_object().expect("names");
    assert_eq!(names.len(), 256);
    for (id_str, name) in names {
        let id: u32 = id_str.parse().expect("id");
        let name = name.as_str().expect("name");
        assert_eq!(
            tok.control_id(name),
            Some(id),
            "control token {name:?} should be id {id}"
        );
        assert_eq!(
            tok.decode(&[id], false).expect("decode"),
            name,
            "id {id} should decode to {name:?}"
        );
        // `skip_special_tokens` drops the whole block.
        assert_eq!(tok.decode(&[id], true).expect("decode"), "");
    }

    assert_eq!(
        tok.control_id("[BOS]").map(u64::from),
        table["bos_id"].as_u64()
    );
    assert_eq!(
        tok.control_id("[EOS]").map(u64::from),
        table["eos_id"].as_u64()
    );
    assert_eq!(
        tok.control_id("[PAD]").map(u64::from),
        table["pad_id"].as_u64()
    );
    assert_eq!(
        tok.control_id("[UNK]").map(u64::from),
        table["unk_id"].as_u64()
    );
}

#[test]
fn encode_text_never_emits_control_ids() {
    let Some(tok) = k3_tokenizer("encode_text_never_emits_control_ids") else {
        return;
    };
    let base = tok.vocab_size() as u32 - 256;
    for spelling in [
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
        "[BOS]",
        "<|kimi_image_placeholder|>",
    ] {
        let ids = tok.encode_text(spelling).expect("encode_text");
        assert!(
            ids.iter().all(|&id| id < base),
            "{spelling:?} leaked a control id: {ids:?}"
        );
        assert_eq!(
            tok.decode(&ids, true).expect("decode"),
            spelling,
            "{spelling:?} did not round-trip as ordinary text"
        );
    }
    // The generic `encode` still recognizes a spelling, which is what a caller
    // passing an already rendered prompt needs.
    assert_eq!(tok.encode("<|open|>", true).expect("encode"), vec![163587]);
    // `encode_without_special_parsing` is the same call as `encode_text`.
    assert_eq!(
        tok.encode_without_special_parsing("<|open|>")
            .expect("encode"),
        tok.encode_text("<|open|>").expect("encode_text")
    );
}

#[test]
fn k3_eos_is_end_of_msg_and_bos_is_never_prepended() {
    let dir = std::path::Path::new(K3_DIR);
    let Some(tok) = k3_tokenizer("k3_eos_is_end_of_msg_and_bos_is_never_prepended") else {
        return;
    };
    // `generation_config.json` carries `eos_token_id: 163586`, which is
    // `<|end_of_msg|>` rather than `[EOS]`.
    assert_eq!(crate::loading::read_eos_token_ids(dir), vec![163586]);
    for text in ["hello", "", "안녕하세요"] {
        let ids = tok.encode(text, true).expect("encode");
        assert!(
            !ids.contains(&163584),
            "encode({text:?}, add_special=true) prepended [BOS]: {ids:?}"
        );
    }
}

#[test]
fn k3_pattern_splits_han_camel_case_and_digits() {
    let Some(tok) = k3_tokenizer("k3_pattern_splits_han_camel_case_and_digits") else {
        return;
    };
    let raw = std::fs::read_to_string(fixture_path("pretokenize_pins.json"))
        .expect("read pretokenize_pins.json");
    let pins: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("parse pins");
    assert!(!pins.is_empty());
    for pin in &pins {
        let text = pin["text"].as_str().expect("pin text");
        let pieces: Vec<&str> = pin["pieces"]
            .as_array()
            .expect("pieces")
            .iter()
            .map(|p| p.as_str().expect("piece"))
            .collect();
        let expected: Vec<u32> = pin["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();

        assert_eq!(
            tok.encode_text(text).expect("encode_text"),
            expected,
            "whole-string ids differ for {text:?}"
        );

        // The reference's own per-piece split must concatenate to the same ids,
        // which is what pins the pre-tokenization boundaries rather than only
        // the final BPE output.
        let mut per_piece = Vec::new();
        for piece in &pieces {
            per_piece.extend(tok.encode_text(piece).expect("encode_text piece"));
        }
        assert_eq!(
            per_piece, expected,
            "per-piece ids differ for {text:?} (pieces {pieces:?})"
        );
    }

    // The documented split for the issue's own example.
    let camel = &pins
        .iter()
        .find(|p| p["text"] == "HelloWorld 漢字テスト 12345 it's")
        .expect("the CamelCase pin");
    let pieces: Vec<&str> = camel["pieces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert_eq!(
        pieces,
        vec![
            "Hello",
            "World",
            " ",
            "漢字",
            "テスト",
            " ",
            "123",
            "45",
            " it's"
        ]
    );
}

#[test]
fn k3_corpus_matches_the_reference_tokenizer() {
    let Some(tok) = k3_tokenizer("k3_corpus_matches_the_reference_tokenizer") else {
        return;
    };
    let corpus = std::fs::read_to_string(fixture_path("corpus.txt")).expect("read corpus.txt");
    let expected =
        std::fs::read_to_string(fixture_path("corpus_ids.jsonl")).expect("read corpus_ids.jsonl");

    let mut lines: Vec<&str> = corpus.split('\n').collect();
    if corpus.ends_with('\n') {
        lines.pop();
    }
    let expected_lines: Vec<&str> = expected.lines().collect();
    assert_eq!(lines.len(), 1000, "the corpus fixture is 1000 lines");
    assert_eq!(lines.len(), expected_lines.len());

    let mut matched = 0usize;
    for (index, (line, want)) in lines.iter().zip(expected_lines.iter()).enumerate() {
        let ids = tok.encode_text(line).expect("encode_text");
        let got = serde_json::to_string(&ids).expect("serialize ids");
        assert_eq!(&got, want, "line {} differs for {line:?}", index + 1);
        matched += 1;
    }
    assert_eq!(matched, 1000);

    // The whole file as one document exercises the pattern's newline
    // alternatives, which per-line encoding never reaches.
    let whole_expected = std::fs::read_to_string(fixture_path("corpus_whole_ids.json"))
        .expect("read corpus_whole_ids.json");
    let whole = tok.encode_text(&corpus).expect("encode whole corpus");
    assert_eq!(
        serde_json::to_string(&whole).expect("serialize ids"),
        whole_expected.trim_end()
    );
}

// ---------------------------------------------------------------------------
// Synthetic vocabularies (no checkpoint required)
// ---------------------------------------------------------------------------

/// Write a minimal `.tiktoken` file: every single byte plus a couple of
/// multi-byte merges, so BPE has something to do.
fn write_synthetic_tiktoken(dir: &std::path::Path) -> PathBuf {
    use base64::Engine;
    let mut lines = String::new();
    let mut rank = 0u32;
    for byte in 0u8..=255 {
        let encoded = base64::engine::general_purpose::STANDARD.encode([byte]);
        lines.push_str(&format!("{encoded} {rank}\n"));
        rank += 1;
    }
    for token in ["ab", "abc", "  ", "hello"] {
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        lines.push_str(&format!("{encoded} {rank}\n"));
        rank += 1;
    }
    let path = dir.join("tiktoken.model");
    std::fs::write(&path, lines).expect("write synthetic tiktoken file");
    path
}

/// A real HunYuan `.tiktoken` checkpoint, if this machine has one.
///
/// Kimi K3 added a second family to this loader; this is the regression guard
/// that the first one still loads from a real vocabulary rather than only from
/// the synthetic one below.
const HUNYUAN_DIR: &str = "models/hunyuan-a13b-instruct-4bit";

#[test]
fn hunyuan_tiktoken_checkpoint_still_loads_as_hunyuan() {
    let dir = std::path::Path::new(HUNYUAN_DIR);
    let vocab = dir.join("hy.tiktoken");
    if !vocab.exists() {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            "hunyuan_tiktoken_checkpoint_still_loads_as_hunyuan",
            "models/hunyuan-a13b-instruct-4bit is absent",
        );
        return;
    }
    // `model_type: hunyuan` is not a K3 spelling, so detection keeps this on
    // the pre-existing path.
    assert_eq!(TiktokenFamily::detect(dir), TiktokenFamily::HunYuan);
    let tok = TiktokenTokenizer::from_file(&vocab, dir).expect("load hy.tiktoken");
    assert_eq!(tok.family(), TiktokenFamily::HunYuan);
    assert!(tok.kimi_k3_control_ids().is_none());
    assert!(tok.control_id("<|eos|>").is_some());

    for text in [
        "Hello, world!",
        "你好，世界",
        "HelloWorld 12345 it's",
        "안녕하세요",
    ] {
        let ids = tok.encode(text, false).expect("encode");
        assert!(!ids.is_empty());
        assert_eq!(tok.decode(&ids, false).expect("decode"), text);
        // The two encode entry points agree on text with no special spelling
        // in it, which is what the K3 refactor rerouted through one function.
        assert_eq!(tok.encode_text(text).expect("encode_text"), ids);
    }
    // A special spelling is still recognized by `encode` and still not by
    // `encode_text`.
    assert_eq!(tok.encode("<|eos|>", false).expect("encode").len(), 1);
    assert!(tok.encode_text("<|eos|>").expect("encode_text").len() > 1);
}

#[test]
fn hunyuan_family_unchanged_without_a_model_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vocab = write_synthetic_tiktoken(dir.path());
    // No config.json at all: detection must land on HunYuan.
    assert_eq!(TiktokenFamily::detect(dir.path()), TiktokenFamily::HunYuan);

    let tok = TiktokenTokenizer::from_file(&vocab, dir.path()).expect("load");
    assert_eq!(tok.family(), TiktokenFamily::HunYuan);
    assert_eq!(tok.control_id("<|eos|>"), Some(260 + 3));
    assert_eq!(tok.control_id("<|extra_204|>"), Some(260 + 209));
    assert!(tok.kimi_k3_control_ids().is_none());
    // The HunYuan pattern splits every letter run as one piece, so `abc`
    // BPE-merges into the single ranked token rather than three bytes.
    assert_eq!(tok.encode_text("abc").expect("encode"), vec![257]);
    assert_eq!(tok.decode(&[257], false).expect("decode"), "abc");
}

#[test]
fn kimi_linear_needs_a_tiktoken_model_to_be_the_k3_family() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"model_type": "kimi_linear"}"#,
    )
    .expect("write config");
    // A `kimi_linear` directory with no `tiktoken.model` is not the K3
    // tiktoken family (it would have been converted with a tokenizer.json).
    assert_eq!(TiktokenFamily::detect(dir.path()), TiktokenFamily::HunYuan);

    write_synthetic_tiktoken(dir.path());
    assert_eq!(TiktokenFamily::detect(dir.path()), TiktokenFamily::KimiK3);
}

#[test]
fn chunk_guard_splits_long_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"model_type": "kimi_k3"}"#,
    )
    .expect("write config");
    let vocab = write_synthetic_tiktoken(dir.path());
    let tok = TiktokenTokenizer::from_file(&vocab, dir.path()).expect("load");
    assert_eq!(tok.family(), TiktokenFamily::KimiK3);

    // 30000 consecutive non-whitespace characters exceed the 25000-character
    // run bound, so the guard splits at 25000 and encoding equals the
    // concatenation of the two pieces.
    let long: String = "x".repeat(30_000);
    let ids = tok.encode_text(&long).expect("encode");
    let mut expected = tok.encode_text(&long[..25_000]).expect("encode head");
    expected.extend(tok.encode_text(&long[25_000..]).expect("encode tail"));
    assert_eq!(ids, expected);
    assert_eq!(tok.decode(&ids, false).expect("decode"), long);

    // A whitespace run is bounded the same way.
    let spaces: String = " ".repeat(30_000);
    let ids = tok.encode_text(&spaces).expect("encode");
    assert_eq!(tok.decode(&ids, false).expect("decode"), spaces);
}

#[test]
fn split_whitespaces_or_nonwhitespaces_matches_the_reference_shape() {
    assert_eq!(split_whitespaces_or_nonwhitespaces("", 4), vec![""]);
    assert_eq!(split_whitespaces_or_nonwhitespaces("abc", 4), vec!["abc"]);
    // A run longer than the bound splits; the class flip resets the counter.
    assert_eq!(
        split_whitespaces_or_nonwhitespaces("aaaaaa", 2),
        vec!["aa", "aa", "aa"]
    );
    assert_eq!(
        split_whitespaces_or_nonwhitespaces("aa  aa", 2),
        vec!["aa  aa"]
    );
}

#[test]
fn char_chunks_slices_by_character_not_byte() {
    assert!(char_chunks("", 3).is_empty());
    assert_eq!(char_chunks("abcdefg", 3), vec!["abc", "def", "g"]);
    // Three-byte characters: a byte-based split would land mid-character.
    assert_eq!(char_chunks("漢字仮名", 2), vec!["漢字", "仮名"]);
}

// ---------------------------------------------------------------------------
// Special-token splitting (#1743 security review)
// ---------------------------------------------------------------------------

/// Load a synthetic K3 vocabulary whose control block is named by an explicit
/// `added_tokens_decoder`, so a test can pin overlapping spellings.
fn synthetic_k3_with_named_controls(
    dir: &std::path::Path,
    named: &[(u32, &str)],
) -> TiktokenTokenizer {
    std::fs::write(dir.join("config.json"), br#"{"model_type": "kimi_k3"}"#).expect("write config");
    let entries: Vec<String> = named
        .iter()
        .map(|(id, content)| format!("\"{id}\": {{\"content\": \"{content}\"}}"))
        .collect();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        format!("{{\"added_tokens_decoder\": {{{}}}}}", entries.join(", ")),
    )
    .expect("write tokenizer_config");
    let vocab = write_synthetic_tiktoken(dir);
    TiktokenTokenizer::from_file(&vocab, dir).expect("load")
}

/// The special-token split keeps leftmost-longest semantics and never slices a
/// multi-byte character.
///
/// The scan walks bytes, so the guard that it only ever cuts on a character
/// boundary is worth pinning: a continuation byte can never equal the first
/// byte of a control token, which is what makes the byte walk safe.
#[test]
fn special_token_split_is_leftmost_longest_and_utf8_safe() {
    let dir = tempfile::tempdir().expect("tempdir");
    // 260 is a prefix of 261's spelling, so a shortest-first scan would split
    // `<|open|>x` into two segments instead of matching the longer control.
    let tok = synthetic_k3_with_named_controls(
        dir.path(),
        &[(260, "<|open|>"), (261, "<|open|>x"), (262, "<|sep|>")],
    );
    let open = tok.control_id("<|open|>").expect("open id");
    let open_x = tok.control_id("<|open|>x").expect("open_x id");
    let sep = tok.control_id("<|sep|>").expect("sep id");
    assert_eq!(open, 260);
    assert_eq!(open_x, 261);

    // Multi-byte text on both sides of a control token, and an unterminated
    // `<|` that must stay ordinary text rather than advance past a boundary.
    let text = "漢字<|open|>x🙂<|sep|>ｱ<|不完全";
    let ids = tok.encode(text, false).expect("encode");
    let want: Vec<u32> = tok
        .encode_text("漢字")
        .expect("han")
        .into_iter()
        .chain([open_x])
        .chain(tok.encode_text("🙂").expect("emoji"))
        .chain([sep])
        .chain(tok.encode_text("ｱ<|不完全").expect("tail"))
        .collect();
    assert_eq!(ids, want);
    assert_eq!(tok.decode(&ids, false).expect("decode"), text);

    // The bare prefix still matches on its own when nothing longer follows.
    let ids = tok.encode("<|open|>y", false).expect("encode");
    assert_eq!(ids[0], open);
}

/// Splitting on special tokens is linear in the input length.
///
/// Text that alternates ordinary characters with control-token spellings used
/// to cost one full scan of the remaining input per occurrence, for all 256
/// control tokens, which is quadratic. `POST /tokenize` defaults
/// `parse_special` to `true`, so that scan ran on arbitrary request text, and
/// it sits ahead of the bounded-chunk guard rather than behind it. At 72 KB the
/// old shape took 8.5 s and grew 4x per doubling; 1 MB of request body was
/// tens of minutes of one core.
///
/// The wall-clock bound is deliberately loose (this machine is shared): the
/// point is the two orders of magnitude between linear and quadratic here, not
/// a precise number.
#[test]
fn special_token_split_is_linear_on_interleaved_spellings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tok = synthetic_k3_with_named_controls(dir.path(), &[(260, "<|open|>")]);
    let open = tok.control_id("<|open|>").expect("open id");
    let unit_ids = tok.encode_text("a").expect("encode a");

    // ~400 KB alternating one ordinary character with one control spelling.
    let repeats = 45_000usize;
    let text = "a<|open|>".repeat(repeats);
    assert!(text.len() > 400_000);

    let started = std::time::Instant::now();
    let ids = tok.encode(&text, false).expect("encode");
    let elapsed = started.elapsed();

    let mut want = Vec::with_capacity(repeats * (unit_ids.len() + 1));
    for _ in 0..repeats {
        want.extend_from_slice(&unit_ids);
        want.push(open);
    }
    assert_eq!(ids, want, "interleaved control spellings must round-trip");
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "special-token split went superlinear: {elapsed:?} for {} bytes",
        text.len()
    );

    // The new scan's own worst case: every byte is a candidate first byte, so
    // every position tries the whole bucket and fails.
    let dense = "<".repeat(400_000);
    let started = std::time::Instant::now();
    let ids = tok.encode(&dense, false).expect("encode");
    let elapsed = started.elapsed();
    assert_eq!(ids, tok.encode_text(&dense).expect("encode_text"));
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "dense candidate-byte input went superlinear: {elapsed:?}"
    );
}

/// The bounded-chunk guard is reached through the special-parsing entry point,
/// not only through `encode_text`.
///
/// `encode` splits on control spellings first and encodes each segment, so the
/// guard has to apply per segment. A pathological run routed through `encode`
/// must give exactly the ids the reference gives, which for a run past the
/// 25 000-character bound is the concatenation of the bounded pieces.
#[test]
fn encode_reaches_the_chunk_guard_through_special_parsing() {
    let Some(tok) = k3_tokenizer("encode_reaches_the_chunk_guard_through_special_parsing") else {
        return;
    };
    let open = tok.control_id("<|open|>").expect("open id");

    // 30 000 non-whitespace characters exceed the 25 000 bound, wrapped in
    // control tokens so the input takes the special-splitting path.
    let run = "x".repeat(30_000);
    let text = format!("<|open|>{run}<|open|>");

    let started = std::time::Instant::now();
    let ids = tok.encode(&text, false).expect("encode");
    let elapsed = started.elapsed();

    let mut want = vec![open];
    want.extend(tok.encode_text(&run[..25_000]).expect("head"));
    want.extend(tok.encode_text(&run[25_000..]).expect("tail"));
    want.push(open);
    assert_eq!(
        ids, want,
        "the guard must bound the run inside `encode` too"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "guarded encode took {elapsed:?}"
    );

    // Text with no control spelling must agree between the two entry points.
    assert_eq!(
        tok.encode(&run, false).expect("encode"),
        tok.encode_text(&run).expect("encode_text")
    );
}
