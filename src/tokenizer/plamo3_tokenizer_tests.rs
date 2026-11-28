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

//! PLaMo 3 tokenizer (`Plamo3Tokenizer`, issue #2374): `CONTROL` rows as
//! special added tokens and the reference's boundary splits.

use super::*;

/// `(text, ids)` captured from `Plamo3Tokenizer.encode(text,
/// add_special_tokens=False)` on `pfnet/plamo-3-610m-fin-instruct`
/// (transformers 5.24.1, `trust_remote_code`).
const PLAMO3_REFERENCE_CASES: &[(&str, &[u32])] = &[
    (
        "The capital of France is Paris.",
        &[1100, 4146, 1088, 5343, 48097, 4944, 46],
    ),
    ("東京は日本の首都です。", &[50349, 68395, 60134, 50111]),
    ("<|plamo:msg|>", &[19]),
    ("<|plamo:tag|>", &[16]),
    (
        "user<|plamo:msg|>こんにちは<|plamo:tag|>assistant",
        &[1667, 19, 54847, 16, 39306, 1335],
    ),
    // Repeated-character and consecutive-space boundaries.
    (
        "aaaaaaab  spaces    here",
        &[45763, 98, 288, 21492, 525, 8908],
    ),
    // `.` does not match a newline, so four newlines are not a run.
    ("----------\n\n\n\nend", &[721, 521, 1254]),
    (
        "def foo(x):\n    return x+1",
        &[1396, 29010, 40, 120, 1222, 10, 525, 1111, 376, 43, 49],
    ),
    // An unknown `<|plamo:...|>` string still gets its own boundaries.
    (
        "<|plamo:unknown_thing|>x",
        &[60, 124, 112, 42029, 58, 17802, 95, 23536, 124, 62, 120],
    ),
    ("Hello world!!!!", &[7272, 1675, 10817]),
];

const PLAMO3_DIR: &str = "models/plamo-3-610m-fin-instruct";

#[test]
fn plamo3_tokenizer_matches_reference_encodings() {
    let model_dir = std::path::Path::new(PLAMO3_DIR);
    if !model_dir.exists() {
        eprintln!("skipping plamo3_tokenizer_matches_reference_encodings: {PLAMO3_DIR} is absent");
        return;
    }
    let tok = load_tokenizer(model_dir).expect("load PLaMo 3 tokenizer");
    for (text, expected) in PLAMO3_REFERENCE_CASES {
        let ids = tok.encode(text, false).expect("encode");
        assert_eq!(&ids, expected, "encode mismatch for {text:?}");
        assert_eq!(
            &tok.decode(&ids, false).unwrap(),
            text,
            "round trip for {text:?}"
        );
    }
}

#[test]
fn plamo3_control_tokens_are_special_and_dropped_on_skip_decode() {
    let model_dir = std::path::Path::new(PLAMO3_DIR);
    if !model_dir.exists() {
        eprintln!("skipping: {PLAMO3_DIR} is absent");
        return;
    }
    let tok = load_tokenizer(model_dir).expect("load PLaMo 3 tokenizer");
    assert_eq!(tok.encode("<|plamo:msg|>", false).unwrap(), vec![19]);
    let ids = tok.encode("<|plamo:msg|>hi<|plamo:tag|>", false).unwrap();
    assert_eq!(ids.first(), Some(&19));
    assert_eq!(ids.last(), Some(&16));
    assert_eq!(tok.decode(&ids, true).unwrap(), "hi");
}

/// A tiny `tokenizer.jsonl` with a `CONTROL` row past the four fixed
/// specials: it must become a special added token.
#[test]
fn control_rows_become_special_added_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let rows = [
        r#"["<|plamo:unk|>", 0.0, "UNKNOWN"]"#,
        r#"["<|plamo:bos|>", 0.0, "CONTROL"]"#,
        r#"["<|plamo:eos|>", 0.0, "CONTROL"]"#,
        r#"["<|plamo:pad|>", 0.0, "CONTROL"]"#,
        r#"["<|plamo:msg|>", 0.0, "CONTROL"]"#,
        r#"["a", -1.0, "NORMAL"]"#,
        r#"["b", -1.0, "NORMAL"]"#,
    ];
    std::fs::write(dir.path().join("tokenizer.jsonl"), rows.join("\n")).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
    let tok = load_tokenizer(dir.path()).unwrap();
    let ids = tok.encode("a<|plamo:msg|>b", false).unwrap();
    assert_eq!(ids, vec![5, 4, 6]);
    assert_eq!(tok.decode(&ids, true).unwrap(), "ab");
    assert_eq!(tok.decode(&ids, false).unwrap(), "a<|plamo:msg|>b");
}

/// The boundary splits are `Plamo3Tokenizer`'s alone: PLaMo 2 keeps no
/// pre-tokenizer, and PLaMo 3 reads its thresholds from the config.
#[test]
fn the_boundary_pre_tokenizer_is_keyed_on_plamo3_tokenizer() {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = dir.path().join("tokenizer_config.json");
    std::fs::write(&config, r#"{"tokenizer_class": "PlamoTokenizer"}"#).unwrap();
    assert!(plamo3_pre_tokenizer(dir.path()).is_null());
    std::fs::write(
        &config,
        r#"{"tokenizer_class": "Plamo3Tokenizer", "break_around_consecutive_spaces_threshold": 2, "break_around_repeated_chars_threshold": 4}"#,
    )
    .unwrap();
    let pre = plamo3_pre_tokenizer(dir.path());
    let steps = pre["pretokenizers"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[1]["pattern"]["Regex"], r"(.)\1{3,}| {2,}");
}
