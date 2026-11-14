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

use std::fs;

use super::{
    GenerationConfigDefaults, merged_eos_token_ids, read_eos_token_ids,
    read_generation_config_defaults,
};

fn write_generation_config(contents: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("generation_config.json"), contents).expect("write config");
    dir
}

#[test]
fn muse_glimmer_generation_defaults_match_pinned_fixture() {
    let dir = write_generation_config(include_str!(
        "../../tests/fixtures/muse_glimmer/generation_config.json"
    ));

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(
        defaults,
        GenerationConfigDefaults {
            eos_token_ids: vec![200001, 200008],
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(64),
        }
    );
    assert_eq!(read_eos_token_ids(dir.path()), vec![200001, 200008]);
}

#[test]
fn generation_defaults_accept_single_eos_and_partial_sampling() {
    let dir = write_generation_config(
        r#"{
            "eos_token_id": 42,
            "temperature": 0.7,
            "top_k": 32
        }"#,
    );

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(defaults.eos_token_ids, vec![42]);
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.top_p, None);
    assert_eq!(defaults.top_k, Some(32));
}

#[test]
fn generation_defaults_ignore_malformed_or_out_of_range_fields() {
    let dir = write_generation_config(
        r#"{
            "eos_token_id": ["bad", 7],
            "temperature": "hot",
            "top_p": null,
            "top_k": 2147483648
        }"#,
    );

    let defaults = read_generation_config_defaults(dir.path());

    assert_eq!(defaults.eos_token_ids, vec![7]);
    assert_eq!(defaults.temperature, None);
    assert_eq!(defaults.top_p, None);
    assert_eq!(defaults.top_k, None);
}

#[test]
fn missing_generation_config_returns_empty_defaults() {
    let dir = tempfile::tempdir().expect("tempdir");

    assert_eq!(
        read_generation_config_defaults(dir.path()),
        GenerationConfigDefaults::default()
    );
}

/// generation_config `[2]`, tokenizer_config `eos_token: "<|im_end|>"` (id 7)
/// and config `[2, 9]` yield the de-duplicated union `[2, 7, 9]` in priority
/// order, not the first non-empty source (#2384, ml-explore/mlx-lm#1414,
/// Blaizzy/mlx-vlm#2112).
#[test]
fn eos_token_ids_are_the_union_of_every_sidecar() {
    let dir = write_generation_config(r#"{ "eos_token_id": [2] }"#);
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {
                "2": { "content": "<|endoftext|>" },
                "7": { "content": "<|im_end|>" }
            }
        }"#,
    )
    .expect("write tokenizer config");
    fs::write(
        dir.path().join("config.json"),
        r#"{ "eos_token_id": [2, 9] }"#,
    )
    .expect("write config");

    assert_eq!(read_eos_token_ids(dir.path()), vec![2, 7, 9]);
}

/// The layout of the Qwen3.5 conversions on the Hub: `tokenizer_config.json`
/// names `eos_token` and has no `added_tokens_decoder`, the added tokens are in
/// `tokenizer.json`, and `config.json` declares only `<|endoftext|>`, under
/// `text_config`, which this reader does not look at (the model reads it and
/// contributes that id itself). The end-of-turn token has to be a stop id; it
/// used to be printed instead.
#[test]
fn an_eos_token_without_a_decoder_table_is_resolved_through_tokenizer_json() {
    let tokenizer_json = r#"{
        "version": "1.0",
        "added_tokens": [
            { "id": 248044, "content": "<|endoftext|>", "special": true },
            { "id": 248045, "content": "<|im_start|>", "special": true },
            { "id": 248046, "content": "<|im_end|>", "special": true }
        ],
        "model": { "type": "BPE", "vocab": { "a": 0, "b": 1 }, "merges": [] }
    }"#;
    for eos_token in [
        r#""<|im_end|>""#,
        r#"{ "content": "<|im_end|>", "lstrip": false }"#,
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("tokenizer_config.json"),
            format!(r#"{{ "backend": "tokenizers", "eos_token": {eos_token}, "pad_token": "<|endoftext|>" }}"#),
        )
        .expect("write tokenizer config");
        fs::write(dir.path().join("tokenizer.json"), tokenizer_json).expect("write tokenizer");
        fs::write(
            dir.path().join("config.json"),
            r#"{ "model_type": "qwen3_5", "text_config": { "eos_token_id": 248044 } }"#,
        )
        .expect("write config");

        assert_eq!(read_eos_token_ids(dir.path()), vec![248046], "{eos_token}");
        // The model's own id first, then the tokenizer's end-of-turn token.
        assert_eq!(
            merged_eos_token_ids(vec![248044], dir.path()),
            vec![248044, 248046]
        );
    }
}

/// A decoder table that lists the token still decides, and a tokenizer file
/// that is missing, unreadable or does not list the token adds nothing.
#[test]
fn tokenizer_json_is_only_a_fallback_for_the_eos_token() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": { "7": { "content": "<|im_end|>" } }
        }"#,
    )
    .expect("write tokenizer config");
    fs::write(
        dir.path().join("tokenizer.json"),
        r#"{ "added_tokens": [ { "id": 99, "content": "<|im_end|>" } ] }"#,
    )
    .expect("write tokenizer");
    assert_eq!(read_eos_token_ids(dir.path()), vec![7]);

    for tokenizer_json in [
        None,
        Some("not json"),
        Some(r#"{ "added_tokens": [ { "id": 5, "content": "<|other|>" } ] }"#),
        Some(r#"{ "model": { "type": "BPE" } }"#),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{ "eos_token": "<|im_end|>" }"#,
        )
        .expect("write tokenizer config");
        if let Some(contents) = tokenizer_json {
            fs::write(dir.path().join("tokenizer.json"), contents).expect("write tokenizer");
        }
        assert_eq!(
            read_eos_token_ids(dir.path()),
            Vec::<i32>::new(),
            "{tokenizer_json:?}"
        );
    }
}

/// A decoder table that does not list the token leaves the lookup to
/// `tokenizer.json`, where entries of an unexpected shape and negative ids
/// are passed over.
#[test]
fn tokenizer_json_resolves_an_eos_token_the_decoder_table_omits() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": { "5": { "content": "<|other|>" } }
        }"#,
    )
    .expect("write tokenizer config");
    fs::write(
        dir.path().join("tokenizer.json"),
        r#"{ "added_tokens": [
            { "content": "<|im_end|>" },
            { "id": 3 },
            { "id": -1, "content": "<|im_end|>" },
            { "id": 5, "content": "<|other|>" },
            { "id": 9, "content": "<|im_end|>", "special": true }
        ] }"#,
    )
    .expect("write tokenizer");
    assert_eq!(read_eos_token_ids(dir.path()), vec![9]);
}

/// The fallback reads `added_tokens` only. An `eos_token` that is an ordinary
/// vocabulary entry (here `.`, which a model emits mid-sentence) must not
/// become a stop id through the base vocabulary.
#[test]
fn tokenizer_json_fallback_ignores_the_base_vocabulary() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{ "eos_token": "." }"#,
    )
    .expect("write tokenizer config");
    fs::write(
        dir.path().join("tokenizer.json"),
        r#"{
            "added_tokens": [ { "id": 100, "content": "<|im_end|>", "special": true } ],
            "model": { "type": "BPE", "vocab": { ".": 13, "a": 0 }, "merges": [] }
        }"#,
    )
    .expect("write tokenizer");
    assert_eq!(read_eos_token_ids(dir.path()), Vec::<i32>::new());
}

/// #2729: the declared ids keep their order and the sidecar ids not already
/// declared follow them.
#[test]
fn merged_eos_token_ids_appends_sidecar_ids() {
    let dir = write_generation_config(r#"{"eos_token_id": [7, 1]}"#);
    assert_eq!(merged_eos_token_ids(vec![1], dir.path()), vec![1, 7]);
    assert_eq!(merged_eos_token_ids(Vec::new(), dir.path()), vec![7, 1]);
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(merged_eos_token_ids(vec![3], empty.path()), vec![3]);
}
