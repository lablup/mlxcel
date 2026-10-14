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

//! Issue #1683: an added token declared only in `tokenizer_config.json` must
//! encode to its declared id, as it does under transformers.

use serde_json::json;

use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// A character-level BPE `tokenizer.json`: seven base pieces (ids 0..=6) plus
/// one added special token `<|eot|>` at id 7, mirroring the Granite Vision
/// layout where `tokenizer.json`'s `added_tokens` stop one short of `<image>`.
fn tokenizer_json() -> serde_json::Value {
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 7, "content": "<|eot|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
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
            "vocab": {"<": 0, ">": 1, "i": 2, "m": 3, "a": 4, "g": 5, "e": 6},
            "merges": []
        }
    })
}

fn decoder_entry(content: &str) -> serde_json::Value {
    json!({"content": content, "lstrip": false, "normalized": false, "rstrip": false,
           "single_word": false, "special": true})
}

fn checkpoint(decoder: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer_json()).expect("serialize tokenizer.json"),
    )
    .expect("write tokenizer.json");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"added_tokens_decoder": decoder}))
            .expect("serialize tokenizer_config.json"),
    )
    .expect("write tokenizer_config.json");
    dir
}

fn encode(tokenizer: &MlxcelTokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).expect("encode")
}

#[test]
fn config_only_added_token_encodes_to_its_declared_id() {
    let dir = checkpoint(json!({
        "7": decoder_entry("<|eot|>"),
        "8": decoder_entry("<image>"),
    }));
    let tokenizer = load_tokenizer(dir.path()).expect("load tokenizer");

    // Before the fix this was [0, 2, 3, 4, 5, 6, 1]: the placeholder split into
    // characters, so a VLM never saw its image token.
    assert_eq!(encode(&tokenizer, "<image>"), vec![8]);
    assert_eq!(encode(&tokenizer, "a<image>e"), vec![4, 8, 6]);
    assert_eq!(tokenizer.decode(&[8], false).expect("decode"), "<image>");
}

#[test]
fn declared_id_the_tokenizer_cannot_assign_is_left_unregistered() {
    // Id 9 skips 8: registering it would land at 8 and shift the vocabulary,
    // so the loader keeps the checkpoint's own split instead.
    let dir = checkpoint(json!({
        "7": decoder_entry("<|eot|>"),
        "9": decoder_entry("<image>"),
    }));
    let tokenizer = load_tokenizer(dir.path()).expect("load tokenizer");

    assert_eq!(encode(&tokenizer, "<image>"), vec![0, 2, 3, 4, 5, 6, 1]);
}

#[test]
fn entries_tokenizer_json_already_resolves_are_left_alone() {
    // `<|eot|>` is already id 7 and `>` is already a base piece at id 1; the
    // config entry naming `>` at id 8 must not re-register it.
    let dir = checkpoint(json!({
        "7": decoder_entry("<|eot|>"),
        "8": decoder_entry(">"),
    }));
    let tokenizer = load_tokenizer(dir.path()).expect("load tokenizer");

    assert_eq!(encode(&tokenizer, "<|eot|>"), vec![7]);
    assert_eq!(encode(&tokenizer, ">"), vec![1]);
}

#[test]
fn mixed_special_runs_land_on_their_declared_ids() {
    // Registration is batched by runs of the `special` flag; a special, a plain,
    // then a special entry must still take 8, 9, 10 in declared order.
    let mut plain = decoder_entry("<video>");
    plain["special"] = json!(false);
    let dir = checkpoint(json!({
        "7": decoder_entry("<|eot|>"),
        "8": decoder_entry("<image>"),
        "9": plain,
        "10": decoder_entry("<audio>"),
    }));
    let tokenizer = load_tokenizer(dir.path()).expect("load tokenizer");

    assert_eq!(encode(&tokenizer, "<image>"), vec![8]);
    assert_eq!(encode(&tokenizer, "<video>"), vec![9]);
    assert_eq!(encode(&tokenizer, "<audio>"), vec![10]);
    // Only the plain entry survives a skip-special decode.
    assert_eq!(
        tokenizer.decode(&[8, 9, 10], true).expect("decode"),
        "<video>"
    );
}
