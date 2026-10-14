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

//! `MlxcelTokenizer::contains_id` membership tests (#2127).

use super::*;

/// Three vocabulary entries (ids 0..=2) plus the added token `<|tail|>` at id 10, so
/// `get_vocab_size(true)` is 4 while ids 3..=9 are unassigned.
fn gap_vocab_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 10, "content": "<|tail|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {"<unk>": 0, "a": 1, "b": 2, "<|tail|>": 10}}
    }"#;
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("gap vocab builds"),
    )
}

#[test]
fn huggingface_contains_id_is_membership_not_a_vocab_size_bound() {
    let tokenizer = gap_vocab_tokenizer();
    assert_eq!(tokenizer.vocab_size(), 4, "entry count, not max id + 1");

    // Below `vocab_size()` but unassigned: a `< vocab_size()` check accepts it.
    assert!(!tokenizer.contains_id(3));
    // Above `vocab_size()` but an added token: a `< vocab_size()` check refuses it.
    assert!(tokenizer.contains_id(10));

    for id in [0, 1, 2] {
        assert!(tokenizer.contains_id(id), "model id {id}");
    }
    for id in [4, 9, 11, u32::MAX] {
        assert!(!tokenizer.contains_id(id), "id {id}");
    }
}
