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

//! `/tokenize` and `/detokenize` schema tests (#1442).
//!
//! Two vocabularies are exercised because their special-token and piece
//! behavior differ in exactly the ways the endpoints expose: a Llama-style
//! SentencePiece-shaped vocabulary with named special tokens, and a
//! Qwen-style byte-level BPE vocabulary whose tokens routinely carry part of a
//! multi-byte character.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::{MlxcelTokenizer, pieces::byte_to_alphabet_char};

/// A Llama-style vocabulary: named special tokens plus whole-word entries.
fn llama_style_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "unk_token": "<unk>",
            "vocab": {"<s>": 0, "</s>": 1, "Hello": 2, "World": 3, "<unk>": 4}
        }
    }"#;
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("stub tokenizer builds"),
    )
}

/// A Qwen-style byte-level BPE vocabulary: one entry per byte, so any input is
/// tokenized one byte at a time and a multi-byte character always straddles
/// token boundaries.
fn byte_level_tokenizer() -> MlxcelTokenizer {
    let entries: Vec<String> = (0u16..=255)
        .map(|byte| {
            let ch = byte_to_alphabet_char(byte as u8);
            format!("{}: {byte}", serde_json::Value::String(ch.to_string()))
        })
        .collect();
    let json = format!(
        r#"{{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {{"id": 256, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
        ],
        "normalizer": null,
        "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}},
        "post_processor": null,
        "decoder": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}},
        "model": {{
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {{{}, "<|im_start|>": 256}},
            "merges": []
        }}
    }}"#,
        entries.join(", ")
    );
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("byte-level stub builds"),
    )
}

fn app_with(tokenizer: MlxcelTokenizer) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        tokenizer,
        PathBuf::from("tokenize-route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn post(
    tokenizer: MlxcelTokenizer,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app_with(tokenizer)
        .oneshot(request)
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

#[tokio::test]
async fn the_default_shape_is_a_flat_id_array() {
    let (status, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "Hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], serde_json::json!([2]));
}

#[tokio::test]
async fn an_absent_content_is_an_empty_tokenization_not_an_error() {
    let (status, body) = post(llama_style_tokenizer(), "/tokenize", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], serde_json::json!([]));
}

#[tokio::test]
async fn empty_content_tokenizes_to_an_empty_list() {
    let (status, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": ""}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], serde_json::json!([]));
}

#[tokio::test]
async fn parse_special_defaults_to_true_and_can_be_turned_off() {
    // Default: the spelling in the text is the special token, one id.
    let (_, parsed) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "<s>Hello"}),
    )
    .await;
    assert_eq!(parsed["tokens"], serde_json::json!([0, 2]));

    // `parse_special: false`: the same text is ordinary characters, so the
    // special id must not appear.
    let (_, plain) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "<s>Hello", "parse_special": false}),
    )
    .await;
    let ids: Vec<i64> = serde_json::from_value(plain["tokens"].clone()).expect("id array");
    assert!(
        !ids.contains(&0),
        "parse_special:false must not produce the <s> id, got {ids:?}"
    );
}

#[tokio::test]
async fn with_pieces_answers_objects_carrying_the_id_and_the_piece() {
    let (status, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "Hello", "with_pieces": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["tokens"],
        serde_json::json!([{"id": 2, "piece": "Hello"}])
    );
}

#[tokio::test]
async fn a_special_token_piece_is_its_own_spelling() {
    let (_, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "<s>", "with_pieces": true}),
    )
    .await;
    assert_eq!(body["tokens"][0]["piece"], "<s>");
}

#[tokio::test]
async fn an_invalid_utf8_piece_comes_back_as_an_array_of_byte_values() {
    // "中" is E4 B8 AD. Every byte is its own token in this vocabulary, so no
    // piece is a whole character and all three must take the array form.
    let (status, body) = post(
        byte_level_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": "\u{4E2D}", "with_pieces": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let tokens = body["tokens"].as_array().expect("token array").clone();
    assert_eq!(tokens.len(), 3, "{body}");
    assert_eq!(tokens[0]["piece"], serde_json::json!([0xE4]));
    assert_eq!(tokens[1]["piece"], serde_json::json!([0xB8]));
    assert_eq!(tokens[2]["piece"], serde_json::json!([0xAD]));
}

#[tokio::test]
async fn the_pieces_of_a_tokenization_concatenate_back_to_the_input() {
    // The reassembly property is what `with_pieces` exists for: a client that
    // concatenates every piece's bytes must recover the original text, whether
    // an individual piece was a string or a byte array.
    for content in ["Hello World", "\u{4E2D}\u{6587}", "caf\u{e9} \u{1F600}", ""] {
        let (_, body) = post(
            byte_level_tokenizer(),
            "/tokenize",
            serde_json::json!({"content": content, "with_pieces": true}),
        )
        .await;
        let mut bytes = Vec::new();
        for token in body["tokens"].as_array().expect("token array") {
            match &token["piece"] {
                serde_json::Value::String(text) => bytes.extend_from_slice(text.as_bytes()),
                serde_json::Value::Array(values) => {
                    bytes.extend(values.iter().map(|v| v.as_u64().expect("byte value") as u8))
                }
                other => panic!("unexpected piece shape {other}"),
            }
        }
        assert_eq!(
            String::from_utf8(bytes).expect("pieces reassemble to UTF-8"),
            content
        );
    }
}

#[tokio::test]
async fn a_mixed_content_array_splices_pre_tokenized_ids() {
    let (status, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": ["Hello", 1, "World"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], serde_json::json!([2, 1, 3]));
}

#[tokio::test]
async fn a_content_of_the_wrong_type_is_a_400() {
    let (status, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": 7}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn detokenize_round_trips_ascii_unicode_and_special_tokens() {
    for content in ["Hello World", "\u{4E2D}\u{6587}", "caf\u{e9} \u{1F600}"] {
        let (_, tokenized) = post(
            byte_level_tokenizer(),
            "/tokenize",
            serde_json::json!({"content": content}),
        )
        .await;
        let (status, detokenized) = post(
            byte_level_tokenizer(),
            "/detokenize",
            serde_json::json!({"tokens": tokenized["tokens"]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{detokenized}");
        assert_eq!(detokenized["content"], content);
    }
}

#[tokio::test]
async fn detokenize_renders_special_tokens_rather_than_skipping_them() {
    let (_, body) = post(
        byte_level_tokenizer(),
        "/detokenize",
        serde_json::json!({"tokens": [256]}),
    )
    .await;
    assert_eq!(body["content"], "<|im_start|>");
}

#[tokio::test]
async fn an_absent_or_empty_token_list_detokenizes_to_the_empty_string() {
    for body in [serde_json::json!({}), serde_json::json!({"tokens": []})] {
        let (status, response) = post(byte_level_tokenizer(), "/detokenize", body).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(response["content"], "");
    }
}

#[tokio::test]
async fn a_spliced_id_consumes_the_add_special_position() {
    // Upstream's `tokenize_mixed` sets its `first` flag on ANY element, not
    // only on a string, so `[id, "text"]` does not get a BOS after the id.
    let (_, body) = post(
        llama_style_tokenizer(),
        "/tokenize",
        serde_json::json!({"content": [0, "Hello"], "add_special": true}),
    )
    .await;
    assert_eq!(body["tokens"], serde_json::json!([0, 2]));
}
