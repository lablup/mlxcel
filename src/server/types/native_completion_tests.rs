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

//! Golden-schema tests for the native completion response types (#1441).
//!
//! The expected key sets and orders come from a capture of the pinned b10621
//! binary; see the module docs on `native_completion.rs`.

use super::{
    NativeCompletionChunk, NativeCompletionResponse, NativeTimings, PromptProgress, StopType,
};

fn sample() -> NativeCompletionResponse {
    NativeCompletionResponse {
        index: 0,
        content: " Paris.".to_string(),
        tokens: Vec::new(),
        id_slot: 3,
        stop: true,
        model: "test-model".to_string(),
        tokens_predicted: 8,
        tokens_evaluated: 5,
        generation_settings: serde_json::json!({"n_predict": 8}),
        prompt: "The capital of France is".to_string(),
        has_new_line: false,
        truncated: false,
        stop_type: StopType::Limit,
        stopping_word: String::new(),
        tokens_cached: 12,
        timings: NativeTimings::new(0, 5, 61.0, 8, 42.0),
    }
}

/// The exact top-level key order the pinned binary emits.
const B10621_KEYS: [&str; 16] = [
    "index",
    "content",
    "tokens",
    "id_slot",
    "stop",
    "model",
    "tokens_predicted",
    "tokens_evaluated",
    "generation_settings",
    "prompt",
    "has_new_line",
    "truncated",
    "stop_type",
    "stopping_word",
    "tokens_cached",
    "timings",
];

fn keys(value: &serde_json::Value) -> Vec<String> {
    value.as_object().expect("object").keys().cloned().collect()
}

#[test]
fn the_response_carries_exactly_the_b10621_key_set_in_order() {
    let json = serde_json::to_value(sample()).expect("serializes");
    assert_eq!(keys(&json), B10621_KEYS, "key set or order drifted");
}

#[test]
fn the_timings_block_leads_with_cache_n() {
    let json = serde_json::to_value(sample()).expect("serializes");
    assert_eq!(
        keys(&json["timings"]),
        [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ]
    );
}

#[test]
fn timings_rates_are_derived_the_way_upstream_derives_them() {
    let t = NativeTimings::new(4, 10, 100.0, 20, 50.0);
    assert_eq!(t.prompt_per_token_ms, 10.0);
    assert_eq!(t.prompt_per_second, 100.0);
    assert_eq!(t.predicted_per_token_ms, 2.5);
    assert_eq!(t.predicted_per_second, 400.0);
}

#[test]
fn zero_counts_produce_zero_rates_rather_than_nan() {
    // A NaN or an infinity is not representable in JSON and would serialize
    // as null, so the divisor guards matter for the wire shape, not just for
    // arithmetic taste.
    let t = NativeTimings::new(0, 0, 0.0, 0, 0.0);
    let json = serde_json::to_value(&t).expect("serializes");
    for key in [
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_per_token_ms",
        "predicted_per_second",
    ] {
        assert!(
            json[key].is_number(),
            "{key} must be a number, got {}",
            json[key]
        );
        assert_eq!(json[key].as_f64(), Some(0.0));
    }
}

#[test]
fn stop_type_serializes_as_the_upstream_lowercase_strings() {
    for (variant, expected) in [
        (StopType::None, "none"),
        (StopType::Limit, "limit"),
        (StopType::Word, "word"),
        (StopType::Eos, "eos"),
    ] {
        assert_eq!(
            serde_json::to_value(variant).expect("serializes"),
            serde_json::Value::String(expected.to_string())
        );
    }
}

#[test]
fn tokens_is_present_and_empty_when_not_requested() {
    // Upstream always emits the key; `return_tokens` only decides whether it
    // has content. A client reading `tokens` unconditionally must not break.
    let json = serde_json::to_value(sample()).expect("serializes");
    assert_eq!(json["tokens"], serde_json::json!([]));
}

#[test]
fn a_streaming_chunk_omits_the_optional_blocks_by_default() {
    let chunk = NativeCompletionChunk {
        index: 0,
        content: "Question".to_string(),
        tokens: vec![14582],
        stop: false,
        id_slot: -1,
        tokens_predicted: 1,
        tokens_evaluated: 1,
        timings: None,
        prompt_progress: None,
    };
    let json = serde_json::to_value(&chunk).expect("serializes");
    assert_eq!(
        keys(&json),
        [
            "index",
            "content",
            "tokens",
            "stop",
            "id_slot",
            "tokens_predicted",
            "tokens_evaluated",
        ]
    );
}

#[test]
fn a_streaming_chunk_carries_timings_and_progress_when_requested() {
    let chunk = NativeCompletionChunk {
        index: 0,
        content: String::new(),
        tokens: vec![0],
        stop: false,
        id_slot: -1,
        tokens_predicted: 0,
        tokens_evaluated: 1,
        timings: Some(NativeTimings::new(0, 1, 8.9, 0, 0.0)),
        prompt_progress: Some(PromptProgress {
            total: 1,
            cache: 0,
            processed: 1,
            time_ms: 8,
        }),
    };
    let json = serde_json::to_value(&chunk).expect("serializes");
    assert!(json["timings"].is_object());
    assert_eq!(
        keys(&json["prompt_progress"]),
        ["total", "cache", "processed", "time_ms"]
    );
}
