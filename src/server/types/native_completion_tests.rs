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

use mlxcel_core::drafter::DrafterKind;

use super::{
    NativeCompletionChunk, NativeCompletionResponse, NativeTimings, PromptProgress, StopType,
    select_response_fields,
};
use crate::server::model_provider::SpeculativeStats;

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
        completion_probabilities: None,
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
fn prompt_n_is_the_prompt_minus_what_the_cache_supplied() {
    // Captured from the pinned binary on a repeated prompt: `tokens_evaluated`
    // 8 with `cache_n` 7 reports `prompt_n` 1, not 8. The constructor takes
    // `tokens_evaluated` and does the subtraction so no call site can report
    // the whole prompt as the part that was processed.
    let t = NativeTimings::new(7, 8, 4.711, 5, 19.309);
    assert_eq!(t.cache_n, 7);
    assert_eq!(t.prompt_n, 1);
    // A cache figure larger than the prompt (which the wire shape allows but
    // no healthy request produces) floors at zero rather than underflowing.
    assert_eq!(NativeTimings::new(9, 8, 1.0, 1, 1.0).prompt_n, 0);
}

#[test]
fn timings_rates_are_derived_the_way_upstream_derives_them() {
    // Both formulas were read off the pinned binary rather than assumed. The
    // prompt rates divide by `prompt_n`; the generation rates divide by
    // `predicted_n - 1`, because `predicted_ms` starts at the first token.
    // Reference sample: prompt_n 17 / prompt_ms 40.387 reports
    // prompt_per_token_ms 2.3757 and prompt_per_second 420.93, while
    // predicted_n 4 / predicted_ms 15.132 reports predicted_per_token_ms
    // 5.044 (15.132 / 3) and predicted_per_second 198.255 (1e3 / 15.132 * 3).
    let t = NativeTimings::new(0, 17, 40.387, 4, 15.132);
    assert!((t.prompt_per_token_ms - 40.387 / 17.0).abs() < 1e-9);
    assert!((t.prompt_per_second - 1000.0 / 40.387 * 17.0).abs() < 1e-9);
    assert!((t.predicted_per_token_ms - 15.132 / 3.0).abs() < 1e-9);
    assert!((t.predicted_per_second - 1000.0 / 15.132 * 3.0).abs() < 1e-9);
}

#[test]
fn a_single_predicted_token_reports_zero_generation_rates() {
    // Upstream's first streaming frame carries `predicted_n: 1` with
    // `predicted_ms: 0.001` and reports both generation rates as 0.0: with one
    // token there is no interval between tokens to measure.
    let t = NativeTimings::new(0, 15, 44.1, 1, 0.001);
    assert_eq!(t.predicted_n, 1);
    assert_eq!(t.predicted_per_token_ms, 0.0);
    assert_eq!(t.predicted_per_second, 0.0);
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
        completion_probabilities: None,
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
        completion_probabilities: None,
    };
    let json = serde_json::to_value(&chunk).expect("serializes");
    assert!(json["timings"].is_object());
    assert_eq!(
        keys(&json["prompt_progress"]),
        ["total", "cache", "processed", "time_ms"]
    );
}

// ---------------------------------------------------------------------------
// `response_fields` projection (#1441)
//
// Every expectation below is a capture of the pinned b10621 binary answering
// the same request, not a reading of the schema's description, which claims
// the slash form "unnests" the value and is wrong about what the binary does.
// ---------------------------------------------------------------------------

fn projected(paths: &[&str]) -> serde_json::Value {
    let owned: Vec<String> = paths.iter().map(|p| (*p).to_string()).collect();
    select_response_fields(serde_json::to_value(sample()).expect("serializes"), &owned)
}

#[test]
fn an_empty_response_fields_list_returns_the_whole_object() {
    // Upstream treats an absent, null, empty, or wrongly typed value the same
    // way: the full object.
    assert_eq!(keys(&projected(&[])), B10621_KEYS);
}

#[test]
fn response_fields_keeps_only_the_named_root_keys() {
    let json = projected(&["content", "tokens_predicted"]);
    assert_eq!(keys(&json), ["content", "tokens_predicted"]);
    assert_eq!(json["content"], " Paris.");
    assert_eq!(json["tokens_predicted"], 8);
}

#[test]
fn a_slashed_path_keys_the_value_under_the_whole_path() {
    // The binary emits `{"generation_settings/n_predict": 8}`, keeping the
    // slash in the key rather than lifting `n_predict` to the root.
    let json = projected(&[
        "content",
        "generation_settings/n_predict",
        "timings/predicted_n",
    ]);
    assert_eq!(
        keys(&json),
        [
            "content",
            "generation_settings/n_predict",
            "timings/predicted_n"
        ]
    );
    assert_eq!(json["generation_settings/n_predict"], 8);
    assert_eq!(json["timings/predicted_n"], 8);
}

#[test]
fn a_missing_path_is_omitted_without_an_error() {
    let json = projected(&[
        "content",
        "no_such_field",
        "generation_settings/no_such_sub",
    ]);
    assert_eq!(keys(&json), ["content"]);
}

#[test]
fn a_path_that_walks_into_a_non_object_is_omitted() {
    // `content` is a string and `tokens` an array, so neither can be indexed
    // further; upstream answers `{}` for both.
    let json = projected(&["content/x", "tokens/0"]);
    assert_eq!(json, serde_json::json!({}));
}

#[test]
fn an_empty_path_selects_nothing() {
    // Measured: `"response_fields": [""]` answers `{}`, not the whole object.
    assert_eq!(projected(&[""]), serde_json::json!({}));
}

#[test]
fn the_projection_preserves_the_order_the_request_listed() {
    // Upstream builds the result in an insertion-ordered JSON object, so a
    // reversed request order comes back reversed rather than sorted.
    let json = projected(&["timings/predicted_n", "content", "index"]);
    assert_eq!(keys(&json), ["timings/predicted_n", "content", "index"]);
}

#[test]
fn a_repeated_path_yields_one_key() {
    let json = projected(&["content", "content"]);
    assert_eq!(keys(&json), ["content"]);
}

#[test]
fn a_nested_object_can_be_selected_whole() {
    let json = projected(&["timings"]);
    assert_eq!(keys(&json), ["timings"]);
    assert!(json["timings"]["cache_n"].is_number());
}

// ---------------------------------------------------------------------------
// Speculative acceptance on the timings block (#1314)
// ---------------------------------------------------------------------------

fn dflash_stats() -> SpeculativeStats {
    SpeculativeStats::from_counts(DrafterKind::Dflash, 9, 72, 55)
        .expect("nine rounds is speculative")
}

#[test]
fn timings_carries_no_draft_keys_without_a_drafter() {
    // The wire shape of a deployment that runs no drafter has to be exactly
    // what it was before #1314: absent keys, not zeros. A zero `draft_n` would
    // be indistinguishable from a drafter that proposed nothing, which is the
    // one distinction the block exists to make.
    let json =
        serde_json::to_value(NativeTimings::new(0, 17, 40.387, 4, 15.132)).expect("serializes");
    let object = json.as_object().expect("timings is an object");
    for key in ["draft_n", "draft_n_accepted", "draft_rounds", "draft_kind"] {
        assert!(!object.contains_key(key), "{key} must be absent: {json}");
    }
    // The nine keys the pinned binary reports are untouched.
    assert_eq!(object.len(), 9, "{json}");
}

#[test]
fn timings_carries_the_b10621_draft_pair_for_a_speculative_request() {
    // `draft_n` / `draft_n_accepted` are b10621's own optional pair and are
    // spelled exactly as upstream spells them, so a client that already reads
    // llama-server timings reads mlxcel's unchanged. `draft_rounds` and
    // `draft_kind` are the mlxcel extension: the first makes the mean accepted
    // length per round computable, the second names the drafter.
    let stats = dflash_stats();
    let json = serde_json::to_value(
        NativeTimings::new(0, 17, 40.387, 64, 800.0).with_speculative(Some(&stats)),
    )
    .expect("serializes");
    assert_eq!(json["draft_n"], 72);
    assert_eq!(json["draft_n_accepted"], 55);
    assert_eq!(json["draft_rounds"], 9);
    assert_eq!(json["draft_kind"], "dflash");
    // Flattened onto the block rather than nested under a key of their own,
    // which is where upstream puts its pair, and appended after the nine base
    // keys in their unchanged order, which is where upstream appends it.
    assert_eq!(
        keys(&json),
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
            "draft_n",
            "draft_n_accepted",
            "draft_rounds",
            "draft_kind",
        ]
    );
    // The nine base keys still report what they always did.
    assert_eq!(json["predicted_n"], 64);
    assert_eq!(json["prompt_n"], 17);
}

#[test]
fn a_zero_round_speculative_run_reports_no_draft_block() {
    // A request that finished inside prefill (immediate EOS, or `n_predict:
    // 1`) never gave the drafter a round to run. Reporting it with zeros would
    // say a drafter ran and accepted nothing.
    assert!(SpeculativeStats::from_counts(DrafterKind::Mtp, 0, 0, 0).is_none());
}

#[test]
fn every_drafter_kind_renders_its_canonical_name() {
    for (kind, name) in [
        (DrafterKind::Dflash, "dflash"),
        (DrafterKind::Mtp, "mtp"),
        (DrafterKind::InternalMtp, "internal-mtp"),
    ] {
        let stats = SpeculativeStats::from_counts(kind, 1, 4, 3).expect("one round is speculative");
        let json = serde_json::to_value(
            NativeTimings::new(0, 1, 1.0, 4, 1.0).with_speculative(Some(&stats)),
        )
        .expect("serializes");
        assert_eq!(json["draft_kind"], name);
    }
}
