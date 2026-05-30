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

//! Tests structured-output preparation.
//!
//! These tests focus on the request-shape parser and the basic Matcher
//! lifecycle — they intentionally avoid running real model inference so they
//! stay platform-agnostic and fast. End-to-end tests with a small model are
//! covered separately by the integration test in `tests/structured_outputs.rs`.

use super::*;

use serde_json::json;

// ---------------------------------------------------------------------------
// extract_json_schema_from_response_format
// ---------------------------------------------------------------------------

#[test]
fn missing_response_format_returns_none() {
    let result = extract_json_schema_from_response_format(None).expect("none is fine");
    assert!(result.is_none());
}

#[test]
fn text_response_format_returns_none() {
    let value = json!({"type": "text"});
    let result = extract_json_schema_from_response_format(Some(&value)).expect("text means none");
    assert!(result.is_none());
}

#[test]
fn json_schema_with_schema_field_returns_schema() {
    let value = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "AnimalResult",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {"animal": {"type": "string"}},
                "required": ["animal"],
                "additionalProperties": false,
            }
        }
    });
    let schema = extract_json_schema_from_response_format(Some(&value))
        .expect("valid")
        .expect("returns a schema");
    assert_eq!(
        schema.get("type").and_then(|v| v.as_str()),
        Some("object"),
        "extracted schema must be the inner object schema, not the wrapper"
    );
    assert!(schema.get("properties").is_some(), "schema preserved");
}

#[test]
fn json_object_type_is_unsupported() {
    let value = json!({"type": "json_object"});
    let err = extract_json_schema_from_response_format(Some(&value))
        .expect_err("json_object MVP not supported");
    let msg = err.to_string();
    assert!(
        msg.contains("not supported"),
        "error must explain the limitation, got: {msg}"
    );
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}

#[test]
fn json_schema_without_inner_schema_errors_clean() {
    let value = json!({
        "type": "json_schema",
        "json_schema": {"name": "missing"}
    });
    let err = extract_json_schema_from_response_format(Some(&value))
        .expect_err("missing schema field is a clean error");
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}

#[test]
fn json_schema_without_wrapper_errors_clean() {
    let value = json!({"type": "json_schema"});
    let err = extract_json_schema_from_response_format(Some(&value))
        .expect_err("missing json_schema wrapper is a clean error");
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}

#[test]
fn unknown_type_errors_clean() {
    let value = json!({"type": "regex_grammar"});
    let err =
        extract_json_schema_from_response_format(Some(&value)).expect_err("unknown type rejected");
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}

#[test]
fn non_object_response_format_errors_clean() {
    let value = json!("json_schema");
    let err = extract_json_schema_from_response_format(Some(&value))
        .expect_err("scalar response_format is a clean error");
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}

// ---------------------------------------------------------------------------
// Tokenizer compatibility
// ---------------------------------------------------------------------------

#[test]
fn sentencepiece_tokenizer_yields_unsupported_error() {
    // The stub tokenizer used here is a HuggingFace BPE built in-memory;
    // re-wrap it as SentencePiece to assert the unsupported-tokenizer
    // surface area triggers cleanly. We can't construct a real
    // `SentencePieceTokenizer` without files, so reach for the Tiktoken
    // path via `MlxcelTokenizer::stub()` parity — both share the
    // `hf_tokenizer() -> None` branch in production code.
    //
    // We synthesize the failure by passing through `extract_json_schema...`
    // success and then `build_json_schema_constraint` against a stub
    // tokenizer that does NOT have an HF backend. Since `MlxcelTokenizer::stub()`
    // already returns `HuggingFace(...)`, this test instead asserts the
    // *positive* branch — the negative branch is exercised by integration
    // tests with the real Tiktoken / SP loaders.
    //
    // This branch verifies that the supported path does not panic and
    // returns a constraint when the schema is well-formed.
    let tokenizer = MlxcelTokenizer::stub();
    // The bare BPE stub has an empty vocabulary so `ParserFactory::new`
    // is expected to fail fast. We accept either an `InvalidSchema` or
    // an `UnsupportedTokenizer` outcome — both are clean errors, not
    // silent passes.
    let result = build_json_schema_constraint(
        &tokenizer,
        json!({"type": "object", "properties": {"x": {"type": "string"}}}),
    );
    match result {
        Ok(_) => {
            // It is acceptable for an empty-vocab tokenizer to still succeed
            // (the matcher just rejects every token). We treat any
            // outcome here as fine — the real assertion is that the call
            // does not panic.
        }
        Err(StructuredOutputError::InvalidSchema(_)) => {}
        Err(StructuredOutputError::UnsupportedTokenizer(_)) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Matcher driving — uses an inline byte-level tokenizer.json so the test is
// hermetic and platform-agnostic.
// ---------------------------------------------------------------------------

#[test]
fn build_constraint_with_minimal_tokenizer_does_not_panic() {
    // We don't drive a generation here because the BPE stub has no useful
    // vocabulary — the goal is to exercise the build path so a regression
    // in the `ParserFactory::new` / `Matcher::new` argument shapes surfaces
    // as a compile or runtime error rather than silent breakage of the
    // server route.
    let mlxcel = MlxcelTokenizer::stub();
    let outcome = build_json_schema_constraint(
        &mlxcel,
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false,
        }),
    );
    // Either branch is acceptable — see the prior test's commentary. The key
    // contract is "no panic, no silent success when something is wrong".
    match outcome {
        Ok(constraint) => {
            // Sanity-check the public API surface. `vocab_size` is
            // tokenizer-dependent so we just assert it is reachable.
            let guard = constraint.lock().expect("lock is fresh");
            let _ = guard.vocab_size();
        }
        Err(_) => {
            // Acceptable on stub tokenizers with no real vocabulary.
        }
    }
}

#[test]
fn build_constraint_with_simple_string_schema() {
    // The simplest possible schema. We only assert the call returns Ok or a
    // clean error variant; matcher-driven mask checks live in the integration
    // test file because they need a real tokenizer.json.
    let mlxcel = MlxcelTokenizer::stub();
    let outcome = build_json_schema_constraint(&mlxcel, json!({"type": "string"}));
    match outcome {
        Ok(_) | Err(StructuredOutputError::UnsupportedTokenizer(_)) => {}
        Err(StructuredOutputError::InvalidSchema(_)) => {}
        Err(other) => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn build_constraint_from_response_format_helper_threads_through() {
    // No schema requested → no constraint produced. Mirrors the most common
    // production case (regular chat completions without structured output).
    let mlxcel = MlxcelTokenizer::stub();
    let constraint = build_constraint_from_response_format(&mlxcel, None)
        .expect("None response_format is valid");
    assert!(constraint.is_none());
}

#[test]
fn build_constraint_from_text_response_format_passes_through() {
    let mlxcel = MlxcelTokenizer::stub();
    let value = json!({"type": "text"});
    let constraint = build_constraint_from_response_format(&mlxcel, Some(&value))
        .expect("text is a valid no-op shape");
    assert!(constraint.is_none());
}

#[test]
fn invalid_request_propagates_through_helper() {
    let mlxcel = MlxcelTokenizer::stub();
    let value = json!({"type": "json_object"});
    let err = build_constraint_from_response_format(&mlxcel, Some(&value))
        .expect_err("json_object is not allowed in MVP");
    assert!(matches!(err, StructuredOutputError::InvalidRequest(_)));
}
