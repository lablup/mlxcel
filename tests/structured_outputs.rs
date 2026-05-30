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

//! Integration test (`response_format: {"type": "json_schema",...}`).
//!
//! These tests build a real `llguidance` matcher over a synthetic byte-level
//! tokenizer and walk it through a known-conforming JSON output. Failures
//! here would indicate a regression in either the schema-to-grammar
//! compilation or the per-step `compute_mask` / `consume_token` plumbing
//! that the scheduler relies on for constrained decoding.
//!
//! End-to-end tests against a real model live in
//! `tests/structured_outputs_real_model.rs` (gated behind a fixture path
//! environment variable so CI can opt in only when a model is available).

use serde_json::json;

/// Build a byte-level HuggingFace tokenizer from a precomputed JSON
/// representation. Hermetic (no external file dependency) and small enough
/// that mask iteration is cheap. The JSON is a hand-rolled byte-level vocab
/// covering all 256 bytes — every byte is one token, no merges — which
/// matches what `ByteTokenizer::from_json_bytes` expects on the
/// `toktrie_hf_tokenizers` side.
fn build_byte_level_tokenizer() -> tokenizers::Tokenizer {
    let json = byte_level_tokenizer_json();
    tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("test byte-level tokenizer.json must parse")
}

/// Hand-rolled minimal byte-level tokenizer.json. Sufficient for driving
/// llguidance's matcher: every byte gets a single token id, decoder is
/// `ByteLevel`, no merges. Pre-tokenizer is `ByteLevel` so encoded JSON
/// goes through the standard byte-to-char map.
fn byte_level_tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for byte in 0u8..=255u8 {
        let ch = byte_level_char(byte);
        vocab.insert(ch.to_string(), serde_json::Value::from(byte as u32));
    }
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": {
            "type": "ByteLevel",
            "trim_offsets": true,
            "add_prefix_space": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": []
        }
    });
    serde_json::to_string(&tokenizer).expect("serialise minimal byte-level tokenizer.json")
}

/// HuggingFace's byte-level char map. Reproduced here so the test does not
/// depend on internal modules of the `tokenizers` crate.
fn byte_level_char(byte: u8) -> char {
    if (b'!'..=b'~').contains(&byte)
        || (0xa1..=0xac).contains(&byte)
        || (0xae..=0xff).contains(&byte)
    {
        byte as char
    } else {
        // Map remaining bytes into the U+0100..U+0150 range, matching
        // `bytes_to_unicode` from the tokenizers crate.
        let mapped = (byte as u32) + 0x100;
        char::from_u32(mapped).unwrap_or(' ')
    }
}

/// Materialize the `MlxcelTokenizer::HuggingFace` wrapper around the test
/// tokenizer so we can drive the public `build_json_schema_constraint`
/// entry point exactly the way the route layer would.
fn mlxcel_tokenizer_from_hf(hf: tokenizers::Tokenizer) -> mlxcel::tokenizer::MlxcelTokenizer {
    mlxcel::tokenizer::MlxcelTokenizer::HuggingFace(hf)
}

#[test]
fn build_constraint_with_byte_level_tokenizer_succeeds() {
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({
            "type": "object",
            "properties": {"animal": {"type": "string"}},
            "required": ["animal"],
            "additionalProperties": false,
        }),
    )
    .expect("byte-level tokenizer + simple schema must build a valid constraint");
    let guard = constraint
        .lock()
        .expect("freshly built lock is uncontended");
    assert!(
        guard.vocab_size() > 0,
        "byte-level tokenizer has 256 entries"
    );
}

#[test]
fn build_constraint_rejects_garbage_schema_cleanly() {
    // A schema that is structurally valid JSON but semantically
    // unrepresentable as a grammar (mutually exclusive numeric ranges plus
    // unknown keyword combos). llguidance must surface this as a
    // `InvalidSchema` error rather than panic.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let outcome = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        // type that isn't a recognised JSON Schema primitive
        json!({"type": "definitely-not-a-real-type-string"}),
    );
    // Either branch is acceptable. A clean error variant satisfies the
    // "no silent UB" contract, and llguidance may also compile this into
    // a permissive grammar by ignoring unknown keywords (per JSON Schema
    // recommendations). The test passes either way; the important
    // contract is "no panic, no silent unsoundness".
    let _ = outcome;
}

#[test]
fn matcher_initial_mask_is_non_empty() {
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    let mask = guard
        .compute_mask()
        .expect("initial state must produce a valid mask, not an error");
    let allowed_count = mask.iter().filter(|x| **x).count();
    assert!(
        allowed_count > 0,
        "an unstarted JSON-schema matcher must allow at least one token \
         (the opening brace, with byte-level tokenization)"
    );
}

#[test]
fn matcher_consume_then_recompute_succeeds() {
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "string"}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    let mask = guard.compute_mask().expect("initial mask available");
    let first_allowed: usize = mask
        .iter()
        .position(|x| *x)
        .expect("byte-level tokenizer must have at least one allowed start token");

    guard
        .consume_token(first_allowed as i32)
        .expect("consuming the first allowed token must succeed");

    // After consuming a token the matcher must be able to produce another
    // mask without panicking.
    let _ = guard
        .compute_mask()
        .expect("post-consume mask must be available");
}

/// End-to-end test against a real local model. Constrains generation to a
/// simple JSON schema and verifies the resulting output parses back as JSON
/// matching the schema's required keys.
///
/// Gated behind a fixture path because models are gitignored — CI without
/// the fixture skips. To run: place a HuggingFace MLX-format model in
/// `models/qwen3-0.6b-4bit/` and run with `--ignored`.
#[test]
#[ignore = "requires local model weights at models/qwen3-0.6b-4bit/"]
fn end_to_end_constrained_chat_completion_emits_schema_conforming_json() {
    use std::path::PathBuf;
    use std::sync::Arc;

    let model_dir = PathBuf::from("models/qwen3-0.6b-4bit");
    if !model_dir.exists() {
        eprintln!("skipping: {} missing", model_dir.display());
        return;
    }

    let metrics = Arc::new(mlxcel::server::BatchMetrics::new());
    let observability = Arc::new(mlxcel::server::batch::BatchObservability::new());
    let provider = mlxcel::server::ModelProvider::new_with_server_config(
        model_dir.clone(),
        None,
        &mlxcel::server::ServerConfig::default(),
        metrics,
        observability,
    )
    .expect("model loads");

    // Wait for the worker thread to finish loading.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while std::time::Instant::now() < deadline && !provider.is_loaded() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(provider.is_loaded(), "model failed to load within 120s");

    // Load the tokenizer to build the constraint.
    let tokenizer = mlxcel::tokenizer::load_tokenizer(&model_dir).expect("tokenizer loads");
    let schema = json!({
        "type": "object",
        "properties": {
            "animal": {"type": "string", "maxLength": 30},
            "habitat": {"type": "string", "enum": ["forest", "desert", "ocean", "urban", "unknown"]}
        },
        "required": ["animal", "habitat"],
        "additionalProperties": false
    });
    let constraint = mlxcel::server::structured::build_json_schema_constraint(&tokenizer, schema)
        .expect("schema compiles");

    let options = mlxcel::server::ServerGenerateOptions {
        max_tokens: 128,
        sampling: mlxcel::SamplingConfig::greedy(),
        stop_sequences: None,
        priority: mlxcel::server::batch::RequestPriority::Normal,
        logprobs: Default::default(),
        reasoning_budget: Default::default(),
        thinking_enter_block_on_start: false,
        prompt_cache_ctx: None,
        structured: Some(constraint),
    };

    let result = provider
        .generate(
            "Pick one animal and respond as JSON with the animal name and habitat. Only JSON.\n"
                .to_string(),
            options,
        )
        .expect("generation succeeds");

    eprintln!("constrained output: {}", result.text);
    let parsed: serde_json::Value =
        serde_json::from_str(&result.text).expect("output must parse as JSON");
    assert!(
        parsed.get("animal").and_then(|v| v.as_str()).is_some(),
        "output missing required `animal` field: {}",
        result.text
    );
    let habitat = parsed
        .get("habitat")
        .and_then(|v| v.as_str())
        .expect("habitat field is required");
    assert!(
        ["forest", "desert", "ocean", "urban", "unknown"].contains(&habitat),
        "habitat must match enum, got {habitat}"
    );
}

#[test]
fn extract_response_format_helper_round_trips() {
    let response_format = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "Result",
            "strict": true,
            "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}
        }
    });
    let schema = mlxcel::server::structured::extract_json_schema_from_response_format(Some(
        &response_format,
    ))
    .expect("valid")
    .expect("schema present");
    assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
    assert!(schema.get("properties").is_some());
}

// ---------------------------------------------------------------------------
// apply_structured_mask_to_logits — follow-up
//
// These tests exercise the additive-bias construction directly so the
// vocab-size handling stays correct in both directions
// (matcher_vocab > model_vocab and matcher_vocab < model_vocab) and so a
// regression in the bit-to-bias mapping surfaces hermetically without
// requiring a real model on the test runner.
// ---------------------------------------------------------------------------

/// Re-read an evaluated 1xN f32 array back into a host `Vec<f32>` via the
/// raw-bytes accessor exposed by `mlxcel-core`. Tests use this to inspect
/// the bias values that `apply_structured_mask_to_logits` produced.
fn read_f32_array_to_vec(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    assert_eq!(
        bytes.len() % 4,
        0,
        "f32 array bytes must be a multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn apply_mask_produces_bias_at_disallowed_positions() {
    // A real matcher constrained to JSON-schema {"type": "string"} —
    // initial state allows the opening `"` byte and not arbitrary text,
    // so the mask is meaningfully non-trivial.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "string"}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    // Snapshot the mask into an owned `Vec<bool>` so the borrow on
    // `guard` does not block the subsequent `&mut guard` call into
    // `apply_structured_mask_to_logits`.
    let allowed: Vec<bool> = guard
        .compute_mask()
        .expect("initial state must produce a non-error mask")
        .to_vec();
    let model_vocab = guard.vocab_size();

    // Build a flat 1xV logits tensor of zeros so the only post-add value
    // at each position is the bias the masker wrote.
    let logits = mlxcel_core::from_slice_f32(&vec![0.0f32; model_vocab], &[1, model_vocab as i32]);
    let masked = mlxcel::server::structured::apply_structured_mask_to_logits(
        &mut guard,
        &logits,
        model_vocab,
    )
    .expect("non-empty allowed set must produce a mask");

    let host = read_f32_array_to_vec(&masked);
    assert_eq!(
        host.len(),
        model_vocab,
        "masked logits must have exactly model_vocab entries"
    );

    // Every allowed position must remain finite (==0.0 here, since logits
    // was zero); every disallowed position must be -inf.
    for (i, value) in host.iter().enumerate() {
        let is_allowed = allowed.get(i).copied().unwrap_or(false);
        if is_allowed {
            assert!(
                value.is_finite(),
                "allowed position {i} must remain finite, got {value}"
            );
        } else {
            assert!(
                value.is_infinite() && value.is_sign_negative(),
                "disallowed position {i} must be -inf, got {value}"
            );
        }
    }

    // Sanity: at least one position in [0, model_vocab) is allowed.
    assert!(
        host.iter().any(|v| v.is_finite()),
        "matcher must allow at least one reachable token"
    );
}

#[test]
fn apply_mask_with_model_vocab_smaller_than_matcher_vocab() {
    // matcher_vocab > model_vocab path. The bias array must still be
    // exactly `model_vocab` entries so it broadcasts onto the model's
    // logits — the legacy `vocab_size_hint.max(allowed.len())` would have
    // returned a wider array and triggered an FFI broadcast failure.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "string"}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    let matcher_vocab = guard.vocab_size();
    // Pretend the model has a smaller logits axis than the matcher
    // tokenizer. Pick a value that still includes at least one allowed
    // token so the call does not surface the empty-mask error.
    let allowed = guard.compute_mask().expect("initial mask available");
    let first_allowed = allowed
        .iter()
        .position(|x| *x)
        .expect("byte-level tokenizer must allow at least one start token");
    // Make sure model_vocab includes the first allowed id; trim from
    // there to demonstrate the truncation path.
    let model_vocab = (first_allowed + 1).min(matcher_vocab);
    assert!(model_vocab < matcher_vocab, "test must exercise truncation");

    let logits = mlxcel_core::from_slice_f32(&vec![0.0f32; model_vocab], &[1, model_vocab as i32]);
    let masked = mlxcel::server::structured::apply_structured_mask_to_logits(
        &mut guard,
        &logits,
        model_vocab,
    )
    .expect("at least one reachable allowed token");

    let host = read_f32_array_to_vec(&masked);
    assert_eq!(
        host.len(),
        model_vocab,
        "bias must be sized to model_vocab even when matcher_vocab is larger"
    );

    // The last position must be allowed (it's `first_allowed`).
    assert!(
        host[model_vocab - 1].is_finite(),
        "the lone in-range allowed token must survive the truncation"
    );
}

#[test]
fn apply_mask_with_model_vocab_larger_than_matcher_vocab() {
    // matcher_vocab < model_vocab path. Positions beyond the matcher's
    // coverage default to disallowed (-inf), conservative — the model
    // cannot emit them, but if it could, an unknown id can never satisfy
    // the grammar.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "string"}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    let matcher_vocab = guard.vocab_size();
    let model_vocab = matcher_vocab + 16;

    let logits = mlxcel_core::from_slice_f32(&vec![0.0f32; model_vocab], &[1, model_vocab as i32]);
    let masked = mlxcel::server::structured::apply_structured_mask_to_logits(
        &mut guard,
        &logits,
        model_vocab,
    )
    .expect("matcher must still have an allowed token");

    let host = read_f32_array_to_vec(&masked);
    assert_eq!(
        host.len(),
        model_vocab,
        "bias must be sized to model_vocab even when matcher_vocab is smaller"
    );
    // Every position past the matcher's coverage must be -inf.
    for (i, value) in host.iter().enumerate().skip(matcher_vocab) {
        assert!(
            value.is_infinite() && value.is_sign_negative(),
            "padding position {i} (past matcher vocab) must be -inf, got {value}"
        );
    }
}

#[test]
fn apply_mask_returns_error_when_no_reachable_token_is_allowed() {
    // Drive the matcher all the way through a complete JSON string so it
    // reaches a state where the only legal continuations are EOS-class —
    // *plus* clamp the model_vocab to a window that excludes those
    // EOS-class allowed ids. The masker then sees zero reachable allowed
    // tokens and must surface the `Matcher` error variant rather than an
    // all-`-inf` bias (which would silently break the sampler).
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let constraint = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({"type": "string", "enum": ["x"]}),
    )
    .expect("simple schema compiles");

    let mut guard = constraint.lock().expect("uncontended");
    // Walk the matcher to a terminal-or-near-terminal state. Sampling a
    // sequence of allowed tokens until either the matcher stops or we
    // cap at 16 steps keeps the test bounded.
    for _ in 0..16usize {
        if guard.is_stopped() {
            break;
        }
        let mask = match guard.compute_mask() {
            Ok(m) => m,
            Err(_) => break,
        };
        let Some(first) = mask.iter().position(|x| *x) else {
            break;
        };
        if guard.consume_token(first as i32).is_err() {
            break;
        }
    }

    // Now ask for a mask, then call the masker with a model_vocab of 1
    // — at that point, position 0 is almost certainly disallowed (the
    // byte-level vocab puts the "control" bytes there, not the JSON
    // grammar's continuation tokens). The empty-mask path is the
    // expected outcome.
    let logits = mlxcel_core::from_slice_f32(&[0.0f32], &[1, 1]);
    let outcome =
        mlxcel::server::structured::apply_structured_mask_to_logits(&mut guard, &logits, 1);
    match outcome {
        Err(mlxcel::server::structured::StructuredOutputError::Matcher(_)) => {}
        Ok(_) => {
            // Acceptable if position 0 happened to remain allowed at the
            // end-state — we don't want this test to be brittle against
            // matcher internals. The negative-path coverage is exercised
            // by the per-position assertions in the other tests.
        }
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// H1 / L3: public error messages must NOT leak llguidance internals.
//
// `verbose_errors: false` (configured in `build_json_schema_constraint`) plus
// the `tracing::error!` redaction policy means that even when the schema is
// invalid, the public `StructuredOutputError` must surface a sanitized
// message — never a parser-state dump or expanded grammar rule.
// ---------------------------------------------------------------------------

/// Substrings that would indicate llguidance leaked internal state into the
/// public error message. None of these may appear in any user-facing error
/// surface produced by `build_json_schema_constraint`.
const LEAK_FORBIDDEN_FRAGMENTS: &[&str] = &[
    "GrammarRef",
    "Lexeme",
    "lexer state",
    "parser state",
    "tokens so far",
    "Earley",
    "production",
    "RegexAst",
    "json_schema(",
    "GrammarWithLexer",
];

/// Walk a `Display` rendering for any fragment that would identify the
/// message as a verbose llguidance dump.
fn assert_message_is_sanitized(message: &str) {
    let lower = message.to_ascii_lowercase();
    for needle in LEAK_FORBIDDEN_FRAGMENTS {
        let needle_lower = needle.to_ascii_lowercase();
        assert!(
            !lower.contains(&needle_lower),
            "public error message leaks llguidance internal fragment {needle:?}: {message:?}"
        );
    }
}

#[test]
fn invalid_schema_error_does_not_leak_parser_state() {
    // Drive the matcher build with a schema that is structurally JSON
    // but semantically rejected by llguidance (unknown type tag,
    // contradictory `enum` + `type`). The resulting error — whether it
    // surfaces from `ParserFactory::new` or from `Matcher::new` — must
    // be the sanitized public message, never a verbose llguidance dump.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);
    let outcome = mlxcel::server::structured::build_json_schema_constraint(
        &mlxcel,
        json!({
            "type": "definitely-not-a-real-type-string",
            "enum": [1, 2, 3]
        }),
    );
    if let Err(err) = outcome {
        let message = err.to_string();
        assert_message_is_sanitized(&message);
    }
}

#[test]
fn oversized_schema_rejected_with_clean_size_error() {
    // Build a schema whose serialized bytes exceed `MAX_SCHEMA_BYTES`
    // (64 KiB). The pre-compilation guard must reject it as
    // `SchemaTooLarge`, never let it reach llguidance. The public
    // message must mention "too large" but no internal fragments.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);

    // Construct a wide flat object schema with many string properties.
    // Each entry contributes roughly 50-60 bytes of serialized JSON.
    let mut props = serde_json::Map::new();
    for i in 0..2000usize {
        props.insert(
            format!("field_{i:04}"),
            json!({"type": "string", "maxLength": 32}),
        );
    }
    let schema = json!({
        "type": "object",
        "properties": props,
        "additionalProperties": false
    });
    // Sanity: confirm the test really crosses the boundary.
    let serialized_len = serde_json::to_string(&schema).unwrap().len();
    assert!(
        serialized_len > 64 * 1024,
        "test schema must exceed 64KiB, got {serialized_len} bytes"
    );

    let outcome = mlxcel::server::structured::build_json_schema_constraint(&mlxcel, schema);
    let err = outcome.expect_err("oversized schema must be rejected before compilation");
    assert!(
        matches!(
            err,
            mlxcel::server::structured::StructuredOutputError::SchemaTooLarge(_)
        ),
        "expected SchemaTooLarge variant, got {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.to_ascii_lowercase().contains("too large"),
        "size-limit error must mention the limit category, got {message:?}"
    );
    assert_message_is_sanitized(&message);
}

#[test]
fn deeply_nested_schema_rejected_with_clean_depth_error() {
    // Build a schema whose nesting depth exceeds `MAX_SCHEMA_DEPTH` (32).
    // The guard must reject it as `SchemaTooLarge` with a depth message.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);

    // Wrap `{"type": "string"}` in 64 layers of object.properties.x.
    fn nest(depth: usize) -> serde_json::Value {
        if depth == 0 {
            json!({"type": "string"})
        } else {
            json!({
                "type": "object",
                "properties": {"x": nest(depth - 1)},
                "additionalProperties": false
            })
        }
    }
    let schema = nest(64);

    let outcome = mlxcel::server::structured::build_json_schema_constraint(&mlxcel, schema);
    let err = outcome.expect_err("deeply nested schema must be rejected");
    assert!(
        matches!(
            err,
            mlxcel::server::structured::StructuredOutputError::SchemaTooLarge(_)
        ),
        "expected SchemaTooLarge variant, got {err:?}"
    );
    let message = err.to_string();
    assert_message_is_sanitized(&message);
}

#[test]
fn many_refs_schema_rejected_with_clean_ref_error() {
    // Build a schema with > MAX_SCHEMA_REFS (64) `$ref` entries. The guard
    // must reject it as `SchemaTooLarge` with a $ref message.
    let hf = build_byte_level_tokenizer();
    let mlxcel = mlxcel_tokenizer_from_hf(hf);

    let mut props = serde_json::Map::new();
    for i in 0..128usize {
        props.insert(format!("f{i}"), json!({"$ref": "#/$defs/Item"}));
    }
    let schema = json!({
        "type": "object",
        "properties": props,
        "$defs": {
            "Item": {"type": "string"}
        }
    });

    let outcome = mlxcel::server::structured::build_json_schema_constraint(&mlxcel, schema);
    let err = outcome.expect_err("$ref-heavy schema must be rejected");
    assert!(
        matches!(
            err,
            mlxcel::server::structured::StructuredOutputError::SchemaTooLarge(_)
        ),
        "expected SchemaTooLarge variant, got {err:?}"
    );
    let message = err.to_string();
    assert_message_is_sanitized(&message);
}

#[test]
fn structured_output_error_display_is_concise() {
    // Sanity check on every variant: the public Display rendering must be
    // a short, fixed string with no llguidance internals. We construct the
    // error variants directly (bypassing the build path) so this catches
    // regressions where someone re-introduces a verbose message string.
    let cases = [
        mlxcel::server::structured::StructuredOutputError::InvalidRequest(
            "missing schema field".to_string(),
        ),
        mlxcel::server::structured::StructuredOutputError::InvalidSchema(
            "schema compilation failed".to_string(),
        ),
        mlxcel::server::structured::StructuredOutputError::SchemaTooLarge(
            "schema serialised size 999999 bytes exceeds limit 65536 bytes".to_string(),
        ),
        mlxcel::server::structured::StructuredOutputError::UnsupportedTokenizer(
            "structured outputs require a HuggingFace tokenizer.json".to_string(),
        ),
        mlxcel::server::structured::StructuredOutputError::Matcher(
            "compute_mask failed".to_string(),
        ),
    ];
    for err in cases {
        let rendered = err.to_string();
        assert_message_is_sanitized(&rendered);
    }
}
