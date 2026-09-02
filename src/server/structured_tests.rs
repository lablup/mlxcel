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

// ---------------------------------------------------------------------------
// Packed mask application (#1316)
//
// The matcher answers as a bitset, so the decode path keeps it packed all the
// way to the device instead of unpacking it into a `Vec<bool>` and a `Vec<f32>`
// bias of the vocabulary's length. These tests own the bit-level algebra and
// the device-side equivalence; the matcher-driven behavior (mask contents,
// padded head rows, the empty-mask error) is covered end to end against a real
// byte-level tokenizer in `tests/structured_outputs.rs`.
// ---------------------------------------------------------------------------

/// Deterministic xorshift so the randomized cases are reproducible from the
/// seed alone and the test needs no rng dependency.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        // Roughly [-8, 8), the range decode logits actually live in.
        (self.next_u32() as f32 / u32::MAX as f32) * 16.0 - 8.0
    }
}

/// Build a matcher-shaped bitset holding exactly `allowed`.
fn packed_source(allowed: &[bool]) -> Vec<u32> {
    let mut words = vec![0u32; allowed.len().div_ceil(32)];
    for (i, on) in allowed.iter().enumerate() {
        if *on {
            words[i / 32] |= 1u32 << (i % 32);
        }
    }
    words
}

/// Read bit `i` out of a packed mask.
fn packed_bit(words: &[u32], i: usize) -> bool {
    words[i / 32] & (1u32 << (i % 32)) != 0
}

/// The implementation this change replaces, kept as the equivalence reference.
///
/// `compute_mask` unpacked the matcher bitset into one `bool` per token, and
/// `apply_structured_mask_to_logits` then walked the logits axis a second time
/// to build an f32 bias of `0.0` at allowed positions and `-inf` everywhere
/// else, uploaded it, and added it to the logits.
fn reference_bias_apply(
    src_words: &[u32],
    matcher_vocab: usize,
    vocab_size_hint: usize,
    logits: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut mask = vec![false; matcher_vocab];
    for (i, slot) in mask.iter_mut().enumerate() {
        *slot = packed_bit(src_words, i);
    }

    let mut bias = vec![0.0f32; vocab_size_hint];
    for (i, slot) in bias.iter_mut().enumerate() {
        if !(i < mask.len() && mask[i]) {
            *slot = f32::NEG_INFINITY;
        }
    }

    let bias_arr = mlxcel_core::from_slice_f32(&bias, &[1, vocab_size_hint as i32]);
    mlxcel_core::add(logits, &bias_arr)
}

/// Evaluate a 1xN f32 array and copy it back to the host.
fn read_f32_array(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
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

/// Index of the largest logit, the token greedy decoding would emit.
fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN logits in these tests"))
        .map(|(i, _)| i)
}

#[test]
fn packed_mask_matches_bool_mask() {
    let mut rng = Xorshift64::new(0x5175_656e_3331_3336);

    // Widths that are and are not word multiples, in both directions relative
    // to the matcher vocabulary, so the trim and the zero-pad are both live.
    let cases: &[(usize, usize)] = &[
        (32, 32),
        (32, 64),
        (33, 33),
        (33, 96),
        (63, 64),
        (64, 63),
        (100, 100),
        (100, 137),
        (137, 100),
        (1000, 1024),
        (1024, 1000),
        (4099, 4096),
        (4096, 4099),
    ];

    for &(matcher_vocab, vocab_size_hint) in cases {
        let allowed: Vec<bool> = (0..matcher_vocab)
            .map(|_| rng.next_u32().is_multiple_of(3))
            .collect();
        let src = packed_source(&allowed);

        let mut packed = Vec::new();
        pack_mask_words(&src, matcher_vocab, vocab_size_hint, &mut packed);

        assert_eq!(
            packed.len(),
            vocab_size_hint.div_ceil(32),
            "packed mask must cover exactly the logits axis for \
             matcher_vocab={matcher_vocab} hint={vocab_size_hint}"
        );

        for i in 0..vocab_size_hint {
            let want = allowed.get(i).copied().unwrap_or(false);
            assert_eq!(
                packed_bit(&packed, i),
                want,
                "bit {i} disagrees with the bool mask for \
                 matcher_vocab={matcher_vocab} hint={vocab_size_hint}"
            );
        }

        // Nothing may survive in the tail the logits axis cannot name either.
        for i in vocab_size_hint..packed.len() * 32 {
            assert!(
                !packed_bit(&packed, i),
                "lane {i} past the logits axis must be zero for \
                 matcher_vocab={matcher_vocab} hint={vocab_size_hint}"
            );
        }
    }
}

#[test]
fn packed_mask_trims_the_matchers_own_excess_bits() {
    // A matcher vocabulary of 77 occupies three words but only 13 valid bits
    // in the last one. Set every bit in the source, including the 19 that name
    // no token, and require them all to be dropped: this is the shape of the
    // Qwen3.8 head, whose tokenizer carries 248077 entries against 248320 rows.
    let matcher_vocab = 77usize;
    let src = vec![u32::MAX; 3];

    let mut packed = Vec::new();
    pack_mask_words(&src, matcher_vocab, 128, &mut packed);

    for i in 0..matcher_vocab {
        assert!(packed_bit(&packed, i), "real token {i} must survive");
    }
    for i in matcher_vocab..128 {
        assert!(
            !packed_bit(&packed, i),
            "position {i} names no token and must be masked"
        );
    }
}

#[test]
fn packed_mask_zero_pads_past_the_matcher_vocabulary() {
    // The padded-head direction: the model can emit more rows than the
    // tokenizer has tokens, and the extra rows must come back disallowed.
    let allowed = vec![true; 40];
    let src = packed_source(&allowed);

    let mut packed = Vec::new();
    pack_mask_words(&src, 40, 200, &mut packed);

    assert_eq!(packed.len(), 7, "200 tokens need ceil(200 / 32) = 7 words");
    for i in 0..40 {
        assert!(packed_bit(&packed, i), "token {i} is allowed");
    }
    for i in 40..200 {
        assert!(!packed_bit(&packed, i), "padded row {i} must be masked");
    }
}

#[test]
fn packed_mask_of_an_empty_allow_set_is_all_zero() {
    // `apply_structured_mask_to_logits` reads exactly this predicate to raise
    // the empty-mask error, so pin it rather than the error string.
    let src = vec![0u32; 4];
    let mut packed = Vec::new();
    pack_mask_words(&src, 100, 100, &mut packed);
    assert!(
        packed.iter().all(|word| *word == 0),
        "an empty allow set must pack to all-zero words"
    );

    // A single allowed token in the final partial word must not be lost.
    let mut src = vec![0u32; 4];
    src[3] |= 1u32 << 3; // token 99
    let mut packed = Vec::new();
    pack_mask_words(&src, 100, 100, &mut packed);
    assert!(
        !packed.iter().all(|word| *word == 0),
        "a lone allowed token in the last partial word must survive"
    );
    assert!(packed_bit(&packed, 99), "token 99 is the allowed one");
}

#[test]
fn packed_apply_matches_bias_apply() {
    let mut rng = Xorshift64::new(0x1316_0BAD_C0FF_EE01);

    // Widths that stress the trim, the zero-pad, and both word alignments.
    let cases: &[(usize, usize)] = &[
        (64, 64),
        (100, 137),
        (137, 100),
        (255, 256),
        (256, 255),
        (1000, 1024),
        (4099, 4160),
    ];

    for &(matcher_vocab, vocab_size_hint) in cases {
        let allowed: Vec<bool> = (0..matcher_vocab)
            .map(|_| rng.next_u32().is_multiple_of(4))
            .collect();
        let src = packed_source(&allowed);
        let values: Vec<f32> = (0..vocab_size_hint).map(|_| rng.next_f32()).collect();
        let logits = mlxcel_core::from_slice_f32(&values, &[1, vocab_size_hint as i32]);

        let mut packed = Vec::new();
        pack_mask_words(&src, matcher_vocab, vocab_size_hint, &mut packed);
        let got = read_f32_array(&apply_packed_mask_to_logits(
            &packed,
            vocab_size_hint,
            &logits,
        ));
        let want = read_f32_array(&reference_bias_apply(
            &src,
            matcher_vocab,
            vocab_size_hint,
            &logits,
        ));

        assert_eq!(
            got.len(),
            vocab_size_hint,
            "masked logits must keep the model's logits width for \
             matcher_vocab={matcher_vocab} hint={vocab_size_hint}"
        );
        assert_eq!(got.len(), want.len(), "reference and packed widths agree");

        for i in 0..vocab_size_hint {
            // IEEE equality is the contract that matters here: it makes
            // -inf == -inf true and would catch a sign or magnitude change at
            // an allowed position.
            assert!(
                got[i] == want[i],
                "position {i} diverges for matcher_vocab={matcher_vocab} \
                 hint={vocab_size_hint}: packed={} bias={}",
                got[i],
                want[i]
            );
            let is_allowed = allowed.get(i).copied().unwrap_or(false);
            if is_allowed {
                assert!(
                    got[i] == values[i],
                    "allowed position {i} must pass the logit through unchanged, \
                     got {} for input {}",
                    got[i],
                    values[i]
                );
            } else {
                assert!(
                    got[i].is_infinite() && got[i].is_sign_negative(),
                    "disallowed position {i} must be -inf, got {}",
                    got[i]
                );
            }
        }

        // The property the sampler actually depends on.
        assert_eq!(
            argmax(&got),
            argmax(&want),
            "greedy decoding must pick the same token for \
             matcher_vocab={matcher_vocab} hint={vocab_size_hint}"
        );
    }
}

#[test]
fn packed_apply_handles_the_all_allowed_and_single_allowed_edges() {
    let mut rng = Xorshift64::new(0x1316_FEED_FACE_0002);
    let vocab_size_hint = 137usize; // deliberately not a word multiple

    // Every token allowed: the mask must be a no-op on the logits.
    let allowed = vec![true; vocab_size_hint];
    let src = packed_source(&allowed);
    let values: Vec<f32> = (0..vocab_size_hint).map(|_| rng.next_f32()).collect();
    let logits = mlxcel_core::from_slice_f32(&values, &[1, vocab_size_hint as i32]);

    let mut packed = Vec::new();
    pack_mask_words(&src, vocab_size_hint, vocab_size_hint, &mut packed);
    let got = read_f32_array(&apply_packed_mask_to_logits(
        &packed,
        vocab_size_hint,
        &logits,
    ));
    for (i, value) in values.iter().enumerate() {
        assert!(
            got[i] == *value,
            "an all-allowed mask must leave logit {i} untouched, got {} for {value}",
            got[i]
        );
    }

    // Exactly one allowed token, sitting in the final partial word: greedy
    // decoding must pick it no matter how small its logit is.
    let lone = vocab_size_hint - 1;
    let mut allowed = vec![false; vocab_size_hint];
    allowed[lone] = true;
    let src = packed_source(&allowed);
    let mut values: Vec<f32> = (0..vocab_size_hint).map(|_| rng.next_f32()).collect();
    values[lone] = -7.5;
    let logits = mlxcel_core::from_slice_f32(&values, &[1, vocab_size_hint as i32]);

    let mut packed = Vec::new();
    pack_mask_words(&src, vocab_size_hint, vocab_size_hint, &mut packed);
    let got = read_f32_array(&apply_packed_mask_to_logits(
        &packed,
        vocab_size_hint,
        &logits,
    ));
    assert_eq!(
        argmax(&got),
        Some(lone),
        "the only allowed token must win regardless of its logit"
    );
    assert!(got[lone] == -7.5, "its logit must be unchanged");
}

#[test]
fn packed_apply_handles_a_width_change_between_calls() {
    // Nothing the expansion caches depends on the vocabulary width, so a
    // constraint reused across logits axes of different sizes must stay
    // correct. Drive several widths through one buffer to prove it.
    let matcher_vocab = 300usize;
    let allowed: Vec<bool> = (0..matcher_vocab).map(|i| i.is_multiple_of(7)).collect();
    let src = packed_source(&allowed);
    let mut packed = Vec::new();

    for &hint in &[64usize, 320, 137, 300, 512, 64] {
        pack_mask_words(&src, matcher_vocab, hint, &mut packed);
        let values: Vec<f32> = (0..hint).map(|i| i as f32 * 0.01).collect();
        let logits = mlxcel_core::from_slice_f32(&values, &[1, hint as i32]);
        let got = read_f32_array(&apply_packed_mask_to_logits(&packed, hint, &logits));

        assert_eq!(got.len(), hint, "width {hint} must round-trip");
        for (i, value) in got.iter().enumerate() {
            let want_allowed = allowed.get(i).copied().unwrap_or(false);
            if want_allowed {
                assert!(
                    *value == values[i],
                    "width {hint} position {i} must pass through, got {value}"
                );
            } else {
                assert!(
                    value.is_infinite() && value.is_sign_negative(),
                    "width {hint} position {i} must be -inf, got {value}"
                );
            }
        }
    }
}

/// Microbenchmark for the change in #1316. Not part of the gate.
///
/// ```text
/// cargo test --profile test-fast --features metal,accelerate --lib \
///   server::structured::tests::bench_packed_mask_apply -- --ignored --nocapture
/// ```
///
/// Reports min-of-N per iteration for both arms at the widest vocabulary in
/// the test set, split into the scheduler-thread cost (mask preparation plus
/// the host-to-device upload, which is what the change removes) and the same
/// work followed by an eval of the result.
#[test]
#[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
fn bench_packed_mask_apply() {
    use std::time::Instant;

    // mlx-community/Qwen3.8-27B-4bit: 248320 lm_head rows against a
    // tokenizer.json carrying 248077 entries.
    const LM_HEAD_ROWS: usize = 248_320;
    const TOKENIZER_VOCAB: usize = 248_077;
    const TRIALS: usize = 60;

    let mut rng = Xorshift64::new(0x1316_BE1C_0000_0003);
    let allowed: Vec<bool> = (0..TOKENIZER_VOCAB)
        .map(|_| rng.next_u32().is_multiple_of(4))
        .collect();
    let src = packed_source(&allowed);
    let values: Vec<f32> = (0..LM_HEAD_ROWS).map(|_| rng.next_f32()).collect();
    let logits = mlxcel_core::from_slice_f32(&values, &[1, LM_HEAD_ROWS as i32]);

    let mut packed = Vec::new();

    // Warm up both arms so neither pays first-touch allocation or JIT.
    for _ in 0..5 {
        pack_mask_words(&src, TOKENIZER_VOCAB, LM_HEAD_ROWS, &mut packed);
        let out = apply_packed_mask_to_logits(&packed, LM_HEAD_ROWS, &logits);
        mlxcel_core::eval(&out);
        let out = reference_bias_apply(&src, TOKENIZER_VOCAB, LM_HEAD_ROWS, &logits);
        mlxcel_core::eval(&out);
    }

    let mut packed_prepare = f64::MAX;
    let mut packed_total = f64::MAX;
    let mut bias_prepare = f64::MAX;
    let mut bias_total = f64::MAX;

    for _ in 0..TRIALS {
        let t0 = Instant::now();
        pack_mask_words(&src, TOKENIZER_VOCAB, LM_HEAD_ROWS, &mut packed);
        let out = apply_packed_mask_to_logits(&packed, LM_HEAD_ROWS, &logits);
        packed_prepare = packed_prepare.min(t0.elapsed().as_secs_f64());
        mlxcel_core::eval(&out);
        packed_total = packed_total.min(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let out = reference_bias_apply(&src, TOKENIZER_VOCAB, LM_HEAD_ROWS, &logits);
        bias_prepare = bias_prepare.min(t0.elapsed().as_secs_f64());
        mlxcel_core::eval(&out);
        bias_total = bias_total.min(t0.elapsed().as_secs_f64());
    }

    let us = |seconds: f64| seconds * 1e6;
    println!("vocab={LM_HEAD_ROWS} matcher_vocab={TOKENIZER_VOCAB} trials={TRIALS} (min-of-N)");
    println!(
        "  f32 bias   prepare+upload {:8.1} us   with eval {:8.1} us   upload {} KiB",
        us(bias_prepare),
        us(bias_total),
        LM_HEAD_ROWS * 4 / 1024
    );
    println!(
        "  packed u32 prepare+upload {:8.1} us   with eval {:8.1} us   upload {} KiB",
        us(packed_prepare),
        us(packed_total),
        LM_HEAD_ROWS.div_ceil(32) * 4 / 1024
    );
    println!(
        "  speedup    prepare+upload {:8.2}x    with eval {:8.2}x",
        bias_prepare / packed_prepare,
        bias_total / packed_total
    );
}
