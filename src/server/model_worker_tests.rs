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

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

use super::{
    StreamingDecodeState, build_generation_result, decode_request_images,
    decode_request_images_with_limits, merge_config_stop_tokens, parse_byte_fallback_token,
    safe_emit_boundary,
};
use crate::SamplingConfig;
use crate::server::media::ImageInputLimits;
use crate::tokenizer::MlxcelTokenizer;
use crate::worker_failfast::run_core_thread_or_abort;

fn encode_png_bytes() -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([0, 0, 0])));
    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png).unwrap();
    cursor.into_inner()
}

fn png_chunk(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(name, data).to_be_bytes());
    out
}

fn crc32(name: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in name.iter().chain(data.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn synthetic_png_bomb_header(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    png
}

#[test]
fn merge_config_stop_tokens_appends_only_missing_ids() {
    let sampling = SamplingConfig {
        stop_token_ids: vec![2, 3],
        ..SamplingConfig::greedy()
    };

    let merged = merge_config_stop_tokens(sampling, &[3, 4, 5]);
    assert_eq!(merged.stop_token_ids, vec![2, 3, 4, 5]);
}

#[test]
fn decode_request_images_keeps_valid_images_and_rejects_all_invalid_input() {
    let decoded = decode_request_images(&[encode_png_bytes(), vec![1, 2, 3]]).unwrap();
    assert_eq!(decoded.len(), 1);

    let err = decode_request_images(&[vec![1, 2, 3]]).unwrap_err();
    assert!(err.to_string().contains("Failed to decode any images"));
}

#[test]
fn decode_request_images_rejects_decompression_bomb_header() {
    let limits = ImageInputLimits {
        max_width: 4096,
        max_height: 4096,
        max_decode_alloc_bytes: 32 * 1024 * 1024,
        ..ImageInputLimits::default()
    };
    let err =
        decode_request_images_with_limits(&[synthetic_png_bomb_header(100_000, 100_000)], limits)
            .unwrap_err();

    assert!(
        err.to_string()
            .contains("Image decode rejected by configured limits"),
        "{err}"
    );
}

#[test]
fn build_generation_result_computes_finish_reason_and_generation_split() {
    let stop = build_generation_result("ok".to_string(), 10, 3, 120, 40, 8);
    assert_eq!(stop.finish_reason, "stop");
    assert_eq!(stop.generation_only_ms, 80);

    let length = build_generation_result("ok".to_string(), 10, 8, 50, 60, 8);
    assert_eq!(length.finish_reason, "length");
    assert_eq!(length.generation_only_ms, 0);
}

#[test]
fn safe_emit_boundary_stops_before_trailing_replacement_chars() {
    // ASCII string: boundary is at the end.
    assert_eq!(safe_emit_boundary("hello"), 5);

    // Replacement char at end: boundary stops before it.
    let with_replacement = "ok\u{FFFD}";
    let expected = "ok".len();
    assert_eq!(safe_emit_boundary(with_replacement), expected);

    // All replacement chars: boundary is 0.
    assert_eq!(safe_emit_boundary("\u{FFFD}\u{FFFD}"), 0);

    // Empty string: boundary is 0.
    assert_eq!(safe_emit_boundary(""), 0);

    // Multi-byte character followed by replacement char.
    let mixed = "\u{AC00}\u{FFFD}"; // Korean syllable + replacement
    assert_eq!(safe_emit_boundary(mixed), "\u{AC00}".len()); // 3 bytes
}

// ── parse_byte_fallback_token tests ─────────────────────────────────────────

/// Tokens in the form `<0xXX>` with a two-digit hex suffix should return the
/// corresponding byte value.
#[test]
fn parse_byte_fallback_token_recognises_hex_tokens() {
    assert_eq!(parse_byte_fallback_token("<0x00>"), Some(0x00));
    assert_eq!(parse_byte_fallback_token("<0x61>"), Some(0x61)); // 'a'
    assert_eq!(parse_byte_fallback_token("<0xE5>"), Some(0xE5));
    assert_eq!(parse_byte_fallback_token("<0xAB>"), Some(0xAB));
    assert_eq!(parse_byte_fallback_token("<0xFF>"), Some(0xFF));
}

/// Tokens that do not match the exact `<0xXX>` pattern must return `None`.
#[test]
fn parse_byte_fallback_token_rejects_non_hex_tokens() {
    assert_eq!(parse_byte_fallback_token("Hello"), None);
    assert_eq!(parse_byte_fallback_token("<BOS>"), None);
    assert_eq!(parse_byte_fallback_token("<0x>"), None); // too short
    assert_eq!(parse_byte_fallback_token("<0xABC>"), None); // too long
    assert_eq!(parse_byte_fallback_token("0xE5"), None); // missing angle brackets
    assert_eq!(parse_byte_fallback_token("<0xGG>"), None); // invalid hex
    assert_eq!(parse_byte_fallback_token(""), None);
}

/// Defense-in-depth: `u8::from_str_radix` accepts a leading `+` sign, so
/// `<0x+f>` would previously parse as `Some(0x0f)`. Byte-level checks now
/// ensure both digit positions must be ASCII hex digits, rejecting `+` and `-`.
#[test]
fn parse_byte_fallback_token_rejects_leading_sign() {
    // These six-character strings match the length and prefix/suffix of a valid
    // byte-fallback token but contain a leading sign in the hex digit area.
    assert_eq!(parse_byte_fallback_token("<0x+f>"), None);
    assert_eq!(parse_byte_fallback_token("<0x-f>"), None);
    assert_eq!(parse_byte_fallback_token("<0x+F>"), None);
    // Valid tokens continue to work after the byte-level guard is applied.
    assert_eq!(parse_byte_fallback_token("<0xE5>"), Some(0xE5));
    assert_eq!(parse_byte_fallback_token("<0x00>"), Some(0x00));
    assert_eq!(parse_byte_fallback_token("<0xff>"), Some(0xff));
}

// ── Byte-fallback streaming regression tests ───────────────────

/// Helper: simulate streaming a sequence of tokens and collect the emitted
/// chunks. Returns (chunks, final_text_from_finish).
///
/// `flush()` is called at the end to simulate end-of-stream, and the final
/// text is obtained via `finish_with_cache()` which is the same method the
/// batch scheduler uses in production.
fn simulate_byte_fallback_stream(
    tokenizer: &MlxcelTokenizer,
    prompt_ids: &[i32],
    gen_ids: &[i32],
) -> (Vec<String>, String) {
    use std::time::Instant;
    let mut state = StreamingDecodeState::new(tokenizer, prompt_ids);
    let mut chunks = Vec::new();
    for &tok in gen_ids {
        if let Some(chunk) = state.on_token(tok, tokenizer) {
            chunks.push(chunk);
        }
    }
    state.flush(tokenizer);
    let start = Instant::now();
    let result = state.finish_with_cache(start, prompt_ids.len(), usize::MAX, 0);
    (chunks, result.text)
}

/// A stream of three byte-fallback tokens [<0xE5>, <0x8F>, <0xAB>] that
/// together encode the CJK character "叫" (U+53EB) must:
/// 1. Produce no chunks for the first two tokens (incomplete UTF-8).
/// 2. Emit exactly "叫" when the third token arrives.
/// 3. Concatenated chunks == non-streaming decode of the same token sequence.
#[test]
fn byte_fallback_cjk_streams_only_complete_sequences() {
    // In the stub tokenizer: 5=<0xE5>, 6=<0x8F>, 7=<0xAB>
    // "叫" = 0xE5 0x8F 0xAB
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[5, 6, 7]; // <0xE5>, <0x8F>, <0xAB>

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    // The character must appear exactly once, emitted as a single chunk.
    assert_eq!(generated, "\u{53EB}", "generated text must be '叫'");

    // All streaming chunks must contain only valid complete characters.
    let concatenated: String = chunks.iter().cloned().collect();
    assert_eq!(
        concatenated, generated,
        "streamed chunks must equal non-streaming result"
    );

    // No individual chunk must contain U+FFFD (premature replacement char).
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "streamed chunk must not contain replacement characters: {chunk:?}"
        );
    }
}

/// A stream of four byte-fallback tokens encoding the emoji "😀" (U+1F600,
/// UTF-8: F0 9F 98 80) must hold all four bytes until the last arrives and
/// then emit the emoji as a single valid chunk.
#[test]
fn byte_fallback_emoji_streams_only_complete_sequences() {
    // In the stub tokenizer: 8=<0xF0>, 9=<0x9F>, 10=<0x98>, 11=<0x80>
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[8, 9, 10, 11];

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    assert_eq!(generated, "\u{1F600}", "generated text must be '😀'");
    let concatenated: String = chunks.iter().cloned().collect();
    assert_eq!(concatenated, generated);
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "streamed chunk must not contain replacement characters: {chunk:?}"
        );
    }
}

/// A stream of regular tokens followed by byte-fallback tokens for "叫" must
/// emit the regular tokens immediately and then the CJK character only once
/// the full byte sequence is received.
#[test]
fn byte_fallback_mixed_regular_and_cjk_streams_correctly() {
    // Token 2 = "Hello", tokens 5/6/7 = "叫"
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[2, 5, 6, 7]; // "Hello" + 叫

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    // Full output must equal "Hello叫".
    assert_eq!(generated, "Hello\u{53EB}");

    // Concatenated chunks must equal the generated text.
    let concatenated: String = chunks.iter().cloned().collect();
    assert_eq!(concatenated, generated);

    // No chunk must contain U+FFFD.
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "streamed chunk must not contain replacement characters: {chunk:?}"
        );
    }
}

/// A single-byte byte-fallback token `<0x61>` (= 'a') must be emitted
/// immediately since it forms a valid one-byte UTF-8 sequence on its own.
#[test]
fn byte_fallback_single_byte_ascii_emits_immediately() {
    // Token 12 = <0x61> = 'a'
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[12]; // <0x61>

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    assert_eq!(generated, "a");
    let concatenated: String = chunks.iter().cloned().collect();
    assert_eq!(concatenated, "a");
}

/// Incomplete byte-fallback sequences at end-of-stream must be flushed by
/// `flush()` as replacement characters, matching the ByteFallback decoder's
/// own behaviour for invalid byte sequences.
///
/// Note: replacement chars from end-of-stream flushing are appended to
/// `generated_text` but not emitted as streaming delta chunks. This is
/// consistent with how the non-streaming decoder handles incomplete sequences
/// (the final result includes U+FFFD; no intermediate chunk is sent).
#[test]
fn byte_fallback_incomplete_sequence_flushed_at_end_of_stream() {
    // Token 5 = <0xE5>: alone, this is an incomplete UTF-8 sequence.
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[5]; // <0xE5> alone (incomplete)

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    // An incomplete sequence must flush as one replacement char in the final text.
    assert_eq!(
        generated, "\u{FFFD}",
        "incomplete byte-fallback must flush as U+FFFD"
    );

    // Incomplete sequences are not emitted as streaming chunks (they only appear
    // in the final text after end-of-stream flush). So the chunks should be empty
    // and the concatenation of streaming chunks does not include the replacement char.
    assert!(
        chunks.is_empty() || chunks.iter().all(|c| !c.contains('\u{FFFD}')),
        "incomplete byte-fallback must not produce replacement chars in streaming chunks: {chunks:?}"
    );
}

/// An incomplete byte-fallback sequence immediately followed by a regular token
/// must emit the replacement character(s) for the invalid bytes AND the regular
/// token text, without dropping either.
///
/// Regression for the F1 bug where `flush_byte_fallback_buffer()` re-synced
/// `prev_decoded_len` via `safe_emit_boundary(full_text)`, which advanced past
/// the trailing regular-token text and caused `emit_regular_token()` to see
/// `safe_len <= prev_decoded_len` and return None, silently dropping it.
#[test]
fn byte_fallback_incomplete_then_regular_token_does_not_lose_text() {
    // Token 5 = <0xE5> (first byte of a 3-byte CJK sequence, alone = incomplete)
    // Token 2 = "Hello"
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[5, 2]; // <0xE5> (incomplete) + "Hello"

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, prompt_ids, gen_ids);

    assert_eq!(
        generated, "\u{FFFD}Hello",
        "incomplete byte-fallback followed by a regular token must emit the regular text"
    );
    let concat: String = chunks.iter().cloned().collect();
    assert_eq!(
        concat, generated,
        "concatenated streaming chunks must equal the final generated text"
    );
}

/// Verify that `token_piece` correctly identifies byte-fallback tokens and
/// returns `None` for regular tokens in the stub tokenizer.
#[test]
fn token_piece_identifies_byte_fallback_tokens() {
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();

    // Byte-fallback tokens must return their piece strings.
    assert_eq!(tokenizer.token_piece(5), Some("<0xE5>".to_string()));
    assert_eq!(tokenizer.token_piece(6), Some("<0x8F>".to_string()));
    assert_eq!(tokenizer.token_piece(7), Some("<0xAB>".to_string()));
    assert_eq!(tokenizer.token_piece(12), Some("<0x61>".to_string()));

    // Regular tokens return their piece strings (not None).
    assert_eq!(tokenizer.token_piece(2), Some("Hello".to_string()));

    // Out-of-vocabulary ID returns None.
    assert_eq!(tokenizer.token_piece(9999), None);
}

/// `run_core_thread_or_abort` runs a non-panicking body to completion and lets
/// it observe its side effects, behaving as a transparent wrapper on the happy
/// path (issue #375).
///
/// The abort path (a panicking body calls `std::process::abort()`) cannot be
/// exercised in-process: `abort` terminates the whole test runner, so a panic
/// test here would kill every other test. It is verified manually instead, by
/// forcing a panic on a core worker thread in a release build and observing the
/// process exit with the "aborting to preserve fail-fast" log line rather than
/// a hung server. A subprocess re-exec harness could assert the abort, but the
/// added flakiness is not worth it for a one-line `process::abort`.
#[test]
fn run_core_thread_or_abort_runs_body_to_completion() {
    use std::cell::Cell;

    let ran = Cell::new(false);
    let mut returned = 0u32;
    run_core_thread_or_abort("test-core-thread", || {
        ran.set(true);
        returned = 7;
    });

    assert!(ran.get(), "the non-panicking body must run to completion");
    assert_eq!(
        returned, 7,
        "side effects of the body must be observable after the wrapper returns"
    );
}

/// A body that returns a recoverable `Err` must not abort: the value is
/// forwarded unchanged to the caller. Only a panic triggers the abort; a
/// normal return, including `Err`, is not a panic.
#[test]
fn run_core_thread_or_abort_forwards_err_without_aborting() {
    let result: Result<(), &str> = run_core_thread_or_abort("test-err-fwd", || Err("recoverable"));
    assert_eq!(
        result,
        Err("recoverable"),
        "a recoverable Err return must be forwarded, not treated as a panic"
    );
}
