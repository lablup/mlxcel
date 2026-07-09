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
    decode_request_images_with_limits, merge_config_stop_tokens, resolve_end_of_turn_token_id,
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
    // Model a correct streaming client: the end-of-stream flush can carry real
    // text the last token held back (complete text plus a trailing incomplete
    // byte), so its return is part of the streamed output, not dropped.
    if let Some(tail) = state.flush(tokenizer) {
        chunks.push(tail);
    }
    let start = Instant::now();
    let result = state.finish_with_cache(start, prompt_ids.len(), usize::MAX, 0);
    (chunks, result.text)
}

/// Build a HuggingFace tokenizer whose vocab contains every byte-fallback token
/// `<0x00>`..`<0xFF>` (token id == byte value) plus a couple of regular ASCII
/// word tokens, with a `ByteFallback` decoder. This lets a test drive the
/// incremental detokenizer with the exact byte sequence of any UTF-8 string, so
/// byte-exactness can be checked against a one-shot decode of the same ids.
fn stub_all_byte_fallback() -> MlxcelTokenizer {
    let mut vocab_entries: Vec<String> = (0u16..=255)
        .map(|b| format!("\"<0x{b:02X}>\": {b}"))
        .collect();
    // Regular word tokens after the 256 byte ids, exercising the mixed
    // regular-piece + byte-fallback path.
    vocab_entries.push("\"Hello\": 256".to_string());
    vocab_entries.push("\"world\": 257".to_string());
    let vocab = vocab_entries.join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {{"type": "ByteFallback"}},
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": {{{vocab}}},
                "merges": []
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("failed to build all-byte-fallback stub tokenizer");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// The `<0xXX>` token ids for the UTF-8 bytes of `s` in [`stub_all_byte_fallback`]
/// (token id == byte value).
fn byte_fallback_ids(s: &str) -> Vec<i32> {
    s.bytes().map(|b| b as i32).collect()
}

/// Core byte-exactness invariant for the incremental detokenizer: streaming a
/// token sequence one token at a time and concatenating the emitted chunks (plus
/// the end-of-stream flush) must reproduce a one-shot `tokenizer.decode` of the
/// whole sequence exactly. `prompt_ids` are treated as already-emitted context
/// and excluded from the streamed output, matching production use.
fn assert_stream_matches_oneshot(tokenizer: &MlxcelTokenizer, prompt_ids: &[i32], gen_ids: &[i32]) {
    let mut all_ids: Vec<u32> = prompt_ids.iter().map(|&x| x as u32).collect();
    all_ids.extend(gen_ids.iter().map(|&x| x as u32));
    let full = tokenizer.decode(&all_ids, false).unwrap_or_default();
    let prompt_decoded = tokenizer
        .decode(
            &prompt_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap_or_default();
    let expected = &full[prompt_decoded.len()..];

    let (chunks, generated) = simulate_byte_fallback_stream(tokenizer, prompt_ids, gen_ids);
    let concatenated: String = chunks.concat();

    assert_eq!(
        concatenated, generated,
        "streamed chunks must equal the finished text"
    );
    assert_eq!(
        generated,
        expected,
        "incremental detokenization must be byte-identical to a one-shot decode \
         (prompt_ids={prompt_ids:?}, gen_ids len={})",
        gen_ids.len()
    );
}

/// Korean (Hangul, 3 bytes/char) streamed byte-by-byte must reconstruct exactly
/// and match a one-shot decode; no chunk may contain a partial (U+FFFD) char.
#[test]
fn incremental_detok_korean_matches_oneshot() {
    let tokenizer = stub_all_byte_fallback();
    let text = "안녕하세요 세계";
    let gen_ids = byte_fallback_ids(text);
    assert_stream_matches_oneshot(&tokenizer, &[], &gen_ids);

    let (chunks, generated) = simulate_byte_fallback_stream(&tokenizer, &[], &gen_ids);
    assert_eq!(generated, text);
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "no streamed chunk may contain a replacement char: {chunk:?}"
        );
    }
}

/// Mixed ASCII, CJK, emoji, and Korean streamed byte-by-byte must be
/// byte-identical to a one-shot decode.
#[test]
fn incremental_detok_mixed_scripts_match_oneshot() {
    let tokenizer = stub_all_byte_fallback();
    let text = "ab叫cd😀ef 안녕 gh€ij";
    assert_stream_matches_oneshot(&tokenizer, &[], &byte_fallback_ids(text));
}

/// A regular word piece adjacent to multibyte byte-fallback runs (the common
/// real-model shape) streams byte-exactly.
#[test]
fn incremental_detok_regular_piece_then_multibyte_matches_oneshot() {
    let tokenizer = stub_all_byte_fallback();
    // "Hello" (id 256), "world" (id 257), then a CJK char and an emoji as bytes.
    let mut gen_ids = vec![256, 257];
    gen_ids.extend(byte_fallback_ids("叫😀"));
    assert_stream_matches_oneshot(&tokenizer, &[], &gen_ids);
}

/// A non-empty prompt is treated as already-emitted context: the streamed
/// output starts after the prompt boundary and stays byte-exact across the
/// prompt/generation seam even when the generation begins with a multibyte char.
#[test]
fn incremental_detok_with_prompt_context_matches_oneshot() {
    let tokenizer = stub_all_byte_fallback();
    let prompt_ids = byte_fallback_ids("prompt: ");
    let gen_ids = byte_fallback_ids("안녕 world 叫");
    assert_stream_matches_oneshot(&tokenizer, &prompt_ids, &gen_ids);
}

/// Stopping the stream mid-multibyte (as a stop-string truncation would) must
/// never leave a partial character in the streamed chunks; the incomplete tail
/// only surfaces (as U+FFFD) after the end-of-stream flush, matching a one-shot
/// decode of the truncated ids.
#[test]
fn incremental_detok_truncated_mid_multibyte_holds_partial() {
    let tokenizer = stub_all_byte_fallback();
    // A regular word piece "Hello" (id 256, the shape real text takes) followed
    // by the first two of the three bytes of "가" (EA B0 80) — i.e. a stop-string
    // truncation that lands in the middle of a multibyte character.
    let mut truncated = vec![256];
    truncated.extend_from_slice(&byte_fallback_ids("가")[..2]);

    let mut state = StreamingDecodeState::new(&tokenizer, &[]);
    let mut chunks = Vec::new();
    for &tok in &truncated {
        if let Some(chunk) = state.on_token(tok, &tokenizer) {
            chunks.push(chunk);
        }
    }
    // Before flush, only the complete "Hello" may have been emitted; the
    // incomplete "가" bytes must be held back — never streamed as a partial char.
    let streamed: String = chunks.concat();
    assert_eq!(streamed, "Hello", "partial multibyte must not be streamed");
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "no streamed chunk may contain a replacement char: {chunk:?}"
        );
    }

    // After flush the held partial surfaces as replacement char(s). The flush
    // return is part of the stream, so a correct client accumulates it; the full
    // streamed text and the non-streaming result.text are both byte-identical to
    // a one-shot decode of exactly the truncated ids.
    let flushed = state.flush(&tokenizer);
    let full_streamed = format!("{streamed}{}", flushed.clone().unwrap_or_default());
    let result = state.finish_with_cache(std::time::Instant::now(), 0, usize::MAX, 0);
    let expected = tokenizer
        .decode(
            &truncated.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap_or_default();
    assert!(result.text.starts_with("Hello"));
    assert_eq!(result.text, expected);
    assert_eq!(
        full_streamed, result.text,
        "streamed deltas plus the flush return must equal the non-streaming text"
    );
}

/// GPT-2/ByteLevel byte -> visible-char map (the map the ByteLevel decoder
/// inverts). Lets a test build a vocab token whose decoded bytes end in the
/// middle of a multibyte UTF-8 character. Unlike byte-fallback tokenizers, a
/// ByteLevel tokenizer (Qwen, GPT-2, ...) can carry complete text plus an
/// incomplete trailing byte inside a single token.
fn bytelevel_byte_to_char() -> [char; 256] {
    let mut is_direct = [false; 256];
    let mut map = ['\0'; 256];
    for (lo, hi) in [(0x21u32, 0x7E), (0xA1, 0xAC), (0xAE, 0xFF)] {
        for b in lo..=hi {
            is_direct[b as usize] = true;
            map[b as usize] = char::from_u32(b).unwrap();
        }
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !is_direct[b] {
            map[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    map
}

fn bytelevel_token(bytes: &[u8], map: &[char; 256]) -> String {
    bytes.iter().map(|&b| map[b as usize]).collect()
}

/// A HuggingFace ByteLevel tokenizer whose token 0 decodes to "abc" plus the
/// first byte of the 3-byte character "가" (i.e. a single token carrying
/// complete text and an incomplete UTF-8 tail), and tokens 1/2 are the two
/// remaining bytes of "가". This is the token shape that produces the
/// "complete text held on a trailing incomplete byte" case the flush-return
/// path must recover.
fn stub_bytelevel_split_char() -> MlxcelTokenizer {
    let map = bytelevel_byte_to_char();
    let ga = "가".as_bytes(); // [0xEA, 0xB0, 0x80]
    let t0 = bytelevel_token(&[b'a', b'b', b'c', ga[0]], &map);
    let t1 = bytelevel_token(&[ga[1]], &map);
    let t2 = bytelevel_token(&[ga[2]], &map);
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}},
            "post_processor": null,
            "decoder": {{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}},
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {{"{t0}": 0, "{t1}": 1, "{t2}": 2}},
                "merges": []
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("failed to build ByteLevel split-char stub tokenizer");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// Regression for the flush-drop bug (PR #713 review): a single final token
/// decodes to complete text ("abc") plus a trailing incomplete UTF-8 byte, then
/// generation stops. `on_token` correctly holds the whole window (it ends in
/// U+FFFD), so the complete "abc" is only recoverable from the `flush` return.
/// A streaming client that ignores the return would lose "abc" while the
/// non-streaming `result.text` keeps it. The streamed deltas plus the flush
/// return must equal both `result.text` and a one-shot decode.
#[test]
fn incremental_detok_flush_returns_held_complete_text() {
    let tokenizer = stub_bytelevel_split_char();
    // Precondition: the stub really produces "complete + incomplete tail".
    assert_eq!(
        tokenizer.decode(&[0], false).unwrap_or_default(),
        "abc\u{FFFD}",
        "ByteLevel stub token 0 must decode to 'abc' + one replacement char"
    );

    let mut state = StreamingDecodeState::new(&tokenizer, &[]);
    let held = state.on_token(0, &tokenizer);
    assert!(
        held.is_none(),
        "a window ending in an incomplete byte must be held, not streamed: {held:?}"
    );

    // Generation stops here; flush must return the held text so it is streamed.
    let flushed = state.flush(&tokenizer);
    assert_eq!(
        flushed.as_deref(),
        Some("abc\u{FFFD}"),
        "flush must return the held complete text plus the U+FFFD tail"
    );

    let streamed: String = flushed.unwrap_or_default();
    let result = state.finish_with_cache(std::time::Instant::now(), 0, usize::MAX, 0);
    assert_eq!(
        streamed, result.text,
        "streamed text must equal result.text"
    );
    assert_eq!(
        result.text,
        tokenizer.decode(&[0], false).unwrap_or_default(),
        "result.text must equal a one-shot decode"
    );
}

/// The mixed complete-plus-incomplete token followed by the completing bytes:
/// mid-stream (generation continues) the completed text is emitted by `on_token`
/// and `flush` has nothing to add. Byte-exact against a one-shot decode.
#[test]
fn incremental_detok_bytelevel_split_char_completes_mid_stream() {
    let tokenizer = stub_bytelevel_split_char();
    assert_eq!(
        tokenizer.decode(&[0, 1, 2], false).unwrap_or_default(),
        "abc가",
        "ByteLevel stub tokens [0,1,2] must decode to 'abc가'"
    );

    let mut state = StreamingDecodeState::new(&tokenizer, &[]);
    let mut chunks = Vec::new();
    for tok in [0, 1, 2] {
        if let Some(c) = state.on_token(tok, &tokenizer) {
            chunks.push(c);
        }
    }
    if let Some(tail) = state.flush(&tokenizer) {
        chunks.push(tail);
    }
    let concatenated: String = chunks.concat();
    let result = state.finish_with_cache(std::time::Instant::now(), 0, usize::MAX, 0);
    assert_eq!(concatenated, "abc가");
    assert_eq!(result.text, "abc가");
    for chunk in &chunks {
        assert!(
            !chunk.contains('\u{FFFD}'),
            "no chunk may contain a replacement char once the char completes: {chunk:?}"
        );
    }
}

/// Fuzz: many random valid UTF-8 strings streamed byte-by-byte must each be
/// byte-identical to a one-shot decode. Uses a deterministic xorshift PRNG so
/// failures reproduce.
#[test]
fn incremental_detok_fuzz_matches_oneshot() {
    let tokenizer = stub_all_byte_fallback();
    // A pool of characters spanning 1/2/3/4-byte UTF-8 lengths plus ASCII
    // whitespace so pieces land on and off char boundaries.
    let pool: Vec<char> = "aZ9 \n.叫가한€😀🎉ßé中".chars().collect();

    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        // xorshift64
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for _ in 0..300 {
        let len = (next() % 40) as usize;
        let mut s = String::new();
        for _ in 0..len {
            let c = pool[(next() as usize) % pool.len()];
            s.push(c);
        }
        assert_stream_matches_oneshot(&tokenizer, &[], &byte_fallback_ids(&s));
    }
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
/// The U+FFFD tail is streamed exactly once, as the `flush` return (the final
/// end-of-stream delta), never as a mid-stream `on_token` chunk. A streaming
/// client that accumulates deltas therefore ends up byte-identical to the
/// non-streaming `result.text` (issue #633 flush-return fix).
#[test]
fn byte_fallback_incomplete_sequence_flushed_at_end_of_stream() {
    // Token 5 = <0xE5>: alone, this is an incomplete UTF-8 sequence.
    let tokenizer = MlxcelTokenizer::stub_with_byte_fallback();
    let prompt_ids: &[i32] = &[0]; // <BOS>
    let gen_ids: &[i32] = &[5]; // <0xE5> alone (incomplete)

    // Stream without the flush merged, to inspect the mid-stream on_token chunks
    // separately from the final flush delta.
    let mut state = StreamingDecodeState::new(&tokenizer, prompt_ids);
    let mut on_token_chunks = Vec::new();
    for &tok in gen_ids {
        if let Some(c) = state.on_token(tok, &tokenizer) {
            on_token_chunks.push(c);
        }
    }
    // Mid-stream, an incomplete sequence yields no chunk (nothing is emitted
    // prematurely with a replacement char).
    assert!(
        on_token_chunks.is_empty(),
        "incomplete byte-fallback must not emit a mid-stream chunk: {on_token_chunks:?}"
    );

    // The flush return carries the U+FFFD tail exactly once, so a streaming
    // client receives it (previously it was appended to result.text but never
    // streamed, dropping it from the client's accumulated text).
    let flushed = state.flush(&tokenizer);
    assert_eq!(flushed.as_deref(), Some("\u{FFFD}"));
    let result =
        state.finish_with_cache(std::time::Instant::now(), prompt_ids.len(), usize::MAX, 0);
    assert_eq!(result.text, "\u{FFFD}");
    assert_eq!(
        flushed.unwrap_or_default(),
        result.text,
        "the streamed flush delta must equal the non-streaming result text"
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

// ── resolve_end_of_turn_token_id tests ───────────────────────────────────────

/// Build a minimal HuggingFace tokenizer stub that has exactly one added
/// special token acting as the end-of-turn marker.  The model vocabulary
/// intentionally contains only ASCII letter tokens so that encoding either
/// Gemma EOT candidate through the BPE path always produces multiple (or
/// zero) pieces, preventing a false-positive from the SentencePiece/Tiktoken
/// fallback branch.
fn stub_eot_tokenizer(eot_content: &str, eot_id: u32) -> MlxcelTokenizer {
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {{"id": {eot_id}, "content": "{eot_content}", "single_word": false,
                  "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {{"a": 0, "b": 1, "{eot_content}": {eot_id}}},
                "merges": []
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("failed to build stub EOT tokenizer");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// Build a minimal HuggingFace tokenizer stub with no EOT-style added tokens
/// and a small ASCII vocabulary.  Used to exercise the `None` path in
/// `resolve_end_of_turn_token_id` for models whose tokenizer carries neither
/// `<end_of_turn>` nor `<turn|>`.
fn stub_tokenizer_no_eot() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
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
            "vocab": {"a": 0, "b": 1},
            "merges": []
        }
    }"#;
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("failed to build stub no-EOT tokenizer");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// Gemma 4 uses `"<turn|>"` (id 106) as its end-of-turn marker instead of
/// `"<end_of_turn>"`.  Before PR #440 `resolve_end_of_turn_token_id` only
/// looked up `"<end_of_turn>"` and returned `None` for Gemma 4 tokenizers,
/// causing the audio block to fall back to a model-turn insertion and
/// producing 0 output tokens.  This test guards that regression: a tokenizer
/// that has `"<turn|>"` but not `"<end_of_turn>"` must return `Some(106)`.
#[test]
fn resolve_end_of_turn_id_handles_gemma4_turn_marker() {
    // Gemma 4 tokenizer: "<turn|>" = id 106, no "<end_of_turn>" anywhere.
    let tokenizer = stub_eot_tokenizer("<turn|>", 106);
    assert_eq!(
        resolve_end_of_turn_token_id(&tokenizer),
        Some(106),
        "Gemma 4 tokenizer must resolve <turn|> (id 106) as the end-of-turn id"
    );
}

/// Gemma 2 and Gemma 3 tokenizers carry `"<end_of_turn>"` as the EOT marker.
/// `resolve_end_of_turn_token_id` must find it via `token_to_id` before even
/// reaching the `"<turn|>"` candidate, so the returned id matches the token's
/// entry in the added-tokens table.
#[test]
fn resolve_end_of_turn_id_handles_gemma23_end_of_turn_marker() {
    // Gemma 2/3 tokenizer: "<end_of_turn>" = id 107.
    let tokenizer = stub_eot_tokenizer("<end_of_turn>", 107);
    assert_eq!(
        resolve_end_of_turn_token_id(&tokenizer),
        Some(107),
        "Gemma 2/3 tokenizer must resolve <end_of_turn> (id 107) as the end-of-turn id"
    );
}

/// For models whose tokenizer contains neither `"<end_of_turn>"` nor
/// `"<turn|>"` as a registered token, `resolve_end_of_turn_token_id` must
/// return `None`.  The caller keeps its own fallback (insert before the last
/// token) in that case.
#[test]
fn resolve_end_of_turn_id_returns_none_when_no_marker_present() {
    // Minimal tokenizer with no EOT-style tokens: neither candidate is in the
    // vocabulary or added-tokens table, and encoding either marker through the
    // BPE path produces zero or multiple pieces, so the fallback also yields None.
    let tokenizer = stub_tokenizer_no_eot();
    assert_eq!(
        resolve_end_of_turn_token_id(&tokenizer),
        None,
        "a tokenizer without <end_of_turn> or <turn|> must return None"
    );
}

/// Nemotron-H Nano Omni uses a ChatML template that closes every turn with
/// `"<|im_end|>"` (id 151 in the released 30B checkpoint).  A tokenizer that
/// registers only `"<|im_end|>"` and no Gemma marker must resolve to that id,
/// because `"<|im_end|>"` is the last candidate in `EOT_CANDIDATES` and is
/// reached after both Gemma entries return `None`.
#[test]
fn resolve_end_of_turn_id_handles_chatml_im_end_marker() {
    // Nemotron tokenizer stub: "<|im_end|>" = id 151, no Gemma markers.
    let tokenizer = stub_eot_tokenizer("<|im_end|>", 151);
    assert_eq!(
        resolve_end_of_turn_token_id(&tokenizer),
        Some(151),
        "Nemotron tokenizer must resolve <|im_end|> (id 151) as the end-of-turn id"
    );
}

/// Build a minimal HuggingFace tokenizer stub with exactly two added special
/// tokens.  Used to verify candidate ordering in
/// `resolve_end_of_turn_token_id`: when the vocabulary contains both a Gemma
/// marker and `"<|im_end|>"`, the function must return the first match in
/// `EOT_CANDIDATES`, which is the Gemma one.
fn stub_two_eot_tokenizer(
    first_content: &str,
    first_id: u32,
    second_content: &str,
    second_id: u32,
) -> MlxcelTokenizer {
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {{"id": {first_id}, "content": "{first_content}", "single_word": false,
                  "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
                {{"id": {second_id}, "content": "{second_content}", "single_word": false,
                  "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {{"a": 0, "b": 1, "{first_content}": {first_id}, "{second_content}": {second_id}}},
                "merges": []
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .expect("failed to build stub two-token EOT tokenizer");
    MlxcelTokenizer::HuggingFace(tokenizer)
}

/// When a tokenizer contains both `"<end_of_turn>"` (Gemma) and `"<|im_end|>"`
/// (ChatML), `resolve_end_of_turn_token_id` must return the Gemma marker
/// because it appears first in `EOT_CANDIDATES`.  This guards against a future
/// ordering change that would regress the Gemma audio path on any tokenizer
/// that also happens to define `"<|im_end|>"`.
#[test]
fn resolve_end_of_turn_id_prefers_gemma_over_chatml() {
    // Tokenizer with both markers: Gemma "<end_of_turn>" = id 107, ChatML "<|im_end|>" = id 151.
    let tokenizer = stub_two_eot_tokenizer("<end_of_turn>", 107, "<|im_end|>", 151);
    assert_eq!(
        resolve_end_of_turn_token_id(&tokenizer),
        Some(107),
        "<end_of_turn> must win over <|im_end|> when both are present (candidate order)"
    );
}
