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

//! Raw per-token piece bytes, for `/tokenize`'s `with_pieces` mode (#1442).
//!
//! `llama-server` b10621 answers `with_pieces: true` with one object per token
//! carrying `id` and `piece`, where `piece` is a JSON **string** when the
//! token's bytes are valid UTF-8 and a JSON **array of byte values** when they
//! are not. A token that carries half of a multi-byte character is the normal
//! case for byte-level BPE, so the array form is not an edge case: it is how a
//! client reassembles text across token boundaries without guessing.
//!
//! Reproducing that needs the token's raw bytes, and the `tokenizers` crate's
//! `decode` cannot supply them: every decoder in it produces a `String`, so an
//! incomplete UTF-8 sequence comes back as U+FFFD and the original bytes are
//! gone. This module recovers them from the raw vocabulary entry instead.
//!
//! Upstream reference: `common_token_to_piece` in
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/common.cpp>

/// The GPT-2 byte-to-unicode alphabet used by every `ByteLevel` BPE
/// tokenizer (GPT-2, Llama 3, Qwen, Mistral's tekken, ...).
///
/// The printable ASCII range plus two Latin-1 runs map to themselves; the
/// remaining 68 byte values are displaced to the private range starting at
/// U+0100 so that no vocabulary entry contains a control character or a bare
/// space. `bytes_char()` in the reference implementation builds exactly this
/// table.
///
/// Upstream reference: `bytes_to_unicode()` in
/// <https://github.com/openai/gpt-2/blob/master/src/encoder.py>, mirrored by
/// `tokenizers`' own private `bytes_char()`.
pub(crate) fn byte_to_alphabet_char(byte: u8) -> char {
    // The three self-mapping runs, in the order the reference builds them.
    if is_direct_alphabet_byte(byte) {
        return byte as char;
    }
    // Everything else is displaced. `n` is the index of `byte` among the
    // displaced values in ascending order, which is what the reference's
    // running counter produces.
    let n = (0u16..byte as u16)
        .filter(|candidate| !is_direct_alphabet_byte(*candidate as u8))
        .count() as u32;
    char::from_u32(256 + n).expect("displaced byte-level code point is valid")
}

/// Whether a byte maps to itself in the byte-level alphabet.
fn is_direct_alphabet_byte(byte: u8) -> bool {
    (b'!'..=b'~').contains(&byte) || (0xA1..=0xAC).contains(&byte) || (0xAE..=0xFF).contains(&byte)
}

/// Reverse of [`byte_to_alphabet_char`], built once from the forward table so
/// the two cannot disagree.
fn alphabet_char_to_byte(ch: char) -> Option<u8> {
    static REVERSE: std::sync::OnceLock<std::collections::HashMap<char, u8>> =
        std::sync::OnceLock::new();
    REVERSE
        .get_or_init(|| {
            (0u16..=255)
                .map(|byte| (byte_to_alphabet_char(byte as u8), byte as u8))
                .collect()
        })
        .get(&ch)
        .copied()
}

/// The byte a `<0xXX>` byte-fallback vocabulary entry stands for.
///
/// SentencePiece byte-fallback models (Llama 2, Gemma, CodeLlama, ...) spell
/// one raw byte as the six-character token `<0x` + two uppercase hex digits +
/// `>`. Everything else returns `None`.
pub(crate) fn byte_fallback_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Decode a raw vocabulary entry as a byte-level alphabet string.
///
/// A character outside the alphabet contributes its own UTF-8 bytes, so a
/// vocabulary that is not byte-level (a WordPiece or plain SentencePiece one)
/// degrades to "the entry's own bytes" instead of producing nonsense.
pub(crate) fn byte_level_bytes(piece: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(piece.len());
    for ch in piece.chars() {
        match alphabet_char_to_byte(ch) {
            Some(byte) => out.push(byte),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

/// True when `decoded` carries a U+FFFD that `raw` did not, which is the
/// `tokenizers` crate reporting that it dropped bytes it could not turn into a
/// `char`.
pub(crate) fn lost_bytes(raw: &str, decoded: &str) -> bool {
    let decoded_replacements = decoded.matches('\u{FFFD}').count();
    let raw_replacements = raw.matches('\u{FFFD}').count();
    decoded_replacements > raw_replacements
}

/// The `piece` value for one token, in b10621's two-shape encoding: a JSON
/// string when the bytes are valid UTF-8, an array of byte values when they
/// are not.
pub fn piece_json(bytes: Vec<u8>) -> serde_json::Value {
    match String::from_utf8(bytes) {
        Ok(text) => serde_json::Value::String(text),
        Err(err) => serde_json::Value::Array(
            err.into_bytes()
                .into_iter()
                .map(|b| serde_json::Value::from(b as u64))
                .collect(),
        ),
    }
}

#[cfg(test)]
#[path = "pieces_tests.rs"]
mod pieces_tests;
