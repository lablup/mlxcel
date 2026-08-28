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

//! Byte-level alphabet and piece-encoding tests (#1442).

use super::*;

#[test]
fn the_byte_level_alphabet_is_a_bijection_over_all_256_bytes() {
    let mut seen = std::collections::HashSet::new();
    for byte in 0u16..=255 {
        let byte = byte as u8;
        let ch = byte_to_alphabet_char(byte);
        assert!(seen.insert(ch), "byte {byte} reused code point {ch:?}");
        assert_eq!(
            alphabet_char_to_byte(ch),
            Some(byte),
            "byte {byte} did not round trip through {ch:?}"
        );
    }
    assert_eq!(seen.len(), 256);
}

#[test]
fn the_alphabet_matches_the_reference_table_at_its_boundaries() {
    // Space is the first displaced value, so it is U+0100; the reference's
    // table is what every ByteLevel vocabulary on disk was written against.
    assert_eq!(byte_to_alphabet_char(b' '), '\u{0120}');
    assert_eq!(byte_to_alphabet_char(b'!'), '!');
    assert_eq!(byte_to_alphabet_char(b'~'), '~');
    assert_eq!(byte_to_alphabet_char(0x00), '\u{0100}');
    assert_eq!(byte_to_alphabet_char(0xA1), '\u{00A1}');
    assert_eq!(byte_to_alphabet_char(0xAD), '\u{0143}');
    assert_eq!(byte_to_alphabet_char(0xFF), '\u{00FF}');
}

#[test]
fn a_character_outside_the_alphabet_contributes_its_own_utf8() {
    // U+4E2D is neither a self-mapping byte nor a displaced code point, so a
    // non-byte-level vocabulary entry degrades to its own bytes.
    assert_eq!(byte_level_bytes("\u{4E2D}"), "\u{4E2D}".as_bytes());
}

#[test]
fn byte_level_bytes_recovers_a_split_multibyte_character() {
    // "中" is E4 B8 AD. A ByteLevel vocabulary spells the first two bytes as
    // the displaced code points for E4 and B8.
    let raw: String = [byte_to_alphabet_char(0xE4), byte_to_alphabet_char(0xB8)]
        .into_iter()
        .collect();
    assert_eq!(byte_level_bytes(&raw), vec![0xE4, 0xB8]);
}

#[test]
fn byte_fallback_values_parse_only_the_exact_spelling() {
    assert_eq!(byte_fallback_value("<0x0A>"), Some(0x0A));
    assert_eq!(byte_fallback_value("<0xff>"), Some(0xFF));
    assert_eq!(byte_fallback_value("<0x0A"), None);
    assert_eq!(byte_fallback_value("0x0A>"), None);
    assert_eq!(byte_fallback_value("<0xG0>"), None);
    assert_eq!(byte_fallback_value("<0x123>"), None);
    assert_eq!(byte_fallback_value("hello"), None);
}

#[test]
fn piece_json_is_a_string_for_valid_utf8_and_an_array_otherwise() {
    assert_eq!(
        piece_json("Hello".as_bytes().to_vec()),
        serde_json::json!("Hello")
    );
    assert_eq!(piece_json(Vec::new()), serde_json::json!(""));
    // The leading two bytes of "中": a valid prefix, not a valid string.
    assert_eq!(
        piece_json(vec![0xE4, 0xB8]),
        serde_json::json!([228, 184]),
        "an incomplete UTF-8 sequence must come back as byte values"
    );
}

#[test]
fn lost_bytes_only_fires_when_the_decoder_added_a_replacement() {
    assert!(lost_bytes("\u{0120}\u{00E4}", "\u{FFFD}"));
    assert!(!lost_bytes("Hello", "Hello"));
    // A vocabulary entry that genuinely contains U+FFFD is not a loss.
    assert!(!lost_bytes("\u{FFFD}", "\u{FFFD}"));
}
