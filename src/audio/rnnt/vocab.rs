// Copyright 2025-2026 Lablup Inc.
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

//! SentencePiece piece helpers for the RNNT transcript vocabulary.
//!
//! Port of `mlx_audio/stt/models/nemotron_asr/tokenizer.py`: the vocabulary is
//! the flat piece list (`rnnt_vocabulary` in `config.json`, or
//! `rnnt_tokenizer/vocab.json`). `<unk>`, `<pad>`, `<s>`, `</s>` and language
//! tags such as `<en-US>` are special and never reach the text; `▁` marks a
//! word boundary and becomes a space.

use std::path::Path;

const OTHER_SPECIAL: [&str; 4] = ["<unk>", "<pad>", "<s>", "</s>"];

/// `^<[a-z]{2,3}-[A-Za-z]{2,4}>$`.
pub fn is_lang_tag(piece: &str) -> bool {
    let Some(inner) = piece.strip_prefix('<').and_then(|p| p.strip_suffix('>')) else {
        return false;
    };
    let Some((lang, region)) = inner.split_once('-') else {
        return false;
    };
    (2..=3).contains(&lang.len())
        && lang.bytes().all(|b| b.is_ascii_lowercase())
        && (2..=4).contains(&region.len())
        && region.bytes().all(|b| b.is_ascii_alphabetic())
}

pub fn is_special_piece(piece: &str) -> bool {
    OTHER_SPECIAL.contains(&piece) || is_lang_tag(piece)
}

/// Out-of-range ids are not special (they are dropped by [`decode_pieces`]).
pub fn is_special_token(id: i32, vocabulary: &[String]) -> bool {
    usize::try_from(id)
        .ok()
        .and_then(|i| vocabulary.get(i))
        .is_some_and(|piece| is_special_piece(piece))
}

/// Join the pieces of `ids`, mapping `▁` to a space and dropping special
/// pieces, language tags and out-of-range ids. Not trimmed.
pub fn decode_pieces(ids: &[i32], vocabulary: &[String]) -> String {
    ids.iter()
        .filter_map(|&id| usize::try_from(id).ok().and_then(|i| vocabulary.get(i)))
        .filter(|piece| !is_special_piece(piece))
        .map(|piece| piece.replace('\u{2581}', " "))
        .collect()
}

/// Read `rnnt_tokenizer/vocab.json` (a JSON list of pieces).
pub fn load_vocabulary_json(path: &Path) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}
