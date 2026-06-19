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

//! Whisper-style byte-level BPE tokenizer plus the special-token vocabulary
//! that drives the transcribe/translate task tokens, language tags, and
//! decode-time token suppression.
//!
//! The byte-level BPE itself is loaded from the checkpoint's `tokenizer.json`;
//! this module resolves the special token ids on top of it and precomputes the
//! suppression sets.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

/// Language codes in their canonical Whisper order. The id of `<|{code}|>` is
/// looked up against the loaded tokenizer; codes absent from a given checkpoint
/// (e.g. English-only models) are simply skipped.
pub(crate) const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Special-token ids and suppression sets resolved against a checkpoint's BPE.
pub(crate) struct WhisperTokenizer {
    tokenizer: Tokenizer,
    /// `<|startoftranscript|>`
    pub sot: i32,
    /// `<|endoftext|>` (end of transcript)
    pub eot: i32,
    /// `<|transcribe|>`
    pub transcribe: i32,
    /// `<|translate|>` (absent on English-only checkpoints)
    pub translate: Option<i32>,
    /// `<|notimestamps|>`
    pub no_timestamps: i32,
    /// First timestamp token `<|0.00|>`; everything `>=` this is a timestamp.
    pub timestamp_begin: i32,
    /// First token produced by encoding a single space (suppressed on the first
    /// generated step). `None` if it could not be resolved.
    pub blank: Option<i32>,
    /// True when the checkpoint carries language tags (multilingual model).
    pub multilingual: bool,
    /// Resolved `(code, id)` pairs for every available language tag.
    pub language_ids: Vec<(&'static str, i32)>,
    /// Always-suppressed token ids (non-speech symbols plus the task/start
    /// tokens). The timestamp range is suppressed separately.
    pub suppress: Vec<i32>,
}

impl WhisperTokenizer {
    /// Load the byte-level BPE from `<dir>/tokenizer.json` and resolve the
    /// Whisper special tokens on top of it.
    pub(crate) fn from_dir(model_path: &Path) -> Result<Self> {
        let path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("Failed to load Whisper tokenizer from {path:?}: {e}"))?;
        Self::from_tokenizer(tokenizer)
    }

    /// Resolve the special tokens against an already-built tokenizer.
    pub(crate) fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let tid = |s: &str| tokenizer.token_to_id(s).map(|v| v as i32);

        let sot = tid("<|startoftranscript|>")
            .ok_or_else(|| anyhow!("not a Whisper tokenizer: missing <|startoftranscript|>"))?;
        let eot = tid("<|endoftext|>")
            .ok_or_else(|| anyhow!("not a Whisper tokenizer: missing <|endoftext|>"))?;
        let transcribe = tid("<|transcribe|>")
            .ok_or_else(|| anyhow!("not a Whisper tokenizer: missing <|transcribe|>"))?;
        let no_timestamps = tid("<|notimestamps|>")
            .ok_or_else(|| anyhow!("not a Whisper tokenizer: missing <|notimestamps|>"))?;
        let timestamp_begin = tid("<|0.00|>")
            .ok_or_else(|| anyhow!("not a Whisper tokenizer: missing <|0.00|> timestamp token"))?;
        let translate = tid("<|translate|>");

        let language_ids: Vec<(&'static str, i32)> = LANGUAGES
            .iter()
            .filter_map(|&code| tid(&format!("<|{code}|>")).map(|id| (code, id)))
            .collect();
        let multilingual = !language_ids.is_empty();

        let blank = encode_first(&tokenizer, " ");
        let non_speech = non_speech_tokens(&tokenizer);
        let suppress = build_suppress_set(
            sot,
            transcribe,
            translate,
            tid("<|startofprev|>"),
            tid("<|startoflm|>"),
            tid("<|nospeech|>").or_else(|| tid("<|nocaptions|>")),
            &non_speech,
        );

        Ok(Self {
            tokenizer,
            sot,
            eot,
            transcribe,
            translate,
            no_timestamps,
            timestamp_begin,
            blank,
            multilingual,
            language_ids,
            suppress,
        })
    }

    /// Resolve the `<|{code}|>` language token id, if present.
    pub(crate) fn language_token(&self, code: &str) -> Option<i32> {
        self.language_ids
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, id)| *id)
    }

    /// The initial decoder prompt for the given language/task. Always ends with
    /// `<|notimestamps|>` (timestamps are out of scope for this first port).
    pub(crate) fn initial_tokens(&self, language: Option<&str>, translate: bool) -> Vec<i32> {
        let mut tokens = vec![self.sot];
        if self.multilingual {
            if let Some(id) = language.and_then(|c| self.language_token(c)) {
                tokens.push(id);
            } else if let Some(id) = self.language_token("en") {
                tokens.push(id);
            }
            let task = if translate {
                self.translate.unwrap_or(self.transcribe)
            } else {
                self.transcribe
            };
            tokens.push(task);
        }
        tokens.push(self.no_timestamps);
        tokens
    }

    /// Decode generated token ids into text, dropping special and timestamp
    /// tokens (everything `>= eot`).
    pub(crate) fn decode_text(&self, ids: &[i32]) -> String {
        let text_ids: Vec<u32> = ids
            .iter()
            .filter(|&&t| t >= 0 && t < self.eot)
            .map(|&t| t as u32)
            .collect();
        self.tokenizer.decode(&text_ids, true).unwrap_or_default()
    }
}

/// Encode `text` and return its first token id (used for the blank/space token).
fn encode_first(tokenizer: &Tokenizer, text: &str) -> Option<i32> {
    tokenizer
        .encode(text, false)
        .ok()
        .and_then(|e| e.get_ids().first().map(|&id| id as i32))
}

/// Best-effort reconstruction of the non-speech symbol suppression set: encode
/// each punctuation/markup symbol and keep the ones that map to a single token,
/// plus the music note glyphs.
fn non_speech_tokens(tokenizer: &Tokenizer) -> Vec<i32> {
    let symbols = [
        "\"",
        "#",
        "(",
        ")",
        "*",
        "+",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "「",
        "」",
        "『",
        "』",
        "<<",
        ">>",
        "<<<",
        ">>>",
        "--",
        "---",
        "-(",
        "-[",
        "('",
        "(\"",
        "((",
        "))",
        "(((",
        ")))",
        "[[",
        "]]",
        "{{",
        "}}",
        "♪♪",
        "♪♪♪",
    ];
    let miscellaneous = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"];

    let mut result: BTreeSet<i32> = BTreeSet::new();
    if let Some(id) = encode_first(tokenizer, " -") {
        result.insert(id);
    }
    if let Some(id) = encode_first(tokenizer, " '") {
        result.insert(id);
    }
    for symbol in symbols.iter().chain(miscellaneous.iter()) {
        let is_misc = miscellaneous.contains(symbol);
        for candidate in [symbol.to_string(), format!(" {symbol}")] {
            if let Ok(encoding) = tokenizer.encode(candidate, false) {
                let ids = encoding.get_ids();
                if (ids.len() == 1 || is_misc) && !ids.is_empty() {
                    result.insert(ids[0] as i32);
                }
            }
        }
    }
    result.into_iter().collect()
}

/// Combine the non-speech symbols with the task/start tokens into a single
/// sorted, de-duplicated suppression list (the `"-1"` suppression policy).
fn build_suppress_set(
    sot: i32,
    transcribe: i32,
    translate: Option<i32>,
    sot_prev: Option<i32>,
    sot_lm: Option<i32>,
    no_speech: Option<i32>,
    non_speech: &[i32],
) -> Vec<i32> {
    let mut set: BTreeSet<i32> = non_speech.iter().copied().collect();
    set.insert(sot);
    set.insert(transcribe);
    for id in [translate, sot_prev, sot_lm, no_speech]
        .into_iter()
        .flatten()
    {
        set.insert(id);
    }
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic tokenizer that carries the Whisper special tokens as
    /// added tokens, plus a tiny regular vocabulary, so the special-token
    /// resolution can be exercised without a real checkpoint.
    fn synthetic_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 4, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 5, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 6, "content": "<|en|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 7, "content": "<|ko|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 8, "content": "<|transcribe|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 9, "content": "<|translate|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 10, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 11, "content": "<|0.00|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
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
                "vocab": {"a": 0, " ": 1, "b": 2},
                "merges": []
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("synthetic tokenizer")
    }

    #[test]
    fn language_table_is_complete_and_ordered() {
        assert_eq!(LANGUAGES.len(), 100, "Whisper ships 100 language tags");
        assert_eq!(LANGUAGES[0], "en");
        assert_eq!(LANGUAGES[5], "ko");
        assert!(LANGUAGES.contains(&"yue"));
        assert!(LANGUAGES.contains(&"ja"));
        // No duplicates.
        let unique: BTreeSet<&&str> = LANGUAGES.iter().collect();
        assert_eq!(unique.len(), LANGUAGES.len());
    }

    #[test]
    fn resolves_special_tokens() {
        // The tokenizers crate assigns added-token ids after the model vocab,
        // so resolve the expected ids from the tokenizer rather than hardcoding.
        let raw = synthetic_tokenizer();
        let want = |s: &str| raw.token_to_id(s).map(|v| v as i32);
        let sot = want("<|startoftranscript|>").unwrap();
        let eot = want("<|endoftext|>").unwrap();
        let transcribe = want("<|transcribe|>").unwrap();
        let translate = want("<|translate|>");
        let no_timestamps = want("<|notimestamps|>").unwrap();
        let timestamp_begin = want("<|0.00|>").unwrap();
        let en = want("<|en|>").unwrap();
        let ko = want("<|ko|>").unwrap();

        let tok = WhisperTokenizer::from_tokenizer(raw).expect("resolve");
        assert_eq!(tok.sot, sot);
        assert_eq!(tok.eot, eot);
        assert_eq!(tok.transcribe, transcribe);
        assert_eq!(tok.translate, translate);
        assert_eq!(tok.no_timestamps, no_timestamps);
        assert_eq!(tok.timestamp_begin, timestamp_begin);
        assert!(tok.multilingual);
        assert_eq!(tok.language_token("en"), Some(en));
        assert_eq!(tok.language_token("ko"), Some(ko));
        assert_eq!(tok.language_token("fr"), None);
        assert_ne!(tok.sot, tok.eot);
    }

    #[test]
    fn initial_tokens_follow_the_sot_sequence() {
        let raw = synthetic_tokenizer();
        let id = |s: &str| raw.token_to_id(s).unwrap() as i32;
        let (sot, en, ko, tr, tl, nts) = (
            id("<|startoftranscript|>"),
            id("<|en|>"),
            id("<|ko|>"),
            id("<|transcribe|>"),
            id("<|translate|>"),
            id("<|notimestamps|>"),
        );
        let tok = WhisperTokenizer::from_tokenizer(raw).expect("resolve");
        // transcribe, Korean hint
        assert_eq!(
            tok.initial_tokens(Some("ko"), false),
            vec![sot, ko, tr, nts]
        );
        // translate task
        assert_eq!(tok.initial_tokens(Some("ko"), true), vec![sot, ko, tl, nts]);
        // unknown hint falls back to English
        assert_eq!(
            tok.initial_tokens(Some("zz"), false),
            vec![sot, en, tr, nts]
        );
    }

    #[test]
    fn suppress_set_includes_task_and_start_tokens() {
        let raw = synthetic_tokenizer();
        let translate = raw.token_to_id("<|translate|>").unwrap() as i32;
        let tok = WhisperTokenizer::from_tokenizer(raw).expect("resolve");
        assert!(tok.suppress.contains(&tok.sot));
        assert!(tok.suppress.contains(&tok.transcribe));
        assert!(tok.suppress.contains(&translate));
        // sorted and de-duplicated
        let mut sorted = tok.suppress.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, tok.suppress);
    }

    #[test]
    fn build_suppress_set_is_sorted_union() {
        let out = build_suppress_set(5, 8, Some(9), Some(50), None, Some(60), &[3, 1, 3]);
        assert_eq!(out, vec![1, 3, 5, 8, 9, 50, 60]);
    }

    #[test]
    fn decode_text_drops_special_and_timestamp_tokens() {
        let tok = WhisperTokenizer::from_tokenizer(synthetic_tokenizer()).expect("resolve");
        // Mix text tokens (ids below eot: "a"=0, "b"=2) with specials/timestamps.
        let ids = vec![
            tok.sot,
            tok.transcribe,
            tok.no_timestamps,
            0,
            2,
            tok.timestamp_begin,
        ];
        let text = tok.decode_text(&ids);
        assert!(
            !text.contains("<|"),
            "special tokens must not survive decode: {text:?}"
        );
    }
}
