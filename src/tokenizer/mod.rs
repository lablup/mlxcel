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

pub mod fim;
pub mod pieces;
mod thinking;
mod tiktoken;

use anyhow::Result;
use hf_hub::api::sync::Api;
use sentencepiece::SentencePieceProcessor;
use std::collections::HashMap;
use std::path::Path;

pub use fim::{FimToken, FimTokens, FimTriple};
pub use thinking::{ThinkingMarkers, find_subseq, rfind_subseq};
pub use tiktoken::TiktokenTokenizer;

/// Unified tokenizer supporting HuggingFace (tokenizer.json), SentencePiece (tokenizer.model),
/// and Tiktoken (.tiktoken) formats
pub enum MlxcelTokenizer {
    HuggingFace(tokenizers::Tokenizer),
    SentencePiece(SentencePieceTokenizer),
    Tiktoken(TiktokenTokenizer),
}

pub struct SentencePieceTokenizer {
    processor: SentencePieceProcessor,
    special_token_to_id: HashMap<String, u32>,
    id_to_special_token: HashMap<u32, String>,
    /// Special tokens sorted by length descending for greedy longest-match-first splitting
    special_tokens_sorted: Vec<(String, u32)>,
    /// Every `added_tokens_decoder` entry (special or not) by id. Added tokens
    /// live OUTSIDE the SentencePiece vocab, so `decode_piece_ids` errors
    /// "Out of range" on them; decode must map them from this table instead.
    /// Non-special added tokens (e.g. ERNIE's `<|IMAGE_PLACEHOLDER|>`, marked
    /// `special: false`) are real text per HF semantics and are never skipped.
    added_token_contents: HashMap<u32, String>,
    bos_id: Option<u32>,
    add_bos: bool,
    /// Byte-fallback token ids, resolved on first use. See
    /// [`SentencePieceTokenizer::byte_fallback_ids`].
    byte_fallback_ids: std::sync::OnceLock<HashMap<u32, u8>>,
}

impl MlxcelTokenizer {
    /// The BOS token id this tokenizer prepends when encoding with special
    /// tokens, or `None` when it prepends none (#1472).
    ///
    /// Used by the batch scheduler's context-retention arithmetic, which
    /// mirrors b10621's `if (add_bos_token) n_keep += 1` so "keep N prompt
    /// tokens" keeps N tokens of the operator's prompt and the BOS on top.
    ///
    /// The HuggingFace backend detects the prefix empirically rather than
    /// from configuration: the id, if any, that both an empty and a
    /// non-empty encoding start with is the prepended special. A
    /// post-processor that only appends (an EOS-only template) yields
    /// different first ids and correctly resolves to `None`.
    pub fn bos_token_id(&self) -> Option<u32> {
        match self {
            Self::SentencePiece(sp) => {
                if sp.add_bos {
                    sp.bos_id
                } else {
                    None
                }
            }
            Self::HuggingFace(tokenizer) => {
                let first_of = |text: &str| {
                    tokenizer
                        .encode(text, true)
                        .ok()
                        .and_then(|encoding| encoding.get_ids().first().copied())
                };
                match (first_of(""), first_of("a")) {
                    (Some(empty_first), Some(probe_first)) if empty_first == probe_first => {
                        Some(empty_first)
                    }
                    _ => None,
                }
            }
            Self::Tiktoken(_) => None,
        }
    }

    /// Create a stub tokenizer for unit tests.
    ///
    /// The stub returns empty/identity results; it exists so that types like
    /// `StreamingDecodeState` can be constructed without loading a real model.
    #[cfg(test)]
    pub(crate) fn stub() -> Self {
        // Build a minimal HuggingFace tokenizer with a single-character
        // alphabet so encode/decode never panic.
        use tokenizers::models::bpe::BPE;
        let model = BPE::default();
        let tokenizer = tokenizers::Tokenizer::new(model);
        Self::HuggingFace(tokenizer)
    }

    /// Create a minimal tokenizer with byte-fallback support for regression tests
    /// The vocabulary includes:
    ///
    /// - Tokens 0/1: `<BOS>` / `<EOS>` (special)
    /// - Token 2: `Hello` (regular ASCII)
    /// - Token 5/6/7: `<0xE5>` / `<0x8F>` / `<0xAB>` → "叫" (CJK, 3 bytes)
    /// - Token 8/9/10/11: `<0xF0>` / `<0x9F>` / `<0x98>` / `<0x80>` → "😀" (emoji, 4 bytes)
    /// - Token 12: `<0x61>` → 'a' (single-byte ASCII via byte-fallback)
    ///
    /// The decoder is set to `ByteFallback` so that sequences of `<0xXX>` tokens
    /// are assembled into bytes and decoded as UTF-8.
    #[cfg(test)]
    pub(crate) fn stub_with_byte_fallback() -> Self {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<BOS>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<EOS>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type": "ByteFallback"},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": {
                    "<BOS>": 0,
                    "<EOS>": 1,
                    "Hello": 2,
                    "▁World": 3,
                    " ": 4,
                    "<0xE5>": 5,
                    "<0x8F>": 6,
                    "<0xAB>": 7,
                    "<0xF0>": 8,
                    "<0x9F>": 9,
                    "<0x98>": 10,
                    "<0x80>": 11,
                    "<0x61>": 12
                },
                "merges": []
            }
        }"#;
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .expect("Failed to build byte-fallback test tokenizer");
        Self::HuggingFace(tokenizer)
    }

    /// Build a tokenizer whose vocab holds every byte-fallback token
    /// `<0x00>`..`<0xFF>` (token id == byte value) plus two regular ASCII word
    /// tokens, with a `ByteFallback` decoder.
    ///
    /// This lets a test drive the incremental detokenizer with the exact byte
    /// sequence of any UTF-8 string, so streaming behavior can be checked at
    /// every token boundary against a one-shot decode of the same ids. Shared by
    /// the detokenizer regression tests and the stop-sequence tests (#1466),
    /// which need the same byte-level control over the decoded stream.
    #[cfg(test)]
    pub(crate) fn stub_all_byte_fallback() -> Self {
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
        Self::HuggingFace(tokenizer)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => {
                let encoding = t
                    .encode(text, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::SentencePiece(t) => t.encode(text, add_special_tokens),
            Self::Tiktoken(t) => t.encode(text, add_special_tokens),
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t
                .decode(ids, skip_special_tokens)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e)),
            Self::SentencePiece(t) => t.decode(ids, skip_special_tokens),
            Self::Tiktoken(t) => t.decode(ids, skip_special_tokens),
        }
    }

    /// Returns the underlying HuggingFace `tokenizers::Tokenizer` when this
    /// instance was constructed from a `tokenizer.json` file.
    ///
    /// `None` for SentencePiece or Tiktoken tokenizers. Used by Axis B
    /// language steering to feed the tokenizer vocabulary into the
    /// [`mlxcel_core::lang_analyzer`] classifier.
    pub fn hf_tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        match self {
            Self::HuggingFace(t) => Some(t),
            Self::SentencePiece(_) | Self::Tiktoken(_) => None,
        }
    }

    /// Look up the raw token string for a given token ID, without applying any
    /// decoder transformations. Returns `None` if the ID is out of vocabulary.
    ///
    /// General vocab-lookup helper. Since issue #633 the streaming detokenizer
    /// no longer inspects individual pieces to detect byte-fallback tokens
    /// (`<0xXX>`): `StreamingDecodeState` holds incomplete UTF-8 by re-decoding a
    /// bounded token window, so this is off the detok hot path.
    ///
    /// Used by: model_worker_tests (byte-fallback token identification)
    pub fn token_piece(&self, id: u32) -> Option<String> {
        match self {
            Self::HuggingFace(t) => t.id_to_token(id),
            // SentencePiece byte-fallback tokens appear directly as <0xXX> in
            // the decoded output; the incremental decoder handles them via the
            // windowed re-decode path rather than per-piece inspection.
            Self::SentencePiece(_) | Self::Tiktoken(_) => None,
        }
    }

    /// Encode with `llama-server`'s two independent switches (#1442).
    ///
    /// `add_special` is the BOS/EOS post-processor, exactly as
    /// [`Self::encode`] takes it. `parse_special` is the separate question of
    /// whether a special token written out in the *input text* is recognized
    /// as that token or tokenized as ordinary characters. b10621 defaults it
    /// to `true` on `/tokenize`, which is what [`Self::encode`] already does,
    /// so only an explicit `parse_special: false` takes the second path.
    ///
    /// Upstream reference: `tokenize_mixed` in
    /// <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/utils.hpp>
    pub fn encode_with_special(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>> {
        if parse_special {
            return self.encode(text, add_special);
        }
        match self {
            Self::HuggingFace(t) => {
                // Encode normally first. When no added-vocabulary token came
                // out, the text held no special-token spelling and the two
                // modes agree, so the answer is already correct.
                let normal = t
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                let added = t.get_added_vocabulary();
                if !normal
                    .get_ids()
                    .iter()
                    .any(|id| added.simple_id_to_token(*id).is_some())
                {
                    return Ok(normal.get_ids().to_vec());
                }
                // A spelling really is present, so the modes differ.
                // `set_encode_special_tokens(true)` is the crate's own name for
                // "do not split on added tokens" and needs `&mut`, so reaching
                // it from a shared `&self` costs a clone of the tokenizer. The
                // check above keeps that off every ordinary request: it is paid
                // only when the caller asked for the non-default AND wrote a
                // marker into the text.
                let mut plain = t.clone();
                plain.set_encode_special_tokens(true);
                let encoding = plain
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::SentencePiece(t) => t.encode_without_special_parsing(text, add_special),
            Self::Tiktoken(t) => t.encode_without_special_parsing(text),
        }
    }

    /// The raw bytes one token stands for, as `common_token_to_piece` returns
    /// them (#1442).
    ///
    /// `None` when the id is outside the vocabulary. The bytes are not
    /// necessarily valid UTF-8: a byte-level BPE token routinely holds part of
    /// a multi-byte character, which is exactly the case `/tokenize`'s
    /// `with_pieces` array form exists for.
    pub fn token_piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        match self {
            Self::HuggingFace(t) => {
                let raw = t.id_to_token(id)?;
                // A SentencePiece-style byte-fallback entry is one raw byte and
                // never survives a decode, which turns it into U+FFFD.
                if let Some(byte) = pieces::byte_fallback_value(&raw) {
                    return Some(vec![byte]);
                }
                match t.decode(&[id], false) {
                    // The decoder ran the vocabulary entry through the model's
                    // own transformations (byte-level unmapping, Metaspace,
                    // WordPiece prefix stripping), so prefer its answer.
                    Ok(text) if !pieces::lost_bytes(&raw, &text) => Some(text.into_bytes()),
                    // It reported a replacement character the entry did not
                    // carry, meaning bytes were dropped; recover them from the
                    // byte-level alphabet instead.
                    _ => Some(pieces::byte_level_bytes(&raw)),
                }
            }
            Self::SentencePiece(t) => t.piece_bytes(id),
            Self::Tiktoken(t) => t.piece_bytes(id),
        }
    }

    /// The number of ids this vocabulary can decode, i.e. the exclusive
    /// upper bound for iterating [`Self::token_piece_bytes`] (#1485).
    ///
    /// Used by the DRY breaker-head derivation
    /// (`crate::server::dry_breakers::decode_vocab_texts`), which scans the
    /// whole vocabulary surface once. The model's logit row may be padded
    /// wider; padded ids decode to nothing and can never be breaker heads,
    /// so the tokenizer's own bound is the right one.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(t) => t.get_vocab_size(true),
            Self::SentencePiece(t) => t.vocab_size(),
            Self::Tiktoken(t) => t.vocab_size(),
        }
    }

    /// The id a vocabulary entry holds, by its exact spelling.
    ///
    /// Used by FIM discovery ([`Self::fim_tokens`]), which asks the vocabulary
    /// about a fixed list of marker spellings.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(t) => t.token_to_id(token),
            Self::SentencePiece(t) => t.token_to_id(token),
            Self::Tiktoken(t) => t.token_to_id(token),
        }
    }

    /// The fill-in-the-middle markers this vocabulary declares (#1442).
    ///
    /// Drives `POST /infill`'s capability gate: a model without the prefix,
    /// suffix and middle tokens cannot be served and is refused with the
    /// upstream diagnostic rather than prompted with markers it would emit as
    /// literal text.
    pub fn fim_tokens(&self) -> FimTokens {
        FimTokens::discover(|spelling| self.token_to_id(spelling))
    }

    /// Resolve think and tool-call markers from this tokenizer's vocab.
    ///
    /// Mirrors the upstream Python helper
    /// `mlx_lm.tokenizer_utils._infer_thinking()` (PR #1114) and the
    /// `tool_call_start_tokens` / `tool_call_end_tokens` encoding done in
    /// `TokenizerWrapper.__init__`.  Recognizes:
    ///
    /// * **Single-token think pairs** — `<think>` / `</think>` (Qwen3.x,
    ///   Exaone4, Hunyuan, GLM4, Nemotron-H, …) and
    ///   `<longcat_think>` / `</longcat_think>`.
    /// * **Multi-token think pair** — `<|channel>thought` (open) /
    ///   `<channel|>` (close), used by Gemma 4 and any future model that
    ///   adopts the same channel-priming convention.  The `thought`
    ///   continuation is appended to the open marker because Gemma 4's
    ///   reasoning channel is always primed with `<|channel>thought\n`;
    ///   detecting just `<|channel>` would leak the priming literal back
    ///   into the prompt downstream.
    ///
    /// `tool_call_start` / `tool_call_end` are encoded into id sequences
    /// only when the caller passes both halves through
    /// [`Self::with_tool_call_markers`].  This mirrors the upstream
    /// `TokenizerWrapper(..., tool_call_start=..., tool_call_end=...)`
    /// constructor — the wrapper itself does not auto-infer tool-call
    /// markers from the chat template; the inference is done by the
    /// model loader via `_infer_tool_parser`.  Today the streaming filter
    /// in `server::tool_calls::stream_filter` already covers tool-call
    /// markers via plain string matching on decoded text, so this method
    /// returns `None` for the tool-call halves unless the caller threaded
    /// markers through.  Once a full tool-parser registry exists the
    /// caller will call [`Self::with_tool_call_markers`] to populate them.
    ///
    /// Returns an empty [`ThinkingMarkers`] for non-thinking models so
    /// callers get a stable type they can pattern-match without `Option`
    /// peeling.  [`ThinkingMarkers::has_thinking`] is the canonical
    /// predicate for "is this a thinking model".
    ///
    /// Used by: `server::chat_template::ChatTemplateProcessor`
    /// (default for the `enable_thinking` Jinja kwarg),
    /// `server::tool_calls::stream_filter` (future hookup for token-id
    /// based marker detection on top of today's text-based scan).
    ///
    /// Note: `server::thinking_budget::resolve_thinking_token_ids` currently
    /// uses bare `<|channel>` / `<channel|>` single-token IDs directly rather
    /// than consuming this method.  Migrating it to use the multi-token
    /// sequences returned here is a separate follow-up task.
    pub fn infer_thinking_markers(&self) -> ThinkingMarkers {
        let Some(hf) = self.hf_tokenizer() else {
            return ThinkingMarkers::default();
        };

        // Single-token modes — first hit wins (matches upstream's THINK_TOKENS
        // ordering: `<think>` before `<longcat_think>`).
        const SINGLE_TOKEN_PAIRS: &[(&str, &str)] = &[
            ("<think>", "</think>"),
            ("<|content_thinking|>", "<|end_message|>"),
            ("<longcat_think>", "</longcat_think>"),
        ];
        for (start, end) in SINGLE_TOKEN_PAIRS {
            if let (Some(open_id), Some(close_id)) = (hf.token_to_id(start), hf.token_to_id(end)) {
                return ThinkingMarkers {
                    think_start: Some(start.to_string()),
                    think_end: Some(end.to_string()),
                    think_start_tokens: Some(vec![open_id]),
                    think_end_tokens: Some(vec![close_id]),
                    ..ThinkingMarkers::default()
                };
            }
        }

        // Multi-token mode (Gemma 4 / `<|channel>thought` family). Both
        // halves of the pipe-delimited channel marker must be present in
        // the vocab as added tokens; the trailing `thought` literal is
        // tokenized through the regular encoder so we get whatever subword
        // pieces the model uses.
        if hf.token_to_id("<|channel>").is_some() && hf.token_to_id("<channel|>").is_some() {
            let think_start = "<|channel>thought";
            let think_end = "<channel|>";
            let start_tokens = hf
                .encode(think_start, false)
                .ok()
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default();
            let end_tokens = hf
                .encode(think_end, false)
                .ok()
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default();
            // Defensive guard: if either side encoded to an empty sequence
            // (e.g. a tokenizer that strips the marker entirely) we cannot
            // safely treat this as a thinking model — fall through to the
            // empty default.
            if !start_tokens.is_empty() && !end_tokens.is_empty() {
                return ThinkingMarkers {
                    think_start: Some(think_start.to_string()),
                    think_end: Some(think_end.to_string()),
                    think_start_tokens: Some(start_tokens),
                    think_end_tokens: Some(end_tokens),
                    ..ThinkingMarkers::default()
                };
            }
        }

        ThinkingMarkers::default()
    }

    /// Encode an explicit tool-call start/end string pair into token-id
    /// sequences and merge them onto an existing [`ThinkingMarkers`].
    ///
    /// Mirrors upstream `TokenizerWrapper.__init__`'s
    /// `_tool_call_start_tokens = tuple(encode(tool_call_start, ...))`
    /// behavior: the caller has already resolved the tool-parser family
    /// (via the chat-template heuristic in `mlx_lm.tokenizer_utils
    /// ._infer_tool_parser`) and now needs the token sequence for the
    /// chosen markers.
    ///
    /// Returns the input markers unchanged when the tokenizer does not
    /// support `encode` for the tool-call strings (e.g. SentencePiece /
    /// Tiktoken paths) so callers can chain this on every load without a
    /// guard.
    ///
    /// **Empty `tool_call_end` handling (Mistral-like tokenizers, upstream
    /// mlx-lm PR #1151 fix):** some tokenizers (Mistral variants) report a
    /// non-empty `tool_call_start` but an empty `tool_call_end` string.
    /// Encoding an empty string can produce a non-empty token sequence on
    /// some tokenizers, but the intent is clear: there is no end marker, so
    /// the `tool → normal` state-machine transition must not be registered,
    /// and the empty sequence must not be inserted into the sequence map.
    /// When `tool_call_end` is empty the end-marker fields are left at their
    /// `None` default so downstream callers can distinguish "no end marker"
    /// from "end marker not yet resolved".
    ///
    /// Currently consumed by unit tests; future wiring point for
    /// `server::startup` after resolving a tool-call format — pass the
    /// canonical start/end strings through here so the resulting
    /// `ThinkingMarkers` can drive both the chat-template default and the
    /// stream-filter token-id matching path.
    pub fn with_tool_call_markers(
        &self,
        mut markers: ThinkingMarkers,
        tool_call_start: &str,
        tool_call_end: &str,
    ) -> ThinkingMarkers {
        let Some(hf) = self.hf_tokenizer() else {
            return markers;
        };
        let Ok(start_enc) = hf.encode(tool_call_start, false) else {
            return markers;
        };
        let start_ids = start_enc.get_ids().to_vec();
        if start_ids.is_empty() {
            // A tokenizer that drops the start marker entirely cannot be
            // matched on an id basis. Leave the markers untouched so the
            // text-based stream filter remains the single source of truth.
            return markers;
        }
        markers.tool_call_start = Some(tool_call_start.to_string());
        markers.tool_call_start_tokens = Some(start_ids);

        // Only register the end marker when `tool_call_end` is non-empty.
        // Some tokenizers (Mistral variants) provide a non-empty start
        // marker but an empty end marker. Encoding "" may still produce a
        // non-empty token sequence on certain tokenizers, so guard on the
        // source string rather than on the encoded ids (mirrors upstream
        // mlx-lm PR #1151: `transitions["tool"] = [(te, "normal")] if te
        // else []` / `if te: sequences[te] = tokenizer.tool_call_end`).
        if !tool_call_end.is_empty()
            && let Ok(end_enc) = hf.encode(tool_call_end, false)
        {
            let end_ids = end_enc.get_ids().to_vec();
            if !end_ids.is_empty() {
                markers.tool_call_end = Some(tool_call_end.to_string());
                markers.tool_call_end_tokens = Some(end_ids);
            }
        }

        markers
    }
}

impl SentencePieceTokenizer {
    /// The exclusive id bound this wrapper can decode: the SentencePiece
    /// vocabulary size, extended past any added-token id living outside it
    /// (#1485; see `MlxcelTokenizer::vocab_size`).
    pub fn vocab_size(&self) -> usize {
        let added_bound = self
            .added_token_contents
            .keys()
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0);
        self.processor.len().max(added_bound)
    }

    fn new(
        processor: SentencePieceProcessor,
        special_tokens: HashMap<String, u32>,
        added_token_contents: HashMap<u32, String>,
        bos_id: Option<u32>,
        add_bos: bool,
    ) -> Self {
        let id_to_special_token: HashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        let mut special_tokens_sorted: Vec<(String, u32)> = special_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        // Sort by length descending for greedy longest-match-first
        special_tokens_sorted.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        Self {
            processor,
            special_token_to_id: special_tokens,
            id_to_special_token,
            special_tokens_sorted,
            added_token_contents,
            bos_id,
            add_bos,
            byte_fallback_ids: std::sync::OnceLock::new(),
        }
    }

    /// Encode without recognizing special-token spellings written into the
    /// input text (`parse_special: false`, #1442).
    ///
    /// The BOS prefix is still governed by `add_special_tokens`, which is a
    /// separate switch upstream too.
    fn encode_without_special_parsing(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        if add_special_tokens
            && self.add_bos
            && let Some(bos) = self.bos_id
        {
            result.push(bos);
        }
        let pieces = self
            .processor
            .encode(text)
            .map_err(|e| anyhow::anyhow!("SentencePiece encode failed: {}", e))?;
        result.extend(pieces.iter().map(|piece| piece.id));
        Ok(result)
    }

    /// The id a vocabulary entry holds, by its exact spelling.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(&id) = self.special_token_to_id.get(token) {
            return Some(id);
        }
        // `piece_to_id` answers `Some(unk_id)` for an unknown piece on some
        // models, so an id equal to `unk_id` only counts when the caller
        // actually asked for the unknown piece.
        match self.processor.piece_to_id(token) {
            Ok(Some(id)) if id != self.processor.unk_id() => Some(id),
            _ => None,
        }
    }

    /// Map of byte-fallback token id to the byte it stands for.
    ///
    /// Built once, lazily, by asking the processor for each `<0xXX>` spelling.
    /// It exists because the crate's `decode_piece_ids` **panics** when the
    /// decoded bytes are not valid UTF-8, which is exactly what a lone
    /// byte-fallback token produces, so `piece_bytes` must answer those ids
    /// without going through a decode.
    fn byte_fallback_ids(&self) -> &HashMap<u32, u8> {
        self.byte_fallback_ids.get_or_init(|| {
            let mut map = HashMap::new();
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let spelling = format!("<0x{byte:02X}>");
                if let Ok(Some(id)) = self.processor.piece_to_id(&spelling) {
                    map.insert(id, byte);
                }
            }
            map
        })
    }

    /// Raw bytes for one token; see `MlxcelTokenizer::token_piece_bytes`.
    fn piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if let Some(&byte) = self.byte_fallback_ids().get(&id) {
            return Some(vec![byte]);
        }
        if let Some(special) = self.id_to_special_token.get(&id) {
            return Some(special.clone().into_bytes());
        }
        if let Some(content) = self.added_token_contents.get(&id) {
            return Some(content.clone().into_bytes());
        }
        self.processor
            .decode_piece_ids(&[id])
            .ok()
            .map(String::into_bytes)
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Prepend BOS if configured
        if add_special_tokens
            && self.add_bos
            && let Some(bos) = self.bos_id
        {
            result.push(bos);
        }

        if self.special_tokens_sorted.is_empty() {
            // No special tokens to handle — encode directly
            let pieces = self
                .processor
                .encode(text)
                .map_err(|e| anyhow::anyhow!("SentencePiece encode failed: {}", e))?;
            for piece in &pieces {
                result.push(piece.id);
            }
            return Ok(result);
        }

        // Split text at special token boundaries (greedy longest-match-first)
        let segments = self.split_with_special_tokens(text);

        for segment in segments {
            if let Some(&id) = self.special_token_to_id.get(&segment) {
                // This segment is a special token — insert its ID directly
                result.push(id);
            } else {
                // Regular text — encode via sentencepiece
                let pieces = self
                    .processor
                    .encode(&segment)
                    .map_err(|e| anyhow::anyhow!("SentencePiece encode failed: {}", e))?;
                for piece in &pieces {
                    result.push(piece.id);
                }
            }
        }

        Ok(result)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();
        let mut regular_ids: Vec<u32> = Vec::new();

        for &id in ids {
            if let Some(special) = self.id_to_special_token.get(&id) {
                // Flush any accumulated regular IDs first
                if !regular_ids.is_empty() {
                    let text = self
                        .processor
                        .decode_piece_ids(&regular_ids)
                        .map_err(|e| anyhow::anyhow!("SentencePiece decode failed: {}", e))?;
                    result.push_str(&text);
                    regular_ids.clear();
                }
                if !skip_special_tokens {
                    result.push_str(special);
                }
            } else if let Some(content) = self.added_token_contents.get(&id) {
                // Non-special added token: outside the SentencePiece vocab
                // (decode_piece_ids would error "Out of range"), but real text
                // per HF semantics, so it is emitted regardless of
                // skip_special_tokens.
                if !regular_ids.is_empty() {
                    let text = self
                        .processor
                        .decode_piece_ids(&regular_ids)
                        .map_err(|e| anyhow::anyhow!("SentencePiece decode failed: {}", e))?;
                    result.push_str(&text);
                    regular_ids.clear();
                }
                result.push_str(content);
            } else {
                regular_ids.push(id);
            }
        }

        // Flush remaining regular IDs
        if !regular_ids.is_empty() {
            let text = self
                .processor
                .decode_piece_ids(&regular_ids)
                .map_err(|e| anyhow::anyhow!("SentencePiece decode failed: {}", e))?;
            result.push_str(&text);
        }

        Ok(result)
    }

    /// Split text into segments, alternating between special tokens and regular text.
    /// Uses greedy longest-match-first strategy.
    fn split_with_special_tokens(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to match a special token at the current position
            let mut matched = false;
            for (token, _id) in &self.special_tokens_sorted {
                if remaining.starts_with(token.as_str()) {
                    segments.push(token.clone());
                    remaining = &remaining[token.len()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Find the next special token occurrence
                let mut next_pos = remaining.len();
                for (token, _id) in &self.special_tokens_sorted {
                    if let Some(pos) = remaining.find(token.as_str())
                        && pos < next_pos
                    {
                        next_pos = pos;
                    }
                }
                // Everything before the next special token is regular text
                segments.push(remaining[..next_pos].to_string());
                remaining = &remaining[next_pos..];
            }
        }

        segments
    }
}

/// Parse special tokens from tokenizer_config.json's `added_tokens_decoder` field
fn parse_special_tokens(model_path: &Path) -> (HashMap<String, u32>, HashMap<u32, String>, bool) {
    let config_path = model_path.join("tokenizer_config.json");
    let mut special_tokens = HashMap::new();
    let mut added_token_contents = HashMap::new();
    let mut add_bos = false;

    if let Ok(content) = std::fs::read_to_string(&config_path)
        && let Ok(config) = serde_json::from_str::<serde_json::Value>(&content)
    {
        // Parse add_bos_token
        if let Some(v) = config.get("add_bos_token").and_then(|v| v.as_bool()) {
            add_bos = v;
        }

        // Parse added_tokens_decoder: { "128132": { "content": "<|im_start|>", "special": true }, ... }
        if let Some(decoder) = config
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
        {
            for (id_str, entry) in decoder {
                if let (Ok(id), Some(content)) = (
                    id_str.parse::<u32>(),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    let is_special = entry
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if is_special {
                        special_tokens.insert(content.to_string(), id);
                    } else {
                        // Non-special added tokens (outside the SentencePiece
                        // vocab) still need an id -> content mapping so decode
                        // can render them instead of erroring "Out of range".
                        added_token_contents.insert(id, content.to_string());
                    }
                }
            }
        }
    }

    (special_tokens, added_token_contents, add_bos)
}

/// Find a `.tiktoken` file in the model directory.
/// Tries `tiktoken.model` first, then any `*.tiktoken` file.
fn find_tiktoken_file(model_path: &Path) -> Option<std::path::PathBuf> {
    // Try tiktoken.model first (standard name used by some models)
    let tiktoken_model = model_path.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Some(tiktoken_model);
    }

    // Try any *.tiktoken file
    let pattern = model_path.join("*.tiktoken");
    if let Ok(paths) = glob::glob(pattern.to_str()?) {
        return paths.flatten().next();
    }
    None
}

fn remote_tokenizer_repo_for_model_type(model_type: &str) -> Option<&'static str> {
    match model_type {
        "moondream3" => Some("moondream/starmie-v1"),
        _ => None,
    }
}

fn remote_tokenizer_repo_for_model(model_path: &Path) -> Option<&'static str> {
    let model_type = read_config_model_type(model_path)?;
    remote_tokenizer_repo_for_model_type(&model_type)
}

fn read_config_model_type(model_path: &Path) -> Option<String> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    config
        .get("model_type")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

/// Repos whose local `tokenizer.json` must be OVERRIDDEN (not merely used as
/// a fallback when absent).
///
/// The official `vikhyatk/moondream2` repository never removed its legacy
/// GPT-2/CodeGen tokenizer files, so a starmie-era snapshot (revision
/// 2025-06-21+) still ships a `tokenizer.json` that does NOT match its
/// weights; the shipped `moondream.py` loads `moondream/starmie-v1` from the
/// Hub instead. Loading the stale local file makes the numerically correct
/// forward pass consume and emit token ids from the wrong vocabulary, which
/// surfaces as pure garbage text (see `crate::moondream2_prompt`).
///
/// Returns the repo to fetch the real tokenizer from, or `None` when the
/// local `tokenizer.json` (if any) is trustworthy:
/// - the checkpoint is not a moondream2-family one, or
/// - it is a legacy-era moondream2 (GPT-2 tokenizer is correct), or
/// - the local `tokenizer.json` is already the starmie one (converted or
///   manually placed), so no fetch is needed.
fn remote_tokenizer_override_for_model(model_path: &Path) -> Option<&'static str> {
    let model_type = read_config_model_type(model_path)?;
    if !matches!(model_type.as_str(), "moondream1" | "moondream2") {
        return None;
    }
    if crate::moondream2_prompt::detect_moondream2_prompt_style(model_path)
        != crate::moondream2_prompt::Moondream2PromptStyle::StarmieTemplates
    {
        return None;
    }
    if let Ok(tokenizer_json) = std::fs::read_to_string(model_path.join("tokenizer.json"))
        && tokenizer_json.contains("<|md_reserved_0|>")
    {
        return None;
    }
    Some("moondream/starmie-v1")
}

fn download_remote_tokenizer(repo_id: &str) -> Result<tokenizers::Tokenizer> {
    // `--offline` / `LLAMA_ARG_OFFLINE` forbids every fetch, and this one is
    // reached from inside the loader rather than through the `-m` resolver, so
    // it consults the process-wide flag directly (issue #1434). Without this,
    // an air-gapped `--offline` run of a starmie-era moondream2 checkpoint
    // still tried to reach huggingface.co and failed to start.
    crate::downloader::ensure_online(&format!("the tokenizer for '{repo_id}'"))?;
    let api = Api::new()
        .map_err(|err| anyhow::anyhow!("Failed to initialize Hugging Face API: {}", err))?;
    let repo = api.model(repo_id.to_string());
    let tokenizer_path = repo.get("tokenizer.json").map_err(|err| {
        anyhow::anyhow!(
            "Failed to download tokenizer.json from {}: {}",
            repo_id,
            err
        )
    })?;
    tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|err| anyhow::anyhow!(err))
}

/// Build a JSON object for one of PLaMo's four special tokens, in the shape the
/// `tokenizers` crate expects inside the top-level `added_tokens` array.
fn plamo_added_token(id: u32, content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true,
    })
}

/// Build a HuggingFace [`tokenizers::Tokenizer`] for PLaMo's custom
/// `PlamoTokenizer` format.
///
/// PLaMo 2 checkpoints ship a `tokenizer.jsonl` Unigram vocabulary (one
/// `[token, score, type]` array per line, where the line index is the token id)
/// plus a `tokenization_plamo.py` reference, instead of a `tokenizer.json`,
/// SentencePiece `tokenizer.model`, or tiktoken vocab. The reference tokenizer
/// is a SentencePiece-style Unigram with byte fallback, run over the raw text
/// (no normalizer, no pre-tokenizer) using Viterbi (maximum-score) decoding;
/// 256 `<0xXX>` byte tokens cover any character the vocab does not.
///
/// We reconstruct that behavior with the `tokenizers` crate's Unigram model:
/// the vocab and scores load verbatim in token-id order, `byte_fallback` routes
/// uncovered characters through the `<0xXX>` tokens, and a `ByteFallback`
/// decoder reassembles those bytes (UTF-8, lossy) exactly like
/// `PlamoTokenizer.convert_tokens_to_string`. The four special tokens (unk=0,
/// bos=1, eos=2, pad=3) are also registered as added/special tokens so
/// `decode(skip_special_tokens=true)` can strip them and EOS detection matches.
///
/// Upstream reference:
/// https://huggingface.co/pfnet/plamo-2-1b/blob/main/tokenization_plamo.py
fn build_plamo_tokenizer(model_path: &Path) -> Result<tokenizers::Tokenizer> {
    use std::io::BufRead;

    let jsonl_path = model_path.join("tokenizer.jsonl");
    let file = std::fs::File::open(&jsonl_path)
        .map_err(|e| anyhow::anyhow!("Failed to open {:?}: {}", jsonl_path, e))?;
    let reader = std::io::BufReader::new(file);

    // The Unigram vocab in token-id order: vocab[i] = [token, score]. Each
    // jsonl line is a `[token, score, type]` array; the line index is the id.
    // Parse via serde_json so tokens containing quotes, backslashes, control
    // characters, or non-BMP code points are handled as real JSON, never
    // hand-formatted. The `type` field ("NORMAL" / "CONTROL" / "UNKNOWN" /
    // "BYTE") is informational: byte tokens stay in the vocab with their scores
    // so `byte_fallback` can resolve them, matching the Python tokenizer, which
    // keeps every entry addressable by id.
    let mut vocab: Vec<serde_json::Value> = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line
            .map_err(|e| anyhow::anyhow!("Failed to read {:?} line {}: {}", jsonl_path, idx, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse {:?} line {} ({:?}): {}",
                jsonl_path,
                idx,
                line,
                e
            )
        })?;
        let entry = row.as_array().ok_or_else(|| {
            anyhow::anyhow!(
                "{:?} line {} is not a JSON array: {:?}",
                jsonl_path,
                idx,
                line
            )
        })?;
        let token = entry.first().and_then(|v| v.as_str()).ok_or_else(|| {
            anyhow::anyhow!(
                "{:?} line {} has no string token: {:?}",
                jsonl_path,
                idx,
                line
            )
        })?;
        let score = entry.get(1).and_then(|v| v.as_f64()).ok_or_else(|| {
            anyhow::anyhow!(
                "{:?} line {} has no numeric score: {:?}",
                jsonl_path,
                idx,
                line
            )
        })?;
        vocab.push(serde_json::json!([token, score]));
    }

    if vocab.is_empty() {
        return Err(anyhow::anyhow!(
            "{:?} contained no vocab entries",
            jsonl_path
        ));
    }

    // Raw text in, raw text out: no normalizer and no pre-tokenizer (PLaMo
    // tokens carry literal spaces, e.g. " of"/"  ", not SentencePiece `_`
    // markers), and a ByteFallback decoder mirrors `convert_tokens_to_string`.
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            plamo_added_token(0, "<|plamo:unk|>"),
            plamo_added_token(1, "<|plamo:bos|>"),
            plamo_added_token(2, "<|plamo:eos|>"),
            plamo_added_token(3, "<|plamo:pad|>"),
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {"type": "ByteFallback"},
        "model": {
            "type": "Unigram",
            "unk_id": 0,
            "byte_fallback": true,
            "vocab": vocab,
        },
    });

    let json_bytes = serde_json::to_vec(&tokenizer_json)
        .map_err(|e| anyhow::anyhow!("Failed to serialize PLaMo tokenizer.json: {}", e))?;

    tokenizers::Tokenizer::from_bytes(json_bytes).map_err(|e| {
        anyhow::anyhow!(
            "Failed to build PLaMo tokenizer from {:?}: {}",
            jsonl_path,
            e
        )
    })
}

/// The pre-tokenization regex `Qwen2Tokenizer` splits on, copied verbatim from
/// `transformers/models/qwen2/tokenization_qwen2.py` (`PRETOKENIZE_REGEX`).
/// `Qwen2Converter` in `convert_slow_tokenizer.py` feeds exactly this string to
/// a `Split(behavior="isolated")` pre-tokenizer.
const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Does this directory hold a Qwen2-family slow tokenizer (`vocab.json` +
/// `merges.txt`) rather than an exported `tokenizer.json`?
///
/// Both files plus a `tokenizer_class` naming the Qwen2 tokenizer are required.
/// The class check keeps this narrow: `vocab.json` and `merges.txt` are the
/// generic GPT-2 slow-tokenizer pair and other families that ship them (with a
/// different normalizer, pre-tokenizer regex, or prefix-space convention) must
/// not be silently tokenized with Qwen2's rules.
fn is_qwen2_slow_tokenizer_dir(model_path: &Path) -> bool {
    if !model_path.join("vocab.json").exists() || !model_path.join("merges.txt").exists() {
        return false;
    }
    let Ok(raw) = std::fs::read_to_string(model_path.join("tokenizer_config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    matches!(
        config.get("tokenizer_class").and_then(|v| v.as_str()),
        Some("Qwen2Tokenizer") | Some("Qwen2TokenizerFast")
    )
}

/// Build a HuggingFace [`tokenizers::Tokenizer`] from a Qwen2 slow-tokenizer
/// directory (`vocab.json` + `merges.txt` + added tokens).
///
/// `nvidia/LocateAnything-3B` and its MLX conversions ship the slow-tokenizer
/// files only, so `Tokenizer::from_file` has nothing to read. Transformers
/// handles that by running `Qwen2Converter`
/// (`transformers/convert_slow_tokenizer.py`) at load time; this is the same
/// construction, component for component:
///
/// - model: byte-level `BPE` from `vocab.json` / `merges.txt`, no unk token,
///   no subword prefix or suffix, `fuse_unk = false`, `byte_fallback = false`;
/// - normalizer: `NFC`;
/// - pre-tokenizer: `Sequence[Split(PRETOKENIZE_REGEX, isolated), ByteLevel(add_prefix_space, use_regex = false)]`;
/// - decoder: `ByteLevel`;
/// - post-processor: `ByteLevel(trim_offsets = false)`.
///
/// `add_prefix_space` comes from `tokenizer_config.json` and defaults to
/// `false`, matching `getattr(self.original_tokenizer, "add_prefix_space", False)`
/// in the converter.
///
/// `vocab.json` is checked to be densely numbered (every id in `0..len` used
/// exactly once) before anything else is built, and a checkpoint that fails
/// that check is refused. `tokenizers` derives the first added-token id from
/// `BPE::get_vocab_size()`, which is `vocab.len()` no matter which ids the file
/// actually occupies, so a gap below `len` hands the first added token an id the
/// base vocab already owns: both contents then resolve to that id and the base
/// one stops decoding. The library exposes no way to steer that starting id, so
/// recomputing a base size here would not help and a sparse `vocab.json` simply
/// cannot be loaded soundly through this path.
///
/// The added tokens (`added_tokens.json`, or `added_tokens_decoder` in
/// `tokenizer_config.json` when it carries per-token flags) are registered in
/// ascending id order and then verified: `tokenizers` assigns added-token ids
/// sequentially from the base vocab size, so an out-of-order id in the
/// checkpoint would silently shift every later token. LocateAnything's 1038
/// added tokens include the 1001 coordinate tokens `<0>`..`<1000>` that carry
/// its box output, so an off-by-one there would corrupt every box rather than
/// fail loudly. Registration is batched over consecutive runs of the same
/// `special` flag; the loop comment records the measured cost of not doing that
/// and why runs rather than two groups.
fn build_qwen2_bpe_tokenizer(model_path: &Path) -> Result<tokenizers::Tokenizer> {
    use tokenizers::Tokenizer;
    use tokenizers::models::bpe::BPE;

    let vocab_path = model_path.join("vocab.json");
    let merges_path = model_path.join("merges.txt");

    let vocab_raw = std::fs::read_to_string(&vocab_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", vocab_path, e))?;
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_raw)
        .map_err(|e| anyhow::anyhow!("Failed to parse {:?}: {}", vocab_path, e))?;
    if vocab.is_empty() {
        return Err(anyhow::anyhow!("{:?} contained no entries", vocab_path));
    }
    validate_dense_vocab_ids(&vocab, &vocab_path)?;

    let merges_raw = std::fs::read_to_string(&merges_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", merges_path, e))?;
    let mut merges: Vec<(String, String)> = Vec::new();
    for line in merges_raw.lines() {
        // The first line is a `#version:` header in every GPT-2-style export.
        if line.is_empty() || line.starts_with("#version") {
            continue;
        }
        let Some((left, right)) = line.split_once(' ') else {
            return Err(anyhow::anyhow!(
                "{:?} line {:?} is not a `left right` merge pair",
                merges_path,
                line
            ));
        };
        merges.push((left.to_string(), right.to_string()));
    }
    if merges.is_empty() {
        return Err(anyhow::anyhow!("{:?} contained no merges", merges_path));
    }

    let base_vocab_size = vocab.len() as u32;
    // `BpeBuilder::vocab_and_merges` takes the crate's own `Vocab` alias
    // (an `AHashMap`), so re-collect rather than passing the `std` map.
    let vocab: tokenizers::models::bpe::Vocab = vocab.into_iter().collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .fuse_unk(false)
        .byte_fallback(false)
        .continuing_subword_prefix(String::new())
        .end_of_word_suffix(String::new())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build Qwen2 BPE model: {}", e))?;

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(Some(tokenizers::normalizers::unicode::NFC));

    let add_prefix_space = std::fs::read_to_string(model_path.join("tokenizer_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|cfg| cfg.get("add_prefix_space").and_then(|v| v.as_bool()))
        .unwrap_or(false);

    let split = tokenizers::pre_tokenizers::split::Split::new(
        tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN2_PRETOKENIZE_REGEX.to_string()),
        tokenizers::SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow::anyhow!("Failed to build Qwen2 Split pre-tokenizer: {}", e))?;
    let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
        add_prefix_space,
        /* trim_offsets */ true,
        /* use_regex */ false,
    );
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(
        vec![split.into(), byte_level.into()],
    )));

    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_post_processor(Some(
        tokenizers::processors::byte_level::ByteLevel::default().trim_offsets(false),
    ));

    let added = read_added_tokens_sorted(model_path)?;
    for (id, token) in &added {
        if *id < base_vocab_size {
            return Err(anyhow::anyhow!(
                "Added token {:?} claims id {} which is inside the {} entry base vocab",
                token.content,
                id,
                base_vocab_size
            ));
        }
    }

    // Register the added tokens a run at a time rather than one at a time.
    // Every `add_tokens` / `add_special_tokens` call ends in
    // `AddedVocabulary::refresh_added_tokens`, which rebuilds both
    // Aho-Corasick automatons over every token accumulated so far (and
    // `add_tokens` first clones the whole existing set), so N separate calls
    // cost N full rebuilds. Measured on the pinned `tokenizers` 0.22.2, release
    // build: 1000 entries took 162 ms one at a time against 0.5 ms batched,
    // 4000 took 2.42 s against 2.3 ms, and 16000 took 46.1 s against 10.4 ms,
    // a clean 4x per doubling. That made a checkpoint with a large
    // `added_tokens_decoder` able to wedge the loader for minutes with no error
    // and no timeout, and it cost LocateAnything's own 1038 tokens 175 ms
    // instead of 0.5 ms.
    //
    // The batches are consecutive runs of the same `special` flag, not one
    // specials group plus one normals group. Ids are handed out in the order
    // the tokens are passed, so regrouping `[A(special), B(normal),
    // C(special)]` declared as 5, 6, 7 would assign A=5, C=6, B=7 and silently
    // mis-map the vocabulary. Runs keep the sorted order intact: a uniformly
    // special checkpoint (LocateAnything, and the common case generally)
    // collapses to a single call, while a perfectly alternating one degrades to
    // the previous one-call-per-token cost without ever moving an id.
    for batch in added.chunk_by(|left, right| left.1.special == right.1.special) {
        let tokens: Vec<tokenizers::AddedToken> =
            batch.iter().map(|(_, token)| token.clone()).collect();
        if batch[0].1.special {
            tokenizer.add_special_tokens(&tokens);
        } else {
            tokenizer.add_tokens(&tokens);
        }
        for (id, token) in batch {
            let assigned = tokenizer.token_to_id(&token.content);
            if assigned != Some(*id) {
                return Err(anyhow::anyhow!(
                    "Added token {:?} landed at id {:?} but the checkpoint declares {}; \
                     the added-token ids in {:?} are not contiguous above the {} entry base vocab",
                    token.content,
                    assigned,
                    id,
                    model_path,
                    base_vocab_size
                ));
            }
        }
    }

    Ok(tokenizer)
}

/// Reject a `vocab.json` whose ids are not exactly `0..len`.
///
/// The file is a `content -> id` map and nothing in the format forces those ids
/// to be dense, but `BPE::get_vocab_size()` reports `vocab.len()` and
/// `AddedVocabulary` starts handing out added-token ids from that number. A gap
/// below `len` therefore aliases: measured on `tokenizers` 0.22.2 with
/// `{"Ġ":0,"h":1,"Ġh":2,"i":3,"COLLIDE":5}` plus an added token declaring id 5,
/// the added token is assigned 5, the loader's contiguity check passes because
/// 5 is what the checkpoint declared, and afterwards `token_to_id` returns 5 for
/// both `COLLIDE` and the added token while `decode([5])` no longer yields
/// `COLLIDE`. The starting id cannot be overridden, so refusing the input is the
/// only sound outcome.
fn validate_dense_vocab_ids(vocab: &HashMap<String, u32>, vocab_path: &Path) -> Result<()> {
    let vocab_len = vocab.len();
    let mut owner: Vec<Option<&str>> = vec![None; vocab_len];
    // Sort so a malformed file always produces the same message, whatever order
    // the hash map happens to iterate in.
    let mut entries: Vec<(&str, u32)> = vocab
        .iter()
        .map(|(content, id)| (content.as_str(), *id))
        .collect();
    entries.sort_unstable_by(|left, right| (left.1, left.0).cmp(&(right.1, right.0)));

    let mut out_of_range: Option<(&str, u32)> = None;
    for (content, id) in entries {
        match owner.get_mut(id as usize) {
            None => {
                out_of_range.get_or_insert((content, id));
            }
            Some(slot) => {
                if let Some(previous) = slot.replace(content) {
                    return Err(anyhow::anyhow!(
                        "{:?} assigns id {} to both {:?} and {:?}; added-token ids are only \
                         unambiguous when every id below the {} entry count is used exactly once",
                        vocab_path,
                        id,
                        previous,
                        content,
                        vocab_len
                    ));
                }
            }
        }
    }

    let unused = owner.iter().position(Option::is_none);
    if unused.is_none() && out_of_range.is_none() {
        return Ok(());
    }
    let stray = out_of_range
        .map(|(content, id)| format!("{content:?} claims id {id}"))
        .unwrap_or_else(|| "no entry claims it".to_string());
    let gap = unused
        .map(|id| format!("id {id} is unused"))
        .unwrap_or_else(|| "every id below the count is used".to_string());
    Err(anyhow::anyhow!(
        "{:?} is not densely numbered: it holds {} entries so its ids must be exactly 0..{}, \
         but {} while {}. `tokenizers` starts assigning added-token ids at the entry count \
         regardless, so loading this would alias an added token onto an id the base vocab \
         already owns",
        vocab_path,
        vocab_len,
        vocab_len,
        gap,
        stray
    ))
}

/// Read the checkpoint's added tokens as `(id, AddedToken)` sorted by id.
///
/// `added_tokens_decoder` in `tokenizer_config.json` is preferred because it
/// carries the per-token `special` / `lstrip` / `rstrip` / `normalized` /
/// `single_word` flags; `added_tokens.json` is the flat `content -> id` map and
/// is used when the decoder table is absent. Every LocateAnything added token
/// is `normalized: false`, which is what keeps `<0>`..`<1000>` from being
/// rewritten by the NFC normalizer.
///
/// `special` defaults to `false` in both branches, matching HuggingFace's
/// `AddedToken` default. Defaulting it to `true` would be actively destructive:
/// `added_tokens.json` carries no flags at all, so every one of its entries
/// would be registered special, and `AddedVocabulary`'s `special_tokens_set` is
/// insert-only (see `demote_tool_parser_markers` and issue #778), so those
/// tokens would then be stripped from every `decode(.., skip_special_tokens =
/// true)` with no way to undo it. LocateAnything is unaffected either way: its
/// `added_tokens_decoder` carries an explicit `special` on all 1038 entries.
fn read_added_tokens_sorted(model_path: &Path) -> Result<Vec<(u32, tokenizers::AddedToken)>> {
    use tokenizers::AddedToken;

    let mut out: Vec<(u32, AddedToken)> = Vec::new();

    let config = std::fs::read_to_string(model_path.join("tokenizer_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok());
    let decoder = config
        .as_ref()
        .and_then(|cfg| cfg.get("added_tokens_decoder"))
        .and_then(|v| v.as_object());

    if let Some(decoder) = decoder {
        for (id, entry) in decoder {
            let id: u32 = id
                .parse()
                .map_err(|e| anyhow::anyhow!("added_tokens_decoder key {:?}: {}", id, e))?;
            let content = entry
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("added_tokens_decoder[{id}] has no content"))?;
            let flag = |name: &str, default: bool| {
                entry.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
            };
            let token = AddedToken::from(content.to_string(), flag("special", false))
                .single_word(flag("single_word", false))
                .lstrip(flag("lstrip", false))
                .rstrip(flag("rstrip", false))
                .normalized(flag("normalized", false));
            out.push((id, token));
        }
    } else if let Ok(raw) = std::fs::read_to_string(model_path.join("added_tokens.json")) {
        let map: HashMap<String, u32> = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("Failed to parse added_tokens.json: {}", e))?;
        for (content, id) in map {
            out.push((id, AddedToken::from(content, false).normalized(false)));
        }
    }

    out.sort_by_key(|(id, _)| *id);
    Ok(out)
}

/// Repair Gemma-family `tokenizer.json` exports that dropped the
/// BOS-inserting post-processor (issue #686).
///
/// `tokenizer_class: "GemmaTokenizer"` semantics in transformers prepend
/// `<bos>` on every encode with special tokens (`add_bos_token` defaults to
/// true), and Gemma model quality collapses without it: measured on the #686
/// docs corpus, dropping BOS costs gemma-3-4b ~3.6 nats/token and
/// gemma-4-12b ~6.6 nats/token of teacher-forced NLL. Gemma 3 checkpoints
/// ship a `TemplateProcessing` post-processor that inserts `<bos>`, but
/// current Gemma 4 exports ship a passthrough post-processor, so every
/// raw-text path (CLI `generate`, `/v1/completions`, teacher-forced scoring)
/// silently ran BOS-less. Chat-template paths were unaffected because the
/// Gemma 4 template emits `{{ bos_token }}` itself.
///
/// The repair installs the exact `TemplateProcessing` Gemma 3 ships
/// (`<bos> $A` single, `<bos> $A <bos>:1 $B:1` pair) when ALL hold:
/// - `tokenizer_config.json` declares a Gemma tokenizer class, or an
///   explicit `"add_bos_token": true`;
/// - `add_bos_token` is not explicitly `false`;
/// - the configured `bos_token` resolves to a vocab id; and
/// - an encode probe shows the loaded post-processor does NOT already
///   insert that id (so correct exports such as Gemma 3 are untouched).
fn ensure_bos_post_processor(tokenizer: &mut tokenizers::Tokenizer, model_path: &Path) {
    let config_path = model_path.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&config_path) else {
        return;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return;
    };

    let add_bos = config.get("add_bos_token").and_then(|v| v.as_bool());
    if add_bos == Some(false) {
        return;
    }
    let tokenizer_class = config
        .get("tokenizer_class")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let is_gemma_class = matches!(tokenizer_class, "GemmaTokenizer" | "GemmaTokenizerFast");
    if add_bos != Some(true) && !is_gemma_class {
        return;
    }

    // bos_token is either a plain string or an AddedToken-style object.
    let bos_token = match config.get("bos_token") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(o)) => o
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    if bos_token.is_empty() {
        return;
    }
    let Some(bos_id) = tokenizer.token_to_id(&bos_token) else {
        return;
    };

    // Probe: a correct export already inserts BOS on encode-with-specials.
    if let Ok(probe) = tokenizer.encode("bos probe", true)
        && probe.get_ids().first() == Some(&bos_id)
    {
        return;
    }

    let template = tokenizers::processors::template::TemplateProcessing::builder()
        .try_single(format!("{bos_token} $A"))
        .and_then(|builder| builder.try_pair(format!("{bos_token} $A {bos_token}:1 $B:1")))
        .and_then(|builder| {
            builder
                .special_tokens(vec![(bos_token.clone(), bos_id)])
                .build()
                .map_err(|e| e.to_string())
        });
    match template {
        Ok(template) => {
            tokenizer.with_post_processor(Some(template));
            tracing::info!(
                model_path = %model_path.display(),
                bos_token,
                bos_id,
                "tokenizer.json lacks the Gemma BOS post-processor; installed \
                 the standard `<bos> $A` TemplateProcessing (issue #686)"
            );
        }
        Err(err) => {
            tracing::warn!(
                model_path = %model_path.display(),
                error = %err,
                "failed to install the Gemma BOS post-processor; raw-text \
                 encodes will remain BOS-less"
            );
        }
    }
}

/// DiffusionGemma tool-parser / reasoning-channel markers that must decode as
/// visible text rather than being stripped as special tokens.
///
/// `<|tool_call>` / `<tool_call|>` / `<|"|>` frame the pipe-delimited Gemma 4
/// tool-call format (`server::tool_calls::formats::try_gemma4`); `<|channel>`
/// / `<channel|>` frame the reasoning channel
/// (`server::thinking_budget::resolve_thinking_token_ids`,
/// `server::chat_template`). All five ship together in the DiffusionGemma
/// `tokenizer.json` as `special: true` added tokens (issue #778).
const DIFFUSION_GEMMA_TOOL_PARSER_MARKERS: [&str; 5] = [
    "<|tool_call>",
    "<tool_call|>",
    "<|\"|>",
    "<|channel>",
    "<channel|>",
];

/// Demote the DiffusionGemma tool-parser markers from `special: true` to
/// `special: false` inside a parsed `tokenizer.json` document, in place.
///
/// WHY this must happen before the tokenizer is deserialized, not after:
/// `tokenizers::Tokenizer::decode(ids, skip_special_tokens=true)` strips any
/// token the `AddedVocabulary` has recorded in its `special_tokens_set`. That
/// set is insert-only: once a content string is registered special, nothing
/// in the crate's public API (`add_tokens`, `add_special_tokens`, ...) ever
/// removes it, because `add_tokens` only ever *inserts* into
/// `special_tokens_set` and re-adding the same content with `special: false`
/// is a no-op for that set. `AddedVocabulary`'s `Deserialize` path rebuilds
/// the vocabulary from scratch by replaying `add_tokens` over each
/// `added_tokens` entry's own `special` field, so the only reliable point to
/// flip the flag is in the raw JSON, before that rebuild runs.
///
/// Demoted tokens remain ordinary added tokens (still `special: false`, not
/// removed), so they still encode atomically and never split across the BPE
/// encoder; they just stop being skipped by `decode(...,
/// skip_special_tokens=true)`, so `server::tool_calls::parser::parse_tool_calls`
/// can see them in generated text.
///
/// Only markers that are both present and currently special are touched, so
/// a future checkpoint that ships a subset of the five (or none) is handled
/// without error. Returns the list of marker strings that were demoted; an
/// empty result means the document was left untouched.
fn demote_tool_parser_markers(tokenizer_json: &mut serde_json::Value) -> Vec<String> {
    let mut demoted = Vec::new();
    let Some(added_tokens) = tokenizer_json
        .get_mut("added_tokens")
        .and_then(|v| v.as_array_mut())
    else {
        return demoted;
    };

    for entry in added_tokens.iter_mut() {
        let Some(obj) = entry.as_object_mut() else {
            continue;
        };
        let is_marker = obj
            .get("content")
            .and_then(|v| v.as_str())
            .is_some_and(|content| DIFFUSION_GEMMA_TOOL_PARSER_MARKERS.contains(&content));
        if !is_marker {
            continue;
        }
        if obj.get("special").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        obj.insert("special".to_string(), serde_json::Value::Bool(false));
        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
            demoted.push(content.to_string());
        }
    }

    demoted
}

/// Build the HuggingFace tokenizer for a DiffusionGemma checkpoint, demoting
/// the tool-parser markers (see [`demote_tool_parser_markers`]) before the
/// `tokenizers::Tokenizer` is deserialized from the patched JSON bytes.
fn build_diffusion_gemma_tokenizer(tokenizer_json_path: &Path) -> Result<tokenizers::Tokenizer> {
    let raw = std::fs::read_to_string(tokenizer_json_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", tokenizer_json_path, e))?;
    let mut json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("Failed to parse {:?}: {}", tokenizer_json_path, e))?;

    let demoted = demote_tool_parser_markers(&mut json);
    if !demoted.is_empty() {
        tracing::info!(
            tokenizer_json = %tokenizer_json_path.display(),
            markers = ?demoted,
            "demoted DiffusionGemma tool-parser markers from special to non-special \
             added tokens so skip-special decode retains them (issue #778)"
        );
    }

    let bytes = serde_json::to_vec(&json)
        .map_err(|e| anyhow::anyhow!("Failed to re-serialize {:?}: {}", tokenizer_json_path, e))?;
    tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!(e))
}

/// `true` when `config.json`'s `model_type` identifies a DiffusionGemma
/// checkpoint (text-only exports use `diffusion_gemma_text`; matches the
/// detection table in `crate::models::detection`).
fn is_diffusion_gemma_model(model_path: &Path) -> bool {
    matches!(
        read_config_model_type(model_path).as_deref(),
        Some("diffusion_gemma") | Some("diffusion_gemma_text")
    )
}

pub fn load_tokenizer(model_path: &Path) -> Result<MlxcelTokenizer> {
    // Model-specific override: some official checkpoints ship a stale
    // tokenizer.json that does not match their weights (starmie-era
    // moondream2). The real tokenizer must be resolved from the Hub (cached
    // by hf-hub after the first fetch) before the local file is considered.
    if let Some(repo_id) = remote_tokenizer_override_for_model(model_path) {
        let tokenizer = download_remote_tokenizer(repo_id).map_err(|err| {
            anyhow::anyhow!(
                "This moondream2 checkpoint pairs starmie-era weights with a stale legacy \
                 tokenizer.json; its text is only coherent with the {repo_id} tokenizer. \
                 Resolving that tokenizer failed: {err}. If this host is offline, download \
                 https://huggingface.co/{repo_id}/resolve/main/tokenizer.json and place it \
                 in {model_path:?} as tokenizer.json."
            )
        })?;
        return Ok(MlxcelTokenizer::HuggingFace(tokenizer));
    }

    // Try HuggingFace tokenizer.json first
    let tokenizer_json_path = model_path.join("tokenizer.json");
    if tokenizer_json_path.exists() {
        let mut tokenizer = if is_diffusion_gemma_model(model_path) {
            build_diffusion_gemma_tokenizer(&tokenizer_json_path)?
        } else {
            tokenizers::Tokenizer::from_file(&tokenizer_json_path)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        ensure_bos_post_processor(&mut tokenizer, model_path);
        return Ok(MlxcelTokenizer::HuggingFace(tokenizer));
    }

    // Fall back to SentencePiece tokenizer.model
    let tokenizer_model_path = model_path.join("tokenizer.model");
    if tokenizer_model_path.exists() {
        let processor = SentencePieceProcessor::open(&tokenizer_model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.model: {}", e))?;

        let bos_id = processor.bos_id();

        let (special_tokens, added_token_contents, add_bos) = parse_special_tokens(model_path);

        let sp_tokenizer = SentencePieceTokenizer::new(
            processor,
            special_tokens,
            added_token_contents,
            bos_id,
            add_bos,
        );
        return Ok(MlxcelTokenizer::SentencePiece(sp_tokenizer));
    }

    // Fall back to tiktoken (.tiktoken files)
    if let Some(tiktoken_path) = find_tiktoken_file(model_path) {
        let tokenizer = TiktokenTokenizer::from_file(&tiktoken_path, model_path)?;
        return Ok(MlxcelTokenizer::Tiktoken(tokenizer));
    }

    // Fall back to PLaMo's `tokenizer.jsonl` (a Unigram vocab shipped instead
    // of tokenizer.json / tokenizer.model; see build_plamo_tokenizer).
    if model_path.join("tokenizer.jsonl").exists() {
        return Ok(MlxcelTokenizer::HuggingFace(build_plamo_tokenizer(
            model_path,
        )?));
    }

    // Fall back to the GPT-2-style slow-tokenizer pair (`vocab.json` +
    // `merges.txt`) that Qwen2-family repositories ship when they never
    // exported a fast tokenizer.
    if is_qwen2_slow_tokenizer_dir(model_path) {
        return Ok(MlxcelTokenizer::HuggingFace(build_qwen2_bpe_tokenizer(
            model_path,
        )?));
    }

    if let Some(repo_id) = remote_tokenizer_repo_for_model(model_path) {
        let tokenizer = download_remote_tokenizer(repo_id).map_err(|err| {
            anyhow::anyhow!(
                "Failed to resolve fallback tokenizer {} for {:?}: {}",
                repo_id,
                model_path,
                err
            )
        })?;
        return Ok(MlxcelTokenizer::HuggingFace(tokenizer));
    }

    Err(anyhow::anyhow!(
        "No tokenizer found in {:?} (tried tokenizer.json, tokenizer.model, *.tiktoken, tokenizer.jsonl, and vocab.json + merges.txt)",
        model_path
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        MlxcelTokenizer, build_qwen2_bpe_tokenizer, is_qwen2_slow_tokenizer_dir,
        read_added_tokens_sorted, remote_tokenizer_override_for_model,
        remote_tokenizer_repo_for_model, remote_tokenizer_repo_for_model_type,
    };
    use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};

    /// Write a minimal Qwen2 slow-tokenizer directory: a byte-level `vocab.json`
    /// covering the ASCII letters plus a `Ġ` (space) marker, one merge, and two
    /// added tokens above the base vocab.
    fn write_qwen2_slow_tokenizer_dir(dir: &std::path::Path, tokenizer_class: &str) {
        std::fs::create_dir_all(dir).unwrap();

        // Byte-level alphabet: the tokens the ByteLevel pre-tokenizer emits for
        // " hi" are "Ġ", "h", "i" (and "Ġh" once the merge applies).
        let mut vocab = serde_json::Map::new();
        let mut next = 0u64;
        for token in ["Ġ", "h", "i", "!", "Ġh", "a", "b"] {
            vocab.insert(token.to_string(), serde_json::json!(next));
            next += 1;
        }
        let base = next as u32;
        std::fs::write(
            dir.join("vocab.json"),
            serde_json::to_string(&serde_json::Value::Object(vocab)).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\nĠ h\n").unwrap();

        let config = serde_json::json!({
            "tokenizer_class": tokenizer_class,
            "add_prefix_space": false,
            "added_tokens_decoder": {
                base.to_string(): {
                    "content": "<|im_start|>",
                    "single_word": false, "lstrip": false, "rstrip": false,
                    "normalized": false, "special": true
                },
                (base + 1).to_string(): {
                    "content": "<0>",
                    "single_word": false, "lstrip": false, "rstrip": false,
                    "normalized": false, "special": true
                }
            }
        });
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();
    }

    /// Write a Qwen2 slow-tokenizer directory with an explicit `vocab.json` id
    /// assignment and an explicit `added_tokens_decoder`, for the cases that
    /// need to control both rather than take the well-formed defaults above.
    fn write_qwen2_dir_with_vocab(
        dir: &std::path::Path,
        vocab_pairs: &[(&str, u32)],
        added_tokens_decoder: serde_json::Value,
    ) {
        std::fs::create_dir_all(dir).unwrap();

        let mut vocab = serde_json::Map::new();
        for (content, id) in vocab_pairs {
            vocab.insert((*content).to_string(), serde_json::json!(id));
        }
        std::fs::write(
            dir.join("vocab.json"),
            serde_json::to_string(&serde_json::Value::Object(vocab)).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\nĠ h\n").unwrap();

        let config = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "add_prefix_space": false,
            "added_tokens_decoder": added_tokens_decoder,
        });
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("mlxcel-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn detects_a_qwen2_slow_tokenizer_directory() {
        let dir = temp_dir("qwen2-slow");
        write_qwen2_slow_tokenizer_dir(&dir, "Qwen2Tokenizer");
        assert!(is_qwen2_slow_tokenizer_dir(&dir));

        // A different family that happens to ship the same GPT-2 file pair must
        // not be tokenized with Qwen2's rules.
        let other = temp_dir("gpt2-slow");
        write_qwen2_slow_tokenizer_dir(&other, "GPT2Tokenizer");
        assert!(!is_qwen2_slow_tokenizer_dir(&other));

        // Missing merges.txt disqualifies the directory.
        let partial = temp_dir("qwen2-partial");
        write_qwen2_slow_tokenizer_dir(&partial, "Qwen2Tokenizer");
        std::fs::remove_file(partial.join("merges.txt")).unwrap();
        assert!(!is_qwen2_slow_tokenizer_dir(&partial));

        for path in [dir, other, partial] {
            let _ = std::fs::remove_dir_all(path);
        }
    }

    #[test]
    fn builds_a_qwen2_tokenizer_from_vocab_and_merges() {
        let dir = temp_dir("qwen2-build");
        write_qwen2_slow_tokenizer_dir(&dir, "Qwen2Tokenizer");

        let tokenizer = build_qwen2_bpe_tokenizer(&dir).expect("build");

        // Added tokens must land on the ids the checkpoint declares, because
        // `tokenizers` assigns them sequentially from the base vocab size.
        assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(7));
        assert_eq!(tokenizer.token_to_id("<0>"), Some(8));

        // The merge `Ġ h` applies, so " hi" is ["Ġh", "i"] and not ["Ġ","h","i"].
        let encoded = tokenizer.encode(" hi", false).expect("encode");
        assert_eq!(encoded.get_tokens(), &["Ġh".to_string(), "i".to_string()]);

        // ByteLevel decoding restores the original text, including the space.
        let decoded = tokenizer.decode(encoded.get_ids(), false).expect("decode");
        assert_eq!(decoded, " hi");

        // An added token is matched whole rather than split into base pieces.
        let encoded = tokenizer.encode("<0>hi", false).expect("encode");
        assert_eq!(encoded.get_ids().first(), Some(&8));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_added_token_ids_that_do_not_continue_the_base_vocab() {
        let dir = temp_dir("qwen2-badids");
        write_qwen2_slow_tokenizer_dir(&dir, "Qwen2Tokenizer");
        // Renumber the added tokens so they leave a gap above the base vocab.
        let config = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "added_tokens_decoder": {
                "99": { "content": "<|im_start|>", "special": true, "normalized": false }
            }
        });
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let err = build_qwen2_bpe_tokenizer(&dir).expect_err("must refuse a shifted id");
        assert!(
            err.to_string().contains("declares 99"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn added_tokens_decoder_defaults_special_to_false() {
        // HuggingFace's `AddedToken` defaults `special=False`, so an entry that
        // omits the key must not be registered special. Defaulting the other
        // way is unrecoverable: `AddedVocabulary`'s special set is insert-only
        // (issue #778), so the token would be stripped from every
        // `skip_special_tokens` decode with no way to demote it afterwards.
        let dir = temp_dir("qwen2-special-default");
        write_qwen2_slow_tokenizer_dir(&dir, "Qwen2Tokenizer");
        let config = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "added_tokens_decoder": {
                "7": { "content": "<|im_start|>", "normalized": false, "special": true },
                // No `special` key at all: this is the default under test.
                "8": { "content": "<0>", "normalized": false }
            }
        });
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let parsed = read_added_tokens_sorted(&dir).expect("read added tokens");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, 7);
        assert!(parsed[0].1.special, "an explicit special:true must survive");
        assert_eq!(parsed[1].0, 8);
        assert!(
            !parsed[1].1.special,
            "a missing `special` key must default to non-special"
        );

        let tokenizer = build_qwen2_bpe_tokenizer(&dir).expect("build");
        let decoder = tokenizer.get_added_tokens_decoder();
        assert!(decoder.get(&7).expect("<|im_start|> entry").special);
        assert!(!decoder.get(&8).expect("<0> entry").special);

        // The consequence that matters: the real control token is stripped from
        // a skip-special decode while the content-bearing token survives.
        assert_eq!(tokenizer.decode(&[7, 8], true).expect("decode"), "<0>");
        assert_eq!(
            tokenizer.decode(&[7, 8], false).expect("decode"),
            "<|im_start|><0>"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn added_tokens_json_fallback_tokens_are_not_special() {
        // `added_tokens.json` is a flat `content -> id` map that carries no
        // flags, so nothing there justifies marking an entry special. Forcing
        // `special = true` here would silently strip every added token of a
        // Qwen2-family checkpoint that ships only this file.
        let dir = temp_dir("qwen2-added-tokens-json");
        write_qwen2_slow_tokenizer_dir(&dir, "Qwen2Tokenizer");
        // Drop `added_tokens_decoder` so the fallback branch is the one taken.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("added_tokens.json"),
            r#"{"<|im_start|>":7,"<0>":8}"#,
        )
        .unwrap();

        let parsed = read_added_tokens_sorted(&dir).expect("read added tokens");
        assert_eq!(parsed.len(), 2);
        assert!(parsed.iter().all(|(_, token)| !token.special));
        // `normalized(false)` still applies, which is what keeps `<0>` intact
        // through the NFC normalizer.
        assert!(parsed.iter().all(|(_, token)| !token.normalized));

        let tokenizer = build_qwen2_bpe_tokenizer(&dir).expect("build");
        let decoder = tokenizer.get_added_tokens_decoder();
        assert!(!decoder.get(&7).expect("<|im_start|> entry").special);
        assert!(!decoder.get(&8).expect("<0> entry").special);
        // Nothing was registered special, so a skip-special decode is lossless.
        assert_eq!(
            tokenizer.decode(&[7, 8], true).expect("decode"),
            "<|im_start|><0>"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_a_vocab_json_whose_ids_are_not_dense() {
        // Five entries but id 4 unused, so `COLLIDE` sits on 5.
        // `BPE::get_vocab_size()` reports the entry count regardless, so
        // `tokenizers` would assign the first added token id 5 as well: a
        // checkpoint declaring 5 passes the contiguity check while `COLLIDE`
        // and the added token end up sharing the id and `decode([5])` stops
        // returning `COLLIDE`. The starting id cannot be steered, so the only
        // sound answer is to refuse the file.
        let dir = temp_dir("qwen2-sparse-vocab");
        write_qwen2_dir_with_vocab(
            &dir,
            &[("Ġ", 0), ("h", 1), ("Ġh", 2), ("i", 3), ("COLLIDE", 5)],
            serde_json::json!({
                "5": { "content": "<x>", "special": true, "normalized": false }
            }),
        );

        let err = build_qwen2_bpe_tokenizer(&dir).expect_err("must refuse a sparse vocab.json");
        let message = err.to_string();
        assert!(
            message.contains("is not densely numbered"),
            "unexpected error: {err}"
        );
        assert!(
            message.contains("id 4 is unused"),
            "the gap must be named: {err}"
        );
        assert!(
            message.contains("\"COLLIDE\" claims id 5"),
            "the stray entry must be named: {err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_a_vocab_json_that_assigns_one_id_twice() {
        // A duplicate leaves the same hole a gap does: four entries covering
        // only three ids means id 3 is free for an added token to alias onto.
        let dir = temp_dir("qwen2-duplicate-vocab-id");
        write_qwen2_dir_with_vocab(
            &dir,
            &[("Ġ", 0), ("h", 1), ("Ġh", 2), ("i", 2)],
            serde_json::json!({}),
        );

        let err = build_qwen2_bpe_tokenizer(&dir).expect_err("must refuse a duplicated id");
        let message = err.to_string();
        assert!(
            message.contains("assigns id 2 to both"),
            "unexpected error: {err}"
        );
        assert!(
            message.contains("\"i\"") && message.contains("Ġh"),
            "both colliding contents must be named: {err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_dense_vocab_registers_every_added_token_at_its_declared_id() {
        // The happy path the density check must not break: a well-formed
        // checkpoint still builds, and each added token answers to exactly the
        // id its `added_tokens_decoder` declares in both directions.
        let dir = temp_dir("qwen2-dense-vocab");
        let declared = [
            (7u32, "<|im_start|>"),
            (8, "<|im_end|>"),
            (9, "<0>"),
            (10, "<1>"),
            (11, "<2>"),
            (12, "<3>"),
        ];
        let mut decoder = serde_json::Map::new();
        for (id, content) in declared {
            decoder.insert(
                id.to_string(),
                serde_json::json!({
                    "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                }),
            );
        }
        write_qwen2_dir_with_vocab(
            &dir,
            &[
                ("Ġ", 0),
                ("h", 1),
                ("i", 2),
                ("!", 3),
                ("Ġh", 4),
                ("a", 5),
                ("b", 6),
            ],
            serde_json::Value::Object(decoder),
        );

        let tokenizer = build_qwen2_bpe_tokenizer(&dir).expect("build");
        for (id, content) in declared {
            assert_eq!(
                tokenizer.token_to_id(content),
                Some(id),
                "{content:?} must answer to its declared id"
            );
            assert_eq!(
                tokenizer.decode(&[id], false).expect("decode"),
                content,
                "id {id} must decode back to {content:?}"
            );
        }

        // The base vocab is untouched: nothing was aliased onto its ids.
        assert_eq!(tokenizer.token_to_id("Ġh"), Some(4));
        assert_eq!(tokenizer.token_to_id("b"), Some(6));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn mixed_special_and_plain_added_tokens_land_on_their_declared_ids() {
        // Added tokens are registered a run at a time (one call per consecutive
        // block of equal `special`) instead of one call per token, because each
        // call rebuilds the added-vocabulary automatons from scratch. Ids are
        // handed out in the order tokens are passed, so regrouping them into a
        // specials batch plus a normals batch would renumber every token that
        // crosses a boundary. This sequence alternates six times, which is the
        // shape that would break under that regrouping.
        let dir = temp_dir("qwen2-mixed-added");
        let declared: [(u32, &str, bool); 9] = [
            (7, "<a>", true),
            (8, "<b>", false),
            (9, "<c>", true),
            (10, "<d>", true),
            (11, "<e>", false),
            (12, "<f>", false),
            (13, "<g>", true),
            (14, "<h>", false),
            (15, "<i>", true),
        ];
        let mut decoder = serde_json::Map::new();
        for (id, content, special) in declared {
            decoder.insert(
                id.to_string(),
                serde_json::json!({
                    "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": special
                }),
            );
        }
        write_qwen2_dir_with_vocab(
            &dir,
            &[
                ("Ġ", 0),
                ("h", 1),
                ("i", 2),
                ("!", 3),
                ("Ġh", 4),
                ("a", 5),
                ("b", 6),
            ],
            serde_json::Value::Object(decoder),
        );

        let tokenizer = build_qwen2_bpe_tokenizer(&dir).expect("build");
        let added = tokenizer.get_added_tokens_decoder();
        for (id, content, special) in declared {
            assert_eq!(
                tokenizer.token_to_id(content),
                Some(id),
                "{content:?} must keep its declared id across the run boundary"
            );
            let entry = added.get(&id).expect("added token entry");
            assert_eq!(entry.content, content, "id {id} must hold {content:?}");
            assert_eq!(
                entry.special, special,
                "{content:?} must keep its declared special flag"
            );
        }

        // The flags survive where they are observable: a skip-special decode
        // drops exactly the special half of the sequence.
        let ids: Vec<u32> = declared.iter().map(|(id, _, _)| *id).collect();
        let kept: String = declared
            .iter()
            .filter(|(_, _, special)| !special)
            .map(|(_, content, _)| *content)
            .collect();
        assert_eq!(tokenizer.decode(&ids, true).expect("decode"), kept);
        assert_eq!(
            tokenizer.decode(&ids, false).expect("decode"),
            declared.iter().map(|(_, c, _)| *c).collect::<String>()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn remote_tokenizer_repo_for_model_type_matches_moondream3() {
        assert_eq!(
            remote_tokenizer_repo_for_model_type("moondream3"),
            Some("moondream/starmie-v1")
        );
        assert_eq!(remote_tokenizer_repo_for_model_type("llama"), None);
    }

    #[test]
    fn remote_tokenizer_repo_for_model_reads_config_json_model_type() {
        let temp_dir =
            std::env::temp_dir().join(format!("mlxcel-tokenizer-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::write(
            temp_dir.join("config.json"),
            r#"{"model_type":"moondream3"}"#,
        )
        .unwrap();

        assert_eq!(
            remote_tokenizer_repo_for_model(&temp_dir),
            Some("moondream/starmie-v1")
        );

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    // ------------------------------------------------------------------
    // Starmie-era moondream2 tokenizer override
    // ------------------------------------------------------------------

    fn override_test_dir(files: &[(&str, &str)]) -> std::path::PathBuf {
        let temp_dir = std::env::temp_dir().join(format!(
            "mlxcel-tokenizer-override-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        for (name, content) in files {
            std::fs::write(temp_dir.join(name), content).unwrap();
        }
        temp_dir
    }

    #[test]
    fn override_fires_for_starmie_era_moondream2_with_stale_local_tokenizer() {
        // The real 2025-06-21 snapshot shape: model_type moondream1,
        // moondream.py naming the starmie repo, and the STALE legacy GPT-2
        // tokenizer.json next to it. The stale file must be overridden.
        let dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"moondream1"}"#),
            (
                "moondream.py",
                "self.tokenizer = Tokenizer.from_pretrained(\"moondream/starmie-v1\")",
            ),
            ("tokenizer.json", r#"{"model":{"vocab":{"!":0}}}"#),
        ]);
        assert_eq!(
            remote_tokenizer_override_for_model(&dir),
            Some("moondream/starmie-v1")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn override_skipped_when_local_tokenizer_is_already_starmie() {
        let dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"moondream1"}"#),
            (
                "moondream.py",
                "self.tokenizer = Tokenizer.from_pretrained(\"moondream/starmie-v1\")",
            ),
            (
                "tokenizer.json",
                r#"{"added_tokens":[{"id":1,"content":"<|md_reserved_0|>"}]}"#,
            ),
        ]);
        assert_eq!(remote_tokenizer_override_for_model(&dir), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn override_skipped_for_legacy_era_moondream2() {
        // 2025-01-09 .. 2025-04-14 snapshots: the GPT-2 tokenizer in the
        // checkpoint is the correct one, so no override.
        let dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"moondream1"}"#),
            (
                "moondream.py",
                "self.tokenizer = Tokenizer.from_pretrained(\n    \"vikhyatk/moondream2\", revision=\"2025-01-09\"\n)",
            ),
            ("tokenizer.json", r#"{"model":{"vocab":{"!":0}}}"#),
        ]);
        assert_eq!(remote_tokenizer_override_for_model(&dir), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn override_skipped_for_non_moondream2_models() {
        // Even with a starmie-looking moondream.py present, other model types
        // never trigger the moondream2 override.
        let dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"llama"}"#),
            (
                "moondream.py",
                "self.tokenizer = Tokenizer.from_pretrained(\"moondream/starmie-v1\")",
            ),
        ]);
        assert_eq!(remote_tokenizer_override_for_model(&dir), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    // ------------------------------------------------------------------
    // ThinkingMarkers / infer_thinking_markers
    //
    // We can't easily construct full `MlxcelTokenizer` instances backed by
    // real model files inside unit tests, so these cases build minimal HF
    // tokenizers with explicit added vocab. The shape mirrors what the
    // production loader produces for each family:
    //
    // - Qwen3 / Exaone / GLM / Hunyuan / Nemotron-H — `<think>` and
    //   `</think>` registered as added tokens.
    // - longcat — `<longcat_think>` / `</longcat_think>` added tokens.
    // - Gemma 4 — `<|channel>` / `<channel|>` added tokens; the literal
    //   `thought` continuation goes through the BPE encoder, which we seed
    //   with a vocab entry to keep the test deterministic.
    // ------------------------------------------------------------------

    fn mlxcel_with_added(tokens: &[&str]) -> MlxcelTokenizer {
        // Minimal BPE base; the underlying model never produces tokens
        // because the test only inspects added-vocab lookups.
        let mut hf = Tokenizer::new(BPE::default());
        let added: Vec<AddedToken> = tokens
            .iter()
            .map(|s| AddedToken::from(*s, /*special=*/ true))
            .collect();
        hf.add_tokens(&added);
        MlxcelTokenizer::HuggingFace(hf)
    }

    #[test]
    fn infer_thinking_markers_recognizes_single_token_qwen_think_pair() {
        let tok = mlxcel_with_added(&["<think>", "</think>"]);
        let markers = tok.infer_thinking_markers();
        assert!(markers.has_thinking());
        assert_eq!(markers.think_start.as_deref(), Some("<think>"));
        assert_eq!(markers.think_end.as_deref(), Some("</think>"));
        // Single-token markers come back as length-1 sequences.
        assert_eq!(markers.think_start_tokens.as_ref().map(Vec::len), Some(1));
        assert_eq!(markers.think_end_tokens.as_ref().map(Vec::len), Some(1));
        // No tool-call markers were threaded through; halves stay None.
        assert!(!markers.has_tool_calling());
    }

    #[test]
    fn infer_thinking_markers_recognizes_inkling_pair() {
        let tok = mlxcel_with_added(&["<|content_thinking|>", "<|end_message|>"]);
        let markers = tok.infer_thinking_markers();
        assert_eq!(markers.think_start.as_deref(), Some("<|content_thinking|>"));
        assert_eq!(markers.think_end.as_deref(), Some("<|end_message|>"));
        assert_eq!(markers.think_start_tokens.as_ref().map(Vec::len), Some(1));
        assert_eq!(markers.think_end_tokens.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn infer_thinking_markers_recognizes_longcat_pair() {
        let tok = mlxcel_with_added(&["<longcat_think>", "</longcat_think>"]);
        let markers = tok.infer_thinking_markers();
        assert!(markers.has_thinking());
        assert_eq!(markers.think_start.as_deref(), Some("<longcat_think>"));
        assert_eq!(markers.think_end.as_deref(), Some("</longcat_think>"));
        assert_eq!(markers.think_start_tokens.unwrap().len(), 1);
        assert_eq!(markers.think_end_tokens.unwrap().len(), 1);
    }

    #[test]
    fn infer_thinking_markers_prefers_qwen_pair_over_longcat() {
        // Both pairs simultaneously is hypothetical, but the precedence
        // contract must match upstream's THINK_TOKENS list order.
        let tok =
            mlxcel_with_added(&["<think>", "</think>", "<longcat_think>", "</longcat_think>"]);
        let markers = tok.infer_thinking_markers();
        assert_eq!(markers.think_start.as_deref(), Some("<think>"));
        assert_eq!(markers.think_end.as_deref(), Some("</think>"));
    }

    #[test]
    fn infer_thinking_markers_recognizes_multi_token_channel_pair() {
        // Gemma 4 / `<|channel>thought` family: the channel delimiters are
        // single tokens, but the open marker (`<|channel>thought`) is
        // multi-token because `thought` falls through to the BPE encoder.
        // We add `thought` as an added token so the encoder produces a
        // deterministic id sequence.
        let tok = mlxcel_with_added(&["<|channel>", "<channel|>", "thought"]);
        let markers = tok.infer_thinking_markers();
        assert!(markers.has_thinking());
        assert_eq!(markers.think_start.as_deref(), Some("<|channel>thought"));
        assert_eq!(markers.think_end.as_deref(), Some("<channel|>"));
        let start = markers.think_start_tokens.expect("start tokens");
        let end = markers.think_end_tokens.expect("end tokens");
        // Gemma 4's open marker spans at least 2 tokens (`<|channel>` and
        // the `thought` continuation) — explicitly assert the multi-token
        // shape so a future tokenizer change that collapses it back to a
        // single id is caught here.
        assert!(
            start.len() >= 2,
            "<|channel>thought must be a multi-token sequence; got {start:?}"
        );
        assert_eq!(end.len(), 1, "<channel|> must remain single-token");
    }

    #[test]
    fn infer_thinking_markers_returns_default_for_non_thinking_tokenizer() {
        let tok = mlxcel_with_added(&["<|user|>", "<|assistant|>"]);
        let markers = tok.infer_thinking_markers();
        assert!(!markers.has_thinking());
        assert!(markers.think_start.is_none());
        assert!(markers.think_end_tokens.is_none());
    }

    #[test]
    fn infer_thinking_markers_partial_channel_pair_does_not_resolve() {
        // Only the open marker present; the loader must not pretend the
        // pair exists.
        let tok = mlxcel_with_added(&["<|channel>"]);
        assert!(!tok.infer_thinking_markers().has_thinking());

        // And the symmetric case — only the close marker.
        let tok2 = mlxcel_with_added(&["<channel|>"]);
        assert!(!tok2.infer_thinking_markers().has_thinking());
    }

    #[test]
    fn with_tool_call_markers_threads_explicit_pair_through() {
        // Hermes-style tool-call markers (`<tool_call>` / `</tool_call>`)
        // are added tokens in the Qwen-coder family. The caller resolves
        // the tool-parser family separately and passes the canonical
        // strings through `with_tool_call_markers`.
        let tok = mlxcel_with_added(&["<think>", "</think>", "<tool_call>", "</tool_call>"]);
        let markers = tok.infer_thinking_markers();
        let merged = tok.with_tool_call_markers(markers, "<tool_call>", "</tool_call>");
        assert!(merged.has_tool_calling());
        assert_eq!(merged.tool_call_start.as_deref(), Some("<tool_call>"));
        assert_eq!(merged.tool_call_end.as_deref(), Some("</tool_call>"));
        assert_eq!(
            merged.tool_call_start_tokens.as_ref().map(Vec::len),
            Some(1)
        );
        assert_eq!(merged.tool_call_end_tokens.as_ref().map(Vec::len), Some(1));
        // Think markers must survive the merge.
        assert!(merged.has_thinking());
    }

    #[test]
    fn with_tool_call_markers_preserves_input_when_tokenizer_lacks_hf() {
        // SentencePiece path: hf_tokenizer() returns None so the helper
        // must short-circuit and return the input unchanged.
        let tok = MlxcelTokenizer::stub();
        let markers = tok.infer_thinking_markers();
        let merged = tok.with_tool_call_markers(markers.clone(), "<tool_call>", "</tool_call>");
        assert_eq!(merged, markers);
    }

    // -- empty tool_call_end (Mistral-like tokenizers) --------

    #[test]
    fn with_tool_call_markers_empty_end_skips_end_transition() {
        // Mistral-like tokenizers report a non-empty tool_call_start but an
        // empty tool_call_end. The state machine must NOT register an
        // empty-sequence tool→normal transition, and tool_call_end /
        // tool_call_end_tokens must remain None (mirrors upstream mlx-lm
        // PR #1151: `transitions["tool"] = [(te, "normal")] if te else []`
        // and `if te: sequences[te] = tokenizer.tool_call_end`).
        let tok = mlxcel_with_added(&["[TOOL_CALLS]"]);
        let markers = tok.infer_thinking_markers();

        // Pass an empty end string (the Mistral case).
        let merged = tok.with_tool_call_markers(markers, "[TOOL_CALLS]", "");

        // Start marker IS populated (we can still enter tool-call mode).
        assert!(merged.has_tool_calling());
        assert_eq!(merged.tool_call_start.as_deref(), Some("[TOOL_CALLS]"));
        assert!(
            merged
                .tool_call_start_tokens
                .as_ref()
                .is_some_and(|v| !v.is_empty())
        );

        // End marker must NOT be populated (no tool→normal transition).
        assert!(
            merged.tool_call_end.is_none(),
            "tool_call_end must be None when end string is empty; got {:?}",
            merged.tool_call_end
        );
        assert!(
            merged.tool_call_end_tokens.is_none(),
            "tool_call_end_tokens must be None when end string is empty; got {:?}",
            merged.tool_call_end_tokens
        );
    }

    #[test]
    fn with_tool_call_markers_nonempty_end_still_registers_transition() {
        // Regression guard: non-empty end markers continue to work correctly
        // after the Mistral empty-end fix. Both start and end fields must be
        // populated when both strings are non-empty (PR #1151 positive path).
        let tok = mlxcel_with_added(&["<tool_call>", "</tool_call>"]);
        let markers = tok.infer_thinking_markers();
        let merged = tok.with_tool_call_markers(markers, "<tool_call>", "</tool_call>");

        assert!(merged.has_tool_calling());
        assert_eq!(merged.tool_call_start.as_deref(), Some("<tool_call>"));
        assert_eq!(merged.tool_call_end.as_deref(), Some("</tool_call>"));
        assert!(
            merged
                .tool_call_start_tokens
                .as_ref()
                .is_some_and(|v| !v.is_empty())
        );
        assert!(
            merged
                .tool_call_end_tokens
                .as_ref()
                .is_some_and(|v| !v.is_empty())
        );
    }

    // -- find_think_* / rfind_think_* via subseq helpers ----------
    //
    // The `ThinkingMarkers::find_*` / `rfind_*` helpers are the Rust analogue
    // of upstream's `TokenizerWrapper.find_think_start` etc. These tests verify
    // the tokenizer-side wiring: encode a Gemma-4-shaped input, resolve the
    // markers, then locate them inside the encoded sequence.

    // -- Real Gemma 4 tokenizer integration (#[ignore]) -------------------
    //
    // Exercises the actual `mlx-community/gemma-4-e4b-it-8bit` tokenizer
    // shipped in `models/gemma-4-e4b-it-8bit/`. Skipped when the directory
    // is missing so the test suite stays portable; run on demand with
    // `cargo test -- --ignored` against a workspace that has the model
    // downloaded (per `docs/testing.md`).

    #[test]
    #[ignore = "requires models/gemma-4-e4b-it-8bit/; run with --ignored"]
    fn gemma4_real_tokenizer_resolves_multi_token_channel_marker() {
        let model_dir = std::path::Path::new("models/gemma-4-e4b-it-8bit");
        assert!(
            model_dir.exists(),
            "this --ignored test needs the Gemma 4 model under models/"
        );
        let tok = super::load_tokenizer(model_dir).expect("load Gemma 4 tokenizer");
        let markers = tok.infer_thinking_markers();
        assert!(
            markers.has_thinking(),
            "Gemma 4 tokenizer must register a thinking marker pair"
        );
        assert_eq!(markers.think_start.as_deref(), Some("<|channel>thought"));
        assert_eq!(markers.think_end.as_deref(), Some("<channel|>"));
        let start = markers.think_start_tokens.expect("start tokens");
        let end = markers.think_end_tokens.expect("end tokens");
        assert!(
            start.len() >= 2,
            "Gemma 4's <|channel>thought open marker must be multi-token; got len={} ids={:?}",
            start.len(),
            start
        );
        assert_eq!(
            end.len(),
            1,
            "Gemma 4's <channel|> close marker must remain single-token; got ids={end:?}"
        );

        // Confirm the resolved id sequence actually matches the bytes the
        // chat template will emit for the channel priming. Encoding the
        // priming substring directly must produce the same prefix that
        // `infer_thinking_markers` resolved; otherwise the stream filter /
        // thinking-budget tracker would miss real markers.
        let hf = tok.hf_tokenizer().unwrap();
        let direct = hf
            .encode("<|channel>thought", false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(start, direct);
    }

    #[test]
    fn find_think_start_locates_multi_token_channel_marker() {
        let tok = mlxcel_with_added(&["<|channel>", "<channel|>", "thought"]);
        let markers = tok.infer_thinking_markers();
        let start_seq = markers.think_start_tokens.clone().unwrap();

        // Encode a synthetic completion: "<|channel>thought<channel|>"
        let hf = tok.hf_tokenizer().unwrap();
        let body = hf
            .encode("<|channel>thought<channel|>", false)
            .unwrap()
            .get_ids()
            .to_vec();

        // The open-marker subsequence must appear at the start (idx 0).
        assert_eq!(markers.find_think_start(&body, None, None), Some(0));
        // The close-marker subsequence must appear after the open marker.
        let close_idx = markers.find_think_end(&body, None, None).unwrap();
        assert!(close_idx >= start_seq.len());
        // rfind variant returns the same index when there is exactly one
        // occurrence.
        assert_eq!(markers.rfind_think_end(&body, None, None), Some(close_idx));
    }

    // ------------------------------------------------------------------
    // DiffusionGemma tool-parser marker demotion (issue #778)
    //
    // Premise confirmed against the real checkpoint
    // (models/diffusiongemma-26b-a4b-it-4bit/tokenizer.json): all five
    // markers ship as `special: true` added tokens (`<|tool_call>` id 48,
    // `<tool_call|>` id 49, `<|"|>` id 52, `<|channel>` id 100, `<channel|>`
    // id 101), and tokenizer_config.json's `added_tokens_decoder` is empty,
    // so this checkpoint loads through the HuggingFace `tokenizer.json` arm
    // of `load_tokenizer`, not the SentencePiece path.
    // ------------------------------------------------------------------

    use super::{
        DIFFUSION_GEMMA_TOOL_PARSER_MARKERS, demote_tool_parser_markers, is_diffusion_gemma_model,
    };

    /// Build a synthetic tokenizer.json shaped like the real DiffusionGemma
    /// checkpoint: the five tool-parser/channel markers as `special: true`
    /// added tokens, plus a handful of plain (non-special) added tokens that
    /// stand in for the literal text a Gemma4-style tool call carries between
    /// the markers. No BPE vocab/merges are needed because every byte of the
    /// test strings is covered by an added token.
    fn diffusion_gemma_style_tokenizer_json() -> String {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<|tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<tool_call|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "<|\"|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 3, "content": "<|channel>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 4, "content": "<channel|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 5, "content": "call:get_weather{location:", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                {"id": 6, "content": "Tokyo", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                {"id": 7, "content": "}", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                {"id": 8, "content": "thought reasoning here", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                {"id": 9, "content": "The weather is sunny.", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            // `Fuse` concatenates decoded token strings with no separator.
            // Without an explicit decoder, the crate's BPE default inserts a
            // space between adjacent tokens, which would make the literal
            // string comparisons below fail for reasons unrelated to the
            // demotion behavior under test.
            "decoder": {"type": "Fuse"},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": {},
                "merges": []
            }
        })
        .to_string()
    }

    /// A synthetic Gemma4-style tool-call completion built entirely from the
    /// added tokens in [`diffusion_gemma_style_tokenizer_json`]; mirrors the
    /// literal string used by `server::tool_calls::parser`'s
    /// `parse_gemma4_format` test.
    const SYNTHETIC_GEMMA4_TOOL_CALL: &str =
        "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";

    #[test]
    fn demote_tool_parser_markers_flips_special_flag_for_present_markers() {
        let mut json: serde_json::Value =
            serde_json::from_str(&diffusion_gemma_style_tokenizer_json()).unwrap();
        let demoted = demote_tool_parser_markers(&mut json);

        // All five markers were present and special, so all five come back.
        let mut demoted_sorted = demoted.clone();
        demoted_sorted.sort();
        let mut expected: Vec<String> = DIFFUSION_GEMMA_TOOL_PARSER_MARKERS
            .iter()
            .map(|s| s.to_string())
            .collect();
        expected.sort();
        assert_eq!(demoted_sorted, expected);

        // The JSON document itself must now carry special: false for each.
        let added_tokens = json["added_tokens"].as_array().unwrap();
        for entry in added_tokens {
            let content = entry["content"].as_str().unwrap();
            if DIFFUSION_GEMMA_TOOL_PARSER_MARKERS.contains(&content) {
                assert_eq!(
                    entry["special"].as_bool(),
                    Some(false),
                    "{content} must be demoted to special:false"
                );
            }
        }
    }

    #[test]
    fn demote_tool_parser_markers_is_a_noop_when_absent() {
        let mut json = serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ]
        });
        let demoted = demote_tool_parser_markers(&mut json);
        assert!(demoted.is_empty());
        // Untouched: the unrelated special token keeps its flag.
        assert_eq!(json["added_tokens"][0]["special"].as_bool(), Some(true));
    }

    #[test]
    fn demote_tool_parser_markers_skips_already_non_special_markers() {
        // A marker that ships special:false already must not be reported as
        // newly demoted (nothing changed).
        let mut json = serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "<|tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
            ]
        });
        let demoted = demote_tool_parser_markers(&mut json);
        assert!(demoted.is_empty());
    }

    #[test]
    fn is_diffusion_gemma_model_matches_both_config_variants() {
        let dir = override_test_dir(&[("config.json", r#"{"model_type":"diffusion_gemma"}"#)]);
        assert!(is_diffusion_gemma_model(&dir));
        let _ = std::fs::remove_dir_all(dir);

        let dir_text =
            override_test_dir(&[("config.json", r#"{"model_type":"diffusion_gemma_text"}"#)]);
        assert!(is_diffusion_gemma_model(&dir_text));
        let _ = std::fs::remove_dir_all(dir_text);

        let dir_other = override_test_dir(&[("config.json", r#"{"model_type":"gemma3"}"#)]);
        assert!(!is_diffusion_gemma_model(&dir_other));
        let _ = std::fs::remove_dir_all(dir_other);

        // No config.json and a config.json without model_type must both resolve
        // to false without panicking, so those checkpoints keep the unchanged
        // `Tokenizer::from_file` path.
        let dir_empty = override_test_dir(&[("placeholder.txt", "x")]);
        assert!(!is_diffusion_gemma_model(&dir_empty));
        let _ = std::fs::remove_dir_all(dir_empty);

        let dir_no_type = override_test_dir(&[("config.json", r#"{"hidden_size":2048}"#)]);
        assert!(!is_diffusion_gemma_model(&dir_no_type));
        let _ = std::fs::remove_dir_all(dir_no_type);
    }

    #[test]
    fn diffusion_gemma_tokenizer_survives_skip_special_decode_round_trip() {
        // Build the tokenizer the same way `build_diffusion_gemma_tokenizer`
        // does: demote before deserializing, never mutate an already-loaded
        // Tokenizer (see the doc comment on `demote_tool_parser_markers` for
        // why post-load `add_tokens` cannot flip the special flag).
        let mut json: serde_json::Value =
            serde_json::from_str(&diffusion_gemma_style_tokenizer_json()).unwrap();
        let demoted = demote_tool_parser_markers(&mut json);
        assert_eq!(demoted.len(), 5);
        let bytes = serde_json::to_vec(&json).unwrap();
        let demoted_tokenizer = Tokenizer::from_bytes(bytes).unwrap();

        let ids = demoted_tokenizer
            .encode(SYNTHETIC_GEMMA4_TOOL_CALL, false)
            .unwrap()
            .get_ids()
            .to_vec();

        // Every marker is still a single atomic id (encode never splits it
        // across the BPE encoder): the id list length must equal the number
        // of added-token pieces the literal string decomposes into.
        assert_eq!(ids.len(), 7, "unexpected token count for {ids:?}");

        // The regression this issue fixes: with skip_special_tokens=true the
        // decoded text must retain the markers (compare to the plain decode
        // to make sure nothing else changed).
        let decoded_plain = demoted_tokenizer.decode(&ids, false).unwrap();
        let decoded_skip_special = demoted_tokenizer.decode(&ids, true).unwrap();
        assert_eq!(decoded_plain, SYNTHETIC_GEMMA4_TOOL_CALL);
        assert_eq!(
            decoded_skip_special, SYNTHETIC_GEMMA4_TOOL_CALL,
            "demoted markers must survive skip_special_tokens=true decode"
        );

        // Sanity control: build the SAME tokenizer WITHOUT demotion and
        // confirm skip_special_tokens=true strips the markers there, so the
        // test would actually fail without the fix (i.e. it is not
        // vacuously true because skip_special_tokens never strips anything
        // in this crate version).
        let undemoted_json: serde_json::Value =
            serde_json::from_str(&diffusion_gemma_style_tokenizer_json()).unwrap();
        let undemoted_tokenizer =
            Tokenizer::from_bytes(serde_json::to_vec(&undemoted_json).unwrap()).unwrap();
        let undemoted_stripped = undemoted_tokenizer.decode(&ids, true).unwrap();
        assert_ne!(
            undemoted_stripped, SYNTHETIC_GEMMA4_TOOL_CALL,
            "control: an un-demoted tokenizer must still strip the special markers"
        );
        assert!(
            !undemoted_stripped.contains("<|tool_call>"),
            "control tokenizer unexpectedly retained a marker: {undemoted_stripped:?}"
        );
    }

    #[test]
    fn diffusion_gemma_tool_call_output_parses_after_demotion() {
        // The decoded text a demoted tokenizer now produces must still parse
        // as a Gemma4-format tool call (mirrors
        // `server::tool_calls::parser::tests::parse_gemma4_format`, whose
        // literal input is reused as `SYNTHETIC_GEMMA4_TOOL_CALL`).
        use crate::server::tool_calls::{ToolCallFormat, parse_tool_calls};
        use crate::server::types::request::{FunctionDefinition, Tool};

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let result = parse_tool_calls(SYNTHETIC_GEMMA4_TOOL_CALL, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
        assert_eq!(result.format, Some(ToolCallFormat::Gemma4));
    }

    #[test]
    fn diffusion_gemma_thinking_channel_survives_demotion() {
        // `infer_thinking_markers` resolves markers by `token_to_id` lookup,
        // never by the special flag, so demotion must not disturb it. Build
        // the demoted tokenizer exactly as `build_diffusion_gemma_tokenizer`
        // does and confirm the channel pair is still recognized.
        let mut json: serde_json::Value =
            serde_json::from_str(&diffusion_gemma_style_tokenizer_json()).unwrap();
        demote_tool_parser_markers(&mut json);
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
        let tok = MlxcelTokenizer::HuggingFace(tokenizer);

        let markers = tok.infer_thinking_markers();
        assert!(markers.has_thinking());
        assert_eq!(markers.think_start.as_deref(), Some("<|channel>thought"));
        assert_eq!(markers.think_end.as_deref(), Some("<channel|>"));

        // Text-based extraction (the server-side reasoning/tool-call parser)
        // must also still strip the reasoning block from decoded text now
        // that the channel markers are non-special: a demoted tokenizer's
        // decode(..., skip_special_tokens=true) keeps the markers in the
        // string, and `parse_tool_calls`'s internal `strip_thinking` pass
        // removes the whole `<|channel>...<channel|>` span by literal text
        // match (it never looked at the special flag either).
        let hf = tok.hf_tokenizer().unwrap();
        let full_text = "<|channel>thought reasoning here<channel|>The weather is sunny.";
        let ids = hf.encode(full_text, false).unwrap().get_ids().to_vec();
        let decoded = hf.decode(&ids, true).unwrap();
        assert_eq!(
            decoded, full_text,
            "demoted channel markers must survive skip_special_tokens=true decode"
        );

        let result = crate::server::tool_calls::parse_tool_calls(&decoded, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "The weather is sunny.");
    }

    #[test]
    fn load_tokenizer_demotes_markers_only_for_diffusion_gemma_model_type() {
        // End-to-end through the real `load_tokenizer` entry point: a
        // diffusion_gemma config.json triggers demotion, so skip-special
        // decode retains the markers.
        let dg_dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"diffusion_gemma"}"#),
            ("tokenizer.json", &diffusion_gemma_style_tokenizer_json()),
        ]);
        let dg_tokenizer = super::load_tokenizer(&dg_dir).expect("load diffusion_gemma tokenizer");
        let ids = dg_tokenizer
            .encode(SYNTHETIC_GEMMA4_TOOL_CALL, false)
            .expect("encode");
        let decoded = dg_tokenizer.decode(&ids, true).expect("decode");
        assert_eq!(
            decoded, SYNTHETIC_GEMMA4_TOOL_CALL,
            "diffusion_gemma load_tokenizer must demote the markers"
        );
        let _ = std::fs::remove_dir_all(dg_dir);

        // Same tokenizer.json, but a non-diffusion_gemma model_type: the
        // markers must be left exactly as the checkpoint shipped them
        // (still special, still stripped by skip_special_tokens=true).
        let other_dir = override_test_dir(&[
            ("config.json", r#"{"model_type":"gemma3"}"#),
            ("tokenizer.json", &diffusion_gemma_style_tokenizer_json()),
        ]);
        let other_tokenizer = super::load_tokenizer(&other_dir).expect("load other tokenizer");
        let other_ids = other_tokenizer
            .encode(SYNTHETIC_GEMMA4_TOOL_CALL, false)
            .expect("encode");
        let other_decoded = other_tokenizer.decode(&other_ids, true).expect("decode");
        assert_ne!(
            other_decoded, SYNTHETIC_GEMMA4_TOOL_CALL,
            "a non-diffusion_gemma model must be unaffected by the demotion gate"
        );
        assert!(
            !other_decoded.contains("<|tool_call>"),
            "non-diffusion_gemma model unexpectedly retained a marker: {other_decoded:?}"
        );
        let _ = std::fs::remove_dir_all(other_dir);
    }

    #[test]
    fn real_diffusion_gemma_checkpoint_demotes_and_retains_markers() {
        // Exercises the production `load_tokenizer` path against the actual
        // published checkpoint's tokenizer.json (not the synthetic JSON
        // above), confirming the premise (all five markers ship
        // `special: true`) and the fix (they now survive skip-special
        // decode). Skips gracefully when the checkpoint is absent, matching
        // the PLaMo integration tests below.
        let model_dir = std::path::Path::new("models/diffusiongemma-26b-a4b-it-4bit");
        if !model_dir.exists() {
            eprintln!(
                "skipping real_diffusion_gemma_checkpoint_demotes_and_retains_markers: \
                 {model_dir:?} is absent"
            );
            return;
        }

        let tok = super::load_tokenizer(model_dir).expect("load DiffusionGemma tokenizer");
        let hf = tok
            .hf_tokenizer()
            .expect("DiffusionGemma loads via the HF tokenizer.json arm");

        for marker in DIFFUSION_GEMMA_TOOL_PARSER_MARKERS {
            let id = hf
                .token_to_id(marker)
                .unwrap_or_else(|| panic!("real checkpoint is missing marker {marker:?}"));
            let decoded = hf.decode(&[id], true).expect("decode");
            assert_eq!(
                decoded, marker,
                "marker {marker:?} (id {id}) must survive skip_special_tokens=true decode \
                 on the real checkpoint"
            );
        }
    }

    // -- Real PLaMo tokenizer integration ---------------------------------
    //
    // PLaMo 2 ships a `tokenizer.jsonl` Unigram vocab and a custom
    // `tokenization_plamo.py`, not a tokenizer.json. `build_plamo_tokenizer`
    // reconstructs the SentencePiece-style Unigram + byte-fallback behavior on
    // top of the `tokenizers` crate. These cases load the real vocab from
    // `models/plamo-2-1b/` and assert exact parity against id sequences and
    // decoded strings captured from PlamoTokenizer's own Aho-Corasick encode.
    // The tokenizer is CPU-only (no MLX/Metal), so this runs in the normal lib
    // test suite; it skips gracefully when the checkpoint is absent.

    /// `(input text, expected token ids)` pairs captured from the reference
    /// PlamoTokenizer's own Aho-Corasick encode.
    const PLAMO_REFERENCE_CASES: &[(&str, &[u32])] = &[
        (
            "The capital of France is Paris.",
            &[1097, 3849, 1079, 7148, 45119, 10188, 46],
        ),
        (
            "def foo(x):\n    return x+1",
            &[1276, 23154, 40, 120, 1189, 45059, 1094, 376, 43, 49],
        ),
        ("東京は日本の首都です。", &[47361, 64657, 58577, 47134]),
        ("Hello world", &[6721, 1462]),
        ("  spaces", &[288, 18541]),
    ];

    #[test]
    fn plamo_tokenizer_matches_reference_encodings() {
        let model_dir = std::path::Path::new("models/plamo-2-1b");
        if !model_dir.exists() {
            eprintln!(
                "skipping plamo_tokenizer_matches_reference_encodings: models/plamo-2-1b is absent"
            );
            return;
        }
        let tok = super::load_tokenizer(model_dir).expect("load PLaMo tokenizer");

        for (text, expected) in PLAMO_REFERENCE_CASES {
            let ids = tok.encode(text, false).expect("encode");
            assert_eq!(
                &ids, expected,
                "encode mismatch for {text:?}: got {ids:?}, want {expected:?}"
            );
        }
    }

    #[test]
    fn plamo_tokenizer_round_trips_decode() {
        let model_dir = std::path::Path::new("models/plamo-2-1b");
        if !model_dir.exists() {
            eprintln!("skipping plamo_tokenizer_round_trips_decode: models/plamo-2-1b is absent");
            return;
        }
        let tok = super::load_tokenizer(model_dir).expect("load PLaMo tokenizer");

        for (text, _) in PLAMO_REFERENCE_CASES {
            let ids = tok.encode(text, false).expect("encode");
            let decoded = tok.decode(&ids, false).expect("decode");
            assert_eq!(
                &decoded, text,
                "decode round-trip mismatch for {text:?}: got {decoded:?}"
            );
        }
    }
}
