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

mod thinking;
mod tiktoken;

use anyhow::Result;
use hf_hub::api::sync::Api;
use sentencepiece::SentencePieceProcessor;
use std::collections::HashMap;
use std::path::Path;

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
}

impl MlxcelTokenizer {
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
        }
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
        "No tokenizer found in {:?} (tried tokenizer.json, tokenizer.model, *.tiktoken, and tokenizer.jsonl)",
        model_path
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        MlxcelTokenizer, remote_tokenizer_override_for_model, remote_tokenizer_repo_for_model,
        remote_tokenizer_repo_for_model_type,
    };
    use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};

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
