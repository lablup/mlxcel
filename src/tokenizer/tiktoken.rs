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

//! Tiktoken BPE tokenizer implementation.
//!
//! Supports loading `.tiktoken` vocabulary files (base64-encoded byte sequences
//! with integer ranks) and performing byte-level BPE encoding/decoding.
//! Used by: HunYuan models, Kimi K3
//!
//! Two families share this loader, selected by [`TiktokenFamily::detect`] from
//! the checkpoint's `config.json`. They differ in the pre-tokenization pattern
//! and in how the control-token block above the BPE ranks is named; everything
//! below that (rank table, BPE merge loop, decoding) is shared.
//!
//! The Kimi K3 pre-tokenization pattern and its 256-entry control-token block
//! are derived from `tokenization_kimi.py` in moonshotai/Kimi-K3
//! (<https://huggingface.co/moonshotai/Kimi-K3/blob/main/tokenization_kimi.py>).

use anyhow::Result;
use base64::Engine;
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// Pre-tokenization regex pattern used by HunYuan's tiktoken tokenizer.
/// Matches contractions, letter sequences, digits, punctuation runs, and whitespace.
const HUNYUAN_PAT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenization regex pattern used by Kimi K3, verbatim from
/// `TikTokenTokenizer.pat_str` in the checkpoint's `tokenization_kimi.py`
/// (alternatives joined with `|`, in the order the reference lists them).
///
/// `&&[^\p{Han}]` is a character-class intersection: the letter runs exclude
/// Han so the first alternative owns every CJK ideograph run. `regex-syntax`
/// parses both the intersection and the nested negated class, and `fancy_regex`
/// supplies the `(?!\S)` lookahead, which is the same split Python's `tiktoken`
/// makes (it runs this pattern through the Rust `fancy_regex` crate as well).
const KIMI_K3_PAT: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Number of control tokens Kimi K3 reserves above the BPE ranks
/// (`TikTokenTokenizer.num_reserved_special_tokens`).
const KIMI_K3_RESERVED_CONTROL_TOKENS: u32 = 256;

/// Chunk width the reference tokenizer feeds to tiktoken at once
/// (`_encode_text_piece::TIKTOKEN_MAX_ENCODE_CHARS`). Measured in characters,
/// not bytes, because the Python slice is a character slice.
const TIKTOKEN_MAX_ENCODE_CHARS: usize = 400_000;

/// Longest run of consecutive whitespace (or consecutive non-whitespace)
/// characters the reference tokenizer hands to one `encode` call
/// (`_encode_text_piece::MAX_NO_WHITESPACES_CHARS`).
const MAX_NO_WHITESPACE_CHARS: usize = 25_000;

/// Which checkpoint family a `.tiktoken` vocabulary belongs to.
///
/// The family selects the pre-tokenization pattern and the naming of the
/// control-token block that sits above the BPE ranks. Anything that is not
/// recognized as Kimi K3 stays [`TiktokenFamily::HunYuan`], which is the
/// behavior every `.tiktoken` checkpoint had before Kimi K3 was added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TiktokenFamily {
    /// HunYuan: 5 named specials plus `<|extra_N|>` filler, HunYuan pattern.
    HunYuan,
    /// Kimi K3: 256 control tokens named from `tokenizer_config.json`
    /// (`<|reserved_token_N|>` for the unnamed ones), K3 pattern.
    KimiK3,
}

impl TiktokenFamily {
    /// Resolve the family from the checkpoint directory.
    ///
    /// `model_type: "kimi_k3"` is the multimodal wrapper config; the text
    /// backbone alone declares `kimi_linear`, which is only a K3 tiktoken
    /// vocabulary when the directory actually ships `tiktoken.model` (a
    /// `kimi_linear` checkpoint converted with a `tokenizer.json` never reaches
    /// this loader at all).
    pub fn detect(model_path: &Path) -> Self {
        match super::read_config_model_type(model_path).as_deref() {
            Some("kimi_k3") => Self::KimiK3,
            Some("kimi_linear") if model_path.join("tiktoken.model").exists() => Self::KimiK3,
            _ => Self::HunYuan,
        }
    }

    /// The pre-tokenization pattern this family splits text with.
    pub fn pattern(self) -> &'static str {
        match self {
            Self::HunYuan => HUNYUAN_PAT,
            Self::KimiK3 => KIMI_K3_PAT,
        }
    }
}

/// The Kimi K3 XTML structural control token ids, resolved once from the
/// loaded vocabulary.
///
/// `Copy` so the chat renderer can hold it by value in its hot path. Produced
/// only for [`TiktokenFamily::KimiK3`]; every other tokenizer reports `None`,
/// which is what downstream code keys the native renderer off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KimiK3ControlIds {
    /// `<|open|>` — opens an XTML tag.
    pub open: u32,
    /// `<|close|>` — opens a closing XTML tag.
    pub close: u32,
    /// `<|sep|>` — terminates a tag header.
    pub sep: u32,
    /// `<|end_of_msg|>` — ends one message; also the generation stop id.
    pub end_of_msg: u32,
    /// `[BOS]`. Never prepended by the renderer or by `encode`.
    pub bos: u32,
    /// `[EOS]`. Distinct from `end_of_msg`, which is what generation stops on.
    pub eos: u32,
}

pub struct TiktokenTokenizer {
    /// Maps byte sequences to their token IDs (ranks)
    encoder: HashMap<Vec<u8>, u32>,
    /// Maps token IDs back to byte sequences
    decoder: HashMap<u32, Vec<u8>>,
    /// Maps special token strings to their IDs
    special_encoder: HashMap<String, u32>,
    /// Maps special token IDs back to strings
    special_decoder: HashMap<u32, String>,
    /// Special tokens sorted by length descending for greedy matching
    special_tokens_sorted: Vec<(String, u32)>,
    /// Pre-tokenization regex
    pat: Regex,
    /// Which family's pattern and control-token naming this instance uses.
    family: TiktokenFamily,
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` BPE file and tokenizer config.
    ///
    /// The `.tiktoken` file format: each line contains `<base64_token> <rank>`.
    /// The family is resolved from `config.json` (see
    /// [`TiktokenFamily::detect`]) and decides both the pre-tokenization
    /// pattern and how the control-token block above the ranks is named.
    ///
    /// Under the HunYuan family the block is derived from the vocabulary size
    /// and the special-token table the checkpoint's `tokenizer_class` names:
    /// `QWenTokenizer` selects the QWen table, anything else keeps the HunYuan
    /// one. Two families ship a bare `.tiktoken` file with an empty
    /// `added_tokens_decoder`, so the table is the only thing that fixes their
    /// ids, and the two tables disagree from the second entry onward.
    pub fn from_file(tiktoken_path: &Path, model_path: &Path) -> Result<Self> {
        Self::from_file_with_family(
            tiktoken_path,
            model_path,
            TiktokenFamily::detect(model_path),
        )
    }

    /// [`Self::from_file`] with the family chosen by the caller.
    ///
    /// Exists so tests can load a synthetic vocabulary under either family
    /// without writing a `config.json` beside it.
    pub fn from_file_with_family(
        tiktoken_path: &Path,
        model_path: &Path,
        family: TiktokenFamily,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(tiktoken_path)?;

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token_b64 = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Invalid tiktoken line: missing token"))?;
            let rank_str = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Invalid tiktoken line: missing rank"))?;

            let token_bytes = base64::engine::general_purpose::STANDARD.decode(token_b64)?;
            let rank: u32 = rank_str.parse()?;

            decoder.insert(rank, token_bytes.clone());
            encoder.insert(token_bytes, rank);
        }

        let special_start_id = encoder.len() as u32;

        let mut special_encoder = HashMap::new();
        let mut special_decoder = HashMap::new();

        match family {
            TiktokenFamily::HunYuan => {
                // Build special tokens: same order as the reference tokenizer
                // for the family the checkpoint declares (Python `HYTokenizer`
                // or `QWenTokenizer`).
                let special_token_names = if Self::declares_qwen_tokenizer_class(model_path) {
                    Self::build_qwen_special_token_list()
                } else {
                    Self::build_special_token_list()
                };
                for (i, name) in special_token_names.iter().enumerate() {
                    let id = special_start_id + i as u32;
                    special_encoder.insert(name.clone(), id);
                    special_decoder.insert(id, name.clone());
                }

                // Override with tokenizer_config.json if available
                Self::load_special_tokens_from_config(
                    model_path,
                    &mut special_encoder,
                    &mut special_decoder,
                );
            }
            TiktokenFamily::KimiK3 => {
                Self::build_kimi_k3_control_tokens(
                    model_path,
                    special_start_id,
                    &mut special_encoder,
                    &mut special_decoder,
                );
            }
        }

        let mut special_tokens_sorted: Vec<(String, u32)> = special_encoder
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        special_tokens_sorted.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let pat = Regex::new(family.pattern())?;

        Ok(Self {
            encoder,
            decoder,
            special_encoder,
            special_decoder,
            special_tokens_sorted,
            pat,
            family,
        })
    }

    /// Build Kimi K3's 256-entry control-token block.
    ///
    /// Mirrors `TikTokenTokenizer.__init__`: every id in
    /// `base .. base + 256` is a control token, named from
    /// `tokenizer_config.json`'s `added_tokens_decoder` when that file names
    /// it and `<|reserved_token_{id}|>` otherwise. The reference builds the
    /// same block regardless of each entry's `special` flag, so `<|open|>`,
    /// `<|close|>` and `<|sep|>` (all `special: false` in the config) are
    /// control tokens here too.
    fn build_kimi_k3_control_tokens(
        model_path: &Path,
        base: u32,
        special_encoder: &mut HashMap<String, u32>,
        special_decoder: &mut HashMap<u32, String>,
    ) {
        let named = Self::read_added_tokens_decoder(model_path);
        for id in base..base + KIMI_K3_RESERVED_CONTROL_TOKENS {
            let name = named
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
            special_encoder.insert(name.clone(), id);
            special_decoder.insert(id, name);
        }
    }

    /// `added_tokens_decoder` from `tokenizer_config.json`, as `id -> content`.
    ///
    /// An absent or malformed file yields an empty map, which leaves every K3
    /// control token on its `<|reserved_token_N|>` fallback name rather than
    /// failing the load.
    fn read_added_tokens_decoder(model_path: &Path) -> HashMap<u32, String> {
        let mut out = HashMap::new();
        let config_path = model_path.join("tokenizer_config.json");
        if let Ok(content) = std::fs::read_to_string(&config_path)
            && let Ok(config) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(decoder_map) = config
                .get("added_tokens_decoder")
                .and_then(|v| v.as_object())
        {
            for (id_str, entry) in decoder_map {
                if let (Ok(id), Some(token_content)) = (
                    id_str.parse::<u32>(),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    out.insert(id, token_content.to_string());
                }
            }
        }
        out
    }

    /// Build the standard HunYuan special token list.
    fn build_special_token_list() -> Vec<String> {
        let mut tokens = vec![
            "<|endoftext|>".to_string(),
            "<|startoftext|>".to_string(),
            "<|bos|>".to_string(),
            "<|eos|>".to_string(),
            "<|pad|>".to_string(),
        ];
        for i in 0..205 {
            tokens.push(format!("<|extra_{i}|>"));
        }
        tokens
    }

    /// Build the QWen special token list, in `tokenization_qwen.py` order.
    ///
    /// `SPECIAL_TOKENS = (ENDOFTEXT, IMSTART, IMEND) + EXTRAS` followed by
    /// `IMAGE_ST = (ref/box/quad/img/imgpad tags)`, all numbered from
    /// `len(mergeable_ranks)`. For the 151643-rank `qwen.tiktoken` GOT-OCR 2.0
    /// ships this yields `<|endoftext|>` = 151643, `<|im_start|>` = 151644,
    /// `<|im_end|>` = 151645, `<|extra_0..204|>` = 151646..151850, then
    /// `<ref>` 151851 .. `<imgpad>` 151859, and 151643 + 217 = 151860 is the
    /// `vocab_size` the checkpoint declares.
    ///
    /// The HunYuan table cannot stand in for this: it has `<|startoftext|>`
    /// and `<|bos|>` where QWen has `<|im_start|>` and `<|im_end|>`, so a GOT
    /// checkpoint loaded under it silently gets no id at all for `<|im_end|>`
    /// (the stop token) or for `<imgpad>` (the image placeholder).
    fn build_qwen_special_token_list() -> Vec<String> {
        let mut tokens = vec![
            "<|endoftext|>".to_string(),
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
        ];
        for i in 0..205 {
            tokens.push(format!("<|extra_{i}|>"));
        }
        for tag in [
            "<ref>", "</ref>", "<box>", "</box>", "<quad>", "</quad>", "<img>", "</img>",
            "<imgpad>",
        ] {
            tokens.push(tag.to_string());
        }
        tokens
    }

    /// Whether `tokenizer_config.json` names the QWen tiktoken tokenizer.
    ///
    /// Both the `tokenizer_class` field and the `auto_map.AutoTokenizer` entry
    /// are accepted: GOT-OCR 2.0 sets both, but a conversion that keeps only
    /// the `auto_map` still has to reach the QWen table.
    fn declares_qwen_tokenizer_class(model_path: &Path) -> bool {
        let Ok(content) = std::fs::read_to_string(model_path.join("tokenizer_config.json")) else {
            return false;
        };
        let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
            return false;
        };
        if config
            .get("tokenizer_class")
            .and_then(|v| v.as_str())
            .is_some_and(|class| class == "QWenTokenizer")
        {
            return true;
        }
        config
            .get("auto_map")
            .and_then(|v| v.get("AutoTokenizer"))
            .map(|entry| entry.to_string().contains("QWenTokenizer"))
            .unwrap_or(false)
    }

    /// Load additional special token mappings from tokenizer_config.json.
    fn load_special_tokens_from_config(
        model_path: &Path,
        special_encoder: &mut HashMap<String, u32>,
        special_decoder: &mut HashMap<u32, String>,
    ) {
        let config_path = model_path.join("tokenizer_config.json");
        if let Ok(content) = std::fs::read_to_string(&config_path)
            && let Ok(config) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(decoder_map) = config
                .get("added_tokens_decoder")
                .and_then(|v| v.as_object())
        {
            for (id_str, entry) in decoder_map {
                if let (Ok(id), Some(token_content)) = (
                    id_str.parse::<u32>(),
                    entry.get("content").and_then(|v| v.as_str()),
                ) {
                    special_encoder.insert(token_content.to_string(), id);
                    special_decoder.insert(id, token_content.to_string());
                }
            }
        }
    }

    /// Encode text into token IDs.
    ///
    /// Special-token spellings written into `text` are recognized and emitted
    /// as their ids, which is what a caller passing an already rendered prompt
    /// wants. `add_special_tokens` is ignored: no `.tiktoken` family here
    /// prepends a BOS (Kimi K3's `[BOS]`, id `base + 0`, is never auto-added,
    /// matching the reference tokenizer and the XTML renderer).
    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Split at special token boundaries first
        let segments = self.split_with_special_tokens(text);

        for segment in segments {
            if let Some(&id) = self.special_encoder.get(&segment) {
                result.push(id);
            } else {
                self.encode_text_into(&segment, &mut result)?;
            }
        }

        Ok(result)
    }

    /// Encode without recognizing special-token spellings written into the
    /// input text (`parse_special: false`, #1442).
    ///
    /// This is also the Kimi K3 XTML renderer's text-segment encoder: every
    /// piece of user, tool or attribute text goes through here, so a control
    /// token spelled out in a message body becomes ordinary byte tokens and can
    /// never be injected into the structural stream.
    pub fn encode_without_special_parsing(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_text(text)
    }

    /// Encode `text` with no special-token matching at all.
    ///
    /// Alias of [`Self::encode_without_special_parsing`] under the name the
    /// reference tokenizer uses (`encode(..., allow_special_tokens=False)`).
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        self.encode_text_into(text, &mut result)?;
        Ok(result)
    }

    /// Append the ids of one text run to `out`, applying the family's
    /// bounded-chunk guard first.
    ///
    /// The guard reproduces the reference's own chunking, so it bounds what
    /// one regex sweep sees: at most `TIKTOKEN_MAX_ENCODE_CHARS` characters,
    /// split again at runs of `MAX_NO_WHITESPACE_CHARS`. Both widths are the
    /// reference's, and narrowing them would change the pre-tokenization and
    /// with it the ids, so they are not tuning knobs.
    ///
    /// It does not bound [`Self::bpe_encode`], which is quadratic in the length
    /// of a single piece. A 25 000-character run of punctuation matches the
    /// ` ?[^\s\p{L}\p{N}]+[\r\n]*` alternative as one piece and stays slow. That
    /// is the pre-existing shape of `bpe_encode` (HunYuan reaches it with no
    /// guard at all), not something this family introduces, and fixing it means
    /// replacing the merge loop for every tiktoken checkpoint at once.
    fn encode_text_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        match self.family {
            // Byte-identical to the pre-K3 path: one regex sweep, no chunking.
            TiktokenFamily::HunYuan => self.pretokenize_into(text, out),
            TiktokenFamily::KimiK3 => {
                for chunk in char_chunks(text, TIKTOKEN_MAX_ENCODE_CHARS) {
                    for piece in split_whitespaces_or_nonwhitespaces(chunk, MAX_NO_WHITESPACE_CHARS)
                    {
                        self.pretokenize_into(piece, out)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Pre-tokenize with the family's regex, then BPE-encode each piece.
    fn pretokenize_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.pat.find_iter(text) {
            let m = m.map_err(|e| anyhow::anyhow!("Regex error: {}", e))?;
            let piece = m.as_str().as_bytes();
            if let Some(&id) = self.encoder.get(piece) {
                // Single-token fast path
                out.push(id);
            } else {
                // Apply BPE merges
                out.extend(self.bpe_encode(piece));
            }
        }
        Ok(())
    }

    /// Which family this instance loaded as.
    pub fn family(&self) -> TiktokenFamily {
        self.family
    }

    /// The id of one control token, by its exact spelling.
    ///
    /// Only control tokens resolve here; an ordinary vocabulary entry with the
    /// same spelling does not (use [`Self::token_to_id`] for that).
    pub fn control_id(&self, name: &str) -> Option<u32> {
        self.special_encoder.get(name).copied()
    }

    /// The Kimi K3 XTML control ids, or `None` for any other family.
    ///
    /// This is the predicate downstream code uses for "the native XTML
    /// renderer is active", so it is deliberately all-or-nothing: a K3-family
    /// vocabulary missing any one of the six spellings reports `None` rather
    /// than half a renderer.
    pub fn kimi_k3_control_ids(&self) -> Option<KimiK3ControlIds> {
        if self.family != TiktokenFamily::KimiK3 {
            return None;
        }
        Some(KimiK3ControlIds {
            open: self.control_id("<|open|>")?,
            close: self.control_id("<|close|>")?,
            sep: self.control_id("<|sep|>")?,
            end_of_msg: self.control_id("<|end_of_msg|>")?,
            bos: self.control_id("[BOS]")?,
            eos: self.control_id("[EOS]")?,
        })
    }

    /// The id a vocabulary entry holds, by its exact spelling.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.special_encoder
            .get(token)
            .copied()
            .or_else(|| self.encoder.get(token.as_bytes()).copied())
    }

    /// The exclusive id bound this wrapper can decode (#1485; see
    /// `MlxcelTokenizer::vocab_size`).
    pub fn vocab_size(&self) -> usize {
        let special_bound = self
            .special_decoder
            .keys()
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0);
        self.decoder.len().max(special_bound)
    }

    /// Raw bytes for one token; see `MlxcelTokenizer::token_piece_bytes`.
    ///
    /// The tiktoken vocabulary is byte sequences by construction, so this is
    /// the one backend where no reconstruction is needed.
    pub fn piece_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if let Some(special) = self.special_decoder.get(&id) {
            return Some(special.clone().into_bytes());
        }
        self.decoder.get(&id).cloned()
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();

        for &id in ids {
            if let Some(special) = self.special_decoder.get(&id) {
                if !skip_special_tokens {
                    bytes.extend_from_slice(special.as_bytes());
                }
            } else if let Some(token_bytes) = self.decoder.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
            // Unknown IDs are silently skipped
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Apply byte-pair encoding to a byte sequence using the rank-based algorithm.
    ///
    /// The algorithm repeatedly finds the pair of adjacent tokens with the lowest
    /// rank in the vocabulary and merges them, until no more merges are possible.
    fn bpe_encode(&self, piece: &[u8]) -> Vec<u32> {
        if piece.is_empty() {
            return vec![];
        }
        if piece.len() == 1 {
            // Single byte — must be in the vocabulary (bytes 0-255)
            return self
                .encoder
                .get(piece)
                .map(|&id| vec![id])
                .unwrap_or_default();
        }

        // Start with each byte as its own part
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();

        loop {
            if parts.len() < 2 {
                break;
            }

            // Find the pair with the minimum rank
            let mut min_rank = u32::MAX;
            let mut min_idx = usize::MAX;

            for i in 0..parts.len() - 1 {
                let mut merged = parts[i].clone();
                merged.extend_from_slice(&parts[i + 1]);
                if let Some(&rank) = self.encoder.get(&merged)
                    && rank < min_rank
                {
                    min_rank = rank;
                    min_idx = i;
                }
            }

            if min_idx == usize::MAX {
                break; // No more merges possible
            }

            // Merge the pair at min_idx
            let merged = {
                let mut m = parts[min_idx].clone();
                m.extend_from_slice(&parts[min_idx + 1]);
                m
            };
            parts[min_idx] = merged;
            parts.remove(min_idx + 1);
        }

        // Convert parts to token IDs
        parts
            .iter()
            .filter_map(|p| self.encoder.get(p.as_slice()).copied())
            .collect()
    }

    /// Split text into segments at special token boundaries (greedy longest-match-first).
    fn split_with_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_tokens_sorted.is_empty() {
            return vec![text.to_string()];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;
            for (token, _) in &self.special_tokens_sorted {
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
                for (token, _) in &self.special_tokens_sorted {
                    if let Some(pos) = remaining.find(token.as_str())
                        && pos < next_pos
                    {
                        next_pos = pos;
                    }
                }
                segments.push(remaining[..next_pos].to_string());
                remaining = &remaining[next_pos..];
            }
        }

        segments
    }
}

/// Split `s` into slices of at most `max_chars` characters.
///
/// Mirrors the reference tokenizer's `text[i:i + TIKTOKEN_MAX_ENCODE_CHARS]`
/// loop, which slices by character rather than by byte. An empty input yields
/// one empty slice, matching Python's `range(0, 0, n)` producing no chunk and
/// the caller then encoding nothing.
fn char_chunks(s: &str, max_chars: usize) -> Vec<&str> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut count = 0usize;
    for (idx, _) in s.char_indices() {
        if count == max_chars {
            out.push(&s[start..idx]);
            start = idx;
            count = 0;
        }
        count += 1;
    }
    out.push(&s[start..]);
    out
}

/// Split `s` so no slice holds more than `max_consecutive` consecutive
/// whitespace or consecutive non-whitespace characters.
///
/// Port of `TikTokenTokenizer._split_whitespaces_or_nonwhitespaces`. The run
/// classification uses `char::is_whitespace` (the Unicode `White_Space`
/// property) where Python uses `str.isspace()`; the two disagree only on the
/// C0 separators `U+001C..U+001F`, which cannot change the result unless a
/// single run already exceeds 25000 characters.
fn split_whitespaces_or_nonwhitespaces(s: &str, max_consecutive: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut current_len = 0usize;
    let mut current_is_space = s.chars().next().is_some_and(char::is_whitespace);
    let mut slice_start = 0usize;

    for (idx, ch) in s.char_indices() {
        let is_now_space = ch.is_whitespace();
        if current_is_space != is_now_space {
            current_len = 1;
            current_is_space = is_now_space;
        } else {
            current_len += 1;
            if current_len > max_consecutive {
                out.push(&s[slice_start..idx]);
                slice_start = idx;
                current_len = 1;
            }
        }
    }
    out.push(&s[slice_start..]);
    out
}

#[cfg(test)]
#[path = "tiktoken_tests.rs"]
mod tests;
