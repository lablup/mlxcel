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
//! Used by: HunYuan models

use anyhow::Result;
use base64::Engine;
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// Pre-tokenization regex pattern used by HunYuan's tiktoken tokenizer.
/// Matches contractions, letter sequences, digits, punctuation runs, and whitespace.
const HUNYUAN_PAT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` BPE file and tokenizer config.
    ///
    /// The `.tiktoken` file format: each line contains `<base64_token> <rank>`.
    /// Special tokens are derived from the vocabulary size and the standard
    /// HunYuan special token set.
    pub fn from_file(tiktoken_path: &Path, model_path: &Path) -> Result<Self> {
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

        // Build special tokens: same order as Python HYTokenizer
        let special_token_names = Self::build_special_token_list();
        let mut special_encoder = HashMap::new();
        let mut special_decoder = HashMap::new();

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

        let mut special_tokens_sorted: Vec<(String, u32)> = special_encoder
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        special_tokens_sorted.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let pat = Regex::new(HUNYUAN_PAT)?;

        Ok(Self {
            encoder,
            decoder,
            special_encoder,
            special_decoder,
            special_tokens_sorted,
            pat,
        })
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
    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Split at special token boundaries first
        let segments = self.split_with_special_tokens(text);

        for segment in segments {
            if let Some(&id) = self.special_encoder.get(&segment) {
                result.push(id);
            } else {
                // Pre-tokenize with regex, then BPE encode each piece
                let matches = self.pat.find_iter(&segment);
                for m in matches {
                    let m = m.map_err(|e| anyhow::anyhow!("Regex error: {}", e))?;
                    let piece = m.as_str().as_bytes();

                    if let Some(&id) = self.encoder.get(piece) {
                        // Single-token fast path
                        result.push(id);
                    } else {
                        // Apply BPE merges
                        let ids = self.bpe_encode(piece);
                        result.extend(ids);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Encode without recognizing special-token spellings written into the
    /// input text (`parse_special: false`, #1442).
    pub fn encode_without_special_parsing(&self, text: &str) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        let matches = self.pat.find_iter(text);
        for m in matches {
            let m = m.map_err(|e| anyhow::anyhow!("Regex error: {}", e))?;
            let piece = m.as_str().as_bytes();
            if let Some(&id) = self.encoder.get(piece) {
                result.push(id);
            } else {
                result.extend(self.bpe_encode(piece));
            }
        }
        Ok(result)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_model_path() -> PathBuf {
        PathBuf::from("models/hunyuan-13b")
    }

    fn tiktoken_file() -> PathBuf {
        test_model_path().join("hy.tiktoken")
    }

    #[test]
    #[ignore] // Requires model files
    fn test_load_tiktoken() {
        let tokenizer = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path());
        assert!(
            tokenizer.is_ok(),
            "Failed to load tiktoken: {:?}",
            tokenizer.err()
        );
        let t = tokenizer.unwrap();
        assert!(!t.encoder.is_empty());
        assert!(!t.special_encoder.is_empty());
        assert!(t.special_encoder.contains_key("<|eos|>"));
    }

    #[test]
    #[ignore] // Requires model files
    fn test_encode_decode_roundtrip() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "Hello, world!";
        let ids = t.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_encode_chinese() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "你好，世界";
        let ids = t.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_special_tokens() {
        let t = TiktokenTokenizer::from_file(&tiktoken_file(), &test_model_path()).unwrap();
        let text = "<|eos|>";
        let ids = t.encode(text, false).unwrap();
        assert_eq!(ids.len(), 1);
        let decoded = t.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }
}
