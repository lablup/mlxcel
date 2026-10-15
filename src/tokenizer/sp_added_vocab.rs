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

//! Added-token matching for the SentencePiece backend (#1758).
//!
//! The checkpoints that ship only `tokenizer.model` (IQuest-Coder, Baichuan-M1,
//! ERNIE-4.5, InternLM3, TeleChat3) declare slow-only `transformers` tokenizer
//! classes, and every `added_tokens_decoder` entry is an `AddedToken` in their
//! added vocabulary whether or not it is `special`.
//! `PreTrainedTokenizer.tokenize` splits the input on all of them with a trie
//! before SentencePiece sees any text, so `<tool_call>` (marked
//! `"special": false` on IQuest-Coder) encodes as its one reserved id exactly
//! like `<|im_start|>` (marked special) does. On decode the `special` flag
//! decides what `skip_special_tokens` drops; on encode it decides what
//! `parse_special: false` stops matching, as the fast tokenizer's
//! `encode_special_tokens` and llama.cpp do.
//!
//! [`SpAddedVocab::split`] reproduces that split, including the per-token
//! `lstrip` / `rstrip` / `single_word` handling, so a non-special token that is
//! declared `single_word` does not fire inside a longer word. `normalized` is
//! read but deliberately not acted on: the slow tokenizer only consults it when
//! `do_lower_case` is set, which no SentencePiece checkpoint does, and every
//! added token is matched against the raw text.

use std::collections::HashMap;
use std::path::Path;

use tokenizers::AddedToken;

/// One piece of an input after the added-token split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Piece {
    /// Ordinary text, to be encoded by the SentencePiece model.
    Text(String),
    /// An added token, emitted as its declared id.
    Token(u32),
}

/// The added vocabulary of a SentencePiece checkpoint, special and not.
#[derive(Debug)]
pub(super) struct SpAddedVocab {
    /// Every entry with non-empty content, in id order.
    tokens: Vec<(u32, AddedToken)>,
    /// Indexed by first byte: the indices into `tokens` whose content starts
    /// with it, longest content first, so a scan tries the longest match
    /// first. A flat table keeps the per-byte lookup off the hasher.
    by_first_byte: Vec<Vec<usize>>,
    /// Content to index into `tokens`. A content declared twice resolves to
    /// the higher id, as `_added_tokens_encoder` (built in id order) does.
    by_content: HashMap<String, usize>,
}

impl SpAddedVocab {
    pub(super) fn new(entries: &[(u32, AddedToken)]) -> Self {
        let mut tokens: Vec<(u32, AddedToken)> = entries
            .iter()
            .filter(|(_, token)| !token.content.is_empty())
            .cloned()
            .collect();
        tokens.sort_by_key(|(id, _)| *id);

        let mut by_content = HashMap::new();
        for (index, (_, token)) in tokens.iter().enumerate() {
            by_content.insert(token.content.clone(), index);
        }
        let mut by_first_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];
        for &index in by_content.values() {
            let first = tokens[index].1.content.as_bytes()[0];
            by_first_byte[usize::from(first)].push(index);
        }
        for indices in &mut by_first_byte {
            indices.sort_by_key(|&index| std::cmp::Reverse(tokens[index].1.content.len()));
        }

        Self {
            tokens,
            by_first_byte,
            by_content,
        }
    }

    /// The id an added token holds, by its exact content, special or not.
    pub(super) fn id_of(&self, content: &str) -> Option<u32> {
        self.by_content
            .get(content)
            .map(|&index| self.tokens[index].0)
    }

    /// Whether an entry takes part in matching. With `parse_special` off only
    /// the non-special entries do, which is both `split_special_tokens` in
    /// `transformers`' fast path and llama.cpp's `parse_special: false`: a
    /// control token written into the text stays text, a user-defined token
    /// is still recognized.
    fn matches(&self, index: usize, parse_special: bool) -> bool {
        parse_special || !self.tokens[index].1.special
    }

    /// The matching entry whose content is exactly `piece`, if any.
    fn lookup(&self, piece: &str, parse_special: bool) -> Option<usize> {
        self.by_content
            .get(piece)
            .copied()
            .filter(|&index| self.matches(index, parse_special))
    }

    /// Split `text` on the added tokens the way `PreTrainedTokenizer.tokenize`
    /// does: leftmost-longest matching, then each matched token's `rstrip`,
    /// `lstrip` and `single_word` applied to its neighbours. Empty pieces are
    /// dropped.
    pub(super) fn split(&self, text: &str, parse_special: bool) -> Vec<Piece> {
        let mut pieces = self.cut(text, parse_special);
        self.apply_flags(&mut pieces, parse_special);
        pieces
            .into_iter()
            .filter(|piece| !piece.is_empty())
            .map(|piece| match self.lookup(&piece, parse_special) {
                Some(index) => Piece::Token(self.tokens[index].0),
                None => Piece::Text(piece),
            })
            .collect()
    }

    /// Cut `text` at every leftmost-longest added-token occurrence, keeping
    /// the tokens as their own pieces (the trie split).
    fn cut(&self, text: &str, parse_special: bool) -> Vec<String> {
        let bytes = text.as_bytes();
        let mut pieces = Vec::new();
        let mut text_start = 0;
        let mut pos = 0;
        while pos < bytes.len() {
            // Every content is valid UTF-8, so its first byte is never a
            // continuation byte and a match can only start on a char
            // boundary; the slices below are therefore always valid.
            let found = self.by_first_byte[usize::from(bytes[pos])]
                .iter()
                .copied()
                .find(|&index| {
                    self.matches(index, parse_special)
                        && bytes[pos..].starts_with(self.tokens[index].1.content.as_bytes())
                });
            match found {
                Some(index) => {
                    if text_start < pos {
                        pieces.push(text[text_start..pos].to_string());
                    }
                    let len = self.tokens[index].1.content.len();
                    pieces.push(text[pos..pos + len].to_string());
                    pos += len;
                    text_start = pos;
                }
                None => pos += 1,
            }
        }
        if text_start < bytes.len() {
            pieces.push(text[text_start..].to_string());
        }
        pieces
    }

    /// Apply the matched tokens' whitespace and word-boundary flags, a
    /// line-for-line port of the loop in `PreTrainedTokenizer.tokenize`
    /// (transformers 4.57): neighbours are read before either is modified,
    /// `rstrip` eats whitespace on the right, `lstrip` on the left, and a
    /// `single_word` token touching a non-space neighbour is folded into that
    /// neighbour, which turns it back into ordinary text.
    fn apply_flags(&self, pieces: &mut [String], parse_special: bool) {
        for i in 0..pieces.len() {
            let Some(index) = self.lookup(&pieces[i], parse_special) else {
                continue;
            };
            let token = &self.tokens[index].1;
            let left = (i > 0 && !pieces[i - 1].is_empty()).then(|| pieces[i - 1].clone());
            let right = pieces.get(i + 1).filter(|right| !right.is_empty()).cloned();

            if token.rstrip
                && let Some(right) = &right
            {
                pieces[i + 1] = right.trim_start().to_string();
            }
            if token.lstrip
                && let Some(left) = &left
            {
                pieces[i - 1] = left.trim_end().to_string();
            }
            if token.single_word && left.as_deref().is_some_and(|l| !l.ends_with(' ')) {
                let current = std::mem::take(&mut pieces[i]);
                pieces[i - 1].push_str(&current);
            } else if token.single_word
                && let Some(right) = right.filter(|r| !r.starts_with(' '))
            {
                pieces[i].push_str(&right);
                pieces[i + 1].clear();
            }
        }
    }
}

/// Read a SentencePiece checkpoint's added tokens with their flags.
///
/// `added_tokens_decoder` is preferred because it carries the flags; when a
/// checkpoint has no decoder table, `added_tokens.json`'s flat
/// `content -> id` map supplies plain non-special tokens, the same precedence
/// and defaults as `read_added_tokens_sorted` on the `tokenizers` path.
/// Unlike that function this one is lenient: a malformed entry is skipped
/// rather than failing the load, because the SentencePiece loader never
/// failed on one and a checkpoint that loads today must keep loading.
///
/// An `added_tokens.json` entry is always non-special here, as on the
/// `tokenizers` path. `transformers`' legacy loader would mark one special
/// when its content is also a declared special token (`eos_token` and the
/// like); no surveyed checkpoint relies on that, because every one of them
/// ships `added_tokens_decoder`.
pub(super) fn read_sp_added_tokens(
    config: Option<&serde_json::Value>,
    model_path: &Path,
) -> Vec<(u32, AddedToken)> {
    let mut out = Vec::new();
    if let Some(decoder) = config
        .and_then(|cfg| cfg.get("added_tokens_decoder"))
        .and_then(|v| v.as_object())
    {
        out.extend(decoder.iter().filter_map(|(id, entry)| {
            super::added_tokens::parse_added_token_entry(id, entry).ok()
        }));
    } else if let Ok(raw) = std::fs::read_to_string(model_path.join("added_tokens.json"))
        && let Ok(map) = serde_json::from_str::<HashMap<String, u32>>(&raw)
    {
        for (content, id) in map {
            out.push((id, AddedToken::from(content, false).normalized(false)));
        }
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

#[cfg(test)]
#[path = "sp_added_vocab_tests.rs"]
mod sp_added_vocab_tests;
