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

//! b10621 DRY sequence-breaker semantics (#1485).
//!
//! `--dry-sequence-breaker` and the native `dry_sequence_breakers` field take
//! breaker *strings*, exactly as llama-server b10621 does. Breaker token data
//! is derived by scanning the vocabulary for tokens whose decoded text
//! carries (or ends inside) a breaker string, the port of upstream
//! `get_overlapping_token_sequences` in
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/src/llama-sampler.cpp>.
//! The result is a head-token map ([`mlxcel_core::SamplingConfig::dry_breaker_heads`]):
//! a token whose text contains the breaker outright is a head with an empty
//! tail and breaks matching on its own; a token whose text ends with a proper
//! prefix of the breaker is a head whose tail (the tokenization of the
//! remainder) must follow in the window for the break to fire.
//!
//! The scan needs the whole vocabulary surface, so it runs where the
//! tokenizer lives: the scheduler derives (and caches) the map at enqueue
//! time. Pre-#1485 this module resolved each string to exactly one token id
//! and failed startup otherwise; that restriction is gone because the head
//! map represents multi-token breakers faithfully.

use std::collections::HashMap;

use crate::tokenizer::MlxcelTokenizer;

/// b10621's default DRY sequence breakers, applied when the flag is absent.
pub(crate) const DEFAULT_DRY_SEQUENCE_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// The default breaker set as owned strings, for
/// [`crate::server::ServerConfig`]'s default.
pub(crate) fn default_breaker_strings() -> Vec<String> {
    DEFAULT_DRY_SEQUENCE_BREAKERS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Upstream caps a breaker at 40 bytes (`MAX_CHAR_LEN`) with a warning.
const MAX_BREAKER_LEN: usize = 40;

/// Upstream caps a derived tail at 20 tokens (`MAX_SEQ_LEN`).
const MAX_TAIL_LEN: usize = 20;

/// Resolve the raw `--dry-sequence-breaker` CLI values into the effective
/// breaker string set, following b10621's flag handler: an absent flag keeps
/// the default set (`\n`, `:`, `"`, `*`); giving the flag replaces the
/// defaults with exactly the given values in order; the sentinel value
/// `none` clears everything accumulated so far (so a lone `none` runs DRY
/// with no breakers). Shell escapes are interpreted per
/// [`unescape_breaker`]; empty entries from a stray comma are skipped.
pub(crate) fn resolve_breaker_strings(raw: &[String]) -> Vec<String> {
    if raw.is_empty() {
        return default_breaker_strings();
    }
    let mut out = Vec::new();
    for entry in raw {
        if entry == "none" {
            out.clear();
            continue;
        }
        if entry.is_empty() {
            continue;
        }
        out.push(unescape_breaker(entry));
    }
    out
}

/// Cap one breaker at upstream's 40-byte truncation, backing off to the
/// nearest character boundary (upstream truncates raw bytes, which can split
/// a multi-byte character; matching on a broken byte sequence is not
/// representable over `&str`, so the boundary floor is the closest faithful
/// form).
fn cap_breaker(s: &str) -> &str {
    if s.len() <= MAX_BREAKER_LEN {
        return s;
    }
    let mut end = MAX_BREAKER_LEN;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    tracing::warn!(breaker = %s, "truncating DRY sequence breaker to {MAX_BREAKER_LEN} bytes");
    &s[..end]
}

/// The b10621 vocabulary scan (`get_overlapping_token_sequences`): derive
/// the DRY head-token map for `breakers` over this tokenizer's vocabulary.
///
/// For every vocabulary token, the token's decoded text is tested against
/// each breaker: containing the breaker outright yields a head with an empty
/// tail, and ending with a proper prefix of the breaker yields a head whose
/// tail is the tokenization (special parsing off, upstream's
/// `tokenize(str.substr(i), false, false)`) of the breaker's remainder,
/// capped at [`MAX_TAIL_LEN`] tokens. Duplicate tails per head are dropped,
/// as upstream drops them.
///
/// `vocab_texts[id]` is the decoded text of token `id`
/// ([`decode_vocab_texts`]); tokens with no text never become heads.
pub(crate) fn derive_breaker_heads(
    tokenizer: &MlxcelTokenizer,
    vocab_texts: &[String],
    breakers: &[String],
) -> HashMap<i32, Vec<Vec<i32>>> {
    let mut heads: HashMap<i32, Vec<Vec<i32>>> = HashMap::new();
    for raw in breakers {
        if raw.is_empty() {
            continue;
        }
        let breaker = cap_breaker(raw);
        let sb = breaker.as_bytes();
        for (tid, word) in vocab_texts.iter().enumerate() {
            if word.is_empty() {
                continue;
            }
            let tid = tid as i32;
            if word.contains(breaker) {
                let tails = heads.entry(tid).or_default();
                if !tails.iter().any(Vec::is_empty) {
                    tails.push(Vec::new());
                }
                continue;
            }
            let wb = word.as_bytes();
            for pos in 0..wb.len() {
                if wb[pos] != sb[0] {
                    continue;
                }
                let mut i = 1;
                let mut matched = true;
                while i < sb.len() && pos + i < wb.len() {
                    if wb[pos + i] != sb[i] {
                        matched = false;
                        break;
                    }
                    i += 1;
                }
                // A full in-word match was handled by `contains` above; here
                // the word ended inside the breaker, leaving a tail.
                if !matched || pos + i < wb.len() {
                    continue;
                }
                let tail_text = String::from_utf8_lossy(&sb[i..]);
                let mut tail: Vec<i32> = tokenizer
                    .encode_with_special(&tail_text, false, false)
                    .map(|ids| ids.into_iter().map(|id| id as i32).collect())
                    .unwrap_or_default();
                tail.truncate(MAX_TAIL_LEN);
                let tails = heads.entry(tid).or_default();
                if !tails.contains(&tail) {
                    tails.push(tail);
                }
            }
        }
    }
    heads
}

/// Decode every vocabulary token's text once, for [`derive_breaker_heads`].
/// Tokens whose bytes are not valid UTF-8 are decoded lossily: a breaker is
/// matched over text, exactly as upstream matches over `detokenize`'s
/// output.
pub(crate) fn decode_vocab_texts(tokenizer: &MlxcelTokenizer) -> Vec<String> {
    let vocab = tokenizer.vocab_size();
    let mut texts = Vec::with_capacity(vocab);
    for id in 0..vocab {
        let text = tokenizer
            .token_piece_bytes(id as u32)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        texts.push(text);
    }
    texts
}

/// Interpret the four C-style escapes an operator can realistically type at a
/// shell prompt.
///
/// `--dry-sequence-breaker '\n'` reaches this process as the two characters
/// `\` and `n`: no common shell expands an escape inside single quotes, and
/// `"\n"` is the same two characters in POSIX shells. The flag's help text has
/// advertised `"\n"` and `"\t"` as its examples since it was added, so reading
/// the value literally would reject the documented usage at startup, or worse,
/// accept it as whatever unrelated single token `\n` happens to encode to.
///
/// Only `\n`, `\t`, `\r` and `\\` are interpreted. Every other backslash
/// sequence is preserved exactly as typed, so a breaker that genuinely
/// contains a backslash is never silently rewritten, and `\\` is the escape
/// hatch for a literal backslash that precedes one of the four.
fn unescape_breaker(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();

    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            // Unknown escape: keep both characters rather than guessing.
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            // Trailing lone backslash keeps itself.
            None => out.push('\\'),
        }
    }

    out
}

#[cfg(test)]
#[path = "dry_breakers_tests.rs"]
mod dry_breakers_tests;
