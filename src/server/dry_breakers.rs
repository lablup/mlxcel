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

//! Resolution of the `--dry-sequence-breaker` CLI strings into token IDs.
//!
//! The flag takes token *strings* (matching llama.cpp's spelling of the same
//! flag), while [`mlxcel_core::SamplingConfig`] and the per-request HTTP field
//! both take token *IDs*. Bridging the two needs the tokenizer, which is why
//! the value cannot simply be copied into [`crate::server::ServerConfig`] at
//! `build_server_config` time: that runs before the model's tokenizer is
//! loaded. The server resolves it immediately after the tokenizer is available
//! and fails startup on anything it cannot represent, rather than dropping the
//! flag silently (#1103).
//!
//! The single-token requirement is forced by the data model. The sampler's
//! breaker check is `config.dry_sequence_breakers.contains(&window[p1])` over
//! a `Vec<i32>`, so a multi-token breaker has no representation at all. A
//! breaker that does not encode to exactly one token is therefore an error
//! naming the offending string, matching the posture `--allowed-origins` takes
//! for a malformed origin (see [`crate::server::cors`]).

use anyhow::{Context, Result, bail};

use crate::tokenizer::MlxcelTokenizer;

/// Resolve the raw `--dry-sequence-breaker` strings into sampler token IDs.
///
/// Empty entries are skipped, which absorbs a stray delimiter such as
/// `--dry-sequence-breaker 'a,'`. A flag that was set but yielded nothing is a
/// configuration error rather than a silent "no breakers" result, because the
/// operator clearly intended a policy. Both rules mirror
/// [`crate::server::cors::parse_allowed_origins`].
pub(crate) fn resolve_dry_sequence_breakers(
    tokenizer: &MlxcelTokenizer,
    raw: &[String],
) -> Result<Vec<i32>> {
    let mut ids = Vec::with_capacity(raw.len());

    for entry in raw {
        // `is_empty`, deliberately not `trim().is_empty()`: a single space is
        // a legitimate breaker in most BPE vocabularies, and trimming would
        // silently discard it.
        if entry.is_empty() {
            continue;
        }

        let text = unescape_breaker(entry);
        let encoded = encode_breaker(tokenizer, &text).with_context(|| {
            format!("failed to tokenize --dry-sequence-breaker value {entry:?}")
        })?;

        if encoded.len() != 1 {
            bail!(
                "invalid --dry-sequence-breaker value {entry:?}: this model encodes it as {} \
                 tokens ({}), but a DRY sequence breaker must be exactly one token because the \
                 sampler matches breakers by token id. Pick a value this model represents as a \
                 single token, or drop this entry.",
                encoded.len(),
                describe_tokens(tokenizer, &encoded)
            );
        }

        let id = encoded[0];
        ids.push(i32::try_from(id).with_context(|| {
            format!(
                "--dry-sequence-breaker value {entry:?} encodes to token id {id}, which does \
                 not fit in the i32 the sampler uses"
            )
        })?);
    }

    if !raw.is_empty() && ids.is_empty() {
        bail!(
            "--dry-sequence-breaker was set but contained no usable breaker (every entry was \
             empty). Note that the flag splits its value on commas, so a bare \",\" produces \
             two empty entries and a comma cannot itself be a breaker. Remove the flag to run \
             DRY without breakers, or pass at least one single-token string."
        );
    }

    Ok(ids)
}

/// Anchor prepended to a breaker before tokenizing, so that what is measured is
/// the breaker's own token contribution.
///
/// A single ASCII letter is used because it is present in every vocabulary this
/// server can load, and because it is the least likely character to merge with
/// an arbitrary breaker.
const BREAKER_ANCHOR: &str = "a";

/// Tokenize one breaker, discounting any fixed prefix the tokenizer's
/// normalizer adds.
///
/// Encoding the breaker on its own is the obvious implementation and is wrong
/// for a whole family of checkpoints. A SentencePiece-derived `tokenizer.json`
/// carries a `Prepend "▁"` normalizer, usually alongside `Replace " " -> "▁"`,
/// so the text handed to the model is not the text that was passed in:
///
/// - On Mixtral, `encode("\n")` yields `["▁", "<0x0A>"]`, two tokens, even
///   though the newline is a single vocabulary entry (id 13). The breaker is
///   perfectly representable; the bare encoding just asks the wrong question,
///   and startup would fail on the example in this flag's own help text.
/// - Worse, `encode(" ")` normalizes to `"▁▁"` and yields ONE token, the
///   double-space entry (id 259). That passes a length check and installs a
///   breaker the operator did not ask for, silently. It is the same class of
///   failure this whole flag was inert for.
///
/// Encoding `anchor + breaker` and subtracting the anchor's own encoding
/// measures the breaker's contribution instead, so a fixed prefix cannot
/// inflate the count or shift the id. When the anchor does not survive as a
/// prefix (it merged with the breaker) or encodes to nothing, there is nothing
/// to subtract and the bare encoding is used, which is the previous behavior.
fn encode_breaker(tokenizer: &MlxcelTokenizer, text: &str) -> Result<Vec<u32>> {
    let bare = tokenizer.encode(text, false)?;

    let anchor = tokenizer.encode(BREAKER_ANCHOR, false)?;
    if anchor.is_empty() {
        return Ok(bare);
    }

    let anchored = tokenizer.encode(&format!("{BREAKER_ANCHOR}{text}"), false)?;
    Ok(anchored
        .strip_prefix(anchor.as_slice())
        .map_or(bare, <[u32]>::to_vec))
}

/// Render token ids with the pieces they decode to, for an error message.
///
/// The pieces are what make a normalizer artifact visible: `"▁"=28705,
/// "<0x0A>"=13` tells an operator that their tokenizer prepended a word
/// boundary marker, where a bare `[28705, 13]` does not.
fn describe_tokens(tokenizer: &MlxcelTokenizer, ids: &[u32]) -> String {
    ids.iter()
        .map(|id| match tokenizer.token_piece(*id) {
            Some(piece) => format!("{piece:?}={id}"),
            None => format!("<unknown>={id}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Public wrapper over [`describe_tokens`] for startup logging, so a resolved
/// breaker can be read back as pieces rather than as bare ids.
pub(crate) fn describe_resolved_breakers(tokenizer: &MlxcelTokenizer, ids: &[i32]) -> String {
    let unsigned: Vec<u32> = ids
        .iter()
        .filter_map(|id| u32::try_from(*id).ok())
        .collect();
    describe_tokens(tokenizer, &unsigned)
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
