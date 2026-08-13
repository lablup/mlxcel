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
        let encoded = tokenizer.encode(&text, false).with_context(|| {
            format!("failed to tokenize --dry-sequence-breaker value {entry:?}")
        })?;

        if encoded.len() != 1 {
            bail!(
                "invalid --dry-sequence-breaker value {entry:?}: it encodes to {} tokens \
                 ({encoded:?}) for this model, but a DRY sequence breaker must be exactly one \
                 token because the sampler matches breakers by token id. Pass a single-token \
                 string (\"\\n\" and \"\\t\" are the usual ones) or drop this entry.",
                encoded.len()
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
             empty); remove the flag to run DRY without breakers, or pass at least one \
             single-token string such as \"\\n\""
        );
    }

    Ok(ids)
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
