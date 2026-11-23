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

//! Tolerant reader for HF `config.json` text (issue #3339).
//!
//! Ungated so the EOS lookup and the CLI layer count can use it without the
//! `iree` feature; the emitter and the runtime config reader share it.

/// Replace the bare `Infinity`, `-Infinity` and `NaN` tokens Python's `json`
/// writes (HF Mamba2 configs carry `"time_step_limit": [0.0, Infinity]`) with
/// strings, so `serde_json` accepts the document. Text inside string literals
/// is left alone. Returns `None` when there is nothing to replace.
pub(crate) fn quote_non_finite_json(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;
    let mut rest = text;
    while let Some(ch) = rest.chars().next() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            rest = &rest[ch.len_utf8()..];
            continue;
        }
        if ch == '"' {
            in_string = true;
            out.push(ch);
            rest = &rest[1..];
            continue;
        }
        let token = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|token| rest.starts_with(token));
        if let Some(token) = token {
            out.push('"');
            out.push_str(token);
            out.push('"');
            rest = &rest[token.len()..];
            changed = true;
            continue;
        }
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    changed.then_some(out)
}

/// Parse an HF `config.json` text. HF writes Python's bare `Infinity` into
/// some configs (Mamba2's `time_step_limit`), which strict JSON rejects; the
/// tokens are quoted only when the strict parse fails, so every other config is
/// parsed from its unchanged text.
pub fn parse_hf_config_json(text: &str) -> Result<serde_json::Value, String> {
    parse_hf_config_json_tracked(text).map(|(value, _)| value)
}

/// [`parse_hf_config_json`], also reporting whether non-finite tokens had to
/// be quoted. A quoted token parses as a string, which an ordinary numeric
/// field reader treats as absent and replaces with its default, so a reader
/// that does not understand the quoted spelling must refuse such a document
/// instead of silently changing the value.
pub(crate) fn parse_hf_config_json_tracked(
    text: &str,
) -> Result<(serde_json::Value, bool), String> {
    match serde_json::from_str(text) {
        Ok(value) => Ok((value, false)),
        Err(error) => match quote_non_finite_json(text) {
            Some(quoted) => serde_json::from_str(&quoted)
                .map(|value| (value, true))
                .map_err(|error| error.to_string()),
            None => Err(error.to_string()),
        },
    }
}
