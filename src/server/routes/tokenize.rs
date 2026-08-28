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

//! Tokenize endpoint (llama-server compatible).
//!
//! This is a thin tokenizer adapter and should not grow generation or chat
//! policy that belongs in other server modules.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{ErrorResponse, TokenizeEntry, TokenizeRequest, TokenizeResponse};
use crate::tokenizer::{MlxcelTokenizer, pieces};

/// POST /tokenize
///
/// b10621's four switches: `content` (a string or a mixed array), `add_special`
/// (default `false`), `parse_special` (default `true`) and `with_pieces`
/// (default `false`). See [`TokenizeRequest`] (#1442).
pub async fn tokenize(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ErrorResponse> {
    let add_special = request.add_special.unwrap_or(false);
    let parse_special = request.parse_special.unwrap_or(true);
    let with_pieces = request.with_pieces.unwrap_or(false);

    let token_ids = match request.content.as_ref() {
        // An absent `content` is an empty tokenization upstream, not an error.
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(content) => tokenize_mixed(&state.tokenizer, content, add_special, parse_special)
            .map_err(|e| {
                ErrorResponse::new(format!("Tokenization error: {e}"), "invalid_request_error")
            })?,
    };

    let tokens = if with_pieces {
        token_ids
            .iter()
            .map(|&id| TokenizeEntry::Piece {
                id: id as i32,
                piece: pieces::piece_json(
                    state.tokenizer.token_piece_bytes(id).unwrap_or_default(),
                ),
            })
            .collect()
    } else {
        token_ids
            .iter()
            .map(|&id| TokenizeEntry::Id(id as i32))
            .collect()
    };

    Ok(Json(TokenizeResponse { tokens }))
}

/// Tokenize b10621's "mixed" prompt shape.
///
/// A JSON string is tokenized. A JSON array is walked element by element: a
/// string element is tokenized and an integer element is already a token id and
/// is taken as-is, which is what lets a client splice pre-tokenized spans into
/// a prompt. `add_special` applies to the FIRST element only, and an id element
/// consumes that position exactly as a string does, matching upstream: it is
/// the BOS position, not a per-segment property, so `[1, "Hello"]` does not get
/// a second BOS after the spliced id.
///
/// Upstream reference: `tokenize_mixed` in
/// <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/utils.hpp>
pub(crate) fn tokenize_mixed(
    tokenizer: &MlxcelTokenizer,
    content: &serde_json::Value,
    add_special: bool,
    parse_special: bool,
) -> anyhow::Result<Vec<u32>> {
    match content {
        serde_json::Value::String(text) => {
            tokenizer.encode_with_special(text, add_special, parse_special)
        }
        serde_json::Value::Array(items) => {
            let mut out = Vec::new();
            let mut first = true;
            for item in items {
                match item {
                    serde_json::Value::String(text) => {
                        let ids = tokenizer.encode_with_special(
                            text,
                            add_special && first,
                            parse_special,
                        )?;
                        first = false;
                        out.extend(ids);
                    }
                    serde_json::Value::Number(number) => {
                        first = false;
                        let id = number.as_i64().ok_or_else(|| {
                            anyhow::anyhow!("token id {number} is not an integer")
                        })?;
                        let id = u32::try_from(id)
                            .map_err(|_| anyhow::anyhow!("token id {id} is out of range"))?;
                        out.push(id);
                    }
                    other => anyhow::bail!(
                        "a mixed prompt element must be a string or a token id, got {other}"
                    ),
                }
            }
            Ok(out)
        }
        other => anyhow::bail!("\"content\" must be a string or an array, got {other}"),
    }
}

#[cfg(test)]
#[path = "tokenize_tests.rs"]
mod tokenize_tests;
