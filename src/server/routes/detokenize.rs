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

//! Detokenize endpoint (llama-server compatible).
//!
//! This route is intentionally narrow: HTTP translation only, with tokenizer
//! ownership and model state handled elsewhere.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{DetokenizeRequest, DetokenizeResponse, ErrorResponse};

/// POST /detokenize
///
/// Special tokens are rendered rather than skipped, as upstream's
/// `tokens_to_str` does. An absent or empty `tokens` answers `{"content": ""}`
/// instead of failing, which is upstream's behavior too (#1442).
///
/// A token whose bytes are not valid UTF-8 on its own is joined with its
/// neighbours before decoding, so a split multi-byte character round trips;
/// bytes that still cannot form a character come back as U+FFFD, which is what
/// upstream emits as well because its JSON writer is configured to replace
/// rather than throw.
pub async fn detokenize(
    State(state): State<AppState>,
    Json(request): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ErrorResponse> {
    let Some(tokens) = request.tokens.as_ref() else {
        return Ok(Json(DetokenizeResponse {
            content: String::new(),
        }));
    };

    let ids: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();

    let content = state.tokenizer.decode(&ids, false).map_err(|e| {
        ErrorResponse::new(
            format!("Detokenization error: {}", e),
            "invalid_request_error",
        )
    })?;

    Ok(Json(DetokenizeResponse { content }))
}
