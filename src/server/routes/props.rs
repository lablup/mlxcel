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

//! Server properties endpoint (llama-server compatible).
//!
//! This route surfaces configuration/state snapshots and should stay as a thin
//! read-only adapter.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::config::ServerConfig;
use crate::server::types::PropsResponse;

/// The `default_generation_settings` object of the `/props` payload.
///
/// Split out of the handler so the reported field set can be asserted without
/// standing up an `AppState`. The set is a contract: an operator reads it to
/// confirm what the server resolved a flag to, so a sampling default that the
/// server honours but does not report here is invisible in exactly the way
/// `--dry-sequence-breaker` was before #1103.
pub(crate) fn default_generation_settings(config: &ServerConfig) -> serde_json::Value {
    serde_json::json!({
        "n_predict": config.default_max_tokens,
        "temperature": config.default_temperature,
        "top_k": config.default_top_k,
        "top_p": config.default_top_p,
        "min_p": config.default_min_p,
        "repeat_penalty": config.default_repetition_penalty,
        "repeat_last_n": config.default_repetition_context_size,
        "seed": config.default_seed.unwrap_or(u64::MAX),
        "frequency_penalty": config.default_frequency_penalty,
        "presence_penalty": config.default_presence_penalty,
        "dry_multiplier": config.default_dry_multiplier,
        "dry_base": config.default_dry_base,
        "dry_allowed_length": config.default_dry_allowed_length,
        "dry_penalty_last_n": config.default_dry_penalty_last_n,
        // Reported as resolved token IDs rather than as the strings the
        // operator typed, because the IDs are what the sampler compares
        // against and what a per-request `dry_sequence_breakers` overrides.
        "dry_sequence_breakers": config.default_dry_sequence_breakers,
    })
}

/// GET /props
pub async fn props(State(state): State<AppState>) -> Json<PropsResponse> {
    Json(PropsResponse {
        default_generation_settings: default_generation_settings(&state.config),
        total_slots: state.config.n_parallel,
    })
}

#[cfg(test)]
#[path = "props_tests.rs"]
mod props_tests;
