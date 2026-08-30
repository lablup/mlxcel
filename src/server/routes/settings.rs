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

//! Opt-in live settings management endpoints.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::server::AppState;
use crate::server::runtime_settings::{self, KnobSpec, Rejected};

#[derive(Serialize)]
pub struct SettingsResponse {
    schema: Vec<KnobSpec>,
    current: Map<String, Value>,
    fingerprint: String,
}

#[derive(Serialize)]
pub struct PatchSettingsResponse {
    applied: Map<String, Value>,
    rejected: Vec<Rejected>,
    current: Map<String, Value>,
    fingerprint: String,
}

fn response(state: &AppState) -> SettingsResponse {
    let live = state.live();
    SettingsResponse {
        schema: runtime_settings::schema(&state.config),
        current: runtime_settings::current(&live, &state.config),
        fingerprint: runtime_settings::fingerprint(&live),
    }
}

/// Return the typed schema, current values, and stable mutable fingerprint.
pub async fn get_settings(State(state): State<AppState>) -> Json<SettingsResponse> {
    Json(response(&state))
}

/// Validate a merge or replace update and publish every accepted setting in
/// one Arc swap. Rejected names do not roll back independent valid names.
pub async fn patch_settings(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<PatchSettingsResponse>, (StatusCode, Json<Value>)> {
    let Value::Object(body) = body else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "settings PATCH body must be a JSON object"})),
        ));
    };
    let (op, values) = runtime_settings::parse_patch_body(body)
        .map_err(|reason| (StatusCode::BAD_REQUEST, Json(json!({"error": reason}))))?;

    let update_state = state.clone();
    let (result, next, old_values) = tokio::task::spawn_blocking(move || {
        let _update_guard = update_state
            .settings_update_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = update_state.live();
        let mut result = runtime_settings::apply(
            &update_state.startup_live,
            &previous,
            op,
            &values,
            &update_state.config,
        );
        result.next.resolved_token_bias =
            crate::server::model_provider::model_worker::resolve_worker_token_bias(
                result.next.lang_bias_config.as_ref(),
                &update_state.tokenizer,
                &update_state.model_path,
            );
        let next = std::sync::Arc::new(result.next.clone());
        let old_values = runtime_settings::mutable_values(&previous);
        let mut guard = update_state
            .current_live
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = next.clone();
        (result, next, old_values)
    })
    .await
    .map_err(|error| {
        tracing::error!("settings update task failed: {error}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "settings update failed"})),
        )
    })?;
    let new_values = runtime_settings::mutable_values(&next);
    for (name, new_value) in &new_values {
        let old_value = old_values.get(name).cloned().unwrap_or(Value::Null);
        if old_value == *new_value {
            continue;
        }
        tracing::info!(
            setting = name,
            old = %old_value,
            new = %new_value,
            "live server setting updated"
        );
    }

    Ok(Json(PatchSettingsResponse {
        applied: result.applied,
        rejected: result.rejected,
        current: runtime_settings::current(&next, &state.config),
        fingerprint: runtime_settings::fingerprint(&next),
    }))
}
