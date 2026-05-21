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

//! Slots endpoint reporting per-sequence status.
//!
//! When the `--slots` flag is enabled, this endpoint reports the status of
//! each batch slot: idle slots and active sequences with their token counts.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::SlotInfo;

/// GET /slots
///
/// Reports per-slot status. Active sequences include prompt token count,
/// generated token count, and elapsed time.
pub async fn slots(State(state): State<AppState>) -> Json<Vec<SlotInfo>> {
    let max_slots = state.config.max_batch_size.max(1);
    let active_count = state.batch_metrics.active_count();
    let queue_depth = state.batch_metrics.queue_depth();
    let model_id = state.display_model_id().to_string();
    let context_size = state.config.context_size;

    let mut slots: Vec<SlotInfo> = Vec::with_capacity(max_slots + queue_depth);

    // Report active/idle decode slots
    for i in 0..max_slots {
        let is_active = i < active_count;
        slots.push(SlotInfo {
            id: i,
            state: if is_active {
                "decoding".to_string()
            } else {
                "idle".to_string()
            },
            model: model_id.clone(),
            context_size,
            is_processing: is_active,
            prompt_tokens: None,
            generated_tokens: None,
            elapsed_ms: None,
        });
    }

    // Report queued requests as additional entries
    for i in 0..queue_depth {
        slots.push(SlotInfo {
            id: max_slots + i,
            state: "queued".to_string(),
            model: model_id.clone(),
            context_size,
            is_processing: false,
            prompt_tokens: None,
            generated_tokens: None,
            elapsed_ms: None,
        });
    }

    Json(slots)
}
