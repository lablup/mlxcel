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

//! LoRA adapter inventory and hot-swap endpoints (llama-server b10621
//! compatible, issue #1439).
//!
//! `GET /lora-adapters` reports the adapters the server was started with, in
//! b10621's entry shape (`id`, `path`, `scale`, `task_name`,
//! `prompt_prefix`), with a not-applied adapter reported at scale 0.0 exactly
//! as upstream reports `--lora-init-without-apply`. `POST /lora-adapters`
//! carries b10621's request contract (an array of `{id, scale}`, unlisted
//! adapters dropping to 0.0). On the unfused runtime path (the default,
//! #1439) the resolved scales replace the server default and apply to every
//! request admitted afterwards, exactly upstream's `SERVER_TASK_TYPE_SET_LORA`
//! semantics: in-flight generations keep the snapshot they were admitted
//! with. Under `--lora-fuse` the adapters are baked into the weights, so
//! only a request that resolves to the configuration already in force can be
//! acknowledged; anything else is refused with a diagnostic.

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};

use super::slots::{llama_invalid_request, llama_not_supported};
use crate::server::AppState;

/// The effective per-adapter scales currently in force: the runtime set's
/// live server scales, or the fused-at-load configuration.
fn current_scales(state: &AppState) -> Vec<f32> {
    if let Some(set) = &state.config.lora_runtime {
        return set.server_scales();
    }
    state
        .config
        .lora_adapters
        .iter()
        .map(|spec| spec.reported_scale())
        .collect()
}

/// The effective scales a b10621 `[{id, scale}]` list asks for: listed ids
/// set their scale, everything else drops to 0.0 (upstream
/// `construct_lora_list`; an id outside the adapter range is ignored there
/// too).
pub(crate) fn requested_scales(entries: &[serde_json::Value], adapter_count: usize) -> Vec<f32> {
    let mut scales = vec![0.0f32; adapter_count];
    for entry in entries {
        let id = entry.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);
        let scale = entry.get("scale").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        if id >= 0 && (id as usize) < adapter_count {
            scales[id as usize] = scale;
        }
    }
    scales
}

/// GET /lora-adapters
pub async fn get_lora_adapters(State(state): State<AppState>) -> Json<serde_json::Value> {
    let scales = current_scales(&state);
    let entries: Vec<serde_json::Value> = state
        .config
        .lora_adapters
        .iter()
        .enumerate()
        .map(|(id, spec)| {
            serde_json::json!({
                "id": id,
                "path": spec.path.to_string_lossy(),
                // The live server scale on the runtime path (a POST swap is
                // visible here, exactly as upstream reports params_base);
                // the fused configuration otherwise.
                "scale": scales.get(id).copied().unwrap_or_else(|| spec.reported_scale()),
                // b10621 reads these from GGUF adapter metadata; MLX adapter
                // directories carry neither, so both are truthfully empty.
                "task_name": "",
                "prompt_prefix": "",
            })
        })
        .collect();
    Json(serde_json::Value::Array(entries))
}

/// POST /lora-adapters
pub async fn post_lora_adapters(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return llama_invalid_request("Request body must be an array");
    };
    let Some(entries) = parsed.as_array() else {
        // Upstream's exact wording for a non-array body.
        return llama_invalid_request("Request body must be an array");
    };
    let current = current_scales(&state);
    let requested = requested_scales(entries, current.len());
    if let Some(set) = &state.config.lora_runtime {
        // Runtime path (#1439): the resolved scales become the server
        // default. Requests admitted afterwards snapshot them; in-flight
        // generations keep the snapshot they were admitted with, exactly
        // like upstream's slots.
        set.set_server_scales(requested);
        return Json(serde_json::json!({ "success": true })).into_response();
    }
    if requested == current {
        // The configuration asked for is the configuration in force:
        // acknowledging it is genuinely inert, and it is what upstream
        // answers for any successful set.
        return Json(serde_json::json!({ "success": true })).into_response();
    }
    llama_not_supported(
        "changing LoRA adapter scales at runtime is not supported under --lora-fuse: the \
         adapters are fused into the model weights at load time. Restart without --lora-fuse, \
         or with --lora-scaled FNAME:SCALE for a different fused configuration",
    )
}

#[cfg(test)]
#[path = "lora_adapters_tests.rs"]
mod lora_adapters_tests;
