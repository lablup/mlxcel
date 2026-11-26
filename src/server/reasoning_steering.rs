// Copyright 2025-2026 Lablup Inc.
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

//! The model-side reasoning effort of a chat request (#3435).
//!
//! Raon-1 mini's template reads only `enable_thinking`; its reasoning level
//! is applied in the weights (`reasoning_effort_steering`), so for a model
//! whose config declares that block the request's level must reach the
//! model, per sequence. [`resolve_model_reasoning_effort`] is the single
//! resolution, shared by request preparation (the value the scheduler hands
//! the model, and the 400 for an unknown level) and the prompt-cache
//! signature (so a prefix computed at one effort is never reused at another).
//!
//! Sources, highest first:
//!
//! 1. Per-request `chat_template_kwargs.reasoning_effort` (also nested or
//!    flattened under `extra_body`). As on the vendor servers, a `none` here
//!    only changes the alpha; thinking stays on unless `enable_thinking` is
//!    also false, which is the model card's documented behavior.
//! 2. The portable fields: top-level `reasoning_effort` (the Responses API's
//!    `reasoning.effort` is translated onto it) or `reasoning`. A disabled
//!    `reasoning: false` selects `none`. The existing template mapping
//!    already renders a portable `none` with `enable_thinking=false`.
//! 3. A per-request `enable_thinking: false` (in `chat_template_kwargs`,
//!    top-level or under `extra_body`) selects `none`, even over a server
//!    default level: the render turns thinking off for this request, and the
//!    model card documents both controls as giving the same result.
//! 4. The server-default `--chat-template-kwargs` `reasoning_effort`, the
//!    operator's level for requests that set none (llama-server's
//!    `--reasoning-effort`).
//! 5. Thinking off with no level selects `none`: both controls give the same
//!    result. Thinking is resolved as the render resolves it: the request's
//!    `enable_thinking`, else its portable `reasoning` toggle, else the
//!    server default.
//! 6. The config's `default` level.
//!
//! No translation: the value must be `none` or one of the config's alpha
//! keys after trimming and lowercasing, so OpenAI's `minimal` and the
//! disabled spellings `off` / `false` are refused with the accepted set.

use serde_json::Value;

use super::chat_template::ChatTemplateProcessor;
use super::chat_template_kwargs::{ChatTemplateKwargs, extract_request_kwargs};
use super::types::request::{ChatCompletionRequest, ReasoningControl};
use crate::models::reasoning_effort_steering::NONE_LEVEL;

const REASONING_EFFORT: &str = "reasoning_effort";
const ENABLE_THINKING: &str = "enable_thinking";

/// The level the model runs this request at, or `None` when the loaded model
/// does not steer by reasoning effort.
///
/// # Errors
/// Returns the client-facing message (a 400) for a level the model does not
/// accept, naming the accepted set.
pub(crate) fn resolve_model_reasoning_effort(
    processor: &ChatTemplateProcessor,
    request: &ChatCompletionRequest,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
    merged_extra_body: &Option<serde_json::Map<String, Value>>,
) -> Result<Option<String>, String> {
    let Some(steering) = processor.reasoning_effort_steering() else {
        return Ok(None);
    };
    let per_request = extract_request_kwargs(
        request.chat_template_kwargs.as_ref(),
        merged_extra_body.as_ref(),
    );
    let control = request.resolve_reasoning_control();
    let request_level = per_request
        .get(REASONING_EFFORT)
        .map(level_text)
        .or_else(|| {
            control.as_ref().and_then(|c| match (&c.effort, c.enabled) {
                (Some(effort), _) => Some(effort.clone()),
                (None, false) => Some(NONE_LEVEL.to_string()),
                (None, true) => None,
            })
        });
    if let Some(raw) = request_level {
        return steering.normalize_level(&raw).map(Some);
    }
    let request_thinking_off =
        per_request.get(ENABLE_THINKING).and_then(Value::as_bool) == Some(false);
    if request_thinking_off {
        return Ok(Some(NONE_LEVEL.to_string()));
    }
    if let Some(value) = server_default_kwargs.and_then(|kwargs| kwargs.get(REASONING_EFFORT)) {
        return steering.normalize_level(&level_text(value)).map(Some);
    }
    let level = if thinking_disabled(&per_request, control.as_ref(), server_default_kwargs) {
        NONE_LEVEL.to_string()
    } else {
        steering.default.clone()
    };
    Ok(Some(level))
}

/// A kwarg's level as text: the string itself, or a non-string's JSON form
/// (which no published level matches, so it is refused with the set).
fn level_text(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), str::to_string)
}

/// `enable_thinking` is `false` for this request, resolved as the render
/// resolves it (`resolve_effective_kwargs`): its own kwarg, else the portable
/// control (`reasoning: true` re-enables thinking over a disabled server
/// default), else the server default.
fn thinking_disabled(
    per_request: &ChatTemplateKwargs,
    control: Option<&ReasoningControl>,
    server_default_kwargs: Option<&ChatTemplateKwargs>,
) -> bool {
    if let Some(enabled) = per_request.get(ENABLE_THINKING) {
        return enabled.as_bool() == Some(false);
    }
    if let Some(control) = control {
        return !control.enabled;
    }
    server_default_kwargs
        .and_then(|kwargs| kwargs.get(ENABLE_THINKING))
        .and_then(Value::as_bool)
        == Some(false)
}

#[cfg(test)]
#[path = "reasoning_steering_tests.rs"]
mod tests;
