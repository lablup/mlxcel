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

//! Tool call output parsing and formatting.
//!
//! This module detects and parses tool call patterns from model output,
//! supporting multiple formats used by popular model families (Hermes/Qwen,
//! Llama 3.x, Mistral Nemo, Functionary, etc.).
//!
//! Used by: routes/chat, chat_request

mod atem;
mod formats;
pub mod parser;
pub mod stream_filter;
pub mod types;

pub use atem::render_muse_channels_for_display;
pub use parser::{clean_structural_tokens, generate_tool_call_id, parse_tool_calls};
pub use types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};

use super::types::request::ChatCompletionRequest;
use super::types::response::{ToolCallFunctionResponse, ToolCallResponse};

/// Infer a default tool-call format from model/config/template identity.
///
/// This is intentionally conservative and never handles an explicit operator
/// parser choice; callers should pass explicit choices through
/// [`resolve_tool_call_format`] so user configuration remains authoritative.
pub fn infer_default_tool_call_format(
    model_id: Option<&str>,
    model_type: Option<&str>,
    template_source: Option<&str>,
) -> Option<ToolCallFormat> {
    if template_source.is_some_and(template_uses_atem)
        || model_type.is_some_and(is_muse_glimmer_identity)
        || model_id.is_some_and(is_muse_glimmer_identity)
    {
        Some(ToolCallFormat::Atem)
    } else {
        None
    }
}

/// Resolve the active tool-call format without overriding an explicit choice.
pub fn resolve_tool_call_format(
    explicit: Option<ToolCallFormat>,
    model_id: Option<&str>,
    model_type: Option<&str>,
    template_source: Option<&str>,
) -> Option<ToolCallFormat> {
    explicit.or_else(|| infer_default_tool_call_format(model_id, model_type, template_source))
}

fn template_uses_atem(template: &str) -> bool {
    template.contains("<atem:function_calls>") && template.contains("<atem:invoke")
}

fn is_muse_glimmer_identity(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase().replace('-', "_");
    normalized.contains("muse_glimmer")
}

/// Check if tool call parsing should be attempted for this request.
///
/// Returns false when no tools are provided or tool_choice is "none".
// Used by: routes/chat
pub fn should_parse_tool_calls(request: &ChatCompletionRequest) -> bool {
    let Some(ref tools) = request.tools else {
        return false;
    };
    if tools.is_empty() {
        return false;
    }
    if let Some(ref tc) = request.tool_choice
        && tc.is_none()
    {
        return false;
    }
    true
}

/// Convert parsed tool calls to response format, filtering by specific
/// function name when tool_choice selects one.
// Used by: routes/chat
pub fn build_tool_call_responses(
    parsed: &ToolCallParseResult,
    request: &ChatCompletionRequest,
) -> Vec<ToolCallResponse> {
    let specific_fn = request
        .tool_choice
        .as_ref()
        .and_then(|tc| tc.specific_function());

    parsed
        .tool_calls
        .iter()
        .filter(|c| {
            if let Some(fn_name) = specific_fn {
                c.name == fn_name
            } else {
                true
            }
        })
        .map(|c| ToolCallResponse {
            id: generate_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunctionResponse {
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            },
        })
        .collect()
}
