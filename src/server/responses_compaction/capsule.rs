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

//! Capsule encoding, replay and input-item JSON for Responses compaction
//! (issue #2397). See the parent module for the format and its rationale.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::{Value, json};

use super::{CAPSULE_VERSION, SUMMARY_PREFIX};
use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputRole, ResponseToolOutput,
};
use crate::server::types::responses_response::ResponseCompactionItem;

// -- Capsule -------------------------------------------------------------------

/// Decoded `encrypted_content`.
#[derive(Debug, Clone)]
pub(crate) struct Capsule {
    pub summary: String,
    pub retained: Vec<ResponseInputItem>,
}

/// Encode a version-1 capsule.
///
/// Used by: `POST /v1/responses/compact`, and the phase-B automatic path.
pub(crate) fn encode_capsule(summary: &str, retained: &[ResponseInputItem]) -> String {
    let payload = json!({
        "v": CAPSULE_VERSION,
        "summary": summary,
        "retained": retained.iter().map(input_item_to_json).collect::<Vec<_>>(),
    });
    URL_SAFE_NO_PAD.encode(payload.to_string())
}

/// Decode a capsule, refusing anything this server did not write.
///
/// `max_encoded_bytes` is the request-body limit: the capsule arrived inside
/// the body, so this only restates that bound before any decoding work. The
/// retained items go through the same `ResponseInputItem` parser as `input`,
/// and a nested capsule is refused so replay stays one level deep.
pub(crate) fn decode_capsule(encoded: &str, max_encoded_bytes: usize) -> Result<Capsule, String> {
    if encoded.len() > max_encoded_bytes {
        return Err(format!(
            "encrypted_content is {} bytes, above the {max_encoded_bytes}-byte request limit",
            encoded.len()
        ));
    }
    // Accept padded base64url too; this server writes it unpadded.
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded.trim_end_matches('='))
        .map_err(|err| format!("encrypted_content is not base64url: {err}"))?;
    let mut payload: Value = serde_json::from_slice(&bytes)
        .map_err(|err| format!("encrypted_content is not a capsule: {err}"))?;
    let version = payload.get("v").and_then(Value::as_u64);
    if version != Some(CAPSULE_VERSION) {
        return Err(format!(
            "unsupported capsule version {}; this server accepts v{CAPSULE_VERSION}",
            payload
                .get("v")
                .map_or("<missing>".to_string(), Value::to_string)
        ));
    }
    let summary = match payload.get("summary") {
        Some(Value::String(summary)) => summary.clone(),
        _ => return Err("capsule summary must be a string".to_string()),
    };
    let retained = payload
        .get_mut("retained")
        .map(Value::take)
        .ok_or_else(|| "capsule retained items are missing".to_string())?;
    let retained: Vec<ResponseInputItem> = serde_json::from_value(retained)
        .map_err(|err| format!("capsule retained items are invalid: {err}"))?;
    if retained
        .iter()
        .any(|item| matches!(item, ResponseInputItem::Compaction { .. }))
    {
        return Err("a capsule must not retain another compaction item".to_string());
    }
    Ok(Capsule { summary, retained })
}

/// The output item that carries `encoded`.
pub(crate) fn compaction_item(encoded: String) -> ResponseCompactionItem {
    ResponseCompactionItem {
        id: format!("cmp_{}", crate::server::responses_translator::short_uuid()),
        encrypted_content: encoded,
    }
}

// -- Replay --------------------------------------------------------------------

/// The assistant message a summary replays as. Empty summaries replay as
/// nothing.
pub(crate) fn summary_message(summary: &str) -> Option<ResponseInputItem> {
    let summary = summary.trim();
    (!summary.is_empty()).then(|| ResponseInputItem::Message {
        role: ResponseInputRole::Assistant,
        content: ResponseInputContent::Text(format!("{SUMMARY_PREFIX}{summary}")),
        name: None,
    })
}

/// Expand the newest compaction item in `items`.
///
/// Everything before it except system and developer messages is dropped,
/// older capsules included; its summary and retained items take its place,
/// followed by the items after it. History without a capsule is returned
/// unchanged.
///
/// System and developer messages that are exactly equal (same role and
/// content) are kept once, at their first occurrence: a client that resends
/// the compact `output` over `conversation` or `previous_response_id` history
/// would otherwise render its instructions twice.
///
/// Used by: `responses_request_to_chat` (so `/v1/responses`,
/// `/v1/responses/input_tokens` and `previous_response_id` chains all replay
/// alike) and `POST /v1/responses/compact`.
pub(crate) fn resolve_compaction(
    mut items: Vec<ResponseInputItem>,
) -> Result<Vec<ResponseInputItem>, String> {
    let Some(canonical) = canonical_compaction_input(&items) else {
        return Ok(items);
    };
    items = canonical;
    let position = items
        .iter()
        .position(|item| matches!(item, ResponseInputItem::Compaction { .. }))
        .expect("canonical compaction input contains its capsule");
    let tail = items.split_off(position + 1);
    let Some(ResponseInputItem::Compaction { encrypted_content }) = items.pop() else {
        unreachable!("rposition found a compaction item at this index");
    };
    let capsule = decode_capsule(
        &encrypted_content,
        crate::server::app::main_json_body_limit_bytes(),
    )?;
    let dropped = items.iter().filter(|item| !is_instruction(item)).count();
    tracing::debug!(
        dropped,
        retained = capsule.retained.len(),
        "expanding Responses compaction capsule"
    );
    let mut out: Vec<ResponseInputItem> = items.into_iter().filter(is_instruction).collect();
    out.extend(summary_message(&capsule.summary));
    out.extend(capsule.retained);
    out.extend(tail);
    let mut seen = std::collections::HashSet::new();
    out.retain(|item| !is_instruction(item) || seen.insert(input_item_to_json(item).to_string()));
    Ok(out)
}

/// Keep the newest capsule and its tail as a self-contained stored input.
/// Exact resent instructions are deduplicated, including their role and name;
/// different instructions remain in order. Without compaction, storage retains
/// its existing current-request-only semantics.
pub(crate) fn canonical_compaction_input(
    items: &[ResponseInputItem],
) -> Option<Vec<ResponseInputItem>> {
    let position = items
        .iter()
        .rposition(|item| matches!(item, ResponseInputItem::Compaction { .. }))?;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for item in items[..position].iter().filter(|item| is_instruction(item)) {
        if seen.insert(input_item_to_json(item).to_string()) {
            out.push(item.clone());
        }
    }
    out.extend_from_slice(&items[position..]);
    Some(out)
}

/// Whether `item` is a system or developer message, which compaction never
/// summarizes away.
pub(crate) fn is_instruction(item: &ResponseInputItem) -> bool {
    matches!(
        item,
        ResponseInputItem::Message {
            role: ResponseInputRole::System | ResponseInputRole::Developer,
            ..
        }
    )
}

/// Wire JSON of an input item, accepted back by the `input` parser.
///
/// Used by: the capsule's `retained` list and the compact response `output`.
pub(crate) fn input_item_to_json(item: &ResponseInputItem) -> Value {
    match item {
        ResponseInputItem::Message {
            role,
            content,
            name,
        } => {
            let role = match role {
                ResponseInputRole::User => "user",
                ResponseInputRole::Assistant => "assistant",
                ResponseInputRole::System => "system",
                ResponseInputRole::Developer => "developer",
            };
            let content = match content {
                ResponseInputContent::Text(text) => json!(text),
                ResponseInputContent::Parts(parts) => {
                    Value::Array(parts.iter().map(|part| part.to_json_value()).collect())
                }
            };
            let mut value = json!({"type": "message", "role": role, "content": content});
            if let Some(name) = name {
                value["name"] = json!(name);
            }
            value
        }
        ResponseInputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }),
        ResponseInputItem::FunctionCallOutput { call_id, output } => {
            let output = match output {
                ResponseToolOutput::Text(text) => json!(text),
                ResponseToolOutput::Parts(parts) => {
                    Value::Array(parts.iter().map(|part| part.to_json_value()).collect())
                }
            };
            json!({"type": "function_call_output", "call_id": call_id, "output": output})
        }
        ResponseInputItem::Reasoning { content } => json!({
            "type": "reasoning",
            "summary": [],
            "content": content
                .iter()
                .map(|part| json!({"type": part.part_type, "text": part.text}))
                .collect::<Vec<_>>(),
        }),
        ResponseInputItem::Compaction { encrypted_content } => json!({
            "type": "compaction",
            "encrypted_content": encrypted_content,
        }),
    }
}
