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

use super::*;

#[test]
fn correlation_id_uniqueness() {
    let id1 = CorrelationId::new();
    let id2 = CorrelationId::new();
    assert_ne!(id1, id2);
}

#[test]
fn correlation_id_display() {
    let id = CorrelationId::new();
    let s = format!("{id}");
    assert!(!s.is_empty());
    // UUID v4 format: 8-4-4-4-12 hex digits
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|c| *c == '-').count(), 4);
}

#[test]
fn correlation_id_from_string() {
    let original = "abc-123-def";
    let id = CorrelationId::from_string(original.to_string()).unwrap();
    assert_eq!(id.as_str(), original);
}

#[test]
fn correlation_id_from_string_rejects_empty() {
    assert!(CorrelationId::from_string(String::new()).is_none());
}

#[test]
fn correlation_id_from_string_rejects_too_long() {
    let long = "x".repeat(257);
    assert!(CorrelationId::from_string(long).is_none());
}

#[test]
fn correlation_id_from_string_accepts_max_length() {
    let max_len = "x".repeat(256);
    assert!(CorrelationId::from_string(max_len).is_some());
}

#[test]
fn correlation_id_serde_roundtrip() {
    let id = CorrelationId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: CorrelationId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

#[test]
fn request_context_new() {
    let ctx = RequestContext::new("node-0");
    assert_eq!(ctx.origin_node, "node-0");
    assert_eq!(ctx.hop_count, 0);
    assert!(ctx.parent_span.is_none());
}

#[test]
fn request_context_from_incoming_increments_hop() {
    let original = RequestContext::new("node-0");
    let forwarded = RequestContext::from_incoming(
        original.correlation_id.clone(),
        "node-1",
        original.hop_count,
    )
    .unwrap();
    assert_eq!(forwarded.correlation_id, original.correlation_id);
    assert_eq!(forwarded.origin_node, "node-1");
    assert_eq!(forwarded.hop_count, 1);
}

#[test]
fn request_context_from_incoming_rejects_max_hops() {
    let id = CorrelationId::new();
    // At max hop count, from_incoming should return None
    let result = RequestContext::from_incoming(id, "node-1", MAX_HOP_COUNT);
    assert!(result.is_none());
}

#[test]
fn request_context_from_incoming_accepts_below_max() {
    let id = CorrelationId::new();
    let result = RequestContext::from_incoming(id, "node-1", MAX_HOP_COUNT - 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().hop_count, MAX_HOP_COUNT);
}

#[test]
fn request_context_from_incoming_saturates_overflow() {
    let id = CorrelationId::new();
    // u32::MAX should saturate and exceed MAX_HOP_COUNT, returning None
    let result = RequestContext::from_incoming(id, "node-1", u32::MAX);
    assert!(result.is_none());
}

#[test]
fn request_context_with_parent_span() {
    let ctx = RequestContext::new("node-0").with_parent_span("span-42".to_string());
    assert_eq!(ctx.parent_span.as_deref(), Some("span-42"));
}

#[test]
fn request_context_bytes_roundtrip() {
    let ctx = RequestContext::new("node-0").with_parent_span("span-1".to_string());
    let bytes = ctx.to_bytes().unwrap();
    let restored = RequestContext::from_bytes(&bytes).unwrap();
    assert_eq!(restored.correlation_id, ctx.correlation_id);
    assert_eq!(restored.origin_node, ctx.origin_node);
    assert_eq!(restored.hop_count, ctx.hop_count);
    assert_eq!(restored.parent_span, ctx.parent_span);
}

#[test]
fn request_context_to_bytes_returns_some() {
    let ctx = RequestContext::new("node-0");
    assert!(ctx.to_bytes().is_some());
}

#[test]
fn request_context_from_invalid_bytes() {
    let result = RequestContext::from_bytes(b"not json");
    assert!(result.is_none());
}

#[test]
fn correlation_id_default() {
    let id: CorrelationId = Default::default();
    assert!(!id.as_str().is_empty());
}
