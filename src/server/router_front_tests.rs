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

//! Unit tests for the disaggregated router front-end's usage accounting
//! (issue #387).
//!
//! Model-free: exercise `resolve_completion_tokens`, the pure function that
//! turns the worker's authoritative wire-carried token count (or the emitted-
//! piece fallback) into the reported `usage.completion_tokens`.

use super::*;

/// When the worker reports an authoritative count, it is used verbatim even if
/// it differs from the emitted-piece count. This is the byte-fallback case: a
/// Gemma `<0xXX>` sequence produces fewer text pieces than model tokens, so the
/// frame count under-counts and the authoritative count corrects it.
#[test]
fn prefers_authoritative_count_over_frame_count() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(10),
    };
    // Three multi-byte characters surfaced as 3 text pieces, but the worker
    // generated 10 model tokens (byte-fallback expansion).
    assert_eq!(resolve_completion_tokens(&outcome, 3, 16), 10);
}

/// With no authoritative count on the wire (a mixed-version cluster running an
/// older prefill or decode node), the router falls back to the emitted-piece
/// count, preserving the prior behavior.
#[test]
fn falls_back_to_frame_count_when_no_authoritative_count() {
    let outcome = HandoffOutcome {
        generated_tokens: None,
    };
    assert_eq!(resolve_completion_tokens(&outcome, 7, 16), 7);
}

/// When no authoritative count is available and the emitted-piece count exceeds
/// `max_tokens` (which can happen for a hostile or buggy node in a mixed-version
/// cluster), the fallback is clamped to `max_tokens` for uniform defensiveness.
#[test]
fn fallback_frame_count_clamped_to_max_tokens() {
    let outcome = HandoffOutcome {
        generated_tokens: None,
    };
    // Frame count exceeds the budget; the fallback must not report more than allowed.
    assert_eq!(resolve_completion_tokens(&outcome, 9999, 16), 16);
}

/// The authoritative count is clamped to `max_tokens`: the router bounds
/// generation to its own budget, so a larger reported count (a buggy or hostile
/// node) must not inflate the usage figure.
#[test]
fn clamps_authoritative_count_to_max_tokens() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(9999),
    };
    assert_eq!(resolve_completion_tokens(&outcome, 4, 16), 16);
}

/// An authoritative count exactly at the budget is reported as-is and drives the
/// "length" finish_reason (`count >= max_tokens`), matching single-node.
#[test]
fn authoritative_count_at_budget_is_length() {
    let outcome = HandoffOutcome {
        generated_tokens: Some(16),
    };
    let max_tokens = 16;
    let completion_tokens = resolve_completion_tokens(&outcome, 16, max_tokens);
    assert_eq!(completion_tokens, 16);
    assert!(
        completion_tokens >= max_tokens,
        "should map to finish=length"
    );
}

/// A byte-fallback under-count that would have flipped the finish_reason: the
/// frame count (15) is below the budget (16) and would report "stop", but the
/// authoritative count (16) correctly reports "length".
#[test]
fn authoritative_count_fixes_finish_reason_flip() {
    let max_tokens = 16;
    let frame_counted = 15; // emitted-piece under-count
    let authoritative = HandoffOutcome {
        generated_tokens: Some(16),
    };
    let no_count = HandoffOutcome {
        generated_tokens: None,
    };

    let fixed = resolve_completion_tokens(&authoritative, frame_counted, max_tokens);
    let legacy = resolve_completion_tokens(&no_count, frame_counted, max_tokens);

    assert!(fixed >= max_tokens, "authoritative count reports length");
    assert!(
        legacy < max_tokens,
        "frame-count fallback would have reported stop"
    );
}

// ── /router/stats topology-disclosure redaction (issue #389) ────────────────

/// Build a small `StatsNode` fixture for the redaction tests.
fn stats_node(id: &str, role: &str, status: &str, address: &str) -> StatsNode {
    StatsNode {
        id: id.to_string(),
        role: role.to_string(),
        status: status.to_string(),
        address: address.to_string(),
    }
}

/// The default (redacted) `/router/stats` body omits every node's raw address
/// but keeps the stable id, role, health, and dispatch counts, so an operator
/// can still read the load spread without learning the internal topology.
#[test]
fn router_stats_redacts_addresses_by_default() {
    let nodes = vec![
        stats_node("prefill-0", "prefill", "online", "10.0.0.1:9001"),
        stats_node("decode-0", "decode", "online", "10.0.0.2:9002"),
    ];
    let mut prefill_hits = HashMap::new();
    prefill_hits.insert("prefill-0".to_string(), 3u64);
    let decode_hits = HashMap::new();
    let metrics = crate::distributed::disaggregated::RouterMetrics::default();

    let body = router_stats_body(&nodes, &prefill_hits, &decode_hits, &metrics, false);

    assert_eq!(body["addresses_redacted"], serde_json::json!(true));
    let nodes_json = body["nodes"].as_array().expect("nodes array");
    assert_eq!(nodes_json.len(), 2);
    for node in nodes_json {
        assert!(
            node.get("address").is_none(),
            "redacted view must not carry a raw address: {node}"
        );
        assert!(node.get("id").is_some(), "id is retained");
        assert!(node.get("status").is_some(), "health is retained");
    }
    // The serialized body must contain neither raw address anywhere.
    let serialized = serde_json::to_string(&body).expect("serialize");
    assert!(
        !serialized.contains("10.0.0.1") && !serialized.contains("10.0.0.2"),
        "no raw address may leak into the redacted body: {serialized}"
    );
    // Dispatch counts (keyed by stable node id) are still reported.
    assert_eq!(body["prefill_hits"]["prefill-0"], serde_json::json!(3));
}

/// The opt-in verbose view (env-gated) restores each node's raw address for
/// trusted-segment debugging.
#[test]
fn router_stats_verbose_includes_addresses() {
    let nodes = vec![stats_node(
        "decode-1",
        "decode",
        "unreachable",
        "10.0.0.3:9003",
    )];
    let metrics = crate::distributed::disaggregated::RouterMetrics::default();

    let body = router_stats_body(&nodes, &HashMap::new(), &HashMap::new(), &metrics, true);

    assert_eq!(body["addresses_redacted"], serde_json::json!(false));
    assert_eq!(
        body["nodes"][0]["address"],
        serde_json::json!("10.0.0.3:9003")
    );
}

/// Only the documented truthy spellings enable the verbose view; anything else
/// (including unset and empty) keeps the redacted default.
#[test]
fn router_stats_verbose_env_parsing() {
    for truthy in ["1", "true", "TRUE", "Yes", "on", " on "] {
        assert!(
            router_stats_verbose_from(Some(truthy)),
            "{truthy:?} should enable verbose"
        );
    }
    for falsy in [
        None,
        Some(""),
        Some("0"),
        Some("false"),
        Some("off"),
        Some("nope"),
    ] {
        assert!(
            !router_stats_verbose_from(falsy),
            "{falsy:?} should keep redaction"
        );
    }
}
