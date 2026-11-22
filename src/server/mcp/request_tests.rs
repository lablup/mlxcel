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
use crate::server::mcp::test_support::fake_config;

fn definition(json: serde_json::Value) -> McpToolDefinition {
    serde_json::from_value(json).expect("definition parses")
}

#[test]
fn approval_defaults_to_always() {
    let parsed = parse_mcp_tool(&definition(serde_json::json!({"server_label": "fake"}))).unwrap();
    assert_eq!(parsed.approval, ApprovalPolicy::Always);
    assert!(parsed.allowed_tools.is_none());
    assert!(parsed.approval.requires_approval("add"));
    assert!(parsed.allows("anything"));
}

#[test]
fn approval_string_and_object_forms_parse() {
    let never = parse_mcp_tool(&definition(
        serde_json::json!({"server_label": "fake", "require_approval": "never"}),
    ))
    .unwrap();
    assert!(!never.approval.requires_approval("add"));

    let mixed = parse_mcp_tool(&definition(serde_json::json!({
        "server_label": "fake",
        "require_approval": {
            "never": {"tool_names": ["add", "lookup"]},
            "always": {"tool_names": ["lookup"]}
        }
    })))
    .unwrap();
    assert!(!mixed.approval.requires_approval("add"));
    // Named on both sides, and not named at all: both need approval.
    assert!(mixed.approval.requires_approval("lookup"));
    assert!(mixed.approval.requires_approval("pid"));
}

#[test]
fn allowed_tools_array_and_filter_forms_parse() {
    for allowed in [
        serde_json::json!(["add"]),
        serde_json::json!({"tool_names": ["add"]}),
    ] {
        let parsed = parse_mcp_tool(&definition(
            serde_json::json!({"server_label": "fake", "allowed_tools": allowed}),
        ))
        .unwrap();
        assert!(parsed.allows("add"));
        assert!(!parsed.allows("lookup"));
    }
}

#[test]
fn remote_server_fields_and_bad_shapes_are_refused() {
    for (extra, needle) in [
        (
            serde_json::json!({"server_url": "https://example.com"}),
            "server_url",
        ),
        (serde_json::json!({"connector_id": "c"}), "connector_id"),
        (
            serde_json::json!({"headers": {"Authorization": "x"}}),
            "headers",
        ),
        (
            serde_json::json!({"authorization": "token"}),
            "authorization",
        ),
        (
            serde_json::json!({"require_approval": "sometimes"}),
            "sometimes",
        ),
        (
            serde_json::json!({"require_approval": 1}),
            "require_approval",
        ),
        (
            serde_json::json!({"require_approval": {"maybe": {}}}),
            "maybe",
        ),
        (serde_json::json!({"allowed_tools": "add"}), "allowed_tools"),
        (
            serde_json::json!({"allowed_tools": {"read_only": true}}),
            "read_only",
        ),
        (serde_json::json!({"allowed_tools": [1]}), "allowed_tools"),
    ] {
        let mut json = serde_json::json!({"server_label": "fake"});
        json.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let error = parse_mcp_tool(&definition(json.clone())).expect_err("must be refused");
        assert!(error.contains(needle), "{json}: {error}");
    }
}

#[test]
fn server_description_is_tolerated() {
    assert!(
        parse_mcp_tool(&definition(
            serde_json::json!({"server_label": "fake", "server_description": "numbers"})
        ))
        .is_ok()
    );
}

#[tokio::test]
async fn plan_lists_allowed_tools_and_records_their_approval_rule() {
    let runtime = McpRuntime::new(fake_config());
    let declaration = definition(serde_json::json!({
        "server_label": "fake",
        "allowed_tools": ["add", "lookup"],
        "require_approval": {"never": {"tool_names": ["add"]}}
    }));
    let plan = plan_request(&runtime, &[&declaration], &HashSet::new())
        .await
        .expect("plans");
    let names: Vec<_> = plan
        .function_tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    assert_eq!(names, ["add", "lookup"]);
    assert!(!plan.tools["add"].requires_approval);
    assert!(plan.tools["lookup"].requires_approval);
    assert_eq!(plan.tools["add"].server_label, "fake");
    assert!(plan.function_tools[0].function.parameters.is_some());
}

#[tokio::test]
async fn plan_refuses_unknown_and_repeated_servers_and_name_collisions() {
    let runtime = McpRuntime::new(fake_config());
    let fake = definition(serde_json::json!({"server_label": "fake", "allowed_tools": ["add"]}));
    let unknown = definition(serde_json::json!({"server_label": "nope"}));

    let error = plan_request(&runtime, &[&unknown], &HashSet::new())
        .await
        .expect_err("unknown server");
    assert!(
        error.contains("'nope'") && error.contains("fake"),
        "{error}"
    );

    let error = plan_request(&runtime, &[&fake, &fake], &HashSet::new())
        .await
        .expect_err("repeated server");
    assert!(error.contains("more than once"), "{error}");

    let error = plan_request(&runtime, &[&fake], &HashSet::from(["add".to_string()]))
        .await
        .expect_err("collision with a function tool");
    assert!(
        error.contains("'add'") && error.contains("function tool"),
        "{error}"
    );
}

#[tokio::test]
async fn plan_reports_an_unavailable_server_without_failing_the_request() {
    let mut config = fake_config();
    config.servers[0].command = "mlxcel-no-such-mcp-command".to_string();
    let runtime = McpRuntime::new(config);
    let declaration = definition(serde_json::json!({"server_label": "fake"}));
    let plan = plan_request(&runtime, &[&declaration], &HashSet::new())
        .await
        .expect("the request still plans");
    assert!(plan.function_tools.is_empty());
    let error = plan.servers[0].tools.as_ref().expect_err("listing failed");
    assert!(error.contains("could not be started"), "{error}");
}

#[tokio::test]
async fn two_servers_offering_one_tool_name_collide() {
    let mut config = fake_config();
    let mut second = config.servers[0].clone();
    second.name = "other".to_string();
    config.servers.push(second);
    let runtime = McpRuntime::new(config);
    let first = definition(serde_json::json!({"server_label": "fake", "allowed_tools": ["add"]}));
    let second = definition(serde_json::json!({"server_label": "other", "allowed_tools": ["add"]}));
    let error = plan_request(&runtime, &[&first, &second], &HashSet::new())
        .await
        .expect_err("same tool on two servers");
    assert!(
        error.contains("'fake'") && error.contains("'other'"),
        "{error}"
    );

    let disjoint =
        definition(serde_json::json!({"server_label": "other", "allowed_tools": ["lookup"]}));
    assert!(
        plan_request(&runtime, &[&first, &disjoint], &HashSet::new())
            .await
            .is_ok()
    );
}
