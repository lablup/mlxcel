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

use super::{
    ChatTemplateKwargs, ChatTemplateKwargsError, LLAMA_ARG_CHAT_TEMPLATE_KWARGS,
    env_fallback_chat_template_kwargs, extract_request_kwargs, merge_server_and_request,
    resolve_server_default_kwargs, strip_rolling_checkpoint, strip_think_block,
};
// Env-var fallback tests must serialize through the crate-wide `ENV_LOCK`
// per-module locks race with env mutations in unrelated
// modules of the same test binary.
use crate::test_support::env_lock::env_lock as lock_env;
use serde_json::{Map, Value, json};

// ---------------------------------------------------------------------------
// ChatTemplateKwargs::from_json_str
// ---------------------------------------------------------------------------

#[test]
fn from_json_str_empty_is_empty() {
    let k = ChatTemplateKwargs::from_json_str("").unwrap();
    assert!(k.is_empty());
    let k = ChatTemplateKwargs::from_json_str("   ").unwrap();
    assert!(k.is_empty());
}

#[test]
fn from_json_str_parses_object() {
    let k = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    assert!(!k.is_empty());
    assert_eq!(k.len(), 1);
    assert!(k.preserve_thinking());
}

#[test]
fn from_json_str_multi_key() {
    let k = ChatTemplateKwargs::from_json_str(
        r#"{"preserve_thinking": true, "enable_thinking": false}"#,
    )
    .unwrap();
    assert_eq!(k.len(), 2);
    assert!(k.preserve_thinking());
    assert_eq!(
        k.get("enable_thinking").and_then(Value::as_bool),
        Some(false)
    );
}

#[test]
fn from_json_str_rejects_non_object() {
    // Arrays are not objects.
    let err = ChatTemplateKwargs::from_json_str(r#"[true]"#).unwrap_err();
    assert!(matches!(err, ChatTemplateKwargsError::NotAnObject(_)));

    // Bare booleans, strings, numbers — all rejected.
    assert!(matches!(
        ChatTemplateKwargs::from_json_str(r#"true"#).unwrap_err(),
        ChatTemplateKwargsError::NotAnObject(_)
    ));
    assert!(matches!(
        ChatTemplateKwargs::from_json_str(r#""hi""#).unwrap_err(),
        ChatTemplateKwargsError::NotAnObject(_)
    ));
    assert!(matches!(
        ChatTemplateKwargs::from_json_str(r#"42"#).unwrap_err(),
        ChatTemplateKwargsError::NotAnObject(_)
    ));
}

#[test]
fn from_json_str_rejects_invalid_json() {
    let err = ChatTemplateKwargs::from_json_str("not json").unwrap_err();
    assert!(matches!(err, ChatTemplateKwargsError::InvalidJson(_)));
}

#[test]
fn preserve_thinking_accessor_ignores_non_bool_values() {
    // An operator typo like `{"preserve_thinking": "true"}` (string, not bool)
    // must not silently flip semantics.
    let k = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": "true"}"#).unwrap();
    assert!(!k.preserve_thinking());
}

// ---------------------------------------------------------------------------
// Env-var fallback
// ---------------------------------------------------------------------------

// Env-var fallback tests share both `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` and
// the process-wide env block. They serialize through the crate-wide
// `ENV_LOCK` imported above — per-module locks would race with env
// mutations in unrelated modules of the same test binary.

#[test]
fn env_fallback_cli_none_takes_env() {
    let _guard = lock_env();

    // SAFETY: single-threaded under the lock for the duration of this test.
    unsafe {
        std::env::set_var(
            LLAMA_ARG_CHAT_TEMPLATE_KWARGS,
            r#"{"preserve_thinking": true}"#,
        );
    }
    let mut cli: Option<String> = None;
    env_fallback_chat_template_kwargs(&mut cli);
    assert_eq!(cli.as_deref(), Some(r#"{"preserve_thinking": true}"#));

    // CLI already set — env is ignored.
    let mut cli = Some(r#"{"preserve_thinking": false}"#.to_string());
    env_fallback_chat_template_kwargs(&mut cli);
    assert_eq!(cli.as_deref(), Some(r#"{"preserve_thinking": false}"#));

    // SAFETY: single-threaded under the lock.
    unsafe {
        std::env::remove_var(LLAMA_ARG_CHAT_TEMPLATE_KWARGS);
    }

    // No env and no CLI → stays None.
    let mut cli: Option<String> = None;
    env_fallback_chat_template_kwargs(&mut cli);
    assert_eq!(cli, None);
}

#[test]
fn env_fallback_parses_identically_to_cli() {
    let _guard = lock_env();

    let raw = r#"{"preserve_thinking": true, "custom_flag": 42}"#;

    // "CLI path": parse directly.
    let from_cli = ChatTemplateKwargs::from_json_str(raw).unwrap();

    // "Env path": set env, run fallback, parse.
    // SAFETY: single-threaded under the lock.
    unsafe {
        std::env::set_var(LLAMA_ARG_CHAT_TEMPLATE_KWARGS, raw);
    }
    let mut cli: Option<String> = None;
    env_fallback_chat_template_kwargs(&mut cli);
    let from_env = ChatTemplateKwargs::from_json_str(cli.as_deref().unwrap()).unwrap();
    // SAFETY: single-threaded under the lock.
    unsafe {
        std::env::remove_var(LLAMA_ARG_CHAT_TEMPLATE_KWARGS);
    }

    assert_eq!(from_cli, from_env);
    assert!(from_env.preserve_thinking());
    assert_eq!(
        from_env.get("custom_flag").and_then(Value::as_i64),
        Some(42)
    );
}

#[test]
fn resolve_server_default_empty_string_is_none() {
    assert_eq!(resolve_server_default_kwargs(None).unwrap(), None);
    assert_eq!(resolve_server_default_kwargs(Some("")).unwrap(), None);
    assert_eq!(resolve_server_default_kwargs(Some("   ")).unwrap(), None);
    // `{}` parses as an empty object, still collapses to None.
    assert_eq!(resolve_server_default_kwargs(Some("{}")).unwrap(), None);
}

#[test]
fn resolve_server_default_valid_object_returns_some() {
    let out = resolve_server_default_kwargs(Some(r#"{"preserve_thinking": true}"#))
        .unwrap()
        .unwrap();
    assert!(out.preserve_thinking());
}

// ---------------------------------------------------------------------------
// extract_request_kwargs — precedence
// ---------------------------------------------------------------------------

fn obj(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), v.clone());
    }
    m
}

#[test]
fn extract_top_level_wins_over_extra_body_nested() {
    let top = obj(&[("preserve_thinking", json!(true))]);
    let extra = obj(&[(
        "chat_template_kwargs",
        json!({ "preserve_thinking": false }),
    )]);
    let out = extract_request_kwargs(Some(&top), Some(&extra));
    assert!(out.preserve_thinking(), "top-level must win");
}

#[test]
fn extract_top_level_wins_over_extra_body_flat() {
    let top = obj(&[("preserve_thinking", json!(true))]);
    let extra = obj(&[("preserve_thinking", json!(false))]);
    let out = extract_request_kwargs(Some(&top), Some(&extra));
    assert!(out.preserve_thinking());
}

#[test]
fn extract_extra_body_nested_wins_over_flat() {
    let extra = obj(&[
        ("chat_template_kwargs", json!({ "preserve_thinking": true })),
        ("preserve_thinking", json!(false)),
    ]);
    let out = extract_request_kwargs(None, Some(&extra));
    assert!(out.preserve_thinking());
}

#[test]
fn extract_extra_body_flat_is_last_resort() {
    let extra = obj(&[("preserve_thinking", json!(true))]);
    let out = extract_request_kwargs(None, Some(&extra));
    assert!(out.preserve_thinking());
    assert_eq!(out.len(), 1);
}

#[test]
fn extract_extra_body_flat_only_recognizes_preserve_thinking() {
    // A random DashScope-style flat key other than preserve_thinking is
    // NOT promoted into kwargs. This guards against the flat shape becoming
    // a general fallback and polluting the kwargs namespace.
    let extra = obj(&[("enable_thinking", json!(false))]);
    let out = extract_request_kwargs(None, Some(&extra));
    assert!(out.is_empty());
}

#[test]
fn extract_extra_body_flat_requires_bool() {
    // Non-bool values for the flat key are ignored, matching the bool-only
    // semantics of the DashScope shape.
    let extra = obj(&[("preserve_thinking", json!("yes"))]);
    let out = extract_request_kwargs(None, Some(&extra));
    assert!(out.is_empty());
}

#[test]
fn extract_empty_inputs_yield_empty_kwargs() {
    let out = extract_request_kwargs(None, None);
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// merge_server_and_request
// ---------------------------------------------------------------------------

#[test]
fn merge_empty_request_keeps_server_defaults() {
    let server = ChatTemplateKwargs::from_json_str(
        r#"{"preserve_thinking": true, "enable_thinking": false}"#,
    )
    .unwrap();
    let out = merge_server_and_request(Some(&server), &ChatTemplateKwargs::new());
    assert_eq!(out, server);
}

#[test]
fn merge_per_request_overrides_per_key() {
    let server = ChatTemplateKwargs::from_json_str(
        r#"{"preserve_thinking": false, "enable_thinking": false}"#,
    )
    .unwrap();
    let req = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let out = merge_server_and_request(Some(&server), &req);
    assert!(out.preserve_thinking());
    // unrelated server-default key persists
    assert_eq!(
        out.get("enable_thinking").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(out.len(), 2);
}

#[test]
fn merge_no_server_default_uses_request() {
    let req = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let out = merge_server_and_request(None, &req);
    assert_eq!(out, req);
}

#[test]
fn merge_no_server_and_empty_request_is_empty() {
    let out = merge_server_and_request(None, &ChatTemplateKwargs::new());
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// Rolling-checkpoint stripper
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Msg {
    role: &'static str,
    #[allow(dead_code)]
    content: &'static str,
}

fn role(m: &Msg) -> &str {
    m.role
}

fn content(m: &Msg) -> &str {
    m.content
}

#[test]
fn rolling_checkpoint_three_turn_conversation() {
    // u0, a0, u1, a1, u2 — threshold is index 4 (the last "user").
    // Before threshold: u0(0), a0(1), u1(2), a1(3). Of those, a0 and a1
    // are assistant replies, so both get stripped.
    let messages = vec![
        Msg {
            role: "user",
            content: "q1",
        },
        Msg {
            role: "assistant",
            content: "<think>t1</think>a1",
        },
        Msg {
            role: "user",
            content: "q2",
        },
        Msg {
            role: "assistant",
            content: "<think>t2</think>a2",
        },
        Msg {
            role: "user",
            content: "q3",
        },
    ];
    let strip_indices = strip_rolling_checkpoint(&messages, role, content);
    assert_eq!(strip_indices, vec![1, 3]);
}

#[test]
fn rolling_checkpoint_tool_call_turn_does_not_anchor() {
    // The tool turn at index 3 must not count as the latest user turn —
    // the anchor must be the genuine "user" turn at index 2.
    let messages = vec![
        Msg {
            role: "user",
            content: "q1",
        },
        Msg {
            role: "assistant",
            content: "<think>t1</think>calling tool",
        },
        Msg {
            role: "user",
            content: "q2",
        },
        Msg {
            role: "tool",
            content: "tool_result",
        },
    ];
    // Threshold = 2 (the last "user"). Indices strictly < 2 that are
    // "assistant": only index 1.
    let strip_indices = strip_rolling_checkpoint(&messages, role, content);
    assert_eq!(strip_indices, vec![1]);
}

#[test]
fn rolling_checkpoint_no_user_turns_strips_nothing() {
    let messages = vec![
        Msg {
            role: "system",
            content: "s",
        },
        Msg {
            role: "assistant",
            content: "<think>t</think>a",
        },
    ];
    let strip_indices = strip_rolling_checkpoint(&messages, role, content);
    assert!(strip_indices.is_empty());
}

#[test]
fn rolling_checkpoint_single_turn_strips_nothing() {
    let messages = vec![Msg {
        role: "user",
        content: "hi",
    }];
    let strip_indices = strip_rolling_checkpoint(&messages, role, content);
    assert!(strip_indices.is_empty());
}

#[test]
fn rolling_checkpoint_pseudo_user_tool_response_does_not_anchor() {
    let messages = vec![
        Msg {
            role: "user",
            content: "q1",
        },
        Msg {
            role: "assistant",
            content: "<think>t1</think>a1",
        },
        Msg {
            role: "user",
            content: "q2",
        },
        Msg {
            role: "assistant",
            content: "<think>t2</think>a2",
        },
        Msg {
            role: "user",
            content: "<tool_response>tool output</tool_response>",
        },
    ];
    let strip_indices = strip_rolling_checkpoint(&messages, role, content);
    assert_eq!(strip_indices, vec![1]);
}

// ---------------------------------------------------------------------------
// strip_think_block
// ---------------------------------------------------------------------------

#[test]
fn strip_think_block_removes_block_and_surrounding_newlines() {
    let content = "<think>planning</think>\n\nthe answer";
    let out = strip_think_block(content);
    assert_eq!(out, "the answer");
}

#[test]
fn strip_think_block_preserves_content_before_block() {
    let content = "prefix\n\n<think>planning</think>\n\nthe answer";
    let out = strip_think_block(content);
    // Up to 2 newlines consumed on each side.
    assert_eq!(out, "prefixthe answer");
}

#[test]
fn strip_think_block_without_block_returns_borrowed() {
    let content = "no block here";
    let out = strip_think_block(content);
    assert_eq!(out, "no block here");
    match out {
        std::borrow::Cow::Borrowed(_) => {}
        std::borrow::Cow::Owned(_) => panic!("expected borrowed Cow for a no-op"),
    }
}

#[test]
fn strip_think_block_malformed_open_without_close_preserves_content() {
    let content = "<think>hanging";
    let out = strip_think_block(content);
    assert_eq!(out, "<think>hanging");
}

#[test]
fn strip_think_block_multiline_block() {
    let content = "<think>\nline 1\nline 2\n</think>\n\nanswer";
    let out = strip_think_block(content);
    assert_eq!(out, "answer");
}
