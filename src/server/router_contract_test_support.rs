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

#![allow(dead_code)]

//! Whole-value producer comparisons against fixtures validated by the pinned
//! JSON Schema gate. Only dynamic IDs, timestamps and revisions are normalized;
//! missing fields, extra fields and all static values must match exactly.

use serde_json::Value;

#[derive(Clone, Copy)]
enum Dynamic {
    Token,
    EventId,
    ModelId,
    Timestamp,
    Revision,
    Sequence,
}

fn fixture(source: &str) -> Value {
    let mut value: Value = serde_json::from_str(source).expect("canonical fixture JSON");
    value
        .as_object_mut()
        .expect("fixture object")
        .remove("$schemaName");
    value
}

fn operation_fixture() -> Value {
    fixture(include_str!(
        "../../tests/fixtures/webui/examples/operation.succeeded.json"
    ))
}

fn download_running_fixture() -> Value {
    fixture(include_str!(
        "../../tests/fixtures/webui/examples/operation.download-running.json"
    ))
}

fn download_succeeded_fixture() -> Value {
    fixture(include_str!(
        "../../tests/fixtures/webui/examples/operation.download-succeeded.json"
    ))
}

fn operation_accepted_fixture() -> Value {
    fixture(include_str!(
        "../../tests/fixtures/webui/examples/operation.accepted.json"
    ))
}

fn download_operation_dynamic(prefix: &str, include_result: bool) -> Vec<(String, Dynamic)> {
    let mut fields = vec![
        ("/operation_id", Dynamic::Token),
        ("/created_at", Dynamic::Timestamp),
        ("/updated_at", Dynamic::Timestamp),
    ];
    if include_result {
        fields.push(("/result/model_id", Dynamic::ModelId));
    }
    fields
        .into_iter()
        .map(|(path, kind)| (format!("{prefix}{path}"), kind))
        .collect()
}

fn operation_dynamic(prefix: &str) -> Vec<(String, Dynamic)> {
    [
        ("/operation_id", Dynamic::Token),
        ("/created_at", Dynamic::Timestamp),
        ("/updated_at", Dynamic::Timestamp),
        ("/target/model_id", Dynamic::ModelId),
        ("/target/requested_revision", Dynamic::Revision),
        ("/target/eviction_target_id", Dynamic::ModelId),
        ("/result/model_id", Dynamic::ModelId),
        ("/result/revision", Dynamic::Revision),
        ("/result/eviction/requested_target_id", Dynamic::ModelId),
        ("/result/eviction/displaced_model_id", Dynamic::ModelId),
    ]
    .into_iter()
    .map(|(path, kind)| (format!("{prefix}{path}"), kind))
    .collect()
}

fn valid_dynamic(value: &Value, kind: Dynamic) -> bool {
    match kind {
        Dynamic::Token => value.as_str().is_some_and(|s| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'~' | b'-'))
        }),
        Dynamic::EventId => value.as_str().is_some_and(|s| {
            s.starts_with("evt_")
                && s.len() > 4
                && s.len() <= 160
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'~' | b'-'))
        }),
        Dynamic::ModelId => value.as_str().is_some_and(|s| {
            s.len() == 47
                && s.starts_with("mdl_")
                && s.as_bytes()[4..]
                    .iter()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
        }),
        Dynamic::Timestamp => value
            .as_str()
            .is_some_and(|s| chrono::DateTime::parse_from_rfc3339(s).is_ok()),
        Dynamic::Revision => value.as_u64().is_some_and(|n| n > 0),
        Dynamic::Sequence => value.as_u64().is_some(),
    }
}

fn matches_fixture(
    actual: &Value,
    expected: &Value,
    dynamic: &[(String, Dynamic)],
) -> Result<(), String> {
    let mut normalized = actual.clone();
    for (path, kind) in dynamic {
        // pointer_mut never inserts absent keys, unlike Value's index operator.
        let field = normalized
            .pointer_mut(path)
            .ok_or_else(|| format!("missing dynamic field {path}"))?;
        if !valid_dynamic(field, *kind) {
            return Err(format!("invalid dynamic field {path}: {field}"));
        }
        *field = expected
            .pointer(path)
            .ok_or_else(|| format!("fixture missing {path}"))?
            .clone();
    }
    if normalized != *expected {
        return Err(format!(
            "whole-value contract mismatch\nactual: {normalized}\nfixture: {expected}"
        ));
    }
    Ok(())
}

pub(super) fn assert_operation_accepted(value: &Value) {
    let expected = operation_accepted_fixture();
    let dynamic = [("/operation_id".into(), Dynamic::Token)];
    matches_fixture(value, &expected, &dynamic).unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_download_operation_running(value: &Value) {
    matches_fixture(
        value,
        &download_running_fixture(),
        &download_operation_dynamic("", false),
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_download_operation_succeeded(value: &Value) {
    matches_fixture(
        value,
        &download_succeeded_fixture(),
        &download_operation_dynamic("", true),
    )
    .unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_download_progress_event(value: &Value) {
    let expected = fixture(include_str!(
        "../../tests/fixtures/webui/examples/event.2.json"
    ));
    let dynamic = [
        ("/server_instance_id".into(), Dynamic::Token),
        ("/event_id".into(), Dynamic::EventId),
        ("/sequence".into(), Dynamic::Sequence),
        ("/emitted_at".into(), Dynamic::Timestamp),
        ("/payload/operation_id".into(), Dynamic::Token),
    ];
    matches_fixture(value, &expected, &dynamic).unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_operation(operation: &Value, target: &str, eviction_target: &str) {
    for path in ["/target/model_id", "/result/model_id"] {
        assert_eq!(
            operation.pointer(path).and_then(Value::as_str),
            Some(target)
        );
    }
    for path in [
        "/target/eviction_target_id",
        "/result/eviction/requested_target_id",
        "/result/eviction/displaced_model_id",
    ] {
        assert_eq!(
            operation.pointer(path).and_then(Value::as_str),
            Some(eviction_target)
        );
    }
    matches_fixture(operation, &operation_fixture(), &operation_dynamic(""))
        .unwrap_or_else(|error| panic!("{error}"));
}

pub(super) fn assert_operation_list(list: &Value, target: &str, eviction_target: &str) {
    let expected = fixture(include_str!(
        "../../tests/fixtures/webui/examples/operations.succeeded-list.json"
    ));
    let mut dynamic = operation_dynamic("/items/0");
    dynamic.extend([
        ("/server_instance_id".into(), Dynamic::Token),
        ("/snapshot_sequence".into(), Dynamic::Sequence),
    ]);
    matches_fixture(list, &expected, &dynamic).unwrap_or_else(|error| panic!("{error}"));
    assert_operation(&list["items"][0], target, eviction_target);
}

pub(super) fn assert_gap_event(chunk: &str, server_instance: &str) {
    let data = chunk
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("SSE JSON data");
    let actual: Value = serde_json::from_str(data).expect("producer event JSON");
    let event_id = chunk
        .lines()
        .find_map(|line| line.strip_prefix("id: "))
        .expect("SSE id");
    assert_eq!(actual["server_instance_id"], server_instance);
    let sequence = actual["sequence"].as_u64().expect("event sequence");
    assert_eq!(event_id, format!("evt_{server_instance}_gap_{sequence:08}"));
    assert_eq!(actual["event_id"], event_id);
    let expected = fixture(include_str!(
        "../../tests/fixtures/webui/examples/event.gap.json"
    ));
    let dynamic = [
        ("/server_instance_id".into(), Dynamic::Token),
        ("/event_id".into(), Dynamic::EventId),
        ("/sequence".into(), Dynamic::Sequence),
        ("/emitted_at".into(), Dynamic::Timestamp),
    ];
    matches_fixture(&actual, &expected, &dynamic).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn whole_value_comparison_rejects_missing_extra_and_discriminator_drift() {
    let expected = operation_fixture();
    let dynamic = operation_dynamic("");
    assert!(matches_fixture(&expected, &expected, &dynamic).is_ok());
    for path in [
        "/error",
        "/cancel_reason",
        "/progress/total_bytes",
        "/result/lifecycle/last_error",
    ] {
        let mut missing = expected.clone();
        let (parent, key) = path.rsplit_once('/').unwrap();
        missing
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(
            matches_fixture(&missing, &expected, &dynamic).is_err(),
            "accepted missing {path}"
        );
    }
    for parent in [
        "",
        "/target",
        "/result",
        "/result/lifecycle",
        "/result/eviction",
    ] {
        let mut extra = expected.clone();
        extra
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), Value::Null);
        assert!(
            matches_fixture(&extra, &expected, &dynamic).is_err(),
            "accepted extra at {parent}"
        );
    }
    for path in [
        "/kind",
        "/target/target_kind",
        "/result/result_kind",
        "/result/lifecycle/state",
        "/result/eviction/outcome",
    ] {
        let mut changed = expected.clone();
        *changed.pointer_mut(path).unwrap() = Value::String("unknown".into());
        assert!(
            matches_fixture(&changed, &expected, &dynamic).is_err(),
            "accepted changed {path}"
        );
    }
}

#[test]
fn normalization_rejects_invalid_or_absent_dynamic_values() {
    let expected = operation_fixture();
    let dynamic = operation_dynamic("");
    for (path, invalid) in [
        ("/operation_id", serde_json::json!("bad\r\ntoken")),
        ("/target/model_id", serde_json::json!("mdl_short")),
        ("/created_at", serde_json::json!("not a timestamp")),
        ("/result/revision", serde_json::json!(0)),
    ] {
        let mut changed = expected.clone();
        *changed.pointer_mut(path).unwrap() = invalid;
        assert!(
            matches_fixture(&changed, &expected, &dynamic).is_err(),
            "accepted invalid {path}"
        );
        let (parent, key) = path.rsplit_once('/').unwrap();
        changed
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(
            matches_fixture(&changed, &expected, &dynamic).is_err(),
            "inserted absent {path}"
        );
    }
}

#[test]
fn list_and_event_comparison_preserves_nullable_and_unknown_fields() {
    for (source, missing_path) in [
        (
            include_str!("../../tests/fixtures/webui/examples/operations.succeeded-list.json"),
            "/pagination/next_cursor",
        ),
        (
            include_str!("../../tests/fixtures/webui/examples/event.gap.json"),
            "/payload/resnapshot",
        ),
    ] {
        let expected = fixture(source);
        let mut changed = expected.clone();
        let (parent, key) = missing_path.rsplit_once('/').unwrap();
        changed
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(matches_fixture(&changed, &expected, &[]).is_err());
        let mut changed = expected.clone();
        changed
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), Value::Null);
        assert!(matches_fixture(&changed, &expected, &[]).is_err());
    }
}
