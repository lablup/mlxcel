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

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use super::{
    ApplyResult, CLASSIFIED_SERVER_CONFIG_FIELDS, Op, api_name, apply, fingerprint, is_mutable,
    mutable_values, parse_patch_body, schema,
};
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::{ReasoningAliasField, ServerConfig};

fn object(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .expect("test fixture must be a JSON object")
        .clone()
}

fn declared_server_config_fields() -> Vec<&'static str> {
    const SOURCE: &str = include_str!("config.rs");
    let (_, after_declaration) = SOURCE
        .split_once("pub struct ServerConfig {")
        .expect("ServerConfig declaration must remain discoverable");

    after_declaration
        .lines()
        .take_while(|line| line.trim() != "}")
        .filter_map(|line| {
            line.trim_start()
                .strip_prefix("pub ")
                .and_then(|field| field.split_once(':'))
                .map(|(name, _)| name.trim())
        })
        .collect()
}

fn rejection_reason<'a>(result: &'a ApplyResult, name: &str) -> &'a str {
    result
        .rejected
        .iter()
        .find(|rejected| rejected.name == name)
        .unwrap_or_else(|| panic!("missing rejection for {name}"))
        .reason
        .as_str()
}

#[test]
fn server_config_schema_classifies_all_93_fields() {
    let declared = declared_server_config_fields();
    assert_eq!(declared.len(), 93, "ServerConfig field count changed");
    assert_eq!(
        declared.as_slice(),
        CLASSIFIED_SERVER_CONFIG_FIELDS,
        "the classification table must match ServerConfig source order exactly"
    );

    let specs = schema(&ServerConfig::default());
    assert_eq!(specs.len(), 93);
    let expected_names: Vec<_> = CLASSIFIED_SERVER_CONFIG_FIELDS
        .iter()
        .copied()
        .map(api_name)
        .collect();
    let actual_names: Vec<_> = specs.iter().map(|spec| spec.name).collect();
    assert_eq!(actual_names, expected_names);
    assert_eq!(
        actual_names.iter().copied().collect::<BTreeSet<_>>().len(),
        93,
        "every management API name must be unique"
    );

    let expected_mutable: BTreeSet<_> = [
        "timeout_seconds",
        "default_temperature",
        "default_top_p",
        "default_top_k",
        "default_min_p",
        "default_repetition_penalty",
        "default_repetition_context_size",
        "default_max_tokens",
        "default_seed",
        "default_frequency_penalty",
        "default_presence_penalty",
        "default_dry_multiplier",
        "default_dry_base",
        "default_dry_allowed_length",
        "default_dry_penalty_last_n",
        "default_dry_sequence_breakers",
        "lang_bias_config",
        "reasoning_budget",
        "chat_template_kwargs",
        "loop_detection",
        "max_denoising_steps",
        "diffusion_sampler",
        "diffusion_threshold",
    ]
    .into_iter()
    .collect();
    let actual_mutable: BTreeSet<_> = specs
        .iter()
        .filter(|spec| spec.mutable)
        .map(|spec| spec.name)
        .collect();
    assert_eq!(actual_mutable, expected_mutable);

    for spec in &specs {
        assert_eq!(spec.mutable, is_mutable(spec.name), "{}", spec.name);
        if spec.mutable {
            assert!(spec.reason.is_none(), "{} is mutable", spec.name);
        } else {
            assert!(
                spec.reason.is_some_and(|reason| !reason.is_empty()),
                "{} needs a read-only reason",
                spec.name
            );
        }
    }

    let reasoning_alias = specs
        .iter()
        .find(|spec| spec.name == "reasoning_alias_field")
        .expect("reasoning_alias_field must be classified");
    assert!(!reasoning_alias.mutable);
    assert!(
        reasoning_alias
            .reason
            .is_some_and(|reason| reason.contains("restart required"))
    );
    assert!(specs.iter().any(|spec| spec.name == "timeout_seconds"));
    assert!(
        !specs
            .iter()
            .any(|spec| spec.name == "decode_timeout_seconds")
    );
}

#[test]
fn merge_applies_valid_values_and_reports_each_rejection() {
    let config = ServerConfig {
        context_size: 128,
        ..ServerConfig::default()
    };
    let startup = config.live_settings();
    let mut current = startup.clone();
    current.timeout_seconds = 44;
    current.default_temperature = 0.9;
    current.default_top_p = 0.75;
    current.default_seed = None;

    let values = object(json!({
        "default_seed": 7,
        "default_temperature": 0.25,
        "default_top_p": "wide",
        "mystery_knob": true,
        "reasoning_alias_field": "none",
        "timeout_seconds": 0
    }));
    let result = apply(&startup, &current, Op::Merge, &values, &config);

    assert_eq!(result.next.default_temperature, 0.25);
    assert_eq!(result.next.default_seed, Some(7));
    assert_eq!(result.next.default_top_p, 0.75);
    assert_eq!(result.next.timeout_seconds, 44);
    assert_eq!(result.applied.len(), 2);
    assert_eq!(result.applied.get("default_seed"), Some(&json!(7)));
    assert_eq!(
        result.applied.get("default_temperature"),
        Some(&json!(0.25))
    );
    assert_eq!(result.rejected.len(), 4);
    assert!(rejection_reason(&result, "default_top_p").contains("JSON number"));
    assert_eq!(rejection_reason(&result, "mystery_knob"), "unknown setting");
    assert!(rejection_reason(&result, "reasoning_alias_field").contains("read-only"));
    assert!(rejection_reason(&result, "timeout_seconds").contains("greater than zero"));
}

#[test]
fn replace_starts_from_the_startup_snapshot() {
    let config = ServerConfig {
        context_size: 1024,
        default_temperature: 0.8,
        default_top_p: 0.95,
        default_max_tokens: 512,
        ..ServerConfig::default()
    };
    let startup = config.live_settings();
    let mut current = startup.clone();
    current.default_temperature = 1.2;
    current.default_top_p = 0.4;
    current.default_max_tokens = 64;
    current.timeout_seconds = 9;

    let values = object(json!({"default_temperature": 0.3}));
    let result = apply(&startup, &current, Op::Replace, &values, &config);
    let mut expected = startup.clone();
    expected.default_temperature = 0.3;

    assert_eq!(mutable_values(&result.next), mutable_values(&expected));
    assert_eq!(result.applied.len(), 1);
    assert_eq!(
        result.applied.get("default_temperature"),
        Some(&json!(0.3_f32))
    );
    assert!(result.rejected.is_empty());
}

#[test]
fn chat_template_kwargs_accepts_an_object_through_the_shared_parser() {
    let config = ServerConfig::default();
    let startup = config.live_settings();
    let kwargs = json!({
        "enable_thinking": true,
        "nested": {"level": 2},
        "template_name": "custom"
    });
    let values = object(json!({"chat_template_kwargs": kwargs.clone()}));

    let result = apply(&startup, &startup, Op::Merge, &values, &config);

    assert!(result.rejected.is_empty());
    assert_eq!(result.applied.get("chat_template_kwargs"), Some(&kwargs));
    assert_eq!(
        result
            .next
            .chat_template_kwargs
            .as_ref()
            .expect("valid object must produce kwargs")
            .as_map(),
        kwargs.as_object().expect("fixture is an object")
    );
}

#[test]
fn chat_template_kwargs_rejects_non_objects_without_mutation() {
    let config = ServerConfig::default();
    let startup = config.live_settings();
    let mut current = startup.clone();
    current.chat_template_kwargs = Some(ChatTemplateKwargs::from_json_object(object(json!({
        "existing": true
    }))));
    let before = mutable_values(&current);

    for invalid in [Value::Null, json!(true), json!(7), json!("bad"), json!([])] {
        let values = object(json!({"chat_template_kwargs": invalid}));
        let result = apply(&startup, &current, Op::Merge, &values, &config);
        assert!(result.applied.is_empty());
        assert_eq!(result.rejected.len(), 1);
        assert!(rejection_reason(&result, "chat_template_kwargs").contains("JSON object"));
        assert_eq!(mutable_values(&result.next), before);
    }
}

#[test]
fn fingerprint_hashes_only_canonical_mutable_values() {
    let base = ServerConfig::default();
    let base_fingerprint = fingerprint(&base.live_settings());
    assert_eq!(base_fingerprint.len(), 64);
    assert!(
        base_fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );

    let read_only_changed = ServerConfig {
        reasoning_alias_field: ReasoningAliasField::None,
        api_prefix: "/private".to_string(),
        enable_props_endpoint: true,
        ..base.clone()
    };
    assert_eq!(
        fingerprint(&read_only_changed.live_settings()),
        base_fingerprint,
        "read-only ServerConfig fields must not affect the fingerprint"
    );

    let mutable_changed = ServerConfig {
        default_temperature: 0.81,
        ..base.clone()
    };
    assert_ne!(
        fingerprint(&mutable_changed.live_settings()),
        base_fingerprint,
        "a mutable value change must affect the fingerprint"
    );

    let mut first = base.live_settings();
    let mut first_nested = Map::new();
    first_nested.insert("z".to_string(), json!(2));
    first_nested.insert("a".to_string(), json!(1));
    let mut first_kwargs = Map::new();
    first_kwargs.insert("z".to_string(), Value::Object(first_nested));
    first_kwargs.insert("a".to_string(), json!(true));
    first.chat_template_kwargs = Some(ChatTemplateKwargs::from_json_object(first_kwargs));

    let mut second = base.live_settings();
    let mut second_nested = Map::new();
    second_nested.insert("a".to_string(), json!(1));
    second_nested.insert("z".to_string(), json!(2));
    let mut second_kwargs = Map::new();
    second_kwargs.insert("a".to_string(), json!(true));
    second_kwargs.insert("z".to_string(), Value::Object(second_nested));
    second.chat_template_kwargs = Some(ChatTemplateKwargs::from_json_object(second_kwargs));

    assert_eq!(
        fingerprint(&first),
        fingerprint(&second),
        "object insertion order must not affect the canonical hash"
    );
}

#[test]
fn patch_body_accepts_flat_and_wrapped_forms() {
    let flat_values = object(json!({"default_temperature": 0.2}));
    let (op, values) = parse_patch_body(flat_values.clone()).expect("flat merge must parse");
    assert_eq!(op, Op::Merge);
    assert_eq!(values, flat_values);

    let (op, values) = parse_patch_body(object(json!({
        "values": {"default_top_p": 0.8}
    })))
    .expect("wrapped body without op defaults to merge");
    assert_eq!(op, Op::Merge);
    assert_eq!(values, object(json!({"default_top_p": 0.8})));

    let (op, values) = parse_patch_body(object(json!({
        "op": "merge",
        "values": {"timeout_seconds": 10}
    })))
    .expect("explicit merge must parse");
    assert_eq!(op, Op::Merge);
    assert_eq!(values, object(json!({"timeout_seconds": 10})));

    let (op, values) = parse_patch_body(object(json!({
        "op": "replace",
        "values": {"default_seed": null}
    })))
    .expect("replace must parse");
    assert_eq!(op, Op::Replace);
    assert_eq!(values, object(json!({"default_seed": null})));
}

#[test]
fn patch_body_rejects_malformed_wrappers() {
    assert_eq!(
        parse_patch_body(object(json!({"op": "merge"}))).expect_err("values are required"),
        "wrapped PATCH body requires a values object"
    );
    assert_eq!(
        parse_patch_body(object(json!({"values": []}))).expect_err("values must be an object"),
        "values must be an object, got array"
    );
    assert_eq!(
        parse_patch_body(object(json!({"op": "reset", "values": {}})))
            .expect_err("unknown operation must fail"),
        "op must be \"merge\" or \"replace\""
    );
    assert_eq!(
        parse_patch_body(object(json!({"op": 1, "values": {}})))
            .expect_err("non-string operation must fail"),
        "op must be \"merge\" or \"replace\""
    );
    assert_eq!(
        parse_patch_body(object(json!({
            "extra": true,
            "op": "merge",
            "values": {}
        })))
        .expect_err("unknown wrapper keys must fail"),
        "wrapped PATCH body has unknown fields: extra"
    );
}
