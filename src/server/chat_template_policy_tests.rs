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

//! Context-sensitive compatibility policy regressions through public APIs.

use serde_json::json;

use super::{ChatMessage, ChatTemplateProcessor};
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::types::request::Tool;

fn checkpoint(template: &str) -> ChatTemplateProcessor {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"chat_template": template}).to_string(),
    )
    .unwrap();
    ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap()
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

fn kwargs(value: serde_json::Value) -> ChatTemplateKwargs {
    ChatTemplateKwargs::from_json_object(value.as_object().unwrap().clone())
}

const THINKING_SCHEMA: &str = concat!(
    "{% if enable_thinking %}STRING:{% else %}TYPED:{% endif %}",
    "{% for m in messages %}",
    "{% if enable_thinking %}{{ m.content }}",
    "{% else %}{% for p in m.content | selectattr('type', 'equalto', 'text') %}",
    "{{ p.text }}{% endfor %}{% endif %}{% endfor %}",
);

#[test]
fn transformed_string_content_is_not_mistaken_for_a_dropped_question() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}",
        "{% if m.content is string %}{{ m.content | upper }}",
        "{% else %}{% for p in m.content | selectattr('type', 'equalto', 'text') %}",
        "{{ p.text }}{% endfor %}{% endif %}{% endfor %}",
    ));
    let messages = [message("user", "Mixed_CASE question")];
    assert_eq!(
        processor.apply(&messages, None).unwrap(),
        "MIXED_CASE QUESTION"
    );
    assert_eq!(
        processor.apply_raw(&json!(messages), None).unwrap(),
        "MIXED_CASE QUESTION"
    );
}

#[test]
fn hybrid_roles_preserve_original_strings_without_poisoning_user_only_policy() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}{{ m.role }}:",
        "{% if m.role == 'user' %}",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% else %}{{ m.content }}{% endif %}|{% endfor %}",
    ));
    let user = [message("user", "QUESTION")];
    let conversation = [
        message("system", "SYSTEM"),
        message("user", "QUESTION"),
        message("assistant", "ANSWER"),
        message("user", "FOLLOWUP"),
    ];
    // User-only input can be repaired. If a real string system/assistant role
    // is incompatible with wrapping, preserve the whole original conversation
    // instead of corrupting previously correct instructions/history.
    assert_eq!(processor.apply(&user, None).unwrap(), "user:QUESTION|");
    let original = "system:SYSTEM|user:|assistant:ANSWER|user:|";
    assert_eq!(processor.apply(&conversation, None).unwrap(), original);
    assert_eq!(
        processor.apply_raw(&json!(conversation), None).unwrap(),
        original
    );
    assert_eq!(processor.apply(&user, None).unwrap(), "user:QUESTION|");
}

#[test]
fn changing_default_thinking_rechecks_the_effective_content_contract() {
    let mut processor = checkpoint(THINKING_SCHEMA);
    let messages = [message("user", "QUESTION")];
    assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
    processor.set_default_enable_thinking(true);
    assert_eq!(processor.apply(&messages, None).unwrap(), "STRING:QUESTION");
    processor.set_default_enable_thinking(false);
    assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
}

#[test]
fn explicit_thinking_kwargs_override_the_default_without_cache_cross_contamination() {
    let mut processor = checkpoint(THINKING_SCHEMA);
    let messages = [message("user", "QUESTION")];
    let enabled = kwargs(json!({"enable_thinking": true}));
    let disabled = kwargs(json!({"enable_thinking": false}));
    assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &enabled)
            .unwrap(),
        "STRING:QUESTION"
    );
    processor.set_default_enable_thinking(true);
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &disabled)
            .unwrap(),
        "TYPED:QUESTION"
    );
    assert_eq!(processor.apply(&messages, None).unwrap(), "STRING:QUESTION");
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &enabled)
            .unwrap(),
        "STRING:QUESTION"
    );
}

#[test]
fn generation_and_history_use_their_own_content_contract() {
    let mut processor = checkpoint(concat!(
        "{% for m in messages %}",
        "{% if add_generation_prompt %}",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% else %}{{ m.content }}{% endif %}{% endfor %}",
        "{% if add_generation_prompt %}|GENERATE{% endif %}",
    ));
    let messages = [message("user", "QUESTION")];
    let raw = json!(messages);
    assert_eq!(
        processor.apply(&messages, None).unwrap(),
        "QUESTION|GENERATE"
    );
    assert_eq!(
        processor
            .apply_history_with_kwargs(&messages, None, &Default::default())
            .unwrap(),
        "QUESTION"
    );
    assert_eq!(
        processor
            .apply_raw_history_with_kwargs(&raw, None, &Default::default())
            .unwrap(),
        "QUESTION"
    );
    processor.set_add_generation_prompt(false);
    assert_eq!(processor.apply(&messages, None).unwrap(), "QUESTION");
    processor.set_add_generation_prompt(true);
    assert_eq!(
        processor.apply_raw(&raw, None).unwrap(),
        "QUESTION|GENERATE"
    );
}

#[test]
fn preserve_thinking_kwargs_participate_in_content_policy() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}{% if preserve_thinking | default(false) %}",
        "KEPT:{{ m.content }}{% else %}STRIPPED:",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% endif %}{% endfor %}",
    ));
    let messages = [message("user", "QUESTION")];
    let kept = kwargs(json!({"preserve_thinking": true}));
    let stripped = kwargs(json!({"preserve_thinking": false}));
    assert_eq!(
        processor.apply(&messages, None).unwrap(),
        "STRIPPED:QUESTION"
    );
    assert_eq!(
        processor.apply_with_kwargs(&messages, None, &kept).unwrap(),
        "KEPT:QUESTION"
    );
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &stripped)
            .unwrap(),
        "STRIPPED:QUESTION"
    );
}

#[test]
fn arbitrary_nested_kwargs_participate_in_content_policy() {
    let processor = checkpoint(concat!(
        "{% set mode = render_mode | default({'schema': 'typed'}) %}",
        "{% for m in messages %}{% if mode.schema == 'string' %}",
        "STRING:{{ m.content }}{% else %}TYPED:",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% endif %}{% endfor %}",
    ));
    let messages = [message("user", "QUESTION")];
    let string = kwargs(json!({"render_mode": {"schema": "string", "revision": 1}}));
    let typed = kwargs(json!({"render_mode": {"schema": "typed", "revision": 1}}));
    assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &string)
            .unwrap(),
        "STRING:QUESTION"
    );
    assert_eq!(
        processor
            .apply_with_kwargs(&messages, None, &typed)
            .unwrap(),
        "TYPED:QUESTION"
    );
    assert_eq!(
        processor
            .apply_raw_with_kwargs(&json!(messages), None, &string)
            .unwrap(),
        "STRING:QUESTION"
    );
}

#[test]
fn tool_definitions_not_just_tool_presence_participate_in_content_policy() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}",
        "{% if tools is defined and tools[0].function.name == 'strings' %}",
        "STRING:{{ m.content }}{% else %}TYPED:",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% endif %}{% endfor %}",
    ));
    let make_tool = |name: &str| -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }))
        .unwrap()
    };
    let strings = [make_tool("strings")];
    let typed = [make_tool("typed")];
    let messages = [message("user", "QUESTION")];
    assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
    assert_eq!(
        processor.apply(&messages, Some(&strings)).unwrap(),
        "STRING:QUESTION"
    );
    assert_eq!(
        processor.apply(&messages, Some(&typed)).unwrap(),
        "TYPED:QUESTION"
    );
    assert_eq!(
        processor.apply(&messages, Some(&[])).unwrap(),
        "TYPED:QUESTION"
    );
    assert_eq!(
        processor.apply(&messages, Some(&strings)).unwrap(),
        "STRING:QUESTION"
    );
}

#[test]
fn cloned_processors_share_cache_without_sharing_effective_thinking_defaults() {
    let processor = checkpoint(THINKING_SCHEMA);
    let mut thinking = processor.clone();
    thinking.set_default_enable_thinking(true);
    let messages = [message("user", "QUESTION")];
    for _ in 0..3 {
        assert_eq!(processor.apply(&messages, None).unwrap(), "TYPED:QUESTION");
        assert_eq!(thinking.apply(&messages, None).unwrap(), "STRING:QUESTION");
    }
}

#[test]
fn many_distinct_contexts_and_revisited_entries_keep_their_content_contract() {
    let processor = checkpoint(THINKING_SCHEMA);
    let messages = [message("user", "QUESTION")];
    // More than the bounded decision cache can retain: evicting a decision
    // must only cause another probe, never change rendering semantics.
    for nonce in 0..48 {
        let enabled = nonce % 2 == 0;
        let context = kwargs(json!({
            "enable_thinking": enabled,
            "cache_test_nonce": nonce,
        }));
        let expected = if enabled {
            "STRING:QUESTION"
        } else {
            "TYPED:QUESTION"
        };
        assert_eq!(
            processor
                .apply_with_kwargs(&messages, None, &context)
                .unwrap(),
            expected
        );
    }
    for nonce in [0, 1, 47, 0, 46, 1] {
        let enabled = nonce % 2 == 0;
        let context = kwargs(json!({
            "enable_thinking": enabled,
            "cache_test_nonce": nonce,
        }));
        let expected = if enabled {
            "STRING:QUESTION"
        } else {
            "TYPED:QUESTION"
        };
        assert_eq!(
            processor
                .apply_with_kwargs(&messages, None, &context)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn oversized_context_keys_still_render_correctly_between_cached_contexts() {
    let processor = checkpoint(THINKING_SCHEMA);
    let messages = [message("user", "QUESTION")];
    let padding = "context-only-padding-".repeat(2048);
    for (oversized, enabled) in [
        (false, false),
        (true, true),
        (true, false),
        (false, true),
        (true, true),
        (false, false),
    ] {
        let context = kwargs(json!({
            "enable_thinking": enabled,
            "cache_test_padding": if oversized { padding.as_str() } else { "" },
        }));
        let expected = if enabled {
            "STRING:QUESTION"
        } else {
            "TYPED:QUESTION"
        };
        assert_eq!(
            processor
                .apply_with_kwargs(&messages, None, &context)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn concurrent_clones_render_different_contexts_without_cache_cross_contamination() {
    let processor = checkpoint(THINKING_SCHEMA);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(6));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
            .map(|worker| {
                let processor = processor.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                scope.spawn(move || {
                    let messages = [message("user", "QUESTION")];
                    let raw = json!(messages);
                    barrier.wait();
                    for turn in 0..32 {
                        let enabled = (worker + turn) % 2 == 0;
                        let context = kwargs(json!({
                            "enable_thinking": enabled,
                            "cache_test_partition": worker % 3,
                        }));
                        let expected = if enabled {
                            "STRING:QUESTION"
                        } else {
                            "TYPED:QUESTION"
                        };
                        let actual = if turn % 2 == 0 {
                            processor.apply_with_kwargs(&messages, None, &context)
                        } else {
                            processor.apply_raw_with_kwargs(&raw, None, &context)
                        }
                        .unwrap();
                        assert_eq!(actual, expected, "worker={worker}, turn={turn}");
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("concurrent renderer must not panic");
        }
    });
}

#[test]
fn tool_call_metadata_schema_does_not_reuse_plain_role_decisions() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}{{ m.role }}:",
        "{% if m.role == 'assistant' and m.tool_calls is defined ",
        "and m.tool_calls | length > 0 and m.tool_calls[0].function.name == 'strings' %}",
        "{{ m.content }}{% else %}",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% endif %}|{% endfor %}{% if add_generation_prompt %}GENERATE{% endif %}",
    ));
    let plain = json!([
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "FOLLOWUP"},
    ]);
    let adapted = "user:QUESTION|assistant:ANSWER|user:FOLLOWUP|";
    let original_mixed = "user:|assistant:ANSWER|user:|";
    let call = |name: &str| {
        json!([{
            "id": "call-1", "type": "function",
            "function": {"name": name, "arguments": {"city": "Seoul"}},
        }])
    };
    // Prime a plain-role decision, then alternate values of the same metadata
    // field. Guarding presence alone, or caching only context+roles, is unsafe.
    for calls in [
        None,
        Some(json!([])),
        Some(call("strings")),
        Some(call("typed")),
        None,
        Some(call("strings")),
    ] {
        let mut raw = plain.clone();
        let expected = if let Some(calls) = calls {
            let strings = calls
                .get(0)
                .and_then(|v| v.pointer("/function/name"))
                .is_some_and(|name| name == "strings");
            raw[1]["tool_calls"] = calls;
            if strings { original_mixed } else { adapted }
        } else {
            adapted
        };
        assert_eq!(
            processor.apply_raw(&raw, None).unwrap(),
            format!("{expected}GENERATE")
        );
        assert_eq!(
            processor
                .apply_raw_history_with_kwargs(&raw, None, &Default::default())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn reasoning_metadata_schema_is_checked_without_retaining_a_stale_role_policy() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}{{ m.role }}:",
        "{% if m.role == 'assistant' and m.reasoning_content | default('') == 'STRING_MODE' %}",
        "{{ m.content }}{% else %}",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}",
        "{% endif %}|{% endfor %}",
    ));
    for reasoning in ["TYPED_MODE", "STRING_MODE", "TYPED_MODE", "STRING_MODE"] {
        let raw = json!([
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "ANSWER", "reasoning_content": reasoning},
            {"role": "user", "content": "FOLLOWUP"},
        ]);
        let expected = if reasoning == "STRING_MODE" {
            "user:|assistant:ANSWER|user:|"
        } else {
            "user:QUESTION|assistant:ANSWER|user:FOLLOWUP|"
        };
        assert_eq!(processor.apply_raw(&raw, None).unwrap(), expected);
    }
}

#[test]
fn string_user_metadata_also_participates_in_the_safety_guard() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}",
        "{% if m.name | default('') == 'string-reader' %}{{ m.content }}",
        "{% else %}{% for p in m.content | selectattr('type', 'equalto', 'text') %}",
        "{{ p.text }}{% endfor %}{% endif %}{% endfor %}",
    ));
    for name in ["typed-reader", "string-reader", "typed-reader"] {
        let raw = json!([{"role": "user", "content": "QUESTION", "name": name}]);
        assert_eq!(processor.apply_raw(&raw, None).unwrap(), "QUESTION");
    }
}

#[test]
fn explicit_assistant_rejection_is_not_bypassed_by_content_adaptation() {
    let processor = checkpoint(concat!(
        "{% for m in messages %}",
        "{% if m.role == 'assistant' and m.content is string %}",
        "{{ raise_exception('assistant string policy rejected') }}",
        "{% endif %}",
        "{% for p in m.content | selectattr('type', 'equalto', 'text') %}",
        "{{ p.text }}{% endfor %}{% endfor %}",
    ));
    // A successful user-only decision must not override a later role's
    // explicit policy rejection, even when typed assistant content would render.
    assert_eq!(
        processor
            .apply(&[message("user", "QUESTION")], None)
            .unwrap(),
        "QUESTION"
    );
    let conversation = [
        message("user", "QUESTION"),
        message("assistant", "ANSWER"),
        message("user", "FOLLOWUP"),
    ];
    let raw = json!(conversation);
    let typed = json!([
        {"role": "user", "content": [{"type": "text", "text": "QUESTION"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ANSWER"}]},
        {"role": "user", "content": [{"type": "text", "text": "FOLLOWUP"}]},
    ]);
    assert_eq!(
        processor.apply_raw(&typed, None).unwrap(),
        "QUESTIONANSWERFOLLOWUP"
    );
    for result in [
        processor.apply(&conversation, None),
        processor.apply_history_with_kwargs(&conversation, None, &Default::default()),
        processor.apply_raw(&raw, None),
        processor.apply_raw_history_with_kwargs(&raw, None, &Default::default()),
    ] {
        let error = result.expect_err("explicit template policy must remain an error");
        assert_eq!(
            super::template_rejection_message(&error),
            Some("assistant string policy rejected")
        );
    }
}
