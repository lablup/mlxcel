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

//! Unit tests for the b10621 chat-template / reasoning group (issue #1447).

use super::*;

fn args() -> ChatCompatArgs {
    ChatCompatArgs::default()
}

fn resolve(args: ChatCompatArgs) -> ChatCompatResolution {
    args.resolve().expect("must resolve")
}

fn kwargs(args: ChatCompatArgs) -> Vec<(String, Value)> {
    resolve(args).template_kwargs
}

fn error(args: ChatCompatArgs) -> ChatCompatError {
    args.resolve().expect_err("must be rejected")
}

// ── defaults ────────────────────────────────────────────────────────────────

#[test]
fn nothing_supplied_resolves_to_b10621s_defaults() {
    let resolved = resolve(args());
    assert_eq!(resolved.reasoning_format, ReasoningFormat::Auto);
    assert!(!resolved.skip_chat_parsing);
    assert!(
        !resolved.no_prefill_assistant,
        "b10621 prefills a trailing assistant message by default"
    );
    assert_eq!(resolved.reasoning_budget_message, None);
    assert!(resolved.template_kwargs.is_empty());
}

#[test]
fn the_inert_positive_halves_change_nothing() {
    // Each of these is what mlxcel already does.
    let resolved = resolve(ChatCompatArgs {
        jinja: true,
        prefill_assistant: true,
        no_skip_chat_parsing: true,
        escape: true,
        ..args()
    });
    assert_eq!(resolved, resolve(args()));
}

#[test]
fn both_escape_positions_are_inert_because_the_server_has_no_prompt_flag() {
    // `--escape` processes escape sequences in `--prompt` / `--in-prefix`,
    // which the server does not have; neither position can change anything.
    for build in [
        ChatCompatArgs {
            escape: true,
            ..args()
        },
        ChatCompatArgs {
            no_escape: true,
            ..args()
        },
    ] {
        assert_eq!(resolve(build), resolve(args()));
    }
}

// ── --reasoning writes the enable_thinking kwarg, as upstream does ──────────

#[test]
fn reasoning_on_and_off_write_the_enable_thinking_kwarg() {
    for value in ["on", "enabled", "true", "1"] {
        assert_eq!(
            kwargs(ChatCompatArgs {
                reasoning: Some(value.to_owned()),
                ..args()
            }),
            vec![("enable_thinking".to_owned(), Value::Bool(true))],
            "--reasoning {value}"
        );
    }
    for value in ["off", "disabled", "false", "0"] {
        assert_eq!(
            kwargs(ChatCompatArgs {
                reasoning: Some(value.to_owned()),
                ..args()
            }),
            vec![("enable_thinking".to_owned(), Value::Bool(false))],
            "--reasoning {value}"
        );
    }
}

#[test]
fn reasoning_auto_writes_nothing_and_leaves_the_template_default() {
    for value in ["auto", "-1"] {
        assert!(
            kwargs(ChatCompatArgs {
                reasoning: Some(value.to_owned()),
                ..args()
            })
            .is_empty(),
            "--reasoning {value} must leave the template's own default"
        );
    }
}

#[test]
fn an_unknown_reasoning_value_is_rejected_with_the_upstream_vocabulary() {
    let err = error(ChatCompatArgs {
        reasoning: Some("maybe".to_owned()),
        ..args()
    });
    assert_eq!(err.option, "--reasoning");
    assert!(err.message.contains("maybe"), "{err}");
    assert!(err.message.contains("on, off, auto"), "{err}");
}

// ── --reasoning-effort ──────────────────────────────────────────────────────

#[test]
fn a_reasoning_effort_level_becomes_the_template_kwarg() {
    assert_eq!(
        kwargs(ChatCompatArgs {
            reasoning_effort: Some("xhigh".to_owned()),
            ..args()
        }),
        vec![(
            "reasoning_effort".to_owned(),
            Value::String("xhigh".to_owned())
        )]
    );
}

#[test]
fn the_literal_default_level_erases_the_kwarg_as_upstream_does() {
    assert!(
        kwargs(ChatCompatArgs {
            reasoning_effort: Some("default".to_owned()),
            ..args()
        })
        .is_empty(),
        "`default` keeps the template's own default rather than setting a level"
    );
}

#[test]
fn the_level_is_passed_through_without_translation() {
    // OpenAI's vocabulary is {minimal, low, medium, high} and Qwen3.8's is
    // {xhigh, medium, low}; remapping one onto the other would silently change
    // the model's reasoning budget. The template decides, exactly as it does
    // for the request-level field (#1164).
    for level in ["minimal", "low", "medium", "high", "xhigh", "max", "wat"] {
        assert_eq!(
            kwargs(ChatCompatArgs {
                reasoning_effort: Some(level.to_owned()),
                ..args()
            }),
            vec![(
                "reasoning_effort".to_owned(),
                Value::String(level.to_owned())
            )],
            "{level} must reach the template verbatim"
        );
    }
}

// ── --reasoning-preserve ────────────────────────────────────────────────────

#[test]
fn both_reasoning_preserve_halves_write_the_kwarg() {
    assert_eq!(
        kwargs(ChatCompatArgs {
            reasoning_preserve: true,
            ..args()
        }),
        vec![("preserve_reasoning".to_owned(), Value::Bool(true))]
    );
    assert_eq!(
        kwargs(ChatCompatArgs {
            no_reasoning_preserve: true,
            ..args()
        }),
        vec![("preserve_reasoning".to_owned(), Value::Bool(false))]
    );
}

#[test]
fn every_reasoning_flag_contributes_to_one_kwarg_map() {
    let resolved = resolve(ChatCompatArgs {
        reasoning: Some("off".to_owned()),
        reasoning_effort: Some("low".to_owned()),
        no_reasoning_preserve: true,
        ..args()
    });
    assert_eq!(
        resolved.template_kwargs,
        vec![
            ("enable_thinking".to_owned(), Value::Bool(false)),
            (
                "reasoning_effort".to_owned(),
                Value::String("low".to_owned())
            ),
            ("preserve_reasoning".to_owned(), Value::Bool(false)),
        ]
    );
}

// ── --reasoning-format ──────────────────────────────────────────────────────

#[test]
fn every_reasoning_format_name_resolves() {
    for (value, expected) in [
        ("none", ReasoningFormat::None),
        ("deepseek", ReasoningFormat::DeepSeek),
        ("deepseek-legacy", ReasoningFormat::DeepSeekLegacy),
        ("auto", ReasoningFormat::Auto),
    ] {
        assert_eq!(
            resolve(ChatCompatArgs {
                reasoning_format: Some(value.to_owned()),
                ..args()
            })
            .reasoning_format,
            expected,
            "{value}"
        );
    }
}

#[test]
fn an_unknown_reasoning_format_names_the_option_and_the_accepted_set() {
    let err = error(ChatCompatArgs {
        reasoning_format: Some("legacy".to_owned()),
        ..args()
    });
    assert_eq!(err.option, "--reasoning-format");
    assert!(err.message.contains("legacy"), "{err}");
    assert!(
        err.message
            .contains("none, deepseek, deepseek-legacy, auto"),
        "{err}"
    );
}

// ── flags mlxcel cannot honour ──────────────────────────────────────────────

#[test]
fn turning_the_jinja_engine_off_is_rejected() {
    let err = error(ChatCompatArgs {
        no_jinja: true,
        ..args()
    });
    assert_eq!(err.option, "--no-jinja");
    assert!(err.message.contains("Jinja"), "{err}");
}

#[test]
fn asking_for_special_tokens_in_the_output_is_rejected() {
    let err = error(ChatCompatArgs {
        special: true,
        ..args()
    });
    assert_eq!(err.option, "--special");
    assert!(err.message.contains("special tokens"), "{err}");
}

// ── the response-shaping settings ───────────────────────────────────────────

#[test]
fn skip_chat_parsing_and_prefill_reach_the_resolution() {
    let resolved = resolve(ChatCompatArgs {
        skip_chat_parsing: true,
        no_prefill_assistant: true,
        reasoning_budget_message: Some("(budget exhausted)".to_owned()),
        ..args()
    });
    assert!(resolved.skip_chat_parsing);
    assert!(resolved.no_prefill_assistant);
    assert_eq!(
        resolved.reasoning_budget_message.as_deref(),
        Some("(budget exhausted)")
    );
}

#[test]
fn a_blank_value_is_treated_as_absent() {
    // An inherited `LLAMA_ARG_REASONING=` must not stop the server.
    let resolved = resolve(ChatCompatArgs {
        reasoning: Some(String::new()),
        reasoning_format: Some("   ".to_owned()),
        reasoning_effort: Some(String::new()),
        reasoning_budget_message: Some(String::new()),
        ..args()
    });
    assert_eq!(resolved, resolve(args()));
}

// ── environment vocabulary ──────────────────────────────────────────────────

fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_owned(), std::env::var(k).ok()))
        .collect();
    unsafe {
        for (key, value) in pairs {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    let out = body();
    unsafe {
        for (key, value) in &saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    out
}

#[test]
fn a_bool_pair_environment_value_reaches_the_resolution() {
    let resolved = with_env(
        &[
            ("LLAMA_ARG_SKIP_CHAT_PARSING", Some("1")),
            ("LLAMA_ARG_NO_SKIP_CHAT_PARSING", None),
            ("LLAMA_ARG_JINJA", None),
            ("LLAMA_ARG_NO_JINJA", None),
            ("LLAMA_ARG_REASONING_PRESERVE", None),
            ("LLAMA_ARG_NO_REASONING_PRESERVE", None),
            ("LLAMA_ARG_PREFILL_ASSISTANT", Some("0")),
            ("LLAMA_ARG_NO_PREFILL_ASSISTANT", None),
        ],
        || {
            let mut args = ChatCompatArgs::default();
            args.apply_env_bindings().expect("recognized values");
            args.resolve().expect("resolves")
        },
    );
    assert!(resolved.skip_chat_parsing);
    assert!(
        resolved.no_prefill_assistant,
        "LLAMA_ARG_PREFILL_ASSISTANT=0 is the negative half"
    );
}

#[test]
fn an_unparseable_bool_pair_value_is_reported_like_b10621_throws() {
    let result = with_env(
        &[
            ("LLAMA_ARG_JINJA", Some("perhaps")),
            ("LLAMA_ARG_NO_JINJA", None),
        ],
        || {
            let mut args = ChatCompatArgs::default();
            args.apply_env_bindings()
        },
    );
    let (var, raw) = result.expect_err("b10621 throws on an unparseable boolean");
    assert_eq!(var, "LLAMA_ARG_JINJA");
    assert_eq!(raw, "perhaps");
}

#[test]
fn an_explicit_flag_wins_over_the_environment() {
    let args = with_env(
        &[
            ("LLAMA_ARG_SKIP_CHAT_PARSING", Some("0")),
            ("LLAMA_ARG_NO_SKIP_CHAT_PARSING", None),
        ],
        || {
            let mut args = ChatCompatArgs {
                skip_chat_parsing: true,
                ..ChatCompatArgs::default()
            };
            args.apply_env_bindings().expect("recognized");
            args
        },
    );
    assert!(args.skip_chat_parsing && !args.no_skip_chat_parsing);
}

// ── diagnostics are readable ────────────────────────────────────────────────

#[test]
fn no_diagnostic_carries_a_run_of_collapsed_indentation() {
    for build in [
        ChatCompatArgs {
            no_jinja: true,
            ..args()
        },
        ChatCompatArgs {
            special: true,
            ..args()
        },
        ChatCompatArgs {
            reasoning: Some("maybe".to_owned()),
            ..args()
        },
        ChatCompatArgs {
            reasoning_format: Some("legacy".to_owned()),
            ..args()
        },
    ] {
        let text = error(build).to_string();
        for line in text.lines() {
            assert!(
                !line.trim().contains("   "),
                "diagnostic line carries collapsed indentation: {line:?}"
            );
        }
    }
}
