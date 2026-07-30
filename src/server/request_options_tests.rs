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
    LOOP_DETECTION_RECOMMENDED, RequestOptionOverrides, build_server_generate_options,
    carries_loop_amplifier, chat_carries_loop_amplifier, loop_detection_from_request,
    resolve_loop_detection,
};
use crate::server::ServerConfig;
use crate::server::types::request::{ChatCompletionRequest, FunctionDefinition, Tool};
use mlxcel_core::{LoopDetectionConfig, detect_repetition_loop};

/// A Gemma 4 server config: the only place `model_is_gemma4_family` is set.
fn gemma4_config() -> ServerConfig {
    ServerConfig {
        model_is_gemma4_family: true,
        ..Default::default()
    }
}

fn make_tool(name: &str) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: None,
        },
    }
}

/// Deserialize a chat request carrying one function tool and the given
/// `tool_choice`, going through serde so the wire shape is what is under test.
/// `tool_choice: None` omits the field entirely.
fn chat_request_with_tools(tool_choice: Option<&str>) -> ChatCompletionRequest {
    let mut body = serde_json::json!({
        "model": "gemma-4-12b-it-4bit",
        "messages": [{"role": "user", "content": "what is the weather"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }]
    });
    if let Some(choice) = tool_choice {
        body["tool_choice"] = serde_json::Value::String(choice.to_string());
    }
    serde_json::from_value(body).expect("chat request fixture deserializes")
}

/// An agent-loop follow-up turn: prior tool calls and their results are replayed
/// in `messages`, but the turn declares no top-level `tools`. `tool_choice` is
/// applied when given. This is the shape `has_tool_fields` routes to the
/// raw-JSON render path, which writes `tool_calls` / `tool_call_id` into the
/// prompt whatever `effective_tools` says.
fn tool_replay_request(tool_choice: Option<&str>) -> ChatCompletionRequest {
    let mut body = serde_json::json!({
        "model": "gemma-4-12b-it-4bit",
        "messages": [
            {"role": "user", "content": "what is the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"}
        ]
    });
    if let Some(choice) = tool_choice {
        body["tool_choice"] = serde_json::Value::String(choice.to_string());
    }
    serde_json::from_value(body).expect("tool replay fixture deserializes")
}

/// The same follow-up turn but with only the `tool_call_id` half: a tool result
/// message and no assistant `tool_calls`.
fn tool_result_only_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "gemma-4-12b-it-4bit",
        "messages": [
            {"role": "user", "content": "what is the weather"},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"}
        ]
    }))
    .expect("tool result fixture deserializes")
}

#[test]
fn build_server_generate_options_uses_server_defaults() {
    let config = ServerConfig::default();

    let options = build_server_generate_options(&config, RequestOptionOverrides::default());

    assert_eq!(options.max_tokens, config.default_max_tokens);
    assert_eq!(options.sampling.temperature, config.default_temperature);
    assert_eq!(options.sampling.top_k, config.default_top_k);
    assert_eq!(options.sampling.top_p, config.default_top_p);
    assert_eq!(options.sampling.min_p, config.default_min_p);
    assert_eq!(
        options.sampling.repetition_penalty,
        config.default_repetition_penalty
    );
    assert_eq!(
        options.sampling.dry_multiplier,
        config.default_dry_multiplier
    );
    // XTC has no server-level CLI default; an absent request field always
    // resolves to the disabled baseline regardless of server configuration.
    assert_eq!(options.sampling.xtc_probability, 0.0);
    assert_eq!(options.sampling.xtc_threshold, 0.1);
    assert_eq!(options.stop_sequences, None);
}

#[test]
fn build_server_generate_options_applies_request_overrides() {
    let config = ServerConfig::default();
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            max_tokens: Some(7),
            temperature: Some(0.0),
            top_k: Some(99),
            top_p: Some(0.5),
            min_p: Some(0.2),
            repetition_penalty: Some(1.3),
            seed: Some(42),
            frequency_penalty: Some(0.4),
            presence_penalty: Some(0.5),
            dry_multiplier: Some(0.9),
            dry_base: Some(2.2),
            dry_allowed_length: Some(5),
            dry_penalty_last_n: Some(17),
            dry_sequence_breakers: Some(vec![1, 2]),
            xtc_probability: Some(0.7),
            xtc_threshold: Some(0.2),
            stop_sequences: Some(vec!["stop".to_string()]),
            priority: crate::server::batch::RequestPriority::High,
            reasoning_budget: crate::server::config::ReasoningBudgetOverride::default(),
            thinking_enter_block_on_start: true,
            loop_detection_request: None,
            request_carries_loop_amplifier: false,
        },
    );

    assert_eq!(options.max_tokens, 7);
    assert_eq!(options.sampling.temperature, 0.0);
    assert_eq!(options.sampling.top_k, 1);
    assert_eq!(options.sampling.top_p, 1.0);
    assert_eq!(options.sampling.min_p, 0.2);
    assert_eq!(options.sampling.seed, Some(42));
    assert_eq!(options.sampling.repetition_penalty, 1.3);
    assert_eq!(options.sampling.frequency_penalty, 0.4);
    assert_eq!(options.sampling.presence_penalty, 0.5);
    assert_eq!(options.sampling.dry_multiplier, 0.9);
    assert_eq!(options.sampling.dry_base, 2.2);
    assert_eq!(options.sampling.dry_allowed_length, 5);
    assert_eq!(options.sampling.dry_penalty_last_n, 17);
    assert_eq!(options.sampling.dry_sequence_breakers, Vec::<i32>::new());
    assert_eq!(options.sampling.xtc_probability, 0.7);
    assert_eq!(options.sampling.xtc_threshold, 0.2);
    assert_eq!(options.stop_sequences, Some(vec!["stop".to_string()]));
}

// -- loop detection (issue #432) --

#[test]
fn loop_detection_disabled_by_default_for_non_gemma() {
    // Default config is a non-Gemma model: loop detection stays disabled so the
    // bit-exact baseline is preserved.
    let config = ServerConfig::default();
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn loop_detection_from_request_none_when_no_fields() {
    assert_eq!(loop_detection_from_request(None, None, None), None);
}

#[test]
fn loop_detection_from_request_some_when_any_field_set() {
    // Even an explicit disable (max_pattern_size = 0) is authoritative.
    let only_disable = loop_detection_from_request(Some(0), None, None);
    assert_eq!(only_disable, Some(LoopDetectionConfig::new(0, 0, 0)));
    assert!(!only_disable.unwrap().is_enabled());

    let full = loop_detection_from_request(Some(20), Some(1), Some(4));
    assert_eq!(full, Some(LoopDetectionConfig::new(1, 20, 4)));
}

#[test]
fn recommended_threshold_survives_a_four_column_markdown_table() {
    // Issue #967: the recommended `min_count` must sit above the repeat count a
    // markdown alignment row produces, or ordinary tables get truncated.
    assert_eq!(
        LOOP_DETECTION_RECOMMENDED,
        LoopDetectionConfig::new(1, 20, 12)
    );
    assert!(LOOP_DETECTION_RECOMMENDED.is_enabled());
}

#[test]
fn resolve_loop_detection_precedence() {
    let req = LoopDetectionConfig::new(2, 5, 3);
    let global = LoopDetectionConfig::new(1, 10, 6);

    // Explicit request wins over everything, including the family default-on and
    // the amplifier gate: it applies whether or not the request is amplified.
    for amplified in [false, true] {
        assert_eq!(
            resolve_loop_detection(Some(req), Some(global), true, amplified),
            req,
            "explicit request beats global, family, and the amplifier gate"
        );
    }

    // Global override beats the family default-on (and may force-disable), and
    // is likewise not subject to the amplifier gate.
    for amplified in [false, true] {
        assert_eq!(
            resolve_loop_detection(None, Some(global), true, amplified),
            global
        );
        let forced_off = LoopDetectionConfig::disabled();
        assert_eq!(
            resolve_loop_detection(None, Some(forced_off), true, amplified),
            forced_off,
            "operator can force-disable even for the Gemma 4 family"
        );
    }

    // Gemma 4 family default-on applies when the request is amplified and
    // nothing higher-precedence is set.
    assert_eq!(
        resolve_loop_detection(None, None, true, true),
        LOOP_DETECTION_RECOMMENDED
    );

    // Disabled baseline for non-family models when nothing applies.
    assert_eq!(
        resolve_loop_detection(None, None, false, true),
        LoopDetectionConfig::disabled()
    );
}

// -- amplifier gate on the family default-on (issue #967) --

#[test]
fn carries_loop_amplifier_predicate() {
    // No tools: a plain or grammar-only request.
    assert!(!carries_loop_amplifier(None));
    // An empty `tools` array is not a tool declaration.
    assert!(!carries_loop_amplifier(Some(&[])));
    // A non-empty slice is. Callers pass what the template will render, so
    // `tool_choice` is already accounted for by the time the slice gets here;
    // see the `tool_choice_*` tests below.
    assert!(carries_loop_amplifier(Some(&[make_tool("get_weather")])));
}

#[test]
fn gemma4_family_default_on_requires_an_amplifier() {
    let tools = [make_tool("get_weather")];
    let with_tools = carries_loop_amplifier(Some(&tools));
    let without_tools = carries_loop_amplifier(None);

    // gemma4 + tools -> recommended threshold.
    assert_eq!(
        resolve_loop_detection(None, None, true, with_tools),
        LOOP_DETECTION_RECOMMENDED
    );
    // Gemma 4 without a tool-shaped prompt stays disabled. This includes plain
    // chat and grammar-only structured output.
    assert_eq!(
        resolve_loop_detection(None, None, true, without_tools),
        LoopDetectionConfig::disabled()
    );
    // non-gemma4 + tools -> disabled; the gate never enables a non-family model.
    assert_eq!(
        resolve_loop_detection(None, None, false, with_tools),
        LoopDetectionConfig::disabled()
    );
}

// -- tool_choice participates in the tools signal (issue #967) --
//
// The gate reads `chat_request::effective_tools`, the same helper that decides
// what the template renders, so a declared tool only counts when the model
// actually sees it. `tool_choice: "none"` produces a prompt indistinguishable
// from plain chat, so it carries no amplifier.

#[test]
fn tool_choice_none_drops_the_tools_signal() {
    let request = chat_request_with_tools(Some("none"));
    assert!(
        request.tools.as_ref().is_some_and(|t| !t.is_empty()),
        "fixture must still declare a tool, only tool_choice suppresses it"
    );

    let amplified = chat_carries_loop_amplifier(&request);
    assert!(
        !amplified,
        "tool_choice=none renders no declarations, so there is no amplifier"
    );
    assert_eq!(
        resolve_loop_detection(None, None, true, amplified),
        LoopDetectionConfig::disabled()
    );
}

#[test]
fn absent_tool_choice_keeps_the_tools_signal() {
    let request = chat_request_with_tools(None);
    let amplified = chat_carries_loop_amplifier(&request);
    assert!(amplified);
    assert_eq!(
        resolve_loop_detection(None, None, true, amplified),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn tool_choice_auto_keeps_the_tools_signal() {
    let request = chat_request_with_tools(Some("auto"));
    let amplified = chat_carries_loop_amplifier(&request);
    assert!(amplified);
    assert_eq!(
        resolve_loop_detection(None, None, true, amplified),
        LOOP_DETECTION_RECOMMENDED
    );
}

// -- tool-shaped message content is its own amplifier (issue #967 follow-up) --
//
// `chat_request::has_tool_fields` routes any request whose messages carry
// `tool_calls` or `tool_call_id` to the raw-JSON render path, and that path
// writes those fields into the prompt independently of `effective_tools`. An
// agent loop replaying prior tool calls therefore feeds Gemma 4 a tool-shaped
// prompt even when the follow-up turn declares no `tools`. #432's unconditional
// default-on covered those turns; the #967 narrowing was meant to exclude plain
// chat only, so they stay covered.

#[test]
fn replayed_tool_calls_amplify_without_a_tools_array() {
    let request = tool_replay_request(None);
    assert!(
        request.tools.is_none(),
        "fixture must declare no top-level tools, only replayed message content"
    );
    assert!(chat_carries_loop_amplifier(&request));
    assert_eq!(
        chat_route_loop_detection(&request),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn replayed_tool_results_amplify_without_a_tools_array() {
    // The `tool_call_id` half on its own, with no assistant `tool_calls`.
    let request = tool_result_only_request();
    assert!(request.tools.is_none());
    assert!(chat_carries_loop_amplifier(&request));
    assert_eq!(
        chat_route_loop_detection(&request),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn tool_choice_none_does_not_disarm_replayed_tool_calls() {
    // `tool_choice: "none"` suppresses the declarations, but the replayed
    // messages still reach the prompt, so the request stays amplified. This is
    // the case the narrowing regressed before message content was added.
    let request = tool_replay_request(Some("none"));
    assert!(chat_carries_loop_amplifier(&request));
    assert_eq!(
        chat_route_loop_detection(&request),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn grammar_constrained_long_uniform_array_does_not_arm_detector() {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "gemma-4-12b-it-4bit",
        "messages": [{
            "role": "user",
            "content": "Return the values array containing the number 0 repeated 30 times."
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vals",
                "schema": {
                    "type": "object",
                    "properties": {"values": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["values"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .expect("json_schema request fixture deserializes");

    assert!(request.response_format.is_some());
    assert!(!chat_carries_loop_amplifier(&request));
    let config = chat_route_loop_detection(&request);
    assert_eq!(
        config,
        LoopDetectionConfig::disabled(),
        "grammar-only structured output must not auto-enable token-level loop detection"
    );
    assert!(
        !detect_repetition_loop(&[42; 30], &config),
        "a schema-valid run of 30 identical values must pass the disabled detector"
    );
}

#[test]
fn global_override_on_enables_plain_chat_despite_the_gate() {
    // `MLXCEL_LOOP_DETECTION=on` resolves to `Some(LOOP_DETECTION_RECOMMENDED)`
    // and applies to every request unconditionally, gate or no gate.
    assert_eq!(
        resolve_loop_detection(None, Some(LOOP_DETECTION_RECOMMENDED), true, false),
        LOOP_DETECTION_RECOMMENDED
    );
    assert_eq!(
        resolve_loop_detection(None, Some(LOOP_DETECTION_RECOMMENDED), false, false),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn global_override_off_disables_an_amplified_gemma4_request() {
    let tools = [make_tool("get_weather")];
    let amplified = carries_loop_amplifier(Some(&tools));
    assert_eq!(
        resolve_loop_detection(None, Some(LoopDetectionConfig::disabled()), true, amplified),
        LoopDetectionConfig::disabled(),
        "MLXCEL_LOOP_DETECTION=off wins over the gated family default-on"
    );
}

#[test]
fn gemma4_plain_chat_stays_disabled_through_build_server_generate_options() {
    // End-to-end through the plumbing, not just `resolve_loop_detection`: a
    // plain Gemma 4 chat leaves the detector off.
    let config = gemma4_config();
    let plain = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!plain.sampling.loop_detection.is_enabled());
    assert_eq!(
        plain.sampling.loop_detection,
        LoopDetectionConfig::disabled()
    );
}

#[test]
fn gemma4_amplified_request_auto_enables_through_build_server_generate_options() {
    // The same plumbing with the flag set: the #432 protection still arrives at
    // the sampling config for tool-bearing and schema-constrained requests.
    let config = gemma4_config();
    let amplified = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            request_carries_loop_amplifier: true,
            ..Default::default()
        },
    );
    assert_eq!(
        amplified.sampling.loop_detection,
        LOOP_DETECTION_RECOMMENDED
    );
    assert!(amplified.sampling.loop_detection.is_enabled());
}

#[test]
fn non_gemma4_amplified_request_stays_disabled_through_build_server_generate_options() {
    let config = ServerConfig::default(); // model_is_gemma4_family = false
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            request_carries_loop_amplifier: true,
            ..Default::default()
        },
    );
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn non_gemma4_family_stays_disabled_by_default() {
    let config = ServerConfig::default(); // model_is_gemma4_family = false
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn explicit_request_disable_overrides_family_default_on() {
    let config = gemma4_config();

    // A per-request explicit disable (max_pattern_size = 0) must win over the
    // Gemma 4 family default-on, including for an amplified request where the
    // gate would otherwise turn detection on.
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            loop_detection_request: loop_detection_from_request(Some(0), None, None),
            request_carries_loop_amplifier: true,
            ..Default::default()
        },
    );
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn explicit_request_tune_overrides_family_default_on() {
    let config = gemma4_config();

    // A per-request tune wins over the family default-on threshold.
    let tuned = loop_detection_from_request(Some(8), Some(2), Some(3));
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            loop_detection_request: tuned,
            request_carries_loop_amplifier: true,
            ..Default::default()
        },
    );
    assert_eq!(
        options.sampling.loop_detection,
        LoopDetectionConfig::new(2, 8, 3)
    );
}

#[test]
fn explicit_request_enable_wins_over_the_amplifier_gate() {
    let config = gemma4_config();

    // A plain chat (no amplifier) that explicitly asks for detection still gets
    // it: the per-request override sits above the gate.
    let tuned = loop_detection_from_request(Some(20), Some(1), Some(4));
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            loop_detection_request: tuned,
            request_carries_loop_amplifier: false,
            ..Default::default()
        },
    );
    assert_eq!(
        options.sampling.loop_detection,
        LoopDetectionConfig::new(1, 20, 4),
        "an operator restoring the pre-#967 threshold per request keeps working"
    );
}

#[test]
fn global_override_applies_to_non_gemma4() {
    let config = ServerConfig {
        loop_detection: Some(LOOP_DETECTION_RECOMMENDED),
        ..Default::default()
    };
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert_eq!(options.sampling.loop_detection, LOOP_DETECTION_RECOMMENDED);
}

#[test]
fn global_override_can_force_disable_gemma4() {
    // An operator can globally force-disable even for the Gemma 4 family.
    let config = ServerConfig {
        model_is_gemma4_family: true,
        loop_detection: Some(LoopDetectionConfig::disabled()),
        ..Default::default()
    };
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            request_carries_loop_amplifier: true,
            ..Default::default()
        },
    );
    assert!(!options.sampling.loop_detection.is_enabled());
}

// -- route-shaped wiring (issue #967) --
//
// The tests above drive the helpers directly. These go one level out and build
// the options exactly as each route does: the same request type the handler
// receives, through the same translator where there is one, into the same
// `chat::build_generate_options`, then assert the resolved `loop_detection` on
// the result. They pin the composition, so a call site that swapped the
// chat-shaped helper for a raw `request.tools` slice would have to change this
// file too.
//
// The async handlers themselves are not invoked: `AppState` requires a real
// `ModelProvider`, which means loading a model from disk. `routes/cache_tests.rs`
// documents the same limitation for the cache routes.

use crate::server::routes::chat::build_generate_options;

/// Resolved loop-detection for a chat-shaped request on a Gemma 4 server, built
/// the way `non_stream_chat_completion` and `stream_chat_completion` build it.
fn chat_route_loop_detection(request: &ChatCompletionRequest) -> LoopDetectionConfig {
    let amplified = chat_carries_loop_amplifier(request);
    let options = build_generate_options(&request.params, &gemma4_config(), amplified);
    options.sampling.loop_detection
}

#[test]
fn chat_route_gates_on_rendered_tools() {
    assert_eq!(
        chat_route_loop_detection(&chat_request_with_tools(Some("none"))),
        LoopDetectionConfig::disabled(),
        "tool_choice=none renders no declarations, so /v1/chat/completions must not arm the detector"
    );
    assert_eq!(
        chat_route_loop_detection(&chat_request_with_tools(None)),
        LOOP_DETECTION_RECOMMENDED
    );
}

#[test]
fn responses_route_gates_on_rendered_tools() {
    // The Responses translator flattens `tools` / `tool_choice` onto the chat
    // request; this covers that mapping as well as the gate.
    let build = |tool_choice: Option<&str>| {
        let mut body = serde_json::json!({
            "model": "gemma-4-12b-it-4bit",
            "input": "what is the weather",
            "tools": [{"type": "function", "name": "get_weather"}]
        });
        if let Some(choice) = tool_choice {
            body["tool_choice"] = serde_json::Value::String(choice.to_string());
        }
        let request = serde_json::from_value(body).expect("responses request deserializes");
        let translated =
            crate::server::responses_translator::responses_request_to_chat(&request, None, None)
                .expect("translates");
        chat_route_loop_detection(&translated.chat_request)
    };

    assert_eq!(build(Some("none")), LoopDetectionConfig::disabled());
    assert_eq!(build(None), LOOP_DETECTION_RECOMMENDED);
}

#[test]
fn anthropic_route_gates_on_rendered_tools() {
    // The Anthropic translator maps `{"type": "none"}` onto `Mode("none")`, so
    // the gate closes on that surface too. This route never has a grammar
    // constraint (`response_format` is always translated to `None`).
    let build = |tool_choice: Option<serde_json::Value>| {
        let mut body = serde_json::json!({
            "model": "gemma-4-12b-it-4bit",
            "messages": [{"role": "user", "content": "what is the weather"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
        });
        if let Some(choice) = tool_choice {
            body["tool_choice"] = choice;
        }
        let request = serde_json::from_value(body).expect("anthropic request deserializes");
        let translated = crate::server::anthropic_translator::anthropic_request_to_chat(&request);
        assert!(
            translated.chat_request.response_format.is_none(),
            "the Anthropic surface carries no response_format"
        );
        chat_route_loop_detection(&translated.chat_request)
    };

    assert_eq!(
        build(Some(serde_json::json!({"type": "none"}))),
        LoopDetectionConfig::disabled()
    );
    assert_eq!(build(None), LOOP_DETECTION_RECOMMENDED);
}

#[test]
fn completions_route_keeps_grammar_only_requests_disabled() {
    // `/v1/completions` has no tool-shaped prompt signal, so both plain and
    // grammar-constrained requests pass `false` to `build_generate_options`.
    let request: crate::server::types::CompletionRequest = serde_json::from_value(
        serde_json::json!({"model": "gemma-4-12b-it-4bit", "prompt": "once upon a time"}),
    )
    .expect("completion request deserializes");

    let constrained = build_generate_options(&request.params, &gemma4_config(), false);
    assert_eq!(
        constrained.sampling.loop_detection,
        LoopDetectionConfig::disabled()
    );
}

#[test]
fn disaggregated_chat_front_gates_on_rendered_tools() {
    // `router_front::route_chat` takes the same `ChatCompletionRequest` and calls
    // the same `build_generate_options`. The resolved value is inert today:
    // `loop_detection` is not carried by the PrefillRequestFrame.
    assert_eq!(
        chat_route_loop_detection(&chat_request_with_tools(Some("none"))),
        LoopDetectionConfig::disabled()
    );
    assert_eq!(
        chat_route_loop_detection(&chat_request_with_tools(None)),
        LOOP_DETECTION_RECOMMENDED
    );
}
