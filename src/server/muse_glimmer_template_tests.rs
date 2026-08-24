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

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::chat_template::ChatTemplateProcessor;
use super::chat_template_kwargs::{
    ChatTemplateKwargs, extract_request_kwargs, merge_server_and_request,
};
use super::tool_calls::{ToolCallFormat, infer_default_tool_call_format, resolve_tool_call_format};
use super::types::request::{FunctionDefinition, Tool};

const TEMPLATE: &str = include_str!("../../tests/fixtures/muse_glimmer/chat_template.jinja");
const TEMPLATE_SHA256: &str = "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e";

fn processor() -> ChatTemplateProcessor {
    ChatTemplateProcessor::with_template(TEMPLATE.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn kwargs(pairs: &[(&str, Value)]) -> ChatTemplateKwargs {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert((*key).to_string(), value.clone());
    }
    ChatTemplateKwargs::from_json_object(map)
}

fn render(messages: Value, tools: Option<&[Tool]>, kwargs: &ChatTemplateKwargs) -> String {
    processor()
        .apply_raw_with_kwargs(&messages, tools, kwargs)
        .expect("render Muse Glimmer chat template")
}

fn weather_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "weather.get_current".to_string(),
            description: Some("Get current weather for a city.".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            })),
        },
    }
}

fn base_kwargs() -> ChatTemplateKwargs {
    kwargs(&[("current_date", json!("2026-08-11"))])
}

/// Pin a rendered prompt by length and digest.
///
/// Two of these pins moved when `tojson` stopped being minijinja's builtin and
/// became mlxcel's CPython-compatible filter (`chat_template_json`): the
/// template serializes each tool's JSON Schema with `fn.parameters | tojson`,
/// and CPython `json.dumps` writes `", "` / `": "` where serde_json writes
/// `","` / `":"`. Substituting the compact schema back into either new render
/// reproduces the previous digest exactly, so the separators are the whole of
/// the difference. The new form is what `transformers` renders from this same
/// checkpoint template, which is what the checkpoint was tokenized against.
fn assert_render_hash(name: &str, rendered: &str, expected_len: usize, expected_sha: &str) {
    assert_eq!(rendered.len(), expected_len, "{name} rendered length");
    assert_eq!(
        sha256_hex(rendered.as_bytes()),
        expected_sha,
        "{name} sha256"
    );
}

#[test]
fn muse_glimmer_template_fixture_hash_matches_checkpoint() {
    assert_eq!(sha256_hex(TEMPLATE.as_bytes()), TEMPLATE_SHA256);
}

#[test]
fn muse_glimmer_template_selects_atem_without_overriding_explicit_format() {
    let processor = processor();
    assert_eq!(
        processor.default_tool_call_format(),
        Some(ToolCallFormat::Atem)
    );
    assert_eq!(
        processor.resolve_tool_call_format(Some(ToolCallFormat::MinimaxM2)),
        Some(ToolCallFormat::MinimaxM2)
    );
    assert_eq!(
        processor.tool_call_parser_name(None).as_deref(),
        Some("atem")
    );
    assert_eq!(
        processor
            .tool_call_parser_name(Some("operator-parser"))
            .as_deref(),
        Some("operator-parser")
    );
    assert_eq!(
        infer_default_tool_call_format(
            Some("./models/mlx/muse-glimmer-30b"),
            Some("muse_glimmer"),
            Some(TEMPLATE),
        ),
        Some(ToolCallFormat::Atem)
    );
    assert_eq!(
        resolve_tool_call_format(
            Some(ToolCallFormat::Hermes),
            Some("./models/mlx/muse-glimmer-30b"),
            Some("muse_glimmer"),
            Some(TEMPLATE),
        ),
        Some(ToolCallFormat::Hermes)
    );
    let generic_tools = ChatTemplateProcessor::with_template(
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}".to_string(),
    );
    assert_eq!(
        generic_tools.tool_call_parser_name(None).as_deref(),
        Some("mlxcel")
    );
    let no_tools = ChatTemplateProcessor::with_template(
        "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
    );
    assert_eq!(no_tools.tool_call_parser_name(None), None);
}

#[test]
fn muse_glimmer_template_renders_default_system_text_image_and_turns() {
    let base = base_kwargs();

    let default_system = render(json!([{"role": "user", "content": "Hello."}]), None, &base);
    assert!(
        default_system.starts_with("<|start|>system<|message|>You are a helpful AI assistant.")
    );
    assert!(default_system.contains("Reasoning strength: high."));
    assert_render_hash(
        "default_system",
        &default_system,
        239,
        "f33717e7fcc4ab99b76f4646157eba6804a8842ac03ff9872dc0d90b8bce2f27",
    );

    let image_content = render(
        json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe "},
                {"type": "image"},
                {"type": "text", "text": " briefly."}
            ]
        }]),
        None,
        &base,
    );
    assert!(image_content.contains("<|start|>user<|message|>Describe <|patch|> briefly."));
    assert_render_hash(
        "image_content",
        &image_content,
        260,
        "dbf9697589db648c0e48b6dca8aae021ce0094b397bfdd0f610309aa2a5dc587",
    );

    let multi_turn = render(
        json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Continue."}
        ]),
        None,
        &base,
    );
    assert!(multi_turn.contains("<|message|>Hello!<|eot|>"));
    assert!(!multi_turn.contains("<|message|>Hello!<|eom|>"));
    assert_render_hash(
        "multi_turn",
        &multi_turn,
        325,
        "433d37ff14caf2f2b177d904726b34ff09cb2aad4426a237a7b27772eab47007",
    );
}

#[test]
fn muse_glimmer_template_renders_tools_calls_and_results() {
    let tool = weather_tool();
    let rendered = render(
        json!([
            {"role": "user", "content": "Weather in Seoul?"},
            {
                "role": "assistant",
                "reasoning_content": "I should call the weather tool.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather.get_current",
                        "arguments": {"city": "Seoul", "units": "celsius"}
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"temp_c\": 29}"
            },
            {"role": "assistant", "content": "It is 29 C."}
        ]),
        Some(&[tool]),
        &base_kwargs(),
    );

    assert!(rendered.contains("// Function schemas"));
    assert!(rendered.contains("<atem:invoke name=\"weather.get_current\">"));
    assert!(rendered.contains("<tool_output name=\"weather.get_current\">"));
    assert!(rendered.contains("<|message|>It is 29 C.<|eot|>"));
    assert!(!rendered.contains("<|message|>It is 29 C.<|eom|>"));
    assert_render_hash(
        "tools_and_results",
        &rendered,
        2340,
        "dc451d3030d24f37ecc20fc0236c0b5fa7f70032d8c5331f8f6690689620d6ae",
    );
}

#[test]
fn muse_glimmer_reasoning_strength_defaults_to_high_and_accepts_all_levels() {
    let mut renders = Vec::new();
    for strength in [
        None,
        Some("low"),
        Some("medium"),
        Some("high"),
        Some("xhigh"),
    ] {
        let mut map = Map::new();
        map.insert("current_date".to_string(), json!("2026-08-11"));
        if let Some(strength) = strength {
            map.insert("reasoning_strength".to_string(), json!(strength));
        }
        renders.push((
            strength,
            render(
                json!([{"role": "user", "content": "Calibrate reasoning."}]),
                None,
                &ChatTemplateKwargs::from_json_object(map),
            ),
        ));
    }

    let expected = [
        (
            None,
            253,
            "fadd31bfcb508997b9bb0d8d6ebf3e9c4143589cca3c2405b6c0c45b216f12a2",
        ),
        (
            Some("low"),
            252,
            "3c7f6659a1e3ff073d9e5b9fced3fe92c77e8a91a0ff92a556848695fe9f4636",
        ),
        (
            Some("medium"),
            255,
            "768aa2e02629a4f4eed8ae2eebb4480507f4f7125f1624ef4d3cbd3e49f7269b",
        ),
        (
            Some("high"),
            253,
            "fadd31bfcb508997b9bb0d8d6ebf3e9c4143589cca3c2405b6c0c45b216f12a2",
        ),
        (
            Some("xhigh"),
            254,
            "5c3ae961e9cf9a312c610bb1417780aa20c4165fa3680983c27a0506833f263e",
        ),
    ];
    for ((strength, rendered), (expected_strength, expected_len, expected_sha)) in
        renders.iter().zip(expected)
    {
        assert_eq!(*strength, expected_strength);
        let label = format!("reasoning_{strength:?}");
        assert_render_hash(&label, rendered, expected_len, expected_sha);
    }
    assert_eq!(renders[0].1, renders[3].1);
}

#[test]
fn muse_glimmer_kwargs_precedence_and_reserved_keys_are_enforced() {
    let server = kwargs(&[
        ("current_date", json!("2026-08-11")),
        ("reasoning_strength", json!("low")),
        ("knowledge_cutoff", json!("2025-12-31")),
        (
            "messages",
            json!([{"role": "user", "content": "MALICIOUS"}]),
        ),
    ]);
    let request = extract_request_kwargs(
        Some(&Map::from_iter([
            ("reasoning_strength".to_string(), json!("xhigh")),
            ("add_generation_prompt".to_string(), json!(false)),
            ("tools".to_string(), json!([])),
        ])),
        None,
    );
    let merged = merge_server_and_request(Some(&server), &request);
    let tool = weather_tool();
    let rendered = render(
        json!([{"role": "user", "content": "Use merged kwargs."}]),
        Some(&[tool]),
        &merged,
    );

    assert!(rendered.contains("Knowledge cutoff: 2025-12-31."));
    assert!(rendered.contains("Reasoning strength: xhigh."));
    assert!(!rendered.contains("Reasoning strength: low."));
    assert!(rendered.contains("Use merged kwargs."));
    assert!(!rendered.contains("MALICIOUS"));
    assert!(rendered.contains("weather.get_current"));
    assert!(rendered.ends_with("<|start|>assistant"));
    assert_render_hash(
        "kwargs_precedence_reserved",
        &rendered,
        1827,
        "97ad54c9ddea6a4be5d8fa79f09d365108389f209871f038d0c1141f27ed1bd6",
    );
}
