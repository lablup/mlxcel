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

//! Tests for the native Kimi K3 XTML chat renderer.
//!
//! The fixtures under `tests/fixtures/kimi_k3/xtml/` are the oracle: each one
//! was produced by the checkpoint's own reference tokenizer
//! (`AutoTokenizer.from_pretrained(..., trust_remote_code=True)` resolving
//! `tokenization_kimi.TikTokenTokenizer`), so the `text` and `ids` in them are
//! the reference's answer rather than this port's. See that directory's
//! `README.md` for the regeneration command.
//!
//! Two tiers of test live here. The fixture tier needs the real vocabulary and
//! is gated on `models/kimi-k3-tokenizer` being present, skipping loudly when
//! it is not. The structural tier builds a synthetic K3 vocabulary in a
//! tempdir, so the grammar, the tool-result reordering and the argument
//! normalization stay covered on a machine that has never downloaded the
//! checkpoint.

use super::*;

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::server::types::Role;
use crate::server::types::request::{
    FunctionDefinition, Message, MessageContent, Tool, ToolCallFunction, ToolCallInMessage,
};
use crate::tokenizer::{TiktokenFamily, TiktokenTokenizer};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The tokenizer-only Kimi K3 checkpoint the fixture tier reads. It ships
/// `config.json`, `generation_config.json`, `tokenizer_config.json` and
/// `tiktoken.model` and no weights, so nothing here loads a model.
const K3_DIR: &str = "models/kimi-k3-tokenizer";

const XTML_DIR: &str = "tests/fixtures/kimi_k3/xtml";

/// One reference rendering: the request as the reference saw it, and the text
/// and ids it produced.
#[derive(Debug, serde::Deserialize)]
struct XtmlFixture {
    name: String,
    messages: Vec<Message>,
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    #[serde(default)]
    kwargs: serde_json::Map<String, Value>,
    text: String,
    ids: Vec<i32>,
}

impl XtmlFixture {
    /// The renderer options the fixture's `kwargs` describe.
    ///
    /// An absent key takes the reference `apply_chat_template` default, so
    /// `thinking_effort` absent means `max` while `thinking_effort: null`
    /// means "omit the thinking-effort message". Conflating the two would let
    /// `user_turn_no_effort` pass against the wrong rendering.
    fn options(&self) -> K3RenderOptions<'_> {
        K3RenderOptions {
            add_generation_prompt: self
                .kwargs
                .get("add_generation_prompt")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            thinking: self
                .kwargs
                .get("thinking")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            thinking_effort: match self.kwargs.get("thinking_effort") {
                None => Some(DEFAULT_THINKING_EFFORT.to_string()),
                Some(Value::Null) => None,
                Some(value) => Some(
                    value
                        .as_str()
                        .expect("thinking_effort is a string")
                        .to_string(),
                ),
            },
            tool_choice: self
                .kwargs
                .get("tool_choice")
                .and_then(Value::as_str)
                .map(str::to_string),
            response_format: self.kwargs.get("response_format"),
            image_prompts: None,
        }
    }

    fn render_with(&self, renderer: &KimiK3Renderer) -> K3Rendered {
        renderer
            .render(&self.messages, self.tools.as_deref(), &self.options())
            .unwrap_or_else(|err| panic!("render fixture {}: {err}", self.name))
    }

    /// Assert both halves of the reference answer.
    ///
    /// The text is checked first because a text mismatch names the exact
    /// divergent substring, while an id mismatch on its own only reports two
    /// long integer vectors.
    fn assert_matches_reference(&self, renderer: &KimiK3Renderer) -> K3Rendered {
        let rendered = self.render_with(renderer);
        assert_eq!(
            rendered.text, self.text,
            "fixture {} rendered different text",
            self.name
        );
        assert_eq!(
            rendered.ids, self.ids,
            "fixture {} rendered different ids",
            self.name
        );
        rendered
    }
}

fn load_xtml_fixture(name: &str) -> XtmlFixture {
    let path = PathBuf::from(XTML_DIR).join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("parse {path:?}: {err}"))
}

/// Every fixture in the directory, sorted by file name so a failure names the
/// same case on every machine.
fn all_xtml_fixtures() -> Vec<XtmlFixture> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(XTML_DIR)
        .unwrap_or_else(|err| panic!("read {XTML_DIR}: {err}"))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let raw =
                std::fs::read_to_string(path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
            serde_json::from_str(&raw).unwrap_or_else(|err| panic!("parse {path:?}: {err}"))
        })
        .collect()
}

/// The renderer bound to the real K3 vocabulary, or `None` with a loud skip.
fn k3_renderer(test_name: &str) -> Option<KimiK3Renderer> {
    let dir = Path::new(K3_DIR);
    if !dir.join("tiktoken.model").exists() {
        crate::test_support::pinned_checkpoint::skip_or_fail_pinned_checkpoint(
            test_name,
            "models/kimi-k3-tokenizer is absent",
        );
        return None;
    }
    let tokenizer = Arc::new(
        crate::tokenizer::load_tokenizer(dir).expect("load the Kimi K3 tiktoken vocabulary"),
    );
    Some(KimiK3Renderer::new(tokenizer).expect("the K3 tokenizer yields a renderer"))
}

// ---------------------------------------------------------------------------
// Synthetic vocabulary (no checkpoint required)
// ---------------------------------------------------------------------------

/// Number of BPE ranks the synthetic vocabulary holds: 256 single bytes plus
/// four merges, so the control block starts at 260 the way the real one starts
/// at 163584.
const SYNTHETIC_BASE: u32 = 260;

/// The first six control tokens, in the order the real
/// `tokenizer_config.json` names them.
const SYNTHETIC_CONTROL_NAMES: [&str; 6] = [
    "[BOS]",
    "[EOS]",
    "<|end_of_msg|>",
    "<|open|>",
    "<|close|>",
    "<|sep|>",
];

/// Build a K3-family tokenizer over a throwaway vocabulary.
///
/// The ids are meaningless, but the rendered `text` is vocabulary independent
/// and the control-id placement is not, so every structural property of the
/// grammar is testable without the 2.8 MB checkpoint.
fn synthetic_renderer() -> (tempfile::TempDir, KimiK3Renderer) {
    synthetic_renderer_named(&SYNTHETIC_CONTROL_NAMES)
}

/// The four media control tokens the image prompt is built from (#1342),
/// appended after the six structural ones.
const SYNTHETIC_MEDIA_NAMES: [&str; 4] = [
    "<|media_begin|>",
    "<|media_content|>",
    "<|media_end|>",
    "<|media_pad|>",
];

/// [`synthetic_renderer`] plus the media control tokens.
fn synthetic_renderer_with_media() -> (tempfile::TempDir, KimiK3Renderer) {
    let names: Vec<&str> = SYNTHETIC_CONTROL_NAMES
        .iter()
        .chain(SYNTHETIC_MEDIA_NAMES.iter())
        .copied()
        .collect();
    synthetic_renderer_named(&names)
}

fn synthetic_renderer_named(control_names: &[&str]) -> (tempfile::TempDir, KimiK3Renderer) {
    use base64::Engine;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut lines = String::new();
    let mut rank = 0u32;
    for byte in 0u8..=255 {
        let encoded = base64::engine::general_purpose::STANDARD.encode([byte]);
        lines.push_str(&format!("{encoded} {rank}\n"));
        rank += 1;
    }
    for token in ["me", "ss", "age", "think"] {
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        lines.push_str(&format!("{encoded} {rank}\n"));
        rank += 1;
    }
    assert_eq!(rank, SYNTHETIC_BASE);
    let vocab = dir.path().join("tiktoken.model");
    std::fs::write(&vocab, lines).expect("write synthetic tiktoken file");

    let decoder: serde_json::Map<String, Value> = control_names
        .iter()
        .enumerate()
        .map(|(offset, name)| {
            (
                (SYNTHETIC_BASE + offset as u32).to_string(),
                json!({"content": name, "special": true}),
            )
        })
        .collect();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({ "added_tokens_decoder": decoder })).expect("serialize config"),
    )
    .expect("write tokenizer_config.json");

    let tiktoken =
        TiktokenTokenizer::from_file_with_family(&vocab, dir.path(), TiktokenFamily::KimiK3)
            .expect("load the synthetic K3 vocabulary");
    let tokenizer = Arc::new(MlxcelTokenizer::Tiktoken(tiktoken));
    let renderer = KimiK3Renderer::new(tokenizer).expect("synthetic renderer");
    (dir, renderer)
}

fn user(content: &str) -> Message {
    Message {
        role: Role::User,
        content: MessageContent::Text(content.to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning: None,
    }
}

fn tool_result(tool_call_id: &str, content: &str) -> Message {
    Message {
        role: Role::Tool,
        content: MessageContent::Text(content.to_string()),
        name: None,
        tool_call_id: Some(tool_call_id.to_string()),
        tool_calls: None,
        reasoning: None,
    }
}

/// `MessageContent` carries no `PartialEq`, so unwrap the text arm to compare.
fn plain_text(content: &MessageContent) -> &str {
    match content {
        MessageContent::Text(text) => text,
        other => panic!("expected plain text content, got {other:?}"),
    }
}

fn assistant_calling(calls: &[(&str, &str)]) -> Message {
    Message {
        role: Role::Assistant,
        content: MessageContent::Text(String::new()),
        name: None,
        tool_call_id: None,
        tool_calls: Some(
            calls
                .iter()
                .map(|(id, name)| ToolCallInMessage {
                    id: (*id).to_string(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: (*name).to_string(),
                        arguments: "{}".to_string(),
                    },
                })
                .collect(),
        ),
        reasoning: None,
    }
}

// ---------------------------------------------------------------------------
// Fixture tier: every rendering matches the reference
// ---------------------------------------------------------------------------

#[test]
fn every_xtml_fixture_matches_the_reference_rendering() {
    let Some(renderer) = k3_renderer("every_xtml_fixture_matches_the_reference_rendering") else {
        return;
    };
    let fixtures = all_xtml_fixtures();
    assert!(
        fixtures.len() >= 20,
        "expected the checked-in XTML fixture set, found {}",
        fixtures.len()
    );
    for fixture in &fixtures {
        fixture.assert_matches_reference(&renderer);
    }
}

#[test]
fn render_user_turn_exact_ids() {
    let Some(renderer) = k3_renderer("render_user_turn_exact_ids") else {
        return;
    };
    // `user_turn_no_effort` passes `thinking_effort: null`, so the rendering is
    // exactly the structure the issue pins with no preamble in front of it.
    let fixture = load_xtml_fixture("user_turn_no_effort");
    let rendered = fixture.assert_matches_reference(&renderer);

    let control = renderer.control_ids();
    let tiktoken = renderer
        .tokenizer()
        .tiktoken()
        .expect("the K3 renderer holds a tiktoken vocabulary");
    let text = |s: &str| -> Vec<i32> {
        tiktoken
            .encode_text(s)
            .expect("encode_text")
            .into_iter()
            .map(|id| id as i32)
            .collect()
    };

    let mut expected = vec![control.open as i32];
    expected.extend(text("message"));
    expected.extend(text(" role"));
    expected.extend(text("=\""));
    expected.extend(text("user"));
    expected.extend(text("\""));
    expected.push(control.sep as i32);
    expected.extend(text("Hello, K3!"));
    expected.push(control.close as i32);
    expected.extend(text("message"));
    expected.push(control.sep as i32);
    expected.push(control.end_of_msg as i32);
    expected.push(control.open as i32);
    expected.extend(text("message"));
    expected.extend(text(" role"));
    expected.extend(text("=\""));
    expected.extend(text("assistant"));
    expected.extend(text("\""));
    expected.push(control.sep as i32);
    expected.push(control.open as i32);
    expected.extend(text("think"));
    expected.push(control.sep as i32);

    assert_eq!(rendered.ids, expected);
    // The generation prompt leaves the think channel open, which is what
    // `prompt_primed_open_thinking` reads off the text form.
    assert!(rendered.text.ends_with("<|open|>think<|sep|>"));
    // `[BOS]` is never prepended.
    assert_ne!(rendered.ids.first(), Some(&(control.bos as i32)));
}

#[test]
fn render_assistant_with_reasoning_and_tool_calls() {
    let Some(renderer) = k3_renderer("render_assistant_with_reasoning_and_tool_calls") else {
        return;
    };
    let fixture = load_xtml_fixture("assistant_with_reasoning_and_tool_calls");
    let rendered = fixture.assert_matches_reference(&renderer);

    // The think channel carries the prior turn's reasoning, and each argument
    // is typed by its JSON type with the original literal preserved.
    assert!(rendered.text.contains("<|open|>think<|sep|>"));
    assert!(rendered.text.contains("<|close|>think<|sep|>"));
    assert!(rendered.text.contains(r#"<|open|>tools<|sep|>"#));
    assert!(
        rendered
            .text
            .contains(r#"<|open|>call tool="get_weather" index="1"<|sep|>"#)
    );
    assert!(
        rendered.text.contains(
            r#"<|open|>argument key="count" type="number"<|sep|>1e2<|close|>argument<|sep|>"#
        ),
        "the number literal must survive verbatim: {}",
        rendered.text
    );

    // A raw-string arguments payload that is not a JSON object renders as one
    // `json` block instead of per-key arguments.
    let raw = load_xtml_fixture("assistant_raw_json_block");
    let raw_rendered = raw.assert_matches_reference(&renderer);
    assert!(
        raw_rendered
            .text
            .contains(r#"<|open|>json type="object"<|sep|>"#)
    );

    // Empty arguments produce a call with no argument entries at all.
    let empty = load_xtml_fixture("assistant_empty_arguments");
    let empty_rendered = empty.assert_matches_reference(&renderer);
    assert!(!empty_rendered.text.contains("<|open|>argument"));
}

#[test]
fn render_tool_results_reordered_by_call_id() {
    let Some(renderer) = k3_renderer("render_tool_results_reordered_by_call_id") else {
        return;
    };
    let fixture = load_xtml_fixture("tool_results_reordered_by_call_id");
    let rendered = fixture.assert_matches_reference(&renderer);

    // The wire order is call_b then call_a; the assistant called a then b, so
    // the rendering swaps them and names each result after the call it matched.
    let first = rendered
        .text
        .find(r#"tool="get_weather" index="1"<|sep|>first result"#)
        .expect("the call_a result renders first");
    let second = rendered
        .text
        .find(r#"tool="search" index="2"<|sep|>second result"#)
        .expect("the call_b result renders second");
    assert!(first < second);

    // The index keeps counting across the intervening user turn, and an
    // unmatched run stays where it was written.
    assert!(
        rendered
            .text
            .contains(r#"tool="search" index="3"<|sep|>unmatched"#)
    );
}

#[test]
fn render_tools_declaration_sorted_compact() {
    let Some(renderer) = k3_renderer("render_tools_declaration_sorted_compact") else {
        return;
    };
    let fixture = load_xtml_fixture("tools_declaration_sorted_compact");
    let rendered = fixture.assert_matches_reference(&renderer);

    // The declaration leads the prompt, is compact (no spaces after `:` or
    // `,`), key-sorted at every depth, and keeps non-ASCII unescaped.
    assert!(rendered.text.starts_with(
        "<|open|>message role=\"system\" type=\"tool-declare\"<|sep|># Tools\n\
         Here are the available tools, described in JSONSchema.\n\n```json\n["
    ));
    assert!(rendered.text.contains(r#"{"function":{"description":"#));
    assert!(rendered.text.contains("날씨를 조회한다"));
    assert!(
        rendered
            .text
            .contains(r#""properties":{"location":{"description":"City name","type":"string"}"#),
        "keys sort at every depth: {}",
        rendered.text
    );
    assert!(!rendered.text.contains(r#"", ""#));
}

#[test]
fn render_tool_choice_and_response_format() {
    let Some(renderer) = k3_renderer("render_tool_choice_and_response_format") else {
        return;
    };
    let required = load_xtml_fixture("tool_choice_required").assert_matches_reference(&renderer);
    assert!(required.text.contains(
        "<|open|>message role=\"system\" type=\"tool-choice\"<|sep|>The system is invoked with \
         `tool_choice=required`.\nYou MUST call tools in the next message."
    ));

    let none = load_xtml_fixture("tool_choice_none").assert_matches_reference(&renderer);
    assert!(none.text.contains(
        "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next \
         message."
    ));

    let object =
        load_xtml_fixture("response_format_json_object").assert_matches_reference(&renderer);
    assert!(object.text.contains(
        "<|open|>message role=\"system\" type=\"response-format\"<|sep|>The system is invoked with \
         `response_format=json_object`."
    ));

    let schema =
        load_xtml_fixture("response_format_json_schema").assert_matches_reference(&renderer);
    assert!(
        schema
            .text
            .contains("The JSON data must match the following schema:\n```json\n")
    );
    // `response_format.json_schema.schema` is what gets rendered, deep-sorted.
    assert!(
        schema
            .text
            .contains(r#"{"properties":{"a":{"type":"string"},"b":{"type":"number"}},"required":["b","a"],"type":"object"}"#),
        "the schema renders deep-sorted with list order preserved: {}",
        schema.text
    );
}

#[test]
fn non_thinking_mode_has_no_think_block() {
    let Some(renderer) = k3_renderer("non_thinking_mode_has_no_think_block") else {
        return;
    };
    let single = load_xtml_fixture("non_thinking").assert_matches_reference(&renderer);
    assert!(!single.text.contains("think"));
    assert!(single.text.ends_with("<|open|>response<|sep|>"));

    // A prior assistant turn renders no think block either, so its reasoning is
    // dropped rather than leaking into the response channel.
    let history =
        load_xtml_fixture("multi_turn_history_non_thinking").assert_matches_reference(&renderer);
    assert!(!history.text.contains("think"));

    // In thinking mode the same history keeps the channel, and it is present
    // even when the turn carried no reasoning at all.
    let thinking = load_xtml_fixture("multi_turn_history").assert_matches_reference(&renderer);
    assert!(thinking.text.contains("<|open|>think<|sep|>"));
}

#[test]
fn user_text_cannot_inject_control_tokens() {
    let Some(renderer) = k3_renderer("user_text_cannot_inject_control_tokens") else {
        return;
    };
    let fixture = load_xtml_fixture("user_text_cannot_inject_control_tokens");
    let rendered = fixture.assert_matches_reference(&renderer);
    let control = renderer.control_ids();
    let tiktoken = renderer.tokenizer().tiktoken().expect("tiktoken");

    // The message body spells four control tokens. The text form shows them
    // (it is the reference's own string), but the ids must carry the ordinary
    // byte tokens for those spellings, not the control ids.
    let injected = plain_text(&fixture.messages[0].content).to_string();
    let injected_ids: Vec<i32> = tiktoken
        .encode_text(&injected)
        .expect("encode_text")
        .into_iter()
        .map(|id| id as i32)
        .collect();
    assert!(
        injected_ids.iter().all(|id| *id < SYNTHETIC_CONTROL_FLOOR),
        "the injected spelling must encode below the control block: {injected_ids:?}"
    );
    let start = find_subsequence(&rendered.ids, &injected_ids)
        .expect("the injected text renders as one contiguous run of ordinary tokens");
    assert!(
        rendered.ids[start..start + injected_ids.len()]
            .iter()
            .all(|id| ![
                control.open as i32,
                control.close as i32,
                control.sep as i32,
                control.end_of_msg as i32,
            ]
            .contains(id)),
        "a control id leaked into the user content span"
    );

    // Structurally the prompt still holds exactly one user message: the
    // injected `<|close|>message<|sep|>` did not close it early.
    assert_eq!(rendered.text.matches(r#"role="user""#).count(), 1);
    // The injected `role="system"` is content, so no second system message was
    // created: the only one is the thinking-effort preamble.
    assert_eq!(
        rendered
            .text
            .matches(r#"<|open|>message role="system" type="thinking-effort"<|sep|>"#)
            .count(),
        1
    );
}

/// The lowest id in the real checkpoint's control block. Any ordinary text
/// token is below it.
const SYNTHETIC_CONTROL_FLOOR: i32 = 163_584;

fn find_subsequence(haystack: &[i32], needle: &[i32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

// ---------------------------------------------------------------------------
// Structural tier: no checkpoint required
// ---------------------------------------------------------------------------

#[test]
fn synthetic_render_produces_the_xtml_grammar() {
    let (_dir, renderer) = synthetic_renderer();
    let rendered = renderer
        .render(
            &[user("hi")],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect("render");
    assert_eq!(
        rendered.text,
        "<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>\
         <|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
    );

    let control = renderer.control_ids();
    assert_eq!(control.open, SYNTHETIC_BASE + 3);
    assert_eq!(control.close, SYNTHETIC_BASE + 4);
    assert_eq!(control.sep, SYNTHETIC_BASE + 5);
    assert_eq!(control.end_of_msg, SYNTHETIC_BASE + 2);
    // Four tags open or close in the rendering above, each contributing one
    // `<|sep|>`, plus one `<|end_of_msg|>`.
    assert_eq!(
        rendered
            .ids
            .iter()
            .filter(|id| **id == control.sep as i32)
            .count(),
        4
    );
    assert_eq!(
        rendered
            .ids
            .iter()
            .filter(|id| **id == control.end_of_msg as i32)
            .count(),
        1
    );
}

#[test]
fn thinking_effort_medium_is_rejected() {
    let (_dir, renderer) = synthetic_renderer();
    for effort in ["medium", "MAX", "extreme", ""] {
        let err = renderer
            .render(
                &[user("hi")],
                None,
                &K3RenderOptions {
                    thinking_effort: Some(effort.to_string()),
                    ..K3RenderOptions::reference_defaults()
                },
            )
            .expect_err("an out-of-set thinking effort is an error");
        assert!(
            err.to_string().contains("thinking_effort"),
            "unhelpful error for {effort:?}: {err}"
        );
    }
    // In non-thinking mode the effort is ignored rather than validated, which
    // is what the reference does.
    let rendered = renderer
        .render(
            &[user("hi")],
            None,
            &K3RenderOptions {
                thinking: false,
                thinking_effort: Some("medium".to_string()),
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect("non-thinking mode ignores the effort");
    assert!(!rendered.text.contains("thinking-effort"));
}

#[test]
fn portable_reasoning_effort_clamps_onto_the_three_rendered_levels() {
    // The two names both ladders share keep their meaning.
    assert_eq!(clamp_portable_reasoning_effort("low"), Some("low"));
    assert_eq!(clamp_portable_reasoning_effort("high"), Some("high"));
    assert_eq!(clamp_portable_reasoning_effort("max"), Some("max"));

    // OpenAI's default effort is the one `VALID_THINKING_EFFORTS` omits, so
    // without the clamp a plain `reasoning_effort` request would 400. It
    // rounds up, toward K3's own `max` default, rather than down.
    assert_eq!(clamp_portable_reasoning_effort("medium"), Some("high"));
    assert_eq!(clamp_portable_reasoning_effort("minimal"), Some("low"));

    // An unrecognized level is not invented into a valid one; it falls
    // through to the renderer, which still rejects it.
    assert_eq!(clamp_portable_reasoning_effort("turbo"), None);
    assert_eq!(clamp_portable_reasoning_effort(""), None);

    // Every level the clamp produces is one the renderer accepts.
    for portable in ["minimal", "low", "medium", "high", "max"] {
        let clamped = clamp_portable_reasoning_effort(portable).expect("clamped");
        assert!(
            VALID_THINKING_EFFORTS.contains(&clamped),
            "{portable} clamped to {clamped}, which the renderer rejects"
        );
    }
}

#[test]
fn unresolvable_tool_name_is_an_error() {
    let (_dir, renderer) = synthetic_renderer();
    let err = renderer
        .render(
            &[user("hi"), tool_result("call_x", "result")],
            None,
            &K3RenderOptions::reference_defaults(),
        )
        .expect_err("a tool result with no name and no matching call cannot render");
    assert!(err.to_string().contains("tool name"), "{err}");
}

#[test]
fn image_placeholder_is_literal_text_without_image_prompts() {
    let (_dir, renderer) = synthetic_renderer();
    let rendered = renderer
        .render(
            &[user("before <|kimi_image_placeholder|> after")],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect("render");
    assert!(
        rendered
            .text
            .contains("before <|kimi_image_placeholder|> after")
    );

    // With prompts supplied, each placeholder consumes one and an unconsumed
    // leftover is an error rather than a silently dropped image.
    let prompts = vec![
        K3ImagePrompt {
            ids: vec![7, 8],
            text: "<IMG>".to_string(),
        },
        K3ImagePrompt {
            ids: vec![9],
            text: "<IMG2>".to_string(),
        },
    ];
    let one = renderer
        .render(
            &[user("a <|kimi_image_placeholder|> b")],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                image_prompts: Some(&prompts),
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect_err("one placeholder cannot consume two prompts");
    assert!(one.to_string().contains("image prompt count"), "{one}");

    let both = renderer
        .render(
            &[user(
                "a <|kimi_image_placeholder|> b <|kimi_image_placeholder|> c",
            )],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                image_prompts: Some(&prompts),
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect("two placeholders consume two prompts");
    assert!(both.text.contains("a <IMG> b <IMG2> c"));
    assert!(both.ids.windows(2).any(|w| w == [7, 8]));
}

#[test]
fn dynamic_tool_declare_body_is_the_lazy_loading_form() {
    let (_dir, renderer) = synthetic_renderer();
    let tools = json!([{"type": "function", "function": {"name": "t"}}]);

    let mut sink = Sink::new(&renderer);
    sink.tool_declare(&tools, true).expect("dynamic declare");
    let dynamic = sink.finish().text;
    assert!(dynamic.contains("## New Tools Available"));
    assert!(dynamic.contains("The system dynamically extends the toolset via lazy-loading."));

    let mut sink = Sink::new(&renderer);
    sink.tool_declare(&tools, false).expect("static declare");
    let static_form = sink.finish().text;
    assert!(static_form.contains("# Tools"));
    assert!(!static_form.contains("New Tools Available"));
    // mlxcel's wire `Message` carries no `tools` field, so only the static form
    // is reachable from an HTTP request; the dynamic body exists for parity
    // with the reference and is covered only here.
    //
    // Both forms serialize the value they were handed verbatim: key sorting is
    // `render`'s job (`deep_sorted_tools`), not this one's, so the caller's
    // `type` before `function` order survives here.
    assert!(static_form.contains(r#"[{"type":"function","function":{"name":"t"}}]"#));
}

// ---------------------------------------------------------------------------
// Tool-result reordering (pure)
// ---------------------------------------------------------------------------

#[test]
fn normalize_reorders_a_fully_matched_run_and_names_each_result() {
    let messages = vec![
        user("go"),
        assistant_calling(&[("call_a", "alpha"), ("call_b", "beta")]),
        tool_result("call_b", "B"),
        tool_result("call_a", "A"),
    ];
    let normalized = normalize_xtml_tool_result_messages(&messages);
    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[2].name.as_deref(), Some("alpha"));
    assert_eq!(normalized[3].name.as_deref(), Some("beta"));
    assert_eq!(plain_text(&normalized[2].content), "A");
    // Side-effect free: the caller's slice is untouched.
    assert_eq!(messages[2].name, None);
}

#[test]
fn normalize_leaves_a_run_with_any_unmatched_id_exactly_as_written() {
    let messages = vec![
        assistant_calling(&[("call_a", "alpha"), ("call_b", "beta")]),
        tool_result("call_b", "B"),
        tool_result("nope", "X"),
        tool_result("call_a", "A"),
    ];
    let normalized = normalize_xtml_tool_result_messages(&messages);
    assert_eq!(
        normalized
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect::<Vec<_>>(),
        vec!["call_b", "nope", "call_a"]
    );
    assert!(normalized.iter().all(|m| m.name.is_none()));
}

#[test]
fn normalize_keeps_the_first_of_a_duplicate_call_id_and_splits_runs_at_a_user_turn() {
    let messages = vec![
        assistant_calling(&[("dup", "first"), ("dup", "second")]),
        tool_result("dup", "one"),
        user("interleaved"),
        tool_result("dup", "two"),
    ];
    let normalized = normalize_xtml_tool_result_messages(&messages);
    // Both runs match the same first-wins entry, and the user turn between
    // them is its own boundary rather than part of either run.
    assert_eq!(normalized[1].name.as_deref(), Some("first"));
    assert_eq!(normalized[2].role, Role::User);
    assert_eq!(normalized[3].name.as_deref(), Some("first"));
}

// ---------------------------------------------------------------------------
// Argument normalization (pure)
// ---------------------------------------------------------------------------

fn entries(arguments: &str) -> Vec<XtmlArgument> {
    match normalize_tool_arguments(arguments) {
        K3Arguments::Entries(entries) => entries,
        K3Arguments::RawJson(raw) => panic!("expected parsed entries, got raw {raw:?}"),
    }
}

#[test]
fn tool_arguments_keep_the_original_literal_for_non_string_values() {
    let parsed = entries(
        r#"{"s": "he said \"hi\" & left", "n": 1e2, "b": true, "z": null, "a": [1, 2,  3], "o": {"b": 2, "a": 1}}"#,
    );
    let by_key: HashMap<&str, &XtmlArgument> = parsed.iter().map(|e| (e.key.as_str(), e)).collect();

    // Strings are decoded; everything else keeps the bytes the caller wrote,
    // spacing included, because that is what the reference renders.
    assert_eq!(by_key["s"].value_type, "string");
    assert_eq!(by_key["s"].text, r#"he said "hi" & left"#);
    assert_eq!(by_key["n"].value_type, "number");
    assert_eq!(by_key["n"].text, "1e2");
    assert_eq!(by_key["b"].value_type, "boolean");
    assert_eq!(by_key["b"].text, "true");
    assert_eq!(by_key["z"].value_type, "null");
    assert_eq!(by_key["z"].text, "null");
    assert_eq!(by_key["a"].value_type, "array");
    assert_eq!(by_key["a"].text, "[1, 2,  3]");
    assert_eq!(by_key["o"].value_type, "object");
    assert_eq!(by_key["o"].text, r#"{"b": 2, "a": 1}"#);

    // Key order is the caller's, not sorted: the reference iterates the parsed
    // mapping in insertion order.
    assert_eq!(
        parsed.iter().map(|e| e.key.as_str()).collect::<Vec<_>>(),
        vec!["s", "n", "b", "z", "a", "o"]
    );
}

#[test]
fn tool_arguments_ignore_trailing_bytes_after_the_closing_brace() {
    let parsed = entries(r#"{"a": 1} trailing garbage"#);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].text, "1");
}

#[test]
fn tool_arguments_that_are_not_an_object_fall_back_to_a_raw_json_block() {
    for arguments in [
        "not json at all",
        "   ",
        "[1, 2, 3]",
        "\"just a string\"",
        "{\"unterminated\": ",
        "{,}",
    ] {
        assert!(
            matches!(normalize_tool_arguments(arguments), K3Arguments::RawJson(raw) if raw == arguments),
            "{arguments:?} should fall back to a raw block"
        );
    }
    // An empty arguments string is no arguments, not a raw block.
    assert_eq!(
        normalize_tool_arguments(""),
        K3Arguments::Entries(Vec::new())
    );
    assert_eq!(
        normalize_tool_arguments("{}"),
        K3Arguments::Entries(Vec::new())
    );
    assert_eq!(
        normalize_tool_arguments("  { }  "),
        K3Arguments::Entries(Vec::new())
    );
}

#[test]
fn pre_parsed_arguments_use_python_default_separators() {
    let parsed = normalize_tool_arguments_value(&json!({
        "s": "raw string",
        "a": [1, 2, 3],
        "o": {"b": 2},
        "n": 1.5,
    }))
    .expect("an object normalizes");
    let by_key: HashMap<&str, &XtmlArgument> = parsed.iter().map(|e| (e.key.as_str(), e)).collect();
    assert_eq!(by_key["s"].text, "raw string");
    assert_eq!(by_key["a"].text, "[1, 2, 3]");
    assert_eq!(by_key["o"].text, r#"{"b": 2}"#);
    assert_eq!(by_key["n"].text, "1.5");
    assert!(normalize_tool_arguments_value(&json!([1])).is_none());
}

// ---------------------------------------------------------------------------
// JSON and escaping helpers (pure)
// ---------------------------------------------------------------------------

#[test]
fn deep_sort_orders_keys_at_every_depth_and_keeps_list_order() {
    let sorted = deep_sort(&json!({
        "b": 1,
        "a": {"z": [{"y": 1, "x": 2}], "y": 2},
        "Z": 3,
        "가": 4,
    }));
    assert_eq!(
        compact_json(&sorted).expect("serialize"),
        r#"{"Z":3,"a":{"y":2,"z":[{"x":2,"y":1}]},"b":1,"가":4}"#
    );
}

#[test]
fn compact_json_matches_python_ensure_ascii_false_escaping() {
    // Python's `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`
    // escapes the same set serde_json does: `\n`, `\t`, `\"`, `\\`, and
    // `\u00XX` for other control characters. `/` is escaped by neither, and
    // non-ASCII passes through.
    let value = json!({"k": "a\nb\tc\"d\\e/f\u{1}g 漢 😀"});
    assert_eq!(
        compact_json(&value).expect("serialize"),
        "{\"k\":\"a\\nb\\tc\\\"d\\\\e/f\\u0001g 漢 😀\"}"
    );
}

#[test]
fn escape_attr_value_escapes_ampersand_before_quote() {
    assert_eq!(escape_attr_value(r#"a & b"#), "a &amp; b");
    assert_eq!(escape_attr_value(r#"say "hi""#), "say &quot;hi&quot;");
    // `&` first, so an input that already spells `&quot;` is escaped once into
    // `&amp;quot;` and round-trips through the stream parser unchanged.
    assert_eq!(escape_attr_value("&quot;"), "&amp;quot;");
    // `<` and `>` are not escaped: the XTML delimiters are control tokens, not
    // angle brackets, so there is nothing for them to break out of.
    assert_eq!(escape_attr_value("<tag>"), "<tag>");
}

#[test]
fn response_format_type_and_schema_follow_the_reference_fallbacks() {
    let object = json!({"type": "json_object"});
    assert_eq!(response_format_type(Some(&object)), Some("json_object"));
    assert_eq!(
        response_format_type(Some(&json!("json_object"))),
        Some("json_object")
    );
    assert_eq!(response_format_type(None), None);
    assert_eq!(response_format_type(Some(&json!({}))), None);

    // `.json_schema.schema` wins, then `.json_schema.json_schema`, then the
    // `json_schema` object itself.
    let with_schema = json!({"json_schema": {"schema": {"a": 1}, "json_schema": {"b": 2}}});
    assert_eq!(
        extract_response_schema(Some(&with_schema)),
        Some(&json!({"a": 1}))
    );
    let nested = json!({"json_schema": {"json_schema": {"b": 2}}});
    assert_eq!(
        extract_response_schema(Some(&nested)),
        Some(&json!({"b": 2}))
    );
    let bare = json!({"json_schema": {"type": "object"}});
    assert_eq!(
        extract_response_schema(Some(&bare)),
        Some(&json!({"type": "object"}))
    );
    assert_eq!(extract_response_schema(Some(&json!({}))), None);
}

#[test]
fn tools_serialize_without_absent_optional_fields() {
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "t".to_string(),
            description: None,
            parameters: None,
        },
    }];
    assert_eq!(
        compact_json(&deep_sorted_tools(&tools).expect("serialize")).expect("compact"),
        r#"[{"function":{"name":"t"},"type":"function"}]"#
    );
}

// ---------------------------------------------------------------------------
// Image prompts (#1342)
// ---------------------------------------------------------------------------

#[test]
fn image_prompt_needs_the_media_control_ids() {
    let (_dir, renderer) = synthetic_renderer();
    assert!(renderer.media_token_ids().is_none());
    let err = renderer
        .image_prompt(100, 100)
        .expect_err("no media ids, no image prompt");
    assert!(err.to_string().contains("<|media_begin|>"), "{err}");
}

#[test]
fn image_prompt_places_media_ids_around_the_pad_run() {
    let (_dir, renderer) = synthetic_renderer_with_media();
    let media = renderer.media_token_ids().expect("media ids");
    let base = SYNTHETIC_BASE as i32 + SYNTHETIC_CONTROL_NAMES.len() as i32;
    assert_eq!(media.begin, base);
    assert_eq!(media.content, base + 1);
    assert_eq!(media.end, base + 2);
    assert_eq!(media.pad, base + 3);

    // 100x100 -> grid (8, 8) -> 16 tokens, and the label is BPE text.
    let prompt = renderer.image_prompt(100, 100).expect("image prompt");
    let label: Vec<i32> = renderer
        .tokenizer()
        .encode_with_special("image 100x100", false, false)
        .expect("label")
        .into_iter()
        .map(|id| id as i32)
        .collect();
    let mut expected = vec![media.begin];
    expected.extend(&label);
    expected.push(media.content);
    expected.extend(std::iter::repeat_n(media.pad, 16));
    expected.push(media.end);
    assert_eq!(prompt.ids, expected);
    assert_eq!(
        prompt.text,
        crate::vision::processors::kimi_k3::image_prompt_text(100, 100, 16)
    );
    assert!(
        prompt
            .text
            .starts_with("<|media_begin|>image 100x100<|media_content|><|media_pad|>")
    );
    assert!(prompt.text.ends_with("<|media_pad|><|media_end|>"));

    // The navit parameters size the run: a 4000x3000 photo costs 15444 pads.
    let big = renderer.image_prompt(4000, 3000).expect("image prompt");
    assert_eq!(
        big.ids.iter().filter(|&&id| id == media.pad).count(),
        15_444
    );
    // A checkpoint-provided budget changes the count through the same rule.
    let tight = synthetic_renderer_with_media().1.with_navit_config(
        crate::vision::processors::kimi_k3::KimiK3NavitConfig {
            in_patch_limit: 64,
            ..Default::default()
        },
    );
    let small = tight.image_prompt(4000, 3000).expect("image prompt");
    assert!(small.ids.iter().filter(|&&id| id == media.pad).count() < 15_444);

    // Rendered into a user turn, the block sits between the text segments
    // and every media id is a control id, not a re-encoding of its spelling.
    let message = Message {
        role: Role::User,
        content: MessageContent::Parts(vec![
            crate::server::types::request::ContentPart::Text {
                text: "look:".to_string(),
            },
            crate::server::types::request::ContentPart::ImageUrl {
                image_url: crate::server::types::request::ImageUrl::new(
                    "data:image/png;base64,AA==",
                ),
            },
            crate::server::types::request::ContentPart::Text {
                text: "what is it?".to_string(),
            },
        ]),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning: None,
    };
    let prompts = vec![prompt.clone()];
    let rendered = renderer
        .render(
            &[message],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                image_prompts: Some(&prompts),
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect("render with one image");
    assert!(
        rendered
            .ids
            .windows(prompt.ids.len())
            .any(|window| window == prompt.ids.as_slice()),
        "the image block must appear verbatim in the rendered ids"
    );
    assert!(
        rendered
            .text
            .contains("look:<|media_begin|>image 100x100<|media_content|>")
    );
    assert!(rendered.text.contains("<|media_end|>what is it?"));
}

#[test]
fn an_image_url_part_without_image_prompts_is_refused() {
    // The literal `<|kimi_image_placeholder|>` fallback belongs to the text
    // path. An `image_url` part rendered with no prompt supplied would put
    // the placeholder in `text` and nothing in `ids`, so the model would
    // never see the image; that has to be an error, not a silent drop.
    let (_dir, renderer) = synthetic_renderer();
    let message = Message {
        role: Role::User,
        content: MessageContent::Parts(vec![
            crate::server::types::request::ContentPart::Text {
                text: "look:".to_string(),
            },
            crate::server::types::request::ContentPart::ImageUrl {
                image_url: crate::server::types::request::ImageUrl::new(
                    "data:image/png;base64,AA==",
                ),
            },
        ]),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning: None,
    };
    let err = renderer
        .render(
            &[message],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                image_prompts: None,
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect_err("an image part with no prompt cannot render");
    assert!(err.to_string().contains("image_url content part"), "{err}");
}

#[test]
fn an_image_on_a_tool_message_is_refused() {
    // A run of tool results is reordered to the assistant's tool-call order,
    // while images stay in wire order, so an image on a tool message would be
    // paired with the wrong prompt without any count check noticing.
    let (_dir, renderer) = synthetic_renderer_with_media();
    let media = renderer.media_token_ids().expect("media ids");
    let prompt = renderer.image_prompt(100, 100).expect("image prompt");
    assert!(prompt.ids.contains(&media.pad));
    let tool_with_image = Message {
        role: Role::Tool,
        content: MessageContent::Parts(vec![
            crate::server::types::request::ContentPart::Text {
                text: "result".to_string(),
            },
            crate::server::types::request::ContentPart::ImageUrl {
                image_url: crate::server::types::request::ImageUrl::new(
                    "data:image/png;base64,AA==",
                ),
            },
        ]),
        name: Some("lookup".to_string()),
        tool_call_id: Some("call_1".to_string()),
        tool_calls: None,
        reasoning: None,
    };
    let prompts = vec![prompt];
    let err = renderer
        .render(
            &[user("go"), tool_with_image],
            None,
            &K3RenderOptions {
                thinking_effort: None,
                image_prompts: Some(&prompts),
                ..K3RenderOptions::reference_defaults()
            },
        )
        .expect_err("an image on a tool message cannot render");
    assert!(err.to_string().contains("tool message"), "{err}");
}
