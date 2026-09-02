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

//! Tool call output parsing and formatting.
//!
//! This module detects and parses tool call patterns from model output,
//! supporting multiple formats used by popular model families (Hermes/Qwen,
//! Llama 3.x, Mistral Nemo, Functionary, etc.).
//!
//! Used by: routes/chat, chat_request

mod atem;
mod formats;
pub mod parser;
pub mod stream_filter;
pub mod types;

pub use atem::render_muse_channels_for_display;
pub use parser::{
    clean_structural_tokens, content_with_thinking_block, generate_tool_call_id, parse_tool_calls,
    thinking_marker_pair, thinking_marker_pair_for_close,
};
pub use types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};

use super::types::request::{ChatCompletionRequest, Tool, ToolChoice};
use super::types::response::{ToolCallFunctionResponse, ToolCallResponse};

/// Infer a default tool-call format from model/config/template identity.
///
/// This is intentionally conservative and never handles an explicit operator
/// parser choice; callers should pass explicit choices through
/// [`resolve_tool_call_format`] so user configuration remains authoritative.
pub fn infer_default_tool_call_format(
    model_id: Option<&str>,
    model_type: Option<&str>,
    template_source: Option<&str>,
) -> Option<ToolCallFormat> {
    if template_source.is_some_and(template_uses_atem)
        || model_type.is_some_and(is_muse_glimmer_identity)
        || model_id.is_some_and(is_muse_glimmer_identity)
    {
        Some(ToolCallFormat::Atem)
    } else {
        None
    }
}

/// Resolve the active tool-call format without overriding an explicit choice.
pub fn resolve_tool_call_format(
    explicit: Option<ToolCallFormat>,
    model_id: Option<&str>,
    model_type: Option<&str>,
    template_source: Option<&str>,
) -> Option<ToolCallFormat> {
    explicit.or_else(|| infer_default_tool_call_format(model_id, model_type, template_source))
}

fn template_uses_atem(template: &str) -> bool {
    template.contains("<atem:function_calls>") && template.contains("<atem:invoke")
}

fn is_muse_glimmer_identity(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase().replace('-', "_");
    normalized.contains("muse_glimmer")
}

/// The llguidance Lark grammar that forces a tool call for a grammar-capable
/// format (#1319), or `None` when nothing can be forced.
///
/// Grammar-capable means the emitted call is one JSON object inside a fixed
/// wrapper, so the wrapper can be spelled as literals around a `%json` schema
/// built from the declared tools:
///
/// ```text
/// start: "<tool_call>" %json { <schema> } "</tool_call>"    # Hermes
/// start: %json { <schema> }                                   # Llama 3
/// start: "[TOOL_CALLS] [" %json { <schema> } "]"              # Mistral Nemo
/// ```
///
/// Hermes additionally allows the newline the Qwen templates put on either
/// side of the JSON object, so the forced shape is the one the model was
/// trained on. The per-tool schema pins `name` to a constant and types the
/// argument object with the tool's own `parameters` (an empty object schema
/// when the tool declares none); Llama 3 spells that key `parameters`, the
/// other two `arguments`. `"required"` becomes an `anyOf` over every declared
/// tool, a named choice is that tool's schema alone. `auto`, `none`, a format
/// without a JSON wire shape (ATEM, Gemma 4, XML dialects, ...), an empty tool
/// list and a named function that is not declared all yield `None`: those
/// cases get the prompt instruction and tool narrowing only.
///
/// This form spells every wrapper marker as bytes. Production callers use
/// [`tool_choice_grammar_with`] so markers the tokenizer carries as special
/// tokens are referenced by id instead; see [`special_token_id`].
// Used by: routes/chat
pub fn tool_choice_grammar(
    format: ToolCallFormat,
    tools: &[Tool],
    choice: &ToolChoice,
) -> Option<String> {
    tool_choice_grammar_with(format, tools, choice, |_| None)
}

/// One element of a format's fixed wrapper around the JSON call.
enum WrapperPiece {
    /// A marker the model may carry as a single special token (`<tool_call>`,
    /// `[TOOL_CALLS]`). Spelled by id when the tokenizer does, as bytes
    /// otherwise.
    Marker(&'static str),
    /// Plain bytes.
    Literal(&'static str),
    /// Plain bytes the model may leave out.
    Optional(&'static str),
}

/// [`tool_choice_grammar`] with the tokenizer's view of the wrapper markers.
///
/// `special_token` answers, for a marker string, the id of the token
/// llguidance treats as special for exactly that text. A special token never
/// matches its own bytes inside a grammar (the trie stores it behind a marker
/// byte), so a byte-literal `"<tool_call>"` would mask out the very token a
/// Qwen or Hermes model emits and the forced call could never start. Such
/// markers are referenced by id (`<[151657]>`) instead; markers the tokenizer
/// does not carry as special tokens stay byte literals, which is what a model
/// that spells them from pieces needs.
// Used by: routes/chat
pub fn tool_choice_grammar_with(
    format: ToolCallFormat,
    tools: &[Tool],
    choice: &ToolChoice,
    special_token: impl Fn(&str) -> Option<u32>,
) -> Option<String> {
    use WrapperPiece::{Literal, Marker, Optional};
    let (prefix, suffix, arguments_key): (&[WrapperPiece], &[WrapperPiece], &str) = match format {
        ToolCallFormat::Hermes => (
            &[Marker("<tool_call>"), Optional("\n")],
            &[Optional("\n"), Marker("</tool_call>")],
            "arguments",
        ),
        ToolCallFormat::Llama3 => (&[], &[], "parameters"),
        ToolCallFormat::MistralNemo => (
            &[Marker("[TOOL_CALLS]"), Optional(" "), Literal("[")],
            &[Literal("]")],
            "arguments",
        ),
        _ => return None,
    };
    let render = |pieces: &[WrapperPiece]| -> String {
        pieces
            .iter()
            .map(|piece| match piece {
                Marker(text) => {
                    special_token(text).map_or_else(|| lark_string(text), |id| format!("<[{id}]>"))
                }
                Literal(text) => lark_string(text),
                Optional(text) => format!("{}?", lark_string(text)),
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let prefix = render(prefix);
    let suffix = render(suffix);
    let selected: Vec<&Tool> = match choice {
        ToolChoice::Specific(named) => tools
            .iter()
            .find(|tool| tool.function.name == named.function.name)
            .into_iter()
            .collect(),
        ToolChoice::Mode(mode) if mode == "required" => tools.iter().collect(),
        _ => return None,
    };
    if selected.is_empty() {
        return None;
    }
    // Pre-flight the declared schemas before any of them is cloned. The
    // finished Lark text is capped by `MAX_SCHEMA_BYTES` as well, but that cap
    // is applied on the blocking pool after this function has already run, so
    // on its own it bounds nothing this function does.
    if !tool_schemas_fit(&selected, crate::server::structured::MAX_SCHEMA_BYTES) {
        tracing::warn!(
            target: "mlxcel::tool_calls",
            tools = selected.len(),
            limit = crate::server::structured::MAX_SCHEMA_BYTES,
            "forced tool_choice: the declared tool schemas are larger than the grammar size \
             limit; falling back to prompt-instruction enforcement"
        );
        return None;
    }
    let single = selected.len() == 1;
    let mut schemas: Vec<serde_json::Value> = selected
        .into_iter()
        .enumerate()
        .map(|(index, tool)| {
            // Where this branch's `arguments` object ends up in the finished
            // document, so the tool's own internal `$ref`s can be re-pointed
            // at it. A single tool is the document root; several tools sit
            // under `anyOf`.
            let base = if single {
                format!("properties/{arguments_key}")
            } else {
                format!("anyOf/{index}/properties/{arguments_key}")
            };
            tool_call_schema(tool, arguments_key, &base)
        })
        .collect();
    let schema = if single {
        schemas.remove(0)
    } else {
        let mut any_of = serde_json::Map::new();
        any_of.insert("anyOf".to_string(), serde_json::Value::Array(schemas));
        serde_json::Value::Object(any_of)
    };
    let mut rule = String::from("start:");
    if !prefix.is_empty() {
        rule.push(' ');
        rule.push_str(&prefix);
    }
    rule.push_str(" %json ");
    rule.push_str(&schema.to_string());
    if !suffix.is_empty() {
        rule.push(' ');
        rule.push_str(&suffix);
    }
    rule.push('\n');
    Some(rule)
}

/// Whether the selected tools' names and `parameters` schemas together
/// serialize to at most `budget` bytes, measured without materializing any of
/// them.
///
/// This runs before [`tool_choice_grammar_with`] clones a single schema, and it
/// is the only thing that bounds that clone. Everything the grammar builder
/// does afterwards -- a deep clone of each `parameters` value, a recursive
/// `$ref` rewrite over the clone, and a full re-serialization of the result --
/// is proportional to the tool schemas the request declared, runs on whichever
/// thread the caller is on (a Tokio runtime worker for every HTTP route), and
/// is then discarded by `build_lark_constraint`'s size cap. `tools` is capped
/// by count (`MAX_TOOLS`) and not by size, and the JSON body limit is sized for
/// base64 image payloads, so a single request can otherwise buy hundreds of
/// milliseconds of blocked reactor time and a multiple of the body size in
/// transient allocation for a grammar that was never going to compile.
///
/// `serde_json::to_writer` stops at the first writer error, so an oversized
/// schema costs `budget` bytes of walking rather than its own size. The measure
/// is a cheap lower bound on the finished Lark text, not a replacement for the
/// cap: the wrapper literals and the per-call JSON scaffolding are not counted,
/// and `build_lark_constraint` stays the authority on the final size.
fn tool_schemas_fit(tools: &[&Tool], budget: usize) -> bool {
    use std::io::Write;

    /// A sink that accepts bytes until `budget` is exhausted and then fails.
    struct Budget(usize);

    impl Write for Budget {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(buf.len()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "tool schema exceeds the grammar size budget",
                )
            })?;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut remaining = Budget(budget);
    tools.iter().all(|tool| {
        remaining.write_all(tool.function.name.as_bytes()).is_ok()
            && tool
                .function
                .parameters
                .as_ref()
                .is_none_or(|parameters| serde_json::to_writer(&mut remaining, parameters).is_ok())
    })
}

/// Quote `text` as a Lark string literal.
fn lark_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Id of the token the llguidance trie treats as special for exactly `text`,
/// if the loaded tokenizer has one.
///
/// Mirrors the classification in `toktrie_hf_tokenizers`: an added token is
/// special when the tokenizer flags it so, or when its text is wrapped in
/// `<...>` regardless of the flag (Qwen marks `<tool_call>` as
/// `special: false`, and the trie still hides it behind its marker byte).
/// Only HuggingFace tokenizers reach the grammar path at all, so a
/// SentencePiece or Tiktoken tokenizer answers `None`.
// Used by: routes/chat
pub fn special_token_id(tokenizer: &crate::tokenizer::MlxcelTokenizer, text: &str) -> Option<u32> {
    let hf = tokenizer.hf_tokenizer()?;
    let id = hf.token_to_id(text)?;
    let added = hf.get_added_vocabulary().get_added_tokens_decoder();
    let token = added.get(&id)?;
    (token.special || (text.starts_with('<') && text.ends_with('>'))).then_some(id)
}

/// JSON schema for one forced call: `{"name": <const>, <arguments_key>: <parameters>}`
/// with both keys required and nothing else allowed, in that order (the order
/// llguidance enforces is the schema's own, which matches how the templates
/// teach the call shape).
///
/// `base` is the JSON Pointer path, relative to the finished document's root,
/// at which the tool's `parameters` schema comes to rest. The tool's own
/// root-relative `$ref`s are re-pointed at it by [`relocate_root_refs`].
fn tool_call_schema(tool: &Tool, arguments_key: &str, base: &str) -> serde_json::Value {
    let mut parameters = tool
        .function
        .parameters
        .clone()
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));
    relocate_root_refs(&mut parameters, base);
    let mut name = serde_json::Map::new();
    name.insert(
        "const".to_string(),
        serde_json::Value::String(tool.function.name.clone()),
    );
    let mut properties = serde_json::Map::new();
    properties.insert("name".to_string(), serde_json::Value::Object(name));
    properties.insert(arguments_key.to_string(), parameters);
    let mut schema = serde_json::Map::new();
    schema.insert(
        "type".to_string(),
        serde_json::Value::String("object".to_string()),
    );
    schema.insert(
        "properties".to_string(),
        serde_json::Value::Object(properties),
    );
    schema.insert(
        "required".to_string(),
        serde_json::json!(["name", arguments_key]),
    );
    schema.insert(
        "additionalProperties".to_string(),
        serde_json::Value::Bool(false),
    );
    serde_json::Value::Object(schema)
}

/// Re-point a schema's root-relative `$ref`s after it is embedded at `base`
/// inside a larger document (#1319).
///
/// A tool's `parameters` schema is written as its own document: the shapes
/// that Pydantic, the OpenAI SDK and LangChain emit for a nested model carry
/// a `$defs` block and refer to it as `#/$defs/Name`. Dropped verbatim under
/// `properties/arguments` of the forced-call wrapper, those pointers resolve
/// against the wrapper's root, where no `$defs` exists, and the grammar fails
/// to compile. Rewriting `#/x` to `#/<base>/x` (and a bare `#` self-reference
/// to `#/<base>`) keeps every pointer aimed at the block it was written for,
/// which also keeps two tools' identically-named definitions apart because
/// each branch's pointers stay inside that branch.
///
/// Absolute refs into another document are left alone: they are not ours to
/// rewrite, and the grammar compiler rejects them on its own.
fn relocate_root_refs(value: &mut serde_json::Value, base: &str) {
    match value {
        serde_json::Value::Object(map) => {
            // Only a string `$ref` is a reference; an entry named `$ref`
            // inside a `properties` map describes a property with that name.
            if let Some(serde_json::Value::String(target)) = map.get_mut("$ref") {
                if target == "#" {
                    *target = format!("#/{base}");
                } else if let Some(rest) = target.strip_prefix("#/") {
                    *target = format!("#/{base}/{rest}");
                }
            }
            for nested in map.values_mut() {
                relocate_root_refs(nested, base);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                relocate_root_refs(item, base);
            }
        }
        _ => {}
    }
}

/// Check if tool call parsing should be attempted for this request.
///
/// Returns false when no tools are provided or tool_choice is "none".
// Used by: routes/chat
pub fn should_parse_tool_calls(request: &ChatCompletionRequest) -> bool {
    let Some(ref tools) = request.tools else {
        return false;
    };
    if tools.is_empty() {
        return false;
    }
    if let Some(ref tc) = request.tool_choice
        && tc.is_none()
    {
        return false;
    }
    true
}

/// Convert parsed tool calls to response format, filtering by specific
/// function name when tool_choice selects one.
// Used by: routes/chat
pub fn build_tool_call_responses(
    parsed: &ToolCallParseResult,
    request: &ChatCompletionRequest,
) -> Vec<ToolCallResponse> {
    let specific_fn = request
        .tool_choice
        .as_ref()
        .and_then(|tc| tc.specific_function());

    parsed
        .tool_calls
        .iter()
        .filter(|c| {
            if let Some(fn_name) = specific_fn {
                c.name == fn_name
            } else {
                true
            }
        })
        .map(|c| ToolCallResponse {
            id: generate_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunctionResponse {
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            },
        })
        .collect()
}

#[cfg(test)]
mod tool_choice_grammar_tests {
    use super::*;
    use crate::server::types::request::{
        FunctionDefinition, ToolChoiceFunction, ToolChoiceFunctionName,
    };

    fn tool(name: &str, parameters: Option<serde_json::Value>) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: None,
                parameters,
            },
        }
    }

    fn named(name: &str) -> ToolChoice {
        ToolChoice::Specific(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName {
                name: name.to_string(),
            },
        })
    }

    fn required() -> ToolChoice {
        ToolChoice::Mode("required".to_string())
    }

    fn two_tools() -> Vec<Tool> {
        vec![
            tool(
                "get_time",
                Some(
                    serde_json::json!({"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}),
                ),
            ),
            tool("get_weather", None),
        ]
    }

    /// The JSON schema embedded after `%json` in a grammar line.
    fn embedded_schema(lark: &str) -> serde_json::Value {
        let start = lark.find("%json ").expect("grammar embeds a %json schema") + "%json ".len();
        serde_json::Deserializer::from_str(&lark[start..])
            .into_iter::<serde_json::Value>()
            .next()
            .expect("a JSON value follows %json")
            .expect("the embedded schema is valid JSON")
    }

    #[test]
    fn tool_choice_grammar_hermes_required_is_anyof_over_all_tools() {
        let lark = tool_choice_grammar(ToolCallFormat::Hermes, &two_tools(), &required())
            .expect("Hermes is grammar-capable");
        assert!(
            lark.starts_with(r#"start: "<tool_call>" "\n"? %json "#),
            "{lark}"
        );
        assert!(
            lark.trim_end().ends_with(r#""\n"? "</tool_call>""#),
            "{lark}"
        );

        let schema = embedded_schema(&lark);
        let branches = schema["anyOf"].as_array().expect("required is an anyOf");
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0]["properties"]["name"]["const"], "get_time");
        assert_eq!(
            branches[0]["properties"]["arguments"]["required"],
            serde_json::json!(["tz"]),
            "the tool's own parameters schema types the arguments"
        );
        assert_eq!(branches[1]["properties"]["name"]["const"], "get_weather");
        assert_eq!(
            branches[1]["properties"]["arguments"],
            serde_json::json!({"type": "object"}),
            "a tool without parameters accepts any object"
        );
        for branch in branches {
            assert_eq!(branch["required"], serde_json::json!(["name", "arguments"]));
            assert_eq!(branch["additionalProperties"], false);
        }
    }

    #[test]
    fn tool_choice_grammar_named_pins_name_const() {
        let lark = tool_choice_grammar(ToolCallFormat::Hermes, &two_tools(), &named("get_weather"))
            .expect("named choice on Hermes");
        let schema = embedded_schema(&lark);
        assert!(
            schema.get("anyOf").is_none(),
            "a single tool is not wrapped in anyOf"
        );
        assert_eq!(schema["properties"]["name"]["const"], "get_weather");
        assert!(
            !lark.contains("get_time"),
            "the other tool is unreachable: {lark}"
        );

        // Llama 3 spells the argument object `parameters` and has no wrapper.
        let lark = tool_choice_grammar(ToolCallFormat::Llama3, &two_tools(), &named("get_time"))
            .expect("named choice on Llama 3");
        assert!(lark.starts_with("start: %json "), "{lark}");
        let schema = embedded_schema(&lark);
        assert_eq!(
            schema["required"],
            serde_json::json!(["name", "parameters"])
        );
        assert!(schema["properties"].get("arguments").is_none());

        // Mistral Nemo wraps the object in its bracketed array.
        let lark = tool_choice_grammar(
            ToolCallFormat::MistralNemo,
            &two_tools(),
            &named("get_time"),
        )
        .expect("named choice on Mistral Nemo");
        assert!(
            lark.starts_with(r#"start: "[TOOL_CALLS]" " "? "[" %json "#),
            "{lark}"
        );
        assert!(lark.trim_end().ends_with(r#" "]""#), "{lark}");
    }

    #[test]
    fn tool_choice_grammar_references_special_wrapper_tokens_by_id() {
        // A Qwen-style tokenizer carries `<tool_call>` / `</tool_call>` as
        // added tokens; the trie hides those behind a marker byte, so the
        // grammar must name them by id rather than by bytes.
        let lookup = |text: &str| match text {
            "<tool_call>" => Some(151657),
            "</tool_call>" => Some(151658),
            _ => None,
        };
        let lark =
            tool_choice_grammar_with(ToolCallFormat::Hermes, &two_tools(), &required(), lookup)
                .expect("Hermes is grammar-capable");
        assert!(
            lark.starts_with(r#"start: <[151657]> "\n"? %json "#),
            "{lark}"
        );
        assert!(lark.trim_end().ends_with(r#""\n"? <[151658]>"#), "{lark}");
        assert!(
            !lark.contains("\"<tool_call>\""),
            "no byte spelling of a special token: {lark}"
        );

        // Mistral Nemo's control token is followed by the bracketed array.
        let lookup = |text: &str| (text == "[TOOL_CALLS]").then_some(9);
        let lark = tool_choice_grammar_with(
            ToolCallFormat::MistralNemo,
            &two_tools(),
            &required(),
            lookup,
        )
        .expect("Mistral Nemo is grammar-capable");
        assert!(
            lark.starts_with(r#"start: <[9]> " "? "[" %json "#),
            "{lark}"
        );

        // Without a special token the byte literal stays, and Llama 3 has no
        // wrapper to resolve at all.
        let lark =
            tool_choice_grammar_with(ToolCallFormat::Llama3, &two_tools(), &required(), |_| {
                Some(1)
            })
            .expect("Llama 3 is grammar-capable");
        assert!(lark.starts_with("start: %json "), "{lark}");
    }

    #[test]
    fn special_token_id_follows_the_trie_classification() {
        // The byte-fallback stub declares `<BOS>` / `<EOS>` as special added
        // tokens and `Hello` as a plain vocabulary entry.
        let tokenizer = crate::tokenizer::MlxcelTokenizer::stub_with_byte_fallback();
        assert_eq!(special_token_id(&tokenizer, "<BOS>"), Some(0));
        assert_eq!(special_token_id(&tokenizer, "<EOS>"), Some(1));
        assert_eq!(
            special_token_id(&tokenizer, "Hello"),
            None,
            "plain vocab is not special"
        );
        assert_eq!(
            special_token_id(&tokenizer, "<tool_call>"),
            None,
            "unknown text"
        );
        let bare = crate::tokenizer::MlxcelTokenizer::stub_all_byte_fallback();
        assert_eq!(special_token_id(&bare, "<tool_call>"), None);
    }

    #[test]
    fn tool_choice_grammar_is_none_for_atem_and_gemma4() {
        for format in [
            ToolCallFormat::Atem,
            ToolCallFormat::Gemma4,
            ToolCallFormat::FunctionGemma,
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::Mistral,
            ToolCallFormat::Pythonic,
        ] {
            assert!(
                tool_choice_grammar(format, &two_tools(), &required()).is_none(),
                "{format} has no JSON wire shape to force"
            );
            assert!(tool_choice_grammar(format, &two_tools(), &named("get_time")).is_none());
        }
    }

    /// A forced choice whose declared schemas are larger than the grammar
    /// size cap is answered before anything is cloned or serialized (#1319).
    /// `build_lark_constraint` rejects the finished text anyway, but only on
    /// the blocking pool and only after this builder has deep-cloned every
    /// selected tool's `parameters`, walked the clone to relocate its `$ref`s
    /// and re-serialized the result -- work proportional to a request body
    /// whose limit is sized for base64 images, paid on a Tokio runtime worker
    /// and then thrown away.
    #[test]
    fn tool_choice_grammar_is_none_when_the_declared_schemas_exceed_the_size_cap() {
        let cap = crate::server::structured::MAX_SCHEMA_BYTES;
        let schema = |bytes: usize| {
            Some(serde_json::json!({
                "type": "object",
                "properties": {"note": {"type": "string", "description": "x".repeat(bytes)}}
            }))
        };

        let one_over = vec![tool("get_time", schema(cap + 1))];
        assert!(tool_choice_grammar(ToolCallFormat::Hermes, &one_over, &required()).is_none());
        assert!(
            tool_choice_grammar(ToolCallFormat::Hermes, &one_over, &named("get_time")).is_none()
        );

        // The budget is the sum over the selected tools, not any single one:
        // two schemas at two thirds of the cap each fit alone and not together.
        let two_thirds = vec![
            tool("get_time", schema(cap / 3 * 2)),
            tool("get_weather", schema(cap / 3 * 2)),
        ];
        assert!(tool_choice_grammar(ToolCallFormat::Hermes, &two_thirds, &required()).is_none());
        assert!(
            tool_choice_grammar(ToolCallFormat::Hermes, &two_thirds, &named("get_time")).is_some(),
            "a named choice only pays for the tool it selects"
        );

        // An unbounded name is measured too: it is cloned into the `const`.
        let long_name = "n".repeat(cap + 1);
        let named_tool = vec![tool(&long_name, None)];
        assert!(tool_choice_grammar(ToolCallFormat::Hermes, &named_tool, &required()).is_none());

        // Ordinary schemas are unaffected.
        assert!(tool_choice_grammar(ToolCallFormat::Hermes, &two_tools(), &required()).is_some());
    }

    #[test]
    fn tool_choice_grammar_is_none_without_a_forced_choice_or_a_matching_tool() {
        for mode in ["auto", "none"] {
            let choice = ToolChoice::Mode(mode.to_string());
            assert!(tool_choice_grammar(ToolCallFormat::Hermes, &two_tools(), &choice).is_none());
        }
        assert!(tool_choice_grammar(ToolCallFormat::Hermes, &[], &required()).is_none());
        assert!(
            tool_choice_grammar(ToolCallFormat::Hermes, &two_tools(), &named("get_stock"))
                .is_none(),
            "an undeclared name forces nothing; validation rejects it upstream"
        );
    }
}
