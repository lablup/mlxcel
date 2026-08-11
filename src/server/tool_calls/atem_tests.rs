use std::str::FromStr as _;

use serde_json::Value;

use super::*;
use crate::server::tool_calls::{ToolCallFormat, clean_structural_tokens, parse_tool_calls};
use crate::server::types::request::{FunctionDefinition, Tool};

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

fn atem_call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<atem:invoke name=\"{name}\">\n");
    for (key, value) in params {
        s.push_str(&format!(
            "<atem:parameter name=\"{key}\">{value}</atem:parameter>\n"
        ));
    }
    s.push_str("</atem:invoke>\n");
    s
}

#[test]
fn muse_recipient_envelope_routes_reasoning_and_hides_tool_header() {
    let raw = concat!(
        "to=self<|message|>We must use the weather tool.<|eom|>",
        "<|start|>assistant to=get_weather<|message|>",
        "<atem:function_calls><atem:invoke name=\"get_weather\">",
        "<atem:parameter name=\"city\">Seoul</atem:parameter>",
        "</atem:invoke></atem:function_calls><|eot|>"
    );

    let parsed = try_atem(raw).expect("Muse ATEM call");
    assert_eq!(parsed.content, "");
    assert_eq!(
        parsed.reasoning_content.as_deref(),
        Some("We must use the weather tool.")
    );
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    assert_eq!(parsed.tool_calls[0].arguments, r#"{"city":"Seoul"}"#);
}

#[test]
fn muse_recipient_envelope_keeps_user_answer_content() {
    let raw = concat!(
        "to=self<|message|>I know the answer.<|eom|>",
        "<|start|>assistant to=user<|message|>Done.<|eot|>",
        "<atem:function_calls><atem:invoke name=\"noop\">",
        "</atem:invoke></atem:function_calls>"
    );

    let parsed = try_atem(raw).expect("Muse ATEM call");
    assert_eq!(parsed.content, "Done.");
    assert_eq!(
        parsed.reasoning_content.as_deref(),
        Some("I know the answer.")
    );
}

fn atem_block(body: &str) -> String {
    format!("<atem:function_calls>\n{body}</atem:function_calls>")
}

fn parsed_args(call: &ParsedToolCall) -> Value {
    match serde_json::from_str(&call.arguments) {
        Ok(value) => value,
        Err(error) => panic!("arguments must be JSON: {error}: {}", call.arguments),
    }
}

#[test]
fn atem_format_display_and_parse_identity() {
    assert_eq!(ToolCallFormat::Atem.as_str(), "atem");
    assert_eq!(ToolCallFormat::Atem.to_string(), "atem");
    assert_eq!(ToolCallFormat::from_str("atem"), Ok(ToolCallFormat::Atem));
}

#[test]
fn atem_parallel_calls_preserve_order() {
    let output = atem_block(&format!(
        "{}{}",
        atem_call("search", &[("q", "rust")]),
        atem_call("calc", &[("expr", "2+2")])
    ));

    let result = try_atem(&output).unwrap_or_else(|| panic!("ATEM parse failed: {output}"));
    assert_eq!(result.format, Some(ToolCallFormat::Atem));
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0].name, "search");
    assert_eq!(result.tool_calls[1].name, "calc");
    assert_eq!(parsed_args(&result.tool_calls[0])["q"], "rust");
    assert_eq!(parsed_args(&result.tool_calls[1])["expr"], "2+2");
}

#[test]
fn atem_nested_and_typed_values_follow_template_contract() {
    let output = atem_block(&atem_call(
        "set_profile",
        &[
            ("name", " Ada Lovelace "),
            ("active", "true"),
            ("score", "42"),
            ("ratio", "0.25"),
            ("nil", "null"),
            ("tags", r#"["math","notes"]"#),
            ("meta", r#"{"nested":{"ok":true},"count":2}"#),
        ],
    ));

    let result = try_atem(&output).unwrap_or_else(|| panic!("ATEM parse failed: {output}"));
    let args = parsed_args(&result.tool_calls[0]);
    assert_eq!(args["name"], " Ada Lovelace ");
    assert_eq!(args["active"], true);
    assert_eq!(args["score"], 42);
    assert_eq!(args["ratio"], 0.25);
    assert!(args["nil"].is_null());
    assert_eq!(args["tags"][1], "notes");
    assert_eq!(args["meta"]["nested"]["ok"], true);
}

#[test]
fn atem_allowlist_filters_and_normalizes_namespace() {
    let output = atem_block(&format!(
        "{}{}",
        atem_call("functions.weather", &[("city", "Seoul")]),
        atem_call("forbidden", &[("path", "/etc/passwd")])
    ));
    let tools = vec![make_tool("weather")];

    let result = parse_tool_calls(&output, Some(&tools));
    assert_eq!(result.format, Some(ToolCallFormat::Atem));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "weather");
    assert_eq!(parsed_args(&result.tool_calls[0])["city"], "Seoul");
}

#[test]
fn atem_duplicate_parameters_last_value_wins() {
    let output = atem_block(&atem_call("set", &[("mode", "first"), ("mode", "second")]));

    let result = try_atem(&output).unwrap_or_else(|| panic!("ATEM parse failed: {output}"));
    assert_eq!(parsed_args(&result.tool_calls[0])["mode"], "second");
}

#[test]
fn atem_malformed_truncated_and_unknown_tags_do_not_panic() {
    let truncated = concat!(
        "<atem:function_calls><atem:invoke name=\"run\">",
        "<atem:unknown>ignored</atem:unknown>",
        "<atem:parameter name=\"script\">echo hi"
    );
    let result = parse_tool_calls(truncated, None);
    assert_eq!(result.format, Some(ToolCallFormat::Atem));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(parsed_args(&result.tool_calls[0])["script"], "echo hi");

    let malformed = "<atem:function_calls><atem:invoke name=\"run\"";
    let result = parse_tool_calls(malformed, None);
    assert!(!result.has_tool_calls());
    assert!(!result.content.contains("<atem:"));

    let bad_attr =
        "<atem:function_calls><atem:invoke badname=\"run\"></atem:invoke></atem:function_calls>";
    let result = parse_tool_calls(bad_attr, None);
    assert!(!result.has_tool_calls());
    assert_eq!(result.content, "");
}

#[test]
fn atem_wrapper_only_and_unknown_tools_are_stripped_from_content() {
    let wrapper_only = "visible<atem:function_calls></atem:function_calls>";
    let result = parse_tool_calls(wrapper_only, None);
    assert!(!result.has_tool_calls());
    assert_eq!(result.content, "visible");

    let unknown = atem_block(&atem_call("unknown", &[("x", "1")]));
    let tools = vec![make_tool("known")];
    let result = parse_tool_calls(&format!("before {unknown} after"), Some(&tools));
    assert!(!result.has_tool_calls());
    assert_eq!(result.content, "before  after");
    assert_eq!(clean_structural_tokens(&unknown), "");
}

#[test]
fn atem_trailing_content_is_visible_without_wrappers() {
    let output = format!(
        "before {} after",
        atem_block(&atem_call("search", &[("q", "rust")]))
    );

    let result = parse_tool_calls(&output, None);
    assert_eq!(result.format, Some(ToolCallFormat::Atem));
    assert_eq!(result.content, "before  after");
    assert!(!result.content.contains("<atem:"));
}

#[test]
fn atem_bounds_huge_call_parameter_and_argument_amplification() {
    let mut invokes = String::new();
    for i in 0..(ATEM_MAX_CALLS + 8) {
        invokes.push_str(&atem_call(&format!("tool_{i}"), &[("x", "1")]));
    }
    let result =
        try_atem(&atem_block(&invokes)).unwrap_or_else(|| panic!("ATEM call-cap parse failed"));
    assert_eq!(result.tool_calls.len(), ATEM_MAX_CALLS);

    let mut params = Vec::new();
    for i in 0..(ATEM_MAX_PARAMS_PER_CALL + 8) {
        params.push((format!("p{i}"), "1".to_string()));
    }
    let param_refs: Vec<(&str, &str)> = params
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let result = try_atem(&atem_block(&atem_call("many", &param_refs)))
        .unwrap_or_else(|| panic!("ATEM param-cap parse failed"));
    let args = parsed_args(&result.tool_calls[0]);
    assert_eq!(
        args.as_object().map(|m| m.len()),
        Some(ATEM_MAX_PARAMS_PER_CALL)
    );

    let huge = "x".repeat(ATEM_MAX_ARGUMENT_BYTES + 1);
    let result = try_atem(&atem_block(&atem_call("huge", &[("blob", &huge)])))
        .unwrap_or_else(|| panic!("ATEM huge-arg parse failed"));
    assert_eq!(parsed_args(&result.tool_calls[0]), serde_json::json!({}));
}

#[test]
fn atem_precedes_minimax_m2_when_both_are_present() {
    let atem = atem_block(&atem_call("muse_tool", &[("x", "1")]));
    let minimax = r#"<invoke name="minimax_tool"><parameter name="y">2</parameter></invoke>"#;
    let tools = vec![make_tool("muse_tool"), make_tool("minimax_tool")];

    let result = parse_tool_calls(&format!("{atem}{minimax}"), Some(&tools));
    assert_eq!(result.format, Some(ToolCallFormat::Atem));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "muse_tool");
}
