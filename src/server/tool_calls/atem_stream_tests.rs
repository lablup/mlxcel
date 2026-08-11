use super::{FilterOutput, StreamFilter};
use crate::server::tool_calls::{ToolCallFormat, parse_tool_calls};

#[test]
fn muse_recipient_channels_split_reasoning_and_content() {
    let mut filter = StreamFilter::new();
    let raw = concat!(
        "to=self<|message|>The image is orange.<|eom|>",
        "<|start|>assistant to=user<|message|>orange<|eot|>"
    );
    let output = filter.feed(raw);
    let tail = filter.flush();
    assert_eq!(output.reasoning.as_deref(), Some("The image is orange."));
    assert_eq!(output.content.as_deref(), Some("orange"));
    assert_eq!(tail.content, None);
    assert_eq!(tail.reasoning, None);
}

#[test]
fn muse_prompt_primed_leading_space_routes_reasoning() {
    let mut filter = StreamFilter::new();
    let output = filter.feed(concat!(
        " to=self<|message|>Think first.<|eom|>",
        "<|start|>assistant to=user<|message|>Done<|eot|>"
    ));
    assert_eq!(output.reasoning.as_deref(), Some("Think first."));
    assert_eq!(output.content.as_deref(), Some("Done"));
    assert_eq!(filter.flush(), FilterOutput::default());
}

#[test]
fn muse_dynamic_tool_recipient_header_is_suppressed() {
    let mut filter = StreamFilter::new();
    let fragments = [
        "to=self<|message|>Use tool.<|eom|><|start|>assistant to=get_",
        "weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\">",
        "<atem:parameter name=\"city\">Seoul</atem:parameter></atem:invoke>",
        "</atem:function_calls><|eot|>",
    ];
    let mut content = String::new();
    let mut reasoning = String::new();
    for fragment in fragments {
        let output = filter.feed(fragment);
        content.push_str(output.content.as_deref().unwrap_or(""));
        reasoning.push_str(output.reasoning.as_deref().unwrap_or(""));
    }
    let tail = filter.flush();
    content.push_str(tail.content.as_deref().unwrap_or(""));
    reasoning.push_str(tail.reasoning.as_deref().unwrap_or(""));
    assert_eq!(content, "");
    assert_eq!(reasoning, "Use tool.");
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

fn atem_block(body: &str) -> String {
    format!("<atem:function_calls>\n{body}</atem:function_calls>")
}

fn collect_visible(chunks: &[&str]) -> (String, usize, usize) {
    let mut filter = StreamFilter::new();
    let mut visible = String::new();
    let mut suppressed = 0usize;
    let mut consumed = 0usize;
    for chunk in chunks {
        let out = filter.feed(chunk);
        if let Some(content) = out.content {
            visible.push_str(&content);
        }
        suppressed += out.suppressed_positions;
        consumed += out.consumed_positions;
    }
    let out = filter.flush();
    if let Some(content) = out.content {
        visible.push_str(&content);
    }
    suppressed += out.suppressed_positions;
    consumed += out.consumed_positions;
    (visible, suppressed, consumed)
}

#[test]
fn atem_single_feed_suppresses_and_resumes_after_wrapper() {
    let block = atem_block(&atem_call("search", &[("q", "rust")]));
    let (visible, suppressed, consumed) = collect_visible(&[&format!("before {block} after")]);
    assert_eq!(visible, "before  after");
    assert!(consumed >= suppressed);
}

#[test]
fn atem_all_fixed_delimiters_split_at_every_byte_boundary() {
    let block = atem_block(&atem_call(
        "set_profile",
        &[
            ("active", "true"),
            ("meta", r#"{"nested":{"ok":true},"count":2}"#),
        ],
    ));
    let full = format!("A{block}B");
    let mut filter = StreamFilter::new();
    let mut visible = String::new();
    let mut suppressed = 0usize;
    for ch in full.chars() {
        let fragment = ch.to_string();
        let out = filter.feed(&fragment);
        if let Some(content) = out.content {
            visible.push_str(&content);
        }
        suppressed += out.suppressed_positions;
    }
    if let Some(content) = filter.flush().content {
        visible.push_str(&content);
    }
    assert_eq!(visible, "AB");
    assert!(
        !visible.contains("atem") && !visible.contains("invoke") && !visible.contains("parameter")
    );
    assert!(suppressed >= 6);
}

#[test]
fn atem_dynamic_openers_split_across_chunk_boundaries() {
    let chunks = [
        "prefix ",
        "<atem:function_calls>\n<atem:in",
        "voke name=\"search\">\n<atem:para",
        "meter name=\"q\">rust</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
        " suffix",
    ];
    let (visible, _, _) = collect_visible(&chunks);
    assert_eq!(visible, "prefix  suffix");
}

#[test]
fn atem_split_delimiters_preserve_suppressed_positions() {
    let mut filter = StreamFilter::new();
    assert_eq!(filter.feed("<atem:function").suppressed_positions, 0);
    assert_eq!(filter.feed("_calls>").suppressed_positions, 2);
    assert_eq!(filter.feed("<atem:in").suppressed_positions, 0);
    assert_eq!(filter.feed("voke name=\"search\">").suppressed_positions, 2);
    assert_eq!(filter.feed("<atem:para").suppressed_positions, 0);
    assert_eq!(filter.feed("meter name=\"q\">").suppressed_positions, 2);
    assert_eq!(filter.feed("rust").suppressed_positions, 0);
    assert_eq!(filter.feed("</atem:para").suppressed_positions, 0);
    assert_eq!(filter.feed("meter>").suppressed_positions, 2);
    assert_eq!(filter.feed("</atem:in").suppressed_positions, 0);
    assert_eq!(filter.feed("voke>").suppressed_positions, 2);
    assert_eq!(filter.feed("</atem:function").suppressed_positions, 0);
    assert_eq!(filter.feed("_calls>").suppressed_positions, 2);
    assert_eq!(filter.feed("done").content.as_deref(), Some("done"));
}

#[test]
fn atem_parallel_calls_parse_order_from_accumulated_stream() {
    let raw = atem_block(&format!(
        "{}{}",
        atem_call("search", &[("q", "rust")]),
        atem_call("calc", &[("expr", "2+2")])
    ));
    let (visible, _, _) = collect_visible(&[&raw]);
    assert_eq!(visible, "");

    let parsed = parse_tool_calls(&raw, None);
    assert_eq!(parsed.format, Some(ToolCallFormat::Atem));
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].name, "search");
    assert_eq!(parsed.tool_calls[1].name, "calc");
}

#[test]
fn atem_nested_typed_values_do_not_leak_when_split_one_byte() {
    let raw = format!(
        "pre{}post",
        atem_block(&atem_call(
            "set_profile",
            &[
                ("active", "true"),
                ("tags", r#"["math","notes"]"#),
                ("meta", r#"{"nested":{"ok":true}}"#),
            ],
        ))
    );
    let chunks: Vec<String> = raw.chars().map(|c| c.to_string()).collect();
    let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    let (visible, _, _) = collect_visible(&refs);
    assert_eq!(visible, "prepost");
}

#[test]
fn atem_stray_invoke_and_parameter_tags_are_suppressed() {
    let (visible, _, _) = collect_visible(&[
        "A",
        "<atem:invoke name=\"f\">hidden</atem:invoke>",
        "B",
        "<atem:parameter name=\"x\">secret</atem:parameter>",
        "C",
    ]);
    assert_eq!(visible, "ABC");
}

#[test]
fn atem_payload_delimiter_lookalikes_do_not_exit_outer_block() {
    let raw = atem_block(&atem_call(
        "write",
        &[(
            "text",
            "before </tool_call> </atem:parameter> </atem:invoke> after <think>hidden</think>",
        )],
    ));
    let (visible, _, _) = collect_visible(&[&format!("A{raw}B")]);
    assert_eq!(visible, "AB");
}

#[test]
fn atem_flush_handles_complete_truncated_malformed_and_wrapper_only() {
    let complete = collect_visible(&[&atem_block(&atem_call("f", &[("x", "1")]))]);
    assert_eq!(complete.0, "");

    let truncated = collect_visible(&["A<atem:function_calls><atem:invoke name=\"f\">hidden"]);
    assert_eq!(truncated.0, "A");

    let malformed = collect_visible(&["A<atem:function_calls><atem:invoke name=\"f\""]);
    assert_eq!(malformed.0, "A");

    let wrapper_only = collect_visible(&["A<atem:function_calls></atem:function_calls>B"]);
    assert_eq!(wrapper_only.0, "AB");
}

#[test]
fn atem_huge_malformed_input_drains_without_unbounded_visible_buffer() {
    let huge = "x".repeat(128 * 1024);
    let (visible, _, consumed) = collect_visible(&["A<atem:invoke name=\"f\">", &huge]);
    assert_eq!(visible, "A");
    assert!(consumed > 0);
}

#[test]
fn atem_exact_eof_ambiguity_is_suppressed_safely() {
    let mut filter = StreamFilter::new();
    let first = filter.feed("text <atem:invoke");
    assert_eq!(first.content.as_deref(), Some("text "));
    let tail = filter.flush();
    assert_eq!(tail, FilterOutput::default());

    let (visible, _, _) = collect_visible(&["text <atem:invocation"]);
    assert_eq!(visible, "text <atem:invocation");
}
