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

//! Streamed vs non-streamed parity (#2394).
//!
//! Every case feeds one model output through [`StreamFilter`] under many
//! chunkings (whole, one character per `feed()`, fixed widths, and every
//! two-piece split, which puts a boundary inside each marker) and through the
//! non-streamed cleanup the routes use, then asserts that both produce the
//! same visible content, the same reasoning, and the same tool calls.

use super::StreamFilter;
use crate::server::tool_calls::{
    ParsedToolCall, clean_structural_tokens, parse_tool_calls, strip_answer_wrapper,
};
use crate::server::types::request::{FunctionDefinition, Tool};

fn tool(name: &str) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: None,
        },
    }
}

/// Every chunking the parity tests feed: whole, per character, fixed widths,
/// and each two-piece split at a character boundary.
fn chunkings(raw: &str) -> Vec<Vec<&str>> {
    let bounds: Vec<usize> = raw
        .char_indices()
        .map(|(i, _)| i)
        .chain([raw.len()])
        .collect();
    let mut out = vec![vec![raw]];
    for width in [1usize, 2, 3, 5, 7] {
        let mut pieces = Vec::new();
        let mut i = 0;
        while i + 1 < bounds.len() {
            let j = (i + width).min(bounds.len() - 1);
            pieces.push(&raw[bounds[i]..bounds[j]]);
            i = j;
        }
        out.push(pieces);
    }
    for &split in &bounds[1..bounds.len() - 1] {
        out.push(vec![&raw[..split], &raw[split..]]);
    }
    out
}

/// What a streaming client receives: concatenated `delta.content` and
/// `delta.reasoning_content`, plus the tool calls the route parses from the
/// accumulated raw text at end of stream (`parse_tools` mirrors
/// `should_parse_tool_calls`).
fn streamed(
    pieces: &[&str],
    tools: Option<&[Tool]>,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    streamed_with(pieces, tools, false)
}

/// [`streamed`] for a model whose tokenizer defines the GLM-4.1V-Thinking
/// `<answer>` wrapper when `answer_wrapper` is set, as the routes configure it.
fn streamed_with(
    pieces: &[&str],
    tools: Option<&[Tool]>,
    answer_wrapper: bool,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    let mut filter = StreamFilter::new().with_answer_wrapper(answer_wrapper);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut accumulated = String::new();
    // Logprob bookkeeping: every fed position is reported exactly once, as
    // emitted or as suppressed, by some `feed()` or the final `flush()`. Held
    // whitespace and trimmed newlines defer or suppress positions; they must
    // never lose or double-count one.
    let mut consumed = 0usize;
    for piece in pieces {
        accumulated.push_str(piece);
        let out = filter.feed(piece);
        assert!(
            out.consumed_positions >= out.suppressed_positions,
            "suppressed exceeds consumed for chunking {pieces:?}"
        );
        consumed += out.consumed_positions;
        content.push_str(out.content.as_deref().unwrap_or(""));
        reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
    }
    let out = filter.flush();
    assert!(
        out.consumed_positions >= out.suppressed_positions,
        "suppressed exceeds consumed at flush for chunking {pieces:?}"
    );
    consumed += out.consumed_positions;
    assert_eq!(
        consumed,
        pieces.len(),
        "positions not reported exactly once for chunking {pieces:?}"
    );
    content.push_str(out.content.as_deref().unwrap_or(""));
    reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
    let calls = tools.map(|tools| parse_tool_calls(&accumulated, Some(tools)).tool_calls);
    (content, reasoning, calls)
}

/// What a non-streaming client receives for the same text: the route's
/// `clean_structural_tokens` on a no-tools request, `parse_tool_calls`
/// otherwise. Reasoning is the parser's own (Harmony) when it has one, else the
/// whole-text `StreamFilter` split the route uses (`extract_reasoning_content`).
fn non_streamed(
    raw: &str,
    tools: Option<&[Tool]>,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    non_streamed_with(raw, tools, false)
}

/// [`non_streamed`] with the routes' `strip_answer_wrapper` post-pass applied
/// when `answer_wrapper` is set.
fn non_streamed_with(
    raw: &str,
    tools: Option<&[Tool]>,
    answer_wrapper: bool,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    let (content, reasoning, calls) = non_streamed_plain(raw, tools);
    let content = if answer_wrapper {
        strip_answer_wrapper(&content)
    } else {
        content
    };
    (content, reasoning, calls)
}

fn non_streamed_plain(
    raw: &str,
    tools: Option<&[Tool]>,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    let filter_reasoning = {
        let mut filter = StreamFilter::new();
        let mut reasoning = filter.feed(raw).reasoning.unwrap_or_default();
        reasoning.push_str(filter.flush().reasoning.as_deref().unwrap_or(""));
        reasoning
    };
    match tools {
        None => (clean_structural_tokens(raw), filter_reasoning, None),
        Some(tools) => {
            let parsed = parse_tool_calls(raw, Some(tools));
            let reasoning = parsed.reasoning_content.clone().unwrap_or(filter_reasoning);
            (parsed.content, reasoning, Some(parsed.tool_calls))
        }
    }
}

/// Assert streamed == non-streamed under every chunking, and return the shared
/// result for case-specific checks.
fn assert_parity(
    raw: &str,
    tools: Option<&[Tool]>,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    assert_parity_with(raw, tools, false)
}

/// [`assert_parity`] for a model with (`answer_wrapper`) or without the
/// GLM-4.1V-Thinking `<answer>` tokens.
fn assert_parity_with(
    raw: &str,
    tools: Option<&[Tool]>,
    answer_wrapper: bool,
) -> (String, String, Option<Vec<ParsedToolCall>>) {
    let expected = non_streamed_with(raw, tools, answer_wrapper);
    for pieces in chunkings(raw) {
        let got = streamed_with(&pieces, tools, answer_wrapper);
        assert_eq!(got.0, expected.0, "content differs for chunking {pieces:?}");
        assert_eq!(
            got.1, expected.1,
            "reasoning differs for chunking {pieces:?}"
        );
        assert_eq!(
            got.2, expected.2,
            "tool calls differ for chunking {pieces:?}"
        );
    }
    expected
}

// -- 1. Whitespace between two tool calls --

const TWO_CALLS: &str = concat!(
    r#"<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>"#,
    "\n",
    r#"<tool_call>{"name": "g", "arguments": {}}</tool_call>"#
);

#[test]
fn whitespace_between_two_tool_calls_is_not_content() {
    let tools = [tool("f"), tool("g")];
    let (content, _, calls) = assert_parity(TWO_CALLS, Some(&tools));
    assert_eq!(content, "");
    assert_eq!(calls.map(|c| c.len()), Some(2));
    // The no-tools request suppresses both blocks as well.
    let (content, _, _) = assert_parity(TWO_CALLS, None);
    assert_eq!(content, "");
}

#[test]
fn whitespace_after_the_last_tool_call_is_dropped_at_flush() {
    let raw = concat!(
        r#"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#,
        "\n\n"
    );
    for pieces in chunkings(raw) {
        assert_eq!(streamed(&pieces, None).0, "", "chunking {pieces:?}");
    }
}

#[test]
fn whitespace_between_a_call_and_visible_text_is_kept() {
    let raw = concat!(
        "A",
        r#"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#,
        "\nB"
    );
    for pieces in chunkings(raw) {
        assert_eq!(streamed(&pieces, None).0, "A\nB", "chunking {pieces:?}");
    }
    let mut filter = StreamFilter::new();
    let first = filter.feed(concat!(
        "A",
        r#"<tool_call>{"name": "f", "arguments": {}}</tool_call>"#
    ));
    assert_eq!(first.content.as_deref(), Some("A"));
    assert_eq!(filter.feed("\n").content, None, "whitespace is held");
    assert_eq!(filter.feed("B").content.as_deref(), Some("\nB"));
}

// -- 2. Newlines after `</think>` in a later chunk --

#[test]
fn newlines_after_think_close_in_a_later_chunk_are_trimmed() {
    let mut filter = StreamFilter::new();
    let mut content = String::new();
    for piece in ["<think>r</think>", "\n\n", "Answer"] {
        content.push_str(filter.feed(piece).content.as_deref().unwrap_or(""));
    }
    content.push_str(filter.flush().content.as_deref().unwrap_or(""));
    assert_eq!(content, "Answer");

    let (content, reasoning, _) = assert_parity("<think>r</think>\n\nAnswer", None);
    assert_eq!(content, "Answer");
    assert_eq!(reasoning, "r");
}

#[test]
fn trimmed_newlines_are_reported_as_the_post_thinking_gap() {
    // `--reasoning-format none` re-emits these after the close marker (#1470).
    let mut filter = StreamFilter::new();
    let close = filter.feed("<think>r</think>");
    assert_eq!(close.thinking_close, Some("</think>"));
    let gap = filter.feed("\n\n");
    assert_eq!(gap.content, None);
    assert_eq!(gap.post_thinking_gap.as_deref(), Some("\n\n"));
    assert_eq!(
        gap.suppressed_positions, 1,
        "the newline token is a placeholder position"
    );
    let answer = filter.feed("Answer");
    assert_eq!(answer.content.as_deref(), Some("Answer"));
    assert_eq!(answer.post_thinking_gap, None);
}

// -- 3. Unclosed call at end of stream, tool markup on a no-tools request --

#[test]
fn unclosed_tool_call_at_end_of_stream_matches_non_streamed() {
    let raw = r#"Let me check.<tool_call>{"name":"f""#;
    let (content, _, _) = assert_parity(raw, None);
    assert_eq!(content, "Let me check.");
    let tools = [tool("f")];
    let (content, _, calls) = assert_parity(raw, Some(&tools));
    assert_eq!(content, "Let me check.");
    assert_eq!(calls, Some(Vec::new()));
}

#[test]
fn tool_markup_on_a_no_tools_request_matches_non_streamed() {
    let raw = concat!(
        "Sure.",
        r#"<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>"#
    );
    let (content, _, _) = assert_parity(raw, None);
    assert_eq!(content, "Sure.");
}

// -- 4. Output markers: GLM-4.xV boxes, Aya Vision response wrappers --

#[test]
fn glm_box_markers_are_stripped_streamed_and_non_streamed() {
    let (content, _, _) = assert_parity("<|begin_of_box|>42<|end_of_box|>", None);
    assert_eq!(content, "42");
    // GLM-4.1V-Thinking: reasoning, then the boxed answer.
    let (content, reasoning, _) = assert_parity(
        "<think>six times seven</think><|begin_of_box|>42<|end_of_box|>",
        None,
    );
    assert_eq!(content, "42");
    assert_eq!(reasoning, "six times seven");
}

/// Recorded from models/mlx/glm-4.1v-9b-thinking-4bit on main: the answer
/// wrapper and the box both reached `content` (the stop token ends the
/// generation before `</answer>`).
const GLM_THINKING_REPLY: &str = "<think>Count them.</think><answer>There are <|begin_of_box|>3<|end_of_box|> shapes in the image.";

#[test]
fn glm_answer_wrapper_is_stripped_when_the_tokenizer_defines_it() {
    let (content, reasoning, _) = assert_parity_with(GLM_THINKING_REPLY, None, true);
    assert_eq!(content, "There are 3 shapes in the image.");
    assert_eq!(reasoning, "Count them.");
    let (content, _, _) = assert_parity_with("<answer>42</answer>", None, true);
    assert_eq!(content, "42");
    let tools = [tool("f")];
    let (content, _, _) = assert_parity_with("<answer>42</answer>", Some(&tools), true);
    assert_eq!(content, "42");
    // Reasoning keeps the tags on both paths: only the answer is unwrapped.
    let (content, reasoning, _) = assert_parity_with(
        "<think>maybe <answer>41</answer></think><answer>42</answer>",
        None,
        true,
    );
    assert_eq!(content, "42");
    assert_eq!(reasoning, "maybe <answer>41</answer>");
}

#[test]
fn answer_tags_are_ordinary_text_for_other_models() {
    // Prompts and eval harnesses ask models to wrap a final answer in
    // `<answer>` tags; a model whose tokenizer does not define them keeps them.
    let (content, _, _) = assert_parity("<answer>42</answer>", None);
    assert_eq!(content, "<answer>42</answer>");
    let tools = [tool("f")];
    let (content, _, _) = assert_parity("The result: <answer>42</answer>", Some(&tools));
    assert_eq!(content, "The result: <answer>42</answer>");
    let (content, reasoning, _) = assert_parity("<think>r</think><answer>42</answer>", None);
    assert_eq!(content, "<answer>42</answer>");
    assert_eq!(reasoning, "r");
}

#[test]
fn aya_response_markers_are_stripped_streamed_and_non_streamed() {
    let (content, _, _) = assert_parity("<|START_RESPONSE|>Hi<|END_RESPONSE|>", None);
    assert_eq!(content, "Hi");
    let mut filter = StreamFilter::new();
    let mut content = String::new();
    for piece in ["<|START_RES", "PONSE|>Hi<|END_", "RESPONSE|>"] {
        content.push_str(filter.feed(piece).content.as_deref().unwrap_or(""));
    }
    content.push_str(filter.flush().content.as_deref().unwrap_or(""));
    assert_eq!(content, "Hi");
}

// -- 5. gpt-oss Harmony channels --

/// Recorded from `mlxcel-server -m models/mlx/gpt-oss-20b-mxfp4` on main
/// (#2394 runtime check): the raw `delta.content` of a no-tools streamed
/// request, temperature 0.
const HARMONY_NO_TOOLS: &str = concat!(
    "<|channel|>analysis<|message|>The user asks: \"What is 17 times 3? Answer briefly.\" ",
    "So answer: 51.<|end|><|start|>assistant<|channel|>final<|message|>51"
);

/// Same server, a `get_weather` tool request. `<|call|>` is a stop token, so
/// the generation ends right after the arguments.
const HARMONY_TOOL_CALL: &str = concat!(
    "<|channel|>analysis<|message|>The user asks: \"What is the weather in Paris right now? ",
    "Use the tool.\" We need to call the get_weather function with city \"Paris\". ",
    "Then respond with the result.<|end|><|start|>assistant<|channel|>commentary ",
    "to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}"
);

#[test]
fn harmony_no_tools_transcript_streams_like_try_harmony() {
    let tools = [tool("get_weather")];
    let (content, reasoning, calls) = assert_parity(HARMONY_NO_TOOLS, Some(&tools));
    assert_eq!(content, "51");
    assert_eq!(
        reasoning,
        "The user asks: \"What is 17 times 3? Answer briefly.\" So answer: 51."
    );
    assert_eq!(calls, Some(Vec::new()));
    // A request without `tools` takes `clean_structural_tokens`, which must
    // route Harmony the same way.
    let (content, reasoning, _) = assert_parity(HARMONY_NO_TOOLS, None);
    assert_eq!(content, "51");
    assert!(reasoning.ends_with("So answer: 51."), "{reasoning:?}");
}

#[test]
fn harmony_tool_call_transcript_streams_like_try_harmony() {
    let tools = [tool("get_weather")];
    let (content, reasoning, calls) = assert_parity(HARMONY_TOOL_CALL, Some(&tools));
    assert_eq!(content, "");
    assert!(reasoning.starts_with("The user asks:"), "{reasoning:?}");
    let calls = calls.unwrap_or_default();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn harmony_terminators_do_not_strip_outside_harmony_output() {
    // `<|end|>` is also Phi-3's end-of-turn token; outside a Harmony message it
    // must not change what the filter emits.
    let mut filter = StreamFilter::new();
    let mut content = filter.feed("a<|end|>b").content.unwrap_or_default();
    content.push_str(filter.flush().content.as_deref().unwrap_or(""));
    assert_eq!(content, "a<|end|>b");
}

#[test]
fn literal_channel_marker_outside_harmony_output_is_kept() {
    // A non-Harmony reply that mentions the marker in plain text never reaches
    // `<|message|>`, so `try_harmony` declines and the non-streamed content
    // keeps the text. The stream must not strip it (and everything after it)
    // as an unterminated Harmony header at end of stream.
    let raw = "gpt-oss opens each message with <|channel|> and a channel name.";
    let (content, _, _) = assert_parity(raw, None);
    assert_eq!(content, raw);
}

#[test]
fn truncated_header_after_harmony_output_is_stripped() {
    // Generation cut off inside the second header: `try_harmony` stops at a
    // header with no `<|message|>`, so neither path shows its text.
    let raw = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>fin";
    let (content, reasoning, _) = assert_parity(raw, None);
    assert_eq!(content, "");
    assert_eq!(reasoning, "Think.");
}

// -- 6. Bounded header hold and Harmony isolation (#2394 security review) --

/// Feed `raw` in pieces of `width` characters, asserting after every `feed()`
/// that the filter buffers no more than one maximal pending header plus the
/// partial-match window. Returns the content streamed before `flush()` and the
/// full streamed content.
fn feed_bounded(raw: &str, width: usize) -> (String, String) {
    let mut filter = StreamFilter::new();
    let bound = super::MAX_HEADER_HOLD
        + super::HARMONY_ASSISTANT_CHANNEL.len()
        + filter.max_delim_len
        + width * 4;
    let chars: Vec<char> = raw.chars().collect();
    let mut live = String::new();
    for piece in chars.chunks(width) {
        let piece: String = piece.iter().collect();
        let out = filter.feed(&piece);
        live.push_str(out.content.as_deref().unwrap_or(""));
        assert!(
            filter.buffer.len() <= bound,
            "buffer grew to {} bytes (bound {bound}) at width {width}",
            filter.buffer.len()
        );
    }
    let mut all = live.clone();
    all.push_str(filter.flush().content.as_deref().unwrap_or(""));
    (live, all)
}

fn filler(words: usize) -> String {
    let mut text = "word ".repeat(words);
    text.push_str("end.");
    text
}

#[test]
fn literal_channel_marker_does_not_hold_the_rest_of_the_stream() {
    // A non-Harmony reply that mentions `<|channel|>` must keep streaming: the
    // marker is held only while it could still be a header, not until the end
    // of generation (which also made every `feed()` rescan the whole reply).
    let raw = format!("Harmony uses <|channel|> markers. {}", filler(2000));
    let (expected, _, _) = non_streamed(&raw, None);
    assert_eq!(expected, raw);
    for width in [1, 7, 64] {
        let (live, all) = feed_bounded(&raw, width);
        assert_eq!(all, expected, "width {width}");
        assert!(
            live.len() + 64 >= raw.len(),
            "only {} of {} bytes streamed before flush at width {width}",
            live.len(),
            raw.len()
        );
    }
    let (live, all) = feed_bounded(&raw, raw.len());
    assert_eq!(all, expected);
    assert!(
        live.len() + 64 >= raw.len(),
        "whole-text feed held the reply"
    );
}

#[test]
fn literal_muse_role_header_does_not_hold_the_rest_of_the_stream() {
    let raw = format!(
        "Muse writes <|start|>assistant to= before a recipient. {}",
        filler(2000)
    );
    let (expected, _, _) = non_streamed(&raw, None);
    assert_eq!(expected, raw);
    for width in [1, 7, 64, raw.len()] {
        let (live, all) = feed_bounded(&raw, width);
        assert_eq!(all, expected, "width {width}");
        assert!(live.len() + 64 >= raw.len(), "width {width} held the reply");
    }
}

#[test]
fn overlong_harmony_header_after_harmony_output_stays_hidden() {
    // After Harmony output, a header that never reaches `<|message|>` is not
    // content (`try_harmony` stops there). Holding it must stay bounded, and
    // the body behind an overlong header must not surface either.
    let unterminated = format!(
        "<|channel|>final<|message|>ok<|end|><|start|>assistant<|channel|>{}",
        "x ".repeat(3000)
    );
    let late_message = format!(
        "<|channel|>final<|message|>ok<|end|><|start|>assistant<|channel|>{}<|message|>secret",
        "x".repeat(1000)
    );
    for raw in [unterminated, late_message] {
        let (expected, _, _) = non_streamed(&raw, None);
        assert_eq!(expected, "ok");
        for width in [1, 7, 64, raw.len()] {
            let (_, all) = feed_bounded(&raw, width);
            assert_eq!(all, expected, "width {width}");
        }
    }
}

#[test]
fn harmony_tool_arguments_keep_foreign_close_tags_out_of_content() {
    // Inside a Harmony message only Harmony markers move the filter: a Hermes
    // `</tool_call>` in the arguments must not end suppression and stream the
    // rest of the call as content (#1743).
    let raw = concat!(
        "<|channel|>analysis<|message|>Call the tool.<|end|><|start|>assistant<|channel|>commentary ",
        "to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"</tool_call> Paris\"}"
    );
    let tools = [tool("get_weather")];
    let (content, reasoning, calls) = assert_parity(raw, Some(&tools));
    assert_eq!(content, "");
    assert_eq!(reasoning, "Call the tool.");
    let calls = calls.unwrap_or_default();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, r#"{"city":"</tool_call> Paris"}"#);
    let (content, _, _) = assert_parity(raw, None);
    assert_eq!(content, "");
}

#[test]
fn harmony_analysis_keeps_foreign_think_close_out_of_content() {
    // A `</think>` the model writes inside its `analysis` channel is text, not
    // the end of reasoning; the rest of the chain-of-thought must not stream as
    // content.
    let raw = concat!(
        "<|channel|>analysis<|message|>The tag </think> closes a block. So answer: 51.<|end|>",
        "<|start|>assistant<|channel|>final<|message|>51"
    );
    let (content, reasoning, _) = assert_parity(raw, None);
    assert_eq!(content, "51");
    assert_eq!(reasoning, "The tag </think> closes a block. So answer: 51.");
    let tools = [tool("get_weather")];
    let (content, _, _) = assert_parity(raw, Some(&tools));
    assert_eq!(content, "51");
}
