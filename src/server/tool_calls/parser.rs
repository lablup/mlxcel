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

//! Core output parsing logic for tool call extraction.
//!
//! Tries each known format in sequence. If tools were requested and the
//! output matches a known pattern, structured tool calls are extracted.
//!
//! Used by: routes/chat, tool_calls::mod

use super::formats;
use super::types::{ParsedToolCall, ToolCallParseResult};
use crate::server::types::request::Tool;

/// Strip thinking/reasoning blocks from model output before parsing.
///
/// Handles two formats:
/// - `<think>...</think>` — DeepSeek R1, Granite 3.3, etc.
/// - `<|channel>...<channel|>` — Gemma 4 thinking channels
///
/// Also handles the **prompt-primed** Gemma 4 case: when the chat template's
/// generation prompt ended with `<|channel>thought\n` (open), the model's
/// generated text starts inside the thinking channel and only carries the
/// closing `<channel|>` — no matching `<|channel>` is ever emitted. Without
/// this case, every token before the first `<channel|>` leaks into the
/// user-visible content. Treat a leading (pre-first-`<|channel>`) `<channel|>`
/// as the close of an implicit open.
///
/// Used by: tool_calls::parser
fn strip_thinking(text: &str) -> String {
    // (open, close) marker pairs by family: Qwen-style `<think>...</think>`
    // and Gemma 4 `<|channel>...<channel|>`.
    let pairs: &[(&str, &str)] = &[("<think>", "</think>"), ("<|channel>", "<channel|>")];

    // Prompt-primed pass: when a close marker appears with no opening marker
    // before it, the generation prompt primed the open (Gemma 4
    // `<|channel>thought\n` or a Qwen-style `<think>\n`), so the model output
    // starts inside the block and carries only the close. Strip everything up
    // to and including that first bare close, for both families. Runs before
    // the balanced-pair pass so any later `open...close` blocks still get
    // handled uniformly, and so a primed `</think>` close is stripped on the
    // non-streaming path the same way `/v1/responses`, `/v1/anthropic`, and the
    // streaming filter already handle it.
    let mut result = text.to_string();
    for &(open, close) in pairs {
        if let Some(close_pos) = result.find(close)
            && !result[..close_pos].contains(open)
        {
            result = result[close_pos + close.len()..].to_string();
        }
    }
    for &(open, close) in pairs {
        let mut new_result = String::with_capacity(result.len());
        let mut remaining = result.as_str();
        while let Some(start) = remaining.find(open) {
            new_result.push_str(&remaining[..start]);
            if let Some(end) = remaining[start..].find(close) {
                remaining = &remaining[start + end + close.len()..];
            } else {
                // Unclosed tag -- strip everything after it
                remaining = "";
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }
    result
}

/// Remove model-specific structural markers from content text.
///
/// Called when tool-call parsing was attempted but no tool calls were found,
/// or when cleaning content extracted alongside tool calls.  Strips markers
/// that would otherwise leak into the response content.
///
/// Note that matched `<|channel>...<channel|>` blocks are removed earlier
/// by [`strip_thinking`]; this function only cleans *stray* markers (e.g.
/// a dangling `<channel|>` the model emits without a matching open tag,
/// which happens with Gemma 4 non-thinking mode).
///
/// Used by: tool_calls::parser
fn clean_content_markers(text: &str) -> String {
    text.replace("<turn|>", "")
        .replace("<|turn>", "")
        .replace("<|think|>", "")
        .replace("<channel|>", "")
        .replace("<|channel>", "")
        .replace("<|tool_call>", "")
        .replace("<tool_call|>", "")
        .trim()
        .to_string()
}

/// Strip all known structural tokens (thinking blocks and stray turn/think
/// markers) from a raw model output.
///
/// This is the non-streaming counterpart to
/// [`StreamFilter`](super::stream_filter::StreamFilter) and should be applied
/// to any chat response text before it is returned to the client, regardless
/// of whether tool-call parsing is enabled.  Without this pass, Gemma 4 leaks
/// `<channel|>`, `<turn|>`, and related markers into plain chat content when
/// no tools are present in the request.
// Used by: routes/chat (non-streaming path)
pub fn clean_structural_tokens(raw: &str) -> String {
    clean_content_markers(&strip_thinking(raw))
}

/// Parse model output for tool calls, trying each known format in order.
///
/// `tools` is the set of tools that were passed in the request.  When
/// provided, the parser will filter out any calls to functions not in
/// the tool set.
///
/// Returns a `ToolCallParseResult` which may contain zero or more parsed
/// calls.
pub fn parse_tool_calls(raw_output: &str, tools: Option<&[Tool]>) -> ToolCallParseResult {
    // Strip thinking blocks first
    let cleaned = strip_thinking(raw_output);
    let text = cleaned.trim();

    if text.is_empty() {
        // All generation was thinking-channel content (common when the Gemma 4
        // generation prompt primes an open `<|channel>thought\n` and the
        // model fills it for the whole window). Returning the raw bytes here
        // would leak every token of reasoning into the user-visible content;
        // return empty content instead so the response is a clean "model
        // thought but produced no answer" rather than a wall of `<|channel>`
        // markers.
        return ToolCallParseResult::none(String::new());
    }

    // Harmony (GPT-OSS) is handled up front, ahead of the single-purpose
    // parsers below. Unlike them it owns the whole channel stream: it routes the
    // `analysis` channel to `reasoning_content` and strips the channel markup
    // from `content` even when the model answers directly without a tool call.
    // Returning its result unconditionally (when the distinctive `<|channel|>`
    // markers are present) is what prevents that markup from leaking through the
    // fall-through cleaner in the no-tool-call case.
    if let Some(mut result) = formats::try_harmony(text) {
        if let Some(tools) = tools {
            result.tool_calls = filter_by_tools(result.tool_calls, tools);
        }
        return result;
    }

    // Try each format in order of specificity (most distinctive markers first)
    let parsers: &[fn(&str) -> Option<ToolCallParseResult>] = &[
        formats::try_granite, // <response><tool_call> — more specific than bare Hermes
        formats::try_gemma4,  // <|tool_call>call:... — pipe-delimited, before Hermes
        formats::try_hermes,  // <tool_call> — Hermes/Qwen/DeepSeek
        formats::try_minimax_m2, // <invoke name=...><parameter name=...>...</parameter></invoke>
        formats::try_kimi_k2, // <|tool_calls_section_begin|>...<|tool_calls_section_end|>
        // [TOOL_CALLS]NAME[ARGS]{json}; declines (returns None) without [ARGS], so the
        // older JSON-array format below still falls through to try_mistral_nemo.
        formats::try_mistral_bracket,
        formats::try_mistral_nemo,    // [TOOL_CALLS]
        formats::try_functionary_v31, // <function=name>{json}
        formats::try_qwen3_coder, // <function=name><parameter=key>val</parameter> (after v31, which declines non-JSON bodies)
        formats::try_functionary_v32, // >>>name\n
        // <|tool_call_start|>[func(arg=value)]<|tool_call_end|> — pythonic, before generic JSON
        formats::try_pythonic,
        formats::try_llama3,       // {"name": ..., "parameters": ...}
        formats::try_generic_json, // {"name": ..., "arguments": ...}
        formats::try_command_r,    // Action: / Action Input: — least specific
    ];

    for parser in parsers {
        if let Some(mut result) = parser(text) {
            // Filter tool calls to only those in the provided tool set
            if let Some(tools) = tools {
                result.tool_calls = filter_by_tools(result.tool_calls, tools);
            }
            if result.has_tool_calls() {
                return result;
            }
        }
    }

    // No tool calls found: return cleaned content (thinking blocks and
    // model-specific markers stripped) so callers never see raw control tokens.
    let content = clean_content_markers(text);
    ToolCallParseResult::none(content)
}

/// Filter parsed tool calls to only include functions that exist in the
/// provided tool definitions.
///
/// Names are normalized before matching: if a call name contains a leading
/// namespace segment (e.g. `functions.get_weather`), the segment up to and
/// including the first `.` is stripped and the bare name is matched against
/// the registered tools. The call is emitted with the stripped name when the
/// match succeeds. The original spelling is kept when neither the full name
/// nor the stripped name matches any registered tool (in which case the call
/// is dropped).
fn filter_by_tools(calls: Vec<ParsedToolCall>, tools: &[Tool]) -> Vec<ParsedToolCall> {
    let tool_names: std::collections::HashSet<&str> =
        tools.iter().map(|t| t.function.name.as_str()).collect();

    calls
        .into_iter()
        .filter_map(|mut c| {
            if tool_names.contains(c.name.as_str()) {
                // Bare name matches directly.
                Some(c)
            } else if let Some(dot_pos) = c.name.find('.') {
                // Strip the leading namespace segment (e.g. `functions.`) and
                // retry. Only normalise when the result actually matches a
                // registered tool; otherwise drop the call as before.
                let stripped = c.name[dot_pos + 1..].to_string();
                if tool_names.contains(stripped.as_str()) {
                    c.name = stripped;
                    Some(c)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

/// Generate a unique tool call ID in the format `call_` + random alphanumeric.
pub fn generate_tool_call_id() -> String {
    let id = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("call_{}", &id[..24])
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn parse_hermes_format() {
        let output =
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_with_thinking_blocks() {
        let output = r#"<think>Let me check the weather API.</think>
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_filters_unknown_tools() {
        let output = r#"<tool_call>{"name": "unknown_fn", "arguments": {}}</tool_call>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_no_tools_provided_accepts_all() {
        let output = r#"<tool_call>{"name": "any_fn", "arguments": {"x": 1}}</tool_call>"#;
        let result = parse_tool_calls(output, None);
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "any_fn");
    }

    #[test]
    fn parse_plain_text_returns_none() {
        let output = "Hello, how can I help you?";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, output);
    }

    #[test]
    fn parse_mistral_nemo_format() {
        let output = r#"[TOOL_CALLS] [{"name": "search", "arguments": {"query": "rust"}}]"#;
        let tools = vec![make_tool("search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "search");
        // Regression: the old JSON-array format must still resolve to
        // MistralNemo, proving try_mistral_bracket correctly falls through.
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::MistralNemo)
        );
    }

    #[test]
    fn parse_mistral_bracket_format() {
        let output = r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Mistral)
        );
    }

    #[test]
    fn parse_mistral_bracket_filters_unknown_tools() {
        let output = r#"[TOOL_CALLS]unknown_fn[ARGS]{}"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_functionary_v31_format() {
        let output = r#"<function=get_weather>{"location": "Berlin"}</function>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parse_generic_json_format() {
        let output = r#"{"name": "calc", "arguments": {"expr": "2+2"}}"#;
        let tools = vec![make_tool("calc")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "calc");
    }

    #[test]
    fn parse_command_r_format() {
        let output = "Action: search\nAction Input: {\"query\": \"rust\"}";
        let tools = vec![make_tool("search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn parse_granite_format() {
        let output = r#"<response><tool_call>{"name": "get_info", "arguments": {"id": 42}}</tool_call></response>"#;
        let tools = vec![make_tool("get_info")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_info");
    }

    #[test]
    fn parse_deepseek_format_via_hermes() {
        // DeepSeek uses <tool_call> (Hermes style) after <think> blocks
        let output = "<think>Reasoning step.</think>\n<tool_call>{\"name\": \"fn\", \"arguments\": {\"k\": \"v\"}}</tool_call>";
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "fn");
    }

    #[test]
    fn parse_gemma4_format() {
        let output = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Gemma4)
        );
    }

    #[test]
    fn parse_gemma4_filters_unknown_tools() {
        let output = "<|tool_call>call:unknown_fn{key:<|\"|>val<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_granite_takes_precedence_over_hermes() {
        // Granite format wraps in <response>; must be matched by Granite handler, not Hermes
        let output =
            r#"<response><tool_call>{"name": "fn", "arguments": {}}</tool_call></response>"#;
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Granite)
        );
    }

    #[test]
    fn generate_tool_call_id_format() {
        let id = generate_tool_call_id();
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 29); // "call_" (5) + 24 hex chars
    }

    #[test]
    fn strip_thinking_removes_think_blocks() {
        let input = "<think>reasoning here</think>actual content";
        let result = strip_thinking(input);
        assert_eq!(result, "actual content");
    }

    #[test]
    fn strip_thinking_handles_multiple_blocks() {
        let input = "<think>first</think>middle<think>second</think>end";
        let result = strip_thinking(input);
        assert_eq!(result, "middleend");
    }

    #[test]
    fn strip_thinking_handles_unclosed_tag() {
        let input = "<think>unclosed thinking";
        let result = strip_thinking(input);
        assert_eq!(result, "");
    }

    #[test]
    fn strip_thinking_passes_through_no_tags() {
        let input = "no thinking tags here";
        let result = strip_thinking(input);
        assert_eq!(result, input);
    }

    // -- Gemma 4 channel/thinking block stripping --

    #[test]
    fn strip_thinking_removes_gemma4_channel() {
        let input = "<|channel>thought\nI should search.<channel|>actual content";
        let result = strip_thinking(input);
        assert_eq!(result, "actual content");
    }

    #[test]
    fn strip_thinking_handles_unclosed_channel() {
        let input = "<|channel>thought\nunclosed channel";
        let result = strip_thinking(input);
        assert_eq!(result, "");
    }

    #[test]
    fn strip_thinking_handles_both_think_and_channel() {
        let input = "<think>think block</think><|channel>thought\nchannel block<channel|>final";
        let result = strip_thinking(input);
        assert_eq!(result, "final");
    }

    #[test]
    fn strip_thinking_primed_qwen_close_only() {
        // Prompt-primed `<think>\n`: the model output carries only the closing
        // `</think>` (no opening `<think>`), so the close-only block must be
        // stripped the same way the primed Gemma 4 `<channel|>` case is. This
        // matches the responses/anthropic split and chat's streaming filter.
        let input = "reasoning text</think>\n\nHello!";
        let result = strip_thinking(input);
        assert_eq!(result, "\n\nHello!");
    }

    #[test]
    fn clean_structural_tokens_primed_qwen_close_only() {
        // The full non-streaming content path: a primed `<think>` close-only
        // output yields the clean answer, not the leaked reasoning + marker.
        let cleaned = clean_structural_tokens("reasoning text</think>\n\nHello!");
        assert_eq!(cleaned, "Hello!");
    }

    // -- Gemma 4 full-path parse tests --

    #[test]
    fn gemma4_thinking_only() {
        let output = "<|channel>thought\nI should search.<channel|><turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "");
    }

    #[test]
    fn gemma4_thinking_then_content() {
        let output = "<|channel>thought\nPlanning.<channel|>Here is the answer.<turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "Here is the answer.");
    }

    #[test]
    fn gemma4_thinking_then_tool_call() {
        let output = "<|channel>thought\nI need to search.<channel|>\
                       <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("web_search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "web_search");
        assert_eq!(result.content, "");
    }

    #[test]
    fn gemma4_content_then_tool_call() {
        let output = "Let me search.\
                       <|tool_call>call:web_search{query:<|\"|>rust<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("web_search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content, "Let me search.");
    }

    #[test]
    fn gemma4_multiple_tool_calls() {
        let output = "<|tool_call>call:search{query:<|\"|>rust<|\"|>}<tool_call|>\
                       <|tool_call>call:calc{expr:<|\"|>2+2<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("search"), make_tool("calc")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "calc");
    }

    #[test]
    fn gemma4_strips_trailing_turn() {
        // Content followed by <turn|> with no tool calls
        let output = "Here is the answer.<turn|>";
        let result = parse_tool_calls(output, None);
        assert!(!result.content.contains("<turn|>"));
        assert_eq!(result.content, "Here is the answer.");
    }

    #[test]
    fn gemma4_strips_trailing_turn_with_tools() {
        // Content + tool call + <turn|> — content must not contain <turn|>
        let output = "Content<|tool_call>call:fn{key:<|\"|>v<|\"|>}<tool_call|><turn|>";
        let tools = vec![make_tool("fn")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert!(!result.content.contains("<turn|>"));
        assert_eq!(result.content, "Content");
    }

    #[test]
    fn clean_content_markers_strips_all() {
        let input = "<|turn>content<turn|><|think|>";
        let result = clean_content_markers(input);
        assert_eq!(result, "content");
    }

    // -- Prompt-primed Gemma 4 (enable_thinking=true) --
    //
    // When the chat template ends the generation prompt with an OPEN
    // `<|channel>thought\n` marker, the model's generated text begins inside
    // the thinking channel and only carries the closing `<channel|>` — no
    // matching open is ever emitted. `strip_thinking` must recognize that
    // lonely close as the end of the primed thinking block.

    #[test]
    fn strip_thinking_prompt_primed_close_drops_preceding_thinking() {
        // Model wrote scratchpad notes, then closed the channel the prompt
        // primed, then wrote its real answer. The scratchpad has no matching
        // `<|channel>` in the generated text (the open was in the prompt).
        let input = "*   Goal: …\n    *   Steps …<channel|>Here is the answer.";
        let result = strip_thinking(input);
        assert_eq!(result, "Here is the answer.");
    }

    #[test]
    fn strip_thinking_prompt_primed_close_then_tool_call() {
        // Same shape but the post-thinking content is a Gemma 4 tool call.
        // The parser below extracts the call; strip_thinking is only
        // responsible for dropping the prompt-primed thinking prefix.
        let input = "thinking about it<channel|><|tool_call>call:fn{k:<|\"|>v<|\"|>}<tool_call|>";
        let result = strip_thinking(input);
        assert_eq!(result, "<|tool_call>call:fn{k:<|\"|>v<|\"|>}<tool_call|>");
    }

    #[test]
    fn strip_thinking_preserves_balanced_channel_after_orphan_close() {
        // A close without a preceding open strips; a second balanced
        // `<|channel>…<channel|>` block afterwards strips normally too.
        let input = "prompt thinking<channel|>middle<|channel>more thought<channel|>tail";
        let result = strip_thinking(input);
        assert_eq!(result, "middletail");
    }

    // -- MiniMax M2 --

    #[test]
    fn parse_minimax_m2_single_call() {
        let output = "<invoke name=\"get_weather\">\n<parameter name=\"location\">Paris</parameter>\n</invoke>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::MinimaxM2)
        );
    }

    #[test]
    fn parse_minimax_m2_parallel_calls() {
        // Multiple <invoke> blocks = parallel tool calls (the fix in PR #1171)
        let output = "<invoke name=\"search\">\n<parameter name=\"query\">weather</parameter>\n</invoke>\n<invoke name=\"read_file\">\n<parameter name=\"path\">/tmp/test.txt</parameter>\n</invoke>";
        let tools = vec![make_tool("search"), make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "read_file");
    }

    #[test]
    fn parse_minimax_m2_filters_unknown_tools() {
        let output = "<invoke name=\"unknown_fn\">\n<parameter name=\"k\">v</parameter>\n</invoke>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    // -- Kimi K2 --

    #[test]
    fn parse_kimi_k2_single_call() {
        let output = "<|tool_calls_section_begin|>\
                       <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"Paris\"}<|tool_call_end|>\
                       <|tool_calls_section_end|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::KimiK2)
        );
    }

    #[test]
    fn parse_kimi_k2_multiple_calls() {
        let output = "<|tool_calls_section_begin|>\
                       <|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"weather\"}<|tool_call_end|>\
                       <|tool_call_begin|>functions.read_file:1<|tool_call_argument_begin|>{\"path\": \"/tmp/test.txt\"}<|tool_call_end|>\
                       <|tool_calls_section_end|>";
        let tools = vec![make_tool("search"), make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "read_file");
    }

    #[test]
    fn parse_kimi_k2_filters_unknown_tools() {
        let output = "<|tool_calls_section_begin|>\
                       <|tool_call_begin|>functions.unknown_fn:0<|tool_call_argument_begin|>{}<|tool_call_end|>\
                       <|tool_calls_section_end|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    // -- Pythonic --

    #[test]
    fn parse_pythonic_format() {
        let output = r#"<|tool_call_start|>[get_weather(city="Paris", days=2)]<|tool_call_end|>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 2);
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Pythonic)
        );
    }

    #[test]
    fn parse_pythonic_filters_unknown_tools() {
        let output = "<|tool_call_start|>[unknown_fn(x=1)]<|tool_call_end|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_pythonic_does_not_steal_json_tool_call() {
        // Regression guard: `try_pythonic` runs before the JSON parsers and its
        // call regex `\[(\w+)\(...\)\]` scans anywhere, so a valid JSON tool
        // call whose string argument merely CONTAINS a `[word(...)]` substring
        // must not be re-routed to the bracketed inner name. Here the real call
        // is `search`, and `calc` (which appears inside the query string) is
        // also a registered tool, so the parser must still return `search`.
        let output = r#"{"name": "search", "arguments": {"query": "[calc(x=1)]"}}"#;
        let tools = vec![make_tool("search"), make_tool("calc")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].name, "search",
            "a JSON `search` call must not be mis-parsed as the bracketed `calc`"
        );
        assert_ne!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Pythonic),
            "a JSON tool call must be claimed by a JSON parser, not the pythonic one"
        );
    }

    #[test]
    fn parse_tool_calls_empty_after_strip_returns_empty_content() {
        // Previously we fell back to `raw_output` when stripping emptied the
        // text, which leaked the scratchpad back into the visible response.
        // Now the empty result is respected so the user sees nothing rather
        // than a wall of thinking markers when the model ran past budget
        // inside a primed channel.
        let raw = "thinking that never closes the channel";
        let result = parse_tool_calls(raw, None);
        // The prep-step in strip_thinking finds no `<channel|>` and leaves
        // text as-is, so for this specific input nothing is stripped.
        assert_eq!(result.content, "thinking that never closes the channel");

        // But when the prompt-primed orphan close IS present and consumes
        // everything, content stays empty (no fallback to raw).
        let raw2 = "all thinking<channel|>";
        let result2 = parse_tool_calls(raw2, None);
        assert_eq!(result2.content, "");
    }

    // ----------------------------------------------------------------
    // Qwen3-Coder XML tool-call format (issue: agentic clients break
    // because tool calls land in `content` instead of `tool_calls`)
    // ----------------------------------------------------------------

    fn arg_obj(call: &crate::server::tool_calls::ParsedToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).expect("arguments must be valid JSON")
    }

    #[test]
    fn qwen3_coder_single_call_single_param() {
        let output = "<tool_call>\n<function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n</function>\n</tool_call>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(arg_obj(&result.tool_calls[0])["location"], "San Francisco");
        // Pure tool-call response: no leaked content.
        assert_eq!(result.content, "");
    }

    #[test]
    fn qwen3_coder_single_call_multiple_params_with_type_coercion() {
        let output = "<tool_call><function=search><parameter=query>rust async</parameter><parameter=limit>5</parameter><parameter=fuzzy>true</parameter></function></tool_call>";
        let tools = vec![make_tool("search")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        let args = arg_obj(&result.tool_calls[0]);
        assert_eq!(args["query"], "rust async");
        assert_eq!(args["limit"], 5); // coerced to integer
        assert_eq!(args["fuzzy"], true); // coerced to boolean
    }

    #[test]
    fn qwen3_coder_multiple_calls_in_one_response() {
        let output = "<tool_call><function=read_file><parameter=path>a.rs</parameter></function></tool_call><tool_call><function=read_file><parameter=path>b.rs</parameter></function></tool_call>";
        let tools = vec![make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(arg_obj(&result.tool_calls[0])["path"], "a.rs");
        assert_eq!(arg_obj(&result.tool_calls[1])["path"], "b.rs");
    }

    #[test]
    fn qwen3_coder_value_with_whitespace_newlines_and_quotes() {
        // A code-snippet parameter: internal whitespace, newlines, and quotes
        // must survive into a valid JSON string.
        let output = "<tool_call><function=write_file><parameter=path>main.rs</parameter><parameter=content>\nfn main() {\n    println!(\"hi\");\n}\n</parameter></function></tool_call>";
        let tools = vec![make_tool("write_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        let args = arg_obj(&result.tool_calls[0]);
        assert_eq!(args["path"], "main.rs");
        assert_eq!(args["content"], "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn qwen3_coder_empty_parameter_value() {
        let output =
            "<tool_call><function=set_note><parameter=text></parameter></function></tool_call>";
        let tools = vec![make_tool("set_note")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(arg_obj(&result.tool_calls[0])["text"], "");
    }

    #[test]
    fn qwen3_coder_malformed_missing_closing_tags_keeps_prior_calls() {
        // First call is well-formed; the second opener is truncated mid-stream.
        // The good call must still be returned (no panic, no total discard).
        let output = "<tool_call><function=read_file><parameter=path>a.rs</parameter></function></tool_call><tool_call><function=read_file";
        let tools = vec![make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(arg_obj(&result.tool_calls[0])["path"], "a.rs");
    }

    #[test]
    fn qwen3_coder_zero_parameter_call() {
        // No `<parameter=` body at all: must still parse as a no-arg call,
        // not fall through to raw content.
        let output = "<tool_call><function=list_files></function></tool_call>";
        let tools = vec![make_tool("list_files")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "list_files");
        assert_eq!(result.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn qwen3_coder_does_not_regress_functionary_v31_json_body() {
        // Functionary v3.1 shares the `<function=` opener but has a JSON body.
        // It must still be claimed by the functionary parser, not mis-parsed
        // by the Qwen3-Coder parser into empty args.
        let output = r#"<function=get_weather>{"location": "Berlin"}</function>"#;
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(arg_obj(&result.tool_calls[0])["location"], "Berlin");
    }

    // -- namespace normalization tests -----------------------------------------

    #[test]
    fn filter_by_tools_strips_leading_namespace() {
        // Gemma 4 can emit `functions.get_weather` when the registered tool is
        // `get_weather`. The filter must strip the `functions.` prefix and
        // emit the call with the bare name so downstream serialisation produces
        // the correct function name.
        let output =
            "<|tool_call>call:functions.get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(
            result.has_tool_calls(),
            "namespaced call should pass the filter"
        );
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].name, "get_weather",
            "name must be normalised to the bare form"
        );
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn filter_by_tools_bare_name_unchanged() {
        // Bare names (no dot) must continue to work as before.
        let output = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn filter_by_tools_unknown_namespaced_name_dropped() {
        // `functions.unknown_fn` is not in the registered tool set even after
        // stripping, so it must be dropped.
        let output = "<|tool_call>call:functions.unknown_fn{key:<|\"|>val<|\"|>}<tool_call|>";
        let tools = vec![make_tool("get_weather")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(
            !result.has_tool_calls(),
            "unknown namespaced call must be dropped"
        );
    }

    // -- Harmony (GPT-OSS) end-to-end --------------------------------------------

    #[test]
    fn parse_harmony_tool_call() {
        let output = "<|channel|>analysis<|message|>Read the file first.<|end|>\
                      <|start|>assistant<|channel|>commentary to=functions.read_file \
                      <|constrain|>json<|message|>{\"path\": \"/etc/hosts\"}<|call|>";
        let tools = vec![make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "read_file");
        assert_eq!(arg_obj(&result.tool_calls[0])["path"], "/etc/hosts");
        assert_eq!(
            result.format,
            Some(crate::server::tool_calls::ToolCallFormat::Harmony)
        );
        // Channel markup must never leak into content.
        assert_eq!(result.content, "");
        assert!(!result.content.contains("<|channel|>"));
        // Analysis channel surfaces as reasoning_content.
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("Read the file first.")
        );
    }

    #[test]
    fn parse_harmony_filters_unknown_tools() {
        let output = "<|channel|>commentary to=functions.unknown_fn \
                      <|constrain|>json<|message|>{}<|call|>";
        let tools = vec![make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
    }

    #[test]
    fn parse_harmony_no_tools_accepts_all() {
        let output = "<|channel|>commentary to=functions.any_fn \
                      <|constrain|>json<|message|>{\"x\": 1}<|call|>";
        let result = parse_tool_calls(output, None);
        assert!(result.has_tool_calls());
        assert_eq!(result.tool_calls[0].name, "any_fn");
    }

    #[test]
    fn parse_harmony_multiple_calls() {
        let output = "<|channel|>commentary to=functions.a <|constrain|>json\
                      <|message|>{\"x\": 1}<|call|><|start|>assistant<|channel|>\
                      commentary to=functions.b <|constrain|>json<|message|>{\"y\": 2}<|call|>";
        let tools = vec![make_tool("a"), make_tool("b")];
        let result = parse_tool_calls(output, Some(&tools));
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn parse_harmony_final_only_does_not_leak_markup() {
        // Tools were requested but the model answered directly via `final`.
        // Without the up-front Harmony handling this markup would leak through
        // the fall-through content cleaner.
        let output = "<|channel|>analysis<|message|>Thinking.<|end|><|start|>assistant\
                      <|channel|>final<|message|>Hello there!<|return|>";
        let tools = vec![make_tool("read_file")];
        let result = parse_tool_calls(output, Some(&tools));
        assert!(!result.has_tool_calls());
        assert_eq!(result.content, "Hello there!");
        assert_eq!(result.reasoning_content.as_deref(), Some("Thinking."));
        assert!(!result.content.contains("<|channel|>"));
    }
}
