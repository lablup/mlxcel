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

//! Known tool call format definitions for popular model families.
//!
//! Each format handler attempts to parse the model output and returns
//! `Some(ToolCallParseResult)` on success or `None` if the format
//! does not match.
//!
//! Used by: tool_calls::parser

use fancy_regex::Regex;

use super::types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};

/// Try parsing Hermes/Qwen format:
/// `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>`
///
/// Multiple tool calls may appear sequentially.
pub fn try_hermes(text: &str) -> Option<ToolCallParseResult> {
    let tag_open = "<tool_call>";
    let tag_close = "</tool_call>";

    if !text.contains(tag_open) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;

    // Collect content before the first tool_call tag
    if let Some(first_pos) = remaining.find(tag_open) {
        let before = remaining[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
        remaining = &remaining[first_pos..];
    }

    while let Some(start) = remaining.find(tag_open) {
        let json_start = start + tag_open.len();
        let end = remaining[json_start..].find(tag_close);

        let json_str = if let Some(end_offset) = end {
            let s = &remaining[json_start..json_start + end_offset];
            remaining = &remaining[json_start + end_offset + tag_close.len()..];
            s
        } else {
            // No closing tag: take the rest
            let s = &remaining[json_start..];
            remaining = "";
            s
        };

        if let Some(call) = parse_hermes_json(json_str.trim()) {
            calls.push(call);
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Hermes),
        tool_calls: calls,
        content,
        reasoning_content: None,
    })
}

/// Parse a single Hermes-format JSON object.
fn parse_hermes_json(json_str: &str) -> Option<ParsedToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments")?;
    Some(ParsedToolCall {
        name,
        arguments: if arguments.is_string() {
            arguments.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(arguments).ok()?
        },
    })
}

/// Try parsing Llama 3.x format:
/// `{"name": "fn", "parameters": {...}}`
/// Optionally preceded by `<|python_tag|>`
pub fn try_llama3(text: &str) -> Option<ToolCallParseResult> {
    let cleaned = text
        .trim()
        .strip_prefix("<|python_tag|>")
        .unwrap_or(text.trim())
        .trim();

    // May be a single call or array of calls
    if cleaned.starts_with('[') {
        let arr: Vec<serde_json::Value> = serde_json::from_str(cleaned).ok()?;
        let calls: Vec<ParsedToolCall> = arr.iter().filter_map(parse_llama3_value).collect();
        if calls.is_empty() {
            return None;
        }
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::Llama3),
            tool_calls: calls,
            content: String::new(),
            reasoning_content: None,
        });
    }

    if cleaned.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(cleaned).ok()?;
        let call = parse_llama3_value(&v)?;
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::Llama3),
            tool_calls: vec![call],
            content: String::new(),
            reasoning_content: None,
        });
    }

    None
}

fn parse_llama3_value(v: &serde_json::Value) -> Option<ParsedToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    // Llama uses "parameters" instead of "arguments"
    let args = v.get("parameters").or_else(|| v.get("arguments"))?;
    Some(ParsedToolCall {
        name,
        arguments: if args.is_string() {
            args.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(args).ok()?
        },
    })
}

/// Try parsing Mistral Nemo format:
/// `[TOOL_CALLS] [{"name": "fn", "arguments": {...}}]`
pub fn try_mistral_nemo(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    let json_part = trimmed.strip_prefix("[TOOL_CALLS]")?.trim();

    let arr: Vec<serde_json::Value> = serde_json::from_str(json_part).ok()?;
    let calls: Vec<ParsedToolCall> = arr
        .iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let arguments = v.get("arguments")?;
            Some(ParsedToolCall {
                name,
                arguments: if arguments.is_string() {
                    arguments.as_str().unwrap_or_default().to_string()
                } else {
                    serde_json::to_string(arguments).ok()?
                },
            })
        })
        .collect();

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::MistralNemo),
        tool_calls: calls,
        content: String::new(),
        reasoning_content: None,
    })
}

/// Try parsing Functionary v3.1 format:
/// `<function=fn_name>{"key": "val"}</function>`
pub fn try_functionary_v31(text: &str) -> Option<ToolCallParseResult> {
    let prefix = "<function=";
    if !text.contains(prefix) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;

    if let Some(first_pos) = remaining.find(prefix) {
        let before = remaining[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
        remaining = &remaining[first_pos..];
    }

    while let Some(start) = remaining.find(prefix) {
        let name_start = start + prefix.len();
        // Use if-let instead of ? to avoid early return from the entire
        // function when a single malformed tag is missing its closing '>'.
        let Some(name_end) = remaining[name_start..].find('>') else {
            // Malformed tag without '>': skip past the prefix and continue
            remaining = &remaining[name_start..];
            continue;
        };
        let name = remaining[name_start..name_start + name_end].to_string();

        let json_start = name_start + name_end + 1; // skip '>'
        let close_tag = "</function>";
        let json_end = remaining[json_start..].find(close_tag);

        let json_str = if let Some(end_offset) = json_end {
            let s = &remaining[json_start..json_start + end_offset];
            remaining = &remaining[json_start + end_offset + close_tag.len()..];
            s
        } else {
            let s = &remaining[json_start..];
            remaining = "";
            s
        };

        let arguments = json_str.trim().to_string();
        // Validate it's valid JSON
        if serde_json::from_str::<serde_json::Value>(&arguments).is_ok() {
            calls.push(ParsedToolCall { name, arguments });
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::FunctionaryV31),
        tool_calls: calls,
        content,
        reasoning_content: None,
    })
}

/// Try parsing Functionary v3.2 format:
/// `>>>fn_name\n{"key": "val"}`
///
/// Only matches `>>>` at the start of a line (position 0 or after `\n`).
/// This prevents false positives when `>>>` appears mid-line (e.g. in shell
/// output or blockquotes).
pub fn try_functionary_v32(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    // Guard: >>> must appear at line start (position 0 or after a newline)
    if !trimmed.starts_with(">>>") && !trimmed.contains("\n>>>") {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut found_first = false;

    // Walk line by line, treating each ">>>" line-start as a segment header.
    // Lines before the first ">>>" header become the `content` prefix.
    let mut current_name: Option<String> = None;
    let mut current_args_lines: Vec<&str> = Vec::new();

    for line in trimmed.lines() {
        if let Some(stripped) = line.strip_prefix(">>>") {
            // Flush any pending call
            if let Some(name) = current_name.take() {
                let json_str = current_args_lines.join("\n");
                let json_str = json_str.trim();
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                    calls.push(ParsedToolCall {
                        name,
                        arguments: json_str.to_string(),
                    });
                }
                current_args_lines.clear();
            }

            found_first = true;
            let name = stripped.trim().to_string();
            // Skip "all" which is a common delimiter for general text
            if name != "all" && !name.is_empty() {
                current_name = Some(name);
            }
        } else if found_first {
            if current_name.is_some() {
                current_args_lines.push(line);
            }
        } else {
            // Content before the first >>> marker
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(line);
        }
    }

    // Flush last pending call
    if let Some(name) = current_name.take() {
        let json_str = current_args_lines.join("\n");
        let json_str = json_str.trim().to_string();
        if serde_json::from_str::<serde_json::Value>(&json_str).is_ok() {
            calls.push(ParsedToolCall {
                name,
                arguments: json_str,
            });
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::FunctionaryV32),
        tool_calls: calls,
        content: content.trim().to_string(),
        reasoning_content: None,
    })
}

/// Try parsing Command R7B format:
/// `Action: function_name\nAction Input: {"key": "val"}`
///
/// Multiple calls may appear sequentially, each starting with `Action:`.
pub fn try_command_r(text: &str) -> Option<ToolCallParseResult> {
    // Guard: text must contain both the "Action:" and "Action Input:" markers.
    // Checking only "Action:" would cause false positives on normal prose
    // such as "Required Action: none".
    if !text.contains("Action:") || !text.contains("Action Input:") {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut found_first = false;
    let mut pending_name: Option<String> = None;

    for line in text.lines() {
        let trimmed_line = line.trim();
        if let Some(name) = trimmed_line.strip_prefix("Action:") {
            // Flush any pending action that had no Action Input line
            pending_name = Some(name.trim().to_string());
            if !found_first {
                found_first = true;
            }
        } else if let Some(json_part) = trimmed_line.strip_prefix("Action Input:") {
            if let Some(name) = pending_name.take() {
                let json_str = json_part.trim();
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                    calls.push(ParsedToolCall {
                        name,
                        arguments: json_str.to_string(),
                    });
                }
            }
        } else if !found_first {
            // Accumulate content before the first Action: marker
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(line);
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::CommandR),
        tool_calls: calls,
        content: content.trim().to_string(),
        reasoning_content: None,
    })
}

/// Try parsing Granite 3.3 format:
/// `<response><tool_call>{"name": ..., "arguments": ...}</tool_call></response>`
///
/// Strips the outer `<response>...</response>` wrapper and delegates to the
/// Hermes parser, which handles the inner `<tool_call>` tags.
pub fn try_granite(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();
    if !trimmed.contains("<response>") {
        return None;
    }

    // Extract the content inside <response>...</response>
    let start_tag = "<response>";
    let end_tag = "</response>";

    let inner_start = trimmed.find(start_tag)? + start_tag.len();
    let inner_end = trimmed[inner_start..].find(end_tag);

    let inner = if let Some(end_offset) = inner_end {
        &trimmed[inner_start..inner_start + end_offset]
    } else {
        // No closing tag: take everything after the opening tag
        &trimmed[inner_start..]
    };

    // Extract any content before the <response> tag
    let prefix = trimmed[..trimmed.find(start_tag).unwrap_or(0)].trim();

    // Delegate to the Hermes parser for the inner content
    let mut result = try_hermes(inner.trim())?;

    // Prepend any prefix content
    if !prefix.is_empty() {
        if result.content.is_empty() {
            result.content = prefix.to_string();
        } else {
            result.content = format!("{prefix}\n{}", result.content);
        }
    }

    // Override the format to Granite
    result.format = Some(ToolCallFormat::Granite);
    Some(result)
}

/// Try parsing Gemma 4 format:
/// `<|tool_call>call:function_name{key:<|"|>val<|"|>}<tool_call|>`
///
/// The pipe-delimited tags (`<|tool_call>` / `<tool_call|>`) distinguish this
/// format from Hermes's `<tool_call>` tag.  Multiple tool calls may appear
/// sequentially.  String values are delimited by `<|"|>` markers; non-string
/// values (numbers, booleans, null) are written without delimiters.
pub fn try_gemma4(text: &str) -> Option<ToolCallParseResult> {
    let tag_open = "<|tool_call>";
    let tag_close = "<tool_call|>";

    if !text.contains(tag_open) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;

    // Collect content before the first tool_call tag
    if let Some(first_pos) = remaining.find(tag_open) {
        let before = remaining[..first_pos].trim();
        if !before.is_empty() {
            content = before.to_string();
        }
        remaining = &remaining[first_pos..];
    }

    while let Some(start) = remaining.find(tag_open) {
        let block_start = start + tag_open.len();
        let block_end = remaining[block_start..].find(tag_close);

        let block = if let Some(end_offset) = block_end {
            let s = &remaining[block_start..block_start + end_offset];
            remaining = &remaining[block_start + end_offset + tag_close.len()..];
            s
        } else {
            // No closing tag: take the rest
            let s = &remaining[block_start..];
            remaining = "";
            s
        };

        if let Some(call) = parse_gemma4_block(block.trim()) {
            calls.push(call);
        }
    }

    if calls.is_empty() {
        return None;
    }

    // Strip Gemma 4 structural markers from content (e.g. trailing `<turn|>`,
    // stray `<channel|>` left over from an out-of-order thinking close when
    // `thinking_budget_tokens` force-injected the close marker). `<|channel>`
    // is included for symmetry so partial/malformed emissions don't leak.
    let content = content
        .replace("<turn|>", "")
        .replace("<|turn>", "")
        .replace("<|think|>", "")
        .replace("<channel|>", "")
        .replace("<|channel>", "");
    let content = content.trim().to_string();

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Gemma4),
        tool_calls: calls,
        content,
        reasoning_content: None,
    })
}

/// Parse a single Gemma 4 tool call block: `call:function_name{args_body}`
fn parse_gemma4_block(block: &str) -> Option<ParsedToolCall> {
    // Must start with "call:"
    let rest = block.strip_prefix("call:")?;

    // Find the opening brace for arguments
    let brace_pos = rest.find('{')?;
    let name = rest[..brace_pos].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let args_with_brace = rest[brace_pos..].trim();
    let arguments = gemma4_args_to_json(args_with_brace)?;

    // Validate that the constructed JSON is well-formed before returning.
    // The Gemma4 format is built by string concatenation, so a malformed
    // model output could produce invalid JSON.  Other format parsers rely
    // on serde_json to parse the original text; we do the equivalent here.
    serde_json::from_str::<serde_json::Value>(&arguments).ok()?;

    Some(ParsedToolCall { name, arguments })
}

/// Convert Gemma 4 argument syntax to a valid JSON object string.
///
/// Input format: `{key:<|"|>string_val<|"|>, key2:42, key3:true}`
/// Output format: `{"key":"string_val","key2":42,"key3":true}`
fn gemma4_args_to_json(args: &str) -> Option<String> {
    // Strip outer braces
    let inner = args.strip_prefix('{')?.strip_suffix('}')?.trim();

    if inner.is_empty() {
        return Some("{}".to_string());
    }

    // Replace the string delimiter with a NUL placeholder that won't appear
    // in normal text, making it easy to detect string boundaries.
    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let normalised = inner.replace("<|\"|>", STR_DELIM);

    let pairs = split_top_level_commas(&normalised);
    let mut json_pairs: Vec<String> = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        // Split on the first `:` to get key and value
        let colon_pos = pair.find(':')?;
        let key = pair[..colon_pos].trim();
        let value_raw = pair[colon_pos + 1..].trim();

        let json_key = format!("\"{}\"", escape_json_string(key));

        let json_value = if value_raw.starts_with(STR_DELIM) {
            // String value: strip delimiters and JSON-escape the content
            let without_open = value_raw.strip_prefix(STR_DELIM)?;
            let content = if let Some(end) = without_open.rfind(STR_DELIM) {
                &without_open[..end]
            } else {
                // Unterminated string: take everything
                without_open
            };
            format!("\"{}\"", escape_json_string(content))
        } else if value_raw.starts_with('{') {
            // Nested object: recurse
            // Restore the original delimiters before recursing
            let restored = value_raw.replace(STR_DELIM, "<|\"|>");
            gemma4_args_to_json(&restored)?
        } else if value_raw.starts_with('[') {
            // Array: restore delimiters and parse via serde_json
            let restored = value_raw.replace(STR_DELIM, "<|\"|>");
            gemma4_array_to_json(&restored)?
        } else {
            // Number, boolean, null — pass through as-is
            value_raw.to_string()
        };

        json_pairs.push(format!("{json_key}:{json_value}"));
    }

    Some(format!("{{{}}}", json_pairs.join(",")))
}

/// Escape a string for inclusion inside JSON double quotes.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Convert a Gemma 4 array literal to a JSON array string.
///
/// Arrays use the same `<|"|>` string escaping as objects.
fn gemma4_array_to_json(arr: &str) -> Option<String> {
    let inner = arr.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some("[]".to_string());
    }

    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let normalised = inner.replace("<|\"|>", STR_DELIM);
    let items = split_top_level_commas(&normalised);
    let mut json_items: Vec<String> = Vec::with_capacity(items.len());

    for item in items {
        let item = item.trim();
        if item.starts_with(STR_DELIM) {
            let without_open = item.strip_prefix(STR_DELIM)?;
            let content = if let Some(end) = without_open.rfind(STR_DELIM) {
                &without_open[..end]
            } else {
                without_open
            };
            json_items.push(format!("\"{}\"", escape_json_string(content)));
        } else if item.starts_with('{') {
            let restored = item.replace(STR_DELIM, "<|\"|>");
            json_items.push(gemma4_args_to_json(&restored)?);
        } else if item.starts_with('[') {
            let restored = item.replace(STR_DELIM, "<|\"|>");
            json_items.push(gemma4_array_to_json(&restored)?);
        } else {
            json_items.push(item.to_string());
        }
    }

    Some(format!("[{}]", json_items.join(",")))
}

/// Split a comma-separated sequence at the top level only
/// (i.e., not inside `{}`, `[]` brackets, or string delimiters).
///
/// String regions are bounded by the `STR_DELIM` placeholder that
/// `gemma4_args_to_json` / `gemma4_array_to_json` substitute for the
/// original `<|"|>` markers.  Commas inside string regions are ignored.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    const STR_DELIM: &str = "<|\"|\u{200b}>";
    let delim_bytes = STR_DELIM.as_bytes();

    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut last = 0;

    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Check for string delimiter boundary
        if i + delim_bytes.len() <= bytes.len() && &bytes[i..i + delim_bytes.len()] == delim_bytes {
            in_string = !in_string;
            i += delim_bytes.len();
            continue;
        }

        if !in_string {
            match bytes[i] {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    parts.push(&s[last..i]);
                    last = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    if last < s.len() {
        parts.push(&s[last..]);
    }
    parts
}

/// Maximum number of `<invoke>` blocks the MiniMax M2 parser will accept from a
/// single model output. Caps memory amplification when an adversarial model
/// emits a long run of well-formed open tags. Real models emit at most a handful
/// of parallel calls; 1024 leaves room for pathological-but-legitimate behavior
/// while preventing a 100k-call DoS.
const MINIMAX_M2_MAX_CALLS: usize = 1024;
/// Same idea for `<parameter>` tags inside one `<invoke>` body.
const MINIMAX_M2_MAX_PARAMS_PER_CALL: usize = 1024;

/// Try parsing MiniMax M2 format:
/// `<invoke name="fn_name"><parameter name="k">v</parameter></invoke>`
///
/// Multiple `<invoke>` blocks may appear sequentially for parallel tool calls.
/// Parameter values are converted to their most likely JSON types (number,
/// boolean, null, object/array via JSON parse) before falling back to string.
///
/// Mirrors the upstream Python rewrite in mlx-lm PR #1171 (commit 6d11468).
pub fn try_minimax_m2(text: &str) -> Option<ToolCallParseResult> {
    let invoke_open = "<invoke name=";
    let invoke_close = "</invoke>";

    if !text.contains(invoke_open) {
        return None;
    }

    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find(invoke_open) {
        let after_tag = start + invoke_open.len();

        // Extract the function name (inside quotes or until ">").
        // Use if-let instead of `?` so that a trailing malformed
        // `<invoke name=` (without a closing `>`) does not discard
        // already-parsed calls — same fix pattern as `try_functionary_v31`.
        let Some(name_end) = remaining[after_tag..].find('>') else {
            // Malformed tag without '>': stop scanning, keep prior calls
            break;
        };
        let raw_name = &remaining[after_tag..after_tag + name_end];
        let function_name = extract_quoted_name(raw_name);
        if function_name.is_empty() {
            remaining = &remaining[after_tag + name_end + 1..];
            continue;
        }

        // Everything up to </invoke> is the body
        let body_start = after_tag + name_end + 1; // skip '>'
        let body_end = remaining[body_start..].find(invoke_close);

        let body = if let Some(end_offset) = body_end {
            let b = &remaining[body_start..body_start + end_offset];
            remaining = &remaining[body_start + end_offset + invoke_close.len()..];
            b
        } else {
            // No closing </invoke>: take the rest as the body
            let b = &remaining[body_start..];
            remaining = "";
            b
        };

        let arguments = extract_minimax_parameters(body);
        calls.push(ParsedToolCall {
            name: function_name,
            arguments,
        });

        // Defensive cap: bound parallel-call memory under adversarial output.
        if calls.len() >= MINIMAX_M2_MAX_CALLS {
            break;
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::MinimaxM2),
        tool_calls: calls,
        content: String::new(),
        reasoning_content: None,
    })
}

/// Extract the name from a possibly-quoted attribute value.
///
/// Handles `"fn_name"`, `'fn_name'`, or bare `fn_name`.  Requires `s.len()
/// >= 2` before stripping outer quotes, otherwise a single lone quote
/// character would panic on the `s[1..s.len() - 1]` slice (`1..0`).
fn extract_quoted_name(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Parse all `<parameter name="k">v</parameter>` tags inside an `<invoke>` body.
///
/// Returns a JSON object string mapping parameter names to their typed values.
/// Type coercion order: null → integer → float → boolean → JSON object/array → string.
fn extract_minimax_parameters(body: &str) -> String {
    let param_open = "<parameter name=";
    let param_close = "</parameter>";

    let mut pairs: Vec<(String, serde_json::Value)> = Vec::new();
    let mut remaining = body;

    while let Some(start) = remaining.find(param_open) {
        let after_tag = start + param_open.len();

        // Find the closing '>' of the opening tag.
        //
        // If there is no '>' anywhere in the remaining buffer, no well-formed
        // `<parameter name=key>val</parameter>` block can follow, so stop
        // scanning. Using `continue` here with `remaining = &remaining[after_tag..]`
        // would still match the next `<parameter name=` prefix in the body,
        // re-scan the buffer for '>' (and again find none), and repeat —
        // producing O(N^2) work in the number of malformed parameter prefixes.
        // An adversarial model output can pin a request handler thread for
        // many seconds this way; `break` matches the outer `try_minimax_m2`
        // loop's behavior for the analogous missing-'>' case.
        let Some(gt_pos) = remaining[after_tag..].find('>') else {
            break;
        };

        let raw_name = &remaining[after_tag..after_tag + gt_pos];
        let param_name = extract_quoted_name(raw_name);

        let value_start = after_tag + gt_pos + 1;
        let value_end = remaining[value_start..].find(param_close);

        let raw_value = if let Some(end_offset) = value_end {
            let v = &remaining[value_start..value_start + end_offset];
            remaining = &remaining[value_start + end_offset + param_close.len()..];
            v
        } else {
            let v = &remaining[value_start..];
            remaining = "";
            v
        };

        // Trim leading/trailing newlines (upstream behaviour)
        let mut raw_value = raw_value;
        if raw_value.starts_with('\n') {
            raw_value = &raw_value[1..];
        }
        if raw_value.ends_with('\n') {
            raw_value = &raw_value[..raw_value.len() - 1];
        }
        let raw_value = raw_value.trim();

        let json_value = coerce_minimax_param(raw_value);
        if !param_name.is_empty() {
            pairs.push((param_name, json_value));
        }

        // Defensive cap: bound per-call parameter memory under adversarial output.
        if pairs.len() >= MINIMAX_M2_MAX_PARAMS_PER_CALL {
            break;
        }
    }

    // Build a JSON object from the collected pairs
    let mut map = serde_json::Map::with_capacity(pairs.len());
    for (k, v) in pairs {
        map.insert(k, v);
    }
    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

/// Convert a raw string parameter value to the most specific JSON type.
///
/// Priority: null → integer → float → boolean → JSON object/array → string.
fn coerce_minimax_param(value: &str) -> serde_json::Value {
    // Null
    let lower = value.to_lowercase();
    if lower == "null" || lower == "none" || lower == "nil" {
        return serde_json::Value::Null;
    }

    // Integer (try before float so we preserve exact integer representation)
    if let Ok(i) = value.parse::<i64>() {
        return serde_json::Value::Number(serde_json::Number::from(i));
    }

    // Float
    if let Ok(f) = value.parse::<f64>()
        && let Some(n) = serde_json::Number::from_f64(f)
    {
        return serde_json::Value::Number(n);
    }

    // Boolean
    if lower == "true" || lower == "1" || lower == "yes" || lower == "on" {
        return serde_json::Value::Bool(true);
    }
    if lower == "false" || lower == "0" || lower == "no" || lower == "off" {
        return serde_json::Value::Bool(false);
    }

    // JSON object or array
    if (value.starts_with('{') || value.starts_with('['))
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(value)
    {
        return v;
    }

    // Fallback: string
    serde_json::Value::String(value.to_string())
}

// Defensive caps mirroring the MiniMax M2 parser: bound parallel-call and
// per-call parameter memory under adversarial model output.
const QWEN3_CODER_MAX_CALLS: usize = 1024;
const QWEN3_CODER_MAX_PARAMS_PER_CALL: usize = 1024;

/// Try parsing the Qwen3-Coder XML tool-call format:
///
/// ```text
/// <tool_call>
/// <function=NAME>
/// <parameter=KEY>
/// VALUE
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// This shares the `<function=...>` opener with Functionary v3.1 but differs in
/// the body: Qwen3-Coder emits `<parameter=key>val</parameter>` XML rather than a
/// JSON object. The dispatcher runs this parser *after* [`try_functionary_v31`],
/// which declines Qwen input because its non-JSON body fails the JSON-validity
/// check, so functionary calls are never stolen, and zero-parameter Qwen calls
/// (an empty `<function=NAME></function>` body) are still handled here.
///
/// The surrounding `<tool_call>` wrapper is not required: scanning for
/// `<function=` naturally skips it, matching Qwen3-Coder variants that omit it.
pub fn try_qwen3_coder(text: &str) -> Option<ToolCallParseResult> {
    let fn_open = "<function=";
    let fn_close = "</function>";

    let fn_pos = text.find(fn_open)?;

    // Capture any prose before the first tool call. Prefer the `<tool_call>`
    // wrapper boundary when it precedes `<function=` so the wrapper tag itself
    // never leaks into the user-visible content.
    let boundary = match text.find("<tool_call>") {
        Some(tc) if tc < fn_pos => tc,
        _ => fn_pos,
    };
    let content = text[..boundary].trim().to_string();

    let mut calls = Vec::new();
    let mut remaining = &text[fn_pos..];

    while let Some(start) = remaining.find(fn_open) {
        let name_start = start + fn_open.len();
        // if-let rather than `?` so a single malformed `<function=` opener
        // without a closing '>' does not discard already-parsed calls.
        let Some(name_end) = remaining[name_start..].find('>') else {
            break;
        };
        let function_name = extract_quoted_name(&remaining[name_start..name_start + name_end]);

        let body_start = name_start + name_end + 1; // skip '>'
        let body = match remaining[body_start..].find(fn_close) {
            Some(end_offset) => {
                let b = &remaining[body_start..body_start + end_offset];
                remaining = &remaining[body_start + end_offset + fn_close.len()..];
                b
            }
            None => {
                // No closing </function>: take the rest as the body.
                let b = &remaining[body_start..];
                remaining = "";
                b
            }
        };

        if !function_name.is_empty() {
            let arguments = extract_qwen_parameters(body);
            calls.push(ParsedToolCall {
                name: function_name,
                arguments,
            });
        }

        if calls.len() >= QWEN3_CODER_MAX_CALLS {
            break;
        }
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Qwen3Coder),
        tool_calls: calls,
        content,
        reasoning_content: None,
    })
}

/// Parse all `<parameter=key>value</parameter>` tags inside a Qwen3-Coder
/// `<function>` body into a JSON object string.
///
/// Mirrors [`extract_minimax_parameters`] but keys on the bare `<parameter=`
/// opener (Qwen3-Coder) rather than `<parameter name=` (MiniMax M2). Values are
/// stripped of one surrounding newline (the model pretty-prints each parameter
/// on its own line) then trimmed, and typed via [`coerce_minimax_param`].
fn extract_qwen_parameters(body: &str) -> String {
    let param_open = "<parameter=";
    let param_close = "</parameter>";

    let mut pairs: Vec<(String, serde_json::Value)> = Vec::new();
    let mut remaining = body;

    while let Some(start) = remaining.find(param_open) {
        let after_tag = start + param_open.len();

        // `break` (not `continue`) on a missing '>': a malformed `<parameter=`
        // prefix with no '>' anywhere left would otherwise re-match and rescan
        // forever: the same O(N^2) guard as `extract_minimax_parameters`.
        let Some(gt_pos) = remaining[after_tag..].find('>') else {
            break;
        };

        let param_name = extract_quoted_name(&remaining[after_tag..after_tag + gt_pos]);

        let value_start = after_tag + gt_pos + 1; // skip '>'
        let raw_value = match remaining[value_start..].find(param_close) {
            Some(end_offset) => {
                let v = &remaining[value_start..value_start + end_offset];
                remaining = &remaining[value_start + end_offset + param_close.len()..];
                v
            }
            None => {
                let v = &remaining[value_start..];
                remaining = "";
                v
            }
        };

        // Strip a single leading/trailing newline (the model emits the value on
        // its own line), then trim surrounding spaces.
        let mut raw_value = raw_value;
        if let Some(stripped) = raw_value.strip_prefix('\n') {
            raw_value = stripped;
        }
        if let Some(stripped) = raw_value.strip_suffix('\n') {
            raw_value = stripped;
        }
        let raw_value = raw_value.trim();

        if !param_name.is_empty() {
            pairs.push((param_name, coerce_minimax_param(raw_value)));
        }

        if pairs.len() >= QWEN3_CODER_MAX_PARAMS_PER_CALL {
            break;
        }
    }

    let mut map = serde_json::Map::with_capacity(pairs.len());
    for (k, v) in pairs {
        map.insert(k, v);
    }
    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

/// Try parsing generic JSON format:
/// `{"name": "fn", "arguments": {...}}` or `{"name": "fn", "parameters": {...}}`
///
/// Also handles arrays: `[{"name": ..., "arguments": ...}, ...]`
pub fn try_generic_json(text: &str) -> Option<ToolCallParseResult> {
    let trimmed = text.trim();

    // Try as array
    if trimmed.starts_with('[')
        && let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed)
    {
        let calls: Vec<ParsedToolCall> = arr.iter().filter_map(parse_generic_value).collect();
        if !calls.is_empty() {
            return Some(ToolCallParseResult {
                format: Some(ToolCallFormat::GenericJson),
                tool_calls: calls,
                content: String::new(),
                reasoning_content: None,
            });
        }
    }

    // Try as single object
    if trimmed.starts_with('{')
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(call) = parse_generic_value(&v)
    {
        return Some(ToolCallParseResult {
            format: Some(ToolCallFormat::GenericJson),
            tool_calls: vec![call],
            content: String::new(),
            reasoning_content: None,
        });
    }

    None
}

fn parse_generic_value(v: &serde_json::Value) -> Option<ParsedToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let args = v.get("arguments").or_else(|| v.get("parameters"))?;
    Some(ParsedToolCall {
        name,
        arguments: if args.is_string() {
            args.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(args).ok()?
        },
    })
}

/// Markers that terminate a Harmony message body. The body after `<|message|>`
/// runs until the earliest of these (or the end of the text): `<|call|>` ends a
/// tool call, `<|end|>` ends an analysis/preamble message, `<|return|>` ends the
/// final answer, and `<|start|>` / `<|channel|>` begin the next message when the
/// model omitted an explicit terminator (or the output was truncated mid-token,
/// as happens when `<|call|>` is a stop token that never reaches the text).
const HARMONY_BODY_TERMINATORS: &[&str] = &[
    "<|call|>",
    "<|end|>",
    "<|return|>",
    "<|start|>",
    "<|channel|>",
];

/// Try parsing the Harmony (GPT-OSS) channel format.
///
/// Harmony interleaves reasoning, tool calls, and the visible answer as a
/// sequence of channel messages:
///
/// ```text
/// <|channel|>analysis<|message|>chain of thought<|end|>
/// <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{…}<|call|>
/// <|start|>assistant<|channel|>final<|message|>visible answer<|return|>
/// ```
///
/// Routing:
/// - a channel header carrying a `to=` recipient is a tool call (the recipient's
///   `functions.` namespace is stripped to the bare function name),
/// - the `analysis` channel is chain-of-thought and surfaces as
///   [`ToolCallParseResult::reasoning_content`],
/// - the `final` channel (and any recipient-less `commentary` preamble) is the
///   visible `content`.
///
/// The distinctive double-pipe `<|channel|>` / `<|message|>` markers never
/// collide with Gemma 4's single-pipe `<|channel>` / `<channel|>`, so a plain
/// `contains` guard is enough to claim only Harmony output. Because this parser
/// also owns the reasoning/content split, the dispatcher runs it up front and
/// returns its result whether or not a tool call is present, so channel markup
/// never leaks into `content`.
pub fn try_harmony(text: &str) -> Option<ToolCallParseResult> {
    const CHANNEL: &str = "<|channel|>";
    const MESSAGE: &str = "<|message|>";

    if !text.contains(CHANNEL) || !text.contains(MESSAGE) {
        return None;
    }

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut reasoning = String::new();

    let mut rest = text;
    while let Some(ch_pos) = rest.find(CHANNEL) {
        // Header = everything between `<|channel|>` and the message marker:
        // the channel name plus any `to=` recipient and `<|constrain|>` hint.
        let after_channel = &rest[ch_pos + CHANNEL.len()..];
        let Some(msg_off) = after_channel.find(MESSAGE) else {
            break;
        };
        let header = &after_channel[..msg_off];
        let body_region = &after_channel[msg_off + MESSAGE.len()..];

        // Body runs until the earliest terminator (or end of text).
        let body_end = HARMONY_BODY_TERMINATORS
            .iter()
            .filter_map(|marker| body_region.find(marker))
            .min()
            .unwrap_or(body_region.len());
        let body = &body_region[..body_end];

        route_harmony_segment(header, body, &mut calls, &mut content, &mut reasoning);

        // Advance past this body; `body_region` starts inside `rest`, and the
        // header consumed at least the `<|channel|>` marker, so `rest` strictly
        // shrinks every iteration (no infinite loop).
        rest = &body_region[body_end..];
    }

    let reasoning = reasoning.trim();
    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Harmony),
        tool_calls: calls,
        content: content.trim().to_string(),
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning.to_string())
        },
    })
}

/// Route one Harmony `(header, body)` segment to a tool call, reasoning, or
/// visible content.
fn route_harmony_segment(
    header: &str,
    body: &str,
    calls: &mut Vec<ParsedToolCall>,
    content: &mut String,
    reasoning: &mut String,
) {
    // A `to=` recipient marks a tool call regardless of channel name.
    if let Some(recipient) = harmony_recipient(header) {
        let name = recipient
            .strip_prefix("functions.")
            .unwrap_or(recipient)
            .to_string();
        if !name.is_empty() {
            calls.push(ParsedToolCall {
                name,
                arguments: harmony_arguments(body),
            });
        }
        return;
    }

    let channel = header.split_whitespace().next().unwrap_or("");
    let body = body.trim();
    if body.is_empty() {
        return;
    }
    // `analysis` is chain-of-thought; `final` is the answer and a recipient-less
    // `commentary` is a user-visible preamble. Unknown channels are dropped so
    // their markup never leaks.
    let sink = match channel {
        "analysis" => reasoning,
        "final" | "commentary" => content,
        _ => return,
    };
    if !sink.is_empty() {
        sink.push('\n');
    }
    sink.push_str(body);
}

/// Extract the `to=` recipient from a Harmony channel header, if present.
///
/// The recipient runs from just after `to=` until the next whitespace or the
/// next `<|` marker (e.g. `<|constrain|>` or `<|message|>`), whichever comes
/// first: `commentary to=functions.get_weather <|constrain|>json` yields
/// `functions.get_weather`.
fn harmony_recipient(header: &str) -> Option<&str> {
    let after = &header[header.find("to=")? + "to=".len()..];
    let end = [after.find(char::is_whitespace), after.find("<|")]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(after.len());
    let recipient = after[..end].trim();
    (!recipient.is_empty()).then_some(recipient)
}

/// Normalize a Harmony tool-call body into a JSON `arguments` string.
///
/// The body is JSON when the header carried `<|constrain|>json` (the common
/// case). Valid JSON is re-serialized to drop the model's pretty-printing (and,
/// with serde_json's `preserve_order`, keep key order); a body that does not
/// parse falls back to its trimmed form, and an empty body to `{}`, so a
/// malformed call never yields invalid `arguments`.
fn harmony_arguments(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "{}".to_string();
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| trimmed.to_string()),
        Err(_) => trimmed.to_string(),
    }
}

/// Try parsing the Kimi K2 sectioned tool-call format:
///
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "Paris"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// The per-call `<|tool_call_begin|>` / `<|tool_call_end|>` wrappers are
/// optional: when the section body carries none, the whole section body is
/// parsed as a single call. Each call id has the shape `functions.NAME:INDEX`
/// (the `functions.` namespace prefix is optional); only the middle `NAME`
/// segment is kept. The model-provided id and index are discarded, matching
/// every other parser in this module — `ParsedToolCall` has no `id` field, so
/// the caller regenerates a fresh id via `generate_tool_call_id`.
pub fn try_kimi_k2(text: &str) -> Option<ToolCallParseResult> {
    const SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
    const SECTION_END: &str = "<|tool_calls_section_end|>";
    const CALL_BEGIN: &str = "<|tool_call_begin|>";
    const CALL_END: &str = "<|tool_call_end|>";

    let section_start = text.find(SECTION_BEGIN)?;
    let content = text[..section_start].trim().to_string();
    let body_start = section_start + SECTION_BEGIN.len();

    let body = match text[body_start..].find(SECTION_END) {
        Some(end_offset) => &text[body_start..body_start + end_offset],
        None => &text[body_start..],
    };

    // Name/argument marker: `functions.NAME:INDEX<|tool_call_argument_begin|>`.
    // Group 1 (unused downstream) is the full id `functions.NAME:INDEX`;
    // group 2 is the bare function NAME. Compiled once and reused across
    // every call segment in this section.
    let Ok(name_re) =
        Regex::new(r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>")
    else {
        return None;
    };

    let mut calls = Vec::new();

    if body.contains(CALL_BEGIN) {
        let mut remaining = body;
        while let Some(start) = remaining.find(CALL_BEGIN) {
            let call_start = start + CALL_BEGIN.len();
            let call_body = match remaining[call_start..].find(CALL_END) {
                Some(end_offset) => {
                    let b = &remaining[call_start..call_start + end_offset];
                    remaining = &remaining[call_start + end_offset + CALL_END.len()..];
                    b
                }
                None => {
                    // No closing wrapper: take the rest as the last call body.
                    let b = &remaining[call_start..];
                    remaining = "";
                    b
                }
            };
            if let Some(call) = parse_kimi_k2_call(call_body, &name_re) {
                calls.push(call);
            }
        }
    } else if let Some(call) = parse_kimi_k2_call(body, &name_re) {
        calls.push(call);
    }

    if calls.is_empty() {
        return None;
    }

    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::KimiK2),
        tool_calls: calls,
        content,
        reasoning_content: None,
    })
}

/// Parse one Kimi K2 call body: `functions.NAME:INDEX<|tool_call_argument_begin|>{...}`.
///
/// Returns `None` when the name marker does not match (e.g. a call segment
/// that carries no `functions.NAME:INDEX<|tool_call_argument_begin|>` header
/// at all), so a single malformed call in a multi-call section is skipped
/// rather than discarding the whole result.
fn parse_kimi_k2_call(call_body: &str, name_re: &Regex) -> Option<ParsedToolCall> {
    let caps = name_re.captures(call_body).ok()??;
    let name = caps.get(2)?.as_str().to_string();
    if name.is_empty() {
        return None;
    }
    let marker_end = caps.get(0)?.end();
    let args_raw = call_body[marker_end..].trim();
    Some(ParsedToolCall {
        name,
        arguments: kimi_k2_arguments(args_raw),
    })
}

/// Convert raw Kimi K2 argument text (everything after
/// `<|tool_call_argument_begin|>`, already trimmed) into a JSON `arguments`
/// string.
///
/// Parse order: strict JSON first (the common case — Kimi K2 emits a JSON
/// object literal); if that fails, a loose literal reparse that tolerates
/// Python-dict-repr spellings (single-quoted strings, `True`/`False`/`None`);
/// if both fail, the raw text is kept verbatim so a malformed call still
/// produces a call instead of being dropped.
fn kimi_k2_arguments(raw: &str) -> String {
    if raw.is_empty() {
        return "{}".to_string();
    }

    if let Some(json) = kimi_k2_try_json(raw) {
        return json;
    }

    let loosened = kimi_k2_loosen_literal(raw);
    if let Some(json) = kimi_k2_try_json(&loosened) {
        return json;
    }

    raw.to_string()
}

/// Parse `raw` as a JSON value, re-serializing to a compact JSON string.
/// A value that is itself a JSON string is unwrapped rather than
/// double-encoded, mirroring `parse_hermes_json` / `try_mistral_nemo`.
fn kimi_k2_try_json(raw: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    Some(if value.is_string() {
        value.as_str().unwrap_or_default().to_string()
    } else {
        serde_json::to_string(&value).ok()?
    })
}

/// Best-effort normalization of Python-dict-repr-style arguments (single
/// quotes, `True`/`False`/`None`) into JSON syntax, used as the second parse
/// attempt in [`kimi_k2_arguments`]. Not a full Python literal evaluator:
/// quotes are swapped verbatim, so a string value containing an apostrophe
/// will not round-trip — the caller's raw-string fallback covers that case.
fn kimi_k2_loosen_literal(raw: &str) -> String {
    let quoted: String = raw
        .chars()
        .map(|c| if c == '\'' { '"' } else { c })
        .collect();
    replace_python_literals(&quoted)
}

/// Replace whole-word Python literal tokens (`True`, `False`, `None`) with
/// their JSON equivalents (`true`, `false`, `null`). A token only matches
/// when it is not adjacent to another identifier character on either side,
/// so `NoneType` or `TrueValue` are left untouched.
fn replace_python_literals(text: &str) -> String {
    const REPLACEMENTS: &[(&str, &str)] = &[("True", "true"), ("False", "false"), ("None", "null")];

    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let mut matched = false;
        for (lit, json) in REPLACEMENTS {
            if text[i..].starts_with(lit) {
                let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
                let after = i + lit.len();
                let after_ok = after >= text.len() || !is_ident_byte(bytes[after]);
                if before_ok && after_ok {
                    out.push_str(json);
                    i = after;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            let ch = text[i..]
                .chars()
                .next()
                .expect("i < text.len() guarantees a char");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Whether a byte can be part of an identifier (used for word-boundary
/// detection in [`replace_python_literals`]).
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hermes / Qwen --

    #[test]
    fn hermes_single_tool_call() {
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Paris"));
        assert_eq!(result.format, Some(ToolCallFormat::Hermes));
    }

    #[test]
    fn hermes_multiple_tool_calls() {
        let text = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn hermes_with_content_before() {
        let text = r#"Let me check the weather.
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>"#;
        let result = try_hermes(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
    }

    #[test]
    fn hermes_no_match() {
        assert!(try_hermes("Hello, world!").is_none());
    }

    // -- Llama 3.x --

    #[test]
    fn llama3_single_call() {
        let text = r#"{"name": "search", "parameters": {"query": "rust"}}"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        assert!(result.tool_calls[0].arguments.contains("rust"));
    }

    #[test]
    fn llama3_with_python_tag() {
        let text = r#"<|python_tag|>{"name": "calc", "parameters": {"expr": "2+2"}}"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls[0].name, "calc");
    }

    #[test]
    fn llama3_array() {
        let text = r#"[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {"x": 1}}]"#;
        let result = try_llama3(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
    }

    // -- Mistral Nemo --

    #[test]
    fn mistral_nemo_single_call() {
        let text = r#"[TOOL_CALLS] [{"name": "get_time", "arguments": {"tz": "UTC"}}]"#;
        let result = try_mistral_nemo(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_time");
    }

    #[test]
    fn mistral_nemo_no_prefix() {
        assert!(try_mistral_nemo(r#"[{"name": "x", "arguments": {}}]"#).is_none());
    }

    // -- Functionary v3.1 --

    #[test]
    fn functionary_v31_single_call() {
        let text = r#"<function=get_weather>{"location": "Berlin"}</function>"#;
        let result = try_functionary_v31(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Berlin"));
    }

    #[test]
    fn functionary_v31_no_match() {
        assert!(try_functionary_v31("Hello!").is_none());
    }

    #[test]
    fn functionary_v31_malformed_trailing_tag_preserves_prior_calls() {
        // A trailing malformed tag (no '>' at all) must not discard already-parsed calls.
        // Before the fix, the '?' operator would abort the entire function, returning None.
        let text =
            r#"<function=get_weather>{"location": "Berlin"}</function><function=broken_no_close"#;
        let result = try_functionary_v31(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Berlin"));
    }

    // -- Functionary v3.2 --

    #[test]
    fn functionary_v32_single_call() {
        let text = ">>>get_weather\n{\"location\": \"London\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn functionary_v32_skips_all_segment() {
        let text = ">>>all\nHello\n>>>get_weather\n{\"location\": \"Tokyo\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn functionary_v32_multiple_calls() {
        let text = ">>>fn_a\n{\"x\": 1}\n>>>fn_b\n{\"y\": 2}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "fn_a");
        assert_eq!(result.tool_calls[1].name, "fn_b");
    }

    #[test]
    fn functionary_v32_no_false_positive_mid_line() {
        // >>> appearing mid-line (e.g. shell output) must NOT match
        let text = "The output was: foo>>>bar\nsome more text";
        assert!(try_functionary_v32(text).is_none());
    }

    #[test]
    fn functionary_v32_no_false_positive_blockquote() {
        // >>> in a blockquote-like context mid-sentence must not match
        let text = "Here is an example: >>>function_name\n{}";
        assert!(try_functionary_v32(text).is_none());
    }

    #[test]
    fn functionary_v32_content_before_first_marker() {
        let text = "Some preamble\n>>>get_weather\n{\"location\": \"Berlin\"}";
        let result = try_functionary_v32(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content, "Some preamble");
    }

    // -- Command R --

    #[test]
    fn command_r_single_call() {
        let text = "Action: get_weather\nAction Input: {\"location\": \"Paris\"}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Paris"));
        assert_eq!(result.format, Some(ToolCallFormat::CommandR));
    }

    #[test]
    fn command_r_multiple_calls() {
        let text = "Action: fn_a\nAction Input: {\"x\": 1}\nAction: fn_b\nAction Input: {\"y\": 2}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "fn_a");
        assert_eq!(result.tool_calls[1].name, "fn_b");
    }

    #[test]
    fn command_r_with_content_before() {
        let text = "I need to check the weather.\nAction: get_weather\nAction Input: {\"city\": \"Tokyo\"}";
        let result = try_command_r(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
    }

    #[test]
    fn command_r_no_match() {
        assert!(try_command_r("Hello, world!").is_none());
        assert!(try_command_r("Action without colon").is_none());
    }

    #[test]
    fn command_r_invalid_json_skipped() {
        let text = "Action: fn\nAction Input: not-json";
        assert!(try_command_r(text).is_none());
    }

    // -- Granite 3.3 --

    #[test]
    fn granite_single_call() {
        let text = r#"<response><tool_call>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Seoul"));
        assert_eq!(result.format, Some(ToolCallFormat::Granite));
    }

    #[test]
    fn granite_multiple_calls() {
        let text = r#"<response><tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn granite_with_content_before_response_tag() {
        let text = r#"Let me help.<response><tool_call>{"name": "search", "arguments": {"q": "rust"}}</tool_call></response>"#;
        let result = try_granite(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("Let me help"));
    }

    #[test]
    fn granite_no_match() {
        assert!(try_granite("Hello, world!").is_none());
        assert!(
            try_granite("<tool_call>{\"name\": \"fn\", \"arguments\": {}}</tool_call>").is_none()
        );
    }

    // -- DeepSeek V3/R1 (Hermes format with think blocks) --

    #[test]
    fn deepseek_tool_call_via_hermes_with_think_block() {
        // DeepSeek wraps reasoning in <think> blocks; tool calls use <tool_call> tags.
        // Hermes parser handles the tool calls after think-block stripping in parser.rs.
        let text = r#"<think>Let me call the weather API.</think>
<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>"#;

        // Strip think blocks as the parser does
        let cleaned: String = {
            let mut result = String::new();
            let mut remaining = text;
            while let Some(start) = remaining.find("<think>") {
                result.push_str(&remaining[..start]);
                if let Some(end) = remaining[start..].find("</think>") {
                    remaining = &remaining[start + end + "</think>".len()..];
                } else {
                    remaining = "";
                }
            }
            result.push_str(remaining);
            result
        };

        let result = try_hermes(cleaned.trim()).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert!(result.tool_calls[0].arguments.contains("Beijing"));
    }

    // -- Gemma 4 --

    #[test]
    fn test_gemma4_no_args() {
        let text = "<|tool_call>call:get_time{}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_time");
        assert_eq!(result.tool_calls[0].arguments, "{}");
        assert_eq!(result.format, Some(ToolCallFormat::Gemma4));
    }

    #[test]
    fn test_gemma4_string_arg() {
        let text = "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
    }

    #[test]
    fn test_gemma4_mixed_types() {
        // string + number + boolean
        let text =
            "<|tool_call>call:search{query:<|\"|>rust<|\"|>,limit:10,active:true}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 10);
        assert_eq!(args["active"], true);
    }

    #[test]
    fn test_gemma4_multiple_calls() {
        let text = "<|tool_call>call:get_time{}<tool_call|><|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_time");
        assert_eq!(result.tool_calls[1].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[1].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_gemma4_with_content_before() {
        let text = "Let me check the weather.\n<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_gemma4_empty_string_value() {
        let text = "<|tool_call>call:fn{key:<|\"|><|\"|>}<tool_call|>";
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["key"], "");
    }

    #[test]
    fn test_gemma4_string_with_special_chars() {
        // Strings containing double quotes and backslashes
        let text = r#"<|tool_call>call:fn{msg:<|"|>hello "world" and \path\<|"|>}<tool_call|>"#;
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["msg"], r#"hello "world" and \path\"#);
    }

    #[test]
    fn test_gemma4_string_with_comma() {
        // Commas inside string values must not split the argument list
        let text = r#"<|tool_call>call:fn{msg:<|"|>hello, world<|"|>,count:3}<tool_call|>"#;
        let result = try_gemma4(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["msg"], "hello, world");
        assert_eq!(args["count"], 3);
    }

    #[test]
    fn test_gemma4_no_match() {
        // Regular Hermes tag should not match
        assert!(try_gemma4(r#"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#).is_none());
        // Plain text should not match
        assert!(try_gemma4("Hello, world!").is_none());
    }

    // -- Generic JSON --

    #[test]
    fn generic_json_single() {
        let text = r#"{"name": "fn1", "arguments": {"a": 1}}"#;
        let result = try_generic_json(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "fn1");
    }

    #[test]
    fn generic_json_array() {
        let text = r#"[{"name": "a", "arguments": {}}, {"name": "b", "parameters": {"x": 1}}]"#;
        let result = try_generic_json(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
    }

    #[test]
    fn generic_json_not_a_tool_call() {
        assert!(try_generic_json(r#"{"key": "value"}"#).is_none());
    }

    // -- MiniMax M2 --

    #[test]
    fn minimax_m2_single_tool_call() {
        // From upstream test_tool_parsing.py: single invoke with a string parameter
        let text = "<invoke name=\"get_current_temperature\">\n<parameter name=\"location\">London</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_current_temperature");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "London");
        assert_eq!(result.format, Some(ToolCallFormat::MinimaxM2));
    }

    #[test]
    fn minimax_m2_parallel_tool_calls() {
        // From upstream test_minimax_m2: two parallel invocations in one response
        let text = "<invoke name=\"search\">\n<parameter name=\"query\">weather</parameter>\n</invoke>\n<invoke name=\"read_file\">\n<parameter name=\"path\">/tmp/test.txt</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        let args0: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args0["query"], "weather");
        assert_eq!(result.tool_calls[1].name, "read_file");
        let args1: serde_json::Value =
            serde_json::from_str(&result.tool_calls[1].arguments).unwrap();
        assert_eq!(args1["path"], "/tmp/test.txt");
    }

    #[test]
    fn minimax_m2_numeric_params() {
        // From upstream test_parsers: multiply with numeric a and b
        let text = "<invoke name=\"multiply\">\n<parameter name=\"a\">12234585</parameter>\n<parameter name=\"b\">48838483920</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "multiply");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        // Numeric values must be coerced to numbers, not kept as strings
        assert_eq!(args["a"], 12234585i64);
        assert_eq!(args["b"], 48838483920i64);
    }

    #[test]
    fn minimax_m2_boolean_param() {
        let text = "<invoke name=\"fn\">\n<parameter name=\"active\">true</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["active"], true);
    }

    #[test]
    fn minimax_m2_null_param() {
        let text = "<invoke name=\"fn\">\n<parameter name=\"val\">null</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["val"], serde_json::Value::Null);
    }

    #[test]
    fn minimax_m2_json_object_param() {
        let text = "<invoke name=\"fn\">\n<parameter name=\"config\">{\"key\": \"val\", \"n\": 1}</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        let config = &args["config"];
        assert_eq!(config["key"], "val");
        assert_eq!(config["n"], 1);
    }

    #[test]
    fn minimax_m2_no_match_plain_text() {
        assert!(try_minimax_m2("Hello, world!").is_none());
    }

    #[test]
    fn minimax_m2_no_match_hermes_format() {
        assert!(
            try_minimax_m2(r#"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#).is_none()
        );
    }

    #[test]
    fn minimax_m2_single_quotes_name() {
        // Name quoted with single quotes (edge case)
        let text = "<invoke name='search'>\n<parameter name='query'>rust</parameter>\n</invoke>";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls[0].name, "search");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["query"], "rust");
    }

    #[test]
    fn minimax_m2_missing_close_invoke() {
        // No </invoke> closing tag: still parses what it can
        let text = "<invoke name=\"search\">\n<parameter name=\"query\">rust</parameter>\n";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn minimax_m2_lone_quote_in_name_does_not_panic() {
        // A model emitting `<invoke name=">...` previously caused a runtime
        // panic in `extract_quoted_name` (`s[1..s.len() - 1]` with len=1 is
        // `1..0`).  The fix gates the slice on `s.len() >= 2` so the lone
        // quote is treated as a bare (invalid) name and the parser either
        // skips that block or returns no calls — but never panics.
        let text = "<invoke name=\">\n<parameter name=\"k\">v</parameter>\n</invoke>";
        // Must not panic.
        let _ = try_minimax_m2(text);

        // Same check for a single apostrophe.
        let text = "<invoke name='>\n<parameter name=\"k\">v</parameter>\n</invoke>";
        let _ = try_minimax_m2(text);

        // Same check inside a parameter name.
        let text = "<invoke name=\"fn\">\n<parameter name=\">v</parameter>\n</invoke>";
        let _ = try_minimax_m2(text);
    }

    #[test]
    fn minimax_m2_trailing_malformed_invoke_preserves_prior_calls() {
        // A trailing `<invoke name=` without a closing `>` previously caused
        // the entire parse to abort via `?`, discarding the already-parsed
        // first call.  Same fix pattern as
        // `functionary_v31_malformed_trailing_tag_preserves_prior_calls`.
        let text = "<invoke name=\"get_weather\">\n<parameter name=\"loc\">Paris</parameter>\n</invoke><invoke name=\"broken_no_close";
        let result = try_minimax_m2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["loc"], "Paris");
    }

    #[test]
    fn minimax_m2_malformed_parameter_no_quadratic_blowup() {
        // Adversarial body filled with `<parameter name=` fragments that never
        // close with '>'. Before the fix, each loop iteration re-scanned the
        // entire remaining buffer for '>' (O(N) per iteration, N iterations =
        // O(N^2)) so a 1 MB run pinned a thread for ~15 s. After the fix, the
        // missing-'>' branch breaks out of the loop on the first hit.
        //
        // Test on a size that would have taken multiple seconds before the
        // fix; with the fix this completes in milliseconds.
        let mut body = String::from("<invoke name=\"fn\">");
        for _ in 0..50_000 {
            body.push_str("<parameter name=hello");
        }
        body.push_str("</invoke>");

        let start = std::time::Instant::now();
        let result = try_minimax_m2(&body);
        let elapsed = start.elapsed();

        // Should complete in well under one second on any reasonable hardware.
        assert!(
            elapsed.as_secs() < 2,
            "parsing took {elapsed:?}; suggests O(N^2) regression"
        );
        // The first malformed `<parameter name=` immediately breaks out, so
        // no parameters are parsed but the invoke itself produces an empty
        // arguments object.
        let calls = result.unwrap().tool_calls;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fn");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn minimax_m2_caps_excessive_parallel_calls() {
        // 10_000 well-formed parallel `<invoke>` blocks should be capped at
        // MINIMAX_M2_MAX_CALLS to bound memory amplification.
        let one = "<invoke name=\"x\"><parameter name=\"a\">1</parameter></invoke>";
        let huge = one.repeat(10_000);
        let result = try_minimax_m2(&huge).unwrap();
        assert!(
            result.tool_calls.len() <= MINIMAX_M2_MAX_CALLS,
            "expected <= {}, got {}",
            MINIMAX_M2_MAX_CALLS,
            result.tool_calls.len()
        );
        assert_eq!(result.tool_calls.len(), MINIMAX_M2_MAX_CALLS);
    }

    #[test]
    fn minimax_m2_caps_excessive_parameters_per_call() {
        // 10_000 well-formed `<parameter>` tags inside a single invoke must
        // be capped to bound per-call memory.
        let mut body = String::from("<invoke name=\"fn\">");
        for i in 0..10_000 {
            body.push_str(&format!("<parameter name=\"k{i}\">v</parameter>"));
        }
        body.push_str("</invoke>");
        let result = try_minimax_m2(&body).unwrap();
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        let obj = args.as_object().unwrap();
        assert!(
            obj.len() <= MINIMAX_M2_MAX_PARAMS_PER_CALL,
            "expected <= {}, got {}",
            MINIMAX_M2_MAX_PARAMS_PER_CALL,
            obj.len()
        );
        assert_eq!(obj.len(), MINIMAX_M2_MAX_PARAMS_PER_CALL);
    }

    // -- Harmony (GPT-OSS) --

    fn harmony_args(call: &ParsedToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).expect("arguments must be valid JSON")
    }

    #[test]
    fn harmony_single_tool_call_matches_issue_repro() {
        // The exact leaked output from the issue: an analysis channel followed
        // by a truncated commentary tool call (no trailing `<|call|>`).
        let text = "<|channel|>analysis<|message|>The user asks to read /etc/hosts and \
                    summarize it. We have a tool to read file. We'll use read_file.<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.read_file \
                    <|constrain|>json<|message|>{\n  \"path\": \"/etc/hosts\"\n}";
        let result = try_harmony(text).unwrap();
        assert_eq!(result.format, Some(ToolCallFormat::Harmony));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "read_file");
        assert_eq!(harmony_args(&result.tool_calls[0])["path"], "/etc/hosts");
        // Pretty-printed body is compacted.
        assert_eq!(result.tool_calls[0].arguments, r#"{"path":"/etc/hosts"}"#);
        // Analysis channel routes to reasoning, not content.
        assert_eq!(result.content, "");
        assert!(
            result
                .reasoning_content
                .as_deref()
                .unwrap()
                .contains("read_file")
        );
    }

    #[test]
    fn harmony_strips_functions_namespace() {
        let text = "<|channel|>commentary to=functions.get_weather <|constrain|>json\
                    <|message|>{\"location\": \"Tokyo\"}<|call|>";
        let result = try_harmony(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(harmony_args(&result.tool_calls[0])["location"], "Tokyo");
    }

    #[test]
    fn harmony_constrain_marker_optional() {
        // Some outputs omit `<|constrain|>json` and go straight to `<|message|>`.
        let text = "<|channel|>commentary to=functions.get_weather<|message|>\
                    {\"location\": \"Paris\"}<|call|>";
        let result = try_harmony(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(harmony_args(&result.tool_calls[0])["location"], "Paris");
    }

    #[test]
    fn harmony_multiple_sequential_tool_calls() {
        let text = "<|channel|>analysis<|message|>Plan the calls.<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.search \
                    <|constrain|>json<|message|>{\"query\": \"rust\"}<|call|>\
                    <|start|>assistant<|channel|>commentary to=functions.calc \
                    <|constrain|>json<|message|>{\"expr\": \"2+2\"}<|call|>";
        let result = try_harmony(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(harmony_args(&result.tool_calls[0])["query"], "rust");
        assert_eq!(result.tool_calls[1].name, "calc");
        assert_eq!(harmony_args(&result.tool_calls[1])["expr"], "2+2");
        assert_eq!(result.reasoning_content.as_deref(), Some("Plan the calls."));
    }

    #[test]
    fn harmony_final_channel_is_content_not_tool_call() {
        // Tools were available but the model answered directly via `final`.
        let text = "<|channel|>analysis<|message|>Simple question.<|end|>\
                    <|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>";
        let result = try_harmony(text).unwrap();
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content, "The answer is 42.");
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("Simple question.")
        );
    }

    #[test]
    fn harmony_analysis_routes_to_reasoning() {
        let text = "<|channel|>analysis<|message|>Deliberating carefully.<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.noop \
                    <|constrain|>json<|message|>{}<|call|>";
        let result = try_harmony(text).unwrap();
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("Deliberating carefully.")
        );
        assert_eq!(result.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn harmony_content_never_leaks_channel_markup() {
        let text = "<|channel|>analysis<|message|>reasoning<|end|>\
                    <|start|>assistant<|channel|>commentary to=functions.fn \
                    <|constrain|>json<|message|>{\"a\": 1}<|call|>";
        let result = try_harmony(text).unwrap();
        assert!(!result.content.contains("<|channel|>"));
        assert!(!result.content.contains("<|message|>"));
        assert!(!result.content.contains("to=functions"));
    }

    #[test]
    fn harmony_no_match_plain_text() {
        assert!(try_harmony("Hello, how can I help you?").is_none());
    }

    #[test]
    fn harmony_no_match_gemma_single_pipe_channel() {
        // Gemma 4 uses single-pipe `<|channel>` / `<channel|>`; Harmony must not
        // claim it (that would steal reasoning the Gemma path already handles).
        let text = "<|channel>thought\nreasoning here<channel|>the answer";
        assert!(try_harmony(text).is_none());
    }

    #[test]
    fn harmony_no_match_hermes_tool_call() {
        assert!(try_harmony(r#"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#).is_none());
    }

    #[test]
    fn harmony_malformed_missing_message_marker_does_not_panic() {
        // A channel header with no following `<|message|>`: must not panic and
        // must not produce a bogus call.
        let text = "<|channel|>commentary to=functions.fn <|constrain|>json";
        // `contains("<|message|>")` is false, so this is not claimed at all.
        assert!(try_harmony(text).is_none());
    }

    #[test]
    fn harmony_recipient_stops_at_constrain_without_space() {
        // Recipient immediately abutting `<|constrain|>` (no separating space).
        let text = "<|channel|>commentary to=functions.fn<|constrain|>json\
                    <|message|>{\"k\": \"v\"}<|call|>";
        let result = try_harmony(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "fn");
    }

    // -- Kimi K2 --

    #[test]
    fn kimi_k2_single_call_with_wrapper() {
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"Paris\"}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.format, Some(ToolCallFormat::KimiK2));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn kimi_k2_multiple_calls_in_one_section() {
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"rust\"}<|tool_call_end|>\
                     <|tool_call_begin|>functions.calc:1<|tool_call_argument_begin|>{\"expr\": \"2+2\"}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[1].name, "calc");
        let args0: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args0["query"], "rust");
        let args1: serde_json::Value =
            serde_json::from_str(&result.tool_calls[1].arguments).unwrap();
        assert_eq!(args1["expr"], "2+2");
    }

    #[test]
    fn kimi_k2_missing_per_call_wrappers_parses_whole_section_as_one_call() {
        // No `<|tool_call_begin|>` / `<|tool_call_end|>` wrappers inside the
        // section: the whole section body is a single call.
        let text = "<|tool_calls_section_begin|>functions.get_time:0<|tool_call_argument_begin|>{}<|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_time");
        assert_eq!(result.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn kimi_k2_malformed_json_args_falls_back_to_raw_string() {
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.broken:0<|tool_call_argument_begin|>{not valid json at all<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "broken");
        // Falls back to the raw (trimmed) argument text rather than panicking
        // or dropping the call.
        assert_eq!(result.tool_calls[0].arguments, "{not valid json at all");
    }

    #[test]
    fn kimi_k2_name_extracted_from_functions_prefixed_id() {
        // The tricky part of the name regex: `functions.NAME:INDEX` must
        // yield the bare middle segment `NAME`, discarding both the
        // `functions.` namespace and the `:INDEX` suffix.
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn kimi_k2_name_extracted_without_functions_prefix() {
        // The `functions.` namespace prefix is optional per the spec regex.
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>get_weather:3<|tool_call_argument_begin|>{}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn kimi_k2_no_match_without_section_marker() {
        assert!(try_kimi_k2("Hello, world!").is_none());
        assert!(try_kimi_k2(r#"<tool_call>{"name": "fn", "arguments": {}}</tool_call>"#).is_none());
    }

    #[test]
    fn kimi_k2_content_before_section_preserved() {
        let text = "Let me check the weather.\
                     <|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Tokyo\"}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert!(result.content.contains("check the weather"));
    }

    #[test]
    fn kimi_k2_missing_section_end_marker_still_parses() {
        // No closing `<|tool_calls_section_end|>`: take the rest of the text.
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Seoul\"}<|tool_call_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn kimi_k2_loose_literal_python_repr_arguments_coerced() {
        // Arguments spelled as a Python dict repr (single-quoted keys/values,
        // `True`/`False`/`None`) fail strict JSON parsing, so this exercises
        // the second parse attempt in `kimi_k2_arguments`: the loose-literal
        // reparse via `kimi_k2_loosen_literal` / `replace_python_literals`.
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.set_profile:0<|tool_call_argument_begin|>{'active': True, 'name': 'Paris', 'note': None}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "set_profile");
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["active"], true);
        assert_eq!(args["name"], "Paris");
        assert!(args["note"].is_null());
    }

    #[test]
    fn kimi_k2_loose_literal_word_boundary_guard_preserves_identifier_substrings() {
        // `NoneType` contains the literal token `None` as a substring, but
        // `replace_python_literals` only replaces whole-word matches, so it
        // must survive the loose-literal reparse untouched while the
        // standalone `True` is still coerced to JSON `true`.
        let text = "<|tool_calls_section_begin|>\
                     <|tool_call_begin|>functions.describe:0<|tool_call_argument_begin|>{'active': True, 'kind': 'NoneType'}<|tool_call_end|>\
                     <|tool_calls_section_end|>";
        let result = try_kimi_k2(text).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&result.tool_calls[0].arguments).unwrap();
        assert_eq!(args["active"], true);
        assert_eq!(args["kind"], "NoneType");
    }
}
