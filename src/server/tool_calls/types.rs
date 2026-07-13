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

//! Internal types for tool call parsing and detection.
//!
//! Used by: tool_calls::parser, tool_calls::formats, routes/chat

/// A parsed tool call extracted from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// Function name
    pub name: String,
    /// Arguments as a JSON string
    pub arguments: String,
}

/// The format detected in the model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallFormat {
    /// Hermes/Qwen: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
    Hermes,
    /// Llama 3.x: `{"name": ..., "parameters": ...}` possibly with `<|python_tag|>`
    Llama3,
    /// Mistral Nemo: `[TOOL_CALLS] [{"name": ..., "arguments": ...}]`
    MistralNemo,
    /// Mistral (bracketed): `[TOOL_CALLS]NAME[ARGS]{json}`, used by Ministral
    /// 2410 and later, Mistral Small 3, Magistral, and Devstral.
    Mistral,
    /// Functionary v3.1: `<function=name>{"key": "val"}</function>`
    FunctionaryV31,
    /// Functionary v3.2: `>>>name\n{"key": "val"}`
    FunctionaryV32,
    /// Command R7B: `Action: fn_name\nAction Input: {"key": "val"}`
    CommandR,
    /// Granite 3.3: `<response><tool_call>...</tool_call></response>`
    Granite,
    /// Gemma 4: `<|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>`
    Gemma4,
    /// Function-calling Gemma: `<start_function_call>call:name{key:value,
    /// key2:<escape>text<escape>}<end_function_call>`. Shares the
    /// `call:name{...}` call syntax with [`Gemma4`](ToolCallFormat::Gemma4)
    /// but uses distinct wrapper markers and a different string-escaping
    /// convention (`<escape>...<escape>` rather than `<|"|>...<|"|>`).
    FunctionGemma,
    /// Generic JSON: raw `{"name": ..., "arguments": ...}` object
    GenericJson,
    /// MiniMax M2: `<invoke name="fn_name"><parameter name="k">v</parameter></invoke>`
    MinimaxM2,
    /// MiniMax M3: a namespaced XML dialect where every tag is prefixed with
    /// the literal namespace token `]<]minimax[>[`, e.g.
    /// `]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="NAME">`
    /// `]<]minimax[>[<param>value]<]minimax[>[</param>]<]minimax[>[</invoke>`
    /// `]<]minimax[>[</tool_call>`. Parameters nest (repeated tags and explicit
    /// `item` tags form arrays; mixed text plus child elements route the text to
    /// a `$text` field) and values are coerced against the request tool schema.
    MinimaxM3,
    /// Qwen3-Coder: `<tool_call><function=name><parameter=key>val</parameter></function></tool_call>`
    ///
    /// Named for the family that introduced it (the format is spelled out in
    /// the Qwen3-Coder chat template), and matches the parser name vLLM and
    /// SGLang use (`--tool-call-parser qwen3_coder`). The parser keys on the
    /// markup, not the model, so it also handles newer Qwen models that share
    /// this template (Qwen3.5 / Qwen3.6).
    Qwen3Coder,
    /// Harmony (GPT-OSS): channel-structured output where a tool call is a
    /// `commentary` message targeting `to=functions.NAME`, e.g.
    /// `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{…}<|call|>`.
    /// The `analysis` channel carries chain-of-thought (routed to
    /// `reasoning_content`) and the `final` channel carries the visible answer.
    Harmony,
    /// Kimi K2: sectioned marker format `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`
    /// wrapping one or more `<|tool_call_begin|>...<|tool_call_end|>` calls
    /// (the per-call wrapper is optional; an unwrapped section body is parsed
    /// as a single call). Each call is shaped
    /// `functions.NAME:INDEX<|tool_call_argument_begin|>{json arguments}`,
    /// e.g. `functions.get_weather:0<|tool_call_argument_begin|>{"location": "Paris"}`.
    KimiK2,
    /// Pythonic (Llama-family pythonic tool use, some Nemotron templates):
    /// a bracketed Python-like call, e.g. `[get_weather(city="Paris", days=2)]`,
    /// wrapped in `<|tool_call_start|>` / `<|tool_call_end|>` markers (the
    /// corresponding tool-list template marker is `<|tool_list_start|>`).
    /// Single call per message.
    Pythonic,
}

/// Result of parsing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolCallParseResult {
    /// The detected format (if any)
    pub format: Option<ToolCallFormat>,
    /// Extracted tool calls
    pub tool_calls: Vec<ParsedToolCall>,
    /// Any text content before/outside the tool calls (may be empty)
    pub content: String,
    /// Chain-of-thought / scratchpad text that the format carries inline and
    /// that should surface as `reasoning_content` rather than `content`.
    ///
    /// Only formats whose reasoning is interleaved with the tool call in a
    /// single stream populate this (currently Harmony / GPT-OSS, whose
    /// `analysis` channel is the reasoning). For families where reasoning is a
    /// separable `<think>` / `<|channel>` block, the routes recover it from the
    /// raw text via the shared `StreamFilter`, so this stays `None`.
    pub reasoning_content: Option<String>,
}

impl ToolCallParseResult {
    /// Returns true if tool calls were found.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Create an empty result (no tool calls found).
    pub fn none(content: String) -> Self {
        Self {
            format: None,
            tool_calls: Vec::new(),
            content,
            reasoning_content: None,
        }
    }
}
