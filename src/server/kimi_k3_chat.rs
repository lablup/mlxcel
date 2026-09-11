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
//
// The XTML grammar rendered here is derived from `encoding_k3.py` in
// moonshotai/Kimi-K3
// (https://huggingface.co/moonshotai/Kimi-K3/blob/main/encoding_k3.py) and its
// caller `tokenization_kimi.TikTokenTokenizer.apply_chat_template`.

//! Native Kimi K3 XTML chat renderer.
//!
//! Kimi K3 ships no Jinja chat template. Its chat format is XTML, a tag
//! language built out of four control tokens (`<|open|>`, `<|close|>`,
//! `<|sep|>`, `<|end_of_msg|>`) that the reference implementation renders in
//! code. This module is the Rust port of that code path, and it emits token
//! **ids** rather than a string: tag names, attribute names and attribute
//! values are ordinary text segments, and only the four structural markers
//! become control ids, so no amount of `<|open|>` written into a user message
//! can inject structure into the prompt.
//!
//! [`K3Rendered::text`] is the same string the reference
//! `apply_chat_template(tokenize=False)` returns. It is not what gets
//! tokenized (the ids are authoritative); it is the diagnostic rendering, the
//! prompt-cache key material, and the string the primed-open-thinking check
//! reads.
//!
//! Used by: `server::chat_request` (chat route), `commands::chat` (CLI REPL).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::multimodal::kimi_k3_prompt::{
    MEDIA_BEGIN, MEDIA_CONTENT, MEDIA_END, MEDIA_PAD, image_block_ids, image_label,
};
use crate::tokenizer::{KimiK3ControlIds, MlxcelTokenizer};
use crate::vision::kimi_k3_vl::KimiK3MediaTokenIds;
use crate::vision::processors::kimi_k3::{KimiK3NavitConfig, image_prompt_text};

use super::types::Role;
use super::types::request::{ContentPart, Message, MessageContent, Tool};

/// The image placeholder Kimi K3's own config declares
/// (`config.json: image_placeholder`).
pub const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";

/// The only three thinking-effort values the reference accepts.
///
/// The rendered message text lists `medium` as well, but
/// `encoding_k3._VALID_THINKING_EFFORTS` does not contain it, so `medium` is a
/// hard error rather than a silently accepted value.
pub const VALID_THINKING_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// The default thinking effort, applied by the reference
/// `apply_chat_template` via `kwargs.setdefault("thinking_effort", "max")`.
pub const DEFAULT_THINKING_EFFORT: &str = "max";

/// Map a portable OpenAI-style `reasoning_effort` onto the three levels this
/// family renders, or `None` when the name is not one mlxcel forwards.
///
/// Only the portable field goes through here. A `thinking_effort` kwarg is
/// K3's own name, so an unsupported value there stays a hard error: the caller
/// asked for this family by name and named a level it does not have. The
/// portable field is different, because a client that never heard of K3 sends
/// OpenAI's ladder, whose default level `medium` is exactly the one
/// `VALID_THINKING_EFFORTS` omits. Erroring on the default effort of the
/// portable API would make plain `reasoning_effort` unusable on this family.
///
/// The map is monotone and keeps the two names both ladders share, so `low`
/// and `high` mean what they say. `minimal` rounds down to `low`; `medium`
/// rounds up to `high` rather than down, because K3's own default is `max` and
/// rounding toward it is the smaller departure.
///
/// Used by: chat_request
pub fn clamp_portable_reasoning_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "minimal" | "low" => Some("low"),
        "medium" | "high" => Some("high"),
        "max" => Some("max"),
        _ => None,
    }
}

/// One pre-encoded image prompt: the ids to splice in, plus the text form for
/// the diagnostic rendering.
///
/// The reference encodes image prompts with `allow_special=True` (they carry
/// `<|media_begin|>` / `<|media_pad|>` / `<|media_end|>` control tokens), which
/// is why this is an id sequence rather than a string. Issue #1338 always
/// passes `None`; #1342 fills these in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct K3ImagePrompt {
    pub ids: Vec<i32>,
    pub text: String,
}

/// Everything the renderer takes besides the messages and the tool list.
#[derive(Debug, Clone, Default)]
pub struct K3RenderOptions<'a> {
    /// Append the assistant generation prompt (`<|open|>message role="assistant"<|sep|>`
    /// then the opening `think` or `response` tag).
    pub add_generation_prompt: bool,
    /// Render the structural `think` channel. The reference default is `true`.
    pub thinking: bool,
    /// `low` / `high` / `max`, or `None` to omit the thinking-effort message
    /// entirely. Ignored when `thinking` is false.
    pub thinking_effort: Option<String>,
    /// `required` / `none` render a tool-choice system message; every other
    /// value (including `auto` and a named function) renders nothing.
    pub tool_choice: Option<String>,
    /// The request's `response_format` object, verbatim.
    pub response_format: Option<&'a Value>,
    /// Pre-encoded image prompts, consumed in order by image content parts and
    /// by `<|kimi_image_placeholder|>` occurrences in text.
    pub image_prompts: Option<&'a [K3ImagePrompt]>,
}

impl K3RenderOptions<'_> {
    /// The reference defaults: generation prompt on, thinking on, effort `max`.
    pub fn reference_defaults() -> Self {
        Self {
            add_generation_prompt: true,
            thinking: true,
            thinking_effort: Some(DEFAULT_THINKING_EFFORT.to_string()),
            tool_choice: None,
            response_format: None,
            image_prompts: None,
        }
    }
}

/// A rendered K3 prompt: the authoritative token ids plus the text form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct K3Rendered {
    pub ids: Vec<i32>,
    pub text: String,
}

/// Fixed literals every chat turn re-renders. Encoding them once at
/// construction keeps the pre-tokenization regex off the per-request scaffold;
/// only genuinely variable text (message bodies, tool names, JSON schemas) is
/// encoded per request.
const CACHED_LITERALS: &[&str] = &[
    // Tag names.
    "message",
    "think",
    "response",
    "tools",
    "call",
    "argument",
    "json",
    // Attribute names (rendered with their leading space).
    " role",
    " name",
    " type",
    " tool",
    " index",
    " key",
    // Attribute punctuation.
    "=\"",
    "\"",
    // Role and type attribute values.
    "user",
    "system",
    "assistant",
    "tool",
    "tool-declare",
    "thinking-effort",
    "tool-choice",
    "response-format",
    "object",
    // XTML argument types.
    "string",
    "number",
    "boolean",
    "null",
    "array",
    // Fixed system-message bodies.
    TOOL_CHOICE_REQUIRED_BODY,
    TOOL_CHOICE_NONE_BODY,
    RESPONSE_FORMAT_JSON_OBJECT_BODY,
];

const TOOL_CHOICE_REQUIRED_BODY: &str = "The system is invoked with `tool_choice=required`.\n\
     You MUST call tools in the next message.";
const TOOL_CHOICE_NONE_BODY: &str = "The system is invoked with `tool_choice=none`.\n\
     You MUST NOT call any tools in the next message.";
const RESPONSE_FORMAT_JSON_OBJECT_BODY: &str = "The system is invoked with `response_format=json_object`.\n\
     Your response must be raw JSON data without markdown code blocks (```json) or any \
     additional formatting.";
const RESPONSE_FORMAT_JSON_SCHEMA_PREFIX: &str = "The system is invoked with `response_format=json_schema`.\n\
     Your response must be raw JSON data without markdown code blocks (```json) or any \
     additional formatting.\n\
     The JSON data must match the following schema:\n";

/// The native XTML renderer, bound to one loaded Kimi K3 tokenizer.
pub struct KimiK3Renderer {
    tokenizer: Arc<MlxcelTokenizer>,
    control: KimiK3ControlIds,
    literals: HashMap<&'static str, Vec<u32>>,
    /// The four media control ids, when the vocabulary names them all.
    /// `None` makes every image request fail with a named error rather than
    /// render a block the model cannot read.
    media: Option<KimiK3MediaTokenIds>,
    /// The navit resize parameters that size each image's `<|media_pad|>`
    /// run. The published values unless `preprocessor_config.json` says
    /// otherwise (`with_navit_config`).
    navit: KimiK3NavitConfig,
}

impl std::fmt::Debug for KimiK3Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KimiK3Renderer")
            .field("control", &self.control)
            .finish_non_exhaustive()
    }
}

impl KimiK3Renderer {
    /// Bind a renderer to `tokenizer`, or `None` when it is not a Kimi K3
    /// tiktoken vocabulary.
    pub fn new(tokenizer: Arc<MlxcelTokenizer>) -> Option<Self> {
        let control = tokenizer.kimi_k3_control_ids()?;
        let tiktoken = tokenizer.tiktoken()?;
        let mut literals = HashMap::with_capacity(CACHED_LITERALS.len());
        for literal in CACHED_LITERALS {
            literals.insert(*literal, tiktoken.encode_text(literal).ok()?);
        }
        let media = match (
            tiktoken.control_id(MEDIA_BEGIN),
            tiktoken.control_id(MEDIA_CONTENT),
            tiktoken.control_id(MEDIA_PAD),
            tiktoken.control_id(MEDIA_END),
        ) {
            (Some(begin), Some(content), Some(pad), Some(end)) => Some(KimiK3MediaTokenIds {
                begin: begin as i32,
                content: content as i32,
                pad: pad as i32,
                end: end as i32,
            }),
            _ => None,
        };
        Some(Self {
            tokenizer,
            control,
            literals,
            media,
            navit: KimiK3NavitConfig::default(),
        })
    }

    /// Size the image prompts with the checkpoint's own navit parameters
    /// (`preprocessor_config.json` `media_proc_cfg`) instead of the published
    /// defaults.
    pub fn with_navit_config(mut self, navit: KimiK3NavitConfig) -> Self {
        self.navit = navit;
        self
    }

    /// The control ids this renderer emits.
    pub fn control_ids(&self) -> KimiK3ControlIds {
        self.control
    }

    /// The media control ids, when the vocabulary names all four.
    pub fn media_token_ids(&self) -> Option<KimiK3MediaTokenIds> {
        self.media
    }

    /// The navit parameters image prompts are sized with.
    pub fn navit_config(&self) -> KimiK3NavitConfig {
        self.navit
    }

    /// The pre-encoded prompt of one `w x h` image (#1342):
    /// `<|media_begin|>image {w}x{h}<|media_content|>` + `<|media_pad|>` x
    /// `grid_h * grid_w / 4` + `<|media_end|>`, the count from the navit rule
    /// over the original size. The worker's processor re-derives the same
    /// grid from the same pixels and refuses a run that disagrees.
    pub fn image_prompt(&self, width: u32, height: u32) -> Result<K3ImagePrompt> {
        let media = self.media.ok_or_else(|| {
            anyhow::anyhow!(
                "the loaded Kimi K3 tokenizer does not name all of {MEDIA_BEGIN}, \
                 {MEDIA_CONTENT}, {MEDIA_PAD} and {MEDIA_END}; image inputs cannot be rendered"
            )
        })?;
        let plan = self
            .navit
            .plan(width, height)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let label: Vec<i32> = self
            .encode_text(&image_label(width, height))?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        Ok(K3ImagePrompt {
            ids: image_block_ids(media, &label, plan.num_tokens as usize),
            text: image_prompt_text(width, height, plan.num_tokens),
        })
    }

    /// The tokenizer this renderer encodes text with.
    pub fn tokenizer(&self) -> &Arc<MlxcelTokenizer> {
        &self.tokenizer
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(cached) = self.literals.get(text) {
            return Ok(cached.clone());
        }
        self.tokenizer.encode_with_special(text, false, false)
    }

    /// Render one conversation into XTML ids plus the reference text form.
    pub fn render(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        opts: &K3RenderOptions<'_>,
    ) -> Result<K3Rendered> {
        let tools = tools.filter(|t| !t.is_empty());
        // A run of tool results is reordered below to follow the assistant's
        // tool-call order, while the caller's image prompts and the worker's
        // pixels are both in wire order. An image on a tool message would
        // therefore be described by the prompt of whichever tool result landed
        // in its place, and two same-sized images would pass every count check
        // while swapping their features. Refuse instead: the XTML grammar
        // renders a tool result as text and has no place to put an image.
        if let Some(index) = messages
            .iter()
            .position(|m| m.role == Role::Tool && !m.content.image_parts().is_empty())
        {
            bail!(
                "Kimi K3 chat rendering does not accept an image_url content part on a tool \
                 message (message {index}); tool results render as text"
            );
        }
        let messages = normalize_xtml_tool_result_messages(messages);

        let mut sink = Sink::new(self);
        let mut images = ImagePromptState::new(opts.image_prompts);

        if let Some(tools) = tools {
            let declared = deep_sorted_tools(tools)?;
            sink.tool_declare(&declared, false)?;
        }

        if let Some(effort) = opts.thinking_effort.as_deref()
            && opts.thinking
        {
            if !VALID_THINKING_EFFORTS.contains(&effort) {
                bail!(
                    "Unsupported thinking_effort={effort:?}; supported values are {:?}",
                    VALID_THINKING_EFFORTS
                );
            }
            sink.internal_system_message("thinking-effort", &thinking_effort_body(effort))?;
        }

        // The `index` attribute of a tool result counts within the run of tool
        // messages since the last assistant message. The reference resets it at
        // an assistant message only, so it keeps counting across an intervening
        // user turn.
        let mut assistant_tool_calls: Option<&[super::types::request::ToolCallInMessage]> = None;
        let mut tool_index = 0usize;

        for (index, message) in messages.iter().enumerate() {
            match message.role {
                Role::User => {
                    sink.open("message", &named_attrs("user", message.name.as_deref()))?;
                    sink.content(&message.content, &mut images)?;
                    sink.close("message")?;
                    sink.end_of_msg();
                }
                Role::System => {
                    sink.open("message", &named_attrs("system", message.name.as_deref()))?;
                    sink.content(&message.content, &mut images)?;
                    sink.close("message")?;
                    sink.end_of_msg();
                }
                Role::Tool => {
                    tool_index += 1;
                    let name = match message.name.as_deref() {
                        Some(name) => name.to_string(),
                        None => assistant_tool_calls
                            .and_then(|calls| calls.get(tool_index - 1))
                            .map(|call| call.function.name.clone())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Kimi K3 tool message at index {index} needs a resolvable \
                                     tool name: carry `name`, or match a preceding assistant \
                                     tool_call by order"
                                )
                            })?,
                    };
                    let idx = tool_index.to_string();
                    sink.open(
                        "message",
                        &[("role", "tool"), ("tool", &name), ("index", &idx)],
                    )?;
                    sink.content(&message.content, &mut images)?;
                    sink.close("message")?;
                    sink.end_of_msg();
                }
                Role::Assistant => {
                    assistant_tool_calls = message.tool_calls.as_deref();
                    tool_index = 0;
                    sink.open(
                        "message",
                        &named_attrs("assistant", message.name.as_deref()),
                    )?;
                    sink.assistant_body(message, opts.thinking, &mut images)?;
                    sink.close("message")?;
                    sink.end_of_msg();
                }
            }
        }

        match opts.tool_choice.as_deref() {
            Some("required") => {
                sink.internal_system_message("tool-choice", TOOL_CHOICE_REQUIRED_BODY)?
            }
            Some("none") => sink.internal_system_message("tool-choice", TOOL_CHOICE_NONE_BODY)?,
            _ => {}
        }

        match response_format_type(opts.response_format) {
            Some("json_object") => {
                sink.internal_system_message("response-format", RESPONSE_FORMAT_JSON_OBJECT_BODY)?
            }
            Some("json_schema") => {
                let schema = extract_response_schema(opts.response_format);
                let sorted = schema.map(deep_sort).unwrap_or(Value::Null);
                let body = format!(
                    "{RESPONSE_FORMAT_JSON_SCHEMA_PREFIX}```json\n{}\n```",
                    compact_json(&sorted)?
                );
                sink.internal_system_message("response-format", &body)?;
            }
            _ => {}
        }

        if opts.add_generation_prompt {
            sink.open("message", &[("role", "assistant")])?;
            sink.open(if opts.thinking { "think" } else { "response" }, &[])?;
        }

        images.assert_consumed()?;
        Ok(sink.finish())
    }
}

/// `[("role", role)]`, plus `[("name", name)]` when the message carries a
/// non-empty one. The reference renders `name` for user, system and assistant
/// messages alike.
fn named_attrs<'a>(role: &'a str, name: Option<&'a str>) -> Vec<(&'a str, &'a str)> {
    let mut attrs = vec![("role", role)];
    if let Some(name) = name.filter(|n| !n.is_empty()) {
        attrs.push(("name", name));
    }
    attrs
}

fn thinking_effort_body(effort: &str) -> String {
    format!(
        "`thinking_effort` guides on how much to think in your thinking channel (not including \
         the response channel), supported values include `low`, `medium`, `high`, and `max`.\n\
         Now the system is invoked with `thinking_effort={effort}`."
    )
}

/// Segment accumulator: ids and the equivalent text, built in one pass.
struct Sink<'a> {
    renderer: &'a KimiK3Renderer,
    ids: Vec<i32>,
    text: String,
}

impl<'a> Sink<'a> {
    fn new(renderer: &'a KimiK3Renderer) -> Self {
        Self {
            renderer,
            ids: Vec::new(),
            text: String::new(),
        }
    }

    fn finish(self) -> K3Rendered {
        K3Rendered {
            ids: self.ids,
            text: self.text,
        }
    }

    fn control(&mut self, id: u32, spelling: &str) {
        self.ids.push(id as i32);
        self.text.push_str(spelling);
    }

    /// One text segment. An empty string emits nothing, matching
    /// `encoding_k3._segment`.
    fn text(&mut self, s: &str) -> Result<()> {
        if s.is_empty() {
            return Ok(());
        }
        self.ids.extend(
            self.renderer
                .encode_text(s)?
                .into_iter()
                .map(|id| id as i32),
        );
        self.text.push_str(s);
        Ok(())
    }

    fn open(&mut self, tag: &str, attrs: &[(&str, &str)]) -> Result<()> {
        let control = self.renderer.control;
        self.control(control.open, "<|open|>");
        self.text(tag)?;
        for (key, value) in attrs {
            self.text(&format!(" {key}"))?;
            self.text("=\"")?;
            self.text(&escape_attr_value(value))?;
            self.text("\"")?;
        }
        self.control(control.sep, "<|sep|>");
        Ok(())
    }

    fn close(&mut self, tag: &str) -> Result<()> {
        let control = self.renderer.control;
        self.control(control.close, "<|close|>");
        self.text(tag)?;
        self.control(control.sep, "<|sep|>");
        Ok(())
    }

    fn end_of_msg(&mut self) {
        let control = self.renderer.control;
        self.control(control.end_of_msg, "<|end_of_msg|>");
    }

    /// `sys_internal(type, body)`: a system message whose body is trimmed.
    fn internal_system_message(&mut self, message_type: &str, body: &str) -> Result<()> {
        self.open("message", &[("role", "system"), ("type", message_type)])?;
        self.text(body.trim())?;
        self.close("message")?;
        self.end_of_msg();
        Ok(())
    }

    /// The tool-declaration system message. `dynamic` renders the lazy-loading
    /// variant the reference emits for a `system` message carrying its own
    /// `tools` field; mlxcel's wire `Message` has no such field, so only the
    /// static form is reachable from an HTTP request.
    fn tool_declare(&mut self, tools: &Value, dynamic: bool) -> Result<()> {
        let json = compact_json(tools)?;
        let body = if dynamic {
            format!(
                "## New Tools Available\nThe system dynamically extends the toolset via \
                 lazy-loading.\nYou have access to all existing and extended tools.\nHere are \
                 the specs for the extended tools.\n\n```json\n{json}\n```"
            )
        } else {
            format!(
                "# Tools\nHere are the available tools, described in JSONSchema.\n\n\
                 ```json\n{json}\n```"
            )
        };
        self.open("message", &[("role", "system"), ("type", "tool-declare")])?;
        // The reference renders the tool-declare body WITHOUT the `.strip()`
        // every other internal system message applies.
        self.text(&body)?;
        self.close("message")?;
        self.end_of_msg();
        Ok(())
    }

    fn content(
        &mut self,
        content: &MessageContent,
        images: &mut ImagePromptState<'_>,
    ) -> Result<()> {
        match content {
            MessageContent::Text(text) => self.text_with_placeholders(text, images),
            MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text } => self.text_with_placeholders(text, images)?,
                        ContentPart::ImageUrl { .. } => {
                            let prompt = images.next_image_part_prompt()?;
                            self.image_prompt(&prompt)?;
                        }
                        ContentPart::VideoUrl { .. } => {
                            bail!("Kimi K3 chat rendering does not support video content parts")
                        }
                        ContentPart::InputAudio { .. } => {
                            bail!("Kimi K3 chat rendering does not support audio content parts")
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Text that may carry `<|kimi_image_placeholder|>` occurrences.
    ///
    /// Splitting only happens when image prompts were supplied; with none, the
    /// placeholder is ordinary text, exactly as `encoding_k3._append_text`
    /// leaves it.
    fn text_with_placeholders(
        &mut self,
        text: &str,
        images: &mut ImagePromptState<'_>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if images.prompts.is_none() || !text.contains(IMAGE_PLACEHOLDER) {
            return self.text(text);
        }
        let parts: Vec<&str> = text.split(IMAGE_PLACEHOLDER).collect();
        let last = parts.len() - 1;
        for (i, part) in parts.iter().enumerate() {
            self.text(part)?;
            if i < last {
                let prompt = images.next_prompt()?;
                self.image_prompt(&prompt)?;
            }
        }
        Ok(())
    }

    fn image_prompt(&mut self, prompt: &K3ImagePrompt) -> Result<()> {
        self.ids.extend_from_slice(&prompt.ids);
        self.text.push_str(&prompt.text);
        Ok(())
    }

    fn assistant_body(
        &mut self,
        message: &Message,
        thinking: bool,
        images: &mut ImagePromptState<'_>,
    ) -> Result<()> {
        if thinking {
            // Structural: in thinking mode the channel is always present, even
            // with no reasoning to put in it. Only an empty string counts as
            // "no reasoning"; whitespace-only reasoning is rendered.
            self.open("think", &[])?;
            if let Some(reasoning) = message.reasoning.as_deref().filter(|r| !r.is_empty()) {
                self.text_with_placeholders(reasoning, images)?;
            }
            self.close("think")?;
        }

        self.open("response", &[])?;
        self.content(&message.content, images)?;
        self.close("response")?;

        let Some(tool_calls) = message.tool_calls.as_deref().filter(|c| !c.is_empty()) else {
            return Ok(());
        };
        self.open("tools", &[])?;
        for (position, call) in tool_calls.iter().enumerate() {
            let index = (position + 1).to_string();
            self.open(
                "call",
                &[("tool", call.function.name.as_str()), ("index", &index)],
            )?;
            match normalize_tool_arguments(&call.function.arguments) {
                K3Arguments::Entries(entries) => {
                    for entry in &entries {
                        self.open(
                            "argument",
                            &[("key", entry.key.as_str()), ("type", entry.value_type)],
                        )?;
                        self.text_with_placeholders(&entry.text, images)?;
                        self.close("argument")?;
                    }
                }
                K3Arguments::RawJson(raw) => {
                    self.open("json", &[("type", "object")])?;
                    self.text_with_placeholders(&raw, images)?;
                    self.close("json")?;
                }
            }
            self.close("call")?;
        }
        self.close("tools")?;
        Ok(())
    }
}

/// Consumes the caller's pre-encoded image prompts in render order.
struct ImagePromptState<'a> {
    prompts: Option<&'a [K3ImagePrompt]>,
    index: usize,
    literal: K3ImagePrompt,
}

impl<'a> ImagePromptState<'a> {
    fn new(prompts: Option<&'a [K3ImagePrompt]>) -> Self {
        Self {
            prompts,
            index: 0,
            literal: K3ImagePrompt {
                ids: Vec::new(),
                text: IMAGE_PLACEHOLDER.to_string(),
            },
        }
    }

    /// The next image prompt, or the literal placeholder text when the caller
    /// supplied none (`_ImagePromptState.next_prompt` with `image_prompts is
    /// None`). The literal form carries no ids of its own; the caller renders
    /// its text through the ordinary text path.
    fn next_prompt(&mut self) -> Result<K3ImagePrompt> {
        let Some(prompts) = self.prompts else {
            return Ok(self.literal.clone());
        };
        let prompt = prompts
            .get(self.index)
            .ok_or_else(|| anyhow::anyhow!("More image placeholders than image prompts"))?;
        self.index += 1;
        Ok(prompt.clone())
    }

    /// The next image prompt for an `image_url` content part.
    ///
    /// The literal fallback of [`Self::next_prompt`] belongs to the text
    /// path, where `<|kimi_image_placeholder|>` is ordinary text that the
    /// caller re-encodes. An `image_url` part has no such text spelling: with
    /// no prompt supplied it would put the placeholder into
    /// [`K3Rendered::text`] and nothing into the ids, dropping the image from
    /// the prompt the model actually reads. Refuse instead (#1342).
    fn next_image_part_prompt(&mut self) -> Result<K3ImagePrompt> {
        if self.prompts.is_none() {
            bail!(
                "Kimi K3: an image_url content part needs a pre-encoded image prompt, and the \
                 renderer was given none"
            );
        }
        self.next_prompt()
    }

    fn assert_consumed(&self) -> Result<()> {
        let Some(prompts) = self.prompts else {
            return Ok(());
        };
        if self.index != prompts.len() {
            bail!(
                "image prompt count {} != consumed placeholder count {}",
                prompts.len(),
                self.index
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tool-result reordering
// ---------------------------------------------------------------------------

/// Re-sort each run of consecutive tool messages into the preceding
/// assistant's `tool_calls` order, and align each matched message's `name`
/// with the call it matched.
///
/// Port of `encoding_k3.normalize_xtml_tool_result_messages`. A run in which
/// any message's `tool_call_id` fails to match is left exactly as it came in.
/// Side-effect free with respect to the caller's slice: the input is borrowed
/// and every message in the returned vector is a clone, so a rewritten `name`
/// never reaches the request the route still holds.
///
/// The clone is of the whole message list, content strings included, which is
/// one copy of the conversation per render. That is well under the cost of
/// tokenizing it, and it buys the reordering a single straightforward pass;
/// revisit it only if a profile puts this path on top.
pub fn normalize_xtml_tool_result_messages(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    // `tool_call_id` -> (1-based position, function name). Every entry advances
    // the position, even an id-less one; duplicate ids keep the first.
    let mut index: HashMap<&str, (usize, &str)> = HashMap::new();
    let mut i = 0usize;

    while i < messages.len() {
        let message = &messages[i];
        if message.role == Role::Assistant {
            index.clear();
            if let Some(calls) = message.tool_calls.as_deref() {
                for (position, call) in calls.iter().enumerate() {
                    index
                        .entry(call.id.as_str())
                        .or_insert((position + 1, call.function.name.as_str()));
                }
            }
            out.push(message.clone());
            i += 1;
            continue;
        }
        if message.role != Role::Tool {
            out.push(message.clone());
            i += 1;
            continue;
        }

        // (position, original offset, message index, matched name)
        let mut run: Vec<(usize, usize, usize, &str)> = Vec::new();
        let mut unresolved = false;
        let run_start = i;
        while i < messages.len() && messages[i].role == Role::Tool {
            let matched = messages[i]
                .tool_call_id
                .as_deref()
                .and_then(|id| index.get(id).copied());
            match matched {
                Some((position, name)) => run.push((position, i - run_start, i, name)),
                None => unresolved = true,
            }
            i += 1;
        }

        if unresolved {
            out.extend(messages[run_start..i].iter().cloned());
            continue;
        }
        run.sort_by_key(|(position, offset, _, _)| (*position, *offset));
        for (_, _, message_index, name) in run {
            // The id-matched call is authoritative, so the rendered `tool`
            // attribute cannot disagree with the reordered position.
            let mut resolved = messages[message_index].clone();
            resolved.name = Some(name.to_string());
            out.push(resolved);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Argument normalization
// ---------------------------------------------------------------------------

/// One normalized tool-call argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XtmlArgument {
    pub key: String,
    /// `boolean` | `null` | `number` | `string` | `object` | `array`.
    pub value_type: &'static str,
    /// The original JSON literal for non-string values, the decoded string for
    /// string values.
    pub text: String,
}

/// The two shapes a call's arguments render as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum K3Arguments {
    /// One `<argument key=... type=...>` per top-level key.
    Entries(Vec<XtmlArgument>),
    /// A single `<json type="object">` block carrying the raw string.
    RawJson(String),
}

/// Normalize the wire's stringified arguments.
///
/// Port of `encoding_k3.normalize_tool_arguments` for the string case: the
/// object is parsed exactly one level deep so a non-string value keeps its
/// original JSON literal (`1e2` stays `1e2`, `[1, 2, 3]` keeps its spacing) and
/// a string value is decoded. Anything after the closing brace is discarded.
/// Any string that is not a well-formed JSON object falls back to the raw
/// `<json>` block.
pub fn normalize_tool_arguments(arguments: &str) -> K3Arguments {
    if arguments.is_empty() {
        return K3Arguments::Entries(Vec::new());
    }
    match parse_arguments_object(arguments) {
        Ok(entries) => K3Arguments::Entries(entries),
        Err(_) => K3Arguments::RawJson(arguments.to_string()),
    }
}

/// Normalize already-parsed arguments (the reference's `dict` branch).
///
/// Non-string values are re-serialized with Python's `json.dumps` default
/// separators (`", "` and `": "`), which is what the reference produces for a
/// caller that passed a mapping rather than a JSON string.
pub fn normalize_tool_arguments_value(arguments: &Value) -> Option<Vec<XtmlArgument>> {
    let object = arguments.as_object()?;
    Some(
        object
            .iter()
            .map(|(key, value)| XtmlArgument {
                key: key.clone(),
                value_type: xtml_type(value),
                text: match value {
                    Value::String(s) => s.clone(),
                    other => python_default_json(other),
                },
            })
            .collect(),
    )
}

fn xtml_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

/// Parse a JSON object string one level deep, keeping each non-string value's
/// original literal text.
fn parse_arguments_object(s: &str) -> Result<Vec<XtmlArgument>, String> {
    let bytes = s.as_bytes();
    let mut idx = 0usize;

    let skip_ws = |idx: &mut usize| {
        while *idx < bytes.len() && matches!(bytes[*idx], b' ' | b'\t' | b'\n' | b'\r') {
            *idx += 1;
        }
    };

    skip_ws(&mut idx);
    if idx >= bytes.len() || bytes[idx] != b'{' {
        return Err("JSON arguments must be an object".to_string());
    }
    idx += 1;
    skip_ws(&mut idx);

    let mut parsed = Vec::new();
    if idx >= bytes.len() {
        return Err("Unexpected end of JSON object".to_string());
    }
    if bytes[idx] == b'}' {
        return Ok(parsed);
    }

    loop {
        let (key, next) = raw_decode(s, idx)?;
        idx = next;
        let Value::String(key) = key else {
            return Err("JSON object key must be a string".to_string());
        };
        skip_ws(&mut idx);
        if idx >= bytes.len() || bytes[idx] != b':' {
            return Err(format!("Expects ':' after {key}"));
        }
        idx += 1;
        skip_ws(&mut idx);

        let value_start = idx;
        let (value, next) = raw_decode(s, idx)?;
        idx = next;
        let text = match &value {
            Value::String(decoded) => decoded.clone(),
            _ => s[value_start..idx].to_string(),
        };
        parsed.push(XtmlArgument {
            key,
            value_type: xtml_type(&value),
            text,
        });

        skip_ws(&mut idx);
        let terminator = bytes.get(idx).copied();
        if terminator.is_some() {
            idx += 1;
        }
        skip_ws(&mut idx);
        match terminator {
            Some(b'}') => break,
            Some(b',') => continue,
            other => return Err(format!("Expect '}}' or ',', got {other:?}")),
        }
    }

    Ok(parsed)
}

/// Decode one JSON value starting at `start`, returning it and the byte offset
/// just past it (`json.JSONDecoder.raw_decode`).
fn raw_decode(s: &str, start: usize) -> Result<(Value, usize), String> {
    if !s.is_char_boundary(start) {
        return Err("value does not start on a character boundary".to_string());
    }
    let mut stream = serde_json::Deserializer::from_str(&s[start..]).into_iter::<Value>();
    let value = match stream.next() {
        Some(Ok(value)) => value,
        Some(Err(err)) => return Err(err.to_string()),
        None => return Err("Unexpected end of JSON input".to_string()),
    };
    Ok((value, start + stream.byte_offset()))
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Recursively sort every object's keys (`encoding_k3.deep_sort_dict`).
///
/// serde_json is built with `preserve_order`, so the insertion order is kept
/// unless it is rebuilt like this. Rust sorts `&str` by UTF-8 bytes, which for
/// valid UTF-8 is the same order Python's `sorted()` gives by code point.
pub fn deep_sort(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::with_capacity(keys.len());
            for key in keys {
                out.insert(key.clone(), deep_sort(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(deep_sort).collect()),
        other => other.clone(),
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`.
pub fn compact_json(value: &Value) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

/// `json.dumps(value, ensure_ascii=False)` with Python's DEFAULT separators.
pub fn python_default_json(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonDefaultFormatter);
    match serde::Serialize::serialize(value, &mut ser) {
        // Serializing a `Value` into a `Vec<u8>` cannot fail.
        Ok(()) => String::from_utf8(buf).unwrap_or_default(),
        Err(_) => String::new(),
    }
}

/// serde_json formatter matching Python's `json.dumps` default separators.
struct PythonDefaultFormatter;

impl serde_json::ser::Formatter for PythonDefaultFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

/// The tools list as the reference serializes it: exactly the caller's own
/// objects, deep-sorted.
fn deep_sorted_tools(tools: &[Tool]) -> Result<Value> {
    Ok(deep_sort(&serde_json::to_value(tools)?))
}

/// `&` then `"`, in that order (`encoding_k3._escape_attr_value`).
pub fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

/// The `type` of a `response_format`, or `None` when there is none to read.
///
/// Mirrors `rf_type = _get_value(rf, "type", rf) if isinstance(rf, dict) else rf`:
/// an object without a `type` key resolves to the object itself, which matches
/// neither branch, and a bare string is its own type.
fn response_format_type(response_format: Option<&Value>) -> Option<&str> {
    match response_format? {
        Value::Object(map) => map.get("type")?.as_str(),
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// `encoding_k3.extract_response_schema`.
fn extract_response_schema(response_format: Option<&Value>) -> Option<&Value> {
    let json_schema = response_format?.get("json_schema")?;
    if json_schema.is_null() {
        return None;
    }
    let Value::Object(map) = json_schema else {
        return Some(json_schema);
    };
    Some(
        map.get("schema")
            .or_else(|| map.get("json_schema"))
            .unwrap_or(json_schema),
    )
}

#[cfg(test)]
#[path = "kimi_k3_chat_tests.rs"]
mod tests;
