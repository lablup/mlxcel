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

//! OpenAI and llama-server compatible request types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::vision::processors::gemma4::{SUPPORTED_IMAGE_SOFT_TOKENS, validate_image_soft_tokens};

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

// ---------------------------------------------------------------------------
// Tool calling types (OpenAI-compatible)
// Used by: ChatCompletionRequest, chat_template, routes/chat
// ---------------------------------------------------------------------------

/// A tool definition (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Human-readable description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool choice specification
///
/// Can be a string ("auto", "none", "required") or an object specifying a
/// particular function.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode: "auto", "none", or "required"
    Mode(String),
    /// Specific function: {"type": "function", "function": {"name": "X"}}
    Specific(ToolChoiceFunction),
}

impl ToolChoice {
    /// Returns the mode string for simple string choices, or "specific" for
    /// named function selections.
    pub fn mode(&self) -> &str {
        match self {
            ToolChoice::Mode(s) => s.as_str(),
            ToolChoice::Specific(_) => "specific",
        }
    }

    /// Returns true if tool calling is effectively disabled.
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "none")
    }

    /// Returns the specific function name if this is a named choice.
    pub fn specific_function(&self) -> Option<&str> {
        match self {
            ToolChoice::Specific(f) => Some(&f.function.name),
            _ => None,
        }
    }
}

/// Named function tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunctionName,
}

/// Function name within a tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

/// A tool call within an assistant message (multi-turn history)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInMessage {
    /// Unique tool call ID
    pub id: String,
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// Function call details
    pub function: ToolCallFunction,
}

/// Function name + arguments within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name
    pub name: String,
    /// Stringified JSON arguments
    pub arguments: String,
}

impl Role {
    /// Convert role to lowercase string for chat templates
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Content part for multimodal messages (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content (supports base64 data URIs, `file://`, local paths,
    /// and `http(s)` URLs)
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Video URL content for VLMs that support video inputs.
    /// Mirrors the `image_url` shape for symmetry. Accepts local paths,
    /// `file://...`, and (where the model supports it) `http(s)://...`.
    /// Frame extraction relies on `ffmpeg` being available on the server
    /// host's PATH; missing `ffmpeg` produces a clean 4xx response rather
    /// than a crash.
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
    /// Audio input content (base64-encoded audio data)
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

/// Image URL reference.
///
/// Both budget fields are optional and default to `None`, so a request that
/// sends only `url` behaves exactly as it did before they existed.
///
/// # Soft-token budget (Gemma 4 only)
///
/// Gemma 4's vision tower is resolution-driven: the number of soft tokens an
/// image contributes to the prompt is a function of the resize target, which
/// is a function of the soft-token budget. These two fields expose that dial
/// per request. Every other VLM family ignores them.
///
/// * [`Self::detail`] is the OpenAI-standard field. `"low"` maps to the
///   smallest supported budget, `"high"` to the largest, and `"auto"` (or an
///   absent field) leaves the checkpoint's configured default in place.
/// * [`Self::max_soft_tokens`] is an **mlxcel extension**, not part of the
///   OpenAI spec. It names an exact budget from the supported ladder and wins
///   over `detail` when both are present.
///
/// Both are validated at the request boundary; an unsupported value is a 400
/// rather than a silent clamp. See
/// [`Self::resolve_soft_token_budget`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL: `data:image/...;base64,...`, `file://...`, bare local path, or
    /// `http(s)://...`
    pub url: String,
    /// OpenAI-standard detail hint: `"low"`, `"high"`, or `"auto"`.
    ///
    /// Maps onto the Gemma 4 soft-token ladder (see [`Self`]). Unknown values
    /// are rejected with a 400 rather than silently treated as `"auto"`, so a
    /// typo cannot quietly downgrade image fidelity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// mlxcel extension: exact Gemma 4 soft-token budget for this image.
    ///
    /// Must be one of
    /// [`mlxcel::vision::processors::gemma4::SUPPORTED_IMAGE_SOFT_TOKENS`].
    /// Takes precedence over [`Self::detail`]. Not part of the OpenAI API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_soft_tokens: Option<usize>,
}

impl ImageUrl {
    /// Construct a plain image reference with no budget override, i.e. the
    /// checkpoint's configured default. Used by translators (e.g. the Anthropic
    /// Messages API) whose wire format has no soft-token dial.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
            max_soft_tokens: None,
        }
    }

    /// Resolve this content part's soft-token budget.
    ///
    /// Returns `Ok(None)` when the caller expressed no preference (neither
    /// field set, or `detail: "auto"`), which means "use the checkpoint's
    /// configured default" and preserves today's behavior exactly.
    ///
    /// The numeric field wins over `detail` when both are present.
    ///
    /// # Errors
    /// Returns `Err` with a caller-facing message (surfaced as a 400) when
    /// `max_soft_tokens` is off the supported ladder or `detail` is not one of
    /// `low` / `high` / `auto`. Both values are untrusted request input and the
    /// budget drives the resize target, so neither is clamped.
    pub fn resolve_soft_token_budget(&self) -> Result<Option<usize>, String> {
        if let Some(requested) = self.max_soft_tokens {
            return validate_image_soft_tokens(requested).map(Some);
        }

        let Some(detail) = self.detail.as_deref() else {
            return Ok(None);
        };

        match detail.trim().to_ascii_lowercase().as_str() {
            // The ladder is non-empty, so first/last always resolve; fall back
            // to "no override" rather than panicking if that ever changes.
            "low" => Ok(SUPPORTED_IMAGE_SOFT_TOKENS.first().copied()),
            "high" => Ok(SUPPORTED_IMAGE_SOFT_TOKENS.last().copied()),
            "auto" => Ok(None),
            other => Err(format!(
                "image_url.detail must be one of [\"low\", \"high\", \"auto\"], got \"{other}\""
            )),
        }
    }
}

/// Resolve a single request-scoped Gemma 4 image soft-token budget from every
/// `image_url` content part in the request.
///
/// The budget is applied per request rather than per image because the Gemma 4
/// preprocessor takes one budget for the whole batch. Parts that express no
/// preference are ignored. When two parts request *different* explicit budgets
/// the request is rejected: silently picking one (or the max) would give the
/// caller a budget they did not ask for on at least one image, and the prompt's
/// placeholder expansion would then be derived from a value the caller cannot
/// predict.
///
/// # Errors
/// Returns `Err` when any part fails [`ImageUrl::resolve_soft_token_budget`],
/// or when two parts disagree on an explicit budget.
pub fn resolve_request_image_soft_tokens(parts: &[ImageUrl]) -> Result<Option<usize>, String> {
    let mut resolved: Option<usize> = None;
    for part in parts {
        let Some(budget) = part.resolve_soft_token_budget()? else {
            continue;
        };
        match resolved {
            Some(existing) if existing != budget => {
                return Err(format!(
                    "conflicting image soft-token budgets in one request: {existing} and {budget}; \
                     all image_url parts must agree"
                ));
            }
            _ => resolved = Some(budget),
        }
    }
    Ok(resolved)
}

/// Video URL reference. Same wire shape as [`ImageUrl`] for
/// symmetry with the OpenAI vision content blocks.
///
/// `fps` is an optional sampling rate. When omitted, the server falls
/// back to [`mlxcel::video::DEFAULT_FPS`] (2.0 fps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    /// URL: `data:video/...;base64,...` (where supported), `file://...`,
    /// bare local path, or `http(s)://...`.
    pub url: String,
    /// Optional sampling rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

/// Audio input reference (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data, or a URL/file path
    pub data: String,
    /// Audio format: "wav", "mp3", etc.
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrManyStrings {
    One(String),
    Many(Vec<String>),
}

fn deserialize_optional_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(
        Option::<OneOrManyStrings>::deserialize(deserializer)?.map(|value| match value {
            OneOrManyStrings::One(stop) => vec![stop],
            OneOrManyStrings::Many(stops) => stops,
        }),
    )
}

fn deserialize_optional_seed<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Number>::deserialize(deserializer)? else {
        return Ok(None);
    };
    // b10621 reads `seed` into a `uint32_t` with an unchecked cast (#1485),
    // so every integer folds modulo 2^32 and only the resulting sentinel
    // `0xFFFF_FFFF` (spelled `-1`) draws a random seed: `-2` is the
    // DETERMINISTIC seed `4294967294`. The pre-#1485 rejection of values
    // below `-1` diverged from that; the fold restores upstream's
    // arithmetic (an `as u32` cast wraps modulo 2^32, congruent with C++'s
    // conversion for every value JSON can carry in an i64/u64).
    let folded = if let Some(signed) = value.as_i64() {
        signed as u32
    } else if let Some(unsigned) = value.as_u64() {
        unsigned as u32
    } else {
        return Err(serde::de::Error::custom("seed must be an integer"));
    };
    if folded == u32::MAX {
        Ok(None)
    } else {
        Ok(Some(u64::from(folded)))
    }
}

/// Message content: either a plain string or multimodal array
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content
    Text(String),
    /// Multimodal content parts (text + images)
    Parts(Vec<ContentPart>),
}

pub(crate) const ORDERED_MEDIA_PREFIX: &str = "<|mlxcel_ordered_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrderedMediaSegment<'a> {
    Text(&'a str),
    Image(usize),
    Audio(usize),
}

impl MessageContent {
    /// Extract the text content from the message
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Returns `true` when the content has at least one non-whitespace text
    /// character, without allocating a `String` the way [`Self::text`] does.
    ///
    /// Equivalent to `!self.text().trim().is_empty()`: joining `Parts` text
    /// parts with `""` is non-whitespace iff at least one part is
    /// non-whitespace on its own, so a per-part `any` check gives the same
    /// answer as trimming the joined string.
    pub fn has_effective_text(&self) -> bool {
        match self {
            MessageContent::Text(s) => !s.trim().is_empty(),
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text { text } if !text.trim().is_empty())),
        }
    }

    /// Extract image data URIs/paths from multimodal content
    pub fn image_urls(&self) -> Vec<String> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract whole `image_url` content parts, preserving the per-part
    /// `detail` / `max_soft_tokens` fields that [`Self::image_urls`] drops.
    pub fn image_parts(&self) -> Vec<ImageUrl> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract audio input data from multimodal content
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::InputAudio { input_audio } => Some(input_audio.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract video URL references from multimodal content.
    pub fn video_urls(&self) -> Vec<VideoUrl> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::VideoUrl { video_url } => Some(video_url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
}

pub(crate) fn ordered_image_sentinel(ordinal: usize) -> String {
    format!("{ORDERED_MEDIA_PREFIX}image_{ordinal}|>")
}

pub(crate) fn ordered_audio_sentinel(ordinal: usize) -> String {
    format!("{ORDERED_MEDIA_PREFIX}audio_{ordinal}|>")
}

pub(crate) fn parse_ordered_media_segments(
    prompt: &str,
) -> Result<Vec<OrderedMediaSegment<'_>>, String> {
    let mut segments = Vec::new();
    let mut remaining = prompt;
    let mut expected_image = 1usize;
    let mut expected_audio = 1usize;
    while let Some(position) = remaining.find(ORDERED_MEDIA_PREFIX) {
        if position > 0 {
            segments.push(OrderedMediaSegment::Text(&remaining[..position]));
        }
        let marker = &remaining[position + ORDERED_MEDIA_PREFIX.len()..];
        let end = marker
            .find("|>")
            .ok_or("malformed ordered media sentinel: missing |>")?;
        let descriptor = &marker[..end];
        let (kind, ordinal) = descriptor
            .rsplit_once('_')
            .ok_or("malformed ordered media sentinel: missing ordinal")?;
        let ordinal = ordinal
            .parse::<usize>()
            .map_err(|_| "malformed ordered media sentinel: invalid ordinal")?;
        match kind {
            "image" if ordinal == expected_image => {
                segments.push(OrderedMediaSegment::Image(ordinal));
                expected_image += 1;
            }
            "audio" if ordinal == expected_audio => {
                segments.push(OrderedMediaSegment::Audio(ordinal));
                expected_audio += 1;
            }
            "image" => {
                return Err(format!(
                    "ordered image sentinel {ordinal} is out of sequence; expected {expected_image}"
                ));
            }
            "audio" => {
                return Err(format!(
                    "ordered audio sentinel {ordinal} is out of sequence; expected {expected_audio}"
                ));
            }
            _ => return Err("malformed ordered media sentinel: unknown kind".to_string()),
        }
        remaining = &marker[end + 2..];
    }
    if !remaining.is_empty() {
        segments.push(OrderedMediaSegment::Text(remaining));
    }
    Ok(segments)
}

pub(crate) fn strip_ordered_media_sentinels(prompt: &str) -> Result<String, String> {
    let segments = parse_ordered_media_segments(prompt)?;
    let mut stripped = String::with_capacity(prompt.len());
    for segment in segments {
        if let OrderedMediaSegment::Text(text) = segment {
            stripped.push_str(text);
        }
    }
    Ok(stripped)
}

impl Default for MessageContent {
    /// Empty text — the canonical "no content" value.
    ///
    /// Assistant messages whose payload is a `tool_calls` array legitimately
    /// omit `content` or send `null` (issue #89); both map to this.
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

/// Deserialize [`MessageContent`], tolerating an explicit JSON `null`.
///
/// Paired with `#[serde(default)]` on the field, this lets `content` accept
/// the three shapes OpenAI-compatible clients emit on assistant messages that
/// carry `tool_calls` (issue #89): the key is absent (handled by `default`),
/// the value is `null` (mapped to empty content here), or the value is a
/// normal string / multimodal array. Without it, axum's `Json` extractor
/// rejects the follow-up request of a tool-calling loop with HTTP 422
/// (`missing field 'content'`).
fn deserialize_message_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<MessageContent>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Deserialize)]
struct MessageWire {
    role: Role,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    content: MessageContent,
    name: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Option<Vec<ToolCallInMessage>>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

/// Chat message
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    /// Message content.
    ///
    /// Optional and nullable per the OpenAI Chat Completions spec: assistant
    /// messages that carry `tool_calls` may omit `content` or send `null`. A
    /// missing or `null` value deserializes to empty text (issue #89), keeping
    /// the `content` field present for Jinja chat templates that read
    /// `message.content`.
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID for `role: "tool"` messages (references a previous tool call)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant (multi-turn history)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInMessage>>,
    /// Prior-turn assistant reasoning ("interleaved reasoning"), forwarded to
    /// chat templates that render `message.get('reasoning')` (e.g. Gemma 4) so
    /// the model can see its own thinking across turns (issue #362).
    ///
    /// Accepts `reasoning`, `reasoning_content`, or an equal pair of both
    /// spellings. The field is dropped from serialized output when absent,
    /// keeping existing wire shapes unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = MessageWire::deserialize(deserializer)?;
        let reasoning = match (wire.reasoning, wire.reasoning_content) {
            (Some(reasoning), Some(reasoning_content)) if reasoning != reasoning_content => {
                return Err(serde::de::Error::custom(
                    "reasoning and reasoning_content must be identical when both are provided",
                ));
            }
            (Some(reasoning), _) | (_, Some(reasoning)) => Some(reasoning),
            (None, None) => None,
        };
        Ok(Self {
            role: wire.role,
            content: wire.content,
            name: wire.name,
            tool_call_id: wire.tool_call_id,
            tool_calls: wire.tool_calls,
            reasoning,
        })
    }
}

/// Sampling parameters shared across endpoints
///
/// All parameters are optional. When not specified in the request,
/// server defaults (from CLI arguments) will be used.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate. Server routes silently clamp an
    /// explicit value to the effective per-slot context window.
    #[serde(alias = "n_predict", alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random)
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold
    pub top_p: Option<f32>,
    /// Top-k sampling (0 = disabled)
    pub top_k: Option<usize>,
    /// Min-p sampling threshold (0.0 = disabled)
    pub min_p: Option<f32>,
    /// Repetition penalty (1.0 = no penalty)
    #[serde(alias = "repeat_penalty")]
    pub repetition_penalty: Option<f32>,
    /// Repetition / frequency / presence penalty lookback window (b10621
    /// `repeat_last_n`, mlx-lm `repetition_context_size`; #1436 wired it
    /// into the sampler after #1430 left it parse-only). `0` disables the
    /// three history penalties; `N > 0` penalizes over the last N tokens.
    /// Absent falls back to the server-wide `--repeat-last-n` default.
    #[serde(alias = "repeat_last_n")]
    pub repetition_context_size: Option<usize>,
    /// Logit bias for specific tokens
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Stop sequences
    #[serde(deserialize_with = "deserialize_optional_stop")]
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility
    #[serde(deserialize_with = "deserialize_optional_seed")]
    pub seed: Option<u64>,

    // DRY (Don't Repeat Yourself) sampling parameters
    /// DRY penalty multiplier (0.0 = disabled, typical: 0.8-1.3)
    pub dry_multiplier: Option<f32>,
    /// DRY penalty base for exponential scaling (typical: 1.75)
    pub dry_base: Option<f32>,
    /// Minimum sequence length before DRY penalties apply (typical: 2)
    pub dry_allowed_length: Option<usize>,
    /// Number of recent tokens to scan for DRY (0 = entire context)
    pub dry_penalty_last_n: Option<usize>,
    /// Sequence breaker tokens for DRY (resets matching)
    pub dry_sequence_breakers: Option<Vec<i32>>,

    // XTC (Exclude Top Choices) sampling parameters
    /// XTC probability (0.0 = disabled)
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold
    pub xtc_threshold: Option<f32>,

    /// Top-n-sigma logit filter (`0.0` = disabled). Keeps only tokens whose
    /// raw logit lies within `top_n_sigma` standard deviations of the row
    /// maximum. Must be finite and `>= 0.0`; validated at the request layer.
    pub top_n_sigma: Option<f32>,

    /// Locally typical sampling cutoff (`1.0` = disabled). Keeps the tokens
    /// whose surprisal is closest to the row entropy until `typical_p`
    /// probability mass accumulates. Must be finite and in `(0.0, 1.0]`;
    /// validated at the request layer.
    pub typical_p: Option<f32>,

    // OpenAI-compatible frequency/presence penalties
    /// Frequency penalty (0.0 = disabled) - penalizes based on frequency
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (0.0 = disabled) - penalizes based on presence
    pub presence_penalty: Option<f32>,

    // vLLM-compatible N-gram repetition / loop detection. When any of the three
    // is present the request is authoritative and overrides server defaults and
    // family auto-enable (see `request_options::resolve_loop_detection`). Field
    // names match vLLM's `SamplingParams` for client compatibility.
    /// Largest N-gram pattern size to scan (`0` disables detection).
    pub max_pattern_size: Option<usize>,
    /// Smallest N-gram pattern size to scan (`0` is treated as `1`).
    pub min_pattern_size: Option<usize>,
    /// Minimum consecutive repeats of a pattern that ends generation early
    /// (must be `>= 2`).
    pub min_count: Option<usize>,

    // thinking-token budget (Qwen3-family reasoning cap).
    //
    // Three aliases accepted, first non-None wins (see
    // `thinking_budget::pick_budget_alias`). llama.cpp-compatible primary
    // name, vLLM alias, and Qwen alias. Value semantics: -1 unrestricted,
    // 0 immediate close, N > 0 cap at N tokens inside the `<think>` block.
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    #[serde(alias = "reasoning_budget_tokens")]
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,

    /// b10621 `reasoning_control` (#1444): arm realtime reasoning control
    /// for this completion, so `POST /v1/chat/completions/control` with
    /// `action: "reasoning_end"` can close the thinking block mid-
    /// generation. Upstream's schema description: "Create the budget
    /// sampler on demand so reasoning can be ended at runtime". Default
    /// `false`, exactly as upstream.
    pub reasoning_control: Option<bool>,
}

/// Stream options for controlling streaming behavior
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    /// Include token usage statistics in the final streaming chunk
    #[serde(default)]
    pub include_usage: bool,
}

/// Effort spellings that disable template-level reasoning.
///
/// Matching applies `trim().to_ascii_lowercase()` before comparing.
pub const DISABLED_REASONING_EFFORTS: [&str; 5] = ["none", "off", "disabled", "false", "0"];

/// Resolved reasoning behavior for one chat-template render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningControl {
    /// Whether the template should enable its thinking path.
    pub enabled: bool,
    /// The caller's raw effort spelling, when one was supplied.
    pub effort: Option<String>,
}

/// Chat completion request (POST /v1/chat/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier
    pub model: String,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether to return log probabilities of output tokens
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probability alternatives to return per token (0–20)
    #[serde(default)]
    pub top_logprobs: Option<u8>,

    // Tool calling fields (OpenAI-compatible)
    /// Tool definitions available for the model to call
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Controls how the model selects tools: "auto", "none", "required", or a
    /// specific function object
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue multiple tool calls in parallel
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// top-level `chat_template_kwargs` (llama.cpp shape).
    ///
    /// A JSON object whose keys are forwarded as Jinja template kwargs when
    /// rendering the conversation. Primary shape; wins over nested
    /// `extra_body.chat_template_kwargs`, flattened OpenAI-SDK `extra_body`
    /// aliases, and DashScope flat `extra_body.preserve_thinking`. See
    /// [`crate::server::chat_template_kwargs::extract_request_kwargs`] for
    /// the full precedence chain.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,

    /// nested `extra_body` compatibility (vLLM / manual callers).
    ///
    /// Some callers send an actual top-level `extra_body` object. Only the
    /// keys we currently recognize are read back out; unknown keys are
    /// silently ignored to match llama.cpp's lenient behavior.
    #[serde(default)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,

    /// OpenAI-compatible prompt-cache key hint.
    ///
    /// Clients can send this to pin a conversation to a specific prompt-cache
    /// session bucket. When present it wins over the standard OpenAI `user`
    /// field in [`crate::server::prompt_cache::key::resolve_session_key`];
    /// when absent the server falls back to `user`, then to an anonymous
    /// bucket sentinel. The string is never echoed back to the client — it is
    /// only used as a session-bucket discriminator inside the cache key hash.
    ///
    /// Also round-trips through the flattened OpenAI-SDK `extra_body`
    /// mechanism ([`Self::extra_body_fields`]): SDKs that pass `extra_body=
    /// {"prompt_cache_key": "..."}` land here via the flatten; SDKs that send
    /// it at the request root land here directly.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,

    /// llama-server b10621 `cache_prompt`: re-use the KV cache from a previous
    /// request when the prompts share a prefix.
    ///
    /// `None` (the default) leaves the server-wide setting in force, which is
    /// on unless `--no-cache-prompt` / `--no-prompt-cache` turned it off.
    /// `false` opts this one request out: it is prefilled from cold and it
    /// donates nothing back, so a client can force a clean evaluation without
    /// disturbing what other requests may reuse.
    ///
    /// `true` is accepted and asserts the default; it cannot switch the cache
    /// back on for one request against a server-wide disable, because there is
    /// no store to look in.
    ///
    /// Reaches the same three ways as [`Self::prompt_cache_key`]: at the
    /// request root, through the flattened OpenAI-SDK `extra_body`, or nested
    /// inside `extra_body`. See [`Self::resolve_cache_prompt`].
    #[serde(default)]
    pub cache_prompt: Option<bool>,

    /// OpenAI-standard stable end-user identifier.
    ///
    /// Used as a session-bucket fallback for the prompt-prefix cache when
    /// `prompt_cache_key` is not supplied. See
    /// [`crate::server::prompt_cache::key::resolve_session_key`] for the
    /// full precedence chain. The value is treated as opaque bytes; the
    /// server never attempts to interpret it as an identity or access control
    /// token.
    #[serde(default)]
    pub user: Option<String>,

    /// OpenAI-standard reasoning-effort hint.
    ///
    /// Resolved into `enable_thinking` plus the level key used by the loaded
    /// chat template in
    /// [`crate::server::chat_request::resolve_effective_kwargs`]. Enabled
    /// values are forwarded verbatim as `reasoning_effort`, or as
    /// `reasoning_strength` when that is the template's supported spelling.
    /// Disabled values (`none`, `off`, `disabled`, `false`, and `0`) set
    /// `enable_thinking=false` and are never forwarded as a level. Explicit
    /// `chat_template_kwargs` entries retain per-key precedence.
    ///
    /// Resolved through the same three-tier chain as [`Self::resolve_user`]:
    /// this field, then the flattened OpenAI-SDK `extra_body`, then a nested
    /// `extra_body`. See [`Self::resolve_reasoning_effort`].
    #[serde(default)]
    pub reasoning_effort: Option<String>,

    /// OpenAI SDK `extra_body={...}` flattened into the request root.
    ///
    /// The official OpenAI Python client merges `extra_body` into the top-level
    /// JSON object instead of emitting a nested `"extra_body": {...}` wrapper.
    /// Capture those unknown root keys here so request-kwarg extraction can
    /// treat them the same as nested `extra_body` aliases.
    #[serde(default, flatten)]
    pub extra_body_fields: serde_json::Map<String, serde_json::Value>,

    /// OpenAI-compatible structured-output spec.
    ///
    /// Accepts the OpenAI Chat Completions shape:
    ///
    /// ```json
    /// {
    ///   "response_format": {
    ///     "type": "json_schema",
    ///     "json_schema": {
    ///       "name": "result",
    ///       "strict": true,
    ///       "schema": { "type": "object", ... }
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// When set with `type: "json_schema"`, generation is constrained via
    /// [`crate::server::structured`] so emitted tokens always conform to the
    /// supplied schema. Other types (`text`, `null`) are no-ops; `json_object`
    /// is rejected as unsupported in the MVP — see `extract_json_schema_from_response_format`.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,

    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

/// Text completion request (POST /v1/completions)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Options controlling streaming behavior (only used when stream=true)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Number of top log-probability alternatives to return (legacy format: 0–5)
    #[serde(default)]
    pub logprobs: Option<u8>,
    /// OpenAI-compatible structured-output spec; see the
    /// matching field on [`ChatCompletionRequest`] for shape details.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Sampling parameters (flattened)
    #[serde(flatten)]
    pub params: SamplingParams,
}

impl ChatCompletionRequest {
    /// Merge nested `extra_body` with flattened OpenAI-SDK root fields.
    ///
    /// Flattened root keys win over the nested object on collision because the
    /// request body already exposed them at the higher-precedence top level.
    pub fn merged_extra_body(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        match (&self.extra_body, self.extra_body_fields.is_empty()) {
            (None, true) => None,
            (Some(extra), true) => Some(extra.clone()),
            (None, false) => Some(self.extra_body_fields.clone()),
            (Some(extra), false) => {
                let mut merged = extra.clone();
                for (key, value) in &self.extra_body_fields {
                    merged.insert(key.clone(), value.clone());
                }
                Some(merged)
            }
        }
    }

    /// Resolve the request-level `prompt_cache_key`.
    ///
    /// Precedence (first non-empty wins):
    ///   1. Top-level `prompt_cache_key`.
    ///   2. Flattened OpenAI-SDK `extra_body` field of the same name.
    ///   3. Nested `extra_body.prompt_cache_key`.
    ///
    /// Empty strings are treated as "not supplied" so a caller can't
    /// accidentally smuggle themselves into an empty-string bucket.
    pub fn resolve_prompt_cache_key(&self) -> Option<&str> {
        if let Some(k) = self.prompt_cache_key.as_deref()
            && !k.is_empty()
        {
            return Some(k);
        }
        if let Some(s) = self
            .extra_body_fields
            .get("prompt_cache_key")
            .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(s) = body
                .get("prompt_cache_key")
                .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        None
    }

    /// Resolve the request-level b10621 `cache_prompt` switch.
    ///
    /// Same precedence as [`Self::resolve_prompt_cache_key`]: top-level field,
    /// then flattened `extra_body`, then nested `extra_body`. `None` means the
    /// request said nothing and the server-wide setting decides.
    ///
    /// A non-boolean value under the key is ignored rather than rejected: the
    /// top-level field is typed, so a wrong type there is already a 400 from
    /// serde, and the `extra_body` paths are an untyped passthrough where an
    /// unrelated key of the same name is likelier than a deliberate lie.
    pub fn resolve_cache_prompt(&self) -> Option<bool> {
        if let Some(v) = self.cache_prompt {
            return Some(v);
        }
        if let Some(v) = self
            .extra_body_fields
            .get("cache_prompt")
            .and_then(serde_json::Value::as_bool)
        {
            return Some(v);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(v) = body
                .get("cache_prompt")
                .and_then(serde_json::Value::as_bool)
        {
            return Some(v);
        }
        None
    }

    /// Resolve the request-level OpenAI-standard `user` identifier.
    ///
    /// Same precedence rules as [`Self::resolve_prompt_cache_key`]: top-level
    /// field, then flattened `extra_body`, then nested `extra_body`. Empty
    /// strings are ignored.
    pub fn resolve_user(&self) -> Option<&str> {
        if let Some(u) = self.user.as_deref()
            && !u.is_empty()
        {
            return Some(u);
        }
        if let Some(s) = self
            .extra_body_fields
            .get("user")
            .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(s) = body.get("user").and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        None
    }

    /// Resolve the request-level OpenAI-standard `reasoning_effort` hint.
    ///
    /// Same precedence rules as [`Self::resolve_prompt_cache_key`] and
    /// [`Self::resolve_user`]: top-level field, then flattened `extra_body`,
    /// then nested `extra_body`. Empty strings are ignored so a caller cannot
    /// push an empty value at a template that would then reject it.
    ///
    /// Non-string values in either `extra_body` shape are ignored rather than
    /// rejected, matching how the other two resolvers treat them.
    pub fn resolve_reasoning_effort(&self) -> Option<&str> {
        if let Some(e) = self.reasoning_effort.as_deref()
            && !e.is_empty()
        {
            return Some(e);
        }
        if let Some(s) = self
            .extra_body_fields
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        if let Some(body) = self.extra_body.as_ref()
            && let Some(s) = body
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
            && !s.is_empty()
        {
            return Some(s);
        }
        None
    }

    /// Resolve the portable reasoning controls carried by this request.
    ///
    /// The existing top-level/flattened/nested [`Self::reasoning_effort`]
    /// chain has first priority. If it is absent, a `reasoning` key from the
    /// flattened OpenAI-SDK fields and then nested `extra_body` may supply
    /// either `{"effort": "..."}` or a boolean. Effort strings retain their
    /// original spelling for the template; only the enabled/disabled decision
    /// uses a trimmed, lowercase copy.
    pub fn resolve_reasoning_control(&self) -> Option<ReasoningControl> {
        if let Some(effort) = self.resolve_reasoning_effort() {
            return Some(reasoning_control_from_effort(effort));
        }

        self.extra_body_fields
            .get("reasoning")
            .and_then(reasoning_control_from_value)
            .or_else(|| {
                self.extra_body
                    .as_ref()
                    .and_then(|body| body.get("reasoning"))
                    .and_then(reasoning_control_from_value)
            })
    }

    /// Convert messages to a prompt string using a simple format
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            let text = msg.content.text();
            match msg.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n\n", text));
                }
                Role::User => {
                    prompt.push_str(&format!("User: {}\n\n", text));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n\n", text));
                }
                Role::Tool => {
                    prompt.push_str(&format!("Tool: {}\n\n", text));
                }
            }
        }
        prompt.push_str("Assistant: ");
        prompt
    }

    /// Extract all image URLs from messages
    pub fn image_urls(&self) -> Vec<String> {
        self.messages
            .iter()
            .flat_map(|m| m.content.image_urls())
            .collect()
    }

    /// Extract all `image_url` content parts from messages, preserving the
    /// per-part budget fields. Same order as [`Self::image_urls`].
    pub fn image_parts(&self) -> Vec<ImageUrl> {
        self.messages
            .iter()
            .flat_map(|m| m.content.image_parts())
            .collect()
    }

    /// Resolve the request-scoped Gemma 4 image soft-token budget.
    ///
    /// # Errors
    /// Returns `Err` when a part carries an unsupported `detail` /
    /// `max_soft_tokens`, or when parts disagree. Routes surface this as a 400.
    pub fn image_soft_tokens(&self) -> Result<Option<usize>, String> {
        resolve_request_image_soft_tokens(&self.image_parts())
    }

    /// Extract all audio inputs from messages
    pub fn audio_inputs(&self) -> Vec<InputAudio> {
        self.messages
            .iter()
            .flat_map(|m| m.content.audio_inputs())
            .collect()
    }

    /// Extract all video URL references from messages.
    pub fn video_urls(&self) -> Vec<VideoUrl> {
        self.messages
            .iter()
            .flat_map(|m| m.content.video_urls())
            .collect()
    }
}

fn reasoning_control_from_effort(effort: &str) -> ReasoningControl {
    let normalized = effort.trim().to_ascii_lowercase();
    ReasoningControl {
        enabled: !DISABLED_REASONING_EFFORTS.contains(&normalized.as_str()),
        effort: Some(effort.to_string()),
    }
}

fn reasoning_control_from_value(value: &serde_json::Value) -> Option<ReasoningControl> {
    if let Some(enabled) = value.as_bool() {
        return Some(ReasoningControl {
            enabled,
            effort: None,
        });
    }

    let effort = value
        .as_object()?
        .get("effort")?
        .as_str()
        .filter(|effort| !effort.is_empty())?;
    Some(reasoning_control_from_effort(effort))
}

/// Native llama-server completion request (POST /completion)
#[derive(Debug, Clone, Deserialize)]
pub struct NativeCompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum number of tokens to predict. The server silently clamps an
    /// explicit value to the effective per-slot context window.
    ///
    /// b10621 declares `max_tokens` and `max_completion_tokens` as aliases of
    /// this field, so a request written for either OpenAI spelling reaches the
    /// native route unchanged (#1441).
    ///
    /// Signed because upstream's hard limits are `-1 <= value <= INT32_MAX`,
    /// where `-1` means "as many as the context allows" and `0` means "evaluate
    /// the prompt into the cache". The route resolves the sign through
    /// [`NativeCompletionRequest::resolve_n_predict`] rather than letting serde
    /// reject a value b10621 serves.
    #[serde(alias = "max_tokens", alias = "max_completion_tokens")]
    pub n_predict: Option<i64>,
    /// Attach the timing block to every streaming frame instead of only the
    /// final one (#1441).
    pub timings_per_token: Option<bool>,
    /// Project the finished response down to the listed paths (#1441).
    ///
    /// Held as a raw value because b10621 reads it through `json_value(...,
    /// std::vector<std::string>())`, which falls back to the default for a
    /// value of any other shape instead of failing the request. A strongly
    /// typed field would turn `"response_fields": "content"` into a 422 that
    /// upstream answers with a normal completion.
    #[serde(default)]
    pub response_fields: Option<serde_json::Value>,
    /// Additional options for streaming responses (#1441).
    ///
    /// Raw for the same reason as `response_fields`: upstream tolerates a
    /// non-object here and ignores it, while rejecting a non-boolean
    /// `include_usage` inside it with a 400. Both halves are reproduced by
    /// `validate_native_stream_options`.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    /// Number of completions to generate. mlxcel serves one completion per
    /// request; a value above 1 is rejected with a diagnostic rather than
    /// silently producing one result where b10621 would produce an array.
    #[serde(alias = "n")]
    pub n_cmpl: Option<i64>,
    /// Minimum line indentation for the generated text, a FIM feature with no
    /// mlxcel equivalent. Rejected when set to a value that would change
    /// behavior.
    pub n_indent: Option<i64>,
    /// Time limit in milliseconds for the prediction phase. mlxcel bounds a
    /// stalled decode with `--decode-timeout` per server, not per request.
    pub t_max_predict_ms: Option<i64>,
    /// Include prompt-processing progress events in stream mode. mlxcel's
    /// scheduler emits no prefill progress events on this path.
    pub return_progress: Option<bool>,
    /// llama-server b10621 `verbose`, accepted and inert on this route.
    ///
    /// Upstream writes its `__verbose` debug block only from the OAI-compat
    /// response builders (`server-task.cpp`); the native `/completion` object
    /// IS `to_json_non_oaicompat()`, so `verbose: true` changes nothing there.
    /// Verified against the pinned binary: the top-level key set with the
    /// field set is identical to the key set without it. mlxcel therefore
    /// accepts it and ignores it, which is exactly what upstream does (#1477).
    pub verbose: Option<bool>,
    /// Return the raw generated token ids in the `tokens` field.
    pub return_tokens: Option<bool>,
    /// llama-server b10621 `cache_prompt`: reuse the KV prefix a previous
    /// request left in the prompt cache, and donate this request's own prefix
    /// back (#1473). Upstream's default is enabled, so absent means "follow
    /// the server-wide `--cache-prompt` / `--no-cache-prompt`"; `false` opts
    /// this one request out of both the lookup and the donate-back. `true`
    /// asserts the default and cannot re-enable the cache against a
    /// server-wide disable, which is the same rule the chat-shaped routes
    /// apply.
    pub cache_prompt: Option<bool>,
    /// Number of leading prompt tokens retained across a context shift
    /// (#1472). `-1` retains the whole initial prompt; absent falls back to
    /// the server's `--keep`. Read only when `--context-shift` is enabled.
    pub n_keep: Option<i64>,
    /// Tokens past the retained prefix each context shift discards (#1472).
    /// `0`, upstream's default, resolves to half of the non-retained window.
    pub n_discard: Option<i64>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Per-request override for the SSE comment ping interval, in seconds
    /// (#1432). `-1` disables the pings for this stream. Absent falls back to
    /// the server's `--sse-ping-interval`.
    pub sse_ping_interval: Option<i64>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<i32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Min-p sampling
    pub min_p: Option<f32>,
    /// Locally typical sampling, parameter p (`1.0` = disabled). b10621
    /// declares the field with no schema limits
    /// (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-schema.cpp>
    /// leaves the range commented out), so the route sanitizes rather than
    /// rejects: a present value outside the enabled range `(0.0, 1.0)`
    /// resolves to the explicit disabled form `1.0`, overriding any
    /// server-wide `--typical` default exactly like an upstream request
    /// value replaces the server default.
    pub typical_p: Option<f32>,
    /// Top-n-sigma logit filter. b10621 declares the field without limits
    /// and its sampler treats every value `<= 0.0` (default `-1.0`) as
    /// disabled, so the route maps non-positive and non-finite values to the
    /// explicit disabled form rather than rejecting them (#1436).
    pub top_n_sigma: Option<f32>,
    /// XTC removal probability. b10621 declares a SOFT `0.0..=1.0` schema
    /// limit, clamping out-of-range values into the domain instead of
    /// rejecting them; the route clamps identically (#1436).
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold. Same soft `0.0..=1.0` clamp as
    /// `xtc_probability`; values above `0.5` are in range and make XTC
    /// inert, matching upstream (#1436).
    pub xtc_threshold: Option<f32>,
    /// Suppress end-of-generation tokens so generation runs to the token
    /// budget or a stop string (#1436).
    pub ignore_eos: Option<bool>,
    /// Sampler chain order (b10621 accepts an array of stage names or a
    /// single character string). mlxcel's chain order is fixed to b10621's
    /// default, so only the default order (either spelling) is accepted as
    /// an inert configuration; any other order is rejected with a 400
    /// instead of silently sampling in a different order than requested
    /// (#1436). Held as a raw value because both shapes must be inspected.
    #[serde(default)]
    pub samplers: Option<serde_json::Value>,
    /// b10621 `lora`: a per-request adapter configuration, an array of
    /// `{id, scale}` where unlisted adapters drop to scale 0.0 (#1439).
    /// On the unfused runtime path (the default) the resolved vector becomes
    /// this request's own scale snapshot and applies to its forwards only.
    /// Under `--lora-fuse` the adapters are baked into the weights, so only a
    /// value resolving to the configuration already in force is accepted as
    /// inert; anything else is refused with a diagnostic rather than silently
    /// served on the wrong weights. Held raw because both checks inspect it.
    #[serde(default)]
    pub lora: Option<serde_json::Value>,
    // b10621 per-request speculative fields (#1433). Upstream registers
    // these seven FLAT dotted top-level keys behind a schema block that is
    // compiled out (`#if 0` in server-schema.cpp), so b10621 itself accepts
    // and ignores them. mlxcel declares the same keys and treats them
    // identically as inert: the real controls are the server-wide
    // --model-draft / --spec-draft-n-max / --draft-kind flags, and matching
    // upstream's accept-and-ignore here is the compatible behavior (a 400
    // would refuse requests b10621 answers). Held as raw values because
    // upstream's disabled block would have taken numbers and strings alike.
    /// b10621 `speculative.n_max` (inert upstream and here; see above).
    #[serde(rename = "speculative.n_max", default)]
    pub speculative_n_max: Option<serde_json::Value>,
    /// b10621 `speculative.n_min` (inert upstream and here).
    #[serde(rename = "speculative.n_min", default)]
    pub speculative_n_min: Option<serde_json::Value>,
    /// b10621 `speculative.p_min` (inert upstream and here).
    #[serde(rename = "speculative.p_min", default)]
    pub speculative_p_min: Option<serde_json::Value>,
    /// b10621 `speculative.type` (inert upstream and here).
    #[serde(rename = "speculative.type", default)]
    pub speculative_type: Option<serde_json::Value>,
    /// b10621 `speculative.ngram_min_hits` (inert upstream and here).
    #[serde(rename = "speculative.ngram_min_hits", default)]
    pub speculative_ngram_min_hits: Option<serde_json::Value>,
    /// b10621 `speculative.ngram_size_m` (inert upstream and here).
    #[serde(rename = "speculative.ngram_size_m", default)]
    pub speculative_ngram_size_m: Option<serde_json::Value>,
    /// b10621 `speculative.ngram_size_n` (inert upstream and here).
    #[serde(rename = "speculative.ngram_size_n", default)]
    pub speculative_ngram_size_n: Option<serde_json::Value>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty last N tokens
    pub repeat_last_n: Option<usize>,
    /// Stop sequences
    #[serde(default, deserialize_with = "deserialize_optional_stop")]
    pub stop: Option<Vec<String>>,
    /// Random seed
    #[serde(default, deserialize_with = "deserialize_optional_seed")]
    pub seed: Option<u64>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// DRY penalty multiplier (0.0 = disabled)
    pub dry_multiplier: Option<f32>,
    /// DRY exponential base
    pub dry_base: Option<f32>,
    /// DRY minimum match length before penalty
    pub dry_allowed_length: Option<usize>,
    /// DRY lookback window (-1 = full context)
    pub dry_penalty_last_n: Option<usize>,
    /// DRY sequence breakers. b10621 value domain (#1485): a non-empty
    /// array of STRINGS; anything else (an id array, an empty array, a
    /// non-array) is refused with upstream's own wording. Held raw because
    /// the route produces that exact diagnostic.
    #[serde(default)]
    pub dry_sequence_breakers: Option<serde_json::Value>,

    // #1485 sampling remainder: mirostat, dynamic temperature, adaptive-p,
    // min_keep, probability reporting, logit_bias, backend_sampling.
    /// Mirostat mode (`0` disabled, `1` Mirostat, `2` Mirostat 2.0). Values
    /// outside that domain are refused with a 400 where b10621 would abort
    /// its own process inside `common_sampler_init`.
    pub mirostat: Option<i32>,
    /// Mirostat target entropy tau.
    pub mirostat_tau: Option<f32>,
    /// Mirostat learning rate eta.
    pub mirostat_eta: Option<f32>,
    /// Dynamic temperature range (`0.0` = disabled; negative values are the
    /// same disabled state upstream's `delta > 0` gate produces).
    pub dynatemp_range: Option<f32>,
    /// Dynamic temperature exponent.
    pub dynatemp_exponent: Option<f32>,
    /// Adaptive-p target. b10621 declares a SOFT upper limit of `1.0`
    /// (values above clamp down); negative disables, and the sampler runs
    /// only when the request's sampler list names `adaptive_p`.
    pub adaptive_target: Option<f32>,
    /// Adaptive-p EMA decay. b10621 declares HARD limits `0.0..=0.99`;
    /// values outside are refused with a 400.
    pub adaptive_decay: Option<f32>,
    /// b10621 `min_keep`: force the truncation samplers to keep at least
    /// this many candidates. HARD limits `0..=2147483647` (a 400 outside).
    pub min_keep: Option<i64>,
    /// b10621 `n_probs` (primary) with the `logprobs` alias: when greater
    /// than zero, each generated token carries the probabilities of the top
    /// N tokens, pre-sampling by default and post-chain under
    /// `post_sampling_probs`.
    pub n_probs: Option<i64>,
    /// The `logprobs` alias b10621 declares for `n_probs`; used only when
    /// `n_probs` itself is absent, upstream's alias order. (A serde alias
    /// would reject requests carrying both keys, which upstream accepts.)
    pub logprobs: Option<i64>,
    /// Return post-sampling-chain probabilities instead of raw-logit
    /// probabilities in the `n_probs` report.
    pub post_sampling_probs: Option<bool>,
    /// b10621 `backend_sampling`: whether to sample on the backend instead
    /// of the CPU sampler chain. mlxcel's sampler IS the backend graph and
    /// has no CPU chain to switch to, so the field is accepted and inert in
    /// both values (`not_applicable` in the compatibility manifest).
    pub backend_sampling: Option<bool>,
    /// Token biases: an array of `[token, bias]` pairs or an object mapping
    /// token (an id, or a string to tokenize) to bias; `false` as a bias
    /// bans the token. Held raw because both shapes and the string-key
    /// tokenization are resolved by the route (#1485).
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,

    // #1485 grammar surfaces: declared so a constrained-decoding request is
    // refused loudly instead of silently ignored; the GBNF engine itself
    // stays deferred under issue #1485.
    /// b10621 `json_schema`: refused with a 400 naming the deferral (the
    /// OpenAI-shaped routes' `response_format` carries mlxcel's schema
    /// support).
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
    /// b10621 `grammar` (the `json_schema` alias key): a non-empty GBNF
    /// grammar string is refused with a 400 naming the deferral; the empty
    /// string is upstream's inert form and passes.
    #[serde(default)]
    pub grammar: Option<serde_json::Value>,
    /// b10621 `grammar_lazy`: `false` is inert and passes; `true` is
    /// refused with a 400 naming the deferral.
    #[serde(default)]
    pub grammar_lazy: Option<bool>,
    /// b10621 `grammar_triggers`: an empty array is inert and passes; a
    /// non-empty one is refused with a 400 naming the deferral.
    #[serde(default)]
    pub grammar_triggers: Option<serde_json::Value>,
    /// b10621 `preserved_tokens`: an empty array is inert and passes; a
    /// non-empty one is refused with a 400 naming the deferral.
    #[serde(default)]
    pub preserved_tokens: Option<serde_json::Value>,

    // thinking-token budget (Qwen3-family reasoning cap).
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    #[serde(alias = "reasoning_budget_tokens")]
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,

    /// b10621 `reasoning_control` (#1444): arm realtime reasoning control.
    /// Declared with upstream's schema (a bool, default `false`) and wired
    /// to the same sampler arming as the chat routes. As in b10621, the
    /// native response never exposes the internal completion id, so a
    /// native client has no id to address a control request to; the field's
    /// observable behavior on this route matches upstream's.
    #[serde(default)]
    pub reasoning_control: Option<bool>,

    /// structured-output `response_format` is **not** supported
    /// on the native llama-server `/completion` endpoint. The field is
    /// captured here only so the route can reject the request with a clear
    /// 400 instead of silently ignoring the schema and emitting
    /// non-conforming output. Use `/v1/chat/completions` for constrained
    /// decoding.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
}

/// Upstream's inclusive hard limits for `n_predict`
/// (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-schema.cpp>).
const N_PREDICT_MIN: i64 = -1;
const N_PREDICT_MAX: i64 = i32::MAX as i64;

impl NativeCompletionRequest {
    /// Resolve `n_predict` into the token budget the generation options take
    /// (#1441).
    ///
    /// Reproduces b10621's value domain rather than serde's: `-1` is the
    /// upstream spelling of "as many as the context allows", so it becomes an
    /// effectively unbounded request that
    /// `resolve_server_max_tokens` clamps to the effective context window; `0`
    /// stays `0`, upstream's prompt-only evaluation; and anything outside
    /// `[-1, INT32_MAX]` is refused with upstream's own diagnostic wording
    /// instead of a serde type error.
    pub fn resolve_n_predict(&self) -> Result<Option<usize>, String> {
        match self.n_predict {
            None => Ok(None),
            Some(value) if !(N_PREDICT_MIN..=N_PREDICT_MAX).contains(&value) => Err(format!(
                "Field 'n_predict': Value must be between {N_PREDICT_MIN} <= value <= \
                 {N_PREDICT_MAX}, but got {value}"
            )),
            // Clamped down to the effective context window by
            // `resolve_server_max_tokens`, which is what upstream's -1 means.
            Some(-1) => Ok(Some(usize::MAX)),
            Some(value) => Ok(Some(value as usize)),
        }
    }

    /// Validate the b10621 context-retention fields (#1472).
    ///
    /// Upstream's schema floors: `n_keep >= -1` (`-1` = retain the whole
    /// initial prompt) and `n_discard >= 0` (`0` = discard half of the
    /// non-retained window). The diagnostics keep upstream's field-error
    /// shape, as `resolve_n_predict` does.
    pub fn validate_retention(&self) -> Result<(), String> {
        if let Some(value) = self.n_keep
            && !(-1..=i64::from(i32::MAX)).contains(&value)
        {
            return Err(format!(
                "Field 'n_keep': Value must be between -1 <= value <= {}, but got {value}",
                i32::MAX
            ));
        }
        if let Some(value) = self.n_discard
            && !(0..=i64::from(i32::MAX)).contains(&value)
        {
            return Err(format!(
                "Field 'n_discard': Value must be between 0 <= value <= {}, but got {value}",
                i32::MAX
            ));
        }
        Ok(())
    }

    /// The `response_fields` projection paths, or an empty list.
    ///
    /// Upstream reads the field with a `std::vector<std::string>` default, so a
    /// value that is not an array of strings is silently ignored and the whole
    /// response is returned. Measured on the pinned binary:
    /// `"response_fields": "content"` and `["content", 5]` both answer the full
    /// object.
    pub fn response_field_paths(&self) -> Vec<String> {
        let Some(serde_json::Value::Array(items)) = self.response_fields.as_ref() else {
            return Vec::new();
        };
        let mut paths = Vec::with_capacity(items.len());
        for item in items {
            match item.as_str() {
                Some(path) => paths.push(path.to_string()),
                // A single non-string entry makes upstream's whole conversion
                // throw, and the field falls back to its default.
                None => return Vec::new(),
            }
        }
        paths
    }

    /// Validate `stream_options` the way b10621 validates it.
    ///
    /// A non-object value is ignored (measured: `"stream_options": "garbage"`
    /// answers a normal completion), while a present `include_usage` that is
    /// not a boolean is refused with upstream's field-named diagnostic.
    pub fn validate_stream_options(&self) -> Result<NativeStreamOptions, String> {
        let Some(value) = self.stream_options.as_ref() else {
            return Ok(NativeStreamOptions::default());
        };
        if !value.is_object() {
            return Ok(NativeStreamOptions::default());
        }
        serde_json::from_value::<NativeStreamOptions>(value.clone()).map_err(|_| {
            let offending = value
                .get("include_usage")
                .unwrap_or(&serde_json::Value::Null);
            format!(
                "Field 'include_usage': type must be boolean, but is {}",
                json_type_name(offending)
            )
        })
    }
}

/// b10621's nested `stream_options` block on the native completion schema
/// (#1441).
///
/// Upstream declares it as one nested field with a single boolean subfield.
/// The subfield is inert on `/completion` in both implementations: the native
/// final frame always carries the counts and the timing block, so there is no
/// usage to include or omit. It is modeled anyway so its type is validated
/// rather than the value being dropped unread.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NativeStreamOptions {
    /// Whether to include usage information in the stream.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// nlohmann-style type name, so the `include_usage` diagnostic reads the way
/// the b10621 message a client may already be matching on reads.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Tokenize request (POST /tokenize).
///
/// b10621's own body, whose four fields are independent: `content` is the
/// "mixed" prompt shape (a string, or an array whose elements are strings or
/// already-tokenized ids), `add_special` runs the BOS/EOS post-processor,
/// `parse_special` decides whether a special-token spelling written into the
/// text is recognized as that token, and `with_pieces` switches the response
/// from a flat id list to one object per token (#1442).
///
/// Every field is optional, `content` included: upstream answers an absent
/// `content` with an empty token list rather than an error.
///
/// Upstream reference:
/// <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server.cpp>
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenizeRequest {
    /// Text content to tokenize: a string, or an array of strings and token ids.
    pub content: Option<serde_json::Value>,
    /// Whether to add special tokens (BOS/EOS). Upstream default: `false`.
    pub add_special: Option<bool>,
    /// Whether special-token spellings in `content` are parsed as those tokens.
    /// Upstream default: `true`.
    pub parse_special: Option<bool>,
    /// Whether to answer with `{id, piece}` objects instead of bare ids.
    /// Upstream default: `false`.
    pub with_pieces: Option<bool>,
}

/// Detokenize request (POST /detokenize).
///
/// `tokens` is optional: upstream answers an absent list with an empty string
/// rather than an error.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DetokenizeRequest {
    /// Token IDs to decode
    pub tokens: Option<Vec<i32>>,
}

// ---------------------------------------------------------------------------
// Audio API request types (OpenAI-compatible)
// Used by: routes/audio.rs
// ---------------------------------------------------------------------------

/// Text-to-speech request (POST /v1/audio/speech).
///
/// JSON body mirroring the OpenAI `audio/speech` payload. The response is
/// binary audio rather than JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioSpeechRequest {
    /// Identifier of the speech model to use.
    pub model: String,
    /// Text to synthesize into audio.
    pub input: String,
    /// Optional named voice.
    #[serde(default)]
    pub voice: Option<String>,
    /// Optional output container (`wav` today; others are a follow-up).
    #[serde(default)]
    pub response_format: Option<String>,
    /// Optional playback-speed multiplier.
    #[serde(default)]
    pub speed: Option<f32>,
}

/// Speech-to-text request schema (POST /v1/audio/transcriptions and
/// /v1/audio/translations).
///
/// This struct mirrors the OpenAI transcription field schema for reference and
/// future reuse. The live multipart handler parses the fields directly from the
/// `multipart/form-data` stream and does not deserialize into this struct.
/// `Deserialize` is derived to keep field naming aligned with the OpenAI JSON
/// schema and to support JSON-based deserialization in tests or future contexts
/// that do not use multipart upload.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AudioTranscriptionRequest {
    /// Identifier of the speech model to use.
    #[serde(default)]
    pub model: String,
    /// Optional ISO-639-1 source-language hint.
    #[serde(default)]
    pub language: Option<String>,
    /// Optional response container (`json`, `text`, `verbose_json`).
    #[serde(default)]
    pub response_format: Option<String>,
    /// Optional sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_accepts_llama_sampling_aliases_and_scalar_stop() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "n_predict": 17,
            "repeat_penalty": 1.2,
            "repeat_last_n": 31,
            "reasoning_budget_tokens": 9,
            "stop": "END",
            "seed": -1
        }))
        .expect("llama.cpp-shaped chat request must deserialize");

        assert_eq!(request.params.max_tokens, Some(17));
        assert_eq!(request.params.repetition_penalty, Some(1.2));
        assert_eq!(request.params.repetition_context_size, Some(31));
        assert_eq!(request.params.thinking_budget_tokens, Some(9));
        assert_eq!(request.params.stop, Some(vec!["END".to_string()]));
        assert_eq!(request.params.seed, None);
    }

    #[test]
    fn completion_request_accepts_alternate_max_tokens_and_stop_array() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "prompt": "hello",
            "max_completion_tokens": 23,
            "repeat_penalty": 1.1,
            "repeat_last_n": 47,
            "reasoning_budget_tokens": 11,
            "stop": ["A", "B"],
            "seed": 42
        }))
        .expect("llama.cpp-shaped completion request must deserialize");

        assert_eq!(request.params.max_tokens, Some(23));
        assert_eq!(request.params.repetition_penalty, Some(1.1));
        assert_eq!(request.params.repetition_context_size, Some(47));
        assert_eq!(request.params.thinking_budget_tokens, Some(11));
        assert_eq!(
            request.params.stop,
            Some(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(request.params.seed, Some(42));
    }

    #[test]
    fn seed_values_fold_into_b10621s_uint32_seed_space() {
        // b10621 reads `seed` into a uint32 with an unchecked cast (#1485):
        // -2 wraps to the DETERMINISTIC seed 4294967294, only the resulting
        // sentinel 0xFFFF_FFFF (spelled -1, or 4294967295 outright) draws a
        // random seed, and larger integers wrap modulo 2^32.
        let seed = |v: serde_json::Value| {
            serde_json::from_value::<CompletionRequest>(serde_json::json!({
                "model": "test",
                "prompt": "hello",
                "seed": v
            }))
            .expect("integer seeds are accepted")
            .params
            .seed
        };
        assert_eq!(seed(serde_json::json!(-2)), Some(4_294_967_294));
        assert_eq!(
            seed(serde_json::json!(-1)),
            None,
            "-1 stays the random sentinel"
        );
        assert_eq!(
            seed(serde_json::json!(4_294_967_295_u64)),
            None,
            "the folded sentinel itself is random too, as upstream"
        );
        assert_eq!(seed(serde_json::json!(4_294_967_296_u64)), Some(0));
        assert_eq!(seed(serde_json::json!(42)), Some(42));
        let error = serde_json::from_value::<CompletionRequest>(serde_json::json!({
            "model": "test",
            "prompt": "hello",
            "seed": 1.5
        }))
        .expect_err("a fractional seed is not an integer");
        assert!(
            error.to_string().contains("seed must be an integer"),
            "{error}"
        );
    }

    #[test]
    fn message_content_default_is_empty_text() {
        let c = MessageContent::default();
        assert_eq!(c.text(), "");
        assert!(matches!(c, MessageContent::Text(_)));
    }

    /// `has_effective_text` must agree with `!text().trim().is_empty()` for
    /// every shape (issue #804): `Text` variants directly, and `Parts`
    /// variants where the borrow-only per-part `any` check has to reach the
    /// same verdict as trimming the fully joined string.
    #[test]
    fn has_effective_text_matches_text_trim_is_empty_for_all_shapes() {
        let cases = [
            MessageContent::Text(String::new()),
            MessageContent::Text("   \n\t  ".to_string()),
            MessageContent::Text("hello".to_string()),
            MessageContent::Parts(vec![]),
            MessageContent::Parts(vec![ContentPart::Text {
                text: String::new(),
            }]),
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "   ".to_string(),
                },
                ContentPart::Text {
                    text: "\n\t".to_string(),
                },
            ]),
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "   ".to_string(),
                },
                ContentPart::Text {
                    text: "hi".to_string(),
                },
            ]),
            MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl::new("data:image/png;base64,aGVsbG8="),
            }]),
        ];
        for content in cases {
            let expected = !content.text().trim().is_empty();
            assert_eq!(
                content.has_effective_text(),
                expected,
                "mismatch for {content:?}: text()={:?}",
                content.text()
            );
        }
    }

    #[test]
    fn ordered_media_parser_preserves_text_and_exact_part_order() {
        let flattened = concat!(
            "alpha<|mlxcel_ordered_audio_1|>",
            "beta<|mlxcel_ordered_image_1|>",
            "<|mlxcel_ordered_audio_2|>gamma"
        );
        assert_eq!(ordered_audio_sentinel(1), "<|mlxcel_ordered_audio_1|>");
        assert_eq!(ordered_image_sentinel(1), "<|mlxcel_ordered_image_1|>");
        assert_eq!(
            parse_ordered_media_segments(flattened).unwrap(),
            vec![
                OrderedMediaSegment::Text("alpha"),
                OrderedMediaSegment::Audio(1),
                OrderedMediaSegment::Text("beta"),
                OrderedMediaSegment::Image(1),
                OrderedMediaSegment::Audio(2),
                OrderedMediaSegment::Text("gamma"),
            ]
        );
    }

    #[test]
    fn ordered_media_parser_rejects_cardinality_drift() {
        let duplicate = concat!("<|mlxcel_ordered_audio_1|>", "<|mlxcel_ordered_audio_1|>");
        assert!(
            parse_ordered_media_segments(duplicate)
                .unwrap_err()
                .contains("out of sequence")
        );
        assert!(
            parse_ordered_media_segments("<|mlxcel_ordered_image_2|>")
                .unwrap_err()
                .contains("expected 1")
        );
    }

    #[test]
    fn message_without_content_field_deserializes_to_empty() {
        // OpenAI-compatible clients omit `content` on assistant messages whose
        // payload is a `tool_calls` array (issue #89).
        let json = r#"{
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_system_info", "arguments": "{}"}}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).expect("missing content must deserialize");
        assert_eq!(msg.content.text(), "");
        assert!(msg.tool_calls.is_some());
    }

    #[test]
    fn message_with_null_content_deserializes_to_empty() {
        // Some clients send `"content": null` rather than omitting the key.
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: Message = serde_json::from_str(json).expect("null content must deserialize");
        assert_eq!(msg.content.text(), "");
        assert!(matches!(msg.content, MessageContent::Text(_)));
    }

    #[test]
    fn message_with_string_content_deserializes() {
        let json = r#"{"role": "user", "content": "hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content.text(), "hello");
        assert!(matches!(msg.content, MessageContent::Text(_)));
    }

    #[test]
    fn message_reasoning_field_and_alias_round_trip() {
        // Issue #362: assistant `reasoning` is accepted under both the
        // `reasoning` and OpenAI-compatible `reasoning_content` spellings, and
        // its presence does not disturb the other fields.
        let from_reasoning: Message = serde_json::from_str(
            r#"{"role":"assistant","content":"hi","reasoning":"because 2+2=4"}"#,
        )
        .expect("`reasoning` must deserialize");
        assert_eq!(from_reasoning.reasoning.as_deref(), Some("because 2+2=4"));
        assert_eq!(from_reasoning.content.text(), "hi");

        let from_alias: Message = serde_json::from_str(
            r#"{"role":"assistant","content":"hi","reasoning_content":"alias text"}"#,
        )
        .expect("`reasoning_content` alias must deserialize");
        assert_eq!(from_alias.reasoning.as_deref(), Some("alias text"));

        let from_equal_pair: Message = serde_json::from_str(
            r#"{"role":"assistant","content":"hi","reasoning":"same","reasoning_content":"same"}"#,
        )
        .expect("an emitted equal reasoning pair must round-trip");
        assert_eq!(from_equal_pair.reasoning.as_deref(), Some("same"));

        let conflicting = serde_json::from_str::<Message>(
            r#"{"role":"assistant","content":"hi","reasoning":"one","reasoning_content":"two"}"#,
        )
        .expect_err("conflicting reasoning spellings must be rejected");
        assert!(
            conflicting
                .to_string()
                .contains("reasoning and reasoning_content must be identical"),
            "{conflicting}"
        );

        // Absent reasoning leaves the field None while other fields still load.
        let absent: Message =
            serde_json::from_str(r#"{"role":"user","content":"q","name":"alice"}"#)
                .expect("missing reasoning must deserialize");
        assert_eq!(absent.reasoning, None);
        assert_eq!(absent.name.as_deref(), Some("alice"));

        // Serialize uses the canonical `reasoning` key and omits it when None.
        let serialized = serde_json::to_string(&from_reasoning).unwrap();
        assert!(
            serialized.contains(r#""reasoning":"because 2+2=4""#),
            "serialized form must carry reasoning: {serialized}"
        );
        let absent_serialized = serde_json::to_string(&absent).unwrap();
        assert!(
            !absent_serialized.contains("reasoning"),
            "absent reasoning must be omitted from output: {absent_serialized}"
        );
        // Full round-trip preserves the value.
        let round_trip: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(round_trip.reasoning.as_deref(), Some("because 2+2=4"));
    }

    #[test]
    fn message_with_multimodal_content_deserializes() {
        let json = r#"{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.content, MessageContent::Parts(_)));
        assert_eq!(msg.content.text(), "describe");
        assert_eq!(
            msg.content.image_urls(),
            vec!["data:image/png;base64,AAAA".to_string()]
        );
    }

    #[test]
    fn audio_transcription_request_deserializes_and_defaults() {
        // All fields present: verify each is captured.
        let full: AudioTranscriptionRequest = serde_json::from_str(
            r#"{"model":"test-model","language":"en","response_format":"json","temperature":0.0}"#,
        )
        .expect("full form deserializes");
        assert_eq!(full.model, "test-model");
        assert_eq!(full.language.as_deref(), Some("en"));
        assert_eq!(full.response_format.as_deref(), Some("json"));
        assert_eq!(full.temperature, Some(0.0_f32));

        // Omitted optional fields must default to None.
        let minimal: AudioTranscriptionRequest =
            serde_json::from_str(r#"{"model":"m"}"#).expect("minimal form deserializes");
        assert_eq!(minimal.model, "m");
        assert!(minimal.language.is_none(), "language defaults to None");
        assert!(
            minimal.response_format.is_none(),
            "response_format defaults to None"
        );
        assert!(
            minimal.temperature.is_none(),
            "temperature defaults to None"
        );
    }

    // -----------------------------------------------------------------------
    // Per-request Gemma 4 image soft-token budget (issue #777)
    // -----------------------------------------------------------------------

    fn chat_request_with_image_part(image_url_json: &str) -> ChatCompletionRequest {
        let json = format!(
            r#"{{
                "model": "gemma-4",
                "messages": [
                    {{"role": "user", "content": [
                        {{"type": "text", "text": "what is this?"}},
                        {{"type": "image_url", "image_url": {image_url_json}}}
                    ]}}
                ]
            }}"#
        );
        serde_json::from_str(&json).expect("request must deserialize")
    }

    #[test]
    fn image_url_without_budget_fields_deserializes_and_means_no_override() {
        // The pre-existing wire shape: bare `url`. Must stay valid and must
        // resolve to "no override" so existing requests are unchanged.
        let req = chat_request_with_image_part(r#"{"url": "data:image/png;base64,aGk="}"#);
        let parts = req.image_parts();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].detail, None);
        assert_eq!(parts[0].max_soft_tokens, None);
        assert_eq!(req.image_soft_tokens(), Ok(None));
        // And the plain URL accessor still sees it.
        assert_eq!(
            req.image_urls(),
            vec!["data:image/png;base64,aGk=".to_string()]
        );
    }

    #[test]
    fn detail_low_and_high_map_to_ladder_ends() {
        let low = chat_request_with_image_part(r#"{"url": "x.png", "detail": "low"}"#);
        assert_eq!(low.image_soft_tokens(), Ok(Some(70)));

        let high = chat_request_with_image_part(r#"{"url": "x.png", "detail": "high"}"#);
        assert_eq!(high.image_soft_tokens(), Ok(Some(1120)));
    }

    #[test]
    fn detail_auto_means_no_override() {
        let auto = chat_request_with_image_part(r#"{"url": "x.png", "detail": "auto"}"#);
        assert_eq!(
            auto.image_soft_tokens(),
            Ok(None),
            "auto must leave the checkpoint default in place"
        );
    }

    #[test]
    fn detail_is_case_insensitive() {
        let req = chat_request_with_image_part(r#"{"url": "x.png", "detail": "HIGH"}"#);
        assert_eq!(req.image_soft_tokens(), Ok(Some(1120)));
    }

    #[test]
    fn unknown_detail_value_is_rejected() {
        let req = chat_request_with_image_part(r#"{"url": "x.png", "detail": "ultra"}"#);
        let err = req
            .image_soft_tokens()
            .expect_err("an unknown detail must be a client error, not silently ignored");
        assert!(err.contains("detail"), "error should name the field: {err}");
        assert!(
            err.contains("ultra"),
            "error should echo the bad value: {err}"
        );
    }

    #[test]
    fn max_soft_tokens_extension_names_an_exact_budget() {
        for budget in [70usize, 140, 280, 560, 1120] {
            let req = chat_request_with_image_part(&format!(
                r#"{{"url": "x.png", "max_soft_tokens": {budget}}}"#
            ));
            assert_eq!(req.image_soft_tokens(), Ok(Some(budget)));
        }
    }

    #[test]
    fn off_ladder_max_soft_tokens_is_rejected() {
        // An unbounded budget scales the resized image and its patch grid, so
        // it is a memory/DoS vector. Reject rather than clamp.
        for bad in ["0", "281", "100000", "18446744073709551615"] {
            let req = chat_request_with_image_part(&format!(
                r#"{{"url": "x.png", "max_soft_tokens": {bad}}}"#
            ));
            let err = req
                .image_soft_tokens()
                .expect_err("off-ladder max_soft_tokens must be rejected");
            assert!(
                err.contains("must be one of"),
                "error should name the supported values: {err}"
            );
        }
    }

    #[test]
    fn numeric_budget_wins_over_detail() {
        let req = chat_request_with_image_part(
            r#"{"url": "x.png", "detail": "low", "max_soft_tokens": 560}"#,
        );
        assert_eq!(
            req.image_soft_tokens(),
            Ok(Some(560)),
            "the mlxcel extension field takes precedence over detail"
        );
    }

    #[test]
    fn an_invalid_numeric_budget_is_rejected_even_when_detail_is_valid() {
        // The numeric field wins, so its validation must not be skipped just
        // because a valid `detail` is also present.
        let req = chat_request_with_image_part(
            r#"{"url": "x.png", "detail": "high", "max_soft_tokens": 999}"#,
        );
        assert!(req.image_soft_tokens().is_err());
    }

    #[test]
    fn agreeing_parts_resolve_to_one_budget() {
        let parts = vec![
            ImageUrl {
                url: "a.png".into(),
                detail: Some("high".into()),
                max_soft_tokens: None,
            },
            ImageUrl {
                url: "b.png".into(),
                detail: None,
                max_soft_tokens: Some(1120),
            },
            // A part with no preference does not veto the others.
            ImageUrl::new("c.png"),
        ];
        assert_eq!(resolve_request_image_soft_tokens(&parts), Ok(Some(1120)));
    }

    #[test]
    fn conflicting_parts_are_rejected() {
        let parts = vec![
            ImageUrl {
                url: "a.png".into(),
                detail: Some("low".into()),
                max_soft_tokens: None,
            },
            ImageUrl {
                url: "b.png".into(),
                detail: Some("high".into()),
                max_soft_tokens: None,
            },
        ];
        let err = resolve_request_image_soft_tokens(&parts).expect_err(
            "two different explicit budgets in one request must not be silently merged",
        );
        assert!(err.contains("conflicting"), "got: {err}");
    }

    #[test]
    fn image_url_serialization_omits_unset_budget_fields() {
        // Round-tripping a plain image part must not introduce `detail: null`
        // or `max_soft_tokens: null` into the wire payload.
        let json = serde_json::to_string(&ImageUrl::new("x.png")).expect("serializes");
        assert_eq!(json, r#"{"url":"x.png"}"#);
    }
}
