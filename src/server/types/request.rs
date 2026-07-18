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

/// Message content: either a plain string or multimodal array
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content
    Text(String),
    /// Multimodal content parts (text + images)
    Parts(Vec<ContentPart>),
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

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Accepts both `reasoning` and the OpenAI-compatible `reasoning_content`
    /// spelling via serde alias. The field is dropped from serialized output
    /// when absent, keeping existing wire shapes unchanged.
    #[serde(
        default,
        alias = "reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning: Option<String>,
}

/// Sampling parameters shared across endpoints
///
/// All parameters are optional. When not specified in the request,
/// server defaults (from CLI arguments) will be used.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate
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
    pub repetition_penalty: Option<f32>,
    /// Context size for repetition penalty
    pub repetition_context_size: Option<usize>,
    /// Logit bias for specific tokens
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility
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
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,
}

/// Stream options for controlling streaming behavior
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    /// Include token usage statistics in the final streaming chunk
    #[serde(default)]
    pub include_usage: bool,
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

/// Native llama-server completion request (POST /completion)
#[derive(Debug, Clone, Deserialize)]
pub struct NativeCompletionRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum number of tokens to predict
    pub n_predict: Option<usize>,
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<i32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Min-p sampling
    pub min_p: Option<f32>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty last N tokens
    pub repeat_last_n: Option<usize>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Random seed
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
    /// DRY sequence breaker token IDs
    pub dry_sequence_breakers: Option<Vec<i32>>,

    // thinking-token budget (Qwen3-family reasoning cap).
    /// Primary / llama.cpp-compatible name for the reasoning-token cap.
    pub thinking_budget_tokens: Option<i32>,
    /// vLLM-compatible alias for `thinking_budget_tokens`.
    pub thinking_token_budget: Option<i32>,
    /// Qwen-official alias for `thinking_budget_tokens`.
    pub thinking_budget: Option<i32>,

    /// structured-output `response_format` is **not** supported
    /// on the native llama-server `/completion` endpoint. The field is
    /// captured here only so the route can reject the request with a clear
    /// 400 instead of silently ignoring the schema and emitting
    /// non-conforming output. Use `/v1/chat/completions` for constrained
    /// decoding.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
}

/// Tokenize request (POST /tokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// Text content to tokenize
    pub content: String,
    /// Whether to add special tokens (BOS/EOS)
    pub add_special: Option<bool>,
}

/// Detokenize request (POST /detokenize)
#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    /// Token IDs to decode
    pub tokens: Vec<i32>,
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
