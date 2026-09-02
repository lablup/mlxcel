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

//! SSE streaming response types

use serde::Serialize;

use super::response::{ChatLogprobs, CompletionLogprobs};
use crate::server::ReasoningAliasField;

/// Delta content for streaming
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reasoning / thinking channel content for models that expose a separate
    /// scratchpad (Gemma 4 `<|channel>thought\n...<channel|>`, Qwen3/DeepSeek
    /// `<think>...</think>`, Gemini `<thought>...</thought>`). Mirrors OpenAI's
    /// `o1` / DeepSeek R1 streaming convention so downstream routers and UIs
    /// can render a "thinking" status without parsing model-specific markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// OpenRouter-compatible alias for `reasoning_content`. When enabled,
    /// both fields carry identical text and are omitted together otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Tool call deltas for streaming tool call output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Incremental tool call data for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    /// Index of this tool call in the parallel array
    pub index: usize,
    /// Tool call ID (only in the first delta for this index)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool type (only in the first delta for this index)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// Incremental function data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Incremental function data for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionDelta {
    /// Function name (only in the first delta)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Incremental arguments string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Streaming choice
///
/// `finish_reason` is always serialized (as `null` in content chunks, as a
/// string value in the final chunk). This is required by the OpenAI spec and
/// expected by clients such as opencode, Continue, and Cursor.
#[derive(Debug, Clone, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    /// Always serialized: `null` in content chunks, "stop"/"length" in final
    pub finish_reason: Option<String>,
    /// Log probabilities for this chunk's token; `null` when not requested
    pub logprobs: Option<ChatLogprobs>,
}

/// Chat completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    /// Always null for now; present to satisfy strict client parsers
    pub system_fingerprint: Option<String>,
    pub choices: Vec<StreamChoice>,
    /// Only present in the final usage chunk (when stream_options.include_usage
    /// is true). Omitted from all other chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::response::Usage>,
    /// Speculative acceptance for this request (issue #1314). Attached to the
    /// finish chunk only, where the totals are final: a mid-stream chunk that
    /// carried a running `draft_n` would read as a total and be wrong. Absent
    /// unless a drafter executed at least one verify round for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<super::native_completion::SpeculativeTimings>,
}

impl ChatCompletionChunk {
    /// Create initial chunk with role
    pub fn initial(id: String, model: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Create content chunk
    pub fn content(id: String, model: String, content: String) -> Self {
        Self::content_with_logprobs(id, model, content, None)
    }

    /// Create content chunk with optional log probabilities
    pub fn content_with_logprobs(
        id: String,
        model: String,
        content: String,
        logprobs: Option<ChatLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(content),
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Create a reasoning-channel chunk.
    ///
    /// Emitted while the model is inside a scratchpad/thinking block (Gemma 4
    /// `<|channel>thought\n...<channel|>`, Qwen3/DeepSeek `<think>...</think>`,
    /// etc.). Carried as `delta.reasoning_content` instead of `delta.content`
    /// so downstream routers and UIs can surface a "thinking" state without
    /// parsing model-specific markers themselves. Matches OpenAI's `o1` and
    /// DeepSeek R1 streaming conventions.
    pub fn reasoning_content(id: String, model: String, text: String) -> Self {
        Self::reasoning_content_with_alias_field(id, model, text, ReasoningAliasField::Reasoning)
    }

    /// Build a reasoning delta while honoring the configured alias policy.
    pub fn reasoning_content_with_alias_field(
        id: String,
        model: String,
        text: String,
        alias_field: ReasoningAliasField,
    ) -> Self {
        let reasoning = alias_field.emits_reasoning().then(|| text.clone());
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: Some(text),
                    reasoning,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Attach this request's speculative acceptance counters as `timings`
    /// (issue #1314), for the finish chunk that closes a streamed response.
    ///
    /// `None` leaves the key absent, so a non-speculative stream is unchanged
    /// frame for frame.
    ///
    /// Used by: chat.rs (streaming path)
    #[must_use]
    pub fn with_speculative_timings(
        mut self,
        stats: Option<&crate::server::model_provider::SpeculativeStats>,
    ) -> Self {
        self.timings = stats.map(super::native_completion::SpeculativeTimings::from);
        self
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Create a tool call delta chunk (first delta with id/type/name).
    pub fn tool_call_start(
        id: String,
        model: String,
        index: usize,
        call_id: String,
        function_name: String,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index,
                        id: Some(call_id),
                        call_type: Some("function".to_string()),
                        function: Some(ToolCallFunctionDelta {
                            name: Some(function_name),
                            arguments: None,
                        }),
                    }]),
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Create a tool call arguments delta chunk (incremental arguments).
    pub fn tool_call_arguments(
        id: String,
        model: String,
        index: usize,
        arguments_chunk: String,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: Some(vec![ToolCallDelta {
                        index,
                        id: None,
                        call_type: None,
                        function: Some(ToolCallFunctionDelta {
                            name: None,
                            arguments: Some(arguments_chunk),
                        }),
                    }]),
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            timings: None,
        }
    }

    /// Create usage chunk sent when `stream_options.include_usage` is true.
    ///
    /// The OpenAI spec requires this to be a separate chunk with an empty
    /// `choices` array and a populated `usage` object.
    pub fn usage(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Self {
        Self::usage_with_cache(id, model, prompt_tokens, completion_tokens, 0, false)
    }

    /// Like [`usage`] but includes `prompt_tokens_details.cached_tokens` when
    /// the prompt-prefix cache feature is active.
    ///
    /// `cache_enabled` controls whether the `prompt_tokens_details` field is
    /// emitted at all: when `false` the field is omitted so existing clients
    /// that do not expect it are unaffected.
    ///
    /// Used by: chat.rs (streaming path)
    pub fn usage_with_cache(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
        cache_enabled: bool,
    ) -> Self {
        let prompt_tokens_details = if cache_enabled {
            Some(super::response::PromptTokensDetails {
                cached_tokens: cached_tokens as u64,
            })
        } else {
            None
        };
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![],
            usage: Some(super::response::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details,
            }),
            timings: None,
        }
    }
}

/// Text completion chunk for streaming
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    /// Always null for now; present to satisfy strict client parsers
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionStreamChoice>,
    /// Only present in the final usage chunk (when stream_options.include_usage
    /// is true). Omitted from all other chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::response::Usage>,
}

/// Streaming choice for text completions
///
/// `finish_reason` is always serialized (as `null` in content chunks) per the
/// OpenAI spec.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionStreamChoice {
    pub index: usize,
    pub text: String,
    /// Always serialized: `null` in content chunks, "stop"/"length" in final
    pub finish_reason: Option<String>,
    /// Log probabilities for this chunk's token; `null` when not requested
    pub logprobs: Option<CompletionLogprobs>,
}

impl CompletionChunk {
    /// Create content chunk
    pub fn content(id: String, model: String, text: String) -> Self {
        Self::content_with_logprobs(id, model, text, None)
    }

    /// Create content chunk with optional log probabilities
    pub fn content_with_logprobs(
        id: String,
        model: String,
        text: String,
        logprobs: Option<CompletionLogprobs>,
    ) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text,
                finish_reason: None,
                logprobs,
            }],
            usage: None,
        }
    }

    /// Create final chunk with finish reason
    pub fn finish(id: String, model: String, finish_reason: String) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![CompletionStreamChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Create usage chunk sent when `stream_options.include_usage` is true.
    pub fn usage(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Self {
        Self::usage_with_cache(id, model, prompt_tokens, completion_tokens, 0, false)
    }

    /// The same chunk, reporting what the prompt cache supplied (#1473).
    ///
    /// `/v1/completions` gained prompt-cache coverage with #1473, so its
    /// streaming usage chunk carries the same `prompt_tokens_details` the
    /// non-streaming response does. With the cache off the field is omitted,
    /// so a client that predates the coverage sees an unchanged chunk.
    ///
    /// Used by: completions.rs (streaming path)
    pub fn usage_with_cache(
        id: String,
        model: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
        cache_enabled: bool,
    ) -> Self {
        let prompt_tokens_details = if cache_enabled {
            Some(super::response::PromptTokensDetails {
                cached_tokens: cached_tokens as u64,
            })
        } else {
            None
        };
        Self {
            id,
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            system_fingerprint: None,
            choices: vec![],
            usage: Some(super::response::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details,
            }),
        }
    }
}
