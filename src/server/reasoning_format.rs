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

//! Where a model's thoughts end up in a chat response (issue #1447).
//!
//! b10621's `--reasoning-format` (`LLAMA_ARG_THINK`) chooses between three
//! placements, and the per-request `reasoning_format` field overrides it:
//!
//! | Value | `message.content` | `message.reasoning_content` |
//! |---|---|---|
//! | `none` | the thoughts, tags and all, left unparsed | absent |
//! | `deepseek` | the answer only | the thoughts |
//! | `deepseek-legacy` | the answer **with** the `<think>` tags | the thoughts |
//! | `auto` (default) | detected from the template; `deepseek` here | the thoughts |
//!
//! mlxcel's pre-#1447 behavior was `deepseek` unconditionally: thoughts always
//! went to `reasoning_content` and never appeared in `content`. A deployment
//! whose client reads thoughts out of `content` (every pre-`reasoning_content`
//! integration does) therefore saw an empty answer where llama-server showed
//! the reasoning, and had no flag to change it.
//!
//! `auto` resolves to `deepseek` rather than being carried as a third state:
//! upstream's `auto` inspects the chat template's own declared format, and
//! mlxcel's reasoning split is the same `<think>` / `<|channel>` marker set for
//! every family it supports, so there is nothing else for it to detect. The
//! entry in `compat/llama-server/b10621/chat-templates.json` records that.

use std::fmt;

/// b10621 `--reasoning-format` placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// `auto`: detect from the template. Resolves to [`Self::DeepSeek`] here.
    #[default]
    Auto,
    /// `none`: leave thoughts unparsed in `message.content`.
    None,
    /// `deepseek`: put thoughts in `message.reasoning_content`.
    DeepSeek,
    /// `deepseek-legacy`: keep the `<think>` tags in `message.content` while
    /// also populating `message.reasoning_content`.
    DeepSeekLegacy,
}

impl ReasoningFormat {
    /// Parse a b10621 `--reasoning-format` / `reasoning_format` value.
    ///
    /// Matches upstream's `common_reasoning_format_from_name`: the four names
    /// exactly, case-sensitively. Returns the requested value in the error so
    /// the caller can quote it.
    pub fn parse(value: &str) -> Result<Self, UnknownReasoningFormat> {
        match value {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "deepseek" => Ok(Self::DeepSeek),
            "deepseek-legacy" => Ok(Self::DeepSeekLegacy),
            other => Err(UnknownReasoningFormat(other.to_owned())),
        }
    }

    /// The name b10621 uses for this format, for diagnostics and `/props`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::DeepSeek => "deepseek",
            Self::DeepSeekLegacy => "deepseek-legacy",
        }
    }

    /// True when thoughts belong in `message.reasoning_content`.
    #[must_use]
    pub const fn emits_reasoning_content(self) -> bool {
        matches!(self, Self::Auto | Self::DeepSeek | Self::DeepSeekLegacy)
    }

    /// True when the thinking block stays in `message.content`.
    #[must_use]
    pub const fn keeps_thoughts_in_content(self) -> bool {
        matches!(self, Self::None | Self::DeepSeekLegacy)
    }
}

impl fmt::Display for ReasoningFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A `--reasoning-format` / `reasoning_format` value outside b10621's four.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownReasoningFormat(pub String);

impl fmt::Display for UnknownReasoningFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown reasoning format '{}'; expected one of: none, deepseek, \
             deepseek-legacy, auto",
            self.0
        )
    }
}

impl std::error::Error for UnknownReasoningFormat {}

/// How one response's thoughts and answer should be reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapedResponse {
    /// The `message.content` value.
    pub content: String,
    /// The `message.reasoning_content` value, absent when the format does not
    /// emit one or the model produced no thoughts.
    pub reasoning_content: Option<String>,
}

/// Place a response's thoughts according to `format`.
///
/// `answer` is the content with the thinking block removed (mlxcel's
/// pre-#1447 `content`), `with_thoughts` produces the same content with the
/// block left in place, and `reasoning` is the extracted thinking text.
///
/// `with_thoughts` is a closure rather than a `String` because two of the four
/// formats never look at it, and building it means a second pass over the
/// generated text on every request. Taking both content forms rather than
/// re-deriving one here keeps this function pure and leaves the marker
/// vocabulary in `tool_calls::parser`, which owns it.
#[must_use]
pub fn shape_response(
    format: ReasoningFormat,
    answer: String,
    with_thoughts: impl FnOnce() -> String,
    reasoning: Option<String>,
) -> ShapedResponse {
    ShapedResponse {
        content: if format.keeps_thoughts_in_content() {
            with_thoughts()
        } else {
            answer
        },
        reasoning_content: format
            .emits_reasoning_content()
            .then_some(reasoning)
            .flatten(),
    }
}

#[cfg(test)]
#[path = "reasoning_format_tests.rs"]
mod tests;
