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

//! llama-server b10621 chat-template, reasoning, and output-parsing options
//! (issue #1447).
//!
//! Matching names do not prove matching behavior, which is the whole reason
//! this group exists: mlxcel already had a Jinja engine, `reasoning_content`,
//! a thinking budget and a `reasoning_effort` kwarg, but `--reasoning-format`,
//! `--reasoning`, `--skip-chat-parsing` and `--prefill-assistant` did not
//! parse at all, so a llama-server command line asking for any of them failed
//! outright and a deployment could not move its thoughts out of
//! `reasoning_content`.
//!
//! # How upstream implements these
//!
//! Three of them are not response-shaping at all: b10621's `--reasoning`,
//! `--reasoning-effort` and `--reasoning-preserve` handlers write into
//! `params.default_template_kwargs`, the same map `--chat-template-kwargs`
//! fills
//! (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>).
//! mlxcel already has that map, its request-level override, and its merge
//! rule, so those three are implemented by writing the same three keys into
//! the same place rather than by inventing a parallel mechanism.
//!
//! The rest do shape the response and are carried as their own settings:
//! `--reasoning-format` chooses where thoughts land, `--skip-chat-parsing`
//! turns every parser off, `--prefill-assistant` decides whether a trailing
//! assistant message is continued or answered, and
//! `--reasoning-budget-message` is injected when the budget runs out.
//!
//! # Environment bindings
//!
//! Value-taking options bind their variable through clap. The four `--x` /
//! `--no-x` pairs do not, for the reason `ggml_compat_args` documents: b10621
//! reads a bool pair through `parse_bool_value` plus a `LLAMA_ARG_NO_*` alias,
//! and clap's boolish parser accepts a wider vocabulary and errors outside it.
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;
use serde_json::Value;

use crate::cli::ggml_compat_args::{FALSEY, TRUTHY, env_bool_pair};
use crate::server::reasoning_format::{ReasoningFormat, UnknownReasoningFormat};

/// llama-server b10621 chat-template / reasoning / parsing options.
///
/// Flattened into both server binaries so the two surfaces cannot drift; see
/// `tests/cli_help_consistency.rs`.
#[derive(Args, Debug, Clone, Default)]
pub struct ChatCompatArgs {
    // ── template engine ─────────────────────────────────────────────────
    /// b10621 `--jinja` (positive half). Inert: mlxcel always renders chat
    /// prompts through its Jinja engine.
    #[arg(long = "jinja", overrides_with = "no_jinja", hide = true)]
    pub jinja: bool,

    /// b10621 `--no-jinja` (negative half).
    #[arg(long = "no-jinja", overrides_with = "jinja", hide = true)]
    pub no_jinja: bool,

    // ── reasoning ───────────────────────────────────────────────────────
    /// b10621 `--reasoning`: on | off | auto.
    ///
    /// Writes the `enable_thinking` chat-template kwarg, exactly as upstream
    /// does; `auto` writes nothing and leaves the template's own default.
    #[arg(
        long = "reasoning",
        env = "LLAMA_ARG_REASONING",
        value_name = "[on|off|auto]",
        hide = true
    )]
    pub reasoning: Option<String>,

    /// b10621 `--reasoning-format`: none | deepseek | deepseek-legacy | auto.
    ///
    /// Chooses whether a model's thoughts are reported in
    /// `message.reasoning_content`, left in `message.content`, or both.
    #[arg(
        long = "reasoning-format",
        env = "LLAMA_ARG_THINK",
        value_name = "FORMAT",
        hide = true
    )]
    pub reasoning_format: Option<String>,

    /// b10621 `--reasoning-effort`: the level handed to the chat template.
    ///
    /// Writes the `reasoning_effort` chat-template kwarg, as upstream does;
    /// `default` erases it and keeps the template's own default.
    #[arg(
        long = "reasoning-effort",
        env = "LLAMA_ARG_REASONING_EFFORT",
        value_name = "LEVEL",
        hide = true
    )]
    pub reasoning_effort: Option<String>,

    /// b10621 `--reasoning-budget-message`: text injected before the
    /// end-of-thinking tag when the reasoning budget is exhausted.
    #[arg(
        long = "reasoning-budget-message",
        env = "LLAMA_ARG_THINK_BUDGET_MESSAGE",
        value_name = "MESSAGE",
        hide = true
    )]
    pub reasoning_budget_message: Option<String>,

    /// b10621 `--reasoning-preserve` (positive half). Writes the
    /// `preserve_reasoning` chat-template kwarg.
    #[arg(
        long = "reasoning-preserve",
        overrides_with = "no_reasoning_preserve",
        hide = true
    )]
    pub reasoning_preserve: bool,

    /// b10621 `--no-reasoning-preserve` (negative half).
    #[arg(
        long = "no-reasoning-preserve",
        overrides_with = "reasoning_preserve",
        hide = true
    )]
    pub no_reasoning_preserve: bool,

    // ── output parsing ──────────────────────────────────────────────────
    /// b10621 `--skip-chat-parsing` (positive half): force a pure content
    /// parser, so reasoning and tool calls stay in `message.content`.
    #[arg(
        long = "skip-chat-parsing",
        overrides_with = "no_skip_chat_parsing",
        hide = true
    )]
    pub skip_chat_parsing: bool,

    /// b10621 `--no-skip-chat-parsing` (negative half).
    #[arg(
        long = "no-skip-chat-parsing",
        overrides_with = "skip_chat_parsing",
        hide = true
    )]
    pub no_skip_chat_parsing: bool,

    // ── assistant prefill ───────────────────────────────────────────────
    /// b10621 `--prefill-assistant` (positive half, the default): a trailing
    /// assistant message is continued rather than answered.
    #[arg(
        long = "prefill-assistant",
        overrides_with = "no_prefill_assistant",
        hide = true
    )]
    pub prefill_assistant: bool,

    /// b10621 `--no-prefill-assistant` (negative half): a trailing assistant
    /// message is a complete message and a fresh reply is generated.
    #[arg(
        long = "no-prefill-assistant",
        overrides_with = "prefill_assistant",
        hide = true
    )]
    pub no_prefill_assistant: bool,

    // ── prompt-text handling (no server-side effect) ────────────────────
    /// b10621 `--escape` (positive half, the default).
    #[arg(long = "escape", overrides_with = "no_escape", hide = true)]
    pub escape: bool,

    /// b10621 `--no-escape` (negative half).
    #[arg(long = "no-escape", overrides_with = "escape", hide = true)]
    pub no_escape: bool,

    /// b10621 `--special`: render special tokens in the output.
    #[arg(long = "special", hide = true)]
    pub special: bool,
}

/// The settings a [`ChatCompatArgs`] resolves to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatCompatResolution {
    /// Where a model's thoughts are reported.
    pub reasoning_format: ReasoningFormat,
    /// `--skip-chat-parsing`: every parser off, everything in `content`.
    pub skip_chat_parsing: bool,
    /// `--no-prefill-assistant`: a trailing assistant message is complete.
    /// `false` (prefill enabled) is b10621's default and mlxcel's behavior.
    pub no_prefill_assistant: bool,
    /// `--reasoning-budget-message`, injected when the budget is exhausted.
    pub reasoning_budget_message: Option<String>,
    /// Chat-template kwargs the reasoning flags contribute, in b10621's own
    /// key vocabulary. Applied *under* an explicit `--chat-template-kwargs`
    /// entry of the same name and under a per-request kwarg.
    pub template_kwargs: Vec<(String, Value)>,
}

/// A value outside the option's accepted set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatCompatError {
    /// The option as the operator wrote it.
    pub option: &'static str,
    /// The message, already naming the value and the accepted set.
    pub message: String,
}

impl std::fmt::Display for ChatCompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.option, self.message)
    }
}

impl std::error::Error for ChatCompatError {}

impl ChatCompatArgs {
    /// Apply the b10621 environment bindings clap cannot express.
    ///
    /// The four `--x` / `--no-x` pairs, for the reason
    /// [`crate::cli::ggml_compat_args`] documents. An explicit command-line
    /// occurrence of either half wins.
    ///
    /// # Errors
    ///
    /// Returns the variable and its value when b10621's `parse_bool_value`
    /// would throw on it, which is what b10621 does.
    pub fn apply_env_bindings(&mut self) -> Result<(), (&'static str, String)> {
        for (var, positive, negative) in [
            ("LLAMA_ARG_JINJA", &mut self.jinja, &mut self.no_jinja),
            (
                "LLAMA_ARG_REASONING_PRESERVE",
                &mut self.reasoning_preserve,
                &mut self.no_reasoning_preserve,
            ),
            (
                "LLAMA_ARG_SKIP_CHAT_PARSING",
                &mut self.skip_chat_parsing,
                &mut self.no_skip_chat_parsing,
            ),
            (
                "LLAMA_ARG_PREFILL_ASSISTANT",
                &mut self.prefill_assistant,
                &mut self.no_prefill_assistant,
            ),
        ] {
            if *positive || *negative {
                continue;
            }
            match env_bool_pair(var) {
                None => {}
                Some(Ok(true)) => *positive = true,
                Some(Ok(false)) => *negative = true,
                Some(Err(raw)) => return Err((var, raw)),
            }
        }
        // `--escape` and `--special` carry no environment binding upstream.
        Ok(())
    }

    /// Resolve into the settings the server acts on.
    ///
    /// # Errors
    ///
    /// Returns the first option whose value b10621 would refuse, or that asks
    /// for behavior mlxcel cannot produce.
    pub fn resolve(&self) -> Result<ChatCompatResolution, ChatCompatError> {
        // ── template engine ─────────────────────────────────────────────
        if self.no_jinja {
            return Err(ChatCompatError {
                option: "--no-jinja",
                message: "mlxcel renders every chat prompt through its Jinja engine and has no \
                     legacy non-Jinja formatter to fall back to, so the template engine cannot \
                     be turned off. Drop the flag; `--jinja` is what mlxcel already does."
                    .to_owned(),
            });
        }

        // ── prompt-text handling ────────────────────────────────────────
        if self.special {
            return Err(ChatCompatError {
                option: "--special",
                message: "mlxcel never renders special tokens in a response; its detokenizer \
                          drops them and the chat routes strip structural markers on top. There \
                          is no switch to turn that off."
                    .to_owned(),
            });
        }

        // ── reasoning ───────────────────────────────────────────────────
        let mut template_kwargs: Vec<(String, Value)> = Vec::new();
        if let Some(raw) = non_empty(self.reasoning.as_deref()) {
            // Upstream: truthy sets `enable_thinking=true`, falsey sets
            // `false`, `auto` writes nothing, anything else throws.
            if TRUTHY.contains(&raw) {
                template_kwargs.push(("enable_thinking".to_owned(), Value::Bool(true)));
            } else if FALSEY.contains(&raw) {
                template_kwargs.push(("enable_thinking".to_owned(), Value::Bool(false)));
            } else if !matches!(raw, "auto" | "-1") {
                return Err(ChatCompatError {
                    option: "--reasoning",
                    message: format!("unknown value '{raw}'; expected one of: on, off, auto"),
                });
            }
        }
        if let Some(raw) = non_empty(self.reasoning_effort.as_deref())
            && raw != "default"
        {
            // Upstream stores the JSON-dumped value, so a template reading
            // `reasoning_effort` sees a string. `default` erases the key,
            // which here means contributing nothing.
            template_kwargs.push(("reasoning_effort".to_owned(), Value::String(raw.to_owned())));
        }
        if self.reasoning_preserve || self.no_reasoning_preserve {
            template_kwargs.push((
                "preserve_reasoning".to_owned(),
                Value::Bool(self.reasoning_preserve),
            ));
        }

        let reasoning_format = match non_empty(self.reasoning_format.as_deref()) {
            None => ReasoningFormat::default(),
            Some(raw) => ReasoningFormat::parse(raw).map_err(|UnknownReasoningFormat(value)| {
                ChatCompatError {
                    option: "--reasoning-format",
                    message: UnknownReasoningFormat(value).to_string(),
                }
            })?,
        };

        Ok(ChatCompatResolution {
            reasoning_format,
            skip_chat_parsing: self.skip_chat_parsing,
            no_prefill_assistant: self.no_prefill_assistant,
            reasoning_budget_message: non_empty(self.reasoning_budget_message.as_deref())
                .map(str::to_owned),
            template_kwargs,
        })
    }
}

/// `Some(trimmed)` when a value is present and not whitespace-only.
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

#[cfg(test)]
#[path = "chat_compat_args_tests.rs"]
mod tests;
