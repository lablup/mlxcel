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

//! Shared request-to-generation option adapters for server routes.
//!
//! Chat and native completion requests expose slightly different field names,
//! but once their overrides are resolved they should map onto the same
//! `ServerGenerateOptions` policy.

use super::{ServerConfig, ServerGenerateOptions};
use crate::sampling::{ResolvedSamplingParams, build_sampling_config};
use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use mlxcel_core::sampling::LogprobsConfig;

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RequestOptionOverrides {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<usize>,
    pub dry_penalty_last_n: Option<usize>,
    pub dry_sequence_breakers: Option<Vec<i32>>,
    pub stop_sequences: Option<Vec<String>>,
    pub priority: RequestPriority,
    /// per-request thinking-token budget override.
    pub reasoning_budget: ReasoningBudgetOverride,
    /// whether the rendered prompt primes `<think>\n` (true for
    /// chat endpoints) vs. takes a free-form prompt (false for raw text
    /// completion endpoints).
    pub thinking_enter_block_on_start: bool,
}

pub(crate) fn build_server_generate_options(
    config: &ServerConfig,
    overrides: RequestOptionOverrides,
) -> ServerGenerateOptions {
    let sampling = build_sampling_config(ResolvedSamplingParams {
        temperature: overrides.temperature.unwrap_or(config.default_temperature),
        top_k: overrides.top_k.unwrap_or(config.default_top_k),
        top_p: overrides.top_p.unwrap_or(config.default_top_p),
        min_p: overrides.min_p.unwrap_or(config.default_min_p),
        seed: overrides.seed.or(config.default_seed),
        repetition_penalty: overrides
            .repetition_penalty
            .unwrap_or(config.default_repetition_penalty),
        dry_multiplier: overrides
            .dry_multiplier
            .unwrap_or(config.default_dry_multiplier),
        dry_base: overrides.dry_base.unwrap_or(config.default_dry_base),
        dry_allowed_length: overrides
            .dry_allowed_length
            .unwrap_or(config.default_dry_allowed_length),
        dry_penalty_last_n: overrides
            .dry_penalty_last_n
            .unwrap_or(config.default_dry_penalty_last_n),
        dry_sequence_breakers: overrides.dry_sequence_breakers.unwrap_or_default(),
        frequency_penalty: overrides
            .frequency_penalty
            .unwrap_or(config.default_frequency_penalty),
        presence_penalty: overrides
            .presence_penalty
            .unwrap_or(config.default_presence_penalty),
        stop_token_ids: Vec::new(),
    });

    ServerGenerateOptions {
        max_tokens: overrides.max_tokens.unwrap_or(config.default_max_tokens),
        sampling,
        stop_sequences: overrides.stop_sequences,
        priority: overrides.priority,
        logprobs: LogprobsConfig::default(),
        reasoning_budget: overrides.reasoning_budget,
        thinking_enter_block_on_start: overrides.thinking_enter_block_on_start,
        // Default unset; chat routes populate this when the prompt cache
        // store is installed (see `src/server/routes/chat.rs`).
        prompt_cache_ctx: None,
        // defaults to `None` (unconstrained generation). Routes
        // that handle `response_format` populate this after the helper
        // returns — see `chat.rs` and `completions.rs`.
        structured: None,
    }
}

#[cfg(test)]
#[path = "request_options_tests.rs"]
mod tests;
