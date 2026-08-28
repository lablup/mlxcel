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

//! Shared sampling-config assembly for CLI and server request surfaces.
//!
//! `SamplingConfig` lives in `mlxcel-core`, but the policy for how user-facing
//! request fields map onto greedy vs sampled generation belongs in this crate's
//! control plane so every entry point applies the same defaults.

use crate::SamplingConfig;
use mlxcel_core::{LoopDetectionConfig, TokenBiasMap};

#[derive(Debug, Clone, PartialEq)]
/// Fully resolved sampling knobs before conversion into `SamplingConfig`.
///
/// The CLI and server each resolve their own defaults first, then pass the
/// merged values through this struct so the greedy/non-greedy branching remains
/// centralized in one place.
pub struct ResolvedSamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_sequence_breakers: Vec<i32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// XTC (Exclude Top Choices) per-step probability (0.0 = disabled).
    pub xtc_probability: f32,
    /// XTC probability threshold (valid range `0.0..=0.5`, enforced at the
    /// request layer). Unused while `xtc_probability == 0.0`.
    pub xtc_threshold: f32,
    /// Top-n-sigma logit filter (`0.0` = disabled). Non-negative and finite,
    /// enforced at the request layer.
    pub top_n_sigma: f32,
    /// Locally typical sampling cutoff (`1.0` = disabled). In `(0.0, 1.0]`
    /// and finite, enforced at the request layer.
    pub typical_p: f32,
    /// Repetition / frequency / presence penalty window (b10621
    /// `repeat_last_n`, #1436): `-1` = full history, `0` = stage disabled,
    /// `N > 0` = last N tokens.
    pub penalty_last_n: i32,
    pub stop_token_ids: Vec<i32>,
}

pub fn build_sampling_config(params: ResolvedSamplingParams) -> SamplingConfig {
    if params.temperature <= 0.0 {
        SamplingConfig {
            min_p: params.min_p,
            seed: params.seed,
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            // The breakers are the DRY backward-match termination condition
            // (`mlxcel_core::sampling::apply_dry_penalty`), not a sampling knob.
            // Leaving them to `SamplingConfig::greedy()`'s empty vector would let
            // the match run past the intended boundary and inflate the penalty,
            // so the greedy branch threads them through with the other four DRY
            // fields above.
            dry_sequence_breakers: params.dry_sequence_breakers,
            // XTC is a logits pre-processing step applied regardless of
            // temperature (like the repetition/DRY/frequency/presence
            // penalties above), so the greedy branch threads it through too.
            xtc_probability: params.xtc_probability,
            xtc_threshold: params.xtc_threshold,
            // Threaded through even though the greedy sampler skips it (the
            // row-filter hook gates on `temperature == 0.0 || top_k == 1`),
            // so the config faithfully mirrors what the request resolved.
            top_n_sigma: params.top_n_sigma,
            typical_p: params.typical_p,
            penalty_last_n: params.penalty_last_n,
            stop_token_ids: params.stop_token_ids,
            ..SamplingConfig::greedy()
        }
    } else {
        SamplingConfig {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            seed: params.seed,
            repetition_penalty: params.repetition_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_penalty_last_n: params.dry_penalty_last_n,
            dry_sequence_breakers: params.dry_sequence_breakers,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            xtc_probability: params.xtc_probability,
            xtc_threshold: params.xtc_threshold,
            top_n_sigma: params.top_n_sigma,
            typical_p: params.typical_p,
            penalty_last_n: params.penalty_last_n,
            stop_token_ids: params.stop_token_ids,
            token_bias: TokenBiasMap::default(),
            // Loop detection defaults to disabled here. The server control
            // plane sets `sampling.loop_detection` after this helper returns,
            // where the loaded model family is visible (see
            // `request_options::build_server_generate_options`).
            loop_detection: LoopDetectionConfig::default(),
            // The special-token allowlist is resolved per-request from the
            // tokenizer and the merged EOS set, which are not visible here;
            // the server control plane sets `sampling.xtc_special_token_ids`
            // after this helper returns (see
            // `BatchScheduler::enqueue_request`).
            xtc_special_token_ids: Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;
