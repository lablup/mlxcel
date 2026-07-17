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

use super::{ResolvedSamplingParams, build_sampling_config};

fn sample_params() -> ResolvedSamplingParams {
    ResolvedSamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.1,
        seed: Some(7),
        repetition_penalty: 1.1,
        dry_multiplier: 0.5,
        dry_base: 1.9,
        dry_allowed_length: 3,
        dry_penalty_last_n: 16,
        dry_sequence_breakers: vec![10, 20],
        frequency_penalty: 0.2,
        presence_penalty: 0.3,
        xtc_probability: 0.4,
        xtc_threshold: 0.15,
        stop_token_ids: vec![1, 2],
    }
}

#[test]
fn build_sampling_config_keeps_sampling_fields_when_temperature_is_positive() {
    let params = sample_params();
    let config = build_sampling_config(params.clone());

    assert_eq!(config.temperature, params.temperature);
    assert_eq!(config.top_k, params.top_k);
    assert_eq!(config.top_p, params.top_p);
    assert_eq!(config.min_p, params.min_p);
    assert_eq!(config.seed, params.seed);
    assert_eq!(config.repetition_penalty, params.repetition_penalty);
    assert_eq!(config.dry_multiplier, params.dry_multiplier);
    assert_eq!(config.dry_base, params.dry_base);
    assert_eq!(config.dry_allowed_length, params.dry_allowed_length);
    assert_eq!(config.dry_penalty_last_n, params.dry_penalty_last_n);
    assert_eq!(config.dry_sequence_breakers, params.dry_sequence_breakers);
    assert_eq!(config.frequency_penalty, params.frequency_penalty);
    assert_eq!(config.presence_penalty, params.presence_penalty);
    assert_eq!(config.xtc_probability, params.xtc_probability);
    assert_eq!(config.xtc_threshold, params.xtc_threshold);
    assert_eq!(config.stop_token_ids, params.stop_token_ids);
}

#[test]
fn build_sampling_config_uses_greedy_defaults_when_temperature_is_zero() {
    let mut params = sample_params();
    params.temperature = 0.0;
    let config = build_sampling_config(params.clone());

    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.dry_sequence_breakers, Vec::<i32>::new());
    assert_eq!(config.min_p, params.min_p);
    assert_eq!(config.seed, params.seed);
    assert_eq!(config.repetition_penalty, params.repetition_penalty);
    assert_eq!(config.frequency_penalty, params.frequency_penalty);
    assert_eq!(config.presence_penalty, params.presence_penalty);
    // XTC is applied regardless of temperature, so the greedy branch still
    // threads the resolved probability/threshold through.
    assert_eq!(config.xtc_probability, params.xtc_probability);
    assert_eq!(config.xtc_threshold, params.xtc_threshold);
    assert_eq!(config.stop_token_ids, params.stop_token_ids);
}
