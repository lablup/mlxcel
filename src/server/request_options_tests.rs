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

use super::{
    LOOP_DETECTION_RECOMMENDED, RequestOptionOverrides, build_server_generate_options,
    loop_detection_from_request, resolve_loop_detection,
};
use crate::server::ServerConfig;
use mlxcel_core::LoopDetectionConfig;

#[test]
fn build_server_generate_options_uses_server_defaults() {
    let config = ServerConfig::default();

    let options = build_server_generate_options(&config, RequestOptionOverrides::default());

    assert_eq!(options.max_tokens, config.default_max_tokens);
    assert_eq!(options.sampling.temperature, config.default_temperature);
    assert_eq!(options.sampling.top_k, config.default_top_k);
    assert_eq!(options.sampling.top_p, config.default_top_p);
    assert_eq!(options.sampling.min_p, config.default_min_p);
    assert_eq!(
        options.sampling.repetition_penalty,
        config.default_repetition_penalty
    );
    assert_eq!(
        options.sampling.dry_multiplier,
        config.default_dry_multiplier
    );
    assert_eq!(options.stop_sequences, None);
}

#[test]
fn build_server_generate_options_applies_request_overrides() {
    let config = ServerConfig::default();
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            max_tokens: Some(7),
            temperature: Some(0.0),
            top_k: Some(99),
            top_p: Some(0.5),
            min_p: Some(0.2),
            repetition_penalty: Some(1.3),
            seed: Some(42),
            frequency_penalty: Some(0.4),
            presence_penalty: Some(0.5),
            dry_multiplier: Some(0.9),
            dry_base: Some(2.2),
            dry_allowed_length: Some(5),
            dry_penalty_last_n: Some(17),
            dry_sequence_breakers: Some(vec![1, 2]),
            stop_sequences: Some(vec!["stop".to_string()]),
            priority: crate::server::batch::RequestPriority::High,
            reasoning_budget: crate::server::config::ReasoningBudgetOverride::default(),
            thinking_enter_block_on_start: true,
            loop_detection_request: None,
        },
    );

    assert_eq!(options.max_tokens, 7);
    assert_eq!(options.sampling.temperature, 0.0);
    assert_eq!(options.sampling.top_k, 1);
    assert_eq!(options.sampling.top_p, 1.0);
    assert_eq!(options.sampling.min_p, 0.2);
    assert_eq!(options.sampling.seed, Some(42));
    assert_eq!(options.sampling.repetition_penalty, 1.3);
    assert_eq!(options.sampling.frequency_penalty, 0.4);
    assert_eq!(options.sampling.presence_penalty, 0.5);
    assert_eq!(options.sampling.dry_multiplier, 0.9);
    assert_eq!(options.sampling.dry_base, 2.2);
    assert_eq!(options.sampling.dry_allowed_length, 5);
    assert_eq!(options.sampling.dry_penalty_last_n, 17);
    assert_eq!(options.sampling.dry_sequence_breakers, Vec::<i32>::new());
    assert_eq!(options.stop_sequences, Some(vec!["stop".to_string()]));
}

// -- loop detection (issue #432) --

#[test]
fn loop_detection_disabled_by_default_for_non_gemma() {
    // Default config is a non-Gemma model: loop detection stays disabled so the
    // bit-exact baseline is preserved.
    let config = ServerConfig::default();
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn loop_detection_from_request_none_when_no_fields() {
    assert_eq!(loop_detection_from_request(None, None, None), None);
}

#[test]
fn loop_detection_from_request_some_when_any_field_set() {
    // Even an explicit disable (max_pattern_size = 0) is authoritative.
    let only_disable = loop_detection_from_request(Some(0), None, None);
    assert_eq!(only_disable, Some(LoopDetectionConfig::new(0, 0, 0)));
    assert!(!only_disable.unwrap().is_enabled());

    let full = loop_detection_from_request(Some(20), Some(1), Some(4));
    assert_eq!(full, Some(LoopDetectionConfig::new(1, 20, 4)));
}

#[test]
fn resolve_loop_detection_precedence() {
    let req = LoopDetectionConfig::new(2, 5, 3);
    let global = LoopDetectionConfig::new(1, 10, 6);

    // Explicit request wins over everything, including the family default-on.
    assert_eq!(
        resolve_loop_detection(Some(req), Some(global), true),
        req,
        "explicit request beats global and family"
    );

    // Global override beats the family default-on (and may force-disable).
    assert_eq!(resolve_loop_detection(None, Some(global), true), global);
    let forced_off = LoopDetectionConfig::disabled();
    assert_eq!(
        resolve_loop_detection(None, Some(forced_off), true),
        forced_off,
        "operator can force-disable even for the Gemma 4 family"
    );

    // Gemma 4 family default-on applies unconditionally when nothing else is set.
    assert_eq!(
        resolve_loop_detection(None, None, true),
        LOOP_DETECTION_RECOMMENDED
    );

    // Disabled baseline for non-family models when nothing applies.
    assert_eq!(
        resolve_loop_detection(None, None, false),
        LoopDetectionConfig::disabled()
    );
}

#[test]
fn gemma4_family_auto_enables_by_default_without_amplifier() {
    let config = ServerConfig {
        model_is_gemma4_family: true,
        ..Default::default()
    };

    // A plain Gemma 4 chat with no tools and no response_format still gets
    // detection enabled by default (the "Best" engine-level activation surface).
    let plain = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert_eq!(plain.sampling.loop_detection, LOOP_DETECTION_RECOMMENDED);
    assert!(plain.sampling.loop_detection.is_enabled());
}

#[test]
fn non_gemma4_family_stays_disabled_by_default() {
    let config = ServerConfig::default(); // model_is_gemma4_family = false
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn explicit_request_disable_overrides_family_default_on() {
    let config = ServerConfig {
        model_is_gemma4_family: true,
        ..Default::default()
    };

    // A per-request explicit disable (max_pattern_size = 0) must win over the
    // Gemma 4 family default-on.
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            loop_detection_request: loop_detection_from_request(Some(0), None, None),
            ..Default::default()
        },
    );
    assert!(!options.sampling.loop_detection.is_enabled());
}

#[test]
fn explicit_request_tune_overrides_family_default_on() {
    let config = ServerConfig {
        model_is_gemma4_family: true,
        ..Default::default()
    };

    // A per-request tune wins over the family default-on threshold.
    let tuned = loop_detection_from_request(Some(8), Some(2), Some(3));
    let options = build_server_generate_options(
        &config,
        RequestOptionOverrides {
            loop_detection_request: tuned,
            ..Default::default()
        },
    );
    assert_eq!(
        options.sampling.loop_detection,
        LoopDetectionConfig::new(2, 8, 3)
    );
}

#[test]
fn global_override_applies_to_non_gemma4() {
    let config = ServerConfig {
        loop_detection: Some(LOOP_DETECTION_RECOMMENDED),
        ..Default::default()
    };
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert_eq!(options.sampling.loop_detection, LOOP_DETECTION_RECOMMENDED);
}

#[test]
fn global_override_can_force_disable_gemma4() {
    // An operator can globally force-disable even for the Gemma 4 family.
    let config = ServerConfig {
        model_is_gemma4_family: true,
        loop_detection: Some(LoopDetectionConfig::disabled()),
        ..Default::default()
    };
    let options = build_server_generate_options(&config, RequestOptionOverrides::default());
    assert!(!options.sampling.loop_detection.is_enabled());
}
