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

use super::default_generation_settings;
use crate::server::config::ServerConfig;

#[test]
fn props_reports_the_resolved_dry_sequence_breakers() {
    let config = ServerConfig {
        default_dry_sequence_breakers: vec![198, 271],
        ..Default::default()
    };

    let settings = default_generation_settings(&config);

    assert_eq!(
        settings["dry_sequence_breakers"],
        serde_json::json!([198, 271]),
        "an operator must be able to read back what --dry-sequence-breaker resolved to"
    );
}

#[test]
fn props_reports_an_unset_breaker_list_as_empty_rather_than_omitting_it() {
    let settings = default_generation_settings(&ServerConfig::default());

    // Present-but-empty and absent are different answers to "is this flag
    // doing anything". The gap #1103 closed was that the field was absent, so
    // there was no way to tell the flag was inert.
    assert_eq!(
        settings["dry_sequence_breakers"],
        serde_json::json!([]),
        "the key must exist even when no breakers are configured"
    );
}

#[test]
fn props_reports_all_five_dry_fields() {
    let config = ServerConfig {
        default_dry_multiplier: 0.8,
        default_dry_base: 1.9,
        default_dry_allowed_length: 3,
        default_dry_penalty_last_n: 64,
        default_dry_sequence_breakers: vec![198],
        ..Default::default()
    };

    let settings = default_generation_settings(&config);

    // Compare against the config values themselves rather than against
    // literals: these are `f32`, and a literal such as `0.8` is an `f64`, so
    // hardcoding one would fail on the widening rather than on the payload.
    assert_eq!(
        settings["dry_multiplier"],
        serde_json::json!(config.default_dry_multiplier)
    );
    assert_eq!(
        settings["dry_base"],
        serde_json::json!(config.default_dry_base)
    );
    assert_eq!(settings["dry_allowed_length"], serde_json::json!(3));
    assert_eq!(settings["dry_penalty_last_n"], serde_json::json!(64));
    assert_eq!(settings["dry_sequence_breakers"], serde_json::json!([198]));
}
