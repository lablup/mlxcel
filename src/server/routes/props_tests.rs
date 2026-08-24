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
use crate::server::types::PropsResponse;

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

/// The reported key set is the contract this function was extracted to make
/// assertable, so assert it. Without this, a regression that dropped `top_k`
/// or `seed` from the payload would pass every other test in this file.
#[test]
fn props_reports_exactly_the_documented_key_set() {
    let settings = default_generation_settings(&ServerConfig::default());
    let mut keys: Vec<&str> = settings
        .as_object()
        .expect("default_generation_settings is a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();

    assert_eq!(
        keys,
        vec![
            "dry_allowed_length",
            "dry_base",
            "dry_multiplier",
            "dry_penalty_last_n",
            "dry_sequence_breakers",
            "frequency_penalty",
            "min_p",
            "n_predict",
            "presence_penalty",
            "repeat_last_n",
            "repeat_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        ],
        "the /props payload key set changed. Adding a key is fine, update this list; \
         losing one is a silent break for anyone reading it back."
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

/// `/props` reports the KV cache mode and `--kv-bits` the server actually
/// resolved (issue #1350).
///
/// `docs/turbo-kv-cache.md` tells operators that the mode announced by the CLI
/// banner, the startup log and `/props` is the effective mode rather than the
/// requested one. `ServerStartupInput::into_startup_config` performs the
/// substitution before `ServerConfig` is built, so the payload only has to
/// carry the resolved value; this test is what keeps the field from being
/// dropped and the doc sentence from going stale.
#[test]
fn props_reports_the_effective_kv_cache_mode_and_kv_bits() {
    let response = PropsResponse {
        default_generation_settings: default_generation_settings(&ServerConfig::default()),
        total_slots: 1,
        kv_cache_mode: mlxcel_core::cache::KVCacheMode::Turbo4Asym.to_string(),
        kv_bits: 4,
    };

    let payload = serde_json::to_value(&response).expect("PropsResponse serializes");
    // The `--kv-cache-mode` spelling, not the Rust variant name: an operator
    // has to be able to paste it straight back onto the flag.
    assert_eq!(payload["kv_cache_mode"], serde_json::json!("fp16+turbo4"));
    assert_eq!(payload["kv_bits"], serde_json::json!(4));
}

/// The default server reports `fp16` and `0`, not an absent key. Present-but-
/// default and absent are different answers to "is a quantized KV cache in
/// force", which is the same gap #1103 closed for `dry_sequence_breakers`.
#[test]
fn props_reports_the_kv_keys_even_on_a_default_server() {
    let config = ServerConfig::default();
    let response = PropsResponse {
        default_generation_settings: default_generation_settings(&config),
        total_slots: config.n_parallel,
        kv_cache_mode: config.kv_cache_mode.to_string(),
        kv_bits: config.batch_kv_quant.bits,
    };

    let payload = serde_json::to_value(&response).expect("PropsResponse serializes");
    assert_eq!(payload["kv_cache_mode"], serde_json::json!("fp16"));
    assert_eq!(payload["kv_bits"], serde_json::json!(0));
}
