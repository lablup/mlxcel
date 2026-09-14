// Copyright 2025-2026 Lablup Inc.
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

use super::*;
use crate::server::router_presets::{PresetSection, apply_section_to_startup};

fn profile(value: serde_json::Value) -> UiLoadProfile {
    serde_json::from_value(value).expect("valid profile DTO")
}

#[test]
fn profile_bounds_and_enum_follow_canonical_request_contract() {
    for value in [
        serde_json::json!({"ctx_size": 0}),
        serde_json::json!({"ctx_size": 262145}),
        serde_json::json!({"n_parallel": 0}),
        serde_json::json!({"n_parallel": 33}),
        serde_json::json!({"kv_cache_mode": "q8_0"}),
        serde_json::json!({"kv_cache_mode": "FP16"}),
    ] {
        assert!(profile(value).validate().is_err());
    }
    for mode in [
        "fp16",
        "float16",
        "int8",
        "i8",
        "turbo4-asym",
        "fp16+turbo4",
        "turbo3-asym",
        "fp16+turbo3",
        "turbo3",
        "turbo4",
        "turbo4-sym",
        "turbo4-delegated",
        "fp16+turbo4-delegated",
    ] {
        assert!(
            profile(serde_json::json!({"kv_cache_mode": mode}))
                .validate()
                .is_ok()
        );
    }
    for value in [
        serde_json::json!({"model_path": "/secret"}),
        serde_json::json!({"ctx_size": -1}),
        serde_json::json!({"n_parallel": 1.5}),
        serde_json::json!({"api_key":"secret"}),
    ] {
        assert!(serde_json::from_value::<UiLoadProfile>(value).is_err());
    }
}

#[test]
fn profile_omission_inherits_and_precedence_preserves_operator_flags() {
    let mut startup = ServerStartupConfig {
        ctx_size: 16384,
        n_parallel: 2,
        ..Default::default()
    };
    let cli = PresetCliOverrides {
        ctx_size: true,
        kv_cache_mode: true,
        ..Default::default()
    };
    let section = PresetSection {
        ctx_size: Some(4096),
        n_parallel: Some(4),
        ..Default::default()
    };
    apply_section_to_startup(&mut startup, &section, &cli);
    assert_eq!((startup.ctx_size, startup.n_parallel), (16384, 4));
    let input =
        profile(serde_json::json!({"ctx_size": 8192, "n_parallel": 8, "kv_cache_mode": "int8"}));
    assert!(input.apply(&mut startup, &cli).is_ok());
    assert_eq!((startup.ctx_size, startup.n_parallel), (16384, 8));
    assert_eq!(startup.kv_cache_mode, KVCacheMode::Fp16);
    let omitted = profile(serde_json::json!({"ctx_size": null, "n_parallel": null}));
    assert!(!omitted.has_overrides());
    assert!(omitted.apply(&mut startup, &cli).is_ok());
    assert_eq!((startup.ctx_size, startup.n_parallel), (16384, 8));
}

#[test]
fn profile_context_reuses_actual_per_slot_floor_and_config_conversion() {
    let mut startup = ServerStartupConfig::default();
    assert!(
        profile(serde_json::json!({"ctx_size": 1, "n_parallel": 32}))
            .apply(&mut startup, &Default::default())
            .is_err()
    );
    let mut startup = ServerStartupConfig::default();
    assert!(
        profile(serde_json::json!({"ctx_size": 8192, "n_parallel": 2}))
            .apply(&mut startup, &Default::default())
            .is_ok()
    );
    let config = crate::server::startup::build_server_config(&startup, Default::default());
    assert_eq!(config.context_size_total, 8192);
    assert_eq!(config.n_parallel, 2);
    assert_eq!(config.context_size, 4096);
    assert_eq!(
        ServerStartupConfig::default().ctx_size,
        0,
        "template not mutated"
    );
}

#[test]
fn profile_rejects_model_kv_substitution_and_read_only_distributed_geometry() {
    let temp = tempfile::tempdir().expect("temp model");
    std::fs::write(
        temp.path().join("config.json"),
        r#"{"model_type":"deepseek_v3"}"#,
    )
    .expect("config");
    let mut startup = ServerStartupConfig {
        model_path: temp.path().to_path_buf(),
        ..Default::default()
    };
    assert!(
        profile(serde_json::json!({"kv_cache_mode":"int8"}))
            .apply(&mut startup, &Default::default())
            .is_err()
    );
    startup.kv_cache_mode = KVCacheMode::Int8;
    assert!(
        profile(serde_json::json!({"ctx_size":8192}))
            .apply(&mut startup, &Default::default())
            .is_err(),
        "context-only profile revalidates inherited unsafe KV policy"
    );
    let mut startup = ServerStartupConfig {
        tp_size: 2,
        ..Default::default()
    };
    assert!(
        profile(serde_json::json!({"ctx_size": 8192}))
            .apply(&mut startup, &Default::default())
            .is_err()
    );
}
