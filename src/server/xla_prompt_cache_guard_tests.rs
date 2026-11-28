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

//! Tests for the OpenXLA recurrent prompt-cache guard (issue #3341).

use std::sync::Arc;

use super::*;
use crate::server::prompt_cache::ApcConfig;

/// Every recurrent `model_type` epic #564 covers.
const EPIC_RECURRENT_TYPES: &[&str] = &[
    "mamba",
    "mamba2",
    "falcon_mamba",
    "jamba",
    "nemotron_h",
    "falcon_h1",
    "plamo2",
    "qwen3_next",
    "qwen3_5",
    "kimi_linear",
    "rwkv7",
    "recurrent_gemma",
    "granitemoehybrid",
];

/// A Mamba2 config the emitter loads, written the way HF serializes it (a
/// bare `Infinity`), so the guard's `has_recurrent_state()` half decides.
const TINY_MAMBA2: &str = r#"{
  "model_type": "mamba2", "hidden_size": 16, "num_hidden_layers": 2,
  "vocab_size": 40, "num_heads": 4, "head_dim": 8, "state_size": 16,
  "n_groups": 2, "conv_kernel": 4, "expand": 2, "use_conv_bias": true,
  "use_bias": false, "time_step_limit": [0.0, Infinity],
  "layer_norm_epsilon": 1e-05, "chunk_size": 256, "rms_norm": true,
  "hidden_act": "silu", "tie_word_embeddings": true
}"#;

const TINY_LLAMA: &str = r#"{"model_type": "llama", "hidden_size": 16,
  "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
  "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
  "vocab_size": 32}"#;

fn model_dir(config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), config).expect("write config.json");
    dir
}

fn store() -> Option<Arc<PromptCacheStore>> {
    Some(Arc::new(PromptCacheStore::with_config(
        PromptCacheConfig::default(),
    )))
}

fn apc_requested() -> PromptCacheConfig {
    let config = PromptCacheConfig {
        apc: ApcConfig {
            enabled: true,
            ..ApcConfig::default()
        },
        ..PromptCacheConfig::default()
    };
    assert!(config.apc_enabled());
    config
}

fn assert_refused(dir: &Path, what: &str) {
    assert!(
        xla_provider_prompt_cache(dir, store()).is_none(),
        "{what}: the OpenXLA provider must not attach a prompt-cache store"
    );
    let mut config = apc_requested();
    assert!(
        disable_prompt_cache_for_xla_recurrent(&mut config, dir).is_some(),
        "{what}: startup must turn the prompt cache off"
    );
    assert!(!config.is_enabled(), "{what}: prompt cache must be off");
    assert!(!config.apc_enabled(), "{what}: APC must be off");
    assert!(
        !config.apc.enabled,
        "{what}: the APC request must be cleared"
    );
}

#[test]
fn every_epic_recurrent_type_gets_no_store_and_no_apc() {
    for &model_type in EPIC_RECURRENT_TYPES {
        let dir = model_dir(&format!(r#"{{"model_type": "{model_type}"}}"#));
        assert_refused(dir.path(), model_type);
    }
}

#[test]
fn an_emitter_loadable_mamba2_is_refused_through_has_recurrent_state() {
    assert_eq!(
        mlxcel_xla::config_json_has_recurrent_state(TINY_MAMBA2),
        Ok(true)
    );
    let dir = model_dir(TINY_MAMBA2);
    assert_eq!(xla_recurrent_family(dir.path()).as_deref(), Some("mamba2"));
    assert_refused(dir.path(), "tiny mamba2");
}

#[test]
fn an_emitter_loadable_nemotron_h_gets_no_prompt_cache_or_apc() {
    let config = include_str!("../lib/mlxcel-xla/assets/nemotron_h/config.json");
    assert_eq!(
        mlxcel_xla::config_json_has_recurrent_state(config),
        Ok(true)
    );
    let dir = model_dir(config);
    assert_refused(dir.path(), "tiny nemotron_h hybrid");
}

/// Jamba (issue #3349) is attention + Mamba1 + MoE: it has a KV cache, so the
/// guard must still refuse it through its recurrent state, for both the tiny
/// MoE fixture and the real 3B checkpoint's dense schedule.
#[test]
fn an_emitter_loadable_jamba_gets_no_prompt_cache_or_apc() {
    let tiny = include_str!("../lib/mlxcel-xla/assets/jamba/config.json");
    let real_3b = r#"{"model_type": "jamba", "hidden_size": 2560,
      "intermediate_size": 8192, "num_hidden_layers": 28, "vocab_size": 65536,
      "num_attention_heads": 20, "num_key_value_heads": 1,
      "attn_layer_period": 14, "attn_layer_offset": 7,
      "expert_layer_period": 2, "expert_layer_offset": 1, "num_experts": 1,
      "num_experts_per_tok": 1, "mamba_d_state": 16, "mamba_d_conv": 4,
      "mamba_expand": 2, "mamba_dt_rank": 160, "mamba_conv_bias": true,
      "mamba_proj_bias": false, "hidden_act": "silu", "rms_norm_eps": 1e-06,
      "sliding_window": null, "tie_word_embeddings": true}"#;
    for (what, config) in [("tiny jamba MoE", tiny), ("jamba 3B dense", real_3b)] {
        assert_eq!(
            mlxcel_xla::config_json_has_recurrent_state(config),
            Ok(true),
            "{what}"
        );
        let dir = model_dir(config);
        assert_eq!(
            xla_recurrent_family(dir.path()).as_deref(),
            Some("jamba"),
            "{what}"
        );
        assert_refused(dir.path(), what);
    }
}

#[test]
fn an_emitter_loadable_plamo2_gets_no_prompt_cache_or_apc() {
    let config = include_str!("../lib/mlxcel-xla/assets/plamo2/config.json");
    assert_eq!(
        mlxcel_xla::config_json_has_recurrent_state(config),
        Ok(true)
    );
    let dir = model_dir(config);
    assert_eq!(xla_recurrent_family(dir.path()).as_deref(), Some("plamo2"));
    assert_refused(dir.path(), "tiny plamo2 hybrid");
}

#[test]
fn an_attention_model_keeps_its_store_and_apc() {
    let dir = model_dir(TINY_LLAMA);
    assert!(xla_recurrent_family(dir.path()).is_none());
    assert!(xla_provider_prompt_cache(dir.path(), store()).is_some());
    let mut config = apc_requested();
    assert!(disable_prompt_cache_for_xla_recurrent(&mut config, dir.path()).is_none());
    assert!(config.apc_enabled());
}

#[test]
fn a_local_hf_mamba2_checkpoint_is_refused() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/hf/mamba2-130m");
    if !dir.join("config.json").is_file() {
        assert!(
            std::env::var_os("MLXCEL_REQUIRE_MODELS").is_none(),
            "models/hf/mamba2-130m is required but missing"
        );
        eprintln!("skipping: models/hf/mamba2-130m not found");
        return;
    }
    assert_eq!(xla_recurrent_family(&dir).as_deref(), Some("mamba2"));
    assert_refused(&dir, "mamba2-130m");
}

/// Sets `MLXCEL_BACKEND` and restores the previous value on drop, so a panic
/// cannot leave it set. Hold the crate-wide env lock for its whole lifetime.
struct BackendEnvGuard(Option<std::ffi::OsString>);

impl BackendEnvGuard {
    fn set(value: &str) -> Self {
        let previous = std::env::var_os("MLXCEL_BACKEND");
        // SAFETY: callers hold the crate-wide env lock, as every
        // env-mutating test in this crate does.
        unsafe {
            std::env::set_var("MLXCEL_BACKEND", value);
        }
        Self(previous)
    }
}

impl Drop for BackendEnvGuard {
    fn drop(&mut self) {
        // SAFETY: the env lock is still held; the guard is declared after it
        // and so drops first.
        unsafe {
            match self.0.take() {
                Some(value) => std::env::set_var("MLXCEL_BACKEND", value),
                None => std::env::remove_var("MLXCEL_BACKEND"),
            }
        }
    }
}

/// `startup::resolve_prompt_cache_store` is where the guard is wired in: with
/// `MLXCEL_BACKEND=xla` a recurrent model gets no store and both flags are
/// cleared, while a dense model keeps its store and APC.
#[test]
fn startup_resolver_builds_no_store_for_an_xla_recurrent_model() {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let _backend = BackendEnvGuard::set("xla");
    let resolve = |config_json: &str| {
        let dir = model_dir(config_json);
        let mut config = crate::server::ServerConfig::default();
        config.prompt_cache.enabled = true;
        config.prompt_cache.apc.enabled = true;
        let metrics = Arc::new(crate::server::state::BatchMetrics::new());
        let store =
            crate::server::startup::resolve_prompt_cache_store(&mut config, dir.path(), &metrics);
        (
            store.is_some(),
            config.prompt_cache.enabled,
            config.prompt_cache.apc.enabled,
        )
    };
    let recurrent = resolve(TINY_MAMBA2);
    let listed = resolve(r#"{"model_type": "granitemoehybrid"}"#);
    let dense = resolve(TINY_LLAMA);
    assert_eq!(recurrent, (false, false, false), "tiny mamba2");
    assert_eq!(listed, (false, false, false), "granitemoehybrid");
    assert_eq!(dense, (true, true, true), "tiny llama");
}

/// A `config.json` the guard cannot read or parse leaves the cache off: the
/// guard cannot rule out recurrent state, so it must not fail open.
#[test]
fn an_unreadable_or_unparseable_config_fails_closed() {
    let missing = tempfile::tempdir().expect("tempdir");
    let garbled = model_dir(r#"{"model_type": "mamba2", "#);
    for (dir, what) in [
        (missing.path(), "missing config.json"),
        (garbled.path(), "unparseable config.json"),
    ] {
        assert_eq!(
            xla_recurrent_family(dir).as_deref(),
            Some(UNDETECTED_FAMILY),
            "{what}"
        );
        assert_refused(dir, what);
    }
}
