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
